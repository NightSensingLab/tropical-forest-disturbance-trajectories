#!/usr/bin/env python3
"""Run the v1.0 reviewer-response analyses.

This module keeps the validated v0.9 Major 2/3 output as the primary result and
adds four auditable analyses requested in the review: an event-pre-period
control risk set with dynamic control censoring, a stabilized inverse-
probability-of-censoring sensitivity analysis, a trajectory-versus-t0
classification check, and region-specific direct contrasts with spatial-block
uncertainty.  It writes a new output directory and never overwrites v0.9.
"""

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from pipeline.major23_revision import _concat_parquet, _feature_paths, _stage03_parts
from pipeline.matched_control import (
    MATCHING_COVARIATES,
    MatchedControlSettings,
    match_risk_set,
    select_all_first_treatments,
)
from pipeline.matched_estimation import (
    ANNUAL_COLUMNS,
    EVI_METRICS,
    _expand_treatment_window,
    apply_dynamic_censoring,
    assign_spatial_blocks,
    common_support_weights,
    metric_baselines,
    summarize_matched_response,
    summarize_standardized_event_type_contrast,
    bootstrap_standardized_event_type_contrast,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = HERE.parent


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_parquet(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    os.replace(temporary, path)


def atomic_write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_catalog(config):
    path = Path(config["output_stage_06"]) / "event_catalog.parquet"
    catalog = pd.read_parquet(path)
    return catalog, path


def _load_features(stage07, event_year, settings):
    return _concat_parquet(
        _feature_paths(stage07, event_year, settings.primary_obs_threshold)
    )


def _pre_event_control_risk_set(features, first_event_year, event_year, treatment_ids):
    """Build a risk set using only threshold history through event_year - 1.

    The first-event catalog is used as an indexed representation of that
    history.  A control whose first threshold crossing is in the event year or
    later remains eligible at matching; its later crossing is handled by
    dynamic censoring in the outcome window.
    """
    required = {"grid_id", "coverage_eligible", "matching_covariates_complete", "lon", "lat"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table lacks pre-event control columns: {sorted(missing)}")
    prior_ids = set(
        first_event_year.loc[
            first_event_year["first_event_year"].lt(int(event_year)), "grid_id"
        ]
    )
    valid_coordinates = np.isfinite(
        features.loc[:, ["lon", "lat"]].to_numpy(dtype=np.float64)
    ).all(axis=1)
    mask = (
        features["coverage_eligible"].astype(bool)
        & features["matching_covariates_complete"].astype(bool)
        & valid_coordinates
        & ~features["grid_id"].isin(prior_ids)
        & ~features["grid_id"].isin(set(treatment_ids))
    )
    return features.loc[mask].sort_values("grid_id", kind="mergesort").reset_index(drop=True)


def _match_pre_event_controls(stage07, eligible, catalog, settings):
    first_event_year = catalog.loc[
        np.isclose(pd.to_numeric(catalog["event_threshold"], errors="coerce"), settings.primary_event_threshold),
        ["grid_id", "event_year"],
    ].rename(columns={"event_year": "first_event_year"})
    pair_frames = []
    treatment_frames = []
    for event_year in range(settings.primary_event_years[0], settings.primary_event_years[1] + 1):
        event_rows = eligible.loc[eligible["event_year"].eq(event_year)].copy()
        if event_rows.empty:
            continue
        features = _load_features(stage07, event_year, settings)
        treatment_features = features.loc[
            features["grid_id"].isin(event_rows["grid_id"])
        ].merge(
            event_rows,
            on=["grid_id", "event_year"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_event"),
        )
        controls = _pre_event_control_risk_set(
            features,
            first_event_year,
            event_year,
            treatment_features["grid_id"],
        )
        pairs = match_risk_set(treatment_features, controls, settings)
        if pairs.empty:
            continue
        pair_frames.append(pairs)
        treatment_frames.append(
            treatment_features.loc[
                treatment_features["grid_id"].isin(pairs["treatment_grid_id"])
            ]
        )
    if not pair_frames:
        raise RuntimeError("Pre-event control matching produced no pairs.")
    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["event_year", "treatment_grid_id", "match_rank"], kind="mergesort"
    ).reset_index(drop=True)
    treatments = pd.concat(treatment_frames, ignore_index=True).drop_duplicates(
        ["grid_id", "event_year"]
    )
    pairs = pairs.merge(
        first_event_year.rename(columns={"grid_id": "control_grid_id"}),
        on="control_grid_id",
        how="left",
        validate="many_to_one",
    )
    pairs["control_censor_relative_year"] = (
        pd.to_numeric(pairs["first_event_year"], errors="coerce")
        - pd.to_numeric(pairs["event_year"], errors="coerce")
    )
    censor = pairs["control_censor_relative_year"]
    pairs.loc[~censor.between(0, 5), "control_censor_relative_year"] = np.nan
    return pairs, treatments, first_event_year


def _load_annual_outcomes(config, grid_ids):
    grid_ids = set(map(int, grid_ids))
    frames = []
    for path in _stage03_parts(config):
        frame = pd.read_parquet(path, columns=list(ANNUAL_COLUMNS))
        selected = frame.loc[frame["grid_id"].isin(grid_ids)]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError("No annual outcomes were found for v1.0 matched IDs.")
    return pd.concat(frames, ignore_index=True)


def _aggregate_control_metric_dynamic(control_metric, pairs):
    """Aggregate controls after removing links at/after their own crossing."""
    linked = pairs.loc[
        :,
        [
            "treatment_grid_id",
            "control_grid_id",
            "event_year",
            "within_set_weight",
            "control_censor_relative_year",
        ],
    ].merge(
        control_metric.rename(columns={"grid_id": "control_grid_id"}),
        on=["control_grid_id", "event_year"],
        how="left",
        validate="many_to_many",
    )
    relative_year = pd.to_numeric(linked["relative_year"], errors="coerce")
    control_censor = pd.to_numeric(
        linked["control_censor_relative_year"], errors="coerce"
    )
    before_control_event = control_censor.isna() | relative_year.lt(control_censor)
    delta = pd.to_numeric(linked["delta"], errors="coerce")
    available = np.where(
        delta.notna() & before_control_event,
        pd.to_numeric(linked["within_set_weight"], errors="coerce"),
        0.0,
    )
    linked["available_weight"] = available
    for column in ("value", "baseline", "delta"):
        values = pd.to_numeric(linked[column], errors="coerce")
        linked[f"weighted_{column}"] = np.where(available > 0, values * available, 0.0)
    grouped = linked.groupby(
        ["treatment_grid_id", "event_year", "relative_year"],
        as_index=False,
        sort=True,
    ).agg(
        available_weight=("available_weight", "sum"),
        usable_controls=("available_weight", lambda value: int((value > 0.0).sum())),
        weighted_value=("weighted_value", "sum"),
        weighted_baseline=("weighted_baseline", "sum"),
        weighted_delta=("weighted_delta", "sum"),
    )
    for column in ("value", "baseline", "delta"):
        grouped[column] = np.where(
            grouped["available_weight"].gt(0.0),
            grouped[f"weighted_{column}"] / grouped["available_weight"],
            np.nan,
        )
    return grouped


def _build_pre_event_dynamic_window(annual, treatments, pairs, settings):
    result = _expand_treatment_window(treatments, pairs, settings.relative_years)
    treatment_keys = result.loc[:, ["treatment_grid_id", "event_year"]].rename(
        columns={"treatment_grid_id": "grid_id"}
    )
    control_keys = pairs.loc[:, ["control_grid_id", "event_year"]].rename(
        columns={"control_grid_id": "grid_id"}
    )
    for metric in EVI_METRICS:
        treatment_metric = metric_baselines(
            annual, treatment_keys, metric, settings.primary_obs_threshold
        ).rename(
            columns={
                "grid_id": "treatment_grid_id",
                "value": f"treated_value_{metric}",
                "baseline": f"treated_baseline_{metric}",
                "baseline_n": f"treated_baseline_n_{metric}",
                "delta": f"treated_delta_{metric}",
                "usable": f"treated_usable_{metric}",
            }
        )
        treated_columns = [
            "treatment_grid_id",
            "event_year",
            "relative_year",
            f"treated_value_{metric}",
            f"treated_baseline_{metric}",
            f"treated_baseline_n_{metric}",
            f"treated_delta_{metric}",
            f"treated_usable_{metric}",
        ]
        result = result.merge(
            treatment_metric.loc[:, treated_columns],
            on=["treatment_grid_id", "event_year", "relative_year"],
            how="left",
            validate="one_to_one",
        )
        control_metric = metric_baselines(
            annual, control_keys, metric, settings.primary_obs_threshold
        )
        aggregate = _aggregate_control_metric_dynamic(control_metric, pairs).rename(
            columns={
                "value": f"control_value_{metric}",
                "baseline": f"control_baseline_{metric}",
                "delta": f"control_delta_{metric}",
                "usable_controls": f"usable_controls_{metric}",
            }
        )
        control_columns = [
            "treatment_grid_id",
            "event_year",
            "relative_year",
            f"control_value_{metric}",
            f"control_baseline_{metric}",
            f"control_delta_{metric}",
            f"usable_controls_{metric}",
        ]
        result = result.merge(
            aggregate.loc[:, control_columns],
            on=["treatment_grid_id", "event_year", "relative_year"],
            how="left",
            validate="one_to_one",
        )
        result[f"pair_effect_{metric}"] = (
            pd.to_numeric(result[f"treated_delta_{metric}"], errors="coerce")
            - pd.to_numeric(result[f"control_delta_{metric}"], errors="coerce")
        )
    return apply_dynamic_censoring(
        result.sort_values(
            ["event_year", "treatment_grid_id", "relative_year"], kind="mergesort"
        ).reset_index(drop=True)
    )


def _auc(y, scores):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    return float((ranks[y == 1].sum() - positives * (positives + 1.0) / 2.0) / (positives * negatives))


def _best_balanced_threshold(scores, y):
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    positive = sorted_y.astype(np.int64)
    negative = 1 - positive
    tp_if_threshold = positive[::-1].cumsum()[::-1]
    fp_if_threshold = negative[::-1].cumsum()[::-1]
    n_positive = max(int(positive.sum()), 1)
    n_negative = max(int(negative.sum()), 1)
    balanced = (tp_if_threshold / n_positive + (n_negative - fp_if_threshold) / n_negative) / 2.0
    index = int(np.nanargmax(balanced))
    return float(sorted_scores[index])


def _stratified_folds(y, n_folds=5, seed=20260725):
    rng = np.random.default_rng(int(seed))
    parts = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        parts.append(np.array_split(indices, int(n_folds)))
    return [np.concatenate([parts[0][i], parts[1][i]]) for i in range(int(n_folds))]


def _cross_validated_classifier(wide, feature_sets, seed=20260725, n_folds=5):
    complete_columns = sorted(set().union(*feature_sets.values()))
    complete = wide.dropna(subset=complete_columns).copy()
    y = (complete["event_type"].to_numpy() == "deforestation").astype(np.int8)
    if len(np.unique(y)) != 2:
        raise RuntimeError("Both event types are required for classification.")
    folds = _stratified_folds(y, n_folds=n_folds, seed=seed)
    rows = []
    for label, columns in feature_sets.items():
        x = complete.loc[:, columns].to_numpy(dtype=np.float64)
        predictions = np.full(len(complete), np.nan, dtype=np.float64)
        fold_balanced = []
        for test in folds:
            train = np.setdiff1d(np.arange(len(complete)), test)
            mean = x[train].mean(axis=0)
            scale = x[train].std(axis=0)
            scale[scale == 0.0] = 1.0
            z = (x - mean) / scale
            beta = np.linalg.solve(
                z[train].T @ z[train] + np.eye(len(columns)),
                z[train].T @ (y[train] - y[train].mean()),
            )
            predictions[test] = z[test] @ beta + y[train].mean()
            threshold = _best_balanced_threshold(z[train] @ beta + y[train].mean(), y[train])
            predicted = predictions[test] >= threshold
            truth = y[test]
            sensitivity = float(predicted[truth == 1].mean())
            specificity = float((~predicted[truth == 0]).mean())
            fold_balanced.append((sensitivity + specificity) / 2.0)
        rows.append(
            {
                "feature_set": label,
                "features": ",".join(columns),
                "sample_n": int(len(complete)),
                "deforestation_n": int(y.sum()),
                "degradation_n": int(len(y) - y.sum()),
                "cv_folds": int(n_folds),
                "auc": _auc(y, predictions),
                "balanced_accuracy": float(np.mean(fold_balanced)),
                "balanced_accuracy_sd": float(np.std(fold_balanced, ddof=1)),
                "seed": int(seed),
                "model": "ridge linear discriminant; equal-event weighting",
                "interpretation": "exploratory discrimination, not causal prediction",
            }
        )
    return pd.DataFrame(rows)


def _weighted_mean(values, weights):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def _regional_contrasts(window, event_table, settings, n_bins=5):
    weights, diagnostics = common_support_weights(event_table, n_bins=n_bins)
    working = window.merge(
        weights.loc[:, ["treatment_grid_id", "event_year", "standardized_event_weight"]],
        on=["treatment_grid_id", "event_year"],
        how="inner",
        validate="many_to_one",
    )
    working["spatial_block"] = assign_spatial_blocks(working, 2.0)
    rows = []
    rng = np.random.default_rng(int(settings.bootstrap_seed))
    for region in sorted(working["macroregion"].dropna().unique()):
        for relative_year in settings.relative_years:
            frame = working.loc[
                working["macroregion"].eq(region)
                & working["relative_year"].eq(relative_year)
                & working["pair_effect_evi_mean"].notna()
            ].copy()
            blocks = np.sort(frame["spatial_block"].unique())
            if len(blocks) == 0:
                continue
            point = {}
            block_sums = {}
            counts = {}
            for event_type in ("degradation", "deforestation"):
                group = frame.loc[frame["event_type"].eq(event_type)]
                values = pd.to_numeric(group["pair_effect_evi_mean"], errors="coerce").to_numpy()
                event_weights = pd.to_numeric(group["standardized_event_weight"], errors="coerce").to_numpy()
                point[event_type] = _weighted_mean(values, event_weights)
                counts[event_type] = int(np.isfinite(values).sum())
                block_sums[event_type] = group.assign(
                    _num=values * event_weights,
                    _den=event_weights,
                ).groupby("spatial_block")[["_num", "_den"]].sum().reindex(blocks, fill_value=0.0)
            if not all(np.isfinite(point.get(kind, np.nan)) for kind in ("degradation", "deforestation")):
                continue
            draw_counts = rng.multinomial(
                len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=settings.bootstrap_replicates
            )
            type_draws = {}
            for event_type in ("degradation", "deforestation"):
                numerators = draw_counts @ block_sums[event_type]["_num"].to_numpy()
                denominators = draw_counts @ block_sums[event_type]["_den"].to_numpy()
                type_draws[event_type] = np.divide(
                    numerators,
                    denominators,
                    out=np.full(len(draw_counts), np.nan),
                    where=denominators > 0.0,
                )
            contrast_draws = type_draws["deforestation"] - type_draws["degradation"]
            finite = contrast_draws[np.isfinite(contrast_draws)]
            rows.append(
                {
                    "macroregion": region,
                    "relative_year": int(relative_year),
                    "degradation_estimate": point["degradation"],
                    "deforestation_estimate": point["deforestation"],
                    "direct_contrast": point["deforestation"] - point["degradation"],
                    "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "valid_bootstrap": int(len(finite)),
                    "spatial_block_count": int(len(blocks)),
                    "degradation_n": counts["degradation"],
                    "deforestation_n": counts["deforestation"],
                    "standardization_strata": int(frame["standardized_event_weight"].notna().sum()),
                    "estimand": "common-support standardized conditional association",
                }
            )
    return pd.DataFrame(rows), diagnostics


def _spatial_block_distribution(window, event_table, settings, n_bins=5):
    weights, _ = common_support_weights(event_table, n_bins=n_bins)
    working = window.merge(
        weights.loc[:, ["treatment_grid_id", "event_year", "standardized_event_weight"]],
        on=["treatment_grid_id", "event_year"],
        how="inner",
        validate="many_to_one",
    )
    working["spatial_block"] = assign_spatial_blocks(working, 2.0)
    rng = np.random.default_rng(int(settings.bootstrap_seed) + 11)
    rows = []
    for region in sorted(working["macroregion"].dropna().unique()):
        for relative_year in (3, 4, 5):
            frame = working.loc[
                working["macroregion"].eq(region)
                & working["relative_year"].eq(relative_year)
                & working["pair_effect_evi_mean"].notna()
            ].copy()
            grouped = frame.groupby(["spatial_block", "event_type"], sort=True).apply(
                lambda group: pd.Series(
                    {
                        "num": float((group["pair_effect_evi_mean"] * group["standardized_event_weight"]).sum()),
                        "den": float(group["standardized_event_weight"].sum()),
                    }
                ),
                include_groups=False,
            ).reset_index()
            pivot = grouped.pivot(index="spatial_block", columns="event_type")
            if not {"degradation", "deforestation"}.issubset(pivot["num"].columns):
                continue
            direct = (
                pivot["num"]["deforestation"] / pivot["den"]["deforestation"]
                - pivot["num"]["degradation"] / pivot["den"]["degradation"]
            ).dropna()
            block_weight = np.minimum(
                pivot["den"]["deforestation"], pivot["den"]["degradation"]
            ).reindex(direct.index)
            blocks = direct.index.to_numpy()
            counts = rng.multinomial(
                len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=settings.bootstrap_replicates
            )
            base_weight = block_weight.to_numpy(dtype=np.float64)
            negative_draws = np.divide(
                counts @ (base_weight * (direct.to_numpy() < 0.0)),
                counts @ base_weight,
                out=np.full(settings.bootstrap_replicates, np.nan),
                where=(counts @ base_weight) > 0.0,
            )
            finite = negative_draws[np.isfinite(negative_draws)]
            direct_values = direct.to_numpy(dtype=np.float64)
            rows.append(
                {
                    "macroregion": region,
                    "relative_year": int(relative_year),
                    "spatial_block_count": int(len(direct)),
                    "q10": float(np.quantile(direct_values, 0.10)),
                    "q25": float(np.quantile(direct_values, 0.25)),
                    "median": float(np.quantile(direct_values, 0.50)),
                    "q75": float(np.quantile(direct_values, 0.75)),
                    "q90": float(np.quantile(direct_values, 0.90)),
                    "weighted_negative_fraction": float(
                        np.average(direct_values < 0.0, weights=base_weight)
                    ),
                    "negative_fraction_ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "negative_fraction_ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "valid_bootstrap": int(len(finite)),
                    "estimand": "direct deforestation-minus-degradation spatial-block effect",
                }
            )
    return pd.DataFrame(rows)


def _stabilized_ipcw_weights(event_table, relative_years):
    """Estimate transparent baseline-stratified stabilized censoring weights.

    The censoring event is a subsequent threshold crossing.  Probabilities are
    estimated within event type, macroregion and broad event-year period so
    that the sensitivity analysis addresses the observed imbalance in repeat
    disturbance without presenting the weights as a fully causal model.
    """
    required = {
        "treatment_grid_id",
        "event_year",
        "event_type",
        "macroregion",
        "censor_relative_year",
    }
    missing = required - set(event_table.columns)
    if missing:
        raise ValueError(f"Event table lacks IPCW columns: {sorted(missing)}")
    events = event_table.loc[:, sorted(required)].drop_duplicates(
        ["treatment_grid_id", "event_year"]
    ).copy()
    events["event_year"] = pd.to_numeric(events["event_year"], errors="coerce")
    events["censor_relative_year"] = pd.to_numeric(
        events["censor_relative_year"], errors="coerce"
    )
    events["censor_year_period"] = pd.cut(
        events["event_year"],
        bins=[2003, 2008, 2013, 2018],
        labels=False,
        include_lowest=True,
    )
    rows = []
    diagnostics = []
    for relative_year in relative_years:
        relative_year = int(relative_year)
        observed = events["censor_relative_year"].isna() | events[
            "censor_relative_year"
        ].gt(relative_year)
        if relative_year <= 0:
            conditional_probability = pd.Series(1.0, index=events.index)
            marginal_probability = pd.Series(1.0, index=events.index)
            stabilized = pd.Series(1.0, index=events.index)
        else:
            conditional_probability = events.assign(
                _observed=observed.astype(float)
            ).groupby(
                ["macroregion", "event_type", "censor_year_period"],
                dropna=False,
            )["_observed"].transform("mean")
            marginal_probability = events.assign(
                _observed=observed.astype(float)
            ).groupby("event_type", dropna=False)["_observed"].transform("mean")
            stabilized = (marginal_probability / conditional_probability).clip(
                lower=0.25, upper=4.0
            )
        rows.append(
            pd.DataFrame(
                {
                    "treatment_grid_id": events["treatment_grid_id"].to_numpy(),
                    "event_year": events["event_year"].to_numpy(),
                    "event_type": events["event_type"].to_numpy(),
                    "macroregion": events["macroregion"].to_numpy(),
                    "censor_year_period": events["censor_year_period"].to_numpy(),
                    "relative_year": relative_year,
                    "observed_uncensored": observed.to_numpy(dtype=bool),
                    "p_uncensored_conditional": conditional_probability.to_numpy(
                        dtype=np.float64
                    ),
                    "p_uncensored_marginal": marginal_probability.to_numpy(
                        dtype=np.float64
                    ),
                    "stabilized_ipcw": stabilized.to_numpy(dtype=np.float64),
                }
            )
        )
        diagnostics_frame = pd.DataFrame(
            {
                "event_type": events["event_type"].to_numpy(),
                "observed_uncensored": observed.to_numpy(dtype=bool),
                "conditional_probability": conditional_probability.to_numpy(
                    dtype=np.float64
                ),
                "stabilized_ipcw": stabilized.to_numpy(dtype=np.float64),
            }
        )
        for event_type, group in diagnostics_frame.groupby("event_type", sort=True):
            diagnostics.append(
                {
                    "event_type": str(event_type),
                    "relative_year": relative_year,
                    "event_n": int(len(group)),
                    "uncensored_n": int(group["observed_uncensored"].sum()),
                    "marginal_uncensored_fraction": float(
                        group["observed_uncensored"].mean()
                    ),
                    "conditional_probability_min": float(
                        group["conditional_probability"].min()
                    ),
                    "conditional_probability_max": float(
                        group["conditional_probability"].max()
                    ),
                    "stabilized_ipcw_mean": float(group["stabilized_ipcw"].mean()),
                    "stabilized_ipcw_max": float(group["stabilized_ipcw"].max()),
                    "weight_truncated_fraction": float(
                        (group["stabilized_ipcw"].isin([0.25, 4.0])).mean()
                    ),
                    "model": "baseline-stratified stabilized IPCW; strata=event type x macroregion x 5-year event period",
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diagnostics)


def _ipcw_event_type_contrast(window, event_table, settings, n_bins=5):
    """Compute a block-bootstrap contrast after stabilized IPCW sensitivity weighting."""
    effect_column = "pair_effect_evi_mean"
    standardized, _ = common_support_weights(event_table, n_bins=n_bins)
    event_weights = standardized.loc[
        :, ["treatment_grid_id", "event_year", "standardized_event_weight"]
    ].merge(
        event_table.loc[
            :, ["treatment_grid_id", "event_year", "event_type", "lon", "lat"]
        ].drop_duplicates(["treatment_grid_id", "event_year"]),
        on=["treatment_grid_id", "event_year"],
        how="left",
        validate="one_to_one",
    )
    event_weights["spatial_block"] = assign_spatial_blocks(event_weights, 2.0)
    ipcw, diagnostics = _stabilized_ipcw_weights(event_table, settings.relative_years)
    rows = []
    rng = np.random.default_rng(int(settings.bootstrap_seed) + 23)
    for relative_year in settings.relative_years:
        frame = window.loc[
            window["relative_year"].eq(relative_year),
            ["treatment_grid_id", "event_year", effect_column],
        ].merge(
            event_weights,
            on=["treatment_grid_id", "event_year"],
            how="inner",
            validate="one_to_one",
        ).merge(
            ipcw.loc[
                ipcw["relative_year"].eq(relative_year),
                ["treatment_grid_id", "event_year", "stabilized_ipcw"],
            ],
            on=["treatment_grid_id", "event_year"],
            how="inner",
            validate="one_to_one",
        )
        frame = frame.loc[frame[effect_column].notna()].copy()
        if frame.empty:
            continue
        frame["ipcw_weight"] = (
            pd.to_numeric(frame["standardized_event_weight"], errors="coerce")
            * pd.to_numeric(frame["stabilized_ipcw"], errors="coerce")
        )
        frame = frame.loc[
            np.isfinite(frame[effect_column].to_numpy(dtype=np.float64))
            & np.isfinite(frame["ipcw_weight"].to_numpy(dtype=np.float64))
            & frame["ipcw_weight"].gt(0.0)
        ].copy()
        blocks = np.sort(frame["spatial_block"].unique())
        if len(blocks) == 0:
            continue
        point = {}
        block_sums = {}
        for event_type in ("degradation", "deforestation"):
            group = frame.loc[frame["event_type"].eq(event_type)].copy()
            values = pd.to_numeric(group[effect_column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            weights = pd.to_numeric(group["ipcw_weight"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            point[event_type] = _weighted_mean(values, weights)
            group["_num"] = values * weights
            group["_den"] = weights
            block_sums[event_type] = group.groupby("spatial_block")[["_num", "_den"]].sum().reindex(
                blocks, fill_value=0.0
            )
        if not all(np.isfinite(point.get(kind, np.nan)) for kind in ("degradation", "deforestation")):
            continue
        draw_counts = rng.multinomial(
            len(blocks),
            np.full(len(blocks), 1.0 / len(blocks)),
            size=settings.bootstrap_replicates,
        )
        draws = {}
        for event_type in ("degradation", "deforestation"):
            numerators = draw_counts @ block_sums[event_type]["_num"].to_numpy()
            denominators = draw_counts @ block_sums[event_type]["_den"].to_numpy()
            draws[event_type] = np.divide(
                numerators,
                denominators,
                out=np.full(len(draw_counts), np.nan),
                where=denominators > 0.0,
            )
        contrast_draws = draws["deforestation"] - draws["degradation"]
        for event_type, estimate in (
            ("degradation", point["degradation"]),
            ("deforestation", point["deforestation"]),
            (
                "deforestation_minus_degradation",
                point["deforestation"] - point["degradation"],
            ),
        ):
            draw_values = (
                contrast_draws
                if event_type == "deforestation_minus_degradation"
                else draws[event_type]
            )
            finite = draw_values[np.isfinite(draw_values)]
            rows.append(
                {
                    "event_type": event_type,
                    "metric": "evi_mean",
                    "relative_year": int(relative_year),
                    "estimate": float(estimate),
                    "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "bootstrap_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "requested_replicates": int(settings.bootstrap_replicates),
                    "valid_replicates": int(len(finite)),
                    "spatial_block_count": int(len(blocks)),
                    "weighting": "standardized event area x stabilized IPCW",
                    "censoring_model": "event type x macroregion x 5-year event period",
                    "treatment_count": int(frame.loc[frame["event_type"].eq(event_type), "treatment_grid_id"].nunique()) if event_type in {"degradation", "deforestation"} else np.nan,
                    "effective_sample_size": float(
                        frame.loc[frame["event_type"].eq(event_type), "ipcw_weight"].sum() ** 2
                        / np.sum(frame.loc[frame["event_type"].eq(event_type), "ipcw_weight"].to_numpy() ** 2)
                    ) if event_type in {"degradation", "deforestation"} else np.nan,
                }
            )
    return pd.DataFrame(rows), ipcw, diagnostics


def _read_main_window(main_root):
    main_root = Path(main_root)
    window = pd.read_parquet(main_root / "all_first_matched_event_window.parquet")
    event_table = window.loc[
        :,
        [
            "treatment_grid_id",
            "event_year",
            "event_type",
            "macroregion",
            "event_area_m2",
            "disturbance_share",
            "degradation_share",
            "deforestation_share",
            "baseline_undist_share",
            "lon",
            "lat",
            "censor_relative_year",
        ],
    ].drop_duplicates(["treatment_grid_id", "event_year"])
    return window, event_table


def _build_completion_figure(classification, pre_bootstrap, regional, block_distribution, main_root, output_root):
    """Create a compact, English-labelled supplement diagnostic figure."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    # Panel a: classification information on the identical complete-trajectory set.
    ax = axes[0, 0]
    labels = ["t0 only", "t0-t5 trajectory"]
    values = []
    for label in ("t0_only", "t0_to_t5_trajectory"):
        row = classification.loc[classification["feature_set"].eq(label)].iloc[0]
        values.append((float(row["auc"]), float(row["balanced_accuracy"]), float(row["balanced_accuracy_sd"])))
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, [row[0] for row in values], width, label="AUC", color="#234f70")
    ax.bar(
        x + width / 2,
        [row[1] for row in values],
        width,
        yerr=[row[2] for row in values],
        capsize=3,
        label="Balanced accuracy",
        color="#d07a32",
    )
    ax.set_ylim(0.45, 0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cross-validated score")
    ax.set_title("a  Event-type discrimination")
    ax.legend(frameon=False, fontsize=8)

    # Panel b: primary estimate versus the event-pre-period-control sensitivity.
    ax = axes[0, 1]
    main_bootstrap = pd.read_csv(Path(main_root) / "standardized_event_type_bootstrap.csv")
    rows = []
    for source, frame in (("Main", main_bootstrap), ("Pre-event controls", pre_bootstrap)):
        selected = frame.loc[
            frame["event_type"].eq("deforestation_minus_degradation")
            & frame["relative_year"].isin([0, 5])
        ]
        for _, row in selected.iterrows():
            rows.append((source, int(row["relative_year"]), float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])))
    plot = pd.DataFrame(rows, columns=["source", "relative_year", "estimate", "ci_low", "ci_high"])
    for offset, source in enumerate(("Main", "Pre-event controls")):
        selected = plot.loc[plot["source"].eq(source)]
        xpos = np.arange(len(selected)) + (offset - 0.5) * 0.18
        ax.errorbar(
            xpos,
            selected["estimate"],
            yerr=[selected["estimate"] - selected["ci_low"], selected["ci_high"] - selected["estimate"]],
            fmt="o",
            capsize=3,
            label=source,
        )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks([0, 1], ["t0", "t5"])
    ax.set_ylabel("Deforestation minus degradation")
    ax.set_title("b  Control-selection sensitivity")
    ax.legend(frameon=False, fontsize=8)

    # Panel c: regional direct contrasts at t5.
    ax = axes[1, 0]
    selected = regional.loc[regional["relative_year"].eq(5)].copy()
    selected["label"] = selected["macroregion"].str.replace("_proxy", "", regex=False).str.replace("Southeast_Asia", "SE Asia", regex=False)
    selected = selected.sort_values("direct_contrast")
    xpos = np.arange(len(selected))
    ax.errorbar(
        xpos,
        selected["direct_contrast"],
        yerr=[selected["direct_contrast"] - selected["ci_low"], selected["ci_high"] - selected["direct_contrast"]],
        fmt="o",
        color="#234f70",
        capsize=3,
    )
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(xpos, selected["label"], rotation=25, ha="right")
    ax.set_ylabel("t5 direct contrast")
    ax.set_title("c  Regional direct contrasts")

    # Panel d: weighted fraction of spatial blocks with a negative direct contrast.
    ax = axes[1, 1]
    selected = block_distribution.loc[block_distribution["relative_year"].eq(5)].copy()
    selected["label"] = selected["macroregion"].str.replace("_proxy", "", regex=False).str.replace("Southeast_Asia", "SE Asia", regex=False)
    selected = selected.sort_values("weighted_negative_fraction")
    xpos = np.arange(len(selected))
    ax.errorbar(
        xpos,
        selected["weighted_negative_fraction"],
        yerr=[selected["weighted_negative_fraction"] - selected["negative_fraction_ci_low"], selected["negative_fraction_ci_high"] - selected["weighted_negative_fraction"]],
        fmt="o",
        color="#d07a32",
        capsize=3,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xpos, selected["label"], rotation=25, ha="right")
    ax.set_ylabel("Weighted negative-block fraction")
    ax.set_title("d  Spatial-block effect distribution")
    figure.suptitle("Reviewer-completion diagnostics", fontsize=13)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".png", 300), (".tiff", 300), (".pdf", None), (".svg", None)):
        kwargs = {"dpi": dpi} if dpi else {}
        figure.savefig(output_root / f"reviewer_completion{suffix}", **kwargs)
    plt.close(figure)


def run_pipeline(config, main_root, output_dir, settings=None):
    settings = settings or MatchedControlSettings()
    config = json.loads(Path(config).read_text(encoding="utf-8")) if isinstance(config, (str, Path)) else dict(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_root = Path(main_root)
    catalog, catalog_path = _load_catalog(config)
    eligible = select_all_first_treatments(catalog, settings)

    pre_pairs, pre_treatments, first_event_year = _match_pre_event_controls(
        Path(config["output_stage_07"]), eligible, catalog, settings
    )
    annual = _load_annual_outcomes(
        config,
        set(pre_pairs["treatment_grid_id"]) | set(pre_pairs["control_grid_id"]),
    )
    pre_window = _build_pre_event_dynamic_window(
        annual, pre_treatments, pre_pairs, settings
    )
    pre_event_table = pre_treatments.rename(columns={"grid_id": "treatment_grid_id"})
    pre_summary, pre_diag, _ = summarize_standardized_event_type_contrast(
        pre_window, pre_event_table, metric="evi_mean"
    )
    pre_bootstrap = bootstrap_standardized_event_type_contrast(
        pre_window,
        pre_event_table,
        metric="evi_mean",
        relative_years=settings.relative_years,
        replicates=settings.bootstrap_replicates,
        seed=settings.bootstrap_seed,
    )

    main_window, main_event_table = _read_main_window(main_root)
    classification_wide = main_window.pivot(
        index=["treatment_grid_id", "event_year", "event_type"],
        columns="relative_year",
        values="pair_effect_evi_mean",
    ).reset_index()
    classification_wide.columns = [
        (f"t{int(value)}" if value >= 0 else f"tm{abs(int(value))}")
        if isinstance(value, (int, np.integer))
        else value
        for value in classification_wide.columns
    ]
    classification = _cross_validated_classifier(
        classification_wide,
        {
            "t0_only": ["t0"],
            "t0_to_t5_trajectory": ["t0", "t1", "t2", "t3", "t4", "t5"],
        },
        seed=settings.bootstrap_seed,
        n_folds=5,
    )
    regional, regional_diag = _regional_contrasts(
        main_window, main_event_table, settings
    )
    block_distribution = _spatial_block_distribution(
        main_window, main_event_table, settings
    )
    ipcw_contrast, ipcw_weights, ipcw_diagnostics = _ipcw_event_type_contrast(
        main_window, main_event_table, settings
    )

    atomic_write_parquet(pre_pairs, output_dir / "pre_event_dynamic_control_pairs.parquet")
    atomic_write_parquet(pre_window, output_dir / "pre_event_dynamic_control_window.parquet")
    atomic_write_csv(pre_summary, output_dir / "pre_event_dynamic_standardized_contrast.csv")
    atomic_write_csv(pre_bootstrap, output_dir / "pre_event_dynamic_standardized_bootstrap.csv")
    atomic_write_csv(classification, output_dir / "trajectory_classification.csv")
    atomic_write_csv(regional, output_dir / "regional_event_type_contrast.csv")
    atomic_write_csv(block_distribution, output_dir / "spatial_block_distribution.csv")
    atomic_write_csv(ipcw_contrast, output_dir / "ipcw_censoring_sensitivity.csv")
    atomic_write_parquet(
        ipcw_weights, output_dir / "ipcw_censoring_weights.parquet"
    )
    atomic_write_csv(
        ipcw_diagnostics, output_dir / "ipcw_censoring_model.csv"
    )
    atomic_write_csv(
        pd.DataFrame(
            [
                {
                    "analysis": "pre_event_dynamic_control",
                    "eligible_treatments": int(len(eligible)),
                    "matched_treatments": int(pre_pairs["treatment_grid_id"].nunique()),
                    "pair_links": int(len(pre_pairs)),
                    "links_with_control_censor": int(pre_pairs["control_censor_relative_year"].notna().sum()),
                    "t5_valid_treatment_events": int(pre_window.loc[pre_window["relative_year"].eq(5), "pair_effect_evi_mean"].notna().sum()),
                    "control_selection": "no threshold crossing through event_year-1",
                    "control_dynamic_rule": "exclude control at and after its first threshold crossing",
                },
                {
                    "analysis": "main_all_first_dynamic",
                    "eligible_treatments": int(main_window["treatment_grid_id"].nunique()),
                    "matched_treatments": int(main_window["treatment_grid_id"].nunique()),
                    "pair_links": np.nan,
                    "links_with_control_censor": np.nan,
                    "t5_valid_treatment_events": int(main_window.loc[main_window["relative_year"].eq(5), "pair_effect_evi_mean"].notna().sum()),
                    "control_selection": "stable through follow-up",
                    "control_dynamic_rule": "none; controls are fixed",
                },
            ]
        ),
        output_dir / "analysis_sample_flow.csv",
    )
    _build_completion_figure(
        classification,
        pre_bootstrap,
        regional,
        block_distribution,
        main_root,
        output_dir / "figures",
    )

    report = {
        "created_at_utc": utc_now(),
        "analysis_version": "2026-08-26-reviewer-completion-v1",
        "primary_input": str(main_root),
        "event_catalog": str(catalog_path),
        "pre_event_dynamic_control": {
            "eligible_treatments": int(len(eligible)),
            "matched_treatments": int(pre_pairs["treatment_grid_id"].nunique()),
            "pair_links": int(len(pre_pairs)),
            "links_with_control_censor": int(pre_pairs["control_censor_relative_year"].notna().sum()),
            "t5_valid_treatment_events": int(pre_window.loc[pre_window["relative_year"].eq(5), "pair_effect_evi_mean"].notna().sum()),
            "standardization": pre_diag,
        },
        "classification": classification.to_dict(orient="records"),
        "regional_contrast_rows": int(len(regional)),
        "spatial_block_distribution_rows": int(len(block_distribution)),
        "ipcw_censoring_sensitivity": {
            "rows": int(len(ipcw_contrast)),
            "model": "event type x macroregion x 5-year event period stabilized IPCW",
            "weight_range": [
                float(ipcw_weights["stabilized_ipcw"].min()),
                float(ipcw_weights["stabilized_ipcw"].max()),
            ],
            "t5_direct_contrast": float(
                ipcw_contrast.loc[
                    ipcw_contrast["event_type"].eq("deforestation_minus_degradation")
                    & ipcw_contrast["relative_year"].eq(5),
                    "estimate",
                ].iloc[0]
            ),
            "t5_ci": [
                float(
                    ipcw_contrast.loc[
                        ipcw_contrast["event_type"].eq(
                            "deforestation_minus_degradation"
                        )
                        & ipcw_contrast["relative_year"].eq(5),
                        "ci_low",
                    ].iloc[0]
                ),
                float(
                    ipcw_contrast.loc[
                        ipcw_contrast["event_type"].eq(
                            "deforestation_minus_degradation"
                        )
                        & ipcw_contrast["relative_year"].eq(5),
                        "ci_high",
                    ].iloc[0]
                ),
            ],
        },
        "settings": asdict(settings),
        "interpretation": {
            "major3": "The pre-event control and stabilized IPCW analyses are sensitivity analyses; the primary estimand remains relative to controls stable through follow-up.",
            "major1": "Trajectory discrimination is exploratory and evaluated on identical complete-trajectory events with equal-event weighting.",
            "major4": "Regional results are direct common-support contrasts with spatial-block intervals; calendar-month patterns remain exploratory.",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    atomic_write_json(report, output_dir / "reviewer_completion_report.json")
    manifest = {
        "analysis_version": report["analysis_version"],
        "artifacts": {},
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "reviewer_completion_manifest.json":
            manifest["artifacts"][str(path.relative_to(output_dir))] = {
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
    atomic_write_json(manifest, output_dir / "reviewer_completion_manifest.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = run_pipeline(args.config, args.main_root, args.output)
    print(
        json.dumps(
            {
                "analysis_version": report["analysis_version"],
                "matched_pre_event": report["pre_event_dynamic_control"]["matched_treatments"],
                "classification_rows": len(report["classification"]),
                "regional_rows": report["regional_contrast_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
