#!/usr/bin/env python3
"""Estimate standardized primary trajectories from annualized monthly matched effects."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "src"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
from pipeline.statistics import common_block_bootstrap, contrast_on_column


RUN_ROOT = PROJECT_ROOT / "data/primary"
OUT = RUN_ROOT / "event_level"
STAGE09 = RUN_ROOT / "outcomes"
STAGE08 = RUN_ROOT / "subannual"
SEED = 20260725
BLOCK_DEGREES = 2.0
REPS = 1000


def weighted_ess(weights: pd.Series) -> float:
    x = pd.to_numeric(weights, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(x.sum() ** 2 / np.square(x).sum()) if len(x) else float("nan")


def prepare_frame() -> pd.DataFrame:
    effects = pd.read_parquet(STAGE09 / "annual_matched_effects.parquet")
    support = pd.read_parquet(OUT / "common_support_weights.parquet")
    links = pd.read_parquet(STAGE08 / "matched_links.parquet", columns=[
        "treatment_grid_id", "event_year", "censor_relative_year"
    ])
    censor = links.drop_duplicates(["treatment_grid_id", "event_year"])
    if censor.duplicated(["treatment_grid_id", "event_year"]).any():
        raise ValueError("Current Stage 08 censoring map is not unique")

    frame = effects.merge(
        support.loc[:, [
            "treatment_grid_id", "event_year", "standardized_event_weight",
            "standardization_multiplier", "disturbance_share",
            "event_composition_purity", "baseline_undist_share",
        ]],
        on=["treatment_grid_id", "event_year"],
        how="inner",
        validate="many_to_one",
    ).merge(censor, on=["treatment_grid_id", "event_year"], how="left", validate="many_to_one")
    frame = frame.loc[
        frame["censor_relative_year"].isna()
        | (frame["relative_year"] < frame["censor_relative_year"])
    ].copy()
    frame = frame.rename(columns={"annual_matched_effect": "pair_effect_evi_mean"})
    frame["scope"] = frame["macroregion"].astype(str)
    frame["treatment_lon"] = pd.to_numeric(frame["treatment_lon"], errors="coerce")
    frame["treatment_lat"] = pd.to_numeric(frame["treatment_lat"], errors="coerce")
    frame = frame.loc[frame["pair_effect_evi_mean"].notna()].copy()
    if frame.duplicated(["treatment_grid_id", "event_year", "relative_year"]).any():
        raise ValueError("Current annual effects contain duplicate treatment-year rows")
    return frame


def write_event_window(frame: pd.DataFrame) -> None:
    columns = [
        "treatment_grid_id", "event_year", "event_type", "macroregion", "scope",
        "event_area_m2", "treatment_lon", "treatment_lat", "relative_year",
        "pair_effect_evi_mean",
        "disturbance_share", "event_composition_purity", "baseline_undist_share",
        "censor_relative_year",
    ]
    frame.loc[:, columns].to_parquet(OUT / "fixed_matched_event_window.parquet", index=False)


def add_common_metadata(summary: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["run_id"] = "expanded_control_pool_full_history_1990"
    summary["metric"] = "evi_mean"
    summary["weighting"] = "standardized_event_area"
    summary["bootstrap_requested"] = REPS
    summary["bootstrap_seed"] = SEED
    summary["block_degrees"] = BLOCK_DEGREES
    summary["valid_bootstrap"] = summary["bootstrap_valid"]
    return summary


def group_bootstrap(frame: pd.DataFrame, groups: tuple[str, ...]):
    bundle = common_block_bootstrap(
        frame,
        "pair_effect_evi_mean",
        groups,
        weight_column="standardized_event_weight",
        weighting="event_area",
        block_degrees=BLOCK_DEGREES,
        replicates=REPS,
        seed=SEED,
        minimum_valid_replicates=950,
    )
    return bundle


def write_global(frame: pd.DataFrame) -> pd.DataFrame:
    bundle = group_bootstrap(frame, ("event_type", "relative_year"))
    summary = add_common_metadata(bundle.summary, frame)
    direct = contrast_on_column(
        bundle, "event_type", "deforestation", "degradation",
        by_columns=("relative_year",), label="deforestation_minus_degradation"
    )
    direct["run_id"] = "expanded_control_pool_full_history_1990"
    direct["metric"] = "evi_mean"
    direct["weighting"] = "standardized_event_area"
    direct["bootstrap_requested"] = REPS
    direct["bootstrap_seed"] = SEED
    direct["block_degrees"] = BLOCK_DEGREES
    direct["valid_bootstrap"] = direct["bootstrap_valid"]
    direct["event_type"] = "deforestation_minus_degradation"
    block_counts = frame.assign(
        spatial_block=(
            np.floor((frame["treatment_lon"] + 180.0) / BLOCK_DEGREES).astype(int).astype(str)
            + ":"
            + np.floor((frame["treatment_lat"] + 90.0) / BLOCK_DEGREES).astype(int).astype(str)
        )
    ).groupby("relative_year")["spatial_block"].nunique()
    direct["spatial_block_count"] = direct["relative_year"].map(block_counts).astype(int)
    direct = direct.rename(columns={"contrast": "contrast_label"})
    group_out = summary.rename(columns={"event_type": "event_type"})
    cols = ["run_id", "event_type", "metric", "relative_year", "estimate", "ci_low", "ci_high", "bootstrap_sd", "bootstrap_requested", "valid_bootstrap", "spatial_block_count", "weighting", "block_degrees"]
    boot_out = pd.concat([
        group_out.loc[:, cols],
        direct.loc[:, cols],
    ], ignore_index=True).sort_values(["event_type", "relative_year"])
    boot_out.to_csv(OUT / "standardized_event_type_bootstrap.csv", index=False)

    support = pd.read_parquet(OUT / "common_support_weights.parquet")
    count = frame.groupby(["event_type", "relative_year"], as_index=False).agg(
        treatment_count=("treatment_grid_id", "nunique"),
        weight_sum=("standardized_event_weight", "sum"),
    )
    count["effective_sample_size"] = frame.groupby(["event_type", "relative_year"])["standardized_event_weight"].apply(weighted_ess).to_numpy()
    count = count.merge(
        summary.loc[:, ["event_type", "relative_year", "estimate"]],
        on=["event_type", "relative_year"], validate="one_to_one",
    )
    matched = pd.read_parquet(OUT / "matched_treatments.parquet")
    raw_area = matched.groupby("event_type", as_index=False)["event_area_m2"].sum().rename(columns={"event_area_m2": "raw_event_area"})
    retained = support.groupby("event_type", as_index=False).agg(
        common_support_strata=("standardization_stratum", "nunique"),
        retained_event_area=("event_area_m2", "sum"),
    ).merge(raw_area, on="event_type", validate="one_to_one")
    retained["retained_event_area_fraction"] = retained["retained_event_area"] / retained["raw_event_area"]
    count = count.merge(retained.loc[:, ["event_type", "common_support_strata", "retained_event_area_fraction"]], on="event_type", validate="many_to_one")
    direct_point = direct.loc[:, ["relative_year", "estimate"]].copy()
    direct_point["event_type"] = "deforestation_minus_degradation"
    direct_point["weight_sum"] = np.nan
    direct_point["effective_sample_size"] = np.nan
    direct_point["treatment_count"] = np.nan
    direct_point["common_support_strata"] = int(support["standardization_stratum"].nunique())
    direct_point["retained_event_area_fraction"] = np.nan
    out = pd.concat([count, direct_point], ignore_index=True)
    out["run_id"] = "expanded_control_pool_full_history_1990"
    out.to_csv(OUT / "standardized_event_type_contrast.csv", index=False)
    return boot_out


def write_regional(frame: pd.DataFrame) -> pd.DataFrame:
    bundle = group_bootstrap(frame, ("scope", "event_type", "relative_year"))
    group = add_common_metadata(bundle.summary, frame)
    direct = contrast_on_column(
        bundle, "event_type", "deforestation", "degradation",
        by_columns=("scope", "relative_year"), label="deforestation_minus_degradation"
    )
    group = group.rename(columns={"scope": "macroregion"})
    direct = direct.rename(columns={"scope": "macroregion", "contrast": "contrast_label"})
    direct["valid_bootstrap"] = direct["bootstrap_valid"]
    block_counts = frame.assign(
        spatial_block=(
            np.floor((frame["treatment_lon"] + 180.0) / BLOCK_DEGREES).astype(int).astype(str)
            + ":"
            + np.floor((frame["treatment_lat"] + 90.0) / BLOCK_DEGREES).astype(int).astype(str)
        )
    ).groupby(["scope", "relative_year"])["spatial_block"].nunique()
    direct["spatial_block_count"] = [
        int(block_counts.loc[(row["macroregion"], row["relative_year"])])
        for _, row in direct.iterrows()
    ]
    piv = group.pivot_table(index=["macroregion", "relative_year"], columns="event_type", values=["estimate", "n"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    out = direct.merge(piv, on=["macroregion", "relative_year"], how="left", validate="one_to_one")
    out = out.rename(columns={"estimate": "direct_contrast", "ci_low": "ci_low", "ci_high": "ci_high"})
    out["run_id"] = "expanded_control_pool_full_history_1990"
    out["degradation_estimate"] = out["estimate_degradation"]
    out["deforestation_estimate"] = out["estimate_deforestation"]
    out["degradation_n"] = out["n_degradation"]
    out["deforestation_n"] = out["n_deforestation"]
    out["standardization_strata"] = int(pd.read_parquet(OUT / "common_support_weights.parquet")["standardization_stratum"].nunique())
    out["estimand"] = "annualized monthly matched EVI; common-support standardized conditional association"
    cols = ["run_id", "macroregion", "relative_year", "degradation_estimate", "deforestation_estimate", "direct_contrast", "ci_low", "ci_high", "valid_bootstrap", "spatial_block_count", "degradation_n", "deforestation_n", "standardization_strata", "estimand"]
    out.loc[:, cols].sort_values(["macroregion", "relative_year"]).to_csv(OUT / "regional_event_type_contrast.csv", index=False)
    return out


def write_spatial_distribution(frame: pd.DataFrame) -> None:
    working = frame.copy()
    lon = working["treatment_lon"].to_numpy(float)
    lat = working["treatment_lat"].to_numpy(float)
    working["spatial_block"] = [f"{int(np.floor((x + 180) / 2))}:{int(np.floor((y + 90) / 2))}" for x, y in zip(lon, lat)]
    block = working.groupby(["scope", "relative_year", "spatial_block", "event_type"], as_index=False).apply(
        lambda g: pd.Series({"estimate": np.average(g["pair_effect_evi_mean"], weights=g["standardized_event_weight"]), "area": g["standardized_event_weight"].sum()}),
        include_groups=False,
    ).reset_index(drop=True)
    piv = block.pivot_table(index=["scope", "relative_year", "spatial_block"], columns="event_type", values=["estimate", "area"], aggfunc="first").dropna().reset_index()
    piv.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in piv.columns]
    piv["direct"] = piv["estimate_deforestation"] - piv["estimate_degradation"]
    rows = []
    for (scope, year), g in piv.groupby(["scope", "relative_year"], sort=True):
        values = g["direct"].to_numpy(float)
        weights = g["area_deforestation"].to_numpy(float) + g["area_degradation"].to_numpy(float)
        neg = (values < 0).astype(float)
        rows.append({
            "run_id": "expanded_control_pool_full_history_1990", "macroregion": scope, "relative_year": int(year),
            "spatial_block_count": int(len(g)), "q10": float(np.quantile(values, .10)), "q25": float(np.quantile(values, .25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, .75)), "q90": float(np.quantile(values, .90)),
            "weighted_negative_fraction": float(np.average(neg, weights=weights)),
            "negative_fraction_ci_low": np.nan, "negative_fraction_ci_high": np.nan,
            "valid_bootstrap": 0, "estimand": "primary block-level direct contrast",
        })
    pd.DataFrame(rows).to_csv(OUT / "spatial_block_distribution.csv", index=False)


def write_capped(frame: pd.DataFrame) -> None:
    support = pd.read_parquet(OUT / "common_support_weights.parquet")
    support["multiplier"] = support["standardized_event_weight"] / support["event_area_m2"]
    caps = [10.0, 20.0, float(support["multiplier"].quantile(.99))]
    rows = []
    for cap in caps:
        f = frame.copy()
        f["capped_weight"] = f["event_area_m2"] * np.minimum(f["standardization_multiplier"], cap)
        b = common_block_bootstrap(f, "pair_effect_evi_mean", ("event_type", "relative_year"), weight_column="capped_weight", weighting="event_area", block_degrees=BLOCK_DEGREES, replicates=REPS, seed=SEED, minimum_valid_replicates=950)
        d = contrast_on_column(b, "event_type", "deforestation", "degradation", by_columns=("relative_year",), label="deforestation_minus_degradation")
        s = b.summary
        for r in s.to_dict("records"):
            rows.append({"run_id": "expanded_control_pool_full_history_1990", "cap": cap, "event_type": r["event_type"], "relative_year": r["relative_year"], "estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "valid_replicates": r["bootstrap_valid"], "effective_sample_size": weighted_ess(f.loc[(f.event_type == r['event_type']) & (f.relative_year == r['relative_year']), 'capped_weight'])})
        for r in d.to_dict("records"):
            rows.append({"run_id": "expanded_control_pool_full_history_1990", "cap": cap, "event_type": "deforestation_minus_degradation", "relative_year": r["relative_year"], "estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "valid_replicates": r["bootstrap_valid"], "effective_sample_size": np.nan})
    pd.DataFrame(rows).sort_values(["cap", "relative_year", "event_type"]).to_csv(OUT / "capped_weight_contrasts.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame()
    write_event_window(frame)
    write_global(frame)
    write_regional(frame)
    write_spatial_distribution(frame)
    write_capped(frame)
    summary = {
        "run_id": "expanded_control_pool_full_history_1990",
        "source": "outcomes/annual_matched_effects.parquet",
        "matching_source": "event_level/common_support_weights.parquet",
        "rows_after_common_support_and_dynamic_censoring": int(len(frame)),
        "event_counts": frame.drop_duplicates(["treatment_grid_id", "event_year", "event_type"]).groupby("event_type").size().astype(int).to_dict(),
        "bootstrap": {"replicates": REPS, "seed": SEED, "block_degrees": BLOCK_DEGREES},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
