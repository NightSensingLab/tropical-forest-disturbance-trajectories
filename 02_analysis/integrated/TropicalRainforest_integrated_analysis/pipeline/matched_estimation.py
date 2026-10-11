from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.matched_control import MATCHING_COVARIATES, MatchedControlSettings


EVI_METRICS = ("evi_mean", "evi_min", "evi_max", "evi_amp", "evi_sd")
ANNUAL_COLUMNS = (
    "grid_id",
    "year",
    "evi_mean",
    "evi_min",
    "evi_max",
    "evi_sd",
    "valid_obs",
)


def apply_dynamic_censoring(window):
    """Mask matched effects at and after an observed repeat disturbance."""
    required = {"relative_year", "censor_relative_year"}
    missing = required - set(window.columns)
    if missing:
        raise ValueError(f"Matched window lacks censoring columns: {sorted(missing)}")
    result = window.copy()
    relative_year = pd.to_numeric(result["relative_year"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    censor_year = pd.to_numeric(
        result["censor_relative_year"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    dynamic_censored = (
        np.isfinite(relative_year)
        & np.isfinite(censor_year)
        & (relative_year >= censor_year)
    )
    result["dynamic_censored"] = dynamic_censored
    for column in result.columns:
        if column.startswith("pair_effect_"):
            result.loc[dynamic_censored, column] = np.nan
    return result


def _pooled_quantile_bins(values, n_bins):
    values = pd.to_numeric(values, errors="coerce")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least two.")
    finite = values.notna()
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    if finite.any():
        clean = values.loc[finite]
        if clean.nunique(dropna=True) == 1:
            result.loc[finite] = 0
        else:
            result.loc[finite] = pd.qcut(
                clean,
                q=min(int(n_bins), int(clean.nunique(dropna=True))),
                labels=False,
                duplicates="drop",
            ).astype("int64")
    return result


def _derive_event_composition_purity(events):
    """Return the dominant D/F share conditional on total D+F disturbance."""
    if "event_composition_purity" in events.columns:
        purity = pd.to_numeric(events["event_composition_purity"], errors="coerce")
    else:
        required = {"degradation_share", "deforestation_share"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(
                "Event table lacks event composition fields needed for common-support "
                f"standardization: {sorted(missing)}"
            )
        degradation = pd.to_numeric(events["degradation_share"], errors="coerce")
        deforestation = pd.to_numeric(events["deforestation_share"], errors="coerce")
        total = degradation + deforestation
        purity = pd.Series(
            np.divide(
                np.maximum(degradation.to_numpy(dtype=np.float64), deforestation.to_numpy(dtype=np.float64)),
                total.to_numpy(dtype=np.float64),
                out=np.full(len(events), np.nan, dtype=np.float64),
                where=total.to_numpy(dtype=np.float64) > 0.0,
            ),
            index=events.index,
            name="event_composition_purity",
        )
    return purity


def common_support_weights(events, n_bins=5):
    """Standardize event types to shared intensity, composition, and baseline support."""
    required = {
        "treatment_grid_id",
        "event_year",
        "event_type",
        "macroregion",
        "event_area_m2",
        "disturbance_share",
        "baseline_undist_share",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Event table lacks standardization columns: {sorted(missing)}")
    working = events.loc[:, sorted(required)].copy()
    if working.duplicated(["treatment_grid_id", "event_year"]).any():
        raise ValueError("Event table must contain one row per treatment event.")
    if not set(working["event_type"].dropna()).issubset(
        {"degradation", "deforestation"}
    ):
        raise ValueError("Event table contains an invalid event type.")
    working["event_area_m2"] = pd.to_numeric(
        working["event_area_m2"], errors="coerce"
    )
    working["disturbance_share"] = pd.to_numeric(
        working["disturbance_share"], errors="coerce"
    )
    working["baseline_undist_share"] = pd.to_numeric(
        working["baseline_undist_share"], errors="coerce"
    )
    working["event_composition_purity"] = _derive_event_composition_purity(events).reindex(
        working.index
    ).to_numpy(dtype=np.float64)
    valid = (
        working["event_area_m2"].gt(0.0)
        & np.isfinite(working["event_area_m2"])
        & np.isfinite(working["disturbance_share"])
        & np.isfinite(working["baseline_undist_share"])
        & np.isfinite(working["event_composition_purity"])
        & working["event_composition_purity"].between(0.5, 1.0)
    )
    working = working.loc[valid].copy()
    if working.empty:
        raise ValueError("No finite events remain for common-support standardization.")

    working["intensity_bin"] = _pooled_quantile_bins(
        working["disturbance_share"], n_bins
    )
    working["baseline_forest_bin"] = _pooled_quantile_bins(
        working["baseline_undist_share"], n_bins
    )
    working["composition_purity_bin"] = _pooled_quantile_bins(
        working["event_composition_purity"], n_bins
    )
    working = working.dropna(
        subset=["intensity_bin", "baseline_forest_bin", "composition_purity_bin"]
    ).copy()
    working["intensity_bin"] = working["intensity_bin"].astype(int)
    working["baseline_forest_bin"] = working["baseline_forest_bin"].astype(int)
    working["composition_purity_bin"] = working["composition_purity_bin"].astype(int)
    working["standardization_stratum"] = (
        working["macroregion"].astype(str)
        + "|year="
        + working["event_year"].astype(str)
        + "|i="
        + working["intensity_bin"].astype(str)
        + "|f="
        + working["baseline_forest_bin"].astype(str)
        + "|p="
        + working["composition_purity_bin"].astype(str)
    )
    presence = working.groupby("standardization_stratum")["event_type"].nunique()
    common = set(presence.loc[presence.eq(2)].index)
    working["common_support"] = working["standardization_stratum"].isin(common)
    supported = working.loc[working["common_support"]].copy()
    if supported.empty:
        raise ValueError("No common-support strata contain both event types.")

    finite_area_totals = working.groupby("event_type")["event_area_m2"].sum()
    base_totals = supported.groupby("event_type")["event_area_m2"].sum()
    stratum_type = supported.groupby(
        ["standardization_stratum", "event_type"], sort=True
    )["event_area_m2"].sum()
    type_share = stratum_type / base_totals
    target_share = type_share.unstack("event_type").mean(axis=1)
    multipliers = (target_share / type_share).rename("standardization_multiplier")
    supported = supported.join(multipliers, on=["standardization_stratum", "event_type"])
    supported["standardized_event_weight"] = (
        supported["event_area_m2"] * supported["standardization_multiplier"]
    )

    diagnostics = {
        "total_events": int(len(events)),
        "finite_events": int(len(working)),
        "retained_events": int(len(supported)),
        "total_strata": int(len(presence)),
        "common_support_strata": int(len(common)),
        "retained_treatments_by_type": {
            str(key): int(value)
            for key, value in supported.groupby("event_type").size().items()
        },
        "retained_event_area_fraction_by_type": {
            str(event_type): float(
                supported.loc[supported["event_type"].eq(event_type), "event_area_m2"].sum()
                / finite_area_totals[event_type]
            )
            for event_type in base_totals.index
        },
    }
    return supported.reset_index(drop=True), diagnostics


def _standardization_weighted_stats(values, weights):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    weights = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan, 0.0, 0.0, 0
    values = values[valid]
    weights = weights[valid]
    total = weights.sum()
    return (
        float(np.sum(values * weights) / total),
        float(total),
        float(total**2 / np.sum(weights**2)),
        int(valid.sum()),
    )


def summarize_standardized_event_type_contrast(
    window, events, metric="evi_mean", relative_years=None
):
    effect_column = f"pair_effect_{metric}"
    required = {"treatment_grid_id", "event_year", "event_type", "relative_year", effect_column}
    missing = required - set(window.columns)
    if missing:
        raise ValueError(f"Matched window lacks standardized contrast columns: {sorted(missing)}")
    weights, diagnostics = common_support_weights(events)
    weight_columns = [
        "treatment_grid_id",
        "event_year",
        "standardization_stratum",
        "standardized_event_weight",
    ]
    working = window.merge(
        weights.loc[:, weight_columns],
        on=["treatment_grid_id", "event_year"],
        how="inner",
        validate="many_to_one",
    )
    if relative_years is not None:
        working = working.loc[working["relative_year"].isin(relative_years)].copy()
    rows = []
    for relative_year, year_frame in working.groupby("relative_year", sort=True):
        estimates = {}
        for event_type, group in year_frame.groupby("event_type", sort=True):
            estimate, weight_sum, ess, n = _standardization_weighted_stats(
                group[effect_column], group["standardized_event_weight"]
            )
            estimates[str(event_type)] = estimate
            rows.append(
                {
                    "event_type": str(event_type),
                    "relative_year": int(relative_year),
                    "estimate": estimate,
                    "weight_sum": weight_sum,
                    "effective_sample_size": ess,
                    "treatment_count": int(
                        group.loc[group[effect_column].notna(), "treatment_grid_id"].nunique()
                    ),
                    "common_support_strata": int(group["standardization_stratum"].nunique()),
                    "retained_event_area_fraction": diagnostics[
                        "retained_event_area_fraction_by_type"
                    ].get(str(event_type), np.nan),
                }
            )
        if {"degradation", "deforestation"}.issubset(estimates):
            rows.append(
                {
                    "event_type": "deforestation_minus_degradation",
                    "relative_year": int(relative_year),
                    "estimate": estimates["deforestation"] - estimates["degradation"],
                    "weight_sum": np.nan,
                    "effective_sample_size": np.nan,
                    "treatment_count": np.nan,
                    "common_support_strata": int(
                        year_frame["standardization_stratum"].nunique()
                    ),
                    "retained_event_area_fraction": np.nan,
                }
            )
    return pd.DataFrame(rows), diagnostics, weights


def bootstrap_standardized_event_type_contrast(
    window,
    events,
    metric="evi_mean",
    relative_years=None,
    n_bins=5,
    block_degrees=2.0,
    replicates=1000,
    seed=20260725,
):
    """Bootstrap the standardized event-type contrast using spatial blocks.

    Treatment-level standardization weights are held fixed while the spatial
    blocks are resampled jointly for both event types. This preserves the
    paired nature of the direct contrast and makes the valid-replicate count
    explicit when dynamic censoring removes all observations in a block/year.
    """
    effect_column = f"pair_effect_{metric}"
    required = {
        "treatment_grid_id",
        "event_year",
        "event_type",
        "relative_year",
        effect_column,
    }
    missing = required - set(window.columns)
    if missing:
        raise ValueError(f"Matched window lacks bootstrap columns: {sorted(missing)}")
    replicates = int(replicates)
    if replicates < 2:
        raise ValueError("replicates must be at least two.")
    block_degrees = float(block_degrees)
    if block_degrees <= 0.0:
        raise ValueError("block_degrees must be positive.")

    weights, diagnostics = common_support_weights(events, n_bins=n_bins)
    weight_columns = [
        "treatment_grid_id",
        "event_year",
        "standardization_stratum",
        "standardized_event_weight",
    ]
    working = window.merge(
        weights.loc[:, weight_columns],
        on=["treatment_grid_id", "event_year"],
        how="inner",
        validate="many_to_one",
    ).copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "event_type",
                "metric",
                "relative_year",
                "estimate",
                "ci_low",
                "ci_high",
                "bootstrap_sd",
                "requested_replicates",
                "valid_replicates",
                "spatial_block_count",
                "weighting",
                "block_degrees",
            ]
        )

    lon_column = "lon" if "lon" in working.columns else "treatment_lon"
    lat_column = "lat" if "lat" in working.columns else "treatment_lat"
    if lon_column not in working.columns or lat_column not in working.columns:
        event_coordinates = events.loc[
            :, ["treatment_grid_id", "event_year", "lon", "lat"]
        ] if {"treatment_grid_id", "event_year", "lon", "lat"}.issubset(events.columns) else None
        if event_coordinates is None:
            raise ValueError(
                "Bootstrap requires lon/lat or treatment_lon/treatment_lat coordinates."
            )
        working = working.merge(
            event_coordinates.drop_duplicates(["treatment_grid_id", "event_year"]),
            on=["treatment_grid_id", "event_year"],
            how="left",
            validate="many_to_one",
        )
        lon_column, lat_column = "lon", "lat"

    event_keys = ["treatment_grid_id", "event_year"]
    event_rows = working.loc[:, event_keys + [
        "event_type",
        lon_column,
        lat_column,
        "standardization_stratum",
        "standardized_event_weight",
    ]].drop_duplicates(event_keys)
    event_rows = event_rows.rename(columns={lon_column: "lon", lat_column: "lat"})
    event_rows["spatial_block"] = assign_spatial_blocks(event_rows, block_degrees)
    blocks = np.sort(event_rows["spatial_block"].unique())
    if relative_years is None:
        relative_years = sorted(pd.to_numeric(working["relative_year"], errors="coerce").dropna().unique())
    else:
        relative_years = [int(value) for value in relative_years]
    rng = np.random.default_rng(int(seed))
    block_counts = rng.multinomial(
        len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=replicates
    )
    rows = []
    for relative_year in relative_years:
        year_frame = working.loc[working["relative_year"].eq(relative_year)].copy()
        year_frame = year_frame.merge(
            event_rows.loc[:, event_keys + ["spatial_block"]],
            on=event_keys,
            how="inner",
            validate="many_to_one",
        )
        type_estimates = {}
        type_bootstrap = {}
        for event_type in ("degradation", "deforestation"):
            group = year_frame.loc[year_frame["event_type"].eq(event_type)]
            effects = pd.to_numeric(group[effect_column], errors="coerce").to_numpy(dtype=np.float64)
            weights_array = pd.to_numeric(
                group["standardized_event_weight"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            valid = np.isfinite(effects) & np.isfinite(weights_array) & (weights_array > 0.0)
            block_frame = pd.DataFrame(
                {
                    "spatial_block": group.loc[valid, "spatial_block"].to_numpy(),
                    "weighted_effect": effects[valid] * weights_array[valid],
                    "weight": weights_array[valid],
                }
            )
            if block_frame.empty:
                numerator_by_block = np.zeros(len(blocks), dtype=np.float64)
                denominator_by_block = np.zeros(len(blocks), dtype=np.float64)
            else:
                block_sums = block_frame.groupby("spatial_block", sort=True).sum().reindex(
                    blocks, fill_value=0.0
                )
                numerator_by_block = block_sums["weighted_effect"].to_numpy(dtype=np.float64)
                denominator_by_block = block_sums["weight"].to_numpy(dtype=np.float64)
            numerators = block_counts @ numerator_by_block
            denominators = block_counts @ denominator_by_block
            estimates = np.divide(
                numerators,
                denominators,
                out=np.full(replicates, np.nan, dtype=np.float64),
                where=denominators > 0.0,
            )
            point = weighted_mean_and_ess(effects[valid], weights_array[valid])["mean"] if valid.any() else np.nan
            type_estimates[event_type] = point
            type_bootstrap[event_type] = estimates
            finite = estimates[np.isfinite(estimates)]
            rows.append(
                {
                    "event_type": event_type,
                    "metric": metric,
                    "relative_year": int(relative_year),
                    "estimate": float(point),
                    "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "bootstrap_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "requested_replicates": replicates,
                    "valid_replicates": int(len(finite)),
                    "spatial_block_count": int(len(blocks)),
                    "weighting": "standardized_event_area",
                    "block_degrees": block_degrees,
                }
            )
        if all(event_type in type_estimates for event_type in ("degradation", "deforestation")):
            contrast = type_estimates["deforestation"] - type_estimates["degradation"]
            contrast_draws = type_bootstrap["deforestation"] - type_bootstrap["degradation"]
            finite = contrast_draws[np.isfinite(contrast_draws)]
            rows.append(
                {
                    "event_type": "deforestation_minus_degradation",
                    "metric": metric,
                    "relative_year": int(relative_year),
                    "estimate": float(contrast) if np.isfinite(contrast) else np.nan,
                    "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "bootstrap_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "requested_replicates": replicates,
                    "valid_replicates": int(len(finite)),
                    "spatial_block_count": int(len(blocks)),
                    "weighting": "standardized_event_area",
                    "block_degrees": block_degrees,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["event_type", "metric", "relative_year"], kind="mergesort"
    ).reset_index(drop=True)


@dataclass(frozen=True)
class SensitivityScenario:
    family: str
    event_threshold: float
    obs_threshold: int
    radius_km: float
    max_controls: int
    weighting: str
    stable_through_2023: bool

    @property
    def label(self):
        horizon = "through2023" if self.stable_through_2023 else "through_t5"
        return (
            f"{self.family}_event{int(round(self.event_threshold * 100)):02d}_"
            f"obs{int(self.obs_threshold):02d}_radius{int(round(self.radius_km)):03d}_"
            f"controls{int(self.max_controls)}_{self.weighting}_{horizon}"
        )


def metric_baselines(annual, grid_event_keys, metric, obs_threshold):
    if metric not in EVI_METRICS:
        raise ValueError(f"Unsupported EVI metric: {metric}")
    missing = set(ANNUAL_COLUMNS) - set(annual.columns)
    if missing:
        raise ValueError(f"Annual outcome table lacks columns: {sorted(missing)}")
    missing_keys = {"grid_id", "event_year"} - set(grid_event_keys.columns)
    if missing_keys:
        raise ValueError(f"Grid-event keys lack columns: {sorted(missing_keys)}")
    keys = grid_event_keys.loc[:, ["grid_id", "event_year"]].drop_duplicates()
    if keys.empty:
        return pd.DataFrame(
            columns=[
                "grid_id",
                "event_year",
                "relative_year",
                "value",
                "baseline",
                "baseline_n",
                "delta",
                "usable",
            ]
        )
    source = annual.loc[:, ANNUAL_COLUMNS].copy()
    source["evi_amp"] = (
        pd.to_numeric(source["evi_max"], errors="coerce")
        - pd.to_numeric(source["evi_min"], errors="coerce")
    )
    window = keys.merge(source, on="grid_id", how="left", validate="many_to_many")
    window["relative_year"] = window["year"] - window["event_year"]
    window = window.loc[window["relative_year"].between(-3, 5)].copy()
    value = pd.to_numeric(window[metric], errors="coerce")
    valid_obs = pd.to_numeric(window["valid_obs"], errors="coerce")
    window["usable"] = value.notna() & valid_obs.ge(int(obs_threshold))
    baseline_rows = window.loc[
        window["relative_year"].between(-3, -1) & window["usable"]
    ]
    grouped = baseline_rows.groupby(["grid_id", "event_year"], sort=False)[metric]
    baseline_count = grouped.count()
    baseline = grouped.mean().where(baseline_count >= 2)
    key_index = pd.MultiIndex.from_frame(window[["grid_id", "event_year"]])
    window["baseline"] = baseline.reindex(key_index).to_numpy()
    window["baseline_n"] = (
        baseline_count.reindex(key_index).fillna(0).to_numpy(dtype=np.int16)
    )
    window["value"] = value.where(window["usable"])
    window["delta"] = (window["value"] - window["baseline"]).where(
        window["usable"] & window["baseline"].notna()
    )
    return window.loc[
        :,
        [
            "grid_id",
            "event_year",
            "relative_year",
            "value",
            "baseline",
            "baseline_n",
            "delta",
            "usable",
        ],
    ].sort_values(["grid_id", "event_year", "relative_year"], kind="mergesort")


def _expand_treatment_window(treatments, pairs, relative_years):
    matched_ids = pairs.loc[:, ["treatment_grid_id", "event_year"]].drop_duplicates()
    treatment_source = treatments.rename(columns={"grid_id": "treatment_grid_id"})
    selected = matched_ids.merge(
        treatment_source,
        on=["treatment_grid_id", "event_year"],
        how="left",
        validate="one_to_one",
    )
    if selected["event_type"].isna().any():
        raise ValueError("Matched treatment identity is absent from the treatment table.")
    repeated = selected.loc[selected.index.repeat(len(relative_years))].copy()
    repeated["relative_year"] = np.tile(np.asarray(relative_years), len(selected))
    return repeated.reset_index(drop=True)


def _aggregate_control_metric(control_metric, pairs):
    linked = pairs.loc[
        :,
        [
            "treatment_grid_id",
            "control_grid_id",
            "event_year",
            "within_set_weight",
        ],
    ].merge(
        control_metric.rename(columns={"grid_id": "control_grid_id"}),
        on=["control_grid_id", "event_year"],
        how="left",
        validate="many_to_many",
    )
    linked["available_weight"] = np.where(
        np.isfinite(pd.to_numeric(linked["delta"], errors="coerce")),
        pd.to_numeric(linked["within_set_weight"], errors="coerce"),
        0.0,
    )
    for column in ("value", "baseline", "delta"):
        linked[f"weighted_{column}"] = np.where(
            linked["available_weight"].gt(0.0),
            pd.to_numeric(linked[column], errors="coerce") * linked["available_weight"],
            0.0,
        )
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


def build_matched_event_window(annual, treatments, pairs, settings=None, obs_threshold=None):
    settings = settings or MatchedControlSettings()
    obs_threshold = int(obs_threshold or settings.primary_obs_threshold)
    if pairs.empty:
        return pd.DataFrame()
    result = _expand_treatment_window(treatments, pairs, settings.relative_years)
    treatment_keys = result.loc[:, ["treatment_grid_id", "event_year"]].rename(
        columns={"treatment_grid_id": "grid_id"}
    )
    control_keys = pairs.loc[:, ["control_grid_id", "event_year"]].rename(
        columns={"control_grid_id": "grid_id"}
    )

    for metric in EVI_METRICS:
        treatment_metric = metric_baselines(
            annual, treatment_keys, metric, obs_threshold
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
        treatment_columns = [
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
            treatment_metric.loc[:, treatment_columns],
            on=["treatment_grid_id", "event_year", "relative_year"],
            how="left",
            validate="one_to_one",
        )

        control_metric = metric_baselines(annual, control_keys, metric, obs_threshold)
        control_aggregate = _aggregate_control_metric(control_metric, pairs).rename(
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
            control_aggregate.loc[:, control_columns],
            on=["treatment_grid_id", "event_year", "relative_year"],
            how="left",
            validate="one_to_one",
        )
        treated_delta = pd.to_numeric(
            result[f"treated_delta_{metric}"], errors="coerce"
        )
        control_delta = pd.to_numeric(
            result[f"control_delta_{metric}"], errors="coerce"
        )
        result[f"pair_effect_{metric}"] = treated_delta - control_delta

    if "censor_relative_year" not in result.columns:
        result["censor_relative_year"] = np.nan
    return apply_dynamic_censoring(
        result.sort_values(
            ["event_year", "treatment_grid_id", "relative_year"], kind="mergesort"
        ).reset_index(drop=True)
    )


def weighted_mean_and_ess(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return {"mean": np.nan, "weight_sum": 0.0, "effective_sample_size": 0.0}
    values = values[valid]
    weights = weights[valid]
    weight_sum = float(weights.sum())
    return {
        "mean": float(np.sum(values * weights) / weight_sum),
        "weight_sum": weight_sum,
        "effective_sample_size": float(weight_sum**2 / np.sum(weights**2)),
    }


def summarize_matched_response(window, weighting="event_area", include_macroregions=True):
    if weighting not in {"event_area", "equal_grid"}:
        raise ValueError("weighting must be event_area or equal_grid.")
    required = {
        "treatment_grid_id",
        "event_type",
        "macroregion",
        "relative_year",
        "event_area_m2",
    }
    missing = required - set(window.columns)
    if missing:
        raise ValueError(f"Matched window lacks columns: {sorted(missing)}")
    metrics = [
        column.removeprefix("pair_effect_")
        for column in window.columns
        if column.startswith("pair_effect_")
    ]
    rows = []
    scopes = [("Global", window)]
    if include_macroregions:
        scopes.extend((str(region), group) for region, group in window.groupby("macroregion"))
    for scope, scope_frame in scopes:
        for (event_type, relative_year), group in scope_frame.groupby(
            ["event_type", "relative_year"], sort=True
        ):
            base_weights = (
                pd.to_numeric(group["event_area_m2"], errors="coerce").to_numpy()
                if weighting == "event_area"
                else np.ones(len(group), dtype=np.float64)
            )
            for metric in metrics:
                effects = pd.to_numeric(
                    group[f"pair_effect_{metric}"], errors="coerce"
                ).to_numpy()
                treated = pd.to_numeric(
                    group[f"treated_delta_{metric}"], errors="coerce"
                ).to_numpy()
                controls = pd.to_numeric(
                    group[f"control_delta_{metric}"], errors="coerce"
                ).to_numpy()
                valid = np.isfinite(effects) & np.isfinite(base_weights) & (base_weights > 0.0)
                effect_stats = weighted_mean_and_ess(effects[valid], base_weights[valid])
                treated_stats = weighted_mean_and_ess(treated[valid], base_weights[valid])
                control_stats = weighted_mean_and_ess(controls[valid], base_weights[valid])
                rows.append(
                    {
                        "scope": scope,
                        "event_type": event_type,
                        "metric": metric,
                        "relative_year": int(relative_year),
                        "weighting": weighting,
                        "treatment_count": int(
                            group.loc[valid, "treatment_grid_id"].nunique()
                        ),
                        "weight_sum": effect_stats["weight_sum"],
                        "observed_event_area_m2": float(
                            pd.to_numeric(
                                group.loc[valid, "event_area_m2"], errors="coerce"
                            ).sum()
                        ),
                        "effective_sample_size": effect_stats["effective_sample_size"],
                        "treated_change": treated_stats["mean"],
                        "control_change": control_stats["mean"],
                        "estimate": effect_stats["mean"],
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["scope", "event_type", "metric", "relative_year"], kind="mergesort"
    ).reset_index(drop=True)


def _weighted_mean_variance(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan, np.nan
    values = values[valid]
    weights = weights[valid]
    total = weights.sum()
    mean = np.sum(values * weights) / total
    variance = np.sum(weights * (values - mean) ** 2) / total
    return float(mean), float(variance)


def _standardized_difference(treated_mean, control_mean, treated_var, control_var):
    denominator = np.sqrt((treated_var + control_var) / 2.0)
    if np.isfinite(denominator) and denominator > 0.0:
        return float((treated_mean - control_mean) / denominator)
    return 0.0 if np.isclose(treated_mean, control_mean, equal_nan=False) else np.inf


def compute_balance(
    treatments,
    controls,
    pairs,
    covariates=MATCHING_COVARIATES,
):
    covariates = tuple(covariates)
    required_treatments = {"grid_id", "event_year", "event_type", *covariates}
    required_controls = {"grid_id", "event_year", *covariates}
    if required_treatments - set(treatments.columns):
        raise ValueError("Treatment feature table is incomplete.")
    if required_controls - set(controls.columns):
        raise ValueError("Control feature table is incomplete.")
    scopes = [("Overall", None)]
    scopes.extend((event_type, event_type) for event_type in sorted(treatments.event_type.unique()))
    rows = []
    for scope, event_type in scopes:
        treated = treatments if event_type is None else treatments.loc[treatments.event_type.eq(event_type)]
        linked = pairs if event_type is None else pairs.loc[pairs.event_type.eq(event_type)]
        if treated.empty or linked.empty:
            continue
        event_years = set(treated.event_year)
        pre_controls = controls.loc[controls.event_year.isin(event_years)]
        post_controls = linked.merge(
            controls,
            left_on=["control_grid_id", "event_year"],
            right_on=["grid_id", "event_year"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_control"),
        )
        control_weights = post_controls.groupby(
            ["control_grid_id", "event_year"], sort=False
        )["within_set_weight"].transform("sum")
        unique_post = post_controls.assign(_weight=control_weights).drop_duplicates(
            ["control_grid_id", "event_year"]
        )
        effective_control_n = weighted_mean_and_ess(
            np.ones(len(unique_post)), unique_post["_weight"]
        )["effective_sample_size"]
        for covariate in covariates:
            treated_values = pd.to_numeric(treated[covariate], errors="coerce").to_numpy()
            pre_values = pd.to_numeric(pre_controls[covariate], errors="coerce").to_numpy()
            post_values = pd.to_numeric(unique_post[covariate], errors="coerce").to_numpy()
            treated_mean, treated_var = _weighted_mean_variance(
                treated_values, np.ones(len(treated_values))
            )
            pre_mean, pre_var = _weighted_mean_variance(
                pre_values, np.ones(len(pre_values))
            )
            post_mean, post_var = _weighted_mean_variance(
                post_values, unique_post["_weight"].to_numpy(dtype=np.float64)
            )
            pre_smd = _standardized_difference(
                treated_mean, pre_mean, treated_var, pre_var
            )
            post_smd = _standardized_difference(
                treated_mean, post_mean, treated_var, pre_var
            )
            if np.isclose(treated_var, 0.0) and np.isclose(post_var, 0.0):
                variance_ratio = 1.0
            elif post_var > 0.0:
                variance_ratio = treated_var / post_var
            else:
                variance_ratio = np.inf
            rows.append(
                {
                    "scope": scope,
                    "covariate": covariate,
                    "treated_mean": treated_mean,
                    "pre_control_mean": pre_mean,
                    "post_control_mean": post_mean,
                    "pre_smd": pre_smd,
                    "post_smd": post_smd,
                    "post_variance_ratio": variance_ratio,
                    "treated_n": int(len(treated)),
                    "pre_control_n": int(len(pre_controls)),
                    "matched_control_n": int(len(unique_post)),
                    "effective_control_n": effective_control_n,
                }
            )
    return pd.DataFrame(rows).sort_values(["scope", "covariate"]).reset_index(drop=True)


def balance_gate(balance, threshold=0.10):
    if balance.empty or "post_smd" not in balance:
        return False
    values = pd.to_numeric(balance["post_smd"], errors="coerce").to_numpy()
    return bool(np.isfinite(values).all() and (np.abs(values) <= float(threshold) + 1e-12).all())


def assign_spatial_blocks(frame, block_degrees=2.0):
    block_degrees = float(block_degrees)
    if block_degrees <= 0.0:
        raise ValueError("block_degrees must be positive.")
    lon_column = "lon" if "lon" in frame.columns else "treatment_lon"
    lat_column = "lat" if "lat" in frame.columns else "treatment_lat"
    missing = {lon_column, lat_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Matched window lacks spatial columns: {sorted(missing)}")
    lon = pd.to_numeric(frame[lon_column], errors="coerce").to_numpy(dtype=np.float64)
    lat = pd.to_numeric(frame[lat_column], errors="coerce").to_numpy(dtype=np.float64)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Spatial bootstrap coordinates must be finite.")
    bx = np.floor((lon + 180.0) / block_degrees).astype(np.int32)
    by = np.floor((lat + 90.0) / block_degrees).astype(np.int32)
    return pd.Series(
        [f"{x}:{y}" for x, y in zip(bx, by)],
        index=frame.index,
        name="spatial_block",
    )


def spatial_block_bootstrap(
    window,
    block_degrees=2.0,
    replicates=1000,
    seed=20260725,
    weighting="event_area",
):
    if weighting not in {"event_area", "equal_grid"}:
        raise ValueError("weighting must be event_area or equal_grid.")
    replicates = int(replicates)
    if replicates < 2:
        raise ValueError("replicates must be at least two.")
    working = window.copy()
    working["spatial_block"] = assign_spatial_blocks(working, block_degrees)
    blocks = np.sort(working["spatial_block"].unique())
    if not len(blocks):
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    bootstrap_counts = rng.multinomial(
        len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=replicates
    )
    metrics = [
        column.removeprefix("pair_effect_")
        for column in working.columns
        if column.startswith("pair_effect_")
    ]
    rows = []
    for (event_type, relative_year), group in working.groupby(
        ["event_type", "relative_year"], sort=True
    ):
        base_weights = (
            pd.to_numeric(group["event_area_m2"], errors="coerce").to_numpy(dtype=np.float64)
            if weighting == "event_area"
            else np.ones(len(group), dtype=np.float64)
        )
        for metric in metrics:
            effects = pd.to_numeric(
                group[f"pair_effect_{metric}"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            valid = np.isfinite(effects) & np.isfinite(base_weights) & (base_weights > 0.0)
            point = weighted_mean_and_ess(effects[valid], base_weights[valid])["mean"]
            block_frame = pd.DataFrame(
                {
                    "spatial_block": group["spatial_block"].to_numpy()[valid],
                    "weighted_effect": effects[valid] * base_weights[valid],
                    "weight": base_weights[valid],
                }
            )
            block_sums = block_frame.groupby("spatial_block", sort=True).sum().reindex(
                blocks, fill_value=0.0
            )
            numerators = bootstrap_counts @ block_sums["weighted_effect"].to_numpy()
            denominators = bootstrap_counts @ block_sums["weight"].to_numpy()
            estimates = np.divide(
                numerators,
                denominators,
                out=np.full(replicates, np.nan, dtype=np.float64),
                where=denominators > 0.0,
            )
            finite = estimates[np.isfinite(estimates)]
            rows.append(
                {
                    "event_type": event_type,
                    "metric": metric,
                    "relative_year": int(relative_year),
                    "weighting": weighting,
                    "block_degrees": float(block_degrees),
                    "replicates": replicates,
                    "estimate": point,
                    "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "bootstrap_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "spatial_block_count": int(len(blocks)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["event_type", "metric", "relative_year"], kind="mergesort"
    ).reset_index(drop=True)


def primary_sensitivity_scenarios(settings=None):
    settings = settings or MatchedControlSettings()
    scenarios = []
    for event_threshold in settings.event_thresholds:
        for obs_threshold in settings.obs_thresholds:
            scenarios.append(
                SensitivityScenario(
                    "threshold_grid",
                    event_threshold,
                    obs_threshold,
                    settings.primary_radius_km,
                    settings.max_controls,
                    "event_area",
                    False,
                )
            )
    for radius in (25.0, settings.primary_radius_km, settings.fallback_radius_km):
        scenarios.append(
            SensitivityScenario(
                "radius",
                settings.primary_event_threshold,
                settings.primary_obs_threshold,
                radius,
                settings.max_controls,
                "event_area",
                False,
            )
        )
    for max_controls in (1, settings.max_controls):
        scenarios.append(
            SensitivityScenario(
                "match_ratio",
                settings.primary_event_threshold,
                settings.primary_obs_threshold,
                settings.primary_radius_km,
                max_controls,
                "event_area",
                False,
            )
        )
    for weighting in ("event_area", "equal_grid"):
        scenarios.append(
            SensitivityScenario(
                "weighting",
                settings.primary_event_threshold,
                settings.primary_obs_threshold,
                settings.primary_radius_km,
                settings.max_controls,
                weighting,
                False,
            )
        )
    for strict in (False, True):
        scenarios.append(
            SensitivityScenario(
                "control_horizon",
                settings.primary_event_threshold,
                settings.primary_obs_threshold,
                settings.primary_radius_km,
                settings.max_controls,
                "event_area",
                strict,
            )
        )
    labels = [scenario.label for scenario in scenarios]
    if len(set(labels)) != len(labels):
        raise RuntimeError("Sensitivity scenario labels must be unique.")
    return tuple(scenarios)
