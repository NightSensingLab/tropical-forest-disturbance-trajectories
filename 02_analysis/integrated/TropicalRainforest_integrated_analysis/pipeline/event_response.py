from dataclasses import dataclass

import numpy as np
import pandas as pd


MACROREGION_DEFINITIONS = (
    ("Amazon_proxy", -82.0, -34.0, -24.0, 13.0),
    ("Congo_proxy", 8.0, 32.0, -14.0, 8.0),
    ("Southeast_Asia_proxy", 90.0, 155.0, -12.0, 23.5),
)

EVENT_INPUT_COLUMNS = (
    "grid_id",
    "grid_row",
    "grid_col",
    "lon",
    "lat",
    "cell_area_m2",
    "year",
    "tmf_domain_pct",
    "tmf_degrad_event_pct",
    "tmf_defor_event_pct",
    "tmf_regrow_event_pct",
    "tmf_disturb_event_pct",
)

CATALOG_COLUMNS = (
    "grid_id",
    "grid_row",
    "grid_col",
    "lon",
    "lat",
    "cell_area_m2",
    "analysis_area_m2",
    "macroregion",
    "event_threshold",
    "event_year",
    "event_type",
    "disturbance_share",
    "degradation_share",
    "deforestation_share",
    "regrowth_share",
    "event_area_m2",
    "primary_window_eligible",
    "next_significant_event_year",
    "repeat_within_5y",
    "censor_relative_year",
    "source_part",
)

WINDOW_ANNUAL_COLUMNS = (
    "grid_id",
    "year",
    "tmf_domain_pct",
    "tmf_forest_pct",
    "tmf_undist_pct",
    "tmf_degrad_state_pct",
    "tmf_defor_state_pct",
    "tmf_regrow_state_pct",
    "tmf_degrad_event_pct",
    "tmf_defor_event_pct",
    "tmf_regrow_event_pct",
    "tmf_disturb_event_pct",
    "evi_mean",
    "evi_min",
    "evi_max",
    "evi_amp",
    "evi_sd",
    "valid_obs",
)

RECOVERY_COLUMNS = (
    "grid_id",
    "event_year",
    "event_type",
    "macroregion",
    "event_area_m2",
    "metric",
    "baseline_n",
    "post_n",
    "baseline",
    "event_value",
    "event_anomaly",
    "trough_value",
    "trough_anomaly",
    "trough_relative_year",
    "latest_post_year",
    "latest_post_value",
    "latest_post_anomaly",
    "decline_present",
    "recovered_to_95pct_baseline",
    "recovery_relative_year",
    "recovery_fraction",
)


@dataclass(frozen=True)
class EventSettings:
    event_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20)
    obs_thresholds: tuple[int, ...] = (8, 12, 16)
    primary_event_threshold: float = 0.10
    primary_obs_threshold: int = 12
    relative_years: tuple[int, ...] = tuple(range(-3, 6))
    primary_event_years: tuple[int, int] = (2004, 2018)
    analysis_years: tuple[int, int] = (2001, 2023)
    evi_metrics: tuple[str, ...] = (
        "evi_mean",
        "evi_min",
        "evi_max",
        "evi_amp",
        "evi_sd",
    )
    tmf_raw_full_cover_value: float = 0.0029050935991108418
    expected_part_count: int = 29
    expected_stage03_rows: int = 38_805_899


def validate_settings(settings):
    if settings.primary_event_threshold not in settings.event_thresholds:
        raise ValueError("primary_event_threshold must occur in event_thresholds.")
    if settings.primary_obs_threshold not in settings.obs_thresholds:
        raise ValueError("primary_obs_threshold must occur in obs_thresholds.")
    if tuple(sorted(settings.event_thresholds)) != settings.event_thresholds:
        raise ValueError("event_thresholds must be sorted.")
    if tuple(sorted(settings.obs_thresholds)) != settings.obs_thresholds:
        raise ValueError("obs_thresholds must be sorted.")
    expected_relative_years = tuple(
        range(settings.relative_years[0], settings.relative_years[-1] + 1)
    )
    if settings.relative_years != expected_relative_years:
        raise ValueError("relative_years must be contiguous.")
    if settings.primary_event_years != (
        settings.analysis_years[0] - settings.relative_years[0],
        settings.analysis_years[1] - settings.relative_years[-1],
    ):
        raise ValueError("primary_event_years do not provide a balanced event window.")
    return settings


def safe_share(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def assign_macroregion(lon, lat, definitions=MACROREGION_DEFINITIONS):
    lon, lat = np.broadcast_arrays(
        np.asarray(lon, dtype=np.float64),
        np.asarray(lat, dtype=np.float64),
    )
    result = np.full(lon.shape, "Other", dtype=object)
    unassigned = np.ones(lon.shape, dtype=bool)
    for label, lon_min, lon_max, lat_min, lat_max in definitions:
        inside = (
            unassigned
            & np.isfinite(lon)
            & np.isfinite(lat)
            & (lon >= lon_min)
            & (lon <= lon_max)
            & (lat >= lat_min)
            & (lat <= lat_max)
        )
        result[inside] = label
        unassigned[inside] = False
    return result


def derive_event_catalog(
    frame,
    event_threshold,
    settings=None,
    source_part="unknown",
):
    settings = settings or EventSettings()
    validate_settings(settings)
    event_threshold = float(event_threshold)
    if not np.isfinite(event_threshold) or not 0.0 < event_threshold <= 1.0:
        raise ValueError("event_threshold must be finite and within (0, 1].")
    missing = set(EVENT_INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Event input lacks columns: {sorted(missing)}")

    working = frame.loc[:, EVENT_INPUT_COLUMNS].copy()
    for column in EVENT_INPUT_COLUMNS:
        if column != "source_part":
            working[column] = pd.to_numeric(working[column], errors="coerce")
    domain = working["tmf_domain_pct"].to_numpy(dtype=np.float64, na_value=np.nan)
    share_columns = {
        "disturbance_share": "tmf_disturb_event_pct",
        "degradation_share": "tmf_degrad_event_pct",
        "deforestation_share": "tmf_defor_event_pct",
        "regrowth_share": "tmf_regrow_event_pct",
    }
    for output_column, input_column in share_columns.items():
        working[output_column] = safe_share(
            working[input_column].to_numpy(dtype=np.float64, na_value=np.nan),
            domain,
        )

    significant = working.loc[
        working["disturbance_share"].ge(event_threshold)
    ].sort_values(["grid_id", "year"], kind="mergesort")
    if significant.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    first = significant.groupby("grid_id", as_index=False, sort=True).first()
    first = first.rename(columns={"year": "event_year"})

    candidates = significant.loc[:, ["grid_id", "year"]].merge(
        first.loc[:, ["grid_id", "event_year"]],
        on="grid_id",
        how="inner",
        validate="many_to_one",
    )
    later = candidates.loc[
        candidates["year"].gt(candidates["event_year"])
        & candidates["year"].le(candidates["event_year"] + 5)
    ]
    next_year = later.groupby("grid_id", sort=True)["year"].min()
    first["next_significant_event_year"] = first["grid_id"].map(next_year)
    first["repeat_within_5y"] = first["next_significant_event_year"].notna()
    first["censor_relative_year"] = (
        first["next_significant_event_year"] - first["event_year"]
    )

    static_domain = working.groupby("grid_id", sort=True)["tmf_domain_pct"].max()
    first["analysis_area_m2"] = (
        first["cell_area_m2"]
        * first["grid_id"].map(static_domain)
        / settings.tmf_raw_full_cover_value
    )
    first["event_area_m2"] = (
        first["cell_area_m2"]
        * first["tmf_disturb_event_pct"]
        / settings.tmf_raw_full_cover_value
    )
    first["event_type"] = np.where(
        first["deforestation_share"].ge(first["degradation_share"]),
        "deforestation",
        "degradation",
    )
    first["event_threshold"] = event_threshold
    first["primary_window_eligible"] = first["event_year"].between(
        settings.primary_event_years[0],
        settings.primary_event_years[1],
    )
    first["macroregion"] = assign_macroregion(first["lon"], first["lat"])
    first["source_part"] = str(source_part)
    return first.loc[:, CATALOG_COLUMNS].sort_values("grid_id").reset_index(drop=True)


def add_metric_baselines(
    window,
    metric,
    obs_threshold,
    minimum_baseline_years=2,
):
    result = window.copy()
    threshold_label = f"obs{int(obs_threshold):02d}"
    usable_column = f"{metric}_usable_{threshold_label}"
    if usable_column not in result.columns:
        raise ValueError(f"Window lacks usability column: {usable_column}")
    baseline_rows = result.loc[
        result["relative_year"].between(-3, -1) & result[usable_column]
    ]
    grouped = baseline_rows.groupby("grid_id", sort=True)[metric]
    baseline_count = grouped.count()
    baseline = grouped.mean().where(baseline_count >= int(minimum_baseline_years))

    count_column = f"{metric}_baseline_n_{threshold_label}"
    baseline_column = f"{metric}_baseline_{threshold_label}"
    anomaly_column = f"{metric}_anomaly_{threshold_label}"
    relative_column = f"{metric}_relative_change_{threshold_label}"
    result[count_column] = result["grid_id"].map(baseline_count).fillna(0).astype(np.int16)
    result[baseline_column] = result["grid_id"].map(baseline)
    usable = result[usable_column].to_numpy(dtype=bool)
    values = pd.to_numeric(result[metric], errors="coerce").to_numpy(dtype=np.float64)
    baselines = pd.to_numeric(result[baseline_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid_effect = usable & np.isfinite(values) & np.isfinite(baselines)
    anomalies = np.full(len(result), np.nan, dtype=np.float64)
    anomalies[valid_effect] = values[valid_effect] - baselines[valid_effect]
    relative = np.full(len(result), np.nan, dtype=np.float64)
    valid_relative = valid_effect & (baselines != 0.0)
    relative[valid_relative] = anomalies[valid_relative] / baselines[valid_relative]
    result[anomaly_column] = anomalies
    result[relative_column] = relative
    return result


def add_all_baselines(window, settings=None, obs_threshold=None):
    settings = settings or EventSettings()
    obs_threshold = int(obs_threshold or settings.primary_obs_threshold)
    result = window.copy()
    for metric in settings.evi_metrics:
        result = add_metric_baselines(result, metric, obs_threshold)
    return result


def build_event_window(frame, catalog, settings=None):
    settings = settings or EventSettings()
    validate_settings(settings)
    missing = set(WINDOW_ANNUAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Window input lacks columns: {sorted(missing)}")
    missing_catalog = set(CATALOG_COLUMNS) - set(catalog.columns)
    if missing_catalog:
        raise ValueError(f"Event catalog lacks columns: {sorted(missing_catalog)}")

    eligible = catalog.loc[catalog["primary_window_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame()
    annual = frame.loc[:, WINDOW_ANNUAL_COLUMNS].copy()
    for column in WINDOW_ANNUAL_COLUMNS:
        annual[column] = pd.to_numeric(annual[column], errors="coerce")
    window = annual.merge(
        eligible,
        on="grid_id",
        how="inner",
        validate="many_to_one",
    )
    window["relative_year"] = window["year"] - window["event_year"]
    window = window.loc[window["relative_year"].isin(settings.relative_years)].copy()
    window = window.rename(columns={"year": "calendar_year"})

    window["evi_amp"] = window["evi_max"] - window["evi_min"]
    domain = window["tmf_domain_pct"].to_numpy(dtype=np.float64, na_value=np.nan)
    annual_share_columns = {
        "tmf_forest_share": "tmf_forest_pct",
        "tmf_undist_share": "tmf_undist_pct",
        "tmf_degrad_state_share": "tmf_degrad_state_pct",
        "tmf_defor_state_share": "tmf_defor_state_pct",
        "tmf_regrow_state_share": "tmf_regrow_state_pct",
        "tmf_degrad_event_share": "tmf_degrad_event_pct",
        "tmf_defor_event_share": "tmf_defor_event_pct",
        "tmf_regrow_event_share": "tmf_regrow_event_pct",
        "tmf_disturb_event_share": "tmf_disturb_event_pct",
    }
    for output_column, input_column in annual_share_columns.items():
        window[output_column] = safe_share(
            window[input_column].to_numpy(dtype=np.float64, na_value=np.nan),
            domain,
        )

    next_year = pd.to_numeric(
        window["next_significant_event_year"], errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    calendar_year = window["calendar_year"].to_numpy(dtype=np.float64, na_value=np.nan)
    window["censored"] = np.isfinite(next_year) & (calendar_year >= next_year)
    valid_obs = pd.to_numeric(window["valid_obs"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    for metric in settings.evi_metrics:
        values = pd.to_numeric(window[metric], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        for threshold in settings.obs_thresholds:
            window[f"{metric}_usable_obs{threshold:02d}"] = (
                np.isfinite(values) & np.isfinite(valid_obs) & (valid_obs >= threshold)
            )

    window = add_all_baselines(window, settings, settings.primary_obs_threshold)
    return window.sort_values(["grid_id", "relative_year"]).reset_index(drop=True)


def compute_recovery_metrics(window, settings=None):
    settings = settings or EventSettings()
    threshold_label = f"obs{settings.primary_obs_threshold:02d}"
    common_required = {
        "grid_id",
        "event_year",
        "event_type",
        "macroregion",
        "event_area_m2",
        "repeat_within_5y",
        "relative_year",
    }
    missing = common_required - set(window.columns)
    if missing:
        raise ValueError(f"Recovery input lacks columns: {sorted(missing)}")

    records = []
    isolated = window.loc[~window["repeat_within_5y"].astype(bool)]
    for _, rows in isolated.groupby("grid_id", sort=True):
        rows = rows.sort_values("relative_year", kind="mergesort")
        first = rows.iloc[0]
        for metric in settings.evi_metrics:
            usable_column = f"{metric}_usable_{threshold_label}"
            baseline_count_column = f"{metric}_baseline_n_{threshold_label}"
            baseline_column = f"{metric}_baseline_{threshold_label}"
            required = {metric, usable_column, baseline_count_column, baseline_column}
            missing_metric = required - set(rows.columns)
            if missing_metric:
                raise ValueError(
                    f"Recovery input lacks metric columns: {sorted(missing_metric)}"
                )
            baseline_n = int(rows[baseline_count_column].iloc[0])
            baseline = float(rows[baseline_column].iloc[0])
            event_rows = rows.loc[
                rows["relative_year"].eq(0) & rows[usable_column].astype(bool)
            ]
            post_rows = rows.loc[
                rows["relative_year"].between(1, 5)
                & rows[usable_column].astype(bool)
            ].copy()
            if (
                baseline_n < 2
                or not np.isfinite(baseline)
                or event_rows.empty
                or len(post_rows) < 4
            ):
                continue

            trough_rows = rows.loc[
                rows["relative_year"].between(0, 2)
                & rows[usable_column].astype(bool)
            ].copy()
            trough_rows = trough_rows.loc[np.isfinite(trough_rows[metric])]
            if trough_rows.empty:
                continue
            trough = trough_rows.sort_values(
                [metric, "relative_year"], kind="mergesort"
            ).iloc[0]
            latest = post_rows.sort_values("relative_year", kind="mergesort").iloc[-1]
            event = event_rows.sort_values("relative_year", kind="mergesort").iloc[0]
            trough_value = float(trough[metric])
            latest_value = float(latest[metric])
            event_value = float(event[metric])
            decline_present = bool(trough_value < baseline)
            recovery_relative_year = np.nan
            recovery_fraction = np.nan
            recovered = False
            if decline_present:
                recovered_rows = post_rows.loc[
                    post_rows[metric].ge(0.95 * baseline)
                ].sort_values("relative_year", kind="mergesort")
                if not recovered_rows.empty:
                    recovery_relative_year = int(
                        recovered_rows["relative_year"].iloc[0]
                    )
                    recovered = True
                denominator = baseline - trough_value
                if denominator > 0.0:
                    recovery_fraction = (latest_value - trough_value) / denominator

            records.append(
                {
                    "grid_id": int(first["grid_id"]),
                    "event_year": int(first["event_year"]),
                    "event_type": str(first["event_type"]),
                    "macroregion": str(first["macroregion"]),
                    "event_area_m2": float(first["event_area_m2"]),
                    "metric": metric,
                    "baseline_n": baseline_n,
                    "post_n": int(len(post_rows)),
                    "baseline": baseline,
                    "event_value": event_value,
                    "event_anomaly": event_value - baseline,
                    "trough_value": trough_value,
                    "trough_anomaly": trough_value - baseline,
                    "trough_relative_year": int(trough["relative_year"]),
                    "latest_post_year": int(latest["relative_year"]),
                    "latest_post_value": latest_value,
                    "latest_post_anomaly": latest_value - baseline,
                    "decline_present": decline_present,
                    "recovered_to_95pct_baseline": recovered,
                    "recovery_relative_year": recovery_relative_year,
                    "recovery_fraction": recovery_fraction,
                }
            )
    return pd.DataFrame.from_records(records, columns=RECOVERY_COLUMNS)


SUMMARY_COLUMNS = (
    "event_threshold",
    "obs_threshold",
    "cohort",
    "scope",
    "macroregion",
    "event_type",
    "metric",
    "relative_year",
    "weighting",
    "grid_count",
    "weight_sum",
    "observed_event_area_m2",
    "evi_weighted_sum",
    "anomaly_weighted_sum",
    "relative_weight_sum",
    "relative_weighted_sum",
    "weighted_mean_evi",
    "weighted_mean_anomaly",
    "weighted_mean_relative_change",
)


def _aggregate_summary_rows(
    frame,
    event_threshold,
    obs_threshold,
    cohort,
    scope,
    metric,
    weighting,
):
    if frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    working = frame.copy()
    if scope == "Global":
        working["macroregion"] = "Global"
    if weighting == "event_area":
        weights = pd.to_numeric(working["event_area_m2"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
    elif weighting == "equal_grid":
        weights = np.ones(len(working), dtype=np.float64)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    valid_weight = np.isfinite(weights) & (weights > 0.0)
    working = working.loc[valid_weight].copy()
    weights = weights[valid_weight]
    if working.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    threshold_label = f"obs{int(obs_threshold):02d}"
    anomaly_column = f"{metric}_anomaly_{threshold_label}"
    relative_column = f"{metric}_relative_change_{threshold_label}"
    values = pd.to_numeric(working[metric], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    anomalies = pd.to_numeric(working[anomaly_column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    relatives = pd.to_numeric(working[relative_column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    relative_valid = np.isfinite(relatives)
    working["__weight"] = weights
    working["__evi_weighted"] = values * weights
    working["__anomaly_weighted"] = anomalies * weights
    working["__relative_weight"] = np.where(relative_valid, weights, 0.0)
    working["__relative_weighted"] = np.where(
        relative_valid, relatives * weights, 0.0
    )
    group_columns = ["macroregion", "event_type", "relative_year"]
    grouped = working.groupby(
        group_columns, as_index=False, sort=True, observed=True
    ).agg(
        grid_count=("grid_id", "nunique"),
        weight_sum=("__weight", "sum"),
        observed_event_area_m2=("event_area_m2", "sum"),
        evi_weighted_sum=("__evi_weighted", "sum"),
        anomaly_weighted_sum=("__anomaly_weighted", "sum"),
        relative_weight_sum=("__relative_weight", "sum"),
        relative_weighted_sum=("__relative_weighted", "sum"),
    )
    grouped["weighted_mean_evi"] = grouped["evi_weighted_sum"] / grouped["weight_sum"]
    grouped["weighted_mean_anomaly"] = (
        grouped["anomaly_weighted_sum"] / grouped["weight_sum"]
    )
    grouped["weighted_mean_relative_change"] = np.where(
        grouped["relative_weight_sum"].gt(0.0),
        grouped["relative_weighted_sum"] / grouped["relative_weight_sum"],
        np.nan,
    )
    grouped.insert(0, "event_threshold", float(event_threshold))
    grouped.insert(1, "obs_threshold", int(obs_threshold))
    grouped.insert(2, "cohort", cohort)
    grouped.insert(3, "scope", scope)
    grouped.insert(6, "metric", metric)
    grouped.insert(8, "weighting", weighting)
    return grouped.loc[:, SUMMARY_COLUMNS]


def summarize_event_response(window, event_threshold, settings=None):
    settings = settings or EventSettings()
    validate_settings(settings)
    required = {
        "grid_id",
        "macroregion",
        "event_type",
        "relative_year",
        "event_area_m2",
        "repeat_within_5y",
        "censored",
    }
    missing = required - set(window.columns)
    if missing:
        raise ValueError(f"Summary input lacks columns: {sorted(missing)}")
    prepared = window.copy()
    for threshold in settings.obs_thresholds:
        for metric in settings.evi_metrics:
            baseline_column = f"{metric}_baseline_obs{threshold:02d}"
            if baseline_column not in prepared.columns:
                prepared = add_metric_baselines(prepared, metric, threshold)

    summaries = []
    cohorts = {
        "censored_full": ~prepared["censored"].astype(bool),
        "isolated": ~prepared["repeat_within_5y"].astype(bool),
    }
    for cohort, cohort_mask in cohorts.items():
        cohort_rows = prepared.loc[cohort_mask]
        for threshold in settings.obs_thresholds:
            threshold_label = f"obs{threshold:02d}"
            for metric in settings.evi_metrics:
                usable_column = f"{metric}_usable_{threshold_label}"
                anomaly_column = f"{metric}_anomaly_{threshold_label}"
                if usable_column not in cohort_rows.columns:
                    raise ValueError(f"Summary input lacks column: {usable_column}")
                valid = (
                    cohort_rows[usable_column].astype(bool)
                    & np.isfinite(cohort_rows[metric])
                    & np.isfinite(cohort_rows[anomaly_column])
                )
                metric_rows = cohort_rows.loc[valid]
                for scope in ("Global", "Macroregion"):
                    for weighting in ("event_area", "equal_grid"):
                        summaries.append(
                            _aggregate_summary_rows(
                                metric_rows,
                                event_threshold,
                                threshold,
                                cohort,
                                scope,
                                metric,
                                weighting,
                            )
                        )
    nonempty = [frame for frame in summaries if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.concat(nonempty, ignore_index=True).sort_values(
        [
            "event_threshold",
            "obs_threshold",
            "cohort",
            "scope",
            "macroregion",
            "event_type",
            "metric",
            "relative_year",
            "weighting",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
