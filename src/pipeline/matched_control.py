from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.event_response import assign_macroregion


TREATMENT_COLUMNS = (
    "grid_id",
    "event_year",
    "event_type",
    "event_area_m2",
    "primary_window_eligible",
    "repeat_within_5y",
    "event_threshold",
    "macroregion",
    "lon",
    "lat",
)

PANEL_COLUMNS = (
    "grid_id",
    "grid_row",
    "grid_col",
    "lon",
    "lat",
    "cell_area_m2",
    "year",
    "tmf_domain_pct",
    "tmf_undist_pct",
    "tmf_disturb_event_pct",
    "evi_mean",
    "evi_min",
    "evi_max",
    "evi_sd",
    "valid_obs",
)

MATCHING_COVARIATES = (
    "baseline_evi_mean",
    "baseline_evi_min",
    "baseline_evi_max",
    "baseline_evi_amp",
    "baseline_evi_sd",
    "pretrend_evi_mean",
    "baseline_undist_share",
    "log_analysis_area",
)

PAIR_COLUMNS = (
    "treatment_grid_id",
    "control_grid_id",
    "event_year",
    "control_event_year",
    "event_type",
    "macroregion",
    "control_macroregion",
    "event_area_m2",
    "treatment_lon",
    "treatment_lat",
    "control_lon",
    "control_lat",
    "geographic_distance_km",
    "covariate_distance",
    "match_rank",
    "within_set_weight",
    "radius_fallback",
    "search_radius_km",
    "control_reuse_count",
)


@dataclass(frozen=True)
class MatchedControlSettings:
    event_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20)
    obs_thresholds: tuple[int, ...] = (8, 12, 16)
    primary_event_threshold: float = 0.10
    primary_obs_threshold: int = 12
    primary_radius_km: float = 50.0
    fallback_radius_km: float = 100.0
    max_controls: int = 3
    relative_years: tuple[int, ...] = tuple(range(-3, 6))
    primary_event_years: tuple[int, int] = (2004, 2018)
    analysis_years: tuple[int, int] = (2001, 2023)
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20260725
    balance_threshold: float = 0.10
    tmf_raw_full_cover_value: float = 0.0029050935991108418
    expected_part_count: int = 29
    expected_stage03_rows: int = 38_805_899
    expected_primary_treatments: int = 15_674


@dataclass(frozen=True)
class SpatialBins:
    cell_degrees: float
    longitude_cells: int
    members: dict


def validate_settings(settings):
    if settings.primary_event_threshold not in settings.event_thresholds:
        raise ValueError("primary_event_threshold must occur in event_thresholds.")
    if settings.primary_obs_threshold not in settings.obs_thresholds:
        raise ValueError("primary_obs_threshold must occur in obs_thresholds.")
    if tuple(sorted(settings.event_thresholds)) != settings.event_thresholds:
        raise ValueError("event_thresholds must be sorted.")
    if tuple(sorted(settings.obs_thresholds)) != settings.obs_thresholds:
        raise ValueError("obs_thresholds must be sorted.")
    if settings.primary_radius_km <= 0.0:
        raise ValueError("primary_radius_km must be positive.")
    if settings.fallback_radius_km < settings.primary_radius_km:
        raise ValueError("fallback_radius_km must not be smaller than primary_radius_km.")
    if settings.max_controls < 1:
        raise ValueError("max_controls must be positive.")
    expected_relative_years = tuple(
        range(settings.relative_years[0], settings.relative_years[-1] + 1)
    )
    if settings.relative_years != expected_relative_years:
        raise ValueError("relative_years must be contiguous.")
    if not 0.0 < settings.balance_threshold < 1.0:
        raise ValueError("balance_threshold must be within (0, 1).")
    return settings


def select_primary_treatments(catalog, recovery, settings=None):
    settings = settings or MatchedControlSettings()
    validate_settings(settings)
    missing_catalog = set(TREATMENT_COLUMNS) - set(catalog.columns)
    if missing_catalog:
        raise ValueError(f"Event catalog lacks columns: {sorted(missing_catalog)}")
    missing_recovery = {"grid_id", "metric"} - set(recovery.columns)
    if missing_recovery:
        raise ValueError(f"Recovery table lacks columns: {sorted(missing_recovery)}")
    if catalog["grid_id"].duplicated().any():
        raise ValueError("Event catalog must contain a unique grid_id per row.")

    recovery_ids = pd.Index(
        recovery.loc[recovery["metric"].eq("evi_mean"), "grid_id"].unique()
    )
    event_threshold = pd.to_numeric(catalog["event_threshold"], errors="coerce")
    mask = (
        catalog["grid_id"].isin(recovery_ids)
        & catalog["primary_window_eligible"].astype(bool)
        & ~catalog["repeat_within_5y"].astype(bool)
        & np.isclose(event_threshold, settings.primary_event_threshold)
    )
    result = catalog.loc[mask, TREATMENT_COLUMNS].copy()
    if not set(result["event_type"]).issubset({"degradation", "deforestation"}):
        raise ValueError("Primary treatments contain an invalid event_type.")
    event_area = pd.to_numeric(result["event_area_m2"], errors="coerce")
    if not (np.isfinite(event_area).all() and event_area.gt(0.0).all()):
        raise ValueError("Primary treatment event_area_m2 must be finite and positive.")
    return result.sort_values("grid_id", kind="mergesort").reset_index(drop=True)


def select_all_first_treatments(catalog, settings=None):
    """Select all first threshold-level events, including later-repeat events."""
    settings = settings or MatchedControlSettings()
    validate_settings(settings)
    missing_catalog = set(TREATMENT_COLUMNS) - set(catalog.columns)
    if missing_catalog:
        raise ValueError(f"Event catalog lacks columns: {sorted(missing_catalog)}")
    if catalog["grid_id"].duplicated().any():
        raise ValueError("Event catalog must contain a unique grid_id per row.")
    event_threshold = pd.to_numeric(catalog["event_threshold"], errors="coerce")
    mask = (
        catalog["primary_window_eligible"].astype(bool)
        & np.isclose(event_threshold, settings.primary_event_threshold)
    )
    result = catalog.loc[mask].copy()
    if not set(result["event_type"]).issubset({"degradation", "deforestation"}):
        raise ValueError("All-first treatments contain an invalid event_type.")
    event_area = pd.to_numeric(result["event_area_m2"], errors="coerce")
    if not (np.isfinite(event_area).all() and event_area.gt(0.0).all()):
        raise ValueError("All-first treatment event_area_m2 must be finite and positive.")
    return result.sort_values("grid_id", kind="mergesort").reset_index(drop=True)


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


def reshape_complete_panel(frame, analysis_years=tuple(range(2001, 2024))):
    missing = set(PANEL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Annual panel lacks columns: {sorted(missing)}")
    working = frame.loc[:, PANEL_COLUMNS].copy()
    if working.duplicated(["grid_id", "year"]).any():
        raise ValueError("Duplicate grid-year rows are not allowed.")
    working = working.sort_values(["grid_id", "year"], kind="mergesort")
    group_sizes = working.groupby("grid_id", sort=False).size()
    expected_years = np.asarray(analysis_years, dtype=np.int16)
    if group_sizes.empty or not group_sizes.eq(len(expected_years)).all():
        raise ValueError(f"Every grid must contain exactly {len(expected_years)} annual rows.")
    n_grids = len(group_sizes)
    year_matrix = working["year"].to_numpy().reshape(n_grids, len(expected_years))
    if not np.array_equal(year_matrix, np.broadcast_to(expected_years, year_matrix.shape)):
        raise ValueError("Every grid must contain the complete ordered analysis-year sequence.")

    static_columns = ("grid_row", "grid_col", "lon", "lat", "cell_area_m2")
    inconsistent = [
        column
        for column in static_columns
        if working.groupby("grid_id", sort=False)[column].nunique(dropna=False).gt(1).any()
    ]
    if inconsistent:
        raise ValueError(f"Static grid fields vary across years: {inconsistent}")

    arrays = {
        "grid_id": working["grid_id"].to_numpy().reshape(n_grids, -1)[:, 0],
        "years": expected_years,
    }
    for column in PANEL_COLUMNS:
        if column in {"grid_id", "year"}:
            continue
        values = pd.to_numeric(working[column], errors="coerce").to_numpy()
        arrays[column] = values.reshape(n_grids, len(expected_years))
    return arrays


def ols_slope(years, values, usable):
    years = np.asarray(years, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool)
    if values.ndim == 1:
        values = values[None, :]
        usable = usable[None, :]
        scalar = True
    else:
        scalar = False
    if values.shape != usable.shape or values.shape[1] != len(years):
        raise ValueError("Slope inputs have incompatible shapes.")
    count = usable.sum(axis=1)
    xsum = np.where(usable, years[None, :], 0.0).sum(axis=1)
    ysum = np.where(usable, values, 0.0).sum(axis=1)
    xbar = np.divide(xsum, count, out=np.zeros_like(xsum), where=count > 0)
    ybar = np.divide(ysum, count, out=np.zeros_like(ysum), where=count > 0)
    xcenter = years[None, :] - xbar[:, None]
    ycenter = values - ybar[:, None]
    numerator = np.where(usable, xcenter * ycenter, 0.0).sum(axis=1)
    denominator = np.where(usable, xcenter**2, 0.0).sum(axis=1)
    result = np.full(len(values), np.nan, dtype=np.float64)
    valid = (count >= 2) & np.isfinite(denominator) & (denominator > 0.0)
    np.divide(numerator, denominator, out=result, where=valid)
    return float(result[0]) if scalar else result


def _masked_mean(values, usable, minimum=1):
    values = np.asarray(values, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool) & np.isfinite(values)
    count = usable.sum(axis=1)
    total = np.where(usable, values, 0.0).sum(axis=1)
    result = np.full(len(values), np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count >= minimum)
    result[count < minimum] = np.nan
    return result, count


def _derive_event_year_features_from_arrays(
    arrays,
    event_year,
    event_threshold,
    obs_threshold,
    settings,
):
    validate_settings(settings)
    event_year = int(event_year)
    event_threshold = float(event_threshold)
    obs_threshold = int(obs_threshold)
    if event_threshold <= 0.0 or event_threshold > 1.0:
        raise ValueError("event_threshold must be within (0, 1].")
    years = arrays["years"]
    matches = np.flatnonzero(years == event_year)
    if len(matches) != 1:
        raise ValueError("event_year must occur exactly once in the analysis years.")
    event_index = int(matches[0])
    baseline_slice = slice(event_index - 3, event_index)
    post_slice = slice(event_index + 1, event_index + 6)
    if baseline_slice.start < 0 or post_slice.stop > len(years):
        raise ValueError("event_year does not support the approved -3..+5 window.")

    valid_obs = np.asarray(arrays["valid_obs"], dtype=np.float64)
    metrics = {
        "evi_mean": np.asarray(arrays["evi_mean"], dtype=np.float64),
        "evi_min": np.asarray(arrays["evi_min"], dtype=np.float64),
        "evi_max": np.asarray(arrays["evi_max"], dtype=np.float64),
        "evi_sd": np.asarray(arrays["evi_sd"], dtype=np.float64),
    }
    metrics["evi_amp"] = metrics["evi_max"] - metrics["evi_min"]
    feature_data = {
        "grid_id": arrays["grid_id"],
        "grid_row": arrays["grid_row"][:, 0].astype(np.int32),
        "grid_col": arrays["grid_col"][:, 0].astype(np.int32),
        "lon": arrays["lon"][:, 0],
        "lat": arrays["lat"][:, 0],
        "cell_area_m2": arrays["cell_area_m2"][:, 0],
        "event_year": np.full(len(arrays["grid_id"]), event_year, dtype=np.int16),
    }
    metric_usable = {}
    for metric, values in metrics.items():
        usable = np.isfinite(values) & np.isfinite(valid_obs) & (valid_obs >= obs_threshold)
        metric_usable[metric] = usable
        baseline, count = _masked_mean(
            values[:, baseline_slice],
            usable[:, baseline_slice],
            minimum=2,
        )
        feature_data[f"baseline_{metric}"] = baseline
        feature_data[f"baseline_n_{metric}"] = count.astype(np.int16)
    feature_data["baseline_n"] = feature_data["baseline_n_evi_mean"]
    feature_data["pretrend_evi_mean"] = ols_slope(
        years[baseline_slice],
        metrics["evi_mean"][:, baseline_slice],
        metric_usable["evi_mean"][:, baseline_slice],
    )

    domain = np.asarray(arrays["tmf_domain_pct"], dtype=np.float64)
    undisturbed = np.asarray(arrays["tmf_undist_pct"], dtype=np.float64)
    disturbance = np.asarray(arrays["tmf_disturb_event_pct"], dtype=np.float64)
    undisturbed_share = safe_share(undisturbed, domain)
    baseline_undist, baseline_undist_n = _masked_mean(
        undisturbed_share[:, baseline_slice],
        np.isfinite(undisturbed_share[:, baseline_slice]),
        minimum=2,
    )
    analysis_area = arrays["cell_area_m2"] * domain / settings.tmf_raw_full_cover_value
    baseline_area, baseline_area_n = _masked_mean(
        analysis_area[:, baseline_slice],
        np.isfinite(analysis_area[:, baseline_slice]) & (analysis_area[:, baseline_slice] > 0.0),
        minimum=2,
    )
    feature_data["baseline_undist_share"] = baseline_undist
    feature_data["baseline_undist_n"] = baseline_undist_n.astype(np.int16)
    feature_data["log_analysis_area"] = np.where(
        np.isfinite(baseline_area) & (baseline_area > 0.0),
        np.log(baseline_area),
        np.nan,
    )
    feature_data["baseline_area_n"] = baseline_area_n.astype(np.int16)

    disturbance_share = safe_share(disturbance, domain)
    history_observed = np.isfinite(disturbance_share)
    followup_stop = event_index + 6
    observed_through_followup = history_observed[:, :followup_stop].all(axis=1)
    observed_through_2023 = history_observed.all(axis=1)
    maximum_through_followup = np.where(
        observed_through_followup,
        np.max(
            np.where(
                history_observed[:, :followup_stop],
                disturbance_share[:, :followup_stop],
                -np.inf,
            ),
            axis=1,
        ),
        np.nan,
    )
    maximum_through_2023 = np.where(
        observed_through_2023,
        np.max(
            np.where(history_observed, disturbance_share, -np.inf), axis=1
        ),
        np.nan,
    )
    feature_data["max_disturbance_share_through_followup"] = maximum_through_followup
    feature_data["max_disturbance_share_through_2023"] = maximum_through_2023
    feature_data["stable_through_followup"] = (
        observed_through_followup
        & ~(disturbance_share[:, :followup_stop] >= event_threshold).any(axis=1)
    )
    feature_data["stable_through_2023"] = (
        observed_through_2023
        & ~(disturbance_share >= event_threshold).any(axis=1)
    )

    mean_usable = metric_usable["evi_mean"]
    feature_data["event_year_usable"] = mean_usable[:, event_index]
    feature_data["post_usable_n"] = mean_usable[:, post_slice].sum(axis=1).astype(np.int16)
    feature_data["coverage_eligible"] = (
        (feature_data["baseline_n"] >= 2)
        & feature_data["event_year_usable"]
        & (feature_data["post_usable_n"] >= 4)
    )
    feature_data["macroregion"] = assign_macroregion(
        feature_data["lon"], feature_data["lat"]
    )
    features = pd.DataFrame(feature_data)
    features["matching_covariates_complete"] = np.isfinite(
        features.loc[:, MATCHING_COVARIATES].to_numpy(dtype=np.float64)
    ).all(axis=1)
    return features.sort_values("grid_id", kind="mergesort").reset_index(drop=True)


def derive_event_year_features(
    frame,
    event_year,
    event_threshold,
    obs_threshold,
    settings=None,
):
    settings = settings or MatchedControlSettings()
    analysis_years = tuple(
        range(settings.analysis_years[0], settings.analysis_years[1] + 1)
    )
    arrays = reshape_complete_panel(frame, analysis_years)
    return _derive_event_year_features_from_arrays(
        arrays,
        event_year,
        event_threshold,
        obs_threshold,
        settings,
    )


def iter_event_year_features(
    frame,
    event_years,
    event_threshold,
    obs_threshold,
    settings=None,
):
    settings = settings or MatchedControlSettings()
    analysis_years = tuple(
        range(settings.analysis_years[0], settings.analysis_years[1] + 1)
    )
    arrays = reshape_complete_panel(frame, analysis_years)
    for event_year in event_years:
        yield int(event_year), _derive_event_year_features_from_arrays(
            arrays,
            event_year,
            event_threshold,
            obs_threshold,
            settings,
        )


def control_risk_set(features, stable_through_2023=False, event_threshold=None):
    stable_column = "stable_through_2023" if stable_through_2023 else "stable_through_followup"
    required = {
        "grid_id",
        "lon",
        "lat",
        "coverage_eligible",
        "matching_covariates_complete",
        stable_column,
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table lacks columns: {sorted(missing)}")
    valid_coordinates = np.isfinite(
        features.loc[:, ["lon", "lat"]].to_numpy(dtype=np.float64)
    ).all(axis=1)
    if event_threshold is None:
        stable = features[stable_column].astype(bool)
    else:
        maximum_column = (
            "max_disturbance_share_through_2023"
            if stable_through_2023
            else "max_disturbance_share_through_followup"
        )
        if maximum_column not in features:
            raise ValueError(f"Feature table lacks column: {maximum_column}")
        maximum = pd.to_numeric(features[maximum_column], errors="coerce")
        stable = maximum.notna() & maximum.lt(float(event_threshold))
    mask = (
        features["coverage_eligible"].astype(bool)
        & features["matching_covariates_complete"].astype(bool)
        & stable
        & valid_coordinates
    )
    return features.loc[mask].sort_values("grid_id", kind="mergesort").reset_index(drop=True)


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = np.broadcast_arrays(
        np.asarray(lon1, dtype=np.float64),
        np.asarray(lat1, dtype=np.float64),
        np.asarray(lon2, dtype=np.float64),
        np.asarray(lat2, dtype=np.float64),
    )
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    delta_lon = (lon2 - lon1 + np.pi) % (2.0 * np.pi) - np.pi
    delta_lat = lat2 - lat1
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * 6371.0088 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def robust_location_scale(frame, covariate_columns=MATCHING_COVARIATES):
    missing = set(covariate_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Matching table lacks covariates: {sorted(missing)}")
    values = frame.loc[:, covariate_columns].apply(pd.to_numeric, errors="coerce")
    locations = values.median(axis=0, skipna=True)
    deviations = values.sub(locations, axis=1).abs()
    scales = deviations.median(axis=0, skipna=True) * 1.4826
    standard_deviation = values.std(axis=0, ddof=0, skipna=True)
    invalid = ~np.isfinite(scales) | scales.le(0.0)
    scales.loc[invalid] = standard_deviation.loc[invalid]
    scales.loc[~np.isfinite(scales) | scales.le(0.0)] = np.nan
    return locations, scales


def standardize_covariates(
    frame,
    locations,
    scales,
    covariate_columns=MATCHING_COVARIATES,
):
    values = frame.loc[:, covariate_columns].apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame(0.0, index=frame.index, columns=covariate_columns)
    active = scales.index[np.isfinite(scales.to_numpy(dtype=np.float64))]
    if len(active):
        result.loc[:, active] = values.loc[:, active].sub(
            locations.loc[active], axis=1
        ).div(scales.loc[active], axis=1)
    if not np.isfinite(result.to_numpy(dtype=np.float64)).all():
        raise ValueError("Matching covariates cannot be standardized to finite values.")
    return result


def build_spatial_bins(controls, cell_degrees=0.25):
    cell_degrees = float(cell_degrees)
    if not 0.0 < cell_degrees <= 5.0:
        raise ValueError("cell_degrees must be within (0, 5].")
    coordinates = controls.loc[:, ["lon", "lat"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Control coordinates must be finite.")
    longitude_cells = int(round(360.0 / cell_degrees))
    bx = np.floor((coordinates[:, 0] + 180.0) / cell_degrees).astype(np.int32)
    bx %= longitude_cells
    by = np.floor((coordinates[:, 1] + 90.0) / cell_degrees).astype(np.int32)
    members = defaultdict(list)
    for index, key in enumerate(zip(bx, by)):
        members[(int(key[0]), int(key[1]))].append(index)
    compact = {key: np.asarray(value, dtype=np.int64) for key, value in members.items()}
    return SpatialBins(cell_degrees, longitude_cells, compact)


def candidate_indices_within_radius(
    treatment,
    controls,
    spatial_bins,
    radius_km,
):
    radius_km = float(radius_km)
    lon = float(treatment["lon"])
    lat = float(treatment["lat"])
    if not np.isfinite([lon, lat, radius_km]).all() or radius_km <= 0.0:
        raise ValueError("Treatment coordinates and radius must be finite and valid.")
    cell = spatial_bins.cell_degrees
    tx = int(np.floor((lon + 180.0) / cell)) % spatial_bins.longitude_cells
    ty = int(np.floor((lat + 90.0) / cell))
    latitude_span = int(np.ceil((radius_km / 110.574) / cell)) + 1
    cosine = max(float(np.cos(np.radians(lat))), 0.10)
    longitude_span = int(np.ceil((radius_km / (111.320 * cosine)) / cell)) + 1
    candidates = []
    for dx in range(-longitude_span, longitude_span + 1):
        bx = (tx + dx) % spatial_bins.longitude_cells
        for dy in range(-latitude_span, latitude_span + 1):
            hit = spatial_bins.members.get((bx, ty + dy))
            if hit is not None:
                candidates.append(hit)
    if not candidates:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    indices = np.unique(np.concatenate(candidates))
    control_coordinates = controls.iloc[indices].loc[:, ["lon", "lat"]]
    distances = haversine_km(
        lon,
        lat,
        control_coordinates["lon"].to_numpy(dtype=np.float64),
        control_coordinates["lat"].to_numpy(dtype=np.float64),
    )
    within = distances <= radius_km + 1e-9
    return indices[within], distances[within]


def match_risk_set(treatments, controls, settings=None):
    settings = settings or MatchedControlSettings()
    validate_settings(settings)
    required = {
        "grid_id",
        "event_year",
        "macroregion",
        "lon",
        "lat",
        *MATCHING_COVARIATES,
    }
    missing_treatments = required - set(treatments.columns)
    missing_controls = required - set(controls.columns)
    if missing_treatments:
        raise ValueError(f"Treatments lack columns: {sorted(missing_treatments)}")
    if missing_controls:
        raise ValueError(f"Controls lack columns: {sorted(missing_controls)}")
    if treatments["grid_id"].duplicated().any():
        raise ValueError("Treatment grid_id values must be unique.")

    rows = []
    grouping = ["event_year", "macroregion"]
    sorted_treatments = treatments.sort_values(
        [*grouping, "grid_id"], kind="mergesort"
    )
    for (event_year, macroregion), treatment_group in sorted_treatments.groupby(
        grouping, sort=True, observed=True
    ):
        control_group = controls.loc[
            controls["event_year"].eq(event_year)
            & controls["macroregion"].eq(macroregion)
        ].sort_values("grid_id", kind="mergesort").reset_index(drop=True)
        if control_group.empty:
            continue
        locations, scales = robust_location_scale(control_group)
        control_standardized = standardize_covariates(
            control_group, locations, scales
        ).to_numpy(dtype=np.float64)
        treatment_standardized = standardize_covariates(
            treatment_group, locations, scales
        )
        spatial_bins = build_spatial_bins(control_group)

        for treatment_index, treatment in treatment_group.iterrows():
            candidate_indices, geographic_distances = candidate_indices_within_radius(
                treatment,
                control_group,
                spatial_bins,
                settings.primary_radius_km,
            )
            radius_fallback = False
            search_radius = settings.primary_radius_km
            if not len(candidate_indices):
                candidate_indices, geographic_distances = candidate_indices_within_radius(
                    treatment,
                    control_group,
                    spatial_bins,
                    settings.fallback_radius_km,
                )
                radius_fallback = True
                search_radius = settings.fallback_radius_km
            if not len(candidate_indices):
                continue

            treatment_values = treatment_standardized.loc[treatment_index].to_numpy(
                dtype=np.float64
            )
            differences = control_standardized[candidate_indices] - treatment_values
            covariate_distances = np.sqrt(np.sum(differences**2, axis=1))
            control_ids = control_group.iloc[candidate_indices]["grid_id"].to_numpy()
            order = np.lexsort(
                (control_ids, geographic_distances, covariate_distances)
            )[: settings.max_controls]
            selected_indices = candidate_indices[order]
            selected_geographic = geographic_distances[order]
            selected_covariate = covariate_distances[order]
            selected = control_group.iloc[selected_indices]
            event_type = treatment.get("event_type", "unknown")
            event_area = float(treatment.get("event_area_m2", 1.0))
            for rank, (control_row, geo_distance, cov_distance) in enumerate(
                zip(
                    selected.itertuples(index=False),
                    selected_geographic,
                    selected_covariate,
                ),
                start=1,
            ):
                rows.append(
                    {
                        "treatment_grid_id": treatment["grid_id"],
                        "control_grid_id": control_row.grid_id,
                        "event_year": int(event_year),
                        "control_event_year": int(control_row.event_year),
                        "event_type": event_type,
                        "macroregion": macroregion,
                        "control_macroregion": control_row.macroregion,
                        "event_area_m2": event_area,
                        "treatment_lon": float(treatment["lon"]),
                        "treatment_lat": float(treatment["lat"]),
                        "control_lon": float(control_row.lon),
                        "control_lat": float(control_row.lat),
                        "geographic_distance_km": float(geo_distance),
                        "covariate_distance": float(cov_distance),
                        "match_rank": rank,
                        "radius_fallback": radius_fallback,
                        "search_radius_km": float(search_radius),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    result = pd.DataFrame(rows).sort_values(
        ["event_year", "treatment_grid_id", "match_rank"], kind="mergesort"
    ).reset_index(drop=True)
    match_count = result.groupby(
        ["event_year", "treatment_grid_id"], sort=False
    )["control_grid_id"].transform("size")
    result["within_set_weight"] = 1.0 / match_count
    result["control_reuse_count"] = result.groupby(
        "control_grid_id", sort=False
    )["treatment_grid_id"].transform("size").astype(np.int32)
    return result.loc[:, PAIR_COLUMNS]
