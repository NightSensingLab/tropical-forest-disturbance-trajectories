from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


MONTHLY_REQUIRED_COLUMNS = (
    "grid_id",
    "event_year",
    "role",
    "relative_year",
    "calendar_month",
    "evi_mean",
    "evi_median",
    "baseline_evi",
    "baseline_year_count",
    "evi_anomaly",
    "valid_composite_count",
    "coverage_fraction",
    "mean_valid_pixel_fraction",
)

LINK_REQUIRED_COLUMNS = (
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
    "within_set_weight",
)

MONTHLY_EFFECT_COLUMNS = (
    "treatment_grid_id",
    "event_year",
    "event_type",
    "macroregion",
    "event_area_m2",
    "treatment_lon",
    "treatment_lat",
    "relative_year",
    "calendar_month",
    "treatment_evi_anomaly",
    "control_evi_anomaly",
    "matched_effect",
    "retained_control_weight",
    "usable_control_count",
)

ANNUAL_EFFECT_COLUMNS = (
    "treatment_grid_id",
    "event_year",
    "event_type",
    "macroregion",
    "event_area_m2",
    "treatment_lon",
    "treatment_lat",
    "relative_year",
    "usable_months",
    "annual_matched_effect",
)

SIXTEEN_DAY_REQUIRED_COLUMNS = (
    "grid_id",
    "event_year",
    "role",
    "calendar_year",
    "relative_year",
    "calendar_month",
    "day_of_year",
    "evi",
    "valid_pixel_fraction",
    "evi_valid",
)

SIXTEEN_DAY_EFFECT_COLUMNS = (
    "treatment_grid_id",
    "event_year",
    "event_type",
    "macroregion",
    "event_area_m2",
    "treatment_lon",
    "treatment_lat",
    "relative_year",
    "calendar_month",
    "composite_slot",
    "treatment_evi_anomaly",
    "control_evi_anomaly",
    "matched_effect",
    "retained_control_weight",
    "usable_control_count",
)


@dataclass(frozen=True)
class SubannualSettings:
    event_years: tuple[int, ...] = tuple(range(2004, 2019))
    relative_years: tuple[int, ...] = tuple(range(-3, 6))
    valid_pixel_threshold: float = 0.50
    coverage_threshold: float = 0.50
    min_baseline_years: int = 2
    min_valid_composites: int = 1
    control_weight_threshold: float = 2.0 / 3.0
    annual_min_months: int = 9
    outcome: str = "mean"
    weighting: str = "event_area"
    block_degrees: float = 2.0
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 20260725
    stage08_signature: str = (
        "fce7f61a50249af3c80f67b0f098a1676e0cbc4413844a5669b34558764d950b"
    )
    stage07_source_sha256: str = (
        "493359a28b473d4a3213c4dceee4dad3b5cbc13476f87c0d860cb811f18a7672"
    )
    expected_pair_links: int = 46_984
    expected_treatments: int = 15_674
    expected_unit_events: int = 49_197
    expected_monthly_rows: int = 5_313_276
    expected_sixteen_day_rows: int = 10_183_779


@dataclass(frozen=True)
class SensitivityScenario:
    name: str
    valid_pixel_threshold: float
    control_weight_threshold: float
    outcome: str
    weighting: str
    annual_min_months: int


def validate_settings(settings):
    if not settings.event_years or tuple(sorted(settings.event_years)) != settings.event_years:
        raise ValueError("event_years must be nonempty and sorted.")
    if settings.relative_years != tuple(
        range(settings.relative_years[0], settings.relative_years[-1] + 1)
    ):
        raise ValueError("relative_years must be contiguous.")
    if not 0.0 <= settings.valid_pixel_threshold <= 1.0:
        raise ValueError("valid_pixel_threshold must be within [0, 1].")
    if not 0.0 <= settings.coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be within [0, 1].")
    if settings.min_baseline_years < 2:
        raise ValueError("min_baseline_years must be at least two.")
    if settings.min_valid_composites < 1:
        raise ValueError("min_valid_composites must be positive.")
    if not 0.0 < settings.control_weight_threshold <= 1.0:
        raise ValueError("control_weight_threshold must be within (0, 1].")
    if not 1 <= settings.annual_min_months <= 12:
        raise ValueError("annual_min_months must be within [1, 12].")
    if settings.outcome not in {"mean", "median"}:
        raise ValueError("outcome must be mean or median.")
    if settings.weighting not in {"event_area", "equal_grid"}:
        raise ValueError("weighting must be event_area or equal_grid.")
    if settings.block_degrees <= 0.0:
        raise ValueError("block_degrees must be positive.")
    if settings.bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be at least two.")
    return settings


def _require_columns(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} lacks columns: {sorted(missing)}")


def monthly_usable_mask(
    frame,
    settings=None,
    anomaly_column="evi_anomaly",
    value_column="evi_mean",
    baseline_column="baseline_evi",
):
    settings = validate_settings(settings or SubannualSettings())
    required = {
        anomaly_column,
        value_column,
        baseline_column,
        "baseline_year_count",
        "valid_composite_count",
        "coverage_fraction",
        "mean_valid_pixel_fraction",
    }
    _require_columns(frame, required, "Monthly frame")
    finite = np.isfinite(
        frame[[anomaly_column, value_column, baseline_column]].apply(
            pd.to_numeric, errors="coerce"
        )
    ).all(axis=1)
    return (
        finite
        & pd.to_numeric(frame["baseline_year_count"], errors="coerce").ge(
            settings.min_baseline_years
        )
        & pd.to_numeric(frame["valid_composite_count"], errors="coerce").ge(
            settings.min_valid_composites
        )
        & pd.to_numeric(frame["coverage_fraction"], errors="coerce").ge(
            settings.coverage_threshold
        )
        & pd.to_numeric(frame["mean_valid_pixel_fraction"], errors="coerce").ge(
            settings.valid_pixel_threshold
        )
    )


def recompute_monthly_anomaly(frame, value_column="evi_median", minimum_years=2):
    _require_columns(
        frame,
        ("grid_id", "event_year", "role", "relative_year", "calendar_month", value_column),
        "Monthly frame",
    )
    result = frame.copy()
    values = pd.to_numeric(result[value_column], errors="coerce")
    pre = result.loc[
        result["relative_year"].between(-3, -1) & np.isfinite(values),
        ["grid_id", "event_year", "role", "calendar_month"],
    ].copy()
    pre["_value"] = values.loc[pre.index].to_numpy()
    keys = ["grid_id", "event_year", "role", "calendar_month"]
    baseline = (
        pre.groupby(keys, sort=False)["_value"]
        .agg([("_analysis_baseline", "mean"), ("_analysis_baseline_count", "size")])
        .reset_index()
    )
    result = result.merge(baseline, on=keys, how="left", validate="many_to_one")
    enough = result["_analysis_baseline_count"].ge(int(minimum_years))
    result["_analysis_value"] = values
    result["_analysis_baseline"] = result["_analysis_baseline"].where(enough)
    result["_analysis_baseline_count"] = (
        result["_analysis_baseline_count"].fillna(0).astype(np.int16)
    )
    result["_analysis_anomaly"] = (
        result["_analysis_value"] - result["_analysis_baseline"]
    )
    return result


def _validate_links(links):
    _require_columns(links, LINK_REQUIRED_COLUMNS, "Matched links")
    if links.duplicated(["treatment_grid_id", "control_grid_id", "event_year"]).any():
        raise ValueError("Duplicate treatment-control links are not allowed.")
    if not (links["event_year"].to_numpy() == links["control_event_year"].to_numpy()).all():
        raise ValueError("Treatment and control event years must agree.")
    if not (links["macroregion"].to_numpy() == links["control_macroregion"].to_numpy()).all():
        raise ValueError("Treatment and control macroregions must agree.")
    weights = pd.to_numeric(links["within_set_weight"], errors="coerce")
    if not (np.isfinite(weights).all() and weights.gt(0.0).all()):
        raise ValueError("within_set_weight must be finite and positive.")
    sums = links.assign(_weight=weights).groupby(
        ["treatment_grid_id", "event_year"], sort=False
    )["_weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("within_set_weight must sum to one within every matched set.")


def _empty_monthly_effects():
    return pd.DataFrame(columns=MONTHLY_EFFECT_COLUMNS)


def build_monthly_matched_effects(monthly, links, settings=None):
    settings = validate_settings(settings or SubannualSettings())
    _require_columns(monthly, MONTHLY_REQUIRED_COLUMNS, "Monthly frame")
    _validate_links(links)
    role_key = ["grid_id", "event_year", "role", "relative_year", "calendar_month"]
    if monthly.duplicated(role_key).any():
        raise ValueError("Duplicate monthly role-specific keys are not allowed.")

    if settings.outcome == "median":
        working = recompute_monthly_anomaly(
            monthly, value_column="evi_median", minimum_years=settings.min_baseline_years
        )
        working["baseline_year_count"] = working["_analysis_baseline_count"]
        usable = monthly_usable_mask(
            working,
            settings,
            anomaly_column="_analysis_anomaly",
            value_column="_analysis_value",
            baseline_column="_analysis_baseline",
        )
        working["_outcome_anomaly"] = working["_analysis_anomaly"]
    else:
        working = monthly.copy()
        usable = monthly_usable_mask(working, settings)
        working["_outcome_anomaly"] = pd.to_numeric(
            working["evi_anomaly"], errors="coerce"
        )
    working["_usable"] = usable.to_numpy(dtype=bool)

    periods = ["event_year", "relative_year", "calendar_month"]
    treatment = working.loc[
        working["role"].eq("treatment") & working["_usable"],
        ["grid_id", *periods, "_outcome_anomaly"],
    ].rename(
        columns={
            "grid_id": "treatment_grid_id",
            "_outcome_anomaly": "treatment_evi_anomaly",
        }
    )
    controls = working.loc[
        working["role"].eq("control") & working["_usable"],
        ["grid_id", *periods, "_outcome_anomaly"],
    ].rename(
        columns={
            "grid_id": "control_grid_id",
            "_outcome_anomaly": "control_evi_anomaly_raw",
        }
    )
    if treatment.empty or controls.empty:
        return _empty_monthly_effects()

    link_columns = list(dict.fromkeys(LINK_REQUIRED_COLUMNS))
    linked = links.loc[:, link_columns].merge(
        controls,
        on=["control_grid_id", "event_year"],
        how="inner",
        validate="many_to_many",
    )
    linked["_weighted_control"] = (
        pd.to_numeric(linked["within_set_weight"], errors="coerce")
        * pd.to_numeric(linked["control_evi_anomaly_raw"], errors="coerce")
    )
    group_keys = ["treatment_grid_id", *periods]
    control_summary = (
        linked.groupby(group_keys, sort=False)
        .agg(
            retained_control_weight=("within_set_weight", "sum"),
            _weighted_control=("_weighted_control", "sum"),
            usable_control_count=("control_grid_id", "nunique"),
        )
        .reset_index()
    )
    control_summary["control_evi_anomaly"] = (
        control_summary["_weighted_control"]
        / control_summary["retained_control_weight"]
    )
    result = treatment.merge(control_summary, on=group_keys, how="inner", validate="one_to_one")

    metadata_columns = [
        "treatment_grid_id",
        "event_year",
        "event_type",
        "macroregion",
        "event_area_m2",
        "treatment_lon",
        "treatment_lat",
    ]
    metadata = links.loc[:, metadata_columns].drop_duplicates()
    if metadata.duplicated(["treatment_grid_id", "event_year"]).any():
        raise ValueError("Matched-set treatment metadata are inconsistent.")
    result = result.merge(
        metadata,
        on=["treatment_grid_id", "event_year"],
        how="left",
        validate="many_to_one",
    )
    threshold = settings.control_weight_threshold
    result = result.loc[result["retained_control_weight"].ge(threshold - 1e-12)].copy()
    result["matched_effect"] = (
        result["treatment_evi_anomaly"] - result["control_evi_anomaly"]
    )
    if result.empty:
        return _empty_monthly_effects()
    if result.duplicated(
        ["treatment_grid_id", "event_year", "relative_year", "calendar_month"]
    ).any():
        raise ValueError("Duplicate monthly matched-effect primary keys are not allowed.")
    return result.loc[:, MONTHLY_EFFECT_COLUMNS].sort_values(
        ["event_year", "treatment_grid_id", "relative_year", "calendar_month"],
        kind="mergesort",
    ).reset_index(drop=True)


def prepare_sixteen_day_anomalies(frame, settings=None):
    settings = validate_settings(settings or SubannualSettings())
    _require_columns(frame, SIXTEEN_DAY_REQUIRED_COLUMNS, "Sixteen-day frame")
    key = ["grid_id", "event_year", "role", "calendar_year", "day_of_year"]
    if frame.duplicated(key).any():
        raise ValueError("Duplicate sixteen-day role-specific observations are not allowed.")
    result = frame.copy()
    calendar = (
        result.loc[:, ["calendar_year", "day_of_year"]]
        .drop_duplicates()
        .sort_values(["calendar_year", "day_of_year"], kind="mergesort")
    )
    calendar["composite_slot"] = (
        calendar.groupby("calendar_year", sort=False).cumcount() + 1
    ).astype(np.int8)
    slot_counts = calendar.groupby("calendar_year")["composite_slot"].max()
    if slot_counts.gt(23).any():
        raise ValueError("A calendar year contains more than 23 MODIS composite slots.")
    result = result.merge(
        calendar,
        on=["calendar_year", "day_of_year"],
        how="left",
        validate="many_to_one",
    )
    values = pd.to_numeric(result["evi"], errors="coerce")
    current_eligible = (
        result["evi_valid"].fillna(False).astype(bool)
        & np.isfinite(values)
        & pd.to_numeric(result["valid_pixel_fraction"], errors="coerce").ge(
            settings.valid_pixel_threshold
        )
    )
    baseline_keys = ["grid_id", "event_year", "role", "composite_slot"]
    pre = result.loc[
        result["relative_year"].between(-3, -1) & current_eligible,
        baseline_keys,
    ].copy()
    pre["_baseline_value"] = values.loc[pre.index].to_numpy()
    baseline = (
        pre.groupby(baseline_keys, sort=False)["_baseline_value"]
        .agg([("baseline_evi_16day", "mean"), ("baseline_year_count_16day", "size")])
        .reset_index()
    )
    result = result.merge(baseline, on=baseline_keys, how="left", validate="many_to_one")
    result["baseline_year_count_16day"] = (
        result["baseline_year_count_16day"].fillna(0).astype(np.int8)
    )
    enough = result["baseline_year_count_16day"].ge(settings.min_baseline_years)
    result["baseline_evi_16day"] = result["baseline_evi_16day"].where(enough)
    result["evi_anomaly_16day"] = values - result["baseline_evi_16day"]
    result["_usable"] = (
        current_eligible
        & enough
        & np.isfinite(pd.to_numeric(result["evi_anomaly_16day"], errors="coerce"))
    )
    return result


def build_sixteen_day_matched_effects(prepared, links, settings=None):
    settings = validate_settings(settings or SubannualSettings())
    _require_columns(
        prepared,
        (
            "grid_id",
            "event_year",
            "role",
            "relative_year",
            "calendar_month",
            "composite_slot",
            "evi_anomaly_16day",
            "_usable",
        ),
        "Prepared sixteen-day frame",
    )
    _validate_links(links)
    role_key = [
        "grid_id",
        "event_year",
        "role",
        "relative_year",
        "composite_slot",
    ]
    if prepared.duplicated(role_key).any():
        raise ValueError("Duplicate sixteen-day role-period keys are not allowed.")
    periods = ["event_year", "relative_year", "calendar_month", "composite_slot"]
    treatment = prepared.loc[
        prepared["role"].eq("treatment") & prepared["_usable"].astype(bool),
        ["grid_id", *periods, "evi_anomaly_16day"],
    ].rename(
        columns={
            "grid_id": "treatment_grid_id",
            "evi_anomaly_16day": "treatment_evi_anomaly",
        }
    )
    controls = prepared.loc[
        prepared["role"].eq("control") & prepared["_usable"].astype(bool),
        ["grid_id", *periods, "evi_anomaly_16day"],
    ].rename(
        columns={
            "grid_id": "control_grid_id",
            "evi_anomaly_16day": "control_evi_anomaly_raw",
        }
    )
    if treatment.empty or controls.empty:
        return pd.DataFrame(columns=SIXTEEN_DAY_EFFECT_COLUMNS)
    linked = links.loc[:, list(dict.fromkeys(LINK_REQUIRED_COLUMNS))].merge(
        controls,
        on=["control_grid_id", "event_year"],
        how="inner",
        validate="many_to_many",
    )
    linked["_weighted_control"] = (
        linked["within_set_weight"] * linked["control_evi_anomaly_raw"]
    )
    group_keys = ["treatment_grid_id", *periods]
    control_summary = (
        linked.groupby(group_keys, sort=False)
        .agg(
            retained_control_weight=("within_set_weight", "sum"),
            _weighted_control=("_weighted_control", "sum"),
            usable_control_count=("control_grid_id", "nunique"),
        )
        .reset_index()
    )
    control_summary["control_evi_anomaly"] = (
        control_summary["_weighted_control"]
        / control_summary["retained_control_weight"]
    )
    result = treatment.merge(control_summary, on=group_keys, how="inner", validate="one_to_one")
    metadata_columns = [
        "treatment_grid_id",
        "event_year",
        "event_type",
        "macroregion",
        "event_area_m2",
        "treatment_lon",
        "treatment_lat",
    ]
    metadata = links.loc[:, metadata_columns].drop_duplicates()
    result = result.merge(
        metadata,
        on=["treatment_grid_id", "event_year"],
        how="left",
        validate="many_to_one",
    )
    result = result.loc[
        result["retained_control_weight"].ge(settings.control_weight_threshold - 1e-12)
    ].copy()
    result["matched_effect"] = (
        result["treatment_evi_anomaly"] - result["control_evi_anomaly"]
    )
    if result.duplicated(
        ["treatment_grid_id", "event_year", "relative_year", "composite_slot"]
    ).any():
        raise ValueError("Duplicate sixteen-day matched-effect keys are not allowed.")
    return result.loc[:, SIXTEEN_DAY_EFFECT_COLUMNS].sort_values(
        ["event_year", "treatment_grid_id", "relative_year", "composite_slot"],
        kind="mergesort",
    ).reset_index(drop=True)


def annualize_monthly_effects(monthly_effects, settings=None):
    settings = validate_settings(settings or SubannualSettings())
    _require_columns(monthly_effects, MONTHLY_EFFECT_COLUMNS, "Monthly matched effects")
    key = ["treatment_grid_id", "event_year", "relative_year", "calendar_month"]
    if monthly_effects.duplicated(key).any():
        raise ValueError("Duplicate monthly matched-effect primary keys are not allowed.")
    if monthly_effects.empty:
        return pd.DataFrame(columns=ANNUAL_EFFECT_COLUMNS)
    group_keys = [
        "treatment_grid_id",
        "event_year",
        "event_type",
        "macroregion",
        "event_area_m2",
        "treatment_lon",
        "treatment_lat",
        "relative_year",
    ]
    result = (
        monthly_effects.groupby(group_keys, sort=False, dropna=False)
        .agg(
            usable_months=("calendar_month", "nunique"),
            annual_matched_effect=("matched_effect", "mean"),
        )
        .reset_index()
    )
    result = result.loc[result["usable_months"].ge(settings.annual_min_months)].copy()
    return result.loc[:, ANNUAL_EFFECT_COLUMNS].sort_values(
        ["event_year", "treatment_grid_id", "relative_year"], kind="mergesort"
    ).reset_index(drop=True)


def weighted_mean_and_ess(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        return {"estimate": np.nan, "effective_n": 0.0, "weight_sum": 0.0, "n": 0}
    weight_sum = float(weights.sum())
    return {
        "estimate": float(np.dot(values, weights) / weight_sum),
        "effective_n": float(weight_sum**2 / np.dot(weights, weights)),
        "weight_sum": weight_sum,
        "n": int(len(values)),
    }


def _scoped_frames(frame):
    global_frame = frame.copy()
    global_frame["scope"] = "Global"
    regional = frame.copy()
    regional["scope"] = regional["macroregion"].astype(str)
    return pd.concat([global_frame, regional], ignore_index=True)


def _summarize(frame, value_column, group_columns, weighting):
    if weighting not in {"event_area", "equal_grid"}:
        raise ValueError("weighting must be event_area or equal_grid.")
    rows = []
    for key, group in frame.groupby(list(group_columns), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        weights = (
            pd.to_numeric(group["event_area_m2"], errors="coerce").to_numpy()
            if weighting == "event_area"
            else np.ones(len(group), dtype=np.float64)
        )
        stats = weighted_mean_and_ess(group[value_column], weights)
        row = dict(zip(group_columns, key))
        row.update(stats)
        row["weighting"] = weighting
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_monthly_effects(monthly_effects, weighting="event_area"):
    _require_columns(monthly_effects, MONTHLY_EFFECT_COLUMNS, "Monthly matched effects")
    scoped = _scoped_frames(monthly_effects)
    return _summarize(
        scoped,
        "matched_effect",
        ("scope", "event_type", "relative_year", "calendar_month"),
        weighting,
    ).sort_values(
        ["scope", "event_type", "relative_year", "calendar_month"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_annual_effects(annual_effects, weighting="event_area"):
    _require_columns(annual_effects, ANNUAL_EFFECT_COLUMNS, "Annual matched effects")
    scoped = _scoped_frames(annual_effects)
    return _summarize(
        scoped,
        "annual_matched_effect",
        ("scope", "event_type", "relative_year"),
        weighting,
    ).sort_values(
        ["scope", "event_type", "relative_year"], kind="mergesort"
    ).reset_index(drop=True)


def assign_spatial_blocks(frame, block_degrees=2.0):
    if float(block_degrees) <= 0.0:
        raise ValueError("block_degrees must be positive.")
    _require_columns(frame, ("treatment_lon", "treatment_lat"), "Effect frame")
    lon = pd.to_numeric(frame["treatment_lon"], errors="coerce").to_numpy()
    lat = pd.to_numeric(frame["treatment_lat"], errors="coerce").to_numpy()
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Spatial bootstrap coordinates must be finite.")
    bx = np.floor((lon + 180.0) / float(block_degrees)).astype(np.int32)
    by = np.floor((lat + 90.0) / float(block_degrees)).astype(np.int32)
    return pd.Series([f"{x}:{y}" for x, y in zip(bx, by)], index=frame.index)


def spatial_block_bootstrap_summary(
    frame,
    value_column,
    group_columns,
    weighting="event_area",
    block_degrees=2.0,
    replicates=1000,
    seed=20260725,
):
    if int(replicates) < 2:
        raise ValueError("replicates must be at least two.")
    _require_columns(
        frame,
        (*group_columns, value_column, "event_area_m2", "treatment_lon", "treatment_lat"),
        "Bootstrap frame",
    )
    working = frame.copy()
    working["spatial_block"] = assign_spatial_blocks(working, block_degrees)
    blocks = np.sort(working["spatial_block"].unique())
    if not len(blocks):
        return pd.DataFrame()
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(
        len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=int(replicates)
    )
    rows = []
    for key, group in working.groupby(list(group_columns), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        values = pd.to_numeric(group[value_column], errors="coerce").to_numpy()
        weights = (
            pd.to_numeric(group["event_area_m2"], errors="coerce").to_numpy()
            if weighting == "event_area"
            else np.ones(len(group), dtype=np.float64)
        )
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        stats = weighted_mean_and_ess(values[valid], weights[valid])
        block_frame = pd.DataFrame(
            {
                "spatial_block": group["spatial_block"].to_numpy()[valid],
                "weighted_value": values[valid] * weights[valid],
                "weight": weights[valid],
            }
        )
        sums = block_frame.groupby("spatial_block", sort=True).sum().reindex(
            blocks, fill_value=0.0
        )
        numerators = counts @ sums["weighted_value"].to_numpy()
        denominators = counts @ sums["weight"].to_numpy()
        estimates = np.divide(
            numerators,
            denominators,
            out=np.full(int(replicates), np.nan),
            where=denominators > 0.0,
        )
        finite = estimates[np.isfinite(estimates)]
        row = dict(zip(group_columns, key))
        row.update(stats)
        row.update(
            {
                "weighting": weighting,
                "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                "bootstrap_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                "spatial_block_count": int(len(blocks)),
                "block_degrees": float(block_degrees),
                "bootstrap_replicates": int(replicates),
                "bootstrap_seed": int(seed),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_columns), kind="mergesort").reset_index(
        drop=True
    )


CONTRAST_YEARS = {
    "pre": (-3, -2, -1),
    "t0": (0,),
    "early": (1, 2),
    "late": (3, 4, 5),
}


def build_treatment_contrasts(annual_effects):
    _require_columns(annual_effects, ANNUAL_EFFECT_COLUMNS, "Annual matched effects")
    metadata = [
        "treatment_grid_id",
        "event_year",
        "event_type",
        "macroregion",
        "event_area_m2",
        "treatment_lon",
        "treatment_lat",
    ]
    rows = []
    for contrast, years in CONTRAST_YEARS.items():
        selected = annual_effects.loc[annual_effects["relative_year"].isin(years)]
        grouped = (
            selected.groupby(metadata, sort=False, dropna=False)
            .agg(
                relative_year_count=("relative_year", "nunique"),
                contrast_effect=("annual_matched_effect", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.loc[grouped["relative_year_count"].eq(len(years))].copy()
        grouped["contrast"] = contrast
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=[*metadata, "contrast", "relative_year_count", "contrast_effect"])
    return pd.concat(rows, ignore_index=True).sort_values(
        ["event_year", "treatment_grid_id", "contrast"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_contrasts(contrasts, weighting="event_area"):
    scoped = _scoped_frames(contrasts)
    return _summarize(
        scoped,
        "contrast_effect",
        ("scope", "event_type", "contrast"),
        weighting,
    ).sort_values(["scope", "event_type", "contrast"], kind="mergesort").reset_index(
        drop=True
    )


def sensitivity_scenarios(settings=None):
    settings = validate_settings(settings or SubannualSettings())
    return (
        SensitivityScenario(
            "primary",
            settings.valid_pixel_threshold,
            settings.control_weight_threshold,
            settings.outcome,
            settings.weighting,
            settings.annual_min_months,
        ),
        SensitivityScenario(
            "valid_pixel_025",
            0.25,
            settings.control_weight_threshold,
            settings.outcome,
            settings.weighting,
            settings.annual_min_months,
        ),
        SensitivityScenario(
            "valid_pixel_075",
            0.75,
            settings.control_weight_threshold,
            settings.outcome,
            settings.weighting,
            settings.annual_min_months,
        ),
        SensitivityScenario(
            "control_weight_any",
            settings.valid_pixel_threshold,
            1e-12,
            settings.outcome,
            settings.weighting,
            settings.annual_min_months,
        ),
        SensitivityScenario(
            "control_weight_all",
            settings.valid_pixel_threshold,
            1.0,
            settings.outcome,
            settings.weighting,
            settings.annual_min_months,
        ),
        SensitivityScenario(
            "outcome_median",
            settings.valid_pixel_threshold,
            settings.control_weight_threshold,
            "median",
            settings.weighting,
            settings.annual_min_months,
        ),
        SensitivityScenario(
            "equal_grid",
            settings.valid_pixel_threshold,
            settings.control_weight_threshold,
            settings.outcome,
            "equal_grid",
            settings.annual_min_months,
        ),
        SensitivityScenario(
            "annual_months_06",
            settings.valid_pixel_threshold,
            settings.control_weight_threshold,
            settings.outcome,
            settings.weighting,
            6,
        ),
        SensitivityScenario(
            "annual_months_12",
            settings.valid_pixel_threshold,
            settings.control_weight_threshold,
            settings.outcome,
            settings.weighting,
            12,
        ),
    )


def settings_for_scenario(settings, scenario):
    return replace(
        settings,
        valid_pixel_threshold=scenario.valid_pixel_threshold,
        control_weight_threshold=scenario.control_weight_threshold,
        outcome=scenario.outcome,
        weighting=scenario.weighting,
        annual_min_months=scenario.annual_min_months,
    )
