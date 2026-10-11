"""History-qualified first-threshold event and control utilities.

This module is deliberately independent of the legacy 2001-era event catalog.
It joins the cached 1990-2000 TMF annual threshold shares to the validated
2001-2023 panel, while preserving the existing denominator and transition
semantics used by the project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline.event_response import (
    CATALOG_COLUMNS,
    EVENT_INPUT_COLUMNS,
    EventSettings,
    assign_macroregion,
    safe_share,
)


HISTORY_COLUMNS = (
    "grid_id",
    "calendar_year",
    "disturbance_share",
    "coverage_available",
)


@dataclass(frozen=True)
class FullHistorySettings:
    first_event_history_start: int = 1990
    first_event_definition: str = "first_threshold_event_over_full_available_tmf_history"
    control_history_start: int = 1990
    analysis_label: str = "TMF full-history first-threshold events"
    run_id: str = "full_history_first_event_1990"
    history_end: int = 2023
    primary_event_years: tuple[int, int] = (2004, 2018)
    tmf_raw_full_cover_value: float = 0.0029050935991108418

    def as_dict(self):
        return asdict(self)


def _numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def validate_history_table(history: pd.DataFrame, history_start: int, history_end: int):
    missing = set(HISTORY_COLUMNS) - set(history.columns)
    if missing:
        raise ValueError(f"History table lacks columns: {sorted(missing)}")
    if history.duplicated(["grid_id", "calendar_year"]).any():
        raise ValueError("History table must contain one row per grid-year.")
    years = pd.to_numeric(history["calendar_year"], errors="coerce")
    if years.dropna().empty:
        raise ValueError("History table contains no calendar years.")
    if years.min() < history_start or years.max() > history_end:
        raise ValueError("History years fall outside the configured history bounds.")
    coverage = history["coverage_available"].astype(bool)
    share = pd.to_numeric(history["disturbance_share"], errors="coerce")
    if (coverage & ~np.isfinite(share)).any():
        raise ValueError("Coverage-available history rows must have finite shares.")


def history_from_screening_cache(cache: pd.DataFrame) -> pd.DataFrame:
    """Normalize the historical-screening cache to one canonical schema."""
    required = {"grid_id", "calendar_year", "disturbance_share", "coverage_available"}
    missing = required - set(cache.columns)
    if missing:
        raise ValueError(f"Historical screening cache lacks columns: {sorted(missing)}")
    result = cache.loc[:, list(required)].copy()
    result = _numeric(result, ["grid_id", "calendar_year", "disturbance_share"])
    result["coverage_available"] = result["coverage_available"].astype(bool)
    return result.sort_values(["grid_id", "calendar_year"], kind="mergesort").reset_index(drop=True)


def _panel_annual_shares(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "grid_id",
        "year",
        "tmf_domain_pct",
        "tmf_degrad_event_pct",
        "tmf_defor_event_pct",
        "tmf_disturb_event_pct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Panel lacks event columns: {sorted(missing)}")
    result = frame.loc[:, list(required)].copy()
    result = _numeric(result, required - {"grid_id", "year"})
    result["grid_id"] = pd.to_numeric(result["grid_id"], errors="coerce").astype("Int64")
    result["calendar_year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    result["degradation_share"] = safe_share(
        result["tmf_degrad_event_pct"].to_numpy(dtype=float, na_value=np.nan),
        result["tmf_domain_pct"].to_numpy(dtype=float, na_value=np.nan),
    )
    result["deforestation_share"] = safe_share(
        result["tmf_defor_event_pct"].to_numpy(dtype=float, na_value=np.nan),
        result["tmf_domain_pct"].to_numpy(dtype=float, na_value=np.nan),
    )
    result["disturbance_share"] = safe_share(
        result["tmf_disturb_event_pct"].to_numpy(dtype=float, na_value=np.nan),
        result["tmf_domain_pct"].to_numpy(dtype=float, na_value=np.nan),
    )
    result["coverage_available"] = result["disturbance_share"].notna()
    return result.loc[:, ["grid_id", "calendar_year", "disturbance_share", "coverage_available", "degradation_share", "deforestation_share"]]


def join_historical_shares(
    panel_2001_2023: pd.DataFrame,
    historical_1990_2000: pd.DataFrame,
    history_start: int = 1990,
    history_end: int = 2023,
) -> pd.DataFrame:
    """Join historical cache and local panel using one canonical S share."""
    history = history_from_screening_cache(historical_1990_2000)
    validate_history_table(history, history_start, min(2000, history_end))
    panel = _panel_annual_shares(panel_2001_2023)
    panel_years = pd.to_numeric(panel["calendar_year"], errors="coerce")
    if panel_years.min() < 2001 or panel_years.max() > history_end:
        raise ValueError("Local panel must cover only 2001 through the configured history end.")
    historical = history.assign(
        degradation_share=np.nan,
        deforestation_share=np.nan,
    )
    combined = pd.concat(
        [historical, panel.loc[:, historical.columns]],
        ignore_index=True,
    )
    combined = combined.sort_values(["grid_id", "calendar_year"], kind="mergesort").reset_index(drop=True)
    if combined.duplicated(["grid_id", "calendar_year"]).any():
        raise ValueError("Historical and local panel contain duplicate grid-years.")
    return combined


def first_threshold_year_by_grid(
    annual_history: pd.DataFrame,
    threshold: float,
    history_start: int,
    history_end: int,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Return each grid's first complete-history threshold crossing."""
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be within (0, 1].")
    validate_history_table(annual_history, history_start, history_end)
    rows = []
    for grid_id, group in annual_history.groupby("grid_id", sort=True):
        group = group.sort_values("calendar_year", kind="mergesort")
        years = group["calendar_year"].to_numpy(dtype=np.int64)
        expected = np.arange(history_start, history_end + 1, dtype=np.int64)
        observed = group["coverage_available"].astype(bool).to_numpy()
        complete = len(years) == len(expected) and np.array_equal(years, expected) and observed.all()
        valid = observed & np.isfinite(pd.to_numeric(group["disturbance_share"], errors="coerce").to_numpy(dtype=float))
        share = pd.to_numeric(group["disturbance_share"], errors="coerce").to_numpy(dtype=float)
        crossing_years = years[valid & (share >= threshold)]
        first_year = int(crossing_years[0]) if len(crossing_years) else np.nan
        rows.append(
            {
                "grid_id": int(grid_id),
                "first_threshold_year": first_year,
                "history_complete": bool(complete),
                "history_years_available": int(observed.sum()),
                "history_years_expected": int(len(expected)),
                "history_max_disturbance_share": float(np.nanmax(share)) if np.isfinite(share).any() else np.nan,
                "history_threshold_event": bool(len(crossing_years)),
            }
        )
    result = pd.DataFrame(rows)
    if require_complete and not result["history_complete"].all():
        incomplete = result.loc[~result["history_complete"], "grid_id"].tolist()[:5]
        raise ValueError(f"Incomplete history for grids, examples={incomplete}")
    return result


def _event_catalog_from_panel(
    panel: pd.DataFrame,
    event_threshold: float,
    settings: EventSettings,
    history_flags: pd.DataFrame,
    source_part: str,
) -> pd.DataFrame:
    missing = set(EVENT_INPUT_COLUMNS) - set(panel.columns)
    if missing:
        raise ValueError(f"Event input lacks columns: {sorted(missing)}")
    working = panel.loc[:, EVENT_INPUT_COLUMNS].copy()
    for column in EVENT_INPUT_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    domain = working["tmf_domain_pct"].to_numpy(dtype=float, na_value=np.nan)
    working["disturbance_share"] = safe_share(working["tmf_disturb_event_pct"], domain)
    working["degradation_share"] = safe_share(working["tmf_degrad_event_pct"], domain)
    working["deforestation_share"] = safe_share(working["tmf_defor_event_pct"], domain)
    working["regrowth_share"] = safe_share(working["tmf_regrow_event_pct"], domain)
    first_2001 = (
        working.loc[working["disturbance_share"].ge(float(event_threshold))]
        .sort_values(["grid_id", "year"], kind="mergesort")
        .groupby("grid_id", as_index=False, sort=True)
        .first()
        .rename(columns={"year": "event_year"})
    )
    if first_2001.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS + ("first_threshold_year", "history_complete", "history_threshold_event"))
    flags = history_flags.loc[:, ["grid_id", "first_threshold_year", "history_complete", "history_threshold_event"]]
    first = first_2001.merge(flags, on="grid_id", how="left", validate="one_to_one")
    first = first.loc[
        first["history_complete"].astype(bool)
        & first["first_threshold_year"].eq(first["event_year"])
    ].copy()
    if first.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS + ("first_threshold_year", "history_complete", "history_threshold_event"))

    all_significant = working.loc[working["disturbance_share"].ge(float(event_threshold)), ["grid_id", "year"]]
    later = all_significant.merge(first[["grid_id", "event_year"]], on="grid_id", how="inner", validate="many_to_one")
    later = later.loc[later["year"].gt(later["event_year"]) & later["year"].le(later["event_year"] + 5)]
    next_year = later.groupby("grid_id", sort=True)["year"].min()
    first["next_significant_event_year"] = first["grid_id"].map(next_year)
    first["repeat_within_5y"] = first["next_significant_event_year"].notna()
    first["censor_relative_year"] = first["next_significant_event_year"] - first["event_year"]
    static_domain = working.groupby("grid_id", sort=True)["tmf_domain_pct"].max()
    first["analysis_area_m2"] = first["cell_area_m2"] * first["grid_id"].map(static_domain) / settings.tmf_raw_full_cover_value
    first["event_area_m2"] = first["cell_area_m2"] * first["tmf_disturb_event_pct"] / settings.tmf_raw_full_cover_value
    first["event_type"] = np.where(first["deforestation_share"].ge(first["degradation_share"]), "deforestation", "degradation")
    first["event_threshold"] = float(event_threshold)
    first["primary_window_eligible"] = first["event_year"].between(settings.primary_event_years[0], settings.primary_event_years[1])
    first["macroregion"] = assign_macroregion(first["lon"], first["lat"])
    first["source_part"] = str(source_part)
    return first.loc[:, CATALOG_COLUMNS + ("first_threshold_year", "history_complete", "history_threshold_event")].sort_values("grid_id", kind="mergesort").reset_index(drop=True)


def derive_full_history_catalog(
    panel_2001_2023: pd.DataFrame,
    historical_1990_2000: pd.DataFrame,
    event_threshold: float,
    settings: EventSettings | None = None,
    history_settings: FullHistorySettings | None = None,
    source_part: str = "full_history",
) -> pd.DataFrame:
    """Derive first events over 1990-2023, retaining only 2004-2018 events."""
    settings = settings or EventSettings()
    history_settings = history_settings or FullHistorySettings()
    # The screening table may contain both candidate and control grids.  Event
    # discovery is defined only for the panel grids supplied here; restrict the
    # historical rows before the completeness check so unrelated controls do
    # not appear as 11-year, pre-panel histories.
    panel_grid_ids = set(pd.to_numeric(panel_2001_2023["grid_id"], errors="coerce").dropna().astype(int))
    historical_1990_2000 = historical_1990_2000.loc[
        pd.to_numeric(historical_1990_2000["grid_id"], errors="coerce").isin(panel_grid_ids)
    ].copy()
    annual = join_historical_shares(
        panel_2001_2023,
        historical_1990_2000,
        history_settings.first_event_history_start,
        history_settings.history_end,
    )
    flags = first_threshold_year_by_grid(
        annual,
        event_threshold,
        history_settings.first_event_history_start,
        history_settings.history_end,
        require_complete=True,
    )
    result = _event_catalog_from_panel(panel_2001_2023, event_threshold, settings, flags, source_part)
    if not result.empty:
        result["history_start"] = int(history_settings.first_event_history_start)
        result["history_end"] = int(history_settings.history_end)
        result["first_event_definition"] = history_settings.first_event_definition
        result["prior_threshold_before_event"] = False
        result["prior_threshold_year"] = np.nan
        validate_first_event_catalog(
            result,
            annual,
            event_threshold,
            history_settings.first_event_history_start,
            history_settings.history_end,
            settings.primary_event_years,
        )
    return result


def validate_first_event_catalog(
    catalog: pd.DataFrame,
    annual_history: pd.DataFrame,
    threshold: float,
    history_start: int,
    history_end: int,
    event_years: tuple[int, int] = (2004, 2018),
):
    required = {"grid_id", "event_year", "first_threshold_year", "event_type", "degradation_share", "deforestation_share", "history_complete"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Catalog lacks validation columns: {sorted(missing)}")
    if catalog["grid_id"].duplicated().any():
        raise AssertionError("Full-history catalog must be unique by grid_id.")
    if not catalog["event_year"].between(event_years[0], event_years[1]).all():
        raise AssertionError("Full-history event years fall outside the primary window.")
    if not catalog["first_threshold_year"].eq(catalog["event_year"]).all():
        raise AssertionError("Catalog event year is not the first complete-history threshold year.")
    if not catalog["history_complete"].astype(bool).all():
        raise AssertionError("Full-history catalog contains incomplete history.")
    prior = annual_history.merge(catalog[["grid_id", "event_year"]], on="grid_id", how="inner", validate="many_to_one")
    prior = prior.loc[prior["calendar_year"] < prior["event_year"]]
    if (pd.to_numeric(prior["disturbance_share"], errors="coerce") >= float(threshold)).any():
        raise AssertionError("At least one retained event has a prior threshold crossing.")
    purity = np.maximum(
        pd.to_numeric(catalog["degradation_share"], errors="coerce"),
        pd.to_numeric(catalog["deforestation_share"], errors="coerce"),
    ) / pd.to_numeric(catalog["disturbance_share"], errors="coerce")
    if not purity.between(0.5, 1.0).all():
        raise AssertionError("Event purity is outside [0.5, 1].")
    expected_type = np.where(
        pd.to_numeric(catalog["deforestation_share"], errors="coerce") >= pd.to_numeric(catalog["degradation_share"], errors="coerce"),
        "deforestation",
        "degradation",
    )
    if not np.array_equal(catalog["event_type"].to_numpy(), expected_type):
        raise AssertionError("Event type is not uniquely determined by D/F at t0.")


def add_history_qualification(
    features: pd.DataFrame,
    history_flags: pd.DataFrame,
    history_start: int,
    event_threshold: float,
) -> pd.DataFrame:
    """Attach prior-history and full stable-through-follow-up flags to features."""
    required = {"grid_id", "event_year", "max_disturbance_share_through_followup", "coverage_eligible", "matching_covariates_complete"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature table lacks history qualification inputs: {sorted(missing)}")
    flags = history_flags.loc[:, ["grid_id", "first_threshold_year", "history_complete"]].drop_duplicates("grid_id")
    result = features.merge(flags, on="grid_id", how="left", validate="many_to_one")
    result["history_start"] = int(history_start)
    result["prior_threshold_before_event"] = (
        pd.to_numeric(result["first_threshold_year"], errors="coerce") < pd.to_numeric(result["event_year"], errors="coerce")
    )
    current_max = pd.to_numeric(result["max_disturbance_share_through_followup"], errors="coerce")
    result["stable_through_followup_full_history"] = (
        result["history_complete"].astype(bool)
        & ~result["prior_threshold_before_event"].astype(bool)
        & current_max.notna()
        & current_max.lt(float(event_threshold))
    )
    result["pre_event_risk_eligible_full_history"] = (
        result["history_complete"].astype(bool)
        & ~result["prior_threshold_before_event"].astype(bool)
        & result["coverage_eligible"].astype(bool)
        & result["matching_covariates_complete"].astype(bool)
    )
    return result


def validate_stable_controls(features: pd.DataFrame, event_threshold: float):
    required = {"stable_through_followup_full_history", "history_complete", "prior_threshold_before_event"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Stable-control table lacks columns: {sorted(missing)}")
    if features.empty:
        return
    if not features["history_complete"].astype(bool).all():
        raise AssertionError("Stable controls include incomplete historical coverage.")
    if features["prior_threshold_before_event"].astype(bool).any():
        raise AssertionError("Stable controls include a prior historical threshold event.")
    if not features["stable_through_followup_full_history"].astype(bool).all():
        raise AssertionError("Stable controls fail full-history follow-up stability.")


def validate_pre_event_controls(features: pd.DataFrame):
    required = {"history_complete", "prior_threshold_before_event", "pre_event_risk_eligible_full_history"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Risk-set table lacks columns: {sorted(missing)}")
    if features.empty:
        return
    if not features["history_complete"].astype(bool).all():
        raise AssertionError("Pre-event controls include incomplete historical coverage.")
    if features["prior_threshold_before_event"].astype(bool).any():
        raise AssertionError("Pre-event controls include a prior threshold crossing.")


__all__ = [
    "FullHistorySettings",
    "add_history_qualification",
    "derive_full_history_catalog",
    "first_threshold_year_by_grid",
    "history_from_screening_cache",
    "join_historical_shares",
    "validate_first_event_catalog",
    "validate_pre_event_controls",
    "validate_stable_controls",
]
