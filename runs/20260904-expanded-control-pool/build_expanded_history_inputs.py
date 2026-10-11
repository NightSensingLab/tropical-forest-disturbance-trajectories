#!/usr/bin/env python3
"""Build full-history treatment candidates and expanded-control flags."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "02_analysis/integrated/TropicalRainforest_integrated_analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from pipeline.event_response import EVENT_INPUT_COLUMNS, EventSettings
from pipeline.full_history_events import (
    FullHistorySettings,
    derive_full_history_catalog,
    first_threshold_year_by_grid,
    history_from_screening_cache,
    join_historical_shares,
    validate_first_event_catalog,
)


HISTORY_YEARS = tuple(range(1990, 2001))
FEATURE_STATUS_COLUMNS = ["grid_id", "max_disturbance_share_through_2023"]
HISTORY_EVENT_COLUMNS = [f"historical_event_{year}" for year in HISTORY_YEARS]
HISTORY_VALID_COLUMNS = [f"historical_valid_{year}" for year in HISTORY_YEARS]


def _config() -> dict:
    return json.loads((RUN_ROOT / "config.json").read_text(encoding="utf-8"))


def _settings(config: dict) -> tuple[EventSettings, FullHistorySettings]:
    event = EventSettings(
        event_thresholds=tuple(config["event_thresholds"]),
        obs_thresholds=tuple(config["obs_thresholds"]),
        primary_event_threshold=config["primary_event_threshold"],
        primary_obs_threshold=config["primary_obs_threshold"],
        relative_years=tuple(config["relative_years"]),
        primary_event_years=tuple(config["primary_event_years"]),
        analysis_years=tuple(config["analysis_years"]),
        tmf_raw_full_cover_value=config["tmf_raw_full_cover_value"],
    )
    history = FullHistorySettings(
        first_event_history_start=config["first_event_history_start"],
        first_event_definition=config["first_event_definition"],
        control_history_start=config["control_history_start"],
        analysis_label=config["analysis_label"],
        run_id=config["run_id"],
        history_end=config["history_end"],
        primary_event_years=tuple(config["primary_event_years"]),
        tmf_raw_full_cover_value=config["tmf_raw_full_cover_value"],
    )
    return event, history


def load_candidates(config: dict) -> pd.DataFrame:
    columns = [
        "grid_id", "grid_row", "grid_col", "lon", "lat", "cell_area_m2", "analysis_area_m2",
        "macroregion", "event_threshold", "event_year", "event_type", "disturbance_share",
        "degradation_share", "deforestation_share", "regrowth_share", "event_area_m2",
        "primary_window_eligible", "source_part",
    ]
    catalog = pd.read_parquet(config["legacy_catalog"], columns=columns)
    candidates = catalog.loc[
        np.isclose(pd.to_numeric(catalog["event_threshold"], errors="coerce"), config["primary_event_threshold"])
        & catalog["primary_window_eligible"].astype(bool)
    ].copy()
    candidates = candidates.drop_duplicates("grid_id", keep="first").sort_values("grid_id", kind="mergesort").reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError("No primary candidate events were found")
    return candidates


def load_local_panel(config: dict, grid_ids: set[int]) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(config["stage03_dir"]).glob("part_*.parquet")):
        frame = pd.read_parquet(path, columns=list(EVENT_INPUT_COLUMNS), filters=[("year", "in", list(range(2001, 2024)))])
        selected = frame.loc[frame["grid_id"].astype(int).isin(grid_ids)].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError("No Stage 03 rows found for candidate grids")
    result = pd.concat(frames, ignore_index=True).sort_values(["grid_id", "year"], kind="mergesort").reset_index(drop=True)
    if result.duplicated(["grid_id", "year"]).any():
        raise ValueError("Candidate panel contains duplicate grid-year rows")
    return result


def _summary_from_long(history: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for grid_id, group in history.groupby("grid_id", sort=True):
        group = group.sort_values("calendar_year", kind="mergesort")
        years = pd.to_numeric(group["calendar_year"], errors="coerce").to_numpy(dtype=float)
        shares = pd.to_numeric(group["disturbance_share"], errors="coerce").to_numpy(dtype=float)
        available = group["coverage_available"].astype(bool).to_numpy()
        crossing = np.isfinite(shares) & (shares >= float(threshold)) & available
        crossing_years = years[crossing]
        rows.append({
            "grid_id": int(grid_id),
            "first_threshold_year": int(crossing_years[0]) if len(crossing_years) else np.nan,
            "history_complete_pre2001": bool(len(years) == len(HISTORY_YEARS) and np.array_equal(years, np.asarray(HISTORY_YEARS, dtype=float)) and available.all()),
            "history_years_available_pre2001": int(available.sum()),
            "history_threshold_event": bool(len(crossing_years)),
            "history_max_disturbance_share_pre2001": float(np.nanmax(shares)) if np.isfinite(shares).any() else np.nan,
        })
    return pd.DataFrame(rows)


def _summary_from_wide(path: Path, threshold: float) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    valid_columns = HISTORY_VALID_COLUMNS if set(HISTORY_VALID_COLUMNS).issubset(available) else ["historical_valid_1990"] * len(HISTORY_YEARS)
    frame = pd.read_parquet(path, columns=["grid_id", *HISTORY_EVENT_COLUMNS, *sorted(set(valid_columns))])
    if frame["grid_id"].duplicated().any():
        raise ValueError(f"Duplicate grid IDs in history part: {path}")
    event = frame.loc[:, HISTORY_EVENT_COLUMNS].to_numpy(dtype=float)
    valid = frame.loc[:, valid_columns].to_numpy(dtype=float)
    shares = np.divide(event, valid, out=np.full(event.shape, np.nan), where=np.isfinite(valid) & (valid > 0.0))
    crossing = np.isfinite(shares) & (shares >= float(threshold))
    has = crossing.any(axis=1)
    first_index = np.argmax(crossing, axis=1)
    first_year = np.where(has, np.asarray(HISTORY_YEARS, dtype=float)[first_index], np.nan)
    return pd.DataFrame({
        "grid_id": frame["grid_id"].astype(int).to_numpy(),
        "first_threshold_year": first_year,
        "history_complete_pre2001": np.isfinite(valid).all(axis=1) & (valid > 0.0).all(axis=1),
        "history_years_available_pre2001": np.isfinite(valid).sum(axis=1).astype(int),
        "history_threshold_event": has,
        "history_max_disturbance_share_pre2001": np.nanmax(shares, axis=1),
    })


def load_control_history_summary(config: dict, control_ids: set[int]) -> pd.DataFrame:
    screening = Path(config["historical_screening_root"]) / "historical_annual_event_shares.parquet"
    long = history_from_screening_cache(pd.read_parquet(screening))
    long = long.loc[long["grid_id"].astype(int).isin(control_ids) & long["calendar_year"].between(1990, 2000)].copy()
    summaries = []
    if not long.empty:
        summaries.append(_summary_from_long(long, config["primary_event_threshold"]))
    for directory in (
        Path(config["restricted_run_root"]) / "control_history_extraction" / "point_parts",
        Path(config["control_history_extraction_root"]) / "point_parts",
        Path(config["control_history_extraction_root"]) / "compute_parts",
        Path(config["control_history_extraction_root"]) / "compute_parts_final",
    ):
        for path in sorted(directory.glob("*.parquet")):
            summary = _summary_from_wide(path, config["primary_event_threshold"])
            summary = summary.loc[summary["grid_id"].isin(control_ids)].copy()
            if not summary.empty:
                summaries.append(summary)
    if not summaries:
        raise RuntimeError("No control history records were found")
    combined = pd.concat(summaries, ignore_index=True)
    combined = combined.loc[combined["grid_id"].isin(control_ids)].sort_values("grid_id", kind="mergesort")
    combined = combined.drop_duplicates("grid_id", keep="first").reset_index(drop=True)
    missing = sorted(control_ids - set(combined["grid_id"].astype(int)))
    if missing:
        raise RuntimeError(f"Control history missing {len(missing)} grids; examples={missing[:10]}")
    if not combined["grid_id"].is_unique:
        raise ValueError("Control history summary is not unique by grid_id")
    return combined


def load_local_completeness(config: dict, universe: pd.DataFrame) -> pd.DataFrame:
    wanted = set(universe["grid_id"].astype(int))
    frames = []
    root = Path(config["feature_root"])
    for year in range(config["feature_years"][0], config["feature_years"][1] + 1):
        for path in sorted((root / f"year{year}").glob("part_*.parquet")):
            frame = pd.read_parquet(path, columns=FEATURE_STATUS_COLUMNS)
            selected = frame.loc[frame["grid_id"].astype(int).isin(wanted)].copy()
            if not selected.empty:
                frames.append(selected)
    status = pd.concat(frames, ignore_index=True).drop_duplicates("grid_id", keep="first")
    status["local_history_complete_2001_2023"] = pd.to_numeric(status["max_disturbance_share_through_2023"], errors="coerce").notna()
    return status.loc[:, ["grid_id", "local_history_complete_2001_2023"]]


def main() -> None:
    config = _config()
    event_settings, history_settings = _settings(config)
    candidates = load_candidates(config)
    candidate_ids = set(candidates["grid_id"].astype(int))
    local_candidates = load_local_panel(config, candidate_ids)
    screening = pd.read_parquet(Path(config["historical_screening_root"]) / "historical_annual_event_shares.parquet")
    candidate_history = history_from_screening_cache(screening)
    candidate_history = candidate_history.loc[candidate_history["grid_id"].astype(int).isin(candidate_ids) & candidate_history["calendar_year"].between(1990, 2000)].copy()
    annual = join_historical_shares(local_candidates, candidate_history, 1990, 2023)
    annual.to_parquet(RUN_ROOT / "history_qualified_annual.parquet", index=False)
    candidate_flags = first_threshold_year_by_grid(annual, config["primary_event_threshold"], 1990, 2023, require_complete=True)
    catalog = derive_full_history_catalog(local_candidates, candidate_history, config["primary_event_threshold"], event_settings, history_settings, source_part="expanded_full_history")
    if catalog.empty:
        raise RuntimeError("No full-history candidates were retained")
    validate_first_event_catalog(catalog, annual, config["primary_event_threshold"], 1990, 2023, tuple(config["primary_event_years"]))
    catalog.to_parquet(RUN_ROOT / "history_qualified_candidates.parquet", index=False)

    universe = pd.read_parquet(RUN_ROOT / "expanded_control_grid_universe.parquet")
    control_ids = set(universe["grid_id"].astype(int))
    control_history = load_control_history_summary(config, control_ids)
    local_status = load_local_completeness(config, universe)
    control_flags = control_history.merge(local_status, on="grid_id", how="left", validate="one_to_one")
    control_flags["history_complete"] = control_flags["history_complete_pre2001"].fillna(False).astype(bool) & control_flags["local_history_complete_2001_2023"].fillna(False).astype(bool)
    control_flags["history_threshold_event"] = control_flags["history_threshold_event"].astype(bool)
    control_flags = control_flags.loc[:, ["grid_id", "first_threshold_year", "history_complete", "history_threshold_event", "history_years_available_pre2001", "history_max_disturbance_share_pre2001", "local_history_complete_2001_2023"]]
    if control_flags["history_complete"].isna().any() or not control_flags["history_complete"].all():
        bad = control_flags.loc[~control_flags["history_complete"], "grid_id"].tolist()[:10]
        raise RuntimeError(f"Expanded control history incomplete, examples={bad}")
    candidate_flag_output = candidate_flags.loc[:, ["grid_id", "first_threshold_year", "history_complete", "history_threshold_event", "history_years_available", "history_max_disturbance_share"]].copy()
    flags = pd.concat([control_flags, candidate_flag_output], ignore_index=True).drop_duplicates("grid_id", keep="last").sort_values("grid_id", kind="mergesort").reset_index(drop=True)
    flags.to_parquet(RUN_ROOT / "history_qualified_controls.parquet", index=False)
    status_counts = control_flags.assign(
        pre2001_threshold_event=control_flags["history_threshold_event"].astype(bool),
    ).groupby("pre2001_threshold_event", as_index=False).size().rename(columns={"size": "n_grids"})
    status_counts.to_csv(RUN_ROOT / "expanded_control_history_status.csv", index=False)
    report = {
        "run_id": config["run_id"],
        "candidate_count_legacy_primary": int(len(candidates)),
        "full_history_candidate_count": int(len(catalog)),
        "candidate_excluded_by_prior_history": int(len(candidates) - len(catalog)),
        "expanded_universe_count": int(len(universe)),
        "expanded_control_flags_count": int(len(control_flags)),
        "expanded_control_pre2001_threshold_event_count": int(control_flags["history_threshold_event"].sum()),
        "expanded_control_pre2001_threshold_event_share": float(control_flags["history_threshold_event"].mean()),
        "all_history_flags_count": int(len(flags)),
        "all_history_flags_unique": bool(flags["grid_id"].is_unique),
        "validation_pass": True,
    }
    (RUN_ROOT / "history_input_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
