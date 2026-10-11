#!/usr/bin/env python3
"""Recompute pre-event, IPCW, regional, and classification sensitivities."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "02_analysis/integrated/TropicalRainforest_integrated_analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from pipeline.full_history_events import add_history_qualification
from pipeline.full_history_sensitivities import (
    evaluate_classifier_validations,
    validate_pre_event_pair_history,
)
from pipeline.matched_control import match_risk_set
from pipeline.matched_estimation import MatchedControlSettings
from pipeline.revision_v1_analysis import (
    _build_pre_event_dynamic_window,
    _ipcw_event_type_contrast,
    _regional_contrasts,
    _spatial_block_distribution,
)


def _load_config() -> dict:
    return json.loads((RUN_ROOT / "config.json").read_text(encoding="utf-8"))


def _settings(config: dict) -> MatchedControlSettings:
    return MatchedControlSettings(
        event_thresholds=tuple(config["event_thresholds"]),
        obs_thresholds=tuple(config["obs_thresholds"]),
        primary_event_threshold=config["primary_event_threshold"],
        primary_obs_threshold=config["primary_obs_threshold"],
        primary_radius_km=config["primary_radius_km"],
        fallback_radius_km=config["fallback_radius_km"],
        max_controls=config["max_controls"],
        relative_years=tuple(config["relative_years"]),
        primary_event_years=tuple(config["primary_event_years"]),
        analysis_years=tuple(config["analysis_years"]),
        bootstrap_replicates=config["bootstrap_replicates"],
        bootstrap_seed=config["bootstrap_seed"],
        tmf_raw_full_cover_value=config["tmf_raw_full_cover_value"],
    )


def _read_features(config: dict, year: int, grid_ids: set[int]) -> pd.DataFrame:
    root = Path(config["legacy_output_root"]).parent / "07_matched_control" / "work" / "features" / "obs12" / f"year{int(year)}"
    paths = sorted(root.glob("part_*.parquet"))
    if len(paths) != 29:
        raise RuntimeError(f"Expected 29 feature parts for {year}, found {len(paths)}")
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        selected = frame.loc[frame["grid_id"].isin(grid_ids)].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError(f"No matching features found for {year}")
    return pd.concat(frames, ignore_index=True)


def _build_pre_event_pairs(config: dict, catalog: pd.DataFrame, history_flags: pd.DataFrame, settings: MatchedControlSettings):
    all_ids = set(history_flags["grid_id"].astype(int))
    pair_frames, treatment_frames = [], []
    diagnostics = []
    for event_year in range(settings.primary_event_years[0], settings.primary_event_years[1] + 1):
        event_rows = catalog.loc[catalog["event_year"].eq(event_year)].copy()
        if event_rows.empty:
            continue
        features = _read_features(config, event_year, all_ids)
        qualified = add_history_qualification(
            features,
            history_flags,
            config["control_history_start"],
            settings.primary_event_threshold,
        )
        treatment_features = qualified.loc[qualified["grid_id"].isin(event_rows["grid_id"])].merge(
            event_rows,
            on=["grid_id", "event_year"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_event"),
        )
        treatment_ids = set(treatment_features["grid_id"].astype(int))
        controls = qualified.loc[
            qualified["history_complete"].astype(bool)
            & ~qualified["prior_threshold_before_event"].astype(bool)
            & qualified["coverage_eligible"].astype(bool)
            & qualified["matching_covariates_complete"].astype(bool)
            & ~qualified["grid_id"].isin(treatment_ids)
        ].copy()
        pairs = match_risk_set(treatment_features, controls, settings)
        if pairs.empty:
            continue
        first_map = history_flags.set_index("grid_id")["first_threshold_year"]
        control_first = pairs["control_grid_id"].map(first_map)
        crossing = pd.to_numeric(control_first, errors="coerce") - pd.to_numeric(pairs["event_year"], errors="coerce")
        pairs["control_censor_relative_year"] = crossing.where(crossing.between(0, 5))
        checked = validate_pre_event_pair_history(pairs, history_flags)
        pair_frames.append(checked)
        matched_ids = set(checked["treatment_grid_id"].astype(int))
        treatment_frames.append(treatment_features.loc[treatment_features["grid_id"].isin(matched_ids)].copy())
        diagnostics.append(
            {
                "event_year": int(event_year),
                "eligible_treatments": int(len(treatment_features)),
                "eligible_controls": int(len(controls)),
                "matched_treatments": int(checked["treatment_grid_id"].nunique()),
                "pair_links": int(len(checked)),
                "links_with_dynamic_censor": int(checked["control_censor_relative_year"].notna().sum()),
            }
        )
    if not pair_frames:
        raise RuntimeError("No pre-event pairs were produced")
    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(["event_year", "treatment_grid_id", "match_rank"], kind="mergesort").reset_index(drop=True)
    treatments = pd.concat(treatment_frames, ignore_index=True).drop_duplicates(["grid_id", "event_year"])
    return pairs, treatments, pd.DataFrame(diagnostics)


def _event_table(treatments: pd.DataFrame) -> pd.DataFrame:
    table = treatments.rename(columns={"grid_id": "treatment_grid_id"}).copy()
    d = pd.to_numeric(table["degradation_share"], errors="coerce")
    f = pd.to_numeric(table["deforestation_share"], errors="coerce")
    table["event_composition_purity"] = np.maximum(d, f) / (d + f)
    return table


def _write(frame: pd.DataFrame, path: Path, run_id: str):
    frame = frame.copy()
    frame.insert(0, "run_id", run_id)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def main():
    config = _load_config()
    settings = _settings(config)
    catalog = pd.read_parquet(RUN_ROOT / "history_qualified_candidates.parquet")
    history_flags = pd.read_parquet(RUN_ROOT / "history_qualified_controls.parquet")
    pairs, treatments, pair_diag = _build_pre_event_pairs(config, catalog, history_flags, settings)
    annual_ids = set(pairs["treatment_grid_id"].astype(int)) | set(pairs["control_grid_id"].astype(int))
    # Reuse the main runner's loader without importing its CLI.
    from run_expanded_matching import load_annual_outcomes

    annual = load_annual_outcomes(config, annual_ids)
    pre_window = _build_pre_event_dynamic_window(annual, treatments, pairs, settings)
    pre_events = _event_table(treatments)
    from pipeline.matched_estimation import bootstrap_standardized_event_type_contrast, summarize_standardized_event_type_contrast

    pre_summary, pre_diag, _ = summarize_standardized_event_type_contrast(pre_window, pre_events, metric="evi_mean", relative_years=settings.relative_years)
    pre_bootstrap = bootstrap_standardized_event_type_contrast(pre_window, pre_events, metric="evi_mean", relative_years=settings.relative_years, block_degrees=config["bootstrap_block_degrees"], replicates=settings.bootstrap_replicates, seed=settings.bootstrap_seed)

    main_window = pd.read_parquet(RUN_ROOT / "fixed_matched_event_window.parquet")
    main_treatments = pd.read_parquet(RUN_ROOT / "matched_treatments.parquet")
    main_events = _event_table(main_treatments)
    wide = main_window.pivot(index=["treatment_grid_id", "event_year", "event_type", "macroregion", "lon", "lat"], columns="relative_year", values="pair_effect_evi_mean").reset_index()
    wide.columns = [f"t{int(c)}" if isinstance(c, (int, np.integer)) and c >= 0 else f"tm{abs(int(c))}" if isinstance(c, (int, np.integer)) else c for c in wide.columns]
    classification = evaluate_classifier_validations(wide, {"t0_only": ["t0"], "t0_to_t5_trajectory": ["t0", "t1", "t2", "t3", "t4", "t5"]}, seed=settings.bootstrap_seed)
    regional, regional_diag = _regional_contrasts(main_window, main_events, settings)
    block_distribution = _spatial_block_distribution(main_window, main_events, settings)
    ipcw_contrast, ipcw_weights, ipcw_diag = _ipcw_event_type_contrast(main_window, main_events, settings)

    _write(pairs, RUN_ROOT / "pre_event_dynamic_control_pairs.parquet", config["run_id"])
    _write(pre_window, RUN_ROOT / "pre_event_dynamic_control_window.parquet", config["run_id"])
    _write(pre_summary, RUN_ROOT / "pre_event_dynamic_standardized_contrast.csv", config["run_id"])
    _write(pre_bootstrap, RUN_ROOT / "pre_event_dynamic_standardized_bootstrap.csv", config["run_id"])
    _write(classification, RUN_ROOT / "trajectory_classification.csv", config["run_id"])
    _write(regional, RUN_ROOT / "regional_event_type_contrast.csv", config["run_id"])
    _write(block_distribution, RUN_ROOT / "spatial_block_distribution.csv", config["run_id"])
    _write(ipcw_contrast, RUN_ROOT / "ipcw_censoring_sensitivity.csv", config["run_id"])
    _write(ipcw_weights, RUN_ROOT / "ipcw_censoring_weights.parquet", config["run_id"])
    _write(ipcw_diag, RUN_ROOT / "ipcw_censoring_model.csv", config["run_id"])
    _write(pair_diag, RUN_ROOT / "pre_event_matching_by_year.csv", config["run_id"])
    report = {
        "run_id": config["run_id"],
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "pre_event": {
            "matched_treatments": int(pairs["treatment_grid_id"].nunique()),
            "pair_links": int(len(pairs)),
            "links_with_dynamic_censor": int(pairs["control_censor_relative_year"].notna().sum()),
            "t5_valid": int(pre_window.loc[pre_window["relative_year"].eq(5), "pair_effect_evi_mean"].notna().sum()),
            "standardization": pre_diag,
        },
        "classification": classification.to_dict(orient="records"),
        "regional_rows": int(len(regional)),
        "ipcw_rows": int(len(ipcw_contrast)),
        "ipcw_weight_range": [float(ipcw_weights["stabilized_ipcw"].min()), float(ipcw_weights["stabilized_ipcw"].max())],
        "ipcw_diagnostics_rows": int(len(ipcw_diag)),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "validation_pass": True,
    }
    (RUN_ROOT / "validation_phase4_5.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
