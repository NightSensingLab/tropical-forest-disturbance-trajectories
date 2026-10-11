#!/usr/bin/env python3
"""Run fresh full-history matching and primary dynamic EVI analysis."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "02_analysis/integrated/TropicalRainforest_integrated_analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from pipeline.full_history_events import (
    add_history_qualification,
    validate_first_event_catalog,
    validate_stable_controls,
)
from pipeline.full_history_sensitivities import capped_standardized_weights, validate_dynamic_window
from pipeline.matched_control import MATCHING_COVARIATES, MatchedControlSettings, match_risk_set
from pipeline.matched_estimation import (
    bootstrap_standardized_event_type_contrast,
    build_matched_event_window,
    common_support_weights,
    spatial_block_bootstrap,
    summarize_matched_response,
    summarize_standardized_event_type_contrast,
)


FEATURE_COLUMNS = [
    "grid_id", "event_year", "grid_row", "grid_col", "lon", "lat", "cell_area_m2",
    *MATCHING_COVARIATES, "max_disturbance_share_through_followup",
    "max_disturbance_share_through_2023", "stable_through_followup", "stable_through_2023",
    "event_year_usable", "post_usable_n", "coverage_eligible", "macroregion",
    "matching_covariates_complete",
]
OUTCOME_COLUMNS = ["grid_id", "year", "evi_mean", "evi_min", "evi_max", "evi_sd", "valid_obs"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_config():
    return json.loads((RUN_ROOT / "config.json").read_text(encoding="utf-8"))


def settings_from_config(config):
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


def read_features_for_year(config, event_year, grid_ids):
    root = Path(config["legacy_output_root"]).parent / "07_matched_control" / "work" / "features" / "obs12" / f"year{int(event_year)}"
    paths = sorted(root.glob("part_*.parquet"))
    if len(paths) != 29:
        raise RuntimeError(f"Expected 29 feature parts for {event_year}, found {len(paths)}")
    wanted = set(map(int, grid_ids))
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=FEATURE_COLUMNS)
        selected = frame.loc[frame["grid_id"].isin(wanted)].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError(f"No feature rows found for event year {event_year}")
    return pd.concat(frames, ignore_index=True)


def load_annual_outcomes(config, grid_ids):
    wanted = set(map(int, grid_ids))
    frames = []
    for path in sorted(Path(config["stage03_dir"]).glob("part_*.parquet")):
        frame = pd.read_parquet(path, columns=OUTCOME_COLUMNS)
        selected = frame.loc[frame["grid_id"].isin(wanted)].copy()
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError("No annual outcome rows for matched IDs")
    return pd.concat(frames, ignore_index=True)


def event_table(treatments):
    result = treatments.rename(columns={"grid_id": "treatment_grid_id"}).copy()
    d = pd.to_numeric(result["degradation_share"], errors="coerce")
    f = pd.to_numeric(result["deforestation_share"], errors="coerce")
    result["event_composition_purity"] = np.maximum(d, f) / (d + f)
    return result


def _match(cohort, catalog, history_flags, config, settings):
    target_ids = set(history_flags["grid_id"].astype(int))
    pair_frames = []
    treatment_frames = []
    qualification_rows = []
    for event_year in range(settings.primary_event_years[0], settings.primary_event_years[1] + 1):
        event_rows = cohort.loc[cohort["event_year"].eq(event_year)].copy()
        if event_rows.empty:
            continue
        features = read_features_for_year(config, event_year, target_ids)
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
        if treatment_features.empty:
            continue
        controls = qualified.loc[
            qualified["stable_through_followup_full_history"].astype(bool)
            & qualified["coverage_eligible"].astype(bool)
            & qualified["matching_covariates_complete"].astype(bool)
        ].copy()
        validate_stable_controls(controls, settings.primary_event_threshold)
        pairs = match_risk_set(treatment_features, controls, settings)
        if pairs.empty:
            continue
        matched_ids = set(pairs["treatment_grid_id"].astype(int))
        pair_frames.append(pairs)
        treatment_frames.append(treatment_features.loc[treatment_features["grid_id"].isin(matched_ids)].copy())
        qualification_rows.append(pd.DataFrame({
            "event_year": int(event_year),
            "feature_rows": int(len(features)),
            "history_qualified_controls": int(len(controls)),
            "treatments_in_year": int(len(treatment_features)),
            "matched_treatments": int(pairs["treatment_grid_id"].nunique()),
            "pair_links": int(len(pairs)),
        }, index=[0]))
    if not pair_frames:
        raise RuntimeError("No full-history matches were produced")
    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(["event_year", "treatment_grid_id", "match_rank"], kind="mergesort").reset_index(drop=True)
    treatments = pd.concat(treatment_frames, ignore_index=True).drop_duplicates(["grid_id", "event_year"])
    treatment_history = treatments.loc[:, [
        "grid_id", "event_year", "next_significant_event_year", "repeat_within_5y", "censor_relative_year",
        "disturbance_share", "degradation_share", "deforestation_share", "regrowth_share",
    ]].rename(columns={"grid_id": "treatment_grid_id"})
    pairs = pairs.merge(treatment_history, on=["treatment_grid_id", "event_year"], how="left", validate="many_to_one")
    return pairs, treatments, pd.concat(qualification_rows, ignore_index=True)


def _repeat_composition(treatments):
    grouped = treatments.groupby(["event_type", "macroregion", "repeat_within_5y"], sort=True).agg(
        treatment_count=("grid_id", "nunique"), event_area_m2=("event_area_m2", "sum"),
        mean_disturbance_share=("disturbance_share", "mean"), mean_baseline_undist_share=("baseline_undist_share", "mean"),
    ).reset_index()
    totals = grouped.groupby(["event_type", "macroregion"], sort=False)["treatment_count"].transform("sum")
    grouped["share_of_type_region"] = grouped["treatment_count"] / totals
    return grouped


def _weighted_mean_ess(values, weights):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    weights = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan, 0.0
    values, weights = values[valid], weights[valid]
    total = weights.sum()
    return float(np.sum(values * weights) / total), float(total**2 / np.sum(weights**2))


def _capped_contrast(window, events, weights, cap, block_degrees, replicates, seed):
    """Re-estimate all relative years after a new multiplier cap."""
    capped = capped_standardized_weights(weights, cap)
    cols = ["treatment_grid_id", "event_year", "standardized_event_weight", "capped_standardization_multiplier"]
    working = window.merge(capped.loc[:, cols], on=["treatment_grid_id", "event_year"], how="inner", validate="many_to_one")
    coords = events.loc[:, ["treatment_grid_id", "event_year", "event_type", "lon", "lat"]].drop_duplicates(["treatment_grid_id", "event_year"]).reset_index(drop=True)
    coords["spatial_block"] = (
        np.floor((pd.to_numeric(coords["lon"], errors="coerce") + 180.0) / float(block_degrees)).astype(int).astype(str)
        + ":"
        + np.floor((pd.to_numeric(coords["lat"], errors="coerce") + 90.0) / float(block_degrees)).astype(int).astype(str)
    )
    working = working.merge(coords.loc[:, ["treatment_grid_id", "event_year", "spatial_block", "event_type"]], on=["treatment_grid_id", "event_year"], how="inner", validate="many_to_one", suffixes=("", "_event"))
    blocks = np.sort(working["spatial_block"].dropna().unique())
    rng = np.random.default_rng(int(seed))
    draw_counts = rng.multinomial(len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=int(replicates))
    rows = []
    for relative_year in sorted(pd.to_numeric(working["relative_year"], errors="coerce").dropna().unique()):
        frame = working.loc[working["relative_year"].eq(relative_year)].copy()
        type_point, type_draws, type_ess = {}, {}, {}
        for event_type in ("degradation", "deforestation"):
            group = frame.loc[frame["event_type"].eq(event_type) & frame["pair_effect_evi_mean"].notna()].copy()
            values = pd.to_numeric(group["pair_effect_evi_mean"], errors="coerce").to_numpy(dtype=np.float64)
            area_weights = pd.to_numeric(group["standardized_event_weight"], errors="coerce").to_numpy(dtype=np.float64)
            point, ess = _weighted_mean_ess(group["pair_effect_evi_mean"], group["standardized_event_weight"])
            type_point[event_type], type_ess[event_type] = point, ess
            if group.empty:
                draws = np.full(int(replicates), np.nan)
            else:
                block_sums = group.assign(_num=values * area_weights, _den=area_weights).groupby("spatial_block")[["_num", "_den"]].sum().reindex(blocks, fill_value=0.0)
                draws = np.divide(draw_counts @ block_sums["_num"].to_numpy(), draw_counts @ block_sums["_den"].to_numpy(), out=np.full(int(replicates), np.nan), where=(draw_counts @ block_sums["_den"].to_numpy()) > 0.0)
            type_draws[event_type] = draws
            finite = draws[np.isfinite(draws)]
            rows.append({"cap": float(cap), "event_type": event_type, "relative_year": int(relative_year), "estimate": point, "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan, "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan, "valid_replicates": int(len(finite)), "effective_sample_size": ess})
        contrast_draws = type_draws["deforestation"] - type_draws["degradation"]
        finite = contrast_draws[np.isfinite(contrast_draws)]
        rows.append({"cap": float(cap), "event_type": "deforestation_minus_degradation", "relative_year": int(relative_year), "estimate": type_point["deforestation"] - type_point["degradation"], "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan, "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan, "valid_replicates": int(len(finite)), "effective_sample_size": np.nan})
    return pd.DataFrame(rows)


def main():
    config = read_config()
    settings = settings_from_config(config)
    output = RUN_ROOT
    catalog = pd.read_parquet(output / "history_qualified_candidates.parquet")
    history_flags = pd.read_parquet(output / "history_qualified_controls.parquet")
    annual_history = pd.read_parquet(output / "history_qualified_annual.parquet")
    if catalog.empty:
        raise RuntimeError("Full-history candidate catalog is empty")
    validate_first_event_catalog(catalog, annual_history, settings.primary_event_threshold, config["first_event_history_start"], config["history_end"], tuple(config["primary_event_years"]))
    # Fresh matching is performed for all retained first events and again for
    # the isolated sensitivity cohort, rather than deleting rows from legacy pairs.
    pairs, treatments, qualification = _match(catalog, catalog, history_flags, config, settings)
    isolated_catalog = catalog.loc[~catalog["repeat_within_5y"].astype(bool)].copy()
    isolated_pairs, isolated_treatments, isolated_qualification = _match(isolated_catalog, catalog, history_flags, config, settings)
    annual = load_annual_outcomes(config, set(pairs["treatment_grid_id"]) | set(pairs["control_grid_id"]))
    window = build_matched_event_window(annual, treatments, pairs, settings, settings.primary_obs_threshold)
    isolated_annual = load_annual_outcomes(config, set(isolated_pairs["treatment_grid_id"]) | set(isolated_pairs["control_grid_id"]))
    isolated_window = build_matched_event_window(isolated_annual, isolated_treatments, isolated_pairs, settings, settings.primary_obs_threshold)
    events = event_table(treatments)
    standardized, diagnostics, weights = summarize_standardized_event_type_contrast(window, events, metric="evi_mean")
    bootstrap = bootstrap_standardized_event_type_contrast(
        window, events, metric="evi_mean", relative_years=settings.relative_years,
        block_degrees=config["bootstrap_block_degrees"], replicates=settings.bootstrap_replicates, seed=settings.bootstrap_seed,
    )
    response = summarize_matched_response(window, weighting="event_area", include_macroregions=True)
    block_bootstrap = spatial_block_bootstrap(window, block_degrees=config["bootstrap_block_degrees"], replicates=settings.bootstrap_replicates, seed=settings.bootstrap_seed, weighting="event_area")
    validate_dynamic_window(window)
    repeat = _repeat_composition(treatments)
    dynamic = window.loc[:, ["treatment_grid_id", "event_year", "event_type", "macroregion", "relative_year", "pair_effect_evi_mean", "event_area_m2", "censor_relative_year"]].copy()
    dynamic["retained"] = dynamic["pair_effect_evi_mean"].notna()
    dynamic_diag = dynamic.groupby(["relative_year", "event_type"], sort=True).agg(
        n=("treatment_grid_id", "nunique"), retained_n=("retained", lambda value: int(value.groupby(dynamic.loc[value.index, "treatment_grid_id"]).any().sum())), event_area_m2=("event_area_m2", "sum")
    ).reset_index()
    multiplier = pd.to_numeric(weights["standardization_multiplier"], errors="coerce")
    multiplier_summary = pd.DataFrame([{
        "n": int(multiplier.notna().sum()),
        "min": float(multiplier.min()), "q25": float(multiplier.quantile(0.25)),
        "median": float(multiplier.quantile(0.50)), "q75": float(multiplier.quantile(0.75)),
        "max": float(multiplier.max()), "pooled_99th_percentile": float(multiplier.quantile(0.99)),
    }])
    pooled99 = float(multiplier.quantile(0.99))
    cap_values = [10.0, 20.0, pooled99]
    capped_results = pd.concat([_capped_contrast(window, events, weights, cap, config["bootstrap_block_degrees"], settings.bootstrap_replicates, settings.bootstrap_seed) for cap in cap_values], ignore_index=True)
    # Persist all outputs with the run identifier in a sidecar manifest.
    for frame, name in ((pairs, "fixed_matched_pairs.parquet"), (treatments, "matched_treatments.parquet"), (window, "fixed_matched_event_window.parquet"), (isolated_pairs, "isolated_matched_pairs.parquet"), (isolated_treatments, "isolated_matched_treatments.parquet"), (isolated_window, "isolated_matched_event_window.parquet"), (weights, "common_support_weights.parquet")):
        frame = frame.copy()
        frame.insert(0, "run_id", config["run_id"])
        frame.to_parquet(output / name, index=False)
    for frame, name in ((qualification, "matching_qualification_by_year.csv"), (isolated_qualification, "isolated_matching_qualification_by_year.csv"), (response, "main_response_summary.csv"), (standardized, "standardized_event_type_contrast.csv"), (bootstrap, "standardized_event_type_bootstrap.csv"), (block_bootstrap, "spatial_block_bootstrap.csv"), (repeat, "repeat_event_composition.csv"), (dynamic_diag, "dynamic_risk_set_diagnostics.csv"), (capped_results, "capped_weight_contrasts.csv"), (multiplier_summary, "support_multiplier_summary.csv")):
        frame = frame.copy()
        frame.insert(0, "run_id", config["run_id"])
        frame.to_csv(output / name, index=False)
    diagnostics["run_id"] = config["run_id"]
    diagnostics["multiplier_distribution"] = multiplier_summary.iloc[0].to_dict()
    diagnostics["cap_values"] = cap_values
    (output / "support_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = {
        "created_at_utc": utc_now(), "run_id": config["run_id"], "analysis_label": config["analysis_label"],
        "candidate_count": int(len(catalog)), "matched_treatments": int(treatments["grid_id"].nunique()),
        "matched_pairs": int(len(pairs)), "matched_controls": int(pairs["control_grid_id"].nunique()),
        "isolated_treatments": int(isolated_treatments["grid_id"].nunique()), "isolated_pairs": int(len(isolated_pairs)),
        "standardization": diagnostics, "bootstrap_replicates": int(settings.bootstrap_replicates),
        "multiplier_summary": multiplier_summary.iloc[0].to_dict(),
        "bootstrap_seed": int(settings.bootstrap_seed), "history_start": int(config["first_event_history_start"]),
        "control_history_start": int(config["control_history_start"]), "validation_pass": True,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (output / "phase3_main_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "created_at_utc": utc_now(), "run_id": config["run_id"], "config": config,
        "inputs": {key: {"path": config[key], "sha256": sha256_file(config[key])} for key in ("legacy_catalog", "legacy_pairs", "legacy_window")},
        "artifacts": {p.name: {"size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(output.iterdir()) if p.is_file() and p.suffix in {".parquet", ".csv", ".json"}},
        "software": report["software"],
    }
    (output / "phase3_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
