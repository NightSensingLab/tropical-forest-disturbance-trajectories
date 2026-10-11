#!/usr/bin/env python3
"""Reproducible Major 2/3 revision analysis.

The runner deliberately writes into a new output directory and only reads the
validated Stage 06/07 artifacts.  The primary cohort contains all eligible
first events; observations at and after a subsequent significant event are
masked in the matched effect columns.  The original isolated-event window is
copied as a sensitivity cohort for transparent comparison.
"""

import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from pipeline.matched_control import (
    MatchedControlSettings,
    control_risk_set,
    match_risk_set,
    select_all_first_treatments,
)
from pipeline.matched_estimation import (
    bootstrap_standardized_event_type_contrast,
    build_matched_event_window,
    summarize_matched_response,
    summarize_standardized_event_type_contrast,
)
from pipeline.summarize_event_comparability import summarize_frame as summarize_event_comparability
from pipeline.major23_revision_figures import (
    build_major23_revision_figure,
    export_revision_figure,
)


ARTIFACT_NAMES = (
    "all_first_matched_pairs.parquet",
    "all_first_matched_event_window.parquet",
    "isolated_matched_event_window.parquet",
    "event_comparability_diagnostics.csv",
    "event_comparability_by_stratum.csv",
    "all_first_response_summary.csv",
    "standardized_event_type_contrast.csv",
    "standardized_event_type_bootstrap.csv",
    "repeat_event_composition.csv",
    "major23_report.json",
    "major23_report.md",
    "major23_manifest.json",
)
FIGURE_ARTIFACT_NAMES = tuple(
    f"figures/major23_revision{suffix}"
    for suffix in (".png", ".pdf", ".svg", ".tiff")
)
HASHED_ARTIFACT_NAMES = tuple(
    name for name in (*ARTIFACT_NAMES, *FIGURE_ARTIFACT_NAMES)
    if name != "major23_manifest.json"
)

OUTCOME_COLUMNS = (
    "grid_id",
    "year",
    "evi_mean",
    "evi_min",
    "evi_max",
    "evi_sd",
    "valid_obs",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


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


def artifact_metadata(path):
    path = Path(path)
    metadata = {"size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    if path.suffix == ".parquet":
        metadata["kind"] = "parquet"
        metadata["row_count"] = int(pd.read_parquet(path).shape[0])
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
        metadata.update({"kind": "csv", "row_count": int(len(frame)), "columns": list(frame.columns)})
    else:
        metadata["kind"] = "json" if path.suffix == ".json" else "markdown"
    return metadata


def _concat_parquet(paths):
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        raise FileNotFoundError("No Parquet inputs were found.")
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else frames[0]


def _feature_paths(stage07, event_year, obs_threshold):
    root = Path(stage07) / "work" / "features" / f"obs{int(obs_threshold):02d}" / f"year{int(event_year)}"
    paths = sorted(root.glob("part_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No cached feature parts found under {root}")
    return paths


def _stage03_parts(config):
    stage03 = Path(config["cloud_analysis_dir"]) / "03_combined_panel" / "parts"
    paths = sorted(stage03.glob("part_*.parquet"))
    if paths:
        return paths
    fallback = Path(config.get("stage03_parts_dir", ""))
    paths = sorted(fallback.glob("part_*.parquet")) if str(fallback) else []
    if not paths:
        raise FileNotFoundError(f"No Stage 03 annual panel parts found under {stage03}")
    return paths


def _load_annual_outcomes(paths, grid_ids):
    grid_ids = set(map(int, grid_ids))
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, columns=list(OUTCOME_COLUMNS))
        selected = frame.loc[frame["grid_id"].isin(grid_ids)]
        if not selected.empty:
            frames.append(selected)
    if not frames:
        raise RuntimeError("No annual outcomes were found for matched grid IDs.")
    return pd.concat(frames, ignore_index=True)


def _load_treatment_features(stage07, treatments, settings):
    frames = []
    event_fields = [
        "grid_id",
        "event_year",
        "event_type",
        "event_area_m2",
        "disturbance_share",
        "degradation_share",
        "deforestation_share",
        "regrowth_share",
        "next_significant_event_year",
        "repeat_within_5y",
        "censor_relative_year",
        "macroregion",
        "lon",
        "lat",
    ]
    events = treatments.loc[:, [column for column in event_fields if column in treatments.columns]].copy()
    for event_year in range(settings.primary_event_years[0], settings.primary_event_years[1] + 1):
        event_rows = events.loc[events["event_year"].eq(event_year)].copy()
        if event_rows.empty:
            continue
        features = _concat_parquet(_feature_paths(stage07, event_year, settings.primary_obs_threshold))
        treatment_features = features.loc[features["grid_id"].isin(event_rows["grid_id"])].merge(
            event_rows,
            on=["grid_id", "event_year"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_event"),
        )
        if not treatment_features.empty:
            frames.append(treatment_features)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["event_year", "grid_id"], kind="mergesort").reset_index(drop=True)


def _match_all_first_events(stage07, treatments, settings):
    pair_frames = []
    treatment_frames = []
    for event_year in range(settings.primary_event_years[0], settings.primary_event_years[1] + 1):
        event_rows = treatments.loc[treatments["event_year"].eq(event_year)].copy()
        if event_rows.empty:
            continue
        features = _concat_parquet(_feature_paths(stage07, event_year, settings.primary_obs_threshold))
        treatment_features = features.loc[features["grid_id"].isin(event_rows["grid_id"])].merge(
            event_rows,
            on=["grid_id", "event_year"],
            how="inner",
            validate="one_to_one",
            suffixes=("", "_event"),
        )
        if treatment_features.empty:
            continue
        controls = control_risk_set(
            features,
            stable_through_2023=False,
            event_threshold=settings.primary_event_threshold,
        )
        pairs = match_risk_set(treatment_features, controls, settings)
        if pairs.empty:
            continue
        matched_ids = set(pairs["treatment_grid_id"])
        pair_frames.append(pairs)
        treatment_frames.append(treatment_features.loc[treatment_features["grid_id"].isin(matched_ids)])
    if not pair_frames:
        return pd.DataFrame(), pd.DataFrame()
    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["event_year", "treatment_grid_id", "match_rank"], kind="mergesort"
    ).reset_index(drop=True)
    treatment_features = pd.concat(treatment_frames, ignore_index=True).drop_duplicates(
        ["grid_id", "event_year"]
    )
    history = treatment_features.loc[:, [
        "grid_id",
        "event_year",
        "next_significant_event_year",
        "repeat_within_5y",
        "censor_relative_year",
        "disturbance_share",
        "degradation_share",
        "deforestation_share",
        "regrowth_share",
    ]].rename(columns={"grid_id": "treatment_grid_id"})
    pairs = pairs.merge(history, on=["treatment_grid_id", "event_year"], how="left", validate="many_to_one")
    return pairs, treatment_features


def _event_table_from_treatments(treatments):
    result = treatments.rename(columns={"grid_id": "treatment_grid_id"}).copy()
    required = {
        "treatment_grid_id",
        "event_year",
        "event_type",
        "event_area_m2",
        "disturbance_share",
        "degradation_share",
        "deforestation_share",
        "baseline_undist_share",
        "macroregion",
    }
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Matched treatment features lack standardization columns: {sorted(missing)}")
    degradation = pd.to_numeric(result["degradation_share"], errors="coerce")
    deforestation = pd.to_numeric(result["deforestation_share"], errors="coerce")
    total = degradation + deforestation
    result["event_composition_purity"] = np.divide(
        np.maximum(
            degradation.to_numpy(dtype=np.float64),
            deforestation.to_numpy(dtype=np.float64),
        ),
        total.to_numpy(dtype=np.float64),
        out=np.full(len(result), np.nan, dtype=np.float64),
        where=total.to_numpy(dtype=np.float64) > 0.0,
    )
    return result


def _repeat_composition(treatments):
    columns = ["event_type", "macroregion", "repeat_within_5y"]
    grouped = treatments.groupby(columns, dropna=False, sort=True).agg(
        treatment_count=("grid_id", "nunique"),
        event_area_m2=("event_area_m2", "sum"),
        mean_disturbance_share=("disturbance_share", "mean"),
        mean_baseline_undist_share=("baseline_undist_share", "mean"),
    ).reset_index()
    totals = grouped.groupby(["event_type", "macroregion"], sort=False)["treatment_count"].transform("sum")
    grouped["share_of_type_region"] = grouped["treatment_count"] / totals
    return grouped


def _comparability_diagnostics(events, weights, all_first_count):
    rows = []
    for event_type, group in events.groupby("event_type", sort=True):
        retained = weights.loc[weights["event_type"].eq(event_type)]
        rows.append({
            "event_type": event_type,
            "all_first_matched_treatments": int(len(group)),
            "common_support_treatments": int(len(retained)),
            "common_support_treatment_fraction": float(len(retained) / len(group)) if len(group) else np.nan,
            "common_support_event_area_fraction": float(
                retained["event_area_m2"].sum() / group["event_area_m2"].sum()
            ) if group["event_area_m2"].sum() else np.nan,
            "standardized_event_area_sum": float(retained["standardized_event_weight"].sum()),
            "all_first_cohort_total": int(all_first_count),
        })
    return pd.DataFrame(rows)


def _report_markdown(report):
    sample = report["sample_flow"]
    standardization = report["standardization"]
    return "\n".join([
        "# Major 2/3 revision analysis report",
        "",
        f"- Revision pass: **{report['major23_pass']}**",
        f"- All-first eligible treatments: **{sample['eligible_all_first_treatments']:,}**",
        f"- All-first matched treatments: **{sample['matched_all_first_treatments']:,}**",
        f"- Isolated sensitivity matched treatments: **{sample['isolated_sensitivity_matched_treatments']:,}**",
        f"- All-first pair links: **{sample['all_first_pair_links']:,}**",
        "",
        "## Estimand",
        "",
        "Primary estimates use all eligible first events and are dynamically censored at the next significant disturbance year. Controls are conditional on remaining below the primary disturbance threshold through the existing follow-up horizon.",
        "",
        "## Major 2 standardization",
        "",
        f"- Common-support strata: **{standardization['common_support_strata']:,}**",
        f"- Standardization variables: **D+F composition, event composition purity, macroregion, event year, and baseline undisturbed-forest share**.",
        "",
        "## Limitations",
        "",
        "Dynamic censoring identifies effects before a subsequent disturbance; it does not estimate post-repeat outcomes. The isolated cohort is retained only as a sensitivity analysis, and residual unmeasured confounding remains possible.",
        "",
    ])


def run_pipeline(config, output_dir=None, settings=None, force_rebuild=False):
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        config = read_json(config_path)
    else:
        config = dict(config)
    settings = settings or MatchedControlSettings()
    output_dir = Path(output_dir or config.get("output_major23_revision", Path(config["project_root"]) / "results" / "major23_revision"))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage06 = Path(config["output_stage_06"])
    stage07 = Path(config["output_stage_07"])
    catalog_path = stage06 / "event_catalog.parquet"
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)
    catalog = pd.read_parquet(catalog_path)
    eligible = select_all_first_treatments(catalog, settings)
    pairs, treatment_features = _match_all_first_events(stage07, eligible, settings)
    if pairs.empty or treatment_features.empty:
        raise RuntimeError("All-first revision matching produced no pairs.")

    annual_parts = _stage03_parts(config)
    matched_ids = set(pairs["treatment_grid_id"]) | set(pairs["control_grid_id"])
    annual = _load_annual_outcomes(annual_parts, matched_ids)
    all_first_window = build_matched_event_window(
        annual,
        treatment_features,
        pairs,
        settings,
        obs_threshold=settings.primary_obs_threshold,
    )
    isolated_window_path = stage07 / "matched_event_window.parquet"
    if not isolated_window_path.is_file():
        raise FileNotFoundError(isolated_window_path)
    isolated_window = pd.read_parquet(isolated_window_path)

    event_table = _event_table_from_treatments(treatment_features)
    standardized_summary, standardization_diagnostics, weights = summarize_standardized_event_type_contrast(
        all_first_window,
        event_table,
        metric="evi_mean",
    )
    bootstrap_replicates = int(settings.bootstrap_replicates)
    bootstrap = bootstrap_standardized_event_type_contrast(
        all_first_window,
        event_table,
        metric="evi_mean",
        relative_years=settings.relative_years,
        replicates=bootstrap_replicates,
        seed=settings.bootstrap_seed,
    )
    response = summarize_matched_response(
        all_first_window,
        weighting="event_area",
        include_macroregions=False,
    )
    response["cohort"] = "all_first_dynamic_censored"
    standardized_summary["cohort"] = "all_first_dynamic_censored"
    bootstrap["cohort"] = "all_first_dynamic_censored"
    repeat_composition = _repeat_composition(treatment_features)
    comparability = _comparability_diagnostics(
        event_table,
        weights,
        len(eligible),
    )
    comparability_by_stratum = summarize_event_comparability(all_first_window)

    # Year-specific audit counts distinguish fixed matched treatments from
    # observations that remain estimable after dynamic censoring.
    year_audit = []
    for event_year, group in all_first_window.groupby("event_year", sort=True):
        for relative_year, year_frame in group.groupby("relative_year", sort=True):
            year_audit.append({
                "event_year": int(event_year),
                "relative_year": int(relative_year),
                "fixed_matched_treatments": int(group["treatment_grid_id"].nunique()),
                "uncensored_treatments": int(
                    year_frame.loc[year_frame["pair_effect_evi_mean"].notna(), "treatment_grid_id"].nunique()
                ),
            })
    year_audit = pd.DataFrame(year_audit)

    artifact_frames = {
        "all_first_matched_pairs.parquet": pairs,
        "all_first_matched_event_window.parquet": all_first_window,
        "isolated_matched_event_window.parquet": isolated_window,
        "event_comparability_diagnostics.csv": comparability,
        "event_comparability_by_stratum.csv": comparability_by_stratum,
        "all_first_response_summary.csv": response,
        "standardized_event_type_contrast.csv": standardized_summary,
        "standardized_event_type_bootstrap.csv": bootstrap,
        "repeat_event_composition.csv": repeat_composition,
    }
    for name, frame in artifact_frames.items():
        if name.endswith(".parquet"):
            atomic_write_parquet(frame, output_dir / name)
        else:
            atomic_write_csv(frame, output_dir / name)

    figure = build_major23_revision_figure(
        response,
        bootstrap,
        repeat_composition,
        event_table,
    )
    export_revision_figure(figure, output_dir / "figures" / "major23_revision")

    matched_counts = treatment_features["event_type"].value_counts().sort_index().to_dict()
    censor_counts = treatment_features.assign(
        uncensored_entry=treatment_features["censor_relative_year"].isna()
    ).groupby("event_type")["uncensored_entry"].sum().to_dict()
    locked_settings = asdict(settings)
    # These legacy expected-count fields validate the original Stage 07 run;
    # they are not analysis settings for the all-first-event revision cohort.
    locked_settings.pop("expected_primary_treatments", None)
    locked_settings.pop("expected_stage03_rows", None)
    report = {
        "created_at_utc": utc_now(),
        "analysis_version": "2026-08-26-major2-major3-revision-v1",
        "major23_pass": bool(
            not all_first_window.duplicated(["treatment_grid_id", "event_year", "relative_year"]).any()
            and np.isfinite(weights["standardized_event_weight"]).all()
            and (weights["standardized_event_weight"] > 0).all()
            and bool((bootstrap["valid_replicates"] >= 2).all())
        ),
        "settings": locked_settings,
        "estimand": {
            "primary_cohort": "all eligible first events",
            "dynamic_censor_rule": "relative_year >= censor_relative_year masks pair_effect_*",
            "sensitivity_cohort": "original isolated events with no repeat within five years",
            "control_horizon": "controls stable through follow-up",
        },
        "sample_flow": {
            "eligible_all_first_treatments": int(len(eligible)),
            "matched_all_first_treatments": int(pairs["treatment_grid_id"].nunique()),
            "isolated_sensitivity_eligible_treatments": int(catalog.loc[catalog["primary_window_eligible"] & ~catalog["repeat_within_5y"]].shape[0]),
            "isolated_sensitivity_matched_treatments": int(isolated_window["treatment_grid_id"].nunique()),
            "all_first_pair_links": int(len(pairs)),
            "all_first_window_rows": int(len(all_first_window)),
            "fixed_matched_treatments_by_event_type": {str(k): int(v) for k, v in matched_counts.items()},
            "uncensored_entry_treatments_by_event_type": {str(k): int(v) for k, v in censor_counts.items()},
        },
        "year_specific_audit": year_audit.to_dict(orient="records"),
        "standardization": {
            **standardization_diagnostics,
            "variables": [
                "D+F disturbance share",
                "event composition purity (degradation_share, deforestation_share)",
                "macroregion",
                "event year",
                "baseline undisturbed-forest share",
            ],
        },
        "bootstrap": {
            "requested_replicates": bootstrap_replicates,
            "minimum_valid_replicates": int(bootstrap["valid_replicates"].min()) if not bootstrap.empty else 0,
            "all_rows_valid": bool(not bootstrap.empty and (bootstrap["valid_replicates"] >= 2).all()),
        },
        "input_paths": {
            "stage06_event_catalog": str(catalog_path),
            "stage07_cache": str(stage07),
            "stage03_parts": [str(path) for path in annual_parts],
            "isolated_window": str(isolated_window_path),
        },
        "interpretation_scope": "Conditional matched associations relative to controls stable through follow-up; dynamic censoring makes post-repeat effects undefined rather than zero.",
        "figure_artifacts": list(FIGURE_ARTIFACT_NAMES),
    }
    atomic_write_json(report, output_dir / "major23_report.json")
    atomic_write_text(_report_markdown(report), output_dir / "major23_report.md")

    source_hashes = {
        "event_catalog_sha256": sha256_file(catalog_path),
        "isolated_window_sha256": sha256_file(isolated_window_path),
    }
    manifest = {
        "created_at_utc": utc_now(),
        "analysis_version": report["analysis_version"],
        "source": source_hashes,
        "settings": locked_settings,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "artifacts": {
            name: artifact_metadata(output_dir / name)
            for name in HASHED_ARTIFACT_NAMES
            if (output_dir / name).is_file()
        },
    }
    atomic_write_json(manifest, output_dir / "major23_manifest.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to analysis_config.json")
    parser.add_argument("--output", default=None, help="Revision output directory")
    parser.add_argument("--force-rebuild", action="store_true", help="Recompute output artifacts")
    args = parser.parse_args(argv)
    report = run_pipeline(args.config, output_dir=args.output, force_rebuild=args.force_rebuild)
    print(json.dumps({"major23_pass": report["major23_pass"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
