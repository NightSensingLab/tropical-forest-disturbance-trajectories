#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from pipeline.subannual_response import (
    LINK_REQUIRED_COLUMNS,
    MONTHLY_REQUIRED_COLUMNS,
    SIXTEEN_DAY_REQUIRED_COLUMNS,
    SubannualSettings,
    annualize_monthly_effects,
    build_monthly_matched_effects,
    build_sixteen_day_matched_effects,
    build_treatment_contrasts,
    monthly_usable_mask,
    prepare_sixteen_day_anomalies,
    sensitivity_scenarios,
    settings_for_scenario,
    spatial_block_bootstrap_summary,
    summarize_annual_effects,
    summarize_contrasts,
    summarize_monthly_effects,
    validate_settings,
    weighted_mean_and_ess,
)
from pipeline.subannual_response_figures import (
    build_main_figure,
    build_validation_figure,
    export_figure,
    validate_figure_bundle,
)


STAGE08_REQUIRED_FILES = (
    "subannual_manifest.json",
    "subannual_report.json",
    "task_manifest.json",
    "input_audit.json",
    "matched_links.parquet",
    "unit_event_catalog.parquet",
    "collection_calendar.csv",
)

TABLE_REPORT_ARTIFACTS = (
    "input_audit.json",
    "monthly_matched_effects.parquet",
    "annual_matched_effects.parquet",
    "monthly_response_summary.csv",
    "annual_response_summary.csv",
    "aggregate_contrasts.csv",
    "coverage_summary.csv",
    "sixteen_day_validation_summary.csv",
    "sensitivity_summary.csv",
    "stage09_report.json",
    "stage09_report.md",
)
FIGURE_ARTIFACTS = tuple(
    f"figures/{stem}{suffix}"
    for stem in ("stage09_subannual_response", "stage09_sixteen_day_validation")
    for suffix in (".png", ".pdf", ".svg", ".tiff")
)
HASHED_ARTIFACTS = TABLE_REPORT_ARTIFACTS + FIGURE_ARTIFACTS
PUBLIC_ARTIFACTS = HASHED_ARTIFACTS + ("stage09_manifest.json",)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_write_parquet(frame, path, compression="zstd"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow", compression=compression)
    os.replace(temporary, path)


def atomic_write_text(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def output_metadata(path):
    path = Path(path)
    return {"size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def artifact_metadata(path):
    path = Path(path)
    metadata = output_metadata(path)
    if path.suffix == ".parquet":
        parquet_file = pq.ParquetFile(path)
        metadata.update(
            {
                "kind": "parquet",
                "row_count": int(parquet_file.metadata.num_rows),
                "columns": parquet_file.schema_arrow.names,
            }
        )
    elif path.suffix == ".csv":
        frame = pd.read_csv(path)
        metadata.update(
            {"kind": "csv", "row_count": int(len(frame)), "columns": list(frame.columns)}
        )
    elif path.suffix in {".png", ".pdf", ".svg", ".tiff"}:
        metadata["kind"] = "figure"
    else:
        metadata["kind"] = "json" if path.suffix == ".json" else "markdown"
    return metadata


def analysis_signature(source_paths, settings):
    settings = validate_settings(settings)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(asdict(settings), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for path in sorted(map(Path, source_paths), key=lambda value: str(value)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def resolve_stage08_path(recorded_path, stage08_root):
    stage08_root = Path(stage08_root)
    recorded = Path(str(recorded_path))
    if recorded.is_file():
        return recorded
    marker = "08_subannual_matched/"
    text = str(recorded_path).replace("\\", "/")
    if marker in text:
        relative = text.split(marker, 1)[1]
        return stage08_root / relative
    return stage08_root / recorded.name


def _manifest_records(manifest):
    records = [
        manifest["collection_calendar"],
        manifest["matched_links"],
        manifest["unit_event_catalog"],
        *manifest["parts_monthly"],
        *manifest["parts_16day"],
    ]
    for part in manifest["parts_16day"]:
        records.extend(part.get("csv_inputs", []))
    return records


def _verify_record(record, stage08_root):
    path = resolve_stage08_path(record["path"], stage08_root)
    checks = {
        "exists": path.is_file(),
        "size": False,
        "sha256": False,
        "rows": True,
    }
    if not path.is_file():
        return path, checks
    checks["size"] = int(path.stat().st_size) == int(record["size_bytes"])
    checks["sha256"] = sha256_file(path) == record["sha256"]
    if "rows" in record:
        if path.suffix == ".parquet":
            actual_rows = int(pq.ParquetFile(path).metadata.num_rows)
        elif path.suffix == ".csv":
            actual_rows = int(len(pd.read_csv(path)))
        else:
            actual_rows = -1
        checks["rows"] = actual_rows == int(record["rows"])
    return path, checks


def audit_stage08_inputs(stage08_root, settings=None):
    settings = validate_settings(settings or SubannualSettings())
    root = Path(stage08_root)
    missing = [name for name in STAGE08_REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 08 input files are missing: {missing}")
    manifest = read_json(root / "subannual_manifest.json")
    report = read_json(root / "subannual_report.json")
    task_manifest = read_json(root / "task_manifest.json")
    source_audit = read_json(root / "input_audit.json")

    identity_checks = {
        "analysis_signature": manifest.get("analysis_signature")
        == settings.stage08_signature
        == report.get("analysis_signature")
        == task_manifest.get("analysis_signature"),
        "source_signature": manifest.get("source_signature")
        == settings.stage07_source_sha256
        == task_manifest.get("source_signature")
        == source_audit.get("source_sha256"),
        "event_years": tuple(manifest.get("event_years", ())) == settings.event_years
        and tuple(report.get("event_years", ())) == settings.event_years,
        "stage08_pass": report.get("stage08_pass") is True,
        "pair_links": int(manifest.get("input", {}).get("pair_links", -1))
        == settings.expected_pair_links
        == int(report.get("pair_links", -2)),
        "unit_events": int(manifest.get("input", {}).get("unit_events", -1))
        == settings.expected_unit_events
        == int(report.get("unit_events", -2)),
        "treatment_unit_events": int(
            manifest.get("input", {}).get("treatment_unit_events", -1)
        )
        == settings.expected_treatments,
        "monthly_rows": int(manifest.get("total_monthly_rows", -1))
        == settings.expected_monthly_rows
        == int(report.get("rows_monthly", -2)),
        "sixteen_day_rows": int(manifest.get("total_16day_rows", -1))
        == settings.expected_sixteen_day_rows
        == int(report.get("rows_16day", -2)),
    }

    task_checks = {}
    for event_year in settings.event_years:
        entry = task_manifest.get("parts", {}).get(str(event_year), {})
        shards = entry.get("shard_tasks", [])
        task_checks[str(event_year)] = bool(
            entry.get("task_state") == "COMPLETED"
            and entry.get("validation_status") == "complete"
            and (not shards or all(shard.get("task_state") == "COMPLETED" for shard in shards))
        )

    file_checks = {}
    resolved_paths = []
    for record in _manifest_records(manifest):
        path, checks = _verify_record(record, root)
        relative = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        file_checks[relative] = checks
        resolved_paths.append(path)

    monthly_parts = {
        int(Path(record["path"]).parent.name.split("=", 1)[1]): resolve_stage08_path(
            record["path"], root
        )
        for record in manifest["parts_monthly"]
    }
    sixteen_day_parts = {
        int(Path(record["path"]).parent.name.split("=", 1)[1]): resolve_stage08_path(
            record["path"], root
        )
        for record in manifest["parts_16day"]
    }
    schema_checks = {
        "monthly_part_count": set(monthly_parts) == set(settings.event_years),
        "sixteen_day_part_count": set(sixteen_day_parts) == set(settings.event_years),
        "monthly_schema": all(
            set(MONTHLY_REQUIRED_COLUMNS).issubset(pq.ParquetFile(path).schema_arrow.names)
            for path in monthly_parts.values()
        ),
        "sixteen_day_schema": all(
            set(SIXTEEN_DAY_REQUIRED_COLUMNS).issubset(pq.ParquetFile(path).schema_arrow.names)
            for path in sixteen_day_parts.values()
        ),
        "links_schema": set(LINK_REQUIRED_COLUMNS).issubset(
            pq.ParquetFile(root / "matched_links.parquet").schema_arrow.names
        ),
    }
    passed = bool(
        all(identity_checks.values())
        and all(task_checks.values())
        and all(all(check.values()) for check in file_checks.values())
        and all(schema_checks.values())
    )
    if not passed:
        failed = {
            "identity": [key for key, value in identity_checks.items() if not value],
            "tasks": [key for key, value in task_checks.items() if not value],
            "files": [
                key for key, value in file_checks.items() if not all(value.values())
            ],
            "schemas": [key for key, value in schema_checks.items() if not value],
        }
        raise RuntimeError(f"Stage 08 input gate failed: {failed}")
    return {
        "pass": True,
        "audited_at_utc": utc_now(),
        "identity_checks": identity_checks,
        "task_checks": task_checks,
        "file_count": len(file_checks),
        "file_checks": file_checks,
        "schema_checks": schema_checks,
        "monthly_parts": monthly_parts,
        "sixteen_day_parts": sixteen_day_parts,
        "manifest": manifest,
    }


def _work_paths(output_dir, event_year):
    directory = Path(output_dir) / "work" / f"event_year={int(event_year)}"
    return directory / "part.parquet", directory / "part.json"


def process_event_year(
    stage08_root,
    output_dir,
    event_year,
    links,
    settings,
    signature,
    force_rebuild=False,
):
    settings = validate_settings(settings)
    source = (
        Path(stage08_root)
        / "parts_monthly"
        / f"event_year={int(event_year)}"
        / "part.parquet"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Missing Stage 08 monthly part: {source}")
    source_sha256 = sha256_file(source)
    output_path, metadata_path = _work_paths(output_dir, event_year)
    if not force_rebuild and output_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if (
            metadata.get("analysis_signature") == signature
            and metadata.get("source_sha256") == source_sha256
            and metadata.get("sha256") == sha256_file(output_path)
            and int(metadata.get("rows", -1))
            == int(pq.ParquetFile(output_path).metadata.num_rows)
        ):
            return {**metadata, "skipped": True}
    monthly = pd.read_parquet(source)
    year_links = links.loc[links["event_year"].eq(int(event_year))].copy()
    effects = build_monthly_matched_effects(monthly, year_links, settings)
    atomic_write_parquet(effects, output_path)
    metadata = {
        "event_year": int(event_year),
        "analysis_signature": signature,
        "source_sha256": source_sha256,
        "rows": int(len(effects)),
        **output_metadata(output_path),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(metadata, metadata_path)
    return {**metadata, "skipped": False}


def _load_primary_work(output_dir, event_years):
    frames = []
    for event_year in event_years:
        path, _ = _work_paths(output_dir, event_year)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 09 work part: {path}")
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["event_year", "treatment_grid_id", "relative_year", "calendar_month"],
        kind="mergesort",
    ).reset_index(drop=True)


def _bootstrap_by_scope(
    frame,
    value_column,
    time_column,
    settings,
):
    rows = []
    scopes = [("Global", frame)]
    scopes.extend(
        (str(scope), group.copy())
        for scope, group in frame.groupby("macroregion", sort=True)
    )
    for scope, selected in scopes:
        if selected.empty:
            continue
        boot = spatial_block_bootstrap_summary(
            selected,
            value_column=value_column,
            group_columns=("event_type", time_column),
            weighting="event_area",
            block_degrees=settings.block_degrees,
            replicates=settings.bootstrap_replicates,
            seed=settings.bootstrap_seed,
        )
        boot["scope"] = scope
        rows.append(boot)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _merge_intervals(summary, bootstrap, time_column):
    interval_columns = [
        "scope",
        "event_type",
        time_column,
        "ci_low",
        "ci_high",
        "bootstrap_sd",
        "spatial_block_count",
        "block_degrees",
        "bootstrap_replicates",
        "bootstrap_seed",
    ]
    available = bootstrap.loc[:, interval_columns] if not bootstrap.empty else pd.DataFrame(
        columns=interval_columns
    )
    result = summary.merge(
        available,
        on=["scope", "event_type", time_column],
        how="left",
        validate="many_to_one",
    )
    return result


def build_primary_summaries(monthly_effects, settings):
    monthly_summary = pd.concat(
        [
            summarize_monthly_effects(monthly_effects, "event_area"),
            summarize_monthly_effects(monthly_effects, "equal_grid"),
        ],
        ignore_index=True,
    )
    annual = annualize_monthly_effects(monthly_effects, settings)
    annual_summary = pd.concat(
        [
            summarize_annual_effects(annual, "event_area"),
            summarize_annual_effects(annual, "equal_grid"),
        ],
        ignore_index=True,
    )
    annual_bootstrap = _bootstrap_by_scope(
        annual, "annual_matched_effect", "relative_year", settings
    )
    annual_summary = _merge_intervals(
        annual_summary, annual_bootstrap, "relative_year"
    )

    treatment_contrasts = build_treatment_contrasts(annual)
    contrast_summary = pd.concat(
        [
            summarize_contrasts(treatment_contrasts, "event_area"),
            summarize_contrasts(treatment_contrasts, "equal_grid"),
        ],
        ignore_index=True,
    )
    contrast_bootstrap = _bootstrap_by_scope(
        treatment_contrasts, "contrast_effect", "contrast", settings
    )
    contrast_summary = _merge_intervals(contrast_summary, contrast_bootstrap, "contrast")
    return monthly_summary, annual, annual_summary, treatment_contrasts, contrast_summary


def build_monthly_coverage(stage08_root, links, settings):
    treatment_map = links.loc[
        :, ["treatment_grid_id", "event_year", "event_type"]
    ].drop_duplicates()
    control_map = links.loc[
        :, ["control_grid_id", "event_year", "event_type"]
    ].drop_duplicates()
    frames = []
    for event_year in settings.event_years:
        path = (
            Path(stage08_root)
            / "parts_monthly"
            / f"event_year={event_year}"
            / "part.parquet"
        )
        monthly = pd.read_parquet(path, columns=list(MONTHLY_REQUIRED_COLUMNS))
        monthly["_usable"] = monthly_usable_mask(monthly, settings)
        treatment = monthly.loc[monthly["role"].eq("treatment")].merge(
            treatment_map,
            left_on=["grid_id", "event_year"],
            right_on=["treatment_grid_id", "event_year"],
            how="inner",
            validate="many_to_one",
        )
        treatment["role_group"] = "treatment"
        control = monthly.loc[monthly["role"].eq("control")].merge(
            control_map,
            left_on=["grid_id", "event_year"],
            right_on=["control_grid_id", "event_year"],
            how="inner",
            validate="many_to_many",
        )
        control["role_group"] = "control"
        frames.append(
            pd.concat([treatment, control], ignore_index=True).loc[
                :, ["event_type", "role_group", "relative_year", "calendar_month", "_usable"]
            ]
        )
    working = pd.concat(frames, ignore_index=True)
    detailed = (
        working.groupby(
            ["event_type", "role_group", "relative_year", "calendar_month"], sort=True
        )["_usable"]
        .agg([("expected_rows", "size"), ("usable_rows", "sum")])
        .reset_index()
    )
    detailed["usable_pct"] = 100.0 * detailed["usable_rows"] / detailed["expected_rows"]
    detailed["summary_type"] = "relative_year_month"
    aggregate = (
        working.groupby(["event_type", "role_group", "calendar_month"], sort=True)[
            "_usable"
        ]
        .agg([("expected_rows", "size"), ("usable_rows", "sum")])
        .reset_index()
    )
    aggregate["relative_year"] = np.nan
    aggregate["usable_pct"] = 100.0 * aggregate["usable_rows"] / aggregate["expected_rows"]
    aggregate["summary_type"] = "calendar_month"
    result = pd.concat([detailed, aggregate], ignore_index=True)
    return result.rename(columns={"role_group": "role"}).loc[
        :,
        [
            "summary_type",
            "event_type",
            "role",
            "relative_year",
            "calendar_month",
            "expected_rows",
            "usable_rows",
            "usable_pct",
        ],
    ]


def _sufficient_statistics(frame, group_columns, value_column):
    if frame.empty:
        return pd.DataFrame(
            columns=[*group_columns, "weighted_sum", "weight_sum", "weight_sq_sum", "n"]
        )
    working = frame.loc[:, [*group_columns, value_column, "event_area_m2"]].copy()
    value = pd.to_numeric(working[value_column], errors="coerce")
    weight = pd.to_numeric(working["event_area_m2"], errors="coerce")
    valid = np.isfinite(value) & np.isfinite(weight) & weight.gt(0.0)
    working = working.loc[valid].copy()
    working["weighted_sum"] = value.loc[valid].to_numpy() * weight.loc[valid].to_numpy()
    working["weight_sum"] = weight.loc[valid].to_numpy()
    working["weight_sq_sum"] = weight.loc[valid].to_numpy() ** 2
    working["n"] = 1
    return (
        working.groupby(list(group_columns), sort=True, dropna=False)[
            ["weighted_sum", "weight_sum", "weight_sq_sum", "n"]
        ]
        .sum()
        .reset_index()
    )


def _combine_sufficient(frames, group_columns):
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.groupby(list(group_columns), sort=True, dropna=False)[
            ["weighted_sum", "weight_sum", "weight_sq_sum", "n"]
        ]
        .sum()
        .reset_index()
    )
    combined["estimate"] = combined["weighted_sum"] / combined["weight_sum"]
    combined["effective_n"] = combined["weight_sum"] ** 2 / combined["weight_sq_sum"]
    return combined


def build_sixteen_day_validation(
    stage08_root,
    links,
    settings,
    monthly_response_summary,
):
    composite_frames = []
    agreement_frames = []
    for event_year in settings.event_years:
        path = (
            Path(stage08_root)
            / "parts_16day"
            / f"event_year={event_year}"
            / "part.parquet"
        )
        frame = pd.read_parquet(path, columns=list(SIXTEEN_DAY_REQUIRED_COLUMNS))
        prepared = prepare_sixteen_day_anomalies(frame, settings)
        year_links = links.loc[links["event_year"].eq(event_year)]
        effects = build_sixteen_day_matched_effects(prepared, year_links, settings)
        composite_frames.append(
            _sufficient_statistics(
                effects,
                ("event_type", "relative_year", "composite_slot"),
                "matched_effect",
            )
        )
        if effects.empty:
            continue
        treatment_month = (
            effects.groupby(
                [
                    "treatment_grid_id",
                    "event_year",
                    "event_type",
                    "event_area_m2",
                    "relative_year",
                    "calendar_month",
                ],
                sort=False,
            )["matched_effect"]
            .mean()
            .reset_index()
        )
        agreement_frames.append(
            _sufficient_statistics(
                treatment_month,
                ("event_type", "relative_year", "calendar_month"),
                "matched_effect",
            )
        )
    composite = _combine_sufficient(
        composite_frames, ("event_type", "relative_year", "composite_slot")
    )
    composite["summary_type"] = "composite_effect"
    composite["scope"] = "Global"
    composite["calendar_month"] = np.nan
    composite["monthly_estimate"] = np.nan
    composite["sixteen_day_estimate"] = np.nan

    agreement = _combine_sufficient(
        agreement_frames, ("event_type", "relative_year", "calendar_month")
    )
    primary_monthly = monthly_response_summary.loc[
        monthly_response_summary["scope"].eq("Global")
        & monthly_response_summary["weighting"].eq("event_area"),
        ["event_type", "relative_year", "calendar_month", "estimate"],
    ].rename(columns={"estimate": "monthly_estimate"})
    agreement = agreement.rename(columns={"estimate": "sixteen_day_estimate"}).merge(
        primary_monthly,
        on=["event_type", "relative_year", "calendar_month"],
        how="left",
        validate="one_to_one",
    )
    agreement["summary_type"] = "monthly_agreement"
    agreement["scope"] = "Global"
    agreement["composite_slot"] = np.nan
    agreement["estimate"] = np.nan
    result = pd.concat([composite, agreement], ignore_index=True, sort=False)
    result["difference"] = result["sixteen_day_estimate"] - result["monthly_estimate"]
    columns = [
        "summary_type",
        "scope",
        "event_type",
        "relative_year",
        "calendar_month",
        "composite_slot",
        "estimate",
        "monthly_estimate",
        "sixteen_day_estimate",
        "difference",
        "n",
        "effective_n",
        "weight_sum",
    ]
    return result.loc[:, columns].sort_values(
        ["summary_type", "event_type", "relative_year", "calendar_month", "composite_slot"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)


def build_sensitivity_summary(stage08_root, links, primary_effects, settings):
    effect_cache = {
        (
            settings.valid_pixel_threshold,
            settings.control_weight_threshold,
            settings.outcome,
        ): primary_effects
    }
    rows = []
    for scenario in sensitivity_scenarios(settings):
        scenario_settings = settings_for_scenario(settings, scenario)
        effect_key = (
            scenario.valid_pixel_threshold,
            scenario.control_weight_threshold,
            scenario.outcome,
        )
        if effect_key not in effect_cache:
            parts = []
            for event_year in settings.event_years:
                path = (
                    Path(stage08_root)
                    / "parts_monthly"
                    / f"event_year={event_year}"
                    / "part.parquet"
                )
                monthly = pd.read_parquet(path)
                year_links = links.loc[links["event_year"].eq(event_year)]
                parts.append(
                    build_monthly_matched_effects(monthly, year_links, scenario_settings)
                )
            effect_cache[effect_key] = pd.concat(parts, ignore_index=True)
        effects = effect_cache[effect_key]
        annual = annualize_monthly_effects(effects, scenario_settings)
        contrasts = build_treatment_contrasts(annual)
        summary = summarize_contrasts(contrasts, scenario.weighting)
        summary = summary.loc[summary["scope"].eq("Global")].copy()
        summary["scenario"] = scenario.name
        summary["valid_pixel_threshold"] = scenario.valid_pixel_threshold
        summary["control_weight_threshold"] = scenario.control_weight_threshold
        summary["outcome"] = scenario.outcome
        summary["annual_min_months"] = scenario.annual_min_months
        summary["retained_treatments"] = int(annual["treatment_grid_id"].nunique())
        rows.append(summary)
    return pd.concat(rows, ignore_index=True).loc[
        :,
        [
            "scenario",
            "event_type",
            "contrast",
            "estimate",
            "n",
            "effective_n",
            "weight_sum",
            "retained_treatments",
            "weighting",
            "valid_pixel_threshold",
            "control_weight_threshold",
            "outcome",
            "annual_min_months",
        ],
    ].sort_values(["scenario", "event_type", "contrast"], kind="mergesort").reset_index(
        drop=True
    )


def _report_markdown(report):
    flow = report["sample_flow"]
    lines = [
        "# Stage 09 Subannual Matched Response",
        "",
        f"- Stage 09 pass: `{report['stage09_pass']}`",
        f"- Analysis signature: `{report['analysis_signature']}`",
        f"- Monthly matched effects: {flow['monthly_matched_rows']:,}",
        f"- Annual matched effects: {flow['annual_matched_rows']:,}",
        f"- Treatments retained monthly: {flow['monthly_treatments']:,}",
        f"- Treatments retained annually: {flow['annual_treatments']:,}",
        f"- Spatial bootstrap: {report['settings']['bootstrap_replicates']:,} replicates, "
        f"{report['settings']['block_degrees']}-degree blocks, seed "
        f"{report['settings']['bootstrap_seed']}",
        "",
        "Monthly matched EVI anomalies are the sole primary subannual estimand. Sixteen-day results are descriptive consistency checks. TMF supplies event year only; no within-year observation is interpreted as an exact disturbance date.",
        "",
        "The estimates are conditional matched associations, not unconditional causal effects.",
    ]
    return "\n".join(lines) + "\n"


def run_pipeline(config, settings=None, force_rebuild=False):
    settings = validate_settings(settings or SubannualSettings())
    stage08_root, output_dir = resolve_paths(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_stage08_inputs(stage08_root, settings)
    links = pd.read_parquet(stage08_root / "matched_links.parquet")
    signature = analysis_signature([stage08_root / "subannual_manifest.json"], settings)
    audit_output = {
        key: value
        for key, value in audit.items()
        if key not in {"monthly_parts", "sixteen_day_parts", "manifest"}
    }
    audit_output.update(
        {
            "analysis_signature": signature,
            "stage08_analysis_signature": settings.stage08_signature,
            "stage07_source_sha256": settings.stage07_source_sha256,
        }
    )
    atomic_write_json(audit_output, output_dir / "input_audit.json")

    work_parts = []
    for event_year in settings.event_years:
        result = process_event_year(
            stage08_root,
            output_dir,
            event_year,
            links,
            settings,
            signature,
            force_rebuild=force_rebuild,
        )
        work_parts.append(result)
        state = "Reused" if result["skipped"] else "Prepared"
        print(f"{state} event year {event_year}: {result['rows']:,} monthly effects")

    monthly_effects = _load_primary_work(output_dir, settings.event_years)
    (
        monthly_summary,
        annual_effects,
        annual_summary,
        treatment_contrasts,
        contrast_summary,
    ) = build_primary_summaries(monthly_effects, settings)
    coverage_summary = build_monthly_coverage(stage08_root, links, settings)
    sixteen_day_summary = build_sixteen_day_validation(
        stage08_root, links, settings, monthly_summary
    )
    sensitivity_summary = build_sensitivity_summary(
        stage08_root, links, monthly_effects, settings
    )

    atomic_write_parquet(monthly_effects, output_dir / "monthly_matched_effects.parquet")
    atomic_write_parquet(annual_effects, output_dir / "annual_matched_effects.parquet")
    atomic_write_csv(monthly_summary, output_dir / "monthly_response_summary.csv")
    atomic_write_csv(annual_summary, output_dir / "annual_response_summary.csv")
    atomic_write_csv(contrast_summary, output_dir / "aggregate_contrasts.csv")
    atomic_write_csv(coverage_summary, output_dir / "coverage_summary.csv")
    atomic_write_csv(
        sixteen_day_summary, output_dir / "sixteen_day_validation_summary.csv"
    )
    atomic_write_csv(sensitivity_summary, output_dir / "sensitivity_summary.csv")

    figure_dir = output_dir / "figures"
    main_paths = export_figure(
        build_main_figure(monthly_summary, annual_summary, contrast_summary),
        figure_dir / "stage09_subannual_response",
    )
    validation_paths = export_figure(
        build_validation_figure(sixteen_day_summary, coverage_summary),
        figure_dir / "stage09_sixteen_day_validation",
    )
    main_figure_check = validate_figure_bundle(main_paths)
    validation_figure_check = validate_figure_bundle(validation_paths)

    expected_scenarios = {scenario.name for scenario in sensitivity_scenarios(settings)}
    checks = {
        "input_gate": audit["pass"],
        "all_work_parts": len(work_parts) == len(settings.event_years)
        and all(int(item["rows"]) >= 0 for item in work_parts),
        "monthly_nonempty": not monthly_effects.empty,
        "monthly_primary_key_unique": not monthly_effects.duplicated(
            ["treatment_grid_id", "event_year", "relative_year", "calendar_month"]
        ).any(),
        "monthly_upper_bound": len(monthly_effects)
        <= settings.expected_treatments * len(settings.relative_years) * 12,
        "annual_nonempty": not annual_effects.empty,
        "sensitivity_catalogue": set(sensitivity_summary["scenario"])
        == expected_scenarios,
        "sixteen_day_validation_nonempty": not sixteen_day_summary.empty,
        "no_exact_date_claim_fields": not bool(
            {"disturbance_date", "time_since_event", "days_since_event"}
            & set(sixteen_day_summary.columns)
        ),
        "figure_bundles_valid": main_figure_check["pass"]
        and validation_figure_check["pass"],
    }
    report = {
        "created_at_utc": utc_now(),
        "analysis_signature": signature,
        "analysis_version": "2026-07-25-stage09-subannual-matched-response-v1",
        "stage08_analysis_signature": settings.stage08_signature,
        "stage07_source_sha256": settings.stage07_source_sha256,
        "stage09_pass": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "settings": asdict(settings),
        "sample_flow": {
            "monthly_matched_rows": int(len(monthly_effects)),
            "annual_matched_rows": int(len(annual_effects)),
            "monthly_treatments": int(monthly_effects["treatment_grid_id"].nunique()),
            "annual_treatments": int(annual_effects["treatment_grid_id"].nunique()),
            "treatment_contrasts": int(len(treatment_contrasts)),
            "sixteen_day_summary_rows": int(len(sixteen_day_summary)),
        },
        "work_parts": [
            {key: value for key, value in item.items() if key != "skipped"}
            for item in work_parts
        ],
        "sensitivity_scenarios": sorted(expected_scenarios),
        "figure_validation": {
            "main": main_figure_check,
            "sixteen_day": validation_figure_check,
        },
        "interpretation_boundary": (
            "TMF supplies event year only; calendar month and composite slot are not exact "
            "disturbance dates. Estimates are conditional matched associations."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    atomic_write_json(report, output_dir / "stage09_report.json")
    atomic_write_text(_report_markdown(report), output_dir / "stage09_report.md")
    manifest = {
        "created_at_utc": utc_now(),
        "analysis_signature": signature,
        "analysis_version": report["analysis_version"],
        "stage08_analysis_signature": settings.stage08_signature,
        "stage07_source_sha256": settings.stage07_source_sha256,
        "artifacts": {
            name: artifact_metadata(output_dir / name) for name in HASHED_ARTIFACTS
        },
    }
    atomic_write_json(manifest, output_dir / "stage09_manifest.json")
    if not report["stage09_pass"]:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Stage 09 checks failed: {failed}")
    return report


def load_config(path):
    return read_json(path)


def resolve_paths(config):
    stage08 = config.get("input_stage_08") or str(
        Path(config["cloud_analysis_dir"]) / "08_subannual_matched"
    )
    stage09 = config.get("output_stage_09") or str(
        Path(config["project_root"]) / "results" / "09_subannual_matched_response"
    )
    return Path(stage08), Path(stage09)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage 09 subannual matched response.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = run_pipeline(config, force_rebuild=args.force_rebuild)
    print(
        json.dumps(
            {
                "analysis_signature": report["analysis_signature"],
                "stage09_pass": report["stage09_pass"],
                "sample_flow": report["sample_flow"],
            }
        )
    )


if __name__ == "__main__":
    main()
