#!/usr/bin/env python3
"""Assemble the current Stage 08 package from exact old-key reuse plus new exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
OLD_ROOT = PROJECT_ROOT / "01_data/upstream/TropicalRainforest_cloud_analysis/08_subannual_matched"
CURRENT_CATALOG = RUN_ROOT / "current_event_role_catalogue.parquet"
MISSING_CATALOG = RUN_ROOT / "current_modis_missing_catalogue.parquet"
TASK_MANIFEST = RUN_ROOT / "current_modis_increment_tasks.json"
MAIN_RUN = PROJECT_ROOT / "runs/20260904-expanded-control-pool"
OUT_ROOT = RUN_ROOT / "stage08_current_full_history"
DRIVE_ROOT = Path(os.environ.get("TFDT_DRIVE_ROOT", str(PROJECT_ROOT / "external_exports")))
DRIVE_FOLDER = "stage08_mod13q1_full_history_1990_20260909"
EVENT_YEARS = tuple(range(2004, 2019))
PERIODS_PER_EVENT = 207
CURRENT_SOURCE = MAIN_RUN / "fixed_matched_pairs.parquet"
STAGE08_SIGNATURE = "current_full_history_1990_stage08_20260909"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task_manifest():
    if not TASK_MANIFEST.is_file():
        raise FileNotFoundError(TASK_MANIFEST)
    return json.loads(TASK_MANIFEST.read_text(encoding="utf-8"))


def task_csvs(manifest):
    files = []
    for task in manifest["tasks"]:
        folder = DRIVE_ROOT / DRIVE_FOLDER
        exact = folder / f"{task['file_prefix']}.csv"
        matches = sorted(folder.glob(task["file_prefix"] + "*.csv")) if folder.is_dir() else []
        # Google Drive can retain a duplicate with a ``(1)`` suffix after a
        # repeated export. Prefer the exact task filename and keep the
        # duplicate available for audit rather than treating it as missing.
        selected = exact if exact in matches else (matches[0] if len(matches) == 1 else None)
        if selected is None or int(sum(1 for _ in open(selected, "rb"))) < 2:
            raise RuntimeError(f"Missing or empty new MODIS export: {task['file_prefix']}")
        if int(task["expected_rows"]) != int(len(pd.read_csv(selected, usecols=["grid_id"]))):
            raise RuntimeError(f"Row count mismatch in {selected}")
        files.append(selected)
    return files


def _monthly_summary(frame: pd.DataFrame) -> pd.DataFrame:
    sys.path.insert(0, str(PROJECT_ROOT / "03_publication/manuscript/TropicalRainforest_manuscript/02_pipeline/tools"))
    from build_subannual_notebook import build_monthly_summary

    return build_monthly_summary(frame)


def _read_new_csvs(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["observation_date"] = pd.to_datetime(frame["observation_date"])
        frame["evi"] = pd.to_numeric(frame["evi"], errors="coerce")
        frame["valid_pixel_fraction"] = pd.to_numeric(frame["valid_pixel_fraction"], errors="coerce")
        frame["evi_valid"] = frame["evi"].notna() & frame["valid_pixel_fraction"].gt(0)
        frame["record_provenance"] = "new_gee_export"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _read_old_common(year, common_keys):
    path = OLD_ROOT / "parts_16day" / f"event_year={year}" / "part.parquet"
    old = pd.read_parquet(path)
    old["observation_date"] = pd.to_datetime(old["observation_date"])
    selected = old.merge(common_keys, on=["grid_id", "event_year", "role"], how="inner", validate="many_to_one")
    selected["record_provenance"] = "reused_old_exact_mod13q1"
    return selected


def build_parts(catalogue, missing, new):
    key = ["grid_id", "event_year", "role"]
    old_keys = catalogue.merge(missing[key], on=key, how="left", indicator=True)
    common_keys = old_keys.loc[old_keys["_merge"].eq("left_only"), key].copy()
    parts = []
    for year in EVENT_YEARS:
        old = _read_old_common(year, common_keys.loc[common_keys["event_year"].eq(year), key])
        current_new = new.loc[new["event_year"].eq(year)].copy()
        frame = pd.concat([old, current_new], ignore_index=True, sort=False)
        frame = frame.sort_values(["role", "grid_id", "observation_date"], kind="mergesort").reset_index(drop=True)
        required = {"grid_id", "event_year", "role", "observation_date", "calendar_year", "calendar_month", "day_of_year", "relative_year", "evi", "valid_pixel_fraction", "evi_valid"}
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            raise RuntimeError(f"Stage 08 year {year} missing columns: {missing_columns}")
        expected = int(catalogue["event_year"].eq(year).sum()) * PERIODS_PER_EVENT
        if len(frame) != expected:
            raise RuntimeError(f"Stage 08 year {year} rows {len(frame)} != {expected}")
        if frame.duplicated(["grid_id", "event_year", "role", "observation_date"]).any():
            raise RuntimeError(f"Stage 08 year {year} contains duplicate unit-date rows")
        parts.append((year, frame))
    return parts


def write_package(parts, catalogue):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "parts_16day").mkdir(exist_ok=True)
    (OUT_ROOT / "parts_monthly").mkdir(exist_ok=True)
    for year, frame in parts:
        sixteen_path = OUT_ROOT / "parts_16day" / f"event_year={year}" / "part.parquet"
        monthly_path = OUT_ROOT / "parts_monthly" / f"event_year={year}" / "part.parquet"
        sixteen_path.parent.mkdir(parents=True, exist_ok=True)
        monthly_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(sixteen_path, index=False, compression="zstd")
        _monthly_summary(frame).to_parquet(monthly_path, index=False, compression="zstd")
    links = pd.read_parquet(CURRENT_SOURCE).drop(columns=["run_id"], errors="ignore")
    links.to_parquet(OUT_ROOT / "matched_links.parquet", index=False, compression="zstd")
    catalogue.to_parquet(OUT_ROOT / "unit_event_catalog.parquet", index=False, compression="zstd")
    calendar = pd.read_csv(OLD_ROOT / "collection_calendar.csv")
    calendar.to_csv(OUT_ROOT / "collection_calendar.csv", index=False)
    source_sha = sha256(CURRENT_SOURCE)
    total_16 = sum(len(frame) for _, frame in parts)
    total_monthly = sum(len(pd.read_parquet(OUT_ROOT / "parts_monthly" / f"event_year={year}" / "part.parquet", columns=["grid_id"])) for year, _ in parts)
    records_16 = [{"path": str(OUT_ROOT / "parts_16day" / f"event_year={year}" / "part.parquet"), "rows": len(frame), "size_bytes": (OUT_ROOT / "parts_16day" / f"event_year={year}" / "part.parquet").stat().st_size, "sha256": sha256(OUT_ROOT / "parts_16day" / f"event_year={year}" / "part.parquet")} for year, frame in parts]
    records_monthly = [{"path": str(OUT_ROOT / "parts_monthly" / f"event_year={year}" / "part.parquet"), "rows": len(pd.read_parquet(OUT_ROOT / "parts_monthly" / f"event_year={year}" / "part.parquet", columns=["grid_id"])), "size_bytes": (OUT_ROOT / "parts_monthly" / f"event_year={year}" / "part.parquet").stat().st_size, "sha256": sha256(OUT_ROOT / "parts_monthly" / f"event_year={year}" / "part.parquet")} for year, _ in parts]
    manifest = {
        "analysis_signature": STAGE08_SIGNATURE,
        "analysis_version": "2026-09-09-stage08-current-full-history-v1",
        "created_at_utc": utc_now(), "event_years": list(EVENT_YEARS),
        "source_signature": source_sha,
        "input": {"pair_links": int(len(links)), "treatment_unit_events": int(catalogue["role"].eq("treatment").sum()), "control_unit_events": int(catalogue["role"].eq("control").sum()), "unit_events": int(len(catalogue)), "unique_grids": int(catalogue["grid_id"].nunique()), "source_sha256": source_sha},
        "matched_links": {"path": str(OUT_ROOT / "matched_links.parquet"), "rows": len(links), "size_bytes": (OUT_ROOT / "matched_links.parquet").stat().st_size, "sha256": sha256(OUT_ROOT / "matched_links.parquet")},
        "unit_event_catalog": {"path": str(OUT_ROOT / "unit_event_catalog.parquet"), "rows": len(catalogue), "size_bytes": (OUT_ROOT / "unit_event_catalog.parquet").stat().st_size, "sha256": sha256(OUT_ROOT / "unit_event_catalog.parquet")},
        "collection_calendar": {"path": str(OUT_ROOT / "collection_calendar.csv"), "rows": len(calendar), "size_bytes": (OUT_ROOT / "collection_calendar.csv").stat().st_size, "sha256": sha256(OUT_ROOT / "collection_calendar.csv")},
        "parts_16day": records_16, "parts_monthly": records_monthly,
        "total_16day_rows": int(total_16), "total_monthly_rows": int(total_monthly),
    }
    (OUT_ROOT / "subannual_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"analysis_signature": STAGE08_SIGNATURE, "event_years": list(EVENT_YEARS), "pair_links": len(links), "unit_events": len(catalogue), "rows_16day": total_16, "rows_monthly": total_monthly, "stage08_pass": True}
    (OUT_ROOT / "subannual_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {"source_sha256": source_sha, "source_path": str(CURRENT_SOURCE), "pair_links": len(links), "treatment_unit_events": int(catalogue["role"].eq("treatment").sum()), "control_unit_events": int(catalogue["role"].eq("control").sum()), "unit_events": len(catalogue), "event_years": list(EVENT_YEARS)}
    (OUT_ROOT / "input_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tasks = {"analysis_signature": STAGE08_SIGNATURE, "source_signature": source_sha, "parts": {str(year): {"event_year": year, "task_state": "COMPLETED", "validation_status": "complete", "expected_rows": len(frame)} for year, frame in parts}}
    (OUT_ROOT / "task_manifest.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": str(OUT_ROOT), "rows_16day": total_16, "rows_monthly": total_monthly, "validation_pass": True}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if OUT_ROOT.exists() and any(OUT_ROOT.iterdir()) and not args.force:
        raise RuntimeError(f"Output exists; pass --force only after reviewing {OUT_ROOT}")
    manifest = load_task_manifest()
    if any(task.get("state") != "COMPLETED" or not task.get("drive_files") for task in manifest["tasks"]):
        raise RuntimeError("New MODIS tasks are not all complete and visible in Drive")
    csvs = task_csvs(manifest)
    catalogue = pd.read_parquet(CURRENT_CATALOG)
    missing = pd.read_parquet(MISSING_CATALOG)
    new = _read_new_csvs(csvs)
    if len(new) != len(missing) * PERIODS_PER_EVENT:
        raise RuntimeError(f"New export rows {len(new)} != {len(missing) * PERIODS_PER_EVENT}")
    write_package(build_parts(catalogue, missing, new), catalogue)


if __name__ == "__main__":
    main()
