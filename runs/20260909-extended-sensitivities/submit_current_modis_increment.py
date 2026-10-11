#!/usr/bin/env python3
"""Submit the current full-history MOD13Q1 queue to Earth Engine.

The frozen historical MOD13Q1 exports are reused only for exact event-role keys
present in both catalogues. New keys are exported under a new Drive namespace;
no old task, file, or Stage 08 package is modified.
"""

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
ANALYSIS_ROOT = PROJECT_ROOT / "02_analysis/integrated/TropicalRainforest_integrated_analysis"
VENV_PYTHON = Path(sys.executable)
CURRENT_CATALOG = RUN_ROOT / "current_event_role_catalogue.parquet"
OLD_CATALOG = PROJECT_ROOT / "01_data/upstream/TropicalRainforest_cloud_analysis/08_subannual_matched/unit_event_catalog.parquet"
OLD_STAGE08 = PROJECT_ROOT / "01_data/upstream/TropicalRainforest_cloud_analysis/08_subannual_matched"
DRIVE_ROOT = Path(os.environ.get("TFDT_DRIVE_ROOT", str(PROJECT_ROOT / "external_exports")))
GEE_PROJECT = os.environ.get("TFDT_EE_PROJECT", "")
GEE_DRIVE_FOLDER = "stage08_mod13q1_full_history_1990_20260909"
TASK_PREFIX = "stage08_full_history_1990"
MODIS_COLLECTION = "MODIS/061/MOD13Q1"
SRTM_COLLECTION = "CGIAR/SRTM90_V4"
GRID_CRS = "EPSG:4326"
GRID_TRANSFORM = [0.05, 0, -180, 0, -0.05, 23.5]
EVI_SCALE = 0.0001
EVI_RANGE = (-0.1, 1.0)
ACCEPTED_SUMMARY_QA = (0, 1)
RELATIVE_YEARS = tuple(range(-3, 6))
EVENT_YEARS = tuple(range(2004, 2019))
PERIODS_PER_EVENT = 207
SHARD_UNITS = 750
MAX_ACTIVE_TASKS = 6
ANALYSIS_VERSION = "2026-09-09-stage08-current-full-history-v1"
TASK_MANIFEST = RUN_ROOT / "current_modis_increment_tasks.json"
MISSING_CATALOG = RUN_ROOT / "current_modis_missing_catalogue.parquet"
COLLECTION_CALENDAR = OLD_STAGE08 / "collection_calendar.csv"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(value: object, path: Path) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_catalogues() -> tuple[pd.DataFrame, pd.DataFrame]:
    current = pd.read_parquet(CURRENT_CATALOG)
    old = pd.read_parquet(OLD_CATALOG)
    key = ["grid_id", "event_year", "role"]
    for frame, label in ((current, "current"), (old, "old")):
        missing = set(key) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} catalogue lacks {sorted(missing)}")
        if frame.duplicated(key).any():
            raise ValueError(f"{label} catalogue contains duplicate event-role keys")
    missing = current.merge(old[key], on=key, how="left", indicator=True)
    missing = missing.loc[missing["_merge"].eq("left_only")].drop(columns="_merge")
    missing = missing.sort_values(["event_year", "role", "grid_id"], kind="mergesort").reset_index(drop=True)
    return current, missing


def task_records(catalogue: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    for year in EVENT_YEARS:
        part = catalogue.loc[catalogue["event_year"].eq(year)].copy()
        if part.empty:
            continue
        part = part.sort_values(["role", "grid_id"], kind="mergesort").reset_index(drop=True)
        shards = [part.iloc[start : start + SHARD_UNITS].copy() for start in range(0, len(part), SHARD_UNITS)]
        for shard_index, shard in enumerate(shards, start=1):
            stem = f"{TASK_PREFIX}_event_{year}_shard_{shard_index:02d}-of-{len(shards):02d}"
            records.append(
                {
                    "event_year": int(year),
                    "shard_index": int(shard_index),
                    "shard_count": int(len(shards)),
                    "unit_count": int(len(shard)),
                    "expected_rows": int(len(shard) * PERIODS_PER_EVENT),
                    "description": f"{stem}_{ANALYSIS_VERSION[-8:]}",
                    "file_prefix": f"{stem}_{ANALYSIS_VERSION[-8:]}",
                    "task_id": None,
                    "state": "NOT_SUBMITTED",
                    "error_message": None,
                    "drive_files": [],
                }
            )
    return records


def build_manifest(current: pd.DataFrame, missing: pd.DataFrame) -> dict:
    manifest = {
        "run_id": "expanded_control_pool_full_history_1990",
        "analysis_version": ANALYSIS_VERSION,
        "gee_project": GEE_PROJECT,
        "drive_root": str(DRIVE_ROOT),
        "drive_folder": GEE_DRIVE_FOLDER,
        "current_catalogue": str(CURRENT_CATALOG),
        "current_catalogue_sha256": sha256_file(CURRENT_CATALOG),
        "old_catalogue": str(OLD_CATALOG),
        "old_catalogue_sha256": sha256_file(OLD_CATALOG),
        "current_catalogue_rows": int(len(current)),
        "reused_event_role_rows": int(len(current) - len(missing)),
        "new_event_role_rows": int(len(missing)),
        "periods_per_event": PERIODS_PER_EVENT,
        "relative_years": list(RELATIVE_YEARS),
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "tasks": task_records(missing),
    }
    return manifest


def _import_ee():
    sys.path = [entry for entry in sys.path if entry not in {str(RUN_ROOT), str(ANALYSIS_ROOT)}]
    import ee

    if not GEE_PROJECT:
        raise ValueError("Set TFDT_EE_PROJECT before submitting Earth Engine tasks")
    ee.Initialize(project=GEE_PROJECT)
    return ee


def _build_ee_helpers(ee, calendar: pd.DataFrame):
    land_mask = ee.Image(SRTM_COLLECTION).select("elevation").mask().gt(0)

    def aggregate_modis_image(image):
        scaled_evi = image.select("EVI").multiply(EVI_SCALE)
        qa = image.select("SummaryQA")
        qa_ok = qa.eq(ACCEPTED_SUMMARY_QA[0]).Or(qa.eq(ACCEPTED_SUMMARY_QA[1]))
        range_ok = scaled_evi.gte(EVI_RANGE[0]).And(scaled_evi.lte(EVI_RANGE[1]))
        valid_mask = qa_ok.And(range_ok).And(land_mask)
        accepted_evi = scaled_evi.updateMask(valid_mask)
        native_valid = valid_mask.unmask(0).toFloat().updateMask(land_mask)
        evi_005 = accepted_evi.reduceResolution(
            reducer=ee.Reducer.mean(), bestEffort=False, maxPixels=65535
        ).reproject(crs=GRID_CRS, crsTransform=GRID_TRANSFORM).rename("evi").unmask(-9999.0)
        valid_fraction = native_valid.reduceResolution(
            reducer=ee.Reducer.mean(), bestEffort=False, maxPixels=65535
        ).reproject(crs=GRID_CRS, crsTransform=GRID_TRANSFORM).rename("valid_pixel_fraction").unmask(0)
        return evi_005.addBands(valid_fraction)

    def build_collection(catalogue_part: pd.DataFrame, event_year: int):
        features = []
        for row in catalogue_part.itertuples(index=False):
            properties = {
                "grid_id": int(row.grid_id),
                "event_year": int(row.event_year),
                "role": str(row.role),
                "lon": float(row.lon),
                "lat": float(row.lat),
                "macroregion": str(row.macroregion),
                "source_signature": str(row.source_signature),
                "analysis_version": ANALYSIS_VERSION,
            }
            features.append(ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), properties))
        points = ee.FeatureCollection(features)
        start = pd.Timestamp(event_year - 3, 1, 1)
        stop = pd.Timestamp(event_year + 6, 1, 1)
        dates = calendar.loc[calendar["observation_date"].between(start, stop, inclusive="left")]
        if len(dates) != PERIODS_PER_EVENT:
            raise ValueError(f"Event year {event_year} has {len(dates)} calendar rows")
        collections = []
        for row in dates.itertuples(index=False):
            image = ee.Image(f"{MODIS_COLLECTION}/{row.modis_image_id}")
            sampled = aggregate_modis_image(image).reduceRegions(
                collection=points,
                reducer=ee.Reducer.first(),
                crs=GRID_CRS,
                crsTransform=GRID_TRANSFORM,
                tileScale=4,
            )
            relative_year = int(row.observation_date.year - event_year)
            sampled = sampled.map(
                lambda feature, row=row, relative_year=relative_year: feature.set(
                    {
                        "observation_date": row.observation_date.strftime("%Y-%m-%d"),
                        "calendar_year": int(row.observation_date.year),
                        "calendar_month": int(row.observation_date.month),
                        "day_of_year": int(row.observation_date.dayofyear),
                        "relative_year": relative_year,
                        "modis_image_id": str(row.modis_image_id),
                        "system_time_start": int(row.system_time_start),
                    }
                )
            )
            collections.append(sampled)
        return ee.FeatureCollection(collections).flatten()

    return build_collection


def _drive_files(prefix: str) -> list[str]:
    folder = DRIVE_ROOT / GEE_DRIVE_FOLDER
    return [str(path) for path in sorted(folder.glob(prefix + "*.csv"))] if folder.is_dir() else []


def reconcile_manifest(manifest: dict, ee) -> None:
    task_ids = [task["task_id"] for task in manifest["tasks"] if task.get("task_id")]
    status_by_id = {}
    if task_ids:
        status_by_id = {status.get("id"): status for status in ee.data.getTaskStatus(task_ids)}
    for task in manifest["tasks"]:
        if task.get("task_id") in status_by_id:
            status = status_by_id[task["task_id"]]
            task["state"] = status.get("state", "UNKNOWN")
            task["error_message"] = status.get("error_message")
        task["drive_files"] = _drive_files(task["file_prefix"])
        if task["drive_files"] and task["state"] in {"COMPLETED", "UNKNOWN", "NOT_SUBMITTED"}:
            task["state"] = "COMPLETED"
    manifest["updated_at_utc"] = utc_now()


def submit_new_tasks(manifest: dict, missing: pd.DataFrame, ee) -> list[str]:
    calendar = pd.read_csv(COLLECTION_CALENDAR, parse_dates=["observation_date"])
    build_collection = _build_ee_helpers(ee, calendar)
    active_states = {"READY", "RUNNING", "SUBMITTED"}
    active = sum(task["state"] in active_states for task in manifest["tasks"])
    submitted: list[str] = []
    for task_record in manifest["tasks"]:
        if active >= MAX_ACTIVE_TASKS:
            break
        if task_record["state"] != "NOT_SUBMITTED" or task_record.get("drive_files"):
            continue
        year = int(task_record["event_year"])
        index = int(task_record["shard_index"])
        count = int(task_record["shard_count"])
        part = missing.loc[missing["event_year"].eq(year)].sort_values(["role", "grid_id"], kind="mergesort").reset_index(drop=True)
        start = (index - 1) * SHARD_UNITS
        shard = part.iloc[start : start + SHARD_UNITS].copy()
        if len(shard) != int(task_record["unit_count"]):
            raise RuntimeError(f"Shard size mismatch for {task_record['description']}")
        collection = build_collection(shard, year)
        task = ee.batch.Export.table.toDrive(
            collection=collection,
            description=task_record["description"],
            folder=GEE_DRIVE_FOLDER,
            fileNamePrefix=task_record["file_prefix"],
            fileFormat="CSV",
            selectors=[
                "grid_id", "event_year", "role", "lon", "lat", "macroregion",
                "observation_date", "calendar_year", "calendar_month", "day_of_year",
                "relative_year", "modis_image_id", "system_time_start", "evi",
                "valid_pixel_fraction", "source_signature", "analysis_version",
            ],
        )
        task.start()
        task_record["task_id"] = task.id
        task_record["state"] = "READY"
        task_record["submitted_at_utc"] = utc_now()
        submitted.append(task.id)
        active += 1
        json_dump(manifest, TASK_MANIFEST)
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if not CURRENT_CATALOG.is_file() or not OLD_CATALOG.is_file():
        raise FileNotFoundError("Current or old Stage 08 catalogue is missing")
    current, missing = load_catalogues()
    missing.to_parquet(MISSING_CATALOG, index=False)
    if TASK_MANIFEST.is_file():
        manifest = json.loads(TASK_MANIFEST.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(current, missing)
        json_dump(manifest, TASK_MANIFEST)
    summary = {
        "current_rows": int(len(current)),
        "missing_rows": int(len(missing)),
        "reused_rows": int(len(current) - len(missing)),
        "tasks": int(len(manifest["tasks"])),
        "expected_new_rows": int(len(missing) * PERIODS_PER_EVENT),
        "by_year": missing.groupby("event_year").size().astype(int).to_dict(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run and not (args.submit or args.reconcile):
        return
    ee = _import_ee()
    reconcile_manifest(manifest, ee)
    if args.submit:
        submitted = submit_new_tasks(manifest, missing, ee)
        print(json.dumps({"submitted": submitted}, ensure_ascii=False, indent=2))
    json_dump(manifest, TASK_MANIFEST)
    if args.reconcile:
        print(json.dumps({"states": pd.Series([t["state"] for t in manifest["tasks"]]).value_counts().to_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
