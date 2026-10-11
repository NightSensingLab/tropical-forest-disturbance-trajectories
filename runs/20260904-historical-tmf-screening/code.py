#!/usr/bin/env python3
"""Screen 2004-2018 TMF candidate events for pre-2001 threshold events.

The Earth Engine part samples candidate centres in 2-degree dispatch bins and
50-point batches.  Each response contains the 1990 event indicator and
1991-2000 annual transition indicators, together with their valid-coverage
denominators.  The local TMF panel supplies 2001-2023 values, so no original
MODIS/TMF rasters are reprocessed locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = ROOT / "02_analysis/integrated/TropicalRainforest_integrated_analysis"
DEFAULT_RUN = ROOT / "runs/20260904-historical-tmf-screening"
DEFAULT_CATALOG = ANALYSIS_ROOT / "results/06_event_response/event_catalog.parquet"
DEFAULT_STAGE03 = ROOT / "01_data/upstream/TropicalRainforest_cloud_analysis/03_combined_panel/parts"
DEFAULT_FIXED = ANALYSIS_ROOT / "results/major23_revision/all_first_matched_event_window.parquet"
DEFAULT_PAIRS = ANALYSIS_ROOT / "results/major23_revision/all_first_matched_pairs.parquet"
DEFAULT_ISOLATED = ANALYSIS_ROOT / "results/major23_revision/isolated_matched_event_window.parquet"

PROJECT_ID = os.environ.get("TFDT_EE_PROJECT", "")
EE_REQUEST_DEADLINE_MS = 120_000
TMF_ANNUAL_BASE = "projects/JRC/TMF/v1_2025/AnnualChanges"
TMF_DEG_YEAR = "projects/JRC/TMF/v1_2025/DegradationYear"
TMF_DEF_YEAR = "projects/JRC/TMF/v1_2025/DeforestationYear"
TRANSFORM = [0.05, 0, -180, 0, -0.05, 90]
TILE_DEGREES = 2
DEFAULT_BATCH_SIZE = 50
DEFAULT_WORKERS = 8
YEARS = tuple(range(1990, 2001))
PANEL_YEARS = tuple(range(2001, 2024))
FULL_COVER_VALUE = 0.0029050935991108418

ANNUAL_ASSETS = {
    "SAM": f"{TMF_ANNUAL_BASE}/SAM",
    "AFR": f"{TMF_ANNUAL_BASE}/AFR",
    "ASIA": f"{TMF_ANNUAL_BASE}/ASIA",
}
DEG_YEAR_ASSETS = {
    "SAM": f"{TMF_DEG_YEAR}/SAM",
    "AFR": f"{TMF_DEG_YEAR}/AFR",
    "ASIA": f"{TMF_DEG_YEAR}/ASIA",
}
DEF_YEAR_ASSETS = {
    "SAM": (f"{TMF_DEF_YEAR}/AOI_SAM1", f"{TMF_DEF_YEAR}/AOI_SAM2"),
    "AFR": (f"{TMF_DEF_YEAR}/AFR",),
    "ASIA": (
        f"{TMF_DEF_YEAR}/AOI_ASIA2",
        f"{TMF_DEF_YEAR}/AOI_ASIA3",
        f"{TMF_DEF_YEAR}/AOI_ASIA4",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def continent_for_lon(lon: float) -> str:
    # Candidate coordinates cover the three TMF continent products.  The
    # -20/60 boundaries only dispatch tiles; macroregions remain unchanged.
    if lon < -20:
        return "SAM"
    if lon < 60:
        return "AFR"
    return "ASIA"


def load_candidates(path: Path) -> pd.DataFrame:
    columns = [
        "grid_id",
        "grid_row",
        "grid_col",
        "lon",
        "lat",
        "cell_area_m2",
        "analysis_area_m2",
        "macroregion",
        "event_year",
        "event_type",
        "disturbance_share",
        "degradation_share",
        "deforestation_share",
        "event_area_m2",
        "primary_window_eligible",
        "source_part",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[frame["primary_window_eligible"].eq(True)].copy()
    frame = frame.drop_duplicates(["grid_id", "event_year"], keep="first")
    if len(frame) != 29738:
        raise RuntimeError(f"Expected 29,738 balanced candidates, found {len(frame)}")
    frame["continent"] = frame["lon"].map(continent_for_lon)
    frame["lon_bin"] = np.floor(frame["lon"] / TILE_DEGREES).astype(int) * TILE_DEGREES
    frame["lat_bin"] = np.floor(frame["lat"] / TILE_DEGREES).astype(int) * TILE_DEGREES
    return frame


def build_gee_image(continent: str):
    # geemap is imported only after the script directory is removed from
    # sys.path, because a local code.py otherwise shadows stdlib code.py.
    import ee

    annual = ee.Image(ANNUAL_ASSETS[continent])
    source_projection = annual.select("Dec1990").projection()
    deg_year = ee.Image(DEG_YEAR_ASSETS[continent])
    def_year = ee.ImageCollection.fromImages(
        [ee.Image(asset) for asset in DEF_YEAR_ASSETS[continent]]
    ).mosaic()
    native_bands = []
    band_names = []

    def valid_class(image):
        return image.gte(1).And(image.lte(6))

    # 1990 cannot be reconstructed from a preceding AnnualChanges band.  The
    # official first-year maps provide the conservative 1990 event indicator.
    base = annual.select("Dec1990")
    valid = valid_class(base)
    event = deg_year.eq(1990).Or(def_year.eq(1990))
    native_bands.extend([event.And(valid).unmask(0), valid.unmask(0)])
    band_names.extend(["historical_event_1990", "historical_valid_1990"])

    # For 1991-2000, reproduce the event definitions used by the local panel:
    # newly degraded = 1 -> 2; newly deforested = (1/2) -> (3/5).
    for year in range(1991, 2001):
        prev = annual.select(f"Dec{year - 1}")
        curr = annual.select(f"Dec{year}")
        valid = valid_class(prev).And(valid_class(curr))
        degradation = prev.eq(1).And(curr.eq(2))
        deforestation = prev.eq(1).Or(prev.eq(2)).And(curr.eq(3).Or(curr.eq(5)))
        event = degradation.Or(deforestation)
        native_bands.append(event.And(valid).unmask(0))
        band_names.append(f"historical_event_{year}")
    # Aggregate the complete stack in one server-side reducer.  Repeating
    # reduceResolution independently for every band can exceed GEE memory even
    # for a small download region.
    return (
        ee.Image.cat(native_bands)
        .rename(band_names)
        .setDefaultProjection(source_projection)
        .reduceResolution(ee.Reducer.mean(), maxPixels=65535)
        .reproject(crs="EPSG:4326", crsTransform=TRANSFORM)
        .toFloat()
    )


def tile_region(west: int, south: int):
    import ee

    return ee.Geometry.Rectangle(
        [west, south, west + TILE_DEGREES, south + TILE_DEGREES], None, False
    )


def download_tile(image, continent: str, west: int, south: int, path: Path):
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return "cached"
    raw_path = path.with_suffix(".raw.tif")
    for stale in (raw_path, path):
        if stale.exists():
            stale.unlink()
    last_error = None
    for attempt in range(1, 4):
        try:
            params = {
                "name": path.stem,
                "filePerBand": False,
                "scale": None,
                "region": tile_region(west, south),
                "crs": "EPSG:4326",
                "crs_transform": TRANSFORM,
                "format": "GEO_TIFF",
            }
            url = image.getDownloadURL(params)
            response = requests.get(url, stream=True, timeout=(15, 180))
            if response.status_code != 200:
                detail = response.text[:1000]
                raise RuntimeError(f"HTTP {response.status_code}: {detail}")
            with raw_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            os.replace(raw_path, path)
            if path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded empty tile: {path}")
            return "downloaded"
        except Exception as exc:  # noqa: BLE001 - retry network/API failures
            last_error = exc
            for stale in (raw_path, path):
                if stale.exists():
                    stale.unlink()
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed tile {continent}/{west}/{south}: {last_error}")


def sample_tile(path: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    import rasterio

    if candidates.empty:
        return pd.DataFrame()
    with rasterio.open(path) as src:
        data = src.read()
        rows, cols = zip(*(src.index(float(lon), float(lat)) for lon, lat in zip(candidates.lon, candidates.lat)))
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
        inside = (
            (rows >= 0)
            & (rows < src.height)
            & (cols >= 0)
            & (cols < src.width)
        )
        if not inside.all():
            missing = candidates.loc[~inside, "grid_id"].tolist()[:5]
            raise RuntimeError(f"Tile sample outside raster bounds, examples={missing}")
        values = data[:, rows, cols]
    out = candidates.loc[:, ["grid_id", "event_year"]].copy().reset_index(drop=True)
    for idx, year in enumerate(YEARS):
        event_raw = values[2 * idx].astype(float)
        valid_raw = values[2 * idx + 1].astype(float)
        # Earth Engine represents pixels outside a continental footprint as
        # non-finite float values in a direct GeoTIFF download.
        event_raw[~np.isfinite(event_raw)] = np.nan
        valid_raw[~np.isfinite(valid_raw)] = np.nan
        out[f"historical_event_{year}_raw"] = event_raw
        out[f"historical_valid_{year}_raw"] = valid_raw
        out[f"historical_event_{year}_share"] = np.divide(
            event_raw,
            valid_raw,
            out=np.full(len(out), np.nan, dtype=float),
            where=np.isfinite(valid_raw) & (valid_raw > 0),
        )
    out["history_tile"] = path.name
    return out


def sample_point_batch(image, candidates: pd.DataFrame, part_path: Path) -> pd.DataFrame:
    """Sample a small candidate batch and cache the returned properties."""
    import ee

    if part_path.exists() and part_path.stat().st_size > 0:
        cached = pd.read_parquet(part_path)
        if "event_year" not in cached.columns:
            cached = cached.merge(
                candidates.loc[:, ["grid_id", "event_year"]],
                on="grid_id",
                how="left",
                validate="one_to_one",
            )
        return cached
    features = [
        ee.Feature(
            ee.Geometry.Point([float(row.lon), float(row.lat)]),
            {"grid_id": int(row.grid_id)},
        )
        for row in candidates.itertuples()
    ]
    collection = ee.FeatureCollection(features)
    last_error = None
    for attempt in range(1, 4):
        try:
            response = image.sampleRegions(
                collection=collection,
                properties=["grid_id"],
                geometries=False,
                tileScale=16,
            ).getInfo()
            properties = [feature["properties"] for feature in response["features"]]
            result = pd.DataFrame(properties)
            if result.empty or "grid_id" not in result.columns:
                result = pd.DataFrame({"grid_id": candidates.grid_id.astype(int)})
            else:
                result["grid_id"] = result["grid_id"].astype(int)
                missing_ids = sorted(set(candidates.grid_id.astype(int)) - set(result.grid_id.astype(int)))
                if missing_ids:
                    result = pd.concat(
                        [result, pd.DataFrame({"grid_id": missing_ids})],
                        ignore_index=True,
                    )
            if result.grid_id.nunique() != len(candidates):
                raise RuntimeError(
                    f"Expected {len(candidates)} sampled candidates, got {result.grid_id.nunique()}"
                )
            result = result.merge(
                candidates.loc[:, ["grid_id", "event_year"]],
                on="grid_id",
                how="left",
                validate="one_to_one",
            )
            part_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = part_path.with_suffix(part_path.suffix + ".tmp")
            result.to_parquet(temporary, index=False)
            os.replace(temporary, part_path)
            return result
        except Exception as exc:  # noqa: BLE001 - retry transient EE/API failures
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed point batch {part_path.name}: {last_error}")


def download_and_sample(
    candidates: pd.DataFrame,
    run_dir: Path,
    project: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = DEFAULT_WORKERS,
    shard_index: int = 0,
    shard_count: int = 1,
) -> pd.DataFrame:
    # Remove the run directory from sys.path before geemap import (see skill).
    run_text = str(run_dir.resolve())
    if sys.path and Path(sys.path[0]).resolve() == run_dir.resolve():
        sys.path.pop(0)
    import ee
    import geemap  # noqa: F401 - verified package is available for GEE helpers

    ee.Initialize(project=project)
    # Bound synchronous aggregation calls so a stalled API request is retried
    # by the per-batch retry loop instead of blocking the worker pool.
    ee.data.setDeadline(EE_REQUEST_DEADLINE_MS)
    ee.data.setMaxRetries(0)
    if batch_size < 1 or workers < 1 or shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("batch_size, workers, shard_count must be positive and shard_index must be valid")
    by_tile = candidates.groupby(["continent", "lon_bin", "lat_bin"], sort=True)
    tasks = []
    part_dir = run_dir / "point_parts"
    for (continent, west, south), tile_candidates in by_tile:
        tile_candidates = tile_candidates.reset_index(drop=True)
        for batch_no, start in enumerate(range(0, len(tile_candidates), batch_size), start=1):
            chunk = tile_candidates.iloc[start : start + batch_size].copy()
            part_name = f"{continent}_lon{int(west):+04d}_lat{int(south):+04d}_batch{batch_no:03d}.parquet"
            tasks.append((continent, chunk, part_dir / part_name))
    tasks = [task for task_no, task in enumerate(tasks) if task_no % shard_count == shard_index]
    images = {continent: build_gee_image(continent) for continent in sorted(candidates.continent.unique())}

    def run_task(task):
        continent, chunk, part_path = task
        result = sample_point_batch(images[continent], chunk, part_path)
        return part_path, result

    sampled = []
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            part_path, result = future.result()
            sampled.append(result)
            if completed == 1 or completed % 20 == 0 or completed == len(futures):
                print(f"sampled point batches shard {shard_index}: {completed}/{len(futures)} ({part_path.name})", flush=True)
    if not sampled:
        raise RuntimeError("No candidate point batches were produced")
    result = pd.concat(sampled, ignore_index=True)
    expected_ids = {
        int(grid_id)
        for _, chunk, _ in tasks
        for grid_id in chunk.grid_id.astype(int).tolist()
    }
    if result["grid_id"].nunique() != len(expected_ids):
        raise RuntimeError("Tile extraction did not return one row per candidate")
    return result


def load_local_early(candidates: pd.DataFrame, stage03_dir: Path) -> pd.DataFrame:
    wanted = set(candidates.grid_id.astype(int))
    rows = []
    for source_part, group in candidates.groupby("source_part", sort=True):
        part = stage03_dir / f"{source_part}.parquet"
        if not part.exists():
            raise FileNotFoundError(part)
        frame = pd.read_parquet(
            part,
            columns=["grid_id", "year", "tmf_domain_pct", "tmf_disturb_event_pct"],
            filters=[("year", "in", list(PANEL_YEARS))],
        )
        frame = frame.loc[frame.grid_id.isin(wanted)].copy()
        if frame.empty:
            continue
        frame["disturbance_share"] = np.divide(
            frame["tmf_disturb_event_pct"],
            frame["tmf_domain_pct"],
            out=np.full(len(frame), np.nan),
            where=np.isfinite(frame["tmf_domain_pct"]) & frame["tmf_domain_pct"].gt(0),
        )
        frame["calendar_year"] = frame["year"].astype(int)
        frame["coverage_fraction"] = np.where(
            np.isfinite(frame["tmf_domain_pct"]),
            frame["tmf_domain_pct"] / FULL_COVER_VALUE,
            np.nan,
        )
        frame["source"] = "local_tmf_panel_2001_2023"
        rows.append(frame.loc[:, ["grid_id", "calendar_year", "disturbance_share", "coverage_fraction", "source"]])
    if not rows:
        raise RuntimeError("No local 2001-2023 rows found for candidates")
    result = pd.concat(rows, ignore_index=True)
    result = result.loc[result.grid_id.isin(wanted)].copy()
    return result


def combine_annual(candidates: pd.DataFrame, gee_wide: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    base = candidates.loc[:, ["grid_id", "event_year"]].drop_duplicates().copy()
    wide = base.merge(gee_wide, on=["grid_id", "event_year"], how="left", validate="one_to_one")
    rows = []
    for row in wide.itertuples(index=False):
        for year in YEARS:
            event_raw = getattr(row, f"historical_event_{year}")
            # The valid TMF footprint is stable across AnnualChanges bands;
            # one denominator from Dec1990 is sufficient and avoids ten
            # repeated coverage reductions.
            valid_raw = getattr(row, "historical_valid_1990")
            share = np.divide(
                event_raw,
                valid_raw,
                out=np.array(np.nan, dtype=float),
                where=np.isfinite(event_raw) and np.isfinite(valid_raw) and valid_raw > 0,
            ).item()
            rows.append(
                {
                    "grid_id": int(row.grid_id),
                    "event_year": int(row.event_year),
                    "calendar_year": year,
                    "disturbance_share": share,
                    "coverage_fraction": valid_raw,
                    "source": "gee_tmf_annualchanges_1990_firstyear_or_transition",
                }
            )
    gee_long = pd.DataFrame(rows)
    # Keep the raw coverage value separately; the summary uses >0 as data
    # availability, while coverage fractions remain diagnostic information.
    gee_long["coverage_available"] = gee_long["coverage_fraction"].gt(0)
    local = local.copy()
    local["coverage_available"] = local["coverage_fraction"].gt(0)
    combined = pd.concat(
        [gee_long.loc[:, ["grid_id", "event_year", "calendar_year", "disturbance_share", "coverage_fraction", "coverage_available", "source"]],
         local.loc[:, ["grid_id", "calendar_year", "disturbance_share", "coverage_fraction", "coverage_available", "source"]].assign(event_year=np.nan)],
        ignore_index=True,
    )
    # event_year is carried from the candidate table for local rows.
    combined = combined.drop(columns=["event_year"]).merge(base, on="grid_id", how="left", validate="many_to_one")
    combined = combined.loc[:, ["grid_id", "event_year", "calendar_year", "disturbance_share", "coverage_fraction", "coverage_available", "source"]]
    combined = combined.sort_values(["grid_id", "calendar_year"]).reset_index(drop=True)
    expected = len(base) * (len(YEARS) + len(PANEL_YEARS))
    if len(combined) != expected:
        raise RuntimeError(f"Expected {expected} combined annual rows, found {len(combined)}")
    return combined


def derive_screening(candidates: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for key, group in annual.groupby(["grid_id", "event_year"], sort=True):
        grid_id, event_year = int(key[0]), int(key[1])
        prior = group.loc[group.calendar_year < event_year].copy()
        found_mask = prior["disturbance_share"].ge(0.10)
        found = bool(found_mask.any())
        complete = bool(prior["coverage_available"].all()) and len(prior) == event_year - 1990
        found_year = int(prior.loc[found_mask, "calendar_year"].min()) if found else np.nan
        max_share = float(prior["disturbance_share"].max()) if prior["disturbance_share"].notna().any() else np.nan
        missing_years = prior.loc[~prior.coverage_available, "calendar_year"].astype(int).tolist()
        if found:
            status = "historical_threshold_event_found"
        elif complete:
            status = "complete_history_no_threshold_event"
        else:
            status = "unknown_incomplete_history"
        summary_rows.append(
            {
                "grid_id": grid_id,
                "event_year": event_year,
                "prior_threshold_status": status,
                "prior_threshold_event": found,
                "prior_threshold_first_year": found_year,
                "prior_threshold_max_share": max_share,
                "history_complete": complete,
                "history_missing_years": ",".join(map(str, missing_years)),
                "history_years_available": int(prior.coverage_available.sum()),
                "history_years_expected": int(event_year - 1990),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return candidates.merge(summary, on=["grid_id", "event_year"], how="left", validate="one_to_one")


def cohort_tables(screened: pd.DataFrame, fixed_path: Path, pairs_path: Path, isolated_path: Path):
    fixed_window = pd.read_parquet(
        fixed_path,
        columns=["treatment_grid_id", "event_year", "baseline_undist_share"],
    ).drop_duplicates(["treatment_grid_id", "event_year"])
    fixed = screened.loc[screened.grid_id.isin(fixed_window.treatment_grid_id)].copy()
    fixed = fixed.merge(
        fixed_window.rename(columns={"treatment_grid_id": "grid_id"}),
        on=["grid_id", "event_year"],
        how="left",
        validate="one_to_one",
    )
    fixed["treatment_grid_id"] = fixed["grid_id"]
    if fixed.grid_id.nunique() != 20054:
        raise RuntimeError(f"Expected 20,054 all-first fixed treatments, found {fixed.grid_id.nunique()}")
    isolated_ids = pd.read_parquet(isolated_path, columns=["treatment_grid_id", "event_year"]).drop_duplicates()
    isolated = fixed.loc[fixed.grid_id.isin(isolated_ids.treatment_grid_id)].copy()
    if isolated.grid_id.nunique() != 15674:
        raise RuntimeError(f"Expected 15,674 isolated treatments, found {isolated.grid_id.nunique()}")

    # Recompute current common-support membership using the canonical function.
    sys.path.insert(0, str(ANALYSIS_ROOT))
    from pipeline.matched_estimation import common_support_weights

    common_input = fixed.loc[:, [
        "treatment_grid_id",
        "event_year",
        "event_type",
        "macroregion",
        "event_area_m2",
        "disturbance_share",
        "degradation_share",
        "deforestation_share",
        "baseline_undist_share",
    ]].copy()
    common, diagnostics = common_support_weights(common_input)
    if len(common) != 13788:
        raise RuntimeError(f"Expected 13,788 common-support treatments, found {len(common)}")
    common_ids = set(common.treatment_grid_id.astype(int))
    fixed["common_support"] = fixed.grid_id.isin(common_ids)
    isolated["common_support"] = isolated.grid_id.isin(common_ids)
    cohorts = {
        "all_candidates": screened,
        "all_first_fixed_matched": fixed,
        "isolated_fixed_matched": isolated,
        "common_support": fixed.loc[fixed.common_support].copy(),
    }
    return cohorts, diagnostics


def make_reports(screened: pd.DataFrame, cohorts: dict[str, pd.DataFrame], diagnostics: dict, run_dir: Path, project: str):
    rows = []
    for cohort_name, frame in cohorts.items():
        for region_name, region in [("Global", frame)] + list(frame.groupby("macroregion", sort=True)):
            for event_type, group in [("All", region)] + list(region.groupby("event_type", sort=True)):
                n = int(len(group))
                found = int(group.prior_threshold_event.sum())
                rows.append(
                    {
                        "cohort": cohort_name,
                        "scope": region_name,
                        "event_type": event_type,
                        "n_events": n,
                        "n_historical_threshold_event": found,
                        "proportion_historical_threshold_event": found / n if n else np.nan,
                        "n_complete_history": int(group.history_complete.sum()),
                        "n_unknown_incomplete_history": int((group.prior_threshold_status == "unknown_incomplete_history").sum()),
                    }
                )
    regional = pd.DataFrame(rows)
    regional.to_csv(run_dir / "historical_screening_by_cohort_region_type.csv", index=False)
    screened.to_parquet(run_dir / "candidate_historical_screening.parquet", index=False)

    status = (
        screened.groupby(["prior_threshold_status"], as_index=False)
        .size()
        .rename(columns={"size": "n_events"})
    )
    status.to_csv(run_dir / "historical_screening_status.csv", index=False)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_id": project,
        "candidate_count": int(len(screened)),
        "status_counts": {str(r.prior_threshold_status): int(r.n_events) for r in status.itertuples()},
        "cohort_counts": {name: int(frame.grid_id.nunique()) for name, frame in cohorts.items()},
        "common_support_diagnostics": diagnostics,
        "interpretation": {
            "first_event_definition": "A current event is first since the 2001-2023 panel began; this screening checks the full 1990-to-event-year-minus-one history.",
            "1990_method": "DegradationYear==1990 or DeforestationYear==1990, constrained by valid Dec1990 AnnualChanges coverage.",
            "1991_2000_method": "AnnualChanges year-to-year transitions: 1->2 for degradation and (1/2)->(3/5) for deforestation.",
            "unknown_status": "No threshold event was observed, but at least one pre-event year lacked valid TMF coverage in the extracted history.",
            "cohort_label_note": "The 20,054 cohort is the all-first fixed-matched sample; 15,674 is the isolated fixed-matched sensitivity cohort; 13,788 is the recomputed common-support subset.",
        },
    }
    (run_dir / "historical_screening_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return regional, report


def write_annual(annual: pd.DataFrame, run_dir: Path):
    annual.to_parquet(run_dir / "historical_annual_event_shares.parquet", index=False)
    annual.to_csv(run_dir / "historical_annual_event_shares.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--stage03-dir", type=Path, default=DEFAULT_STAGE03)
    parser.add_argument("--fixed-window", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--isolated-window", type=Path, default=DEFAULT_ISOLATED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.catalog)
    tile_count = candidates.loc[:, ["continent", "lon_bin", "lat_bin"]].drop_duplicates().shape[0]
    print(json.dumps({"project": args.project, "candidates": len(candidates), "tiles": int(tile_count), "tile_degrees": TILE_DEGREES, "years": list(YEARS)}, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    # A force flag removes only this run's tile cache, never source data.
    if args.force_download:
        for tile in (run_dir / "tiles").glob("*.tif"):
            tile.unlink()
        for part in (run_dir / "point_parts").glob("*.parquet"):
            part.unlink()
    gee_wide = download_and_sample(
        candidates,
        run_dir,
        args.project,
        args.batch_size,
        args.workers,
        args.shard_index,
        args.shard_count,
    )
    if args.extract_only:
        print(f"extract-only complete for shard {args.shard_index}/{args.shard_count}: {len(gee_wide)} rows", flush=True)
        return
    local = load_local_early(candidates, args.stage03_dir)
    annual = combine_annual(candidates, gee_wide, local)
    write_annual(annual, run_dir)
    screened = derive_screening(candidates, annual)
    cohorts, diagnostics = cohort_tables(screened, args.fixed_window, args.pairs, args.isolated_window)
    _, report = make_reports(screened, cohorts, diagnostics, run_dir, args.project)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    # Prevent the local code.py filename from shadowing Python's stdlib code.
    if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
        sys.path.pop(0)
    main()
