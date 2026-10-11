#!/usr/bin/env python3
"""Extract 1990-2000 TMF history for the expanded local control universe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


RUN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_ROOT.parents[1]
SCREENING_DIR = PROJECT_ROOT / "runs/20260904-historical-tmf-screening"
SCREENING_CODE = SCREENING_DIR / "code.py"
OLD_EXTRACTION = PROJECT_ROOT / "runs/20260904-full-history-first-event/control_history_extraction/point_parts"


def _load_screening_module():
    spec = importlib.util.spec_from_file_location("historical_screening_code_expanded", SCREENING_CODE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _covered_ids(cache_path: Path, old_parts: Path, new_parts: Path) -> set[int]:
    covered: set[int] = set()
    if cache_path.exists():
        covered.update(pd.read_parquet(cache_path, columns=["grid_id"])["grid_id"].astype(int).unique().tolist())
    for directory in (old_parts, new_parts):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("**/*.parquet")):
            covered.update(pd.read_parquet(path, columns=["grid_id"])["grid_id"].astype(int).unique().tolist())
    return covered


def _continent(lon: float) -> str:
    if lon < -20.0:
        return "SAM"
    if lon < 60.0:
        return "AFR"
    return "ASIA"


def prepare_targets(universe_path: Path, cache_path: Path, old_parts: Path, new_parts: Path) -> tuple[pd.DataFrame, dict]:
    universe = pd.read_parquet(universe_path)
    required = {"grid_id", "lon", "lat"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"Expanded universe lacks columns: {sorted(missing)}")
    covered = _covered_ids(cache_path, old_parts, new_parts)
    missing_universe = universe.loc[~universe["grid_id"].astype(int).isin(covered)].copy()
    missing_universe["grid_id"] = missing_universe["grid_id"].astype(int)
    missing_universe["event_year"] = 2018
    missing_universe["continent"] = missing_universe["lon"].map(_continent)
    missing_universe["lon_bin"] = np.floor(missing_universe["lon"] / 2.0).astype(int) * 2
    missing_universe["lat_bin"] = np.floor(missing_universe["lat"] / 2.0).astype(int) * 2
    targets = missing_universe.loc[:, ["grid_id", "lon", "lat", "event_year", "continent", "lon_bin", "lat_bin"]].sort_values(["continent", "lon_bin", "lat_bin", "grid_id"], kind="mergesort").reset_index(drop=True)
    diagnostics = {
        "universe_grids": int(len(universe)),
        "covered_grid_ids": int(len(set(universe["grid_id"].astype(int)) & covered)),
        "missing_grid_ids": int(len(targets)),
        "existing_new_part_ids": int(len(_covered_ids(cache_path, Path("/does/not/exist"), new_parts))),
    }
    return targets, diagnostics


def download_and_sample_tiles(screen, targets: pd.DataFrame, run_dir: Path, project: str, workers: int, limit_tiles: int = 0) -> pd.DataFrame:
    """Download one 2-degree multi-band tile per occupied bin and sample locally."""
    import ee

    if targets.empty:
        raise RuntimeError("No tile targets were supplied")
    ee.Initialize(project=project)
    ee.data.setDeadline(120_000)
    ee.data.setMaxRetries(0)
    tile_dir = run_dir / "tiles"
    part_dir = run_dir / "tile_parts"
    grouped = list(targets.groupby(["continent", "lon_bin", "lat_bin"], sort=True))
    if limit_tiles:
        if limit_tiles < 1:
            raise ValueError("--limit-tiles must be positive when supplied")
        grouped = grouped[:limit_tiles]
    images = {continent: screen.build_gee_image(continent) for continent in sorted(targets["continent"].unique())}

    def one_tile(item):
        (continent, west, south), candidates = item
        tile_name = f"{continent}_lon{int(west):+04d}_lat{int(south):+04d}"
        tile_path = tile_dir / f"{tile_name}.tif"
        part_path = part_dir / f"{tile_name}.parquet"
        if part_path.exists() and part_path.stat().st_size > 0:
            return part_path, pd.read_parquet(part_path)
        screen.download_tile(images[continent], continent, int(west), int(south), tile_path)
        sampled = screen.sample_tile(tile_path, candidates.reset_index(drop=True))
        part_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = part_path.with_suffix(part_path.suffix + ".tmp")
        sampled.to_parquet(temporary, index=False)
        temporary.replace(part_path)
        return part_path, sampled

    outputs = []
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(one_tile, item) for item in grouped]
        for completed, future in enumerate(as_completed(futures), start=1):
            path, sampled = future.result()
            outputs.append(sampled)
            if completed == 1 or completed % 20 == 0 or completed == len(futures):
                print(f"sampled tiles: {completed}/{len(futures)} ({path.name})", flush=True)
    if not outputs:
        raise RuntimeError("No tile outputs were produced")
    result = pd.concat(outputs, ignore_index=True)
    if result["grid_id"].nunique() != int(targets["grid_id"].nunique() if not limit_tiles else result["grid_id"].nunique()):
        raise RuntimeError("Tile extraction returned duplicate or missing grid IDs")
    return result


def _compute_feature_batch(image, candidates: pd.DataFrame, part_path: Path) -> pd.DataFrame:
    """Use the Cloud computeFeatures endpoint, which is much faster than getInfo()."""
    import ee

    if part_path.exists() and part_path.stat().st_size > 0:
        cached = pd.read_parquet(part_path)
        cached_ids = set(pd.to_numeric(cached.get("grid_id"), errors="coerce").dropna().astype(int))
        requested_ids = set(candidates["grid_id"].astype(int))
        # Older smoke runs used the same tile/batch names with smaller chunks.
        # Reuse a checkpoint only when it is exactly the requested batch.
        if cached_ids == requested_ids and len(cached_ids) == len(candidates):
            return cached
    features = [
        ee.Feature(ee.Geometry.Point([float(row.lon), float(row.lat)]), {"grid_id": int(row.grid_id)})
        for row in candidates.itertuples()
    ]
    expression = image.sampleRegions(
        collection=ee.FeatureCollection(features),
        properties=["grid_id"],
        geometries=False,
        tileScale=16,
    )
    last_error = None
    for attempt in range(1, 4):
        try:
            result = ee.data.computeFeatures({
                "expression": expression,
                "fileFormat": "PANDAS_DATAFRAME",
                "pageSize": max(100, len(candidates) + 25),
            })
            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)
            if "grid_id" not in result.columns:
                result = pd.DataFrame({"grid_id": candidates["grid_id"].astype(int)})
            result["grid_id"] = pd.to_numeric(result["grid_id"], errors="coerce").astype("Int64")
            result = result.dropna(subset=["grid_id"]).copy()
            result["grid_id"] = result["grid_id"].astype(int)
            missing = sorted(set(candidates["grid_id"].astype(int)) - set(result["grid_id"]))
            if missing:
                result = pd.concat([result, pd.DataFrame({"grid_id": missing})], ignore_index=True)
            if result["grid_id"].nunique() != len(candidates):
                raise RuntimeError(f"Expected {len(candidates)} rows, got {result['grid_id'].nunique()}")
            result = result.merge(candidates.loc[:, ["grid_id", "event_year"]], on="grid_id", how="left", validate="one_to_one")
            part_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = part_path.with_suffix(part_path.suffix + ".tmp")
            result.to_parquet(temporary, index=False)
            temporary.replace(part_path)
            return result
        except Exception as exc:  # noqa: BLE001 - retry transient EE/API failures
            last_error = exc
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed computeFeatures batch {part_path.name}: {last_error}")


def _build_fast_history_image(screen, continent: str):
    """Build the same history stack with reduction performed per year."""
    import ee

    annual = ee.Image(screen.ANNUAL_ASSETS[continent])
    projection = annual.select("Dec1990").projection()
    degradation_year = ee.Image(screen.DEG_YEAR_ASSETS[continent])
    deforestation_year = ee.ImageCollection.fromImages(
        [ee.Image(asset) for asset in screen.DEF_YEAR_ASSETS[continent]]
    ).mosaic()
    images = []
    names = []

    def valid_class(image):
        return image.gte(1).And(image.lte(6))

    for year in range(1990, 2001):
        if year == 1990:
            previous = annual.select("Dec1990")
            valid = valid_class(previous)
            event = degradation_year.eq(1990).Or(deforestation_year.eq(1990))
        else:
            previous = annual.select(f"Dec{year - 1}")
            current = annual.select(f"Dec{year}")
            valid = valid_class(previous).And(valid_class(current))
            degradation = previous.eq(1).And(current.eq(2))
            deforestation = previous.eq(1).Or(previous.eq(2)).And(current.eq(3).Or(current.eq(5)))
            event = degradation.Or(deforestation)
        pair = (
            ee.Image.cat([event.And(valid).unmask(0), valid.unmask(0)])
            .rename([f"historical_event_{year}", f"historical_valid_{year}"])
            .setDefaultProjection(projection)
            .reduceResolution(ee.Reducer.mean(), maxPixels=65535)
            .reproject(crs="EPSG:4326", crsTransform=screen.TRANSFORM)
        )
        images.append(pair)
        names.extend([f"historical_event_{year}", f"historical_valid_{year}"])
    return ee.Image.cat(images).rename(names).toFloat()


def download_and_sample_compute(
    screen,
    targets: pd.DataFrame,
    run_dir: Path,
    project: str,
    workers: int,
    batch_size: int,
    part_subdir: str = "compute_parts",
) -> pd.DataFrame:
    """Batch point samples through ee.data.computeFeatures for fast pagination."""
    import ee

    if targets.empty:
        raise RuntimeError("No computeFeatures targets were supplied")
    if batch_size < 1 or workers < 1:
        raise ValueError("batch_size and workers must be positive")
    ee.Initialize(project=project)
    ee.data.setDeadline(120_000)
    ee.data.setMaxRetries(0)
    part_dir = run_dir / part_subdir
    tasks = []
    for (continent, lon_bin, lat_bin), tile_targets in targets.groupby(["continent", "lon_bin", "lat_bin"], sort=True):
        tile_targets = tile_targets.sort_values(["lon", "lat", "grid_id"], kind="mergesort").reset_index(drop=True)
        tile_key = f"{continent}_lon{int(lon_bin):+04d}_lat{int(lat_bin):+04d}"
        for batch_no, start in enumerate(range(0, len(tile_targets), batch_size), start=1):
            chunk = tile_targets.iloc[start : start + batch_size].copy()
            tasks.append((continent, tile_key, batch_no, chunk, part_dir / f"{tile_key}_batch{batch_no:03d}.parquet"))
    images = {continent: _build_fast_history_image(screen, continent) for continent in sorted(targets["continent"].unique())}

    def run_task(task):
        continent, tile_key, batch_no, chunk, path = task
        return path, _compute_feature_batch(images[continent], chunk, path)

    outputs = []
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            path, frame = future.result()
            outputs.append(frame)
            if completed == 1 or completed % 10 == 0 or completed == len(futures):
                print(f"sampled computeFeatures batches: {completed}/{len(futures)} ({path.name})", flush=True)
    result = pd.concat(outputs, ignore_index=True)
    expected = int(targets["grid_id"].nunique())
    if result["grid_id"].nunique() != expected:
        raise RuntimeError(f"computeFeatures extraction returned {result['grid_id'].nunique()} of {expected} grids")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.environ.get("TFDT_EE_PROJECT", ""))
    parser.add_argument("--universe", type=Path, default=RUN_ROOT / "expanded_control_grid_universe.parquet")
    parser.add_argument("--history-cache", type=Path, default=SCREENING_DIR / "historical_annual_event_shares.parquet")
    parser.add_argument("--old-parts", type=Path, default=OLD_EXTRACTION)
    parser.add_argument("--run-dir", type=Path, default=RUN_ROOT / "control_history_extraction")
    parser.add_argument("--part-subdir", default="compute_parts", help="Subdirectory under --run-dir for compute checkpoints")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", choices=("compute", "tile", "point"), default="compute")
    parser.add_argument("--limit-tiles", type=int, default=0, help="Only process the first N occupied tiles, for a smoke test")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N missing grids, for a smoke test")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1 or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid batch/shard settings")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    new_parts = args.run_dir
    targets, diagnostics = prepare_targets(args.universe, args.history_cache, args.old_parts, new_parts)
    if args.limit:
        if args.limit < 1:
            raise ValueError("--limit must be positive when supplied")
        targets = targets.head(args.limit).copy()
    if args.mode == "compute":
        tile_keys = targets.loc[:, ["continent", "lon_bin", "lat_bin"]].drop_duplicates().sort_values(["continent", "lon_bin", "lat_bin"], kind="mergesort").reset_index(drop=True)
        assigned_keys = tile_keys.iloc[[i for i in range(len(tile_keys)) if i % args.shard_count == args.shard_index]]
        assigned = targets.merge(assigned_keys, on=["continent", "lon_bin", "lat_bin"], how="inner", validate="many_to_one")
    else:
        assigned = targets.iloc[[i for i in range(len(targets)) if i % args.shard_count == args.shard_index]].copy()
    assigned_batches = int((len(assigned) + args.batch_size - 1) // args.batch_size)
    if args.mode == "compute" and not assigned.empty:
        assigned_batches = int(sum((len(group) + args.batch_size - 1) // args.batch_size for _, group in assigned.groupby(["continent", "lon_bin", "lat_bin"])))
    plan = {
        "run_id": "expanded_control_pool_full_history_1990",
        "project": args.project,
        "batch_size": int(args.batch_size),
        "workers": int(args.workers),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "limit": int(args.limit),
        "target_diagnostics": diagnostics,
        "assigned_grid_ids": int(len(assigned)),
        "assigned_batches": assigned_batches,
        "mode": args.mode,
        "occupied_tiles": int(targets.loc[:, ["continent", "lon_bin", "lat_bin"]].drop_duplicates().shape[0]),
        "limit_tiles": int(args.limit_tiles),
    }
    (args.run_dir / f"extraction_plan_shard{args.shard_index:03d}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run or assigned.empty:
        return
    screen = _load_screening_module()
    if args.mode == "compute":
        result = download_and_sample_compute(screen, assigned, args.run_dir, args.project, args.workers, args.batch_size, args.part_subdir)
    elif args.mode == "tile":
        result = download_and_sample_tiles(screen, assigned, args.run_dir, args.project, args.workers, args.limit_tiles)
    else:
        result = screen.download_and_sample(
            assigned,
            args.run_dir,
            args.project,
            batch_size=args.batch_size,
            workers=args.workers,
            shard_index=0,
            shard_count=1,
        )
    print(json.dumps({"sampled_rows": int(len(result)), "sampled_unique_grids": int(result["grid_id"].nunique())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
