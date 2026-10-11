#!/usr/bin/env python3
"""Build the full locally available Stage 07 control-grid universe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATIC_COLUMNS = ["grid_id", "grid_row", "grid_col", "lon", "lat", "cell_area_m2", "macroregion"]
FILTER_COLUMNS = ["coverage_eligible", "matching_covariates_complete"]
READ_COLUMNS = STATIC_COLUMNS + FILTER_COLUMNS


def _as_bool(value: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(value):
        return value.fillna(False)
    return value.astype(str).str.lower().isin({"true", "1", "yes"})


def build_universe(feature_root: Path, year_range: tuple[int, int]) -> tuple[pd.DataFrame, dict]:
    """Return one row per eligible grid and deterministic coverage diagnostics."""
    feature_root = Path(feature_root)
    if year_range[0] > year_range[1]:
        raise ValueError("year_range must be ascending")
    frames: list[pd.DataFrame] = []
    per_year: list[dict] = []
    for year in range(int(year_range[0]), int(year_range[1]) + 1):
        year_dir = feature_root / f"year{year}"
        paths = sorted(year_dir.glob("part_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No feature parts found for {year}: {year_dir}")
        year_frames: list[pd.DataFrame] = []
        for path in paths:
            frame = pd.read_parquet(path, columns=READ_COLUMNS)
            mask = _as_bool(frame["coverage_eligible"]) & _as_bool(frame["matching_covariates_complete"])
            selected = frame.loc[mask, STATIC_COLUMNS].copy()
            if selected.empty:
                continue
            selected["feature_year"] = int(year)
            year_frames.append(selected)
        if not year_frames:
            raise RuntimeError(f"No eligible feature rows for {year}")
        year_frame = pd.concat(year_frames, ignore_index=True)
        if year_frame["grid_id"].duplicated().any():
            raise ValueError(f"Duplicate grid_id within feature year {year}")
        year_frames = None
        per_year.append({"feature_year": int(year), "eligible_rows": int(len(year_frame)), "unique_grids": int(year_frame["grid_id"].nunique())})
        frames.append(year_frame)

    combined = pd.concat(frames, ignore_index=True)
    for col in ["grid_id", "grid_row", "grid_col"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    for col in ["lon", "lat", "cell_area_m2"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    if combined["grid_id"].isna().any() or not np.isfinite(combined[["lon", "lat", "cell_area_m2"]].to_numpy()).all():
        raise ValueError("Eligible universe contains missing or non-finite grid coordinates/area")
    conflict_counts = combined.groupby("grid_id", sort=True)[["grid_row", "grid_col", "lon", "lat", "cell_area_m2"]].nunique(dropna=False)
    conflicts = conflict_counts.gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"Static grid fields conflict for grid IDs: {conflicts.index[conflicts].tolist()[:10]}")
    universe = combined.sort_values(["grid_id", "feature_year"], kind="mergesort").drop_duplicates("grid_id", keep="first").copy()
    year_counts = combined.groupby("grid_id", sort=True)["feature_year"].agg(feature_year_count="nunique", feature_year_first="min", feature_year_last="max").reset_index()
    universe = universe.drop(columns=["feature_year"]).merge(year_counts, on="grid_id", how="left", validate="one_to_one")
    universe = universe.sort_values("grid_id", kind="mergesort").reset_index(drop=True)
    diagnostics = {
        "feature_year_start": int(year_range[0]),
        "feature_year_end": int(year_range[1]),
        "eligible_rows": int(len(combined)),
        "unique_grids": int(len(universe)),
        "per_year": per_year,
        "coordinate_conflict_grids": int(conflicts.sum()),
    }
    return universe, diagnostics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    feature_root = Path(config["feature_root"])
    universe, diagnostics = build_universe(feature_root, tuple(config["feature_years"]))
    if len(universe) < int(config["minimum_expected_control_grids"]):
        raise RuntimeError(f"Expanded universe unexpectedly small: {len(universe)}")
    universe.to_parquet(root / "expanded_control_grid_universe.parquet", index=False)
    universe.to_csv(root / "expanded_control_grid_universe.csv", index=False)
    (root / "expanded_control_universe_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": config["run_id"],
        "config": config,
        "diagnostics": diagnostics,
        "artifact": {"path": str(root / "expanded_control_grid_universe.parquet"), "sha256": _sha256(root / "expanded_control_grid_universe.parquet")},
    }
    (root / "expanded_control_universe_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
