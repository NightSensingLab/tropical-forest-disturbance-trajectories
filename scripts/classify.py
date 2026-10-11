#!/usr/bin/env python3
"""Evaluate event-type discrimination using complete t0-t5 trajectories."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "src"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
from pipeline.trajectory_classification import (
    _auc,
    _balanced_accuracy,
    _fit_predict,
    _region_folds,
    _spatial_folds,
    _stratified_folds,
)

RUN_ROOT = PROJECT_ROOT / "data/primary"
OUT = RUN_ROOT / "event_level" / "diagnostics"
SEED = 20260725


def block(lon, lat):
    return f"{int(math.floor((float(lon) + 180.0) / 2.0))}:{int(math.floor((float(lat) + 90.0) / 2.0))}"


def cohort() -> pd.DataFrame:
    frame = pd.read_parquet(RUN_ROOT / "outcomes/annual_matched_effects.parquet")
    frame = frame.rename(columns={"treatment_lon": "lon", "treatment_lat": "lat", "annual_matched_effect": "pair_effect_evi_mean"})
    censor = pd.read_parquet(RUN_ROOT / "subannual/matched_links.parquet", columns=["treatment_grid_id", "event_year", "censor_relative_year"]).drop_duplicates(["treatment_grid_id", "event_year"])
    frame = frame.merge(censor, on=["treatment_grid_id", "event_year"], how="left", validate="many_to_one")
    frame = frame.loc[frame["censor_relative_year"].isna() | (frame["relative_year"] < frame["censor_relative_year"])].copy()
    index = ["treatment_grid_id", "event_year", "event_type", "macroregion", "lon", "lat"]
    wide = frame.pivot(index=index, columns="relative_year", values="pair_effect_evi_mean").reset_index()
    wide.columns = [f"t{int(c)}" if isinstance(c, (int, np.integer)) else str(c) for c in wide.columns]
    cols = ["t0", "t1", "t2", "t3", "t4", "t5"]
    out = wide.dropna(subset=cols).copy()
    out["event_id"] = out["treatment_grid_id"].astype(int).astype(str) + ":" + out["event_year"].astype(int).astype(str)
    out["block_2deg"] = [block(x, y) for x, y in zip(out["lon"], out["lat"])]
    return out.reset_index(drop=True)


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = (frame["event_type"].astype(str).to_numpy() == "deforestation").astype(np.int8)
    folds = {
        "random_stratified_5fold": _stratified_folds(y, 5, SEED),
        "spatial_block_2deg_5fold": _spatial_folds(frame, y, 5, SEED),
        "region_leave_one_out": _region_folds(frame),
    }
    feature_sets = {"t0_only": ["t0"], "t0_to_t5_trajectory": ["t0", "t1", "t2", "t3", "t4", "t5"]}
    summary, fold_rows = [], []
    for validation, test_folds in folds.items():
        for feature_set, cols in feature_sets.items():
            x = frame.loc[:, cols].to_numpy(float)
            oof = np.full(len(frame), np.nan)
            aucs, bas = [], []
            for fold_id, test in enumerate(test_folds):
                test = np.asarray(test, dtype=int)
                train = np.setdiff1d(np.arange(len(frame)), test)
                scores, threshold = _fit_predict(x, y, train, test)
                valid = np.isfinite(scores) & np.isfinite(threshold)
                auc = _auc(y[test][valid], scores[valid]) if valid.any() else np.nan
                ba = _balanced_accuracy(y[test][valid], scores[valid] >= threshold) if valid.any() else np.nan
                oof[test] = scores
                aucs.append(auc); bas.append(ba)
                fold_rows.append({"validation": validation, "feature_set": feature_set, "fold_id": fold_id, "test_n": len(test), "deforestation_n": int(y[test].sum()), "degradation_n": int(len(test)-y[test].sum()), "auc": auc, "balanced_accuracy": ba})
            valid = np.isfinite(oof)
            summary.append({"validation": validation, "feature_set": feature_set, "sample_n": len(frame), "deforestation_n": int(y.sum()), "degradation_n": int(len(y)-y.sum()), "folds_used": len(aucs), "auc": _auc(y[valid], oof[valid]), "auc_sd": float(np.nanstd(aucs, ddof=1)), "balanced_accuracy": float(np.nanmean(bas)), "balanced_accuracy_sd": float(np.nanstd(bas, ddof=1)), "seed": SEED, "model": "ridge linear discriminant; equal-event weighting"})
    return pd.DataFrame(summary), pd.DataFrame(fold_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = cohort()
    summary, folds = evaluate(c)
    summary.to_csv(OUT / "classification_summary.csv", index=False)
    folds.to_csv(OUT / "classification_fold_results.csv", index=False)
    c.to_csv(OUT / "classification_cohort.csv", index=False)
    summary_info = {"sample_n": int(len(c)), "event_type_counts": c["event_type"].value_counts().astype(int).to_dict(), "seed": SEED}
    print(json.dumps(summary_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
