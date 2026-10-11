#!/usr/bin/env python3
"""Estimate month-threshold and year-specific standardization intervals using 2-degree spatial blocks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/primary"
OUT = ROOT / "results/sensitivity"
MONTH = RUN / "outcomes/monthly_matched_effects.parquet"
SUPPORT = RUN / "event_level/common_support_weights.parquet"
PAIRS = RUN / "event_level/fixed_matched_pairs.parquet"
WINDOW = RUN / "event_level/fixed_matched_event_window.parquet"
PIPELINE = ROOT / "src"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))
from pipeline.statistics import common_block_bootstrap, contrast_on_column

SEED = 20260725
REPS = 1000
YEARS = [0, 3, 5]
THRESHOLDS = [6, 8, 9, 10, 12]


def ess(weights: pd.Series) -> float:
    w = pd.to_numeric(weights, errors="coerce").to_numpy(float)
    w = w[np.isfinite(w) & (w > 0)]
    return float(w.sum() ** 2 / np.square(w).sum()) if len(w) else np.nan


def weighted_smd(frame: pd.DataFrame, column: str, weight: str = "standardized_event_weight") -> float:
    moments = {}
    for event_type, group in frame.groupby("event_type"):
        x = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
        w = pd.to_numeric(group[weight], errors="coerce").to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not ok.any():
            return np.nan
        x, w = x[ok], w[ok]
        mu = np.sum(w * x) / np.sum(w)
        var = np.sum(w * (x - mu) ** 2) / np.sum(w)
        moments[event_type] = (mu, var)
    if not {"degradation", "deforestation"}.issubset(moments):
        return np.nan
    den = np.sqrt((moments["degradation"][1] + moments["deforestation"][1]) / 2)
    return float((moments["deforestation"][0] - moments["degradation"][0]) / den) if den else np.nan


def load_threshold_events() -> pd.DataFrame:
    monthly = pd.read_parquet(MONTH)
    monthly = monthly[np.isfinite(monthly["matched_effect"])].copy()
    key = ["treatment_grid_id", "event_year", "event_type", "macroregion",
           "event_area_m2", "treatment_lon", "treatment_lat", "relative_year"]
    annual = monthly.groupby(key, as_index=False).agg(
        usable_months=("calendar_month", "nunique"),
        annual_matched_effect=("matched_effect", "mean"),
    )
    support = pd.read_parquet(SUPPORT)[[
        "treatment_grid_id", "event_year", "event_type", "standardized_event_weight",
        "disturbance_share", "event_composition_purity", "baseline_undist_share",
    ]]
    censor = pd.read_parquet(PAIRS, columns=["treatment_grid_id", "event_year", "censor_relative_year"])
    censor = censor.drop_duplicates(["treatment_grid_id", "event_year"])
    out = annual.merge(support, on=["treatment_grid_id", "event_year", "event_type"],
                       how="inner", validate="many_to_one")
    out = out.merge(censor, on=["treatment_grid_id", "event_year"], how="left",
                    validate="many_to_one")
    out = out[np.isfinite(out["standardized_event_weight"]) & (out["standardized_event_weight"] > 0)]
    out = out[out["censor_relative_year"].isna() | (out["relative_year"] < out["censor_relative_year"])].copy()
    return out


def point_and_ci(frame: pd.DataFrame, value: str, weight: str = "standardized_event_weight") -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = common_block_bootstrap(
        frame, value, ("event_type", "relative_year"), weight_column=weight,
        weighting="event_area", block_degrees=2.0, replicates=REPS,
        seed=SEED, minimum_valid_replicates=950,
    )
    contrast = contrast_on_column(
        bundle, "event_type", "deforestation", "degradation",
        by_columns=("relative_year",), label="deforestation_minus_degradation",
    )
    return bundle.summary, contrast


def run_thresholds(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        frame = events[events["usable_months"] >= threshold].copy()
        groups, contrasts = point_and_ci(frame, "annual_matched_effect")
        for _, row in contrasts[contrasts.relative_year.isin(YEARS)].iterrows():
            r = int(row.relative_year)
            sub = frame[frame.relative_year == r]
            q = groups.query("relative_year == @r")
            rows.append({
                "analysis": "frozen_standardization",
                "month_threshold": threshold,
                "relative_year": r,
                "degradation_n": int((sub.event_type == "degradation").sum()),
                "deforestation_n": int((sub.event_type == "deforestation").sum()),
                "degradation_ess": ess(sub.loc[sub.event_type == "degradation", "standardized_event_weight"]),
                "deforestation_ess": ess(sub.loc[sub.event_type == "deforestation", "standardized_event_weight"]),
                "degradation_estimate": float(q.query("event_type == 'degradation'").estimate.iloc[0]),
                "deforestation_estimate": float(q.query("event_type == 'deforestation'").estimate.iloc[0]),
                "direct_contrast": float(row.estimate),
                "ci_low": float(row.ci_low),
                "ci_high": float(row.ci_high),
                "bootstrap_sd": float(row.bootstrap_sd),
                "bootstrap_valid": int(row.bootstrap_valid),
                "bootstrap_seed": SEED,
                "block_degrees": 2.0,
                "max_weighted_smd": float(max(abs(weighted_smd(sub, c)) for c in ["disturbance_share", "event_composition_purity", "baseline_undist_share", "event_year"])),
                "event_year_smd": float(weighted_smd(sub, "event_year")),
            })
    return pd.DataFrame(rows)


def run_year_specific() -> pd.DataFrame:
    table = pd.read_parquet(OUT / "year_specific_event_table.parquet")
    rows = []
    for r in YEARS:
        frame = table[table.relative_year == r].copy()
        frame["relative_year"] = r
        groups, contrasts = point_and_ci(frame, "annual_matched_effect", weight="year_weight")
        contrast = contrasts.iloc[0]
        for event_type in ["degradation", "deforestation"]:
            g = groups[groups.event_type == event_type].iloc[0]
            sub = frame[frame.event_type == event_type]
            rows.append({
                "analysis": "year_specific_standardization",
                "month_threshold": np.nan,
                "relative_year": r,
                "event_type": event_type,
                "n": int(len(sub)),
                "ess": ess(sub["year_weight"]),
                "estimate": float(g.estimate),
                "group_ci_low": float(g.ci_low),
                "group_ci_high": float(g.ci_high),
                "direct_contrast": float(contrast.estimate),
                "contrast_ci_low": float(contrast.ci_low),
                "contrast_ci_high": float(contrast.ci_high),
                "bootstrap_sd": float(contrast.bootstrap_sd),
                "bootstrap_valid": int(contrast.bootstrap_valid),
                "bootstrap_seed": SEED,
                "block_degrees": 2.0,
                "max_weighted_smd": float(max(abs(weighted_smd(frame, c, "year_weight")) for c in ["disturbance_share", "event_composition_purity", "baseline_undist_share", "event_year"])),
                "event_year_smd": float(weighted_smd(frame, "event_year", "year_weight")),
            })
    return pd.DataFrame(rows)


def verify_primary() -> dict[str, float]:
    window = pd.read_parquet(WINDOW)
    support = pd.read_parquet(SUPPORT)[["treatment_grid_id", "event_year", "standardized_event_weight"]]
    frame = window.merge(support, on=["treatment_grid_id", "event_year"], how="left", validate="many_to_one")
    _, contrast = point_and_ci(frame, "pair_effect_evi_mean")
    selected = contrast[contrast.relative_year.isin(YEARS)].set_index("relative_year")
    expected = pd.read_csv(RUN / "event_level/standardized_event_type_bootstrap.csv")
    expected = expected[(expected.event_type == "deforestation_minus_degradation") & expected.relative_year.isin(YEARS)].set_index("relative_year")
    diffs = {f"t{r}_estimate_abs_diff": abs(float(selected.loc[r, "estimate"]) - float(expected.loc[r, "estimate"])) for r in YEARS}
    diffs.update({f"t{r}_ci_abs_diff": max(abs(float(selected.loc[r, "ci_low"]) - float(expected.loc[r, "ci_low"])), abs(float(selected.loc[r, "ci_high"]) - float(expected.loc[r, "ci_high"]))) for r in YEARS})
    if max(diffs.values()) > 1e-12:
        raise RuntimeError(f"Primary bootstrap regression failed: {diffs}")
    return diffs


def main() -> None:
    primary = verify_primary()
    events = load_threshold_events()
    threshold = run_thresholds(events)
    threshold.to_csv(OUT / "month_threshold_sensitivity.csv", index=False)
    year = run_year_specific()
    year.to_csv(OUT / "year_specific_standardization_bootstrap.csv", index=False)
    print(threshold.to_string(index=False))
    print(year.to_string(index=False))


if __name__ == "__main__":
    main()
