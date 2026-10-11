#!/usr/bin/env python3
"""Year-specific primary cohort composition, balance and effective sample size."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = ROOT / "results/composition"
CURRENT = ROOT / "data/primary/event_level"
EVENT_KEYS = ["treatment_grid_id", "event_year"]
EVENT_TYPES = ("degradation", "deforestation")
REGIONS = ("Amazon_proxy", "Congo_proxy", "Southeast_Asia_proxy", "Other")
COMPOSITION_VARIABLES = ("disturbance_share", "event_composition_purity", "baseline_undist_share")

def weighted_moments(values: pd.Series, weights: pd.Series) -> tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    x = x[valid]
    w = w[valid]
    if not len(x):
        return np.nan, np.nan, 0.0
    total = w.sum()
    mean = np.dot(x, w) / total
    variance = np.dot(np.square(x - mean), w) / total
    ess = total**2 / np.dot(w, w)
    return float(mean), float(np.sqrt(variance)), float(ess)

def weighted_smd(
    deforestation: pd.DataFrame,
    degradation: pd.DataFrame,
    variable: str,
) -> float:
    d_mean, d_sd, _ = weighted_moments(
        deforestation[variable], deforestation["standardized_event_weight"]
    )
    g_mean, g_sd, _ = weighted_moments(
        degradation[variable], degradation["standardized_event_weight"]
    )
    pooled_sd = np.sqrt((d_sd**2 + g_sd**2) / 2.0)
    return float((d_mean - g_mean) / pooled_sd) if pooled_sd > 0.0 else np.nan

def stage09_composition() -> pd.DataFrame:
    """Describe the actual Stage 09 year-specific primary estimation cohorts."""
    window = pd.read_parquet(CURRENT / "fixed_matched_event_window.parquet")
    support = pd.read_parquet(CURRENT / "common_support_weights.parquet")
    contrast = pd.read_csv(CURRENT / "standardized_event_type_contrast.csv")

    if window.duplicated(EVENT_KEYS + ["relative_year"]).any():
        raise ValueError("Stage 09 event window contains duplicate event-year-relative-year rows")
    if window["pair_effect_evi_mean"].isna().any():
        raise ValueError("Stage 09 event window should contain only valid matched effects")

    support_columns = EVENT_KEYS + [
        "standardized_event_weight",
        "standardization_stratum",
    ]
    working = window.merge(
        support[support_columns], on=EVENT_KEYS, how="inner", validate="many_to_one"
    )
    if len(working) != len(window):
        raise ValueError("Some Stage 09 effect rows lack frozen common-support weights")

    rows: list[dict[str, object]] = []
    for relative_year in sorted(working["relative_year"].unique()):
        year_groups: dict[str, pd.DataFrame] = {}
        year_rows: list[dict[str, object]] = []
        for event_type in EVENT_TYPES:
            group = working.loc[
                working["relative_year"].eq(relative_year)
                & working["event_type"].eq(event_type)
            ].copy()
            if group.empty:
                raise ValueError(f"No {event_type} rows at relative year {relative_year}")
            year_groups[event_type] = group
            weights = group["standardized_event_weight"]
            _, _, ess = weighted_moments(
                group["disturbance_share"], weights
            )
            row: dict[str, object] = {
                "outcome_family": "annualized monthly matched effects; usable_months>=9",
                "relative_year": int(relative_year),
                "event_type": event_type,
                "n": int(len(group)),
                "ess": ess,
                "weight_sum": float(weights.sum()),
                "common_support_strata_present": int(
                    group["standardization_stratum"].nunique()
                ),
            }
            for variable in COMPOSITION_VARIABLES:
                mean, sd, _ = weighted_moments(group[variable], weights)
                row[f"{variable}_weighted_mean"] = mean
                row[f"{variable}_weighted_sd_population"] = sd
            for region in REGIONS:
                row[f"region_{region}_fraction"] = float(
                    group.loc[group["macroregion"].eq(region), "standardized_event_weight"].sum()
                    / weights.sum()
                )
            year_rows.append(row)

        smds = {
            variable: weighted_smd(
                year_groups["deforestation"], year_groups["degradation"], variable
            )
            for variable in COMPOSITION_VARIABLES
        }
        for row in year_rows:
            for variable, value in smds.items():
                row[f"smd_deforestation_minus_degradation_{variable}"] = value
            row["max_abs_smd"] = max(abs(value) for value in smds.values())
            rows.append(row)

    result = pd.DataFrame(rows)
    expected = contrast.loc[
        contrast["event_type"].isin(EVENT_TYPES),
        ["event_type", "relative_year", "treatment_count", "effective_sample_size"],
    ].copy()
    checked = result.merge(
        expected, on=["event_type", "relative_year"], how="left", validate="one_to_one"
    )
    if not np.array_equal(checked["n"].to_numpy(), checked["treatment_count"].to_numpy()):
        raise AssertionError("Derived Stage 09 counts do not match primary contrast output")
    if not np.allclose(
        checked["ess"], checked["effective_sample_size"], rtol=0.0, atol=1e-9
    ):
        raise AssertionError("Derived Stage 09 ESS does not match primary contrast output")
    return result


if __name__ == "__main__":
    stage09_composition().to_csv(OUTPUT / "risk_set_composition.csv", index=False)
