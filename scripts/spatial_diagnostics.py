#!/usr/bin/env python3
"""Regional negative-block fractions and matching-distance sensitivity."""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = ROOT / "results/spatial"
RUN = ROOT / "data/primary"
ANALYSIS = RUN / "event_level"
STAGE08 = RUN / "subannual"
STAGE09 = RUN / "outcomes"
PIPELINE_ROOT = ROOT / "src"
sys.path.insert(0, str(PIPELINE_ROOT))
from pipeline.matched_estimation import common_support_weights

def current_dynamic_outcomes() -> pd.DataFrame:
    effects = pd.read_parquet(STAGE09 / "annual_matched_effects.parquet")
    censor = pd.read_parquet(
        STAGE08 / "matched_links.parquet",
        columns=["treatment_grid_id", "event_year", "censor_relative_year"],
    ).drop_duplicates(["treatment_grid_id", "event_year"])
    frame = effects.merge(censor, on=["treatment_grid_id", "event_year"], how="left", validate="many_to_one")
    frame = frame.loc[
        frame["censor_relative_year"].isna()
        | (frame["relative_year"] < frame["censor_relative_year"])
    ].copy()
    return frame.rename(columns={"annual_matched_effect": "pair_effect_evi_mean"})

def negative_fraction_bootstrap() -> pd.DataFrame:
    frame = pd.read_parquet(ANALYSIS / "fixed_matched_event_window.parquet")
    support = pd.read_parquet(ANALYSIS / "common_support_weights.parquet")
    keys = ["treatment_grid_id", "event_year"]
    frame = frame.merge(
        support.loc[:, keys + ["standardized_event_weight"]],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    frame["spatial_block"] = [
        f"{int(np.floor((x + 180.0) / 2.0))}:{int(np.floor((y + 90.0) / 2.0))}"
        for x, y in zip(frame["treatment_lon"], frame["treatment_lat"])
    ]
    grouped = (
        frame.groupby(["scope", "relative_year", "spatial_block", "event_type"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    "estimate": np.average(
                        group["pair_effect_evi_mean"], weights=group["standardized_event_weight"]
                    ),
                    "area": group["standardized_event_weight"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    pivot = grouped.pivot_table(
        index=["scope", "relative_year", "spatial_block"],
        columns="event_type",
        values=["estimate", "area"],
        aggfunc="first",
    ).dropna().reset_index()
    pivot.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in pivot.columns]
    pivot["direct"] = pivot["estimate_deforestation"] - pivot["estimate_degradation"]
    rng = np.random.default_rng(20260725 + 11)
    rows = []
    for (scope, year), group in pivot.groupby(["scope", "relative_year"], sort=True):
        if int(year) not in (3, 4, 5):
            continue
        direct = group["direct"].to_numpy(float)
        # Supported blocks are weighted by the summed event area of both types.
        weight = (group["area_deforestation"] + group["area_degradation"]).to_numpy(float)
        draws = rng.multinomial(len(group), np.full(len(group), 1.0 / len(group)), size=1000)
        denominator = draws @ weight
        negative = np.divide(
            draws @ (weight * (direct < 0.0)),
            denominator,
            out=np.full(1000, np.nan),
            where=denominator > 0,
        )
        finite = negative[np.isfinite(negative)]
        point = float(np.average(direct < 0.0, weights=weight))
        rows.append(
            {
                "macroregion": scope,
                "relative_year": int(year),
                "spatial_block_count": int(len(group)),
                "weighted_negative_fraction": point,
                "ci_low": float(np.quantile(finite, 0.025)),
                "ci_high": float(np.quantile(finite, 0.975)),
                "valid_bootstrap": int(len(finite)),
                "weight_definition": "sum of standardized deforestation and degradation event-area weights per supported block",
                "bootstrap_definition": "within-region 2-degree block multinomial bootstrap; 1000 replicates; seed 20260736",
            }
        )
    return pd.DataFrame(rows).sort_values(["macroregion", "relative_year"])

def event_id(grid_id, event_year) -> str:
    return f"{int(grid_id)}:{int(event_year)}"

def extreme_distance_current_outcomes() -> pd.DataFrame:
    treatments = pd.read_parquet(ANALYSIS / "matched_treatments.parquet").rename(
        columns={"grid_id": "treatment_grid_id"}
    )
    distance = pd.read_csv(ANALYSIS / "diagnostics/matching_distance_event_summary.csv")
    distance_index = distance.set_index("event_id")
    ids = treatments.apply(lambda row: event_id(row["treatment_grid_id"], row["event_year"]), axis=1)
    max_distance = ids.map(distance_index["max_distance"])
    max_distance.index = treatments.index
    n = len(treatments)
    scenarios: list[tuple[str, pd.Series, str]] = [
        ("S0", pd.Series(False, index=treatments.index), "all fixed-matched events")
    ]
    for label, fraction in (("S1", 0.01), ("S2", 0.005)):
        k = max(1, int(math.ceil(n * fraction)))
        excluded = max_distance.sort_values(ascending=False, kind="mergesort").index[:k]
        mask = pd.Series(False, index=treatments.index)
        mask.loc[excluded] = True
        scenarios.append((label, mask, f"exclude top {fraction:.1%} event-level max distances (n={k})"))
    gt10 = set(distance.loc[distance["max_distance"] > 10.0, "event_id"])
    scenarios.append(("S3", ids.isin(gt10), f"exclude events containing a link >10 (n={ids.isin(gt10).sum()})"))

    outcomes = current_dynamic_outcomes()
    rows = []
    for label, exclude, description in scenarios:
        retained = treatments.loc[~exclude].copy()
        support, diagnostics = common_support_weights(retained)
        work = outcomes.merge(
            support.loc[:, ["treatment_grid_id", "event_year", "standardized_event_weight"]],
            on=["treatment_grid_id", "event_year"],
            how="inner",
            validate="many_to_one",
        )
        for year in (0, 5):
            estimates = {}
            valid_counts = {}
            for event_type in ("deforestation", "degradation"):
                group = work.loc[(work["relative_year"] == year) & (work["event_type"] == event_type)]
                value = pd.to_numeric(group["pair_effect_evi_mean"], errors="coerce")
                weight = pd.to_numeric(group["standardized_event_weight"], errors="coerce")
                valid = value.notna() & weight.notna() & (weight > 0)
                estimates[event_type] = float(np.average(value[valid], weights=weight[valid]))
                valid_counts[event_type] = int(valid.sum())
            rows.append(
                {
                    "scenario": label,
                    "description": description,
                    "relative_year": year,
                    "deforestation": estimates["deforestation"],
                    "degradation": estimates["degradation"],
                    "direct_difference": estimates["deforestation"] - estimates["degradation"],
                    "retained_event_n": int(len(retained)),
                    "retained_event_n_deforestation": int((retained["event_type"] == "deforestation").sum()),
                    "retained_event_n_degradation": int((retained["event_type"] == "degradation").sum()),
                    "common_support_strata": int(diagnostics["common_support_strata"]),
                    "valid_n_deforestation": valid_counts["deforestation"],
                    "valid_n_degradation": valid_counts["degradation"],
                    "outcome_source": "annualized monthly matched effects; >=9 usable months; dynamic censoring",
                }
            )
    result = pd.DataFrame(rows)
    for year in (0, 5):
        base = result.loc[(result["scenario"] == "S0") & (result["relative_year"] == year), "direct_difference"].iloc[0]
        result.loc[result["relative_year"] == year, "delta_vs_S0"] = (
            result.loc[result["relative_year"] == year, "direct_difference"] - base
        )
    return result.sort_values(["scenario", "relative_year"])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    negative_fraction_bootstrap().to_csv(OUTPUT / "negative_fraction_ci.csv", index=False)
    extreme_distance_current_outcomes().to_csv(OUTPUT / "extreme_match_sensitivity.csv", index=False)

if __name__ == "__main__":
    main()
