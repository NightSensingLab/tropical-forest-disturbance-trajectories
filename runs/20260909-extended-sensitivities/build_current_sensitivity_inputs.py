#!/usr/bin/env python3
"""Build non-destructive inputs for current-run monthly, climate/fire and threshold analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
MAIN_RUN = PROJECT / "runs/20260904-expanded-control-pool"
OUTPUT = Path(__file__).resolve().parent
STAGE03 = PROJECT / "01_data/upstream/TropicalRainforest_cloud_analysis/03_combined_panel/parts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_current_unit_catalogue() -> pd.DataFrame:
    """Create the role-specific event-year catalogue required by Stage 10."""
    pairs = pd.read_parquet(MAIN_RUN / "fixed_matched_pairs.parquet")
    treatments = pd.read_parquet(MAIN_RUN / "matched_treatments.parquet")
    treatment_cols = [
        "grid_id", "event_year", "lon", "lat", "macroregion", "cell_area_m2",
    ]
    treatment = treatments.loc[:, treatment_cols].drop_duplicates(
        ["grid_id", "event_year"], keep="first"
    ).rename(columns={"grid_id": "grid_id"})
    treatment["role"] = "treatment"
    treatment["source_signature"] = "expanded_control_pool_full_history_1990"

    control = pairs.loc[:, [
        "control_grid_id", "event_year", "control_lon", "control_lat",
        "control_macroregion",
    ]].drop_duplicates(["control_grid_id", "event_year"], keep="first")
    control = control.rename(
        columns={
            "control_grid_id": "grid_id",
            "control_lon": "lon",
            "control_lat": "lat",
            "control_macroregion": "macroregion",
        }
    )
    universe = pd.read_parquet(
        MAIN_RUN / "expanded_control_grid_universe.parquet",
        columns=["grid_id", "cell_area_m2"],
    )
    control = control.merge(universe, on="grid_id", how="left", validate="many_to_one")
    control["role"] = "control"
    control["source_signature"] = "expanded_control_pool_full_history_1990"

    catalogue = pd.concat([treatment, control], ignore_index=True)
    catalogue = catalogue.loc[
        :, ["grid_id", "event_year", "role", "lon", "lat", "macroregion", "source_signature", "cell_area_m2"]
    ].sort_values(["event_year", "role", "grid_id"], kind="mergesort").reset_index(drop=True)
    if catalogue.duplicated(["grid_id", "event_year", "role"]).any():
        raise ValueError("Current event-role catalogue contains duplicate keys")
    if catalogue["cell_area_m2"].isna().any():
        raise ValueError("Current event-role catalogue has missing control cell areas")
    return catalogue


def build_threshold_candidates() -> pd.DataFrame:
    """Identify full-history first events for 5%, 10% and 20% in the candidate panel."""
    annual = pd.read_parquet(MAIN_RUN / "history_qualified_annual.parquet")
    annual = annual.sort_values(["grid_id", "calendar_year"], kind="mergesort")
    records = []
    for threshold in (0.05, 0.10, 0.20):
        hit = annual["coverage_available"].astype(bool) & annual["disturbance_share"].ge(threshold)
        first_year = annual.loc[hit].groupby("grid_id", sort=False)["calendar_year"].min()
        event = annual.loc[
            annual["grid_id"].map(first_year).eq(annual["calendar_year"])
            & annual["calendar_year"].between(2004, 2018)
        ].copy()
        event["threshold"] = threshold
        event["event_type"] = np.where(
            event["deforestation_share"].ge(event["degradation_share"]),
            "deforestation",
            "degradation",
        )
        event["first_threshold_year"] = event["calendar_year"]
        records.append(event.loc[:, [
            "grid_id", "calendar_year", "threshold", "first_threshold_year",
            "event_type", "disturbance_share", "degradation_share", "deforestation_share",
        ]].rename(columns={"calendar_year": "event_year"}))
    return pd.concat(records, ignore_index=True).sort_values(
        ["threshold", "event_year", "grid_id"], kind="mergesort"
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    catalogue = build_current_unit_catalogue()
    candidates = build_threshold_candidates()
    catalogue.to_parquet(OUTPUT / "current_event_role_catalogue.parquet", index=False)
    catalogue.to_csv(OUTPUT / "current_event_role_catalogue.csv", index=False)
    candidates.to_parquet(OUTPUT / "full_history_threshold_candidates.parquet", index=False)
    candidates.to_csv(OUTPUT / "full_history_threshold_candidates.csv", index=False)

    threshold_summary = (
        candidates.groupby(["threshold", "event_type"], as_index=False)
        .agg(n_events=("grid_id", "nunique"))
        .sort_values(["threshold", "event_type"])
    )
    threshold_summary.to_csv(OUTPUT / "threshold_candidate_summary.csv", index=False)
    report = {
        "run_id": "expanded_control_pool_full_history_1990",
        "main_run": str(MAIN_RUN),
        "fixed_matched_events": int(catalogue.loc[catalogue["role"].eq("treatment"), "grid_id"].nunique()),
        "fixed_matched_links": int(len(pd.read_parquet(MAIN_RUN / "fixed_matched_pairs.parquet"))),
        "catalogue_rows": int(len(catalogue)),
        "treatment_catalogue_rows": int(catalogue["role"].eq("treatment").sum()),
        "control_catalogue_rows": int(catalogue["role"].eq("control").sum()),
        "expected_annual_climate_fire_rows": int(len(catalogue) * 9),
        "expected_monthly_rows_before_quality_filter": int(len(catalogue) * 9 * 12),
        "threshold_candidates": {
            str(threshold): int(group["grid_id"].nunique())
            for threshold, group in candidates.groupby("threshold")
        },
        "inputs": {
            "fixed_matched_pairs_sha256": sha256(MAIN_RUN / "fixed_matched_pairs.parquet"),
            "matched_treatments_sha256": sha256(MAIN_RUN / "matched_treatments.parquet"),
            "history_qualified_annual_sha256": sha256(MAIN_RUN / "history_qualified_annual.parquet"),
        },
        "validation_pass": True,
    }
    (OUTPUT / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
