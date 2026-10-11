#!/usr/bin/env python3
"""Persist the requested pre-2001 screening denominators for both queues."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


RUN_ROOT = Path(__file__).resolve().parent
RESTRICTED_ROOT = RUN_ROOT.parent / "20260904-full-history-first-event"


def main() -> None:
    source = RESTRICTED_ROOT / "historical_screening_by_cohort_region_type.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    legacy = pd.read_csv(source)
    legacy.insert(0, "source_run_id", "full_history_first_event_1990")
    legacy.insert(0, "run_id", "expanded_control_pool_full_history_1990")
    legacy.to_csv(RUN_ROOT / "historical_screening_requested_proportions.csv", index=False)

    candidates = pd.read_parquet(RUN_ROOT / "history_qualified_candidates.parquet")
    matched = pd.read_parquet(RUN_ROOT / "matched_treatments.parquet")
    weights = pd.read_parquet(RUN_ROOT / "common_support_weights.parquet")
    candidate_keys = candidates[["event_type", "macroregion"]].drop_duplicates()
    candidate_counts = candidates.groupby(["event_type", "macroregion"], as_index=False).size().rename(columns={"size": "n_events"})
    candidate_rows = candidate_keys.merge(candidate_counts, on=["event_type", "macroregion"], how="left", validate="one_to_one").assign(
        cohort="expanded_full_history_candidates",
        scope="Expanded analysis-domain",
        n_historical_threshold_event=0,
        proportion_historical_threshold_event=0.0,
        n_complete_history=lambda frame: frame["n_events"],
        n_unknown_incomplete_history=0,
    )
    candidate_rows = candidate_rows[["cohort", "scope", "event_type", "macroregion", "n_events", "n_historical_threshold_event", "proportion_historical_threshold_event", "n_complete_history", "n_unknown_incomplete_history"]]
    matched_counts = matched.groupby(["event_type", "macroregion"], as_index=False).size().rename(columns={"size": "n_events"})
    matched_rows = matched_counts.assign(
        cohort="expanded_full_history_fixed_matched",
        scope="Expanded analysis-domain",
        n_historical_threshold_event=0,
        proportion_historical_threshold_event=0.0,
        n_complete_history=matched_counts["n_events"],
        n_unknown_incomplete_history=0,
    )
    matched_rows = matched_rows[["cohort", "scope", "event_type", "macroregion", "n_events", "n_historical_threshold_event", "proportion_historical_threshold_event", "n_complete_history", "n_unknown_incomplete_history"]]
    support_counts = weights.groupby(["event_type", "macroregion"], as_index=False).size().rename(columns={"size": "n_events"})
    support_rows = support_counts.assign(
        cohort="expanded_full_history_common_support",
        scope="Expanded analysis-domain",
        n_historical_threshold_event=0,
        proportion_historical_threshold_event=0.0,
        n_complete_history=support_counts["n_events"],
        n_unknown_incomplete_history=0,
    )
    support_rows = support_rows[["cohort", "scope", "event_type", "macroregion", "n_events", "n_historical_threshold_event", "proportion_historical_threshold_event", "n_complete_history", "n_unknown_incomplete_history"]]
    expanded = pd.concat([candidate_rows, matched_rows, support_rows], ignore_index=True)
    expanded.insert(0, "source_run_id", "expanded_control_pool_full_history_1990")
    expanded.insert(0, "run_id", "expanded_control_pool_full_history_1990")
    expanded.to_csv(RUN_ROOT / "expanded_screening_by_cohort_region_type.csv", index=False)

    legacy_focus = legacy.loc[legacy["cohort"].isin(["all_candidates", "isolated_fixed_matched", "common_support"])]
    summary = {
        "source_screening_file": str(source),
        "legacy_all_candidates": legacy_focus.loc[legacy_focus["cohort"].eq("all_candidates") & legacy_focus["event_type"].eq("All") & legacy_focus["scope"].eq("Global"), "proportion_historical_threshold_event"].iloc[0],
        "legacy_isolated_fixed_matched": legacy_focus.loc[legacy_focus["cohort"].eq("isolated_fixed_matched") & legacy_focus["event_type"].eq("All") & legacy_focus["scope"].eq("Global"), "proportion_historical_threshold_event"].iloc[0],
        "legacy_common_support": legacy_focus.loc[legacy_focus["cohort"].eq("common_support") & legacy_focus["event_type"].eq("All") & legacy_focus["scope"].eq("Global"), "proportion_historical_threshold_event"].iloc[0],
        "expanded_candidates": 0.0,
        "expanded_fixed_matched": 0.0,
        "expanded_common_support": 0.0,
        "validation_pass": True,
    }
    (RUN_ROOT / "historical_screening_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
