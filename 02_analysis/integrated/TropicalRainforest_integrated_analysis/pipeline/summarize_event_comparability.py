#!/usr/bin/env python3
"""Summarize observed event intensity, composition and baseline support.

The summaries are descriptive diagnostics for the fixed matched event cohort.
They are not additional adjustment variables and do not replace the
common-support standardization used for the event-type contrast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_COLUMNS = [
    "treatment_grid_id",
    "event_year",
    "event_type",
    "macroregion",
    "disturbance_share",
    "degradation_share",
    "deforestation_share",
    "baseline_undist_share",
]


def summarize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, EVENT_COLUMNS].copy()
    events = frame.drop_duplicates(["treatment_grid_id", "event_year"]).copy()
    if events.empty:
        raise ValueError("The matched event window contains no event records.")
    total = events["degradation_share"] + events["deforestation_share"]
    events["degradation_composition"] = np.divide(
        events["degradation_share"],
        total,
        out=np.full(len(events), np.nan),
        where=total.gt(0).to_numpy(),
    )
    events["deforestation_composition"] = np.divide(
        events["deforestation_share"],
        total,
        out=np.full(len(events), np.nan),
        where=total.gt(0).to_numpy(),
    )

    def _summarize_group(group: pd.DataFrame, dimension: str, value: object) -> dict:
        def q(column: str, probability: float) -> float:
            return float(pd.to_numeric(group[column], errors="coerce").quantile(probability))

        return {
            "stratum_dimension": dimension,
            "stratum": str(value),
            "event_type": str(group["event_type"].iloc[0]),
            "event_n": int(len(group)),
            "d_plus_f_mean": float(group["disturbance_share"].mean()),
            "d_plus_f_median": float(group["disturbance_share"].median()),
            "d_plus_f_q25": q("disturbance_share", 0.25),
            "d_plus_f_q75": q("disturbance_share", 0.75),
            "degradation_composition_mean": float(group["degradation_composition"].mean()),
            "deforestation_composition_mean": float(group["deforestation_composition"].mean()),
            "baseline_undist_mean": float(group["baseline_undist_share"].mean()),
            "baseline_undist_median": float(group["baseline_undist_share"].median()),
            "baseline_undist_q25": q("baseline_undist_share", 0.25),
            "baseline_undist_q75": q("baseline_undist_share", 0.75),
        }

    rows: list[dict] = []
    for (region, event_type), group in events.groupby(
        ["macroregion", "event_type"], sort=True
    ):
        rows.append(_summarize_group(group, "macroregion", region))
    for (event_year, event_type), group in events.groupby(
        ["event_year", "event_type"], sort=True
    ):
        rows.append(_summarize_group(group, "event_year", int(event_year)))
    return pd.DataFrame(rows)


def summarize(window_path: str | Path) -> pd.DataFrame:
    return summarize_frame(pd.read_parquet(window_path, columns=EVENT_COLUMNS))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window", type=Path, help="all_first_matched_event_window.parquet")
    parser.add_argument("output", type=Path, help="CSV output path")
    args = parser.parse_args(argv)
    result = summarize(args.window)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} comparability rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
