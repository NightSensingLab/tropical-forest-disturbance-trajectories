"""Reproduce annual EVI risk-set sensitivities from frozen matched windows."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline.annual_sensitivity import _ipcw_event_type_contrast
from pipeline.matched_control import MatchedControlSettings
from pipeline.matched_estimation import (
    bootstrap_standardized_event_type_contrast,
    summarize_standardized_event_type_contrast,
)

INPUT = ROOT / "data/annual_evi"
OUTPUT = ROOT / "results/recomputed/annual_evi"
RUN_ID = "expanded_control_pool_full_history_1990"


def save(frame, name):
    frame = frame.copy()
    frame.insert(0, "run_id", RUN_ID)
    frame.to_csv(OUTPUT / name, index=False)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    settings = MatchedControlSettings(bootstrap_replicates=1000, bootstrap_seed=20260725)
    for prefix, name in [
        ("", "fixed_matched_event_window.parquet"),
        ("pre_event_dynamic_", "pre_event_dynamic_control_window.parquet"),
    ]:
        window = pd.read_parquet(INPUT / name)
        events = window.drop_duplicates(["treatment_grid_id", "event_year"]).copy()
        summary, _, _ = summarize_standardized_event_type_contrast(
            window, events, metric="evi_mean", relative_years=settings.relative_years,
        )
        bootstrap = bootstrap_standardized_event_type_contrast(
            window, events, metric="evi_mean", relative_years=settings.relative_years,
            block_degrees=2.0, replicates=1000, seed=20260725,
        )
        stem = "pre_event_dynamic_standardized" if prefix else "standardized_event_type"
        save(summary, stem + "_contrast.csv")
        save(bootstrap, stem + "_bootstrap.csv")
        if not prefix:
            ipcw, weights, model = _ipcw_event_type_contrast(window, events, settings)
            save(ipcw, "ipcw_censoring_sensitivity.csv")
            save(model, "ipcw_censoring_model.csv")
            weights.to_parquet(OUTPUT / "ipcw_censoring_weights.parquet", index=False)
    print(f"Annual EVI sensitivities written to {OUTPUT}")


if __name__ == "__main__":
    main()
