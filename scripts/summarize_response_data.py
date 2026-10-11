"""Summarize response constructions and classification metrics as CSV tables."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "data/primary/event_level"
ANNUAL = ROOT / "data/annual_evi"
RUN_ID = "expanded_control_pool_full_history_1990"
METHODS = {
    "primary": {
        "label": "Primary monthly dynamic censoring",
        "construction": "Annualized monthly matched effects; usable_months >= 9",
        "file": PRIMARY / "standardized_event_type_bootstrap.csv",
        "counts": PRIMARY / "standardized_event_type_contrast.csv",
    },
    "annual_dynamic": {
        "label": "Annual dynamic censoring",
        "construction": "Separate annual-EVI response; valid_obs >= 12",
        "file": ANNUAL / "standardized_event_type_bootstrap.csv",
        "counts": ANNUAL / "standardized_event_type_contrast.csv",
    },
    "annual_pre_event": {
        "label": "Annual pre-event control risk set",
        "construction": "Separate annual-EVI response; valid_obs >= 12",
        "file": ANNUAL / "pre_event_dynamic_standardized_bootstrap.csv",
        "counts": ANNUAL / "pre_event_dynamic_standardized_contrast.csv",
    },
    "annual_ipcw": {
        "label": "Annual stabilized IPCW",
        "construction": "Separate annual-EVI response; valid_obs >= 12",
        "file": ANNUAL / "ipcw_censoring_sensitivity.csv",
        "counts": ANNUAL / "ipcw_censoring_sensitivity.csv",
    },
}


def summarize():
    trajectories = []
    for method, config in METHODS.items():
        frame = pd.read_csv(config["file"])
        selected = frame.loc[
            frame.run_id.eq(RUN_ID)
            & frame.event_type.eq("deforestation_minus_degradation")
            & frame.metric.eq("evi_mean")
        ].sort_values("relative_year").copy()
        if selected.relative_year.tolist() != list(range(-3, 6)):
            raise ValueError(f"Incomplete response trajectory: {method}")
        if not np.isfinite(selected[["estimate", "ci_low", "ci_high"]]).all().all():
            raise ValueError(f"Nonfinite response estimates: {method}")
        if not ((selected.ci_low <= selected.estimate) & (selected.estimate <= selected.ci_high)).all():
            raise ValueError(f"Invalid confidence interval: {method}")
        selected["method"] = method
        selected["method_label"] = config["label"]
        selected["outcome_construction"] = config["construction"]
        selected["source_file"] = str(config["file"].relative_to(ROOT))
        counts = pd.read_csv(config["counts"])
        for event_type in ("deforestation", "degradation"):
            rows = counts.loc[
                counts.run_id.eq(RUN_ID) & counts.event_type.eq(event_type)
            ].set_index("relative_year")
            selected[f"{event_type}_n"] = selected.relative_year.map(rows.treatment_count)
            selected[f"{event_type}_ess"] = selected.relative_year.map(rows.effective_sample_size)
        trajectories.append(selected)
    direct = pd.concat(trajectories, ignore_index=True)
    classification_path = PRIMARY / "diagnostics/classification_summary.csv"
    classification = pd.read_csv(classification_path)
    if classification[["sample_n", "deforestation_n", "degradation_n"]].drop_duplicates().values.tolist() != [[3881, 1204, 2677]]:
        raise ValueError("Classification summaries do not share the reported cohort")
    classification["source_file"] = str(classification_path.relative_to(ROOT))
    return direct, classification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/recomputed/response_summary")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    direct, classification = summarize()
    direct.to_csv(output / "figure4_full_trajectories.csv", index=False)
    direct.loc[direct.relative_year.ge(0)].to_csv(output / "figure4a_trajectories.csv", index=False)
    direct.loc[direct.relative_year.eq(5)].to_csv(output / "figure4b_t5_contrasts.csv", index=False)
    classification.to_csv(output / "figure4cd_classification.csv", index=False)
    print(f"Response and classification summary CSVs written to {output}")


if __name__ == "__main__":
    main()
