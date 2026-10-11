#!/usr/bin/env python3
"""Redraw Figure 4 from preserved estimates; do not refit any analysis."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CURRENT = ROOT / "runs/20260909-extended-sensitivities/analysis_current_full_history"
ANNUAL = ROOT / "runs/20260904-expanded-control-pool"
FIGURES = HERE / "figures"
SOURCE = HERE / "source_data"
QA = HERE / "qa"
STEM = FIGURES / "Figure4_response_constructions_and_classification"
RUN_ID = "expanded_control_pool_full_history_1990"
METHODS = {
    "primary": {
        "label": "Primary monthly dynamic censoring",
        "short": "Primary monthly",
        "construction": "Annualized monthly matched effects; usable_months >= 9",
        "file": CURRENT / "standardized_event_type_bootstrap.csv",
        "counts": CURRENT / "standardized_event_type_contrast.csv",
        "color": "#355F8A", "marker": "o", "line": "-",
        "expected": (-0.01880, -0.02283, -0.01474),
    },
    "annual_dynamic": {
        "label": "Annual dynamic censoring",
        "short": "Annual dynamic",
        "construction": "Separate annual-EVI response; valid_obs >= 12",
        "file": ANNUAL / "standardized_event_type_bootstrap.csv",
        "counts": ANNUAL / "standardized_event_type_contrast.csv",
        "color": "#A45528", "marker": "s", "line": "--",
        "expected": (-0.02232, -0.02653, -0.01779),
    },
    "annual_pre_event": {
        "label": "Annual pre-event control risk set",
        "short": "Annual pre-event",
        "construction": "Separate annual-EVI response; valid_obs >= 12",
        "file": ANNUAL / "pre_event_dynamic_standardized_bootstrap.csv",
        "counts": ANNUAL / "pre_event_dynamic_standardized_contrast.csv",
        "color": "#D09A3C", "marker": "^", "line": "-.",
        "expected": (-0.02111, -0.02556, -0.01670),
    },
    "annual_ipcw": {
        "label": "Annual stabilized IPCW",
        "short": "Annual IPCW",
        "construction": "Separate annual-EVI response; valid_obs >= 12",
        "file": ANNUAL / "ipcw_censoring_sensitivity.csv",
        "counts": ANNUAL / "ipcw_censoring_sensitivity.csv",
        "color": "#7B5EA7", "marker": "D", "line": ":",
        "expected": (-0.02271, -0.02733, -0.01795),
    },
}
VALIDATIONS = ["random_stratified_5fold", "spatial_block_2deg_5fold", "region_leave_one_out"]
FEATURES = ["t0_only", "t0_to_t5_trajectory"]
CLASS_EXPECTED = {
    "auc": [[0.4914, 0.4375, 0.3940], [0.6955, 0.6807, 0.6004]],
    "balanced_accuracy": [[0.4930, 0.4756, 0.4599], [0.6463, 0.6331, 0.5714]],
    "balanced_accuracy_sd": [[0.0108, 0.0353, 0.0123], [0.0138, 0.0078, 0.0640]],
}
INK = "#27323A"
GRID = "#E0E4E7"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_data():
    reconciliation = []
    trajectories = []
    provenance = []
    for key, config in METHODS.items():
        frame = pd.read_csv(config["file"])
        selected = frame.loc[
            frame.run_id.eq(RUN_ID)
            & frame.event_type.eq("deforestation_minus_degradation")
            & frame.metric.eq("evi_mean")
        ].sort_values("relative_year").copy()
        assert selected.relative_year.tolist() == list(range(-3, 6)), config["label"]
        assert np.isfinite(selected[["estimate", "ci_low", "ci_high"]]).all().all()
        assert (selected.ci_low <= selected.estimate).all()
        assert (selected.estimate <= selected.ci_high).all()
        # Preserve the full source trajectory, although the requested plot is t0-t5.
        selected["method"] = key
        selected["method_label"] = config["label"]
        selected["outcome_construction"] = config["construction"]
        selected["source_file"] = relative(config["file"])
        counts = pd.read_csv(config["counts"])
        for group in ("deforestation", "degradation"):
            rows = counts.loc[counts.run_id.eq(RUN_ID) & counts.event_type.eq(group)].set_index("relative_year")
            selected[f"{group}_n"] = selected.relative_year.map(rows.treatment_count)
            selected[f"{group}_ess"] = selected.relative_year.map(rows.effective_sample_size)
        endpoint = selected.loc[selected.relative_year.eq(5)].iloc[0]
        for field, expected in zip(("estimate", "ci_low", "ci_high"), config["expected"]):
            raw = float(endpoint[field])
            assert round(raw, 5) == expected, (key, field, raw, expected)
            reconciliation.append({"panel": "b", "method": key, "metric": field,
                                   "raw_value": raw, "requested_value": expected,
                                   "absolute_difference": abs(raw - expected),
                                   "rounding_match": True})
        trajectories.append(selected)
        provenance.append({"method": key, "file": relative(config["file"]),
                           "sha256": digest(config["file"]), "rows_selected": 9,
                           "rows_plotted": 6, "filter": "run_id; direct EVI contrast; t0-t5"})
    direct = pd.concat(trajectories, ignore_index=True)
    classification_path = CURRENT / "audit/classification_current_summary.csv"
    classification = pd.read_csv(classification_path)
    assert len(classification) == 6
    assert set(classification.validation) == set(VALIDATIONS)
    assert set(classification.feature_set) == set(FEATURES)
    assert classification[["sample_n", "deforestation_n", "degradation_n"]].drop_duplicates().values.tolist() == [[3881, 1204, 2677]]
    for metric, feature_values in CLASS_EXPECTED.items():
        for feature, values in zip(FEATURES, feature_values):
            rows = classification.loc[classification.feature_set.eq(feature)].set_index("validation").reindex(VALIDATIONS)
            for validation, raw, expected in zip(VALIDATIONS, rows[metric], values):
                assert round(float(raw), 4) == expected
                reconciliation.append({"panel": "c" if metric == "auc" else "d",
                                       "method": f"{validation}/{feature}", "metric": metric,
                                       "raw_value": raw, "requested_value": expected,
                                       "absolute_difference": abs(raw - expected), "rounding_match": True})
    classification["source_file"] = relative(classification_path)
    provenance.append({"method": "classification", "file": relative(classification_path),
                       "sha256": digest(classification_path), "rows_selected": 6,
                       "rows_plotted": 6, "filter": "none"})
    for folder in (FIGURES, SOURCE, QA):
        folder.mkdir(parents=True, exist_ok=True)
    direct.to_csv(SOURCE / "figure4_full_preserved_trajectories.csv", index=False)
    direct.loc[direct.relative_year.ge(0)].to_csv(SOURCE / "figure4a_trajectories.csv", index=False)
    direct.loc[direct.relative_year.eq(5)].to_csv(SOURCE / "figure4b_t5_contrasts.csv", index=False)
    classification.to_csv(SOURCE / "figure4cd_classification.csv", index=False)
    pd.DataFrame(reconciliation).to_csv(SOURCE / "number_reconciliation.csv", index=False)
    pd.DataFrame(provenance).to_csv(SOURCE / "provenance_sha256.csv", index=False)
    return direct, classification


def draw(direct, classification):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.65, "axes.edgecolor": "#858E94",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.2, 6.15))
    ax_a = fig.add_axes([0.11, 0.515, 0.345, 0.265])
    ax_b = fig.add_axes([0.645, 0.515, 0.345, 0.265])
    ax_c = fig.add_axes([0.11, 0.125, 0.345, 0.265])
    ax_d = fig.add_axes([0.645, 0.125, 0.345, 0.265])
    axes = [ax_a, ax_b, ax_c, ax_d]
    for ax, label in zip(axes, "abcd"):
        ax.text(-0.12, 1.08, f"({label})", transform=ax.transAxes,
                ha="left", va="bottom", fontweight="bold", fontsize=9, color=INK)
        ax.set_axisbelow(True)
    ax_a.set_title("Response contrast trajectories", loc="left", pad=8, fontweight="bold")
    for key, config in METHODS.items():
        rows = direct.loc[direct.method.eq(key) & direct.relative_year.ge(0)]
        ax_a.fill_between(rows.relative_year, rows.ci_low, rows.ci_high,
                          color=config["color"], alpha=0.07, linewidth=0)
        ax_a.plot(rows.relative_year, rows.estimate, color=config["color"],
                  linestyle=config["line"], marker=config["marker"],
                  linewidth=1.5 if key == "primary" else 1.1,
                  markersize=3.4 if key == "primary" else 3.0,
                  markerfacecolor=config["color"] if key == "primary" else "white",
                  markeredgewidth=0.75, zorder=5 if key == "primary" else 4)
    ax_a.axhline(0, color="#899197", linewidth=0.65, linestyle="--")
    ax_a.set_xlim(-0.2, 5.2)
    ax_a.set_ylim(-0.031, 0.002)
    ax_a.set_xticks(range(6), [f"t{r}" for r in range(6)])
    ax_a.set_yticks([-0.03, -0.02, -0.01, 0])
    ax_a.set_xlabel("Relative year")
    ax_a.set_ylabel("Deforestation − degradation contrast\nin EVI response")
    ax_a.grid(axis="y", color=GRID, linewidth=0.4)

    ax_b.set_title("t5 between-type contrasts", loc="left", pad=8, fontweight="bold")
    positions = [3.5, 2.0, 1.0, 0.0]
    ax_b.axhspan(-0.5, 2.5, color="#F4F1F7", zorder=0)
    for y, (key, config) in zip(positions, METHODS.items()):
        row = direct.loc[direct.method.eq(key) & direct.relative_year.eq(5)].iloc[0]
        ax_b.errorbar(row.estimate, y,
                      xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                      fmt=config["marker"], color=config["color"], markersize=4.5,
                      markerfacecolor=config["color"] if key == "primary" else "white",
                      elinewidth=1.2, capsize=2.5, zorder=4)
        ax_b.text(0.93, y, f"{row.estimate:.5f}".replace("-", "−"),
                  transform=ax_b.get_yaxis_transform(), ha="right", va="center",
                  fontsize=6.4, color=config["color"])
    ax_b.axvline(0, color="#899197", linewidth=0.65, linestyle="--")
    ax_b.set_xlim(-0.031, 0.001)
    ax_b.set_ylim(-0.65, 4.25)
    ax_b.set_yticks(positions, [c["short"] for c in METHODS.values()])
    ax_b.set_xticks([-0.03, -0.02, -0.01, 0])
    ax_b.set_xlabel("t5 contrast in EVI response (95% CI)")
    ax_b.grid(axis="x", color=GRID, linewidth=0.4)

    feature_labels = ["t0 only", "Complete t0–t5 trajectory"]
    colors = ["#B2BAC0", "#355F8A"]
    labels = ["Stratified random\n5-fold", "2° spatial-block\n5-fold", "Leave-one-\nregion-out"]
    x = np.arange(3)
    width = 0.34
    for ax, metric, title in [(ax_c, "auc", "Pooled out-of-fold AUC"),
                               (ax_d, "balanced_accuracy", "Balanced accuracy")]:
        ax.set_title(title, loc="left", pad=8, fontweight="bold")
        for offset, feature, color in zip([-width / 2, width / 2], FEATURES, colors):
            rows = classification.loc[classification.feature_set.eq(feature)].set_index("validation").reindex(VALIDATIONS)
            kwargs = {}
            if metric == "balanced_accuracy":
                kwargs = {"yerr": rows.balanced_accuracy_sd.to_numpy(),
                          "error_kw": {"ecolor": INK, "elinewidth": 0.8, "capsize": 2, "capthick": 0.8}}
            ax.bar(x + offset, rows[metric], width=width, color=color, zorder=3, **kwargs)
            for xi, value in zip(x + offset, rows[metric]):
                sd = float(rows.iloc[int(round(xi - offset))].balanced_accuracy_sd) if metric == "balanced_accuracy" else 0
                label_y = value + sd + 0.010
                if metric == "balanced_accuracy":
                    label_y = max(label_y, 0.535)
                ax.text(xi, label_y, f"{value:.4f}", ha="center", va="bottom", fontsize=6.0, color=INK)
        ax.axhline(0.5, color="#899197", linewidth=0.65, linestyle="--", zorder=2)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 0.79)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        ax.set_ylabel(title)
        ax.grid(axis="y", color=GRID, linewidth=0.4)
    fig.text(0.11, 0.97, "Primary analysis", fontsize=7, fontweight="bold", color=METHODS["primary"]["color"])
    fig.text(0.52, 0.97, "Separate annual-EVI response construction", fontsize=7, fontweight="bold", color=INK)
    def handle(config):
        return Line2D([0], [0], color=config["color"], linestyle=config["line"],
                      marker=config["marker"], markersize=3.3, linewidth=1.2,
                      markerfacecolor=config["color"] if config is METHODS["primary"] else "white",
                      label=config["label"])
    fig.legend(handles=[handle(METHODS["primary"])], bbox_to_anchor=(0.105, 0.955),
               loc="upper left", frameon=False, borderaxespad=0, fontsize=6.8)
    fig.legend(handles=[handle(c) for key, c in METHODS.items() if key != "primary"],
               bbox_to_anchor=(0.515, 0.955), loc="upper left", frameon=False,
               borderaxespad=0, fontsize=6.8, labelspacing=0.7)
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, feature_labels)],
               loc="center", bbox_to_anchor=(0.55, 0.442), ncol=2, frameon=False,
               fontsize=7, columnspacing=2)
    fig.text(0.5, 0.025,
             "(a–b) 95% spatial-block bootstrap CIs.  (c) Pooled OOF AUC, without error bars.\n"
             "(d) Mean test-fold balanced accuracy ± fold SD.  Classification cohort: n = 3,881.",
             ha="center", va="bottom", fontsize=6.2, color="#5C656B", linespacing=1.4)
    fig.canvas.draw()
    fig.savefig(str(STEM) + ".pdf")
    fig.savefig(str(STEM) + ".svg")
    fig.savefig(str(STEM) + ".png", dpi=600)
    fig.savefig(str(STEM) + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(FIGURES / "Figure4_preview.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    direct, classification = load_data()
    draw(direct, classification)
    print(f"Figure 4 generated in {FIGURES}")
