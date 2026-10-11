from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.matched_control_figures import COLORS, MM_PER_INCH, publication_rcparams


def _panel_label(axis, label):
    axis.text(
        -0.11,
        1.04,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def _series_ci(frame):
    frame = frame.copy()
    if "ci_low" not in frame:
        frame["ci_low"] = frame["estimate"]
    if "ci_high" not in frame:
        frame["ci_high"] = frame["estimate"]
    return frame


def build_major23_revision_figure(response, contrast, composition, events):
    required_response = {"event_type", "relative_year", "estimate", "treatment_count"}
    required_contrast = {"event_type", "relative_year", "estimate"}
    required_composition = {"event_type", "macroregion", "repeat_within_5y", "treatment_count"}
    required_events = {"event_type", "disturbance_share", "baseline_undist_share"}
    for name, frame, required in (
        ("response", response, required_response),
        ("contrast", contrast, required_contrast),
        ("composition", composition, required_composition),
        ("events", events, required_events),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table lacks columns: {sorted(missing)}")

    source = response.copy()
    if "cohort" in source:
        source = source.loc[source["cohort"].eq("all_first_dynamic_censored")]
    if "metric" in source:
        source = source.loc[source["metric"].eq("evi_mean")]
    if "weighting" in source:
        source = source.loc[source["weighting"].eq("event_area")]
    source = _series_ci(source)
    with plt.rc_context(publication_rcparams()):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(183.0 / MM_PER_INCH, 126.0 / MM_PER_INCH),
            layout="constrained",
        )
        axes = axes.ravel()

        # A: all-first dynamic-censor trajectories, with coverage counts.
        axis = axes[0]
        for event_type in ("degradation", "deforestation"):
            rows = source.loc[source["event_type"].eq(event_type)].sort_values("relative_year")
            if rows.empty:
                continue
            color = COLORS[event_type]
            x = rows["relative_year"].to_numpy(dtype=float)
            estimate = rows["estimate"].to_numpy(dtype=float)
            low = rows["ci_low"].to_numpy(dtype=float)
            high = rows["ci_high"].to_numpy(dtype=float)
            axis.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
            axis.plot(x, estimate, color=color, marker="o", markersize=2.4, linewidth=1.2, label=event_type.capitalize())
        axis.axhline(0.0, color=COLORS["ink_soft"], linewidth=0.55, linestyle=(0, (3, 2)))
        axis.axvline(0.0, color=COLORS["grid"], linewidth=0.55)
        axis.set_xticks(range(-3, 6))
        axis.set_xlabel("Years since first event")
        axis.set_ylabel("Matched EVI difference")
        axis.set_title("All-first dynamic-censor trajectories", loc="left", fontweight="bold", pad=4)
        axis.legend(loc="best", handlelength=1.7)
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
        _panel_label(axis, "a")

        # B: standardized direct contrast at t5 or all available years.
        axis = axes[1]
        rows = contrast.loc[contrast["event_type"].eq("deforestation_minus_degradation")].copy()
        rows = _series_ci(rows).sort_values("relative_year")
        if not rows.empty:
            x = rows["relative_year"].to_numpy(dtype=float)
            axis.fill_between(x, rows["ci_low"], rows["ci_high"], color=COLORS["post"], alpha=0.18, linewidth=0)
            axis.plot(x, rows["estimate"], color=COLORS["post"], marker="o", markersize=2.7, linewidth=1.3)
        axis.axhline(0.0, color=COLORS["ink_soft"], linewidth=0.55, linestyle=(0, (3, 2)))
        axis.axvline(0.0, color=COLORS["grid"], linewidth=0.55)
        axis.set_xticks(range(-3, 6))
        axis.set_xlabel("Years since first event")
        axis.set_ylabel("Deforestation − degradation")
        axis.set_title("Standardized event-type contrast", loc="left", fontweight="bold", pad=4)
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
        _panel_label(axis, "b")

        # C: composition of repeat events by type and macroregion.
        axis = axes[2]
        comp = composition.copy()
        totals = comp.groupby(["event_type", "macroregion"], sort=False)["treatment_count"].transform("sum")
        comp["share"] = comp["treatment_count"] / totals.replace(0, np.nan)
        comp = comp.loc[comp["repeat_within_5y"].astype(bool)]
        pivot = comp.pivot_table(index="macroregion", columns="event_type", values="share", aggfunc="sum", fill_value=0.0)
        regions = list(pivot.index)
        pos = np.arange(len(regions))
        width = 0.36
        for offset, event_type in ((-width / 2, "degradation"), (width / 2, "deforestation")):
            values = pivot[event_type].to_numpy() if event_type in pivot else np.zeros(len(regions))
            axis.bar(pos + offset, values, width=width, color=COLORS[event_type], label=event_type.capitalize())
        axis.set_xticks(pos)
        axis.set_xticklabels(regions, rotation=35, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Share with repeat event")
        axis.set_title("Repeat-event composition", loc="left", fontweight="bold", pad=4)
        axis.legend(loc="best", handlelength=1.7)
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
        _panel_label(axis, "c")

        # D: pooled distributions before and after common-support retention.
        axis = axes[3]
        finite_events = events.copy()
        for event_type, group in finite_events.groupby("event_type", sort=True):
            color = COLORS[event_type]
            axis.scatter(
                group["disturbance_share"],
                group["baseline_undist_share"],
                s=8,
                alpha=0.24,
                color=color,
                label=event_type.capitalize(),
                linewidths=0,
            )
        axis.set_xlabel("D+F disturbance share")
        axis.set_ylabel("Baseline undisturbed-forest share")
        axis.set_title("Event-composition support", loc="left", fontweight="bold", pad=4)
        axis.legend(loc="best", handlelength=1.7)
        axis.grid(color=COLORS["grid"], linewidth=0.4)
        _panel_label(axis, "d")

        figure.canvas.draw()
        return figure


def export_revision_figure(figure, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
        (".svg", {}),
        (".tiff", {"dpi": 300}),
    ):
        path = stem.with_suffix(suffix)
        figure.savefig(path, **kwargs)
        paths.append(path)
    return tuple(paths)
