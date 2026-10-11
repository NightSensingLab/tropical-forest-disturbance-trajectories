from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


MM_PER_INCH = 25.4
DOUBLE_COLUMN_MM = 183.0
FIGURE_HEIGHT_MM = 126.0

COLORS = {
    "ink": "#282C2F",
    "ink_soft": "#646B70",
    "grid": "#E4E7E8",
    "degradation": "#287B78",
    "deforestation": "#C35B4D",
    "pre": "#AEB4B7",
    "post": "#3C6E8F",
    "gate": "#8C4242",
    "paper": "#FFFFFF",
}

COVARIATE_LABELS = {
    "baseline_evi_mean": "Mean EVI",
    "baseline_evi_min": "Minimum EVI",
    "baseline_evi_max": "Maximum EVI",
    "baseline_evi_amp": "EVI amplitude",
    "baseline_evi_sd": "EVI variability",
    "pretrend_evi_mean": "Pre-event EVI trend",
    "baseline_undist_share": "Undisturbed share",
    "log_analysis_area": "Analysis area (log)",
}


def publication_rcparams():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "legend.fontsize": 6.3,
        "legend.frameon": False,
        "figure.facecolor": COLORS["paper"],
        "savefig.facecolor": COLORS["paper"],
    }


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


def _base_axis(axis):
    axis.axhline(0.0, color=COLORS["ink_soft"], linewidth=0.55, linestyle=(0, (3, 2)), zorder=0)
    axis.axvline(0.0, color=COLORS["grid"], linewidth=0.55, zorder=0)
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.4, zorder=0)
    axis.set_xticks(range(-3, 6))
    axis.tick_params(length=2.2, pad=2)


def _response_source(response):
    required = {
        "scope",
        "event_type",
        "metric",
        "relative_year",
        "weighting",
        "treatment_count",
        "estimate",
        "ci_low",
        "ci_high",
    }
    missing = required - set(response.columns)
    if missing:
        raise ValueError(f"Response table lacks columns: {sorted(missing)}")
    return response.loc[
        response["scope"].eq("Global")
        & response["metric"].eq("evi_mean")
        & response["weighting"].eq("event_area")
    ].copy()


def build_matched_response_figure(response):
    source = _response_source(response)
    with plt.rc_context(publication_rcparams()):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(DOUBLE_COLUMN_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH),
            layout="constrained",
        )
        axes = axes.ravel()
        titles = {
            "degradation": "Degradation-dominant events",
            "deforestation": "Deforestation-dominant events",
        }
        finite_limits = []
        for event_type in ("degradation", "deforestation"):
            rows = source.loc[source["event_type"].eq(event_type)].sort_values(
                "relative_year"
            )
            finite_limits.extend(pd.to_numeric(rows["ci_low"], errors="coerce").dropna())
            finite_limits.extend(pd.to_numeric(rows["ci_high"], errors="coerce").dropna())
        limit = max(abs(float(np.min(finite_limits))), abs(float(np.max(finite_limits))))
        limit = max(limit * 1.15, 0.005)

        for axis, event_type, panel in zip(
            axes[:2], ("degradation", "deforestation"), ("a", "b")
        ):
            rows = source.loc[source["event_type"].eq(event_type)].sort_values(
                "relative_year"
            )
            x = rows["relative_year"].to_numpy(dtype=np.float64)
            estimate = rows["estimate"].to_numpy(dtype=np.float64)
            low = rows["ci_low"].to_numpy(dtype=np.float64)
            high = rows["ci_high"].to_numpy(dtype=np.float64)
            color = COLORS[event_type]
            axis.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
            axis.plot(
                x,
                estimate,
                color=color,
                linewidth=1.35,
                marker="o",
                markersize=2.7,
                markeredgewidth=0,
            )
            _base_axis(axis)
            axis.set_ylim(-limit, limit)
            axis.set_title(titles[event_type], loc="left", fontweight="bold", pad=4)
            axis.set_xlabel("Years since first disturbance")
            axis.set_ylabel("Matched EVI difference")
            _panel_label(axis, panel)

        contrast_axis = axes[2]
        contrast_rows = source.loc[source["relative_year"].isin([0, 5])].copy()
        contrast_rows["label"] = contrast_rows.apply(
            lambda row: (
                ("Degradation" if row.event_type == "degradation" else "Deforestation")
                + ("  t0" if row.relative_year == 0 else "  t+5")
            ),
            axis=1,
        )
        contrast_rows = contrast_rows.sort_values(
            ["event_type", "relative_year"], ascending=[False, True]
        )
        y = np.arange(len(contrast_rows))[::-1]
        for yi, row in zip(y, contrast_rows.itertuples(index=False)):
            color = COLORS[row.event_type]
            contrast_axis.plot([row.ci_low, row.ci_high], [yi, yi], color=color, linewidth=1.2)
            contrast_axis.plot(row.estimate, yi, "o", color=color, markersize=3.2)
        contrast_axis.axvline(
            0.0, color=COLORS["ink_soft"], linewidth=0.6, linestyle=(0, (3, 2))
        )
        contrast_axis.set_yticks(y)
        contrast_axis.set_yticklabels(contrast_rows["label"])
        contrast_axis.set_xlabel("Matched EVI difference")
        contrast_axis.set_title("Event-year and year-five contrasts", loc="left", fontweight="bold", pad=4)
        contrast_axis.grid(axis="x", color=COLORS["grid"], linewidth=0.4)
        contrast_axis.tick_params(length=2.2, pad=2)
        _panel_label(contrast_axis, "c")

        coverage_axis = axes[3]
        for event_type in ("degradation", "deforestation"):
            rows = source.loc[source["event_type"].eq(event_type)].sort_values(
                "relative_year"
            )
            coverage_axis.plot(
                rows["relative_year"],
                rows["treatment_count"],
                color=COLORS[event_type],
                linewidth=1.25,
                marker="o",
                markersize=2.5,
                label=event_type.capitalize(),
            )
        coverage_axis.axvline(0.0, color=COLORS["grid"], linewidth=0.55)
        coverage_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
        coverage_axis.set_xticks(range(-3, 6))
        coverage_axis.set_xlabel("Years since first disturbance")
        coverage_axis.set_ylabel("Matched treatment grids")
        coverage_axis.set_title("Pair-year outcome coverage", loc="left", fontweight="bold", pad=4)
        coverage_axis.legend(loc="best", handlelength=1.7)
        coverage_axis.tick_params(length=2.2, pad=2)
        _panel_label(coverage_axis, "d")
        figure.canvas.draw()
        return figure


def build_balance_diagnostics_figure(balance, pairs):
    required_balance = {"scope", "covariate", "pre_smd", "post_smd"}
    required_pairs = {
        "treatment_grid_id",
        "control_grid_id",
        "geographic_distance_km",
        "control_reuse_count",
    }
    if required_balance - set(balance.columns):
        raise ValueError("Balance table lacks required columns.")
    if required_pairs - set(pairs.columns):
        raise ValueError("Pair table lacks required columns.")
    source = balance.loc[balance["scope"].eq("Overall")].copy()
    source["label"] = source["covariate"].map(COVARIATE_LABELS).fillna(
        source["covariate"]
    )
    source = source.sort_values("post_smd", key=lambda value: value.abs())
    with plt.rc_context(publication_rcparams()):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(DOUBLE_COLUMN_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH),
            layout="constrained",
        )
        axes = axes.ravel()
        y = np.arange(len(source))
        axes[0].plot(
            source["pre_smd"].abs(),
            y,
            "o",
            color=COLORS["pre"],
            markersize=3.2,
            label="Before matching",
        )
        axes[0].plot(
            source["post_smd"].abs(),
            y,
            "o",
            color=COLORS["post"],
            markersize=3.2,
            label="After matching",
        )
        axes[0].axvline(
            0.10, color=COLORS["gate"], linewidth=0.7, linestyle=(0, (3, 2))
        )
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(source["label"])
        axes[0].set_xlabel("Absolute standardized difference")
        axes[0].set_title("Covariate balance", loc="left", fontweight="bold", pad=4)
        axes[0].legend(loc="lower right", handletextpad=0.4)
        axes[0].grid(axis="x", color=COLORS["grid"], linewidth=0.4)
        _panel_label(axes[0], "a")

        distance = pd.to_numeric(pairs["geographic_distance_km"], errors="coerce").dropna()
        axes[1].hist(distance, bins=24, color=COLORS["degradation"], edgecolor="white", linewidth=0.25)
        axes[1].axvline(50.0, color=COLORS["gate"], linewidth=0.7, linestyle=(0, (3, 2)))
        axes[1].set_xlabel("Treatment-control distance (km)")
        axes[1].set_ylabel("Pair links")
        axes[1].set_title("Geographic proximity", loc="left", fontweight="bold", pad=4)
        _panel_label(axes[1], "b")

        controls_per_treatment = pairs.groupby("treatment_grid_id")["control_grid_id"].nunique()
        control_counts = controls_per_treatment.value_counts().sort_index()
        axes[2].bar(
            control_counts.index.astype(str),
            control_counts.values,
            color=COLORS["post"],
            width=0.65,
        )
        axes[2].set_xlabel("Controls per treatment")
        axes[2].set_ylabel("Treatment grids")
        axes[2].set_title("Variable-ratio matching", loc="left", fontweight="bold", pad=4)
        _panel_label(axes[2], "c")

        reuse = pairs.drop_duplicates("control_grid_id")["control_reuse_count"]
        maximum = max(int(pd.to_numeric(reuse).max()), 1)
        axes[3].hist(
            reuse,
            bins=np.arange(0.5, maximum + 1.6, 1.0),
            color=COLORS["deforestation"],
            edgecolor="white",
            linewidth=0.25,
        )
        axes[3].set_xlabel("Treatments sharing a control")
        axes[3].set_ylabel("Unique controls")
        axes[3].set_title("Control reuse", loc="left", fontweight="bold", pad=4)
        _panel_label(axes[3], "d")
        for axis in axes[1:]:
            axis.grid(axis="y", color=COLORS["grid"], linewidth=0.4)
            axis.tick_params(length=2.2, pad=2)
        figure.canvas.draw()
        return figure


def export_figure(figure, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    with plt.rc_context(publication_rcparams()):
        for suffix, dpi in ((".png", 300), (".pdf", 300), (".svg", 300), (".tiff", 600)):
            output = base_path.with_suffix(suffix)
            temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
            kwargs = {"dpi": dpi, "facecolor": COLORS["paper"], "edgecolor": "none"}
            if suffix == ".tiff":
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            figure.savefig(temporary, **kwargs)
            temporary.replace(output)
            outputs.append(output)
    plt.close(figure)
    return outputs


def validate_figure_bundle(paths):
    paths = [Path(path) for path in paths]
    suffixes = {path.suffix for path in paths}
    checks = {
        "formats_complete": suffixes == {".png", ".pdf", ".svg", ".tiff"},
        "files_nonempty": all(path.is_file() and path.stat().st_size > 1000 for path in paths),
    }
    raster_valid = True
    for path in paths:
        if path.suffix in {".png", ".tiff"}:
            image = np.asarray(Image.open(path).convert("RGB"))
            raster_valid = raster_valid and image.shape[0] > 500 and image.shape[1] > 1000
            raster_valid = raster_valid and float(image.std()) > 2.0
    checks["rasters_nonblank"] = raster_valid
    svg_paths = [path for path in paths if path.suffix == ".svg"]
    checks["svg_text_editable"] = bool(
        svg_paths and "<text" in svg_paths[0].read_text(encoding="utf-8")
    )
    return {"pass": all(checks.values()), "checks": checks}
