from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image, ImageStat


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

EVENT_COLORS = {
    "degradation": "#217A8A",
    "deforestation": "#B44A3A",
}
EVENT_LABELS = {
    "degradation": "Degradation",
    "deforestation": "Deforestation",
}
REGION_LABELS = {
    "Amazon_proxy": "Amazon",
    "Congo_proxy": "Congo",
    "Southeast_Asia_proxy": "Southeast Asia",
}
RESPONSE_CMAP = LinearSegmentedColormap.from_list(
    "stage09_response", ["#2B6C9E", "#D9E8EE", "#F7F7F4", "#F1D5C7", "#A33B2F"]
)


def apply_publication_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _new_grid(bottom=0.085):
    apply_publication_style()
    width_in = 183.0 / 25.4
    height_in = 154.0 / 25.4
    figure, axes = plt.subplots(2, 2, figsize=(width_in, height_in))
    figure.subplots_adjust(left=0.085, right=0.975, bottom=bottom, top=0.95, wspace=0.29, hspace=0.34)
    figure._stage09_panel_axes = tuple(axes.flat)
    return figure, axes


def _panel_label(axis, label):
    axis.text(
        -0.13,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _global_primary(frame):
    result = frame.loc[frame["scope"].eq("Global")].copy()
    if "weighting" in result:
        result = result.loc[result["weighting"].eq("event_area")]
    return result


def _heatmap_array(frame, event_type):
    selected = _global_primary(frame)
    selected = selected.loc[selected["event_type"].eq(event_type)]
    matrix = selected.pivot(index="calendar_month", columns="relative_year", values="estimate")
    return matrix.reindex(index=range(1, 13), columns=range(-3, 6))


def build_main_figure(monthly_summary, annual_summary, contrast_summary):
    required_monthly = {"scope", "event_type", "relative_year", "calendar_month", "estimate"}
    required_annual = {"scope", "event_type", "relative_year", "estimate", "ci_low", "ci_high"}
    required_contrast = {"scope", "event_type", "contrast", "estimate", "ci_low", "ci_high"}
    for frame, required, label in (
        (monthly_summary, required_monthly, "Monthly summary"),
        (annual_summary, required_annual, "Annual summary"),
        (contrast_summary, required_contrast, "Contrast summary"),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} lacks columns: {sorted(missing)}")
    figure, axes = _new_grid(bottom=0.125)
    degradation = _heatmap_array(monthly_summary, "degradation")
    deforestation = _heatmap_array(monthly_summary, "deforestation")
    finite = np.concatenate(
        [
            degradation.to_numpy().ravel(),
            deforestation.to_numpy().ravel(),
        ]
    )
    finite = np.abs(finite[np.isfinite(finite)])
    limit = float(np.quantile(finite, 0.98)) if len(finite) else 0.01
    limit = max(limit, 1e-6)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    for axis, matrix, event_type, label in zip(
        axes[0],
        (degradation, deforestation),
        ("degradation", "deforestation"),
        ("a", "b"),
    ):
        image = axis.imshow(
            matrix.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap=RESPONSE_CMAP,
            norm=norm,
            interpolation="nearest",
            extent=(-3.5, 5.5, 0.5, 12.5),
        )
        axis.axvline(-0.5, color="#333333", linewidth=0.65, linestyle=(0, (2, 2)))
        axis.set_xticks(range(-3, 6))
        axis.set_yticks([1, 3, 5, 7, 9, 11])
        axis.set_xlabel("Relative year")
        axis.set_ylabel("Calendar month")
        axis.set_title(EVENT_LABELS[event_type], loc="left", fontweight="bold")
        _panel_label(axis, label)
    color_axis = figure.add_axes([0.35, 0.035, 0.30, 0.016])
    colorbar = figure.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Monthly matched EVI anomaly", labelpad=1)
    colorbar.ax.tick_params(length=2, pad=1)

    axis = axes[1, 0]
    annual = _global_primary(annual_summary)
    for event_type in ("degradation", "deforestation"):
        selected = annual.loc[annual["event_type"].eq(event_type)].sort_values("relative_year")
        x = selected["relative_year"].to_numpy(dtype=float)
        y = selected["estimate"].to_numpy(dtype=float)
        low = selected["ci_low"].to_numpy(dtype=float)
        high = selected["ci_high"].to_numpy(dtype=float)
        color = EVENT_COLORS[event_type]
        axis.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        axis.plot(x, y, color=color, marker="o", markersize=3.1, linewidth=1.25, label=EVENT_LABELS[event_type])
    axis.axhline(0.0, color="#777777", linewidth=0.65)
    axis.axvline(-0.5, color="#777777", linewidth=0.65, linestyle=(0, (2, 2)))
    axis.set_xticks(range(-3, 6))
    axis.set_xlabel("Relative year")
    axis.set_ylabel("Annual matched EVI anomaly")
    axis.set_title("Annual response", loc="left", fontweight="bold")
    axis.legend(loc="best")
    _panel_label(axis, "c")

    axis = axes[1, 1]
    contrast = contrast_summary.loc[
        contrast_summary["scope"].isin(REGION_LABELS)
        & contrast_summary["contrast"].isin(["early", "late"])
    ].copy()
    if "weighting" in contrast.columns:
        contrast = contrast.loc[contrast["weighting"].eq("event_area")].copy()
    regions = list(REGION_LABELS)
    base = np.arange(len(regions), dtype=float)
    offsets = {
        ("degradation", "early"): -0.24,
        ("degradation", "late"): -0.08,
        ("deforestation", "early"): 0.08,
        ("deforestation", "late"): 0.24,
    }
    markers = {"early": "o", "late": "s"}
    for event_type in ("degradation", "deforestation"):
        for period in ("early", "late"):
            selected = (
                contrast.loc[
                    contrast["event_type"].eq(event_type)
                    & contrast["contrast"].eq(period)
                ]
                .set_index("scope")
                .reindex(regions)
            )
            y = base + offsets[(event_type, period)]
            estimate = selected["estimate"].to_numpy(dtype=float)
            low = selected["ci_low"].to_numpy(dtype=float)
            high = selected["ci_high"].to_numpy(dtype=float)
            axis.errorbar(
                estimate,
                y,
                xerr=np.vstack([estimate - low, high - estimate]),
                fmt=markers[period],
                color=EVENT_COLORS[event_type],
                markerfacecolor="white" if period == "late" else EVENT_COLORS[event_type],
                markeredgewidth=0.8,
                markersize=3.4,
                linewidth=0.8,
                capsize=1.5,
            )
    axis.axvline(0.0, color="#777777", linewidth=0.65)
    axis.set_yticks(base, [REGION_LABELS[value] for value in regions])
    axis.invert_yaxis()
    axis.set_xlabel("Matched EVI anomaly (95% CI)")
    axis.set_title("Regional post-event contrasts", loc="left", fontweight="bold")
    axis.text(
        0.99,
        1.025,
        "filled, early; open, late",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )
    _panel_label(axis, "d")

    figure._stage09_heatmap_limits = (-limit, limit)
    return figure


def build_validation_figure(validation_summary, coverage_summary):
    required_validation = {
        "summary_type",
        "scope",
        "event_type",
        "relative_year",
        "calendar_month",
        "composite_slot",
        "estimate",
        "monthly_estimate",
        "sixteen_day_estimate",
    }
    required_coverage = {"event_type", "role", "calendar_month", "usable_pct"}
    if required_validation - set(validation_summary.columns):
        raise ValueError("Sixteen-day validation summary has an invalid schema.")
    if required_coverage - set(coverage_summary.columns):
        raise ValueError("Coverage summary has an invalid schema.")
    figure, axes = _new_grid()
    curves = validation_summary.loc[
        validation_summary["summary_type"].eq("composite_effect")
        & validation_summary["scope"].eq("Global")
        & validation_summary["relative_year"].eq(0)
    ]
    for axis, event_type, label in zip(
        axes[0], ("degradation", "deforestation"), ("a", "b")
    ):
        selected = curves.loc[curves["event_type"].eq(event_type)].sort_values("composite_slot")
        axis.plot(
            selected["composite_slot"],
            selected["estimate"],
            color=EVENT_COLORS[event_type],
            marker="o",
            markersize=2.3,
            linewidth=1.1,
        )
        axis.axhline(0.0, color="#777777", linewidth=0.65)
        axis.set_xlim(1, 23)
        axis.set_xticks([1, 5, 9, 13, 17, 21, 23])
        axis.set_xlabel("Calendar composite slot")
        axis.set_ylabel("Matched EVI anomaly")
        axis.set_title(f"{EVENT_LABELS[event_type]} event year", loc="left", fontweight="bold")
        _panel_label(axis, label)

    axis = axes[1, 0]
    if "summary_type" in coverage_summary.columns:
        coverage_summary = coverage_summary.loc[
            coverage_summary["summary_type"].eq("calendar_month")
        ].copy()
    line_styles = {"treatment": "-", "control": (0, (3, 2))}
    for event_type in ("degradation", "deforestation"):
        for role in ("treatment", "control"):
            selected = coverage_summary.loc[
                coverage_summary["event_type"].eq(event_type)
                & coverage_summary["role"].eq(role)
            ].sort_values("calendar_month")
            axis.plot(
                selected["calendar_month"],
                selected["usable_pct"],
                color=EVENT_COLORS[event_type],
                linestyle=line_styles[role],
                linewidth=1.1,
                label=f"{EVENT_LABELS[event_type]}, {role}",
            )
    axis.set_xticks(range(1, 13))
    axis.set_xlabel("Calendar month")
    axis.set_ylabel("Usable observations (%)")
    axis.set_title("Observation coverage", loc="left", fontweight="bold")
    axis.legend(loc="best", ncols=2, columnspacing=0.7, handlelength=1.6)
    _panel_label(axis, "c")

    axis = axes[1, 1]
    agreement = validation_summary.loc[
        validation_summary["summary_type"].eq("monthly_agreement")
        & validation_summary["scope"].eq("Global")
    ]
    for event_type in ("degradation", "deforestation"):
        selected = agreement.loc[agreement["event_type"].eq(event_type)]
        axis.scatter(
            selected["monthly_estimate"],
            selected["sixteen_day_estimate"],
            s=8,
            color=EVENT_COLORS[event_type],
            alpha=0.65,
            linewidth=0,
            label=EVENT_LABELS[event_type],
        )
    values = np.concatenate(
        [
            agreement["monthly_estimate"].to_numpy(dtype=float),
            agreement["sixteen_day_estimate"].to_numpy(dtype=float),
        ]
    )
    values = values[np.isfinite(values)]
    low, high = (float(values.min()), float(values.max())) if len(values) else (-0.1, 0.1)
    pad = max((high - low) * 0.05, 1e-4)
    axis.plot([low - pad, high + pad], [low - pad, high + pad], color="#555555", linewidth=0.8, linestyle=(0, (3, 2)))
    axis.set_xlim(low - pad, high + pad)
    axis.set_ylim(low - pad, high + pad)
    axis.set_xlabel("Monthly matched estimate")
    axis.set_ylabel("Reconstructed 16-day estimate")
    axis.set_title("Monthly agreement", loc="left", fontweight="bold")
    axis.legend(loc="best")
    _panel_label(axis, "d")
    return figure


def export_figure(figure, base_path):
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [base_path.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg", ".tiff")]
    figure.savefig(paths[0], dpi=300, facecolor="white")
    figure.savefig(paths[1], facecolor="white")
    figure.savefig(paths[2], facecolor="white")
    figure.savefig(paths[3], dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    return paths


def validate_figure_bundle(paths):
    paths = [Path(path) for path in paths]
    expected = {".png", ".pdf", ".svg", ".tiff"}
    suffixes = {path.suffix for path in paths}
    checks = {
        "formats": suffixes == expected,
        "files_exist": all(path.is_file() for path in paths),
        "files_nonempty": all(path.is_file() and path.stat().st_size > 100 for path in paths),
        "raster_nonblank": True,
        "svg_editable_text": False,
        "pdf_header": False,
    }
    for path in paths:
        if path.suffix in {".png", ".tiff"} and path.is_file():
            with Image.open(path) as image:
                variance = ImageStat.Stat(image.convert("RGB")).var
                checks["raster_nonblank"] &= bool(max(variance) > 0.0)
        elif path.suffix == ".svg" and path.is_file():
            text = path.read_text(encoding="utf-8")
            checks["svg_editable_text"] = "<text" in text and "font-family" in text
        elif path.suffix == ".pdf" and path.is_file():
            checks["pdf_header"] = path.read_bytes()[:4] == b"%PDF"
    return {"pass": bool(all(checks.values())), "checks": checks}
