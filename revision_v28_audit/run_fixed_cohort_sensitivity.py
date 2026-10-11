#!/usr/bin/env python3
"""Revision v2.8 audit: fixed-observation-cohort sensitivity.

This is an independent audit calculation.  It reads the frozen Stage 09
annual matched-effect panel and the frozen v2.7 common-support weights, and it
does not overwrite any upstream result.  A1 requires all six post/event years
(t0--t5) and no dynamic censoring through t5.  A2 requires t0, t3 and t5 and
uses the same no-censoring-through-t5 rule, so that it isolates the three
reported endpoint years while preserving one event cohort.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RUN = PROJECT / "runs/20260909-extended-sensitivities"
ANALYSIS = RUN / "analysis_current_full_history"
STAGE09 = RUN / "stage09_current_full_history"
PIPELINE = PROJECT / "02_analysis/integrated/TropicalRainforest_integrated_analysis"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from pipeline.manuscript_v07_statistics import common_block_bootstrap, contrast_on_column


OUT = HERE
KEY = ["treatment_grid_id", "event_year"]
EVENT_TYPES = ("degradation", "deforestation")
SEED = 20260725
REPS = 1000
BLOCK_DEGREES = 2.0
PRIMARY_T5_CONTRAST = -0.0188019903723987


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ess(w: pd.Series) -> float:
    x = pd.to_numeric(w, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if not len(x):
        return float("nan")
    return float(x.sum() ** 2 / np.square(x).sum())


def weighted_moments(x: pd.Series, w: pd.Series) -> tuple[float, float, float]:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    wv = pd.to_numeric(w, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xv) & np.isfinite(wv) & (wv > 0)
    xv, wv = xv[ok], wv[ok]
    if not len(xv):
        return (float("nan"), float("nan"), float("nan"))
    mu = float(np.dot(xv, wv) / wv.sum())
    sd = float(np.sqrt(np.dot((xv - mu) ** 2, wv) / wv.sum()))
    return mu, sd, float(wv.sum() ** 2 / np.square(wv).sum())


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = pd.read_parquet(ANALYSIS / "fixed_matched_event_window.parquet")
    support = pd.read_parquet(ANALYSIS / "common_support_weights.parquet")
    required_effects = set(KEY + [
        "event_type", "macroregion", "relative_year", "pair_effect_evi_mean",
        "treatment_lon", "treatment_lat", "censor_relative_year",
        "disturbance_share", "event_composition_purity", "baseline_undist_share",
    ])
    missing = required_effects - set(effects.columns)
    if missing:
        raise ValueError(f"Stage09 effect panel lacks columns: {sorted(missing)}")
    if effects.duplicated(KEY + ["relative_year"]).any():
        raise ValueError("Stage09 effect panel contains duplicate event-year-relative-year rows")
    if support.duplicated(KEY).any():
        raise ValueError("common-support table contains duplicate event keys")
    support_cols = KEY + [
        "standardized_event_weight", "event_type", "macroregion",
        "disturbance_share", "event_composition_purity", "baseline_undist_share",
    ]
    support = support[support_cols].copy()
    support["standardized_event_weight"] = pd.to_numeric(
        support["standardized_event_weight"], errors="coerce"
    )
    if not np.isfinite(support["standardized_event_weight"]).all() or (support["standardized_event_weight"] <= 0).any():
        raise ValueError("Support weights must be finite and positive")
    frame = effects.merge(
        support,
        on=KEY,
        how="inner",
        validate="many_to_one",
        suffixes=("", "_support"),
    )
    if len(frame) != len(effects):
        raise ValueError("Some Stage09 effect rows lack frozen common-support weights")
    # Keep the authoritative event-level fields from the support table.  Values
    # in the Stage09 window are checked below rather than silently reconciled.
    for col in ["event_type", "event_year", "macroregion", "disturbance_share", "event_composition_purity", "baseline_undist_share"]:
        alt = f"{col}_support"
        if alt in frame.columns:
            left = frame[col].astype(str)
            right = frame[alt].astype(str)
            if not (left == right).all():
                raise ValueError(f"Effect and support tables disagree for {col}")
            frame = frame.drop(columns=[alt])
    frame["pair_effect_evi_mean"] = pd.to_numeric(frame["pair_effect_evi_mean"], errors="coerce")
    frame["relative_year"] = pd.to_numeric(frame["relative_year"], errors="coerce").astype("Int64")
    frame["censor_relative_year"] = pd.to_numeric(frame["censor_relative_year"], errors="coerce")
    return frame, support


def complete_event_keys(frame: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Return event-level keys meeting fixed-cohort requirements."""
    # The Stage09 window has already dropped rows at and after a censor year.
    # Require explicit rows for every requested relative year and explicitly
    # exclude censor_relative_year <= 5 to make the no-censoring rule auditable.
    group = frame.groupby(KEY, sort=False)
    present = group["relative_year"].agg(lambda x: set(int(v) for v in x.dropna()))
    enough = present.map(lambda s: set(years).issubset(s))
    censor = group["censor_relative_year"].first()
    no_censor = censor.isna() | (censor > max(years))
    keys = pd.concat([enough.rename("has_all_years"), no_censor.rename("no_censor_through_t5")], axis=1)
    keys["eligible"] = keys["has_all_years"] & keys["no_censor_through_t5"]
    out = keys.reset_index()
    return out.loc[out["eligible"], KEY].copy()


def subset_cohort(frame: pd.DataFrame, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = complete_event_keys(frame, years)
    out = frame.merge(keys, on=KEY, how="inner", validate="many_to_many")
    out = out.loc[out["relative_year"].isin(years)].copy()
    # A finite annual response is required for every retained event-year row.
    if out["pair_effect_evi_mean"].isna().any():
        raise ValueError("Fixed cohort includes non-finite annual effects")
    if out.duplicated(KEY + ["relative_year"]).any():
        raise ValueError("Fixed cohort contains duplicate event-year-relative-year rows")
    return out, keys


def estimate(cohort: pd.DataFrame, analysis_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = common_block_bootstrap(
        cohort,
        "pair_effect_evi_mean",
        ("event_type", "relative_year"),
        weight_column="standardized_event_weight",
        weighting="event_area",
        block_degrees=BLOCK_DEGREES,
        replicates=REPS,
        seed=SEED,
        minimum_valid_replicates=950,
    )
    group = bundle.summary.copy()
    group["analysis"] = analysis_id
    group["estimate_type"] = "event_type"
    direct = contrast_on_column(
        bundle, "event_type", "deforestation", "degradation",
        by_columns=("relative_year",), label="deforestation_minus_degradation",
    ).copy()
    direct["analysis"] = analysis_id
    direct["event_type"] = "deforestation_minus_degradation"
    direct["estimate_type"] = "direct_contrast"
    block_counts = cohort.assign(
        spatial_block=(
            np.floor((cohort["treatment_lon"] + 180.0) / BLOCK_DEGREES).astype(int).astype(str)
            + ":"
            + np.floor((cohort["treatment_lat"] + 90.0) / BLOCK_DEGREES).astype(int).astype(str)
        )
    ).groupby("relative_year")["spatial_block"].nunique()
    direct["spatial_block_count"] = direct["relative_year"].map(block_counts).astype(int)
    direct["bootstrap_block_universe_count"] = int(len(bundle.block_ids))
    # Make a common, explicit schema; all n/ESS values are event-level for each
    # year and the intervals are percentile intervals from the shared block draw.
    group = group.rename(columns={"treatment_count": "n_events", "weight_sum": "weight_sum", "estimate": "estimate"})
    group["ess"] = cohort.groupby(["event_type", "relative_year"], sort=True)["standardized_event_weight"].apply(ess).to_numpy()
    direct["n_events"] = np.nan
    direct["ess"] = np.nan
    direct["weight_sum"] = np.nan
    cols = [
        "analysis", "relative_year", "event_type", "estimate_type", "estimate",
        "ci_low", "ci_high", "bootstrap_sd", "n_events", "ess", "weight_sum",
        "spatial_block_count", "bootstrap_block_universe_count", "bootstrap_requested",
        "bootstrap_valid", "bootstrap_seed", "block_degrees",
    ]
    return pd.concat([group[cols], direct[cols]], ignore_index=True).sort_values(
        ["analysis", "relative_year", "estimate_type", "event_type"]
    ).reset_index(drop=True), bundle.replicates


def estimate_regional(cohort: pd.DataFrame, analysis_id: str) -> pd.DataFrame:
    """Regional direct contrasts using the same 2-degree block bootstrap."""
    bundle = common_block_bootstrap(
        cohort,
        "pair_effect_evi_mean",
        ("macroregion", "event_type", "relative_year"),
        weight_column="standardized_event_weight",
        weighting="event_area",
        block_degrees=BLOCK_DEGREES,
        replicates=REPS,
        seed=SEED,
        minimum_valid_replicates=950,
    )
    direct = contrast_on_column(
        bundle, "event_type", "deforestation", "degradation",
        by_columns=("macroregion", "relative_year"), label="deforestation_minus_degradation",
    ).copy()
    direct["analysis"] = analysis_id
    direct["estimate_type"] = "regional_direct_contrast"
    direct["event_type"] = "deforestation_minus_degradation"
    piv = bundle.summary.pivot_table(
        index=["macroregion", "relative_year"], columns="event_type", values=["estimate", "treatment_count"], aggfunc="first"
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    direct = direct.merge(piv, on=["macroregion", "relative_year"], how="left", validate="one_to_one")
    direct = direct.rename(columns={"treatment_count_deforestation": "deforestation_n", "treatment_count_degradation": "degradation_n"})
    # The bootstrap implementation reports the number of blocks represented in
    # each regional/year group; add the universe count explicitly for audit.
    block_frame = cohort.assign(
        spatial_block=(
            np.floor((cohort["treatment_lon"] + 180.0) / BLOCK_DEGREES).astype(int).astype(str)
            + ":"
            + np.floor((cohort["treatment_lat"] + 90.0) / BLOCK_DEGREES).astype(int).astype(str)
        )
    )
    block_counts = block_frame.groupby(["macroregion", "relative_year"])["spatial_block"].nunique()
    direct["spatial_block_count"] = [int(block_counts.loc[(r.macroregion, r.relative_year)]) for r in direct.itertuples()]
    direct["bootstrap_block_universe_count"] = int(len(bundle.block_ids))
    return direct[[
        "analysis", "macroregion", "relative_year", "event_type", "estimate_type",
        "estimate", "ci_low", "ci_high", "bootstrap_sd", "deforestation_n", "degradation_n",
        "spatial_block_count", "bootstrap_block_universe_count", "bootstrap_requested",
        "bootstrap_valid", "bootstrap_seed", "block_degrees",
    ]].sort_values(["analysis", "macroregion", "relative_year"]).reset_index(drop=True)


def balance(cohort: pd.DataFrame, analysis_id: str) -> pd.DataFrame:
    variables = [
        ("disturbance_share", "D+F total disturbance share"),
        ("event_composition_purity", "purity P"),
        ("baseline_undist_share", "baseline undisturbed forest share"),
        ("event_year", "event year"),
    ]
    rows: list[dict[str, object]] = []
    for rel in sorted(cohort["relative_year"].unique()):
        per_type: dict[str, dict[str, float]] = {}
        for typ in EVENT_TYPES:
            g = cohort.loc[(cohort["relative_year"] == rel) & (cohort["event_type"] == typ)].copy()
            per_type[typ] = {"n": int(g[KEY].drop_duplicates().shape[0]), "ess": ess(g["standardized_event_weight"])}
            for col, label in variables:
                mu, sd, e = weighted_moments(g[col], g["standardized_event_weight"])
                per_type[typ][f"{col}_mean"] = mu
                per_type[typ][f"{col}_sd"] = sd
                per_type[typ][f"{col}_ess"] = e
        for col, label in variables:
            d = per_type["deforestation"]
            g = per_type["degradation"]
            pooled = float(np.sqrt((d[f"{col}_sd"] ** 2 + g[f"{col}_sd"] ** 2) / 2.0))
            smd = (d[f"{col}_mean"] - g[f"{col}_mean"]) / pooled if pooled > 0 else np.nan
            rows.append({
                "analysis": analysis_id,
                "relative_year": int(rel),
                "variable": col,
                "variable_label": label,
                "deforestation_n": d["n"],
                "degradation_n": g["n"],
                "deforestation_ess": d["ess"],
                "degradation_ess": g["ess"],
                "deforestation_weighted_mean": d[f"{col}_mean"],
                "degradation_weighted_mean": g[f"{col}_mean"],
                "smd_deforestation_minus_degradation": smd,
                "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
            })
    return pd.DataFrame(rows)


def make_figure(results: pd.DataFrame) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.27, wspace=0.27)
    colors = {"degradation": "#2F6F9F", "deforestation": "#C65A3A"}
    labels = {"degradation": "Degradation-dominated", "deforestation": "Deforestation-dominated"}
    for ax, analysis, title in zip(axes, ["A1", "A2"], ["A1: t0–t5 complete cohort", "A2: t0/t3/t5 complete cohort"]):
        sub = results.loc[(results["analysis"] == analysis) & (results["estimate_type"] == "event_type")]
        for typ in EVENT_TYPES:
            g = sub.loc[sub["event_type"] == typ].sort_values("relative_year")
            if g.empty:
                continue
            ax.errorbar(g["relative_year"], g["estimate"], yerr=[g["estimate"] - g["ci_low"], g["ci_high"] - g["estimate"]], marker="o", ms=3.2, lw=1.2, capsize=2, color=colors[typ], label=labels[typ])
        ctr = results.loc[(results["analysis"] == analysis) & (results["estimate_type"] == "direct_contrast")].sort_values("relative_year")
        ax.axhline(0, lw=0.6, color="#444444")
        if not ctr.empty:
            ax.errorbar(ctr["relative_year"], ctr["estimate"], yerr=[ctr["estimate"] - ctr["ci_low"], ctr["ci_high"] - ctr["estimate"]], marker="D", ms=2.8, lw=0.9, ls="--", color="#252525", capsize=2, label="Direct contrast")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Relative year", labelpad=10)
        ax.set_ylabel("Annualized matched EVI effect")
        ax.set_xticks(sorted(sub["relative_year"].unique()))
        ax.grid(axis="y", color="#dddddd", lw=0.5)
        ax.spines["top"].set_visible(False)
        if analysis == "A2":
            ax.set_xticks([0, 3, 5])
    handles, labs = axes[0].get_legend_handles_labels()
    # The direct-contrast line is documented in the caption/report; keep the
    # figure legend limited to the two event types for legibility.
    fig.legend(handles, labs, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.canvas.draw()
    fig.savefig(OUT / "figures/fixed_cohort_trajectory.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "figures/fixed_cohort_trajectory.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures/fixed_cohort_trajectory.svg", bbox_inches="tight")
    fig.savefig(OUT / "figures/fixed_cohort_trajectory.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(all_results: pd.DataFrame, counts: pd.DataFrame, frame: pd.DataFrame, regional: pd.DataFrame) -> None:
    a1 = all_results.loc[(all_results.analysis == "A1") & (all_results.estimate_type == "direct_contrast")]
    a2 = all_results.loc[(all_results.analysis == "A2") & (all_results.estimate_type == "direct_contrast")]
    def line(df, rel):
        r = df.loc[df.relative_year == rel].iloc[0]
        return f"{float(r.estimate):.8f} (95% CI {float(r.ci_low):.8f}, {float(r.ci_high):.8f})"
    def nline(analysis):
        c = counts.loc[counts.analysis.eq(analysis)].set_index("event_type")
        return f"deforestation={int(c.loc['deforestation','n_events'])}, degradation={int(c.loc['degradation','n_events'])}"
    t5_a1 = float(a1.loc[a1.relative_year == 5, "estimate"].iloc[0])
    diff = t5_a1 - PRIMARY_T5_CONTRAST
    a1_change = t5_a1 - float(a1.loc[a1.relative_year == 0, "estimate"].iloc[0])
    regional_lines = []
    for analysis_id in ("A1", "A2"):
        rsub = regional.loc[(regional.analysis == analysis_id) & (regional.relative_year == 5)]
        parts = []
        for row in rsub.itertuples(index=False):
            parts.append(f"{row.macroregion.replace('_proxy','')}={row.estimate:.6f} [{row.ci_low:.6f},{row.ci_high:.6f}]")
        regional_lines.append(f"- {analysis_id} regional t5 contrasts: " + "; ".join(parts) + ".")
    text = f"""# Fixed-observation-cohort sensitivity (revision v2.8 audit)

## Definition

This independent calculation uses `runs/20260909-extended-sensitivities/analysis_current_full_history/fixed_matched_event_window.parquet` and its frozen `common_support_weights.parquet`. The primary analysis event-area standardization weights are retained without refitting. A1 requires a finite Stage09 annualized matched effect at every relative year t0–t5 and no dynamic censoring at or before t5. A2 applies the same no-censoring-through-t5 rule but requires finite effects only at t0, t3 and t5. Both cohorts are subsets of the 10,970 common-support events.

## Results

- A1 event counts: {nline('A1')} (2,469/10,970 = 22.5% of common-support events); A2 event counts: {nline('A2')} (3,468/10,970 = 31.6%).
- A1 direct contrasts: t0 {line(a1,0)}; t1 {line(a1,1)}; t2 {line(a1,2)}; t3 {line(a1,3)}; t4 {line(a1,4)}; t5 {line(a1,5)}.
- A2 direct contrasts: t0 {line(a2,0)}; t3 {line(a2,3)}; t5 {line(a2,5)}.
- Regional t5 contrasts (95% CI):
{chr(10).join(regional_lines)}
- A1 t5 contrast differs from the v2.7 primary estimate ({PRIMARY_T5_CONTRAST:.8f}) by {diff:+.8f} EVI units.

The fixed cohort therefore tests the same-event composition directly. In A1, the direct contrast changes from -0.008654 at t0 to -0.019528 at t5 (t5 minus t0 = {a1_change:+.8f}), so the negative type difference still expands within the same selected events. This is a conditional sensitivity result, not a replacement for the year-specific primary trajectory. The cohort is substantially smaller than the primary t0 cohort, so the analysis does not remove outcome-availability selection; it quantifies the direction of the complete-observation selection. Frozen weights leave appreciable complete-cohort imbalance (A1 purity SMD -0.403 and baseline-forest SMD -0.342), which should be considered alongside the year-specific restandardization diagnostic.

## Inference and balance

Point estimates use frozen standardized event-area weights. Confidence intervals use 1,000 percentile 2° spatial-block bootstrap replicates (seed {SEED}), with one shared block draw matrix across event types and relative years. `fixed_cohort_balance.csv` reports weighted means, ESS and standardized mean differences for disturbance share, purity P, baseline undisturbed forest share and event year. This file is diagnostic and does not alter the primary cohort or weights.

## Provenance

- Source effects SHA256: `{sha256(ANALYSIS / 'fixed_matched_event_window.parquet')}`
- Source weights SHA256: `{sha256(ANALYSIS / 'common_support_weights.parquet')}`
- Source matching table SHA256: `{sha256(ANALYSIS / 'matched_treatments.parquet')}`
- Calculation script: `{Path(__file__).name}` (stored in this audit directory).

No original code, data, v2.7 manuscript or frozen result was overwritten.
"""
    (OUT / "fixed_cohort_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    frame, support = load_panel()
    outputs = []
    regional_outputs = []
    balances = []
    counts = []
    for analysis_id, years in [("A1", [0, 1, 2, 3, 4, 5]), ("A2", [0, 3, 5])]:
        cohort, keys = subset_cohort(frame, years)
        result, reps = estimate(cohort, analysis_id)
        outputs.append(result)
        regional_outputs.append(estimate_regional(cohort, analysis_id))
        balances.append(balance(cohort, analysis_id))
        event_level = cohort.drop_duplicates(KEY).copy()
        c = event_level.groupby("event_type", as_index=False).agg(n_events=("treatment_grid_id", "nunique"), weight_sum=("standardized_event_weight", "sum"))
        c["analysis"] = analysis_id
        c["ess"] = event_level.groupby("event_type")["standardized_event_weight"].apply(ess).to_numpy()
        c["years_required"] = ",".join(str(x) for x in years)
        c["no_censor_through_t5"] = True
        c["cohort_event_count_total"] = int(len(keys))
        counts.append(c)
    all_results = pd.concat(outputs, ignore_index=True)
    all_bal = pd.concat(balances, ignore_index=True)
    all_counts = pd.concat(counts, ignore_index=True)
    all_results.to_csv(OUT / "fixed_cohort_sensitivity.csv", index=False)
    all_bal.to_csv(OUT / "fixed_cohort_balance.csv", index=False)
    all_counts.to_csv(OUT / "fixed_cohort_counts.csv", index=False)
    regional = pd.concat(regional_outputs, ignore_index=True)
    regional.to_csv(OUT / "fixed_cohort_regional.csv", index=False)
    make_figure(all_results)
    write_report(all_results, all_counts, frame, regional)
    provenance = pd.DataFrame([
        {"role": "annual_effect_panel", "path": str(ANALYSIS / "fixed_matched_event_window.parquet"), "sha256": sha256(ANALYSIS / "fixed_matched_event_window.parquet")},
        {"role": "frozen_common_support_weights", "path": str(ANALYSIS / "common_support_weights.parquet"), "sha256": sha256(ANALYSIS / "common_support_weights.parquet")},
        {"role": "matched_treatment_metadata", "path": str(ANALYSIS / "matched_treatments.parquet"), "sha256": sha256(ANALYSIS / "matched_treatments.parquet")},
        {"role": "existing_bootstrap_implementation", "path": str(PIPELINE / "pipeline/manuscript_v07_statistics.py"), "sha256": sha256(PIPELINE / "pipeline/manuscript_v07_statistics.py")},
    ])
    provenance.to_csv(OUT / "fixed_cohort_provenance.csv", index=False)
    (OUT / "fixed_cohort_manifest.json").write_text(json.dumps({
        "analysis": "fixed_observation_cohort_sensitivity",
        "seed": SEED,
        "replicates": REPS,
        "block_degrees": BLOCK_DEGREES,
        "A1_years": [0, 1, 2, 3, 4, 5],
        "A2_years": [0, 3, 5],
        "no_censor_rule": "censor_relative_year is NA or >5",
        "weights": "frozen common-support standardized_event_weight",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
