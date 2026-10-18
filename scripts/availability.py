"""Audit t0 monthly outcome availability for the frozen common-support cohort.

This is a read-only diagnostic.  It reconstructs the Stage 09 monthly gate from
the preserved Stage 08 monthly panel and the preserved matched-link table.  It
does not modify any upstream result or manuscript.

The event-level primary reason is deliberately hierarchical:
  available -> no failure;
  treated_and_control_gate -> both treatment and control gates contribute;
  treated_gate -> fewer than nine treatment-usable months;
  control_support_gate -> treatment has >=9 usable months but <9 months retain
                          at least 2/3 of the original control-link weight;
  discordant_monthly_gates -> both individual gates have >=9 months, but the
                              months do not overlap sufficiently to produce 9
                              matched monthly effects;
  other -> a residual integrity category (expected to be zero).

All failure flags are also retained as non-exclusive event-level counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/primary"
CS_PATH = RUN / "event_level/common_support_weights.parquet"
LINK_PATH = RUN / "subannual/matched_links.parquet"
MONTHLY_ROOT = RUN / "subannual/parts_monthly"
OUT = ROOT / "results/sensitivity"

EVENT_KEYS = ["treatment_grid_id", "event_year"]
MONTH_KEYS = EVENT_KEYS + ["relative_year", "calendar_month"]


def read_t0_months() -> pd.DataFrame:
    """Read only relative-year t0 monthly rows from each frozen Stage 08 part."""
    frames: list[pd.DataFrame] = []
    columns = [
        "grid_id", "event_year", "role", "macroregion", "calendar_month",
        "relative_year", "evi_mean", "baseline_evi", "evi_anomaly",
        "baseline_year_count", "valid_composite_count",
        "expected_composite_count", "coverage_fraction",
        "mean_valid_pixel_fraction",
    ]
    for part in sorted(MONTHLY_ROOT.glob("event_year=*/part.parquet")):
        frame = pd.read_parquet(part, columns=columns, filters=[("relative_year", "=", 0)])
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No t0 monthly parts under {MONTHLY_ROOT}")
    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(["grid_id", "event_year", "role", "calendar_month"]).any():
        raise AssertionError("Duplicate Stage 08 role-month keys")
    return out


def gate_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact Stage 09 monthly_usable_mask as decomposed flags."""
    numeric = frame.copy()
    for col in [
        "evi_mean", "baseline_evi", "evi_anomaly", "baseline_year_count",
        "valid_composite_count", "coverage_fraction", "mean_valid_pixel_fraction",
    ]:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    numeric["fail_evi"] = ~np.isfinite(numeric[["evi_mean", "evi_anomaly"]]).all(axis=1)
    numeric["fail_baseline"] = (
        ~np.isfinite(numeric["baseline_evi"])
        | numeric["baseline_year_count"].lt(2)
    )
    # Stage 09 has two coverage-related gates: at least one valid composite and
    # at least 50% of expected composites represented in the month.
    numeric["fail_composite"] = (
        numeric["valid_composite_count"].lt(1)
        | numeric["coverage_fraction"].lt(0.50)
        | ~np.isfinite(numeric[["valid_composite_count", "coverage_fraction"]]).all(axis=1)
    )
    numeric["fail_pixel"] = (
        numeric["mean_valid_pixel_fraction"].lt(0.50)
        | ~np.isfinite(numeric["mean_valid_pixel_fraction"])
    )
    numeric["usable"] = ~numeric[["fail_evi", "fail_baseline", "fail_composite", "fail_pixel"]].any(axis=1)
    return numeric


def reconstruct() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    common = pd.read_parquet(CS_PATH)
    common = common.loc[:, [
        "treatment_grid_id", "event_year", "event_type", "macroregion",
        "disturbance_share", "event_composition_purity", "baseline_undist_share",
        "event_area_m2", "standardized_event_weight",
    ]].copy()
    if len(common) != 10970:
        raise AssertionError(f"Expected 10,970 common-support events, got {len(common)}")
    if common.duplicated(EVENT_KEYS).any():
        raise AssertionError("Common-support event keys are not unique")

    links = pd.read_parquet(LINK_PATH)
    link_keep = EVENT_KEYS + ["control_grid_id", "within_set_weight", "covariate_distance"]
    links = links.loc[:, [c for c in link_keep if c in links.columns]].copy()
    if links.duplicated(EVENT_KEYS + ["control_grid_id"]).any():
        raise AssertionError("Duplicate treatment-control links")
    common_links = common[EVENT_KEYS].merge(links, on=EVENT_KEYS, how="left", validate="one_to_many")
    if common_links["control_grid_id"].isna().any():
        raise AssertionError("Common-support events missing matched links")

    months = gate_flags(read_t0_months())
    treatment = months.loc[months["role"].eq("treatment")].copy()
    controls = months.loc[months["role"].eq("control")].copy()
    tr = treatment.rename(columns={"grid_id": "treatment_grid_id"})
    tr = tr.loc[:, EVENT_KEYS + ["calendar_month", "usable", "fail_evi", "fail_baseline", "fail_composite", "fail_pixel"]]
    tr = tr.rename(columns={
        "usable": "treated_month_usable", "fail_evi": "treated_fail_evi",
        "fail_baseline": "treated_fail_baseline", "fail_composite": "treated_fail_composite",
        "fail_pixel": "treated_fail_pixel",
    })
    # The control gate is based on the original within-set weights, summed only
    # over individually usable control records, exactly as Stage 09 does.
    ctrl = controls.rename(columns={"grid_id": "control_grid_id"})
    ctrl = common_links.merge(ctrl, on=["control_grid_id", "event_year"], how="left", validate="many_to_many")
    ctrl["usable_weight"] = ctrl["within_set_weight"].where(ctrl["usable"].fillna(False), 0.0)
    ctrl_month = ctrl.groupby(EVENT_KEYS + ["calendar_month"], as_index=False).agg(
        retained_control_weight=("usable_weight", "sum"),
        n_linked_controls=("control_grid_id", "nunique"),
        n_usable_controls=("usable", lambda s: int(s.fillna(False).sum())),
    )
    ctrl_month["control_support_ok"] = ctrl_month["retained_control_weight"].ge(2.0 / 3.0 - 1e-12)
    ctrl_month["control_support_fail"] = ~ctrl_month["control_support_ok"]

    detail = common.merge(tr, on=EVENT_KEYS, how="left", validate="one_to_many")
    detail = detail.merge(ctrl_month, on=EVENT_KEYS + ["calendar_month"], how="left", validate="one_to_one")
    detail["treated_month_usable"] = detail["treated_month_usable"].fillna(False)
    detail["control_support_ok"] = detail["control_support_ok"].fillna(False)
    detail["matched_month_usable"] = detail["treated_month_usable"] & detail["control_support_ok"]
    detail["treated_failure_any"] = ~detail["treated_month_usable"]
    detail["control_failure_any"] = ~detail["control_support_ok"]

    # Aggregate all twelve calendar months for each event.  Missing role-month
    # rows are treated as a failed gate and are retained in diagnostics.
    agg = detail.groupby(EVENT_KEYS, as_index=False).agg(
        event_type=("event_type", "first"), macroregion=("macroregion", "first"),
        disturbance_share=("disturbance_share", "first"),
        event_composition_purity=("event_composition_purity", "first"),
        baseline_undist_share=("baseline_undist_share", "first"),
        event_area_m2=("event_area_m2", "first"), standardized_event_weight=("standardized_event_weight", "first"),
        treated_usable_months=("treated_month_usable", "sum"),
        matched_usable_months=("matched_month_usable", "sum"),
        control_support_ok_months=("control_support_ok", "sum"),
        treated_fail_evi_months=("treated_fail_evi", "sum"),
        treated_fail_baseline_months=("treated_fail_baseline", "sum"),
        treated_fail_composite_months=("treated_fail_composite", "sum"),
        treated_fail_pixel_months=("treated_fail_pixel", "sum"),
        control_support_fail_months=("control_support_fail", "sum"),
    )
    agg["outcome_available_t0"] = agg["matched_usable_months"].ge(9)
    agg["treated_gate_failure"] = agg["treated_usable_months"].lt(9)
    agg["control_support_failure"] = agg["control_support_ok_months"].lt(9)
    agg["discordant_monthly_gates"] = (
        ~agg["outcome_available_t0"]
        & ~agg["treated_gate_failure"]
        & ~agg["control_support_failure"]
    )
    agg["nonexclusive_treated_evi_failure"] = agg["treated_fail_evi_months"].gt(0)
    agg["nonexclusive_treated_baseline_failure"] = agg["treated_fail_baseline_months"].gt(0)
    agg["nonexclusive_treated_composite_failure"] = agg["treated_fail_composite_months"].gt(0)
    agg["nonexclusive_treated_pixel_failure"] = agg["treated_fail_pixel_months"].gt(0)
    agg["nonexclusive_control_support_failure"] = agg["control_support_fail_months"].gt(0)
    agg["primary_reason"] = np.select(
        [
            agg["outcome_available_t0"],
            agg["treated_gate_failure"] & agg["control_support_failure"],
            agg["treated_gate_failure"],
            agg["control_support_failure"],
            agg["discordant_monthly_gates"],
        ],
        ["available", "treated_and_control_gate", "treated_gate", "control_support_gate", "discordant_monthly_gates"],
        default="other",
    )
    # Add only pre-event/fixed-matching diagnostics available in the preserved
    # treatment table. These are used for the missingness model, not the event
    # decomposition itself.
    mt_path = RUN / "event_level/matched_treatments.parquet"
    if mt_path.exists():
        mt = pd.read_parquet(mt_path)
        mt = mt.rename(columns={"grid_id": "treatment_grid_id"})
        keep = [
            "treatment_grid_id", "event_year", "baseline_evi_mean", "baseline_evi_min",
            "baseline_evi_max", "baseline_evi_amp", "baseline_evi_sd", "pretrend_evi_mean",
            "log_analysis_area",
        ]
        keep = [c for c in keep if c in mt.columns]
        agg = agg.merge(mt[keep].drop_duplicates(EVENT_KEYS), on=EVENT_KEYS, how="left", validate="one_to_one")
    # Matching support diagnostics are summarized over fixed links and are
    # determined before t0 outcome availability.
    link_diag = links.groupby(EVENT_KEYS, as_index=False).agg(
        n_matched_controls=("control_grid_id", "nunique"),
        max_covariate_distance=("covariate_distance", "max") if "covariate_distance" in links.columns else ("within_set_weight", "max"),
    ) if "covariate_distance" in links.columns else links.groupby(EVENT_KEYS, as_index=False).agg(n_matched_controls=("control_grid_id", "nunique"))
    agg = agg.merge(link_diag, on=EVENT_KEYS, how="left", validate="one_to_one")
    # Internal integrity checks reproduce the manuscript counts exactly.
    expected = {"deforestation": 2769, "degradation": 3812}
    got = agg.loc[agg["outcome_available_t0"]].groupby("event_type").size().to_dict()
    if got != expected:
        raise AssertionError(f"t0 availability does not reproduce expected counts: {got}")
    if len(agg) != 10970 or agg["outcome_available_t0"].sum() != 6581:
        raise AssertionError("Common-support to t0 cohort count mismatch")
    return agg, detail, {
        "common_support_n": int(len(agg)), "t0_available_n": int(agg["outcome_available_t0"].sum()),
        "t0_missing_n": int((~agg["outcome_available_t0"]).sum()),
        "treated_month_rows": int(len(treatment)), "control_month_rows": int(len(controls)),
    }


def summarize(agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = agg.loc[~agg["outcome_available_t0"]].copy()
    # Non-exclusive event-level decomposition, with the primary reason as an
    # explicitly declared mutually exclusive classification.
    flag_cols = [
        "nonexclusive_treated_evi_failure", "nonexclusive_treated_baseline_failure",
        "nonexclusive_treated_composite_failure", "nonexclusive_treated_pixel_failure",
        "nonexclusive_control_support_failure",
    ]
    rows = []
    for typ, g in agg.groupby("event_type", sort=True):
        for status, h in [("all_common_support", g), ("t0_available", g[g.outcome_available_t0]), ("t0_missing", g[~g.outcome_available_t0])]:
            row = {"scope": "overall", "event_type": typ, "status": status, "n_events": len(h),
                   "availability_rate": float(h.outcome_available_t0.mean()) if len(h) else np.nan,
                   "mean_treated_usable_months": h.treated_usable_months.mean() if len(h) else np.nan,
                   "mean_matched_usable_months": h.matched_usable_months.mean() if len(h) else np.nan,
                   "primary_reason": ""}
            for col in flag_cols:
                row[col] = int(h[col].sum())
            rows.append(row)
    for (typ, reg, yr), g in agg.groupby(["event_type", "macroregion", "event_year"], sort=True):
        h = g
        row = {"scope": "event_type_region_year", "event_type": typ, "macroregion": reg, "event_year": yr,
               "status": "all_common_support", "n_events": len(h), "n_available_t0": int(h.outcome_available_t0.sum()),
               "n_missing_t0": int((~h.outcome_available_t0).sum()), "availability_rate": float(h.outcome_available_t0.mean()),
               "mean_treated_usable_months": h.treated_usable_months.mean(), "mean_matched_usable_months": h.matched_usable_months.mean()}
        for col in flag_cols:
            row[col] = int(h[col].sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    return missing, summary


def logistic_summary(agg: pd.DataFrame) -> str:
    """Fit an unregularized IRLS logistic diagnostic with convergence checks."""
    d = agg.copy()
    d["available"] = d["outcome_available_t0"].astype(int)
    predictors = [
        "event_type_bin", "event_year_c", "disturbance_share",
        "event_composition_purity", "baseline_undist_share", "baseline_evi_mean",
        "pretrend_evi_mean",
        "log_analysis_area", "n_matched_controls", "max_covariate_distance",
    ]
    d["event_type_bin"] = d["event_type"].eq("deforestation").astype(float)
    d["event_year_c"] = d["event_year"] - d["event_year"].mean()
    # Add region one-hot indicators, with Amazon as the reference category.
    regions = sorted(d["macroregion"].dropna().unique())
    for region in regions[1:]:
        d[f"region_{region}"] = d["macroregion"].eq(region).astype(float)
        predictors.append(f"region_{region}")
    predictors = [p for p in predictors if p in d.columns]
    work = d[["available"] + predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 100 or work["available"].nunique() < 2:
        return "OBSERVATION-MODEL NOT RELIABLE: insufficient complete cases or outcome variation.\n"
    y = work["available"].to_numpy(float)
    continuous = [p for p in predictors if p not in {"event_type_bin", *[x for x in predictors if x.startswith("region_")]}]
    X = work[predictors].to_numpy(float, copy=True)
    # Standardize only continuous predictors for numerical conditioning; binary
    # indicators remain on their original scale.
    scales = {}
    for j, p in enumerate(predictors):
        if p in continuous:
            mu = float(X[:, j].mean()); sd = float(X[:, j].std(ddof=0))
            scales[p] = (mu, sd)
            if sd > 0:
                X[:, j] = (X[:, j] - mu) / sd
    X = np.column_stack([np.ones(len(X)), X])
    names = ["intercept"] + predictors
    beta = np.zeros(X.shape[1], dtype=float)
    converged = False
    for _ in range(100):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(p * (1.0 - p), 1e-8)
        z = eta + (y - p) / W
        xtw = X.T * W
        try:
            new = np.linalg.solve(xtw @ X, xtw @ z)
        except np.linalg.LinAlgError:
            new = np.linalg.pinv(xtw @ X) @ (xtw @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new; converged = True; break
        beta = new
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    W = np.maximum(p * (1.0 - p), 1e-8)
    fisher = (X.T * W) @ X
    try:
        cov = np.linalg.inv(fisher)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher); se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    cond = float(np.linalg.cond(fisher))
    out = [
        "t0 outcome-availability logistic diagnostic (NumPy IRLS)",
        "Outcome: available_t0 = 1 when >=9 Stage 09 matched monthly effects are usable.",
        "This is a missingness diagnostic, not a causal model; no post-t0 outcome or censoring variable is used.",
        f"Complete cases: {len(work):,}/{len(d):,}; event-type variation: {work['available'].value_counts().to_dict()}",
        f"Converged: {converged}; Fisher information condition number: {cond:.3e}",
        "Continuous predictors are z-standardized for numerical fitting; binary indicators are unscaled.",
        "Reference region: Amazon_proxy. Odds ratios below correspond to one SD for continuous predictors.",
        "",
        "term\tcoefficient\tSE\todds_ratio\t95%_low\t95%_high",
    ]
    for name, b, s in zip(names, beta, se):
        lo, hi = b - 1.96 * s, b + 1.96 * s
        out.append(f"{name}\t{b:.8g}\t{s:.8g}\t{np.exp(np.clip(b, -50, 50)):.8g}\t{np.exp(np.clip(lo, -50, 50)):.8g}\t{np.exp(np.clip(hi, -50, 50)):.8g}")
    return "\n".join(out) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    agg, detail, meta = reconstruct()
    stage09_t0 = pd.read_parquet(
        RUN / "outcomes/monthly_matched_effects.parquet",
        filters=[("relative_year", "=", 0)],
    ).groupby(EVENT_KEYS).size().rename("stage09_matched_months").reset_index()
    check = agg.merge(stage09_t0, on=EVENT_KEYS, how="left", validate="one_to_one")
    check["stage09_matched_months"] = check["stage09_matched_months"].fillna(0).astype(int)
    if not np.array_equal(check["matched_usable_months"].to_numpy(int), check["stage09_matched_months"].to_numpy(int)):
        raise AssertionError("Reconstructed matched-month counts differ from Stage 09 t0 rows")
    meta["stage09_t0_month_count_exact_match"] = True
    missing, summary = summarize(agg)
    # Requested decomposition is event-level; retain month counts and all flags
    # so the 4,389-event total can be independently audited.
    decomposition_cols = [
        "treatment_grid_id", "event_year", "event_type", "macroregion", "event_area_m2",
        "disturbance_share", "event_composition_purity", "baseline_undist_share", "standardized_event_weight",
        "treated_usable_months", "matched_usable_months", "control_support_ok_months",
        "treated_fail_evi_months", "treated_fail_baseline_months", "treated_fail_composite_months",
        "treated_fail_pixel_months", "control_support_fail_months", "outcome_available_t0",
        "treated_gate_failure", "control_support_failure", "primary_reason",
        "nonexclusive_treated_evi_failure", "nonexclusive_treated_baseline_failure",
        "nonexclusive_treated_composite_failure", "nonexclusive_treated_pixel_failure",
        "nonexclusive_control_support_failure",
    ]
    missing.loc[:, decomposition_cols].sort_values(EVENT_KEYS).to_csv(
        OUT / "t0_month_availability_decomposition.csv", index=False
    )
    summary.to_csv(OUT / "outcome_availability_by_type_region_year.csv", index=False)
    (OUT / "availability_model_summary.txt").write_text(logistic_summary(agg), encoding="utf-8")
    report = {
        **meta,
        "source_files": {
            "common_support": CS_PATH.relative_to(ROOT).as_posix(),
            "matched_links": LINK_PATH.relative_to(ROOT).as_posix(),
            "monthly_stage08_glob": (MONTHLY_ROOT / "event_year=*/part.parquet").relative_to(ROOT).as_posix(),
            "gate_implementation": "src/pipeline/subannual_response.py:monthly_usable_mask and build_monthly_matched_effects",
        },
        "gate": {
            "finite": ["evi_mean", "baseline_evi", "evi_anomaly"],
            "baseline_year_count_min": 2,
            "valid_composite_count_min": 1,
            "coverage_fraction_min": 0.5,
            "mean_valid_pixel_fraction_min": 0.5,
            "control_weight_min": 2.0 / 3.0,
            "annual_min_usable_months": 9,
        },
        "primary_reason_hierarchy": ["available", "treated_and_control_gate", "treated_gate", "control_support_gate", "discordant_monthly_gates", "other"],
        "counts_by_type": agg.groupby("event_type").size().to_dict(),
        "missing_primary_reason": missing["primary_reason"].value_counts().to_dict(),
        "missing_nonexclusive_flags": {c: int(missing[c].sum()) for c in [
            "nonexclusive_treated_evi_failure", "nonexclusive_treated_baseline_failure",
            "nonexclusive_treated_composite_failure", "nonexclusive_treated_pixel_failure",
            "nonexclusive_control_support_failure",
        ]},
        "missing_failure_month_totals": {
            c: int(missing[c].sum()) for c in [
                "treated_fail_evi_months", "treated_fail_baseline_months",
                "treated_fail_composite_months", "treated_fail_pixel_months",
                "control_support_fail_months",
            ]
        },
    }
    (OUT / "t0_availability_decomposition_provenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
