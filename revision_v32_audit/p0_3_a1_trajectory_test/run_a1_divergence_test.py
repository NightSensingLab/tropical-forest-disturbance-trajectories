#!/usr/bin/env python3
"""P0-3: paired spatial-block bootstrap for the existing A1 t5-t0 contrast."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
RUN = PROJECT / "runs/20260909-extended-sensitivities"
ANALYSIS = RUN / "analysis_current_full_history"
STAGE09 = RUN / "stage09_current_full_history"
PIPELINE = PROJECT / "02_analysis/integrated/TropicalRainforest_integrated_analysis/pipeline"
OLD_A1_SCRIPT = PROJECT / "revision_v28_audit/run_fixed_cohort_sensitivity.py"
KEY = ["treatment_grid_id", "event_year"]
ROW_KEY = KEY + ["relative_year"]
SEED = 20260725
REPLICATES = 1000
BLOCK_DEGREES = 2.0
YEARS = list(range(6))
EXPECTED_COUNTS = {"deforestation": 766, "degradation": 1703}
OUTPUT_NAMES = [
    "run.log", "summary.csv", "bootstrap_distribution.csv",
    "a1_event_data.csv", "cohort_counts.csv", "reproduction_check.csv",
    "bootstrap_block_draws.npz", "validation.json", "provenance.csv",
    "manifest.json", "REPORT.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    existing_outputs = [name for name in OUTPUT_NAMES if (OUT / name).exists()]
    if existing_outputs:
        raise FileExistsError(f"Refusing to overwrite outputs: {existing_outputs}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(OUT / "run.log", mode="x"), logging.StreamHandler()],
    )
    logger = logging.getLogger("p0_3")
    sources = {
        "existing_a1_definition": OLD_A1_SCRIPT,
        "annualized_monthly_effect_window": ANALYSIS / "fixed_matched_event_window.parquet",
        "frozen_common_support_weights": ANALYSIS / "common_support_weights.parquet",
        "annualized_monthly_effect_month_counts": STAGE09 / "annual_matched_effects.parquet",
        "annualization_code": PIPELINE / "subannual_response.py",
        "existing_bootstrap_code": PIPELINE / "manuscript_v07_statistics.py",
        "existing_a1_estimates": PROJECT / "revision_v28_audit/fixed_cohort_sensitivity.csv",
        "existing_a1_manifest": PROJECT / "revision_v28_audit/fixed_cohort_manifest.json",
        "existing_a1_counts": PROJECT / "revision_v28_audit/fixed_cohort_counts.csv",
        "p0_3_calculation_script": Path(__file__).resolve(),
    }
    source_hashes = {role: sha256(path) for role, path in sources.items()}
    logger.info("Starting P0-3; Python=%s NumPy=%s pandas=%s", platform.python_version(), np.__version__, pd.__version__)

    # Import the original cohort functions without running their output-writing main().
    spec = importlib.util.spec_from_file_location("existing_a1_definition", OLD_A1_SCRIPT)
    require(spec is not None and spec.loader is not None, "Cannot load existing A1 script")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    require(original.SEED == SEED and original.REPS == REPLICATES, "Original bootstrap settings changed")
    require(original.BLOCK_DEGREES == BLOCK_DEGREES, "Original spatial block scale changed")
    from pipeline.manuscript_v07_statistics import _spatial_blocks, common_block_bootstrap, contrast_on_column

    frame, support = original.load_panel()
    cohort, keys = original.subset_cohort(frame, YEARS)
    require(len(support) == 10970, "Common-support eligibility count differs from 10,970")
    require(not keys.duplicated(KEY).any(), "Duplicate A1 event keys")
    require(not keys.treatment_grid_id.duplicated().any(), "Duplicate A1 treatment event IDs")
    require(not cohort.duplicated(ROW_KEY).any(), "Duplicate A1 annual rows")
    require(len(keys) == 2469 and len(cohort) == 2469 * 6, "Unexpected A1 sample count")
    require(np.isfinite(cohort.pair_effect_evi_mean).all(), "Nonfinite A1 responses")
    require(np.isfinite(cohort.standardized_event_weight).all() and cohort.standardized_event_weight.gt(0).all(), "Invalid frozen weights")
    require((cohort.censor_relative_year.isna() | cohort.censor_relative_year.gt(5)).all(), "A1 contains censoring through t5")
    for col in ["event_type", "standardized_event_weight", "treatment_lon", "treatment_lat"]:
        require(cohort.groupby(KEY)[col].nunique(dropna=False).eq(1).all(), f"Within-event {col} differs across years")
    endpoint_keys = [cohort.loc[cohort.relative_year.eq(r), KEY].sort_values(KEY).reset_index(drop=True) for r in [0, 5]]
    require(endpoint_keys[0].equals(endpoint_keys[1]), "t0 and t5 event sets differ")

    annual = pd.read_parquet(sources["annualized_monthly_effect_month_counts"])
    require(not annual.duplicated(ROW_KEY).any(), "Annual source has duplicate rows")
    month_check = cohort.merge(
        annual[ROW_KEY + ["usable_months", "annual_matched_effect"]],
        on=ROW_KEY, how="left", validate="one_to_one",
    )
    require(month_check.usable_months.notna().all() and month_check.usable_months.ge(9).all(), "A1 annual month requirement not met")
    require(np.array_equal(month_check.pair_effect_evi_mean.to_numpy(), month_check.annual_matched_effect.to_numpy()), "A1 response differs from monthly annualization")
    cohort["spatial_block"] = _spatial_blocks(cohort, BLOCK_DEGREES)
    event_level = cohort.drop_duplicates(KEY)
    counts = event_level.groupby("event_type").agg(
        n_events=("treatment_grid_id", "size"),
        weight_sum=("standardized_event_weight", "sum"),
        spatial_blocks=("spatial_block", "nunique"),
    ).reset_index()
    require(counts.set_index("event_type").n_events.to_dict() == EXPECTED_COUNTS, "A1 event-type counts changed")
    counts["weighted_ESS"] = counts.event_type.map(
        event_level.groupby("event_type").standardized_event_weight.apply(
            lambda w: float(w.sum() ** 2 / np.square(w).sum())
        )
    )
    logger.info("A1=%d events; deforestation=%d degradation=%d; usable_months=%d..%d", len(keys), EXPECTED_COUNTS["deforestation"], EXPECTED_COUNTS["degradation"], month_check.usable_months.min(), month_check.usable_months.max())

    bundle = common_block_bootstrap(
        cohort, "pair_effect_evi_mean", ("event_type", "relative_year"),
        weight_column="standardized_event_weight", weighting="event_area",
        block_degrees=BLOCK_DEGREES, replicates=REPLICATES, seed=SEED,
        minimum_valid_replicates=1,
    )
    require(len(bundle.block_ids) == 235, "A1 spatial-block universe differs from existing 235 blocks")
    direct = contrast_on_column(
        bundle, "event_type", "deforestation", "degradation",
        by_columns=("relative_year",), label="deforestation_minus_degradation",
    )
    previous = pd.read_csv(sources["existing_a1_estimates"], float_precision="round_trip")
    previous = previous.loc[previous.analysis.eq("A1") & previous.estimate_type.eq("direct_contrast")].set_index("relative_year")
    points = direct.set_index("relative_year")
    reconciliation = []
    for r in YEARS:
        for field in ["estimate", "ci_low", "ci_high", "bootstrap_sd"]:
            observed = float(points.loc[r, field])
            existing = float(previous.loc[r, field])
            diff = observed - existing
            require(np.isclose(observed, existing, rtol=0, atol=1e-12), f"A1 t{r} {field} does not reproduce: {diff}")
            reconciliation.append({"relative_year": r, "metric": field, "existing": existing, "recomputed": observed, "difference": diff, "matches_at_1e_12": True})

    # Recreate the helper's single multinomial draw matrix for an explicit pairing audit.
    # Multiplicity counts are equivalent to sampling all represented blocks with replacement.
    block_draws = np.random.default_rng(SEED).multinomial(
        len(bundle.block_ids), np.repeat(1.0 / len(bundle.block_ids), len(bundle.block_ids)),
        size=REPLICATES,
    )
    require(np.all(block_draws.sum(axis=1) == len(bundle.block_ids)), "Bootstrap draw size changed")
    means = {}
    pairing_max_difference = 0.0
    for typ in EXPECTED_COUNTS:
        for r in [0, 5]:
            g = cohort.loc[cohort.event_type.eq(typ) & cohort.relative_year.eq(r)].copy()
            g["weighted_response"] = g.standardized_event_weight * g.pair_effect_evi_mean
            block_totals = g.groupby("spatial_block")[["weighted_response", "standardized_event_weight"]].sum().reindex(bundle.block_ids, fill_value=0.0)
            numer = block_draws @ block_totals.weighted_response.to_numpy()
            denom = block_draws @ block_totals.standardized_event_weight.to_numpy()
            audit_draws = np.divide(numer, denom, out=np.full(REPLICATES, np.nan), where=denom > 0)
            helper_draws = bundle.replicates.loc[bundle.replicates.event_type.eq(typ) & bundle.replicates.relative_year.eq(r)].sort_values("replicate").estimate.to_numpy()
            require(np.allclose(audit_draws, helper_draws, rtol=0, atol=1e-12, equal_nan=True), "Shared block draw reproduction failed")
            finite_diff = np.abs(audit_draws - helper_draws)
            if np.isfinite(finite_diff).any():
                pairing_max_difference = max(pairing_max_difference, float(np.nanmax(finite_diff)))
            means[(typ, r)] = helper_draws

    distribution = pd.DataFrame({"replicate_id": np.arange(1, REPLICATES + 1)})
    for typ in EXPECTED_COUNTS:
        for r in [0, 5]:
            distribution[f"{typ}_mean_t{r}"] = means[(typ, r)]
    distribution["delta_t0"] = means[("deforestation", 0)] - means[("degradation", 0)]
    distribution["delta_t5"] = means[("deforestation", 5)] - means[("degradation", 5)]
    distribution["delta_change"] = distribution.delta_t5 - distribution.delta_t0
    endpoint_cols = [f"{typ}_mean_t{r}" for typ in EXPECTED_COUNTS for r in [0, 5]]
    valid = np.isfinite(distribution[endpoint_cols].to_numpy()).all(axis=1)
    distribution["valid"] = valid
    distribution["status"] = ["valid" if good else "invalid_missing_type_support" for good in valid]
    distribution["invalid_reason"] = [
        "" if good else ";".join(col for col in endpoint_cols if not np.isfinite(row[col]))
        for good, (_, row) in zip(valid, distribution.iterrows())
    ]
    valid_count, invalid_count = int(valid.sum()), int((~valid).sum())
    logger.info("Bootstrap: %d blocks; seed=%d; requested=%d valid=%d invalid=%d", len(bundle.block_ids), SEED, REPLICATES, valid_count, invalid_count)
    if invalid_count:
        logger.warning("Invalid draws retained in CSV; no redraw or silent deletion")
    require(valid_count == REPLICATES, "Fewer than 1,000 valid paired draws; inspect the run log before reporting an interval")

    delta_t0 = float(points.loc[0, "estimate"])
    delta_t5 = float(points.loc[5, "estimate"])
    delta_change = delta_t5 - delta_t0
    summary_rows = []
    for label, column, point in [
        ("A1_delta_t0", "delta_t0", delta_t0),
        ("A1_delta_t5", "delta_t5", delta_t5),
        ("A1_delta_change_t5_minus_t0", "delta_change", delta_change),
    ]:
        draws = distribution.loc[valid, column].to_numpy()
        lo, hi = np.quantile(draws, [0.025, 0.975], method="linear")
        summary_rows.append({
            "Metric": label, "Estimate": point, "CI_lower": float(lo), "CI_upper": float(hi),
            "Bootstrap_SE": float(np.std(draws, ddof=1)), "CI_crosses_zero": bool(lo <= 0 <= hi),
            "bootstrap_requested": REPLICATES, "bootstrap_valid": valid_count,
            "bootstrap_invalid": invalid_count, "spatial_block_count": len(bundle.block_ids),
            "seed": SEED, "block_degrees": BLOCK_DEGREES,
        })
    summary = pd.DataFrame(summary_rows)
    result = summary.iloc[2]
    logger.info("delta_t0=%.17g delta_t5=%.17g delta_change=%.17g", delta_t0, delta_t5, delta_change)
    logger.info("Paired percentile 95%% CI=[%.17g, %.17g]; SE=%.17g; crosses_zero=%s", result.CI_lower, result.CI_upper, result.Bootstrap_SE, result.CI_crosses_zero)

    event_data = event_level[KEY + ["event_type", "macroregion", "treatment_lon", "treatment_lat", "spatial_block", "censor_relative_year", "standardized_event_weight"]].copy()
    for value, prefix in [("pair_effect_evi_mean", "effect_t"), ("usable_months", "usable_months_t")]:
        wide = month_check.pivot(index=KEY, columns="relative_year", values=value).rename(columns=lambda r: f"{prefix}{r}").reset_index()
        event_data = event_data.merge(wide, on=KEY, validate="one_to_one")
    event_data = event_data.sort_values(KEY).reset_index(drop=True)
    validation = {
        "common_support_eligibility_n": len(support), "a1_n": len(keys),
        "a1_type_counts": EXPECTED_COUNTS, "event_ids_unique": True,
        "six_valid_years_per_event": True, "same_t0_t5_events": True,
        "frozen_weights_identical_across_years": True, "usable_months_min": int(month_check.usable_months.min()),
        "usable_months_max": int(month_check.usable_months.max()),
        "annualized_monthly_response_exact_match": True,
        "no_censoring_through_t5": True, "spatial_blocks": len(bundle.block_ids),
        "paired_draws_verified_with_shared_multiplicity_matrix": True,
        "pairing_audit_max_absolute_difference": pairing_max_difference,
        "all_six_existing_a1_contrasts_and_ci_reproduced_at_1e_12": True,
        "bootstrap_requested": REPLICATES, "bootstrap_valid": valid_count, "bootstrap_invalid": invalid_count,
    }
    for role, path in sources.items():
        require(sha256(path) == source_hashes[role], f"Source changed during run: {path}")
    validation["all_source_hashes_unchanged_after_calculation"] = True
    provenance = pd.DataFrame([
        {"role": role, "source_file": str(path), "sha256_before": source_hashes[role], "sha256_after": sha256(path)}
        for role, path in sources.items()
    ])
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.17g")
    distribution.to_csv(OUT / "bootstrap_distribution.csv", index=False, float_format="%.17g")
    event_data.to_csv(OUT / "a1_event_data.csv", index=False, float_format="%.17g")
    counts.to_csv(OUT / "cohort_counts.csv", index=False, float_format="%.17g")
    pd.DataFrame(reconciliation).to_csv(OUT / "reproduction_check.csv", index=False, float_format="%.17g")
    np.savez_compressed(OUT / "bootstrap_block_draws.npz", block_ids=np.asarray(bundle.block_ids), multiplicities=block_draws, replicate_id=distribution.replicate_id.to_numpy())
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    provenance.to_csv(OUT / "provenance.csv", index=False)
    manifest = {
        "task": "P0-3 A1 fixed-cohort paired t5-minus-t0 divergence test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_source": str(OLD_A1_SCRIPT),
        "cohort": "Existing A1: common-support eligibility, valid annualized monthly responses t0-t5, no censoring through t5",
        "outcome": "pair_effect_evi_mean (annualized monthly matched EVI effects)",
        "weight": "Frozen standardized_event_weight; not re-estimated",
        "block_formula": "floor((treatment_lon+180)/2):floor((treatment_lat+90)/2)",
        "block_universe": "235 blocks represented in the existing A1 cohort",
        "sampling": "One shared multinomial block-multiplicity matrix for both event types and all years; 235 blocks per draw with replacement",
        "seed": SEED, "replicates": REPLICATES,
        "CI": "2.5 and 97.5 percentiles; numpy linear quantile interpolation",
        "invalid_draw_policy": "Retain and label all invalid draws; no redraw; if fewer than 1000 valid draws, stop interval/report generation",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "python_executable": sys.executable},
        "outputs": OUTPUT_NAMES,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    interpretation = (
        "The two-sided paired percentile interval excludes zero and is wholly negative; the between-type contrast becomes more negative from t0 to t5 within this selected fixed cohort."
        if result.CI_upper < 0 else
        "The two-sided paired percentile interval does not establish a negative t5-minus-t0 change."
    )
    report = f"""# P0-3: A1 fixed-cohort t0-t5 divergence test

## Result

| Metric | Raw-precision estimate | Paired percentile 95% CI |
|---|---:|---:|
| A1 delta(t0) | {delta_t0:.17g} | [{summary.iloc[0].CI_lower:.17g}, {summary.iloc[0].CI_upper:.17g}] |
| A1 delta(t5) | {delta_t5:.17g} | [{summary.iloc[1].CI_lower:.17g}, {summary.iloc[1].CI_upper:.17g}] |
| A1 delta(t5) minus delta(t0) | {delta_change:.17g} | [{result.CI_lower:.17g}, {result.CI_upper:.17g}] |

- Bootstrap SE for the change: {result.Bootstrap_SE:.17g} EVI units.
- Requested/valid/invalid replicates: {REPLICATES}/{valid_count}/{invalid_count}.
- CI crosses zero: {bool(result.CI_crosses_zero)}.
- {interpretation}
- Raw-precision change rounds to {delta_change:.5f}. Subtraction of separately rounded endpoints (-0.01953 minus -0.00865 = -0.01088) is not the final point estimate.

## Cohort and reproduction

The existing A1 definition was called directly via `load_panel()` and `subset_cohort(frame, [0,1,2,3,4,5])` in `{OLD_A1_SCRIPT.relative_to(PROJECT)}`. It selects 2,469 unique events from the 10,970 common-support eligibility events: deforestation-dominated 766; degradation-dominated 1,703. Each event has six valid primary annualized monthly matched-effect responses, 9-12 usable months in every year, and no dynamic censoring through t5. All A1 `censor_relative_year` values are missing in the saved panel. No cohort definition was changed.

t0 and t5 have exactly the same event IDs, original frozen standardized event-area weights and spatial coordinates. All 14,814 annual values match the preserved Stage09 monthly-effect annualization output exactly. The six existing A1 direct contrasts and their original bootstrap CIs/SDs reproduce within 1e-12 numerical tolerance, including t0 rounding to -0.00865 and t5 to -0.01953. `reproduction_check.csv` gives the original, recomputed and difference values.

## Paired spatial-block bootstrap

The existing `common_block_bootstrap()` and `contrast_on_column()` in `{sources['existing_bootstrap_code'].relative_to(PROJECT)}` were reused. The A1 universe consists of 235 occupied 2-degree by 2-degree blocks with IDs `floor((lon+180)/2):floor((lat+90)/2)`. Seed {SEED} and {REPLICATES} draws match the prior A1 settings. Each draw samples 235 whole blocks with replacement. One common block-multiplicity matrix is applied to both types and all years; repeated blocks multiply the contribution of every event in them. Frozen weights are used to recompute each type's weighted mean before calculating delta(t0), delta(t5) and their difference.

The shared block draws were independently reconstructed and checked against all four endpoint/type bootstrap mean arrays (maximum absolute difference {pairing_max_difference:.17g}). `bootstrap_block_draws.npz` preserves the draw multiplicities and block IDs. `bootstrap_distribution.csv` retains all requested draws, four type/time means, the two contrasts, their paired change and validity status. No draw failed. Were a draw invalid, it would remain labelled in the CSV with no replacement draw; fewer than 1,000 valid draws would stop final interval/report generation. The CI uses the 2.5 and 97.5 percentiles of the paired changes, not subtraction of endpoint CI bounds.

## Sources and execution

- Primary response: `{sources['annualized_monthly_effect_window'].relative_to(PROJECT)}` (`pair_effect_evi_mean`).
- Frozen weights: `{sources['frozen_common_support_weights'].relative_to(PROJECT)}` (`standardized_event_weight`).
- Month-count/value verification: `{sources['annualized_monthly_effect_month_counts'].relative_to(PROJECT)}` (`usable_months`, `annual_matched_effect`).
- Prior A1 estimates: `{sources['existing_a1_estimates'].relative_to(PROJECT)}`.
- Event key: `treatment_grid_id` plus `event_year`; `treatment_grid_id` is also unique in A1.
- Calculation: `run_a1_divergence_test.py`; log: `run.log`; complete event inputs: `a1_event_data.csv`.
- Input/code SHA-256 checks before and after computation are identical (`provenance.csv`). Software/settings are recorded in `manifest.json`; validation checks in `validation.json`.
- Run from the project root: `.venv-figure4/bin/python revision_v32_audit/p0_3_a1_trajectory_test/run_a1_divergence_test.py`. The script refuses to overwrite existing outputs.

## Interpretation boundary

This test concerns the conditional, fixed complete-observation/no-repeat-disturbance A1 cohort, using preserved common-support standardized event-area weights. It does not generalize directly to all initial events or correct complete-observation/no-repeat selection. Frozen eligibility weights do not guarantee covariate balance in the selected A1 subset. A negative endpoint change establishes a difference between t0 and t5 under the specified bootstrap, not monotonic divergence at every intervening year or a causal effect. This is the single requested comparison; no new model or multiple-test search was performed. The separate annual-EVI sensitivity outcome was not used. No manuscript, original analysis/data or existing result was overwritten.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    logger.info("P0-3 completed; source hashes unchanged; outputs=%s", OUT)


if __name__ == "__main__":
    main()
