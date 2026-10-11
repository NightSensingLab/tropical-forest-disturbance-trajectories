#!/usr/bin/env python3
"""Reconstruct frozen matching distances and candidate-pool balance diagnostics."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(__file__).resolve().parent / "analysis_current_full_history"
AUDIT_ROOT = RUN_ROOT / "audit"
sys.path.insert(0, str(PROJECT_ROOT / "02_analysis/integrated/TropicalRainforest_integrated_analysis"))
from pipeline.full_history_events import add_history_qualification
from pipeline.matched_control import (
    MATCHING_COVARIATES, build_spatial_bins, candidate_indices_within_radius,
    robust_location_scale, standardize_covariates,
)
FEATURE_COLUMNS = [
    "grid_id", "event_year", "grid_row", "grid_col", "lon", "lat", "cell_area_m2",
    *MATCHING_COVARIATES, "max_disturbance_share_through_followup",
    "max_disturbance_share_through_2023", "stable_through_followup",
    "stable_through_2023", "event_year_usable", "post_usable_n",
    "coverage_eligible", "macroregion", "matching_covariates_complete",
]
EVENT_KEY = ["treatment_grid_id", "event_year"]

def read_year_features(config: dict, year: int) -> pd.DataFrame:
    root = Path(config["feature_root"]) / f"year{int(year)}"
    paths = sorted(root.glob("part_*.parquet"))
    if len(paths) != 29:
        raise RuntimeError(f"Expected 29 feature parts for {year}, found {len(paths)}")
    frames = [pd.read_parquet(path, columns=FEATURE_COLUMNS) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    if result["grid_id"].duplicated().any():
        raise AssertionError(f"Duplicate feature grid_id in event year {year}")
    return result

def event_id(grid_id, event_year) -> str:
    return f"{int(grid_id)}:{int(event_year)}"

def raw_mad_scales(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    values = frame.loc[:, MATCHING_COVARIATES].apply(pd.to_numeric, errors="coerce")
    locations = values.median(axis=0, skipna=True)
    mad_scales = values.sub(locations, axis=1).abs().median(axis=0, skipna=True) * 1.4826
    sd = values.std(axis=0, ddof=0, skipna=True)
    fallback = ~np.isfinite(mad_scales) | mad_scales.le(0.0)
    scales = mad_scales.copy()
    scales.loc[fallback] = sd.loc[fallback]
    return locations, scales, fallback

def reconstruct_distances(config: dict, pairs: pd.DataFrame, treatments: pd.DataFrame, history_flags: pd.DataFrame):
    link_rows = []
    event_rows = []
    moments = {(scope, cov): [0, 0.0, 0.0] for scope in ["Overall", "deforestation", "degradation"] for cov in MATCHING_COVARIATES}
    for year in range(int(config["primary_event_years"][0]), int(config["primary_event_years"][1]) + 1):
        features = read_year_features(config, year)
        qualified = add_history_qualification(features, history_flags, config["control_history_start"], config["primary_event_threshold"])
        controls = qualified.loc[
            qualified["stable_through_followup_full_history"].astype(bool)
            & qualified["coverage_eligible"].astype(bool)
            & qualified["matching_covariates_complete"].astype(bool)
        ].copy()
        tx = treatments.loc[treatments["event_year"].eq(year)].copy().reset_index(drop=True)
        py = pairs.loc[pairs["event_year"].eq(year)].copy()
        if tx.empty:
            continue
        if not set(tx["grid_id"].astype(int)).issubset(set(qualified["grid_id"].astype(int))):
            raise AssertionError(f"Treatment feature rows missing in year {year}")
        for scope in ["Overall", "deforestation", "degradation"]:
            scoped = controls if scope == "Overall" else controls
            for cov in MATCHING_COVARIATES:
                arr = pd.to_numeric(scoped[cov], errors="coerce").to_numpy(dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                moments[(scope, cov)][0] += int(len(arr))
                moments[(scope, cov)][1] += float(arr.sum())
                moments[(scope, cov)][2] += float(np.sum(arr * arr))
        for macroregion, treatment_group in tx.groupby("macroregion", sort=True):
            control_group = controls.loc[controls["macroregion"].eq(macroregion)].sort_values("grid_id", kind="mergesort").reset_index(drop=True)
            if control_group.empty:
                raise AssertionError(f"No eligible controls for year {year}, region {macroregion}")
            locations, scales = robust_location_scale(control_group)
            _, _, fallback_flags = raw_mad_scales(control_group)
            control_std = standardize_covariates(control_group, locations, scales).to_numpy(dtype=np.float64)
            treatment_std = standardize_covariates(treatment_group, locations, scales)
            spatial_bins = build_spatial_bins(control_group)
            for treatment_index, treatment in treatment_group.iterrows():
                treatment_id = int(treatment["grid_id"])
                candidate_indices, geo = candidate_indices_within_radius(treatment, control_group, spatial_bins, config["primary_radius_km"])
                radius_fallback = False
                search_radius = float(config["primary_radius_km"])
                if len(candidate_indices) == 0:
                    candidate_indices, geo = candidate_indices_within_radius(treatment, control_group, spatial_bins, config["fallback_radius_km"])
                    radius_fallback = True
                    search_radius = float(config["fallback_radius_km"])
                if len(candidate_indices) == 0:
                    raise AssertionError(f"No candidates for matched treatment {treatment_id}/{year}")
                tvals = treatment_std.loc[treatment_index].to_numpy(dtype=np.float64)
                differences = control_std[candidate_indices] - tvals
                distances = np.sqrt(np.sum(differences ** 2, axis=1))
                control_ids = control_group.iloc[candidate_indices]["grid_id"].to_numpy()
                order = np.lexsort((control_ids, geo, distances))[: int(config["max_controls"])]
                selected_indices = candidate_indices[order]
                selected_geo = geo[order]
                selected_dist = distances[order]
                selected_ids = control_ids[order]
                actual = py.loc[py["treatment_grid_id"].astype(int).eq(treatment_id)].sort_values("match_rank", kind="mergesort")
                if len(actual) != len(selected_ids):
                    raise AssertionError(f"Selected-link count mismatch for {treatment_id}/{year}")
                if not np.array_equal(actual["control_grid_id"].astype(int).to_numpy(), selected_ids.astype(int)):
                    raise AssertionError(f"Control selection mismatch for {treatment_id}/{year}")
                if not np.allclose(actual["covariate_distance"].to_numpy(dtype=float), selected_dist, atol=1e-10, rtol=1e-10):
                    raise AssertionError(f"Frozen distance mismatch for {treatment_id}/{year}")
                event_rows.append({
                    "event_id": event_id(treatment_id, year), "treatment_grid_id": treatment_id,
                    "event_year": int(year), "event_type": str(treatment["event_type"]),
                    "macroregion": str(macroregion), "candidate_count": int(len(candidate_indices)),
                    "radius_fallback": bool(radius_fallback), "search_radius_km": search_radius,
                    "matched_control_n": int(len(selected_ids)), "min_distance": float(np.min(selected_dist)),
                    "mean_distance": float(np.mean(selected_dist)), "median_distance": float(np.median(selected_dist)),
                    "max_distance": float(np.max(selected_dist)),
                })
                for rank, index, control_id, geo_distance, distance in zip(range(1, len(selected_indices) + 1), selected_indices, selected_ids, selected_geo, selected_dist):
                    control = control_group.iloc[int(index)]
                    diff = differences[order[rank - 1]]
                    rec = {
                        "event_id": event_id(treatment_id, year), "treatment_grid_id": treatment_id,
                        "control_grid_id": int(control_id), "event_year": int(year),
                        "event_type": str(treatment["event_type"]), "macroregion": str(macroregion),
                        "geographic_distance_km_recomputed": float(geo_distance),
                        "geographic_distance_km_frozen": float(actual.loc[actual["control_grid_id"].astype(int).eq(int(control_id)), "geographic_distance_km"].iloc[0]),
                        "covariate_distance_recomputed": float(distance),
                        "covariate_distance_frozen": float(actual.loc[actual["control_grid_id"].astype(int).eq(int(control_id)), "covariate_distance"].iloc[0]),
                        "match_rank": int(rank), "candidate_count": int(len(candidate_indices)),
                        "radius_fallback": bool(radius_fallback), "search_radius_km": search_radius,
                        "control_reuse_count": int(actual.loc[actual["control_grid_id"].astype(int).eq(int(control_id)), "control_reuse_count"].iloc[0]),
                    }
                    sq = diff * diff
                    for j, cov in enumerate(MATCHING_COVARIATES):
                        rec[f"{cov}_treated_raw"] = float(treatment[cov])
                        rec[f"{cov}_control_raw"] = float(control[cov])
                        rec[f"{cov}_center"] = float(locations[cov])
                        rec[f"{cov}_scale"] = float(scales[cov])
                        rec[f"{cov}_treated_std"] = float(tvals[j])
                        rec[f"{cov}_control_std"] = float(control_std[int(index), j])
                        rec[f"{cov}_abs_std_diff"] = float(abs(diff[j]))
                        rec[f"{cov}_squared_contribution"] = float(sq[j])
                        rec[f"{cov}_scale_fallback"] = bool(fallback_flags[cov])
                    rec["dominant_distance_covariate"] = MATCHING_COVARIATES[int(np.argmax(sq))]
                    rec["dominant_contribution"] = float(np.max(sq))
                    rec["distance_abs_error"] = abs(float(distance) - float(actual.loc[actual["control_grid_id"].astype(int).eq(int(control_id)), "covariate_distance"].iloc[0]))
                    rec["geographic_abs_error"] = abs(float(geo_distance) - float(actual.loc[actual["control_grid_id"].astype(int).eq(int(control_id)), "geographic_distance_km"].iloc[0]))
                    link_rows.append(rec)
    links = pd.DataFrame(link_rows)
    events = pd.DataFrame(event_rows)
    if len(links) != 50198 or len(events) != 16747:
        raise AssertionError(f"Distance reconstruction counts differ: links={len(links)}, events={len(events)}")
    return links, events, moments

def distance_diagnostics(links: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    thresholds = [3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    rows = []
    for threshold in thresholds:
        mask = links["covariate_distance_recomputed"].gt(threshold)
        event_mask = events["max_distance"].gt(threshold)
        rows.append({
            "summary_type": f"link_gt_{threshold:g}", "threshold": threshold,
            "link_n": int(mask.sum()), "link_pct": float(mask.mean() * 100.0),
            "unique_event_n": int(links.loc[mask, "event_id"].nunique()),
            "event_pct": float(links.loc[mask, "event_id"].nunique() / len(events) * 100.0),
            "unique_control_n": int(links.loc[mask, "control_grid_id"].nunique()),
        })
    distances = links["covariate_distance_recomputed"]
    for q in [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]:
        rows.append({"summary_type": f"link_quantile_{q:g}", "threshold": np.nan, "link_n": np.nan, "link_pct": np.nan, "unique_event_n": np.nan, "event_pct": np.nan, "unique_control_n": np.nan, "quantile": q, "value": float(distances.quantile(q))})
    rows.append({"summary_type": "link_max", "threshold": np.nan, "link_n": np.nan, "link_pct": np.nan, "unique_event_n": np.nan, "event_pct": np.nan, "unique_control_n": np.nan, "quantile": 1.0, "value": float(distances.max())})
    summary = pd.DataFrame(rows)
    event_summary = events.copy()
    for threshold in thresholds:
        event_summary[f"max_gt_{threshold:g}"] = event_summary["max_distance"].gt(threshold)
    event_summary = event_summary.sort_values(["max_distance", "event_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    extreme = links.loc[links["covariate_distance_recomputed"].gt(10.0)].copy()
    if not extreme.empty:
        def classify(row):
            cov = row["dominant_distance_covariate"]
            treated = abs(float(row[f"{cov}_treated_std"]))
            control = abs(float(row[f"{cov}_control_std"]))
            scale = float(row[f"{cov}_scale"])
            if not np.isfinite(row["covariate_distance_recomputed"]):
                return "data_anomaly_nonfinite"
            if scale <= 1e-6:
                return "standardization_scale_anomaly"
            if int(row["candidate_count"]) <= 3:
                return "candidate_support_scarcity"
            if bool(row["radius_fallback"]):
                return "spatial_radius_constraint"
            if treated >= 5.0 and treated > control:
                return "extreme_treated_covariate"
            if control >= 5.0 and control > treated:
                return "extreme_control_covariate"
            return "covariate_support_tail"
        extreme["diagnostic_category"] = extreme.apply(classify, axis=1)
        keep = ["event_id", "treatment_grid_id", "control_grid_id", "event_type", "macroregion", "event_year", "covariate_distance_recomputed", "geographic_distance_km_recomputed", "candidate_count", "match_rank", "dominant_distance_covariate", "dominant_contribution", "diagnostic_category", "radius_fallback", "search_radius_km"]
        extreme = extreme.loc[:, keep].sort_values("covariate_distance_recomputed", ascending=False, kind="mergesort")
    else:
        extreme = pd.DataFrame(columns=["event_id", "treatment_grid_id", "control_grid_id", "event_type", "macroregion", "event_year", "covariate_distance_recomputed", "geographic_distance_km_recomputed", "candidate_count", "match_rank", "dominant_distance_covariate", "dominant_contribution", "diagnostic_category", "radius_fallback", "search_radius_km"])
    return summary, event_summary, extreme

def concentration_diagnostics(event_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in [5.0, 10.0, 20.0]:
        flagged = event_summary.loc[event_summary["max_distance"].gt(threshold)].copy()
        for dimension in ["event_type", "macroregion", "event_year"]:
            counts = flagged.groupby(dimension, dropna=False).size().rename("extreme_event_n")
            totals = event_summary.groupby(dimension, dropna=False).size().rename("all_event_n")
            joined = pd.concat([counts, totals], axis=1).fillna(0.0).reset_index()
            joined["threshold"] = threshold
            joined["dimension"] = dimension
            joined["category"] = joined[dimension].astype(str)
            joined["extreme_event_pct_within_category"] = np.divide(
                joined["extreme_event_n"], joined["all_event_n"],
                out=np.zeros(len(joined), dtype=float), where=joined["all_event_n"].gt(0),
            ) * 100.0
            joined["extreme_event_pct_of_all_extremes"] = joined["extreme_event_n"] / float(max(len(flagged), 1)) * 100.0
            rows.append(joined[["threshold", "dimension", "category", "extreme_event_n", "all_event_n", "extreme_event_pct_within_category", "extreme_event_pct_of_all_extremes"]])
    return pd.concat(rows, ignore_index=True)

def balance_from_moments_local(treatments: pd.DataFrame, matched_controls: pd.DataFrame, moments: dict) -> pd.DataFrame:
    """Same moment-based balance calculation used by Stage 07.

    Kept local so this audit does not import the plotting stack from
    ``run_matched_control.py``.
    """
    def mean_variance(values, weights):
        values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not valid.any():
            return np.nan, np.nan
        values, weights = values[valid], weights[valid]
        total = weights.sum()
        mean = np.sum(values * weights) / total
        variance = np.sum(weights * (values - mean) ** 2) / total
        return float(mean), float(variance)

    def smd(treated_mean, control_mean, treated_var, control_var):
        scale = np.sqrt((treated_var + control_var) / 2.0)
        if np.isfinite(scale) and scale > 0.0:
            return float((treated_mean - control_mean) / scale)
        return 0.0 if np.isclose(treated_mean, control_mean) else np.inf

    rows = []
    scopes = [("Overall", None)]
    scopes.extend((value, value) for value in sorted(treatments.event_type.unique()))
    for scope, event_type in scopes:
        treated = treatments if event_type is None else treatments.loc[treatments.event_type.eq(event_type)]
        post = matched_controls if event_type is None else matched_controls.loc[matched_controls.event_type.eq(event_type)]
        if treated.empty or post.empty:
            continue
        aggregated = post.groupby(["control_grid_id", "event_year"], sort=False)["within_set_weight"].transform("sum")
        post_unique = post.assign(_weight=aggregated).drop_duplicates(["control_grid_id", "event_year"])
        effective_n = float(post_unique["_weight"].sum() ** 2 / np.sum(post_unique["_weight"].to_numpy(dtype=np.float64) ** 2))
        for covariate in MATCHING_COVARIATES:
            treated_mean, treated_var = mean_variance(treated[covariate], np.ones(len(treated)))
            n, total, square_total = moments[(scope, covariate)]
            pre_mean = total / n
            pre_var = max(square_total / n - pre_mean ** 2, 0.0)
            post_mean, post_var = mean_variance(post_unique[covariate], post_unique["_weight"])
            rows.append({
                "scope": scope, "covariate": covariate,
                "treated_mean": treated_mean, "pre_control_mean": pre_mean,
                "post_control_mean": post_mean,
                "pre_smd": smd(treated_mean, pre_mean, treated_var, pre_var),
                "post_smd": smd(treated_mean, post_mean, treated_var, pre_var),
                "post_variance_ratio": treated_var / post_var if post_var > 0.0 else np.inf,
                "treated_n": int(len(treated)), "pre_control_n": int(n),
                "matched_control_n": int(len(post_unique)), "effective_control_n": effective_n,
            })
    return pd.DataFrame(rows).sort_values(["scope", "covariate"]).reset_index(drop=True)


def main():
    config = json.loads((PROJECT_ROOT / "runs/20260904-expanded-control-pool/config.json").read_text())
    config["feature_root"] = str(PROJECT_ROOT / config["feature_root"])
    pairs = pd.read_parquet(RUN_ROOT / "fixed_matched_pairs.parquet")
    treatments = pd.read_parquet(RUN_ROOT / "matched_treatments.parquet")
    history = pd.read_parquet(RUN_ROOT / "history_qualified_controls.parquet")
    links, events, moments = reconstruct_distances(config, pairs, treatments, history)
    tail, events, extremes = distance_diagnostics(links, events)
    controls = links[["control_grid_id", "event_year", "event_type"]].copy()
    controls["within_set_weight"] = 1.0 / links.groupby(EVENT_KEY)["control_grid_id"].transform("size").to_numpy()
    for covariate in MATCHING_COVARIATES:
        controls[covariate] = links[f"{covariate}_control_raw"].to_numpy()
    balance = balance_from_moments_local(treatments, controls, moments)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    links.to_parquet(AUDIT_ROOT / "matching_distance_links_reconstructed.parquet", index=False)
    for filename, frame in [
        ("matching_distance_tail_summary.csv", tail),
        ("matching_distance_event_summary.csv", events),
        ("matching_distance_extreme_links.csv", extremes),
        ("matching_distance_concentration.csv", concentration_diagnostics(events)),
        ("matching_balance_recomputed.csv", balance),
    ]:
        frame.to_csv(AUDIT_ROOT / filename, index=False)

if __name__ == "__main__":
    main()
