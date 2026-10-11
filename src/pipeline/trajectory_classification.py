"""History-aware sensitivity and trajectory classification functions."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def capped_standardized_weights(weights: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Apply a non-negative multiplier cap and return area-standardized weights."""
    required = {"event_area_m2", "standardization_multiplier"}
    if missing := required - set(weights.columns):
        raise ValueError(f"Weight table lacks columns: {sorted(missing)}")
    cap = float(cap)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("cap must be finite and positive")
    result = weights.copy()
    multiplier = pd.to_numeric(result["standardization_multiplier"], errors="coerce")
    area = pd.to_numeric(result["event_area_m2"], errors="coerce")
    result["capped_standardization_multiplier"] = multiplier.clip(lower=0.0, upper=cap)
    result["standardized_event_weight"] = area * result["capped_standardization_multiplier"]
    return result


def validate_dynamic_window(window: pd.DataFrame) -> None:
    """Assert that pair effects are absent at and after the repeat event."""
    required = {"treatment_grid_id", "event_year", "relative_year", "censor_relative_year"}
    if missing := required - set(window.columns):
        raise ValueError(f"Dynamic window lacks columns: {sorted(missing)}")
    effect_columns = [column for column in window.columns if column.startswith("pair_effect_")]
    if not effect_columns:
        raise ValueError("Dynamic window contains no pair-effect columns")
    rel = pd.to_numeric(window["relative_year"], errors="coerce")
    censor = pd.to_numeric(window["censor_relative_year"], errors="coerce")
    masked = censor.notna() & rel.notna() & rel.ge(censor)
    for column in effect_columns:
        if pd.to_numeric(window.loc[masked, column], errors="coerce").notna().any():
            raise AssertionError(f"Dynamic censoring violation in {column}")


def validate_pre_event_pair_history(pairs: pd.DataFrame, history_flags: pd.DataFrame) -> pd.DataFrame:
    """Validate pre-event controls and attach their true dynamic censor year.

    Controls are eligible only when their first full-history threshold crossing
    is at or after the treatment event year. A crossing in ``event_year`` is
    censored at relative year 0, matching the project rule.
    """
    required_pairs = {"control_grid_id", "event_year", "control_censor_relative_year"}
    required_flags = {"grid_id", "first_threshold_year", "history_complete"}
    if missing := required_pairs - set(pairs.columns):
        raise ValueError(f"Pairs lack history columns: {sorted(missing)}")
    if missing := required_flags - set(history_flags.columns):
        raise ValueError(f"History flags lack columns: {sorted(missing)}")
    flags = history_flags.loc[:, sorted(required_flags)].drop_duplicates("grid_id")
    if flags["grid_id"].duplicated().any():
        raise ValueError("History flags must be unique by grid_id.")
    result = pairs.copy().merge(
        flags.rename(columns={"grid_id": "control_grid_id"}),
        on="control_grid_id",
        how="left",
        validate="many_to_one",
    )
    if result["history_complete"].isna().any() or not result["history_complete"].astype(bool).all():
        raise AssertionError("Pre-event pairs include missing or incomplete full history.")
    first = pd.to_numeric(result["first_threshold_year"], errors="coerce")
    event_year = pd.to_numeric(result["event_year"], errors="coerce")
    prior = first.notna() & first.lt(event_year)
    if prior.any():
        raise AssertionError("Pre-event pairs include a control with a prior threshold crossing.")
    expected = first - event_year
    expected = expected.where(expected.between(0, 5))
    actual = pd.to_numeric(result["control_censor_relative_year"], errors="coerce")
    equal = (actual.isna() & expected.isna()) | (actual.notna() & expected.notna() & actual.eq(expected))
    if not equal.all():
        raise AssertionError("Control censor relative year does not equal the true first crossing year.")
    result["prior_threshold_before_event"] = prior
    result["control_censor_relative_year_expected"] = expected
    return result


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return np.nan
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    return float((ranks[y == 1].sum() - positives * (positives + 1.0) / 2.0) / (positives * negatives))


def _balanced_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    predicted = np.asarray(predicted, dtype=bool)
    recalls = []
    for label in (0, 1):
        mask = y == label
        if mask.any():
            recalls.append(float((predicted[mask] == bool(label)).mean()))
    return float(np.mean(recalls)) if len(recalls) == 2 else np.nan


def _best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    positives = sorted_y.astype(np.int64)
    negatives = 1 - positives
    tp = positives[::-1].cumsum()[::-1]
    fp = negatives[::-1].cumsum()[::-1]
    n_pos = max(int(positives.sum()), 1)
    n_neg = max(int(negatives.sum()), 1)
    balanced = (tp / n_pos + (n_neg - fp) / n_neg) / 2.0
    return float(sorted_scores[int(np.nanargmax(balanced))])


def _stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    parts = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        parts.append(np.array_split(indices, int(n_folds)))
    return [np.concatenate([parts[0][i], parts[1][i]]) for i in range(int(n_folds))]


def _spatial_folds(frame: pd.DataFrame, y: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    block_x = np.floor((pd.to_numeric(frame["lon"], errors="coerce") + 180.0) / 2.0).astype(int)
    block_y = np.floor((pd.to_numeric(frame["lat"], errors="coerce") + 90.0) / 2.0).astype(int)
    blocks = pd.Series([f"{x}:{z}" for x, z in zip(block_x, block_y)], index=frame.index)
    block_table = pd.DataFrame({"block": blocks.to_numpy(), "label": y}).groupby("block", sort=True)["label"].agg(["sum", "count"])
    rng = np.random.default_rng(int(seed))
    order = list(block_table.index)
    rng.shuffle(order)
    fold_blocks = [[] for _ in range(int(n_folds))]
    fold_counts = np.zeros((int(n_folds), 2), dtype=float)
    for block in sorted(order, key=lambda key: (-int(block_table.loc[key, "count"]), str(key))):
        counts = np.array([block_table.loc[block, "count"] - block_table.loc[block, "sum"], block_table.loc[block, "sum"]], dtype=float)
        target = int(np.lexsort((np.arange(int(n_folds)), fold_counts.sum(axis=1)))[0])
        fold_blocks[target].append(block)
        fold_counts[target] += counts
    return [np.flatnonzero(blocks.isin(selected).to_numpy()) for selected in fold_blocks]


def _region_folds(frame: pd.DataFrame) -> list[np.ndarray]:
    return [np.flatnonzero(frame["macroregion"].astype(str).eq(region).to_numpy()) for region in sorted(frame["macroregion"].dropna().astype(str).unique())]


def _fit_predict(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, float]:
    if len(train) == 0 or len(test) == 0 or len(np.unique(y[train])) < 2:
        return np.full(len(test), np.nan), np.nan
    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale == 0.0] = 1.0
    z = (x - mean) / scale
    beta = np.linalg.solve(z[train].T @ z[train] + np.eye(x.shape[1]), z[train].T @ (y[train] - y[train].mean()))
    train_scores = z[train] @ beta + y[train].mean()
    test_scores = z[test] @ beta + y[train].mean()
    return test_scores, _best_threshold(train_scores, y[train])


def _evaluate_scheme(frame: pd.DataFrame, columns: Sequence[str], folds: list[np.ndarray], seed: int, scheme: str) -> dict:
    clean = frame.dropna(subset=list(columns)).reset_index(drop=True)
    y = (clean["event_type"].astype(str).to_numpy() == "deforestation").astype(np.int8)
    if len(clean) == 0 or len(np.unique(y)) < 2:
        return {"validation_scheme": scheme, "sample_n": int(len(clean)), "deforestation_n": int(y.sum()), "degradation_n": int(len(y) - y.sum()), "folds_used": 0, "auc": np.nan, "auc_sd": np.nan, "balanced_accuracy": np.nan, "balanced_accuracy_sd": np.nan, "seed": int(seed)}
    if scheme != "random_stratified_5fold":
        # Folds are computed on the cleaned frame so index filtering remains exact.
        if scheme == "spatial_block_2deg_5fold":
            folds = _spatial_folds(clean, y, 5, seed)
        elif scheme == "region_leave_one_out":
            folds = _region_folds(clean)
    x = clean.loc[:, list(columns)].to_numpy(dtype=np.float64)
    oof = np.full(len(clean), np.nan)
    fold_auc, fold_ba = [], []
    for test in folds:
        test = np.asarray(test, dtype=int)
        train = np.setdiff1d(np.arange(len(clean)), test)
        scores, threshold = _fit_predict(x, y, train, test)
        oof[test] = scores
        if np.isfinite(scores).any() and np.isfinite(threshold):
            valid = np.isfinite(scores)
            fold_auc.append(_auc(y[test][valid], scores[valid]))
            fold_ba.append(_balanced_accuracy(y[test][valid], scores[valid] >= threshold))
    valid_oof = np.isfinite(oof)
    return {
        "validation_scheme": scheme,
        "sample_n": int(len(clean)),
        "deforestation_n": int(y.sum()),
        "degradation_n": int(len(y) - y.sum()),
        "folds_used": int(len(fold_auc)),
        "auc": _auc(y[valid_oof], oof[valid_oof]),
        "auc_sd": float(np.std(fold_auc, ddof=1)) if len(fold_auc) > 1 else np.nan,
        "balanced_accuracy": float(np.nanmean(fold_ba)) if fold_ba else np.nan,
        "balanced_accuracy_sd": float(np.nanstd(fold_ba, ddof=1)) if len(fold_ba) > 1 else np.nan,
        "seed": int(seed),
    }


def evaluate_classifier_validations(frame: pd.DataFrame, feature_sets: Mapping[str, Sequence[str]], seed: int = 20260725) -> pd.DataFrame:
    """Evaluate t0 and trajectory classifiers under three locked CV schemes."""
    required = {"event_type", "lon", "lat", "macroregion"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Classifier frame lacks columns: {sorted(missing)}")
    rows = []
    schemes = ("random_stratified_5fold", "spatial_block_2deg_5fold", "region_leave_one_out")
    for label, columns in feature_sets.items():
        for scheme in schemes:
            clean = frame.dropna(subset=list(columns)).reset_index(drop=True)
            y = (clean["event_type"].astype(str).to_numpy() == "deforestation").astype(np.int8)
            folds = _stratified_folds(y, 5, seed) if scheme == "random_stratified_5fold" else []
            row = _evaluate_scheme(clean, columns, folds, seed, scheme)
            row["feature_set"] = str(label)
            row["features"] = ",".join(columns)
            row["model"] = "ridge linear discriminant; equal-event weighting"
            rows.append(row)
    return pd.DataFrame(rows).loc[:, ["feature_set", "validation_scheme", "features", "sample_n", "deforestation_n", "degradation_n", "folds_used", "auc", "auc_sd", "balanced_accuracy", "balanced_accuracy_sd", "seed", "model"]]


__all__ = [
    "capped_standardized_weights",
    "evaluate_classifier_validations",
    "validate_dynamic_window",
    "validate_pre_event_pair_history",
]
