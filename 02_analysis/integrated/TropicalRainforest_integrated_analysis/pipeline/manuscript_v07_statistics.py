from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapBundle:
    summary: pd.DataFrame
    replicates: pd.DataFrame
    block_ids: tuple[str, ...]
    seed: int
    block_degrees: float


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric_weights = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(numeric_values)
        & np.isfinite(numeric_weights)
        & (numeric_weights > 0)
    )
    if not valid.any():
        return np.nan
    return float(
        np.sum(numeric_values[valid] * numeric_weights[valid])
        / numeric_weights[valid].sum()
    )


def build_supported_late_blocks(
    frame: pd.DataFrame,
    *,
    minimum_events: int = 10,
    minimum_event_years: int = 3,
) -> pd.DataFrame:
    """Build supported block records and distinguish mean deficit from strict negativity."""
    required = {
        "treatment_grid_id",
        "event_year",
        "event_type",
        "spatial_block",
        "event_area_m2",
        "effect_t3",
        "effect_t4",
        "effect_t5",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Persistent treatment table lacks columns: {sorted(missing)}")
    if minimum_events < 1 or minimum_event_years < 1:
        raise ValueError("Spatial support thresholds must be positive")
    if frame.duplicated(["treatment_grid_id", "event_year", "event_type"]).any():
        raise ValueError("Persistent treatment rows are duplicated")

    rows: list[dict[str, object]] = []
    for (event_type, spatial_block), group in frame.groupby(
        ["event_type", "spatial_block"], sort=True, dropna=False
    ):
        event_count = int(group["treatment_grid_id"].nunique())
        event_year_count = int(group["event_year"].nunique())
        supported = (
            event_count >= int(minimum_events)
            and event_year_count >= int(minimum_event_years)
        )
        if not supported:
            continue
        weights = group["event_area_m2"]
        estimates = {
            relative_year: _weighted_mean(group[f"effect_t{relative_year}"], weights)
            for relative_year in (3, 4, 5)
        }
        if not np.isfinite(list(estimates.values())).all():
            raise ValueError(
                f"Supported block {event_type}/{spatial_block} has nonfinite year estimates"
            )
        rows.append(
            {
                "event_type": str(event_type),
                "spatial_block": str(spatial_block),
                "event_count": event_count,
                "event_year_count": event_year_count,
                "event_area_m2": float(
                    pd.to_numeric(weights, errors="coerce").sum()
                ),
                "estimate_t3": estimates[3],
                "estimate_t4": estimates[4],
                "estimate_t5": estimates[5],
                "late_mean": float(np.mean(list(estimates.values()))),
                "supported_flag": True,
                "strict_all_year_negative": bool(
                    all(value < 0 for value in estimates.values())
                ),
            }
        )
    columns = [
        "event_type",
        "spatial_block",
        "event_count",
        "event_year_count",
        "event_area_m2",
        "estimate_t3",
        "estimate_t4",
        "estimate_t5",
        "late_mean",
        "supported_flag",
        "strict_all_year_negative",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["event_type", "spatial_block"], kind="mergesort"
    ).reset_index(drop=True)


def _spatial_blocks(frame: pd.DataFrame, block_degrees: float) -> pd.Series:
    if not np.isfinite(block_degrees) or block_degrees <= 0:
        raise ValueError("block_degrees must be finite and positive")
    lon = pd.to_numeric(frame["treatment_lon"], errors="coerce").to_numpy(
        dtype=float
    )
    lat = pd.to_numeric(frame["treatment_lat"], errors="coerce").to_numpy(
        dtype=float
    )
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Bootstrap coordinates must be finite")
    if ((lon < -180) | (lon > 180) | (lat < -90) | (lat > 90)).any():
        raise ValueError("Bootstrap coordinates are outside EPSG:4326")
    bx = np.floor((lon + 180.0) / block_degrees).astype(np.int32)
    by = np.floor((lat + 90.0) / block_degrees).astype(np.int32)
    return pd.Series(
        [f"{x}:{y}" for x, y in zip(bx, by)],
        index=frame.index,
        name="spatial_block",
    )


def common_block_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: tuple[str, ...],
    *,
    weight_column: str = "event_area_m2",
    weighting: str = "event_area",
    block_degrees: float = 2.0,
    replicates: int = 1000,
    seed: int = 20260725,
    minimum_valid_replicates: int = 1,
) -> BootstrapBundle:
    """Estimate every requested group from one common spatial-block draw matrix."""
    if not group_columns:
        raise ValueError("group_columns must be nonempty")
    if weighting not in {"event_area", "equal_grid"}:
        raise ValueError("weighting must be event_area or equal_grid")
    required = {value_column, "treatment_lon", "treatment_lat", *group_columns}
    if weighting == "event_area":
        required.add(weight_column)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Bootstrap frame lacks columns: {sorted(missing)}")
    if int(replicates) < 2:
        raise ValueError("replicates must be at least two")
    if not 1 <= int(minimum_valid_replicates) <= int(replicates):
        raise ValueError(
            "minimum_valid_replicates must be between one and replicates"
        )

    working = frame.copy()
    working["spatial_block"] = _spatial_blocks(working, float(block_degrees))
    block_ids = tuple(sorted(str(value) for value in working.spatial_block.unique()))
    if not block_ids:
        raise ValueError("Bootstrap frame has no spatial blocks")

    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(
        len(block_ids),
        np.repeat(1.0 / len(block_ids), len(block_ids)),
        size=int(replicates),
    )
    summaries: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    grouper: str | list[str]
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for key, group in working.groupby(grouper, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        values = pd.to_numeric(group[value_column], errors="coerce").to_numpy(
            dtype=float
        )
        weights = (
            pd.to_numeric(group[weight_column], errors="coerce").to_numpy(
                dtype=float
            )
            if weighting == "event_area"
            else np.ones(len(group), dtype=float)
        )
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            raise ValueError(
                f"Bootstrap group {key_tuple} has no valid weighted observations"
            )
        block_frame = pd.DataFrame(
            {
                "spatial_block": group.loc[valid, "spatial_block"].to_numpy(),
                "weighted_value": values[valid] * weights[valid],
                "weight": weights[valid],
            }
        )
        block_sums = (
            block_frame.groupby("spatial_block", sort=True)
            .sum()
            .reindex(block_ids, fill_value=0.0)
        )
        numerators = counts @ block_sums["weighted_value"].to_numpy(dtype=float)
        denominators = counts @ block_sums["weight"].to_numpy(dtype=float)
        estimates = np.divide(
            numerators,
            denominators,
            out=np.full(int(replicates), np.nan),
            where=denominators > 0,
        )
        finite = estimates[np.isfinite(estimates)]
        if len(finite) < int(minimum_valid_replicates):
            raise RuntimeError(
                f"Bootstrap group {key_tuple} has only {len(finite)} valid "
                f"bootstrap replicates; minimum is {minimum_valid_replicates}"
            )
        group_values = dict(zip(group_columns, key_tuple))
        summaries.append(
            {
                **group_values,
                "estimate": float(
                    np.sum(values[valid] * weights[valid]) / weights[valid].sum()
                ),
                "ci_low": float(np.quantile(finite, 0.025)),
                "ci_high": float(np.quantile(finite, 0.975)),
                "bootstrap_sd": float(np.std(finite, ddof=1))
                if len(finite) > 1
                else np.nan,
                "n": int(valid.sum()),
                "treatment_count": int(
                    group.loc[valid, "treatment_grid_id"].nunique()
                )
                if "treatment_grid_id" in group.columns
                else int(valid.sum()),
                "weight_sum": float(weights[valid].sum()),
                "weighting": weighting,
                "spatial_block_count": int(
                    block_frame["spatial_block"].nunique()
                ),
                "bootstrap_block_universe_count": int(len(block_ids)),
                "bootstrap_requested": int(replicates),
                "bootstrap_valid": int(len(finite)),
                "bootstrap_failed": int(replicates - len(finite)),
                "bootstrap_seed": int(seed),
                "block_degrees": float(block_degrees),
            }
        )
        replicate_rows.extend(
            {
                **group_values,
                "replicate": int(replicate),
                "estimate": float(estimate),
            }
            for replicate, estimate in enumerate(estimates)
        )

    summary = pd.DataFrame(summaries).sort_values(
        list(group_columns), kind="mergesort"
    )
    replicate_frame = pd.DataFrame(replicate_rows).sort_values(
        [*group_columns, "replicate"], kind="mergesort"
    )
    return BootstrapBundle(
        summary=summary.reset_index(drop=True),
        replicates=replicate_frame.reset_index(drop=True),
        block_ids=block_ids,
        seed=int(seed),
        block_degrees=float(block_degrees),
    )


def contrast_on_column(
    bundle: BootstrapBundle,
    compare_column: str,
    left_value: object,
    right_value: object,
    *,
    by_columns: tuple[str, ...],
    label: str,
) -> pd.DataFrame:
    """Return left-minus-right estimates and intervals from paired replicates."""
    if not by_columns:
        raise ValueError("by_columns must be nonempty")
    required = {compare_column, *by_columns, "estimate"}
    if not required.issubset(bundle.summary.columns):
        raise ValueError("Contrast columns are absent from the bootstrap summary")

    left_point = bundle.summary.loc[
        bundle.summary[compare_column].eq(left_value)
    ].set_index(list(by_columns))
    right_point = bundle.summary.loc[
        bundle.summary[compare_column].eq(right_value)
    ].set_index(list(by_columns))
    if left_point.index.has_duplicates or right_point.index.has_duplicates:
        raise ValueError("Contrast point-estimate keys are duplicated")
    if set(left_point.index) != set(right_point.index):
        raise ValueError("Contrast point-estimate keys do not match")

    join_columns = [*by_columns, "replicate"]
    left_draws = bundle.replicates.loc[
        bundle.replicates[compare_column].eq(left_value),
        [*join_columns, "estimate"],
    ].rename(columns={"estimate": "left_estimate"})
    right_draws = bundle.replicates.loc[
        bundle.replicates[compare_column].eq(right_value),
        [*join_columns, "estimate"],
    ].rename(columns={"estimate": "right_estimate"})
    if left_draws.duplicated(join_columns).any() or right_draws.duplicated(
        join_columns
    ).any():
        raise ValueError("Contrast replicate keys are duplicated")
    merged = left_draws.merge(
        right_draws,
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(left_draws) or len(merged) != len(right_draws):
        raise ValueError("Contrast replicate keys do not match")
    merged["estimate"] = merged["left_estimate"] - merged["right_estimate"]

    rows: list[dict[str, object]] = []
    grouper: str | list[str]
    grouper = by_columns[0] if len(by_columns) == 1 else list(by_columns)
    for key, draws in merged.groupby(grouper, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        index_key = key_tuple if len(key_tuple) > 1 else key_tuple[0]
        values = draws["estimate"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise RuntimeError(f"Contrast {key_tuple} has no valid replicates")
        rows.append(
            {
                **dict(zip(by_columns, key_tuple)),
                "contrast": label,
                "left_value": left_value,
                "right_value": right_value,
                "estimate": float(
                    left_point.loc[index_key, "estimate"]
                    - right_point.loc[index_key, "estimate"]
                ),
                "ci_low": float(np.quantile(finite, 0.025)),
                "ci_high": float(np.quantile(finite, 0.975)),
                "bootstrap_sd": float(np.std(finite, ddof=1))
                if len(finite) > 1
                else np.nan,
                "bootstrap_requested": int(len(values)),
                "bootstrap_valid": int(len(finite)),
                "bootstrap_failed": int(len(values) - len(finite)),
                "bootstrap_seed": int(bundle.seed),
                "block_degrees": float(bundle.block_degrees),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        list(by_columns), kind="mergesort"
    ).reset_index(drop=True)
    result.attrs["replicate_table"] = merged.loc[
        :, [*by_columns, "replicate", "estimate"]
    ].copy()
    if len(result) == 1:
        result.attrs["replicate_estimates"] = merged["estimate"].to_numpy(
            dtype=float
        )
    return result
