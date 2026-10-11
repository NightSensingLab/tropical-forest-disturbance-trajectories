"""Annual EVI censoring sensitivity functions."""
import numpy as np
import pandas as pd
from pipeline.matched_estimation import assign_spatial_blocks, common_support_weights

def _weighted_mean(values, weights):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def _stabilized_ipcw_weights(event_table, relative_years):
    """Estimate transparent baseline-stratified stabilized censoring weights.

    The censoring event is a subsequent threshold crossing.  Probabilities are
    estimated within event type, macroregion and broad event-year period so
    that the sensitivity analysis addresses the observed imbalance in repeat
    disturbance without presenting the weights as a fully causal model.
    """
    required = {
        "treatment_grid_id",
        "event_year",
        "event_type",
        "macroregion",
        "censor_relative_year",
    }
    missing = required - set(event_table.columns)
    if missing:
        raise ValueError(f"Event table lacks IPCW columns: {sorted(missing)}")
    events = event_table.loc[:, sorted(required)].drop_duplicates(
        ["treatment_grid_id", "event_year"]
    ).copy()
    events["event_year"] = pd.to_numeric(events["event_year"], errors="coerce")
    events["censor_relative_year"] = pd.to_numeric(
        events["censor_relative_year"], errors="coerce"
    )
    events["censor_year_period"] = pd.cut(
        events["event_year"],
        bins=[2003, 2008, 2013, 2018],
        labels=False,
        include_lowest=True,
    )
    rows = []
    diagnostics = []
    for relative_year in relative_years:
        relative_year = int(relative_year)
        observed = events["censor_relative_year"].isna() | events[
            "censor_relative_year"
        ].gt(relative_year)
        if relative_year <= 0:
            conditional_probability = pd.Series(1.0, index=events.index)
            marginal_probability = pd.Series(1.0, index=events.index)
            stabilized = pd.Series(1.0, index=events.index)
        else:
            conditional_probability = events.assign(
                _observed=observed.astype(float)
            ).groupby(
                ["macroregion", "event_type", "censor_year_period"],
                dropna=False,
            )["_observed"].transform("mean")
            marginal_probability = events.assign(
                _observed=observed.astype(float)
            ).groupby("event_type", dropna=False)["_observed"].transform("mean")
            stabilized = (marginal_probability / conditional_probability).clip(
                lower=0.25, upper=4.0
            )
        rows.append(
            pd.DataFrame(
                {
                    "treatment_grid_id": events["treatment_grid_id"].to_numpy(),
                    "event_year": events["event_year"].to_numpy(),
                    "event_type": events["event_type"].to_numpy(),
                    "macroregion": events["macroregion"].to_numpy(),
                    "censor_year_period": events["censor_year_period"].to_numpy(),
                    "relative_year": relative_year,
                    "observed_uncensored": observed.to_numpy(dtype=bool),
                    "p_uncensored_conditional": conditional_probability.to_numpy(
                        dtype=np.float64
                    ),
                    "p_uncensored_marginal": marginal_probability.to_numpy(
                        dtype=np.float64
                    ),
                    "stabilized_ipcw": stabilized.to_numpy(dtype=np.float64),
                }
            )
        )
        diagnostics_frame = pd.DataFrame(
            {
                "event_type": events["event_type"].to_numpy(),
                "observed_uncensored": observed.to_numpy(dtype=bool),
                "conditional_probability": conditional_probability.to_numpy(
                    dtype=np.float64
                ),
                "stabilized_ipcw": stabilized.to_numpy(dtype=np.float64),
            }
        )
        for event_type, group in diagnostics_frame.groupby("event_type", sort=True):
            diagnostics.append(
                {
                    "event_type": str(event_type),
                    "relative_year": relative_year,
                    "event_n": int(len(group)),
                    "uncensored_n": int(group["observed_uncensored"].sum()),
                    "marginal_uncensored_fraction": float(
                        group["observed_uncensored"].mean()
                    ),
                    "conditional_probability_min": float(
                        group["conditional_probability"].min()
                    ),
                    "conditional_probability_max": float(
                        group["conditional_probability"].max()
                    ),
                    "stabilized_ipcw_mean": float(group["stabilized_ipcw"].mean()),
                    "stabilized_ipcw_max": float(group["stabilized_ipcw"].max()),
                    "weight_truncated_fraction": float(
                        (group["stabilized_ipcw"].isin([0.25, 4.0])).mean()
                    ),
                    "model": "baseline-stratified stabilized IPCW; strata=event type x macroregion x 5-year event period",
                }
            )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(diagnostics)


def _ipcw_event_type_contrast(window, event_table, settings, n_bins=5):
    """Compute a block-bootstrap contrast after stabilized IPCW sensitivity weighting."""
    effect_column = "pair_effect_evi_mean"
    standardized, _ = common_support_weights(event_table, n_bins=n_bins)
    event_weights = standardized.loc[
        :, ["treatment_grid_id", "event_year", "standardized_event_weight"]
    ].merge(
        event_table.loc[
            :, ["treatment_grid_id", "event_year", "event_type", "lon", "lat"]
        ].drop_duplicates(["treatment_grid_id", "event_year"]),
        on=["treatment_grid_id", "event_year"],
        how="left",
        validate="one_to_one",
    )
    event_weights["spatial_block"] = assign_spatial_blocks(event_weights, 2.0)
    ipcw, diagnostics = _stabilized_ipcw_weights(event_table, settings.relative_years)
    rows = []
    rng = np.random.default_rng(int(settings.bootstrap_seed) + 23)
    for relative_year in settings.relative_years:
        frame = window.loc[
            window["relative_year"].eq(relative_year),
            ["treatment_grid_id", "event_year", effect_column],
        ].merge(
            event_weights,
            on=["treatment_grid_id", "event_year"],
            how="inner",
            validate="one_to_one",
        ).merge(
            ipcw.loc[
                ipcw["relative_year"].eq(relative_year),
                ["treatment_grid_id", "event_year", "stabilized_ipcw"],
            ],
            on=["treatment_grid_id", "event_year"],
            how="inner",
            validate="one_to_one",
        )
        frame = frame.loc[frame[effect_column].notna()].copy()
        if frame.empty:
            continue
        frame["ipcw_weight"] = (
            pd.to_numeric(frame["standardized_event_weight"], errors="coerce")
            * pd.to_numeric(frame["stabilized_ipcw"], errors="coerce")
        )
        frame = frame.loc[
            np.isfinite(frame[effect_column].to_numpy(dtype=np.float64))
            & np.isfinite(frame["ipcw_weight"].to_numpy(dtype=np.float64))
            & frame["ipcw_weight"].gt(0.0)
        ].copy()
        blocks = np.sort(frame["spatial_block"].unique())
        if len(blocks) == 0:
            continue
        point = {}
        block_sums = {}
        for event_type in ("degradation", "deforestation"):
            group = frame.loc[frame["event_type"].eq(event_type)].copy()
            values = pd.to_numeric(group[effect_column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            weights = pd.to_numeric(group["ipcw_weight"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            point[event_type] = _weighted_mean(values, weights)
            group["_num"] = values * weights
            group["_den"] = weights
            block_sums[event_type] = group.groupby("spatial_block")[["_num", "_den"]].sum().reindex(
                blocks, fill_value=0.0
            )
        if not all(np.isfinite(point.get(kind, np.nan)) for kind in ("degradation", "deforestation")):
            continue
        draw_counts = rng.multinomial(
            len(blocks),
            np.full(len(blocks), 1.0 / len(blocks)),
            size=settings.bootstrap_replicates,
        )
        draws = {}
        for event_type in ("degradation", "deforestation"):
            numerators = draw_counts @ block_sums[event_type]["_num"].to_numpy()
            denominators = draw_counts @ block_sums[event_type]["_den"].to_numpy()
            draws[event_type] = np.divide(
                numerators,
                denominators,
                out=np.full(len(draw_counts), np.nan),
                where=denominators > 0.0,
            )
        contrast_draws = draws["deforestation"] - draws["degradation"]
        for event_type, estimate in (
            ("degradation", point["degradation"]),
            ("deforestation", point["deforestation"]),
            (
                "deforestation_minus_degradation",
                point["deforestation"] - point["degradation"],
            ),
        ):
            draw_values = (
                contrast_draws
                if event_type == "deforestation_minus_degradation"
                else draws[event_type]
            )
            finite = draw_values[np.isfinite(draw_values)]
            rows.append(
                {
                    "event_type": event_type,
                    "metric": "evi_mean",
                    "relative_year": int(relative_year),
                    "estimate": float(estimate),
                    "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "bootstrap_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                    "requested_replicates": int(settings.bootstrap_replicates),
                    "valid_replicates": int(len(finite)),
                    "spatial_block_count": int(len(blocks)),
                    "weighting": "standardized event area x stabilized IPCW",
                    "censoring_model": "event type x macroregion x 5-year event period",
                    "treatment_count": int(frame.loc[frame["event_type"].eq(event_type), "treatment_grid_id"].nunique()) if event_type in {"degradation", "deforestation"} else np.nan,
                    "effective_sample_size": float(
                        frame.loc[frame["event_type"].eq(event_type), "ipcw_weight"].sum() ** 2
                        / np.sum(frame.loc[frame["event_type"].eq(event_type), "ipcw_weight"].to_numpy() ** 2)
                    ) if event_type in {"degradation", "deforestation"} else np.nan,
                }
            )
    return pd.DataFrame(rows), ipcw, diagnostics
