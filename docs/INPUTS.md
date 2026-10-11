# Analysis Inputs

## Primary Analysis

| Path under `data/primary/` | Purpose |
| --- | --- |
| `event_level/history_qualified_candidates.parquet` | 24,406 history-qualified first-event candidates |
| `event_level/matched_treatments.parquet` | 16,747 matched events and their covariates |
| `event_level/fixed_matched_pairs.parquet` | 50,198 treatment-control links and weights |
| `event_level/common_support_weights.parquet` | 10,970 common-support events and standardization weights |
| `event_level/fixed_matched_event_window.parquet` | Annualized monthly response window |
| `subannual/matched_links.parquet` | Event censoring map |
| `outcomes/annual_matched_effects.parquet` | Annualized monthly matched effects and usable-month counts |
| `outcomes/monthly_matched_effects.parquet` | Monthly matched effects for threshold sensitivity |

The primary analysis requires at least nine usable months in an event-year.
Responses at and after a recurrent threshold-level disturbance are censored.
The annualized window can be regenerated with `scripts/standardize.py`.

`scripts/sensitivity.py` creates the year-specific event table used by
`scripts/bootstrap_sensitivity.py`. Fixed-cohort definitions and supplied results
are in `results/sensitivity/`. Paired A1 results and bootstrap draws are in
`results/a1_paired/`.

## Annual EVI Sensitivities

`data/annual_evi/` contains the separate annual EVI matched window, common-support
weights, pre-event control pairs and window, and stabilized IPCW weights.
Its annual observation threshold is 12. `scripts/annual_evi.py` reproduces the
annual dynamic-censoring, pre-event control and IPCW estimates from these windows.

## Additional Monthly Inputs

`scripts/availability.py` decomposes event-year response availability using
`data/primary/subannual/parts_monthly/event_year=YYYY/part.parquet`, for event years
2004-2018. These grid-level monthly panels are additional intermediates available
from the corresponding author. The resulting availability tables are supplied
in `results/sensitivity/`.

Satellite extraction and reconstruction of matching candidates require the
annual TMF/EVI grid panels, candidate feature parts and extraction scripts.
The supplied inputs support reproduction after those processing steps.
