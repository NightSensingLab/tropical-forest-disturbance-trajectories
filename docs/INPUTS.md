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
`scripts/summarize_response_data.py` combines those saved result tables with the
primary and classification summaries for Figures 4 and S1.

## Monthly Availability Inputs

`scripts/availability.py` decomposes event-year response availability using
`data/primary/subannual/parts_monthly/event_year=YYYY/part.parquet`, for event years
2004-2018. The included panels contain only the t0 rows and 14 columns read by
this script, preserving the monthly observations and their quality indicators.
They contain 626,976 treatment/control role-month records in total. The other
relative years and unused columns are not needed for this diagnostic. The
resulting availability tables are supplied in `results/sensitivity/`.

The repository does not redistribute the original TMF or MODIS source products.
The supplied derived inputs support reproduction of the reported analyses,
while the public source products can be obtained from the provider links in
`README.md`.
