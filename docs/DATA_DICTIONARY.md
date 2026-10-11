# Frozen Analysis Data Dictionary

`PARQUET_SCHEMAS.csv` records every included Parquet file's row count, column
name and Arrow type. The key scientific meanings below apply across the
primary tables; source code contains the full eligibility and weighting rules.

| Field | Meaning / unit |
| --- | --- |
| `grid_id`, `treatment_grid_id`, `control_grid_id` | Grid identifiers; treatment/control roles are explicit in the field name |
| `event_year` | Calendar year of the first qualifying event |
| `relative_year` | Calendar year relative to the event; t0 is the event year |
| `event_type` | Degradation- or deforestation-dominated event label |
| `macroregion` | Prespecified region code; proxy labels are retained in analytical files |
| `event_area_m2`, `cell_area_m2` | Event or cell area in square metres |
| `treatment_lon`, `treatment_lat`, `lon`, `lat` | Coordinates in degrees |
| `disturbance_share` | Annual D+F disturbance fraction |
| `event_composition_purity` | Dominant disturbance process share |
| `baseline_undist_share` | Baseline undisturbed forest fraction |
| `within_set_weight` | Frozen normalized control weight within a treatment's match set |
| `standardization_multiplier` | Common-support event-type standardization factor |
| `standardized_event_weight` | Event area multiplied by the frozen standardization factor |
| `censor_relative_year` | Recurrent threshold-level disturbance censor year; retain outcome years strictly before it |
| `calendar_month` | Calendar month of a monthly EVI observation |
| `matched_effect` | Monthly treatment response relative to the weighted matched controls |
| `annual_matched_effect` | Mean of usable monthly matched effects, at least nine months for the primary annual construction |
| `usable_months` | Number of usable monthly matched effects in an event-year |
| `pair_effect_evi_mean` | Outcome field in the matched event window; construction depends on its run directory |

EVI and EVI differences are dimensionless, with the MODIS native scale factor
already applied. Fractions are on 0-1 unless a CSV header explicitly states
percent. Missing outcome values are unavailable observations, not zero effects.
A missing censor year indicates no recorded subsequent censoring in the
observed history. No-data scientific quantities remain missing in source files.

## Construction-Specific Paths

- `runs/20260909-extended-sensitivities/analysis_current_full_history/fixed_matched_event_window.parquet`
  contains the primary annualized monthly response after common support and
  dynamic censoring.
- `runs/20260904-expanded-control-pool/fixed_matched_event_window.parquet`
  contains the separate annual EVI response. Its annual observation threshold
  is 12 and its risk set differs from the primary response.
- `runs/20260904-expanded-control-pool/pre_event_dynamic_control_window.parquet`
  contains the separate annual EVI response under the pre-event control rule.

Identical field names do not make these outcomes interchangeable. The full
matched cohort, common-support eligibility cohort, year-specific observable
samples and complete-trajectory classification cohort are distinct.

## Result Tables

Unrounded result CSVs use `estimate`, `ci_low`, `ci_high`, `relative_year`
and event-type/region/method labels. `deforestation_minus_degradation` denotes
the direct contrast in that order. CI columns contain percentile bootstrap
intervals where the analysis specifies bootstrap inference; between-fold
classification SD is not a confidence interval. Rows or fields labeled ESS
report an effective sample size rather than an event count.
