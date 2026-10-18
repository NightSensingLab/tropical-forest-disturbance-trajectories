# Tropical Forest Disturbance Trajectories

Analysis code and data for **Beyond event-year greenness loss: post-disturbance
trajectories diverge between degradation- and deforestation-dominated events
in tropical moist forests**.

## Contents

| Directory | Contents |
| --- | --- |
| `src/pipeline/` | Event definitions, matching, response construction, estimation and classification |
| `scripts/` | Scripts for reproducing reported analyses |
| `data/primary/` | Matched events, common-support weights and monthly/annualized responses |
| `data/annual_evi/` | Separate annual EVI windows and censoring sensitivity inputs |
| `results/` | Fixed-cohort, availability, composition and spatial sensitivity results |
| `source_data/` | CSV data for Figures 1-4, Figures S1-S6, Table 1 and Tables S1-S15 |

[`source_data/DISPLAY_INDEX.csv`](source_data/DISPLAY_INDEX.csv) maps each figure
and table to its files. [`manuscript_tables.xlsx`](source_data/tables/manuscript_tables.xlsx)
contains all 16 manuscript tables and an index. Table CSVs preserve the displayed
values; analytical exports retain unrounded estimates.

## Reproduction

Use Python 3.12 and run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/standardize.py
python scripts/classify.py
python scripts/fixed_cohorts.py
python scripts/paired_divergence.py
python scripts/sensitivity.py
python scripts/availability.py
python scripts/bootstrap_sensitivity.py
python scripts/composition.py
python scripts/spatial_diagnostics.py
python scripts/annual_evi.py
python scripts/summarize_response_data.py
```

The scripts use the supplied derived inputs. Primary and classification scripts
write results alongside their input tables. Annual EVI and paired-divergence
recalculations write to `results/recomputed/`; the paired-divergence script accepts
`--output PATH` and requires an empty output directory. Use a separate checkout
for recalculation when retaining the supplied reference outputs.

`summarize_response_data.py` exports the response-construction and classification
summary CSVs used by Figure 4 and the endpoint comparisons in Figure S1 to
`results/recomputed/response_summary/`. `--output PATH` selects another directory.

The primary outcome is the annual mean of monthly matched EVI effects with at
least nine usable months. The annual EVI sensitivity construction uses a separate
annual panel with at least 12 valid observations. The classification cohort
contains 3,881 complete t0-t5 trajectories and is selected independently of the
common-support restriction. Uncertainty follows the analysis-specific spatial
block bootstrap or classification validation described in the manuscript.

[`docs/INPUTS.md`](docs/INPUTS.md) lists the included inputs, including the
event-year monthly panels used by `scripts/availability.py`.
[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) describes units and fields.
Reproduction starts from the supplied derived inputs. The repository does not
redistribute the original TMF or MODIS source products; those products remain
available from their public providers at the links below and should be obtained
under the providers' terms. The included analysis inputs, source data, and
scripts support reproduction of the reported results without access to private
files.

## Licensing

Original analysis code is released under the [MIT License](LICENSE-CODE).
Author-generated derived tables and data products are released under
[Creative Commons Attribution 4.0 International](LICENSE-DATA).
Third-party source products remain under their original provider terms.

## Data Sources

- [JRC Tropical Moist Forest Annual Changes](https://forobs.jrc.ec.europa.eu/TMF/data)
  (Vancutsem et al., 2021), 1990-2023 history from the 2025 update.
- [MODIS/Terra MOD13Q1 Collection 6.1](https://doi.org/10.5067/MODIS/MOD13Q1.061)
  (Didan, 2021), EVI observations for 2001-2023.
- [Natural Earth Admin 0, release v5.1.2](https://github.com/nvkelso/natural-earth-vector/tree/v5.1.2/10m_cultural),
  1:10 million map background. Natural Earth data are in the
  [public domain](https://www.naturalearthdata.com/about/terms-of-use/).
