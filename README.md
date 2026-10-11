# Tropical Forest Disturbance Trajectories

Analysis code, figure/table source data and frozen derived inputs for the
final manuscript, assembled on 2026-10-11.

## Source Data

`source_data/DISPLAY_INDEX.csv` maps all 26 displays: Figure 1-4, Figure S1-S6,
Table 1 and Table S1-S15. Figure CSVs are under `source_data/figures/`.
Current manuscript table cells are provided as individual CSVs and a
17-sheet Excel workbook at `source_data/tables/manuscript_tables.xlsx`.
Table directories also contain the corresponding unrounded analytical exports.
The table snapshot is the selected v3.2 manuscript; Figure 4 uses the final
2026-10-08 source data. Read `source_data/README.md` for versions and definitions.

Frozen primary event, matching, weight and outcome files are included at
their original relative paths under `runs/20260909-extended-sensitivities/`.
The separate annual EVI windows/weights and reference tables are under
`runs/20260904-expanded-control-pool/`. Data source hashes and operations are
recorded in `docs/DATA_MANIFEST.csv`.

## Analysis Entry Points

Paths are relative to this repository root. Run commands from this root.
The original directory layout is retained to preserve imports and input paths.

| Analysis | Entry point |
| --- | --- |
| Historical TMF screening, 1990-2023 | `runs/20260904-historical-tmf-screening/code.py` |
| Expanded control universe and history | `runs/20260904-expanded-control-pool/build_expanded_control_universe.py`, `extract_expanded_control_history.py`, `build_expanded_history_inputs.py` |
| First-event matching and frozen common support | `runs/20260904-expanded-control-pool/run_expanded_matching.py` |
| Current MODIS catalog and incremental extraction | `runs/20260909-extended-sensitivities/build_current_sensitivity_inputs.py`, `submit_current_modis_increment.py`, `assemble_current_stage08.py` |
| Monthly matched effects and annualization | `runs/20260909-extended-sensitivities/run_current_stage09.py` |
| Primary standardized trajectories, regions and bootstrap | `runs/20260909-extended-sensitivities/recompute_current_standardized.py` |
| Complete-trajectory classification | `runs/20260909-extended-sensitivities/recompute_current_classification.py` |
| Separate annual EVI risk sets and IPCW | `runs/20260904-expanded-control-pool/run_expanded_sensitivities.py` |
| Matching distance and balance diagnostics | `runs/20260909-extended-sensitivities/run_matching_diagnostics.py` |
| Fixed A1/A2 cohorts | `revision_v28_audit/run_fixed_cohort_sensitivity.py` |
| A1 paired t5-minus-t0 divergence | `revision_v32_audit/p0_3_a1_trajectory_test/run_a1_divergence_test.py` |
| Month thresholds and year-specific standardization | `revision_v28_audit/run_v28_core.py`, then `bootstrap_v28.py` |
| t0 monthly availability | `revision_v28_audit/t0_availability_decomposition.py` |
| Regional negative-block fractions and distance sensitivity | `audit/v27/figures_tables/run_analysis_diagnostics.py` |
| Year-specific cohort composition | `audit/v27/risksets/run_composition_diagnostics.py` |
| Final Figure 4 | `03_publication/figure4_revision_20261008/generate_figure4.py` |

Shared analysis modules are under
`02_analysis/integrated/TropicalRainforest_integrated_analysis/pipeline/`.
Names such as `manuscript_v07_statistics`, `revision_v1_analysis` and
`major23_revision` are retained because the selected current scripts import
their implementations. They are dependencies, not alternative primary runs.

## Environment

The package was checked with Python 3.12.14 and the versions in
`requirements.txt`. These are installed local versions; a clean environment
installation has not yet been tested.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B tools/check_code_package.py
python -B tools/check_publication_data.py
```

Earth Engine extraction additionally needs `requirements-extraction.txt`,
Earth Engine authentication, and an authorized project:

```bash
python -m pip install -r requirements-extraction.txt
export TFDT_EE_PROJECT=your-earth-engine-project
export TFDT_DRIVE_ROOT=/path/to/local/export/directory
```

`TFDT_DRIVE_ROOT` defaults to `external_exports/`. The named export subfolder
is retained in the extraction scripts. Extraction is incremental and also
needs archived previous Stage08 exports; see `docs/INPUTS.md`.

## Reproduction From Frozen Inputs

The included frozen inputs support these commands from the repository root:

```bash
python runs/20260909-extended-sensitivities/recompute_current_standardized.py
python runs/20260909-extended-sensitivities/recompute_current_classification.py
python revision_v28_audit/run_fixed_cohort_sensitivity.py
python revision_v32_audit/p0_3_a1_trajectory_test/run_a1_divergence_test.py
python revision_v28_audit/run_v28_core.py
python revision_v28_audit/bootstrap_v28.py
python audit/v27/risksets/run_composition_diagnostics.py
python audit/v27/figures_tables/run_analysis_diagnostics.py
python 03_publication/figure4_revision_20261008/generate_figure4.py
```

These commands write analysis outputs and replace reference CSVs. Use a fresh
working copy when reproducing results and preserve the archived files for
comparison. The A1 divergence script refuses to
overwrite an existing output set. Stage09 can be rebuilt with
`run_current_stage09.py` when both monthly and 16-day Stage08 parts are present.
Annual EVI sensitivity, extraction, distance reconstruction and Figure 4 have
additional inputs listed in `docs/INPUTS.md`.

Important distinctions:

- Primary responses are annualized **monthly** matched effects, requiring at
  least nine usable months. Figure 4 also compares a separate annual EVI
  construction. Identically named event-window files in the two run directories
  contain different responses and must not be interchanged.
- The classification cohort requires a complete t0-t5 trajectory and is
  independent of the frozen common-support eligibility filter.
- This collection supports work from frozen gridded inputs. The original
  annual EVI exporter and original 2001-2023 TMF pixel exporter were not recovered;
  this package does not establish end-to-end reproduction from satellite pixels.

## Provenance And Status

`docs/SOURCE_MANIFEST.csv` records each selected code source, destination, original
SHA-256, packaged SHA-256 and portability adaptation. `docs/PROVENANCE.md`
describes the selection and known limitations. `docs/VALIDATION.md` records
the checks performed during assembly. `SHA256SUMS.txt` covers the published
files. `python -B tools/check_code_package.py --reference-check` checks primary
endpoint estimates and cohort counts using the included frozen inputs.

This repository provides reproduction from frozen derived inputs. Rebuilding
satellite extraction, candidate searches and grid-level monthly response
aggregation requires additional intermediates listed in `docs/INPUTS.md`.
Figure 1-3 and supplementary figure source CSVs are included; their rendering
scripts have not been assembled as a complete figure-generation workflow.
No code/data license or DOI has been assigned in this upload.

## Upstream Data

- [JRC Tropical Moist Forest Annual Changes](https://forobs.jrc.ec.europa.eu/TMF/data)
  (Vancutsem et al., 2021).
- [MODIS/Terra MOD13Q1 Collection 6.1](https://doi.org/10.5067/MODIS/MOD13Q1.061)
  (Didan, 2021).

Target repository:
[NightSensingLab/tropical-forest-disturbance-trajectories](https://github.com/NightSensingLab/tropical-forest-disturbance-trajectories).
