# Selection And Provenance

Assembly date: 2026-10-11. Target: the final manuscript's current analysis
chain, including the 2026-09-09 monthly EVI run, v28 fixed-cohort diagnostics,
v32 A1 paired inference and the 2026-10-08 Figure 4 generator.

The package contains 37 selected Python source files and one matching
configuration. Packaging documentation and the checker are new files.
`SOURCE_MANIFEST.csv` provides one row for each selected source/configuration
file. Its source paths are relative to the original project; destination
paths are relative to this package. Both versions are hashed with SHA-256.

## Adaptations

- Preserve the original project-relative layout and script names.
- Resolve the historical screening root relative to its script.
- Read Earth Engine projects from `TFDT_EE_PROJECT` and the export directory
  from `TFDT_DRIVE_ROOT`; use the active Python interpreter.
- Convert configuration paths to repository-relative paths. Run configured
  historical workflows from the repository root.
- Remove machine-local Codex figure-alignment imports and checks from the
  fixed-cohort and Figure 4 scripts. These were presentation audit hooks.
- Extract existing scientific functions from manuscript audit scripts into
  independent analysis entry points. Omit DOCX and historical package audits.
- Extract matching diagnostics into a standalone entry point, correcting its
  root resolution. Matching selection, distance calculations, equal within-set
  weighting and balance formulas are retained from the source.

The original project scripts and scientific data were not edited. No fitting,
bootstrap, extraction or complete scientific pipeline was rerun during
assembly. Read-only numerical checks are described in `VALIDATION.md`.

## Known Limitations

1. Selected frozen inputs and all current figure/table CSVs are included.
   Additional extraction/rematching inputs are listed in `INPUTS.md`.
2. Original annual EVI pixel-export code and original 2001-2023 TMF pixel-export
   code were not recovered. Reproduction starts from frozen gridded inputs.
3. `pipeline/full_history_sensitivities.py::_auc()` uses ordinal ranks for tied
   scores. Existing project audits found no resulting change in the six
   reported classification AUCs. The manuscript implementation is retained;
   this function is not a general tie-correct AUC implementation. A later
   correction should verify the reported outputs before changing the release.
4. Local versions are recorded, but installation in a new environment and
   optional network extraction dependencies have not been tested.
5. Current figure/table source data were added for this upload. A complete
   figure-rendering workflow, DOI/release metadata and author-selected
   licensing remain to be prepared.
