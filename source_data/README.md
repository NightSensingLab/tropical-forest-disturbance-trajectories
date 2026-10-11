# Figure And Table Source Data

`DISPLAY_INDEX.csv` covers Figure 1-4, Figure S1-S6, Table 1 and Table S1-S15.
Each row records a manuscript caption and the relative paths of its data.

## Tables

`tables/Table_*_manuscript.csv` contains the exact displayed cells extracted
from the selected v3.2 manuscript. `tables/manuscript_tables.xlsx` contains
the same 16 tables plus an index. Single numeric cells are typed as numbers
in Excel, with displayed decimal precision retained; confidence intervals,
combined counts and percentages within compound entries remain text.
Blank cells preserve the manuscript's layout and are not numeric zeros.

These display tables contain rounded values. The `Table_*/source_data/`
subdirectories and included analysis result CSVs provide unrounded estimates
where available. The two forms serve different purposes and should not be
substituted for each other in numerical analysis.

## Figures

The retained v2.5 figure-source exports are combined with v2.7 corrections
for regional negative-block fractions, common-support eligibility and extreme
matching distance. Figure 4 uses the final source exports dated 2026-10-08.
Figure S1 endpoint comparisons use the same four outcome constructions as
the final Figure 4. The selected manuscript table snapshot is documented by
its original DOCX SHA-256 in `docs/DATA_MANIFEST.csv`; the manuscript is not
distributed in this repository.

Primary monthly responses and separate annual EVI responses are distinguished
in Figure 4's `method` and `outcome_construction` fields. A matching event
directory can be shared between constructions without their outcomes being
interchangeable. Counts are event counts unless explicitly labeled ESS.
Classification uses one independent complete t0-t5 cohort of 3,881 events.

`diagnostics/A1_paired/` archives the latest paired A1 t5-minus-t0 inference,
event data and bootstrap draws. Other fixed-cohort and month-threshold
diagnostics are retained at `revision_v28_audit/` for their script inputs.

## Provenance

`docs/DATA_MANIFEST.csv` records original and packaged hashes. Most files
are exact copies; local path prefixes/BOMs are removed where necessary,
and current table cells are extracted without statistical recomputation.
`SHA256SUMS.txt` covers the files in this upload. Source CSV filenames can
retain old generator versions when the final manuscript retained those data.
