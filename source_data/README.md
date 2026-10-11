# Figure and Table Data

`DISPLAY_INDEX.csv` maps Figures 1-4, Figures S1-S6, Table 1 and Tables S1-S15
to their source files, using the captions in the submitted manuscript and
supplementary materials.

`tables/Table_*_manuscript.csv` contains the displayed table cells.
`tables/manuscript_tables.xlsx` contains the same 16 tables and an index.
Numeric cells retain the manuscript's decimal precision. Compound entries such
as confidence intervals remain text, and blank cells remain blank.

`figures/` and `tables/Table_*/source_data/` provide analytical source data.
EVI differences are dimensionless. Event counts and effective sample sizes are
distinct. Primary monthly responses and separate annual EVI responses are
identified by the `method` and `outcome_construction` fields in Figure 4's files.
Classification compares the same 3,881 events across all validation schemes.
