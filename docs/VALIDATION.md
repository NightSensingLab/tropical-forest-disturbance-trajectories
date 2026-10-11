# Local Assembly Validation

Date: 2026-10-11. Environment: Python 3.12.14; dependencies recorded in
`requirements.txt`. Initial code collection was followed by the authorized
figure/table data upload. Checks read frozen inputs without writing scientific
results. Selected inputs were copied into the publication package.

From the original project root, the successful verification command was:

```bash
02_analysis/integrated/TropicalRainforest_integrated_analysis/.venv/bin/python -B github_submission/tools/check_code_package.py --source-root . --reference-check
```

## Package Checks

| Check | Result |
| --- | --- |
| Python syntax/compilation | 39 files passed: 37 selected scripts/modules plus two checkers |
| Selected module imports | All 37 selected scripts and the data checker passed |
| Manifest entries | 38: 37 Python files plus configuration |
| Original and packaged SHA-256 | All manifest entries matched |
| Scientific function AST comparison | 370 retained functions matched their source |
| Private machine paths/key patterns | None detected in packaged content |
| Private paths or caches | None detected; documented Parquet/XLSX inputs are intentional |
| Final checker errors | 0 |

The function comparison normalizes the documented Earth Engine project
replacement and removed figure-alignment calls. It excludes the removed
Figure 4 label-audit helper and the added Earth Engine configuration guard.
New diagnostic entry points wrap unchanged extracted functions. Imports were
checked with result writes and network connections blocked; Matplotlib could
query system fonts and write only to a temporary cache.

## Read-Only Reference Checks

| Quantity | Reconstructed value |
| --- | ---: |
| First-event candidates | 24,406 |
| Matched treatments | 16,747 |
| Fixed matched links | 50,198 |
| Common-support eligibility events | 10,970 |
| Primary t0 observable events | 6,581 |
| Primary t5 observable events | 5,000 |
| Primary t0 deforestation-minus-degradation | -0.005737585035259646 |
| Primary t5 deforestation-minus-degradation | -0.018801990372398703 |
| Complete-trajectory classification events | 3,881 |
| Classification degradation / deforestation | 2,677 / 1,204 |
| Fixed A1 complete t0-t5 events | 2,469 |
| Fixed A2 complete t0, t3, t5 events | 3,468 |

Primary type-specific means, contrasts and sample counts matched the saved
current reference table, with numerical tolerances of `atol=1e-12` and
`rtol=1e-10`. Classification and fixed-cohort checks rebuilt eligibility only;
they did not fit a classifier or run bootstrap inference.

## Scope

No complete scientific run, bootstrap CI recomputation, annual EVI sensitivity
run, figure rendering, satellite extraction or clean-environment installation
was performed. These checks establish the selected code's consistency with
its sources and the listed frozen-input results. Full reproduction requires
the inputs in `INPUTS.md` and further execution in a release environment.

## Publication Data Checks

The current data package maps 10 figures and 16 tables in `DISPLAY_INDEX.csv`.
It contains 139 selected data CSVs, 13 frozen Parquet inputs, the fixed-cohort
manifest and a 17-sheet Excel workbook. A separate data manifest records
source and packaged hashes. Workbook verification compared all 1,146 table
cells against the exact extracted CSV cells and found zero differences.
Every worksheet was rendered for visual inspection; all table sheets contain
the expected headers and data. No spreadsheet formulas were introduced.

`tools/check_publication_data.py` verifies file hashes, complete display
coverage, rectangular CSV records, private paths, workbook cells and release
checksums. The package is intended for direct upload to the repository's main
branch under the user's authorization.
