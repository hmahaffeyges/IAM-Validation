# DEV-POSITION-EXPR-01: each healthy cell type's copy error from its eraser-to-writer balance, read from public RNA (2026-10-14; development)

**Why.** IAM-A needs each cell type's healthy position measured on purified single-molecule data, the scarcest data there is. If the copy error
is the writer's floor plus the eraser's share (PHY-040; the middle mark in DEV_ERASE_01.md), a cell's position follows from how much eraser it
makes against writer: RNA expression, which public atlases hold for hundreds of cell types. That would replace the scarcest input with the most
plentiful one.

**Prediction (committed before any expression value is read):** eps_cell = (1 + r) / (D + 1 + r), D = 63.7 (DEV_WRITER_01_OUTCOME.md),
r_cell = X_cell / median X, X = (TET1 + TET2 + TET3) / (DNMT1 + UHRF1) from the Human Protein Atlas single-cell-type table. Nothing fitted to
any copy error. MET if the one-sided permutation p of Spearman rho(predicted, measured) over the matched points is below 0.01. The full rule,
crosswalk and reported checks (writer alone, eraser alone, a proliferation confound) are in `data/DEV_POSITION_EXPR_01/position_expr_01.py`.

**Synthetic test:** DEV_POSITION_EXPR_01/planted_position_expr_01.py, DEV_POSITION_EXPR_01/planted_position_expr_01.txt

**Prior reading:** the 56 measured copy errors (PROC-CHANNEL-01, Loyfer 2023 GSE186458) were read before; this is development on them. The
Human Protein Atlas tables were downloaded and only their cell-type names listed (to write the crosswalk) before this was committed.

**What it decides.** MET: expression carries the cell's position, and a derived reference for any cell type from public RNA becomes the next
step (with a held-out cell set). NOT MET: the position is not read from bulk expression of these five genes; the middle mark (DEV-ERASE-01)
is unaffected, because it concerns the average, not the spread between cell types.

## Outcome, run once (2026-10-14): NOT MET

`position_expr_01_summary.txt`, `position_expr_01_points.csv`. The rule was committed locally (e3243ef1, 04:48 Pacific) before the run; the push
gate refused it for a missing plain-words paragraph and it was pushed with this outcome, unchanged.

| | |
|---|---|
| matched points | 30 |
| Spearman rho, predicted against measured | **+0.101**, one-sided p = 0.29650, bar p < 0.01: **NOT MET** |
| measured copy error | median 0.0314, range 0.0241-0.0417 |
| predicted copy error | median 0.0304, range 0.0174-**0.1873** (r from 0.13 to 13.7) |
| reported: writer alone (1 / (DNMT1 + UHRF1)) | rho +0.358 |
| reported: eraser alone (TET sum) | rho -0.084 |
| reported: proliferation (MKI67) | rho -0.109; prediction with it partialled out +0.068 |
| secondary, memory minus naive | CD4 same sign, B same sign, CD8 opposite |

**In plain words.** We asked whether a cell type's healthy copy error can be read from how much of the erasing enzymes it makes against the
writing ones, using public RNA. It cannot, in this form: the ranking is close to chance, and the predicted errors spread more than ten times
wider than the measured ones, which stay within about 12 percent of each other. The cells hold their copy error far more tightly than their
enzyme RNA varies, so the balance is not set by bulk enzyme amounts. The average middle mark (DEV-ERASE-01) is a separate question and is
unaffected. The writer-alone reading (rho +0.36) was not the test and was seen only after the run, so it is a lead, not a result.
What it would take to succeed: a reading of each cell's actual erasing and writing activity at its methylated sites (for example
5hmC maps, or same-cell enzyme protein), not bulk RNA, tested by a new rule committed before that data is read; this test is finished.
