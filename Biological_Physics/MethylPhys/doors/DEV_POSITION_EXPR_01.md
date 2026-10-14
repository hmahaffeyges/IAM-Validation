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
