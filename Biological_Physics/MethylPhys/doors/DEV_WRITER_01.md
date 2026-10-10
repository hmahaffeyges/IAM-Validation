# DEV-WRITER-01 — ε0 from the writer's discrimination measured outside the cell (book open item D1; development)

**Sealed 2026-10-10, before any enzyme measurement is looked up** (`development/sims/writer_01.py`, output `writer_01_output.txt`).

**Model (conjecture, the book's form).** One discriminating selection per site per copy, no proofreading (Hopfield's single-step error):
ε = 1/(1+D), so E_hold = k_BT ln D, where D = (k_cat/K_M on hemimethylated CpG) ÷ (k_cat/K_M on unmethylated CpG) of the writer, DNMT1,
measured on the enzyme in vitro. Nothing in D comes from any cell's copy error.
**Simulation finding.** A writer acting on independent sites has one steady state, the same in every territory, so it cannot hold methylated
and unmethylated territories at once; the territories are held by neighbour coupling. Model A therefore claims only that the error at a held
site is set by the writer's single-step discrimination, the territory being held by coupling. A coupled model is the next step if A fails.

**Prediction.** The cells hold E_hold = 3.41 k_BT (canon; healthy cell types 3.18-3.71, copy error 0.024-0.042). Model A requires D = 30.3.
**What counts (fixed before lookup):** human (or mammalian) DNMT1, in vitro, steady-state specificity k_cat/K_M (or the ratio of initial rates at
equal, sub-saturating substrate) on hemimethylated against unmethylated CpG of the same sequence. Every qualifying measurement found is listed;
the median is scored. Truncated catalytic domains without the replication-foci and CXXC regions are listed separately and not scored.
**Bar.** Median D in 15-60 (E_hold 2.72-4.10, a factor of 2 about 30.3): consistent with Model A. Median D below 15: the cell holds its sites
better than the writer alone can, so the extra comes from the partners (UHRF1 recruitment) and costs energy; Model A is not enough. Median D
above 60: the writer alone would hold better than cells do; cells lose more than the writer's errors (loss by other routes). Either outcome is
recorded; neither changes a constant.
