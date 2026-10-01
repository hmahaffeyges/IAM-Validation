# PROC-SCORE-01 — outcome (2026-09-30): per-cell floors work on pure cells; reading a cell out of a mixture does not yet

Array scale. 15 blood cell types from the Salas purified arrays (GSE110554, GSE167998). Identity loci per cell: stable across its purified arrays
(SD ≤ 0.05) and distinct from the other blood cells in the atlas (|Δβ| ≥ 0.20); 8,000–60,000 loci. Floor = mean per-locus entropy on its purified
arrays (0.60–0.89 bits on these loci).

**Pure cells: PASS.** Each purified array, held out and read against the others of its cell: A median 1.001, SD 0.029, 90 % within 0.95–1.05.

**Mixtures: FAIL, every method.** 190 cell readings in the 24 known mixtures, against the same cell's purified A:
| method | median |error| (cells 10–20 % of mix) | within ±0.05 overall |
|---|---|---|
| S1 read the mixture directly | 0.27 | 9 % |
| S2/S3 separate β locus by locus (true / solved fractions) | 0.19 / 0.20 | 6 % / 6 % |
| S4/S5 one-parameter fidelity fit (true / solved fractions) | 0.19 / 0.08 | 11 % / 19 % |
Errors carry a sign per cell (CD8 T −0.17 to −0.40; monocytes +0.24 to +0.36) even with the true fractions, so the dominant error is not the solver.

**Why.** Identity loci chosen for being distinct sit near β = 0 or 1, where entropy is steepest (dH/dβ = 4.2 bits per unit β at β = 0.05).
A 1 % error in the separated β there — the atlas mean of the other cells not matching these donors' cells, array noise divided by the cell's
fraction — moves A by far more than the ±0.05 tolerance. The gauge amplifies exactly where these loci sit.

**What this decides.**
1. A is readable today on specimens that are one cell type, or dominated by one: purified or sorted cells, cultures, and (to be tested) tissue.
2. For minor cells in a mixture, report the fraction and withhold A until a method passes this test. Candidates: loci at intermediate β where
   the slope is lower; errors propagated per reading so the interval decides whether a tier is printed.
3. Measuring the Warburg and Breach lines uses cultured or sorted series (one cell type each), so it does not wait on mixture scoring.
