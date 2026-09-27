# PROC-UNMIX-01 — pre-registration: does inverting the dilution line put every present cell's A at 1.00 on constructed truth?

**Written 2026-09-26, before any change to the scorer.** Runs after PROC-TARE-01 has finished scoring (the chain is not edited while a procedure scores). Nothing below moves after results are visible.

## The defect, measured

Three constructed whole-blood mixes composed from the atlas's own cell profiles (no departure in them, so every cell's A should read 1.00) were run through the full chain on 2026-09-26 (results/comp_truth). Majority cells read 1.00; minority cells did not:

| cell | fraction | A read |
|---|---|---|
| Neutrophils_reinius | 0.40-0.76 | 0.997-1.011 |
| CD14_monocytes | 0.04-0.09 | 1.000-1.010 |
| CD4_T-cells | 0.09-0.22 | 1.029-1.032 |
| CD8_T-cells | 0.03-0.14 | 0.961-0.962 |
| CD56_NK-cells | 0.05-0.10 | 0.945-0.947 |

The same ordering was measured on 80 real arrays (PERCELL_CENTRE_HYPOTHESIS_TEST.json): the only cell-specific signal in a cell's A is its own fraction. A minority cell's identity loci carry the majority cells' bytes; A sits on the dilution line between what the rest of the specimen reads at those loci and the cell's own value (FRACTION_AND_A.md).

## What changes, in order

1. **Re-zero the identity loci** so each cell's own atlas profile reads A = 1.000 on its identity loci (today 0.936-1.020, median 0.990, by construction: loci chosen within +-0.05 of H_min_beta). The standard must read 1.00 before an inversion can be judged against 1.00.
2. **Invert the dilution line per present cell**: with fractions f from Stage 2 and the atlas profiles of the other present cells, remove the other cells' contribution at the cell's identity loci before H is taken. The form is fixed here as the linear mixture inversion already used in FRACTION_AND_A.md; no other form is tried after results are visible.

## Bars, fixed now

| bar | requirement |
|---|---|
| B1 | on the three constructed mixes, every present cell reads A within 0.95-1.04 (NORMAL) after (1)+(2) |
| B2 | max abs(A - 1.00) over all present cells in all three mixes <= 0.015 |
| B3 | majority cells (neutrophils, monocytes) move by < 0.005 from their pre-change reading |
| B4 | a constructed mix with a known departure (one cell's profile displaced so its true A = 1.06) reads that cell at 1.06 +- 0.015 and every other cell NORMAL - the inversion must not erase a real departure |
| B5 | on the 48 healthy arrays (four laboratories, mapped), the spread of A per cell does not widen (p90-p10 after <= before for >= 4 of 5 blood cells) |
| B6 | the instrument-unchanged check: PROC-SYNTH-01's pure-cell readings (f = 1.000) change by < 1e-6 - at f = 1 the inversion is the identity |

**Decision rule.** B1-B6 met -> adopted into stage_a_cells; the Every-cell legend gains one sentence saying fraction is removed from A by inversion, and fraction remains a detection gate. Any bar failing -> not adopted; the failing bar recorded; the CD4 ELEVATED on the real array stays flagged as "possible fraction confound" on the page until it is.

## Evidence files (named before they exist)

In the kit folder: PROC_UNMIX_01.py. In the kit results folder: PROC_UNMIX_01.json. In the plates folder: PROC_UNMIX_01.png. Linked from the outcome once they exist.
