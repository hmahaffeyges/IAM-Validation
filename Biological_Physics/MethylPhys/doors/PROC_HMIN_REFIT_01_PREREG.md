# PROC-HMIN-REFIT-01 — pre-registration: measuring the methylation floors from sorted cells (option A), before anything moves

**Written 2026-09-30, before any sample is read for this purpose.** Why: the 37 values G-002 was fitted on do not match the data they cite
(`hmin_calibration/provenance_check/`). Author, 2026-09-30: "Lets do A … I dont want to do anything to move the floors too drastically
without being CERTAIN that we are correct." This procedure **measures**. It changes no floor, no identity locus and no reading. What
it produces goes to the author; adoption is a separate decision.

## The quantity (option A)
For one purified-cell sample s: h_s = mean over loci i in U of H(β_si), H the binary entropy in bits. No locus is selected by its value,
so the floor does not choose its own loci. For comparison only (option B): H(mean over U of β_si).

- **U** = the atlas v2 locus universe: array CpGs with Loyfer depth ≥ 10 in > 90 % of Loyfer samples (the loci stage B fitted), the
  same for every sample. A sample contributes the loci it measured.
- **Samples** = every atlas v2 admitted sample (roster_samples.csv, qc True, admitted cell) plus GSE63409 normal HSC/progenitors, each
  on its own platform: arrays through our Stage 1; WGBS at depth ≥ 10.
- **Cell value** h_c = median of its samples' h_s. **Class floor** = the G-002 likelihood optimum over its cells,
  ĥ_k = Σ h_c² / Σ h_c (closed form of G-002's likelihood; the MCMC is run afterwards on the same inputs as a check).

## What could make A wrong, and the check for each (all reported; none tuned after reading)
1. **Platform.** Arrays read a true 0 or 1 away from the extremes (background, dye); WGBS reads them near exactly. Most CpGs sit near 0
   or 1, so this can dominate h. Check C1: the 17 cells measured on both platforms — h on array vs h on WGBS, per cell.
2. **Sequencing depth.** A WGBS β at depth n is a binomial estimate; its entropy is biased by depth. Check C2: WGBS h_s at depth ≥ 10,
   ≥ 20, ≥ 30 on the same samples.
3. **Scale.** WGBS β put on the array scale by its source term ((β − a)/b, clipped) against raw. Check C3: h both ways.
4. **Donors.** Check C4: spread of h_s within a cell against spread between cells of one class and between classes.
5. **Sort purity.** A sorted population contaminated by other cells reads higher h. Reported per cell against its roster metric.

## What goes to the author
Per sample, per cell and per class: A (raw and on the array scale, each depth), B, and C1–C5, beside the current floors. No adoption is
proposed unless C1 shows the two platforms agree on h per cell; if they do not, A must be measured on one platform only, and that choice
comes back to the author with the numbers.
