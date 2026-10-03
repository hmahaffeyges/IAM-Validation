# DEV-SKY-01 — stage 11 sky map on chain v3 (development; check written 2026-10-03 before the data were read)

**Commissioning step 5.** Check: healthy replicates give a residual sky consistent with the spatially shuffled null.

**v3 wrapper (no panel, no population).** For one array: f = stage 2 fractions (NNLS8, 8 groups); mu_g = atlas v2 means of the 8 parent entries
(neutrophils, eosinophils, basophils, monocytes, b cells, nk cells, cd4 t cells, cd8 t cells); residual r_i = beta_i - sum_g f_g mu_g,i;
sigma_i = sqrt(sum_g f_g^2 (sd_g,i^2 + donor_sd_g,i^2) + 0.02^2) (atlas posterior and donor SD plus the solver's 0.02 floor; the array's
SNP-probe noise is not available after Stage 1, which keeps cg probes only); z_i = r_i / sigma_i; sky = mean z per HEALPix NSIDE 128 RING pixel
via `Runtime Matrices/Patient_CMB/iamatlas_cpg_to_healpix_nside128.npz`; unmapped pixels masked. Statistics: `sky_statistics.py` (masked
pseudo-C_l, six bands, 20 within-mask permutations). Arrays: GSE250556 (every array read in DEV-BASE-CHAIN-01).

**Bar.** For each of the six bands, the median over arrays of (band power / mean of the array's own 20 shuffles) lies in 0.9-1.1.
Also recorded: the fraction of arrays above the maximum of their shuffles, per band. Pass -> wired as stage 11 (sky record and plate in the
bundle/report). Fail -> not wired (behind a flag only). Prior record: PROC-CLS-01 found large-scale structure on 450K skies (class-era scale).

---
## Outcome (recorded 2026-10-03 after the run; nothing above the line was changed)
Box job 91d7b32f (chain commit 63d55fa; beta vectors from DEV-BASE-CHAIN-01). Records: `data/DEV_TOOLKIT_01/` (`scores.csv`, `repeatability.csv`,
`agreement.csv`, `fractions_long.csv`, `summary.json`, script `toolkit_b.py`).

63 GSE250556 skies (f_sky and pixel counts in `sky_GSE250556.csv`). Median over arrays of band power / own-null mean:
l 2-8 **6.02**, 9-24 **2.95**, 25-64 **1.96**, 65-128 **1.51**, 129-191 **1.14**, 192-255 0.97. Fraction of arrays above the maximum of their 20
shuffles: 100 % in bands 1-5, 0 % in band 6. **FAIL** (bar 0.9-1.1 in every band). The healthy residual sky is not spatially random at large and
intermediate scales, as PROC-CLS-01 found on the class-era sky; the structure is in the healthy replicates, so it is a property of the
expectation (atlas parent templates x stage 2 fractions) or of genomic order, not a departure. **Stage 11 is not wired.** healpy 1.17 was installed
into the box's cosmo-venv for this run (the chain env has no healpy; wiring stage 11 would need it there).
