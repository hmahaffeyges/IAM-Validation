# DEV-SKY-02 - sky map against a within-chromosome block-shuffle null; sky statistics (development, 2026-10-04)

**DEVELOPMENT - not commissioned.** Development round 2 of chain v3 (author ruling O: test-only mode; no sealed pre-registration, no verdict words). Checks written 2026-10-04 before any data were read; the outcome goes under the line in this note; nothing above the line changes after reading.

**Why (author decision J).** DEV-SKY-01 tested the healthy residual sky against a null that permutes pixel values freely. That null
destroys the correlation neighbouring sites share along a chromosome; real genomic order keeps it. The null that matches the physics keeps each
site's chromosome and its local run of neighbours and moves only whole runs.

**Sky (unchanged from DEV-SKY-01).** Residual z_i = (beta_i - sum_g f_g mu_g,i) / sigma_i (atlas v2 parent means, NNLS8 fractions, atlas posterior
and site SD plus 0.02), mean z per HEALPix NSIDE 128 RING pixel through `iamatlas_cpg_to_healpix_nside128.npz`; unmapped pixels masked; six bands.

**Null (set now).** Sites in genomic order (EPIC hg38 manifest position) within each chromosome are cut into runs of 50 consecutive measured sites;
the runs are permuted within the chromosome; each site keeps its z and takes the pixel of the position it lands on. 20 shuffles per array.
**Bar (as DEV-SKY-01):** for each band, the median over the GSE250556 arrays of (band power / mean of its own 20 shuffles) in 0.9-1.1.

**Sky statistics (stage 12, built now).**
- Mask: unmapped pixels (as stage 11); the pixel count and sky fraction are printed.
- Spectrum: masked pseudo-C_l (`sky_statistics.masked_spectrum`), six bands.
- Look-elsewhere by simulation: T = max over the six bands of (band power / mean band power of the shuffles); p = fraction of 100 block-shuffles of
  the same array with T_shuffle >= T. Structure beyond the null is read when p < 0.05.
- **Bar:** on healthy arrays (GSE250556 and the healthy-labelled whole bloods of DEV-BASE-CHAIN-01 in other series, up to 100), the rate of
  "structure beyond the null" <= 0.05 + 2 x its binomial standard error.
Stage 11 enters behind a development flag if its bar holds; stage 12 likewise; otherwise both stay out of the reading.

---
## Outcome (recorded 2026-10-04 after the run; nothing above the line was changed)
Box job 5bd710b8. Records: `data/DEV_SKY_02/sky02.csv` (one row per array: band ratios to the block null, to the free null for GSE250556, look-elsewhere T and p).

| band (l) | 2-8 | 9-24 | 25-64 | 65-128 | 129-191 | 192-255 |
|---|---|---|---|---|---|---|
| GSE250556 median, power / block-shuffle null (bar 0.9-1.1) | 1.84 | 1.03 | 1.14 | 1.15 | 1.08 | 1.05 |
| GSE250556 median, power / free shuffle (DEV-SKY-01 null) | 6.26 | 2.96 | 1.96 | 1.51 | 1.14 | 0.97 |

- \measured Against the block-shuffle null, bands 2, 5 and 6 sit inside 0.9-1.1; bands 1, 3 and 4 sit outside (1.84, 1.14, 1.15). The bar is not met.
- \observed Keeping the local run of neighbours removes most of the excess DEV-SKY-01 saw (l 9-24: 2.96 -> 1.03; l 2-8: 6.26 -> 1.84). What is left is at the
  largest scales: structure longer than a run of 50 sites inside a chromosome.
- \measured Look-elsewhere by simulation (built): structure beyond the null (p < 0.05) on **91 %** of 163 healthy arrays (GSE250556 100 %, 100 other healthy
  whole bloods 86 %; bar <= 8.4 %). The statistic is driven by band 1.
- \conjecture The large-scale residual is the expectation's own error spread along whole chromosome arms (atlas parent templates x stage 2 fractions), not
  noise; a null with longer runs, or a per-arm expectation, is the next test.

**Wiring.** Stage 11 and stage 12 behind `--dev-sky` (needs `--atlas-v2` and healpy). Not part of the reading.
