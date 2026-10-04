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
