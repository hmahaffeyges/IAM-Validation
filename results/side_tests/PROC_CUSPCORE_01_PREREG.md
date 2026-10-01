# PROC-CUSPCORE-01 — pre-registration (written 2026-10-01, before any slope is fitted)

**Claim tested (dated 16 Mar 2026, tests/iam_cusp_core_sigma2_prediction.py):** r_core ∝ σ² exactly, zero free parameters; normalisation open
(~130×). Also claim 4: galaxies with the same σ have the same core radius regardless of morphology or history.

**Data:** Oh et al. 2015 (AJ 149, 180), LITTLE THINGS, Table 2, all 26 galaxies: pseudo-isothermal core radius R_C (col. 9, with error) and
V_max (col. 3). Rows parsed from the published PDF. Galaxies with R_C error ≥ R_C are reported but excluded from the fit (stated now).
**σ (fixed now):** σ = V_max/√2 (isothermal halo). The constant does not change the slope. The script's own σ values (22–90 km/s for dwarfs)
are of this kind; there is no measured central-black-hole σ for these dwarfs.

**Predictions.**
- P1 (scaling): ordinary least-squares slope of log R_C on log σ, with bootstrap 95 % CI. PASS if the CI contains 2.0.
- P2 (claim 4, same σ → same core): the scatter of log R_C about the best line is ≤ 0.15 dex (a factor 1.4). Reported descriptively alongside
  the measurement-error contribution.
- Also reported: the slope with orthogonal (BCES-bisector-type) regression, and the normalisation r_obs/r_IAM per galaxy.
**Stated limits now:** 26 dwarfs over a factor ~10 in V_max; R_C depends on the halo model (ISO); no black-hole masses for these galaxies, so
the M–σ step inside the claim is not tested, only the end-to-end scaling.
