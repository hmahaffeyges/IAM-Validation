# THREE_WAY_CLUSTER_CHECK — "Three-Way Mass Discrepancy in Galaxy Clusters as a Test of µ < 1, Σ = 1: eROSITA X-ray, Planck SZ, and DES Weak Lensing"
(25 Feb 2026, 8 pp). Read in full 2026-10-02 (PDF text 304 lines, 50-line ledger, no gaps). Placement: predictions appendix (author, 2026-10-02).

## Reproduced
- Table 1 R(z) = 1/µ(z): 1.158 (z 0) … 1.002 (z 2). dR/dz at z = 0.3 = −0.183 (paper −0.18).
- Table 2 IAM-only and IAM × C_NT, C_NT = 1 + 0.20(1+z)^0.2 at bin centres: 1.116/1.346, 1.094/1.323, 1.068/1.297, 1.039/1.269 (all four exactly).
  dC_NT/dz = +0.027 to +0.036 (paper +0.02 to +0.04).

## Corrections and holds
1. **Form of the term (hold; same as LENSING_DYNAMICS_CHECK #1 and the Virial check).** §2 states the Level 2 form (a friction term in the matter
   equations, photon sector untouched); §3.1 then uses M_hydro = µM_true, which needs µ in the Poisson equation (G_eff form). In the friction form the
   potential obeys the standard Poisson equation, gas in hydrostatic equilibrium feels the true acceleration and R = 1. The abstract's own statement
   ("offsets follow the same collisionless dynamics as in ΛCDM; differences arise only through growth history") agrees with R = 1, not with Eq. 8.
2. **§6.2 "M_SZ/M_hydro = 0.99 ± 0.04 — the first test already passes"**: §1 says SZ masses are calibrated on hydrostatic X-ray masses through Y–M, so
   M_SZ/M_hydro ≈ 1 holds by calibration in any theory; it cannot test the sector split. Value and source (Bulbul 2024) to be traced.
3. **Table 2 "observed" values** (1.28 ± 0.15, 1.22 ± 0.12, 1.25 ± 0.10, 1.18 ± 0.13) are attributed jointly to four papers without per-bin sources; trace.
4. **§5, §8 forecasts** (σ(dR/dz) ±0.03 from ~180 clusters → 6σ; ±0.008 from ~5,000 → 22σ): no calculation given; not reproduced.
5. **§9.2 SMBH as a "local encoding surface" regulating AGN feedback; the M–σ reference**: speculation, and M–σ is abandoned (author); cut.
6. **"15 chains"** → 18; **"verified against 15 MCMC chains"** for µ(a): the chains fix β_m and test the prediction; they do not verify the formula.
