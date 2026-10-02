# DUAL_SECTOR_VALIDATION_CHECK — "Dual-Sector Expansion: Type Ia Supernovae Validate Matter-Sector H0 Normalization with ΛCDM Geometric Consistency"
(23 Feb 2026, 12 pp), read in full 2026-10-02 (all 1,239 extracted lines, including the three appendix scripts; the first pass saw previews only and was re-read completely before this version; confirmed again in 50-line chunks with a line ledger, no gaps). Reproduction: `scripts/verify_dual_sector_validation.py` (public Pantheon+ release, ~10 s); output beside it.

## The three tests do not measure H0
In the paper's χ², m_b(model) = M + 5 log₁₀ d_L + 25 with d_L ∝ 1/H0, and M is free. H0 and M enter only as M − 5 log₁₀ H0, so the supernova magnitudes
carry **no information on H0**: the χ² profile is flat, 721.1166 at H0 = 67.4, 70 and 73.04 (M − 5 log₁₀ H0 = −28.935 in every case). The H0 in Tests A and B
is set entirely by the prior; M absorbs it. Consequences:
1. **Test B's result is an optimizer stop, not a minimum.** The paper reports Ω_m 0.3736, β −0.0005, χ² 723.16. The same code from the same start reaches
   Ω_m 0.2049, β −0.30, χ² 721.12 under the SH0ES prior — identical to Test A. Tests A and B have the same best fit; neither prior is preferred.
2. **"SNe reject the photon-sector expansion rate" does not follow.** Test A's χ² (721.12) is lower than the paper's Test B (723.16). β → −0.30 in Test A is
   the Ω_m–β shape degeneracy (Ω_m 0.205 with β at its bound), present for any H0.
3. **Test C's "H0 at the boundary"** is the flat direction drifting; it says nothing about which H0 the supernovae prefer.
4. **§VI.A "β and M are not degenerate"**: the degeneracy that matters is H0–M, and it is exact.

## What the supernovae do measure (full STAT+SYS covariance, zHD > 0.01, 1590 SNe, M marginalised)
- **Shape of the Hubble diagram at Planck Ω_m = 0.315:** best β = −0.035, 68 % range −0.065 to 0.000. Supernova distances follow ΛCDM geometry. The paper's
  Prediction 2 (β_distance ≈ 0) holds.
- **The matter-sector expansion rate applied to supernova distances** (H² = H²_ΛCDM + β_m E(a) H0², β_m = 0.15765): Δχ² = +23.6 against ΛCDM at Ω_m = 0.315.
  It can be offset only with Ω_m = 0.41 (Δχ² +0.8), which Planck excludes. This bears on how "supernovae on the matter ruler" is implemented (see below).
- **H0 = 73.04** comes from the Cepheid calibration of M (SH0ES, Riess et al. 2022), not from the Hubble-flow fit. In the dual-sector picture that measurement
  is the matter-sector one; the Hubble-flow data neither confirm nor reject it.

## Reproduced
- 1588 SNe with 0.01 < zCMB < 2.5; Test A numbers (Ω_m 0.2049, β −0.30, M −19.79, χ² 721.12) exactly.
- H0_matter = 67.161 × √1.15765 = 72.26; µ(0) = 0.864, µ(1) = 0.982.
- ΛCDM with Ω_m free on the full covariance: Ω_m = 0.330 (Pantheon+ published 0.334 ± 0.018) — the reproduction is sound.

## Other corrections
5. **§VIII.D "Euclid/LSST S8 = 0.78 ± 0.01"**: S8 = σ8 √(Ω_m/0.3) = 0.7998 × √(0.3166/0.3) = **0.822** for the Level 2 IAM chain. 0.78 does not follow.
6. **Table V vs Fig. 4** disagree: the table gives N = 1094/419/75 and β = −0.007/−0.004/+0.037; the figure gives N = 892/486/210 and β = −0.001/+0.002/+0.001.
7. **Table VI "BAO — matter sector, H0 = 72.5"** contradicts the Level 1 paper (§3.1), where BAO angles are photon paths and unchanged. The book follows the
   Level 1 paper.
8. **β_γ < 1.4 × 10⁻⁶ (95 % CL, MCMC) and β_γ/β_m < 8.5 × 10⁻⁶.** Source: `tests/mcmc_final_iam.py` (emcee, 32 walkers × 5000 steps) →
   `results/mcmc_results_final.npz` (95 % = 1.425 × 10⁻⁶). Its θ_s integral reverses both arrays (`np.trapz(integrand[::-1], z_array[::-1])`), so the
   distance to last scattering comes out negative: θ_s(β_γ = 0) = −0.01025, 6,666σ from Planck. Against that χ² any β_γ ≈ 10⁻⁶ moves χ² by 4.
   **Corrected** (same model and data, `scripts/verify_beta_gamma.py`): **β_γ < 0.0039 (95 %)**, β_γ/β_m < 0.025, the same as the grid scan in
   `development/archive/tests_27-29/test_29_beta_gamma_constraint.py`. Photons couple at under 2.5 % of the matter coupling.
   8b. `tests/iam_validation.py` hard-codes 1.4e-6 as "MCMC 95 % upper limit"; its Figure 9 corner plot is drawn from synthetic samples
   (`np.random.exponential(3.3e-7)`), not a chain.
   8c. **"36σ"**: `results/test_27_results.txt`, `test_28_output.txt` — β = 0.18 applied to photon paths with r_s = 144.43 Mpc and all other
   parameters fixed shifts θ_s by +1.08 % from the observed value (+1.02 % from ΛCDM). With today's β_m = 0.15765: **+0.90 % = 30σ**. With the
   parameters free, the CMB compensates through H0 ≈ 61.5 (Level 2b runs), which the distance ladder and BAO exclude. Print as "at fixed parameters".
9. "15 converged chains", "Δχ² = +0.54", "σ8 0.809 → 0.800", "H0 67.16 / 72.26" match the Level 2 chains.

## Added after the complete read
10. **§V.A "the geometric modification to d_L(z) is subdominant (< 1 % for z < 2)"**: with β = 0.15765 in H(z) (the paper's Eq. 9), at fixed H0 d_L changes
    by −6.5 % at z = 0.1 and −2.3 % at z = 2; with the normalisation absorbed by M, the Hubble-diagram shape changes by +2.3 % (0.048 mag) at z = 0.5 and
    +4.8 % (0.10 mag) at z = 2 relative to z = 0.05. That is far above Pantheon+ precision, which is why the full-covariance fit gives Δχ² = +23.6. The
    supernovae show that their distances do NOT follow H with the β term (item 5), not that the term is too small to see.
11. **§I "photons couple at least 100,000× more weakly than matter"** (β_γ/β_m < 8.5 × 10⁻⁶): with the corrected bound β_γ/β_m < 0.025, photons couple at
    least 40× more weakly (item 8).
12. **Figs. 3 and 5 "β_m = 0.157 ± 0.029 (RSD/growth)"**: the ±0.029 is from the early emcee fit (`results/mcmc_results*.npz`), not from β_m = Ω_m/2, which is
    fixed; label it or remove.
13. Appendix code reads `parts[4]` (zCMB), `parts[8]` (m_b_corr), `parts[9]` (diagonal error) with 0.01 < zCMB < 2.5: matches the reproduction here (1588 SNe,
    Test A χ² 721.12). §III.A's "median σ_mb = 0.21 mag" ✓ (0.212, `m_b_corr_err_DIAG`, 1588 SNe).

14. (ledger read) **§VIII.D CMB-S4 "β_γ < 10⁻⁷"** scales the buggy 1.4 × 10⁻⁶ bound (item 8); with the corrected 0.0039 a CMB-S4 forecast must be redone.
    **"Standard sirens should yield H0 ≈ 73"**: the chain value is 72.26; GW170817 analyses span 68–75.5 (one event). **"DESI Y5 … σ8 = 0.800, confirmed by
    Level 2"**: 0.800 is the chain posterior, not a confirmation.
15. (ledger read) **§VII.C** "modified gravity affects all matter equally … no mechanism for photon/matter separation": µ–Σ models do separate growth (µ) from
    lensing (Σ); the paper's own §VIII.E says so. **§VII.D** "IAM improves S8": the Level 2 S8 is 0.822 vs ΛCDM 0.832, a ~1σ shift.
16. Acknowledgments credit an AI assistant for discussion; not a private correspondent (N1 does not apply).

## For the author
- The paper's title and conclusions 1–3 and 5 rest on the three tests. The chapter keeps the paper's structure and Prediction 2, reports what the tests
  show, and holds the title and conclusions for the author.
- **The +23.6 result** applies to the TWO_RULER_DESI_TEST mock, which put supernovae on H̃²_m = H²_ΛCDM + β_m E(a) H0². Real Pantheon+ shape disfavours that
  form at Planck Ω_m. The chains (Level 1 run J, background unmodified) are consistent (Δχ² +1.58). Which form "supernovae sit on the matter ruler" means —
  the normalisation only (the ladder H0), or the full H̃_m(z) — is the author's decision; the data favour normalisation only.
