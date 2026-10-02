# LATE_TIME_GROWTH_CHECK — "Constraints on Late-Time fσ8 Suppression from µ < 1, Σ = 1" (19 Feb 2026, 13 pp), read in full 2026-10-02

Chapter: `docs/book/part2_drafts/p2_late_time_growth.tex` (the paper carried in its order). Chain numbers: `mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv`
and `CHAIN_PAIRS_FINAL.csv` (final files, 30 % burn-in; all final R − 1 ≤ 0.0099).

## Reproduced
- µ(0) = 1/(1 + Ω_m/2) = 0.864 (paper 0.865, from Ω_m rounding); µ0 = −0.13495 in every run.
- MGCAMB settings in all twelve `yaml_configs/run_*.yaml`: MG_flag 1, pure_MG_flag 2, musigma_par 1, GRtrans 0.001; Rminus1_stop 0.01.
- MGCAMB approximation (source: sfu-cosmo/MGCAMB `fortran/mgcamb.f90` line 761, `MGCAMB_Mu = 1 + mu0*omegaDE_t/omegav`): agrees at z = 0 (0.865 vs 0.864)
  and z ≫ 1; maximum gap 2.8 % at z ≈ 0.65, 2.5 % at z = 1, < 1 % above z = 2.5. The paper's "1–2.5 % at 0.5 ≲ z ≲ 2, exact at z = 0" is right.
- σ8 per run agrees with the final chains to ≤ 0.002; Δσ8 = −0.0128 to −0.0129 (−1.6 %) in every combination.
- CMB lensing "changes by < 0.5 %": Limber estimate at fixed primordial amplitude gives 0.05–0.3 % (L 30–1000) for both the µG and friction forms.
  (The CAMB Technical Note's validation panel reads "IAM reduces lensing by 2.0 %"; to be checked when that paper is read.)
- Literature: DESI 2024 VII µ0 = 0.11 (+0.45/−0.54) (arXiv 2411.12022 eq. 5.5) ✓; Andrade et al. µ0 − 1 = 0.02 ± 0.19 ✓ (value).

## Corrected in the chapter
1. **Eq. 6 typo:** printed µ = 1 + µ0 Ω_DE(a); MGCAMB and DES use µ = 1 + µ0 Ω_DE(a)/Ω_Λ (needed for µ(0) = 1 + µ0).
2. **Chain numbers updated to the final files** (the chains ran longer after 19 Feb): Δχ² (chain minima) +0.96 / +0.56 / +1.73 / +1.58 (paper +1.43 / +1.34 /
   +2.32 / +1.58); free µ0 +0.015 ± 0.156 / +0.039 ± 0.125 / +0.011 ± 0.152 / −0.008 ± 0.163 (paper +0.006 / +0.024 / +0.002 / −0.005); prediction 1.0 / 1.4 /
   1.0 / 0.8 σ from the mean (paper 0.9 / 1.3). H0 and σ8 to the final values. The free-µ0 Δχ² entries (−1.90, −3.99, +0.19, −0.60) are not printed: a free
   parameter's χ²_min depends on how deeply the chain sampled its minimum.
3. **"p = 0.23 for Δχ² = 1.43 with zero additional parameters":** a χ²₁ p-value does not apply to two non-nested models with the same parameter count.
   Replaced by the likelihood ratio e^(−Δχ²/2). Figure 4's "95 % threshold 3.84" framing likewise not printed.
4. **DES Y3:** "µ0 = −0.4 ± 0.4" is not in the source. DES Y3 alone does not constrain µ0; DES Y3 + external gives µ0 = 0.08 (+0.21/−0.19) (Abbott et al.
   2023, arXiv 2207.05766 abstract and eq. 38).
5. **Andrade et al. citation:** MNRAS 529, 831 (2024), not PRD 109, 063518; dataset ACT + WMAP + SDSS + SN.
6. **Euclid reference:** arXiv 2512.09748 is "Euclid preparation. Review of forecast constraints on dark energy and modified gravity" (Frusciante et al.).
   The σ(µ0) table is the paper's own Fisher estimate and is labelled so.
7. **Supernovae:** the paper calls them photon-sector observables. By the author's ruling supernovae sit on the matter ruler; the chapter states only what
   the run shows — with the background unmodified, luminosity distances equal ΛCDM's.
8. µ(0) 0.865 → 0.864 and 13.5 % → 13.6 % (Ω_m = 0.315 throughout).

## Open
- None for this paper. The 2.0 % lensing figure belongs to the CAMB Technical Note (G1, to be read).
