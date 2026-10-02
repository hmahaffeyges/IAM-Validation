# Paper errata — corrections to apply to the papers in `docs/papers/` (and their LaTeX in `docs/papers/latex/`)

Every confirmed correction found while auditing the papers for the book is recorded here the moment it is found, against the paper it belongs to.
The book chapters already carry the corrected values. **The papers themselves are updated from this list after the book is finished**, so the repository
papers, the book and the verification files all say the same thing.

Status: **confirmed** = recomputed or traced to the source, evidence in the linked check; **author** = needs the author's decision before the paper changes;
**pending re-read** = found in an earlier audit, to be reconfirmed when the paper is read in full for its chapter. Applied entries are ticked ☑ and dated.

| # | Paper (file) | Where | Printed | Correct | Status | Evidence |
|---|---|---|---|---|---|---|
| **IAM Theory Paper** (`IAM_Theory_Paper.pdf`, 14 Apr 2026) |||||||
| T1 | | §6.2 Eq. 41 | n − 9/2 = −1 ⟹ n = 5/2 | n = 7/2 | confirmed, author approved 2026-10-02 | `theory/EXPONENT_LINE_BY_LINE.md` |
| T2 | | §6.2 / §10 | "Sheth–Tormen n_eff ≈ 3.5" | remove (no source reports it; ST slope ≈ 1 at 10¹² h⁻¹M⊙) | confirmed | `virial/NBODY_TRACE.md` |
| T3 | | §8.5 | w0 = −1.07, w_a = +0.04 | w0 = −1.062, w_a = −Ω_m²/[3(2−Ω_m)²] = −0.012 | confirmed | `theory/THEORY_CHECK.md` #2 |
| T4 | | §9.3, §10.5 | six-study η_vir table, four-study n_eff table as confirmations | η_vir = 1/(2f_coll) is a definition; published 2T/\|U\| ≥ 1 (Neto 2007, Power 2012); tables removed | confirmed | `virial/NBODY_TRACE.md` |
| T5 | | §11.5 | D2 ratios 1.014/1.052/1.075; bispectrum 1.033/1.052/1.072; crossover near z ≈ 0.2 | ratios are D⁴ with the whole growth equation on H_IAM and amplitude matched today; Level 2 form gives 1.015/1.020/1.025; no crossover with one normalisation | confirmed | `theory/THEORY_CHECK.md` #5, `scripts/verify_theory_paper.py` |
| T6 | | §12.3 Fig. 2 | "Δχ² = +0.75" | label from the final chain extraction (Planck only +0.96) | confirmed | `tests/plot_cl_comparison.py` line 319 (typed text) |
| T7 | | §12 | σ8 0.813 → 0.800 (1.6 %) | per level: L1 0.814 → 0.802 (−1.6 %), L2 0.809 → 0.800 (−1.1 %) | confirmed | `mgcamb_validation/CHAIN_EXTRACTION_FINAL.csv` |
| T8 | | §13.5 | "potential … actualized" | physics terms (records, low-entropy state) | confirmed (wording rule) | `part5_drafts/p5_theory_interpretation.tex` |
| **Late-Time Growth Suppression** (`Late_Time_Growth_Suppression_in_the_mu_Sigma_Framework…pdf`, 19 Feb 2026) |||||||
| L1 | | Eq. 6 | µ = 1 + µ0 Ω_DE(a) | µ = 1 + µ0 Ω_DE(a)/Ω_Λ (MGCAMB `mgcamb.f90` l. 761) | confirmed | `chains/LATE_TIME_GROWTH_CHECK.md` #1 |
| L2 | | Tables 1–5, Fig. 4, abstract | Δχ² +1.43/+1.34/+2.32/+1.58; free µ0 +0.006/+0.024/+0.002/−0.005 ± σ | final chains: Δχ² +0.96/+0.56/+1.73/+1.58; free µ0 as median and 90 % bound (posterior reaches the +0.2 prior edge) | confirmed | `LATE_TIME_GROWTH_CHECK.md` #2, #9 |
| L3 | | §4.1 | "p = 0.23 for Δχ² = 1.43" | likelihood ratio e^(−Δχ²/2); χ²₁ p-value does not apply at equal parameter count | confirmed | #3 |
| L4 | | §5.3 | DES Y3 "µ0 = −0.4 ± 0.4" | DES Y3 + external µ0 = 0.08 (+0.21/−0.19); DES alone does not constrain µ0 | confirmed | #4 (arXiv 2207.05766) |
| L5 | | refs | Andrade et al., PRD 109, 063518 | MNRAS 529, 831 (2024); data ACT + WMAP + SDSS + SN | confirmed | #5 |
| L6 | | refs | Frusciante 2025 "Modified gravity forecasts with the µ–Σ parameterisation" | "Euclid preparation. Review of forecast constraints on dark energy and modified gravity" (arXiv 2512.09748) | confirmed | #6 |
| L7 | | Fig. 3 legend | "SDSS BOSS/eBOSS" | the z = 0.07 point is 6dFGS (Beutler 2012) | confirmed | #10 |
| L8 | | Fig. 5 | Gaussians from mean ± σ | actual posteriors (reach the prior edge) | confirmed | `docs/book/part2_drafts/figs_late_time_growth/fig_mu0_posterior_final.pdf` |
| L9 | | §3.1 | supernovae "photon-sector" | supernovae on the matter ruler (author ruling); distances follow ΛCDM because the background is unmodified | confirmed (ruling) | #7 |
| **Dual-Sector Validation** (`Dual_Sector_Validation_Paper.pdf`, 23 Feb 2026) |||||||
| D1 | | title, abstract, §IV, §IX | "SNe reject photon-sector H0, validate matter-sector H0" | with M free the SN χ² is exactly flat in H0; the tests cannot select H0; 73.04 comes from the Cepheid calibration | author | `chains/DUAL_SECTOR_VALIDATION_CHECK.md` |
| D2 | | Table II | Test B: Ω_m 0.3736, β −0.0005, χ² 723.16 | optimizer stop; same code reaches Ω_m 0.2049, β −0.30, χ² 721.12 | confirmed | `scripts/verify_dual_sector_validation.py` |
| D3 | | §VI.A | "β and M are not degenerate" | H0–M degeneracy is exact | confirmed | same |
| D4 | | §VIII.D | Euclid/LSST "S8 = 0.78 ± 0.01" | S8 = 0.822 (σ8 0.7998, Ω_m 0.3166, Level 2) | confirmed | same, #5 |
| D5 | | Table V vs Fig. 4 | two different bin counts and β values | one set, recomputed | confirmed (inconsistent) | #6 |
| D6 | | Table VI | BAO "matter sector, H0 = 72.5" | BAO angles are photon paths (Level 1 paper) | confirmed | #7 |
| D7 | | §I, §II, Table VI | "β_γ < 1.4 × 10⁻⁶ (95 % CL, MCMC)"; "β_γ/β_m < 8.5 × 10⁻⁶" | β_γ < 0.0039 (95 %), β_γ/β_m < 0.025; the 1.4e-6 is a sign error in the emcee θ_s integral | confirmed | #8, `scripts/verify_beta_gamma.py` |
| D8 | | §VIII.A | "catastrophic 36σ CMB acoustic scale tension" with uniform β | β = 0.18 at fixed parameters: +1.08 % (36σ); with β_m = 0.1577: +0.90 % (30σ); state "at fixed parameters" (free parameters → H0 ≈ 61.5, Level 2b) | confirmed | #8c |
| D9 | | §V.A | "geometric modification to d_L subdominant (< 1 % for z < 2)" | shape change +2.3 % at z 0.5, +4.8 % at z 2 (0.05–0.10 mag), excluded by Pantheon+ (Δχ² +23.6); SN distances do not follow the β-modified H(z) | confirmed | #10 |
| D10 | | §I | "photons couple at least 100,000× more weakly" | at least 40× (β_γ/β_m < 0.025) | confirmed | #11, D7 |
| D11 | | Figs. 3, 5 | β_m = 0.157 ± 0.029 (growth) | the ±0.029 is the early emcee fit; β_m is fixed at Ω_m/2 | confirmed | #12 |
| **Wording across papers (author, 2026-10-02)** |||||||
| W1 | `iam_law_v2.tex` (l. 54, 114), `Evidence_Baryon.tex` (l. 94), `iam_cosmological_constant.tex` (l. 275) | IAM's law statement | "from quantum potential to classical actuality" | "from a quantum superposition to a classical record" | author approved | `CANON/iam_canon.json` |
| D12 | Dual_Sector_Validation_Paper | title, abstract, §IX | "Type Ia Supernovae Validate Matter-Sector H0 Normalization"; conclusions 1–3, 5 | "Type Ia Supernovae in the Dual-Sector Picture: ΛCDM Distances with a Locally Calibrated H0"; conclusions: (1) SN distances follow ΛCDM geometry; (2) β applied to SN distances excluded (Δχ² +23.6); wording final after the matter-ruler read | author approved (title) | `chains/DUAL_SECTOR_VALIDATION_CHECK.md` |
| **IAM Dual Sector Note** (`IAM_Dual_Sector_Note.pdf`, March 2026) |||||||
| S1 | | §3 | "µ < 1, Σ = 1 unique; f(R) µ > 1, Σ > 1; DGP µ > 1" | f(R): Σ = 1, µ 1–4/3; DGP: Σ = 1, self-accelerating branch µ < 1; IAM-specific is Eq. 5 with no free parameter | confirmed (f(R)); DGP pending trace | `chains/DUAL_SECTOR_NOTE_CHECK.md` #1 |
| S2 | | Table 1, Fig. 1(a), §5, §7 | β_γ < 1.4 × 10⁻⁶, ratio > 10⁵, "100,000×" | β_γ < 0.0039, β_γ/β_m < 0.025, ≥ 40× | confirmed | #2 |
| S3 | | Table 1, §7 | "Planck recovers β_m without fitting … strongest single result" | consistency of fixed β_m with Ω_m/2 (0.2σ) | confirmed | #3 |
| S4 | | Table 1, §7 | "1,588 supernovae select matter sector" | SNe: ΛCDM distances; β on distances excluded | confirmed | #4 |
| S5 | | Table 1, §5, Fig. 2(g) | DESI phantom crossing "predicted artifact" | open: mock gave opposite quadrant; real-data two-ruler test pending | pending test | #5 |
| S6 | | Figs. 1, 2 | 72.5/72.48; σ8 0.7901 (−2.6 %); lensing −2.0 %; Δχ² 79.8 (8.9σ); S8 0.753; Pantheon+ as photon sector | chain values (72.26; 0.800, −1.1/−1.6 %); lensing 0.05–0.3 %; remove 8.9σ panel; SNe matter-normalised | confirmed | #6 |
| S7 | | §4 | "below 95 % exclusion threshold 3.84" | likelihood ratio | confirmed | #7 |
| S8 | | §6 | β_γ "above 10⁻⁴"; "fitted β_m" | detection of β_γ > 0; β_m not fitted | confirmed | #8 |
| **Virial papers, complete re-read 2026-10-02** |||||||
| V13 | PRL Version (Thermodynamic Identity) | Step 1 | Q = −E_f for any 1/r system | state scope: systems that release the binding energy; collisionless halos relax without radiating | confirmed | `virial/VIRIAL_CHECK.md` #13 |
| V14 | Dark Matter and Dark Energy as Virial Partners | §4.1, Fig. 1, Table 2 | IAM 0.2952 at z ≈ 0.5; DESI at 0.02σ | 0.2990 at z = 0.5; 0.3σ | confirmed | #14 |
| V15 | Wide Domains; Virial Partners | Test 1, Table 3; Table 2 | σ8 = 0.802 ± 0.020 (2025 joint) | trace source; KiDS-Legacy S8 0.815 vs IAM S8 0.822 | pending trace | #15 |
| V16 | Wide Domains; Virial Partners; Grav. Decoherence | abstracts, §7 | Euclid DR1 October 2026, σ(µ0) 0.04 at DR1 | full DR1 mid-2027; 0.04 is the final-survey forecast | confirmed | #16 |
| V17 | Wide Domains §5 Eq. 7 (and the BH group papers, G6) | Landauer identity for black holes | E_Landauer = (ln 2/2) M c² = 34.7 % of rest energy | with S_BH counted in bits and k_B T_H ln 2 per bit, E = T_H S_BH = ½ M c² exactly (Smarr 1973 for Schwarzschild, M c² = 2 T_H S); the ln 2 was applied twice. The black hole carries the virial ½ | confirmed (computed 1 M⊙: 0.5000) | reconfirm when G6 papers are read |
