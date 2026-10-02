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
| T9 | | Eqs. 12, 15, 16 | G = 1/(4ħη); η = c⁴/(4ħG); δA_min via κ = c²/ℓ_P; one bit per 4ℓ_P² | G = c³/(4ħη); η = c³/(4ħG); δA_min = 4ħG/c³ for any κ; one nat per 4ℓ_P² (bit: 4 ln2 ℓ_P²) | confirmed | `theory/THEORY_CHECK.md` #6 |
| T10 | | Eq. 58 | βE(a) | βE(a)H0² | confirmed | `theory/THEORY_CHECK.md` #7 |
| T11 | | Abstract, §9.2, §16 | β_m verified to 0.3 % (N-body mass functions, MCMC) | β_m fixed; identity via η_vir definition; remove | confirmed | `theory/THEORY_CHECK.md` #8 |
| T12 | | §9.4, §15.5 | fitted β_m shifts with Ω_m | β_m is not fitted; test via free µ0 / growth data | confirmed | `theory/THEORY_CHECK.md` #9 |
| T13 | | §11.4, Fig. 1b, §15.2, §15.6 | f(R) Σ > 1; unique; DGP scale-dependent | f(R) Σ = 1; sDGP µ<1, Σ=1 (ghost); DGP scale-independent (quasi-static) | confirmed | `theory/THEORY_CHECK.md` #10 |
| T14 | | §12.6 | 75.5 cited to Abbott 2017 / Nicolaou 2023, "0.5σ" | 2024 afterglow analysis; other analyses 68–70; consistent with both rates | confirmed | `theory/THEORY_CHECK.md` #11 |
| T15 | | §12.7 | DESI DR1 σ(µ0) = 0.22 (0.6σ) | DESI FS µ0 = 0.11 (+0.45/−0.54) | confirmed | `theory/THEORY_CHECK.md` #12 |
| T16 | | §12.1 | Δχ² +1.43/+1.34; µ0 = 0.033 ± 0.125 (1.3σ) | final chains: 0.039 ± 0.125 (1.4σ); Δχ² from final extraction | confirmed | `theory/THEORY_CHECK.md` #13 |
| T17 | | Eq. 75, §6.6 | ∫ R/(T_H A_H) da′; coefficients within 5–10 % | per dt (dt = da/aH); within 5 % for D^(7/2) | confirmed | `theory/THEORY_CHECK.md` #14 |
| T18 | | §5.3, §10.5, §11.5, Fig. 1, §11.2 | PS n ≈ 2.5–4; 1-loop −2.5 %; MGCAMB 1–2.5 %; H_IAM > H_ΛCDM | not derived / not reproduced / 2.8 % max / matter-sector rate | confirmed | `theory/THEORY_CHECK.md` #15 |
| **Late-Time Growth Suppression** (`Late_Time_Growth_Suppression_in_the_mu_Sigma_Framework…pdf`, 19 Feb 2026) |||||||
| LG1 | | Eq. 6 | µ = 1 + µ0 Ω_DE(a) | µ = 1 + µ0 Ω_DE(a)/Ω_Λ (MGCAMB `mgcamb.f90` l. 761) | confirmed | `chains/LATE_TIME_GROWTH_CHECK.md` #1 |
| LG2 | | Tables 1–5, Fig. 4, abstract | Δχ² +1.43/+1.34/+2.32/+1.58; free µ0 +0.006/+0.024/+0.002/−0.005 ± σ | final chains: Δχ² +0.96/+0.56/+1.73/+1.58; free µ0 as median and 90 % bound (posterior reaches the +0.2 prior edge) | confirmed | `LATE_TIME_GROWTH_CHECK.md` #2, #9 |
| LG3 | | §4.1 | "p = 0.23 for Δχ² = 1.43" | likelihood ratio e^(−Δχ²/2); χ²₁ p-value does not apply at equal parameter count | confirmed | #3 |
| LG4 | | §5.3 | DES Y3 "µ0 = −0.4 ± 0.4" | DES Y3 + external µ0 = 0.08 (+0.21/−0.19); DES alone does not constrain µ0 | confirmed | #4 (arXiv 2207.05766) |
| LG5 | | refs | Andrade et al., PRD 109, 063518 | MNRAS 529, 831 (2024); data ACT + WMAP + SDSS + SN | confirmed | #5 |
| LG6 | | refs | Frusciante 2025 "Modified gravity forecasts with the µ–Σ parameterisation" | "Euclid preparation. Review of forecast constraints on dark energy and modified gravity" (arXiv 2512.09748) | confirmed | #6 |
| LG7 | | Fig. 3 legend | "SDSS BOSS/eBOSS" | the z = 0.07 point is 6dFGS (Beutler 2012) | confirmed | #10 |
| LG8 | | Fig. 5 | Gaussians from mean ± σ | actual posteriors (reach the prior edge) | confirmed | `docs/book/part2_drafts/figs_late_time_growth/fig_mu0_posterior_final.pdf` |
| LG9 | | §3.1 | supernovae "photon-sector" | supernovae on the matter ruler (author ruling); distances follow ΛCDM because the background is unmodified | confirmed (ruling) | #7 |
| LG10 | | Eq. 1 | µ = H²/(H² + βE(a)) | µ = H²/(H² + βE(a)H0²) | confirmed | `chains/LATE_TIME_GROWTH_CHECK.md` #11 |
| LG11 | | §3.1 | RSD = fσ8 from BOSS DR12 and eBOSS DR16 | three BOSS DR12 fσ8 points (consensus final); DR16 entries are BAO distances | confirmed (Cobaya 3.5 data file) | #12 |
| LG12 | | §1, §5.3 | f(R), DGP predict µ ≥ 1 | normal-branch DGP; self-accelerating DGP gives µ < 1, Σ = 1 (ghost, excluded) | confirmed | #13 |
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
| **Dual-Sector Perturbation Cosmology, Level 2** (`Dual_Sector_Perturbation_Cosmology_CAMB.pdf`, 28 Feb 2026) |||||||
| P1 | | §2.3 code listing | `grho_0 = 3.0`; friction in CDM/baryon velocity equations | print the source: extra density βE a² × today's total density; metric source z divided by ℋ_m in the CDM and baryon density equations | confirmed | `chains/DUAL_SECTOR_PERTURBATION_CHECK.md` #1 |
| P2 | | §2.2 Eq. 5 | µ(a) = H²/(H² + βE) for the implementation | the code's growth: σ8 −1.2 % at fixed parameters (Eq. 5: −0.8 %); same redshift dependence; state the coded change and its measured growth | confirmed | #2 |
| P3 | | Tables 1, 6 | E(a) 0.6977 / 0.3679 / 0.1353 / 0.0498 (z 0.2 / 0.5 / 1 / 2); µ 0.893, 0.942; H_m/H 1.058, 1.030; Table 6 H_photon 87.45 / 117.68 / 199.00 / 305.00, H_m 89.73 / 118.72 / 199.22 / 305.06 (z 0.5 / 1 / 2 / 3) | E = e^−z 0.8187 / 0.6065 / 0.3679 / 0.1353; µ 0.905, 0.948; H_m/H 1.051, 1.027; H_photon 88.89 / 120.44 / 204.06 / 307.37; H_m 91.29 / 121.53 / 204.29 / 307.43 | confirmed | #3 |
| P4 | | §7, Fig. 7 | background runs "6σ" | 10.9σ (H0 61.45 ± 0.42 vs 67.36 ± 0.54) | confirmed | #4 |
| P5 | | §3.1, refs | "Planck 2018 CamSpec (Efstathiou & Gratton 2021)" | NPIPE/PR4 CamSpec (Rosenberg et al. 2022) | confirmed | #5 |
| P6 | | §3.1 | seven fσ8 points "from BOSS DR12 and eBOSS DR16" | includes 6dFGS (z 0.067) and SDSS MGS (z 0.15); diagonal errors | confirmed | #6 |
| P7 | | §5.2 Table 4, Fig. 5, Table 7 | RSD +3.08, total +2.92, "validated by Level 1 (+1.34)" | mixed statistics (chain-average vs single point); Level 1 used different data; redo with a ΛCDM + RSD chain or best points | confirmed | #7 |
| P8 | | abstract, §5, Fig. 5 | "below the 95 % exclusion threshold of 3.84" | likelihood ratio; equal parameter count | confirmed | #8 |
| P9 | | §5.3, §6.1 | "posterior returns β_m = 0.1583" | Ω_m/2 of the posterior; consistency with the fixed value | confirmed | #9 |
| P10 | | §2.3 | µ = 1 + µ0 Ω_DE | /Ω_Λ (as L1) | confirmed | #10 |
| P11 | | refs | Frusciante title; DESI JCAP volume | as L6; DESI to trace | confirmed / pending trace | #11 |
| P12 | | §5.2, Table 4, Fig. 5, Run D | Run D tests the IAM growth rate with fσ8 from CAMB | CAMB's fσ8 comes from velocities the modification does not touch (+8.8 % vs the density growth at z = 0); Run D did not test IAM growth; redo with fσ8 = dσ8/d ln a | confirmed | #2 |
| P13 | | §8.2 item 2 | "σ8 suppression of 0.009 (1.5 %)" | 1.1 % (1.51σ) | confirmed | #12 |
| P14 | | §1, refs | KiDS-1000 S8 = 0.759 ± 0.021 (Heymans 2021) | cosmic shear 0.759 +0.024/−0.021 (Asgari 2021) or 3×2pt 0.766 +0.020/−0.014 (Heymans 2021) | pending trace | #13 |
| P15 | | §8.5 item 2 | "Σ ≠ 1 at > 10⁻⁴ falsifies" | state forecast precision of Σ0 | confirmed | #14 |
| P16 | | refs | Wang et al. 2023 "MGCAMB v2", JCAP 01, 036 | "New MGCAMB tests of gravity with CosmoMC and Cobaya", JCAP 08, 038 | confirmed | `chains/DUAL_SECTOR_PERTURBATION_CHECK.md` #15 |
| **Wording across papers (author, 2026-10-02)** |||||||
| W1 | `iam_law_v2.tex` (l. 54, 114), `Evidence_Baryon.tex` (l. 94), `iam_cosmological_constant.tex` (l. 275) | IAM's law statement | "from quantum potential to classical actuality" | "from a quantum superposition to a classical record" | author approved | `CANON/iam_canon.json` |
| D12 | Dual_Sector_Validation_Paper | title, abstract, §IX | "Type Ia Supernovae Validate Matter-Sector H0 Normalization"; conclusions 1–3, 5 | "Type Ia Supernovae in the Dual-Sector Picture: ΛCDM Distances with a Locally Calibrated H0"; conclusions: (1) SN distances follow ΛCDM geometry; (2) β applied to SN distances excluded (Δχ² +23.6); wording final after the matter-ruler read | author approved (title) | `chains/DUAL_SECTOR_VALIDATION_CHECK.md` |
| D13 | | §VIII.D | CMB-S4 β_γ < 10⁻⁷; sirens ≈ 73; σ8 0.800 'confirmed' | forecast from corrected 0.0039; 72.26 (one event spans 68–75.5); chain value | confirmed | `chains/DUAL_SECTOR_VALIDATION_CHECK.md` #14 |
| D14 | | §VII.C–D | MG affects all matter equally; IAM improves S8 | µ–Σ separates growth and lensing; S8 0.822 vs 0.832 (~1σ) | confirmed | `chains/DUAL_SECTOR_VALIDATION_CHECK.md` #15 |
| **IAM Dual Sector Note** (`IAM_Dual_Sector_Note.pdf`, March 2026) |||||||
| S1 | | §3 | "µ < 1, Σ = 1 unique; f(R) µ > 1, Σ > 1; DGP µ > 1" | f(R): Σ = 1, µ 1–4/3; DGP: Σ = 1, self-accelerating branch µ < 1; IAM-specific is Eq. 5 with no free parameter | confirmed (f(R)); DGP pending trace | `chains/DUAL_SECTOR_NOTE_CHECK.md` #1 |
| S2 | | Table 1, Fig. 1(a), §5, §7 | β_γ < 1.4 × 10⁻⁶, ratio > 10⁵, "100,000×" | β_γ < 0.0039, β_γ/β_m < 0.025, ≥ 40× | confirmed | #2 |
| S3 | | Table 1, §7 | "Planck recovers β_m without fitting … strongest single result" | consistency of fixed β_m with Ω_m/2 (0.2σ) | confirmed | #3 |
| S4 | | Table 1, §7 | "1,588 supernovae select matter sector" | SNe: ΛCDM distances; β on distances excluded | confirmed | #4 |
| S5 | | Table 1, §5, Fig. 2(g) | DESI phantom crossing "predicted artifact" | open: mock gave opposite quadrant; real-data two-ruler test pending | pending test | #5 |
| S6 | | Figs. 1, 2 | 72.5/72.48; σ8 0.7901 (−2.6 %); lensing −2.0 %; Δχ² 79.8 (8.9σ); S8 0.753; Pantheon+ as photon sector | chain values (72.26; 0.800, −1.1/−1.6 %); lensing 0.05–0.3 %; remove 8.9σ panel; SNe matter-normalised | confirmed | #6 |
| S7 | | §4 | "below 95 % exclusion threshold 3.84" | likelihood ratio | confirmed | #7 |
| S8 | | §6 | β_γ "above 10⁻⁴"; "fitted β_m" | detection of β_γ > 0; β_m not fitted | confirmed | #8 |
| S9 | | Eq. 5; Eq. 3 | µ without H0²; S_geo = A/4G | βmE(a)H0²; state units (A/4ℓ_P²) | confirmed | `chains/DUAL_SECTOR_NOTE_CHECK.md` #10 |
| S10 | | Table 1 row 4 | DESI DR2 "full-shape fσ8 + BAO" phantom crossing | DR2 preference is distance-only (BAO + SN + CMB) | confirmed | #11 |
| **Virial papers** (5 files, 25 Feb – 18 Mar 2026) |||||||
| V1 | PRL Thermodynamic Identity; Virial Partners §2; Grav. Decoherence §2.1 | | "17 MCMC chains return β_m = 0.1583 ± 0.0033" | β_m is fixed in every chain; 0.1583 is Ω_m/2 of the L2 posterior; ΛCDM gives 0.1581 | confirmed | `virial/VIRIAL_CHECK.md` #1 |
| V2 | Virial Efficiency; PRL Table I row 8 | eq. 3, tables | 2K/\|U\| = 0.815 ± 0.025 from six N-body studies; n_eff table | sources report 2T/\|U\| ≈ 1.05–1.4; 0.76–0.90 match \|U\|/2T; n_eff not reported | confirmed | `virial/NBODY_TRACE.md` |
| V3 | Virial Efficiency eq. 3; other papers | | n = 5/2 (analytic) | n = 7/2 (see T1) | confirmed | T1 |
| V4 | Virial Partners | Table 1, E(a) column | 0.050 / 0.182 / 0.274 | 0.135 / 0.497 / 0.741 | confirmed | #3 |
| V5 | Virial Partners | | "23 % of total dark energy today" | 18.7 % (23 % is β_m/Ω_Λ) | confirmed | #5 |
| V6 | Virial Partners §4.2 | | w_info = −4/3, z_t = 0.718 | need a modified background; unmodified gives z_t = 0.632 | confirmed | #6 |
| V7 | Wide Domains | Tables 4, 6 | SNe photon-sector; H0LiCOW matter-sector | by the worldline rule: SNe matter ruler, time delays photon paths | confirmed (ruling) | #7 |
| V8 | Wide Domains §4.6 | | Δχ² +61.2 as a fit result | dominated (+49.6) by the sector assignment; state as such | confirmed | #8 |
| V9 | Grav. Decoherence §5.3 | | "best-fit improvement Δχ² = +0.54" | IAM χ² higher by 0.54: "consistent with Planck" | confirmed | #9 |
| V10 | Wide Domains, Virial Partners, Grav. Decoherence | | Euclid σ(µ0) DR1 ±0.04 / ±0.08 mixed | DR1 ≈ ±0.08; final survey ≈ ±0.04 | confirmed | #10 |
| V11 | Grav. Decoherence §4.3; Missing Satellites | refs | "Kim & Peter 2021" for halo occupation | that paper is on SIDM cluster mergers; correct source to find | confirmed | #11 |
| V12 | Wide Domains §3.3; Virial Partners; Grav. Decoherence §6 | | three-channel β_m split; DM/DE halves; arrow-of-time sections | speculative, discuss with the author (Part 5) | author | #12 |
| **Virial papers, complete re-read 2026-10-02** |||||||
| V13 | PRL Version (Thermodynamic Identity) | Step 1 | Q = −E_f for any 1/r system | state scope: systems that release the binding energy; collisionless halos relax without radiating | confirmed | `virial/VIRIAL_CHECK.md` #13 |
| V14 | Dark Matter and Dark Energy as Virial Partners | §4.1, Fig. 1, Table 2 | IAM 0.2952 at z ≈ 0.5; DESI at 0.02σ | 0.2990 at z = 0.5; 0.3σ | confirmed | #14 |
| V15 | Wide Domains; Virial Partners | Test 1, Table 3; Table 2 | σ8 = 0.802 ± 0.020 (2025 joint) | trace source; KiDS-Legacy S8 0.815 vs IAM S8 0.822 | pending trace | #15 |
| V16 | Wide Domains; Virial Partners; Grav. Decoherence | abstracts, §7 | Euclid DR1 October 2026, σ(µ0) 0.04 at DR1 | full DR1 mid-2027; 0.04 is the final-survey forecast | confirmed | #16 |
| V17 | Wide Domains §5 Eq. 7 (and the BH group papers, G6) | Landauer identity for black holes | E_Landauer = (ln 2/2) M c² = 34.7 % of rest energy | with S_BH counted in bits and k_B T_H ln 2 per bit, E = T_H S_BH = ½ M c² exactly (Smarr 1973 for Schwarzschild, M c² = 2 T_H S); the ln 2 was applied twice. The black hole carries the virial ½ | confirmed (computed 1 M⊙: 0.5000) | reconfirm when G6 papers are read |
| V18 | Virial Efficiency Table 1; PRL Table I; Wide Domains | N-body row | 2K/\|U\| = 0.815 ± 0.025 (six studies) | published 2T/\|U\| ≈ 1.1–1.3 within r_vir (Bett 2007, Neto 2007, Power 2012, Klypin 2016); 1.02–1.17 with the surface-pressure term (Klypin 2016); print these. 0.815 = 1/(2 f_coll) is a definition, not a measurement | confirmed | `virial/NBODY_TRACE.md` |
| V19 | PRL Table I; Wide Domains Table 1 | Chandrasekhar row | 1.44 M⊙ | 1.456 M⊙ for μ_e = 2 (ω₃⁰ = 2.01824, CODATA 2018) | confirmed | `scripts/verify_virial_atoms_to_horizon.py` |
| V20 | Universal Landauer Identity (archive/derivation suite) | S_BH | 4πGM²/(ħc³) | 4πGk_BM²/(ħc) | confirmed (found 2026-09-19, logged 2026-10-02) | same script |
| V21 | Virial Partners Eq. 6; Grav. Decoherence Eq. 4 | | µ without H0² | βmE(a)H0² | confirmed | `virial/VIRIAL_CHECK.md` #19 |
| V22 | Virial Partners Eq. 9; Grav. Decoherence §6.5 | | fσ8 suppression 7.9 %→0.6 %; 13.6 % growth suppression confirmed | 1 − µ, not growth; fσ8 −4.2 % (z 0) … −0.4 % (z 1); not yet measured | confirmed | `virial/VIRIAL_CHECK.md` #20 |
| V23 | Virial Partners §4.1, Table 2, Fig. 1 | | DESI Ω_m 0.2962 = growth Ω_m, 0.02σ | FS+BAO value dominated by geometry; needs growth-only fit with IAM template | confirmed | `virial/VIRIAL_CHECK.md` #21 |
| V24 | Virial Partners §4.3 | | phantom crossing from distances + growth | DR2 preference is distance-only; two-ruler test | confirmed | `virial/VIRIAL_CHECK.md` #22 |
| V25 | Wide Domains Table 4, §4.6 | | 7 H0 points; χ² 5.57 vs 55.20 | 6 listed; χ² not reproducible; Δχ² not printed | confirmed | `virial/VIRIAL_CHECK.md` #23 |
| V26 | Wide Domains Table 3, Table 9 | | ΛCDM σ8 0.45σ; ±0.22 → 0.85σ | 0.59σ; 0.62σ | confirmed | `virial/VIRIAL_CHECK.md` #24 |
| V27 | Wide Domains §4.2 | | KiDS-Legacy convergence supports Σ = 1 | lensing S8 probes matter clustering; does not separate models | confirmed | `virial/VIRIAL_CHECK.md` #25 |
| V28 | Wide Domains §6.2, §3.3 | | R1, R3 suppressed by µ | holds only for µ·G form; Level 2 leaves lensing = hydrostatic | confirmed | `virial/VIRIAL_CHECK.md` #26 |
| V29 | Wide Domains Table 1, §2, §8 | | equipartition kT/2 as virial ½; kT/2 per event; Coulomb +1/r | remove row; k_BT ln 2 per bit; −1/r | confirmed | `virial/VIRIAL_CHECK.md` #27 |
| V30 | Grav. Decoherence Eq. 11, §4 | | friction wording on µ·G equation; σ_crit prefactor | state implementation; σ_crit out | confirmed | `virial/VIRIAL_CHECK.md` #28 |
| V31 | Virial Efficiency Table 1 | | Power 2012 GIMIC/OWLS; Table 1 values for all six studies | cosmological N-body; Bett/Power report 2T/\|U\| ≈ 1.15–1.3 (cited correctly in the chapters); Ludlow cut only; Bryan & Norman a different quantity | confirmed | `virial/VIRIAL_CHECK.md` #29 |
| **Black-hole papers (G6), read 2026-10-02** |||||||
| B1 | IAM_BH_Thermodynamics §3; Info Paradox §3 | P_SB = P_Hawking "physical, not a check" | identity | the Hawking luminosity used is the black-body formula itself (ratio 1 by construction); real emission has greybody factors (Page 1976) | confirmed | `black_holes/BLACK_HOLES_CHECK.md` #3 |
| B2 | Info Paradox §6, Result 1 | Page curve; S_rad = S0/2 at τ/2; S0 2.6e76 bits | — | monotonic first-law transfer, S0/2 at 0.646 τ; not the Page curve (no turnover); S0 = 1.51e77 bits | confirmed | #4 |
| B3 | M–σ §3–4, Tables 1–2 | M ∝ σ⁴ from v²/c²; 2.00e8 at 200 km/s "to 2 %"; slope 4.05 ± 0.07 | — | dimensional slip (c³); with correct S_BH needs unstated η = 0.0057; radiation reaction is 2.5PN; observed slope 5.64 ± 0.32; 4.05 unsourced | abandoned; paper removed | #5 |
| B4 | Bekenstein coefficient Eqs. 3, 16, §6.1 | G = c⁴/(4ħη); c⁴/(4ħG) = 1/(4ℓ_P²); κ_min | — | G = c³/(4ħη); factor c; κ_max = c²/ℓ_P; η still needs G as input | confirmed | #6 |
| B5 | Bekenstein coefficient, Acknowledgments | named private correspondent | — | removed from the LaTeX (2026-10-02) | done | #7 |
| B6 | IAM_Black_Hole_Information_Paradox | Eq. 8, §7 | Mc² = k_B T ln2 × S^(1/2) (fixed point with the electron) | Mc² = 2 T_BH S_BH (Smarr); not carried | confirmed | #8 |
| B7 | IAM_Black_Hole_Information_Paradox | §9, §5.3 | island formula gives the same Page curve; temperature drift encodes the history | island curve turns over; drift records M(t) only | confirmed | #9 |
| B8 | IAM_BH_Thermodynamics | §5 vs §8 | "Resolution of the Information Paradox" vs "does not claim to resolve" | follow §8 | confirmed | #10 |
| M1 | other paper sources and scripts | M–σ mentions (abandoned 2026-10-02) | — | remove when these sources are updated: `docs/papers/latex/IAM_Virial_37_Orders_Paper` §sec:msigma and Table tab:msigma; `docs/papers/latex/arXiv_Master_file` §sec:msigma, eqs. msigma_exponent/norm, two later references; `iam_missing_satellites` (one citation + bib); `iam_thermodynamic_identity` (table row + bib); `code/Koide/scripts/Virial tests/cross_scale_validation.py` Test 5 (+ outputs, `sector_probe_census`, `honest_assessment.txt`); `tests/bh_bridge_formal.py` Part 9; Info Paradox §8.1–8.2 and BH Thermodynamics §§6.3, 9.2; `Biological_Physics/MethylPhys/papers/IAM_Hubble2Methyl_Alpha_Omega_5.tex` (one word). `tests/iam_cusp_core_sigma2_prediction.py` stays as the record of the failed cusp–core prediction | author | `black_holes/BLACK_HOLES_CHECK.md` #5 |
| N1 | all papers | names of private correspondents (author, 2026-10-02: "remove all names") | personal responses / correspondence / early feedback named in acknowledgments and text | removed from the LaTeX of the Bekenstein coefficient, Decoherence–Virial Partition, Entanglement, IAM's Law v2, Quantum Darwinism and Theory papers, the repo README and one script label. The PDFs in `docs/papers/` still carry them: rebuild each from its LaTeX when the papers are updated. Published work stays cited (Jacobson, Zurek, Smolin, England, Saridakis, Barbour–Bertotti) | LaTeX done; PDFs pending | — |
| **IAM's Law (March 2026), read 2026-10-02** |||||||
| L1 | IAM_Law | Abstract, §§1–2, throughout | "quantum potential to classical actuality" | "from quantum superposition to classical record" (canon) | confirmed | `theory/IAM_LAW_CHECK.md` |
| L2 | IAM_Law | §1, §5.4 | Landauer is a special case of IAM's Law | IAM's Law applies Landauer with the horizon as the reservoir (Assumption 2) | confirmed | `theory/IAM_LAW_CHECK.md` |
| L3 | IAM_Law | §5.3–5.8 | temperature cancels; unique partition | identity; ⟨K⟩ = Q from first law + Euler; scope as ch. 1.3 | confirmed | `theory/IAM_LAW_CHECK.md` |
| L4 | IAM_Law | Table 1, §§9, 12.1, 13, 14.3–14.4, Table 2 | β_m 0.1583 ± 0.0033 at 0.2σ; N-body 0.815; M_Ch 1.44; Sun ~10 % | β_m fixed (0.1583 = Ω_m/2); published halo ratios; 1.456; Kelvin–Helmholtz | confirmed | `theory/IAM_LAW_CHECK.md` |
| L5 | IAM_Law | §6.1 Eqs. 17–19 | G = 1/(4ħη); η = c⁴/(4ħG); κ_min | G = c³/(4ħη); η = c³/(4ħG); κ_max | confirmed | `theory/IAM_LAW_CHECK.md` |
| L6 | IAM_Law | Eq. 26 | ∫₀^a | ∫₁^a | confirmed | `theory/IAM_LAW_CHECK.md` |
| L7 | IAM_Law | §§3, 6.4, 8, 13, 14.6 | n = 5/2; ν ∝ D^−1/2; ST and N-body n_eff support | n = 7/2; ν ∝ D⁻¹; support not reproduced / different quantity | confirmed | `theory/IAM_LAW_CHECK.md` |
| L8 | IAM_Law | §7.4 | unique µ < 1, Σ = 1; f(R) Σ > 1 | f(R) Σ = 1; sDGP also µ < 1, Σ = 1 (ghost, excluded) | confirmed | `theory/IAM_LAW_CHECK.md` |
| L9 | IAM_Law | §12.3 | √(1+β_m) = 1.073; siren 75.5 "0.5σ" | 1.076; one analysis of one event, others 68–70 | confirmed | `theory/IAM_LAW_CHECK.md` |
| L10 | IAM_Law | §12.4 | DESI σ 0.22, "consistent at 0.6σ" | DESI FS µ0 = 0.11 (+0.45/−0.54), consistent with −0.135 and GR | confirmed | `theory/IAM_LAW_CHECK.md` |
| L11 | IAM_Law | §12.2 | N-body Ω_m f_coll η_vir = 0.159 | removed (V18) | confirmed | `theory/IAM_LAW_CHECK.md` |
| L12 | IAM_Law | §12.5, Table 2 | CC 0.07 %, η 0.36 % as two derived quantities | one observed relation Ω_b/Ω_m ≈ (3/16)√Ω_Λ; derivation open | confirmed | `theory/IAM_LAW_CHECK.md` |
| L13 | IAM_Law | §14.3 | strong force via Chandrasekhar; weak-force exception | electron degeneracy + gravity; exception = interpretation | confirmed | `theory/IAM_LAW_CHECK.md` |
| L14 | IAM_Law | §14.4–14.5 | second law a corollary; law not framework | interpretation → Part 5 | confirmed | `theory/IAM_LAW_CHECK.md` |
| L15 | IAM_Law | §14.6 | ST confirms 1/a to 1 %; M–σ | not reproduced; M–σ removed | confirmed | `theory/IAM_LAW_CHECK.md` |
| L16 | IAM_Law | §14.7 | M = E/(k_BT ln2) = 30.2 via the retired cell drive symbol; A ≥ 1 breach | canon M = E/(k_BT) = 20.94; that symbol and the breach criterion retired | confirmed | `theory/IAM_LAW_CHECK.md` |
| L17 | IAM_Law | Eq. 33 | β_m E(a) | β_m E(a) H0² | confirmed | `theory/IAM_LAW_CHECK.md` #17 |
| L18 | IAM_Law | Table 2; §13 A3 vs §12.2 | Δχ² prediction "≤ 0"; N-body 0.3 % vs 1.0 % | Δχ² = +0.54 with no added parameter; N-body row withdrawn | confirmed | #18 |
| **Missing Satellites** (`Missing_Satellites.pdf`, Mar 2026) — book inclusion is the author's decision |||||||
| M2 | | §4.4 | "raw prediction 10^6.4, ~100× below" | Eq. 11 gives 10^8.44 at 4 km/s; offset not in the paper's equation | confirmed | same |
| M3 | | §4 Mechanism B | no halo virialises below σ_crit ≈ 4 km/s | 25 of 54 MW satellites below 4 km/s today; test against σ at infall is open | author | same |
| M4 | | | "all 17 converged R−1 < 0.01"; "Euclid DR1 October 2026" | 14 of 17 at the paper's date (all 18 today); DR1 mid-2027 | confirmed | same |
| M5 | | §3.2, abstract | ΔD/D = −7.4 % | −0.78 % (exact coupling, growth equation) | confirmed | `observations/MISSING_SATELLITES_CHECK.md` |
| M6 | | §2, §3.3, Table 1, §6 | β_m confirmed at 0.2σ; 17 chains; Euclid DR1 Oct 2026 | fixed in every chain; 18; complete DR1 mid-2027 | confirmed | `observations/MISSING_SATELLITES_CHECK.md` |
| M7 | | §4.3, §7, refs | σ³/σ²/σ⁴ family; M–σ paper | M–σ abandoned; not carried | confirmed | `observations/MISSING_SATELLITES_CHECK.md` |
| M8 | | §6 (Mechanism B) | σ_crit ≈ 4 km/s floor | rejected by census (46 % below 4 km/s); infall-σ re-test only | confirmed | `observations/MISSING_SATELLITES_CHECK.md` |
| **CAMB Technical Note** (`IAM_CAMB_Technical_Note.pdf`) — from the 2026-10-01 audit |||||||
| N2 | | µ(z) table | 0.884, 0.920 at z = 0.2, 0.5 | 0.905, 0.948 | pending re-read | same |
| N3 | | Fig. (g) CMB lensing | "IAM reduces lensing by 2.0 %" | Limber estimate at fixed amplitude 0.05–0.3 %; recompute with MGCAMB | pending re-read | `chains/LATE_TIME_GROWTH_CHECK.md` |
| N4 | | binned µ; peak dµ/dz | "~15 % precision" vs fig. ~54 %; peak z ≈ 1.0 vs fig. 0.05 | reconcile | pending re-read | audit |
| **Code** |||||||
| X1 | `tests/plot_cl_comparison.py` | line 319 | hard-coded "Δχ² = +0.75" | compute from the chain files | confirmed | T6 |
| X2 | `tests/mcmc_final_iam.py` | `compute_theta_s` | `np.trapz(integrand[::-1], z_array[::-1])` → negative distance, θ_s = −0.01025 | `np.trapz(integrand, z_array)`; re-run → β_γ < 0.0039 | confirmed | D7 |
| X3 | `tests/iam_validation.py` | l. 392–393, Figure 9 | BETA_GAMMA_95CL = 1.4e-6, SECTOR_RATIO = 8.5e-6; corner plot from synthetic exponential samples | 0.0039, 0.025; plot the real chain | confirmed | D7 |
| X4 | every file quoting 1.4 × 10⁻⁶ / 8.5 × 10⁻⁶ | Dual_Sector_Note, IAM_CAMB_Technical_Note, Supplementary_Methods, Test_Validation_Compendium, Variational_Derivation, iam_desi_paper (LaTeX); development/IAM_Manuscript.tex; docs/README.md; CANON/PREDICTIONS_REGISTER COS-017, COS-255; code/Koide/scripts/Virial tests/cross_scale_validation* | 1.4 × 10⁻⁶; 8.5 × 10⁻⁶ | 0.0039; 0.025 | confirmed | D7 |
| X5 | `mgcamb_validation/yaml_configs/run_d/e/f` ("Planck + RSD") | likelihood block | label "fσ8 from BOSS DR12 and eBOSS DR16" | growth data = BOSS DR12 final consensus only (3 fσ8 points); the DR16 likelihoods are BAO distances; relabel "Planck + BOSS DR12 fσ8 + BAO" (L-paper §3, tables) | confirmed (Cobaya 3.5 data file; LG11) | `LATE_TIME_GROWTH_CHECK.md` |
| X6 | `camb_validation/likelihood_rsd.py` | `get_fsigma8` | fσ8 from CAMB velocities | fσ8 = −(1+z) dσ8/dz from `get_sigma8_z` for the modified code | confirmed | P12 |
| **Cosmological Constant** (`The_Cosmological_Constant_as_Actualized_Vacuum_Energy.pdf`) and **Baryon Asymmetry** (`Baryon_Asymmetry_as_a_Derived_Quantity…pdf`) |||||||
| C1 | Cosmological Constant | 2/π factor | (2/π)(l_P/l_H)² | 2(l_P/l_H)² with A_eff = 2π l_H² | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` |
| C2 | Cosmological Constant | history integral | as written | coefficient 3 × 10³⁰ vs required 0.523; derivation open | confirmed | same |
| C3 | Baryon Asymmetry | analytic η | 6.079, 6.115 × 10⁻¹⁰ | eq. 3 inverted 5.04, eq. 5 inverted 6.09 × 10⁻¹⁰ | confirmed | same |
| C4 | Baryon Asymmetry | abstract, §3.1, §4, §6 | "the BBN prior on Ω_b h² is removed"; §3.1 lists the standard setting as ref N(0.02242, 0.00014), prior [0.020, 0.025] (listed correctly) | neither is a BBN prior: N(0.02242, 0.00014) is Cobaya's `ref` (where walkers start); the prior in every Level 1 run is flat 0.020–0.025, and the 18th chain widens it to 0.010–0.040. Helium Y_He is set by CAMB's stock BBN consistency relation from Ω_b h² (standard; enters only the damping tail). Correct wording: "with a flat Ω_b h² prior and no abundance data" | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` |
| C5 | Both | framing | two confirmations | one observed present-epoch relation Ω_b/Ω_m ≈ (3/16)√Ω_Λ (0.7σ); derivation open | confirmed | same |
| C6 | Baryon Asymmetry | abstract, §3.1, §6 | "a prior four times wider than standard" | six times (0.010–0.040 = 0.030 wide vs 0.020–0.025 = 0.005) | confirmed | `yaml_configs/iam_baryon_test.updated.yaml` |
| C7 | Baryon Asymmetry | §3.1 | "all standard parameters at their Planck 2018 reference values" | all standard parameters are sampled; the reference values are where the walkers start | confirmed | same |
| C8 | Baryon Asymmetry | §4, Table 1 | "three independent frameworks"; MCMC row as an IAM result | the chain contains no IAM constraint on Ω_b (eq. 3/5 is not in it); it measures Ω_b h² from the acoustic peaks, and the ΛCDM chains give the same η (6.120–6.139); what it shows is CMB-only η agreeing with BBN, as in Planck 2018 | confirmed | `CC_AND_BARYON_CHECK.md` #1 |
| C9 | Baryon Asymmetry | abstract, §6 | "no nuclear physics input" | no abundance data and no Ω_b h² constraint; helium is set by CAMB's stock BBN relation from Ω_b h² (damping tail only) | confirmed | yaml (no YHe input; `yhe: YHe` is a name alias) |
| C10 | Baryon Asymmetry | Table 1 | "Observed (BBN) 6.137 ± 0.017" | source value to trace (Cyburt et al. 2016) | pending trace | — |
| C11 | Baryon Asymmetry | Data availability | `mgcamb_validation/iam_planck_chains/iam_baryon_test` | `mgcamb_validation/iam_baryon_test.*` and `mgcamb_validation/chains/iam_baryon_test.*` | confirmed | repo |
| C12 | Baryon Asymmetry | §1, §5 | "prediction stated in advance [Mahaffey 2026c]" | check the Matter–Antimatter paper's date against the chain (run 2026-03-20) when that paper is read (G4 #14) | confirmed (#23) | `iam_baryon_test.progress` |
| C13 | CC §2.1, §3.1 | | N_max = A_H/l_P²; f_geo → 2/π with no step | A_H/(4l_P²); f_geo = (1/4π)(l_P/l_H)² | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #10 |
| C14 | CC §5.1 | | static patch subtends 2π sr | bounded by the full horizon (4π sr); paper's own A_eff gives 2(l_P/l_H)² | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #11 |
| C15 | CC §5.3 | | holographic round trip 2/π × π/2; Koide | not a holographic result; remove | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #12 |
| C16 | CC abstract, §3.3, §6.3, §10 | | √Ω_Λ "not fitted", "0.07 %", exponent 0.502 | introduced to close 1.22; +0.79 %; 0.521 | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #13 |
| C17 | CC §6.3 | | "vanishingly small" chance of 10⁻¹²³ by accident | 10⁻¹²³ is the critical-density identity; 2 of 540 O(1) forms within 1 % | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #15 |
| C18 | CC abstract, §7, §10 | | accumulating Λ ⇒ w > −1 | growing ρ_de ⇒ w < −1; DESI comparison open | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #16 |
| C19 | CC §2.1; MA §2 | | β_m "confirmed at 0.2σ"; ratio "asymptotes to 2" | fixed in every chain; 2 by definition at a = 1 | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #18 |
| C20 | MA §7 | | l_H 10⁻³ pc; 10⁴⁰ bits; a_QCD 1.6e-12 | 5 × 10⁻¹³ pc (15 km); 2.9 × 10⁷⁸; ~1.0 × 10⁻¹² | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #19 |
| C21 | MA §5, §10 | | 10⁹ annihilation partners = dark sector | annihilation energy is the CMB; DM present at z ≈ 1100; remove | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #20 |
| C22 | MA, BA | | "prediction stated in advance" | advance statement (CMB fit returns η ≈ 6e-10) holds for every CMB fit | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #23 |
| C23 | MA §8.3, §10 | | stray "IAM posteriors at < 0.1σ"; duplicated paragraph | delete | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #22 |
| C24 | CC, MA refs | | MGCAMB "JCAP 2023, 022"; 2026d/2026e swapped | JCAP 08, 038; relabel | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` #25 |
| **Floor Breach** (`floor_breach_derivation.pdf`, Apr 2026), read in full 2026-10-02 |||||||
| FB1 | | §2 Step 3, Step 6, Table | N_CpG = 19.6 × 10⁶; E_floor 5.82 × 10⁻¹⁴ J | 2.82 × 10⁷ (hg19 CpG index, 28,217,448); 8.37 × 10⁻¹⁴ J ≈ 1.0 × 10⁶ ATP | confirmed | `scripts/verify_encoding_ladder.py` |
| FB2 | | §5 | bit counts differ by 10²⁷ | 5.4 × 10⁶⁹ (1 M☉ horizon 1.5 × 10⁷⁷ bits vs 2.8 × 10⁷) | confirmed | same |
| FB3 | | Step 5 | modified Friedmann H² = … + β_mE(a)H0²; β_m "confirmed to 0.2σ" | term acts on matter perturbations (background form gives H0 ≈ 61.5); β_m fixed at Ω_m/2 in every chain | confirmed | Level 2b chains |
| FB4 | | Steps 6, §3–4 | H_min(class) from G-002 calibration; A > 1.05 / 1.10; TCGA 27/28 | per-cell measured floor (Met-A); one gauge 0.95–1.05; cancer results re-run through chain v3 before citing | confirmed | CANON |
| **Dark Energy Evolution / Far Future** (`wz_far_future.pdf`, Feb 2026), read in full 2026-10-02 |||||||
| WZ1 | | §2.3, §6.2, Fig. 3, §8 | "small positive wa" | wa = −1/3 (negative) | confirmed | `observations/WZ_FAR_FUTURE_CHECK.md` #1 |
| WZ2 | | Eq. 14, §4.1, §7.2 | d(E/e)/da = 1/(e a²), peaks at a = 1 | E/(e a²), peaks a = 0.5; per e-fold peaks today; per Gyr peaks z = 1.26 | confirmed | `observations/WZ_FAR_FUTURE_CHECK.md` #2 |
| WZ3 | | §5.1, §6 | DR2 values (are DR1); DESI hints at / iam predicts exactly this | DR2: −0.838/−0.62, −0.667/−1.09, −0.752/−0.86; w0 7.6–10.2σ from −4/3; photon ruler predicts w = −1 | confirmed | `observations/WZ_FAR_FUTURE_CHECK.md` #3 |
| WZ4 | | §6.4 item 2 | Roman tests strongly phantom early w | ρ_info → 0 at high z (1.2 % of ρ_Λ at z = 3); remove | confirmed | `observations/WZ_FAR_FUTURE_CHECK.md` #4 |
| WZ5 | | §7.2 | E(1) = 1 from the Planck-epoch reference; observers at peak | normalisation at a = 1; remove anthropic remark | confirmed | `observations/WZ_FAR_FUTURE_CHECK.md` #5 |
| WZ6 | | §2.1 | scalar field on the encoding surface | from ρ_info ∝ E(a) and energy conservation | confirmed | `observations/WZ_FAR_FUTURE_CHECK.md` #6 |
| WZ7 | | abstract | 17 chains | 18 | confirmed | `observations/WZ_FAR_FUTURE_CHECK.md` #7 |
| **Redshift-Dependent S8 Trend** (`The_Redshift_Dependent_S_8_Trend...pdf`, Mar 2026), read in full 2026-10-02 |||||||
| ST1 | | Eq. 5, §3, Figs. 2 | S8 = S8_Planck × µ(z); 0.719 at z = 0 (Fig. 2: 0.702) | S8 × D_IAM/D_ΛCDM: 0.8255 (0.78 %) today | confirmed | `observations/S8_TREND_CHECK.md` #1 |
| ST2 | | abstract, §3, §9 | "predicts this redshift dependence" | shape matches; amplitude ~1/10 of low-z deficit; fσ8 −4.2 %; γ_eff 0.585 | confirmed | `observations/S8_TREND_CHECK.md` #2 |
| ST3 | | §4, §9 | β_m returned by 17 chains within 0.2σ | fixed in every chain | confirmed | `observations/S8_TREND_CHECK.md` #4 |
| ST4 | | §7 | ISW enhancement 10–30 % | ~3 % | confirmed | `observations/S8_TREND_CHECK.md` #5 |
| ST5 | | Fig. 3 | E(a) recovered to ~1 % by Sheth–Tormen | not reproduced; remove | confirmed | `observations/S8_TREND_CHECK.md` #7 |
| ST6 | | title, acknowledgements | addressed to a named cosmologist | cite compilation by journal/arXiv only | confirmed | `observations/S8_TREND_CHECK.md` #8 |
| **Dark Energy or Sector Tension?** (`Dark_Energy_or_Sector_Tension.pdf`, Mar 2026), read in full 2026-10-02 |||||||
| SX1 | | Table 5, §5.2, Fig. 1–2 | DR2 w0wa rows mislabelled / wrong | DR2: +CMB −0.42/−1.75; +P+ −0.838/−0.62; +U3 −0.667/−1.09; +Y5 −0.752/−0.86 | confirmed | `observations/SECTOR_TENSION_CHECK.md` #1 |
| SX2 | | abstract, Tables 1, 5, §2.4, §7 | 13.6 % / 7–8 % growth suppression | 1 − µ; fσ8 deficit 2.2 % (BGS) … 4.2 % today | confirmed | `observations/SECTOR_TENSION_CHECK.md` #2 |
| SX3 | | §5.1, §7 | phantom crossing from distance–growth mixing | DR2 fits are distance-only; SN matter-ruler shape gives opposite quadrant; Pantheon+ rejects it | confirmed | `observations/SECTOR_TENSION_CHECK.md` #3–4 |
| SX4 | | §5.4 | lensing, CMB lensing, E_G identical to ΛCDM | lensing follows lower δ_m (C_φφ −0.08 %); E_G +1.9 % at z = 0.3 | confirmed | `observations/SECTOR_TENSION_CHECK.md` #5 |
| SX5 | | abstract, §2.5, §7 | β_m recovered at 0.2σ | fixed in every chain | confirmed | `observations/SECTOR_TENSION_CHECK.md` #6 |
| SX6 | | §4.3 | β_γ/β_m < 8.5e-6 | 0.025 (corrected bound) | confirmed | `observations/SECTOR_TENSION_CHECK.md` #7 |
| SX7 | | §6.2 | µ<1, Σ=1 unachievable; DGP µ>1 | sDGP gives µ<1, Σ=1 | confirmed | `observations/SECTOR_TENSION_CHECK.md` #8 |
| SX8 | | §2.5, Table 2 | Δχ² ≤ 2.32; µ0 0.006, 0.033; 17 chains | ≤ 1.73; 0.015, 0.039; 18 | confirmed | `observations/SECTOR_TENSION_CHECK.md` #9 |
| SX9 | | Table 3 | DESI DR1 fσ8 values | differ from 2411.12021-derived values in 5/6 bins; trace | confirmed | `observations/SECTOR_TENSION_CHECK.md` #10 |
| SX10 | | §1, §6.1 | named cosmologist | cite by journal/arXiv | confirmed | `observations/SECTOR_TENSION_CHECK.md` #11 |
| **Survey Predictions** (`IAM_Survey_Predictions_Paper.pdf`, 25 Feb 2026), read in full 2026-10-02 |||||||
| SP1 | | Table 5 | ΔD/D −6.8 %, Δfσ8 −10.2 % today | −0.78 %, −4.25 % (growth equation) | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #1 |
| SP2 | | Table 5 | ΔΦ/Φ = Δµ | ΔΦ/Φ = ΔD/D (Σ = 1) | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #2 |
| SP3 | | abstract, §3.2, Fig. 2 | A_ISW = 1.134 | ~1.03; sign stands | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #3 |
| SP4 | | §5.1 | |dµ/dz| peaks at z ≈ 0.05 | largest at z = 0 | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #4 |
| SP5 | | Table 2 | σ(µ0) timeline, 5.4σ, 7.5σ | unsourced, non-monotonic; only Euclid full survey 0.04 is sourced | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #5 |
| SP6 | | Table 1 | current µ0 constraints | uncited; trace | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #6 |
| SP7 | | §4.2, Fig. 3 | tomographic mock | no code; not reproduced | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #7 |
| SP8 | | §6.2 | Σmν < 0.07–0.08 eV | no calculation; remove | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #8 |
| SP9 | | §6.3 | CMB lensing identical | −0.08 % | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #9 |
| SP10 | | Table 6 | scorecard statuses | S8 0.822; KiDS 2σ; ISW unsourced; 18 chains | confirmed | `observations/SURVEY_PREDICTIONS_CHECK.md` #11 |
| **Lensing Dynamics** (`IAM_Lensing_Dynamics_Paper.pdf`, 25 Feb 2026), read in full 2026-10-02 |||||||
| LD1 | | §2.3, abstract, §7 | M_dyn ∝ µM_true with Einstein equations for the potentials unmodified | contradictory: ratio = 1/µ only in the G_eff form; hold | confirmed | `observations/LENSING_DYNAMICS_CHECK.md` #1 |
| LD2 | | §3.3, Table 2 | f(R) Σ > 1; IAM unique | f(R) Σ = 1, µ ≤ 4/3; sDGP µ < 1, Σ = 1 | confirmed | `observations/LENSING_DYNAMICS_CHECK.md` #2 |
| LD3 | | §4.1 | Planck SZ resolved | reduced (σ8 −1.4 %), not resolved | confirmed | `observations/LENSING_DYNAMICS_CHECK.md` #3 |
| LD4 | | §4.2–4.3 | CCCP 1.20 ± 0.12; WtG 1.31 ± 0.11 | trace to source tables | confirmed | `observations/LENSING_DYNAMICS_CHECK.md` #4 |
| LD5 | | §1, §6 | 15 chains | 18 | confirmed | `observations/LENSING_DYNAMICS_CHECK.md` #5 |
| **Three-Way Cluster Mass** (`3Way_Mass_Discrepancy_in_Galaxy_Clusters.pdf`, 25 Feb 2026), read in full 2026-10-02 |||||||
| TW1 | | §2 vs §3.1 | friction form with M_hydro = µM_true | contradictory: R = 1/µ only in the G_eff form; hold | confirmed | `observations/THREE_WAY_CLUSTER_CHECK.md` #1 |
| TW2 | | §6.2, §9.1, §10 | M_SZ/M_hydro = 0.99 confirms the first condition | holds by Y–M calibration in any theory; not a test | confirmed | `observations/THREE_WAY_CLUSTER_CHECK.md` #2 |
| TW3 | | Table 2 | observed ratios, 4 bins | trace per-bin sources | confirmed | `observations/THREE_WAY_CLUSTER_CHECK.md` #3 |
| TW4 | | §5, §8 | 6σ, 22σ forecasts | no calculation; not reproduced | confirmed | `observations/THREE_WAY_CLUSTER_CHECK.md` #4 |
| TW5 | | §9.2, refs | SMBH as encoding surface; M–σ paper | speculation; M–σ abandoned; cut | confirmed | `observations/THREE_WAY_CLUSTER_CHECK.md` #5 |
| TW6 | | §2, §1 | 15 chains; 'verified' | 18; fixed β_m tests, does not verify | confirmed | `observations/THREE_WAY_CLUSTER_CHECK.md` #6 |
| **Entropic Gravity Note** (`A_Note_on_Entropic_Gravity__Saridakis_.pdf`, Mar 2026), read in full 2026-10-02 |||||||
| EG1 | | Eq. 3 | friction (2 + β_mE)Hδ' in cosmic time | dimensionally mixed; third implementation; state the form | confirmed | `theory/ENTROPIC_GRAVITY_NOTE_CHECK.md` #1 |
| EG2 | | §4 | N-body η 0.815 ± 0.025, β_m 0.159 | sources report 1.15–1.3; remove | confirmed | `theory/ENTROPIC_GRAVITY_NOTE_CHECK.md` #2 |
| EG3 | | §4, §8, §9 | β_m recovered at 0.2σ | fixed in every chain | confirmed | `theory/ENTROPIC_GRAVITY_NOTE_CHECK.md` #3 |
| EG4 | | §5 | inflection = peak production; n_eff 3.22 | per e-fold only; n = 7/2 | confirmed | `theory/ENTROPIC_GRAVITY_NOTE_CHECK.md` #4 |
| EG5 | | §8.1 | 17 chains (12+3+2) | 18 | confirmed | `theory/ENTROPIC_GRAVITY_NOTE_CHECK.md` #5 |
| EG6 | | Table 2, §6 | 72.5; µ0/Σ0 data; DESI DR2 growth | 72.26; trace; DR1 full shape | confirmed | `theory/ENTROPIC_GRAVITY_NOTE_CHECK.md` #6 |
| EG7 | | §8.2 | f(R) Σ > 1; DGP µ > 1 | f(R) Σ = 1; sDGP µ < 1, Σ = 1 | confirmed | `theory/ENTROPIC_GRAVITY_NOTE_CHECK.md` #7 |
| EG8 | | Acknowledgements | personal thanks to a named researcher | remove (names rule); cite the papers | confirmed | `theory/ENTROPIC_GRAVITY_NOTE_CHECK.md` — |
| **Technical Reference for Physicists** (27 Sep 2026), read in full 2026-10-02 |||||||
| TR1 | | §5.2 | β_m posterior 0.2σ; η_vir 6 N-body | fixed; untraced | confirmed | `theory/TECH_REFERENCE_CHECK.md` #1 |
| TR2 | | §5.2 | 11/13, 14/17 below R−1 0.01 | all 18 final ≤ 0.010 | confirmed | `theory/TECH_REFERENCE_CHECK.md` #2 |
| TR3 | | §5.3, §8 | M_lens/M_dyn 15.7 % near-term | G_eff form only | confirmed | `theory/TECH_REFERENCE_CHECK.md` #3 |
| TR4 | | §5.5, §8 | DESI DR2 w0–wa tests w_info | photon ruler; w_info is matter ruler | confirmed | `theory/TECH_REFERENCE_CHECK.md` #4 |
| TR5 | | §5.4–5.5 | inflection = peak production | per e-fold only | confirmed | `theory/TECH_REFERENCE_CHECK.md` #5 |
| TR6 | | §5.5, §8 | σ_crit consistent | rejected by census | confirmed | `theory/TECH_REFERENCE_CHECK.md` #6 |
| TR7 | | §5.5 | n = 5/2 | 7/2 | confirmed | `theory/TECH_REFERENCE_CHECK.md` #7 |
| TR8 | | §8 | Euclid DR1 Oct 2026 | mid-2027 | confirmed | `theory/TECH_REFERENCE_CHECK.md` #9 |
| TR9 | | §2.1, §5.11 | class H_min A-score | Met-A / IAM-A (CANON) | confirmed | `theory/TECH_REFERENCE_CHECK.md` #10 |
| **Electron Rest Mass (`Electron_Rest_Mass_from__IAM.pdf`, Feb 2026)**, read in full 2026-10-02 |||||||
| EM1 | | abstract, §5, §9 | 6.6 ppm, no free parameters | within 0.3 % set by H0; one factor identified numerically | confirmed | `particle/ELECTRON_MASS_CHECK.md` |
| EM2 | | §7 | BH obeys mc² = E_bit N | Mc²/2 (Smarr) | confirmed | `particle/ELECTRON_MASS_CHECK.md` |
| EM3 | | §1, §4, §3.3 | empty §1; eq. (??); paragraph printed 3× | fix | confirmed | `particle/ELECTRON_MASS_CHECK.md` |
| EM4 | | §9 | n = 3 generations from companion | at most three (KO2) | confirmed | `particle/ELECTRON_MASS_CHECK.md` |
| **Koide (`Koide_Mahaffey.pdf`, 22 Apr 2026)**, read in full 2026-10-02 |||||||
| KO1 | | §V.D | phase origin fixed, δ = 0 | δ = 0 gives m_e = m_µ = 26.9 MeV; δ = 0.2223 open | confirmed | `particle/KOIDE_CHECK.md` |
| KO2 | | Theorem 1, abstract | exactly three generations | at most three; n = 2 allowed for δ ≠ 0 | confirmed | `particle/KOIDE_CHECK.md` |
| KO3 | | §III D | E_bit = k_BT (no ln 2) | bits vs nats | confirmed | `particle/KOIDE_CHECK.md` |
| KO4 | | Acknowledgements | named correspondent | remove | confirmed | `particle/KOIDE_CHECK.md` |
| **Electroweak (`Electroweak_Symmetry_Breaking_and_the_Matter_Sector.pdf`, Oct 2026 rev.)**, read in full 2026-10-02 |||||||
| EW1 | | Eq. 2 | Ω_dm/2 = 0.1332 | 0.1330 | confirmed | `particle/ELECTROWEAK_CHECK.md` |
| EW2 | | §8 test 3 | β_m/Ω_m = 1/2 testable | definition; remove | confirmed | `particle/ELECTROWEAK_CHECK.md` |
| **x_qp (`IAM_Xqp_Mahaffey.pdf`, 14 Apr 2026)**, read in full 2026-10-02 |||||||
| XQ1 | | Eq. 13 | ln 2 quasiparticles per Δ ln2 | below 2Δ pair-breaking threshold; hold | confirmed | `particle/XQP_CHECK.md` |
| XQ2 | | Eq. 16 | QPs confined to πλ_L³ | diffusion 100–1,000 µm; device average; hold | confirmed | `particle/XQP_CHECK.md` |
| XQ3 | | temperature section | 10^-630 | 10^-61 | confirmed | `particle/XQP_CHECK.md` |
| XQ4 | | Prediction 2 | T_fridge independence discriminates | also non-thermal models | confirmed | `particle/XQP_CHECK.md` |
| XQ5 | | Prediction 3 | ratio depends only on λ³n_cp | also τ_TLS, τ_qp | confirmed | `particle/XQP_CHECK.md` |
| XQ6 | | Numerical evaluation, Eq. 17 | n_cp = 9.03e28 m⁻³ (n_e/2) | field n_cp = 2ν₀Δ ≈ 4e6 µm⁻³; prediction 15,000× the floor; corrected model in XQP_REFEREE_NOTE | confirmed | `particle/XQP_REFEREE_NOTE.md` |
| XQ7 | | Eq. 17–18 | x_qp = ln2 τ_qp/(n_cp τ_TLS πλ³); invariant = ln 2 | x_qp = 2Nτ_qp/(τ_TLS n_cp V); invariant = 2 with N, V measured | confirmed | `particle/XQP_REFEREE_NOTE.md` |
| **Gravitational Decoherence** (`Gravitational_Decoherence_Quantum_Level.pdf`, Feb 2026), read in full 2026-10-02 |||||||
| GD1 | | §2.2, Fig. 1 | τ_IAM ≈ 560 µs (1e-12 kg, 10 mK) | 559 s | confirmed | `quantum/GRAV_DECOHERENCE_CHECK.md` #1 |
| GD2 | | §3.3, Fig. 5, Table 1, §8 | τ ∝ m⁻⁶, Δα 4.33 | m⁻⁵, Δα 3.33 | confirmed | `quantum/GRAV_DECOHERENCE_CHECK.md` #2 |
| GD3 | | Eq. 6–8 | E_q = exp(1 − 1/η) from the integral | integral gives e^(t/τ); ramp not derived (hold) | confirmed | `quantum/GRAV_DECOHERENCE_CHECK.md` #3 |
| GD4 | |  Eq. 6 | S_boundary = k_BT/E_G | underived | confirmed | `quantum/GRAV_DECOHERENCE_CHECK.md` #4 |
| GD5 | | §1.2, §4.2 | 17 chains; rate peak 0.23 | 18; purity-difference peak | confirmed | `quantum/GRAV_DECOHERENCE_CHECK.md` #5 |
| **Measurement Problem** (`IAM_Measurement_Problem_Quantum.pdf`, Feb 2026), read in full 2026-10-02 |||||||
| MP1 | | §3.3, §4.1–4.2, §6.2 | erasure fails after spontaneous emission; untested | contradicted: Blinov 2004, Moehring 2007, Hensen 2015 | confirmed | `quantum/MEASUREMENT_PROBLEM_CHECK.md` #1 |
| MP2 | | §3.5 | matter entanglement ~1.3 m | 1.3 km (Hensen 2015) | confirmed | `quantum/MEASUREMENT_PROBLEM_CHECK.md` #2 |
| MP3 | | §3.7 | no Zeno with dispersive readout | observed: Slichter 2016, Harrington 2017 | confirmed | `quantum/MEASUREMENT_PROBLEM_CHECK.md` #3 |
| MP4 | | §2.2, abstract | measurement = Q ≥ k_BT ln2 | cost paid on erasure/reset (Bennett); record written on absorption | confirmed | `quantum/MEASUREMENT_PROBLEM_CHECK.md` #4 |
| MP5 | | Eq. 2, §4.3 | F(Q,T) ramp form | underived (GD3) | confirmed | `quantum/MEASUREMENT_PROBLEM_CHECK.md` #5 |
| MP6 | | §3.4 | cat decoheres by self-gravity | environmental decoherence dominates | confirmed | `quantum/MEASUREMENT_PROBLEM_CHECK.md` #6 |
| MP7 | | §2.2, Eq. 3 | µ/Σ as state labels; 17 chains | conflation; 18 | confirmed | `quantum/MEASUREMENT_PROBLEM_CHECK.md` #7 |
| **Entanglement** (`Entanglement_Decoherence_and_Classical_Records.pdf`, Oct 2026 rev.), read in full 2026-10-02 |||||||
| EN1 | | §3, §6 | β_m posterior 0.2σ | fixed in every chain | confirmed | `quantum/ENTANGLEMENT_CHECK.md` |
| EN2 | | §6 | 17 chains; two at 0.023 | 18; all ≤ 0.010 | confirmed | `quantum/ENTANGLEMENT_CHECK.md` |
| EN3 | | §6 | Euclid + DESI 5.4σ | unsourced | confirmed | `quantum/ENTANGLEMENT_CHECK.md` |
| EN4 | | Eq. 3 | S(t) ramp | inherits GD3 | confirmed | `quantum/ENTANGLEMENT_CHECK.md` |
| EN5 | | §3 Step 3 | kinetic half = heat | law first | confirmed | `quantum/ENTANGLEMENT_CHECK.md` |
| EN6 | | §4, refs | Sci. Adv. 2025 | incomplete reference | confirmed | `quantum/ENTANGLEMENT_CHECK.md` |
| EN7 | | Acknowledgements | named correspondent | remove | confirmed | `quantum/ENTANGLEMENT_CHECK.md` |
| **Two Faces of Time** (`The_Two_Faces_of_Time.pdf`, Oct 2026 rev.), read in full 2026-10-02 |||||||
| TF1 | | §5 Eq. 5 | µ in Poisson term with 'friction' text | state one form | confirmed | `quantum/TWO_FACES_CHECK.md` |
| TF2 | | §4 | H split derived, zero free parameters | predicted with β_m fixed (chain value) | confirmed | `quantum/TWO_FACES_CHECK.md` |
| TF3 | | §2.3, refs | written on the cosmic horizon; Sakharov uncited | nearest encoding surface; drop ref | confirmed | `quantum/TWO_FACES_CHECK.md` |
