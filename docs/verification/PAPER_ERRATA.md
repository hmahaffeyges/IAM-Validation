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
| LG9 | | Eq. 1 | µ = H²/(H² + βE(a)) | µ = H²/(H² + βE(a)H0²) | confirmed | `chains/LATE_TIME_GROWTH_CHECK.md` #11 |
| LG10 | | §3.1 | RSD = fσ8 from BOSS DR12 and eBOSS DR16 | three BOSS DR12 fσ8 points (consensus final); DR16 entries are BAO distances | confirmed (Cobaya 3.5 data file) | #12 |
| LG11 | | §1, §5.3 | f(R), DGP predict µ ≥ 1 | normal-branch DGP; self-accelerating DGP gives µ < 1, Σ = 1 (ghost, excluded) | confirmed | #13 |
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
| V31 | Virial Efficiency Table 1 | | Power 2012 GIMIC/OWLS; Bett/Ludlow/Bryan values | N-body; not tabulated | confirmed | `virial/VIRIAL_CHECK.md` #29 |
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
