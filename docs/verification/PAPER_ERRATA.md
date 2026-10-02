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
| D7 | | §I, §II | "β_γ < 1.4 × 10⁻⁶ (95 % CL, MCMC)", "36σ" | no chain samples β_γ; source script needed | author | #8 |
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
| **Cosmological Constant** (`The_Cosmological_Constant_as_Actualized_Vacuum_Energy.pdf`) and **Baryon Asymmetry** (`Baryon_Asymmetry_as_a_Derived_Quantity…pdf`) |||||||
| C1 | Cosmological Constant | 2/π factor | (2/π)(l_P/l_H)² | 2(l_P/l_H)² with A_eff = 2π l_H² | confirmed | `cosmological_constant_and_baryon/CC_AND_BARYON_CHECK.md` |
| C2 | Cosmological Constant | history integral | as written | coefficient 3 × 10³⁰ vs required 0.523; derivation open | confirmed | same |
| C3 | Baryon Asymmetry | analytic η | 6.079, 6.115 × 10⁻¹⁰ | eq. 3 inverted 5.04, eq. 5 inverted 6.09 × 10⁻¹⁰ | confirmed | same |
| C4 | Baryon Asymmetry | "standard configuration" BBN prior | N(0.02242, 0.00014) called a prior | Cobaya `ref` (starting point); prior flat 0.020–0.025 | confirmed | same |
| C5 | Both | framing | two confirmations | one observed present-epoch relation Ω_b/Ω_m ≈ (3/16)√Ω_Λ (0.7σ); derivation open | confirmed | same |
| **Missing Satellites** (`Missing_Satellites.pdf`, Mar 2026) — book inclusion is the author's decision |||||||
| M1 | | §3 Mechanism A | ΔD/D(z=0) = −7.4 % | −0.78 % (µG, L1) / −0.67 % (friction, L2); σ8 −1.6 % / −1.1 % in the chains | confirmed | `observations/MISSING_SATELLITES_CHECK.md` |
| M2 | | §4.4 | "raw prediction 10^6.4, ~100× below" | Eq. 11 gives 10^8.44 at 4 km/s; offset not in the paper's equation | confirmed | same |
| M3 | | §4 Mechanism B | no halo virialises below σ_crit ≈ 4 km/s | 25 of 54 MW satellites below 4 km/s today; test against σ at infall is open | author | same |
| M4 | | | "all 17 converged R−1 < 0.01"; "Euclid DR1 October 2026" | 14 of 17 at the paper's date (all 18 today); DR1 mid-2027 | confirmed | same |
| **CAMB Technical Note** (`IAM_CAMB_Technical_Note.pdf`) — from the 2026-10-01 audit |||||||
| N1 | | Fig. 1 summary panel | "Δχ² = 30.01 (5.5σ), ΛCDM 444,000× less likely" | remove (pre-Planck Python test; contradicts the note's own conclusion) | pending re-read | `book3/audit/IAM_CAMB_Technical_Note.md` |
| N2 | | µ(z) table | 0.884, 0.920 at z = 0.2, 0.5 | 0.905, 0.948 | pending re-read | same |
| N3 | | Fig. (g) CMB lensing | "IAM reduces lensing by 2.0 %" | Limber estimate at fixed amplitude 0.05–0.3 %; recompute with MGCAMB | pending re-read | `chains/LATE_TIME_GROWTH_CHECK.md` |
| N4 | | binned µ; peak dµ/dz | "~15 % precision" vs fig. ~54 %; peak z ≈ 1.0 vs fig. 0.05 | reconcile | pending re-read | audit |
| **Code** |||||||
| X1 | `tests/plot_cl_comparison.py` | line 319 | hard-coded "Δχ² = +0.75" | compute from the chain files | confirmed | T6 |
