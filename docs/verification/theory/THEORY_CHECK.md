# THEORY_CHECK — IAM Theory paper (14 Apr 2026, 34 pp)

**Read status.** PDF text 1,916 lines. The 2026-10-01 version of this file said "read in full"; that was not shown (only §5.3–6.4 had been
read line by line). Read in full 2026-10-02 in 50-line chunks, ledger 1–1916 with no gaps (`book3/fullread/LEDGER_G2_07_Theory.md`); items 6–15 below are
from that read, and item 6 corrects an error in this file's own "Reproduced" list.

Chapter: `part2_drafts/p2_theory.tex` carries §1–12, §14–15 in the paper's order; §13 (interpretation) is in `part5_drafts/p5_theory_interpretation.tex`.

**Reproduced:** Jacobson algebra (Eqs. 5–12); η δA_min = 1 (Eqs. 14–16; see item 6 for the constants); Cai–Kim (Eqs. 17–25); Eqs. 36–39; E(a) from
ρ̇_info = ρ_info H/a (Eqs. 43–48); E(z) values; µ(0, 0.5, 1) = 0.864/0.948/0.982; µ0 = −β/(1+β) = −0.136; w_info = −1 − 1/(3a), w(1) = −4/3;
w_eff(1) = −1.062; continuity identity (§12.4); H0 sirens 72.51; M_eq = 2.32 × 10²² M⊙; β = Ω_m/2 = 0.1575; Ω_m f_coll = 0.195 (+24 %).

**Corrected in the chapter (author authorised corrections he can defend, 2026-10-02):**
1. Eq. 41: n = 7/2 (EXPONENT_LINE_BY_LINE.md). Support printed: the paper's own Eqs. 36–39, the full-ΛCDM integration, and its Table 2. The 'Sheth–Tormen n_eff ≈ 3.5' is not printed: no source reports it, and d ln F/d ln D from ST is ≈ 1 at 10¹² h⁻¹M⊙ (NBODY_TRACE.md). The paper's σ* = 1.2 fit exp(0.925 − 1.009/a) is labelled as not yet reproduced.
2. §8.5: w0 = −1.062 (paper −1.07), w_a = −dw/da|₁ = −Ω_m²/[3(2 − Ω_m)²] = −0.012 (paper +0.04). Analytic sympy derivative of the paper's Eq. 68, checked
   numerically, Ω_m 0.30–0.32 gives −0.010 to −0.012. A least-squares CPL fit over a 0.5–1 gives +0.017; neither gives +0.04.
3. §9.3 / §10.5: η_vir stated as the definition 1/(2 f_coll). Published 2T/|U| verified by me from arXiv full text: Neto 2007 p5 ("median 2T/|U| is slightly
   greater than unity", relaxed cut < 1.35); Power, Knebe & Knollmann 2012 p9 eqs 17–18 (η ≈ 1.15 at 10^12, 1.25 at 10^15, "systematically greater than unity").
   The six-study η table and the four-study n_eff table are not printed (NBODY_TRACE.md: no source reports n_eff as a collapse-rate slope). f_coll noted as an
   extrapolation below Tinker's calibrated range.
4. §12.3 Fig. 2: the "Δχ² = +0.75" is typed text in Cosmological_Physics/tests/plot_cl_comparison.py line 319, not computed from a chain. The book labels the figure from
   CHAIN_EXTRACTION_FINAL (Planck-only chain-minimum +0.96; paper table minimizer +1.43).

5. §11.5 bispectrum (no source script). Three implementations of the growth equation, fixed parameters, ΔD/D at z = 0:
   G_eff = µG on the ΛCDM background (Level 1, MGCAMB) −0.78 %; friction 2H_IAM with the ΛCDM clock (Level 2, background unchanged) −0.67 %;
   the whole equation, clock included, on H_IAM −1.87 %. The paper's ratios 1.033 / 1.052 / 1.072 reproduce only with the last (1.038 / 1.053 / 1.071,
   same amplitude today); the Level 2 form gives 1.015 / 1.020 / 1.025. The paper took its z = 0 value from σ8 (early normalisation), which produced the
   "crossover near z ≈ 0.2"; with one implementation and one normalisation there is none. The "D2 ratios" 1.052/1.075 match D⁴ of the last form, not D2.
   The chapter states all of this and keeps the unchanged F2 shape and the σ8(z) amplitude. (An earlier version of this file used the −1.87 % form as
   if it were the chains' implementation; corrected 2026-10-02.)

**sigma8 by level (final extraction):** L1 Planck-only 0.8143 → 0.8015 (−1.6 %); L2 0.8087 → 0.7998 (−1.1 %, = paper §12 '0.809 to 0.800'). The first draft printed '0.813 → 0.800 (1.6 %)', mixing the levels; corrected.
**Updated to the final chains:** Δχ² and free-µ0 values cite the Part 2 chain chapter (one extraction of the final files) rather than the paper's tables.
**Wording:** §13.5 "potential … actualized" rewritten in physics terms (records, low-entropy state).

**From the complete read (2026-10-02):**
6. **Constants (Eqs. 12, 15, 16).** G = 1/(4ħη) → c³/(4ħη); η = c⁴/(4ħG) → **c³/(4ħG)** (= 1/(4ℓ_P²), ℓ_P² = ħG/c³). This file previously listed c⁴
   as reproduced; that was wrong. Eq. 16 reaches 4ℓ_P² only by inserting κ = c²/ℓ_P into a dimensionally inconsistent expression. With the horizon first law
   δE = (κc³/8πG) δA and δE = ħκ/2π for one unit of entropy, δA_min = 4ħG/c³ = 4ℓ_P² for every κ — a cleaner result. "One bit per 4ℓ_P²": the unit is one
   nat (k_B); one bit is 4 ln2 ℓ_P². Same slips in IAM's Law (L5) and the Bekenstein paper (B4).
7. **Eq. 58** µ = H²_ΛCDM/(H²_ΛCDM + βE(a)) omits H0² (Eq. 82's E² form is right).
8. **Abstract, §9.2, §16:** "β_m verified against N-body-calibrated mass functions to 0.3 %", "overshoots the MCMC value … matches to 0.3 %": β_m is fixed in
   every chain (0.1583 = Ω_m/2 of the Level 2 chain); Ω_m f_coll η_vir with η_vir ≡ 1/(2f_coll) equals Ω_m/2 identically. Not a verification.
9. **§9.4, §15 item 5:** "the fitted β_m should shift with the Ω_m prior": β_m is not fitted. The test is free µ0 (done) and future growth data.
10. **§11.4, Fig. 1(b), §15 item 2:** f(R) gives µ > 1 with **Σ = 1** (not Σ > 1); self-accelerating DGP gives µ < 1, Σ = 1 (ghost, excluded). "Unique" → the
    signature IAM predicts among viable models. §15 item 6: DGP growth is scale-independent in the quasi-static regime; only f(R)-type models are scale-dependent.
11. **§12.6:** GW170817 75.5 (+5.3/−5.4) is cited to Abbott 2017 (70 +12/−8) and Nicolaou 2023; it is a 2024 afterglow analysis. Other analyses of the event
    give 68–70. One event is consistent with both 67 and 72.
12. **§12.7:** "DESI DR1 σ(µ0) = 0.22, 0.6σ": 0.6 is the prediction's distance from GR in units of σ, and 0.22 has no source; DESI 2024 full shape gives
    µ0 = 0.11 (+0.45/−0.54).
13. **§12.1:** Δχ² +1.43/+1.34 and free µ0 = 0.033 ± 0.125 ("1.3σ") are the companion paper's values at its date; final chains: Planck+RSD µ0 = 0.039 ± 0.125
    (1.4σ), Δχ² from CHAIN_EXTRACTION_FINAL.
14. **Eq. 75** integrates R/(T_H A_H) da′, which differs from Eq. 29 (per dt) by 1/(aH). Table 2's best fit at D^(7/2) agrees with per-unit-time accumulation
    (EXPONENT_LINE_BY_LINE). Print Eq. 75 per dt. §6.6 "coefficients within 5–10 %" → within 5 % for D^(7/2) (Table 2).
15. **§5.3** "Press–Schechter gives n ≈ 2.5–4" and §10.5 "PS extended formalism n ≈ 2.5": no derivation given; PS/ST d ln F/d ln D ≈ 1 at galaxy scale
    (NBODY_TRACE). Not carried. **§11.5 1-loop** (c₂ = −61/630, σ²_NL 0.30, −2.5 % both models): not reproduced; labelled as the source paper's.
    **Fig. 1 / §14.5:** MGCAMB form within "1–2.5 %": exact at z = 0, largest gap 2.8 % near z ≈ 0.65. **§11.2–11.3** "H_IAM > H_ΛCDM … Ω_m^IAM < Ω_m^ΛCDM":
    the matter-sector rate, background unchanged. **§15 item 4** β_γ < 10⁻⁴ (CMB-S4) stands as a forecast; current bound 0.0039 (D7).
    **Acknowledgments:** private correspondent named in the PDF (N1).
