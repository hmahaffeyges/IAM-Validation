# THEORY_CHECK — IAM Theory paper (14 Apr 2026, 34 pp), read in full 2026-10-01, all 1,916 lines

Chapter: `part2_drafts/p2_theory.tex` carries §1–12, §14–15 in the paper's order; §13 (interpretation) is in `part5_drafts/p5_theory_interpretation.tex`.

**Reproduced:** Jacobson algebra (Eqs. 5–12); η = c⁴/4ħG and η δA_min = 1 (Eqs. 14–16); Cai–Kim (Eqs. 17–25); Eqs. 36–39; E(a) from
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
4. §12.3 Fig. 2: the "Δχ² = +0.75" is typed text in tests/plot_cl_comparison.py line 319, not computed from a chain. The book labels the figure from
   CHAIN_EXTRACTION_FINAL (Planck-only chain-minimum +0.96; paper table minimizer +1.43).

5. §11.5 bispectrum (no source script; author: not essential, 2026-10-02). Reproduced: normalising IAM and ΛCDM to the same amplitude today, B ∝ D⁴
   gives 1.038 / 1.053 / 1.071 at z = 0.3 / 0.5 / 1 (paper 1.033 / 1.052 / 1.072). The paper then applied a z = 0 suppression from σ8 (a different, early
   normalisation), which created the "crossover near z ≈ 0.2". With one normalisation there is no crossover; with the CMB-fixed (early) amplitude IAM is
   below ΛCDM at all z. The "D2 ratios" 1.052/1.075 match D⁴ (bispectrum), not D2 (today-normalised D2: 1.026/1.035); 1.014 at z = 0 matches nothing.
   The chapter keeps the unchanged F2 shape and states the amplitude follows σ8(z); the ratio and D2 tables and the crossover claim are not printed.

**sigma8 by level (final extraction):** L1 Planck-only 0.8143 → 0.8015 (−1.6 %); L2 0.8087 → 0.7998 (−1.1 %, = paper §12 '0.809 to 0.800'). The first draft printed '0.813 → 0.800 (1.6 %)', mixing the levels; corrected.
**Updated to the final chains:** Δχ² and free-µ0 values cite the Part 2 chain chapter (one extraction of the final files) rather than the paper's tables.
**Wording:** §13.5 "potential … actualized" rewritten in physics terms (records, low-entropy state).
