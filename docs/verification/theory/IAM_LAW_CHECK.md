# IAM_LAW_CHECK — *IAM's Law: The Thermodynamic Cost of Classical Existence* (March 2026, 29 pp)
Read status: 1,506 extracted lines. The first pass (100-line chunks) returned 11 chunks only as truncated previews; this file at first said
"no truncated reads", which was wrong. Every gap was then read in 50-line chunks the same day; ledger 1–1506 complete with no gaps (2026-10-02). Numbers recomputed (CODATA 2018, Planck 2018).
Prior checks it depends on: `EXPONENT_LINE_BY_LINE.md`, `THEORY_CHECK.md`, `virial/VIRIAL_CHECK.md`, `virial/NBODY_TRACE.md`, `cc_baryon/CC_AND_BARYON_CHECK.md`,
`chains/LATE_TIME_GROWTH_CHECK.md`, `black_holes/BLACK_HOLES_CHECK.md`.

## Reproduced
- µ(a) = H²_ΛCDM/(H²_ΛCDM + β_m E): 0.864 / 0.948 / 0.982 at z = 0 / 0.5 / 1; µ(z=0) − 1 = −β_m/(1+β_m) = −0.136 (β_m = 0.1575).
- w_info = −1 − 1/(3a) from ρ ∝ E(a) and the continuity equation; E(z) = e^(−z); E → e.
- Hoop/holographic identity Eq. 45; M_eq = c³/(4GH) = 2.3×10²² M⊙; Γ = c³/(1920 GM ln2).
- H0 × √(1+β_m): 67.4 → 72.51; 67.161 → 72.26 (Level 2 chain, the book's value).
- η = 2.738×10⁻⁸ Ω_b h²: 0.022319 → 6.113×10⁻¹⁰.
- Euclid σ(µ0) 0.04 → 0.136/0.04 = 3.4σ; Euclid + DESI Y5 0.03 → 4.5σ.

## Corrected (errata L1–L16)
1. **Wording** (abstract, §§1–2, throughout): "quantum potential to classical actuality" → the canon statement, "from quantum superposition to classical
   record" (CANON `IAM's law`). Physics terms only; "actualization", "cost of becoming", "potential does not reclaim" go.
2. **Landauer "a special case of IAM's Law"** (§1, §5.4): Landauer's bound uses the temperature of the reservoir that receives the heat. IAM's Law identifies
   that reservoir as the nearest horizon. This is the paper's Assumption 2 applied, not a derivation that contains Landauer; state it so.
3. **§5 the virial completion**: the second law reads ΔS ≥ Q/T_res with T_res the temperature of the surroundings, not the system's virial temperature.
   Steps 2–3 (ΔS = Q/T, E = TΔS) are an identity, so "the temperature cancels" carries no physics; ⟨K⟩ = Q follows from the first law plus Euler's theorem.
   "Unique partition … any other would violate Landauer" is not shown: Q = ⟨K⟩ holds for degree −1 by mechanics alone. Carry as in ch. 1.3 (identification,
   scoped to systems that release their binding energy). T_vir = mσ²/k_B for a 1-D dispersion (mσ²/3k_B is for the 3-D σ).
4. **Table 1**: electron 6.6 ppm — particle group, audited there; Sun "~10 %" → Kelvin–Helmholtz 24 Myr; Chandrasekhar 1.44 → 1.456 M⊙ (µ_e = 2);
   N-body 0.815 ± 0.025 → published 2T/|U| 1.1–1.3, 1.02–1.17 with surface pressure (V18); "β_m = 0.1583 ± 0.0033, 0.2σ, 17 chains" → β_m is fixed in every
   chain; 0.1583 is Ω_m/2 of the Level 2 chain (same in ΛCDM) (VIRIAL_CHECK #1). Applies also to §9, §12.1, Table 2, §13 A3, §14.3, §14.4.
5. **§6.1 Jacobson constants**: G = 1/(4ħη) → G = c³/(4ħη); Eq. 18 η = c⁴/(4ħG) → c³/(4ħG); "κ_min = c²/ℓ_P" is the maximum (Planck) surface gravity (B4).
   (A) S ∝ A "from causal accessibility" is interpretation; (B) the 1/4 = 2π/8π reading still takes G as input.
6. **Eq. 26**: ∫₀^a da′/a′² diverges; E(a) = exp(∫₁^a da′/a′²) = exp(1 − 1/a). The constraint ρ̇_info = ρ_info H/a (Eq. 25) is postulated here; its derivation
   is the Theory paper's Eqs. 28–39.
7. **§8 exponent**: Eq. 43 n − 9/2 = −1 ⇒ **n = 7/2** (author-approved 2026-10-02; EXPONENT_LINE_BY_LINE). Eq. 23 and §§3, 6.4, 13 change with it.
   Bottom-up: in Press–Schechter ν = δ_c/(σD) ∝ D⁻¹, not D^(−1/2); the "ST σ* = 1.2 fit r = 0.992" is not reproduced; the "four N-body studies n_eff = 3.33"
   report the power-spectrum slope (−1.4 to −1.7), a different quantity (NBODY_TRACE). The "two routes converge" claim is not carried.
8. **§7.4 uniqueness**: f(R) gives µ > 1 with Σ = 1 (not Σ > 1); normal-branch DGP µ > 1, Σ = 1; self-accelerating DGP gives µ < 1, Σ = 1 — the same sign as IAM —
   but carries a ghost and is excluded by data. Print "unusual", with sDGP named.
9. **§12.3**: √(1+β_m) = 1.076, not 1.073. Sirens: 75.46 (+5.34/−5.39) is one afterglow analysis of GW170817 (Phys. Rev. D 109, 063508, 2024); other
   analyses of the same event give 68–70 (e.g. 70.3 +5.3/−5.0). One siren is consistent with both 67 and 72; state as future test.
10. **§12.4**: "DESI DR1 σ(µ0) = 0.22, consistent at 0.6σ" — 0.136/0.22 is the distance of the prediction from GR in σ, not a consistency test; σ = 0.22 needs
    its source. DESI 2024 full shape: µ0 = 0.11 (+0.45/−0.54), consistent with both −0.135 and GR.
11. **§12.2 N-body confirmation** (Ω_m f_coll η_vir = 0.159): the η_vir values are not the published ones (V18); removed.
12. **§12.5 CC and baryon asymmetry**: the formula reduces to Ω_b/Ω_m = (3/16)√Ω_Λ, an observed present-epoch relation (0.5 %, 0.7σ on the CMB-only chain);
    "0.07 %" depends on the parameter set (Planck 67.4 / 0.0493 / 0.315 gives 0.9 %). The 2/π and the history weighting are not derived. The baryon chain
    does not test IAM (ΛCDM chains return the same Ω_b h²); the standard runs had no BBN prior either (Cobaya `ref` only); helium enters through CAMB's BBN
    relation. One relation, not two derived quantities (CC_AND_BARYON_CHECK).
13. **§14.3 forces**: the Chandrasekhar mass is electron degeneracy against gravity; the strong force does not enter. The weak-force "exception" is
    interpretation (out, as in the virial chapter).
14. **§14.4–14.5**: "the second law is a corollary", "a law, not a framework", the measurement problem and the "boundary between quantum and classical"
    landscape are interpretation → Part 5.
15. **§14.6**: "1/a exponent confirmed to 1 % by Sheth–Tormen" not reproduced (item 7); M–σ removed (M1).
16. **§14.7 Mahaffey number**: canon M = E_drive/(k_B T) (no ln 2); cell value 20.94 (30.2 only in Landauer units); the old cell drive symbol is retired; the cell criterion
    "A = H(β)/H_min(c) ≥ 1 breach" is retired (Met-A and IAM-A now); the quantum-processor report/the semiconductor report's "IAM's Law" relation is the IAM floor. Black hole: Mc²/(k_B T_BH) = 8πGM²/(ħc)
    = 2 S_BH/k_B (Smarr), and "M ≥ 1/(4ℓ_P²)" mixes a number with an inverse area.

17. (complete read) **Eq. 33** µ = H²_ΛCDM/(H²_ΛCDM + β_m E(a)) drops H0²: → H²_ΛCDM/(H²_ΛCDM + β_m E(a) H0²) (as Eq. 39).
18. (complete read) **Table 2** lists the Δχ² prediction as "≤ 0" against the result +0.54: the prediction is no added parameter; print Δχ² = +0.54
    as consistent (Δχ² < 2 with zero extra parameters), not as a met "≤ 0" prediction. The "0.3 %" N-body agreement in §13 A3 and "1.0 %" in §12.2 disagree
    with each other; both are withdrawn with the N-body row (item 11).

## For the book (Part 1, ch. "IAM's Law")
Carried: the law (canon wording); decoherence and the bit (Eqs. 2–5); the cost per bit at the horizon (Eq. 6); the virial completion (cross-reference
ch. 1.3); the Jacobson → Cai–Kim → S_info chain with the 1/4 reading; two information sources; E(a) and its properties with the corrected integral; the
variational form and w_info; β_m = Ω_m/2; the dual-sector split and δφ = 0; µ and Σ; numerical validation; n = 7/2 top-down; the three assumptions and
three failure levels; µ0 forecasts; the Mahaffey number per canon. Part 5: the landscape, law-vs-framework, arrow of time, coincidence, weak force.
