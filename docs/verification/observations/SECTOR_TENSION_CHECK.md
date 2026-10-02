# SECTOR_TENSION_CHECK — "Dark Energy or Sector Tension? IAM Confrontation with DESI Full-Shape Growth Rates and Joint Weak Lensing" (Mar 2026, 20 pp)
Read in full 2026-10-02 (PDF text 934 lines, 50-line ledger, no gaps). Reproduction: `scripts/verify_sector_tension.py`; mock: `TWO_RULER_DESI_TEST.md`.

## Reproduced
- µ(z) at the six DESI DR1 tracer redshifts (Table 1, as 1 − µ). H_matter = 67.16√(1+β_m) = 72.26; SH0ES offset 0.75σ.
- Level 2 σ8 0.7998 ± 0.0058 vs 0.8087; S8 0.822; joint lensing σ8 0.802 ± 0.020 at 0.1σ; survey S8 tensions 1.3–2.5σ.
- DESI DR1 fσ8: neither model preferred (our test: DESI χ² 4.51 ΛCDM, 5.24 MGCAMB-form; SDSS 6.19, 6.95).

## Corrections
1. **Table 5 / §5.2 DESI DR2 values**: "BAO+CMB −0.838/−0.62" is DESI+CMB+Pantheon+; "Union3 −0.752/−0.82" and "DES 5YR −0.734/−1.05" match no DR2 row.
   DR2 (arXiv:2503.14738): DESI+CMB −0.42 ± 0.21, −1.75 ± 0.58 (z_cross 0.50); +Pantheon+ −0.838, −0.62 (0.35); +Union3 −0.667, −1.09 (0.44);
   +DES Y5 −0.752, −0.86 (0.41).
2. **"13.6 % growth suppression", "7–8 % at z ≈ 0.3", Table 1, Table 5 last column, §7**: these are 1 − µ. fσ8 deficit: 2.2 % (BGS), 1.3, 0.8, 0.5,
   0.2, 0.1 %; 4.2 % today. The paper's own Table 3 (≲ 2 % at z < 0.5) contradicts its Table 1 wording.
3. **§5.1 mechanism**: DESI DR2 w0wa fits use BAO + CMB + SN distances only; no growth data, so distance–growth mixing cannot produce them (V24, S10).
4. **§5.1 sector list puts SNe in the photon sector**: author ruling 2026-10-01, SNe are on the matter ruler. With the matter-ruler shape H_m(z) the mock
   gives w0 −1.19, wa +0.38 (opposite quadrant); Pantheon+ rejects that shape (Δχ² +23.6, DSV check). The term does not produce the DESI preference.
5. **§5.4 "lensing convergence, CMB lensing, E_G identical to ΛCDM"** contradicts §5.1 (κ listed as matter sector). Σ = 1 fixes the potential–density
   relation; δ_m is lower, so lensing power is lower (CMB C_φφ −0.08 %) and E_G = Ω_m0Σ/f rises (+1.9 % at z = 0.3, +3.6 % today).
6. **β_m "recovered by the posterior at 0.2σ"** (abstract, §2.5, §7): β_m is fixed in every chain.
7. **β_γ/β_m < 8.5 × 10⁻⁶** (§4.3): from the reversed-array bound; corrected β_γ < 0.0039 gives 0.025.
8. **§6.2 "µ < 1, Σ = 1 not achievable in any published framework"; "DGP produces µ > 1"**: self-accelerating DGP gives µ < 1, Σ = 1 (LG12).
9. **Chain numbers**: "Δχ² ≤ +2.32" → ≤ +1.73 (final chains); free µ0 0.006 → 0.015 ± 0.156 (Planck), +0.039 ± 0.125 (Planck + RSD; paper 0.033); 17 → 18 chains.
10. **Table 3 fσ8 values** (0.377, 0.514, 0.484, 0.422, 0.377, 0.435) differ in 5 of 6 bins from the values derived from arXiv:2411.12021 ShapeFit
    ratios in our fσ8 test (0.397, 0.549, 0.479, 0.438, 0.373, 0.435); to be traced to the DESI table before any fσ8 value is printed.
11. **§2.3 "E(a) recovered to 1 % by Sheth–Tormen"**: not reproduced (ST7). **Naming**: §1, §6.1 name a cosmologist; the book cites the papers by
    journal/arXiv only.
