# BLACK_HOLES_CHECK — the four black-hole papers (G6), read in full 2026-10-02 in 100-line chunks
Papers: *The Cessation of Projection … Information Paradox* (Feb 2026, 453 lines); *Black Hole Horizons as Thermodynamic Encoding Surfaces* (25 Feb 2026,
431 lines); *The M–σ Relation from Gravitational Decoherence Thermodynamics* (25 Feb 2026, 367 lines); *The Geometric Origin of the Bekenstein–Hawking
Entropy Coefficient* (Apr 2026, 596 lines). Numbers: `scripts/verify_virial_atoms_to_horizon.py` and the cells below (CODATA 2018).

## Reproduced
- T_BH = ħc³/(8πGMk_B): 6.17×10⁻⁸ K (1 M⊙). S_BH/k = 4πGM²/(ħc) = 1.049×10⁷⁷ (1 M⊙) = 1.513×10⁷⁷ bits. Table 1 (BH Thermo) S, Γ, τ_evap all ✓.
- σ_SB A T⁴ = ħc⁶/(15360πG²M²) ✓ (ratio 1). Γ = P/(k_BT ln2) = c³/(1920 GM ln2) = 152.5 bits/s (1 M⊙) ✓. τ_evap = 5120πG²M³/(ħc⁴) = 2.10×10⁶⁷ yr ✓.
- T_GH = ħH0/(2πk_B) = 2.66×10⁻³⁰ K; M_eq = c³/(4GH0) = 2.32×10²² M⊙ ✓. Hoop/holographic identity Eq. 11 ✓. Seed formula Eq. 12: 10⁷⁷ nats → 0.98 M⊙ ✓.
- T = ħκ/(2πk_B c) for black-hole, de Sitter and Rindler horizons (standard; with κ in acceleration units the c belongs in the denominator).
- Smarr: T_H S = ½Mc² exactly (Schwarzschild); Kerr share √(1−χ²)/2.

## Corrected
1. **Landauer energy of a black hole** (M–σ §2 Eq. 5, §4.3, Table 2; BH group): (ln 2/2) Mc² = 34.7 % → **½ Mc²**; ln 2 applied twice. Also not spin-independent:
   for Kerr the share is √(1−χ²)/2. (errata V17)
2. **S_BH written 4πGM²/(ħc³)** (M–σ Eq. 2, Eqs. 4, 9): → 4πGk_BM²/(ħc). (errata V20)
3. **P_SB = P_Hawking "is a physical statement, not a consistency check"**: the formula ħc⁶/(15360πG²M²) used as "the Hawking luminosity" is itself the
   black-body estimate σ_SB A T⁴, so the ratio is 1 by construction. Hawking's emission includes greybody factors and per-species counting (Page 1976);
   the true luminosity differs from the black-body value. Print: "in the black-body approximation". Γ = P/(k_BT ln2) is the first law: c²|dM/dt|/T_H = −dS_BH/dt.
4. **"Page curve from IAM"** (Info Paradox §6): S_rad(t) = ∫ Γ dt = S_BH,0 − S_BH(t) rises monotonically and reaches S_BH,0/2 at t = (1 − 2^(−3/2)) τ = 0.646 τ,
   not τ/2. It is the first-law transfer of coarse-grained entropy; the Page curve is the fine-grained entropy, which rises and then FALLS to zero. Not a
   derivation of the Page curve. Also S_BH,0 (1 M⊙) "2.6×10⁷⁶ bits" → 1.51×10⁷⁷.
5. **M–σ — abandoned 2026-10-02 (author: "if it doesn't look like it's going to be in our favor we just abandon M–σ and remove it").** The paper's Eq. 12
   (from the c³ typo) is not dimensionally a mass; with the correct S_BH and η = 1 it gives 2.7×10⁹ M⊙ at 200 km/s (needs unstated η = 0.006). Redone at the
   correct order of the irreversible channel, radiation reaction (σ/c)⁵: exponent 5.5 (observed 5.64 ± 0.32, McConnell & Ma 2013) but M_BH(200 km/s) =
   4.7×10⁴ M⊙ against 2.1×10⁸ — 4,500× low, needing η = 2×10⁷ where η ≤ 1; the most generous inputs (K_FJ 3×10¹¹, f = 1) give 3.6×10⁵, 580× low.
   The paper's "slope 4.05 ± 0.07 from 115 galaxies" is a fit to 18 galaxies hand-entered in `code/Koide/scripts/Virial tests/cross_scale_validation.py`.
   The M–σ paper (PDF, LaTeX) and its register entries were removed from the repo; remaining mentions in other sources are listed in PAPER_ERRATA (M1).
6. **Bekenstein coefficient**: Eq. 3 G = c⁴/(4ħη) → c³/(4ħη) (Unruh T = ħκ/(2πck_B)); Eq. 16 drops a factor c (c⁴/(4ħG) ≠ 1/(4ℓ_P²)). Substance: Jacobson's
   relation η = c³/(4ħG) still takes G as input; the paper shows where the 1/4 comes from (the 2π of the Unruh period over the 8π of the field equations),
   which is a clear statement of Jacobson's result, not removal of the input. "κ_min = c²/ℓ_P" is the maximum (Planck) surface gravity.
   The S ∝ A argument (§2, causal accessibility) is interpretation.
7. **Acknowledgments quote a private letter** (Bekenstein paper): correspondents are not named without permission (author rule).

## For the book
Carried: horizon thermodynamics (Bekenstein, Hawking, Gibbons–Hawking, Unruh), T = ħκ/2πk_Bc, the encoding throughput as the first law, M_eq and the
two-horizon flow, hoop = holographic bound, Smarr ½ with Kerr, the 1/4 = 2π/8π reading of Jacobson. Part 5: projection-cessation picture, paradox
dissolution, cosmic censorship, BH–electron "fixed point". Held: M–σ derivation, seed masses, z-evolution, JWST claims.
