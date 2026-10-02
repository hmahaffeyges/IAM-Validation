# ELECTRON_MASS_CHECK — "Electron Rest Mass from Holographic Horizon Thermodynamics" (Feb 2026, 10 pp)
Read in full 2026-10-02 (PDF text 485 lines, ledger complete). Carried: `part2_drafts/p2_particle_masses.tex` §"The electron as a fixed point".
Reproduction: `scripts/verify_electron_mass.py`.
## Reproduced
- Eq. 14 at H0 = 67.4: 9.10944e-31 kg, +6.6 ppm. Without the prefactor: 1.2018 m_e; required factor 0.832107 vs (2π)^(-1/10) = 0.832112.
## Corrections
1. **"6.6 ppm, no free parameters"**: m_e ∝ H0^(2/5); σ(H0) = 0.54 → ±3,200 ppm. H0 = 67.36 (Planck best) → −231 ppm; 73.0 → +3.3 %. The (2π)^(-1/10)
   is identified by search (§3.5 "78× closer than any other combination"), and (2π)^(3/10) of it is underived; §8 "trusted because it reproduces the
   mass" makes it a fitted factor. Statement: agreement within the 0.3 % set by H0, one factor identified numerically.
2. **§7 "the same fixed point mc² = E_bit × N holds for the black hole"**: N k_BT_H ln2 = Mc²/2 (Smarr). The black hole closes with ½.
3. §1 is empty in the PDF (heading only); "equation (??)" broken reference; the 3/2-origin paragraph is printed three times (§3.3, §3.4, §4).
4. §9 "the same ingredients give n = 3 generations in the companion paper": see KOIDE_CHECK (at most three).
5. §3.3 "scale invariance across 37 orders verified in Ref. 12": the virial ½ holds for 1/r systems; it does not verify N ∝ (m_P/m)^(3/2).
