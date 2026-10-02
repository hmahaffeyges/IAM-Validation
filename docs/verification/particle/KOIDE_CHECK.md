# KOIDE_CHECK — "Three Charged Lepton Generations and the Koide Ratio from Horizon Information Equipartition on a Charge Orbit" (22 Apr 2026, 7 pp)
Read in full 2026-10-02 (PDF text 846 lines, ledger complete). Carried: `p2_particle_masses.tex` §"The charged-lepton pattern". Script: `scripts/verify_koide.py`.
## Reproduced
- Q_obs = 0.66666051 (PDG 2022 pole masses). Theorem 2 (Q = 2/3 for n = 3 and y²/x² = 2) exact, and it holds for any phase offset δ.
- Theorem 1 for δ = 0 (n = 2 and n ≥ 4 excluded).
## Corrections
1. **§V.D fixes δ = 0** ("maximum of √m at φ = 0", with a generation at φ = 0). With δ = 0 the two lighter masses are equal (26.9 MeV each). The measured
   masses need δ = 0.22227 rad (≈ 2/9, Brannen) and x² = 313.84 MeV. The paper's "no free assumption" chain does not reproduce the masses; δ and x are open.
2. **Theorem 1 with δ free**: n ≥ 4 is excluded for every δ (one phase is always within π/4 of π); n = 2 is admissible for half of all δ. Derived:
   at most three generations. "Exactly three" is not derived.
3. §III D: E_bit = k_BT_H (no ln 2) in the one-bit-per-area step, while §III A prices bits at k_BT ln 2; the "one bit per 4ℓ_P²" normalisation counts nats.
4. §III B "volume encoding is causally inaccessible, so S ∝ A is derived": this is the holographic argument of 't Hooft and Susskind, cited; not new.
5. Acknowledgements: personal thanks naming a correspondent; not carried (names rule).
