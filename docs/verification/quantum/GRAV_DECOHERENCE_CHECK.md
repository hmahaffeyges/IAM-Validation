# GRAV_DECOHERENCE_CHECK — "Gravitational Decoherence from Dual-Sector Thermodynamics: Predictions for Optomechanical Experiments and Quantum
Computing Architecture" (Feb 2026, 22 pp). Read in full 2026-10-02 (PDF text 478 lines, ledger complete; pages are figure-heavy, page 6 rendered to
confirm no text is missing). Placement: Part 2 §2.11, `part2_drafts/p2_quantum_records.tex` §"Massive superpositions". Script: `scripts/verify_grav_decoherence.py`.
## Reproduced
- τ ∝ T² (×4 at 20 mK, ×16 at 40 mK); dE_q/dη peaks at η = 0.5; crossover with Penrose–Diósi at 2.3e-10 kg (10 mK).
- §7.1: gravitational decoherence is irrelevant for current qubits; photons exempt (E_G = 0, Σ = 1).
## Corrections
1. **τ_IAM for 1e-12 kg at 10 mK = 559 s**, not 560 µs (text) or 4,687 µs (Fig. 1 label). τ_PD = 7.8 µs. The "mesoscopic frontier ~100 µs" framing
   and the §5.4 detection timeline need recomputing with the correct τ.
2. **Mass scaling m⁻⁵**, not m⁻⁶ (E_G ∝ m^(5/3), τ ∝ E_G⁻³); the paper's own crossover reproduces with m⁻⁵. Δα = 3.33, not 4.33. Fix in §3.3, Fig. 5,
   Table 1, §8 item 2.
3. **The ramp does not follow from Eq. 6** (hold). The integrand is constant, so ln E_q = t/τ and E_q = e^(t/τ); exp(1 − 1/η) is asserted by analogy
   with E(a). The ramp is the main discriminator; it needs its own derivation at the superposition boundary (the analogue of the 1/a² horizon density).
4. **S_boundary = k_BT/E_G** (Eq. 6) is introduced without derivation; it sets the T² law.
5. §1.2 "17 chains": 18. §4.2 rate peak η ≈ 0.23 (Fig. 6d) vs 0.5 (Fig. 2): 0.23 is the purity-difference peak, not the rate peak; label.
6. τ ∝ T² (hotter = more coherent) runs against every environmental decoherence channel; it must be stated as the sharpest test and the most exposed claim.
