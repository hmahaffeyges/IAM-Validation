# ENTANGLEMENT_CHECK — "Entanglement, Decoherence, and the Thermodynamic Cost of Classical Records" (Mar 2026, revised Oct 2026, 6 pp)
Read in full 2026-10-02 (PDF text 253 lines, ledger complete). The October revision already carries τ_IAM(1 pg, 10 mK) = 559 s, τ_PD = 7.8 µs, m⁻⁵,
crossover 2.3e-10 kg, Hensen 1.3 km, n = 7/2. Placement: Part 2 §2.11, `part2_drafts/p2_quantum_records.tex` §"Light writes nothing in flight".
## Reproduced
Table 1 (5.6e17 s / 0.78 s at 1 fg; 559 s, 2,238 s / 7.8 µs at 1 pg); E(z = 10) = 4.5e-5; n = 7/2 from S_info ∝ a^(n−9/2).
## Corrections
1. §3, §6 "β_m posterior 0.1583 ± 0.0033, 0.2σ": β_m is fixed at Ω_m/2 in every chain (V-series, TR1).
2. §6 "17 chains; R−1 < 0.01 for 15, 0.023–0.024 for two": 18 chains, all ≤ 0.010 in their final files (the two at 0.023 were superseded by r2).
3. §6 "Euclid + DESI 5.4σ": unsourced (SP5); only Euclid σ(µ0) ≈ 0.04 is sourced.
4. Eq. 3 S(t) = 2√2(1 − E(t/τ)/e): inherits the underived ramp (GD3).
5. §3 Step 3 "the kinetic half is the heat dissipated": state IAM's Law first; the virial half sets how much bound matter writes (law-first rule).
6. §4 "Sci. Adv. 2025" reference has no authors or article number; trace or drop.
7. Acknowledgements name a correspondent: remove (names rule).
## For the book
§7 states IAM's Law with T = the temperature of the system that absorbs the record: the same correction the measurement chapter needs (MP4). Carried.
## Book chapter (2026-10-02)
Full treatment: `docs/book/part4/p4_21_entanglement_records.tex` (\label{ch:entanglement}); the n = 7/2 exponent, "light writes nothing in flight" and
τ_IAM at 1 pg stay in `p4_14_quantum_records.tex`. Numbers: `docs/verification/scripts/verify_entanglement_electroweak.py` (E1–E7).
## Found while writing the chapter (in PAPER_ERRATA.md as EN8-EN9, 2026-10-02)
8. Eq. 3, S(t) = 2√2(1 − D(t)): this is the CHSH maximum for isotropic (white) noise. Gravitational decoherence is dephasing in the pointer
   (which-path) basis: correlation matrix diag(c, −c, 1), Horodecki criterion gives S_max = 2√(1 + c²) (> 2 for every c > 0; = 2 at c = 0, never 0).
   With the pure-state optimal settings S = √2(1 + c), lost below c = 0.414. The "|S| < 2 once D > 0.293" threshold applies to isotropic noise only.
   Book files that still carry the isotropic form: `part4/p4_06_nonlocality.tex` (equation and "falls below 2 when D > 0.293"), `part7/p7_07_predictions.tex` Q5.
9. Item 6 resolved: the "Sci. Adv. 2025" reference is K. Wang et al., Sci. Adv. 11, eadr1794 (2025), doi:10.1126/sciadv.adr1794 (CrossRef-checked);
   the violation is computed on postselected four-photon coincidences.
