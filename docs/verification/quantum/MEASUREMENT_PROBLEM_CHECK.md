# MEASUREMENT_PROBLEM_CHECK — "The Measurement Problem Dissolved: Quantum Decoherence as Irreversible Sector Crossing" (Feb 2026, 21 pp)
Read in full 2026-10-02 (PDF text 388 lines, ledger complete). Placement: outline Part 5 (interpretation). Output: `scripts/verify_measurement_problem_output.txt`.
## Reproduced
F(Q,T) values (0.1 eV: 0.000 / 0.0024 / 0.164 at 10 mK / 4 K / 300 K); Q_L(300 K) = 0.018 eV; cat τ_IAM = 3.5e-51 s (with the paper's E_G).
## Contradicted by published experiments
1. **§3.3, §4.1–4.2 — erasure fails after spontaneous emission (F < 0.02).** Atom–photon entanglement from a spontaneously emitted photon (Blinov et al.,
   Nature 428, 153, 2004) and remote matter entanglement heralded by interfering spontaneously emitted photons (Moehring et al., Nature 449, 68, 2007;
   Hensen et al., Nature 526, 682, 2015: Bell S = 2.42 ± 0.20, electron spins 1.3 km apart). Which-path information carried by a ~eV emitted photon is
   erased by measuring the photon in a conjugate basis. §6.2 "predictions diverge only for experiments not yet performed" is not so.
2. **§3.5 "matter entanglement limited to ~1.3 m (2019)"**: 1.3 km in 2015 (Hensen).
3. **§3.7 — no Zeno effect with dispersive (Q ≪ Q_L) readout.** Observed with linear dispersive cQED readout and a near-quantum-limited amplifier
   (Slichter et al., NJP 18, 053031, 2016).
## Corrections of principle
4. **"A measurement is any interaction dissipating Q ≥ k_BT ln 2."** Landauer–Bennett: a measurement can be made with no minimum dissipation; the cost
   k_BT ln 2 per bit is paid when the record is erased or reset (Bennett 1982). The paper's own rule puts a photon in flight in the Σ = 1 sector, so a
   spontaneously emitted photon is not yet a record; the irreversible write happens where the photon is absorbed in matter. Read this way, IAM's Law
   agrees with items 1 and 3: the record is written in the detector or amplifier chain, and erasure is possible until then.
5. **F(Q,T) = 1 − exp(1 − Q_L/Q)/e** is the E_q ramp form reused; no derivation (GD3). The temperature test (§4.3) rests on it.
6. **§3.4 "the cat decoheres by its own gravity"**: environmental decoherence of a cat (air molecules, thermal photons) is faster than any gravitational
   channel by many orders of magnitude; the gravitational τ is not the operative one.
7. §2.2 "S transitions to the µ < 1 sector": µ and Σ are cosmological perturbation functions; using them as names for coherent/decohered states of a
   laboratory system conflates two things. §3.5 Eq. 3 uses D = E_q/e, inheriting GD3. "17 chains" → 18.
## For the book
Carried (Part 5, interpretation): IAM's Law gives "measurement" a physical definition — a measurement is complete when a record is written
irreversibly into matter, at k_BT ln 2 per bit at the surface where it is written; no observer is needed. This agrees with every experiment in items 1–3.
Not carried: the three "discriminating" experiments as stated (two are already performed and contradict them), the F(Q,T) formula, the cat timescale.
