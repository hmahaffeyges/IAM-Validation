# x_qp paper — what a superconducting-qubit referee will ask, and the corrected model
"Landauer-Based Thermodynamic Model for the Minimum Quasiparticle Density in Al/AlOx/Al Josephson Junction Transmon Qubits" (14 Apr 2026, submitted).
Written 2026-10-02 after a full read (854 lines). Numbers: `scripts/verify_xqp.py` (output beside it). Note: my own review two weeks ago passed this
paper with one point (the τ_TLS range); the three points below were missed then.

## 1. The Cooper-pair density (decisive)
The paper uses n_cp = n_e/2 = 9.03 × 10²⁸ m⁻³ = 9.0 × 10¹⁰ µm⁻³ (all conduction electrons). In the qubit literature that defines the measured floor,
x_qp = n_qp/n_cp with n_cp = 2ν₀Δ ≈ 4 × 10⁶ µm⁻³ (only electrons within ~Δ of the Fermi surface): Wang et al., Nat. Commun. 5, 5836 (2014);
arXiv:2402.15471; arXiv:2208.02790; Glazman & Catelani, arXiv:2003.04366. The two differ by 22,600×.
- The formula as printed gives a quasiparticle density of **5,900 µm⁻³** at a site. The measured floor x_qp ~ 10⁻⁷ is **0.4 µm⁻³**: 15,000× lower.
- With the field's n_cp the same formula gives x_qp = 1.5 × 10⁻³.
The factor-1.5 agreement comes from comparing a number computed with one n_cp against a floor measured with the other.

## 2. Energy per event
One erasure dissipates at least Δ ln 2 = 126 µeV. Creating quasiparticles means breaking a pair: 2Δ = 364 µeV, two quasiparticles. Energy released
below 2Δ goes into sub-gap phonons, which cannot break pairs. Landauer fixes the minimum heat, not the quasiparticle yield. "ln 2 quasiparticles per
event" (Eq. 13) needs a pooling mechanism; without one, an event either breaks a pair (yield 2) or yields none.

## 3. Volume
Eq. 16 keeps each site's quasiparticles inside πλ_L³ (λ_L = 50 nm) for τ_qp = 100 µs. Quasiparticles in Al diffuse √(Dτ_qp) ≈ 250–800 µm in that time
(D ≈ 0.6–6 µm²/ns), so the qubit literature treats electrodes as having a uniform density (arXiv:2402.15471). The measured x_qp is the island
average; the steady-state density depends on the number of active sites N and the island volume V, neither of which is in the formula.

## The corrected model
With one pair broken per erasure event, N active sites, island volume V, and the field's n_cp:
  x_qp = 2 N τ_qp / (τ_TLS n_cp V)
| island volume V | sites needed for x_qp = 10⁻⁷ | x_qp from one site |
|---|---|---|
| 10³ µm³ | 60 | 1.7 × 10⁻⁹ |
| 10⁴ µm³ | 600 | 1.7 × 10⁻¹⁰ |
| 10⁵ µm³ | 6,000 | 1.7 × 10⁻¹¹ |
A handful of pinholes cannot supply the floor. Hundreds to thousands of active two-level systems on the junction leads and island surfaces could, and
that number is measurable by TLS spectroscopy (Lisenfeld et al. map TLS mostly on the junction leads).

## What stands, stated defensibly
- The idea: part of the floor is endogenous, the dissipation of maintaining coherence against the TLS bath, paid at the junction. Compatible with, but
  not established by, the evidence: the underground and shielding results (Gran Sasso; Gordon et al. 2022) show T1 is not radiation-limited at present
  lifetimes and that much of the measured QP tunnelling is external; a thermalised gap-edge distribution follows from fast phonon relaxation for any
  source (Connolly et al. 2024 does not single out an internal one). Corrected 2026-10-02 after reading the encoding-surface working notes.
- The test becomes sharper, not weaker: on one device, measure x_qp, τ_qp, τ_TLS, the active TLS count N and the island volume V. The prediction is
  x_qp n_cp V τ_TLS / (N τ_qp) = 2. It is falsified if x_qp does not scale with N/V across devices.
- Not parameter-free: N is a device property that must be measured. That is honest and still a single-device test.
- The Landauer link: the erasure sets the event rate (1/τ_TLS per site) and the minimum dissipation; the pair-breaking quantum 2Δ sets the yield.

## Recommendation
The paper is under review. Ask the editor to let you replace it with the corrected version before a referee reports, or withdraw and resubmit.
Points 1–3 are the first things a referee in the field checks; a revision that states them and gives the corrected model is defensible.
