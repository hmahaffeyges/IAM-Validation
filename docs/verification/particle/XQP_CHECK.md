# XQP_CHECK — "Landauer-Based Thermodynamic Model for the Minimum Quasiparticle Density in Al/AlOx/Al Josephson Junction Transmon Qubits" (14 Apr 2026, 8 pp)
Read in full 2026-10-02 (PDF text 854 lines, ledger complete). Placement: opens Part 3. Carried with the corrected model as `part3_drafts/p3_xqp.tex` (author, 2026-10-02: 'I just want it fixed for the book').
## Reproduced
x_qp = ln2·τ_qp/(n_cp τ_TLS πλ_L³) = 6.516e-8 at τ_TLS = 30 µs; 1.00e-7 at 19.5 µs; 1.96e-6 to 1.96e-8 over 1–100 µs. V_eff = ∫e^(−2r/λ)d³r = πλ³ exact.
T_gap = 2.112 K. T1 = 1/(x_qp ω_q) = 0.32 ms at 1e-7, 5 GHz. Invariant Eq. 18 = 0.694.
## Physics hold (must be resolved before the chapter)
0. **n_cp definition (decisive)**: paper 9.03e28 m⁻³ (n_e/2); field n_cp = 2ν₀Δ ≈ 4e6 µm⁻³. Predicted site density 5,900 µm⁻³ vs measured floor
   0.4 µm⁻³ (15,000×). Full note and corrected model: `XQP_REFEREE_NOTE.md`.
1. **Energy per event below pair breaking.** E = Δ ln2 = 126 µeV; breaking a Cooper pair costs 2Δ = 364 µeV and yields two quasiparticles. "ln 2
   quasiparticles per event" (Eq. 13) counts energy/Δ; Landauer gives a minimum dissipation, not a quantised deposit. The mechanism needs either
   accumulation of sub-threshold dissipation into pair breaking or a different counting.
2. **Volume.** Eq. 16 keeps each site's quasiparticles inside πλ_L³ (λ_L = 50 nm) for τ_qp, but quasiparticles diffuse √(Dτ_qp) ≈ 100–1,000 µm in
   100 µs. Measured x_qp is a device average; the formula has no site count and no device volume. As written it gives the local fraction at one site.
3. §"temperature structure" e^(−Δ/k_BT) at 15 mK is 10^(−61) (paper 10^(−630); the later section has 10^(−61)).
4. Prediction 2 (no T_fridge dependence below 150 mK) is also what non-thermal (generated) quasiparticle models predict; it does not discriminate.
5. Prediction 3: the floor ratio between materials also depends on τ_TLS and τ_qp, not only λ_L³n_cp.
## For the appendix
The single-device invariant test (Eq. 18) stands as a test once items 1–2 are resolved in the formula it tests.
