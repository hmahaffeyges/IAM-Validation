# TECH_REFERENCE_CHECK — "The Informational Actualization Model: A Technical Reference for Physicists" (27 Sep 2026, 28 pp)
Read in full 2026-10-02 (PDF text 1,298 lines, 50-line ledger, no gaps). A summary of ~40 papers; no new derivation. Not carried as a chapter. Its
weakness list (§10) and errata (§12) are checked here against the book's current state.

## Its §10/§12 items, status now
| item | status |
|---|---|
| Smarr: TS = Mc²/2, ln 2 not twice | applied (V17); black-hole chapter |
| S_BH formula 1/c² low (M–σ, Hubble2Methyl) | M–σ abandoned and removed; Hubble2Methyl not carried |
| M–σ exponent from Faber–Jackson, not PN order | M–σ abandoned |
| Δχ² = 79.8 "8.9σ" in root README and chains README | **fixed 2026-10-02**: labelled a simplified compilation, the MCMC +0.54 stated as the result |
| mgcamb README "all 12 runs R−1 < 0.01" | **fixed**: true for the final files (r2 for runs A, B); all 18 final chains ≤ 0.010 |
| Enigma `_decode_n` constants in `qproc_derivation_tests.py`, `chip_derivation_tests.py` | **still present**; the quantum-processor report/the semiconductor report parked (author); must be removed with n values published before any the quantum-processor report/the semiconductor report submission |
| two A-score definitions; class H_min; AIBL 46 % below 1; seminoma inversion | superseded by Met-A (per-cell floors, CANON 2026-10-01); class H_min retired |
| β_m uses present-day Ω_m (epoch-specific) | open; carried into the predictions/derivations appendix |
| dual-sector consistency (∇T = 0, Bianchi) | the covariant scalar functional (BH Thermodynamics) is cited in `p2_blackholes`; the Lensing/3-Way "linearized Einstein equations unmodified" argument implies M_lens/M_dyn = 1 (LD1, TW1), so it cannot also support the 15.7 % prediction |

## Corrections to the Technical Reference itself
1. §5.2 "β_m Planck posterior 0.1583 ± 0.0033, 0.2σ"; "β_m closure"; "η_vir 0.815 ± 0.025 (6 N-body studies)": β_m is fixed in every chain; the N-body values do not trace (V31, EG2).
2. §5.2 "11 of 13 below threshold" / "14 of 17": superseded by the r2 files; all 18 final chains ≤ 0.010.
3. §5.3, §8 #2 "M_lens/M_dyn 15.7 % — most near-term test": holds only in the G_eff form (LD1, TW1).
4. §5.5, §8 #5 "DESI DR2 w0–wa is a live test of w_info": DESI distances follow the photon ruler (w = −1); w_info is the matter-ruler equation of state (WZ3).
5. §5.4, §5.5 "E(1)/e at the inflection point, peak production": per e-fold only (WZ2).
6. §5.5, §8 #4 σ_crit ≈ 4 km/s "consistent with the observed absence": rejected by the satellite census (MISSING_SATELLITES_CHECK).
7. §5.5 "zurek_paper supplies n = 5/2": the exponent is n = 7/2 (THEORY_CHECK).
8. §5.6 "~10² offset is one problem" (cusp-core × missing satellites): both are side tests outside the book (author rule).
9. §8 #1 "Euclid DR1 (Oct 2026)": complete DR1 mid-2027; σ(µ0) ≈ 0.04 is the full-survey forecast.
10. §2.1, §5.11 cellular column (H_min(class), A = H(β)/H_min, five substrates, IAMAtlas v0.1): replaced by Met-A, IAM-A and the C-scores (CANON).

## For the appendix (future derivations and tests named here)
- 1/4 = 2π/8π derived independently (P3); β_m in a time-invariant form; the (2π)^(3/10) electron-mass factor (particle group);
  the x_qp single-device invariant (quantum group); gravitational-decoherence τ ∝ T² and ramp profile (quantum group); DNMT1 fidelity 15–42 °C.
