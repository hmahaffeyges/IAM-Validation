# WZ_FAR_FUTURE_CHECK — "Dark Energy Evolution and the Far Future of an IAM Universe" (Feb 2026, 19 pp)
Read in full 2026-10-02 (PDF text 609 lines, 50-line ledger, no gaps). Reproduction: `scripts/verify_wz_far_future.py` (output beside it).

## Reproduced
- w_info = −1 − 1/(3a) follows from ρ_info ∝ E(a) and energy conservation (dlnE/dlna = 1/a). w_info(1) = −4/3; dw/da = 1/(3a²) > 0; no Big Rip (E ≤ e).
- H∞ = H0√Ω_Λ = 55.78. Maturity table (Table 1): every scale factor, redshift and age reproduces to the printed digit.
- Today's rate (1/e)H0 = 2.536 % Gyr⁻¹.
- The matter-sector rate H_m² = H² + β_mE H0²: 72.52 today (H0 = 67.4), → 71.12 asymptotically; photon sector → 55.78.

## Corrections
1. **wa sign words** (§2.3, §6.2, Fig. 3 caption, §8): "small positive wa" → wa = −1/3 (negative). The value is printed correctly.
2. **Eq. 14** d(E/e)/da = 1/(e a²) → E/(e a²) = e^{−1/a}/a². It peaks at a = 0.5, not a = 1. "Inflection point at a = 1" is true in ln a only:
   dE/dln a peaks today. In cosmic time dE/dt peaks at z = 1.26 (3.38 % Gyr⁻¹). "Maturing faster than at any other epoch" holds per e-fold, not per Gyr.
3. **§6, §5.1 DESI**: DESI DR2 has w0 > −1 today (−0.83, −0.75), crossing −1 near a ≈ 0.77; w_info is −4/3 today, 7.7–9.1σ from DESI's w0.
   "DESI already hints at this … iam predicts exactly this" is not supported. More basically, the paper states the background is ΛCDM (Eq. 10), so on
   photon-ruler distances the informational term predicts w = −1; w_info belongs to the matter-sector rate. The CPL comparison is a category mismatch;
   the test is the two-ruler comparison (V24, S10).
4. **§6.4 item 2** (Roman high-z test of the strongly phantom early w): ρ_info ∝ E → 0 (1.2 % of ρ_Λ at z = 3); not observable.
5. **§7.2** "E(1) = 1 follows from the choice of the Planck epoch as reference": E(1) = 1 because 1 − 1/a = 0 at a = 1 (today's normalisation).
   The anthropic remark (observers appearing at peak production) is not carried (speculation rule).
6. **§2.1** w_info "from a scalar field constrained to the encoding surface": no field is introduced; the result follows from ρ_info ∝ E(a) alone.
7. Abstract "17 MCMC chains" → 18. DESI Year 5 "∼1 % w at multiple redshifts" cites the 2013 Snowmass white paper; not carried as a figure.

## Interpretation (to the Part 5 closing chapter)
§5.2–5.3 heat-death arguments ("equilibrium with the horizon", "maximally encoded, not maximally disordered"): interpretation. The photon-sector far
future is the ΛCDM de Sitter state; what ends is the writing (E → e).
