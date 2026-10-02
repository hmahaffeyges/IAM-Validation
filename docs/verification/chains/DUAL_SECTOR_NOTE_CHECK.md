# DUAL_SECTOR_NOTE_CHECK — "On the Dual-Sector Structure of IAM: A Note on Null Geodesics, Proper Time, and Why the Sector Split Is Already in Einstein's Field Equations" (March 2026, 9 pp, informal)
Read in full 2026-10-02 (all 573 extracted lines, in 100-line chunks).

## What stands
- §1, §3 physics: timelike worldlines accumulate proper time (g_µν ẋ^µ ẋ^ν = −1, dτ > 0), null worldlines do not (= 0, dτ = 0); a process that needs Δτ > 0 to
  complete cannot run on a null worldline; gravitational decoherence is irreversible and needs finite Δτ; so S_info is produced by the matter sector only
  (Eq. 4: S_info > 0 iff dτ > 0) and the first law −dE = T_H d(S_geo + S_info) reduces to the standard form for photons. µ(a) Eq. 5, Σ = 1 Eq. 6 as in
  the Level 1/2 chapters.
- Chain bullets (§4): 17 converged chains, Δχ² +0.54 (Level 2), σ8 0.809 → 0.800, H0 67.16 ± 0.47 / 72.26, background version H0 ≈ 61.5 (now 10.9σ): ✓.
- §6 "potential tensions" (Σ ≠ 1; β_γ; β_m/Ω_m; scale dependence) — the right list of tests; wording corrected below.

## Corrected
1. **§3 "µ < 1 with Σ = 1 is unique … f(R) predicts µ > 1, Σ > 1; DGP predicts µ > 1"**: in f(R) gravity in the quasi-static regime lensing is unmodified,
   Σ = 1, with µ between 1 and 4/3 (e.g. Pogosian & Silvestri 2008). In DGP light deflection is also unmodified (Σ = 1); the self-accelerating branch has
   weaker effective gravity, µ < 1 (normal branch µ > 1). So µ < 1 with Σ = 1 is not unique to IAM; what is specific is µ(a) of Eq. 5 with no free parameter.
   The DGP statements are to be traced to a source before printing; the f(R) statement is standard.
2. **Table 1 row 1**: β_γ < 1.4 × 10⁻⁶ and "ratio > 10⁵" → β_γ < 0.0039 (95 %), β_γ/β_m < 0.025: photons couple at least 40× more weakly
   (`DUAL_SECTOR_VALIDATION_CHECK.md` #8; `scripts/verify_beta_gamma.py`). Same in Fig. 1(a) ("< 10⁻⁶"), §5, §7.
3. **Table 1 row 2 "Planck recovers β_m without fitting … the strongest single result"**: β_m is fixed in every chain; Ω_m/2 of the posterior (0.1583 ± 0.0033)
   is 0.2σ from the fixed 0.15765 — a consistency, not a recovery (`virial/VIRIAL_CHECK.md` #1). Same in §7.
4. **Table 1 row 3 "1,588 supernovae select matter sector independently"**: not supported; with M free the SN magnitudes carry no H0 information. What the
   supernovae show: ΛCDM distances, and the β term on distances excluded (`DUAL_SECTOR_VALIDATION_CHECK.md`). Same in §7.
5. **Table 1 row 4, §5, Fig. 2(g) "DESI phantom crossing is a predicted artifact at z ≈ 0.33–0.43"**: not yet shown. The mock two-ruler test
   (`observations/TWO_RULER_DESI_TEST.md`) gave the opposite quadrant (w0 −1.19, wa +0.38) when supernovae carried the full matter-sector H(z), and Pantheon+
   excludes that form. The real-data two-ruler test (matter normalisation only) is the test; status: open.
6. **Figs. 1 and 2 are pre-chain Python results**: H0(matter) 72.5 / 72.48 (chains: 72.26); σ8 = 0.7901, "2.6 % below ΛCDM" (chains: 0.800, −1.1 % Level 2,
   −1.6 % Level 1); "IAM reduces CMB lensing by 2.0 %" (Level 1 Limber estimate 0.05–0.3 %); "Δχ² = 79.8, 8.9σ, 65 points, 6 probes" (the pre-chain
   compilation; not reproduced by any chain); S8 "physical Ω_m = 0.753"; Fig. 2(f) labels Pantheon+ as photon sector. Not carried; the chapter uses chain values.
7. **"well below the 95 % exclusion threshold of 3.84"**: equal parameter count; likelihood ratio (as L3, P8).
8. **§6** "β_γ detected above 10⁻⁴": the current bound is 0.0039; restate as "a detection of β_γ > 0 by CMB-S4". "the fitted β_m": β_m is not fitted; restate as
   "if Ω_m is revised and the growth data require a β_m different from Ω_m/2".
9. **Language (physics-terms rule)**: "energy that has crossed into the timelike sector… duration, history", "actual timelike worldlines", "The philosophical
   argument explains why IAM is right. The numbers are the verdict." — not carried in Part 2; the physical content (timelike vs null, proper time,
   irreversibility) is kept.
