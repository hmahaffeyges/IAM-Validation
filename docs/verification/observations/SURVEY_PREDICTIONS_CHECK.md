# SURVEY_PREDICTIONS_CHECK — "Falsifiable Predictions of the IAM for Euclid, DESI, and Next-Generation Surveys" (25 Feb 2026, 11 pp)
Read in full 2026-10-02 (PDF text 736 lines, 50-line ledger, no gaps). Reproduction: `scripts/verify_survey_predictions.py`.
**Placement (author, 2026-10-02):** prediction papers and prediction figures go into the predictions appendix, built after the cosmology,
particle and quantum groups. This file fixes the values the appendix will carry.

## Reproduced
- Table 3 µ(z) (0.886 at z 0.1 … 0.998 at z 2). Table 4 activation milestones (1 %–90 % of 1 − µ0: z 2.28 … 0.06, lookbacks 10.9 … 0.9 Gyr) exactly.
- Fig. 4(a) E(a) milestones 10/50/90 % at z 2.30/0.69/0.11. Sirens 67.161√1.1575 = 72.26.

## Corrections
1. **Table 5 growth deviations** are not the growth equation's: ΔD/D −0.78 % today (paper −6.8), −0.37 % at z 0.3 (−3.9), −0.06 % at z 1 (−0.9);
   Δ(fσ8)/fσ8 −4.25 % today (paper −10.2), −2.17 % at z 0.3 (−5.9), −0.41 % at z 1 (−1.3). The paper's columns are ≈ Δµ/2 and 3Δµ/4.
2. **Table 5 ΔΦ/Φ = Δµ**: with Σ = 1 the lensing potential follows δ_m, so ΔΦ/Φ = ΔD/D (−0.78 % today), not −13.6 %.
3. **ISW A = 1.134 (+13.4 %)**: the growth-equation ISW source is ~3 % larger (S8_TREND_CHECK #5); Fig. 2's 10–30 % kernel enhancement is not reproduced.
   The sign (enhanced, opposite to f(R)) stands.
4. **§5.1 steepest |dµ/dz| at z ≈ 0.05**: |dµ/dz| is largest at z = 0 (monotonic).
5. **Table 2 timeline**: σ(µ0) for DESI Y5 (0.207) is larger than for DESI DR2 (0.100); the DR1/DR2/Y5/Euclid DR2 entries and the 5.4σ and 7.5σ
   combinations need a sourced Fisher forecast. The only sourced number is Euclid's full-survey σ(µ0) ≈ 0.04 (A&A 642, A191). "Euclid DR1 2027"
   → complete DR1 (lensing, clustering) mid-2027.
6. **Table 1 current constraints** (Planck +0.000 ± 0.200; DES Y3 −0.40 ± 0.40; DESI DR1 +0.11 ± 0.50; ACT+WMAP+SDSS+SN +0.02 ± 0.19) carry no
   citations; trace each to its paper before printing. Our chains: µ0 = +0.015 ± 0.156 (Planck), +0.039 ± 0.125 (Planck + RSD); paper 0.006.
7. **§4.2 tomographic mock (βm 0.1617 ± 0.0867; 1.2–1.9σ)** and Fig. 3(d) (βm 0.0588): no code in the repository; not reproduced.
8. **§6.2 neutrino bound Σmν < 0.07–0.08 eV**: no calculation given; not reproduced.
9. **§6.3 "CMB lensing identical to ΛCDM"**: Σ = 1 fixes the potential–density relation; C_φφ is lower by 0.08 % (SECTOR_TENSION_CHECK #5).
10. **§6.4 w_eff consistent with DESI hints**: DESI's w0 > −1; w_info = −4/3 today is the matter-ruler equation of state (WZ_FAR_FUTURE_CHECK #3).
11. **Scorecard (Table 6)**: S8 ~0.79 → 0.822 (Level 2 chains); KiDS σ8 0.76 ± 0.02 is 2σ from 0.800, not "consistent" without qualification;
    ISW "+15 ± 30 %" has no source; "β_m from the virial theorem" → fixed at Ω_m/2 in every chain (prediction, not a fit); 15 chains → 18.
