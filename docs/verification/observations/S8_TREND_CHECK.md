# S8_TREND_CHECK — "The Redshift-Dependent S8 Trend in the Context of IAM" (Mar 2026, 8 pp)
Read in full 2026-10-02 (PDF text 323 lines, 50-line ledger, no gaps). Reproduction: `scripts/verify_s8_trend.py` (output beside it).

## Reproduced
- µ(z) values (0.864 today; 0.982 at z = 1; 0.998 at z = 2). µ0 = −0.1362 exact form (paper −0.1349 rounds the MGCAMB amplitude).
- Σ = 1 ⇒ BAO, CMB acoustic scale, lensing geometry = ΛCDM. Background modification excluded (H0 ≈ 61.5, Level 2b).
- Scale independence (µ depends on a only).

## Corrections
1. **Eq. 5 S8_inferred = S8_Planck × µ(z)** treats the instantaneous coupling as the amplitude. The amplitude is the integrated growth,
   S8 × D_IAM/D_ΛCDM from the linear growth equation: deficit 0.78 % today (0.8255), 0.22 % at z = 0.5, 0.06 % at z = 1. The paper's 0.719 at z = 0
   overstates the effect ~17×. The paper's text (0.719) and its Fig. 2 label (0.702) also disagree.
2. **Abstract, §3, §9 "predicts this redshift dependence"**: the shape matches; the amplitude is ~1/10 of the low-z deficit (6–9 %).
   fσ8 deficit today 4.25 %; growth index γ_eff = 0.585 (ΛCDM 0.554, measured 0.633): ~40 % of the measured departure.
3. **MGCAMB form vs exact µ**: the same integration with the Level 1 form gives an amplitude deficit of 1.68 % today, reproducing the chains'
   σ8 shift (1.54 %). The exact form gives half. Book chains' σ8 = 0.800 is the MGCAMB form.
4. **§4, §9 "17 chains independently return β_m within 0.2σ"**: β_m is fixed in every chain (errata, Theory and CC checks).
5. **§7 ISW "10–30 % enhancement"**: ISW source (1 − f)D/a is 3.4 % larger at z ≤ 0.5; cross-correlation ~3 % (uniform weight z 0.05–1.5).
6. **§5, §7 cluster M_lens/M_dyn = 1/µ**: 1.16 today, 1.06 at z = 0.5 in the linear coupling; applicability inside virialised clusters is open
   (Virial check: cluster three-way test held).
7. **Fig. 3 "E(a) recovered to ~1 % by the Sheth–Tormen mass function"**: not reproduced in this repository; not carried.
8. **Naming**: the paper is addressed to, and acknowledges, a named cosmologist; the book cites the compilation by journal and arXiv number only.
