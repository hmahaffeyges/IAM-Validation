# PROC-SKY-01 — pre-registration: the sky's zero and spread with no population in them

**Written 2026-09-27, before any array was scored under these bars.** PLAN item 5. Fixed here; may be clarified before
data is read, never after.

## Why

The residual sky draws z per address. Today z = (β − Σ f_c μ_c − m_lab) / s_lab, where m_lab and s_lab are the mean and
spread of the residual across a 40-array healthy panel of the laboratory (`residual_scale_<lab>.npz`). That is a population
layer — the last one on the report after 2026-09-27. It goes regardless of how this procedure ends: the panel scales are
retired with this seal, and if the construction below fails its bars the sky is **withheld** ("sky not commissioned"), not
kept on the old scale.

## The construction (fixed)

1. **Zero.** m = 0. The composition expectation Σ f_c μ_{c,i} is the zero at every address. No laboratory constant.
2. **Spread.** σ_i² = σ_atlas,i² + σ_array²(β_i), with
   - σ_atlas,i² = Σ_c f_c² · sd_{c,i}² — the atlas's own per-address posterior SD, propagated through this specimen's
     composition (the `<cell>_sd` columns of IAMAtlasREBUILD; class-level `<class>_sd` where a class fraction is used);
   - σ_array²(β) = a + b·β(1 − β), fitted on **this array's own 65 SNP probes**: within-cluster SD of the β = 0 and β = 1
     clusters gives a (their β(1 − β) ≈ 0), the β = ½ cluster gives a + b/4. Two-iteration nearest-ideal clustering, the
     same as PROC-TARE-01. An array whose SNP probes are not present (a betas-only input) gets **no sky** — "sky needs the
     array's SNP probes", never a borrowed or pooled σ_array.
3. **Everything else unchanged**: pipeline map before the residual; presence-floor gating of the eight class panels;
   HEALPix mapping; the nine-panel plate; ALL LOCI + per-class summaries (median z, % |z| > 2).

Nothing in σ comes from any person or group. σ_atlas is the reference cell methylomes; σ_array is the array in hand.

## Bars

- **B1 — the zero is the atlas.** A constructed specimen made of exact atlas means at the "typical" blood composition
  (PROC-UNMIX-01 mixes), σ_array set to the value fitted on any one whole-blood array: median z within ±0.05 and
  |z| > 2 on ≤ 0.5 % of addresses, on ALL LOCI and on every assessable class panel.
- **B2 — σ explains the array's own scatter.** On the 48 commissioning arrays (12 per laboratory, the same arrays as
  the detection panel), the robust SD of ALL-LOCI z (1.4826 × MAD) lies in [0.7, 1.4] on ≥ 40 of 48. This is a check that
  the noise model is on the array's scale — not a definition of anything about people.
- **B3 — no laboratory file.** `patient_sky` takes no laboratory argument and reads no per-laboratory file; the four
  `residual_scale_*.npz` are not on the search path (mechanical check in the kit).
- **B4 — serial draws.** (a) The same array twice: z₂ − z₁ = 0 at every address. (b) Two constructed specimens with the
  same cell profiles and different fractions (typical vs neutrophil-heavy): median |z₂ − z₁| < 0.05 on ALL LOCI — the
  composition drops out of the difference, as it must for a patient-against-himself picture.
- **B5 — the plate is the same object.** Nine panels, presence gating unchanged; a class below its floor is black and
  NOT ASSESSABLE exactly as today (mechanical: same panel set, same masks on the same array).

## Decision rule

B1–B5 met → adopted into [`stage_4_6_patient_cmb.py`](../chain/stage_4_6_patient_cmb.py); the caption reads "z = (β − Σ f_c μ_c) / σ, σ from the atlas
posterior and this array's SNP probes"; the four panel scales move to RETIRED. Any bar failed → the panel scales still
move to RETIRED, the sky is withheld with one sentence, and the failed bar is the finding. B2 is the one that can fail:
if the SNP-probe noise does not carry to the cg probes, the sky needs an on-array noise term this procedure did not find.
