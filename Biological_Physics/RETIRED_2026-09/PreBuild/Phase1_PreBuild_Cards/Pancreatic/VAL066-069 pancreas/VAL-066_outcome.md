# VAL-066 Outcome — pancreatic-epic Tissue Arm on TCGA-PAAD HM450

**Date:** 2026-04-25
**Cohort:** TCGA-PAAD HM450 matched tumor/normal — n=7 amended (per `VAL-066_PREREG_AMENDMENT.md`), n=5 effective after QC
**Card:** pancreatic-epic v0.1
**Pre-registration SHA:** `694206201d45c1e3cbced1ef17b565b99e5d7f86a96b29fd58f6ba6050ea887e`
**Amendment SHA:** `9533d64cc98d361a168ee941bcb737156b8410f655a15d2f878297734f5c344b`
**Status:** **EXPLORATORY** — small cohort, classified O5_UNEXPECTED per pre-reg; supplemented by VAL-067 (n=196 GSE49149) and VAL-068 (n=28 GSE74071) and VAL-069 (directional fallback)

## Result

n=5 effective matched pairs after QC (TCGA-FZ-5920, FZ-5922, FZ-5923, FZ-5924, YB-A89D). Two patients failed QC for insufficient Xu-538 panel coverage (TCGA-FZ-5919, FZ-5926).

| Metric | Value |
|---|---|
| Paired Cohen's d | **+1.18** (Hedges +0.95) |
| 95% CI | [+0.04, +2.32] (lower bound just barely above zero) |
| Mean ΔA | +0.0294 |
| Welch t, p | t = +2.65, p = 8.2e-03 |
| Per-CpG positive-direction split | 46.9% (160 / 341 CpGs) |
| Per-CpG negative-direction split | 53.1% (181 / 341 CpGs) |

## Pre-registered outcome classification

H1 (paired d > 0.3 AND lower CI > 0): PASS (technically — d = +1.18, lower CI = +0.04 just above zero). H2 (>50% positive direction): FAIL (46.9% positive, slight negative-majority). Both required for O1; only one passes; both required for O2; H2 fails. **Classified as O5_UNEXPECTED** per the prereg matrix.

## Why this is more informative than it looks

The pooled paired d of +1.18 looks strong, but the per-CpG split is 47%/53% — the pooled magnitude is being driven by larger-magnitude-on-positive CpGs while the *count* of negative-direction CpGs is slightly higher. This is exactly the bidirectional-cancellation pattern that CCL-027 (the new mandatory per-card check) flags as a Stage 1 design risk.

The n=5 caveat dominates everything else: at n=5, 4 patients showed ΔA in [+0.009, +0.049] (all positive but small) and 1 patient (TCGA-YB-A89D) was essentially flat at −0.004. The variance is small (sd = 0.025) which is why the d looks large despite small mean.

## Why the result feeds into a multi-cohort synthesis rather than a card direction

VAL-066 alone cannot tell us whether PDAC tumor architecture is genuinely positive-direction (n=5 too small to be sure) or whether the bidirectional pattern means something. VAL-067 (n=196 case-control on GSE49149) was run as a follow-up to address this with a much larger cohort. VAL-068 (n=28 multi-substrate on GSE74071) was run to add a third independent cohort and pancreatic-juice supplementary data. VAL-069 built a directional fallback panel from VAL-067 training data and tested it on VAL-066 and VAL-068 holdouts.

The full synthesis lives in the pancreatic-epic v0.1 card README. VAL-066 contributes one of four cohort lines to that synthesis.

## Reproduction

Script: `val066_pancreatic_epic_tcga_paad.py` (Python 3 stdlib only). Inputs: TCGA-PAAD HM450 β files for the 7 amended patient IDs (downloadable from NIH GDC public access, no dbGaP), Xu-538 panel JSON. RNG seed 20260425.
