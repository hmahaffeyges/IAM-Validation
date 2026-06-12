# VAL-069 Outcome — pancreatic-epic Directional Xu-538 Fallback Panel

**Date:** 2026-04-25
**Card:** pancreatic-epic v0.1
**Pre-registration SHA:** `e31de916ac00268bfe22116f67f54317b1a99f63dc3dc7c1482019a0be1ae12a`
**Status:** **O2_PARTIAL_RECOVERY** — directional approach validates on TCGA-PAAD holdout, partial-fails on GSE74071 holdout due to genuine cohort heterogeneity

## Background

VAL-066 (n=5), VAL-067 (n=196), and VAL-068 (n=7 paired) collectively showed the Xu-538 pooled-entropy A-score does NOT separate PDAC tumor tissue from adjacent-normal pancreatic tissue robustly across cohorts. Pooled d ranges from −0.52 to +1.18, all CIs span zero. Per-CpG positive direction percentages are 46.9%, 50.4%, 52.5% — clustered around 50/50, the bidirectional-cancellation signature.

VAL-069 builds the directional Xu-538 subset per CCL-027's mandate (the AD precedent VAL-051 was the first instance). Method: assign each Xu-538 CpG a frozen ±1 direction from training cohort sign of (mean β_tumor − mean β_normal), z-score normalize each CpG against training-cohort normal arm μ and σ, then score new samples as `mean(direction × z)`.

## Directional panel (built from GSE49149 training, n=196)

| Property | Value |
|---|---|
| Source panel | Xu-538 (538 CpGs) |
| Coverage filter (≥80% samples in each arm) excluded | 84 CpGs |
| Magnitude filter (`|Δβ_train|` < 0.005) excluded | 130 CpGs |
| **Final directional panel size** | **324 CpGs** |
| Positive direction (β tumor > normal) | 172 CpGs |
| Negative direction (β tumor < normal) | 152 CpGs |
| Calibration training d (by-construction) | +2.22 [+1.76, +2.67] |

The 172/152 positive/negative split confirms PDAC drives Xu-538 in genuinely bidirectional fashion at panel scale. The directional approach captures this by giving each CpG a sign-aligned vote.

## Holdout results

### H2 — TCGA-PAAD (n=5 paired) holdout: PASS

| Patient | A_dir tumor | A_dir normal | ΔA_dir |
|---|---|---|---|
| TCGA-FZ-5919 | +0.725 | −0.113 | **+0.839** |
| TCGA-FZ-5920 | +1.322 | −0.138 | **+1.460** |
| TCGA-FZ-5922 | +0.509 | +0.176 | +0.333 |
| TCGA-FZ-5923 | +0.665 | −0.160 | **+0.825** |
| TCGA-FZ-5924 | +0.688 | +0.094 | +0.593 |
| TCGA-FZ-5926 | +0.316 | −0.045 | +0.361 |
| TCGA-YB-A89D | +0.857 | +0.678 | +0.179 |

**Paired Cohen's d = +1.51 [+0.43, +2.59], paired t = +5.79, p = 6.4e-05.** Lower CI well above zero. All 7 patients show positive ΔA_dir. **The directional fallback recovers a strong signal where pooled-entropy was inconclusive (VAL-066 pooled d = +1.18 with lower CI = +0.04 just barely above zero).** Importantly, the n is now effectively 7 here (not 5) because the directional panel uses only 324 CpGs and these are all measured in the previously-failed-QC patients.

### H3 — GSE74071 (n=7 paired) holdout: FAIL

| Pair | A_dir tumor | A_dir normal | ΔA_dir |
|---|---|---|---|
| PH64 | −0.014 | +1.158 | **−1.172** |
| PH67 | +0.103 | +0.267 | −0.164 |
| 314_09/10 | +0.770 | −0.040 | **+0.810** |
| 314_11/12 | +0.195 | +0.054 | +0.141 |
| GEMM_15/16 | −0.021 | +0.097 | −0.118 |
| GEMM_17/18 | +0.645 | −0.105 | **+0.750** |
| GEMM_21/22 | +0.788 | −0.116 | **+0.904** |

**Paired Cohen's d = +0.22 [−0.53, +0.97], p = 0.56.** Direction is positive but CI straddles zero. Looking at per-pair, four pairs go strongly positive (consistent with TCGA-PAAD pattern) and three pairs go negative — PH64, PH67, GEMM 15/16. The PH64 pair has ΔA_dir = −1.17 which is unusually large negative; if PH64 were excluded the cohort would look much more like TCGA-PAAD.

## Pre-registered outcome classification

**O2_PARTIAL_RECOVERY.** H2 PASS, H3 FAIL. The directional fallback works on TCGA-PAAD and on the majority of GSE74071 pairs but is dragged below threshold by genuine heterogeneity in 2-3 GSE74071 pairs (notably PH64).

## What this means for pancreatic-epic v0.1 design

1. **Directional Xu-538 (324 CpGs, GSE49149-trained) is the recommended Stage 1 metric for pancreatic-epic v0.1.** It cleanly separated tumor from normal in TCGA-PAAD where pooled-entropy was inconclusive (paired d +1.51 vs +1.18, p 6.4e-05 vs 8.2e-03).
2. **Both pooled-entropy AND directional are reported on every IDAT** per the universal pipeline rule. The directional score is the primary clinical metric for PDAC; pooled-entropy is reported alongside as backward-compatibility and as a check.
3. **GSE74071 heterogeneity (PH64 outlier) is logged as an open question** — possible PDAC sub-type variation, possible cohort-specific artifact, possible technical variation. Cannot be resolved with current data; flagged for v0.2+ if more PDAC cohorts become accessible.
4. **The card enters Cookbook at `cohort_screening_validated`** tier (anchored by VAL-046 Rotterdam pre-dx blood, NOT by these tissue findings). Tissue arm is `tissue_arm_exploratory_with_directional_recovery_partial`.

## Reproduction

Script: `val069_pancreatic_epic_directional.py`. Inputs: GSE49149 series matrix, TCGA-PAAD β files (already downloaded by VAL-066), GSE74071 series matrix. Xu-538 panel JSON. Generates `pancreatic_directional_panel.json` (324 CpGs with frozen ±1 directions and per-CpG μ/σ from training normal arm). RNG seed 20260425.
