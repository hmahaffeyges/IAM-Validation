# VAL-067 Outcome — pancreatic-epic Large-Cohort Tissue Case-Control on GSE49149

**Date:** 2026-04-25
**Cohort:** GSE49149 (Mishra/Wood lab, PMID 24500968 / 26909576) — 167 PDAC tumors + 29 adjacent non-tumor on Illumina HM450 (GPL13534)
**Card:** pancreatic-epic v0.1
**Pre-registration SHA:** `f0de98bd22c98bf1a48100387e6a9acf79aa24c4591608552085d8c0c0ba2efb`
**Status:** Large-cohort tissue case-control — null at pooled-entropy

## Result

n=196 (167 tumor + 29 normal), 100% QC pass (≥400 valid Xu-538 CpGs per sample, all 196 pass).

| Metric | Value |
|---|---|
| Unpaired Cohen's d | **+0.2485** |
| 95% CI | [−0.1465, +0.6436] (CI straddles zero) |
| Mean ΔA | +0.00962 |
| Welch t, p | t = +1.24, p = 0.22 |
| Per-CpG positive-direction split | 50.4% (229 / 454 CpGs) |
| Per-CpG negative-direction split | 49.6% (225 / 454 CpGs) |

## Pre-registered outcome classification

**O3_TISSUE_NULL_LARGE_COHORT.** At n=196 (the largest tissue cohort tested for pancreatic-epic), the Xu-538 immune-class pooled-entropy A-score does NOT significantly separate PDAC tumor tissue from adjacent non-tumor pancreatic tissue. Direction is faintly positive (+0.25) but CI straddles zero. Per-CpG direction split is essentially 50/50.

## Why this matters more than VAL-066's large d

VAL-066 had n=5 and showed pooled d = +1.18. VAL-067 has n=196 and shows pooled d = +0.25 with CI straddling zero. The large-n cohort is the more reliable estimator. The VAL-066 d was driven by small variance and small effect rather than by a robust cross-cohort signal.

The 50.4%/49.6% per-CpG split is the key Stage 1 design finding: PDAC drives Xu-538 CpGs almost exactly evenly bidirectionally, which is the signature pattern that pooled-entropy A-score nulls out (CCL-027 / VAL-050 AD precedent).

## What this triggers

VAL-067 is the data evidence that motivates VAL-069 (directional fallback build). At a 50/50 per-CpG split, the pooled-entropy metric cannot recover signal even with the largest cohort. A directional approach — assign each CpG a frozen direction from training cohort, then score new samples with sign-aligned z-scores — is the recommended fix per CCL-027, demonstrated to work for AD (VAL-051).

## Reproduction

Script: `val067_pancreatic_epic_gse49149.py` (Python 3 stdlib). Input: GSE49149_series_matrix.txt downloaded from NCBI GEO public access (944 MB uncompressed). Xu-538 panel JSON. RNG seed 20260425.
