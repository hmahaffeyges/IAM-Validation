# VAL-068 Outcome — pancreatic-epic Multi-Substrate on GSE74071

**Date:** 2026-04-25
**Cohort:** GSE74071 (Tjensvoll lab) — 14 PDAC tumor + 7 adjacent normal + 4 pancreatic juice circulating cancer cells + 3 cancer-associated fibroblasts + 1 primary culture = 28 samples on Illumina HM450
**Card:** pancreatic-epic v0.1
**Pre-registration SHA:** `50c0c7e8afccc2a5dfc407bf95e29b846cb1f3effc1458484e28e88f3cbaedfc`
**Status:** Multi-substrate exploratory — null at pooled-entropy on tumor vs normal arm; supplementary substrate observations

## Primary result (paired tumor vs normal, n=7 pairs)

| Metric | Value |
|---|---|
| Paired Cohen's d | **−0.31** (Hedges −0.27) |
| 95% CI | [−1.07, +0.45] (straddles zero) |
| Mean ΔA | −0.0071 |
| Welch t, p | t = −0.82, p = 0.41 |
| Per-CpG positive-direction split (mean Δβ) | 52.5% (271 / 516 CpGs) |
| Per-CpG negative-direction split | 47.5% (245 / 516 CpGs) |

**Pre-registered outcome: O3_TUMOR_NULL.** Direction goes opposite the VAL-066 result, with CI straddling zero. Consistent with the broader finding that PDAC pooled-entropy on Xu-538 is genuinely heterogeneous across cohorts.

## Per-pair detail (interesting heterogeneity)

| Pair | A_t | A_n | ΔA |
|---|---|---|---|
| PH64 | 0.761 | 0.807 | **−0.0456** |
| PH67 | 0.793 | 0.806 | −0.0130 |
| 314_09/10 | 0.788 | 0.759 | **+0.0282** |
| 314_11/12 | 0.797 | 0.802 | −0.0046 |
| GEMM_15/16 | 0.740 | 0.730 | +0.0104 |
| GEMM_17/18 | 0.751 | 0.768 | −0.0167 |
| GEMM_21/22 | 0.800 | 0.808 | −0.0084 |

PH64 is a strong negative outlier dragging the mean. The 314 series and GEMM 15/16 go positive; PH64, PH67, GEMM 17/18, GEMM 21/22 go negative or flat. **This per-pair heterogeneity is itself the key finding** — suggests genuine sub-type variation in PDAC that the Xu-538 immune panel pools over.

## Supplementary substrate findings

**Pancreatic juice circulating cancer cells (n=4) vs adjacent normal (n=8):** unpaired d = −0.72 [−1.95, +0.51]. Same direction as paired result, larger magnitude, CI still straddles zero. Pancreatic juice does not show the urine-sediment-style dramatic A-score collapse seen in VAL-065 prostate (d = −2.39). At n=4 this is exploratory only.

**Cancer-associated fibroblasts (n=3) at H_min(stromal) = 0.862950:** A_CAF values 0.816, 0.779, 0.809. Cross-class scoring is descriptive only; cannot be directly compared against secretory-class adjacent-normal arm. Demonstrates the framework correctly handles per-substrate H_min selection at the script level.

## What this triggers

GSE74071 is the second cohort in the pancreatic-epic synthesis to show the Xu-538 pooled-entropy metric does not produce a uniform-direction PDAC signal. Combined with VAL-067 (n=196 null) and VAL-066 (n=5 positive but mixed-direction), this strongly motivates VAL-069 directional fallback as the recommended Stage 1 metric for pancreatic-epic v0.1.

## Reproduction

Script: `val068_pancreatic_epic_gse74071.py`. Input: GSE74071_series_matrix.txt from NCBI GEO public access. Xu-538 panel JSON. RNG seed 20260425.
