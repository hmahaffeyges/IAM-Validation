# CPG-VAL-139 — Pre-Registration

**VAL ID:** CPG-VAL-139
**Title:** Pooled-entropy vs directional bidirectional comparison on AIBL
**Date sealed:** 2026-06-06

## Cohort

- **Source:** AIBL GSE153712 (Nabais 2021) 18-CpG IMM panel (from sealed VAL-050)
- **Contrast:** Alzheimer's disease (n=161) vs healthy control (n=471)
- **MCI excluded:** n=94 mild cognitive impairment samples not in this contrast

## Signal

- **Signal A (directional):** a_dir_immune = mean of sign-multiplied z-scores against frozen AIBL HC training distribution (Stage 4.5 bidirectional)
- **Signal B (pooled-entropy):** a_pool_aibl_18cpg = H(β_mean over the 18-CpG panel) / H_min(immune)
- **Hypothesis:** The framework predicts FLAG_BIDIRECTIONAL when class has opposing-direction drift — pooled (which averages β) cancels while directional (which preserves sign) survives. AD-immune fits this pattern per VAL-051.

## Decision rule

- **Pass:** |d_directional| > 2 × |d_pooled| (directional substantially exceeds pooled)
- **Interpretation:** A ratio > 2 confirms FLAG_BIDIRECTIONAL is correctly triggered for the AD-immune class.

## Observed outcome

- **d(a_dir_immune):** 0.616
- **d(a_pool_aibl_18cpg):** 0.077
- **Ratio:** 8.01
- **N1 directional p-value:** 0.0
- **N1 pooled p-value:** 0.388
- **Outcome code:** O1_PRIMARY_VALIDATED
