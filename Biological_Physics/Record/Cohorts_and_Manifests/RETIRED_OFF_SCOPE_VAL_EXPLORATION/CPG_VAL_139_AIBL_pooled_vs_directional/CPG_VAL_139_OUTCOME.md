# CPG-VAL-139 — Directional outperforms pooled by 8.0× on AIBL AD discrimination

**Cohort:** AIBL GSE153712, n_AD=161 vs n_HC=471
**Date sealed:** 2026-06-06
**Outcome code:** O1_PRIMARY_VALIDATED

## Headline result

| Signal form | Cohen's d (AD vs HC) | N1 p-value | Interpretation |
|---|---|---|---|
| **a_dir_immune** (Stage 4.5 directional, sign-multiplied z) | **+0.616** | 0.0 | LOUD |
| a_pool_aibl_18cpg (pooled entropy H(β_mean)/H_min) | +0.077 | 0.388 | MUTED |
| **Directional / Pooled ratio** | **8.01×** | — | FLAG_BIDIRECTIONAL trigger pattern |

## Interpretation

PASS — The 18-CpG immune panel shows a 8.0× larger directional Cohen's d than pooled-entropy on the same data. This is the architectural signature the framework was designed to detect: a class where sub-panels of CpGs drift in opposite directions (some up in AD, others down) creates a pattern that cancels under pooled-mean β computation but survives under sign-multiplied z-scoring. This pattern triggers FLAG_BIDIRECTIONAL in Stage 4.5.

The framework prediction: "When pooled mute + directional loud, FLAG_BIDIRECTIONAL fires." Observed here on AIBL — pooled gives +0.077, directional gives +0.616.

## Cohort linkage

- Per-sample data: `CPG_VAL_139_per_sample.csv` (n=726 × 4 columns)
- VAL-051 panel anchor: d = +0.624 (sealed); this VAL produces directional d = +0.616 (reproduces VAL-135 result with consistent processing)
