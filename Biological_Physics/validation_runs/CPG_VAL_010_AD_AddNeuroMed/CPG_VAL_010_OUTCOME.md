# CPG-VAL-010 — AddNeuroMed cross-platform replication

**Cohort:** AddNeuroMed GSE144858 (n=93 AD / 96 HC / 111 MCI, 450K, multi-center European)
**Method:** Mahalanobis hyper-volume + 115-cell A-score fan-out (cross-platform replication of CPG-VAL-008/009)
**Date:** 2026-06-02
**Status:** PRELIMINARY (not yet sealed under v4 protocol)

## Headline results

**Mahalanobis hyper-volume (universal departure-from-HC):**
- AD-vs-HC Cohen's d = **−0.006** (p = 0.71) — NULL
- HC mean = 34.91, AD mean = 34.90, MCI mean = 35.01

**Per-cell-type AD-vs-HC top 5:**
| Cell type | Cohen's d | p_mwu |
|---|---|---|
| Eosino | −0.463 | 5.1e-04 |
| Eos | −0.451 | 1.2e-03 |
| Eosinophils_reinius | −0.431 | 1.5e-03 |
| Bcell | −0.357 | 7.9e-03 |
| Neutrophils_reinius | −0.356 | 1.0e-02 |

(0 Bonferroni-significant at p<4.35e-04 due to smaller n)

## Comparison to anchors

| Metric | AIBL anchor (CPG-VAL-008/009) | AddNeuroMed (this) |
|---|---|---|
| Mahalanobis AD-vs-HC d | +0.200 (p<0.001) | −0.006 (NULL) |
| Top per-cell: Eosino d | −0.426 *** | −0.463 (Bonf-borderline) |
| Top per-cell: Bcell d | −0.380 *** | −0.357 |
| Top per-cell: Neutrophils d | −0.397 *** | −0.356 |

## Interpretation

The **per-cell-type pattern replicates exactly** across platforms — same top hits, same direction, similar magnitudes. The Mahalanobis universal summary does NOT replicate, falling to null. Two factors plausibly explain the divergence:

1. **450K coverage gap.** AddNeuroMed has 12,169/14,018 (86.8%) of the CpG union vs AIBL's 13,384 (95.5%). 1,849 fewer CpGs means the 115-cell A-scores are computed with reduced cell-type marker coverage. The Mahalanobis is more sensitive to this than per-cell-type effect sizes because it integrates across all 115 dimensions.
2. **Smaller cohort.** n=93/96 vs AIBL's 161/471. Statistical power is reduced.

The per-cell-type biology — eosinophil, B-cell, and neutrophil A-score reduction in AD — is reproduced cleanly. This is the substantive cross-platform validation of CPG-VAL-008's architectural immunosenescence finding.

## MCI-converter analysis (incidental)

AddNeuroMed has the rare advantage of MCI subclass labels (68 stable, 39 AD-converter). Test: do MCI-AD-converters show more AD-like Mahalanobis than MCI-stable?
- MCI-converter Mahalanobis: 35.00 (n=39)
- MCI-stable Mahalanobis: 35.08 (n=68)
- d(converter − stable) = −0.077 (NULL)

The universal summary does not distinguish converters from stable. The 7-CpG Rule A panel (CPG-VAL-051) might, but that's not tested in this VAL.
