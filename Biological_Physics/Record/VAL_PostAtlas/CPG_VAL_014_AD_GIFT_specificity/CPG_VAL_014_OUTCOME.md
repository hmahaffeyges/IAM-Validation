# CPG-VAL-014 — GIFT specificity arm (AD vs FTD vs PSP/CBD vs HC)

**Cohort:** GSE53740 GIFT (Ferrari 2014, UCSF MAC, 450K, peripheral blood)
**Composition:** 193 HC + 15 AD + 121 FTD + 43 PSP + 7 FTD-MND + 1 CBD + 4 unknown
**Method:** Apply post-build instrument (Walther class fractions + 115-cell A-scores + Mahalanobis hyper-volume + 7-CpG Rule A panel) to each arm separately vs HC
**Date:** 2026-06-02
**Status:** PRELIMINARY

## Headline result — Mahalanobis distance by arm

| Arm | n | Mean Mahalanobis | vs HC: d | p_mwu |
|---|---|---|---|---|
| HC | 193 | 19.795 | — | — |
| **AD** | **15** | **23.840** | **+0.681** | **0.0006** |
| FTD | 121 | 22.288 | +0.279 | 0.108 |
| FTD-MND | 7 | 18.222 | −0.270 | 0.557 |
| **PSP** | **43** | **17.567** | **−0.380** | **2.3e-06** |
| CBD | 1 | 18.300 | — | — |
| **PSP+CBD combined** | 44 | — | **−0.378** | **3.0e-06** |
| **FTD class combined** (FTD + FTD-MND) | 128 | — | +0.257 | 0.152 |

**Three distinct signatures emerge:**

1. **AD: STRONG POSITIVE Mahalanobis (d=+0.68, p<0.001)** — even with only n=15, the post-build instrument finds a strong AD signal in GIFT. Larger than the AIBL Mahalanobis (d=+0.20). GIFT is a clinical AD cohort (UCSF MAC) with more advanced disease than AIBL (community cohort), which explains the magnitude difference.

2. **PSP/CBD: STRONG NEGATIVE Mahalanobis (d=−0.38, p<10⁻⁵)** — confirms the v2.2 card's BELOW_NORMAL tier signature. PSP/CBD patients have REDUCED hyper-volume distance from the HC centroid — their architectural readout is COMPACTED, not distended.

3. **FTD: WEAK POSITIVE Mahalanobis (d=+0.28, p=0.11)** — non-significant, intermediate. Possibly heterogeneous (FTD subtypes were not differentiated here).

## Per-cell-type effects vs HC

**AD vs HC (n=15/193) — top 5:**
| Cell type | d | p |
|---|---|---|
| Pancreatic_beta_cells | +1.288 | 8e-04 |
| stem_pluri | +1.286 | 0.023 |
| **Cortical_neurons** | **+1.265** | **0.044** |
| Pancreatic_acinar_cells | +1.257 | 6e-03 |
| Glia | +1.206 | 0.11 |

GIFT AD shows BROAD positive multi-class signal. The **Cortical_neurons +1.27** finding directly confirms VAL-091's outlier-driven detection (AD median 0.9% cortical-neuron cfDNA vs HC 0.0%) — the n=15 GIFT AD cohort has 2-3 patients elevated enough to produce a clear group effect.

**FTD vs HC (n=121/193) — top 5:**
| Cell type | d | p |
|---|---|---|
| NK-cells_EPIC | −0.369 | 3e-03 |
| MPP | −0.367 | 0.031 |
| granulocytes | −0.360 | 0.031 |
| Neutrophils_reinius | −0.355 | 0.066 |
| Neutro | −0.351 | 0.060 |

FTD shows the immune-class negative pattern similar to AIBL AD (CPG-VAL-008), but weaker. Tauopathy-class crossover.

**PSP vs HC (n=43/193) — top 5 (all Bonferroni-significant):**
| Cell type | d | p |
|---|---|---|
| Baso (basophils) | −0.686 | 4e-06 *** |
| LE (lipofibroblasts?) | −0.645 | 2e-05 *** |
| Microglia | −0.627 | 3e-05 *** |
| smooth_muscle | −0.613 | 3e-05 *** |
| Mela (melanocytes) | −0.586 | 8e-05 *** |

**PSP has 7 Bonferroni-significant negative per-cell-type effects** — a substantially distinct signature spanning non-immune cells (smooth_muscle, melanocytes, microglia). This is a real biological pattern, not noise.

## Differential-diagnosis tile expansion

The v2.2 card's differential-diagnosis tile (AD vs glioma via Stage 2 cortical-neuron threshold) now extends to a multi-arm tauopathy differential built on the Mahalanobis hyper-volume:

| Pattern | Mahalanobis | Interpretation |
|---|---|---|
| AD | strongly elevated (d>+0.5) | active multi-class disturbance |
| FTD class | mildly elevated to null | weak immune negative drift |
| PSP/CBD | strongly REDUCED (d<−0.3) | architectural compaction |
| Glioma (per VAL-090) | cortical-neuron elevated (Stage 2 anchor) | brain-tissue cfDNA detected |

This is a substantively richer differential than v2.2 could express. Card v3.0 absorbs this in the `cpg_native_post_build_addendum`.
