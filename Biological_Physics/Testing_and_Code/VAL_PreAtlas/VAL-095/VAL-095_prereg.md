# VAL-095 — UniLIFE Stage 3 Head-to-Head vs Salas Blood.EPIC IDOL at >10yr Breast Pre-Diagnostic

**Status:** PRE-REG (sealed before β-access)
**Card:** breast-epic v0.2 → v0.3 candidate; immune-class atlas integration test
**RNG seed:** 20260426
**Date sealed:** 2026-04-26

---

## Background and motivation

VAL-093 ran the production Stage 3 panel (Salas Blood.EPIC IDOL, 6 cell types: B, CD4T, CD8T, Mono, NK, Neu) and the Loyfer/Moss EPIC immune tiles. UniLIFE Guo 2025 provides 19-cell immune resolution (7 pan-lifespan + 12 adult-specific subtypes). The hypothesis: **does 19-cell immune resolution surface a >10yr breast pre-dx pattern that 6-cell resolution missed?**

This is a **resolution test**, not an outcome test. Production Stage 3 (Salas) remains baseline; UniLIFE is Queue-1 #1 for v0.3 integration.

## Cohorts and specimen pathway

- **GSE51057** (n=329, 450K)
- **GSE51032** (n=845, 450K)

Specimen: buffy-coat whole blood. **CHK-0.5 transferability:** UniLIFE was trained on adult whole blood spanning birth → old age (Guo 2025 Genome Med 17:63), with separate pan-lifespan and adult-specific markers. Buffy-coat ↔ adult whole blood is direct match. On-pathway.

**Coverage caveat (CHK-1.2):** The pre-extracted Loyfer-subset cohort data has only 9% UniLIFE CpG overlap (172/1906). To run UniLIFE properly, this VAL **requires re-extraction of UniLIFE CpGs from the original GSE51057 + GSE51032 series matrices**. The streaming-extractor is part of this VAL's deliverables.

If the re-extraction proves intractable (network/disk), the VAL falls to O5_TECHNICAL_DEFERRED rather than running on degraded coverage.

## Atlas and method

- **Production baseline (Stage 3):** Salas Blood.EPIC IDOL (450 EPIC CpGs × 6 cell types) — re-run on the original cohorts, output kept as VAL-093 baseline.
- **Test arm:** UniLIFE (1,906 CpGs × 19 immune cell types) — RPC deconvolution via the EpiDISH method (per Guo 2025 protocol).
- **Comparison metric:** per-cell-type case-vs-control Cohen's d at >10yr breast pre-dx for each panel; head-to-head report compares which cell types each panel calls.

UniLIFE 19 cell types:
- **Pan-lifespan (7):** B, CD4T, CD8T, Mono, nRBC, Gran, NK
- **Adult-specific (12):** aCD4Tnv, aCD4Tmem, aTreg, aCD8Tnv, aCD8Tmem, aBnv, aBmem, aBaso, aEos, aNeu, aMono, aNK

## Pre-locked decision criteria

**Primary outcome — does UniLIFE 19-cell resolution surface signal that Salas 6-cell missed?**

| Outcome | Condition |
|---|---|
| **O1_RESOLUTION_GAIN** | At least one UniLIFE cell type (one of the 12 adult-specific or one of the 7 pan-lifespan) shows |d| ≥ 0.5 at >10yr in GSE51057 AND replicates direction with |d| ≥ 0.3 in GSE51032, where the corresponding Salas 6-cell aggregate (e.g., Salas CD4T vs UniLIFE aCD4Tnv + aCD4Tmem + aTreg) shows |d| < 0.3 |
| **O2_RESOLUTION_NEUTRAL** | UniLIFE and Salas agree directionally on all overlapping cell types; UniLIFE adult-specific subtypes do not produce stronger signals than Salas aggregates |
| **O3_RESOLUTION_LOSS** | UniLIFE per-subtype signal is weaker than Salas aggregate (deconvolution noise dominates at 19-cell resolution) |
| **O4_DISCORDANT** | UniLIFE and Salas disagree by sign on a major cell type (B, CD4T, CD8T, Mono, NK, Neu) |
| **O5_TECHNICAL_DEFERRED** | Re-extraction unable to achieve >70% UniLIFE CpG coverage in either cohort |

## CHK-3.2 cross-cohort baseline

Per-cell-type healthy fraction in GSE51057 vs GSE51032: if differ by more than 1 SD, flag baseline mismatch. This is critical for UniLIFE because adult-specific subtypes (aBaso, aTreg) have low fractions overall and small absolute differences can produce inflated d.

## CHK-3.5 saturation flag

Stage 3 cell-fraction estimates are bounded [0, 1] by RPC constraint. Saturation = a fraction estimate at 0 or 1 in >5% of samples. Reported in results JSON.

## Test 2 (CCL-030)

UniLIFE adult-specific subtypes provide direct lymphoid-vs-myeloid discrimination via aCD4T/aCD8T/aB (lymphoid) vs aMono/aNeu/aEos (myeloid). **However, OQ-2026-01 immune-atlas staging gates Test 2 framework-level claims; this VAL surfaces lymphoid-vs-myeloid patterns as observation only, not as Test 2 evaluation.** Bidirectional cancellation cannot be claimed from this VAL.

## Card-specific routing (heme-LL-001)

This is a SOLID-ORGAN card (breast-epic) Stage 3 immune analysis. Per breast-epic README §B (case 1: pre-dx low-grade), expected pattern at >10yr is mild Stage 1 immune elevation; UniLIFE resolution test asks whether that mild Stage 1 elevation is uniformly distributed across immune cell types or concentrated in a specific subtype.

## Substrate scope (heme-LL-009)

Single-substrate methyl-only, 450K platform. UniLIFE was published with EPIC compatibility but is 450K-compatible per Guo 2025 supplementary. v1 single-substrate reading.

## Deliverables

1. `val_095.py` — VAL execution script with streaming UniLIFE re-extraction + RPC deconvolution
2. `VAL-095_results.json` — per-cell-type per-cohort Cohen's d, p-values, healthy baseline, head-to-head Salas vs UniLIFE table
3. `VAL-095_per_sample.csv` — per-sample 19-cell UniLIFE fractions + 6-cell Salas fractions, side by side
4. `VAL-095_outcome.md` — outcome label + head-to-head interpretation
5. **GSE51057_betas_unilife.csv + GSE51032_betas_unilife.csv** — the UniLIFE-CpG-subset cohort data (deliverable of the streaming extractor; reusable across future VALs)

## Rules followed

- CHK-2.1 (decision criteria pre-locked) ✓
- CHK-2.2 (cross-cohort baseline check declared) ✓
- CHK-2.3 (saturation flag declared) ✓
- CHK-2.4 (specimen on-pathway: buffy-coat ↔ adult whole blood UniLIFE) ✓
- CHK-2.5 (Test 2 placeholder declared, observation-only flagging allowed) ✓
- CHK-2.6 (atlas declared: UniLIFE 19-cell vs Salas Blood.EPIC IDOL 6-cell, head-to-head) ✓
