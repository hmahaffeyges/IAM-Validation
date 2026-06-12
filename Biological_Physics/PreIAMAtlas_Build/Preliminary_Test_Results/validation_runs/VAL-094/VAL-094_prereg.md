# VAL-094 — EpiSCORE Breast Stage 2 at >10yr Breast Pre-Diagnostic Window

**Status:** PRE-REG (sealed before β-access)
**Card:** breast-epic v0.2 → v0.3 candidate
**Card class:** secretory
**RNG seed:** 20260426
**Date sealed:** 2026-04-26

---

## Background and motivation

VAL-093 established that at the >10yr breast pre-dx window, the breast tile (Loyfer/Moss array atlas) reads NULL (d=+0.198 GSE51057, d=+0.100 GSE51032) while several non-breast tiles (pancreatic beta/acinar/duct, cycling-class kidney/colon/HN-larynx) show concordant elevation. This is the `O2_SECRETORY_DISTRIBUTED` outcome.

The Loyfer/Moss breast tile uses one bulk-breast cell type. EpiSCORE provides finer-grained breast resolution: **BreastRef** mref matrix (DNAm-derived) covers 8 cell types — Adipocytes, Endothelial, Epithelial luminal, Epithelial basal, Fibroblasts, Immune lymphocyte, Immune myeloid, Immune plasma. (Source: vault MANIFEST.json, BreastRef__mrefBreast_m.csv, 468 markers × 8 cell types.)

**Prediction within the framework:** if VAL-093's breast-tile null at >10yr is real (not a coverage artifact), then EpiSCORE's 8-cell-type breast resolution will also read null at >10yr.

**Counter-prediction:** if VAL-093's breast-tile null was a Loyfer-bulk-tissue resolution artifact, EpiSCORE could surface a sub-tissue arm (e.g., epithelial luminal vs basal) that is elevated where bulk-breast is not.

This is a **resolution test**, not an outcome test. The breast card's at-diagnosis arm (VAL-060 paired d=+0.676) is unchanged by this VAL.

## Cohorts and specimen pathway

- **GSE51057** (EPIC-Italy DCSE breast pre-dx, n=329, 450K) — full TTD distribution preserved in `/home/claude/run_everything/VAL-093_per_sample.csv`. >10yr breast pre-dx subset: n=11.
- **GSE51032** (Rotterdam EPIC-Italy n=845, 450K) — TTD distribution preserved. >10yr breast pre-dx subset: ~85 (per VAL-093).

Specimen: buffy-coat whole blood, both cohorts.

**CHK-0.5 caveat:** EpiSCORE BreastRef was trained on tissue-of-origin breast samples, not on buffy-coat. Reading buffy-coat against a tissue reference is exactly what cell-of-origin deconvolution is designed for, so this is on-pathway. Null findings on this VAL are interpreted as breast tissue-of-origin signal absence in plasma, not as transferability failure.

## Atlas and method

- **Stage 2 atlas:** EpiSCORE BreastRef mref matrix (DNAm-derived, 468 markers × 8 cell types, Entrez-ID indexed)
- **Bridge:** Entrez-ID → CpG mapping via probeInfo450k.rda. The probeInfo bridge is the connector between gene-indexed reference and CpG-indexed customer β. **The bridge file is large (3.3 MB) and lives in vault scratch (`/home/claude/atlases/episcore/EpiSCORE-master/data/probeInfo450k.rda`); it must be added to the durable vault as part of this VAL's deliverables.**
- **Scoring method:** for each of the 8 cell types in BreastRef, compute the per-cell-type A-score using the entropy of the cohort's β values at that cell-type's discriminating CpGs against the secretory H_min anchor (0.8433). Identical method to VAL-093 per-tile scoring on Loyfer.
- **Window:** >10yr breast pre-dx vs cancer-free controls

## Pre-locked decision criteria

**Primary outcome — does any EpiSCORE breast sub-tile show elevation that bulk-breast missed?**

| Outcome | Condition |
|---|---|
| **O1_RESOLUTION_GAIN** | At least one of the 8 EpiSCORE breast sub-tiles shows |d| ≥ 0.5 in the case-vs-control comparison at >10yr in GSE51057, AND replicates direction with |d| ≥ 0.3 in GSE51032 |
| **O2_DISTRIBUTED_AS_LOYFER** | All 8 sub-tiles have |d| < 0.5 in both cohorts (consistent with VAL-093 breast null — finer resolution does not surface a hidden breast arm) |
| **O3_INVERTED** | At least one breast sub-tile shows d ≤ −0.5 (homogenization in case group) in both cohorts (parallel to VAL-047 Phase 6 secretory-class result) |
| **O4_NULL_REPLICATION_FAIL** | GSE51032 disagrees with GSE51057 by sign on all 8 sub-tiles |
| **O5_TECHNICAL_FAIL** | EpiSCORE coverage <50% of CpGs map through probeInfo450k bridge → mark VAL inconclusive |

## CHK-3.2 healthy-baseline check

Before scoring case-vs-control, compute per-sub-tile healthy A-score in both cohorts. If any sub-tile differs by more than 1 anchor-SD between GSE51057 healthy and GSE51032 healthy, flag as cross-cohort baseline mismatch.

## Saturation flag

For each per-sample A-score, flag if A ≥ 1.1809 (secretory ceiling 1.1859 − 0.005). Per-group saturation fraction reported in results JSON.

## Test 2 placeholder (CCL-030)

This VAL is Stage 2 cell-of-origin only. Stage 1 immune Test 2 (lymphoid vs myeloid) remains blocked on OQ-2026-01. No claim of bidirectional cancellation can be made from this VAL.

## Card-specific routing (heme-LL-001)

This is a SOLID-ORGAN card (breast-epic). Stage 2 elevation on the matching solid organ is the positive call. Stage 2 null is a negative finding for that card. NOT inverted heme-routing logic.

## Substrate scope (heme-LL-009)

Single-substrate methyl-only buffy-coat, 450K platform. Issue 002 framework-level secretory predictions refer to 5-substrate L2/L3 platform; this VAL is v1 single-substrate.

## Deliverables

1. `val_094.py` — VAL execution script
2. `VAL-094_results.json` — per-sub-tile per-window per-cohort d, p, n, healthy-baseline A
3. `VAL-094_per_sample.csv` — per-sample EpiSCORE 8-cell A-score
4. `VAL-094_outcome.md` — outcome label + rationale
5. **probeInfo450k.rda → CSV bridge** added to atlas vault as part of integration

## Rules followed

- CHK-2.1 (decision criteria pre-locked) ✓
- CHK-2.2 (cross-cohort baseline check declared) ✓
- CHK-2.3 (saturation flag declared) ✓
- CHK-2.4 (specimen pathway transferability — buffy-coat ↔ breast tissue reference is on-pathway) ✓
- CHK-2.5 (Test 2 placeholder declared) ✓
- CHK-2.6 (atlas declared explicitly: EpiSCORE BreastRef mref) ✓
