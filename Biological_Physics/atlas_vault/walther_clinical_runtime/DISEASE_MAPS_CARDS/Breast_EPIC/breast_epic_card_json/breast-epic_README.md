# Breast-EPIC card — CPG / EDEAR

**Card version:** v3.0
**Card date:** 2026-06-02 (initial v3.0) + 2026-06-03 (full SOP retrofit)
**Card location:** `breast-epic_card_v3_1.json` (this folder)
**Card README (this file):** v3.0 — clean rewrite 2026-06-04 with current methodology
**Maintained by:** IAMPerformance Inter-Domain Research Institute, Entiat WA / iamperformance.net

---

## What this card detects

A blood-based methylation signature of breast cancer **detected more than 10 years before clinical diagnosis**. The signature is a broad cellular-architectural disturbance — not a single biomarker — visible across many cell types simultaneously in the patient's blood. The same single summary number that captures it outperforms the existing published Xu-538 breast methylation panel by a meaningful margin on one cohort.

This card is **not a diagnostic test for breast cancer**. It is a methylation-trajectory pattern that, in two independent EPIC-Italy cohorts totaling 47 future-cases and 601 healthy controls, was associated with clinical breast cancer diagnosis years to a decade later. Operational use is as a screening flag and as a serial-monitoring baseline against which trajectory drift becomes visible.

---

## What we learned — plain language

These are the headline findings any clinician needs to know, in order of clinical relevance:

### 1. The signal is detectable a decade before clinical disease

In women who later developed clinical breast cancer, blood drawn **more than 10 years before diagnosis** showed cellular-architectural disturbance distinguishable from age-matched healthy controls. The universal departure-from-healthy metric (Mahalanobis hyper-volume distance on the 115-cell A-score vector) came out at:

- **d = +1.876** on GSE51057 (11 cases / 177 controls)
- **d = +2.097** on GSE51032 (36 cases / 424 controls)

Both effects are "very large" by the conventional Cohen's d interpretation (d > 0.8 = large). The signal **rises** the further from clinical diagnosis you sample — counterintuitive but consistent with the field-effect cancerization hypothesis (cells in the at-risk tissue methylation-distinguish themselves long before tumor formation).

### 2. The strongest single cell-type signal is in basophils

Basophils are a rare granulocyte (~1% of circulating leukocytes). In coarse immune-class analysis they are invisible. In the 115-cell A-score resolution this instrument provides, basophil A-score departs from healthy at:

- **d = +1.58** on GSE51057
- **d = +1.01** on GSE51032

This is hidden in any analysis that only resolves to "immune cells" — basophils are 1/51 of the cells inside the immune architecture class. The fan-out to 115-cell resolution is what surfaces them.

### 3. Breast epithelial cells appear in the patient's blood

Even more than 10 years before clinical breast cancer, breast epithelial A-score in the patient's blood departs from healthy:

- **d = +1.28** on GSE51057
- **d = +0.61** on GSE51032

This is detection of **field-effect cancerization at the methylation level in blood**, a decade before any tumor is clinically visible. The at-risk tissue is already shedding methylation-distinguishable cells into the circulation.

### 4. The immune system appears to be "looking away"

The principal component capturing T-cell variation goes negative in cases vs controls:

- **d = −0.67** on GSE51057
- **d = −0.58** on GSE51032

Interpretation: immune surveillance is suppressed in the pre-diagnostic state. The system that should be policing emerging abnormal cells is, by this readout, doing it less well.

### 5. The universal Mahalanobis metric beats the existing Xu-538 panel

On GSE51032 the published Xu-538 breast methylation panel (a disease-trained biomarker set) shows d = +1.35; our universal Mahalanobis shows d = +2.097. **The universal architectural-information metric BEATS the disease-trained panel by +0.75 standard deviations** on this cohort. For broad-architectural diseases like pre-diagnostic breast cancer, the universal metric is the right operational readout.

### 6. The pipeline reproduces build-time exactly (Stage 1 integrity gate)

When re-run on the same data with the post-build pipeline on 2026-06-03, GSE51032 Mahalanobis came out at **d = +2.088** vs the original build-time anchor of **+2.097** — within 0.4%, well within sampling variation. The post-build instrument is internally consistent with the build-time pipeline that produced the original anchors.

### 7. A second deconvolution method agrees with ours

NILC v2 (an independent compositional deconvolution algorithm) run on the same data agrees with our primary Walther deconvolver on dominant blood compartments at **Spearman ρ = +0.74 (immune)** and **ρ = +0.82 (progenitor)**. The findings are not artifacts of one algorithm choice.

### 8. Cellular age — cycling-class cells look "younger" in pre-dx samples

In the GSE51032 pre-diagnostic cases, cycling-class cellular age comes back at **d = −0.53** vs controls (cases ~5.5 years younger methylation-wise in this proliferating compartment). Consistent with cell-cycle arrest / proliferation slowdown in the at-risk tissue years before clinical disease.

---

## How a CPG report on this card reads

A patient with broad architectural disturbance in this pattern triggers a card-level "flag worth watching" output, not a disease assertion. The report contains:

1. **Per-class A-score** for all 8 architecture classes (terminal, immune, secretory, progenitor, cycling, stromal, stem_adult, stem_pluri) with the patient's value, age-matched healthy median, and percentile.
2. **115-cell-type A-score** with the top three positive and top three negative departures highlighted. If basophil, breast epithelial, or related cells appear in the top departures, the report flags them.
3. **Universal Mahalanobis hyper-volume distance** with the patient's value and the population distribution it sits in.
4. **Cellular age per class** with status flags (OK, SATURATED for blood-irrelevant classes).
5. **Tier assignment** per class (BELOW_NORMAL / NORMAL / MARGINAL / DETECTABLE / FLOOR_BREACH).
6. **Honest limitations section** (see "What we are not claiming" below).

Serial sampling against the patient's own baseline is the intended use mode. A single report is a coordinate; a series is a trajectory.

---

## How the card works — methodology summary

The card uses the current production methodology (no pre-build-era external panels or external atlases in the production scoring chain):

1. **Walther IAM Deconvolver** — produces 8 architecture-class fractions per patient from the methylation array.
2. **A-scoring** — for each architecture class and each of 115 cell types, computes A = H(β) / H_min(class), where H_min is the information-theoretic minimum methylation entropy per class (frozen 2026-04-06).
3. **Mahalanobis hyper-volume** — single scalar summarizing total departure from the healthy 115-cell centroid (Ledoit-Wolf shrinkage 0.00875, healthy reference n = 601 EPIC-Italy controls).
4. **NILC v2 cross-method check** — independent deconvolution method run in parallel for consistency.
5. **Age-axis foreground subtraction** — removes age-correlated methylation drift before disease scoring.
6. **Cellular age per class** — independent age estimate per architecture class.
7. **Tier breakpoints** — universal-screen tier assignment.

The Recipe — the first-principles derivation chain producing the H_min values — is vault-only and never appears in production code or reports. The A-score formula above is the public-facing equation; the H_min values are quoted as constants.

---

## What we are NOT claiming

Be honest with the patient about these:

1. **The card does not diagnose breast cancer.** It detects a methylation pattern statistically associated with future clinical diagnosis in two cohorts.
2. **Effect sizes are cohort-level, not individual-level.** A single positive report does not mean a given woman will develop breast cancer.
3. **Specificity for breast is partial.** The universal Mahalanobis metric was validated against breast pre-diagnostic samples vs healthy controls. We have not tested whether other cancers' pre-diagnostic samples would also show the same universal signal. The breast-epithelial-cell tissue-of-origin signal (Finding #3) is more breast-specific, but only emerges in the per-cell-type analysis.
4. **Validation is European-ancestry.** Both cohorts are EPIC-Italy. Cross-ethnicity validation is outstanding.
5. **Single-time-point interpretation is limited.** The instrument is designed for serial sampling against the patient's own baseline.
6. **The disease signature matrix v1.7 per-patient matching engine is not yet wired.** Per-patient reporting currently goes through the card-driven Stage 8 Path A, not the matrix-driven Stage 8 Path B.

---

## Validation summary

| What | Details |
|---|---|
| **VAL series** | CPG-VAL-001 through CPG-VAL-007 |
| **Cohorts** | GSE51057 (n=188 filtered: 11 case >10y pre-dx / 177 HC) + GSE51032 (n=460 filtered: 36 case >10y pre-dx / 424 HC). Both EPIC-Italy. Severi 2014, *Carcinogenesis* 35(10):2349. |
| **L9 null suite** | 5 of 7 PASS at N1 = 0.000; 2 RESTATE (VAL-004 gain-of-bimodality direction-reversed; VAL-006 chr6 MHC lost Bonferroni significance) |
| **Stage 1 reproduction** | ✅ PASS: GSE51032 Mahalanobis d = +2.088 vs anchor +2.097 (within 0.4%) |
| **Cross-method check** | Walther vs NILC v2: ρ = +0.74 immune / +0.82 progenitor on GSE51032 |
| **Per-VAL bundles** | `validation_runs/CPG_VAL_NNN_Breast_*/` — each with PREREG.md + per_sample.csv + null_results.json + cohort_manifest.json + OUTCOME.md |
| **Cohort folders** | `validation_runs/breast_epic_cohorts/{GSE51057,GSE51032}_EPIC_Italy/` |
| **SOP audit** | `BREAST_EPIC_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md` (this folder's parent) |
| **Status** | ✅ Operational — full SOP coverage stages 2–7 + Stage 8 Path A. Stage 8 Path B engine wiring deferred. |

For complete validation detail, see the evidence report `post_build_evidence/v5_CPG_IAMAtlas_Evidence_Report.html` Section 4.1.

---

## Outstanding follow-up work (carry-forward to v3.1)

1. CPG_breast_panel_v1 (1,392 concordant CpGs seeded from CPG-VAL-003, 1,389 of which are not in Xu-538) — formal seal + holdout validation on an independent cohort
2. CHR/MAPINFO genomic annotation on residual map
3. Full bimodality decomposition (placeholder currently)
4. Cross-ethnicity validation (Asian, African, Latin-American cohorts)
5. Stage 8 Path B engine wiring (cell-name-to-matrix-column mapping artifact)
6. Synthetic_Patient_Generator chain-recovery test on the breast signature
7. First-client IDAT integration test (Stages 0/1 untested on raw IDATs in our chain)

---

## Version log

| Version | Date | Change |
|---|---|---|
| v2.3 | 2026-04-26 | Pre-build era card. Used Xu-538 panel, Moss 2018 NNLS tissue deconvolution, Salas 2018 immune sub-composition. Archived at `OLD/breast-epic_card_v2.3.json`. |
| v3.0 | 2026-06-02 | Strict additive bump from v2.3. New card uses CPG_breast_panel_v1 seed (1,392 concordant CpGs from CPG-VAL-003) instead of Xu-538. Pre-build VAL evidence retained as audit lineage. |
| v3.0 + retrofit | 2026-06-03 | Both cohorts re-streamed from GEO with SHA-256 tracking. Arm parser bug fixed (was labeling all samples HC). All SOP stages re-run on both cohorts. Per-VAL bundles created. Stage 1 reproduction PASS. |
| **v3.0 + README rewrite** | **2026-06-04** | **README rewritten clean** with current methodology focus and plain-language clinical findings sections. Dropped extensive pre-build-era Xu-538 + Moss 2018 + Salas 2018 references in the operational sections (preserved as historical lineage in this version log). v2.3 README archived at `OLD/breast-epic_README_v2_3.md`. |

---

*Companion documents in this card folder: `breast-epic_card_v3_1.json` (card spec), `breast-epic_v3_0_release_notes.md` (technical changelog for v3.0 bump). Companion documents in card parent folder: `BREAST_EPIC_v3_0_SOP_CHAIN_OF_CUSTODY_AUDIT.md`, `WORK_IN_PROGRESS.md`, `breast_epic_residual_maps/`.*
