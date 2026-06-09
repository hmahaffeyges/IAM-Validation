# Session Summary — 2026-04-28 PM (Cardio + CHK-3.1 Split Decision)

## What was decided

**CHK-3.1 split convention adopted (locked 2026-04-28).** Single CHK-3.1 → CHK-3.1A (full-genome bimodality, substrate gating) + CHK-3.1B (card-specific marker subset bimodality, panel-coverage gating). Both must pass for sample to clear data-integrity gating. Decision rationale and Phase 1/2/3 rollout plan documented in `PHASE_2_PENDING/CHK_3_1_SPLIT_DECISION_2026_04_28.md`.

## What was completed (Phase 1)

**VAL-106 — CHK-3.1A calibration on TCGA HM450K sesame Level 3**
- Sealed prereg SHA: `0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a`
- 210 samples (160 KIRC + 50 PRAD adjacent-normal), all SHA-tracked, all NIH GDC public
- Full-genome bimodality: KIRC f_extreme 56.32% / f_middle 7.35%; PRAD 54.58% / 7.64%
- Outcome: O3_CALIBRATION_DEGENERATE under sealed prereg's conflated-convention bounds (which were derived from CpG-subset prior data points)
- Reclassified as the CHK-3.1A baseline anchor for TCGA HM450K sesame Level 3 substrate

**VAL-107 — CHK-3.1B calibration for cardio-epic on TCGA HM450K sesame Level 3**
- Sealed prereg SHA: `b58ce4dbd422198c7cbd6e7d1ee1cdbed86a758afc204189f8a9e070fd700d82`
- Subset SHA: `5a00e29ace75daae5a5bf7e3cfca26c16aa6dbd92750d16ebeaba4e874c48511`
- Cardio-epic CHK-3.1B subset: 8,100 unique CpGs (Loyfer 25-tile 6,105 + UniLIFE 1,906 + Salas 350)
- Same 210 samples (manifest reused from VAL-106)
- Subset bimodality: KIRC 63.62% extreme / 5.74% middle; PRAD 62.92% / 5.90%
- All 210 samples cleared coverage (n_subset_valid ≥ 7000 of 8100)
- Outcome: O2_PLATFORM_DIVERGENCE_DOCUMENTED (Mann-Whitney p=0.034 on f_extreme; practical difference 0.7 percentage points)
- **Established threshold for cardio-epic CHK-3.1B on TCGA HM450K sesame Level 3:** extreme≥55.0%, middle≤8.5%, coverage n≥7000

## Cardio-epic landscape survey (preserved)

Comprehensive GEO + ArrayExpress + EWAS literature survey:
- 51 candidate GEO Series across 8 cardiovascular subdomain queries
- 19 actually-cardiovascular cohorts after false-positive filtering
- 7-cohort curated ranked landscape with analyst notes per cohort

Top three cardio test cohorts (all HM450K with raw IDAT or AVG_Beta):
- **GSE69138** n=589 (404 + 185) ischemic stroke 3-subtype (large-artery vs small-artery vs cardio-embolic), whole blood
- **GSE84395** n=39 PAH on cultured pulmonary endothelial cells (Stage 2 cell-of-origin direct test)
- **GSE84274** n=24 ascending aorta (normal / dissection / BAV) Stage 2 cell-of-origin direct test

## What is BLOCKED pending Heath review

**Substrate-equivalence question for cardio-epic disease VALs.** GSE69138 ave_beta is GenomeStudio AVG_Beta (Illumina raw pipeline) — not TCGA sesame Level 3. These are different substrates per the split convention. The VAL-106/107 thresholds apply to TCGA sesame Level 3 specifically. Three options for moving forward:

(a) **Substrate-equivalence demonstration.** Run a quick check: does GSE69138 ave_beta full-genome f_extreme match TCGA sesame Level 3 (~55.87%) within tolerance? If yes, treat as substrate-equivalent and apply VAL-106/107 thresholds. If no, we need option (b) or (c).

(b) **Fresh CHK-3.1A calibration for GenomeStudio AVG_Beta substrate.** Run a calibration VAL on a structurally-separated cohort that is also GenomeStudio AVG_Beta format (e.g., GSE128235 healthy controls n=209, GSE40360 controls). Establish a separate CHK-3.1A threshold for this substrate.

(c) **Treat GSE69138 ave_beta as a new substrate.** Run fresh CHK-3.1A and CHK-3.1B calibrations specifically for GenomeStudio AVG_Beta on cohorts structurally separated from cardio test cohorts. Most rigorous, most expensive.

My recommendation: option (a) first. If GSE69138 ave_beta full-genome f_extreme is within ±5 percentage points of TCGA sesame Level 3 (55.87%), call it substrate-equivalent and proceed. If it's significantly different, fall back to option (b).

## What is on the horizon (Phase 2 + Phase 3)

**Phase 2 — Cookbook-wide convention update.** PENDING completion of cardio testing.
- TESTING_CHECKLIST.md: split CHK-3.1 into 3.1A and 3.1B
- EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md: new Part 17 documenting split rollout
- LESSONS_LEARNED.md: CCL-040/041 reclassified, new CCL-042 for the split decision
- README_MASTER.md: v2.4 amended line
- GAPE_Evidence_Report_UPDATED.html: VAL-100/101 retroactive split-classification footnotes
- GAPE_Reproduction_Paper_v1.md: section on CHK-3.1 updated

**Phase 3 — Per-card retroactive review.** PENDING completion of Phase 2.
- breast-epic v2.3 → v2.4
- lung-epic v0.2 → v0.3
- ad-immune
- hcc-epic v0.3 → v0.4
- crc-epic v2.4 → v2.5
- kidney-epic, cervical-epic
- cardio-epic v0.2 (already built under split convention; no retroactive update needed)

## Session metrics

- 210 TCGA adjacent-normal HM450K methylation files downloaded and SHA-tracked (~2.7 GB on disk, NIH GDC public)
- 2 calibration VALs sealed and executed
- 1 cookbook-wide policy decision recorded
- 51 GEO Series + 19 cardiovascular cohorts surveyed
- 0 disease findings (calibration session, not biology session)
- 0 sealed VALs unsealed; 0 retroactive changes to existing outcomes

## EDEAR commercial deployment

Per CCL-037 — unaffected. All Phase 1 activity is retrospective cookbook calibration architecture work.
