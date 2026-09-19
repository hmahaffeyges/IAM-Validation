# VAL-108 — Cardio-epic CHK-3.1A self-calibration + Stage 1+2+3 run-everything on GSE69138 ischemic stroke 3-subtype

**Status:** PRE-REGISTERED, sealed before any framework scoring on GSE69138 cohort  
**Card:** cardio-epic v0.1 (initial card build; first cardio disease VAL)  
**Date:** 2026-04-28  
**RNG seed:** 20260428

## Substrate context — why this VAL uses self-calibration

VAL-106 established CHK-3.1A baseline for **TCGA HM450K sesame Level 3 substrate** at f_extreme 55.87% ± 2.44% / f_middle 7.42% ± 0.75% on full ~408K-CpG genome (n=210 KIRC + PRAD adjacent-normal).

VAL-107 established CHK-3.1B threshold for **cardio-epic on TCGA HM450K sesame Level 3 substrate** at extreme≥55.0% / middle≤8.5% / coverage≥7000 of 8100 (n=210 same cohort, cardio-epic 8,100-CpG subset).

A substrate-equivalence test against GSE69138 ave_beta (GenomeStudio AVG_Beta substrate) was run on full genome on all 404 samples:
- GSE69138 ave_beta: f_extreme **31.81% ± 3.54%** / f_middle 8.36%
- TCGA sesame Level 3:        f_extreme 55.87% ± 2.44% / f_middle 7.42%
- Δ f_extreme = **24.06 percentage points** (4.8× the 5pp tolerance)
- **Verdict: NOT EQUIVALENT.** GenomeStudio AVG_Beta (`M / (M + U + 100)` formula with +100 offset) compresses extreme values toward the middle compared to sesame's dye-bias-corrected output.

Per the CHK-3.1A/B split convention (locked 2026-04-28), GenomeStudio AVG_Beta is a **distinct substrate category** from TCGA HM450K sesame Level 3. The VAL-106/107 thresholds do not transfer.

A separate CHK-3.1A calibration on a structurally-separated **GenomeStudio AVG_Beta** cohort would require either GSE40360 (n=30 controls, brain tissue — structurally separate but tissue-type-divergent from blood) or GSE128235 (n=209 controls, but only normalized matrix + raw IDATs publicly released, requiring re-processing). Both are multi-VAL workstreams.

Per the cookbook discipline that calibration must come from a structurally-separated cohort, this VAL adopts a **within-cohort self-substrate-anchor** approach for GSE69138:

- The 404 samples in GSE69138 are all the same substrate (GenomeStudio AVG_Beta on whole blood, ischemic stroke patients, single cohort processing pipeline).
- The internal distribution (mean 31.81%, median 32.09%, IQR [30.14%, 33.73%], 10-90 percentile [27.74%, 35.39%]) is tight enough to define within-cohort substrate normality.
- Outlier samples (12 with f_extreme < 25%, 3 with f_extreme < 20%) self-flag as substrate-divergent and are excluded from biological analysis as data-integrity failures.
- Within-cohort multi-subtype contrast (large-artery atherosclerosis n=132, small-artery disease n=141, cardio-embolic n=127) is the framework biology test; the within-cohort design eliminates need for an external healthy baseline.

This approach is honest about its limitations: it does not establish a generalizable GenomeStudio AVG_Beta CHK-3.1A platform threshold (that remains TBD pending a future calibration VAL on a structurally-separated GenomeStudio cohort). It does establish data integrity for THIS cohort by virtue of the cohort's internal distribution tightness.

## Pre-locked CHK-3.1A self-calibration thresholds for GSE69138 ave_beta

Computed from the 404-sample distribution observed during substrate-equivalence test:
- **CHK-3.1A pass criterion (within-GSE69138 substrate)**: f_extreme ≥ 0.25 (25%, the lower bound of the 95% range, mean − 2*SD = 24.7%) AND f_middle ≤ 0.13 (13%, mean + 2*SD = 11.8% rounded up).
- **n_valid coverage**: ≥ 400,000 (cookbook standard).
- Samples failing these criteria self-flag as substrate-divergent and are excluded from biological analysis.

These thresholds are DERIVED from the cohort's internal distribution and are NOT generalizable to other GenomeStudio AVG_Beta cohorts. Other cohorts on this substrate would need their own self-calibration (or a proper structurally-separated calibration VAL).

## Pre-locked CHK-3.1B subset thresholds for GSE69138 ave_beta cardio-epic

The cardio-epic CHK-3.1B subset (8,100 CpGs, frozen at SHA `5a00e29ace75daae5a5bf7e3cfca26c16aa6dbd92750d16ebeaba4e874c48511`) will be applied to GSE69138 with thresholds derived from the same 404-sample within-cohort distribution at the subset level:

- Pre-locked rule: compute f_extreme_subset and f_middle_subset on the cardio-epic 8,100-CpG subset for each of the 404 GSE69138 samples
- Threshold: extreme_threshold_B = mean(f_extreme_subset) − 2*SD(f_extreme_subset), middle_threshold_B = mean(f_middle_subset) + 2*SD(f_middle_subset)
- Coverage threshold: n_subset_valid ≥ 7000 of 8100 (matches VAL-107)
- These threshold values will be derived from the cohort itself but their **definition rule is pre-locked here** before observation; values get locked in results.json at run time

## Pre-locked biological analysis methodology

**Cohort:** GSE69138 ischemic stroke patients, 404 discovery samples (replication cohort 185 deferred to VAL-108b).

**Subtypes (TOAST classification per source paper):**
- Large-artery atherosclerosis: n=132
- Small-artery disease: n=141
- Cardio-embolic: n=127
- Other / undetermined: remainder

**Stage 1 — Pooled-entropy A-score (universal):**
- Xu-538 immune panel intersected with GSE69138 ave_beta CpG IDs
- H_min(immune) = 0.838889 (frozen H_min anchor)
- Per-sample A-score = pooled-entropy score against H_min
- Subtype contrast: Cohen's d (large-artery vs small-artery), (large-artery vs cardio-embolic), (small-artery vs cardio-embolic), bootstrap 10,000-iteration 95% CI per pair

**Stage 2 — Cell-of-origin tile A-scores (Loyfer 25-tile run-everything):**
- All 25 Loyfer tiles scored per-sample (run-everything per Heath signoff 2026-04-26)
- Per-tile per-class A-score against frozen H_min anchor for each class
- Cardio-relevant tiles emphasized: Vascular_endothelial_cells, Left_atrium, Adipocytes
- Subtype contrast on per-tile A-scores: Cohen's d for each tile across each subtype pairing

**Stage 3 — Immune subcomposition (UniLIFE 19-cell + Salas 6-cell run-everything):**
- UniLIFE 19-cell A-scores per sample
- Salas 6-cell A-scores per sample  
- Subtype contrast on each immune cell-type A-score

**EpiSCORE HeartRef + Caggiano CelFiE:** scoring-time integration deferred to v0.2+ per cookbook decision (cardio-epic acquisition-time CHK-3.1B uses the 8,100-CpG subset only; EpiSCORE HeartRef CM/EC/FB/MP/SMC and Caggiano heart_meth/endothelial_meth are evaluated at scoring time via probe-mapping mechanisms not yet integrated into the pipeline).

## Pre-locked outcomes (per CCL-032 diagnostic order: data integrity → biology → framework)

**O1_CARDIO_EPIC_3SUBTYPE_DIFFERENTIATED_AT_STAGE_1**: Stage 1 immune A-score |d| ≥ 0.5 across at least one subtype pair, CI excludes zero — framework-Stage-1 differentiates ischemic stroke subtypes via immune class.

**O2_CARDIO_EPIC_3SUBTYPE_DIFFERENTIATED_AT_STAGE_2_TILES**: Stage 1 |d| < 0.5 OR Stage 1 not significant, but at least one Loyfer tile (especially Vascular_endothelial_cells, Left_atrium) shows |d| ≥ 0.5 with CI excluding zero across at least one subtype pair — framework discriminates at cell-of-origin level even when Stage 1 immune class is uniform.

**O3_CARDIO_EPIC_3SUBTYPE_UNDIFFERENTIATED**: Neither Stage 1 nor Stage 2 produces |d| ≥ 0.5 across any subtype pair — ischemic stroke subtypes are framework-equivalent at the assayed substrates. Cardio-epic can pool subtypes for clinical reporting; no subtype-stratified clinical action.

**O4_STAGE_3_DIFFERENTIATING**: Stage 1 + Stage 2 are both undifferentiating, but Stage 3 immune subcomposition (UniLIFE 19-cell or Salas 6-cell) shows |d| ≥ 0.5 across at least one subtype pair — discrimination at deep immune-subcomposition level.

**O5_DATA_INTEGRITY_FLAG**: ≥10% of samples fail CHK-3.1A self-calibration criteria; cohort substrate quality issue; biological analysis suspended pending substrate review.

## Pre-locked retrospective cardio-epic learning

Regardless of outcome category, this VAL produces three pieces of cardio-epic foundational information that propagate to v0.1 card:

1. **Healthy-substrate baseline for GenomeStudio AVG_Beta whole blood** (the 404-sample within-cohort mean f_extreme on full genome = 31.81% baseline; subset f_extreme TBD)
2. **Stage 1 + Stage 2 + Stage 3 A-score profiles for ischemic stroke** at the 3-subtype level (architectural map of how the cell responds to each stroke etiology)
3. **Within-subtype variance estimates** (per-class A-score SD within each subtype) — necessary for downstream deployment confidence intervals

## What does NOT propagate from this VAL

- No "stroke vs healthy" comparison (the cohort has no healthy controls; the design is multi-subtype within stroke)
- No substrate-level generalization to other GenomeStudio AVG_Beta cohorts (the self-calibration applies only to this cohort)
- No EpiSCORE HeartRef or Caggiano CelFiE biology (deferred to v0.2+)

## Reproducibility per CHK-7.6

- Inputs: GSE69138_ave_beta.txt.gz (SHA computed at acquisition), cardio_epic_chk31b_subset.txt (SHA `5a00e29...`), per_sample CHK-3.1A from substrate-equivalence run, GSE69138 sample-to-subtype clinical metadata extracted from GEO Series Matrix
- Environment: Python 3 stdlib + math/statistics
- Output: results.json + per_sample.csv + stratified.json (per-subtype tile A-scores)

## Pre-registration seal

This prereg is sealed via SHA-256 of its content prior to any framework scoring (Stage 1 A-score computation, Stage 2 tile scoring, Stage 3 immune subcomposition) on GSE69138 samples. The substrate-level CHK-3.1A statistics (f_extreme distribution per sample) were observed before this seal but are not framework outputs; framework scoring under this seal commits to the methodology and outcome categories specified above.
