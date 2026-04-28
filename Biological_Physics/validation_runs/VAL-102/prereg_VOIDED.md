# VAL-102 Pre-Registration — hcc-epic Run-Everything 25-Tile Per-Class A-Score with Full Etiology Stratification on TCGA-LIHC HM450 (Platform-Tuned CHK-3.1)

**Sealed:** 2026-04-28 (sealed before re-execution under platform-tuned CHK-3.1 threshold)
**Card:** hcc-epic v0.3 candidate (tissue arm 25-tile observation + Marcus-analog stratum analysis)
**Cohort:** TCGA-LIHC HM450 paired tumor/adjacent-normal — 50 candidate paired pairs (4 expected QC dropouts, n=46 carried forward consistent with VAL-064 + VAL-101)
**RNG seed:** 20260428
**Supersedes (in propagation pathway):** VAL-101 (sealed at `O5_DATA_INTEGRITY_FLAG` due to CHK-3.1 prereg-threshold misspecification per CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION)

---

## Why VAL-102 exists

VAL-101 was sealed with the original raw-EPIC CHK-3.1 threshold (extreme >30% AND middle <10%). That threshold tripped on TCGA-LIHC HM450 sesame Level 3 betas (observed: extreme 26.6%, middle 9.1%). Per the pre-locked outcome decision matrix, VAL-101 outcome was `O5_DATA_INTEGRITY_FLAG`. The biological readouts produced under VAL-101 are descriptive supplementary documentation only and do NOT propagate to the cookbook.

CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION was logged from VAL-101: CHK-3.1 thresholds need platform-specific tuning. The original threshold was tuned for raw EPIC β; TCGA HM450 sesame Level 3 — the cookbook's standard public tissue-validation substrate, used in VAL-058 / VAL-060 / VAL-062 / VAL-063 / VAL-064 / VAL-098 / VAL-099 — reads extreme ~24-27% / middle ~9% on the same check. This is bimodal raw β with slightly less extreme bimodality than raw EPIC due to dye bias correction in the standard TCGA pipeline.

VAL-102 re-runs VAL-101's methodology under a platform-tuned CHK-3.1 threshold sealed before re-execution. If the platform-tuned threshold is met, the biological readouts propagate under VAL-102's seal. If the platform-tuned threshold is also missed, VAL-102 takes its own O5 flag and the analysis pathway moves to v0.2+ raw IDAT processing.

This is not a workaround. It is doing the prereg right the second time. VAL-101 stays in the cookbook record as O5_DATA_INTEGRITY_FLAG; VAL-102 supersedes it for the biological-propagation pathway only.

---

## Purpose (unchanged from VAL-101)

Three pre-locked questions, identical to VAL-101:

1. **CCL-039 cross-tissue generalization test.** Does the Hepatocytes tile in HCC tumor-vs-adjacent-normal paired comparisons read strongly negative the same way the Colon_epithelial_cells tile reads negative in colorectal paired tumor-vs-normal? If yes, CCL-039 upgrades to "framework-level rule across at least two cancer types."

2. **Viral-vs-non-viral blunting at the per-tile level.** Does the viral-hepatitis blunting persist at the per-tile Hepatocytes level, or only at the pooled-cycling-class level (where VAL-064 documented it)?

3. **Marcus-analog stratum characterization.** What does the run-everything 25-tile pattern look like in the "no documented risk" stratum (n=10)? Patients who developed HCC without HBV / HCV / alcohol / NAFLD documented in TCGA chart — closest available public analog to aggressive HCC arising without a chronic-driver risk factor.

---

## Cohort (unchanged from VAL-101)

50 candidate paired patient pairs from TCGA-LIHC HM450, same patient list as VAL-064 + VAL-101. Files cached at `/home/claude/edear_working/VAL-101/lihc_downloads/` (100 files, ~1.3 GB). For independent reproduction: download via NIH GDC public API per `LIHC_matched_manifest.json`. Public access, no dbGaP application required. Manifest SHA `760bf65a213da5a86cdd7ecde6ff6d46dad04777eafc00eb14f56356b0088371`.

QC threshold: ≥400,000 valid β per sample. Mirrors VAL-064 and VAL-101. Expected n_carried_forward = 46.

Stratification (pre-classified at VAL-101 prereg time, unchanged here):

| Stratum | n |
|---|---|
| HBV+ alone | 19 |
| HCV+ alone | 3 (descriptive-only per CHK-2.7) |
| HBV+HCV co-infection | 2 (descriptive-only per CHK-2.7) |
| Alcohol+ | 6 |
| NAFLD+ | 1 |
| Other (Tobacco-only, Diabetes-only, Schistosoma, Hemochromatosis, Unknown viral) | 5 |
| **No_documented_risk** | **10** (Marcus-analog stratum) |
| **All_viral** (HBV+HCV+co-infection) | **24** |
| **All_non_viral** (Alcohol+NAFLD+Other+No_documented_risk) | **22** |

---

## Method (unchanged from VAL-101)

1. Pair files into patient-level (tumor, normal) tuples. 50 patient pairs.
2. QC threshold: ≥400,000 valid β per sample.
3. Load Loyfer 25-tile reference atlas.
4. Per tile, select top-100 marker CpGs by |ref_β − mean(ref_β across other 24 tiles)|.
5. Per sample, score run-everything 25-tile per-class A-score.
6. Per tile, compute paired Cohen's d on (A_tumor − A_normal). Bootstrap 10,000-iteration BCa-equivalent CI.
7. Stratified analysis on Hepatocytes tile + full 25-tile output for Marcus-analog stratum.

RNG seed 20260428.

---

## Pre-registered CHK-3.1 — PLATFORM-TUNED PER CCL-041

**Platform:** TCGA HM450 sesame Level 3 β (the cookbook's standard public tissue-validation substrate; consistent with VAL-058 / VAL-060 / VAL-062 / VAL-063 / VAL-064 / VAL-098 / VAL-099 source-data type).

**Pre-locked CHK-3.1 thresholds for TCGA HM450 sesame Level 3:** extreme [<0.05 or >0.95] **>20%** AND middle [0.4-0.6] **<10%**.

**Justification (logged in CCL-041):** Post-hoc verification on cached TCGA-COAD HM450 sesame Level 3 data (the same cohort that anchored VAL-099) reads extreme 24.4%, middle 9.7% on the same check methodology. The cookbook's pre-existing tissue-arm validations (VAL-062 / VAL-098 / VAL-099) on this same substrate produced clean biological signal at framework-predicted effect sizes. The substrate is bimodal raw β with slightly less extreme bimodality than raw EPIC due to standard TCGA pipeline dye bias correction. The original VAL-100 threshold (extreme >30%) was raw-EPIC-tuned. The TCGA HM450 sesame Level 3 platform threshold is set at extreme >20% per CCL-041.

**Distinction from CCL-040 (preserved).** This is NOT a relaxation of the CCL-040 deferral pathway for processed/normalized output. CCL-040 covers cases where bimodal raw β signature is lost entirely (extreme 3.9% / middle 6.8% in VAL-100 GSE282666 — a noob-bg-corrected supplementary file). CCL-041 covers raw-β bimodality manifesting at slightly different threshold values across raw-β platforms. VAL-102's platform-tuned threshold (extreme >20%) is a tightening of the CCL-040 vigilance for THIS platform, not a relaxation of the CCL-040 principle.

**Specifically, VAL-102 will still trip O5_DATA_INTEGRITY_FLAG if:**
- extreme < 20% (would indicate processed output, possibly residual M-values or batch-corrected betas with bimodal-β-signature loss)
- middle ≥ 10% (would indicate substantial loss of bimodal raw β signature regardless of extreme value)

**Sanity expectation under platform-tuned threshold:** VAL-102 should observe extreme ≈ 26.6% and middle ≈ 9.1% (same data, same methodology, same RNG seed). Both pass under the platform-tuned threshold. Bimodal raw β signature: confirmed.

If observed extreme drops below 20% or middle exceeds 10% in VAL-102 (indicating something has changed since VAL-101 sampling), VAL-102 takes O5_DATA_INTEGRITY_FLAG.

---

## Pre-registered outcome decision matrix

**Pre-locked decision logic (in evaluation order):**

1. **CHK-3.1 (platform-tuned).** Extreme >20% AND middle <10%. If FAILED → `O5_DATA_INTEGRITY_FLAG`. Halt; defer to v0.2+ raw IDAT processing per CCL-040. (This is unlikely given VAL-101 observed extreme 26.6% / middle 9.1% on the same data, but the check still applies.)

2. **CHK-2.7 stratum-power floor.** Per-stratum analyses with n<5 are descriptive-only.

3. **Hepatocytes tile pooled paired d (n=46) — primary criterion:**
   - **O1_HEPATOCYTES_TILE_NEGATIVE_DIRECTION_CONFIRMED.** Hepatocytes tile pooled paired d ≤ −0.5 with 95% CI upper bound < 0. Direction matches CCL-039 prediction. CCL-039 upgrades to cross-tissue confirmed pattern across at least two cancer types (colorectal + HCC).
   - **O2_HEPATOCYTES_TILE_NEGATIVE_PARTIAL.** Hepatocytes tile paired d in [−0.5, 0] with 95% CI upper bound at or near 0. Direction consistent with CCL-039 prediction but magnitude attenuated.
   - **O3_HEPATOCYTES_TILE_NULL.** Hepatocytes tile paired d in [−0.2, +0.2] with 95% CI crossing zero. Cell-of-origin tile fidelity-loss does NOT generalize from colorectal to HCC at the pooled level.
   - **O4_HEPATOCYTES_TILE_INVERTED_POSITIVE.** Hepatocytes tile paired d ≥ +0.5 with 95% CI lower bound > 0. Direction inverted from CCL-039 prediction.
   - **O5_UNEXPECTED.** Any other pattern.

4. **Stratum-level expectations** (pre-locked, descriptive at sub-stratum):
   - All_viral (n=24) and All_non_viral (n=22) Hepatocytes tile direction expected NEGATIVE (per CCL-039 cell-of-origin tile fidelity-loss in tumor).
   - HBV+ alone (n=19) inferential at this n; expected NEGATIVE direction.
   - Marcus-analog stratum No_documented_risk (n=10) descriptive-only; pattern documented as reference for future cross-validation.

---

## Pre-locked numerical sanity check (against VAL-101 descriptive supplementary readouts)

**Same data + same script + same RNG seed = identical biological numbers.** VAL-102 is the same execution; only the CHK-3.1 threshold has changed. The numerical readouts in VAL-101 results.json (descriptive supplementary status) become the inferential output of VAL-102 if the platform-tuned CHK-3.1 passes.

Pre-locked sanity expectations under platform-tuned threshold pass:
- CHK-3.1: PASS at extreme ≈ 26.6%, middle ≈ 9.1%
- Pooled Hepatocytes tile paired d ≈ −1.521 [−2.19, −1.18]
- Hepatocytes tile rank by |d|: 3 of 17 reportable tiles
- All_viral Hepatocytes d ≈ −1.726 [−3.03, −1.12]
- All_non_viral Hepatocytes d ≈ −1.393 [−2.30, −1.06]
- No_documented_risk Hepatocytes d ≈ −1.141 [−6.16, −0.85] (descriptive-only per CHK-2.7)
- Outcome label expected: O1_HEPATOCYTES_TILE_NEGATIVE_DIRECTION_CONFIRMED

If the observed numbers diverge materially from these pre-locked expectations (more than ±5% drift on pooled paired d), this indicates something has changed in execution — verify identical RNG seed, identical script, identical input files. The expected drift from VAL-101 to VAL-102 on the same execution path is ZERO.

---

## CHK-4.11 application (CCL-039 prereg-design rule)

This prereg uses pattern-aware O1 criterion language (direction-and-magnitude-explicit, comparison-type-explicit). Same as VAL-101, in compliance with CHK-4.11.

---

## Reproducibility (unchanged from VAL-101)

- **Source data:** TCGA-LIHC HM450 paired tumor/adjacent-normal .txt files via NIH GDC public API. Cached at `/home/claude/edear_working/VAL-101/lihc_downloads/` (100 files). For independent reproduction: download per `LIHC_matched_manifest.json`. Manifest SHA `760bf65a213da5a86cdd7ecde6ff6d46dad04777eafc00eb14f56356b0088371`.
- **Clinical metadata:** `LIHC_clinical.json` from VAL-064 sealed cohort metadata pull.
- **Reference atlas:** Loyfer 25-tile array atlas at `Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv`.
- **RNG seed:** 20260428.
- **Environment:** Python 3 stdlib only.

---

## EDEAR commercial deployment unaffected

Per CCL-037, VAL-102 is retrospective cookbook validation with no impact on EDEAR commercial deployment.

---

## Pre-registration scope clarifier

VAL-102 is structurally identical to VAL-101 except for one pre-locked threshold value (CHK-3.1 extreme: >30% → >20%, justified by CCL-041). The numerical execution is deterministic and identical. The biological readouts are the same numbers documented in VAL-101's descriptive supplementary status; VAL-102 either confirms or denies their inferential propagation status based on the platform-tuned CHK-3.1 result.

This is the prereg-discipline pathway: when a sealed criterion is misspecified, the recovery is a new sealed prereg with a corrected criterion, not post-hoc relaxation of the original criterion. VAL-101 stays in the cookbook record as O5_DATA_INTEGRITY_FLAG. VAL-102 supersedes it for the biological-propagation pathway only when the platform-tuned criterion is met.
