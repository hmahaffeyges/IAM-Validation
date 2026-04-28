# VAL-099 Pre-Registration — crc-epic Age-Stratified Re-Analysis on TCGA-COAD HM450

**Sealed:** 2026-04-28 (sealed before re-scoring beyond the previously-published VAL-062 result)
**Card:** crc-epic v2.4 (early-onset rectal subsection, age-stratified colon arm)
**Cohort:** TCGA-COAD HM450 paired tumor/adjacent-normal — 26 paired pairs (the existing VAL-061 + VAL-062 sealed cohort, no new data)
**RNG seed:** 20260428

---

## Purpose

VAL-098 (TCGA-READ paired tumor/normal, n=7) confirmed the cycling-class architectural drift signal in the rectal subsite at paired d = +0.612 [+0.227, +1.882]. The under-50 stratum in VAL-098 was n=1 (TCGA-AG-A01Y, 49.6 y) — descriptive-only per CHK-2.7.

VAL-099 re-analyzes the existing TCGA-COAD 26-paired-pair cohort (the cohort that anchors VAL-061/VAL-062) by age decile and anatomic subsite. No new data download; pure re-execution of the VAL-062 cycling-class methodology on the same 52 .txt files (already cached in `/home/claude/edear_working/VAL-062_revisit/coad_downloads/`) plus a re-score of the run-everything 25-tile output, followed by stratified analysis using GDC clinical metadata pulled at run time.

The purpose is to characterize the under-50 stratum in colon cancer at modest cohort size (n=3 confirmed under-50, 2 patients with NA age) before VAL-100 attempts the under-50 buffy-coat polyp arm. VAL-099 is descriptive at the under-50 level (n is too small for inferential claims) and confirmatory at the 50+ level (where it should reproduce VAL-062's +0.724 result up to small numerical drift from RNG re-seeding).

This VAL is **descriptive-only at the under-50 stratum** per CHK-2.7. The under-50 evidence chain for crc-epic v2.4 relies on VAL-099 (descriptive direction signal at n=3 colon under-50) + VAL-100 (n=51 buffy-coat polyp under-50 EPIC) for the full early-onset evidence base.

---

## Cohort

**TCGA-COAD HM450 paired tumor/adjacent-normal — 26 paired pairs.** Same patient list as VAL-061/VAL-062. Files cached locally at `/home/claude/edear_working/VAL-062_revisit/coad_downloads/` (52 .txt files). No new download required.

Patient list (all 26): TCGA-A6-2671, TCGA-A6-2675, TCGA-A6-2679, TCGA-A6-2680, TCGA-A6-2685, TCGA-A6-2686, TCGA-A6-4107, TCGA-A6-5667, TCGA-AA-3492, TCGA-AA-3495, TCGA-AA-3510, TCGA-AA-3655, TCGA-AA-3660, TCGA-AA-3663, TCGA-AA-3697, TCGA-AA-3712, TCGA-AZ-6598, TCGA-AZ-6599, TCGA-AZ-6600, TCGA-AZ-6601, TCGA-G4-6295, TCGA-G4-6298, TCGA-G4-6311, TCGA-G4-6314, TCGA-G4-6320, TCGA-G4-6625.

**Clinical metadata** pulled fresh from NIH GDC public API at run time (no dbGaP required) — age_at_diagnosis, tissue_or_organ_of_origin (subsite), ajcc_pathologic_stage, gender, race, ethnicity. Pre-fetched and saved to `clinical_metadata.json`. Pre-known cohort age distribution (from pre-seal fetch):

| Stratum | n | Patients |
|---|---|---|
| under_50 | 3 | TCGA-A6-2685 (48.6), TCGA-A6-5667 (40.4), TCGA-AA-3663 (42.9) |
| age_50_plus | 21 | (all remaining patients with confirmed age ≥ 50) |
| age_NA | 2 | TCGA-AZ-6601, TCGA-G4-6625 (excluded from age stratification; included in pooled analysis) |

**Note on TCGA-G4-6625:** GDC clinical metadata reports tissue_or_organ_of_origin = "Skin, NOS" for this patient. This is anomalous for a TCGA-COAD case. The patient's methylation files are part of the TCGA-COAD project; we include the patient in the pooled analysis (consistent with VAL-062) and in the by_subsite stratification under "Anomalous" rather than excluding, but flag the discrepancy for the outcome.md.

---

## Class assignment

Cycling class. H_min = 0.856055 (G-002 MCMC posterior, R-hat = 1.0003). Reference β = 0.738 from TCGA-LUAD matched normal (cycling-class universal reference). Identical to VAL-062 methodology.

---

## Method

1. **Score every sample** with full-HM450 cycling-class A-score: `A_cycling(sample) = mean over valid HM450 CpGs of [ H(β) / 0.856055 ]` where valid CpGs are the per-sample set with ≥400,000 valid β values per sample (matches VAL-062 exactly).
2. **Score every sample with run-everything 25-tile** per-class A-score using Loyfer 25-tile reference atlas (same atlas as VAL-098).
3. **Compute paired d for the pooled cohort** (n=26 pairs) — should reproduce VAL-062's +0.724 within RNG / numerical tolerance.
4. **Compute paired d for stratified subsets** (pre-locked stratifications):
   - by_age: under_50 (n=3) vs age_50_plus (n=21). under_50 is descriptive-only per CHK-2.7 (n < 5 threshold for inferential claim).
   - by_subsite: Cecum (n=5), Ascending colon (n=7), Sigmoid colon (n=3), Hepatic flexure (n=2), Descending colon (n=1), Colon NOS (n=5), Anomalous-Skin NOS (n=1), Not Reported (n=2).
   - by_sex: female vs male.
5. **Bootstrap 95% CIs** with 10,000 iterations BCa-equivalent on the pooled and 50+ strata. Under-50 stratum at n=3 produces wide CIs; report point estimate + descriptive note rather than precise CI claims.
6. **Outcome label decision matrix** (pre-locked, see below).
7. **Compare to VAL-062 anchor** — pooled paired d should match VAL-062 (+0.724) within ±0.05 (RNG drift tolerance). Material divergence is a flag, not an expected result.

RNG seed 20260428.

---

## Pre-registered outcomes

**O1_AGE_STRATIFIED_DIRECTION_CONFIRMED.** Pooled paired d ≥ +0.5 with 95% CI lower bound > 0 (reproduces VAL-062). AND: under_50 stratum direction descriptively positive (mean ΔA > 0). AND: age_50_plus stratum paired d ≥ +0.5 with 95% CI lower bound > 0.

**O2_AGE_STRATIFIED_50PLUS_ONLY.** Pooled and 50+ strata reproduce VAL-062 direction and magnitude, but under_50 stratum direction is null or negative (n=3 too small to confirm direction; descriptive flag only). Outcome label set, but card v2.4 commentary explicitly notes the under-50 direction was descriptive and inconclusive at this n.

**O3_VAL_062_NON_REPRODUCED.** Pooled paired d differs from VAL-062 +0.724 by more than ±0.05 in either direction. Material flag — investigation required before card update.

**O5_UNEXPECTED.** Any other pattern. Convene with Heath before card update direction.

---

## Why under-50 stratum is descriptive-only here (CHK-2.7)

n=3 produces 95% CIs wide enough that any claim about "the under-50 colon stratum direction" from this VAL alone is statistically descriptive, not inferential. The pre-locked rule: report point estimate + direction; do NOT claim "under-50 colon cycling-class signal validated" from VAL-099 alone. The under-50 evidence chain in crc-epic v2.4 is:

- **VAL-098** (sealed 2026-04-28): rectal subsite paired d = +0.612 at pooled n=7; under-50 n=1 ΔA = +0.016 descriptive-only.
- **VAL-099** (this prereg): colon subsite under-50 stratum descriptive at n=3; pooled n=26 reproduces VAL-062 anchor.
- **VAL-100** (next): under-50 buffy-coat polyp Stage 1 immune n=51 EPIC, the proper-power under-50 confirmation arm.

VAL-099's contribution to the chain is direction concordance at modest n in the colon subsite. Statistical power for the under-50 claim comes from VAL-100.

---

## CCL-039 application (CHK-4.11)

Per CHK-4.11, future preregs with run-everything 25-tile per-class output on tumor-vs-adjacent-normal paired comparisons must NOT pre-lock "cell-of-origin tile shows positive d" as O1 criterion. VAL-099 expects the Colon_epithelial_cells tile to read paired d in the negative direction (consistent with VAL-098 rectal cohort and VAL-062 revisit cookbook-wide CCL-039 finding). The 25-tile output is supplementary documentation; the pre-registered O1 criterion is the full-HM450 cycling-class result.

---

## Reproducibility

- **Source data:** Already on disk at `/home/claude/edear_working/VAL-062_revisit/coad_downloads/` (52 .txt files). For independent reproduction: download via NIH GDC public API per `COAD_matched_manifest.json` from VAL-061/VAL-062. Public access, no dbGaP required.
- **Clinical metadata:** GDC public API at run time (`clinical_metadata.json` saved as part of VAL-099 deliverables).
- **Reference atlas:** Loyfer 25-tile (same as VAL-098), `atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv`.
- **RNG seed:** 20260428.
- **Environment:** Python 3 stdlib + numpy + pandas + scipy + matplotlib.

---

## EDEAR commercial deployment unaffected

Per CCL-037, VAL-099 is retrospective cookbook validation with no impact on EDEAR commercial deployment (single-pipeline patient-vs-internal-reference, structurally insulated from cookbook-validation cohort coverage).
