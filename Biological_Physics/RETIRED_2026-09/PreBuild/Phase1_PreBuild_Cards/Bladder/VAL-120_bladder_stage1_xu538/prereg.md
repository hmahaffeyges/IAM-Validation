# VAL-120 — Pre-Registration

**VAL ID:** VAL-120
**Card target:** bladder-epic v0.1 (Phase C — Stage 1 immune red flag)
**Substrate cohort:** TCGA-BLCA (n=440 = 418 primary tumor + 21 solid tissue normal + 1 metastatic), Illumina HM450K sesame Level 3 from GDC
**Sealed:** [SEAL_TIMESTAMP at execution]
**Sealed BEFORE β read:** YES (this prereg.md is sealed before val120 script reads any β values)

---

## Question

Does the universal Stage 1 immune red-flag score (Xu-538 panel, pooled-entropy A_immune via Shannon binary entropy averaged over CpGs present in sample, normalized by H_min(immune) = 0.838889) fire on TCGA-BLCA tumor tissue compared to adjacent-normal? At what magnitude — paired (n=21 patients with both tumor + adjacent-normal samples) and unpaired (Welch d on 418 tumor vs 21 normal)?

This is the patient flow Stage 1 — the immune red flag. Heath's framing 2026-05-01: "follow the patient flow. Phase 1: red flag immune response and those tests associated with the immune class cells. Then Stage 2 cell-of-origin run-everything. Then Stage 3 immune fine-tune."

---

## Why this matters operationally

Stage 1 Xu-538 is the universal immune-architecture entropy panel — it fires on disease classes where the tumor microenvironment shows architectural drift relative to healthy substrate. Bladder is a high-prior cancer for Stage 1 firing because:

- Urothelial carcinoma is heavily immune-infiltrated (TILs, BCG immunotherapy is standard of care for NMIBC, PD-L1 checkpoint inhibitors approved for advanced UC)
- Christensen lab Chen 2022 NMIBC blood EPIC n=603 shows mdNLR (methylation-derived neutrophil-to-lymphocyte ratio) is a hazard factor for recurrence — direct evidence of methylation-detectable immune architecture in NMIBC patients
- Bladder is the most expensive cancer to manage per capita largely because of repeated cystoscopy surveillance — a methylation-detectable immune red flag has direct clinical-utility implications

VAL-120 produces the bladder-specific Stage 1 reading that anchors v0.1 card claims.

---

## Calibration discipline (CCL-041 + within-cohort self-cal)

Stage 1 Xu-538 has NOT been calibrated against a structurally-separated healthy cohort yet. The Wave 1 Shared Task A (VAL-114, on Hannum 2013 GSE40279 n=656 healthy aging blood) is queued as the v0.X+1 promotion path. **Until VAL-114 lands, all Stage 1 readings are within-cohort self-calibrated.** Per DISC-CARDIO-005 + CCL-041, this is documented as the v0.1 limitation up front. The bladder-epic card README will explicitly state the Stage 1 pre-locked claim: "consistent with architectural drift; within-cohort self-cal at v0.1; Wave 1 calibration on Hannum 2013 is the v0.X+1 promotion path."

---

## Cohort inventory + provenance

### TCGA-BLCA (Phase C primary)
- **Source:** GDC API `https://api.gdc.cancer.gov/data/{file_id}` with filter `cases.project.project_id=TCGA-BLCA AND data_type=Methylation Beta Value AND platform=Illumina Human Methylation 450`
- **Acquired:** 2026-04-30 / 2026-05-01 via fetch_blca_resume.py (parallel ThreadPoolExecutor, 8 workers; all 440 files cached in 38 sec)
- **Manifest:** `/home/claude/edear_working/bladder_epic/blca_manifest.json` (440 entries with file_id, file_name, file_size, case_id, sample_id, sample_type, local_path)
- **Total files:** 440 (verified by GDC API)
- **By sample type:**
  - Primary Tumor: 418
  - Solid Tissue Normal: 21
  - Metastatic: 1 (held aside as exploratory; not in primary contrast)
- **Paired patients (have both adjacent-normal + primary tumor):** 21
- **Substrate:** TCGA HM450K sesame Level 3 (canonical substrate baseline established by VAL-106; substrate-matched to VAL-119 calibration cohort)
- **Local path:** `/home/claude/edear_working/bladder_epic/blca_betas/`

### Tier classification (CHK-1.6)
- TCGA-BLCA = **Tier 1** (open access, GDC public data portal). No biobank application required.

---

## Atlas inventory

### Stage 1 Xu-538 panel
- **Source:** Xu Z, Sandler DP, Taylor JA. *JNCI* 2020. DOI: 10.1093/jnci/djz065
- **Panel file:** `/home/claude/IAM-Validation/Biological_Physics/validation_runs/xu538_panel.json`
- **Panel ID:** `Xu2020_breast_cancer_replicated_full`
- **n CpGs:** 538
- **Selection criterion:** All unique CpG IDs from Xu 2020 djz065 Supplementary Table 1 (dmCpGs at p<1e-7 in Sister Study, replicated in EPIC-Italy)
- **Methodology:** Pooled-entropy A_immune = mean(Shannon_H(β_i) for i in panel ∩ sample) / H_min(immune); H_min(immune) = 0.838889 (G-003b MCMC posteriors, frozen 2026-04-06)

---

## Pre-locked outcomes

Per CHK-2.7 (magnitude-based |d| thresholds with direction labels for direction-ambiguity cases). Cancer-vs-normal Stage 1 Xu-538 has historical positive direction precedent (VAL-058 prostate paired d=+0.497; VAL-008 specimen matrix |ΔA| 0.132-0.301 across 19 cancer types; VAL-001 6/6 cancer types). Direction expectation: **POSITIVE (tumor A > adjacent-normal A)**. But locked as magnitude with direction labels for surfacing potential direction flips (CCL-041 / CHK-2.7 lesson from VAL-118 LE-NEGATIVE).

### O1 — `STAGE1_IMMUNE_FIRES_POSITIVE`

|d_paired| ≥ 0.30 AND direction = POSITIVE on the n=21 paired contrast. Stage 1 immune red flag fires consistent with bladder cancer architectural drift expectation. Confirmation that the universal Stage 1 panel transfers to bladder substrate.

### O2 — `STAGE1_IMMUNE_FIRES_NEGATIVE`

|d_paired| ≥ 0.30 AND direction = NEGATIVE on the n=21 paired contrast. Stage 1 fires but in opposite direction from prior cancer-vs-normal expectation. This would be surprising — would indicate bladder tumor methylation-architecture is more ordered than adjacent-normal, which contradicts general cancer field-effect expectation. Direction-flip surfacing per CHK-2.7. If fires, convene with Heath; classify as O5 (UNEXPECTED) for sealing direction.

### O3 — `STAGE1_IMMUNE_NULL`

|d_paired| < 0.30 on the n=21 paired contrast. Bladder Stage 1 immune signal does not reach the magnitude threshold. Two interpretations: (a) bladder is a low-magnitude Stage 1 cancer; (b) within-cohort self-cal is hiding the signal because adjacent-normal in TCGA-BLCA carries field-effect drift (VAL-003 lesson: 28/28 cancer types show 20.2% elevation in adjacent normal). Direction labeled per observation. Card v0.1 claims documented as Stage-1-null with Wave-1 calibration as v0.X+1 promotion path.

### O4 — `STAGE1_DATA_INTEGRITY_FAILURE`

CHK-3.1A on TCGA-BLCA fails on >25% of samples; or CHK-3.1B coverage on Xu-538 panel fails on >25% of samples; or paired pair count <15 (loss of statistical power). Data-integrity halt; v0.1 deferred pending re-fetch and re-verification.

### O5 — `STAGE1_UNEXPECTED`

Anything not anticipated in O1-O4. Per CCL-032 (data integrity → biology → framework), classify as O5 if data integrity is uncertain or result contradicts expected biology. Convene with Heath before sealing direction.

---

## Pre-locked thresholds (CHK-2.1 + CHK-2.7)

| Threshold | Pre-locked value | Rationale |
|---|---|---|
| Magnitude threshold for "fires" | |d_paired| ≥ 0.30 | Same as VAL-118 LE prostate threshold; VAL-058 sealed paired d=+0.497 reference |
| Direction labels | POSITIVE / NEGATIVE | CHK-2.7 |
| Minimum paired pairs | n ≥ 15 | Statistical power floor |
| CHK-3.1A pass rate | ≥ 75% (substrate-permissive at Phase C) | Phase C scoring; not Phase B calibration |
| CHK-3.1B Xu-538 coverage per sample | ≥ 80% (≥430/538 CpGs present per sample) | HM450K → Xu-538 panel is well-covered (538 CpGs all from 450K design) |
| RNG seed | 20260420 | Cookbook standard |

---

## Statistical methodology

### Primary contrast — paired Cohen's d
- 21 patients with both adjacent-normal and primary tumor sample
- Per-patient: paired_diff_i = A_immune(tumor) - A_immune(adjacent-normal)
- d_paired = mean(paired_diff) / sd(paired_diff)
- 95% CI via t-distribution
- p-value via two-tailed paired t-test
- Direction = sign(mean(paired_diff))

### Secondary contrast — unpaired Welch d
- All 418 tumor vs all 21 normal
- d_welch = (mean_tumor - mean_normal) / pooled_sd
- 95% CI via Welch's t-distribution
- p-value via Welch's two-sample t-test
- Direction = sign(mean_tumor - mean_normal)
- Higher noise than paired but covers full cohort

### Exploratory — metastatic single-sample
- A_immune of the 1 metastatic sample, reported but not part of primary outcome assessment

### Substrate baseline check (CHK-3.1A)
- All 440 samples scored against full-genome baseline
- Reported as observed_f_extreme_mean ± sd, observed_f_middle_mean ± sd
- Compare to VAL-106 sealed baseline (55.87% ± 2.44%, 7.42% ± 0.75%)

---

## Reproducibility triple (CHK-7.6)

### Source code
`val120_bladder_stage1_xu538.py` — Python 3.12 stdlib + numpy + scipy.stats. Loads Xu-538 panel JSON, loads TCGA-BLCA β files via the manifest at `/home/claude/edear_working/bladder_epic/blca_manifest.json`, computes per-sample CHK-3.1A on full genome + per-sample CHK-3.1B on Xu-538 subset + A_immune score using H_min(immune) = 0.838889. Identifies paired patients (case_id present in both Solid Tissue Normal and Primary Tumor groups). Computes paired d, unpaired Welch d, 95% CIs, p-values.

### Inputs
1. **Xu-538 panel:** `/home/claude/IAM-Validation/Biological_Physics/validation_runs/xu538_panel.json` (538 CpG IDs)
2. **TCGA-BLCA β files:** `/home/claude/edear_working/bladder_epic/blca_betas/` × 440 files
3. **BLCA manifest:** `/home/claude/edear_working/bladder_epic/blca_manifest.json` (sample type + case_id per file)

### Environment
- Python 3.12.3
- numpy 2.4.4
- scipy 1.17.1
- Expected runtime: ~5-8 minutes for n=440 cohort
- Expected memory: ~500 MB peak

### Expected headline output
- `VAL-120_results.json` — per-contrast d / CI / p / direction; per-tile CHK-3.1A summary; per-sample A_immune
- `VAL-120_per_sample.csv` — sample_id, case_id, sample_type, n_cpgs_genome, f_extreme, f_middle, n_xu538_present, A_immune, paired_diff (where applicable)
- `VAL-120_paired_pairs.json` — manifest of 21 patient pairs with case_id, normal sample_id, tumor sample_id, A_normal, A_tumor, paired_diff
- `outcome.md` — sealed outcome class

---

## RNG seed

20260420 (cookbook standard).

---

## SHA-256 of this prereg

To be computed at seal time and recorded in `PREREG_SEAL.txt` before val120 script reads any β files.

---

## Pre-registered audit chain

This prereg seals against:
- bladder-epic v0.1 Phase 0 cohort survey (signed off 2026-04-30)
- VAL-119 BladderRef calibration (sealed 2026-05-01T03:46:00Z)
- Calibration TODO v0.5 Phase C requirement
- Guardrail #11 (calibration before testing — Stage 1 within-cohort self-cal documented as v0.1 limitation)
- CCL-041 (prereg locked before β read)
- CCL-046 (atlas selection traces to canonical-document-named candidates)
- CHK-2.7 (magnitude-based |d| with direction labels)
- DISC-CARDIO-005 (within-cohort self-cal documentation requirement when calibration anchor is not in scope)

val120 script execution begins ONLY after this prereg.md is sealed and SHA-hashed. Outcome sealed against pre-locked thresholds above.

**No outcome may be added post-hoc. No threshold may be relaxed post-hoc. No exception under CCL-041.**
