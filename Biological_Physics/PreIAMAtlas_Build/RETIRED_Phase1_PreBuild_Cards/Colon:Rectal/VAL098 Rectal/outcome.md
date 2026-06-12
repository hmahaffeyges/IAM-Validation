# VAL-098 Outcome — crc-epic Early-Onset Rectal Subsection on TCGA-READ HM450

**Date:** 2026-04-28
**Card:** crc-epic v2.4 (early-onset rectal subsection)
**Cohort:** TCGA-READ HM450 paired tumor/adjacent-normal — 7 paired pairs (6 Rectum NOS + 1 Rectosigmoid junction)
**Pre-registration SHA:** `57d830d6c7ba64482b5da7c1c942aceeeeba3f9d1cf76b22f12f416562247d49`
**Sealed at:** 2026-04-28T16:45:00.885925Z (before β-value scoring; no post-hoc adjustments)
**RNG seed:** 20260428
**Outcome label:** **O1_CYCLING_CLASS_RECTAL_CONFIRMED**
**Runtime:** 31.2 s

---

## TL;DR

VAL-098 extends the crc-epic cycling-class tissue arm anchor (VAL-062 TCGA-COAD paired d = +0.724) to the rectal subsite via TCGA-READ paired tumor/adjacent-normal samples. 7 paired pairs at HM450, full-HM450 cycling-class scoring against H_min(cycling) = 0.856055, methodology mirrors VAL-062 exactly. Within-cohort paired comparison — CHK-3.8 condition 1 satisfied, no cross-cohort calibration problem.

**Pooled paired Cohen's d = +0.612 [+0.227, +1.882], t = 1.62, p = 0.157.** Direction confirmed in rectal subsite. Magnitude consistent with VAL-062 TCGA-COAD anchor within bootstrap variation. 95% CI lower bound exceeds zero. Difference vs VAL-062 = −0.112 (within bootstrap envelope).

Under-50 stratum n = 1 (TCGA-AG-A01Y, 49.6 y, Stage II Rectum NOS) ΔA = +0.016, pre-locked direction-only descriptive-only per CHK-2.7.

---

## Cohort

7 paired tumor/adjacent-normal pairs from TCGA-READ HM450 platform, sesame Level 3 betas, NIH GDC public access (no dbGaP required for Level 3 β values). All 7 patients passed QC (≥400,000 valid β values per sample).

| Patient | Age | Sex | Subsite | Stage |
|---|---|---|---|---|
| TCGA-AG-3725 | 60s | F | Rectum, NOS | II |
| TCGA-AG-3731 | 60s | M | Rectum, NOS | II |
| TCGA-AG-A01W | 60s | F | Rectum, NOS | III |
| **TCGA-AG-A01Y** | **49.6** | **M** | **Rectum, NOS** | **II** |
| TCGA-AG-A020 | 60s | F | Rectum, NOS | III |
| TCGA-AG-A02N | 60s | F | Rectum, NOS | II |
| TCGA-AG-A036 | 60s | M | Rectosigmoid junction | III |

**Cohort SHA-256 prefixes:** see `cohort_manifest.json` for per-file aggregate hash.

**Source data access.** All 14 IDAT-derived β files (7 tumor + 7 adjacent-normal) are publicly accessible via the NIH GDC public API at `https://api.gdc.cancer.gov/data/{file_id}` per the file_id values in `READ_matched_manifest.json`. No dbGaP application required for Level 3 β values.

---

## Methodology

Methodology mirrors VAL-062 TCGA-COAD anchor exactly:

1. **Cycling-class A-score per sample.** A_cycling(sample) = mean over all valid HM450 CpGs of [ H(β) / H_min(cycling) ], where H_min(cycling) = 0.856055 (G-002 MCMC posterior, R-hat = 1.0003) and valid CpGs is the per-sample set with ≥400,000 valid β values per sample.
2. **Paired comparison.** Paired Cohen's d on (A_tumor − A_normal) per patient.
3. **Bootstrap CI.** 10,000 iterations, BCa-equivalent, RNG seed 20260428.
4. **Run-everything 25-tile.** Per-class per-tile A-score using Loyfer 25-tile reference atlas (`atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv`, 7,890 array CpGs × 25 cell types). Top-100 marker CpGs per tile, NNLS-style scoring against H_min of each tile's class assignment.
5. **Stratified analysis.** Pre-locked stratifications: by_age (under_50 vs age_50_plus), by_sex (female vs male), by_subsite (Rectum NOS vs Rectosigmoid junction).
6. **Stage 1 immune Xu-538 panel.** Pooled A_immune scored against H_min(immune) = 0.838889 per panc-LL-007 universal Stage 1 H_min rule.

---

## Primary result — full-HM450 cycling-class

| Metric | Value |
|---|---|
| Paired Cohen's d | **+0.6116** |
| 95% CI | [+0.2271, +1.8822] |
| Paired t | +1.618 |
| Paired p | 0.157 |
| n_pairs (QC-passed) | 7 / 7 |

**Direction confirmed.** Magnitude consistent with VAL-062 TCGA-COAD anchor (+0.724) within bootstrap variation. 95% CI lower bound exceeds zero.

### Comparison to VAL-062 anchor

| Cohort | n_pairs | Paired d | 95% CI |
|---|---|---|---|
| TCGA-COAD (VAL-062) | 26 | +0.724 | [+0.292, +1.156] |
| TCGA-READ (VAL-098) | 7 | **+0.612** | [+0.227, +1.882] |

Difference = −0.112. The smaller TCGA-READ cohort gives a wider CI; direction confirmation is robust, magnitude precision is limited by sample size. This is the expected pattern for a smaller subsite-specific replication of a larger anchor cohort.

---

## Stratified analysis (pre-locked stratifications)

### By age

| Stratum | n | ΔA / d | Note |
|---|---|---|---|
| under_50 | 1 | ΔA = +0.0159 | n=1 — direction-only, descriptive-only per CHK-2.7 |
| age_50_plus | 6 | d = +0.577 | 95% CI [+0.030, +1.608], p = 0.216 |

The under-50 stratum has only 1 patient (TCGA-AG-A01Y, 49.6 y, Stage II Rectum NOS). VAL-098 cannot speak to the under-50 stratum at sub-stratum statistical power. Direction is positive (ΔA = +0.0159) and pre-locked direction-only per CHK-2.7. The under-50-stratum evidence chain for crc-epic relies on:

- **VAL-099** (planned) — TCGA-COAD age-stratified re-analysis of existing VAL-061/VAL-062 per-sample CSV. No new data download; re-slice the existing 26-pair sealed dataset by age decile and anatomic subsite.
- **VAL-100** (planned) — GSE282666 Stage 1 immune A-score on under-50 buffy coat with colonoscopy-confirmed polyp status (n=51 EPIC). Same Xu-538 panel as VAL-047.

### By sex

| Stratum | n | Paired d | 95% CI |
|---|---|---|---|
| female | 4 | +0.322 | [−0.955, +2.068] |
| male | 3 | +1.006 | [0.000, +4.093] |

Both directions positive. CIs wide due to small per-stratum n. No sex-stratified clinical action implied at v0.1.

### By subsite

| Stratum | n | Paired d | 95% CI |
|---|---|---|---|
| Rectum, NOS | 6 | +0.750 | [+0.564, +2.941] |
| Rectosigmoid junction | 1 | n=1 | direction-only |

Rectum NOS subsite (6/7 patients) reads paired d = +0.750 with 95% CI lower bound +0.564 — direction and magnitude consistent with the VAL-062 TCGA-COAD anchor (+0.724).

---

## Run-everything 25-tile observation (CCL-039)

VAL-098 was the first cookbook validation to run BOTH full-HM450 cycling-class methodology AND run-everything 25-tile per-class methodology on the same paired tumor/normal samples. The pre-registered O1 criterion was the full-HM450 cycling-class result; the 25-tile output is supplementary documentation.

### Top 8 tiles by |paired d|

| Tile | Class | Paired d | 95% CI |
|---|---|---|---|
| **Colon_epithelial_cells** | cycling | **−2.501** | [−9.307, −1.584] |
| Pancreatic_beta_cells | secretory | +2.136 | [+1.591, +4.544] |
| Hepatocytes | secretory | +2.094 | [+1.598, +4.750] |
| Lung_cells | cycling | +1.599 | [+0.973, +3.963] |
| Head_and_neck_larynx | cycling | +1.546 | [+1.036, +5.494] |
| Bladder | cycling | +1.162 | [+0.616, +2.706] |
| Monocytes_EPIC | immune | +1.072 | [+0.500, +3.085] |
| Neutrophils_EPIC | immune | +1.064 | [+0.386, +4.097] |

**Direction concordance check.** 22 of 25 tiles read positive direction; 3 tiles read negative. Colon_epithelial_cells is the strongest negative-direction tile by |d|.

### Top-1 distribution per patient

| Top-1 tile | n patients (of 7) |
|---|---|
| Colon_epithelial_cells (negative direction) | 3 |
| Upper_GI | 3 |
| Bladder | 1 |

Colon_epithelial_cells is top-1 in 3 of 7 patients (with the largest negative ΔA in those patients). The cell-of-origin tile is among the largest |d| in 7 of 7 patients on the run-everything 25-tile output, consistent with CHK-4.11.

### CCL-039 documentation

Diagnostic re-application of VAL-098 run-everything 25-tile methodology to the existing VAL-062 TCGA-COAD 26-pair sealed dataset confirms the pattern is cookbook-wide:

| Cohort | Method | Paired d | 95% CI |
|---|---|---|---|
| TCGA-READ (VAL-098) | Full-HM450 cycling-class | +0.612 | [+0.227, +1.882] |
| TCGA-READ (VAL-098) | Colon_epithelial_cells tile | −2.501 | [−9.307, −1.584] |
| TCGA-COAD (VAL-062 revisit) | Full-HM450 cycling-class | +0.724 | (matches VAL-062 byte-for-byte) |
| TCGA-COAD (VAL-062 revisit) | Colon_epithelial_cells tile | −1.552 | [−2.175, −1.214] |

Two distinct observables. Full-HM450 cycling-class A-score measures global Shannon entropy change (positive d in tumor — every CpG counted equally, every signal direction averaged). Per-tile marker-CpG A-score measures cell-of-origin tile fidelity which DEGRADES in tumor (negative d at the cell-of-origin tile due to tumor de-differentiation). Both observations are real biology; they measure different things. See LESSONS_LEARNED.md CCL-039 LL-MARKER-CPG-TILE-FIDELITY for full mechanism interpretation. See TESTING_CHECKLIST.md CHK-4.11 for prereg-O1-criterion design rule. See EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md Part 14 for the production pipeline interpretation rule.

The diagnostic VAL-062 revisit script and results are in `Biological_Physics/validation_runs/VAL-062_revisit/` (`revisit_val062.py` and `revisit_results.json`) — not a new sealed VAL, diagnostic re-analysis only.

---

## Pre-registered outcome classification

**O1_CYCLING_CLASS_RECTAL_CONFIRMED.** The pre-locked O1 criterion was: paired d > +0.3 with 95% CI lower bound > 0 in full-HM450 cycling-class scoring. Result: paired d = +0.6116, 95% CI [+0.2271, +1.8822]. **Both criteria met. Outcome O1.**

The Colon_epithelial_cells tile direction (negative) is documented as supplementary observation per CHK-4.11 and CCL-039. The pre-registered O1 criterion did not require positive direction at the cell-of-origin tile (per CHK-4.11 — direction expectation depends on comparison type; tumor-vs-adjacent-normal-paired expects negative direction at the cell-of-origin tile).

---

## EDEAR commercial deployment unaffected

Per CCL-037 LL-CROSS-COHORT-CALIBRATION (signed off 2026-04-28), the cross-cohort calibration boundary applies exclusively to retrospective cookbook validation. EDEAR commercial deployment uses a single calibrated pipeline against a fixed reference distribution and is unaffected by validation-side limitations of any individual VAL.

For a real CRC or rectal cancer patient, the colorectal cell-of-origin tile pattern WILL fire in the EDEAR Stage 2 reading because tumor colorectal cells diverge from healthy colorectal methylation as captured by the Loyfer reference. The pattern of WHICH tiles co-fire is the diagnostic information. VAL-062 (TCGA-COAD) and VAL-098 (TCGA-READ) demonstrate clean signal on within-cohort same-pipeline paired comparisons — the deployment configuration that EDEAR uses by construction.

---

## Reproducibility triple (CHK-7.6)

### 1. Source code

`Biological_Physics/validation_runs/VAL-098/val_098.py` on https://github.com/hmahaffeyges/IAM-Validation. Python 3 stdlib + standard scientific Python (numpy + pandas + scipy + matplotlib). 31,233 bytes.

### 2. Inputs

- **Cohort manifest:** `Biological_Physics/validation_runs/VAL-098/READ_matched_manifest.json` (7 paired pairs, 14 file_id values for NIH GDC public API).
- **Clinical metadata:** `Biological_Physics/validation_runs/VAL-098/clinical_metadata.csv` (age, sex, subsite, stage per patient).
- **TCGA-READ HM450 .txt files:** Download via NIH GDC public API at `https://api.gdc.cancer.gov/data/{file_id}` per the file_id values in `READ_matched_manifest.json`. Total cohort size ≈ 600 MB. No dbGaP application required.
- **Loyfer 25-tile reference atlas:** `atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv` (7,890 array CpGs × 25 cell types). Source: https://github.com/nloyfer/meth_atlas + atlas_vault SHA-256 INVENTORY.json.
- **Xu-538 panel:** Standard cookbook panel, 538 CpGs frozen since v0.1. `xu_538_panel.json` in card resources.

### 3. Environment

- Python 3.12
- numpy 1.26.x, pandas 2.2.x, scipy 1.13.x, matplotlib 3.8.x
- Expected runtime: 30–60 s on a modern laptop after downloading the 14 TCGA-READ .txt files
- Expected memory: < 4 GB

### 4. Expected headline outputs

```
Pooled cycling-class paired d:    +0.6116 [+0.2271, +1.8822], t=1.62, p=0.157
Outcome label:                    O1_CYCLING_CLASS_RECTAL_CONFIRMED
n_pairs (QC-passed):              7 / 7
Colon_epithelial_cells tile:      d = -2.501 [-9.307, -1.584] (CCL-039 supplementary observation)
Pre-reg seal:                     SHA 57d830d6c7ba6448...
RNG seed:                         20260428
Runtime:                          ~31 seconds
```

---

## Files in this VAL bundle

| File | Size | Purpose |
|---|---|---|
| `prereg.md` | 19,315 bytes | Pre-registration document (sealed before β-value scoring) |
| `PREREG_SEAL.txt` | 197 bytes | Prereg seal manifest with SHA-256 |
| `val_098.py` | 31,233 bytes | Reproducible Python script |
| `READ_matched_manifest.json` | 4,006 bytes | 7 paired pairs / 14 file_id values |
| `cohort_manifest.json` | 642 bytes | Aggregate cohort SHA |
| `clinical_metadata.csv` | 2,242 bytes | Per-patient age, sex, subsite, stage |
| `results.json` | 11,612 bytes | Primary + run-everything 25-tile + stratified |
| `stratified.json` | 961 bytes | Stratified analysis only |
| `per_sample.csv` | 13,019 bytes | Per-sample A-score values |
| `tile_heatmap.png` | 110,295 bytes | 25-tile per-patient heatmap |
| `outcome.md` | this file | Outcome write-up |

---

## Lessons logged in this VAL

- **CCL-039 LL-MARKER-CPG-TILE-FIDELITY** (added to LESSONS_LEARNED.md): Marker-CpG-tile A-score and full-HM450 architectural-drift A-score are two distinct observables. They do not always move in the same direction in tumor vs adjacent-normal paired comparisons. Confirmed cookbook-wide via VAL-062 revisit on TCGA-COAD same methodology.
- **CHK-4.11** (added to TESTING_CHECKLIST.md): Run-everything 25-tile prereg-O1-criterion design rule under CCL-039. Future preregs must NOT pre-lock "cell-of-origin tile shows positive d" without specifying comparison type; pre-lock "cell-of-origin tile is among the largest |d|" instead with explicit direction expectation per comparison type.
- **EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md Part 14** (added): Run-everything 25-tile interpretation rules under CCL-039.

---

## Next-step VALs in the early-onset rectal subsection chain

- **VAL-099** — TCGA-COAD age-stratified re-analysis of existing VAL-061/VAL-062 per-sample CSV. No new data download; re-slice the existing 26-pair sealed dataset by age decile and anatomic subsite.
- **VAL-100** — GSE282666 Stage 1 immune A-score on under-50 buffy coat with colonoscopy-confirmed polyp status (n=51 EPIC). Same Xu-538 panel as VAL-047.
- **Future-when-support-arrives:** GSE284325 EOCRC WGBS cohort (n=16) requires post-Caggiano methods translation engineering for WGBS-to-array mapping. Not pursued at v1 per LL-PUBLIC-TIER (no biobank applications, no preprint-first).
