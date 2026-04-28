# VAL-098 — Pre-registration

**Title:** TCGA-READ paired tumor-vs-adjacent-normal cycling-class architectural drift validation, with run-everything 25-tile per-class A-score and age-stratified reporting

**Card:** crc-epic (early-onset rectal subsection — Phase 2 anchor)
**Date sealed:** 2026-04-28 UTC, pre-β-access
**RNG seed:** 20260428
**Analyst:** Walther / Heath W. Mahaffey

---

## Operating context

This VAL operates under the IAMPerformance public-tier-only operational reset (LL-PUBLIC-TIER, signed off 2026-04-28). Cohort access is restricted to public Tier 1 GDC public-API data; biobank-gated cohorts are logged in `crc-epic/future_when_support_arrives.md` and not pursued. This VAL is the rectal-subsite extension of the existing crc-epic tissue arm (VAL-061 immune compartment + VAL-062 cycling architecture, both on TCGA-COAD); it adds a within-cohort paired comparison on TCGA-READ that satisfies CHK-3.8 condition 1 (within-cohort paired tumor-vs-adjacent-normal in a single cohort, same pipeline, same population). No cross-cohort calibration problem applies.

EDEAR commercial deployment is unaffected by validation-side cohort coverage gaps. EDEAR is a health-and-wellness early-detection tool, not a regulated diagnostic device. Public-tier validation strength is sufficient for the early-detection claim because EDEAR's deployment makes the patient-vs-internal-reference comparison apples-to-apples by construction.

---

## Background

VAL-062 (crc-epic tissue arm anchor, 2026-04-24) reported TCGA-COAD paired tumor-vs-adjacent-normal Cohen's d = +0.7241, 95% CI [+0.2922, +1.1559], p = 2.23e-04 on cycling-class A-score across 26 matched pairs. This was the first cookbook validation of cycling-class architectural drift in a tissue arm and the canonical anchor for crc-epic's `cycling_class_tissue_validated` tier.

The crc-epic v2.3 record does NOT yet have a tissue-arm validation on the rectal subsite specifically. TCGA-READ (Rectum Adenocarcinoma) is the rectal counterpart to TCGA-COAD and is publicly accessible via the NIH GDC public API at `https://api.gdc.cancer.gov/data/{file_id}`. Per Phase 1 cohort survey (2026-04-28), TCGA-READ contains 7 paired tumor/adjacent-normal pairs with HM450 methylation files. This is small but sufficient for direction-confirmation extension of VAL-062's cycling-class signal to the rectal anatomic subsite.

VAL-098 asks: does the cycling-class architectural drift signal documented in VAL-062 on TCGA-COAD extend to TCGA-READ on the same paired-tumor-vs-adjacent-normal methodology, and does the run-everything 25-tile per-class A-score on the Loyfer atlas surface any rectal-specific pattern relative to the established colon-cancer tile pattern?

The atlas vault has zero rectum-specific cell types across every reference matrix surveyed (Loyfer 25-tile, EpiSCORE ColonRef, Caggiano CelFiE TIM, MARLIN, Sabedot GeLB, all Stage 3 immune atlases). Loyfer's `Colon_epithelial_cells` tile is the production cell-of-origin reference for both colon and rectal adenocarcinoma at v1. Any tile-level pattern divergence between rectum and colon at the methylation level is therefore not directly testable; what IS testable is whether rectal tumor architecture shows the same cycling-class drift magnitude and direction as colon tumor architecture.

---

## Hypotheses

**H_A — direction-confirmed cycling-class signal at the rectal subsite.** Paired Cohen's d > 0 with 95% CI lower bound > 0, d ≥ +0.5, magnitude consistent with or comparable to VAL-062 TCGA-COAD result (+0.724). This would extend the cycling-class tissue-arm finding to the rectal anatomic subsite and support the early-onset rectal subsection's biology layer at the architectural drift level.

**H_B — direction-confirmed but magnitude divergent from colon.** Paired d > 0 with 95% CI lower bound > 0, but |d − d_VAL062| > 0.5. This would suggest rectal-vs-colon biology distinction at the architectural drift level even without atlas-resolution support for cell-of-origin tile separation.

**H_C — direction-weak or null.** 0 < d < +0.5 OR 95% CI crosses zero. Underpowered cohort (n=7 paired pairs) limits the inference; direction would still be reported but no tier promotion claimed.

**H_D — direction-inverted.** Paired d < 0. Framework inconsistency requiring immediate investigation. Unlikely given VAL-062 anchor and cycling-class biology.

**H_E — Stage 2 run-everything tile pattern reveals rectum-specific signal NOT captured by Loyfer Colon_epithelial_cells tile.** Per CCL-033 run-everything architecture, all 25 Loyfer tiles are scored on every sample. If a non-colon tile (e.g. Upper_GI, Bladder, or any other cycling-class tile) shows comparable or larger |d| than Colon_epithelial_cells in the rectal cohort, this would be unexpected and worth documenting as a tile-pattern observation (descriptive only — atlas resolution does not support rectum-specific cell-of-origin biology claims).

**H_F — age-stratum signal divergence (descriptive only).** Per CHK-2.7, the under-50 stratum in TCGA-READ is structurally underpowered (n=1 paired pair at age 49.6). The under-50 stratum reading is reported as direction-only, descriptive-only, NOT interpretable for early-onset claims at this cohort size.

---

## Method

### Cohort

**Primary cohort — TCGA-READ paired tumor/adjacent-normal:**
- 7 patients with both Primary Tumor AND Solid Tissue Normal HM450 methylation files
- Platform: Illumina HumanMethylation450 (HM450, GPL13534), sesame level3 betas
- Source: NIH GDC public API at `https://api.gdc.cancer.gov/data/{file_id}`
- Manifest: `READ_matched_manifest.json` constructed from GDC API query 2026-04-28 (sealed in this VAL's working directory; SHA-256 recorded in PREREG_SEAL.txt)
- Per-patient breakdown (from GDC API metadata):

| Patient | Age (yr) | Anatomic site | Sex | Stage |
|---|---|---|---|---|
| TCGA-AG-3725 | 90.0 | Rectosigmoid junction | F | III |
| TCGA-AG-3731 | 65.5 | Rectum, NOS | M | IV |
| TCGA-AG-A01W | 67.7 | Rectum, NOS | F | II |
| TCGA-AG-A01Y | 49.6 | Rectum, NOS | F | II |
| TCGA-AG-A020 | 57.1 | Rectum, NOS | F | III |
| TCGA-AG-A02N | 67.6 | Rectum, NOS | M | II |
| TCGA-AG-A036 | 71.7 | Rectum, NOS | M | III |

**Anatomic subsite distribution among paired pairs:** Rectum NOS = 6, Rectosigmoid junction = 1. Predominantly clean-rectum cohort.

**Age stratification:**
- Under-50 stratum: n=1 (TCGA-AG-A01Y, age 49.6, Stage II rectum NOS) — **structurally underpowered, pre-locked direction-only, descriptive-only per CHK-2.7**
- 50+ stratum: n=6 — pooled paired-d primary reading
- Pooled all-ages: n=7 — secondary reading for direct comparison to VAL-062's pooled n=26

### Reference

**Cycling-class H_min:** 0.856055 (frozen from G-002 + G-003b MCMC posteriors, R-hat < 1.001; byte-match GAPE_WEB_v13 `_H_MIN_GRID["cycling"]["methyl"]` and VAL-062 prereg constant)

**Reference β for cycling-class calibration:** 0.740 (inherited from VAL-062, TCGA-COAD matched-normal calibration)

### Run-everything 25-tile Stage 2 architecture (per CCL-033)

Per CCL-033 run-everything architecture (signed off 2026-04-26): every sample is scored against ALL 25 Loyfer cell types in addition to the primary cycling-class scoring. Per-class A-score = mean(H(β)/H_min(class)) across top-100 marker CpGs per tile, where H_min is the architecture-class H_min for that cell type:

| Loyfer cell type | Architecture class | H_min |
|---|---|---|
| Cortical_neurons | terminal | 0.7728 |
| Left_atrium | terminal | 0.7728 |
| Hepatocytes | secretory | 0.843264 |
| Breast | secretory | 0.843264 |
| Prostate | secretory | 0.843264 |
| Pancreatic_acinar_cells | secretory | 0.843264 |
| Pancreatic_duct_cells | secretory | 0.843264 |
| Pancreatic_beta_cells | secretory | 0.843264 |
| Thyroid | secretory | 0.843264 |
| Bladder | cycling | 0.856055 |
| Colon_epithelial_cells | cycling | 0.856055 |
| Lung_cells | cycling | 0.856055 |
| Head_and_neck_larynx | cycling | 0.856055 |
| Upper_GI | cycling | 0.856055 |
| Uterus_cervix | cycling | 0.856055 |
| Kidney | cycling | 0.856055 |
| Adipocytes | stromal | 0.862950 |
| Vascular_endothelial_cells | stromal | 0.862950 |
| Erythrocyte_progenitors | progenitor | 0.852216 |
| Monocytes_EPIC | immune | 0.838889 |
| B-cells_EPIC | immune | 0.838889 |
| CD4T-cells_EPIC | immune | 0.838889 |
| NK-cells_EPIC | immune | 0.838889 |
| CD8T-cells_EPIC | immune | 0.838889 |
| Neutrophils_EPIC | immune | 0.838889 |

Reference atlas: Loyfer 2023 array atlas at `Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv` (7,890 array CpGs × 25 cell types).

### Primary test (mirrors VAL-062)

**Cycling-class paired Cohen's d on all valid HM450 CpGs.** For each patient, compute A_cycling = mean(H(β)/0.856055) across all valid HM450 CpGs in tumor and in adjacent-normal sample. Compute paired Cohen's d on (A_tumor − A_adjacent_normal) per patient. Bootstrap 10000-iteration paired-d 95% CI. RNG seed 20260428.

**QC threshold:** minimum 400,000 valid β values per sample (HM450 has ~485K total CpGs; 400K = ~82% coverage threshold per VAL-062 standard). Pairs failing QC are dropped before paired-d computation.

### Secondary tests (run-everything Stage 2)

**Per-tile paired Cohen's d on Loyfer 25-tile.** For each Loyfer cell type, compute paired d on (A_tumor − A_adjacent_normal) using the top-100 marker CpGs identified per tile. Bootstrap CI per tile. Report the full 25-tile ranked d list.

**Top-1 ΔA call per patient.** For each patient, identify the tile with the largest |A_tumor − A_adjacent_normal|. Report the distribution of top-1 calls across the 7 patient pairs. Expected pattern (if H_A holds): Colon_epithelial_cells dominates as top-1 in majority of patients; immune tiles (B-cells_EPIC, CD4T-cells_EPIC, etc.) may also surface due to tumor-infiltrating-lymphocyte signal in the tumor side.

### Stratification block (CHK-2.7 mandatory)

| Stratum | n | Pre-locked status |
|---|---|---|
| Pooled all-ages | 7 | Primary reading |
| Age 50+ | 6 | Primary reading (sufficient n for direction confirmation) |
| Age under 50 | 1 | **Direction-only, descriptive-only, NOT interpretable per CHK-2.7 underpower floor** |
| Sex (Female) | 4 | Secondary |
| Sex (Male) | 3 | Secondary |
| Stage II | 3 | Secondary, descriptive |
| Stage III | 3 | Secondary, descriptive |
| Stage IV | 1 | Direction-only |
| Anatomic subsite (Rectum NOS) | 6 | Primary |
| Anatomic subsite (Rectosigmoid junction) | 1 | Direction-only |

Per CHK-2.7 honest underpower disclosure: any stratum with n<5 is reported with d + 95% CI but not interpreted. The headline d is the pooled all-ages all-strata reading.

### Cross-cohort baseline check (CHK-3.2 — sanity only, not blocker)

This VAL is within-cohort paired (CHK-3.8 condition 1 satisfied). Cross-cohort baseline check is NOT structurally required — the comparison is paired patient-against-self-tissue. CHK-3.2 is run as sanity check on the run-everything Stage 2 layer comparing VAL-098 healthy (adjacent-normal) baseline against VAL-062 healthy (TCGA-COAD adjacent-normal) baseline, both HM450 sesame, both TCGA pipeline. Any tile delta >1 anchor-SD between the two healthy baselines is flagged but does NOT trigger O5_BASELINE_DOMINATED — the primary paired-within-patient comparison is unaffected.

### Substrate scope (CHK-3.7 absolute)

v1 methyl-only HM450 reading. L1 deployment scope. Any comparison to Issue 002 multi-substrate framework predictions states the L3-scope translation explicitly: direction transfers, magnitude does not.

### Data integrity (CHK-3.6)

- β distribution health check on all 14 samples (7 tumor + 7 adjacent-normal): raw β > 30% extremes (< 0.2 OR > 0.8) and < 30% in [0.4, 0.6]; flat distributions flagged as residuals not raw, abort if detected. (Tissue β distributions typically show larger middle-range fraction than blood; threshold relaxed accordingly per VAL-062 standard.)
- SHA-256 of input files recorded in `outcome.md`.
- File count check: 7 paired pairs = 14 files downloaded. Expected total download size ~280 MB (~20 MB per HM450 sesame level3 .txt file).
- HM450 CpG mapping: full HM450 array (~485K CpGs) is the input; intersection with Loyfer atlas (7,890 array CpGs) is the Stage 2 marker set.

---

## Pre-locked decision criteria

| Outcome | Primary cycling-class paired-d criterion | Run-everything 25-tile criterion | Hypothesis |
|---|---|---|---|
| O1_CYCLING_CLASS_RECTAL_CONFIRMED | Paired d ≥ +0.5, 95% CI lower bound > 0 | Colon_epithelial_cells tile is largest |d| OR among top-3 tiles by |d|; majority of top-1 ΔA calls land on cycling-class tiles | H_A |
| O2_DIRECTION_DIVERGENT_FROM_COLON | Paired d > 0, 95% CI > 0, but \|d − 0.724\| > 0.5 | — | H_B |
| O3_RECTAL_DIRECTION_WEAK | 0 < d < +0.5 OR 95% CI crosses zero | — | H_C |
| O4_DIRECTION_INVERTED | Paired d < 0 | — | H_D |
| O5_BASELINE_DOMINATED | Does NOT apply structurally — within-cohort paired comparison; placeholder per CHK-4.10 | — | — |
| O6_DATA_INTEGRITY | β distribution health check failure, manifest mismatch, file SHA mismatch, HM450 coverage < 400K CpGs in any sample, atlas marker coverage < 80% per tile | — | — |
| O7_TILE_PATTERN_UNEXPECTED | (Descriptive only) | A non-cycling tile shows |d| larger than Colon_epithelial_cells AND larger than all other cycling tiles | H_E |

**CHK-4.10 pattern-based baseline-dominated check (mandatory placeholder):** Per CCL-038, the runtime check applies: ≥3 tiles >3 anchor-SD on CHK-3.2 simultaneously AND ≥80% same-direction = baseline-dominated. **Within-cohort paired comparison structurally avoids this; criterion is logged as placeholder for the Stage 2 sanity-check layer only.**

**CHK-2.7 underpower disclosure (mandatory):** Outcome assignment is based on the pooled all-ages reading (n=7) and the 50+ stratum (n=6). The under-50 stratum (n=1) is direction-only and descriptive-only; it does NOT contribute to outcome assignment. The early-onset rectal subsection v0.1 will reference this VAL's pooled rectal cycling-class signal as the rectal-subsite anchor; it will NOT cite VAL-098 as evidence for under-50-stratum-specific biology. The under-50 stratum's evidence comes from VAL-099 (TCGA-COAD age-stratified re-analysis, 26-pair cohort with several under-50 patients) and VAL-100 (GSE282666 Stage 1 immune blood, n=51 all under-50).

---

## Outputs

1. **`val_098.py`** — full source code with frozen constants, atlas paths, RNG seed, decision-criteria implementation. Mirrors VAL-062 paired-d structure plus VAL-093 run-everything 25-tile structure.
2. **`VAL-098_results.json`** — primary paired-d, bootstrap CI, p-value, per-tile run-everything d table, top-1 distribution, age-stratified results, anatomic-subsite-stratified results.
3. **`VAL-098_stratified.json`** — per-stratum (age × sex × stage × subsite) per-tile A-score statistics.
4. **`VAL-098_per_sample.csv`** — per-patient: TCGA submitter ID, tumor A_cycling, normal A_cycling, paired ΔA, age, sex, stage, anatomic subsite, top-1 tile.
5. **`VAL-098_tile_heatmap.png`** — 25-tile paired-d visualization, tumor-vs-adjacent-normal in TCGA-READ.
6. **`VAL-098_outcome.md`** — outcome interpretation per CHK-4.9 12-section template.
7. **`VAL-098_cohort_manifest.json` / READ_matched_manifest.json` — 7 paired pairs with file IDs, file sizes, clinical metadata.
8. **`VAL-098_clinical_metadata.csv`** — per-sample clinical metadata extracted from GDC API.
9. **`VAL-098_PREREG_SEAL.txt`** — SHA-256 of this prereg.md file.

---

## Caveats declared in advance

- **Specimen pathway (CHK-0.5):** TCGA-READ is bulk rectal tissue (HM450, sesame level3). Same specimen pathway as VAL-062 TCGA-COAD. No cross-substrate caveat per CCL-010.
- **Sample size n=7 paired pairs.** Smaller than VAL-062's n=26 by a factor of ~4. Bootstrap 95% CI will be wider; direction confirmation is the realistic primary goal, not magnitude precision.
- **Under-50 stratum n=1.** Pre-locked direction-only, descriptive-only. The early-onset rectal subsection's biology layer relies on VAL-099 + VAL-100 for under-50-stratum evidence, NOT on VAL-098.
- **Anatomic subsite asymmetry:** 6 Rectum NOS + 1 Rectosigmoid junction. Per CHK-2.7, the rectosigmoid case is direction-only.
- **Stage distribution:** 3 Stage II + 3 Stage III + 1 Stage IV. Stage IV n=1 is direction-only.
- **Sex stratification:** 4 Female + 3 Male. Per CCL-002, sex stratification is mandatory and will be reported even though both strata are below the n=5 interpretation floor.
- **Atlas resolution constraint:** Loyfer atlas has Colon_epithelial_cells but no Rectum_epithelial_cells. Per Phase 1 cohort survey atlas vault check, zero rectum-distinct cell types exist in any reference matrix in the vault. Cell-of-origin tile separation between rectum and colon is not testable at v1; the run-everything 25-tile reading documents the rectal cohort's tile pattern against the existing colon tile, not against a rectum-specific tile.
- **No new prereg structure invented.** This prereg mirrors VAL-062 (paired tissue methodology) and VAL-093 (run-everything 25-tile architecture). No methodology innovation is claimed.
- **EDEAR commercial deployment unaffected.** Per CCL-037: cookbook validation cohort coverage gaps and underpowered strata do not affect EDEAR's single-pipeline patient-vs-internal-reference deployment. The crc-epic early-onset rectal subsection uses VAL-098's rectal-subsite tile pattern as part of the evidence chain for the clinical-action routing layer, not as direct deployment calibration data.

---

## Reproducibility triple (CHK-7.6 absolute)

To be filled in `val_098.py` and `VAL-098_outcome.md`:

1. **Source code:** `val_098.py` inline + GitHub URL `Biological_Physics/validation_runs/VAL-098/`.
2. **Inputs:** `READ_matched_manifest.json` (7 patient pairs with GDC file_ids); GDC public API endpoint `https://api.gdc.cancer.gov/data/{file_id}` for sesame level3 .txt files; Loyfer reference atlas at `Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv` (SHA-256 prefix from atlas vault INVENTORY.json).
3. **Environment:** Python 3, NumPy, Pandas, SciPy, Matplotlib (Agg backend); standard scientific Python; ~30 seconds runtime estimated; ~50 MB peak memory.
4. **Expected headline output:** Pooled cycling-class paired d, bootstrap 95% CI, p-value; per-tile run-everything d table; top-1 distribution; outcome label.

---

## Stratification underpower disclosure (CHK-2.7 — explicit)

Pre-locked: under-50 stratum n=1 is reported with point estimate but explicitly marked as **direction-only, descriptive-only, NOT interpretable for early-onset claims at this cohort size**. The pooled all-ages headline d is the primary outcome assignment basis. The under-50 stratum's value to the early-onset rectal subsection is documenting that the Tier 1 public anatomic-rectal cohort does not have sufficient under-50 cases for stratum-level inference, justifying the cookbook's reliance on VAL-099 (TCGA-COAD, 295 cases including 62 in 30-49 range) and VAL-100 (GSE282666, n=51 all under-50) for the under-50-stratum evidence chain.

---

**SEAL:** 2026-04-28 UTC. SHA-256 of this file recorded in `PREREG_SEAL.txt`.
