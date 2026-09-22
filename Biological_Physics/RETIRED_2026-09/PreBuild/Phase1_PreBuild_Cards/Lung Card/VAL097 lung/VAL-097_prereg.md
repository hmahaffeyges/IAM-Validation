# VAL-097 — Pre-registration

**Never-smoker LUAD tissue 25-tile per-class A-score characterization on GSE256092 with cross-cohort baseline against TCGA-LUAD adjacent-normal**

**Date sealed:** 2026-04-28
**RNG seed:** 20260428
**Run-everything architecture:** This VAL is designed under CCL-033 — every IDAT is processed against ALL Stage 2 tissue tiles in the Loyfer 25-cell array atlas regardless of cohort anchor.
**Operating context:** This VAL operates under the IAMPerformance public-tier-only operational reset (LL-PUBLIC-TIER, signed off 2026-04-28). Cohort access is restricted to public Tier 1 GEO data; biobank-gated cohorts are logged in `lung-epic/future_when_support_arrives.md` and not pursued. Data-availability gaps in metadata (e.g. smoking strata, driver mutations) are pre-locked as honest CHK-2.7 caveats, not blockers. EDEAR is a health-and-wellness early-detection tool; this VAL contributes to the lung-epic card as a never-smoker LUAD tissue tile-pattern characterization, not a regulated diagnostic claim.

---

## Background

VAL-063 (lung-epic anchor) reported TCGA-LUAD paired tumor-vs-adjacent-normal Cohen's d = +1.020 on the cycling-class A-score, ever-smoker stratum (n=22). The lifelong non-smoker stratum within VAL-063 was n=2 — structurally underpowered. The never-smoker LUAD pattern at the lung-cell-of-origin tile is therefore not characterized in the existing cookbook record.

GSE256092 (Korean Cancer Genome Atlas Consortium, 2024, n=141 NSLA tissue, EPIC, deposited 2024-03-11) is a single-stratum cohort: every sample is "Never-smoker lung adenocarcinoma" by cohort definition. Per-sample metadata in the series matrix carries stage (I–IV), age (range 37–85), and gender (M+F). The cohort is enriched for never-smoker LUAD without EGFR or ALK alterations (NENA, the publication's primary focus). Driver mutations (TP53, KRAS, STK11, ERBB2, ROS1 fusion) are reported in the publication abstract and present in the supplementary clinical metadata; they are not surfaced in the series matrix headers and are not used in this VAL's primary analysis (driver-stratified analysis is deferred to a follow-up VAL).

GSE256092 has no internal healthy-tissue controls. Per CCL-034, within-cohort statistics are primary; per CCL-010, the cross-substrate caveat does not apply because both reference cohorts are tissue. The cross-cohort healthy comparator is TCGA-LUAD adjacent-normal lung tissue (HM450, n=29 per VAL-063), which differs from GSE256092 on three axes simultaneously: ethnicity (Korean vs Western), smoking status (all never-smoker vs majority ever-smoker), and platform (EPIC 850K vs HM450). The cross-cohort baseline check (CHK-3.2) is expected to flag at >1 anchor-SD on at least one tile because of these structural differences. Per the operational stance applied here, the CHK-3.2 breach is pre-locked as expected and reported as a feature of the comparison — the cross-cohort baseline difference IS part of what we are characterizing — not as a data-integrity flag.

VAL-097 asks: at the per-tile level on the Loyfer 25-cell array atlas, what does the never-smoker LUAD tissue tile pattern look like, both within-cohort across stage/sex/age strata and against the TCGA-LUAD adjacent-normal cross-cohort reference?

---

## Hypotheses

**H_A — lung-localized signal.** The Lung_cells tile (cycling class, H_min 0.856055) shows the strongest tile A-score elevation in GSE256092 vs TCGA-LUAD adjacent-normal cross-cohort reference. Top-1 ΔA call at the patient level returns Lung_cells as the most-departed tile in the majority of GSE256092 cases. Pattern is consistent with VAL-063 ever-smoker direction (positive d on cycling class) at smaller magnitude given the never-smoker driver biology.

**H_B — cycling-class-distributed signal.** Multiple cycling-class tiles (Lung_cells, Bladder, Colon_epithelial_cells, Head_and_neck_larynx, Upper_GI, Uterus_cervix, Kidney) show comparable A-score elevations. The signal is class-localized but not tissue-localized. Top-1 ΔA calls distribute across cycling-class tiles.

**H_C — non-cycling tissue dominance.** A non-cycling tile (e.g. secretory class, immune class) shows the strongest |d|. This would indicate that never-smoker LUAD biology is dominated by a different cellular signature than smoker-driven LUAD — consistent with CCL-025 chronic disease-driver field defects: never-smoker LUAD is mechanistically distinct from smoker-driven LUAD.

**H_D — direction inversion.** The Lung_cells tile or cycling class shows negative d (lower A-score in NSLA than TCGA adjacent-normal). This would indicate either (a) a never-smoker-specific direction inversion, (b) the cross-cohort baseline difference dominating the case-vs-reference contrast, or (c) homogenization rather than departure (CCL-019 panel choice vs class choice phenomenon).

**H_E — cross-cohort baseline failure.** Cross-cohort baseline check (CHK-3.2) flags ≥3 tiles at >1 anchor-SD healthy-vs-healthy difference, magnitude exceeds the case-vs-reference signal, and the within-cohort variance structure does not support a tile-pattern reading. Result is recorded as descriptive cohort characterization only with explicit cross-cohort baseline incompatibility statement.

---

## Method

### Cohorts

**Primary cohort — GSE256092 (Korean NSLA, EPIC tissue, all never-smoker):**
- n=141 NSLA tissue samples, EPIC 850K (GPL21145)
- Sample IDs: GSM8085657 – GSM8085797
- Per-sample metadata from series matrix: Stage (I, II, III, IV), age (range 37–85), gender (M+F), disease (constant: Never-smoker lung adenocarcinoma)
- IDATs at `ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256092/suppl/GSE256092_RAW.tar`
- SWAN-normalized β matrix at `ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256092/suppl/GSE256092_SWAN.txt.gz`
- Series matrix at `ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256092/matrix/GSE256092_series_matrix.txt.gz`

**Cross-cohort healthy reference — TCGA-LUAD adjacent-normal (per VAL-063):**
- n=29 paired adjacent-normal lung tissue samples
- Platform: HM450 (GPL13534)
- Source: GDC TCGA-LUAD project, β values per VAL-063 input chain
- Used as healthy reference for cross-cohort baseline check and case-vs-reference contrast

### Reference atlas

Loyfer 2023 array atlas (`reference_atlas.csv`), 25 cell types, 7,890 array-indexed CpGs. Same atlas used in VAL-093 / VAL-094 / VAL-095 / VAL-096 — frozen reference, no modifications.

### Per-tile per-class A-score

For each cell type in the Loyfer atlas:
1. Identify top-100 discriminating CpGs maximizing `|β(target_cell) − mean(β(other_24_cells))|`.
2. For each patient β vector at those marker CpGs, compute `A_class = mean(H(β) / H_min(class))` where H is binary Shannon entropy and H_min is the architecture-class H_min from frozen G-002 + G-003b MCMC posteriors:

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

H_min values byte-match GAPE_WEB_v13.py `_H_MIN_GRID` (G-002 + G-003b MCMC posteriors, R-hat < 1.001).

### Stratification block (CHK-2.7 mandatory)

Per CHK-2.7, the prereg pre-locks the stratification structure. GSE256092 is a single-stratum cohort on smoking (all never-smoker — explicit cohort-definition caveat) and on ethnicity (all Korean — explicit single-population caveat). Stratification on within-cohort variables is therefore done on stage, sex, and age:

| Stratification axis | Levels | n per level (from series matrix) | Mandatory per CCL |
|---|---|---|---|
| Smoking | All never-smoker | 141 / 0 / 0 / 0 | CCL-009 absolute — single-stratum pre-locked caveat |
| Sex | Female, Male | F-dominant per series matrix; counted at runtime | CCL-002 absolute |
| Age | Decade bins (30s, 40s, 50s, 60s, 70s, 80s) | counted at runtime, range 37–85 | VAL-052 R²=26% age regression anchor; CCL applies |
| Stage | I, II, III, IV | counted at runtime | Disease-specific covariate per CCL-006 |
| Driver mutation | Not analyzed in primary | — | Deferred to follow-up VAL per "primary unstratified, follow-up stratified" decision |
| Ethnicity | All Korean | 141 | Single-population pre-locked caveat |

Per CHK-2.7 honest underpower language: any sub-stratum with n<5 is reported with d + 95% CI but not interpreted; the headline d is the all-cohort within-stratum aggregate.

### Cross-cohort baseline check (CHK-3.2 mandatory)

GSE256092 healthy-tissue reference is not internal — TCGA-LUAD adjacent-normal is used as the cross-cohort healthy comparator. Cross-cohort baseline alignment is computed per-tile:

`baseline_delta_tile = (A_GSE256092_mean − A_TCGA_LUAD_normal_mean) / anchor_SD`

where `anchor_SD` is the pooled within-tile standard deviation across both cohorts.

**Pre-locked expectation:** ≥1 tile breaches |baseline_delta| > 1 anchor-SD. The breach is expected because of structural cohort differences (ethnicity Korean vs Western, platform EPIC vs HM450, never-smoker enrichment vs smoker-enriched). The breach is documented as a feature of the comparison, not a data-integrity flag. The breach magnitude per tile is reported in the outcome file; its presence does not automatically trigger O6_UNEXPECTED unless the breach is >3 anchor-SDs simultaneously across ≥3 tiles AND drives the case-vs-reference signal (in which case the within-cohort variance reading takes primacy per CCL-034).

### Within-cohort vs cross-cohort hierarchy (CCL-034)

Primary reading: within-cohort stratum-by-stratum tile pattern in GSE256092. Reports A-score distributions across the 25 Loyfer tiles for the full cohort, and stratified by sex / age decade / stage. Variance structure within GSE256092 is the primary signal.

Secondary reading: case-vs-reference Cohen's d on each tile, GSE256092 vs TCGA-LUAD adjacent-normal. Reported with bootstrap 95% CI (n_boot=10000). Baseline-corrected d (per-tile baseline subtracted from raw d) reported alongside raw d.

### Top-1 ΔA call per patient

For each GSE256092 patient, identify the tile with the largest |A_patient − A_TCGA_normal_mean|. Report the distribution of top-1 calls across the 141 cases.

### Substrate scope (CHK-3.7 absolute)

v1 methyl-only EPIC reading. L1 deployment scope (lab partnership, EPIC β-matrix). Any comparison to Issue 002 multi-substrate framework predictions states the L3-scope translation explicitly: direction transfers, magnitude does not.

### Data integrity (CHK-3.6)

Per VAL-063 ad VAL-093 patterns:
- β distribution check: raw β > 30% extremes < 10% in [0.4, 0.6]; flat distributions flagged as residuals not raw, abort if detected.
- SHA-256 of input files recorded in outcome.md.
- Sample count check: 141 IDATs in GSE256092_RAW.tar must match 141 GSM IDs in series matrix.
- Cross-platform CpG mapping: EPIC → HM450 intersection used for cross-cohort comparison. Loyfer atlas CpGs are array-indexed for both platforms; intersection size reported in results.

---

## Pre-locked decision criteria

| Outcome | Within-cohort + cross-cohort criteria | Hypothesis |
|---|---|---|
| O1_LUNG_LOCALIZED | Lung_cells tile case-vs-reference d ≥ +0.5; |d| on Lung_cells is largest absolute among 25 tiles; majority (>50%) of GSE256092 patients have Lung_cells as top-1 ΔA call | H_A |
| O2_CYCLING_DISTRIBUTED | ≥3 cycling-class tiles show case-vs-reference |d| ≥ +0.3, with no single tile uniquely largest; top-1 calls distributed across cycling tiles | H_B |
| O3_NON_CYCLING_DOMINANT | A non-cycling tile shows |d| ≥ +0.5, magnitude greater than Lung_cells | H_C |
| O4_DIRECTION_INVERTED | Lung_cells tile shows d ≤ −0.3, OR ≥3 cycling-class tiles show negative d | H_D |
| O5_BASELINE_DOMINATED | Cross-cohort baseline check flags ≥3 tiles at >3 anchor-SDs simultaneously, AND case-vs-reference d magnitudes are smaller than the baseline deltas, AND within-cohort stratification does not surface a clear tile pattern | H_E |
| O6_DATA_INTEGRITY | β distribution check fails, IDAT count mismatch, SHA-256 verification fails, atlas CpG intersection < 80% of Loyfer 25-tile marker CpGs | revisit data integrity stage |

**CHK-2.7 honest underpower note:** outcome assignment requires that the headline d be supported by within-cohort variance structure (the d is consistent across age decade bins and across stage strata, no single sub-stratum drives the headline). If the headline d is driven entirely by one stratum (e.g. only Stage IV, or only women >70), the result is recorded with explicit sub-stratum localization and the outcome label is qualified.

---

## Outputs

1. **`val_097.py`** — full source code with frozen constants, atlas paths, RNG seed, decision-criteria implementation.
2. **`VAL-097_results.json`** — per-tile within-cohort distributions, cross-cohort baseline deltas, case-vs-reference d with bootstrap CI, top-1 distribution, stratified results across sex / age / stage.
3. **`VAL-097_stratified.json`** — per-stratum (sex × age decade × stage) per-tile A-score statistics.
4. **`VAL-097_per_sample.csv`** — per-patient: GSM ID, sex, age, stage, A-score per tile, top-1 call.
5. **`VAL-097_tile_heatmap.png`** — 25-tile by case heatmap visualization, GSE256092 vs TCGA-LUAD-normal reference.
6. **`VAL-097_outcome.md`** — outcome interpretation per CHK-4.9 run-everything 12-section template.
7. **`VAL-097_cohort_manifest.json`** — cohort manifest (GSM IDs, IDAT URLs, sizes, SHA-256).
8. **`VAL-097_clinical_metadata.csv`** — per-sample metadata extracted from series matrix.
9. **`VAL-097_PREREG_SEAL.txt`** — SHA-256 of this prereg.md file.

---

## Caveats declared in advance

- **Specimen pathway (CHK-0.5):** GSE256092 is bulk lung tissue. TCGA-LUAD adjacent-normal is also bulk lung tissue. Specimen pathway matches; cross-substrate caveat per CCL-010 does not apply.
- **Platform asymmetry:** GSE256092 is EPIC 850K; TCGA-LUAD is HM450. Cross-cohort comparison uses platform intersection at the Loyfer atlas CpG set. Platform difference is documented as a known confounder; cross-platform validation per VAL-052 anchor R²=26% applies.
- **Ethnicity asymmetry:** GSE256092 is Korean; TCGA-LUAD is mostly Western. This is the second structural confounder driving the expected CHK-3.2 baseline breach. Single-population caveat is pre-locked.
- **Smoking asymmetry:** GSE256092 is all never-smoker; TCGA-LUAD is majority ever-smoker. Per CCL-009 absolute and CCL-025 (chronic disease-driver field defects: never-smoker LUAD vs ever-smoker LUAD is mechanistically distinct), smoking-asymmetric cross-cohort comparison is structurally limited. Pre-locked caveat: the case-vs-reference d combines a within-disease-subtype contrast (never-smoker LUAD vs healthy lung tissue) AND a cross-subtype confound (never-smoker driver biology vs ever-smoker driver biology). The CCL-019 panel-choice caution applies: the Lung_cells tile reading does not isolate the disease signal from the driver-subtype confound.
- **Sample size n=141 cases / n=29 cross-cohort reference.** Cross-cohort reference n is the limiting factor for baseline-corrected d precision. Bootstrap 95% CI reported on every headline d.
- **Driver mutations not used in primary analysis.** Per the "primary unstratified, follow-up stratified" decision: TP53 / KRAS / STK11 / ERBB2 / ROS1 status from the GSE256092 supplementary clinical metadata is logged in the cohort manifest but not used to stratify the primary analysis. Driver-stratified follow-up VAL is recorded as `lung-epic/future_VAL_when_driver_metadata_processed.md` but not blocking VAL-097 closure.
- **No internal healthy controls in GSE256092.** Within-cohort case-vs-control is structurally not available. Within-cohort signal is reported as stratification (sex × age × stage variance structure) only.
- **CHK-3.2 baseline breach pre-locked as expected.** Per the operational stance applied here, the cross-cohort baseline delta is part of what is being characterized, not a data-integrity flag, unless the breach is severe (>3 anchor-SDs across ≥3 tiles AND dominates the case-vs-reference signal).
- **Heath sign-off recorded:** the operational stance — public Tier 1 only, biobank-gated logged, smoking-NA pre-locked, document everything, EDEAR is health-and-wellness early-detection — was confirmed 2026-04-28 in the IAMPerformance operational reset (LL-PUBLIC-TIER, to be added to LESSONS_LEARNED.md as a new CCL after VAL-097 closure).

---

## Reproducibility triple (CHK-7.6 absolute)

To be filled in `val_097.py` and `VAL-097_outcome.md`:

1. **Source code:** `val_097.py` inline + GitHub URL.
2. **Inputs:** GSE256092_RAW.tar URL + size + SHA-256; GSE256092_SWAN.txt.gz URL + size + SHA-256; Loyfer reference_atlas.csv path + SHA-256; TCGA-LUAD adjacent-normal β path + SHA-256.
3. **Environment:** Python version, NumPy / Pandas / SciPy / Matplotlib versions, runtime, peak memory.
4. **Expected headline output:** per-tile case-vs-reference d table; top-1 distribution; outcome label.

---

## Stratification underpower disclosure (CHK-2.7)

Pre-locked: any sub-stratum with n<5 has its d reported but not interpreted. The all-cohort within-stratum aggregate is the headline. Sub-strata are reported as descriptive only.

Expected n per sub-stratum from series matrix counts (computed at runtime; pre-registration uses estimates):
- Female: ~80–100 (cohort is female-enriched per the F-dominant pattern in the series matrix preview)
- Male: ~40–60
- Stage I: ~50–60
- Stage II: ~30–40
- Stage III: ~30–40
- Stage IV: ~3–5 (likely underpowered, n<5 floor possible)
- Age 30s: ~3–5
- Age 40s: ~10–15
- Age 50s: ~25–35
- Age 60s: ~40–55
- Age 70s: ~40–50
- Age 80s: ~10–20

Stratum cells with n<5 (likely Stage IV, Age 30s) are pre-locked as descriptive-only.
