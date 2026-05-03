# VAL-126 — Phase C Run-Everything on TCGA-STAD Gastric Adenocarcinoma

**Sprint:** gastric+esophageal-epic v0.1 sprint, Phase C (disease cohort scoring)
**Card target:** gastric+esophageal-epic v0.1
**VAL ID:** VAL-126
**Cohort:** TCGA-STAD HM450 — n=395 primary tumor + n=2 paired adjacent-normal
**Substrate:** TCGA HM450K sesame Level 3
**RNG seed:** 20260502
**Prereg version:** v1.0 (this is the canonical version; no prior prereg was sealed for this VAL)

**SEALED BEFORE β DATA OBSERVED.** CCL-041 compliance.

---

## 1. Hypothesis (pre-locked, BIDIRECTIONAL per CHK-2.7 + Heath sign-off Q2)

**Stage 1 hypothesis (Xu-538 architectural drift):** TCGA-STAD tumor samples show |Cohen's d_unpaired| ≥ 0.5 vs TCGA-KIRC+PRAD adjacent-normal anchor (VAL-106 substrate) on Stage 1 Xu-538 pooled-entropy A-score. Direction expected POSITIVE (cycling-class tumor signature) based on TCGA-COAD VAL-062 d=+0.7241 paired and TCGA-LIHC non-viral VAL-064 d=+0.664 paired secretory-class precedents.

**Stage 2 hypothesis (cell-of-origin tile pattern, run-everything regime):**
- **BoccellatoStomachRef_HM450 (6 gastric tiles)** read NEGATIVE direction (de-differentiation degrades cell-of-origin tile fidelity per CCL-039)
- **EpiSCORE EsoRef Epi_* (4 squamous tiles)** read NULL — gastric tumor is columnar-lineage; if these tiles read POSITIVE/NEGATIVE-DIFFERENTIATING this is the cross-tissue overread signature flagged for kidney-card sprint follow-up
- **EpiSCORE OEref Basal** reads NULL (oral squamous tile not relevant to gastric tumor)
- **Loyfer 25-tile Upper_GI** reads NEGATIVE (cell-of-origin tile per CCL-039)
- **Loyfer 25-tile Bladder, Lung_cells, Hepatocytes, Pancreatic_beta** read POSITIVE (homogenization toward generic-tumor methylation per CCL-039 colorectal precedent)

**Stage 3 hypothesis (immune sub-composition):** Salas IDOL + UniLIFE 19-cell head-to-head; expect Stage 3 lineage-specific shifts in tumor vs control (T-cell exhaustion signature, NK shift, regulatory T-cell expansion). Magnitude direction not pre-locked (BIDIRECTIONAL).

**Subtype-stratified hypotheses (CCL-025):**
- **SUBTYPE=STAD_EBV (n=29):** highest Stage 1 |d| expected; high-CIMP epigenotype amplifies methylation drift signal
- **SUBTYPE=STAD_MSI (n=59) / MSI_SENSOR ≥ 4.0 (n=67):** elevated mutation burden + CIMP overlap; expect Stage 1 d intermediate-to-high
- **SUBTYPE=STAD_CIN (n=202):** dominant subtype; expected Stage 1 d at population baseline
- **SUBTYPE=STAD_GS (n=46):** Lauren diffuse-dominant; expect cell-of-origin tile pattern shift toward diffuse-type signature
- **Lauren intestinal-pooled (n=158) vs diffuse-pooled (n=78):** Boccellato tile pattern divergence — diffuse-type expected to show stronger Antrum/Corpus tile shift due to signet-ring/diffuse-type loss of glandular structure
- **H. pylori Yes (n=20) vs No (n=168):** exploratory; small Yes-group power-limited; report as descriptive

**Bidirectionality declaration:** All outcome thresholds use magnitude-based |d| with explicit direction labels.

---

## 2. Pre-locked decision criteria (CHK-2.1 + CHK-2.7 + CHK-4.11)

### Stage 1 outcomes (cycling-class architectural drift)
- **O1_STAGE1_PASS**: |d_unpaired| ≥ 0.5, lower 95% CI bound away from 0 in observed direction. Direction label STAGE1_POSITIVE if d>0, STAGE1_NEGATIVE if d<0.
- **O2_STAGE1_PARTIAL**: 0.2 ≤ |d_unpaired| < 0.5
- **O3_STAGE1_NULL**: |d_unpaired| < 0.2
- **O5_STAGE1_NEGATIVE_UNEXPECTED**: |d| ≥ 0.5 in direction NEGATIVE (when POSITIVE expected)

### Stage 2 outcomes (per atlas, per tile)
For each Stage 2 tile, compute |d_unpaired| Welch effect size and assign:
- `{tile}_DIFFERENTIATING_POSITIVE`: |d| ≥ 0.5, d > 0
- `{tile}_DIFFERENTIATING_NEGATIVE`: |d| ≥ 0.5, d < 0
- `{tile}_PARTIAL`: 0.2 ≤ |d| < 0.5
- `{tile}_NULL`: |d| < 0.2

**Pre-registered cell-of-origin tiles (NEGATIVE direction expected):** BoccellatoStomachRef_HM450 all 6 tiles, Loyfer 25-tile Upper_GI, Loyfer 25-tile Colon_epithelial_cells.

**Pre-registered "homogenization positive" tiles (POSITIVE direction expected):** Loyfer 25-tile Bladder, Lung_cells, Hepatocytes, Pancreatic_beta.

**Pre-registered "orthogonal/cross-tissue test" tiles (NULL expected):** EsoRef Epi_basal/stratified/suprabasal/upper, OEref Basal — these are squamous-lineage references; gastric tumor (columnar adenocarcinoma) should not register meaningful signal on squamous-lineage tiles. **NULL on these tiles is the hypothesis confirmation.** If POSITIVE/NEGATIVE-DIFFERENTIATING fires here, this is the cross-tissue overread signature to be cross-tested in the kidney-card sprint.

### Stage 3 outcomes (per atlas, per cell type)
- `{cell_type}_DIFFERENTIATING`: |d| ≥ 0.5 with explicit direction label
- `{cell_type}_PARTIAL`: 0.2 ≤ |d| < 0.5
- `{cell_type}_NULL`: |d| < 0.2

### Multi-disease detection patterns enumerated (run-everything mandate)
1. **STAD primary signature:** Stage 1 POSITIVE + cell-of-origin gastric tiles NEGATIVE + homogenization tiles POSITIVE
2. **Lauren-stratified discrimination:** intestinal-type vs diffuse-type cell-of-origin tile pattern difference
3. **Subtype-stratified Stage 1 amplification:** EBV+ > MSI > CIN > GS hypothesis test
4. **Stage 3 immune microenvironment characterization:** patient-level immune profile reported alongside tumor signature
5. **Cross-organ orthogonality test:** EsoRef squamous tiles read NULL on gastric tumor; if POSITIVE-DIFFERENTIATING, log for kidney-card cross-card calibration follow-up

---

## 3. Pre-locked stratifications (CHK-2.2) — ALL DATA PULLED, NO DEFERRALS

Strata sourced from cBioPortal (`stad_tcga_pan_can_atlas_2018`, `stad_tcga`, `stad_tcga_pub` studies) joined with GDC HM450 manifest at prereg seal time. n=395 tumor samples.

### Subtype (PanCanAtlas SUBTYPE call, harmonized)

| SUBTYPE | n in HM450 cohort |
|---------|------------------:|
| STAD_CIN (Chromosomal Instability) | 202 |
| STAD_MSI (Microsatellite Instability) | 59 |
| STAD_GS (Genomically Stable) | 46 |
| STAD_EBV (Epstein-Barr virus positive) | 29 |
| STAD_POLE (POLE ultramutator) | 7 |
| Not Reported | 50 |
| No cBioPortal match (TCGA-CG-5716, TCGA-HF-7131) | 2 |

### MSI quantitative
- **MSI_SENSOR_SCORE ≥ 4.0 (MSI-H per Niu 2014 + Cortes-Ciriano 2017):** 67 MSI-H + 326 MSS + 2 unavailable
- **MSI_SCORE_MANTIS:** continuous metric reported per-sample
- 8 cases discordant between SUBTYPE=MSI and MSI_SENSOR≥4 → both definitions reported

### EBV serology + subtype concordance
- **SUBTYPE=STAD_EBV (n=29) — primary stratum** (harmonized PanCanAtlas call)
- EBV_PRESENT=1 from stad_tcga_pub: 25 cases — secondary check
- Concordance: 23 cases both, 6 SUBTYPE-EBV but pub-EBV negative/missing, 2 pub-EBV+ but different SUBTYPE call

### H. pylori serology (stad_tcga)
| H_PYLORI_INFECTION | n |
|--------------------|--:|
| Yes | 20 |
| No | 168 |
| NotReported | 207 |

### Lauren classification (constructed from primary_diagnosis)

| Lauren category | Source diagnoses | n |
|-----------------|-----------------|--:|
| Lauren intestinal pooled | Adenocarcinoma intestinal type (76) + Tubular (74) + Papillary NOS (8) | **158** |
| Lauren diffuse pooled | Carcinoma diffuse type (64) + Signet ring cell (14) | **78** |
| Mucinous | Mucinous adenocarcinoma | 20 |
| Adenocarcinoma NOS | unclassifiable, separate stratum | 134 |
| Other | Basal cell carcinoma NOS (2) + Not Reported (3) | 5 |

LAUREN_CLASS field directly from cBioPortal was empty for all cases → primary_diagnosis is the operational source.

### Demographics
- Sex: Male 259 / Female 136 (per CCL-002)
- Age (cBioPortal AGE field), AJCC pathologic stage (I/II/III/IV), ICD-10 anatomic subsite (C16.0 cardia / C16.1 fundus / C16.2 body / C16.3 antrum / C16.4 pylorus / C16.5 lesser curvature / C16.6 greater curvature / C16.8 overlapping / C16.9 NOS)

### Sample types
- Primary Tumor: 395
- Solid Tissue Normal (paired adjacent-normal): 2 — structurally underpowered for paired-d; reported descriptively, not the primary statistic

---

## 4. Run-everything atlas stack (per Heath sign-off 2026-04-26)

Phase C VAL-126 runs every calibrated atlas on every TCGA-STAD IDAT, no gating.

| Stage | Atlas | Tiles | Calibration anchor SHA |
|-------|-------|------:|-----------------------|
| 1 | Xu-538 panel | 1 (immune-class pooled-entropy) | Within-cohort + VAL-106 cross-anchor |
| 2 | Layered Moss+Loyfer 25-tile | 25 | sealed VAL-112/113 |
| 2 | BoccellatoStomachRef_HM450 | 6 | sealed VAL-123 (`f5a620a93a...`) |
| 2 | EpiSCORE EsoRef bridged | 8 | sealed VAL-124 (`6e650bd78e...`) |
| 2 | EpiSCORE OEref bridged | 9 | sealed VAL-125 (`8f4e34ef63...`) |
| 2 | Caggiano CelFiE TIM | 19 | sealed VAL-113 |
| 3 | Salas IDOL Blood.EPIC (450K variant `IDOLOptimizedCpGs450k_compTable.csv`) | 6 | sealed VAL-106 |
| 3 | UniLIFE Guo 2025 | 19 | sealed VAL-082+ |

Total: **1 Stage 1 panel + 67 Stage 2 tiles + 25 Stage 3 cell types per IDAT.**

Every atlas runs on every sample regardless of any other tile/atlas's outcome. This is the run-everything mandate per Heath sign-off.

---

## 5. CHK-3.1A / CHK-3.1B / CHK-3.1C gates (pre-locked)

Following VAL-118 prostate Phase C precedent for tumor / cross-substrate cohorts:

- **CHK-3.1A:** TCGA-STAD HM450 sesame Level 3 substrate baseline is **documented, not gated**. The KIRC+PRAD adjacent-normal threshold (f_extreme ≥ 0.505) is a healthy-substrate gate; tumor and gastric-tissue substrate distributions are expected to differ. Per-atlas CHK-3.1A reported in results JSON for transparency. **Hard fail only on single-tone failure** (all-NaN beta values or median β > 0.95). Pre-flight observed STAD f_extreme mean = 0.4358 (tumor n=5) and 0.4231 (normal n=2) — ~9 percentage points lower than VAL-106 anchor 0.5587. This is the substrate baseline shift to be documented in CHK-3.2.

- **CHK-3.1B per-sample atlas-CpG-coverage:** ≥ 0.80 per CHK-2.8 substrate floor. Run separately per atlas's CpG list. Pass rate ≥ 95% per atlas. **Atlas substrate-mismatch failures** (e.g., panel built on EPIC 850K applied to HM450) flagged BEFORE seal at CHK-2.17 pre-flight gate.

- **CHK-3.1C:** all 8 atlases verified PASS at calibration time (zero duplicates each).

---

## 6. CHK-2.17 cohort-substrate-coverage pre-flight (BLOCKING per bladder-LL precedent)

7 random TCGA-STAD HM450 β files (5 random tumor + 2 paired normal — full normal cohort) from VAL-126 manifest. Verify per-sample coverage ≥80% on each calibrated atlas. If any atlas fails on >20% of preflight samples, halt seal and route to repair pathway.

**Pre-flight executes BEFORE prereg seal goes hot.** Pre-flight files already downloaded + MD5-verified (logged in `preflight/` directory).

---

## 7. Comparison strategy

- **Primary statistic:** Welch unpaired d on tumor (n=395) vs TCGA-KIRC+PRAD adjacent-normal anchor (VAL-106 n=210, sealed substrate calibration)
- **Secondary descriptive:** paired-d on n=2 paired HM450 STAD adjacent-normals — structurally underpowered; reported descriptively only
- **Direction labels per CHK-2.7. 95% CI computed per Welch t.**
- **Stratified analyses:** identical Welch unpaired d computed per SUBTYPE / Lauren / sex / MSI status / H. pylori / EBV groups; effect sizes reported separately per stratum.

---

## 8. CHK-3.2 cross-cohort baseline check

For every atlas/tile combination:
- Compute TCGA-STAD adjacent-normal-only mean A-score (n=2; report with sample-size caveat)
- Compare to VAL-106 anchor mean (TCGA-KIRC+PRAD adjacent-normal n=210)
- Report difference in anchor-SD units
- Flag tier per CCL-025 + Stage 3 elevation rule:
  - <1 SD: report only
  - 1-3 SD: `baseline_mismatch_flag: true`, downgrade cross-cohort comparison
  - ≥3 SD: invalidate cross-cohort absolute comparisons; within-cohort only

---

## 9. CHK-7.6 reproducibility triple

- **Source code:** `val126_stad_phase_c.py` + `val126_chunk_runner.py`
- **Inputs:**
  - 397 TCGA-STAD HM450 sesame Level 3 β files (manifest sealed at prereg time, file_id + MD5 + clinical metadata in `tcga_stad_hm450_manifest_FINAL.json`)
  - 8 calibrated atlases (SHA-sealed in atlas_vault INVENTORY.json)
  - VAL-106 sealed anchor distributions for CHK-3.2
- **Environment:** Python 3.x, NumPy, scipy.stats.
- **Expected output:** `VAL-126_phase_c_results.json`, `VAL-126_per_sample_phase_c.csv`, `VAL-126_stratified_results.json`, `outcome.md`

---

## 10. Specimen pathway compliance (CHK-2.4)

Specimen: bulk tumor / adjacent-normal tissue (not blood, not ccfDNA). Tissue substrate is Xu-538-validated per cycling-class TCGA precedents.

---

## 11. Test 2 placeholder

CCL-030 / CHK-2.5: Test 2 (lymphoid vs myeloid sub-panel) BLOCKED on OQ-2026-01. N/A here.

---

## 12. Logged follow-ups (NOT scope of this VAL)

- **EsoRef cross-tissue overread test** — per Heath quick thought 2026-05-02: EsoRef cross-tile separation 0.099 on TCGA-PRAD adjacent-normal exceeds ProstateRef's own cross-tile separation. Possible explanations: (A) atlas overreads, (B) genuine cross-tissue gene-promoter biology, (C) bridging math artifact. Test = run EsoRef on TCGA-PRAD tumor cohort (already-downloaded for prostate-epic prior calibration) and TCGA-KIRC tumor (will be downloaded for kidney card). Logged for kidney-card sprint, NOT this VAL. Will be added to CROSS_CARD_CALIBRATION_TODO at canonical-file update time.
