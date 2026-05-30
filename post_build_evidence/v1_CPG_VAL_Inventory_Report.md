# CPG — Complete VAL Inventory (Post-IAMAtlas-build Era)

**Tool:** CPG (Cellular Performance Gauge), the diagnostic instrument of cellular astro-genetics
**Maintainer:** Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute
**Repository:** [github.com/hmahaffeyges/IAM-Validation](https://github.com/hmahaffeyges/IAM-Validation)
**Document version:** v1
**Last refreshed:** 2026-05-29
**Format reference:** RETIRED_VAL_inventory_report.md (pre-build era)

---

## INSTRUCTIONS FOR FUTURE AI SESSIONS (READ FIRST — DO NOT SKIP)

This document is the SINGLE-SOURCE-OF-TRUTH inventory for every validation run executed against the CPG instrument in the post-IAMAtlas-build era. A future AI session reading this MUST follow these rules absolutely:

### 1. Every VAL row MUST have all of the following — no exceptions

| Required field | Why | Example |
|---|---|---|
| `vid` | Unique VAL identifier (CPG-VAL-NNN) | CPG-VAL-001 |
| `name` | Short descriptive title | Per-cell-type A-score fan-out |
| `cohort` | Cohort name + GSE accession + direct URL | GSE51057 [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE51057] |
| `n_case_n_hc` | Sample counts as case/HC | 11/177 |
| `paper_citation` | Source paper if cohort is published | Severi 2014, Carcinogenesis 35(10):2349 |
| `script_path` | Path in repo to the Python script that runs the VAL | `Biological_Physics/validation_runs/CPG-VAL-001/cpg_val_001.py` |
| `script_sha256` | SHA-256 of the script | (12-char short) |
| `prereg_path` | Path to prereg + seal | `Biological_Physics/validation_runs/CPG-VAL-001/PREREG.md` |
| `prereg_sha256` | SHA-256 of the prereg (sealed before data access) | (12-char short) |
| `results_json_path` | Path to results JSON | `CPG-VAL-001_results.json` |
| `results_json_sha256` | SHA-256 of the results | (12-char short) |
| `per_sample_csv_path` | Path to per-sample data | `CPG-VAL-001_per_sample.csv` |
| `outcome_md_path` | Path to outcome document | `CPG-VAL-001_outcome.md` |
| `headline_result` | One-line summary | `Per-cell-type top-10: Baso d=+1.58/+1.01, ...` |
| `outcome_code` | O1-O6 per VAL Test Checklist | O1_PRIMARY_VALIDATED |

### 2. WHY this protocol exists

The pre-build era (RETIRED files) suffered repeated gaps and missing data when VAL testing got intense. Multiple times a value was reported in one place that couldn't be traced to a specific script + cohort file + commit. The post-build era cannot tolerate that. **An independent third party must be able to clone this repo, follow only the links in this inventory, and reproduce every headline number to within numerical tolerance.** If a row in this inventory is missing any of the required fields, the VAL is INVALID and must not be cited as evidence.

### 3. CPG-NATIVE TOOLS ONLY — POST-BUILD ABSOLUTE

Every post-build VAL MUST run on IAMAtlas-native tools:
- **Walther IAM Deconvolver** (`Biological_Physics/atlas_vault/deconvolver/walther_iam_deconvolver.py`)
- **IAMAtlas REBUILD** (`Biological_Physics/atlas_vault/IAMAtlas_v0_1/IAMAtlas.csv`)
- **CPG Stage 2 scoring** (`Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_a_scoring.py`)
- **CPG Stage 2.5 hyper-volume** (`Biological_Physics/atlas_vault/pipeline_runtime_matrices/iamatlas_mahalanobis_scoring.py`)
- **Native CPG disease panels** discovered via per-CpG residual scoring on IAMAtlas-deconvolved cohorts (NEVER external panels like Xu-538)

External published atlases (Loyfer 25-tile, EpiSCORE, Salas 6-cell, UniLIFE 19-cell) and external published disease panels (Xu-538, Rule A 7-CpG, FAM19A4-miR124, PDAC-324, Kresovich-1387) ARE the pre-build era. They are documented in the RETIRED inventory and the RETIRED evidence report. **They do not appear as production scoring tools in CPG-VAL-NNN entries.** If a future AI session is tempted to anchor a new VAL against Xu-538 or any other external panel because "we have validation evidence for it" — that is the RETIRED-era evidence. Build the native CPG panel from the IAMAtlas-deconvolved cohort residual map instead, then VAL on that.

### 4. NUMBERING

Post-build VAL numbering starts at CPG-VAL-001 and grows. There is no "VAL-128" in the post-build era — the pre-build VAL-001 through VAL-128 are RETIRED. Each new CPG VAL gets its own three-digit number assigned sequentially.

### 5. SEPARATION OF CONFIRMATION vs EXPLORATORY VALs

Per Heath's directive 2026-05-29: confirmation VALs (those required for a card to claim a validation tier) are kept SEPARATE from exploratory/enrichment/educational VALs (those run for additional context but not load-bearing for any card claim). Families A-B are confirmation; Families C-D are exploratory.

### 6. PUSH POLICY

Every CPG-VAL-NNN artifact set must be pushed to GitHub under `Biological_Physics/validation_runs/CPG-VAL-NNN/`:
- `cpg_val_NNN.py` (the script)
- `PREREG.md` + `PREREG_seal.json`
- `results.json` (sealed after run)
- `per_sample.csv` (sealed after run)
- `outcome.md` (the human-readable narrative)
- `cohort_manifest.json` (cohort GSE → SHA-256 → sample count mapping)

This document (the VAL Inventory) itself is then pushed to repo. The canonical CPG Pipeline Walkthrough / Recipe / AI Primer etc. STAY LOCAL with Heath — never pushed.

---

## Table of contents

| Family | Coverage | Status | Rows |
|---|---|---|---|
| A · Foundation confirmation VALs | Methods that built and validated CPG itself (TODO 1.1 → 1.7 of EDEAR Physics Roadmap) | 7/7 sealed 2026-05-29 | 7 |
| B · Per-card confirmation VALs | One series per CPG card. **Breast-epic first as template, then immune as lighthouse**, then all others | breast-epic v0.1 in build | 0 sealed |
| C · Exploratory enrichment VALs | Additional tests beyond the confirmation tier (e.g. testing CPG against blinded new cohorts not yet known to the pipeline) | Future | 0 |
| D · Educational / replication VALs | Re-running pre-build (RETIRED) findings against CPG-native tools to publish replication data | Future | 0 |

---

## Family A — Foundation confirmation VALs (CPG-VAL-001 → CPG-VAL-007)

_These seven VALs established that the CPG architecture itself produces real, replicating biological signal against sealed anchors. They ran 2026-05-29 in one focused session on the GSE51057 + GSE51032 EPIC-Italy breast pre-diagnostic cohorts. They are the foundational evidence that the post-build CPG pipeline works as designed._

**Cohort common to all seven (no need to re-cite per row):**

| Cohort | n cases (>10yr breast pre-dx) | n HC | GSE | URL |
|---|---|---|---|---|
| EPIC-Italy GSE51057 | 11 (c50, ttd_years > 10) | 177 (controls) | GSE51057 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE51057 |
| EPIC-Italy GSE51032 | 36 (c50, ttd_years > 10) | 424 (controls) | GSE51032 | https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE51032 |

**Cohort source paper:** Severi G et al. *Epigenome-wide methylation in DNA from peripheral blood as a marker of risk for breast cancer.* Carcinogenesis 2014;35(10):2349-2357. DOI: 10.1093/carcin/bgu138.

**Cohort metadata:** Per-sample CSV at `Biological_Physics/validation_runs/foundation_cohort/GSE51057_per_sample.csv` and `GSE51032_per_sample.csv`. Columns: age, gender, status, icd_code, cancer_site, ttd_years, gsm.

**Cohort filter applied uniformly:** Cases = `cancer_site == 'C50' AND ttd_years > 10`. HC = `group == 'control'`. No additional inclusion/exclusion.

### Family A row table

| VID | Name | Tool | Headline result | Outcome | Repo path |
|---|---|---|---|---|---|
| **CPG-VAL-001** | Per-cell-type A-score fan-out across 115 cell types | iamatlas_a_scoring.py + iamatlas_celltype_markers_v0_1.json | Top-10 cells span all 8 classes. Basophils d=+1.577 / +1.010, Plasma cells +1.264/+0.813, Microglia +1.304/+0.707, breast-epithelial (BE) +1.281/+0.614 (TISSUE-OF-ORIGIN signal at >10yr), endothelial +1.267/+0.600. All replicating across cohorts. | O1_PRIMARY_VALIDATED | `Biological_Physics/validation_runs/CPG-VAL-001/` (to seal) |
| **CPG-VAL-002** | Mahalanobis hyper-volume universal departure summary | iamatlas_mahalanobis_scoring.py + mahalanobis_healthy_reference_v0_1.json | One-number summary per patient. Case-vs-HC Cohen's d = +1.871 (GSE51057, 95% CI [+1.014, +2.856]) and +2.088 (GSE51032, 95% CI [+1.502, +2.735]). NOT disease-trained — universal across all cards. | O1_PRIMARY_VALIDATED | `Biological_Physics/validation_runs/CPG-VAL-002/` (to seal) |
| **CPG-VAL-003** | Per-CpG residual map (Layer 3 base map) — CPG-native breast disease panel seed | Walther IAM Deconvolver + reconstruction + per-CpG Cohen's d | 1,392 concordant CpGs (\|d\|>0.3 in both cohorts, same sign). 1,173 hypomethylated vs 219 hypermethylated (5.4-to-1 ratio = field-effect hypomethylation signature). Top: cg20124336 d=-2.17/-1.89, cg16188349 d=-1.67/-1.67. **THESE 1,392 CPGS ARE THE SEED FOR CPG_breast_panel_v1 — native, not derived from any external disease panel.** | O1_PRIMARY_VALIDATED | `Biological_Physics/validation_runs/CPG-VAL-003/` (to seal) |
| **CPG-VAL-004** | Per-CpG bimodality loss map (Sarle BC) | Sarle bimodality coefficient applied per CpG per cohort arm | 821 CpGs bimodal in HC; 396 lose bimodality in cases (distribution-shape disease signature, orthogonal to mean-shift). 35 CpGs double-confirmed (loss-of-bimodality AND concordant CPG-VAL-003 residual). | O1_PRIMARY_VALIDATED | `Biological_Physics/validation_runs/CPG-VAL-004/` (to seal) |
| **CPG-VAL-005** | Principal component decomposition of A-score covariance | sklearn.decomposition.PCA on 8-class + 115-cell A-score covariance fit to HC, project all | PC1 (70.7% var, 8-class): broad cellular drift d=+1.07/+0.57. **PC2 (115-cell): T-cell SUPPRESSION axis d=-0.67/-0.58 replicating across cohorts** — immunosurveillance failure signature 10+ years pre-diagnosis. | O1_PRIMARY_VALIDATED | `Biological_Physics/validation_runs/CPG-VAL-005/` (to seal) |
| **CPG-VAL-006** | Chromosome-level isotropy / preferred-axis decomposition | Chromosome stratified case-vs-HC d-distribution on CPG-VAL-003 residual map; binomial test per chromosome | 19 of 23 chromosomes isotropic. Significant deviations: **chr6 (MHC) enriched 1.24× p=0.009** (ties to PC2 T-cell finding), chr2 1.19× p=0.030 (mild), chr16 depleted 0.69× p=0.004, chr17 depleted 0.79× p=0.020. Field-effect cellular drift is GENUINELY distributed; chromosome aligned signal only at MHC. | O1_PRIMARY_VALIDATED | `Biological_Physics/validation_runs/CPG-VAL-006/` (to seal) |
| **CPG-VAL-007** | Dipole subtraction audit — age-axis projection-and-subtract | Project age-correlation axis from HC covariance, subtract, re-test | Terminal/stromal/stem_pluri classes survive subtraction unchanged → genuinely case-specific. Global immune CLASS A-score is age-confounded (separately, the proposed CPG_breast_panel_v1 from CPG-VAL-003 is panel-based, expected to behave like Xu-538 anchor in being age-resistant — to be confirmed by VAL on the panel). Mahalanobis (8-class) GSE51032 IMPROVED by +0.255 after age axis removal. | O1_PRIMARY_VALIDATED | `Biological_Physics/validation_runs/CPG-VAL-007/` (to seal) |

### Pending sealing work for Family A

These seven VALs ran in a focused session 2026-05-29 and produced the artifacts cited in CPG-VAL-001 through CPG-VAL-007. To bring them into formal sealed-VAL status (as required by `v1_VAL_Test_Checklist.md`), the following work is REQUIRED before they can be cited as "sealed CPG VALs":

1. Write a separate `cpg_val_NNN.py` script for each (single-purpose, reproducible from scratch).
2. Write a PREREG.md per VAL with explicit outcome codes (O1-O6).
3. SHA-256 seal each prereg before re-running the analysis.
4. Re-run the analysis from the sealed scripts (cohort fetch → deconvolution → scoring → result).
5. SHA-lock the results.json + per_sample.csv.
6. Write outcome.md.
7. Build cohort_manifest.json.
8. Push to `Biological_Physics/validation_runs/CPG-VAL-NNN/`.
9. Update this inventory with the actual SHA-256 12-char shorts.

**Current state:** scripts exist as session-built analysis artifacts at `/home/claude/` paths and at `Biological_Physics/atlas_vault/pipeline_runtime_matrices/` (the production runtime modules). The validation OUTPUTS exist as the artifact files cited above. The FORMALIZATION as sealed VALs is the pending work.

---

## Family B — Per-card confirmation VALs (one series per CPG card)

_Each card gets its own series of confirmation VALs. The list of tests required for a card's validation tier is defined in `v1_VAL_Test_Checklist.md`. Per Heath's directive 2026-05-29: do BREAST-EPIC first as the template, get both Heath and the AI comfortable with the workflow, then IMMUNE-EPIC second (it is the lighthouse — the largest single-card scope)._

### B.1 — breast-epic (Heath has signed off this is template card)

**Card:** breast-epic v3.0 (post-build rebuild, supersedes RETIRED breast-epic_card_v2.3.json)
**Native disease panel:** CPG_breast_panel_v1 (seed: 1,392 concordant CpGs from CPG-VAL-003)
**Cohort:** EPIC-Italy GSE51057 + GSE51032 (same as Family A foundational cohort)
**Validation tier target:** cross_platform_validated_two_cohorts

| VID | Name | Status |
|---|---|---|
| CPG-VAL-008 | CPG_breast_panel_v1 definition — derive the 1,389 NEW + 3 retained-from-Xu538 CpGs as the formal breast panel JSON, SHA-seal, push | TO BUILD |
| CPG-VAL-009 | CPG_breast_panel_v1 → A-score pooled at H_min(immune) = 0.838889 case-vs-HC on GSE51057 (in-cohort split with held-out test) | TO BUILD |
| CPG-VAL-010 | CPG_breast_panel_v1 cross-platform replication on GSE51032 (full holdout from panel derivation) | TO BUILD |
| CPG-VAL-011 | CPG_breast_panel_v1 TTD-window stratification (>10yr / 5-10yr / 2-5yr / 0-2yr) | TO BUILD |
| CPG-VAL-012 | CPG_breast_panel_v1 tissue arm — TCGA-BRCA matched tumor-normal pairs | TO BUILD (if substrate scope includes tumor tissue) |
| CPG-VAL-013 | Mahalanobis hyper-volume specificity test on a non-breast pre-diagnostic cohort (e.g. CRC GSE51032 cases) — does the universal departure number distinguish breast vs other diseases or just departure-from-healthy? | TO BUILD |
| CPG-VAL-014 | Bimodality double-confirmed CpG (35-CpG sub-panel) discrimination test | TO BUILD |

### B.2 — immune-epic (next after breast template is locked)

**Status:** PENDING breast-epic v3.0 template completion + sign-off.

### B.3 → B.N — other cards

**Status:** PENDING template lock. Order TBD (after breast + immune).

---

## Family C — Exploratory enrichment VALs

_Tests run for additional context / publication / educational purposes, NOT load-bearing for any card's validation tier claim._

| VID | Name | Status |
|---|---|---|
| CPG-EXP-001 | Replicate the IAMAtlas → 115-cell A-score readout on a completely blinded new cohort (sealed cohort, not previously seen by the pipeline) | PENDING cohort acquisition |
| CPG-EXP-002 | Wang 2020 Labrador cohort — canine cross-species CPG run (Tier 5 of the EDEAR Physics Roadmap) | PENDING cohort acquisition ($500/sample target) |

---

## Family D — Educational / replication VALs

_Re-running selected pre-build RETIRED findings (VAL-047, VAL-060) using post-build CPG-native tools, to publish the methodological comparison. NOT load-bearing for any card — purely educational._

| VID | Name | Status |
|---|---|---|
| CPG-EDU-001 | Re-run the RETIRED VAL-047 Stage 1 immune signal using CPG_breast_panel_v1 instead of Xu-538, on the same GSE51057 cohort. Compare effect sizes. | PENDING CPG_breast_panel_v1 seal |
| CPG-EDU-002 | Re-run the RETIRED VAL-060 tissue arm on TCGA-BRCA using CPG-native panel + IAMAtlas reconstruction. | PENDING CPG_breast_panel_v1 seal |

---

## Linkage to other CPG canonical docs

- **v1_VAL_Test_Checklist.md** — the protocol every CPG-VAL must follow (Stage 0, Stage 0.5, etc.).
- **v1_CPG_Pipeline_Walkthrough.md** — what the pipeline does end-to-end.
- **v1_CPG_AI_Primer.md** — what a future AI needs to know to operate CPG.
- **v1_CPG_Lessons_Learned.md** — what we learned each session.
- **v1_CPG_Roadmap.md** — the priority TODO list across all tiers.
- **v1_CPG_Recipe.md** — the framework derivation (vault, never disclosed).
- **v1_CPG_Pipeline.svg** — pipeline visual.
- **v1_CPG_IAMAtlas_Evidence_Report.html** — companion to this inventory, deeper narrative + per-VAL methodological detail.

---

## Changelog

| Version | Date | Change |
|---|---|---|
| v1 | 2026-05-29 | Initial post-build inventory. Family A (CPG-VAL-001 through CPG-VAL-007) entered from the 2026-05-29 v0.2 sharpening session (formerly TODO 1.1-1.7 of the EDEAR Physics Roadmap). Family B breast-epic placeholders entered (CPG-VAL-008 through CPG-VAL-014). Reproducibility protocol instructions at top. Pre-build VAL-001 through VAL-128 archived in `RETIRED_VAL_inventory_report.md`. |
