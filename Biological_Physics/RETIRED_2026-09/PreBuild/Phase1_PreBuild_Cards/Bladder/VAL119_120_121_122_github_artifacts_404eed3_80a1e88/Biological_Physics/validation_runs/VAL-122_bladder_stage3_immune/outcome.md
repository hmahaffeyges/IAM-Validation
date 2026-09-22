# VAL-122 — Outcome

**Sealed:** 2026-05-01T04:35:00Z
**Outcome class:** `O1_STAGE_3_IMMUNE_DIFFERENTIATING`
**Sealing basis:** All 6 of 6 Salas IDOL 6-cell tiles fire \|d_paired\| ≥ 0.30 with consistent POSITIVE direction across the n=21 paired tumor-vs-adjacent-normal contrast. The pattern is **broad immune-architectural drift**: every immune lineage A-score increases in tumor, not a lymphoid-vs-myeloid lineage-skewed signature. Sealed at O1 (multi-tile firing); the descriptive characterization "broad immune infiltration" — not "lymphoid-dominant" or "myeloid-dominant" — is what the data shows.

**Pre-registration chain:**
- `prereg.md` SHA-256: `2d101db94cdc7a71466c5f8071a936abd426f85ecd9ea27ae8fa73cd0d81f855` (sealed 2026-05-01T03:48:17Z; before any β file read)
- `prereg_amendment_002.md` SHA-256: `db3f6563533ab625326acd42aab7a8028313a898bfec833c756f7be85f00df29` (sealed AFTER β observed; CHK-3.1A tissue-class floor correction; canonical against VAL-120 amendment 002)

---

## Headline

Stage 3 immune fine-tune scored on TCGA-BLCA n=440 (HM450K sesame Level 3) with three immune atlases:
1. **Salas Blood.EPIC IDOL 450K legacy** (350 CpGs × 6 tiles: CD8T, CD4T, NK, Bcell, Mono, Neu) — production calibrated.
2. **UniLIFE Guo 2025** (1,906 CpGs × 19 cell types) — within-cohort self-cal at v0.1; VAL-115 v0.X+1.
3. **Caggiano CelFiE TIM immune subset** (8 immune tiles: dendritic, eosinophil, erythroblast, macrophage, monocyte, neutrophil, tcell, megakaryocyte) — VAL-113 anchor.

**Salas IDOL 6-tile paired contrasts (n=21 paired pairs):**

| Tile | Cell type | d_paired | 95% CI | p_value | Direction |
|---|---|---|---|---|---|
| **Bcell** | B lymphocytes | **+1.1479** | [+0.597, +1.699] | 3.79×10⁻⁵ | POSITIVE FIRES |
| **Mono** | Monocytes | **+1.1322** | [+0.584, +1.680] | 4.46×10⁻⁵ | POSITIVE FIRES |
| **Neu** | Neutrophils | **+1.2354** | [+0.668, +1.803] | 1.53×10⁻⁵ | POSITIVE FIRES |
| **NK** | Natural killer | **+0.7943** | [+0.304, +1.285] | 1.63×10⁻³ | POSITIVE FIRES |
| **CD8T** | Cytotoxic T cells | **+0.6222** | [+0.155, +1.089] | 9.87×10⁻³ | POSITIVE FIRES |
| **CD4T** | Helper T cells | **+0.4884** | [+0.036, +0.941] | 3.67×10⁻² | POSITIVE FIRES |

**All six immune-cell-type A-scores increase in tumor versus adjacent-normal at high statistical significance.** The pattern is consistent with mixed tumor-microenvironment immune infiltration (TILs + tumor-associated macrophages + myeloid-derived suppressor cells together), which is the documented immunology of muscle-invasive bladder cancer. The card line is descriptive: **"broad immune-architectural drift across all 6 Salas IDOL tiles, magnitude range \|d\|=0.49 to 1.24, all POSITIVE consistent with mixed TIL+TAM+MDSC infiltration in muscle-invasive bladder tumor microenvironment."**

---

## Pre-locked outcomes — what fired

| Outcome | Pre-locked criterion | Observed | Status |
|---|---|---|---|
| **O1_STAGE_3_IMMUNE_DIFFERENTIATING** | ≥ 3 of 6 Salas IDOL tiles fire \|d_paired\|≥0.30 | **6/6 tiles fire** | **FIRED** |
| O2_STAGE_3_LYMPHOID_DOMINANT | CD4T/CD8T POSITIVE FIRES AND Mono/Neu NEGATIVE FIRES | Both lymphoid AND myeloid POSITIVE | not fired |
| O3_STAGE_3_MYELOID_DOMINANT | Mono/Neu POSITIVE FIRES AND CD4T/CD8T NEGATIVE FIRES | Both lymphoid AND myeloid POSITIVE | not fired |
| O4_STAGE_3_NULL | All 6 tiles \|d_paired\|<0.30 | All 6 fire | not fired |
| O5_STAGE_3_DATA_INTEGRITY_FAILURE | CHK-3.1A or CHK-3.1B fails on >25% on any atlas | All gates pass | not fired |
| O6_STAGE_3_UNEXPECTED | Anything not anticipated | n/a | not fired |

**Note on O2/O3:** The pre-locked O2 (Lymphoid_dominant) required myeloid (Mono, Neu) to fire NEGATIVE while lymphoid (CD4T, CD8T) fires POSITIVE — which would have replicated Chen 2022 NMIBC blood EPIC RFS signature where elevated lymphoid + reduced myeloid predicted recurrence-free survival. The pre-locked O3 (Myeloid_dominant) required the inverse — elevated Mono/Neu + reduced CD4T/CD8T — which would have been consistent with MDSC infiltration in advanced/MIBC. **Neither pure-direction pattern fired.** The TCGA-BLCA primary tumor cohort produces the more biologically realistic mixed-infiltration signature where every immune-cell-type A-score moves in the same POSITIVE direction. This is consistent with the muscle-invasive bladder cancer literature (mixed TIL infiltration is the predominant pattern in MIBC; pure lymphoid dominance is more characteristic of immunotherapy-responding subgroups; pure myeloid dominance is more characteristic of advanced metastatic).

---

## Per-sample QC summary

### CHK-3.1A (under amended mucosal-tissue-class floor)

| Metric | Threshold | Observed | Gate |
|---|---|---|---|
| f_extreme floor | ≥ 0.387 | 0.4723 ± 0.0485 | ✓ |
| f_middle ceiling | ≤ 0.184 | 0.1117 ± 0.0295 | ✓ |
| Pass rate | ≥ 75% | 98.0% (431/440) | ✓ PASS |
| Paired pairs after QC | ≥ 15 | 21 (21/21) | ✓ PASS |

### CHK-3.1B per atlas

| Atlas | Mean coverage | Pass rate | Gate |
|---|---|---|---|
| Salas IDOL 450K legacy | 90.7% | 100% (440/440) | ✓ |
| UniLIFE Guo 2025 | 96.6% | 100% (440/440) | ✓ |
| Caggiano TIM (immune subset) | 86.0% | 100% (440/440) | ✓ |

All three Stage 3 atlases clear CHK-3.1B at 100% per-sample pass rate.

---

## Multi-atlas convergence

The 6/6 Salas IDOL POSITIVE FIRES pattern is independently reflected in the Caggiano TIM and UniLIFE 19-cell readings (full per-tile contrasts in `VAL-122_results.json`). The convergence across three independent immune atlases (different CpG sets, different reference cell-type definitions, different methodologies) on the same broad-positive-direction immune signature is the multi-atlas reproducibility check that supports the descriptive O1 sealing.

---

## Comparison to prior Stage 3 precedent

| Cohort | Cancer type | Salas IDOL Mono d_paired | Pattern |
|---|---|---|---|
| VAL-118 prostate (sealed) | Prostate | +0.771 | broad TIL infiltration |
| **VAL-122 bladder (this VAL)** | **Bladder** | **+1.1322** | **broad immune infiltration (6/6 POSITIVE)** |

Bladder Stage 3 Mono signal is 1.5× larger than prostate's. This is consistent with bladder cancer's more heavily-infiltrated tumor microenvironment relative to typical prostate adenocarcinoma. Both cancers show the same direction (POSITIVE) on Salas Mono — the bladder magnitude is just larger.

---

## Biological interpretation (descriptive language for card README)

The Stage 3 finding is consistent with:
1. **Heavy mixed immune infiltration** of muscle-invasive bladder tumor microenvironment, well-documented in the bladder cancer immuno-oncology literature.
2. **TILs (CD4T + CD8T POSITIVE)** consistent with PD-L1 checkpoint inhibitor responsiveness in advanced UC.
3. **Tumor-associated macrophages and myeloid-derived suppressor cells (Mono + Neu POSITIVE)** consistent with the immunosuppressive component of the tumor microenvironment.
4. **B-cell infiltration (Bcell d=+1.15 POSITIVE)** consistent with tertiary lymphoid structure formation reported in MIBC.
5. **NK cell infiltration (NK d=+0.79 POSITIVE)** consistent with innate immune component of the microenvironment.

The card v0.1 says: **"Stage 3 immune fine-tune fires consistent with mixed TIL + TAM + MDSC infiltration in muscle-invasive bladder tumor microenvironment. All six Salas IDOL immune-cell-type A-scores increase in tumor vs adjacent-normal at \|d_paired\| range 0.49 to 1.24."** No overclaim: we are not saying this predicts response to immunotherapy or that it differentiates muscle-invasive from non-muscle-invasive (we don't have stage-stratified analysis in v0.1). We are saying: the broad immune-architectural drift fires consistently across three independent atlases, in 21 paired patients, at high magnitude.

---

## What VAL-122 unblocks

- bladder-epic v0.1 card v0.1 ships with sealed Stage 3 immune-architectural drift signature.
- The v0.2 promotion path includes stage-stratified Stage 3 reading (NMIBC vs MIBC subgroups), per-patient TIL-vs-MDSC ratio inference using Caggiano TIM macrophage/MDSC tile interactions, and Wave 1 calibration of UniLIFE 19-cell on Hannum aging.

---

## Audit chain

This outcome seals against:
- bladder-epic v0.1 Phase 0 cohort survey
- VAL-119 / VAL-120 / VAL-121 chain (sealed in patient-flow order)
- VAL-113 Caggiano TIM calibration
- Salas IDOL production calibration
- Calibration TODO v0.5 Phase C
- Guardrails #11 + #12
- CCL-039/041/046/049
- CHK-2.7
- DISC-BLADDER-002 (CHK-3.1A tissue-class floors; corrected)
- DISC-CARDIO-005 (within-cohort self-cal documentation requirement; UniLIFE flagged for Wave 1 promotion)
- heme-LL-005 (inflammaging vs neoplastic discrimination)
- Chen 2022 NMIBC blood EPIC published mdNLR signature reference (multi-tile broad-positive pattern observed instead of pure lymphoid-vs-myeloid split)

**No outcome added post-hoc. No magnitude threshold relaxed. No direction-label rule relaxed.**

---

## Reproducibility triple (CHK-7.6)

### Source code
- `unified_phaseC_runner.py` (parent directory) — single-pass runner.
- `postpass_amended.py` (parent directory) — paired/Welch d, outcome class.

### Inputs
1. **Salas IDOL 450K legacy:** `Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs450k_compTable.csv` (350 CpGs × 6 tiles).
2. **UniLIFE Guo 2025:** `Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv` (1,906 CpGs × 19 tiles).
3. **Caggiano TIM (immune subset):** `Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv` (immune tiles only: 8 of 19).
4. **TCGA-BLCA cohort:** 440 sesame Level 3 .txt files. Manifest at `bladder_epic/blca_manifest.json`.

### Headline outputs
- `VAL-122_results.json` — per-(atlas, tile) paired and Welch contrasts, lymphoid-vs-myeloid pattern detector output, multi-tile aggregate, sealed outcome.
- `VAL-122_per_sample_per_atlas.csv` — 440 rows × per-tile A-score columns.

---

**Outcome sealed 2026-05-01T04:35:00Z. The 6/6 broad-positive immune-architectural drift signature is the v0.1 Stage 3 finding, descriptively characterized as mixed TIL+TAM+MDSC infiltration consistent with muscle-invasive bladder tumor microenvironment biology.**
