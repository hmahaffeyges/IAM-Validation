# VAL-121 — Pre-Registration

**VAL ID:** VAL-121
**Card target:** bladder-epic v0.1 (Phase C — Stage 2 cell-of-origin run-everything)
**Substrate cohort:** TCGA-BLCA (n=440 = 418 primary tumor + 21 solid tissue normal + 1 metastatic), Illumina HM450K sesame Level 3 from GDC
**Sealed:** [SEAL_TIMESTAMP at execution]
**Sealed BEFORE β read:** YES

---

## Question

Under EDEAR's run-everything discipline (Guardrail #12 + 2026-04-26 sign-off), every TCGA-BLCA sample scores against every calibrated Stage 2 atlas in bladder-epic v0.1's `atlases_run` block:

1. **Layered Moss+Loyfer** (calibration anchor VAL-112; 6,105 CpGs × 25 cell types including `Bladder` tile from Loyfer 2023 n=5 sorted bladder epithelium WGBS)
2. **EpiSCORE BladderRef CpG-bridged** (calibration anchor VAL-119, sealed 2026-05-01T03:46:00Z; 2,696 CpGs × 4 bladder cell types EC/Epi/Fib/IC)
3. **Caggiano CelFiE TIM array-bridged** (calibration anchor VAL-113; 254 CpGs × 19 cell types — bladder tumor microenvironment context)

The critical question Phase C answers: **Does the bladder cell-of-origin signal converge under multi-atlas triangulation, or does it diverge in a way that exposes single-atlas confounders?**

---

## Why this matters operationally

VAL-120 (Stage 1) tells us whether bladder cancer fires a Stage 1 immune red flag and at what magnitude. VAL-121 (Stage 2) localizes the cell of origin:

- **Loyfer `Bladder` tile** (bulk urothelial reference, WGBS-sorted gold standard): does it read tumor as architectural drift relative to adjacent-normal? Direction expectation per CCL-039: tumor-vs-adjacent-normal-paired = NEGATIVE on cell-of-origin tile (urothelial dedifferentiation pattern, analog to prostate's LE_NEGATIVE finding in VAL-118).
- **BladderRef Epi tile** (sub-cell-type urothelial epithelium, calibrated VAL-119): same direction expectation as Loyfer `Bladder`. If both fire NEGATIVE consistent magnitudes, bladder-epic has a robust cell-of-origin signature.
- **BladderRef EC/Fib/IC tiles** (vascular endothelial, fibroblast stromal, intra-bladder immune): direction expectation = POSITIVE per CCL-039 (tumor microenvironment develops architectural complexity, analog to prostate's BE/EC/Fib/Leu/SM all-positive pattern in VAL-118).
- **Loyfer non-bladder tiles** (Prostate, Kidney, Breast, Thyroid, Colon, etc.): CHK-3.2 cross-tile sanity check. Bladder tumor should NOT read positive on prostate-tile or kidney-tile. If it does, exposes a substrate confounder.
- **Caggiano TIM tiles** (endothelial, fibroblast, t-cell, monocyte, macrophage, neutrophil, etc.): tumor microenvironment immune infiltration signature. Direction expectation = POSITIVE on most tiles for tumor vs adjacent-normal.

The five-vs-one direction split that prostate VAL-118 surfaced (LE_NEGATIVE + BE/EC/Fib/Leu/SM all POSITIVE) is the prostate analog. Bladder will produce its own characteristic split — that is what this VAL surfaces.

---

## Cohort + atlas inventory

### TCGA-BLCA cohort
- Same as VAL-120. 440 files cached at `/home/claude/edear_working/bladder_epic/blca_betas/`. Manifest at `/home/claude/edear_working/bladder_epic/blca_manifest.json`. 21 paired patients (have both adjacent-normal + tumor).

### Stage 2 atlases (all calibrated against TCGA HM450K sesame Level 3)

| Atlas | n_CpGs | n_tiles | Calibration anchor VAL | Atlas family |
|---|---|---|---|---|
| Layered Moss+Loyfer | 6,105 | 25 cell types (incl. Bladder, Prostate, Kidney, Breast, Thyroid, Colon, Vascular_endothelial, etc.) | VAL-112 (sealed 2026-04-29) | tile-coverage WGBS |
| EpiSCORE BladderRef | 2,696 | 4 bladder cell types (EC, Epi, Fib, IC) | VAL-119 (sealed 2026-05-01) | gene-promoter |
| Caggiano CelFiE TIM | 254 | 19 cell types (incl. endothelial, fibroblast, tcell, monocyte, macrophage, neutrophil, etc.) | VAL-113 (sealed 2026-04-29) | tile-coverage WGBS |

**Substrate match.** All three atlases calibrated on TCGA HM450K sesame Level 3 (the same cohort + substrate as TCGA-BLCA). No cross-substrate caveats apply here — VAL-121 is the cleanest Phase C substrate-match scoring run we have for bladder.

---

## Pre-locked outcomes

Per CHK-2.7 (magnitude-based |d| thresholds with direction labels per CCL-039 cell-of-origin direction expectation).

### O1 — `MULTI_ATLAS_CONVERGENT_BLADDER_TILE_FIRES`

Loyfer `Bladder` tile paired d direction = NEGATIVE AND |d| ≥ 0.30 AND BladderRef Epi tile paired d direction = NEGATIVE AND |d| ≥ 0.30 AND at least one BladderRef stromal/vascular tile (EC, Fib) shows direction = POSITIVE with |d| ≥ 0.30. Multi-atlas convergence on bladder cell-of-origin signature. Bladder-epic v0.1 promotes from `stage_1_only_validated` to `multi_modal_validated + multi_atlas_calibrated`.

### O2 — `BLADDER_TILE_DIFFERENTIATING_DIRECTION_AMBIGUOUS`

Loyfer `Bladder` tile and BladderRef Epi tile both fire with |d| ≥ 0.30 BUT directions diverge across atlases (one POSITIVE, one NEGATIVE). Atlas-specific direction-flip; v0.1 card flagged with CCL-049 multi-atlas direction-divergence note. Promote analysis to v0.2 atlas-specific deep-dive.

### O3 — `STAGE_2_NULL`

Both Loyfer `Bladder` tile and BladderRef Epi tile fall below |d| < 0.30 on the n=21 paired contrast. Bladder Stage 2 cell-of-origin signal does not reach magnitude threshold under multi-atlas Phase C. Direction labeled per observation. Card v0.1 claims documented as Stage-2-null with v0.X+1 next-step (re-test on additional cohorts: GSE52955 multi-cancer, Bryan UK NMIBC, Chen 2022 NMIBC blood).

### O4 — `STAGE_2_DATA_INTEGRITY_FAILURE`

CHK-3.1A or CHK-3.1B fails on >25% of TCGA-BLCA samples on any atlas; or paired pair count <15. Halt and re-fetch.

### O5 — `STAGE_2_UNEXPECTED`

Anything not anticipated in O1-O4. Per CCL-032 (data integrity → biology → framework), classify as O5 if data integrity is uncertain or result contradicts expected biology. Convene with Heath before sealing direction.

---

## Pre-locked thresholds (CHK-2.1 + CHK-2.7)

| Threshold | Pre-locked value | Rationale |
|---|---|---|
| Magnitude threshold for "fires" | |d_paired| ≥ 0.30 | Same as VAL-118 LE prostate threshold and VAL-120 |
| Direction labels | POSITIVE / NEGATIVE | CHK-2.7 |
| Cell-of-origin direction expectation | NEGATIVE for Loyfer `Bladder` and BladderRef `Epi` (urothelium = bladder cancer cell of origin) | CCL-039 + DISC-PROSTATE-002 |
| Microenvironment direction expectation | POSITIVE for BladderRef `EC` `Fib` `IC` and Caggiano TIM immune+stromal tiles | CCL-039 |
| Cross-tile sanity (CHK-3.2) | Bladder tumor MUST NOT read |d_paired| ≥ 0.30 POSITIVE on Loyfer non-bladder tiles (Prostate, Kidney, Breast, Thyroid, Colon, Liver, Lung, Pancreas, etc.) | CHK-3.2 substrate confounder check |
| Minimum paired pairs | n ≥ 15 | Statistical power floor |
| CHK-3.1A pass rate | ≥ 75% | Phase C substrate-permissive |
| CHK-3.1B coverage per sample per atlas | ≥ 80% | CHK-2.8 substrate-floor |

---

## Statistical methodology

### Per-tile per-atlas paired contrast
For each (atlas, tile) combination:
- n=21 patients with both adjacent-normal and primary tumor
- paired_diff_i = A_tile(tumor) - A_tile(adjacent-normal)
- d_paired, 95% CI, p, direction = sign(mean(paired_diff))

### Per-tile per-atlas unpaired Welch contrast
For each (atlas, tile) combination:
- 418 tumor vs 21 normal Welch d, 95% CI, p, direction

### CHK-3.2 cross-tile sanity check
For Loyfer atlas: report d_paired for ALL 25 tiles (not just bladder). Bladder cancer should fire only on bladder-relevant tiles, not on prostate/kidney/breast/thyroid/etc. tiles. Out-of-tissue positive |d_paired| ≥ 0.30 is flagged.

### Multi-atlas convergence summary
For the bladder cell-of-origin question, primary scorers are:
- Loyfer `Bladder` (bulk urothelial WGBS reference)
- BladderRef `Epi` (sub-cell-type urothelial epithelium, gene-promoter)

Convergence = both NEGATIVE direction with |d| ≥ 0.30. Divergence is reported with atlas-specific d.

---

## Reproducibility triple (CHK-7.6)

### Source code
`val121_bladder_stage2_multiatlas.py` — Python 3.12 stdlib + numpy + scipy.stats. Loads three calibrated atlas matrices, loads TCGA-BLCA β files via manifest, computes per-sample CHK-3.1A + per-atlas CHK-3.1B coverage + per-tile A-scores per atlas. Identifies paired pairs. Computes paired d, unpaired Welch d, 95% CIs, p-values per (atlas, tile) combination. Cross-tile CHK-3.2 sanity output for Loyfer atlas.

### Inputs
1. **Layered Moss+Loyfer atlas:** `/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv` (deduped 6,105 CpGs × 25 cell types per VAL-112)
2. **EpiSCORE BladderRef CpG-bridged:** `/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_bladderref/episcore_bladderref_cpg_bridged.csv` SHA-256 `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6` (calibration anchor VAL-119)
3. **Caggiano CelFiE TIM array-bridged:** `/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv` (calibration anchor VAL-113)
4. **TCGA-BLCA β files:** `/home/claude/edear_working/bladder_epic/blca_betas/` × 440 files
5. **BLCA manifest:** `/home/claude/edear_working/bladder_epic/blca_manifest.json`

### Environment
- Python 3.12.3
- numpy 2.4.4
- scipy 1.17.1
- Expected runtime: ~10-15 minutes for n=440 cohort × 3 atlases × multi-tile
- Expected memory: ~1 GB peak

### Expected headline output
- `VAL-121_results.json` — per-(atlas, tile) d / CI / p / direction
- `VAL-121_per_sample_per_atlas.csv` — sample_id, case_id, sample_type, A_<atlas>_<tile> for every (atlas, tile) combination
- `VAL-121_cross_tile_sanity.json` — Loyfer 25-tile cross-check
- `outcome.md` — sealed outcome class

---

## RNG seed

20260420 (cookbook standard).

---

## SHA-256 of this prereg

To be computed at seal time and recorded in `PREREG_SEAL.txt` before val121 script reads any β files.

---

## Pre-registered audit chain

This prereg seals against:
- bladder-epic v0.1 Phase 0 cohort survey
- VAL-112 layered Moss+Loyfer calibration anchor (sealed 2026-04-29)
- VAL-113 Caggiano CelFiE TIM calibration anchor (sealed 2026-04-29)
- VAL-119 EpiSCORE BladderRef calibration anchor (sealed 2026-05-01T03:46:00Z)
- VAL-120 Stage 1 Xu-538 (sealed BEFORE this prereg per patient flow order)
- Calibration TODO v0.5 Phase C run-everything requirement
- Guardrail #11 (calibration before testing)
- Guardrail #12 (run-everything Phase C)
- CCL-039 (cell-of-origin direction expectation)
- CCL-041 (prereg before β read)
- CCL-046 (atlas selection traces to canonical-document-named candidates)
- CCL-049 (multi-atlas reporting)
- CHK-2.7 (magnitude-based |d| with direction labels)
- CHK-3.2 (cross-tile sanity check)
- DISC-PROSTATE-001 + DISC-PROSTATE-002 (gene-promoter atlas family + cell-of-origin direction-flip discipline)

val121 script execution begins ONLY after this prereg.md is sealed and SHA-hashed. Outcome sealed against pre-locked thresholds above.

**No outcome may be added post-hoc. No threshold may be relaxed post-hoc. No exception under CCL-041.**
