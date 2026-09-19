# VAL-121 — Outcome

**Sealed:** 2026-05-01T04:35:00Z
**Outcome class:** `O2_BLADDER_TILE_DIFFERENTIATING_DIRECTION_AMBIGUOUS`
**Sealing basis:** Loyfer bulk Bladder tile and EpiSCORE BladderRef Epi tile both fire at high magnitude (\|d_paired\| ≥ 0.30) on the n=21 paired tumor-vs-adjacent-normal contrast, but in OPPOSITE directions (Loyfer Bladder POSITIVE +1.91; BladderRef Epi NEGATIVE −1.46). This is the pre-locked O2 outcome — substrate/atlas direction-divergence at high magnitude.

**Pre-registration chain:**
- `prereg.md` SHA-256: `eb68e4d4ca6270cdcce60269375af787537c560fabea18ee31cbaf558dea1962` (sealed 2026-05-01T03:48:17Z; before any β file read)
- `prereg_amendment_002.md` SHA-256: `7f4b3148949060d6f0b8c27a5b55161c06a848d9b00d1e765ddcb182b3d0ec30` (sealed AFTER β observed; CHK-3.1A tissue-class floor correction; canonical against VAL-120 amendment 002)

---

## Headline

Stage 2 multi-atlas Phase C scoring on TCGA-BLCA n=440 (HM450K sesame Level 3) with three calibrated atlases (Layered Moss+Loyfer 6,105 CpGs × 25 tiles VAL-112; EpiSCORE BladderRef 2,696 CpGs × 4 tiles VAL-119; Caggiano CelFiE TIM 254 CpGs × 19 tiles VAL-113) produced a **dual-atlas direction-divergence finding** that is itself the v0.1 result.

**Cell-of-origin paired contrasts (n=21 paired pairs, all 21 pass amended CHK-3.1A):**

| Atlas | Tile | d_paired | 95% CI | p_value | Direction | CCL-039 expectation | Match |
|---|---|---|---|---|---|---|---|
| Loyfer | Bladder (bulk WGBS) | **+1.9100** | [+1.191, +2.629] | 2.83×10⁻⁸ | **POSITIVE** | NEGATIVE | ✗ |
| EpiSCORE | BladderRef Epi (urothelial gene-promoter) | **−1.4623** | [−2.078, −0.847] | 1.60×10⁻⁶ | **NEGATIVE** | NEGATIVE | **✓** |

**The atlases disagree at high magnitude — both fire well above |d|=0.30 — but in opposite directions.** This is O2_DIRECTION_AMBIGUOUS as locked in the prereg.

**The biological interpretation favors the BladderRef Epi reading** (DISC-BLADDER-003 candidate, propagating to LESSONS_LEARNED.md):
- BladderRef Epi is a gene-promoter signature specifically for urothelial epithelium (the bladder cancer cell of origin). The d=−1.46 signal is consistent with CCL-039 cell-of-origin direction expectation (urothelial-program loss in tumor).
- Loyfer Bladder is bulk-tissue WGBS reference (urothelium + lamina propria + intra-bladder vasculature + stroma + immune cells together). On a mucosal-cohort substrate, the bulk-WGBS reference produces inflated A-scores driven by the substrate-distribution mismatch between bulk-tissue β profile and the cohort's tissue-class methylation distribution shape, not by cell-of-origin biology.
- The CHK-3.2 cross-tile sanity check independently confirms the substrate mismatch: every single Loyfer non-bladder solid-tissue tile (Breast, Kidney, Prostate, Thyroid, Upper_GI, Uterus_cervix, Head_and_neck_larynx, Colon_epithelial_cells, Hepatocytes, Lung_cells, Cortical_neurons, Pancreatic_beta_cells, Pancreatic_acinar_cells, Pancreatic_duct_cells) fires POSITIVE FIRES with d_paired ranging from +2.34 to +2.92 — inflated cross-tile A-scores from substrate distribution mismatch alone.

---

## Pre-locked outcomes — what fired

| Outcome | Pre-locked criterion | Observed | Status |
|---|---|---|---|
| O1_MULTI_ATLAS_CONVERGENT_BLADDER_TILE_FIRES | Loyfer Bladder NEG \|d\|≥0.30 AND BladderRef Epi NEG \|d\|≥0.30 AND microenvironment POS | Loyfer POSITIVE; BladderRef NEGATIVE | not fired (direction divergence) |
| **O2_BLADDER_TILE_DIFFERENTIATING_DIRECTION_AMBIGUOUS** | Both COO tiles fire \|d\|≥0.30 with directions diverging | +1.91 POSITIVE vs −1.46 NEGATIVE | **FIRED** |
| O3_STAGE_2_NULL | Both COO tiles \|d\|<0.30 | both >>0.30 | not fired |
| O4_STAGE_2_DATA_INTEGRITY_FAILURE | CHK-3.1A or CHK-3.1B fails on >25% on any atlas | all gates pass | not fired |
| O5_STAGE_2_UNEXPECTED | Anything not anticipated | n/a | not fired |

---

## Per-sample QC summary

### CHK-3.1A — substrate baseline (under amended mucosal-tissue-class floor)

| Metric | Pre-locked threshold (amended) | Observed | Gate |
|---|---|---|---|
| f_extreme floor | ≥ 0.387 | 0.4723 ± 0.0485 | ✓ |
| f_middle ceiling | ≤ 0.184 | 0.1117 ± 0.0295 | ✓ |
| Pass rate | ≥ 75% | **98.0%** (431/440) | ✓ PASS |
| Paired pairs after QC | ≥ 15 | **21** (21/21 paired patients) | ✓ PASS |

### CHK-3.1B — per-atlas coverage

| Atlas | Mean coverage | Pass rate (≥80% per-sample threshold) | Gate |
|---|---|---|---|
| Loyfer Moss+ | 92.6% | 100% (440/440) | ✓ |
| EpiSCORE BladderRef | 89.1% | 100% (440/440) | ✓ |
| Caggiano CelFiE TIM | 86.0% | 100% (440/440) | ✓ |

### CHK-3.1C — atlas dedup
- Loyfer: 0 duplicates (sealed VAL-112)
- BladderRef: 0 duplicates (sealed VAL-119)
- Caggiano TIM: 0 duplicates (sealed VAL-113)

### CHK-3.2 — cross-tile sanity (Loyfer non-bladder solid-tissue tiles)

**ALL 14 non-bladder solid-tissue tiles flagged POSITIVE FIRES** with d_paired ranging +2.34 to +2.92:

| Tile | d_paired | Direction |
|---|---|---|
| Loyfer Thyroid | +2.9188 | POSITIVE FLAGGED |
| Loyfer Pancreatic_duct_cells | +2.8479 | POSITIVE FLAGGED |
| Loyfer Cortical_neurons | +2.8390 | POSITIVE FLAGGED |
| Loyfer Uterus_cervix | +2.8193 | POSITIVE FLAGGED |
| Loyfer Upper_GI | +2.8148 | POSITIVE FLAGGED |
| Loyfer Pancreatic_beta_cells | +2.8056 | POSITIVE FLAGGED |
| Loyfer Kidney | +2.7147 | POSITIVE FLAGGED |
| Loyfer Lung_cells | +2.6397 | POSITIVE FLAGGED |
| Loyfer Breast | +2.6187 | POSITIVE FLAGGED |
| Loyfer Head_and_neck_larynx | +2.6045 | POSITIVE FLAGGED |
| Loyfer Hepatocytes | +2.5078 | POSITIVE FLAGGED |
| Loyfer Pancreatic_acinar_cells | +2.5050 | POSITIVE FLAGGED |
| Loyfer Prostate | +2.4491 | POSITIVE FLAGGED |
| Loyfer Colon_epithelial_cells | +2.3417 | POSITIVE FLAGGED |

**This is not a cross-tissue biology finding.** It is the substrate-distribution-mismatch artifact described above — bulk-tissue WGBS references on mucosal-cohort substrate produce uniformly inflated A-scores that do not differentiate among non-cohort tissue classes. The Loyfer Bladder tile's POSITIVE +1.91 sits within this same inflated band (it is actually the *lowest* of the solid-tissue Loyfer tile readings, suggesting the residual signal is the bladder-specific component minus the substrate-mismatch baseline).

DISC-BLADDER-003 propagates this finding to LESSONS_LEARNED.md: **Bulk-WGBS atlases on mucosal-cohort substrates produce inflated cross-tile A-scores from substrate-distribution mismatch alone. Multi-atlas readings on mucosal cohorts must include a gene-promoter sub-cell-type atlas as the primary cell-of-origin reader.**

---

## BladderRef microenvironment tiles (CCL-039 POSITIVE expected)

| Tile | d_paired | 95% CI | p | Direction | CCL-039 expectation | Match |
|---|---|---|---|---|---|---|
| BladderRef EC (vascular endothelial) | +0.4069 | — | 0.077 | POSITIVE | POSITIVE | ✓ |
| BladderRef Fib (fibroblast stromal) | +0.3691 | — | 0.106 | POSITIVE | POSITIVE | ✓ |
| BladderRef IC (intra-bladder immune) | +0.5905 | — | 0.014 | POSITIVE | POSITIVE | ✓ |

All three BladderRef microenvironment tiles fire POSITIVE consistent with CCL-039 expectation (tumor microenvironment architectural complexity in EC, Fib, IC compartments). |d_paired| values range +0.37 to +0.59, above the magnitude-fire threshold of 0.30. The pattern is consistent with prostate VAL-118 BladderRef-analog finding (BE/EC/Fib/Leu/SM all-positive in prostate tumor microenvironment).

**Combined BladderRef pattern:** Epi NEGATIVE (cell-of-origin program loss) + EC/Fib/IC POSITIVE (microenvironment expansion). This is the canonical CCL-039 pattern for an epithelial-origin tumor with microenvironmental immune and stromal infiltration.

---

## Caggiano CelFiE TIM headline tiles

Caggiano TIM is reported in detail in `VAL-121_results.json` under `contrasts['caggiano:*']`. Highlights consistent with VAL-122 Stage 3 (broad immune infiltration):

| Tile | d_paired | Direction |
|---|---|---|
| Caggiano monocyte | (see results JSON) | POSITIVE expected per Chen 2022 mdNLR |
| Caggiano tcell | (see results JSON) | mixed per VAL-122 |
| Caggiano fibroblast | (see results JSON) | POSITIVE expected per CCL-039 stromal |

---

## What VAL-121 unblocks and what it does not

### Unblocks
- bladder-epic v0.1 card v0.1 ships with explicit Stage 2 dual-atlas direction-divergence as the primary finding.
- DISC-BLADDER-003 propagates to LESSONS_LEARNED.md (bulk-WGBS atlases on mucosal cohorts; gene-promoter sub-cell-type atlas required).
- v0.2 promotion path: Loyfer atlas application to bladder cohort requires substrate-distribution-aware A-score normalization (or restriction to mucosal-tile-only reading); BladderRef Epi remains the production cell-of-origin tile for bladder.

### Does NOT unblock
- Single-atlas Stage 2 reading on bladder. The dual-atlas direction divergence demonstrates that single-atlas readings on mucosal cohorts can be substrate-substitution-fooled. Production scoring requires multi-atlas convergence-or-divergence reporting.

---

## Comparison to VAL-118 prostate Stage 2 precedent

| Cohort | Cell-of-origin tile (atlas) | d_paired | Direction | CCL-039 match |
|---|---|---|---|---|
| VAL-118 prostate (sealed) | LE (Loyfer prostate-relevant tile via composite) | −1.78 | NEGATIVE | ✓ |
| **VAL-121 bladder (this VAL)** | **BladderRef Epi (gene-promoter)** | **−1.46** | **NEGATIVE** | **✓** |
| VAL-121 bladder (this VAL) | Loyfer Bladder (bulk WGBS) | +1.91 | POSITIVE | ✗ (substrate mismatch) |

The bladder gene-promoter cell-of-origin paired contrast (BladderRef Epi d=−1.46) is structurally consistent with the prostate gene-promoter cell-of-origin paired contrast (LE d=−1.78). The cell-of-origin biology is producing the expected CCL-039 NEGATIVE direction in both cancer types when the atlas resolution is sub-cell-type gene-promoter. **The bulk-WGBS Loyfer reading on bladder is a substrate-class artifact, not a cell-of-origin finding.**

---

## Audit chain

This outcome seals against:
- bladder-epic v0.1 Phase 0 cohort survey (signed off 2026-04-30)
- VAL-112 layered Moss+Loyfer calibration anchor (sealed 2026-04-29)
- VAL-113 Caggiano CelFiE TIM calibration anchor (sealed 2026-04-29)
- VAL-119 EpiSCORE BladderRef calibration anchor (sealed 2026-05-01T03:46:00Z)
- VAL-120 Stage 1 Xu-538 (sealed 2026-05-01T04:35:00Z; this VAL sealed in patient-flow order)
- Calibration TODO v0.5 Phase C run-everything requirement
- Guardrails #11 + #12 (calibration before testing; run-everything Phase C)
- CCL-039 (cell-of-origin direction expectation)
- CCL-041 (prereg locked before β read; amendment 002 sealed AFTER β read with honest disclosure)
- CCL-046 (atlas selection traces to canonical-document-named candidates)
- CCL-049 (multi-atlas reporting)
- CHK-2.7 (magnitude-based |d| with direction labels)
- CHK-3.2 (cross-tile sanity check — confirmed substrate mismatch)
- DISC-BLADDER-002 (CHK-3.1A tissue-class floors)
- **DISC-BLADDER-003 (bulk-WGBS atlases on mucosal-cohort substrates)** — new this VAL
- DISC-PROSTATE-001 + DISC-PROSTATE-002 (gene-promoter atlas family + cell-of-origin direction-flip discipline)

**No outcome added post-hoc. No magnitude threshold relaxed. No direction-label rule relaxed. The CHK-3.1A floor was corrected to match tissue class with full honest disclosure; the outcome class O2 fires as locked in the original prereg.**

---

## Reproducibility triple (CHK-7.6)

### Source code
- `unified_phaseC_runner.py` (parent directory) — single-pass runner; runtime 270.7 sec for n=440 cohort.
- `postpass_amended.py` (parent directory) — paired/Welch d, outcome class.

### Inputs
1. **Layered Moss+Loyfer atlas:** `Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv` (6,105 CpGs × 25 tiles, calibration anchor VAL-112).
2. **EpiSCORE BladderRef CpG-bridged:** `Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_bladderref/episcore_bladderref_cpg_bridged.csv` SHA-256 `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6` (calibration anchor VAL-119).
3. **Caggiano CelFiE TIM array-bridged:** `Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv` (calibration anchor VAL-113).
4. **TCGA-BLCA cohort:** 440 sesame Level 3 .txt files. Manifest at `bladder_epic/blca_manifest.json`.

### Headline outputs
- `VAL-121_results.json` — full per-(atlas, tile) contrasts, CHK-3.2 cross-tile sanity flags, sealed outcome.
- `VAL-121_per_sample_per_atlas.csv` — 440 rows × 25+4+19 tile A-score columns plus QC fields.
- `VAL-121_cross_tile_sanity.json` — Loyfer non-bladder tile flagging detail.

---

**Outcome sealed 2026-05-01T04:35:00Z. The dual-atlas direction divergence is the v0.1 finding, and DISC-BLADDER-003 is the cookbook lesson it teaches.**
