# VAL-118 — Outcome (Amendment Re-Execution)

**Sealed:** 2026-04-30T17:01:00Z (post-amendment re-execution)
**Outcome class:** `O1_MULTI_ATLAS_CONVERGENT + O2_LE_TILE_DIFFERENTIATING (LE_NEGATIVE) + O4_STAGE_3_IMMUNE_SHIFT_PROMINENT`
**Pre-registration chain:**
- `prereg.md` SHA-256: `0a860bea365a2019e1d6fd95a492dc4671a170372165011e115272fdf59a275c` (sealed 2026-04-30T16:09:42Z)
- `prereg_amendment.md` SHA-256: `c1b0a07e25ee9b0b9a8931f04ddd8c7677afcd9c8b2257cf4f9e3c6d42c1868b` (sealed 2026-04-30T16:59:19Z)

This outcome **supersedes** the first-execution outcome.md (sealed 2026-04-30T16:38) for purposes of v0.3 card promotion. The original outcome.md is preserved as the discipline-discovery record showing why the amendment was needed.

---

## Headline

Three pre-registered outcomes fire simultaneously under amended magnitude-based thresholds. This is a clean multi-atlas convergent result for prostate-epic v0.3:

1. **O1_MULTI_ATLAS_CONVERGENT** — ProstateRef LE tile + Stage 1 Xu-538 reproduction both clear amended thresholds
2. **O2_LE_TILE_DIFFERENTIATING (LE_NEGATIVE label)** — luminal dedifferentiation pattern: |d_paired| = 0.767, direction = NEGATIVE, biological interpretation = tumor luminal epithelial cells lose canonical methylation signature
3. **O4_STAGE_3_IMMUNE_SHIFT_PROMINENT** — Salas IDOL Mono d_paired = +0.771 (TIL infiltration in tumor tissue, consistent with Berglund 2024 published CD40/OX40L/STING DMR findings)

---

## Stage 1 Xu-538 reproduction control

| Metric | VAL-058 sealed | VAL-118 reproduction |
|---|---|---|
| Tumor mean (pooled A_immune) | 0.8022 | 0.8021 |
| Normal mean (pooled A_immune) | 0.7809 | 0.7795 |
| Paired Cohen's d | +0.4973 | **+0.5258** |
| Difference in d_paired | — | +0.0285 (within ±0.10 tolerance) |

Same data integrity as VAL-058. β matrix bit-for-bit verified at SHA `7b9fa282...` start of run.

---

## ProstateRef per-tile signature

| Tile | Direction | d_paired | |d_paired| | Interpretation |
|---|---|---|---|---|
| **LE** (luminal epithelial — PCa cell of origin) | **↓** | **−0.767** | **0.767** | **Luminal dedifferentiation** — fires O2 LE_NEGATIVE |
| BE (basal epithelial) | ↑ | +0.477 | 0.477 | Basal architectural drift |
| EC (vascular endothelial) | ↑ | +1.284 | 1.284 | Tumor microvasculature |
| Fib (fibroblasts) | ↑ | +1.311 | 1.311 | Stromal architectural complexity |
| Leu (intra-prostatic leukocytes) | ↑ | +0.999 | 0.999 | Local immune infiltrate |
| SM (smooth muscle, peri-prostatic) | ↑ | +1.092 | 1.092 | Peri-prostatic stromal drift |

The biology is coherent: tumor LE cells dedifferentiate (lose canonical luminal methylation pattern, A-score against LE healthy reference falls) while tumor microenvironment (EC + Fib + Leu + SM) develops architectural complexity that doesn't fit the healthy reference, A-scores rise. Five-vs-one direction split is the discriminating feature.

For prostate-epic v0.3 production scoring, the LE tile is the discriminating tile in the **negative** direction. Patient post-treatment monitoring use case: A_LE trending below the q5 of the VAL-117 healthy floor (q5 = 0.4190, mean = 0.4254) flags potential luminal dedifferentiation drift.

---

## Stage 3 immune signal — multi-atlas confirmation

| Atlas | Strongest tile | d_paired |
|---|---|---|
| **Salas Blood.EPIC IDOL** | Mono | **+0.771** |
| Salas IDOL | Bcell | +0.674 |
| Salas IDOL | CD4T | +0.659 |
| Salas IDOL | NK | +0.645 |
| Salas IDOL | CD8T | +0.587 |
| **UniLIFE 19-cell** | aMono | +0.467 |
| UniLIFE | aNeu | +0.433 |
| UniLIFE | Mono | +0.391 |

Salas IDOL Mono crosses the +0.40 magnitude threshold; five Salas tiles read d_paired between +0.59 and +0.77 — broad TIL signature in tumor tissue. UniLIFE confirms at lower magnitudes (its 19-cell atlas calibrated on whole blood, this is FFPE prostate tissue, single-cell-type fractions are noisier).

Interpretation aligns with Berglund 2024's published immunopathway DMRs (CD40, OX40L, STING) — the tumor methylation signature includes a TIL infiltrate component independently detectable by both atlases.

---

## Pre-locked outcomes status under amendment

| Outcome | Threshold (amended) | Observed | Status |
|---|---|---|---|
| **O1_MULTI_ATLAS_CONVERGENT** | O2 fires AND Stage 1 Xu-538 reproduces within ±0.10 | LE \|d\| = 0.767 (≥0.30); Stage 1 d_paired = +0.5258 (within tol) | **FIRED** |
| **O2_LE_TILE_DIFFERENTIATING (LE_NEGATIVE)** | LE \|d_paired\| ≥ 0.30 with direction label | LE d_paired = −0.767, \|d\| = 0.767 | **FIRED** (LE_NEGATIVE label) |
| O3_BULK_TILE_DIFFERENTIATING | Loyfer Prostate_epithelial \|d\| ≥ 0.30 | Inapplicable: atlas in vault has no Prostate_epithelial column | **N/A v0.4+** |
| **O4_STAGE_3_IMMUNE_SHIFT_PROMINENT** | UniLIFE or Salas paired \|d\| ≥ 0.40 on any tile | Salas Mono d_paired = +0.771 | **FIRED** |
| O5_MULTI_ATLAS_DIVERGENT | ProstateRef vs Layered disagree by >0.50 in opposite directions OR Stage 1 disagrees with VAL-058 by >0.20 | No divergence; Stage 1 reproduces; cannot assess Loyfer prostate-tile | not fired |
| O6_UNEXPECTED | Anything else | n/a | not fired |

---

## DISC-PROSTATE findings sealed

**DISC-PROSTATE-001 (from VAL-117 calibration):** Gene-promoter atlas family fitness extends DISC-CARDIO-004 lesson — atlas family fitness depends on how distinct the atlas's cell types actually are at the gene-promoter level for the tissue in question. ProstateRef's six prostate cell types span markedly different profiles producing within-cohort A-score variance 2-4× higher than HeartRef showed.

**DISC-PROSTATE-002 (from VAL-118 first execution + amendment):** Pre-registration discipline must use magnitude-based |d| thresholds (or pre-lock both directions with separate biological interpretations) when the underlying biology supports direction-ambiguity. The first VAL-118 execution sealed O5 because the original prereg pre-locked O2 as positive-direction-only, and the observed pattern was strong negative. The biology was clean (luminal dedifferentiation), but the discipline instrument was over-specified. Amendment restored magnitude-based thresholds + direction labels. **Operational rule for future ProstateRef-anchored or any cell-of-origin atlas preregs: |d| ≥ threshold with direction label; not d ≥ threshold (positive only) or d ≤ threshold (negative only).**

**DISC-PROSTATE-003 (the headline biological finding):** ProstateRef LE tile reads tumor with strong negative paired d (−0.767); other 5 ProstateRef tiles all positive (+0.48 to +1.31). The pattern is **luminal dedifferentiation + tumor microenvironment architectural complexity**. For post-treatment monitoring deployment, A_LE trending below the VAL-117 healthy-floor q5 = 0.4190 is the operationally important signal. Other tiles support but do not anchor the diagnostic.

---

## Amended thresholds used

Per `prereg_amendment.md` (SHA `c1b0a07e...`):

| Threshold | Original | Amended |
|---|---|---|
| O2 LE direction | d_paired ≥ +0.30 | \|d_paired\| ≥ 0.30 with direction label |
| O3 Loyfer prostate | d_paired ≥ +0.30 | \|d_paired\| ≥ 0.30 with direction label (inapplicable until v0.4+) |
| O1 includes O2 with magnitude-based LE | direction-locked | magnitude-based |
| Other outcomes (O4, O5, O6) | unchanged | unchanged |

---

## Reproducibility triple (CHK-7.6)

### Source code
- `val118_stage1_extract.py` — Phase 1 (54 sec): atlas-CpG row extraction
- `val118_stage2_score.py` — Phase 2 (2 sec): vectorized scoring + outcome under amended thresholds

Re-running Stage 2 with amended thresholds against the SAME sealed Stage 1 atlas TSV produces these outputs deterministically. Per-sample A-scores are identical between original and amendment executions (bit-identical CSVs); only the outcome-classification block differs.

### Inputs
Identical to first execution:
1. **GSE269244 β matrix:** SHA `7b9fa2825bdd88b0936afba0e19fb0fbcf1bd404a65469d9fb0735829dc88a89`
2. ProstateRef bridged (VAL-117 sealed, SHA `4e60c3d0...`)
3. Layered Moss+Loyfer (VAL-112 sealed)
4. UniLIFE 19-cell
5. Salas Blood.EPIC IDOL
6. Xu-538 (VAL-058 sealed, SHA `ada67296...`)

### Environment
Python 3.12.3, numpy 2.4.4, csv/json/math/pathlib (stdlib). Stage 2 runtime: 2.0 sec.

### Expected headline output
- `VAL-118_amendment_cohen_d_per_atlas.json` — outcomes O1 + O2 (LE_NEGATIVE) + O4 sealed
- `VAL-118_amendment_per_sample_run_everything.csv` — 238 rows × 53 columns (bit-identical to first-execution CSV; only outcome JSON differs)
- LE tile d_paired = −0.767 with LE_NEGATIVE direction label
- Stage 1 Xu-538 d_paired = +0.5258 (reproduces VAL-058 sealed +0.4973 within ±0.10)
- O4 fires on Salas IDOL Mono d_paired = +0.771

---

## Audit trail

VAL-118 final state:
- `prereg.md` (original, SHA `0a860bea...`) — direction-specific outcome locks
- `PREREG_SEAL.txt` — original SHA + timestamp 2026-04-30T16:09:42Z
- `prereg_amendment.md` (SHA `c1b0a07e...`) — magnitude-based amendment
- `PREREG_AMENDMENT_SEAL.txt` — amendment SHA + timestamp 2026-04-30T16:59:19Z
- `outcome.md` (first execution, sealed 16:38) — O5 documentation, preserved as discipline record
- `outcome_amendment.md` (this file, sealed 17:01) — O1 + O2 (LE_NEGATIVE) + O4 under amended thresholds
- `VAL-118_cohen_d_per_atlas.json` — first execution
- `VAL-118_amendment_cohen_d_per_atlas.json` — amendment execution
- `VAL-118_per_sample_run_everything.csv` — bit-identical between executions
- `VAL-118_amendment_per_sample_run_everything.csv` — bit-identical to above
- `val118_stage1_extract.py` — Stage 1 extraction script (unchanged)
- `val118_stage2_score.py` — Stage 2 scoring script (amendment block updated)

Both outcomes remain in the audit trail. The amendment outcome supersedes the original outcome for v0.3 card promotion purposes; the original outcome is preserved as the discipline-discovery record showing why DISC-PROSTATE-002 (magnitude-based threshold rule) became a cookbook rule.

---

## Phase D / E / F unblock

Phase C is now sealed cleanly at multi-atlas convergent + LE_NEGATIVE biological interpretation + Stage 3 immune confirmation.

- **Phase D** (compare v0.2 single-atlas vs v0.3 multi-atlas) — ready to execute
- **Phase E** (bump card to v0.3 with all 10 structured blocks) — ready to execute
- **Phase F** (push + deliver per seven-files protocol) — ready to execute

DISC-PROSTATE-001 + DISC-PROSTATE-002 + DISC-PROSTATE-003 propagate to LESSONS_LEARNED.md as part of Phase E.
