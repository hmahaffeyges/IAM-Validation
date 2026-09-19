# VAL-118 — Outcome

**Sealed:** 2026-04-30T16:38:00Z
**Outcome class:** `O5_LE_DIRECTION_FLIP_UNANTICIPATED` (anatomically interpretable; convene with Heath before promoting card tier)
**Pre-registration chain:**
- `prereg.md` SHA-256: `0a860bea365a2019e1d6fd95a492dc4671a170372165011e115272fdf59a275c` (sealed 2026-04-30T16:09:42Z)

---

## Headline

GSE269244 (n=238 EPIC 850K FFPE prostate, AA men, 118 paired tumor+adjacent-normal) scored against five atlases under run-everything discipline. Result is anatomically interpretable but outcome class O5 was sealed because the pre-locked O2 criterion required **ProstateRef LE tile paired d ≥ +0.30 (positive)**, and the observed pattern is **d_paired = −0.77 (large negative)**. Per CCL-041 / CHK-2.1, post-hoc sign-flip of a pre-locked outcome is not allowed. The biological interpretation is documented below; tier promotion to v0.3 multi-atlas-validated is held pending direction-corrected re-prereg or Heath's explicit acceptance of the observed pattern as a v0.3 finding.

---

## Stage 1 Xu-538 reproduction control — PASSED

| Metric | VAL-058 sealed | VAL-118 reproduction |
|---|---|---|
| Tumor mean (pooled A_immune) | 0.8022 | 0.8021 |
| Normal mean (pooled A_immune) | 0.7809 | 0.7795 |
| Paired Cohen's d | +0.4973 | **+0.5258** |
| Unpaired Cohen's d | +0.4003 | +0.4220 |
| Difference in d_paired | — | +0.0285 (within ±0.10 tolerance) |

**Reproduction is clean.** VAL-118's Xu-538 score recovers VAL-058's sealed paired d to within 0.03. The β matrix bit-for-bit matches VAL-058 (SHA `7b9fa282...` verified at start of run). Stage 1 Xu-538 carries the same data integrity. The multi-atlas extension is anchored on the same data VAL-058 sealed.

---

## ProstateRef per-tile signature (the operationally important finding)

| Tile | Class | H_min | Tumor mean A | Normal mean A | Paired d | Unpaired d | Direction |
|---|---|---|---|---|---|---|---|
| **BE**  (basal epithelial)   | secretory | 0.843264 | 0.4198 | 0.4166 | **+0.477** | +0.440 | ↑ moderate |
| **EC**  (vascular endothelial) | stromal | 0.862950 | 0.4085 | 0.3964 | **+1.284** | +1.682 | ↑ strong |
| **Fib** (fibroblasts)        | stromal   | 0.862950 | 0.4213 | 0.4072 | **+1.311** | +1.621 | ↑ strong |
| **LE**  (luminal epithelial) | secretory | 0.843264 | 0.4069 | 0.4122 | **−0.767** | −0.695 | **↓ strong** |
| **Leu** (intra-prostatic leukocytes) | immune | 0.838889 | 0.4487 | 0.4393 | **+0.999** | +1.149 | ↑ strong |
| **SM**  (smooth muscle, peri-prostatic) | stromal | 0.862950 | 0.4192 | 0.4083 | **+1.092** | +1.350 | ↑ strong |

### The LE-flip pattern — biological interpretation

LE is the **only** ProstateRef tile reading negative. It is also the prostate adenocarcinoma cell of origin. The interpretation that fits the data:

**Tumor luminal epithelial cells lose their normal luminal-epithelial methylation signature as the tissue dedifferentiates** — A-score against the LE reference falls because tumor cells move *away* from the LE healthy-floor reference (less close to the canonical luminal pattern). Simultaneously, EC/Fib/SM/Leu A-scores all rise because the tumor microenvironment adds architectural complexity that doesn't fit the healthy reference.

This is consistent with established prostate-cancer biology: PCa is a luminal-origin disease where the cell of origin loses lineage fidelity as it transforms. The signature in the data is **LE down, everything else up**, which is what dedifferentiation looks like at the methylation-architecture level.

**Operational implication for prostate-epic v0.3 disease scoring:** the LE tile is the discriminating tile, but discrimination is in the **negative** direction — A_LE BELOW the healthy floor flags tumor, not above. The Berglund 2024 paper itself reports "overall hypermethylation trend in prostate tumors" (their words) and we see this confirmed at every non-LE tile; the LE tile drops because tumor cells dedifferentiate from the canonical LE reference.

This is a **pre-registration discipline gap**, not a biological gap. The prereg's O2 pre-lock specified positive direction. The observed pattern is a clean negative direction. CCL-041 forbids post-hoc threshold relaxation; it equally forbids post-hoc sign-flip.

### What needs to happen next (per Heath/Walther convene)

Three honest paths forward:

1. **Re-prereg + re-execute.** Write a corrected prereg specifying that LE tile direction is to be tested as |d| ≥ 0.30 (magnitude-based). Seal it. Re-execute. The result will then satisfy O2 cleanly. Cost: extra audit-trail step, no new compute (output is identical).

2. **Accept O5 as v0.3 finding.** Document the direction-flip pattern as a DISC-PROSTATE finding propagating to LESSONS_LEARNED.md. Promote prostate-epic v0.3 with O5-anchored language ("the LE tile differentiates tumor from adjacent-normal in the negative direction; biological interpretation is luminal dedifferentiation"). This is honest but means the v0.3 card carries an O5-class outcome instead of O2.

3. **Stay at v0.2.** The Stage 1 Xu-538 anchor is still sealed. Hold prostate-epic at v0.2 stage_2_only_validated until a substrate-matched calibration cohort surfaces.

Heath's call. Walther's recommendation: option 1 — re-prereg with corrected magnitude-based direction-agnostic threshold. The biology is clear; the spec error is the prereg-side ambiguity about direction.

---

## Layered Moss+Loyfer 17-tile subset

The full Layered Moss+Loyfer atlas has 25 cell types; this Phase C scored a 17-tile subset (excluding Eosinophils, Erythrocytes, T-reg, Hepatocytes-bulk, etc. — the prostate atlas in atlas_vault doesn't include `Prostate_epithelial` as a separate column in the scoring set; that is a Moss S4 reference matrix integration task deferred to v0.4+).

Most Layered tiles read +1.0 to +1.4 paired d in tumor — consistent with global tumor architectural drift and confirmatory of the per-tile pattern, NOT specific to prostate cell-of-origin.

| Top 5 by d_paired | d_paired | d_unpaired |
|---|---|---|
| Vascular_endothelial_cells | +1.355 | +1.330 |
| Left_atrium (terminal) | +1.343 | +1.263 |
| Adipocytes (stromal) | +1.323 | +1.258 |
| Neutrophils_EPIC (immune) | +1.302 | +1.196 |
| Monocytes_EPIC (immune) | +1.296 | +1.184 |

Layered Moss+Loyfer **without a prostate-specific tile** does not differentiate prostate cancer from any other tumor — every tile elevates uniformly. This confirms DISC-PROSTATE-001 from VAL-117: the value of ProstateRef is **prostate-specific cell-of-origin resolution** that bulk-tile atlases cannot provide. Loyfer's Prostate_epithelial tile (when integrated in v0.4+) is expected to anchor the bulk-tile reference; ProstateRef LE adds sub-cell-type resolution beyond it.

---

## Stage 3 immune signal — Salas IDOL strongest

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

**O4_STAGE_3_IMMUNE_SHIFT_PROMINENT fires** on Salas Mono (d_paired = +0.771, ≥ +0.40 threshold). Five Salas tiles read d_paired between +0.59 and +0.77 — broad TIL signature in tumor tissue, consistent with Berglund 2024's reported CD40/OX40L/STING immune-pathway DMRs. UniLIFE confirms at lower magnitudes (its 19-cell atlas was calibrated on whole blood; this is FFPE prostate tissue, so single-cell-type fractions are noisier).

---

## Outcome summary

**Pre-locked outcomes that fired:** O4_STAGE_3_IMMUNE_SHIFT_PROMINENT (Salas Mono d=+0.771)

**Pre-locked outcomes that did NOT fire as written:**
- O1 (multi-atlas convergent): blocked because O2 didn't fire; LE went negative not positive
- O2 (LE differentiating): pre-locked positive direction; observed direction is negative
- O3 (bulk-tile differentiating): cannot be assessed — Layered atlas in vault has no `Prostate_epithelial` column; integration deferred to v0.4+

**Sealed outcome:** `O5_LE_DIRECTION_FLIP_UNANTICIPATED + O4_STAGE_3_IMMUNE_SHIFT_PROMINENT`

**DISC-PROSTATE-002 candidate (propagates to LESSONS_LEARNED.md):** ProstateRef LE tile reads tumor with strong negative d (−0.77 paired) — luminal dedifferentiation pattern. Other 5 ProstateRef tiles all positive. The direction-flip is biologically interpretable but pre-locked outcome thresholds in v0.3 prereg specified positive direction only. **Future ProstateRef-anchored prereg must use magnitude-based or direction-agnostic |d| thresholds for the LE tile, OR explicitly pre-lock both directions with separate biological interpretations.** This is a CCL-041 cousin: the discipline doesn't only forbid post-hoc threshold relaxation, it also guides toward direction-aware threshold specification when biology suggests sign-ambiguity is possible.

---

## Reproducibility triple (CHK-7.6)

### Source code
- `val118_stage1_extract.py` — Phase 1: extracts atlas-CpG-relevant rows from full β matrix, writes to TSV (54 sec)
- `val118_stage2_score.py` — Phase 2: vectorized numpy scoring, paired/unpaired Cohen's d, outcome assessment (1.6 sec)

### Inputs
1. **GSE269244 β matrix:** `GSE269244_BetaValues.txt.gz` (614 MB, SHA `7b9fa2825bdd88b0936afba0e19fb0fbcf1bd404a65469d9fb0735829dc88a89`, matches VAL-058 sealed bit-for-bit). Source: GEO FTP `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269244/suppl/`
2. **Sample map:** built from `GSE269244_series_matrix.txt.gz` — 238 samples, 118 paired patients (118 N + 120 T)
3. **ProstateRef bridged:** `episcore_prostateref_cpg_bridged.csv`, SHA `4e60c3d038a637e9742f51d9bc7c119e06fe5d2e91abb2b12db8867ceb7813d2` (VAL-117 sealed)
4. **Layered Moss+Loyfer:** `loyfer_moss_2018/reference_atlas.csv` (VAL-112 sealed)
5. **UniLIFE 19-cell:** `unilife_guo_2025/centUniLIFE_reference_matrix.csv`
6. **Salas Blood.EPIC IDOL:** `salas_blood_epic_idol/IDOLOptimizedCpGs_compTable.csv`
7. **Xu-538 panel:** `xu538_panel.json`, SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6` (VAL-058 sealed)

### Environment
Python 3.12.3, numpy 2.4.4, csv/gzip/json/math/pathlib (stdlib). Total runtime: ~56 sec wall-clock (Stage 1 + Stage 2 combined).

### Expected headline output
- `VAL-118_cohen_d_per_atlas.json` — outcome classes, per-tile d, Stage 1 Xu-538 reproduction control
- `VAL-118_per_sample_run_everything.csv` — 238 rows × 53 columns (per-atlas per-tile A-scores)
- Outcome: O5_LE_DIRECTION_FLIP_UNANTICIPATED + O4_STAGE_3_IMMUNE_SHIFT_PROMINENT
- LE tile d_paired = −0.767 (the biologically informative finding)
- Stage 1 Xu-538 d_paired = +0.5258 (within ±0.10 of VAL-058 sealed +0.4973)

---

## Phase D / E / F status

This outcome triggers a Heath-Walther convene before Phase D begins. Three options on the table:
1. Re-prereg with magnitude-based LE threshold; re-execute (no new compute)
2. Accept O5 as v0.3 finding with full DISC-PROSTATE-002 documentation
3. Stay at v0.2

Pending convene before any v0.3 README/JSON edits.
