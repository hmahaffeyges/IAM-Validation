# VAL-118 — Pre-Registration

**VAL ID:** VAL-118
**Card target:** prostate-epic v0.3 (Phase C re-scoring under run-everything discipline)
**Cohort:** GSE269244 (Berglund/Yamoah/Kresovich 2024 PMID 39162297) — n=238 EPIC 850K FFPE prostate tissue, African-American men, 118 paired tumor + adjacent-normal + 2 tumor-only
**β matrix SHA-256:** `7b9fa2825bdd88b0936afba0e19fb0fbcf1bd404a65469d9fb0735829dc88a89` (matches VAL-058 sealed bit-for-bit; same file from GEO public deposit)
**Sealed:** [SEAL_TIMESTAMP at execution]
**Sealed BEFORE β read:** YES

---

## Question

Under EDEAR's run-everything discipline (Guardrail #12 + 2026-04-26 sign-off), every IDAT scores against every atlas in the card's `atlases_run` block. VAL-058 (sealed 2026-04-24) covered ONE atlas: Stage 1 Xu-538. VAL-118 extends to ALL atlases the prostate-epic v0.3 card will declare:

1. **Stage 1 Xu-538** (already done in VAL-058; re-runs as control)
2. **EpiSCORE ProstateRef CpG-bridged** (calibration anchor VAL-117, sealed 2026-04-30) — 6 prostate cell types
3. **Layered Moss+Loyfer** (calibration anchor VAL-112) — 25 cell types including `Prostate_epithelial` tile
4. **UniLIFE 19-cell Stage 3** (Guo 2025, on disk in atlas_vault) — fine-grained immune subset
5. **Salas Blood.EPIC IDOL Stage 3** (production atlas) — coarse 6-class immune

**The critical question Phase C answers:** Does the prostate signal in this cohort converge under multi-atlas triangulation, or does it diverge in a way that exposes single-atlas confounders?

---

## Why this matters operationally

VAL-058 sealed Stage 1 paired d=+0.497 (p=0.0001) on Xu-538 immune. Under the v0.2 card, this was the only number prostate-epic claimed. v0.3 needs to know:

- Does **ProstateRef LE tile** (luminal epithelial — prostate adenocarcinoma cell of origin) separate tumor from adjacent-normal, and if so by how much vs the other 5 sub-tiles (BE/EC/Fib/Leu/SM)? This is the operationally most-important scoring tile for post-treatment monitoring use case (the wife's-uncle case).
- Does **Layered Moss+Loyfer Prostate_epithelial tile** read consistently with VAL-058?
- Does **UniLIFE Stage 3** show fine-grained immune lineage shifts in tumor vs normal?
- Does **Salas IDOL Stage 3** corroborate UniLIFE at coarser resolution?

Multi-atlas convergence = clean signal; multi-atlas divergence = single-atlas confounder, requires CCL-049 multi-atlas reporting flag.

---

## Atlas inventory (every atlas Phase C scores)

| Atlas | n_CpGs | n_tiles | Calibration anchor VAL | Substrate notes |
|---|---|---|---|---|
| Xu-538 (Stage 1 immune) | 538 | 1 (pooled) | VAL-058 self-cal | EPIC 80% coverage (481/538) |
| ProstateRef CpG-bridged | 2,603 | 6 (BE/EC/Fib/LE/Leu/SM) | VAL-117 (TCGA HM450K sesame Level 3 anchor) | EPIC: cross-substrate documentation required per CCL-041 |
| Layered Moss+Loyfer | 6,105 | 25 cell types (incl. Prostate_epithelial) | VAL-112 (TCGA HM450K sesame Level 3 anchor) | EPIC: cross-substrate documentation required |
| UniLIFE 19-cell | 1,906 | 19 immune cell types | None yet (VAL-115 reserved) | EPIC: this VAL doubles as UniLIFE smoke-test on prostate substrate |
| Salas Blood.EPIC IDOL | 450 | 6 (B/CD4T/CD8T/Mono/Neu/NK) | Production atlas | EPIC native |

**CCL-041 substrate-platform note.** Phase B calibrations are anchored on TCGA HM450K sesame Level 3 (the same calibration cohort used for cardio-epic). GSE269244 is **EPIC 850K** (Illumina Methylation EPIC V1 from FFPE prostate). This is a substrate mismatch from the Phase B calibration anchor. Per CCL-041 / DISC-CARDIO-005, this Phase C run is documented as **within-cohort self-calibrated** for EPIC 850K substrate; the TCGA HM450K calibration anchors are used as cross-substrate reference, not as direct production thresholds. The DISC-CARDIO-005 lesson (substrate envelopes work but are not generalizable across pre-processing platforms) applies. Future v0.4+ work surfaces a structurally-separated EPIC 850K healthy prostate cohort to anchor a Phase B calibration that generalizes; until then, the v0.3 EPIC scoring is Phase C self-cal anchored.

**CHK-3.1A baseline check.** Each atlas's CHK-3.1A on this cohort produces an EPIC 850K FFPE substrate baseline. Reported in results JSON for documentation; not used as a production threshold.

---

## Pre-locked outcomes

Per CHK-2.1 (all outcomes pre-locked):

### O1 — `MULTI_ATLAS_CONVERGENT`
ProstateRef LE tile paired d ≥ +0.30 AND Layered Moss+Loyfer Prostate_epithelial tile paired d ≥ +0.30 AND VAL-058 Stage 1 Xu-538 paired d reproduces within ±0.10 of sealed +0.497. Multi-atlas convergence on prostate-tile signal. v0.3 card promotes to `multi_modal_validated + multi_atlas_calibrated`.

### O2 — `LE_TILE_DIFFERENTIATING`
ProstateRef LE tile paired d ≥ +0.30 BUT Layered Moss+Loyfer `Prostate_epithelial` tile paired d < +0.20. ProstateRef sub-cell-type resolution adds discrimination beyond bulk tile. Operationally important: post-treatment monitoring use case benefits from LE-specific scoring.

### O3 — `BULK_TILE_DIFFERENTIATING`
Layered Moss+Loyfer `Prostate_epithelial` tile paired d ≥ +0.30 BUT ProstateRef LE tile paired d < +0.20. Reverse pattern: bulk tile carries signal, sub-cell-type resolution doesn't add. Calls into question DISC-PROSTATE-001 finding (gene-promoter sub-cell discrimination).

### O4 — `STAGE_3_IMMUNE_SHIFT_PROMINENT`
UniLIFE 19-cell OR Salas IDOL show paired |d| ≥ +0.40 on any single immune tile (TIL-driven signal in tumor tissue). Documents intra-prostatic immune compartment shift. Not mutually exclusive with O1/O2/O3.

### O5 — `MULTI_ATLAS_DIVERGENT`
ProstateRef and Layered atlases disagree by paired d > 0.50 in opposite directions, OR Stage 1 Xu-538 reproduces at paired d that disagrees with VAL-058 sealed by > 0.20. Single-atlas confounder pattern. Triggers CCL-049 multi-atlas reporting flag. Card delays v0.3 promotion pending DISC-PROSTATE finding investigation.

### O6 — `UNEXPECTED`
Anything not anticipated above. Convene before sealing.

**Multiple outcomes can fire simultaneously** (e.g., O1 + O4). Outcome class block in results JSON is a list, not a single value.

---

## Pre-locked thresholds

| Threshold | Pre-locked value | Source |
|---|---|---|
| LE tile differentiating d_paired | ≥ +0.30 | One-quarter of the Phase B calibration within-cohort range (0.0293) translated to standardized d via ProstateRef SD; conservative proxy for "above floor noise" |
| Bulk Prostate_epithelial differentiating d_paired | ≥ +0.30 | Same |
| Stage 1 reproduction tolerance | ±0.10 of VAL-058 sealed +0.497 | Tightest SHA-locked replication window |
| Stage 3 immune-shift threshold | ≥ +0.40 paired |d| | Standard Cohen's d "moderate-to-large" |
| Multi-atlas divergence threshold | > 0.50 paired d in opposite directions | Standard "structurally inconsistent" threshold |

---

## CHK gate plan per atlas

Per atlas, on the GSE269244 EPIC 850K substrate:
- **CHK-3.1A** (full-genome substrate baseline): EPIC 850K FFPE substrate is a new substrate for this card. Documented, not gated. Hard fail if all-NaN or median β > 0.95 (single-tone failure).
- **CHK-3.1B** (atlas-subset coverage): per-sample atlas-CpG-intersection coverage reported. Pre-locked threshold ≥ 80% per amended VAL-117 protocol (TCGA HM450K substrate floor; expected to be HIGHER on EPIC 850K due to better probe coverage of bridged matrices).
- **CHK-3.1C** (atlas dedup): each atlas has been sealed dedup-passed in its calibration VAL — pass-through.
- **CHK-3.2** (cross-cohort baseline): per-tile healthy-mean A-score from this Phase C compared to VAL-117 / VAL-112 healthy-floor distributions. Documented as cross-substrate baseline shift; not gated (this is the v0.3 finding documentation).

---

## Reproducibility triple (CHK-7.6)

### Source code
`val118_prostateref_phaseC.py` — Python 3.12 stdlib + numpy. Loads β matrix in chunked CpG passes (avoids loading 6 GB raw); per-atlas filtering then per-sample A-score; SHA-verifies all atlas inputs at start.

### Inputs
1. **GSE269244 β matrix:** `GSE269244_BetaValues.txt.gz`, 614 MB, SHA-256 `7b9fa2825bdd88b0936afba0e19fb0fbcf1bd404a65469d9fb0735829dc88a89` (matches VAL-058 sealed bit-for-bit). Source: GEO FTP `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE269nnn/GSE269244/suppl/`
2. **Sample map:** `gse269244_sample_map.json` (built from series matrix `GSE269244_series_matrix.txt.gz`, SHA `3450e486...`). 238 samples, 118 paired patients.
3. **ProstateRef bridged:** `episcore_prostateref_cpg_bridged.csv`, SHA `4e60c3d038a637e9742f51d9bc7c119e06fe5d2e91abb2b12db8867ceb7813d2` (calibration anchor VAL-117 sealed)
4. **Layered Moss+Loyfer:** `loyfer_moss_2018/reference_atlas.csv` (calibration anchor VAL-112 sealed)
5. **UniLIFE:** `unilife_guo_2025/centUniLIFE_reference_matrix.csv` (Guo 2025 Genome Med)
6. **Salas IDOL Stage 3:** `salas_blood_epic_idol/IDOLOptimizedCpGs_compTable.csv`
7. **Xu-538 panel JSON:** SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6` (matches VAL-058)

### Environment
Python 3.12.3, numpy 2.4.4, csv/gzip/hashlib/json/math (stdlib). Expected runtime: 15-20 min on this filesystem. Expected memory: 2-3 GB peak (chunked β read + atlas matrices held).

### Expected headline output
- `VAL-118_per_sample_run_everything.csv` — 238 rows × ~50 columns (per-atlas per-tile A-scores, CHK-3.1A/B per atlas)
- `VAL-118_cohen_d_per_atlas.json` — per-atlas per-tile paired/unpaired d, CIs, perm p
- `VAL-118_outcome.md` — sealed outcome class(es), Stage 1 Xu-538 reproduction check, headline d table

---

## Stage 1 Xu-538 reproduction control

Per CCL-038 audit-trail discipline: VAL-118 re-runs Stage 1 Xu-538 on the same β matrix as VAL-058. Stage 1 paired d should reproduce within ±0.10 of sealed +0.497. If it does, the multi-atlas extension carries the same data integrity as VAL-058. If it doesn't, the data pipeline differs from VAL-058 in some way and Phase C cannot proceed without isolating the discrepancy.

This is a built-in self-check, not a separate outcome.

---

## RNG seed

20260420 (cookbook standard).

---

## SHA-256 of this prereg

To be computed at seal time and recorded in PREREG_SEAL.txt before val118 script reads any β values.

---

## CCL-041 honored

Phase B calibration sealed BEFORE Phase C scoring (VAL-117 sealed 2026-04-30 before this VAL-118 prereg seals). No threshold relaxation post-hoc. EPIC 850K substrate self-cal documented up-front as v0.3 acknowledged limitation.
