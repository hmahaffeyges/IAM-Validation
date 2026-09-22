# VAL-122 — Pre-Registration

**VAL ID:** VAL-122
**Card target:** bladder-epic v0.1 (Phase C — Stage 3 immune fine-tune)
**Substrate cohort:** TCGA-BLCA (n=440), Illumina HM450K sesame Level 3 from GDC
**Sealed:** [SEAL_TIMESTAMP at execution]
**Sealed BEFORE β read:** YES

---

## Question

After Stage 1 (Xu-538 immune red flag) and Stage 2 (multi-atlas cell-of-origin), Stage 3 fine-tunes the immune signature using the patient flow's last layer:

1. **Salas Blood.EPIC IDOL 6-cell** (production atlas) — coarse 6-class immune (B cells, CD4T, CD8T, Mono, Neu, NK)
2. **UniLIFE Guo 2025 19-cell** (atlas vault Queue-1) — fine-grained immune subset across lifespan
3. **Caggiano CelFiE TIM immune subset** (calibrated VAL-113) — immune cell types: dendritic, eosinophil, erythroblast, macrophage, monocyte, neutrophil, tcell, megakaryocyte

Does Stage 3 reveal lymphoid-vs-myeloid shifts in TCGA-BLCA tumor vs adjacent-normal that fine-tune the Stage 1 immune red flag? Specifically: do CD8T, Mono, Neu fractions shift in directions consistent with the Chen 2022 NMIBC blood EPIC n=603 finding (CD4T+CD8T = decreased hazard, monocyte + mdNLR = increased hazard)?

---

## Why this matters operationally

The patient flow third stage is what tells the clinician whether the bladder Stage 1 + Stage 2 finding has a tractable immune-architecture interpretation. Specifically:

- **Lymphoid-vs-myeloid split.** Bladder cancer literature (Chen 2022) shows tumor-recurrence-free-survival correlates with elevated lymphoid (CD4T, CD8T) and reduced myeloid (Mono, Neu) fractions. Stage 3 fine-tune tells us whether this same signature shows up in the tumor-vs-adjacent-normal contrast.
- **Inflammaging vs neoplastic immune drift.** Per heme-LL-005, age-driven inflammaging produces a different immune-architecture signature than tumor-driven immune-architecture. UniLIFE 19-cell is uniquely positioned to surface this distinction because it spans birth-to-old-age immune cell composition reference.
- **Prostate VAL-118 precedent.** Prostate Stage 3 Salas IDOL Mono d_paired = +0.771 (broad TIL infiltration consistent with Berglund 2024 published CD40/OX40L/STING DMR findings). Bladder Stage 3 needs to surface its own immune-shift pattern.

---

## Calibration discipline (CCL-041)

- **Salas Blood.EPIC IDOL 6-cell** is the EDEAR production Stage 3 atlas — production-calibrated, no per-cohort calibration required for Phase C scoring.
- **UniLIFE Guo 2025 19-cell** is Queue-1 in the atlas vault; not yet calibrated against a structurally-separated cohort. Wave 1 Shared Task B (VAL-115 reserved) is the v0.X+1 promotion path. Bladder Phase C use is **within-cohort self-cal** with v0.1 limitation documented per DISC-CARDIO-005.
- **Caggiano CelFiE TIM** is calibrated VAL-113 (sealed 2026-04-29); immune subset of TIM (8 of 19 cell types are immune lineages) reuses VAL-113 calibration anchor.

---

## Cohort + atlas inventory

### TCGA-BLCA cohort
Same as VAL-120 / VAL-121.

### Stage 3 atlases

| Atlas | n_CpGs | n_immune_tiles | Calibration | Source |
|---|---|---|---|---|
| Salas Blood.EPIC IDOL | 450 EPIC / 350 HM450 legacy | 6 (B/CD4T/CD8T/Mono/Neu/NK) | Production calibrated | Salas et al. 2018 *Genome Biol* PMC6012921 |
| UniLIFE Guo 2025 19-cell | 1,906 | 19 (lifespan-spanning) | Within-cohort self-cal at v0.1; VAL-115 v0.X+1 | Guo et al. 2025 |
| Caggiano TIM immune subset | 254 (subset) | 8 (dendritic, eosinophil, macrophage, monocyte, neutrophil, tcell, erythroblast, megakaryocyte) | VAL-113 sealed | Caggiano et al. 2021 *Genome Biol* PMC8480087 |

---

## Pre-locked outcomes

Per CHK-2.7 (magnitude-based |d| with direction labels). Cancer-vs-normal Stage 3 immune contrast direction is biology-dependent — both tumor-promoting (myeloid expansion) and tumor-suppressing (lymphoid infiltration) signatures are documented in bladder cancer literature. Direction-aware reporting per CHK-2.7.

### O1 — `STAGE_3_IMMUNE_DIFFERENTIATING`

At least 3 of the 6 Salas IDOL tiles show |d_paired| ≥ 0.30 with directionally consistent labels. Stage 3 fine-tune resolves the bladder immune signature — multi-tile immune shift in tumor vs adjacent-normal. v0.1 card claims include the specific lymphoid-vs-myeloid pattern observed.

### O2 — `STAGE_3_LYMPHOID_DOMINANT`

CD4T or CD8T tiles d_paired direction = POSITIVE with |d| ≥ 0.30 AND Mono or Neu tiles d_paired direction = NEGATIVE with |d| ≥ 0.30. Tumor shows lymphoid infiltration with myeloid reduction — consistent with Chen 2022 NMIBC blood EPIC finding for recurrence-free-survival. Direct biological corroboration with published bladder cancer immune signatures.

### O3 — `STAGE_3_MYELOID_DOMINANT`

Mono or Neu d_paired direction = POSITIVE with |d| ≥ 0.30 AND CD4T or CD8T d_paired direction = NEGATIVE with |d| ≥ 0.30. Tumor shows myeloid expansion with lymphoid suppression — consistent with myeloid-derived suppressor cell (MDSC) infiltration patterns in advanced/MIBC.

### O4 — `STAGE_3_NULL`

All 6 Salas IDOL tiles have |d_paired| < 0.30. Stage 3 fine-tune does not reach magnitude threshold under within-cohort self-cal. Direction labels reported per observation. Card v0.1 documented as Stage-3-null with v0.X+1 next step (Wave 1 calibration + larger blood cohort like Chen 2022 if accessible).

### O5 — `STAGE_3_DATA_INTEGRITY_FAILURE`

CHK-3.1A or CHK-3.1B fails on >25% of TCGA-BLCA samples on any Stage 3 atlas; or paired pair count <15. Halt and re-fetch.

### O6 — `STAGE_3_UNEXPECTED`

Anything not anticipated. Per CCL-032 classify with Heath sign-off.

---

## Pre-locked thresholds

| Threshold | Pre-locked value | Rationale |
|---|---|---|
| Magnitude threshold for "fires" | |d_paired| ≥ 0.30 | Same as VAL-118 + VAL-120 + VAL-121 |
| Direction labels | POSITIVE / NEGATIVE | CHK-2.7 |
| Multi-tile firing threshold | ≥3 of 6 Salas IDOL tiles for O1 | Robustness against single-tile noise |
| Lymphoid vs myeloid pattern (O2/O3) | At least one CD4T/CD8T tile + at least one Mono/Neu tile fire opposite directions | Chen 2022 NMIBC blood signature replication test |
| Minimum paired pairs | n ≥ 15 | Statistical power floor |
| CHK-3.1A pass rate | ≥ 75% | Phase C substrate-permissive |
| CHK-3.1B coverage per sample per atlas | ≥ 80% | CHK-2.8 substrate-floor |

---

## Statistical methodology

Per (atlas, tile) compute:
- Paired d (n=21 paired patients)
- Unpaired Welch d (418 tumor vs 21 normal)
- 95% CI, p-value
- Direction = sign(mean(paired_diff))

Multi-tile aggregate:
- Count of Salas IDOL tiles firing |d_paired| ≥ 0.30 (for O1 threshold)
- Lymphoid-vs-myeloid pattern detector (for O2/O3)

---

## Reproducibility triple (CHK-7.6)

### Source code
`val122_bladder_stage3_immune.py` — Python 3.12 + numpy + scipy.stats + pyreadr (for Salas .rda atlases). Loads three Stage 3 immune atlases. Loads TCGA-BLCA β files. Per-sample CHK-3.1A on full genome + per-atlas CHK-3.1B coverage + per-tile A-scores. Paired + Welch d per (atlas, tile). Multi-tile aggregation.

### Inputs
1. **Salas Blood.EPIC IDOL 450K legacy:** `/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs450k_compTable.csv` (350 CpGs × 6 cell types — HM450 substrate match)
2. **UniLIFE Guo 2025:** `/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv` (1,906 CpGs × 19 cell types)
3. **Caggiano TIM (immune subset):** `/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv` (immune cell types: dendritic, eosinophil, erythroblast, macrophage, monocyte, neutrophil, tcell, megakaryocyte)
4. **TCGA-BLCA β files:** `/home/claude/edear_working/bladder_epic/blca_betas/` × 440 files
5. **BLCA manifest:** `/home/claude/edear_working/bladder_epic/blca_manifest.json`

### Environment
- Python 3.12.3, numpy 2.4.4, scipy 1.17.1, pyreadr 0.5.6
- Expected runtime: ~10-15 minutes
- Expected memory: ~1 GB peak

### Expected headline output
- `VAL-122_results.json` — per-(atlas, tile) d/CI/p/direction; multi-tile aggregate
- `VAL-122_per_sample_per_atlas.csv`
- `outcome.md` — sealed outcome class

---

## RNG seed

20260420.

---

## SHA-256 of this prereg

To be computed at seal time and recorded in `PREREG_SEAL.txt` before val122 reads any β files.

---

## Pre-registered audit chain

Seals against: bladder-epic v0.1 Phase 0 cohort survey, VAL-119/120/121 chain (sealed in patient-flow order), VAL-113 Caggiano TIM calibration, Salas IDOL production calibration, Calibration TODO v0.5 Phase C, Guardrails #11+#12, CCL-039/041/046/049, CHK-2.7, DISC-CARDIO-005 (within-cohort self-cal documentation), heme-LL-005 (inflammaging vs neoplastic discrimination), Chen 2022 NMIBC blood EPIC published signature reference.

val122 script execution begins ONLY after this prereg.md is sealed and SHA-hashed.

**No outcome may be added post-hoc. No threshold may be relaxed post-hoc. No exception under CCL-041.**
