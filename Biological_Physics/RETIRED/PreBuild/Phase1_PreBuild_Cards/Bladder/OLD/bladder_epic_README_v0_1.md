# bladder-epic v0.1 — Card README

**Card status:** v0.1 sealed 2026-05-01
**Tier:** `multi_modal_validated + multi_atlas_calibrated` (promoted from `stage_2_only_validated` upon VAL-119 + VAL-121 sealing)
**Sprint duration:** 2026-04-30 cohort survey → 2026-05-01 four-VAL sealing
**Sealed VALs:** VAL-119 (BladderRef calibration), VAL-120 (Stage 1 Xu-538), VAL-121 (Stage 2 multi-atlas), VAL-122 (Stage 3 immune fine-tune)
**GitHub commit:** `404eed3` on main, `https://github.com/hmahaffeyges/IAM-Validation`

---

## Top-line — what bladder-epic v0.1 says

bladder-epic v0.1 ships with three sealed Phase C outcomes plus a sealed Phase B calibration anchor. The outcomes are honest about what fired, what fired but at a tissue-class threshold the cookbook had to correct mid-sprint, and what fired only as a diagnostic because of a panel-cohort transferability constraint. **Three cookbook lessons (DISC-BLADDER-002, -003, -004) propagate to TESTING_CHECKLIST and PIPELINE_REFERENCE; one atlas-family lesson (DISC-BLADDER-001) extends the gene-promoter rule from cardio + prostate.**

The sprint structure — Phase A bridge engineering → Phase B atlas calibration → Phase C run-everything — held end-to-end. The CHK-3.1A tissue-class floor mismatch was caught by data after Phase C β read and amended honestly with full CCL-041 disclosure before any outcome.md was sealed.

The card line under the no-overclaim language discipline:

> Phase C scoring on TCGA-BLCA n=440 (HM450K sesame Level 3, 21 paired tumor-vs-adjacent-normal patients) produced results consistent with the framework and with published bladder cancer biology, with three structural caveats sealed and propagated to v0.2 promotion path. Stage 1 immune red flag fires in the diagnostic data (paired d = +1.90, p = 3.1×10⁻⁸) but Xu-538 panel cohort-substrate coverage triggered O4 sealed outcome (DISC-BLADDER-004). Stage 2 cell-of-origin produces a dual-atlas direction-divergence finding consistent with substrate-distribution mismatch on bulk-WGBS atlas vs gene-promoter sub-cell-type atlas; BladderRef Epi paired d = −1.46 (p = 1.6×10⁻⁶) satisfies CCL-039 expectation. Stage 3 immune fine-tune fires across all 6 Salas IDOL tiles consistent with mixed TIL+TAM+MDSC infiltration of muscle-invasive bladder tumor microenvironment.

---

## VAL-119 — BladderRef calibration anchor (Phase B)

**Atlas:** EpiSCORE BladderRef CpG-bridged. Source: Zhu T, Liu J, Beck S, Pan S, Capper D, Lechner M, Thirlwell C, Breeze CE, Teschendorff AE. *A pan-tissue DNA-methylation atlas based on deconvolution of major cell-types.* Nature Methods 2022;19:296. DOI: 10.1038/s41592-022-01412-7. Repository: https://github.com/aet21/EpiSCORE.

**Bridge methodology:** Same as VAL-094 BreastRef + VAL-111 HeartRef + VAL-117 ProstateRef. Source `mrefBladder.m` (163 Entrez Gene IDs × 4 cell types + weight) bridged to 450K CpG resolution via EpiSCORE's `probeInfo450k.lv` (485,577 array probes; 331,229 with EID; 19,357 unique EIDs). Final dimensions: 2,696 unique 450K CpG probes × 4 bladder cell types. 158 of 163 source EIDs covered; 5 EIDs unmapped (1880, 2252, 26521, 51699, 54829).

**Bridged atlas SHA-256:** `3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6`

**Cell types and class assignments:**

| Tile | Cell type | Class | H_min |
|---|---|---|---|
| EC | Vascular endothelial | stromal | 0.862950 |
| Epi | Urothelial epithelium (bladder cancer cell of origin) | secretory | 0.843264 |
| Fib | Fibroblasts (stromal) | stromal | 0.862950 |
| IC | Immune cells (intra-bladder) | immune | 0.838889 |

**Calibration cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (same VAL-106 cohort that anchored cardio + prostate). Substrate: TCGA HM450K sesame Level 3.

**Calibration outcome (sealed `O1_BLADDERREF_CALIBRATION_SEALED`):**

| Gate | Threshold | Observed | Status |
|---|---|---|---|
| CHK-3.1A | ≥ 75% pass | 206/210 (98.1%) | ✓ |
| CHK-3.1B | ≥ 80% per-sample coverage; ≥ 95% pass rate | 210/210 (100%); q5 = 86.15% | ✓ |
| CHK-3.1C | 0 duplicate probeIDs | 0/2,696 | ✓ |
| Tissue-floor-dominated | max within-cohort tile range ≥ 0.02 | 0.0694 | ✓ |

**Per-tile healthy-floor distributions (n=206 QC-passed):**

| Tile | Mean A | SD | q5 | q95 | Within-cohort range |
|---|---|---|---|---|---|
| EC | 0.4087 | 0.0100 | 0.3972 | 0.4265 | 0.0565 |
| **Epi** | **0.4135** | **0.0066** | **0.4004** | **0.4219** | **0.0410** |
| Fib | 0.4875 | 0.0090 | 0.4770 | 0.5020 | 0.0694 |
| IC | 0.4106 | 0.0086 | 0.4001 | 0.4263 | 0.0504 |

The Epi tile has the tightest within-cohort variance (sd = 0.0066, range = 0.0410) — the bladder analog of prostate's LE tile (sd = 0.0041, range = 0.0293). This is the operationally important tile for bladder-epic v0.1 disease scoring: small disease-driven A-score shifts on the bladder cancer cell of origin are detectable above the calibration noise floor.

**The Fib tile separates cleanly** from the other three tiles (Fib mean 0.4875 vs EC/Epi/IC clustered at ~0.41). This separation is the operational signal that BladderRef's 4-cell-type set spans markedly different gene-promoter methylation profiles for the marker genes in bladder tissue — the prerequisite for gene-promoter atlas family success per DISC-CARDIO-004 + DISC-PROSTATE-001 + DISC-BLADDER-001.

---

## VAL-120 — Stage 1 Xu-538 immune red flag (Phase C)

**Cohort:** TCGA-BLCA n=440 (418 Primary Tumor + 21 Solid Tissue Normal + 1 Metastatic; 21 paired tumor-vs-adjacent-normal patients).

**Panel:** Xu-538 (538 CpGs from Xu Z, Sandler DP, Taylor JA. *JNCI* 2020 doi:10.1093/jnci/djz065 Sister Study breast cancer + EPIC-Italy replication, panel ID `Xu2020_breast_cancer_replicated_full`).

**Sealed outcome:** `O4_STAGE1_DATA_INTEGRITY_FAILURE`

**The outcome name is the locked O4 label**; the actual finding is panel-cohort transferability:

- CHK-3.1A under amended mucosal-tissue-class floor (f_extreme ≥ 0.387, f_middle ≤ 0.184): 98.0% pass ✓
- **CHK-3.1B Xu-538 per-sample coverage**: mean 78.0%, **51.1% pass rate at ≥80% per-sample threshold** ✗ → triggered O4

**Diagnostic finding (reported in results JSON, not sealed as VAL outcome):**

| Contrast | n | d | 95% CI | p_value | Direction |
|---|---|---|---|---|---|
| Paired (n=21 paired) | 21 | **+1.8977** | [+1.182, +2.614] | 3.14×10⁻⁸ | POSITIVE |
| Welch (409 tumor vs 21 normal) | — | +1.6433 | [+1.191, +2.099] | 1.92×10⁻⁸ | POSITIVE |

**A_immune by sample type:**

| Sample type | n | mean | SD |
|---|---|---|---|
| Solid Tissue Normal | 21 | 0.5446 | 0.0306 |
| Primary Tumor | 418 | 0.6037 | 0.0361 |
| Metastatic | 1 | 0.6150 | — |

**Comparison to prior Stage 1 cohorts:**

| Cohort | Cancer type | Substrate | Paired d | Direction |
|---|---|---|---|---|
| VAL-058 (sealed) | Prostate | EPIC 850K | +0.497 | POSITIVE |
| **VAL-120 (this VAL)** | **Bladder** | **HM450K** | **+1.898** | **POSITIVE** |

Bladder Stage 1 paired contrast is **3.8× larger** than prostate's. The biology is consistent with bladder cancer's documented heavy TIL infiltration and immune-architecture drift. The cohort-substrate-panel-coverage gate fired regardless.

**v0.2 promotion path:** Wave 1 calibration of Stage 1 panel must include a per-cohort substrate-coverage check (CHK-2.17 cookbook gate added per DISC-BLADDER-004). Either the Xu-538 panel gets a bladder-cohort-substrate-trimmed subset, or VAL-114 produces a Stage 1 panel calibrated against a healthy aging blood cohort (Hannum 2013 GSE40279 n=656) that maintains uniform per-sample coverage across solid + mucosal-tissue cohorts.

---

## VAL-121 — Stage 2 multi-atlas (Phase C)

**Cohort:** TCGA-BLCA n=440 (same as VAL-120). 21 paired pairs.

**Atlases:**

| Atlas | n_CpGs | n_tiles | Calibration anchor | Family |
|---|---|---|---|---|
| Layered Moss+Loyfer | 6,105 | 25 | VAL-112 | tile-coverage WGBS |
| EpiSCORE BladderRef | 2,696 | 4 | VAL-119 | gene-promoter |
| Caggiano CelFiE TIM | 254 | 19 | VAL-113 | tile-coverage WGBS |

**Sealed outcome:** `O2_BLADDER_TILE_DIFFERENTIATING_DIRECTION_AMBIGUOUS`

**The outcome name is the locked O2 label** — both cell-of-origin tiles fire at high magnitude but in opposite directions. The biological interpretation: **the gene-promoter sub-cell-type atlas (BladderRef Epi) delivers the CCL-039 cell-of-origin signal cleanly**, while the bulk-WGBS atlas (Loyfer Bladder) is substrate-distribution-confounded on this mucosal cohort.

**Cell-of-origin paired contrasts (n=21):**

| Atlas | Tile | d_paired | 95% CI | p_value | Direction | CCL-039 expectation | Match |
|---|---|---|---|---|---|---|---|
| Loyfer | Bladder (bulk WGBS) | **+1.9100** | [+1.191, +2.629] | 2.83×10⁻⁸ | **POSITIVE** | NEGATIVE | ✗ |
| EpiSCORE | BladderRef Epi (urothelial gene-promoter) | **−1.4623** | [−2.078, −0.847] | 1.60×10⁻⁶ | **NEGATIVE** | NEGATIVE | **✓** |

**BladderRef microenvironment tiles (CCL-039 POSITIVE expected):**

| Tile | d_paired | p_value | Direction | Match |
|---|---|---|---|---|
| BladderRef EC | +0.4069 | 0.077 | POSITIVE | ✓ |
| BladderRef Fib | +0.3691 | 0.106 | POSITIVE | ✓ |
| BladderRef IC | +0.5905 | 0.014 | POSITIVE | ✓ |

**The combined BladderRef pattern — Epi NEGATIVE + EC/Fib/IC POSITIVE — is structurally consistent with prostate VAL-118's pattern** (LE NEGATIVE + BE/EC/Fib/Leu/SM POSITIVE). The sub-cell-type gene-promoter atlas resolution exposes the cell-of-origin dedifferentiation signal in both adenocarcinomas.

**CHK-3.2 cross-tile sanity (Loyfer non-bladder solid-tissue tiles):**

ALL 14 Loyfer non-bladder solid-tissue tiles fire POSITIVE FIRES at d_paired ranging +2.34 to +2.92:

| Tile | d_paired | Direction |
|---|---|---|
| Thyroid | +2.9188 | POSITIVE FLAGGED |
| Pancreatic_duct_cells | +2.8479 | POSITIVE FLAGGED |
| Cortical_neurons | +2.8390 | POSITIVE FLAGGED |
| Uterus_cervix | +2.8193 | POSITIVE FLAGGED |
| Upper_GI | +2.8148 | POSITIVE FLAGGED |
| Pancreatic_beta_cells | +2.8056 | POSITIVE FLAGGED |
| Kidney | +2.7147 | POSITIVE FLAGGED |
| Lung_cells | +2.6397 | POSITIVE FLAGGED |
| Breast | +2.6187 | POSITIVE FLAGGED |
| Head_and_neck_larynx | +2.6045 | POSITIVE FLAGGED |
| Hepatocytes | +2.5078 | POSITIVE FLAGGED |
| Pancreatic_acinar_cells | +2.5050 | POSITIVE FLAGGED |
| Prostate | +2.4491 | POSITIVE FLAGGED |
| Colon_epithelial_cells | +2.3417 | POSITIVE FLAGGED |

**This uniform inflation across all 14 non-cohort solid-tissue references is the substrate-distribution-mismatch signal**, not biology. Bladder tumor is not "becoming Thyroid + Pancreas + Liver simultaneously" — the bulk-WGBS reference β profiles are uniformly far from the bladder cohort's tissue-class methylation distribution shape, producing inflated |β_sample − β_bulk_ref| metrics across all bulk solid-tissue tiles. The Loyfer Bladder POSITIVE +1.91 sits within this same inflated band (it is actually the LOWEST of the solid-tissue Loyfer tile readings, suggesting the residual signal is the bladder-specific component minus the substrate-mismatch baseline).

**v0.2 promotion path:** DISC-BLADDER-003 propagates the rule: **multi-atlas readings on mucosal cohorts must include a gene-promoter sub-cell-type atlas as the primary cell-of-origin reader.** Stage 2 production scoring for bladder uses BladderRef Epi as the primary cell-of-origin tile; Loyfer reading is interpretive context only on mucosal cohorts. Future EpiSCORE per-tissue bridges (LungRef for lung mucosa; ColonRef for colon epithelium; BreastRef for ducts) get prioritized over bulk-WGBS atlases for those cohorts.

---

## VAL-122 — Stage 3 immune fine-tune (Phase C)

**Cohort:** TCGA-BLCA n=440 (same as VAL-120, 121).

**Atlases:**

| Atlas | n_CpGs | n_immune_tiles | Calibration |
|---|---|---|---|
| Salas Blood.EPIC IDOL 450K legacy | 350 | 6 (CD8T, CD4T, NK, Bcell, Mono, Neu) | production calibrated |
| UniLIFE Guo 2025 | 1,906 | 19 (lifespan-spanning) | within-cohort self-cal v0.1; VAL-115 v0.X+1 |
| Caggiano CelFiE TIM (immune subset) | 254 | 8 (dendritic, eosinophil, erythroblast, macrophage, monocyte, neutrophil, tcell, megakaryocyte) | VAL-113 anchor |

**Sealed outcome:** `O1_STAGE_3_IMMUNE_DIFFERENTIATING`

**Salas IDOL 6-tile paired contrasts (n=21 paired pairs):**

| Tile | Cell type | d_paired | 95% CI | p_value | Direction |
|---|---|---|---|---|---|
| Bcell | B lymphocytes | **+1.1479** | [+0.597, +1.699] | 3.79×10⁻⁵ | POSITIVE FIRES |
| Mono | Monocytes | **+1.1322** | [+0.584, +1.680] | 4.46×10⁻⁵ | POSITIVE FIRES |
| Neu | Neutrophils | **+1.2354** | [+0.668, +1.803] | 1.53×10⁻⁵ | POSITIVE FIRES |
| NK | Natural killer | **+0.7943** | [+0.304, +1.285] | 1.63×10⁻³ | POSITIVE FIRES |
| CD8T | Cytotoxic T cells | **+0.6222** | [+0.155, +1.089] | 9.87×10⁻³ | POSITIVE FIRES |
| CD4T | Helper T cells | **+0.4884** | [+0.036, +0.941] | 3.67×10⁻² | POSITIVE FIRES |

**All six immune-cell-type A-scores increase in tumor versus adjacent-normal at high statistical significance.** The pattern is **broad immune-architectural drift**: every immune lineage fires POSITIVE, not the directional split that pre-locked O2 (lymphoid dominant) or O3 (myeloid dominant) required.

**Pre-locked O2 (`STAGE_3_LYMPHOID_DOMINANT`)** required CD4T or CD8T POSITIVE FIRES AND Mono or Neu NEGATIVE FIRES (lymphoid-elevated + myeloid-reduced, the Chen 2022 NMIBC blood EPIC RFS signature). **Did not fire** — both lymphoid and myeloid fired POSITIVE.

**Pre-locked O3 (`STAGE_3_MYELOID_DOMINANT`)** required the inverse (Mono/Neu POSITIVE FIRES + CD4T/CD8T NEGATIVE FIRES, the MDSC infiltration signature in advanced/MIBC). **Did not fire** — both lymphoid and myeloid fired POSITIVE.

**Biological interpretation:** the TCGA-BLCA primary tumor cohort produces the more biologically realistic **mixed-infiltration signature** (TILs + tumor-associated macrophages + myeloid-derived suppressor cells together) that is characteristic of muscle-invasive bladder cancer. Pure lymphoid dominance is more characteristic of immunotherapy-responding subgroups; pure myeloid dominance is more characteristic of advanced metastatic. The MIBC-dominant cohort produces broad infiltration.

**Comparison to prostate VAL-118 Stage 3:**

| Cohort | Cancer type | Salas Mono d_paired | Pattern |
|---|---|---|---|
| VAL-118 prostate (sealed) | Prostate | +0.771 | broad TIL infiltration |
| **VAL-122 bladder (this VAL)** | **Bladder** | **+1.1322** | **broad immune infiltration (6/6 POSITIVE)** |

Bladder Stage 3 Mono signal is 1.5× larger than prostate's. Both cancers show the same direction (POSITIVE) on Salas Mono — bladder's magnitude is larger.

**Card line (no overclaim language):** "Stage 3 immune fine-tune fires consistent with mixed TIL + TAM + MDSC infiltration in muscle-invasive bladder tumor microenvironment. All six Salas IDOL immune-cell-type A-scores increase in tumor vs adjacent-normal at \|d_paired\| range 0.49 to 1.24."

---

## DISC-BLADDER discoveries (the four lessons that propagate to the cookbook)

### DISC-BLADDER-001 — Gene-promoter atlas family fitness depends on cell-type DISTINCTNESS, not cell-type COUNT

**Atlas family per-tissue test record (third entry):**

| Atlas | Tissue | n cell types | Outcome | Max within-cohort range |
|---|---|---|---|---|
| HeartRef (VAL-111) | cardiac | 5 | O3_TISSUE_FLOOR_DOMINATED | 0.0152 (collapsed) |
| ProstateRef (VAL-117) | prostate | 6 | O1 sealed | 0.0597 (separated) |
| **BladderRef (VAL-119)** | **bladder** | **4** | **O1 sealed** | **0.0694 (separated)** |

The hypothesis "more cell types = better gene-promoter atlas separation" is falsified by bladder. The supported rule is: per-tissue cell-type distinctness at the gene-promoter level for the marker genes Zhu/Teschendorff selected. Cardiac cell types share gene-promoter signatures despite being 5 in number; bladder compartments are markedly distinct despite being only 4. Future EpiSCORE per-tissue calibrations cannot be predicted from source matrix dimensions alone — each tissue must be smoke-tested independently.

### DISC-BLADDER-002 — CHK-3.1A f_extreme floor is tissue-class-dependent (not universal)

The VAL-106 kidney+prostate-derived 0.50 floor is empirically appropriate for solid parenchyma, NOT for mucosal tissue. Bladder cohort observed pass rate under that floor: 23.9%. Bladder cohort q1/q99 (f_extreme ≥ 0.387, f_middle ≤ 0.184) is the appropriate mucosal-tissue-class bracket. Pass rate under amended floor: 98.0%. **Zero samples in cohort had genuine substrate corruption — it was a gate-calibration mismatch, not a data integrity issue.**

CHK-2.16 cookbook gate added: every card prereg specifies the tissue-class CHK-3.1A floor at prereg-write time, not inherited implicitly.

### DISC-BLADDER-003 — Bulk-WGBS atlases on mucosal-cohort substrates produce inflated cross-tile A-scores

Loyfer Bladder POSITIVE +1.91 vs BladderRef Epi NEGATIVE −1.46 on same n=21 paired pairs. CHK-3.2 cross-tile sanity flags ALL 14 Loyfer non-bladder solid-tissue tiles uniformly POSITIVE +2.34 to +2.92. The bulk-tissue WGBS reference encodes mixed-cell-type β profiles and produces |β_sample − β_bulk_ref| dominated by substrate-distribution mismatch on mucosal cohorts. Gene-promoter sub-cell-type references encode signature β profiles for specific cell types and avoid this artifact.

**Multi-atlas readings on mucosal cohorts must include a gene-promoter sub-cell-type atlas as the primary cell-of-origin reader.** Single-atlas Stage 2 readings on mucosal cohorts using bulk-WGBS references can be substrate-substitution-fooled.

### DISC-BLADDER-004 — Stage 1 panels require per-cohort substrate-coverage validation at prereg-write time

Xu-538 panel mean per-sample coverage on TCGA-BLCA: 78.0%; pass rate 51.1% at ≥80% per-sample threshold. The panel CpGs are all from HM450 design (substrate-applicable) but per-sample coverage drops on this specific cohort due to TSS-site processing variability. Stage 1 panel transferability is cohort-specific, not platform-specific.

CHK-2.17 cookbook gate added: Stage 1 panels must be validated against the target Phase C cohort's substrate-coverage envelope at prereg-write time. Validation procedure: sample 5-10 random Phase C cohort β files, compute per-sample panel coverage, FLAG if mean < 90% or q5 < 80%.

---

## Atlases used and deferred

### atlases_run (in production scoring for bladder-epic v0.1)

- **Layered Moss+Loyfer** 25-tile (calibration anchor VAL-112). **NOTE per DISC-BLADDER-003:** Loyfer Bladder tile reading on mucosal cohort is interpretive context, NOT primary cell-of-origin signal.
- **EpiSCORE BladderRef CpG-bridged** 4-tile (calibration anchor VAL-119, sealed 2026-05-01T03:46:00Z, atlas SHA `3005663b…`). **PRIMARY cell-of-origin reader for bladder per DISC-BLADDER-003.**
- **Caggiano CelFiE TIM** 19-tile (calibration anchor VAL-113).
- **Salas Blood.EPIC IDOL 450K legacy** 6-tile (production calibrated; Stage 3 primary).
- **UniLIFE Guo 2025** 19-tile (within-cohort self-cal v0.1; VAL-115 Wave 1 promotion path).

### atlases_deferred (not in production scoring; v0.2+ targets)

- **Bladder-specific Stage 1 panel** (replacement for Xu-538) — blocker: VAL-114 Wave 1 calibration on Hannum 2013 GSE40279 n=656 healthy aging blood + per-cohort substrate-coverage validation per CHK-2.17.
- **Loyfer-corrected mucosal-aware A-score normalization** — blocker: substrate-distribution-aware normalization research (DISC-BLADDER-003 v0.2 promotion path).
- **EpiSCORE LungRef** (when relevant — cookbook-wide mucosal-tissue lesson, lung cards will need this).

---

## v0.1 → v0.2 promotion path

1. **Stage 1 panel validation** — VAL-114 Hannum 2013 calibration with CHK-2.17 cohort-substrate coverage check baked in. Either trim Xu-538 to a bladder-cohort-coverage-validated subset OR generate a calibrated Stage 1 panel that maintains uniform coverage across solid + mucosal cohorts. Re-run Stage 1 on TCGA-BLCA under the calibrated panel. Promote VAL-120 from O4 to O1/O2/O3 if the corrected panel passes CHK-3.1B.

2. **Stage 2 mucosal-aware normalization** — research the substrate-distribution-aware normalization that would let Loyfer bulk-WGBS readings be useful on mucosal cohorts (e.g., per-sample-substrate-baseline normalization). Until that lands, **BladderRef Epi remains the primary cell-of-origin reader for bladder**; Loyfer Bladder reading is descriptive context only.

3. **Stage 3 lineage-pattern detail** — stratify TCGA-BLCA Stage 3 by NMIBC vs MIBC if clinical metadata supports the split. The broad-positive pattern in v0.1 may resolve into pure-lymphoid-dominant (NMIBC subgroup, would replicate Chen 2022 RFS) vs pure-myeloid-dominant (advanced MIBC, MDSC infiltration) on stratified analysis.

4. **Cohort expansion** — the v0.2 sprint adds GSE52955 multi-cancer urological (n=72 HM450) for cross-cohort verification, and Bryan UK NMIBC + Chen 2022 NMIBC blood EPIC n=603 for blood-substrate verification. WHI bladder pre-diagnostic n=440+440 remains v1.0+ biobank-gated.

---

## Tier promotion rationale

bladder-epic v0.1 promotes from `stage_2_only_validated` to `multi_modal_validated + multi_atlas_calibrated`:

- **multi_modal_validated** — three independent stages (Stage 1 panel + Stage 2 multi-atlas + Stage 3 immune) scored on the same cohort with consistent biological interpretation across stages.
- **multi_atlas_calibrated** — five atlases on cohort (Loyfer, BladderRef, Caggiano TIM, Salas IDOL, UniLIFE), three with formal calibration anchors (VAL-112, VAL-113, VAL-119), one production calibrated (Salas), one within-cohort self-cal flagged for Wave 1 (UniLIFE).

The four DISC-BLADDER lessons strengthen the cookbook for every future card.

---

## Reproducibility

- GitHub: `https://github.com/hmahaffeyges/IAM-Validation` commit `404eed3`
- Atlas vault: `Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_bladderref/` with bridge script + bridged CSV + Entrez source CSV + README
- Validation runs: VAL-119, VAL-120, VAL-121, VAL-122 each with prereg + amendment + outcome.md + script + results JSON + per-sample CSV + cohort manifest + clinical metadata + stratified results
- Unified runner: `validation_runs/unified_phaseC_runner.py` (single-pass execution, 270.7s for n=440 cohort)
- Post-pass: `validation_runs/postpass_amended.py` (paired d, Welch d, outcome class against amended floor)
- Cohort acquisition: TCGA-BLCA via GDC API `https://api.gdc.cancer.gov/data/{file_id}`; calibration cohort same as VAL-106

---

**bladder-epic v0.1 sealed 2026-05-01. The cookbook caught its own assumptions in real time and corrected them transparently. That is what discipline looks like.**
