# VAL-128 Outcome — GSE87650 Crohn's Disease Blood Methylation

**Sprint:** gastric+esophageal-epic v0.1
**VAL ID:** VAL-128
**Prereg SHA:** `e7cdb09082d39bdb0c82d4465ffd43a9cc12b79c1b56a5dcd23f22a0086da7bc`
**Cohort:** GSE87650 GPL13534 sorted-cell sub-experiment — n=240 (60 monocytes + 59 CD4 + 56 CD8 + 65 wh-blood-companion)
**Source paper:** Ventham NT et al. Nat Commun 2016;7:13507 (PMID 27886173)
**Substrate:** Illumina HumanMethylation450 (GPL13534)
**Status:** SEALED 2026-05-02

---

## Primary outcome class

**`O1_CROHNS_LANGUAGE_SUPPORTED + O5_MIXTURE_ATTENUATION_REVERSAL`**

The Crohn's-pathway methylation signature DOES fire (max |d_CD-HC| = 1.72) but exclusively through Stage 3 immune-composition atlases, not through Stage 1 cycling-class architectural drift. The pre-locked mixture-attenuation test FAILS in the opposite direction predicted: **whole blood shows STRONGER d than sorted cells**, indicating the Crohn's signature is a population-fraction shift, not a within-cell-type methylation drift.

---

## Headline 1: Stage 1 architectural drift is NULL in Crohn's blood

| Cell type | n CD | n HC | d_CD-HC | p | Outcome |
|-----------|-----:|-----:|--------:|---|---------|
| Monocytes | 20 | 20 | −0.16 | 0.62 | NULL |
| CD4 | 20 | 20 | +0.06 | 0.84 | NULL |
| CD8 | 18 | 19 | −0.46 | 0.17 | PARTIAL (negative) |
| Whole blood | 19 | 25 | −0.21 | 0.51 | NULL |

**The Xu-538 cycling-class architectural-drift framework does NOT detect Crohn's in blood.** This is itself a finding: the Stage 1 panel is a tissue/cancer-driven cycling signature (validated previously on TCGA-COAD, TCGA-LIHC, etc.), not a generic chronic-inflammation marker. Crohn's chronic intestinal inflammation does not produce the same architectural-drift signature as solid-tumor cycling acceleration.

UC vs HC shows similar near-null pattern across all cell types (|d| range 0.25-0.52), confirming this is a class-of-disease finding (IBD does not register on cycling-class Stage 1 panel) not specific to CD vs UC distinction.

---

## Headline 2: Stage 3 immune-composition signature FIRES strongly in whole blood

The Crohn's signal is captured by Stage 3 immune-composition deconvolution atlases, predominantly in whole blood:

### Whole blood — strongest fires (CD vs HC)

| Tile | d_CD-HC | Direction interpretation |
|------|--------:|---------------------------|
| A_uni_aCD8Tnv (UniLIFE activated naive CD8) | **+1.72** | T-cell activation/expansion |
| A_uni_CD8T (UniLIFE total CD8) | **+1.70** | CD8 T-cell expansion |
| A_loyfer_CD4T-cells_EPIC | **+1.66** | CD4 expansion |
| A_uni_CD4T (UniLIFE total CD4) | **+1.64** | CD4 expansion |
| A_uni_aCD4Tmem | +1.61 | Memory CD4 |
| A_salas_CD4T | +1.58 | CD4 expansion |
| A_uni_aCD4Tnv | +1.58 | Naive CD4 |
| A_uni_aTreg | +1.56 | Regulatory T expansion |
| A_uni_NK | +1.55 (approx) | NK expansion |
| A_salas_Mono | **−1.14** | Monocyte proportion decrease |
| A_salas_Neu | **−1.34** | Neutrophil proportion decrease |
| A_loyfer_Neutrophils_EPIC | −1.01 | Neutrophil proportion decrease |

**Bidirectional pattern: T-cells (CD4/CD8/Treg/NK) UP, monocytes/neutrophils DOWN.** This is the classic immune-population-shift signature of active inflammatory bowel disease in peripheral blood.

### Compared to UC (whole blood)

UC vs HC shows similar direction with comparable magnitude (e.g., A_salas_CD8T d_UC-HC ≈ +1.0-1.2). CD vs UC contrast in whole blood reaches d_CD-UC = +0.26 on Stage 1 — modest subtype distinction.

---

## Headline 3: Mixture-attenuation REVERSAL — pre-lock fails opposite direction

**Pre-locked test:** sorted-cell d should be ≥ 1.5x whole-blood d (mixture-attenuation hypothesis: separating cells should AMPLIFY the within-cell-type drift signal).

**Observed:** sorted-cell d is SMALLER than whole-blood d on most tiles. Only 40/93 (43%) of tiles show sorted ≥ 1.5x whole-blood. **The opposite of the pre-locked direction.**

### Why the prereg expected the wrong direction

The prereg assumed Crohn's produces within-cell-type methylation drift that gets diluted by mixture in whole blood. The data shows the opposite: **Crohn's produces methylation signature primarily through population-fraction shifts** — relative abundance of T-cells vs myeloid cells changes — not through individual cell methylation drift.

When you sort the cells, you've already enriched for one cell type — the population-shift signal you're trying to detect is gone by definition. The Stage 3 atlases (Salas IDOL, Loyfer immune tiles, UniLIFE 19-cell, Caggiano TIM) are methylation-based cell-type deconvolution panels: they detect proportional shifts in mixed populations. In whole blood (mixed), they detect the IBD shift. In sorted cells (single population), there is no shift to detect.

This is not a refutation of the Stage 3 atlases — it is a clarification of what Stage 3 measures. **Stage 3 atlases measure cell-type composition; they do not measure within-cell-type chronic-inflammation drift.**

---

## Headline 4: Within-cell-type CD vs HC findings (modest but biologically informative)

### CD8 (sorted)
- A_salas_Bcell d=−1.10 — CD8-sorted CD samples show LESS B-cell contamination than HC samples; suggests cleaner CD8 sorting in CD or B-cell sequestration in inflamed tissue
- A_uni_aBmem d=−0.87, A_uni_aBnv d=−0.81 — same B-cell depletion pattern across UniLIFE B-cell sub-types
- A_cag_neutrophil d=+0.75 — slight neutrophil contamination in CD8-sorted CD samples
- A_esoref_Fib d=+0.68, A_esoref_EC d=+0.65 — EsoRef stromal/endothelial signal in CD8 cells (cross-tissue overread continues here, lower magnitude)

### CD4 (sorted)
- A_salas_CD4T d=+0.69 (note: positive within sorted CD4), A_uni_aCD8Tnv d=+0.58 — modest CD4-internal heterogeneity differences
- A_salas_Neu d=−0.66, A_loyfer_Neutrophils_EPIC d=−0.54 — neutrophil contamination decreased in CD4-sorted CD samples
- 21 tiles with |d| ≥ 0.5

### Monocytes (sorted)
- A_loyfer_Breast d=−0.72, A_loyfer_Head_and_neck_larynx d=−0.69, A_loyfer_Bladder d=−0.63 — multiple secretory-tissue tiles read NEGATIVE in CD monocytes (interpretation unclear; consistent with substrate baseline shift in monocytes)
- 19 tiles with |d| ≥ 0.5
- The monocyte-internal direction is the inverse of the whole-blood pattern (whole blood showed monocyte FRACTION down; sorted monocytes show signal in different tiles)

---

## Headline 5: CD vs UC subtype discrimination — modest

| Cell type | d_CD-UC Stage 1 | Notes |
|-----------|----------------:|-------|
| Monocytes | +0.35 | Partial |
| CD4 | +0.33 | Partial |
| **CD8** | **−0.72** | **Differentiating in CD8** |
| Whole blood | +0.26 | Partial |

CD8 cells in CD show methylation patterns distinct from CD8 in UC (d=−0.72), suggesting **CD8 immune phenotype is the cleanest CD-vs-UC discriminator at v0.1**. Other cell types show only partial separation. This is consistent with literature that CD has stronger Th1/CD8 effector memory signatures than UC's more Th2-skewed presentation.

---

## Pre-locked outcome assignments

| Pre-locked outcome | Status |
|-------------------|--------|
| O1_CROHNS_LANGUAGE_SUPPORTED | **FIRES** (max |d_CD-HC| = 1.72 on Stage 3 in whole blood) |
| O2_CROHNS_LANGUAGE_PARTIAL | not applicable (O1 fires) |
| O3_NO_CROHNS_SIGNATURE | not applicable |
| Stage 1 expected modest |d| ≥ 0.2 | **FAILS** — Stage 1 essentially NULL across all cell types |
| Stage 3 immune shift expected | **FIRES STRONGLY** in whole blood; absent in sorted cells |
| Mixture-attenuation pre-lock (sorted ≥ 1.5x whole-blood) | **FAILS in OPPOSITE direction** — whole blood is stronger; signature is population-fraction shift |
| O1_CD_UC_DISCRIMINATION (|d_CD-UC| ≥ 0.4) | **FIRES** in CD8 (d=−0.72); modest elsewhere |

---

## Marcus-pathway language amendment to gastric+esophageal-epic v0.1

VAL-128 supports adding the following to gastric+esophageal-epic v0.1 known_limitations:

> **Crohn's-pathway methylation signature.** Patients with Inflammatory Bowel Disease (Crohn's or UC) carry a peripheral-blood methylation signature detectable through immune-cell-fraction deconvolution panels (Salas IDOL, Loyfer immune tiles, UniLIFE 19-cell). The signature is captured as a T-cell expansion + myeloid-fraction depletion pattern in whole-blood substrate (max |d| = 1.72 on activated CD8 T-cell tiles per VAL-128). The Stage 1 cycling-class architectural-drift panel does NOT detect IBD; the IBD signature is a population-composition shift not a within-cell-type drift. Card v0.1 reports Stage 3 immune-fraction shifts where applicable but does not currently report a dedicated IBD-detection score.

Same amendment language applies to hcc-epic v0.3.1 known_limitations (Marcus's stepbrother had Crohn's + ileostomy prior to HCC).

---

## Logged follow-ups (NOT in scope of this VAL)

1. **GSE87650 main whole-blood cohort (n=384) — v0.1.1 expansion sprint.** The full Ventham cohort (n=384: CD=103 + UC=101 + HL=105 + HS=75) lives in the supplementary file `GSE87650_processedMethCombinedWb.txt.gz` (1.4 GB compressed), separate from the GPL13534 series matrix used here. Pulling and scoring that cohort would extend Stage 3 statistical power 4-5x over the 65-sample wh-blood-companion subset analyzed here. Logged for v0.1.1 expansion sprint with proper compute allocation.

2. **IBD-dedicated panel construction** — VAL-128 demonstrates that the existing Stage 3 immune-fraction atlases capture IBD signature in whole blood with d ≈ 1.5-1.7. A future card-construction VAL could test whether a panel composed of the top 20-30 most-discriminating CpGs from the Stage 3 detection produces a clean IBD-specific A-score with single-cell granularity. Logged as candidate for IBD-epic v0.0 design phase (not scope of gastric+esophageal-epic).

3. **CD vs UC dedicated discrimination** — CD8 shows d=−0.72 within-cohort, suggesting a CD-specific CD8 effector phenotype distinguishable from UC's CD8. Worth expanded analysis if IBD-epic is pursued.

---

## Reproducibility (CHK-7.6)

- **Source code:**
  - `val128_chunked_scorer.py` — chunked-pass scorer (4 passes × 60 samples), final version
  - `val128_results_build.py` — d-value computation + cell-type stratification + mixture-attenuation test
  - Atlas pipeline shared with VAL-126/127 via `val126_stad_phase_c.py` (reused as module)
- **Inputs:**
  - GSE87650 GPL13534 series matrix (positions 0-239, sorted-cell sub-experiment)
  - 8 calibrated atlases (SHA-sealed in atlas_vault INVENTORY.json)
  - GSE87650 sample metadata table (parsed from series matrix !Sample_* lines)
- **Environment:** Python 3, NumPy, scipy.stats
- **Headline outputs:** `VAL-128_results.json`, `VAL-128_per_sample.csv`, `val128_per_sample.ndjson`

---

## Final language

VAL-128 finds that the Crohn's-pathway methylation signature in peripheral blood is captured by Stage 3 immune-cell-fraction deconvolution atlases (Salas IDOL, Loyfer immune tiles, UniLIFE 19-cell) primarily in whole-blood substrate, with maximum effect size |d_CD-HC| = 1.72 on activated CD8 T-cell tiles. The Stage 1 cycling-class architectural-drift panel does not detect Crohn's, indicating the IBD signature is a population-composition shift rather than within-cell-type methylation drift. The pre-locked mixture-attenuation test fails in the opposite direction predicted: whole blood shows stronger d than sorted cells, because the signature is fundamentally a population-fraction phenomenon that disappears when cells are pre-sorted by type. The data support adding Crohn's-pathway language to the gastric+esophageal-epic v0.1 known_limitations and to hcc-epic v0.3.1 known_limitations, framing IBD detection as accessible through Stage 3 deconvolution panels rather than through Stage 1 architectural-drift scoring.
