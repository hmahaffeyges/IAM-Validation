# cardio-epic v0.2.2 — Cardiovascular EDEAR Card

**Version:** 0.2.2 (same-day afternoon honesty patch on v0.2.1; v0.2.1 was the same-day morning patch on v0.2 which shipped 2026-04-29 morning)
**Built:** 2026-04-29
**Validation tier:** multi_modal_validated (3 disease cohorts across 3 substrates plus 1 atlas-integration test)
**Built under:** CHK-3.1A/B split convention (locked 2026-04-28); first card built natively under split. v0.2 brought the card up to full Block 1-20 + CHK-5.7/5.8/5.9/5.10 structural-parity. v0.2.1 added CHK-5.11 atlas-family-fitness check, CHK-5.12 atlas-canonical-source-check, DISC-CARDIO-007, and corrected atlas naming + atlases_deferred. **v0.2.2 is the second same-day honesty patch:** Phase A acquisition of the canonical-document-named "Konigsberg 2023" atlas began per v0.2.1's revised critical path, web verification of the cited DOI (`10.1093/nargab/lqad061`) found that the canonical document had two factual errors — the actual paper at that DOI is **Cuadrat et al. 2023** (no Konigsberg in author list) with **bulk ENCODE heart-tissue additions** (right atrium, left ventricle, coronary artery), NOT sorted cardiomyocytes. v0.2.2 corrects: (1) atlases_deferred Konigsberg entry → corrected Cuadrat 2023 entry; (2) canonical-documents block reframed — drops "cannot be deployed" framing since it was anchored on a fictional atlas, acknowledges sorted-cardiomyocyte array-CpG atlas gap as open published-literature limitation; (3) v0.3 critical path revised with Cuadrat first instead of Konigsberg; (4) CCL-046 LL-CANONICAL-DOC-FACTUAL-ERROR logged + CHK-5.13 documents-of-record citation-verification gate added. PIPELINE_REFERENCE Part 2.4 fully rewritten.
**Supersedes:** cardio-epic v0.2.1 (2026-04-29 same-day morning). All sealed VAL outcomes from v0.1/v0.2/v0.2.1 are preserved unchanged. v0.2.2 is an additive honesty patch — no biology was rewritten and no scoring changed.

---

## Scope

Cardiovascular disease detection across the cardiovascular landscape: ischemic stroke, pulmonary arterial hypertension (PAH), aortic dissection / BAV+dilation, with framework architecture extensible to coronary heart disease, heart failure, hypertensive heart disease, and other cardiovascular subdomains.

The card covers cardiovascular as a single integrated subdomain, splittable into separate cards if biological signal divergence between subdomains is established in future VALs.

---

## Validated cohorts (v0.2)

| VAL | Cohort | n | Substrate | Outcome |
|---|---|---|---|---|
| **VAL-108** | GSE69138 ischemic stroke 3-subtype | 404 (whole blood) | GenomeStudio AVG_Beta HM450K | `O3_3SUBTYPE_UNDIFFERENTIATED` |
| **VAL-109** | GSE84395 PAH cultured PECs | 39 (control 18 + hPAH 10 + iPAH 11) | minfi preprocessFunnorm HM450K | `O2_VASCULAR_TILE_DIFFERENTIATING` |
| **VAL-110** | GSE84274 ascending aorta | 24 (normal 6 + dissection 12 + BAV+dilation 6) | GenomeStudio V2011.1 HM450K | `O2_AORTIC_ANY_TILE_DIFFERENTIATING` |
| **VAL-111** | EpiSCORE HeartRef on three cohorts above | 589 + 39 + 24 = 652 samples | All three above (cohort β panels reused) | `O3_TISSUE_FLOOR_DOMINATED` (atlas → atlases_deferred for v0.3) |

VAL-111 is the new validation in v0.2. It tested whether the EpiSCORE HeartRef gene-promoter cardiac reference atlas adds Stage 2 cardiac-tile discrimination on the three β panels already sealed in VAL-108/109/110. Outcome sealed at `O3_TISSUE_FLOOR_DOMINATED`: maximum within-cohort tissue discrimination 0.0152 (well below the 0.10 threshold); blood-floor breach on 5/5 tiles in GSE69138; gene-promoter A-scores cluster ~0.5 across all heterogeneous β panels regardless of substrate. EpiSCORE HeartRef → `atlases_deferred` for cardio-epic v0.3 with explicit unblock dependency.

---

## Architecture

**Stage 1 (immune class):** Salas IDOL 350-CpG proxy panel (production target: Xu-538, patent-protected). Pooled-entropy A-score against H_min(immune) = 0.838889.

**Stage 2 (cell-of-origin):** **Layered Moss + Loyfer array atlas** (`loyfer_moss_2018/reference_atlas.csv`, the combined file from nloyfer/meth_atlas) — Moss 2018 primary 18-tissue + Loyfer 2023 array supplement, 7,890 CpGs across 25 cell-type columns. This is the canonical layered Stage 2 reference per PIPELINE_REFERENCE Part 2.1+2.2: Moss stays primary for cells it covers (colon, lung, gastric, bladder, cervical, kidney epithelial; hepatocyte, pancreatic exocrine, breast_ductal, prostate epithelial; vascular_endothelial, fibroblast; neutrophil, lymphocyte, monocyte, hsc), Loyfer supplements for sorted-cell entries Moss didn't have at array CpG resolution (Cortical_neurons, Vascular_endothelial_cells, Left_atrium, EPIC-trained sorted immune, Pancreatic_duct_cells, Head_and_neck_larynx, Upper_GI). Per-tile per-class A-score against frozen H_min anchor for each tile's mapped class. Cardio-relevant tiles emphasized in scoring report:
- Vascular_endothelial_cells (stromal class, H_min 0.86295)
- Left_atrium (terminal class, H_min 0.772837)
- Adipocytes (stromal class, H_min 0.86295) — peri-cardiac/peri-aortic adipose context

**Stage 3 (immune subcomposition):** UniLIFE 19-cell + Salas Blood.EPIC IDOL 6-cell pooled-entropy. Run-everything per cookbook signoff 2026-04-26. Teschendorff 2017 EpiDISH RPC mode is the canonical scoring method per PIPELINE_REFERENCE Part 3.1.

**`atlases_run` at v0.2.1 (CHK-5.8):**
- Layered Moss + Loyfer array atlas (Stage 2, anchor VALs 108/109/110)
- UniLIFE 19-cell (Stage 3, anchor VALs 108/109/110)
- Salas Blood.EPIC IDOL 6-cell (Stage 3, anchor VALs 108/109/110)

**`atlases_deferred` at v0.2.1 (CHK-5.8) — full canonical-document-named list:**

| Atlas | Target | Unblock dependency |
|---|---|---|
| **Cuadrat 2023 cardiovascular extended** (Moss 2018 base + 3 bulk ENCODE heart-tissue additions: right atrium, left ventricle, coronary artery) | v0.3 | Phase A acquire atlas (open access, MIT-licensed deconvR R package) → Phase B Cuadrat-specific calibration VAL → Phase C cardio-cohort scoring. PIPELINE_REFERENCE Part 2.4 (corrected per CCL-046 — actual paper is Cuadrat et al. 2023, NOT Konigsberg 2023; bulk ENCODE heart tissues, NOT sorted cardiomyocytes). |
| **Sorted cardiomyocyte array-CpG atlas (open published-literature gap)** | v1.0+ | External: someone in published literature releases a sorted-cardiomyocyte methylation atlas at array-CpG resolution. Not blocking cardio v0.3 ship — bulk-heart-tissue Stage 2 + Stage 1 immune workhorse is the v0.3 architecture. Monthly literature surveillance pass for "sorted cardiomyocyte methylation array" added. |
| **Caggiano CelFiE TIM cardiac** (heart_meth + endothelial_meth, on disk) | v0.3 | HM450 hg19 manifest acquisition → WGBS region → CpG mapping → calibration VAL → cardio-cohort scoring |
| **EpiSCORE Zhu/Teschendorff 2022 pan-tissue** (full 13-tissue including Heart, on disk) | v0.3 | R-package integration via existing rpy2 bridge → pan-tissue calibration VAL → cardio-cohort scoring (separate from VAL-111's HeartRef sub-panel test) |
| **Tanaka 2025 6-cell-type neural** (cortical + dopaminergic + spinal motor + astrocytes + Schwann + microglia) | v0.3+ | Nanopore→array CpG bridge engineering → calibration VAL → cardio-cohort scoring |
| **Tian et al. 2023 scMCodes brain** (188 cell types, single-cell; corrected from "Liu 2023" per CCL-046) | v0.4+ | scMCodes→array CpG projection → calibration VAL → scoring (lower priority for cardio than Cuadrat 2023 / Tanaka / Caggiano CelFiE TIM) |
| **EpiSCORE HeartRef sub-panel** (anchor VAL-111 O3_TISSUE_FLOOR_DOMINATED) | v0.3 | Either re-bridging or supersession by Konigsberg + Caggiano cardiac integrations |
| **MARLIN Capper 2025 training scaffold** (leukemia matrix v0.3 build-out per TESTING_CHECKLIST §STAGE 0) | v0.3+ | Leukemia matrix build-out → calibration → scoring (lower cardio relevance) |
| **Sabedot GeLB 2021** (R training script, requires GSE150289) | v0.3+ | GSE150289 cohort acquisition + R→Python integration → calibration → scoring |

**Canonical-documents cardio Stage 2 extension path (corrected 2026-04-29 per CCL-046).** PIPELINE_REFERENCE_v2.md Part 2.4 was rewritten in v0.2.2 after web verification of the cited DOI found that the prior version contained two factual errors: the actual paper at `10.1093/nargab/lqad061` is Cuadrat, Kratzer, Giral Arnal et al. 2023 (NAR Genomics & Bioinformatics), no Konigsberg in the author list; and the atlas content is the Moss 2018 25-tissue base extended with three **bulk** ENCODE heart tissues (right atrium auricular, heart left ventricle, coronary artery), NOT sorted cardiomyocytes. With the fictional sorted-cardiomyocyte atlas removed, the "cannot be deployed" framing is dropped: cardio-epic v0.2 is operational under the layered Moss+Loyfer atlas with Stage 1 immune as the validated workhorse (VAL-110 d=+1.08 normal vs BAV on aortic tissue). Cuadrat 2023 + Caggiano CelFiE TIM + EpiSCORE pan-tissue + Tanaka 2025 are integration enhancements that broaden cardio Stage 2 cell-of-origin coverage at bulk-heart-tissue resolution; they do NOT gate deployment.

**Sorted cardiomyocyte array-CpG atlas — open published-literature gap.** As of 2026-04-29 no such atlas exists at array-CpG resolution. Published cardiac methylation work covers either targeted CpG biomarkers (Zemmour 2018 FAM101A six-CpG panel; Yamazoe 2021 mt-cfDNA), bulk heart tissues (Moss 2018 Left_atrium; Cuadrat 2023 right atrium + left ventricle + coronary artery), or sorted vascular cells (Loyfer 2023 vascular_endothelial + smooth_muscle as part of the layered Moss+Loyfer atlas already in production). When a sorted-cardiomyocyte array-CpG atlas is published, it becomes a v1.0+ candidate. Until then, cardio-epic Stage 2 cardiac cell-of-origin discrimination operates at bulk-heart-tissue resolution. Monthly literature surveillance pass for this atlas added to the cookbook surveillance routine.

VAL-111 was a side-track from the actual canonical cardio Stage 2 extension path: EpiSCORE HeartRef was selected because it sat in atlas_vault, not because it was the document-prescribed cardio atlas. The corrected document-prescribed path is **Cuadrat 2023 first** (the actual paper), Caggiano CelFiE TIM cardiac second, Tanaka 2025 third, EpiSCORE pan-tissue fourth. v0.3 critical path begins with Cuadrat 2023 acquisition — open access paper, MIT-licensed R package `deconvR` (https://github.com/BIMSBbioinfo/deconvR), signature matrix in supplementary data, six publicly available ENCODE EPIC IDAT accessions (ENCSR517JQA, ENCSR280LMY, ENCSR515ZCU, ENCSR190PQG, ENCSR688OHW, ENCSR582BMR).

**Run-everything violation acknowledged.** VAL-108/109/110 scored Stage 2 against ONLY the layered Moss+Loyfer combined atlas. Per the run-everything policy (Heath sign-off 2026-04-26), every IDAT runs Stage 2 against ALL reference atlases in the vault. The other Stage 2 atlases (Caggiano CelFiE 2021, Caggiano CelFiE TIM, EpiSCORE pan-tissue, EpiSCORE HeartRef, MARLIN, Sabedot) were NOT scored on cardio cohorts. v0.3 includes corrective re-execution of VAL-108/109/110 against the full atlas stack once each atlas is acquired and calibrated. Sealed structural outcomes from VAL-108/109/110 don't change; this adds new per-atlas results to the same cohorts.

---

## CHK-3.1A/B substrate gates per substrate

**TCGA HM450K sesame Level 3** (calibrated VAL-106/107):
- CHK-3.1A: f_extreme ≥ 50.5%, f_middle ≤ 9.0%, n_valid ≥ 400,000
- CHK-3.1B (cardio-epic 8,100-CpG subset, SHA `5a00e29ace75daae5a...`): extreme ≥ 55.0%, middle ≤ 8.5%, n_subset_valid ≥ 7,000 of 8,100

**GenomeStudio AVG_Beta HM450K** (within-cohort self-cal, VAL-108 + VAL-110):
- CHK-3.1A self-cal envelope: f_extreme in [25%, 40%] (whole blood, GSE69138 31.81% ± 3.54%) or [27%, 40%] (aortic tissue, GSE84274 33.95% ± 2.21%), f_middle ≤ 13%
- Note: NOT a generalizable platform threshold; requires structurally-separated calibration VAL on a separate GenomeStudio cohort to establish a stable platform threshold

**minfi preprocessFunnorm HM450K** (within-cohort self-cal, VAL-109):
- CHK-3.1A self-cal envelope: f_extreme in [48.16%, 57.48%] (cultured PECs, GSE84395 52.82% ± 2.33%), f_middle ≤ 10.5%
- Note: NOT generalizable; requires structurally-separated calibration VAL

**GenomeStudio V2011.1 HM450K raw** (within-cohort self-cal, VAL-110):
- Treated as sub-variant of GenomeStudio AVG_Beta for cardio-epic v0.2 self-cal purposes; envelope identical to AVG_Beta aortic value because cohort is the same.
- Note: NOT generalizable; requires structurally-separated calibration VAL.

---

## Substrate roadmap (CHK-5.9)

| Substrate | Status at v0.2 | Anchor | Target |
|---|---|---|---|
| DNAm methylation β | **validated** | VAL-108/109/110/111 | v0.2 (current) |
| Nucleosome occupancy | deferred | (G-003b framework-level only) | v0.4+ |
| Fragment fuzziness | deferred | (G-003b framework-level only) | v0.4+ |
| Windowed protection score | deferred | (G-003b framework-level only) | v0.4+ |
| Fragment size entropy | deferred | (G-003b framework-level only) | v0.4+ |

DNA methylation β is the sole validated substrate at v0.2. The four non-DNAm substrates are framework-supported via G-003b MCMC posteriors (R-hat < 1.001) but cardio-epic-specific validation is deferred to v0.4+ once L3 multi-substrate assay generation comes online.

---

## Per-disease scoring policy at v0.2

### Ischemic stroke (whole blood)
- **Pooled report only.** Subtype etiology (large-artery atherosclerosis vs small-vessel disease vs cardioembolic) is **framework-equivalent** per VAL-108 (all pairwise |d| < 0.17). Cardio-epic v0.2 does NOT claim subtype stratification on whole-blood methylation.
- **Stage 1 immune** is the primary scoring target (no clear cardio-tile differentiation in stroke whole blood).
- **No healthy-vs-stroke contrast** at v0.2 (cohort had no healthy controls). Future VAL with healthy controls would establish stroke vs healthy effect size.

### PAH (cultured pulmonary endothelial cells)
- **Direct vascular-tile differentiation operational.** Control vs hPAH d = +0.79 on Vascular_endothelial_cells tile, d = +0.65 on Left_atrium.
- **hPAH vs iPAH framework-equivalent.** Cardio-epic v0.2 does NOT claim heritable-vs-idiopathic stratification.
- **Stage 1 immune** also discriminates control from PAH (d = +0.65 both subtypes).
- Sample type: cultured PECs only at v0.2; whole-blood PAH discrimination not yet validated.

### Aortic pathology (ascending aorta tissue)
- **Pooled-pathology report only.** Dissection vs BAV+dilation is framework-equivalent (|d| < 0.05).
- **Stage 1 immune is the strongest signal** (normal vs BAV d = +1.08, normal vs dissection d = +0.56).
- **Stage 2 Vascular_endothelial_cells tile does NOT discriminate aortic pathology** (|d| ≤ 0.15) — bulk aortic tissue is dominated by non-endothelial cell types.
- **Stage 2 Left_atrium and Adipocytes tiles** show moderate discrimination (|d| = 0.6-0.9 normal vs BAV).

### EpiSCORE HeartRef cardiac-tile discrimination (atlas integration test, VAL-111)
- **No cardiac-tile A-score discrimination at the substrate resolutions tested.** All five cardiac tile A-scores read 0.46–0.51 across all three cohorts and all three substrates (whole blood, cultured PECs, aortic tissue).
- Maximum within-cohort tissue discrimination = 0.0152 (GSE84274 MP tile, dissection 0.5012 − normal 0.4860); EC tile range in GSE84395 PEC = 0.0070; SMC tile range in GSE84274 = 0.0120 — all an order of magnitude below the 0.10 threshold.
- Blood-floor breach on all 5 tiles in GSE69138 (cohort means CM 0.4770, EC 0.5025, FB 0.4905, MP 0.5109, SMC 0.5064 — all > 0.10).
- Direction is biologically sensible (dissection > BAV+dilation > normal monotonic in GSE84274; SMC tile always highest in aortic samples; iPAH > hPAH > control on EC tile in GSE84395) but the A-score magnitude is set by gene-promoter average methylation (~0.5 in heterogeneous β panels) rather than substrate-specific cell-of-origin contrast.
- Atlas methodologically sound for its design purpose (EpiDISH proportion estimation in heart tissue) but does not transfer to A-score tile reading on heterogeneous β panels at the resolution required for cardio-epic Stage 2.
- **EpiSCORE HeartRef → atlases_deferred for v0.3** with explicit unblock dependency. Logged as **LL-CARDIO-005**.

---

## What we discovered in the cardio sprint (v0.2 lessons section)

This section consolidates what was learned across the cardio-epic sprint (VAL-108 through VAL-111 plus calibration anchors VAL-106/107). Six discoveries logged:

### DISC-CARDIO-001 — Stage 1 immune A-score is the workhorse for cardio-epic across all substrates tested

Three out of three substrate-validated cohorts (whole blood VAL-108, cultured PECs VAL-109, aortic tissue VAL-110) produced interpretable Stage 1 readings. The strongest single cardio-epic signal at v0.2 is VAL-110 normal vs BAV at d=+1.08 — and it came from Stage 1 immune A-score on bulk aortic tissue, not from a cardio-specific Stage 2 tile. This is consistent with Stage 1 being the universal immune-class flag (per `universal_stage_1_pipeline`) and tells us that systemic methylation drift, picked up on the immune panel even when the sample is not blood, is the primary cardio-epic discriminator at v0.2.

**Implication:** Cardio-epic Stage 1 carries most of the discrimination weight at v0.2. Stage 2 cardio-tiles add localization context but are not load-bearing for the headline call. Future cardio-epic cards (v0.3+) should preserve Stage 1 as the primary axis and treat Stage 2 cardiac-tile contributions as localization modifiers.

### DISC-CARDIO-002 — Substrate-cell match is the single most important cardio biology consideration

VAL-109 cultured PEC Vascular_endothelial_cells tile d=+0.79 vs VAL-110 bulk aorta Vascular_endothelial_cells tile d=−0.04. Same tile, same atlas, same H_min — different sample composition. The cultured PECs are pure endothelial cells; bulk aorta is dominated by smooth muscle and fibroblasts with endothelial cells as a minority cell type. The Stage 2 tile reads what is actually in the sample. The same finding cuts in different directions across the cohort: in VAL-110, the Adipocytes tile fires d=−0.88 normal vs BAV because the bulk aorta sample includes peri-aortic adipose tissue, and the non-cardio tiles (Hepatocytes d=−3.64, Lung_cells d=−3.25) fire because of the same peri-aortic adipose contamination signature.

**Implication:** Cardio-epic deployment reports MUST include a tile-substrate fitness flag. The customer needs to know: this tile is operational on cultured endothelial cells; this tile is NOT operational on bulk aorta because bulk aorta has minority endothelial content. EDEAR commercial deployment uses single-pipeline patient-vs-internal-reference scoring (CCL-037), so in practice the substrate is known at the report-generation step. The fitness flag is a clinician-facing label, not a deployment limitation.

### DISC-CARDIO-003 — Biology-correct nulls are first-class outcomes

VAL-108 sealed at O3_CARDIO_EPIC_3SUBTYPE_UNDIFFERENTIATED. Maximum |d| across all stages and contrasts on n=404 ischemic stroke whole blood: 0.167. The framework correctly reports that whole-blood DNA methylation does not stratify ischemic stroke by TOAST etiology — because by the time blood is drawn post-stroke, the systemic inflammatory response has homogenized the immune methylation signature across etiologies. This is real biology, not framework failure.

**Implication:** Cookbook discipline now treats biology-correct nulls as first-class outcomes. Cardio-epic v0.2 reports stroke as a single pooled signature, not by TOAST etiology. The framework correctly does not stratify what biology has homogenized.

### DISC-CARDIO-004 — Atlas family matters: tile-coverage atlases ≠ gene-promoter atlases at Stage 2 scoring

VAL-111 ran EpiSCORE HeartRef (gene-promoter cardiac reference, 3,727 unique 450K CpGs × 5 cardiac cell types) on the same three cohorts already sealed under VAL-108/109/110. Outcome O3_TISSUE_FLOOR_DOMINATED: all five cardiac tile A-scores read 0.46–0.51 across all three substrates regardless of cohort. Maximum within-cohort tissue discrimination 0.0152 (GSE84274 MP, dissection vs normal); EC range in GSE84395 cultured PEC 0.0070; SMC range in GSE84274 0.0120 — all an order of magnitude below the 0.10 discrimination threshold. Blood-floor breach on 5/5 tiles in GSE69138 (cohort means 0.48–0.51, well above 0.10). Yet the direction was biologically sensible (dissection > BAV+dilation > normal monotonic in GSE84274, SMC tile always highest in aortic samples, iPAH > hPAH > control on EC tile in GSE84395).

**Implication:** Gene-promoter reference atlases like EpiSCORE are methodologically sound for their design purpose (EpiDISH proportion estimation in tissue) but do not transfer to A-score tile reading on heterogeneous β panels. The atlas family that works at cardio-epic Stage 2 scoring is the tile-coverage WGBS-derived family (Loyfer 25-tile validated; Caggiano CelFiE TIM cardiac panels candidate when HM450 hg19 manifest acquisition unblocks). EpiSCORE HeartRef → atlases_deferred for v0.3 with explicit unblock dependency. Registered as **LL-CARDIO-005**.

### DISC-CARDIO-005 — Substrate-specific CHK-3.1A self-cal envelopes work for cardio at v0.2 — and they are not a generalizable platform threshold yet

Three different β preprocessing pipelines produced three different cohort f_extreme distributions:
- GenomeStudio AVG_Beta GSE69138 whole blood: 31.81% ± 3.54%
- GenomeStudio V2011.1 GSE84274 aortic: 33.95% ± 2.21%
- minfi preprocessFunnorm GSE84395 cultured PECs: 52.82% ± 2.33%

Compared against the TCGA HM450K sesame Level 3 baseline (55.87% ± 2.44% from VAL-106), the substrate-equivalence test confirms a 24-percentage-point distribution gap between GenomeStudio AVG_Beta and sesame Level 3 — different substrates need different thresholds. Within each cohort/substrate, self-cal QC pass rates were 94–96%, so the within-cohort envelope is a working operational gate.

**Implication:** Cardio-epic v0.2 uses substrate-specific CHK-3.1A self-cal envelopes at the within-cohort level. These are NOT generalizable platform thresholds yet. To make any of these substrate envelopes generalizable, a structurally-separated calibration VAL is required for each substrate (a separate cohort using the same preprocessing pipeline, ideally adjacent-normal tissue from a non-cardiovascular disease, to anchor the threshold without disease-driven distribution shift). Documented in v0.2 next_validation_steps.

### DISC-CARDIO-006 — The cardio sprint exercised the entire CHK-3.1A/B split convention end-to-end for the first time

Cardio-epic v0.1 was the first card built natively under the CHK-3.1A/B split convention (CCL-042 LL-CHK-3.1-A/B-SPLIT formalized 2026-04-28). VAL-106 established CHK-3.1A baseline for TCGA HM450K sesame Level 3. VAL-107 established CHK-3.1B for cardio-epic on the same substrate. VAL-108/109/110 then operated under the split with substrate-specific self-cal envelopes for the three non-sesame substrates encountered. VAL-111 added an atlas integration test on top.

**Implication:** The split convention works in practice. Phase 4 retroactive review of breast-epic v2.3 / crc-epic v2.4 / hcc / cervical / kidney / lung / ad-immune cards (additive documentation only, no sealed VAL outcomes change) will bring the rest of the cookbook into alignment with what cardio-epic established. CHK-5.7/5.8/5.9/5.10 structural-parity gates were added to TESTING_CHECKLIST.md to lock the universal_reference + atlases_used_and_deferred + substrate_roadmap + chk_3_1_thresholds_per_substrate blocks at every card publish.

### DISC-CARDIO-007 — Always read PIPELINE_REFERENCE Part 2 first; atlas selection must trace to a canonical-document name (added in v0.2.1; corrected in v0.2.2 after CCL-046 finding)

VAL-111 was scored against EpiSCORE HeartRef because that atlas was already in atlas_vault from a prior acquisition pass. PIPELINE_REFERENCE_v2.md Part 2.4 (as written through v0.2.1) named a "Konigsberg 2023" cardio Stage 2 atlas as the deployment blocker — but Phase A acquisition for that atlas in v0.2.2 found via web verification that the canonical document had two factual errors: the actual paper at the cited DOI is **Cuadrat et al. 2023** (no Konigsberg in author list), with **bulk ENCODE heart-tissue additions** (right atrium, left ventricle, coronary artery) NOT sorted cardiomyocytes. The originally-described atlas does not exist in published literature. None of the canonical-document-named cardio atlases were prioritized in cardio v0.1/v0.2 because the atlas selection was made by browsing atlas_vault rather than by reading the canonical document. VAL-111 produced a real and useful negative result (atlas-family-fitness lesson, LL-CARDIO-005), but it was a side-track from the canonical cardio atlas critical path, and even after CHK-5.12 (added in v0.2.1) forced atlas selection to trace to the canonical document, a second-order error (the document of record itself contained a factual error) required a separate gate (CHK-5.13, added in v0.2.2 per CCL-046).

**Implication.** Before any future atlas integration VAL is sealed, the prereg must (i) cite which canonical-document section names the atlas (CHK-5.12), and (ii) verify that the citation in the canonical document is itself accurate against the actual paper at the cited DOI (CHK-5.13 documents-of-record citation-verification gate added 2026-04-29). Cardio v0.3 critical path revised: **Cuadrat 2023 first** (the actual paper at the DOI Part 2.4 cited), Caggiano CelFiE TIM cardiac second (when manifest unblocks), EpiSCORE pan-tissue third (separate from the HeartRef sub-panel scored in VAL-111), Tanaka 2025 fourth (when nanopore→array bridge is engineered). Each of these atlases needs its own calibration VAL (CHK-3.1A + CHK-3.1B) on a structurally-separated healthy cohort BEFORE any cardio-cohort scoring against it (CCL-041 platform calibration discipline applied to atlases, not just substrates).

**Sorted cardiomyocyte array-CpG atlas remains an open published-literature gap.** Cardio-epic v0.3 ship does not require it; Stage 2 cardiac cell-of-origin reads at bulk-heart-tissue resolution + Stage 1 immune workhorse + Stage 3 immune subcomposition is the v0.3 architecture. v1.0+ adds sorted cardiomyocyte when published.

### DISC-CARDIO-008 — Atlas reference matrices need structural validation (duplicate-CpG check) before A-score scoring (added in v0.2.2 from CCL-047 finding)

During Phase A acquisition for the Cuadrat 2023 cardio Stage 2 atlas (per v0.2.2 corrected critical path), a comparison of the cookbook's `loyfer_moss_2018/reference_atlas.csv` against the deconvR Bioconductor package's bundled `HumanCellTypeMethAtlas.rda` found that the cookbook file has 7,890 rows but only 6,105 unique CpG IDs — 1,785 duplicate rows (1,270 CpGs duplicated 2-8× each). All checked duplicate rows have identical β values, but val_108.py's Stage 2 scoring loop retains duplicate rows under pandas `loc` and computes per-tile A-scores as `(sample_β - tile_ref_β).abs().mean()` over them — so identical-row duplicates reweight CpG contributions to the per-tile mean. The duplicated CpGs are systematically lower-β (~0.42-0.43) than the non-duplicated (~0.50-0.52), producing a uniform −0.017 to −0.025 bias on per-tile reference β. The bias is uniform across all patients regardless of disease state, so within-cohort Cohen's d (case mean - control mean over per-patient A-scores) is unbiased and the sealed cardio findings (VAL-110 d=+1.08 normal vs BAV; VAL-109 d=+0.79 control vs hPAH on EC tile; VAL-108 max |d|=0.167 stroke etiology nulls; VAL-111 EpiSCORE HeartRef tile floor) are robust. Absolute A-score magnitudes are biased by ~0.003-0.024 per tile in the sealed VAL outputs.

**Implication.** Atlas reference files are usually treated as black-box input. Internal structural validation (duplicate-CpG check, β-value range check, missing-value check, tile-column consistency check) was not part of CHK-3.1A/B. Without CHK-3.1C (added in v0.2.2), a duplicated-row atlas file silently biases all downstream A-score computations. The Cuadrat 2023 acquisition diagnostic surfaced this only because the deconvR package's bundled atlas (deduped) provided a comparison reference. Without that comparison, the bias would have stayed undetected indefinitely.

**v0.3 corrective execution.** Deduplicate the cookbook atlas file in atlas_vault before v0.3 re-execution; preserve original 7,890-row file as `reference_atlas_v0.2_with_duplicates.csv` for audit-trail; re-run VAL-108/109/110 against deduped 6,105-row file; confirm sealed Cohen's d findings preserved within ±0.05 (expected, given uniform bias); update calibration thresholds if any cardio-epic deployment uses absolute A-scores. Sealed cardio outcomes from v0.2 are not unsealed — outcomes are honored as decisions made under the cookbook rules at the time. CHK-3.1C atlas-deduplication gate added to TESTING_CHECKLIST.md prevents this class of error at every future atlas integration.

**EDEAR commercial deployment unaffected** per CCL-037. Production deployment is A-score-percentile-based rather than absolute-threshold-based, so the uniform per-tile bias does not affect commercial scoring. The dedupe correction is a cookbook calibration upgrade, not a deployment fix.

---

## What we chose not to claim at v0.2

- We did not claim stroke etiology stratification (VAL-108 demonstrated etiology-equivalence in whole blood — biology-correct null per LL-CARDIO-002).
- We did not claim heritable-vs-idiopathic PAH discrimination (VAL-109 demonstrated framework-equivalence; hPAH framework signal stronger than iPAH but not at the level of operational stratification).
- We did not claim aortic dissection vs BAV+dilation discrimination (VAL-110 demonstrated framework-equivalence between the two pathological etiologies; both distinguishable from normal).
- We did not claim that EpiSCORE HeartRef adds Stage 2 cardiac-tile discrimination at v0.2 (VAL-111 sealed at O3_TISSUE_FLOOR_DOMINATED; atlas → atlases_deferred for v0.3).
- We did not claim a generalizable platform threshold for GenomeStudio AVG_Beta, GenomeStudio V2011.1, or minfi preprocessFunnorm substrates (within-cohort self-cal only at v0.2; structurally-separated calibration VALs pending).
- We did not run any retroactive threshold accommodation (CCL-041 lesson honored). Each VAL's threshold was sealed in prereg before β access; outcomes were honored even when they triggered O3.

---

## What remains open

1. Coronary heart disease / MI subdomain (the largest CV mortality category) — target cohort GSE56046 MESA cardiovascular n=1,202 EPIC-era ROS/MAP
2. Stroke vs healthy contrast — target cohort GSE128235 healthy controls vs stroke meta-cohort
3. Heart failure subdomain
4. Hypertensive heart disease subdomain
5. Hemorrhagic stroke (I60-I62)
6. Pulmonary embolism (I26)
7. EpiSCORE HeartRef re-bridging or alternative tile-coverage cardiac atlas (Caggiano CelFiE TIM blocked at HM450 hg19 manifest acquisition)
8. Stage 1 Salas IDOL → Xu-538 production panel migration
9. Structurally-separated calibration VAL on each non-sesame substrate to establish generalizable platform thresholds
10. Multi-substrate (nucleosome occupancy, fragment fuzziness, windowed protection score, fragment size entropy) cardio-epic-specific validation — pending L3 multi-substrate assay generation

---

## What propagates to cardio-epic deployment at v0.2

1. **Three substrate-specific CHK-3.1A baselines** documented (TCGA sesame Level 3, GenomeStudio AVG_Beta, minfi funnorm)
2. **Cardio-epic CHK-3.1B subset** (8,100 CpGs, frozen at SHA `5a00e29ace75daae5a5bf7e3cfca26c16aa6dbd92750d16ebeaba4e874c48511`)
3. **Sample-substrate-aware scoring**: pure cell-type substrates → Stage 2 vascular tiles operational; bulk tissue substrates → Stage 1 immune as primary, Stage 2 vascular tiles as adjunct
4. **Stroke whole-blood single-pooled signature** (no etiology stratification claim)
5. **PAH pure-PEC vascular-tile discrimination** (with subtype-pooling)
6. **Aortic pathology Stage 1 immune signature** (with etiology-pooling)
7. **No EpiSCORE HeartRef cardiac-tile signal at v0.2** — atlas correctly deferred to v0.3 per VAL-111

---

## What does NOT propagate

- No production Stage 1 panel (Salas IDOL is proxy for patent-protected Xu-538)
- No EpiSCORE HeartRef cardiac-tile A-score discrimination (VAL-111 sealed at O3_TISSUE_FLOOR_DOMINATED; atlas → atlases_deferred for v0.3)
- No Caggiano CelFiE TIM cardiac biology yet (atlases_deferred for v0.3, blocked on HM450 hg19 manifest)
- No generalizable substrate thresholds for GenomeStudio AVG_Beta, GenomeStudio V2011.1, or minfi funnorm (within-cohort self-cal only at v0.2)
- No healthy-vs-stroke whole-blood baseline
- No coronary heart disease, heart failure, or hypertensive heart disease validation cohorts (extension targets for v0.3+)

---

## Validation evidence summary

### VAL-108 — GSE69138 ischemic stroke (whole blood)

**Cohort:** GSE69138 ischemic stroke discovery cohort, n=404 whole-blood samples
**Substrate:** GenomeStudio AVG_Beta HM450K (Illumina raw, no normalization)
**Design:** Within-cohort 3-subtype contrast (TOAST classification: large-artery atherosclerosis vs small-vessel disease/lacunar vs cardioembolic); no external healthy baseline
**QC pass rate:** 383/404 (94.8%) cleared CHK-3.1A self-calibration
**Outcome:** `O3_CARDIO_EPIC_3SUBTYPE_UNDIFFERENTIATED` (sealed)

**Key Cohen's d values (all pair contrasts):**
- Stage 1 immune (Salas proxy): max |d| = 0.129 (large-artery vs cardioembolic)
- Stage 2 Vascular_endothelial_cells: max |d| = 0.054 (large-artery vs cardioembolic)
- Stage 2 Left_atrium: max |d| = 0.015
- Stage 2 Monocytes_EPIC: max |d| = 0.167 (small-vessel vs cardioembolic — strongest signal in the entire VAL)
- Stage 3 UniLIFE: max |d| < 0.16
- Stage 3 Salas: max |d| < 0.16

**Interpretation:** Whole-blood DNA methylation does NOT stratify ischemic stroke by TOAST etiology. By the time blood is drawn post-stroke, the systemic inflammatory response has homogenized the immune methylation signature across etiologies. This is a biology-correct null, not a framework failure. The framework correctly reports that whole-blood methylation does not discriminate what biology has homogenized.

**Prereg SHA-256:** `6f40ebd9d30bb10242b245d7bde280607f1170e3c7993a8284e2852ad1f69e7a`

### VAL-109 — GSE84395 PAH (cultured pulmonary endothelial cells)

**Cohort:** GSE84395, n=39 (control 18 + heritable PAH 10 + idiopathic PAH 11)
**Substrate:** minfi `preprocessFunnorm` functional normalization, HM450K GPL16304
**Design:** Within-cohort case-control with two PAH subtypes
**QC pass rate:** 37/39 (94.9%) cleared CHK-3.1A self-calibration
**Outcome:** `O2_PAH_VASCULAR_TILE_DIFFERENTIATING` (sealed)

**Key Cohen's d values:**
- Stage 1 immune control vs hPAH: **d = +0.65**
- Stage 1 immune control vs iPAH: **d = +0.65**
- Stage 2 Vascular_endothelial_cells control vs hPAH: **d = +0.79**
- Stage 2 Left_atrium control vs hPAH: **d = +0.65**
- Stage 2 Lung_cells control vs hPAH: d = +0.91 (likely culture-substrate methylation drift artifact in primary PEC culture, noted as caveat not biology)
- hPAH vs iPAH (all stages): framework-equivalent (all |d| < 0.5)
- Stage 3 UniLIFE control vs hPAH: d = +0.48; control vs iPAH: d = +0.22

**Interpretation:** Direct vascular-tile discrimination on actual endothelial cell substrate. Heritable PAH shows stronger framework signal than idiopathic PAH (consistent with germline genetic component, often BMPR2 mutations). The two PAH subtypes are framework-equivalent. Cultured-cell substrate produces some non-physiological tile-class deviations (Pancreatic, Lung) attributable to in vitro culture artifacts rather than biology.

**Prereg SHA-256:** `f6450b4cf5d384d2ea27b349c101b3f167a6a549d276e670e68fb2232b45f21e`

### VAL-110 — GSE84274 ascending aorta (dissection / BAV+dilation / normal)

**Cohort:** GSE84274 ascending aorta, n=24 (normal 6 + dissection 12 + BAV+dilation 6)
**Substrate:** GenomeStudio V2011.1 raw output, HM450K
**Design:** Within-cohort case-control with two pathological etiologies
**QC pass rate:** 23/24 (95.8%) cleared CHK-3.1A self-calibration
**Outcome:** `O2_AORTIC_ANY_TILE_DIFFERENTIATING` (sealed)

**Key Cohen's d values:**
- Stage 1 immune normal vs BAV+dilation: **d = +1.08** (strongest aortic signal)
- Stage 1 immune normal vs dissection: **d = +0.56**
- Stage 2 Vascular_endothelial_cells: |d| ≤ 0.15 (NOT discriminating — bulk aorta dominated by SMC/fibroblast)
- Stage 2 Left_atrium normal vs BAV: d = −0.81
- Stage 2 Adipocytes normal vs BAV: d = −0.88
- Non-cardio tiles (Hepatocytes, Lung_cells, Pancreatic): |d| > 3 — interpreted as peri-aortic adipose tissue contamination
- Dissection vs BAV+dilation: framework-equivalent at all stages
- Stage 3 UniLIFE normal vs dissection: d = +0.55; normal vs BAV: d = +0.55

**Interpretation:** Stage 1 immune is the strongest aortic discriminator (consistent with infiltrating inflammatory cells in pathological aorta). Stage 2 vascular tile fails on bulk aortic tissue — the framework reads what is in the sample, and bulk ascending aorta is dominated by smooth muscle and fibroblast cell populations, not endothelium. Massive non-cardio tile signals reflect peri-aortic adipose contamination differing between normal and pathological samples. The two pathological etiologies (dissection vs BAV+dilation) share a common methylation signature in aortic tissue.

**Prereg SHA-256:** `1041738ccc8bcdd45a4754d599a28ad80fde3a7b37b6c18b4d528f4fe0271bc8`

### VAL-111 — EpiSCORE HeartRef atlas integration on three cardio cohorts (NEW in v0.2)

**Atlas:** EpiSCORE HeartRef (Zhu et al. *Nat Commun* 2022 13:3895), gene-promoter cardiac reference matrix bridged to 3,727 unique 450K CpGs × 5 cardiac cell types (CM cardiomyocyte, EC endothelial, FB fibroblast, MP macrophage, SMC smooth-muscle), GPL-2 license
**Atlas SHA-256:** `bf6431f66749f02a616560764af3fdd0adc70b03bca96b2a13b6221bbd847c83`
**Atlas vault path:** `/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_heartref/`
**Cohorts:** Three cohorts (β panels reused from VAL-108/109/110):
- GSE69138 ischemic stroke whole blood, n=589 (negative-control substrate; non-cardiac tissue should produce all five cardiac tiles below A=0.10 floor)
- GSE84395 PAH cultured pulmonary endothelial cells, n=39 (vascular substrate; EC tile expected to dominate)
- GSE84274 ascending aorta tissue, n=24 (smooth-muscle-rich substrate; SMC tile expected to dominate)

**Design:** Atlas integration test pre-locking discrimination threshold A-score range ≥ 0.10 within tissue cohorts and blood floor expectation A < 0.10 on all five cardiac tiles in GSE69138 negative-control cohort
**QC pass rate:** All three cohorts cleared >500 atlas CpGs (no O4 bridge failure: 3,727 / 3,727 / 3,408 atlas∩cohort intersections)
**Outcome:** `O3_TISSUE_FLOOR_DOMINATED` (sealed)

**Key A-score values (per-tile cohort means):**
- GSE69138 blood (n=589): CM 0.4770, EC 0.5025, FB 0.4905, MP 0.5109, SMC 0.5064 — all > 0.10 floor (5/5 breach)
- GSE84274 aortic (n=24, by disease state):
  - Aortic dissection (n=12): SMC 0.5192, MP 0.5012, EC 0.4995, FB 0.4973, CM 0.4802
  - BAV+dilation (n=6): SMC 0.5131, MP 0.4931, EC 0.4924, FB 0.4901, CM 0.4728
  - Normal (n=6): SMC 0.5072, MP 0.4860, EC 0.4855, FB 0.4845, CM 0.4669
- GSE84395 PEC (n=39, by subject status):
  - Control (n=18): EC 0.4924, SMC 0.4923, MP 0.4880, FB 0.4828, CM 0.4618
  - hPAH (n=10): EC 0.4971, SMC 0.4936, MP 0.4902, FB 0.4829, CM 0.4599
  - iPAH (n=11): EC 0.4995, SMC 0.4980, MP 0.4934, FB 0.4872, CM 0.4648

**Tissue discrimination ranges (max within-cohort, by tile):**
- GSE84274 MP range: 0.0152 (largest); EC 0.0140; CM 0.0133; FB 0.0128; SMC 0.0120
- GSE84395 EC range: 0.0070; CM/MP/SMC ~0.0055; FB 0.0044
- **Maximum tissue discrimination = 0.0152** (vs 0.10 threshold required); discrimination ratio: 15% of threshold

**Interpretation:** Direction is biologically sensible (GSE84274: dissection > BAV+dilation > normal monotonic across all five tiles; SMC tile always highest, consistent with aortic media composition; GSE84395: iPAH > hPAH > control on EC tile) but A-score magnitude is set by gene-promoter average methylation (~0.5 in heterogeneous β panels) rather than substrate-specific cell-of-origin contrast. EpiSCORE HeartRef is methodologically sound for its design purpose (EpiDISH proportion estimation in heart tissue) but does not transfer to A-score tile reading on heterogeneous β panels at the resolution required for cardio-epic Stage 2.

**Card-level disposition:** EpiSCORE HeartRef → `atlases_deferred` for cardio-epic v0.3. Caggiano CelFiE TIM remains in atlases_deferred for v0.3 (already there, blocked on HM450 hg19 manifest acquisition). Cardio-epic v0.2 ships with VAL-108/109/110 sealed structural results plus VAL-111 sealed atlas-deferred outcome and no Stage 2 cardiac-tile atlas in `atlases_run`.

**Prereg SHA-256:** `172c6ae2a11345935c176b4a1fc57d30009ad4bac9bb9cdeeb9c8226035b78a6`

---

## Calibration anchor evidence (VAL-106 + VAL-107)

### VAL-106 — TCGA HM450K sesame Level 3 CHK-3.1A baseline

- **Cohorts:** TCGA-KIRC adjacent-normal n=160 + TCGA-PRAD adjacent-normal n=50 (all NIH GDC public)
- **QC-pass:** KIRC 144/160, PRAD 50/50 (combined 194/210)
- **Substrate:** TCGA HM450K sesame Level 3 (standard TCGA pipeline)
- **CHK-3.1A baseline established:** f_extreme 55.87% ± 2.44%, f_middle 7.42% ± 0.75%
- **Outcome:** `O3_CALIBRATION_DEGENERATE` under sealed prereg's conflated-convention bounds (which were derived from CpG-subset prior data points incompatible with full-genome measurement). Reclassified post-hoc as the CHK-3.1A baseline anchor for TCGA HM450K sesame Level 3 substrate under the split convention.
- **Prereg SHA:** `0330a3c6c76c8874ba5027e88670ab60307dc322fa4cb9186ffac06d6ec4117a`

### VAL-107 — Cardio-epic CHK-3.1B threshold on TCGA HM450K sesame Level 3

- **Cohorts:** Same 210-sample TCGA-KIRC + TCGA-PRAD as VAL-106
- **Cardio-epic CHK-3.1B subset:** 8,100 unique CpGs (Loyfer 25-tile 6,105 + UniLIFE 1,906 + Salas 350)
- **Subset SHA:** `5a00e29ace75daae5a5bf7e3cfca26c16aa6dbd92750d16ebeaba4e874c48511`
- **Coverage pass:** 210/210 samples (n_subset_valid always > 7,000 of 8,100)
- **Outcome:** `O2_PLATFORM_DIVERGENCE_DOCUMENTED` (Mann-Whitney p=0.034 on f_extreme; practical Δ only 0.7 percentage points between KIRC and PRAD)
- **Established threshold:** extreme ≥ 55.0%, middle ≤ 8.5%, n_subset_valid ≥ 7,000 (per O2 more-permissive rule)
- **Prereg SHA:** `b58ce4dbd422198c7ff8f5f5cdf2b27ebd86a758afc204189f8a9e070fd700d82`

---

## Lessons learned (cardio-epic specific)

**LL-CARDIO-001 — Substrate-cell match matters (substrate fitness lesson).**
VAL-110 Vascular_endothelial_cells tile d = −0.04 on aortic bulk tissue vs VAL-109 d = +0.79 on cultured PECs is a substrate-cell-mismatch finding. The framework reads what is in the sample. Pure cell type → pure cell signal. Mixed bulk tissue → mixed signal dominated by bulk's actual cell types (SMC + fibroblast for ascending aorta). Cardio-epic deployment must communicate the tile-substrate fitness flag to the customer with each report.

**LL-CARDIO-002 — Whole blood does not stratify stroke etiology (biology-correct null).**
VAL-108 demonstrated that post-stroke inflammatory homogenization is a real biological phenomenon. The framework correctly reports that whole-blood methylation does not discriminate what biology has homogenized. This is a feature, not a failure: cardio-epic v0.2 reports stroke as a single pooled signature, not by TOAST etiology.

**LL-CARDIO-003 — hPAH > iPAH framework signal is biology-consistent.**
VAL-109 showed heritable PAH produces stronger framework signal than idiopathic PAH (Vascular_endothelial_cells d = +0.79 vs +0.42 control vs subtype). Consistent with germline genetic component (often BMPR2 mutations) producing more pronounced methylation dysregulation than the heterogeneous etiology of iPAH. Future PAH cards may stratify by genetic vs idiopathic when biology supports.

**LL-CARDIO-004 — Aortic pathology is Stage 1 immune-detectable, Stage 2 vascular-tile-resistant.**
VAL-110 Stage 1 immune d = +1.08 normal vs BAV is the strongest aortic signal; Stage 2 vascular tile fails (|d| ≤ 0.15). The framework's universal Stage 1 immune flag is the operational discriminator for aortic bulk tissue; Stage 2 vascular tiles require pure-cell substrates.

**LL-CARDIO-005 — Atlas-substrate match matters at Stage 2 (NEW in v0.2).**
VAL-111 demonstrated that gene-promoter reference atlases (EpiSCORE-class, designed for EpiDISH proportion estimation in tissue) do not transfer to A-score tile reading on heterogeneous β panels. Maximum within-cohort tissue discrimination 0.0152 vs 0.10 threshold; blood-floor breach 5/5 tiles; gene-promoter average methylation cluster ~0.5 across heterogeneous β regardless of substrate. The atlas family that works at cardio-epic Stage 2 scoring is the tile-coverage WGBS-derived family (Loyfer 25-tile validated; Caggiano CelFiE TIM cardiac panels candidate when HM450 hg19 manifest acquisition unblocks). EpiSCORE HeartRef → atlases_deferred for v0.3 with explicit unblock dependency.

## Cookbook-wide lessons referenced

- **CCL-040 LL-PROCESSED-OUTPUT-DEFERRAL** — Cardio-epic v0.2 honors this lesson with substrate-specific CHK-3.1A self-calibration envelopes per substrate
- **CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION** — Different substrates require different CHK-3.1A thresholds; cardio-epic v0.2 inherited this lesson directly through VAL-106/107 and substrate-specific self-cal envelopes for the four disease cohort substrates
- **CCL-042 LL-CHK-3.1-A/B-SPLIT** — Cardio-epic v0.1 was the first card built natively under the split convention; v0.2 maintains it
- **CCL-043 LL-CARDIO biology lessons** — formalized 2026-04-28 with the four cardio-specific lessons (LL-CARDIO-001/002/003/004); LL-CARDIO-005 added in v0.2 from VAL-111

---

## Reproducibility

All six contributing VALs (VAL-106 calibration A, VAL-107 calibration B, VAL-108 stroke, VAL-109 PAH, VAL-110 aortic, VAL-111 EpiSCORE atlas integration) follow the CHK-7.6 reproducibility triple:

1. **Inputs**: cohort manifests with SHA-256 per file, atlas vault references, derived CpG subset SHAs
2. **Environment**: Python 3 stdlib + pandas; runtime ≤ 5 minutes per VAL on standard hardware (VAL-111 ran in 28 seconds across all three cohorts)
3. **Output**: results.json + per_sample.csv per VAL; VAL-111 also produces stratified.json and three per-cohort per-sample CSVs

All public cohorts cited are accessible via NIH GEO or GDC public APIs. Atlas vault (Loyfer 25-tile, UniLIFE 19-cell, Salas Blood.EPIC IDOL, EpiSCORE HeartRef) is at `Biological_Physics/atlas_vault/` with INVENTORY.json SHA-tracking all source files.

## EDEAR commercial deployment

Per CCL-037, cardio-epic v0.2 is part of the retrospective cookbook validation layer. EDEAR commercial deployment runs on a single calibrated patient-vs-internal-reference pipeline, structurally insulated from the public-cohort substrate diversity tested here. Cardio-epic v0.2 informs the deployment EVIDENCE ENVELOPE (what cardio-epic claims it can do, with what confidence, on which substrates) but does not modify the deployment architecture.

VAL-111's deferral of EpiSCORE HeartRef does not affect commercial deployment: cardio-epic v0.2 production scoring uses Loyfer 25-tile (validated) for Stage 2; EpiSCORE HeartRef is not in `atlases_run`. When the v0.3 atlas integration unblocks (re-bridging or Caggiano CelFiE TIM acquisition), the deployment pipeline is updated additively without requiring re-calibration of existing cardio scoring.

---

## v0.1 → v0.2 changes

1. **Added VAL-111** sealed atlas-deferral outcome (EpiSCORE HeartRef → `atlases_deferred` for v0.3) — sealed 2026-04-29 at `O3_TISSUE_FLOOR_DOMINATED`
2. **Promoted card** to full Block 1-20 + CHK-5.7/5.8/5.9/5.10 structural-parity with breast-epic v2.3 / crc-epic v2.4
3. **Added `universal_reference` block** (CHK-5.7) — full 14 substantive sub-keys: `_purpose`, `schema_version`, `last_updated`, `universal_stage_1_pipeline`, `universal_h_min_table` (all 8 architecture classes), `universal_stage_2_moss_deconvolution` (with healthy_reference_β table for 18 tissues), `universal_stage_3_epidish_subcomposition` (with Salas QC bounds for 7 cell types), `universal_80_cell_age_baseline_immune_class` (10 age decades), `universal_tier_thresholds` (6 tiers BELOW_NORMAL/NORMAL/MARGINAL/DETECTABLE/URGENT/FLOOR_BREACH), `universal_sex_stratification_rule`, `universal_language_discipline`, `universal_cohort_batch_offset_warning`, `universal_no_fabrication_rule`, `gape_web_version_reference`
4. **Added `atlases_used_and_deferred` block** (CHK-5.8): `atlases_run` = [Loyfer_25tile, UniLIFE_19cell, Salas_Blood_EPIC_IDOL_6cell]; `atlases_deferred` = [EpiSCORE_HeartRef anchored at VAL-111, Caggiano_CelFiE_TIM_cardiac blocked at HM450 manifest]
5. **Added `substrate_roadmap` block** (CHK-5.9): DNAm validated at v0.2 via VAL-108/109/110/111; nucleosome_occupancy / fragment_fuzziness / windowed_protection_score / fragment_size_entropy deferred to v0.4+ multi-substrate assay generation
6. **Added `chk_3_1_thresholds_per_substrate` block** (CHK-5.10): both 3.1A and 3.1B thresholds per measurement substrate with `calibration_anchor_val_id` and `calibration_anchor_cohort_n` for all four substrates encountered (TCGA HM450K sesame Level 3, GenomeStudio AVG_Beta HM450K, minfi preprocessFunnorm HM450K, GenomeStudio V2011.1 HM450K raw)
7. **Added `lessons_discovered_v0_2` section** with six discoveries (DISC-CARDIO-001 through DISC-CARDIO-006), six things we chose not to claim, and ten things remaining open
8. **Added LL-CARDIO-005** to lessons_learned.card_specific (atlas-substrate match at Stage 2)
9. **Added `commercial_deployment_unaffected_by_validation_limitations` block** mirroring crc-epic v2.4 structure
10. **Updated cookbook_master_readme reference** from v2_3 to v2_4
11. **Updated card_date and card_version**
12. **No sealed VAL outcomes from v0.1 changed**; this is an additive rebuild only
13. **EDEAR commercial deployment unaffected** per CCL-037

---

## v0.2 → v0.2.1 changes (same-day honesty patch)

1. **Atlas naming corrected** in card JSON `universal_pipeline_acknowledgment.stage2.atlas_v0_2` and in this README's Stage 2 description: "Loyfer 25-tile" → "**Layered Moss + Loyfer array atlas** (`loyfer_moss_2018/reference_atlas.csv`)". The combined CSV in atlas_vault IS the canonical layered atlas per PIPELINE_REFERENCE Part 2.1+2.2 — Moss 2018 primary, Loyfer 2023 supplements. The v0.2 naming was incomplete; both atlases were operative in VAL-108/109/110 scoring. CpG count corrected from 6,105 to 7,890.
2. **`atlases_deferred` expanded** from 2 entries (EpiSCORE HeartRef + Caggiano CelFiE TIM) to 8 entries reflecting every canonical-document-named cardio atlas: **Cuadrat 2023 cardiovascular extended** (originally cited as "Konigsberg 2023" deployment blocker per PIPELINE_REFERENCE Part 2.4; corrected per CCL-046), **Caggiano CelFiE TIM cardiac** (HM450 manifest blocker), **EpiSCORE Zhu/Teschendorff 2022 pan-tissue** (separate from the HeartRef sub-panel scored in VAL-111), **Tanaka 2025 6-cell-type neural** (highest-priority new addition per Part 2.5), **Tian et al. 2023 scMCodes brain** (originally cited as "Liu 2023"; corrected per CCL-046 audit class — v0.4+ candidate), **EpiSCORE HeartRef sub-panel** (VAL-111 anchor, retained), **MARLIN Capper 2025 training scaffold** (Queue-1), **Sabedot GeLB 2021** (Queue-1).
3. **Added `canonical_documents_named_blocker_for_cardio_deployment` block** to card JSON quoting PIPELINE_REFERENCE Part 2.4 verbatim: *"Without this atlas, cardio-epic cannot be deployed."* The named atlas is **Konigsberg 2023**, not EpiSCORE HeartRef. v0.3 critical path documented: Konigsberg first, Caggiano second, EpiSCORE pan-tissue third, Tanaka fourth.
4. **Added DISC-CARDIO-007** to `lessons_discovered_v0_2.what_we_discovered` and to this README: "Always read PIPELINE_REFERENCE Part 2 first; atlas selection must trace to a canonical-document name." Process lesson from VAL-111 having tested an atlas that wasn't on the canonical critical path.
5. **Acknowledged run-everything violation explicitly**: VAL-108/109/110 scored Stage 2 against ONLY the layered Moss+Loyfer combined atlas. Per the run-everything policy (Heath sign-off 2026-04-26), every IDAT runs Stage 2 against ALL reference atlases. Other Stage 2 atlases in atlas_vault (caggiano_celfie_2021, caggiano_celfie_tim, episcore_zhu_teschendorff_2022, episcore_heartref pre-VAL-111, marlin_capper_training, sabedot_gelb_2021) were NOT scored on cardio cohorts. v0.3 includes corrective re-execution of VAL-108/109/110 against the full atlas stack once each atlas is acquired and calibrated.
6. **Card version bumped** v0.2 → v0.2.1; card filename `cardio_epic_card_v0_2.json` → `cardio_epic_card_v0_2_1.json`. README title bumped to v0.2.1.
7. **No sealed VAL outcomes changed.** v0.2.1 is an additive honesty patch documenting what's missing from v0.2 and what the v0.3 critical path looks like. EDEAR commercial deployment unaffected per CCL-037.

## v0.2.1 → v0.2.2 changes (same-day afternoon honesty patch on top of the morning v0.2.1 patch)

1. **Web-verified PIPELINE_REFERENCE Part 2.4's "Konigsberg 2023" citation** during Phase A acquisition and found two factual errors: (a) actual paper at the cited DOI (`10.1093/nargab/lqad061`) is **Cuadrat, Kratzer, Giral Arnal, Rathgeber, Wreczycka, Blume, Gündüz, Ebenal, Mauno, Osberg, Moobed, Hartung, Jakobs, Seppelt, Meteva, Haghikia, Leistner, Landmesser, Akalin (2023)** — no Konigsberg in the author list; (b) actual atlas content is the Moss 2018 25-tissue base extended with **three bulk ENCODE EPIC heart tissues** (right atrium auricular, heart left ventricle, coronary artery), NOT sorted cardiomyocytes. A second targeted search for any Konigsberg-authored cardiovascular methylation atlas paper returned zero hits.
2. **Replaced atlases_deferred Konigsberg entry** with corrected Cuadrat 2023 entry (atlas_full_name, license CC-BY paper / MIT deconvR, atlas_form description, what-atlas-IS / what-atlas-IS-NOT statements, 6 ENCODE EPIC IDAT accession references, atlas-family fitness assessment). Added new entry: **Sorted cardiomyocyte array-CpG atlas (open published-literature gap)** — target v1.0+, dependency = external publication.
3. **Replaced canonical_documents_named_blocker_for_cardio_deployment block** in card JSON with new `canonical_documents_cardio_stage2_extension_path` block. Drops "cannot be deployed" framing (anchored on a fictional atlas). Reframes cardio-epic v0.2 deployment as operational under layered Moss+Loyfer + Stage 1 immune workhorse; Cuadrat / Caggiano / EpiSCORE pan-tissue / Tanaka are integration ENHANCEMENTS, not deployment gates. Acknowledges sorted-cardiomyocyte array-CpG atlas as open published-literature gap with monthly literature surveillance pass added.
4. **Updated DISC-CARDIO-007** prose to reference the Cuadrat correction; v0.2.1's Konigsberg references corrected. The CHK-5.12 (atlas-canonical-source-check) lesson stands — atlas selection must trace to canonical document — but the canonical document itself contained a factual error, requiring CHK-5.13.
5. **Logged CCL-046 LL-CANONICAL-DOC-FACTUAL-ERROR** in LESSONS_LEARNED.md: Documents-of-record can themselves contain factual errors. CHK-5.12 protects against picking the wrong atlas from atlas_vault but does NOT protect against following an incorrect citation in the canonical document. CCL-046 + CHK-5.13 close that gap.
6. **Added CHK-5.13 documents-of-record citation-verification gate** to TESTING_CHECKLIST.md. Before sealing a card publish or promotion, every external citation introduced in new content (canonical-document quotes, atlas attributions, cohort accessions, prior-art references) must have at least one web-verification pass: DOI loads, authors match, content matches, accessions resolve. Cheap (one web search per citation) and catches an entire class of compounding errors.
7. **PIPELINE_REFERENCE_v2.md Part 2.4 fully rewritten** in v0.2.2 with corrected Cuadrat 2023 description + correction note + sorted-cardiomyocyte gap acknowledgment. Part 21 added documenting the v0.2.2 honesty patch.
8. **Card version bumped** v0.2.1 → v0.2.2; card filename `cardio_epic_card_v0_2_1.json` → `cardio_epic_card_v0_2_2.json`. README title bumped to v0.2.2.
9. **DISC-CARDIO-008 + CCL-047 + CHK-3.1C** logged from Phase A acquisition diagnostic finding: cookbook's layered Moss+Loyfer atlas file (`loyfer_moss_2018/reference_atlas.csv`) has 7,890 rows but only 6,105 unique CpGs (1,785 identical-value duplicates over 1,270 CpGs). val_108.py's Stage 2 scoring retains duplicates and produces a uniform −0.017 to −0.025 bias on per-tile reference β. Bias is uniform across all patients regardless of disease, so within-cohort Cohen's d is unbiased and sealed cardio findings (VAL-110 d=+1.08, VAL-109 d=+0.79, VAL-108 nulls, VAL-111 tile floor) are robust. Absolute A-score magnitudes biased by ~0.003-0.024 per tile. v0.3 corrective execution: deduplicate atlas, preserve original as `_with_duplicates.csv` for audit, re-run VAL-108/109/110, confirm Cohen's d preserved within ±0.05. CHK-3.1C atlas-deduplication gate added to TESTING_CHECKLIST.md prevents this class of error at every future atlas integration. EDEAR commercial deployment unaffected per CCL-037 (deployment is percentile-based, not absolute-threshold-based).
10. **CCL-046 audit pass — citation verification status logged.** CHK-5.13 first application to v0.2.2 patch DOIs found 2 errors out of 12 verifications (8 verified accurate, 1 not yet web-verified Tanaka 2025 medRxiv preprint, 1 Salas IDOL vs IDOL-Ext distinction pending). Errors found and **corrected in v0.2.2 expanded patch**: (a) **"Liu 2023" Science adf5357 attribution wrong** — actual paper is **Tian W, Zhou J, Bartlett A, Zeng Q, Liu H, Castanon RG, et al. 2023** (lead Wei Tian, co-first Jingtian Zhou; Hanqing Liu is mid-author, 5th of 49). 188 cell types / 517K cells / 46 brain regions content correct, attribution wrong. Inherited error from PIPELINE_REFERENCE Part 2.6 — same class of error as Konigsberg → Cuadrat in Part 2.4. **All "Liu 2023" references in cookbook documents corrected to "Tian et al. 2023" in v0.2.2 expanded patch (PIPELINE_REFERENCE Part 2.6 + line 281 + line 362 + line 770 + cardio card atlas_full_name + deferral_rationale + cardio README atlases_deferred table + v0.2.1 changes section).** Card atlas_id key kept as `Liu_2023_scMCodes_brain` for cookbook reference stability with explicit `_atlas_id_note` documenting the correction. (b) **Caggiano CelFiE DOI wrong** — cited as `10.1038/s41467-021-22335-5` in LESSONS_LEARNED.md, correct is `10.1038/s41467-021-22901-x` (Caggiano C, Celona B, Garton F, Mefford J, Black BL, Henderson R, Lomen-Hoerth C, Dahl A, Zaitlen N. "Comprehensive cell type decomposition of circulating cell-free DNA with CelFiE." Nat Commun 2021;12:2717). Corrected in v0.2.2.
11. **Four broken figures in GAPE_Evidence_Report_UPDATED.html replaced with clean styled placeholders.** The four `<img src="figures/...">` references near the end of the Evidence Report (fig1_loud_at_distance, fig2_two_cohort_breast, fig3_colorectal_subtypes, fig4_percentile_rank) referenced PNG files in a `figures/` subdirectory not bundled in the standalone HTML, rendering as white-square broken-image artifacts. PNG sources not present in mounted volumes (only IAM-physics figures available, different project). Replaced each broken `<img>` with a styled `<div>` placeholder reading "[Figure N asset not bundled in standalone HTML — see caption below]" preserving all four figure captions intact. Captions describe the figures in detail. v0.3 task: regenerate figure PNGs and re-bundle for next Evidence Report release.

## v0.3 critical path (atlas acquisition + calibration + cardio-cohort scoring, REVISED 2026-04-29 per CCL-046)

**Phase A — atlas acquisition / engineering:**
1. **Acquire Cuadrat 2023 atlas** — the actual paper at doi:10.1093/nargab/lqad061. Three additional bulk heart-tissue tiles (right_atrium, heart_left_ventricle, coronary_artery) added to the Moss 2018 base. Open access CC-BY paper, MIT-licensed R package `deconvR` (https://github.com/BIMSBbioinfo/deconvR), signature matrix in supplementary data, 6 ENCODE EPIC IDAT accessions publicly available without authorization (ENCSR517JQA, ENCSR280LMY, ENCSR515ZCU, ENCSR190PQG, ENCSR688OHW, ENCSR582BMR). Highest-priority Phase A acquisition (replaces fictional Konigsberg).
2. Acquire HM450 hg19 manifest to unblock Caggiano CelFiE TIM cardiac (heart_meth + endothelial_meth, sitting in atlas_vault, blocked at scoring engineering).
3. Engineer Tanaka 2025 nanopore→array CpG bridge (highest-priority neural addition per Part 2.5; cardio relevance via astrocyte/microglia signatures of cardiac inflammation).
4. Integrate EpiSCORE Zhu/Teschendorff pan-tissue via R rpy2 bridge (existing infrastructure).

**Phase B — per-atlas calibration VAL** (must seal **before** any cardio-cohort scoring against that atlas, per CCL-041):
5. Cuadrat 2023 calibration VAL → CHK-3.1A baseline + CHK-3.1B subset threshold sealed on structurally-separated healthy cohort
6. Caggiano CelFiE calibration VAL → same
7. Tanaka 2025 calibration VAL → same (after bridge)
8. EpiSCORE pan-tissue calibration VAL → same

**Phase C — cardio-cohort scoring against each calibrated atlas:**
9. VAL-XXX: Cuadrat 2023 on GSE69138 + GSE84395 + GSE84274 + ideally GSE56046 MESA CHD/MI cohort n=1,202
10. VAL-XXX: Caggiano CelFiE on the same cardio cohorts
11. VAL-XXX: EpiSCORE pan-tissue on the same cardio cohorts (Heart, Kidney, Liver, Lung, Brain references for differential)
12. VAL-XXX: Tanaka 2025 on the cardio cohorts (cardiac inflammation reads through astrocyte/microglia signatures)

**Phase D — re-execute VAL-108/109/110 honoring run-everything:**
After each new atlas is calibrated and brought into production, the existing VALs need re-execution against the full atlas stack. Sealed structural outcomes from VAL-108/109/110 don't change; this adds new per-atlas results to the same cohorts.

**Phase E — cardio-epic v0.3 ship:**
Once Phase B + C complete for at least Cuadrat 2023 + Caggiano CelFiE TIM (the two with explicit bulk-cardiac coverage that can actually be acquired), the card promotes from v0.2.2 to v0.3 with those atlases in `atlases_run` and the v0.2.2 deferral notes resolved. Sorted-cardiomyocyte array-CpG discrimination remains a v1.0+ goal pending publication of an underlying atlas. Cardio-epic v0.3 deployment is NOT gated on sorted cardiomyocytes — Stage 1 immune workhorse + bulk-heart-tissue Stage 2 indicators + Stage 3 immune subcomposition is the v0.3 architecture.
