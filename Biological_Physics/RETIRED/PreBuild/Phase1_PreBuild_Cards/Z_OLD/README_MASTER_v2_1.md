# IAMPerformance EDEAR Cookbook — Master README

**Generated:** 2026-04-23
**Amended:** 2026-04-24 (v2.1: universal-pipeline rule promoted to top; kidney-epic added to expansion table; lung-epic v0.2 added at multi_modal_validated tier after VAL-056 4/4 predictions pass; multi_modal_validated tier definition added; all 4 cards updated to embed full-inline universal_reference block and per-card lessons_learned section; VAL-057 consolidated external specificity test on GSE53740 added — ad-immune stays cross_platform_validated with new sex-stratification, tauopathy-specificity, and cross-cohort-normalization requirements; LESSONS_LEARNED.md master catalog created)
**Amended:** 2026-04-26 (v2.2 prep: **run-everything pipeline architecture signed off** — every IDAT runs Stage 1 + Stage 2 + Stage 3 with all panels and reference atlases regardless of any single-stage result, no conditional gating, per-class A-scores computed every tile every IDAT; spec doc `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md` is authoritative; CHK-3.2 cross-cohort baseline check promoted to mandatory-every-run; **Queue-1 atlas integration approved** for v0.3 — Tanaka 2025 6-cell neural / Konigsberg 2023 cardiac (= Loyfer cardiomyocyte sorted samples) / Zhu-Teschendorff 2022 EpiSCORE pan-tissue / Caggiano 2021 CelFiE TIM matrix / Capper 2025 MARLIN leukemia / Sabedot 2021 GeLB; **downloaded atlas inventory:** Loyfer/Moss array atlas (in production), EpiSCORE 13-tissue R-package data, Caggiano CelFiE TIM matrix (1,580 markers × 19 tissues, WGBS-region-based — caveat), Sabedot GeLB R training script (requires GSE150289 to train classifier); externally accessible but not downloaded: Tanaka 2025 (EGA controlled access), Konigsberg/Cuadrat 2023 (uses Loyfer atlas), Capper 2025 MARLIN (atlas matrix not yet released), Liu 2023 brain scMCodes (Allen Brain Cell Atlas, Queue 2); VAL-092 added — Stage 2 per-class A-score `A_terminal` on cortical-neuron-discriminating CpGs across glioma blood + glioma tissue + AD blood + healthy reference, outcome `O1_DRIFT_DISCRIMINATOR` with within-cohort vs cross-cohort asymmetry annotated; **VAL-093 added — first multi-cohort run-everything demonstration** — full 25-tile per-class A-score on Loyfer atlas at >10yr breast pre-dx window (n=47 cases, n=601 HC across GSE51057 + GSE51032), outcome `O2_SECRETORY_DISTRIBUTED`; **headline finding broader than the outcome label:** at >10yr breast pre-dx, the strongest per-tile signals are on **non-breast tiles** — Pancreatic_beta_cells d=+1.020/+0.939, Pancreatic_acinar_cells d=+0.913/+1.025, Pancreatic_duct_cells d=+0.991/+0.705 (concordant in both cohorts at p<1e-4); cycling-class tiles (Kidney d=+0.726/+0.902, Head_and_neck_larynx d=+0.746/+0.814, Colon, Upper_GI, Uterus_cervix) also fire concordantly; **Breast tile itself null: GSE51057 d=+0.198, GSE51032 d=+0.100**; Breast as top-1 ΔA call only 2/47 = 4.3% of cases; CHK-3.2 cross-cohort baseline passes cleanly (max 0.24 anchor-SDs across 25 tiles — first clean cross-cohort baseline alignment in cookbook); breast-epic claim that >10yr signal is breast-localized at the per-tile level needs softening — the framework detects multi-class drift at this window, not breast-specific localization (at-diagnosis tissue arm VAL-060 paired d=+0.676 unaffected); CCL-035 candidate established for Heath review (per-tile vs panel-CpG findings can both be true on different CpG sets); **psp-epic v0.1 stub at `exploratory_pending_replication` tier** — PSP shows replicable architectural homogenization on cortical-neuron-discriminating CpGs at fraction (VAL-091 d=−0.51) and per-CpG drift (VAL-092 d=−0.43) levels, FTD null d=−0.004 confirms PSP-specific not generic tauopathy; **surveillance + acquisition pass 2026-04-26 PM**: UniLIFE (Guo 2025 Genome Med, **NOW ON DISK** at `/home/claude/atlases/unilife/centUniLIFE_reference_matrix.csv`, 1,906 CpGs × 19 immune cell types via EpiDISH GitHub clone), Salas Blood.EPIC IDOL baseline (450 EPIC CpGs × 6 cell types, **NOW ON DISK** for direct UniLIFE-vs-Salas comparison), Salas IDOL-Ext metadata + R wrapper (**ON DISK**, RGChannelSet via lazy ExperimentHub fetch), Capper mnp_training (**ON DISK** 2.3MB, MARLIN building block); **NOT yet on disk** (network 503 from this container, retry next session): 17-tissue Ageing Atlas (Jacques bioRxiv 2025.07.21.665830), MethAgingDB (Zenodo DOI 10.5281/zenodo.15714493), Ontology-aware 190-CpG Kim 2025-26, Cuadrat 2026 Comm Bio guidelines; **EGA controlled**: Tanaka 2025, Konigsberg/Loyfer 2023; **Queue-2**: Zhou Body 206-subtype, 223-cell WGBS, Liu brain scMCodes, Guo 2025 Adv Sci inflammaging, Cell Reports 2026 PBMC scRNA+scATAC; **canonical inventory** at `/home/claude/atlases/ATLAS_DOWNLOAD_MANIFEST.md`; **revised Queue-1 #1 integration target**: UniLIFE Stage 3 head-to-head VAL vs Salas Blood.EPIC IDOL baseline on AIBL HC + GSE51057 HC (both already on disk); **surveillance is recurring** — monthly + start-of-new-card-build + on-demand, findings appended to GAPE Reproduction Paper as §7.17, §7.18, etc.; full acquisition log in §7.17)
**Version:** v2.1 (corrects v1 errors; rebuilds against Evidence Report ground truth; makes the universal rule explicit)
**Author:** Heath W. Mahaffey, IAMPerformance Inter-Domain Research Institute

---

## ABSOLUTE RULE — Stage 1 terminology (CCL-030 + CCL-031, formalized 2026-04-25)

**Read this before any card review or new card build.** Earlier cards drifted into using "bidirectional" loosely. This rule is what fixes that drift. It applies to every Cookbook card, every VAL outcome, every LESSONS_LEARNED entry, and every customer-facing documentation block going forward.

### The two distinct Stage 1 tests

**Test 1 — pooled A_immune on the full Xu-538 panel.** Standard scoring. `A_pooled = mean over Xu-538 CpGs of [ H(β) / H_min(immune) ]` where H_min(immune) = 0.838889. **Direction-agnostic at the per-patient level due to Shannon symmetry** — H(β) peaks at β = 0.5, so a CpG moving from β = 0.7 → 0.5 produces the same per-patient entropy elevation as a CpG moving from β = 0.3 → 0.5. **Operational on every disease in the record. This is the test every Stage 1 validation actually runs.**

**Test 2 — lymphoid-marker vs myeloid-marker sub-panel split.** Run pooled A_immune separately on the lymphoid-assigned subset of Xu-538 CpGs and on the myeloid-assigned subset; compare directions. Opposite directions with comparable magnitudes = AD-style lineage-level bidirectional cancellation. **Test 2 requires per-CpG lineage assignment from an immune-cell-type methylation atlas (Salas IDOL-Ext or equivalent). This is OQ-2026-01 immune-atlas staging — currently NOT runnable on any disease.** Until OQ-2026-01 is operational, NO card can claim lineage-level bidirectional cancellation as a confirmed mechanism.

### What "bidirectional cancellation" means (CCL-031, the rule that closes prior loose usage)

The phrase **"bidirectional cancellation" is reserved EXCLUSIVELY** for one specific operational pattern:

> **Test 1 (pooled A_immune on the full Xu-538 panel) NULLS on a cohort where it was expected to pass, AND a directional ±1 z-scored panel built on the same Stage 1 panel PASSES on the same cohort or independent holdout.**

This is the AD-instance pattern. AD via VAL-050 (pooled d = +0.077, AIBL holdout) + VAL-051 (directional 7-CpG Rule A d = +0.624, same AIBL holdout) is the canonical example. PDAC via VAL-066/067/068 (pooled CIs straddle zero) + VAL-069 (directional 324-CpG panel d = +1.51 on TCGA-PAAD holdout) is the second case exhibiting this pattern. **The mechanism for the AD-instance pattern is currently unresolved between AD-style lineage cancellation, z-scoring sensitivity gain, and cohort/batch structure** (per CCL-028). Test 2 is what would distinguish them; Test 2 is pending OQ-2026-01.

### What is NOT bidirectional cancellation, even though it superficially looks similar

| Pattern | Cards exhibiting it | Correct terminology |
|---|---|---|
| Pooled Test 1 negative in blood, positive in tumor TIL (same disease, different compartment) | crc-epic | **compartment-direction-flip (CCL-019)** — NOT bidirectional cancellation |
| Pooled Test 1 different sign across diseases on same panel | breast (positive) vs CRC (negative) | **cross-disease direction difference (CCL-006)** — NOT bidirectional cancellation |
| Pooled Test 1 passes cleanly with negative-direction-dominant cohort-mean Δβ | cervical-epic VAL-073 (37% positive Δβ, pooled d = +0.73) | **pooled-positive (Shannon-symmetric)** — NOT bidirectional cancellation |
| Per-CpG cohort Δβ direction percentage clustered near 50% (without a pooled null) | n/a (not a finding by itself) | **descriptive only, NOT a mechanism diagnostic** — NOT bidirectional cancellation |

### Diseases currently in each category (record as of 2026-04-25)

- **Pooled-positive (Test 1 passes cleanly):** breast-epic, lung-epic, prostate-epic (Stage 2), hcc-epic, **cervical-epic** (VAL-073 anchor)
- **Pooled-negative compartment-flip (CCL-019):** crc-epic (blood d = −0.33; tumor TIL d = +1.066)
- **Pooled-null + directional-pass (AD-instance pattern; mechanism unresolved per CCL-028):** ad-immune (VAL-050/051), pancreatic-epic (VAL-066/067/068/069)
- **Lineage-confirmed bidirectional cancellation:** **NONE** — Test 2 not yet operational on any disease

### The single-sentence summary for any documentation block

> Bidirectional cancellation is the AD-instance pattern: Test 1 pooled A_immune nulls cross-cohort AND a directional ±1 z-scored panel built on the same Stage 1 panel passes on holdout. Compartment-direction-flips, cross-disease direction differences, and negative-direction-dominant cohort-mean Δβ are NOT bidirectional cancellation, even when they superficially resemble it.

### Operational checklist for every future card or session

1. When describing CRC's blood-vs-tumor sign difference: use **"compartment-direction-flip"** (CCL-019), never "bidirectional."
2. When describing breast vs CRC's panel-direction difference: use **"cross-disease direction difference"** (CCL-006), never "bidirectional."
3. When describing AD or PDAC: use **"pooled-null + directional-pass operational pattern; mechanism unresolved pending OQ-2026-01"**, NOT "confirmed bidirectional-cancellation disease."
4. When reporting per-CpG cohort Δβ direction percentages: explicitly mark as **"DESCRIPTIVE ONLY, NOT a mechanism diagnostic per CCL-030."**
5. When answering CCL-027 question (iv) lymphoid/myeloid expected pattern: explicitly flag as **"Test 2 placeholder; pending OQ-2026-01 immune-atlas staging; literature-anchored expected pattern only at v0.1."**

Cards verified clean under this rule (2026-04-25): **pancreatic-epic v0.1, crc-epic v2.3, cervical-epic v0.1.**

---

## ABSOLUTE RULE — Diagnostic order before any null-finding outcome (CCL-032, formalized 2026-04-25)

**Read this before publishing any null or negative-direction VAL outcome.** Cervical-epic burned ~4 hours on VAL-076/077 because Walther treated framework numbers as biology before checking whether the data was interpretable as biology. CCL-032 is the rule that prevents the next card from repeating that mistake.

### The diagnostic order is fixed: data integrity → biology → framework

Every cohort run with a null or negative-direction reading must complete these three checks in sequence BEFORE the outcome is drafted. Skipping or reordering produces overclaim+revert cycles.

**1. Data integrity check.** Verify the file is what you think it is. Check the source paper's Methods to find the exact pipeline that produced the deposited file. Run the β distribution sanity check (CHK-3.1: real raw β has >30% at extremes [<0.1 or >0.9] and <10% in [0.4, 0.6]; flat near 0.5 = residual M-values, NOT raw β). Run the cross-cohort healthy baseline check (CHK-3.2: if healthy mean A differs by >1 SD from anchor cohort, the cohorts are not directly comparable). Run the panel coverage report. Run the saturation flag check. Spot-check sample-group assignments.

**2. Biology consistency check.** If data integrity passes, ask: is the result consistent with the published clinical-grade panels for this disease? Is it consistent with the cohort's own published findings? Is it consistent with the established disease immunology literature? If clinical-grade panels achieve strong signal on the same cohort where the framework reads null, **the framework's panel does not transfer** — that is a transferability finding, not a "the disease has no signal" finding.

**3. Framework finding (last, not first).** Only after data integrity AND biology consistency are both validated can a null/negative-direction reading be claimed as a framework-relevant finding. The outcome label O3_NULL is reserved for nulls on validated specimen pathways with clinical-grade-corroborated biology. The outcome label O5_NEGATIVE_DIRECTION is reserved for negative-direction readings with verified data and corroborated biology. The outcome label O6_UNEXPECTED is the correct label whenever data integrity is uncertain or the result contradicts well-characterized disease biology.

### What CCL-032 forbids

- **Drafting an outcome.md as O3_NULL or O5_NEGATIVE without first running CHK-3.1 / CHK-3.2 / CHK-3.5.**
- **Treating a null reading on a novel specimen pathway (LBC, urine, saliva, stool, CSF) as a framework finding without explicit panel-transferability evaluation.**
- **Using Cohen's d as biological evidence when the input β values are residual/processed, not raw.** Residual M-values from EWAS regression pipelines map to β ≈ 0.5 across the panel under the standard β = 2^M / (1+2^M) conversion, producing artifactual A-scores that look like "no signal" when the underlying biology is intact.
- **Ignoring published clinical-grade panels.** If FAM19A4/miR124-2 (cervical), SEPT9 (CRC), ADAMTS1/BNC1 (PDAC), SHOX2/PTGER4 (lung), or PITX2 (breast) achieve strong signal on the cohort, the cervical/CRC/PDAC/lung/breast immune signal IS there — a framework null on that cohort is a transferability finding.

### What CCL-032 requires

- **Every null/negative outcome.md cites the CHK items it passed.** This is verifiable: future cards with null outcomes must show their work.
- **TESTING_CHECKLIST.md is the FIRST tool call** at the start of any new card or new VAL session. The checklist is in `/mnt/user-data/outputs/cookbook_v2.1/TESTING_CHECKLIST.md` and is the persistent memory across compactions. Per memory #9 absolute rule.
- **LESSONS_LEARNED.md is the SECOND tool call.** Read the master cookbook `LESSONS_LEARNED.md` and any per-card analog before starting.

### Cards that have applied CCL-032 retroactively

cervical-epic v0.1: VAL-076 reclassified from O3_NULL to O6_UNEXPECTED (panel transferability flag). VAL-077 reclassified from O3_NULL to O6_UNEXPECTED (residual-M-values data-integrity flag, defer to v0.2+ raw IDAT processing). VAL-074 and VAL-081 reclassified to O5_NEGATIVE_DIRECTION with explicit cohort-baseline-heterogeneity flag. Card tier set to `exploratory_with_cohort_heterogeneity` rather than the inflated `cross_platform_validated` that single-cohort VAL-073 anchor would have produced under loose application.

---

## ABSOLUTE RULE — Run-everything pipeline architecture (CCL-033, signed off 2026-04-26)

**Heath signed off 2026-04-26.** Every IDAT runs Stage 1 + Stage 2 + Stage 3 with **all panels and all reference atlases regardless of any single-stage result**. No conditional gating. Per-class A-scores are computed for every tissue every IDAT. Display logic in the patient report can collapse uninformative tiles ("17 tissues NORMAL — collapsed"); the underlying scoring is exhaustive on every IDAT.

The authoritative pipeline reference document is **`EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md`** (2026-04-26). Any conditional-gating language in pre-2026-04-26 cards or READMEs is superseded — the v2 reference doc is the operating spec. Card READMEs and `commercial.web.py` retain conditional-gating language pending the v0.3 propagation pass; until that pass lands, scoring code follows the v2 reference, display code follows the older language, and the result is a slight asymmetry that future cards close at next version bump.

### Why this is the right architecture

Stage 1 immune A-score can null when a disease drives CpGs bidirectionally (the AD-instance pattern, CCL-030 — VAL-050 was null at d=+0.077, VAL-051 directional was d=+0.624). The pre-diagnostic breast secretory-class signal at >10yr (d=−1.226 in VAL-047 Phase 6) is on the *negative* side and would never be reached by an elevation-gated Stage 2. Heme-epic explicitly looks for Stage 2 NULL on solid organs as the *diagnostic feature*. PSP/CBD reads cortical-neuron at d=−0.51 (BELOW_NORMAL) — gating on elevation misses every below-normal pattern. **A patient with early AD + early breast cancer + chronic inflammation + cardiovascular drift fires four anomaly patterns simultaneously; gating on the first signal that crosses threshold filters the others through that first signal's lens.** Run-everything surfaces multi-disease patterns that gating hides.

### What this requires operationally

1. **CHK-3.2 (cross-cohort baseline) is mandatory every run, never optional.** Under run-everything a single platform-induced baseline shift on a single tile silently corrupts every multi-disease pattern that uses that tile. CHK-3.2 is the structural defense. See `TESTING_CHECKLIST.md` Stage 3 for the full mandatory-every-run rules.
2. **Per-class A-scores must be computed for every tile, every IDAT.** The 8 architecture-class H_min values (terminal=0.7728, immune=0.838889, secretory=0.843264, cycling=0.856055, stromal=0.862950, progenitor=0.852216, stem_adult=0.873718, stem_pluri=0.982166) anchor a per-class A-score on every Stage 2 tissue tile that the patient's IDAT touches.
3. **Pre-registered outcome criteria must enumerate multi-disease detection patterns,** not only "disease X vs HC."
4. **Patient-facing reports must surface anomaly stack, not single-disease verdict.** Display logic collapses NORMAL tiles; underlying scoring exposes everything.

### What this does NOT change

- The 8 architecture classes and their H_min values. Frozen.
- The Stage 1 panels (Xu-538 + AD Rule A 7-CpG + PDAC 324-CpG + Kresovich comparator).
- The Stage 2 layered atlas (Moss 2018 primary + Loyfer 2023 array supplementary).
- The Stage 3 EpiDISH RPC + Salas 2018 6-cell reference.
- The 80-cell healthy baseline.
- The Test 1 vs Test 2 distinction (CCL-030/031).
- The data-integrity → biology → framework diagnostic order (CCL-032).

### First demonstration

**VAL-092 (2026-04-26)** is the first VAL that explicitly runs under the run-everything architecture. Stage 2 per-class `A_terminal` on cortical-neuron-discriminating CpGs computed for every IDAT in the glioma blood + glioma tissue + 3 AD blood + healthy reference cohort set, regardless of Stage 1 status. Outcome `O1_DRIFT_DISCRIMINATOR` per pre-registered criteria with within-cohort vs cross-cohort asymmetry annotated. Demonstrated detection of the PSP/CBD class-specific architectural homogenization (d=−0.43, p=0.010 BELOW_NORMAL) that would have been missed under elevation-gated Stage 2 — and the AD-blood null on per-CpG architectural drift that closes the AD vs glioma differential when combined with VAL-091's fraction null.

### Queue-1 atlas integration approved (v0.3 task list)

Six published reference atlases approved 2026-04-26 for v0.3 integration into the run-everything Stage 2 reference layer:

1. **Tanaka 2025 6-cell neural cfDNA atlas** (medRxiv 2025.10.07.25337503v2, nanopore methylation atlas, AD/PD/ALS discrimination AUC > 0.98). HIGHEST priority — answers the AD-vs-LGG-vs-PD-vs-ALS-vs-MS differential directly via cortical / dopaminergic / spinal motor / astrocyte / Schwann / microglia separation.
2. **Konigsberg 2023 cardiac extended atlas** (NAR Genomics 10.1093/nargab/lqad061, 28-cell-type extended atlas with sorted cardiomyocytes). Cardio-epic deployment dependency.
3. **Zhu-Teschendorff 2022 EpiSCORE pan-tissue atlas** (Nat Methods 10.1038/s41592-022-01412-7, 42 cell types × 13 solid tissues, R package `aet21/EpiSCORE` v0.9.6). Same Teschendorff lab as EpiDISH.
4. **Caggiano 2021 array-native neural references** (referenced in glioma-epic v0.3 task list). Oligodendrocyte / astrocyte / microglia separation.
5. **Capper 2025 MARLIN leukemia 450K/EPIC reference** (n=2,540 acute leukemia, 1,461 AML / 686 B-ALL / 266 T-ALL). Heme-epic v0.2 myeloid arm cross-cohort replication.
6. **Sabedot 2021 GeLB external classifier** (Mendeley deposit cgrz6zztfg, EPIC-array glioma blood classifier). Already accessible Tier 1 — engineering, not validation.

Liu 2023 brain scMCodes (Science 10.1126/science.adf5357, 188 single-cell brain types) is Queue 2 — discriminator regions can be downsampled to array CpGs but the engineering is heavier than the Queue 1 set.

**No Queue-1 atlas is in production scoring as of 2026-04-26.** A VAL that names a Queue-1 atlas may use the published external classifier (Sabedot GeLB output as a comparator arm, MARLIN as a leukemia subtype anchor) but cannot claim integrated A-score scoring against H_min until the atlas-integration VAL has landed.

---

## What this Cookbook is

A disease-by-disease operational reference for EDEAR clinical deployment. Each card takes the same IDAT input through the same universal pipeline and produces a disease-specific call. This Master README documents the pipeline; each card specifies the disease-specific direction, expected tissue localization, and validation tier.

---

## Glossary — terms and abbreviations used throughout the Cookbook

Cards and validation records use a large shared vocabulary. Terms are grouped by category for quick reference.

**Platforms and file formats.**
- **Illumina EPIC 850K** — the current standard DNA methylation microarray, measures β values at approximately 865,000 CpG sites. Current-generation platform for clinical and research methylation.
- **Illumina HM450** — HumanMethylation450, the predecessor to EPIC. Measures β at approximately 485,000 CpG sites. All TCGA methylation data from before 2016 is HM450. About 90% of HM450 probes are retained in EPIC.
- **Illumina HM27** — HumanMethylation27, the oldest methylation array. Only 27,000 CpGs, too thin for Xu-538 panel coverage. Exclude from analyses.
- **IDAT** — Illumina intensity data file. The raw output from a methylation array — one red-channel and one green-channel file per sample, containing fluorescence intensities at each probe. β values are computed from IDATs.
- **CpG** — a cytosine nucleotide followed by a guanine nucleotide in the DNA sequence, the primary site of DNA methylation in mammalian genomes. Each methylation array probe targets one CpG.
- **β value** — the methylation fraction at a CpG, ranging 0 (fully unmethylated) to 1 (fully methylated). Computed as β = M / (M + U + 100) where M is methylated-probe intensity and U is unmethylated-probe intensity.

**Data sources and repositories.**
- **TCGA** — The Cancer Genome Atlas. A public NIH-funded database of tumor samples with matched methylation, RNA-seq, mutation, and clinical data for approximately 30 cancer types. Each cancer type has its own accession code: TCGA-BRCA (breast invasive carcinoma), TCGA-COAD (colon adenocarcinoma), TCGA-LUAD (lung adenocarcinoma), TCGA-LUSC (lung squamous cell carcinoma), TCGA-PRAD (prostate adenocarcinoma), TCGA-LIHC (liver hepatocellular carcinoma), TCGA-STAD (stomach/gastric adenocarcinoma), TCGA-BLCA (bladder urothelial carcinoma), TCGA-CESC (cervical squamous cell and endocervical adenocarcinoma), TCGA-PAAD (pancreatic adenocarcinoma), TCGA-GBM (glioblastoma), TCGA-LGG (lower-grade glioma), and more. The codes are 3-4 letter tissue abbreviations.
- **GDC** — Genomic Data Commons (`https://api.gdc.cancer.gov`). NIH's open-access portal for TCGA data. Methylation β values are open access with no application required.
- **GEO** — Gene Expression Omnibus (`https://www.ncbi.nlm.nih.gov/geo/`). NIH's public repository for individual published methylation studies. Each study has a GSE accession (e.g. GSE51057, GSE269244).
- **dbGaP** — database of Genotypes and Phenotypes. NIH's gated-access repository for sensitive human data (UK Biobank, Health ABC, Rotterdam Study, Framingham, Sister Study). Requires a data-use application taking 2 to 12 weeks for approval.
- **ROSMAP** — Religious Orders Study and Memory and Aging Project. A longitudinal Alzheimer's cohort with brain cortex methylation data from over 700 deceased participants. Provides the primary AD tissue-level reference.
- **BDR** — Brains for Dementia Research. A UK-based AD brain-tissue repository with methylation data on approximately 1,400 samples.
- **AIBL** — Australian Imaging, Biomarkers and Lifestyle flagship study of ageing. Longitudinal AD cohort with blood methylation data. Primary validation cohort for VAL-051.
- **AddNeuroMed** — European multicenter longitudinal AD cohort. Blood methylation; cross-platform replication cohort for VAL-052.
- **Sister Study** — US prospective cohort of approximately 50,000 women whose sister has had breast cancer. Blood methylation at EPIC. Source of the Xu-538 panel (Xu 2020 JNCI).
- **EPIC-Italy** — European Prospective Investigation into Cancer and Nutrition, Italian arm. Pre-diagnostic blood methylation for breast, CRC, and other cancers. Source of GSE51057 and GSE51032 used in VAL-047 Phase 9/12.
- **UK Biobank** — large UK prospective cohort (approximately 500,000 participants) with methylation on a subset. dbGaP-gated. Source of VAL-046 lung arm at cohort level.

**Biological and methodological terms.**
- **ccfDNA** — circulating cell-free DNA. Fragments of DNA shed from cells throughout the body and carried in the plasma. Different from "whole-blood leukocyte DNA" which is genomic DNA extracted from blood immune cells.
- **buffy coat** — the thin layer of white blood cells and platelets separated by centrifugation between plasma and red blood cells. Standard substrate for whole-blood leukocyte methylation. Approximately 99% immune cells.
- **NAF** — nipple aspirate fluid. A small volume of fluid extracted from the nipple by gentle suction or massage, containing epithelial cells shed directly from the breast ductal and lobular system. A minimally invasive specimen pathway for breast-cell-specific methylation.
- **Ductal lavage** — saline flush of a breast duct through a catheter inserted at the nipple. Yields more breast epithelial cells than NAF but is more invasive. Research use mainly.
- **FNA** — fine-needle aspirate. A thin needle samples cells from a specific tissue area; commonly used for palpable or imaged lesions in breast, thyroid, and other organs.
- **Bronchoalveolar lavage (BAL)** — saline flush of a lung airway through a bronchoscope. Recovers lung epithelial and alveolar cells; a lung-specific specimen pathway.
- **NNLS** — non-negative least squares, the numerical method Moss 2018 uses to deconvolve plasma ccfDNA methylation into per-tissue fractions.
- **Moss 2018 atlas** — the reference set of approximately 7,890 tissue-specific marker CpGs across 25 human tissues used for NNLS deconvolution. Public; published in Moss et al. 2018 Nature Communications Supplementary Table S4, mirrored on GitHub at `nloyfer/meth_atlas`.
- **Xu-538 panel** — the 538 CpG immune-class panel derived from Xu 2020 Sister Study breast cancer plus EPIC-Italy replication. Used in Stage 1 of every Cookbook card. Panel SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`.
- **H_min** — the information-theoretic architectural floor for a cell identity class, derived from G-003b MCMC on reference cell populations. Used to normalize the A-score. Values per class: immune 0.838889, secretory 0.843264, cycling 0.856055, terminal 0.772837, stromal 0.862950, stem_adult 0.873718, progenitor 0.852216, stem_pluri 0.982166. The derivation method is proprietary (US Provisional Patents 64/012,720 and 64/014,568); the values themselves are documented here.
- **A-score** — architectural drift score. A = mean(H(β) / H_min) across a panel of CpGs, where H(β) is the Shannon entropy of the β value. A > 1 indicates drift above the architectural floor.
- **Shannon entropy H(β)** — −β·log₂(β) − (1−β)·log₂(1−β). Peaks at β = 0.5 (maximum uncertainty) and equals 0 at β = 0 or β = 1 (full certainty).
- **Cohen's d** — standardized effect size. d = (mean_case − mean_control) / pooled_SD. Values of 0.2 small, 0.5 medium, 0.8 large. Used for all Cookbook effect-size reporting.
- **MCMC** — Markov Chain Monte Carlo. Bayesian sampling method used to derive posterior distributions for H_min values from reference cell population methylation data. The G-003b MCMC posteriors are the source of the Cookbook's frozen H_min constants.
- **EpiDISH RPC** — Epigenetic Dissection of Intra-Sample Heterogeneity, Robust Partial Correlations mode. A reference-based method for decomposing whole-blood methylation into immune cell-type fractions (CD4, CD8, NK, B-cell, monocyte, neutrophil). Used in Stage 3.
- **Salas 2018 bounds** — the expected healthy ranges for immune cell fractions in whole blood, used as a QC check on EpiDISH output.

**Statistical and validation terms.**
- **Pre-registration (pre-reg)** — the practice of SHA-256-hashing the analysis plan before any data access, committing it to a dated public record (GitHub), and running the analysis exactly as specified. Protects against post-hoc cherry-picking. Every Cookbook validation run is pre-registered.
- **VAL-XXX** — validation run numbering. VAL-001 through VAL-036 are framework-level validations of the IAM architecture. VAL-037 onward are more recent per-card or per-cohort runs. Each has a pre-registration document, a seal file with a SHA-256 hash and timestamp, a reproducible analysis script, and a SHA-locked results JSON.
- **Outcome codes O1-O5** — pre-specified outcome categories locked in the pre-registration. Typically O1 = validated at primary threshold, O2 = partial or below-threshold-positive, O3 = null or opposite direction, O4 = substrate-specific or sub-cohort finding, O5 = unexpected pattern requiring investigation.

**Cookbook terminology.**
- **Card** — the operational specification for one disease. A card JSON contains the Stage 1 panel, Stage 1 expected direction, Stage 2 expected tissue localization, tier thresholds, clinical action paths, validation evidence summary, known limitations, and embedded universal reference block.
- **Tier** — the validation strength a card has earned. See the "Validation tiers per card" section below for the full list (cross_platform_validated, cohort_screening_validated, stage_2_only_validated, multi_modal_validated, tissue_arm_validated, substrate_restricted, post_dx_only, exploratory, null_documented).
- **EDEAR** — Early Detection, End-of-life, Aging, Research. The clinical platform the Cookbook operationalizes. Covers the full life-course of architectural drift detection.
- **GAPE** — Generalized Architectural Performance Engine. The underlying analytical engine that operates on IDATs; the Cookbook is the disease-facing application layer on top of GAPE.
- **CCL-### / PL-### / disease-LL-###** — lessons-learned numbering. CCL = cross-card lesson (applies to multiple cards), PL = process lesson, disease-LL = per-card disease-specific lesson.

---

**Security boundary.** Cards contain operational constants (panels, H_min values, tier thresholds, expected directions, localization targets, QC bounds). Cards do NOT contain framework derivations, H_min MCMC protocol, class-assignment rule, or IAM physics. Those are held upstream under USPTO provisional patents 64/012,720 and 64/014,568.

---

## THE UNIVERSAL RULE — READ THIS FIRST

Every patient's first test is the same. There is no disease-specific Stage 1 substrate, no disease-specific Stage 1 class, no disease-specific Stage 1 H_min. The first line of defense is always the immune class because buffy-coat DNA is approximately 70% immune cells and the immune compartment sees everything — breast, colorectal, lung, pancreas, liver, prostate, gastric, bladder, cervical, kidney, glioma, AD, chronic infection, autoimmune — via its response to upstream tissue drift.

**Stage 1 is always:**
- **Substrate:** buffy-coat DNA (peripheral blood leukocytes)
- **Class measured:** immune
- **Primary panel:** Xu-538 (panel_id `Xu2020_breast_cancer_replicated_full`, SHA-256 `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`)
- **H_min:** 0.838889 (G-003b MCMC posterior, R-hat < 1.001)
- **Scoring:** pooled-entropy A_immune AND directional A_dir (both reported on every IDAT, per the Directional-Score Principle)
- **Disease-specific directional panels** (e.g., the 7-CpG AD Rule A panel) run alongside the Xu-538 pooled score on the same IDAT. They do not replace Stage 1; they supplement it.

**What varies by card:**
- The expected **direction** of the Stage 1 departure (breast positive, CRC negative, AD directional-positive on AD-specific CpGs with pooled-null expected)
- The expected **Stage 2 tissue localization** (breast_ductal for breast, colon_epithelial for CRC, NULL for AD)
- The card's **tier thresholds** applied to the disease-appropriate scoring direction
- The **recommended clinical workup** when the card fires

**What does not vary:** the Stage 1 substrate, class, primary panel, H_min, or scoring method. The universal pipeline is identical for every patient, every card, every disease. This is what makes the Cookbook a single reusable instrument rather than 12 independent assays.

---

## The universal pipeline — three stages

### Stage 1 — Immune-class A-score flag (every IDAT goes here first)

**Input:** Illumina 450K or EPIC 850K IDAT from buffy-coat DNA (peripheral blood leukocytes). Buffy coat is approximately 70% immune cells, so the architectural class being measured in Stage 1 is **immune**.

**Computation:** For each CpG in the Xu-538 panel, compute H(β) = −β·log₂(β) − (1−β)·log₂(1−β). Divide by H_min(immune) = 0.838889 (G-003b MCMC posterior, R-hat < 1.001). Sample A-score is mean across panel (pooled entropy). Additionally, for each disease-specific directional panel the patient's clinician has requested (e.g., AD 7-CpG Rule A), compute A_dir = sum of z(β_cpg) × direction_cpg across the directional panel CpGs.

**Outputs:**
- A_immune_pooled (pooled entropy score on Xu-538)
- A_dir (directional score for each directional panel loaded; AD panel is the current standard directional panel)
- Age-matched percentile against the 80-cell healthy baseline reference
- Tier call: NORMAL (A < 1.01) / MARGINAL (≥1.01) / DETECTABLE (≥1.05) / URGENT (≥1.07) / FLOOR BREACH (≥1.10). For diseases with expected negative direction (e.g., CRC), tier call uses depression-from-baseline per the CRC card's inverted tier thresholds.
- Per-CpG delta-β table for QC

**Interpretation:** Stage 1 flags architectural departure in the immune compartment. It does NOT identify the source. A flag means "something is driving immune drift"; it does NOT distinguish breast cancer from CRC from AD from chronic infection from autoimmune. That is Stage 2's job for solid-organ cancers and Stage 3's job for non-solid-organ immune conditions.

**Validation anchor:** VAL-047 Phase 9 + Phase 12 (live re-run 2026-04-23 on SHA-locked GEO matrices, GSE51057 and GSE51032, combined n=1,174). Breast d = +0.45 to +0.71 pooled pre-dx, +1.36 to +1.78 at >10yr. CRC d = −0.33 pooled pre-dx (inverted), p = 0.009.

**Universal Stage 1 caveat — bidirectional drift cancellation.** Pooled-entropy A-score is the correct primary metric when a disease drives its immune-class CpGs in a uniform direction (breast up, CRC down — both are uniform-direction patterns at the panel level; the binary Shannon entropy captures both). Pooled-entropy A-score FAILS when a disease drives CpGs bidirectionally within the same panel — some up, some down — because H(β) is symmetric around β = 0.5 and signed Δβ contributions cancel. VAL-050 demonstrated this on AD: 10 of 18 CpGs had positive Δβ, 8 had negative, and the pooled A-score gave d = +0.077 (null) despite real per-CpG signal. VAL-051 recovered the signal by assigning each CpG a frozen direction (+1 or −1) and multiplying before summing: same CpGs, same cohort, directional d = +0.624.

Every disease card must specify BOTH the pooled-entropy and the directional expectation. For a disease with uniform-direction drift (breast, CRC so far), pooled-entropy IS the directional answer because the direction is uniform. For a disease with bidirectional drift (AD so far, likely other neurodegenerative and autoimmune conditions), only the directional panel recovers the signal. Stage 1 reports BOTH scores on every IDAT to guard against this failure mode. If pooled-entropy is null but the directional A-score for any established-card disease exceeds tier threshold, that disease's card fires regardless of pooled A-score. This is the Directional-Score Principle (VAL-051).

### Stage 2 — Moss 2018 NNLS tissue-of-origin deconvolution

**Input:** Same IDAT, now decomposed into estimated per-tissue β values using the Moss 2018 25-tissue methylation atlas as the NNLS reference (`scipy.optimize.nnls`). Output is an 18-tissue β vector: colon_epithelial, lung_epithelial, gastric_epithelial, bladder_epithelial, cervical_epithelial, kidney_epithelial, hepatocyte, pancreatic_exocrine, breast_ductal, prostate_epithelial, neuron, oligodendrocyte, vascular_endothelial, fibroblast, neutrophil, lymphocyte, monocyte, hsc.

**Computation:** For each tissue, compute per-tissue A-score using that tissue's class H_min:
- cycling epithelium (colon, lung, gastric, bladder, cervical, kidney): H_min = 0.856055
- secretory (breast_ductal, hepatocyte, pancreatic_exocrine, prostate_epithelial): H_min = 0.843264
- terminal (neuron, oligodendrocyte): H_min = 0.772837
- stromal (vascular_endothelial, fibroblast): H_min = 0.862950
- immune (neutrophil, lymphocyte, monocyte): H_min = 0.838889
- stem_adult (hsc): H_min = 0.873718

ΔA_tissue = A_tissue(patient) − A_tissue(healthy reference β). Healthy reference β per tissue, from Moss 2018 Table S1: colon 0.741, lung 0.738, gastric 0.739, bladder 0.737, cervical 0.740, kidney 0.739, hepatocyte 0.742, pancreatic 0.738, breast_ductal 0.744, prostate 0.743, neuron 0.779, oligodendrocyte 0.775, vascular_endothelial 0.731, fibroblast 0.728, neutrophil 0.762, lymphocyte 0.751, monocyte 0.758, hsc 0.734.

**Outputs:**
- 18-tissue β vector
- 18-tissue A-score vector
- 18-tissue ΔA vector
- Top-1 localization: tissue with max ΔA
- Top-3 localization: top three tissues with their ΔA values
- Confidence: ΔA ratio of top-1 to top-2 (if top-1 ΔA > 2× top-2 ΔA, high confidence; if within 1.5×, ambiguous)
- Salas 2018 QC gate: neutrophil 45–75%, lymphocyte 20–40% (subdivided CD4+ 10–30%, CD8+ 5–25%, NK 3–15%, B 3–15%), monocyte 3–12%. IDATs outside these bounds get a QC flag and Stage 2 output is not released.

**Validation anchor:** VAL-041 (10/10 top-1 correct across 10 cancer types, mean max ΔA = +0.174). Colorectal plasma: colon_epithelial β drops to 0.612 (max ΔA). Breast: breast_ductal β drops to 0.621 (max ΔA). Lung: lung_epithelial β drops to 0.628. Glioma: neuron β drops to 0.521.

**Production status:** G-DECONV-001 module in the IAM-Validation GitHub roadmap is OPEN — DEFERRED. VAL-041 proved the workflow using Moss 2018 markers + scipy.optimize.nnls at the published-β level. Productionizing requires (1) the 30 MB Moss 2018 reference matrix locked into the repo, (2) the Salas 2018 QC harness implemented, (3) version-tagged assay (L1 Illumina EPIC with Moss markers; L2 custom capture panel; L3 full 5-substrate MESA+DELFI). Patients get a clear assay version in their report.

### Stage 3 — EpiDISH RPC immune sub-composition

Runs when Stage 1 flags and Stage 2 returns no solid-organ localization. Teschendorff 2017 EpiDISH RPC mode decomposes the IDAT into CD4+ T, CD8+ T, NK, B, monocyte, neutrophil fractions against the Salas 2018 reference. Output supports AD-type patterns (brain tissue not in sample) and distinguishes chronic inflammation (neutrophil shift), hematologic immune drift (lymphocyte composition shift), and autoimmune patterns.

---

## Specimen selection and the cfDNA detection floor — WHY PLASMA WORKS FOR MOST CLASSES AND NOT OTHERS

**This section is mandatory reading for any future AI or analyst working on Cookbook deployment.** The Stage 1 + Stage 2 pipeline described above assumes a single blood draw as the input specimen. That is true for most classes — but not all. The specimen selection is physics-driven, and getting it wrong means missing the disease. Here is exactly why.

### The fundamental question: is the signal in the blood?

Stage 1 reads the immune response in the blood. Stage 2 mathematically deconvolves the same blood sample into per-tissue β estimates — Moss 2018 NNLS extracts the prostate-specific, breast-specific, colon-specific, lung-specific, (and every other tissue's) β fraction from bulk plasma cfDNA. This is the validated workflow (VAL-041, 10/10 top-1 correct localization across 10 cancer types).

But for Stage 2 to work, **enough cfDNA from the target tissue has to actually be present in the plasma.** Every cell that dies releases cfDNA into the bloodstream, so every tissue contributes something — but not in equal amounts. The table below, from the GAPE blueprint (`GAPE_WEB_v13.py` constant `CFDNA_PCT`, derived from Moss 2018 atlas plus Snyder 2016 quantification), shows the per-class contribution to total plasma cfDNA in a healthy adult:

| Class | Tissues in this class (Moss 2018 localizable) | % of total plasma cfDNA | Stage 2 from plasma? |
|---|---|---|---|
| immune | neutrophil, lymphocyte, monocyte | **70%** | Reliable (this is Stage 1) |
| cycling | colon_epithelial, lung_epithelial, gastric_epithelial, bladder_epithelial, cervical_epithelial, kidney_epithelial | **12%** | Reliable with Moss deconvolution |
| secretory | breast_ductal, prostate_epithelial, hepatocyte, pancreatic_exocrine | **8%** | Reliable with Moss deconvolution |
| stromal | vascular_endothelial, fibroblast | **4%** | **Detection floor — exploratory only** |
| stem_adult | hsc (hematopoietic stem cells) | **3%** | Below floor — exploratory only |
| progenitor | CMP, GMP, NPC, etc. | **2%** | Below floor — exploratory only |
| terminal | neuron, oligodendrocyte, cardiomyocyte | **0.5%** | **Too dilute — plasma fails, CSF required** |
| stem_pluri | ESC, iPSC | **0.5%** | Too dilute — plasma fails |

### The 4% detection floor

The framework has a physics-derived threshold. From `GAPE_WEB_v13.py`: *"Everything below 4% cfDNA should be treated as exploratory only."* Below 4%, the tissue-of-origin signal is too dilute to extract reliably through Moss NNLS — the 96%-plus cfDNA from other tissues adds enough stochastic fluctuation that a real architectural shift in the source tissue gets buried under noise. Above 4%, deconvolution works: Moss NNLS pulls out the tissue-specific β with enough signal-to-noise that the per-class A-score is meaningful.

**This directly determines specimen selection per card:**

- **Blood (plasma or buffy coat) — primary specimen** for immune (70%), cycling (12%), and secretory (8%). That covers most adult cancers: breast, prostate, lung, colorectal, HCC, pancreatic, bladder, cervical, gastric, kidney. It also covers AD's peripheral immune signature. These cards all take a single blood draw and run it through Stage 1 + Stage 2. The tumor itself is never touched in the standard workflow — deconvolution recovers its architectural state mathematically from its shed cfDNA.

- **CSF (cerebrospinal fluid) — gold-standard specimen** for terminal class cancers (glioma, GBM, LGG) and for terminal-class neurodegenerative imaging beyond the peripheral immune signature. CSF contains brain-derived cfDNA directly, at workable concentrations for methylation, and is the most sensitive specimen pathway for early-window detection. **However, VAL-090 (2026-04-25) demonstrated that standard EPIC peripheral blood is also viable for glioma at the time of clinical diagnosis** when the Stage 2 reference atlas includes a sorted-cell `Cortical_neurons` entry (Loyfer/Moss array atlas, `nloyfer/meth_atlas/reference_atlas.csv`). Glioma plasma reads 1.09% mean cortical-neuron cfDNA fraction vs 0.28% in healthy reference (Cohen's d = +1.96 [+1.62, +2.31], n=76 glioma vs n=177 healthy). The earlier framing that "plasma fails because brain cfDNA is below the 4% detection floor" was incorrect — the floor is reachable when the right reference atlas is used. Pre-diagnostic, sub-clinical detection windows still require CSF or specialized chemistry (cfMeDIP-seq enrichment, Nassiri 2020 / Sabedot 2021); for at-diagnosis glioma confirmation, standard array peripheral blood now works. Glioma-epic v0.2 supports both blood and CSF specimen pathways (and others — see card §Pathways).

- **Urine — alternative specimen** for bladder (urothelial cells shed directly into urine) and prostate (post-DRE urine contains prostate-shed secretory cells). Urine-based methylation concentrates the tissue-specific signal to a level that outperforms plasma deconvolution for these two tissues specifically.

- **Stool — alternative specimen** for colon (CRC-epic). Colon epithelial cells shed directly into stool. Cologuard is already FDA-approved for stool-based CRC methylation screening. A GAPE cycling-class stool panel would use the same specimen pathway with physics-derived thresholds.

- **Cervical mucosa (Pap smear) — preferred specimen** for cervical-epic. Cervical epithelial cells shed directly onto a swab. WID-CIN (Widschwendter 2021) validated this at n=2,254 with cleaner signal than blood-based cervical deconvolution. The FAM19A4/miR124-2 panel is already in European clinical guidelines for HPV triage.

- **Sputum — alternative specimen** for lung (bronchial cycling cells in induced sputum).

- **NAF (nipple aspirate fluid) — alternative specimen** for breast (direct ductal epithelial shedding).

- **Tissue biopsy — optional higher-fidelity alternative** for any card where biopsy tissue is already available (pathology workflow, not screening). 450K/EPIC run directly on tumor tissue gives clean per-class β without needing Moss deconvolution. Used when biopsy is indicated for other clinical reasons and methylation analysis is added.

### Why the blood-first screen is still the right default

The blood screen works for the cancers that kill the most people — breast, lung, prostate, colorectal, pancreatic, gastric, hepatocellular, bladder. Together these account for the majority of global cancer mortality. All of them are detectable via a single blood draw through Stage 1 (immune response) + Stage 2 (Moss deconvolution of cycling/secretory tissues above 4%). One tube of blood, one 450K/EPIC IDAT, all 10 cards above fire if they should.

Terminal class cancers (gliomas) are rare. Progenitor class cancers (MDS, some leukemias) have their own presentations that bring patients to medical attention through CBC/WBC before the EDEAR screen would matter. Stem_pluri cancers (germ-cell tumors, some pediatric) are rarer still. The 4% detection floor excludes the rare cases and catches the common ones — that is not a limitation, it is the framework correctly reporting where the physics puts the signal.

### The terminal-class paradox — huge signal, wrong specimen

**A future AI should understand this apparent contradiction before acting on it.** Terminal-class cancers have the **largest** per-cell architectural departure of any class in GAPE — glioma ΔA = +0.217, LGG ΔA = +0.239 (Ceccarelli 2016 TCGA). This is because the terminal class floor H_min = 0.7728 is so low that any cancer arising in that compartment sits enormously above it. The physics is extremely loud.

But plasma cfDNA carries only 0.5% brain-derived fraction, which is below the 4% Stage 2 detection floor. So even though the per-cell signal is the loudest in the framework, the specimen delivers the quietest detectable version of it. CSF reverses this: CSF cfDNA is predominantly brain-derived, so the loud per-cell signal comes through at workable signal-to-noise. **The signal is in the specimen, not just in the tissue.** Choosing the right specimen for the class is as important as choosing the right class for the tissue.

### What cfDNA percentage means for validation runs

When running validation tests on TCGA or other published cohorts, note:

- **TCGA matched tumor/normal β values are TISSUE, not blood.** TCGA-BRCA, TCGA-PRAD, TCGA-COAD, TCGA-LUAD and similar provide the tumor's own β directly from biopsy. These validations test the Stage 2 output (what deconvolution WOULD recover from plasma if the tumor shed enough cfDNA) but they are not themselves blood-based tests. A VAL-XXX run on TCGA tumor tissue tests the tissue-arm pathway; the blood-arm pathway requires a blood-methylation cohort with case/control status (VAL-047 GSE51057/GSE51032 is the paradigm).
- **The per-patient cohort validation with VAL-047-style design (blood, pre-dx, Phase 9/12) is the gold standard.** Tissue-level TCGA validation confirms the Stage 2 tissue-signal is real but does not prove blood-based deployment will work — that requires VAL-046-style cohort confirmation followed by VAL-047-style per-patient confirmation.
- **Before committing a card to `cross_platform_validated` tier, the blood-arm Phase 9/12 must pass on the correct specimen for that class.** Secretory and cycling classes: blood. Terminal class: CSF. Cervical/bladder/colon: tissue-direct specimen (swab/urine/stool) is acceptable substitute when blood deconvolution is marginal.

### File pointers for this section

- `GAPE_WEB_v13.py` — `CFDNA_PCT` constant with per-class plasma fractions
- `IAMPerformance_GAPEIssue002.pdf` — terminal-class specimen discussion (page 28, CSF-not-plasma rationale)
- `IAM_Hubble2GAPE_Alpha_Omega_v3.tex` — §E.3 Reference-Based Deconvolution Methods (Moss 2018 NNLS, EpiDISH RPC, Salas 2018 QC)
- `VAL-041` — Moss deconvolution validated 10/10 across 10 cancer types (mean max ΔA = +0.174)
- `VAL-046` — multi-class signature pre-dx cohort (breast, lung, CRC, pancreatic, prostate — all above 4% floor)
- `VAL-047` — blood-arm per-patient validation paradigm (Phase 9 + Phase 12 on GSE51057/GSE51032)
- `EXPANSION_ROADMAP_8_MISSING_CARDS.md` — per-card specimen decisions (glioma CSF-required, bladder urine-preferred, cervical mucosa-preferred)

---

## Serial sampling and trajectory-based diagnostics — THE SUBSCRIPTION MODEL

**Single-timepoint EDEAR is a flag. Serial-timepoint EDEAR is a diagnostic trajectory.** This section documents why trajectory is a first-class diagnostic modality inside the Cookbook, not an optional follow-up feature.

### The fundamental idea

Every EDEAR card returns a single-draw A-score against an age-matched baseline. That is useful but bounded: the tier system catches patients already in DETECTABLE, URGENT, or FLOOR BREACH tiers, but it has limited resolution in MARGINAL or NORMAL patients who may be drifting upward in a disease-specific pattern that has not yet crossed a tier threshold. The signal is real; the threshold has not been reached.

Serial sampling changes this completely. With two readings taken 6 months apart, you have:
- **Rate of change (ΔA/month)** — is the patient drifting upward, stable, or returning to baseline?
- **Acceleration (second derivative)** — is the rate of change itself increasing (characteristic of progressive disease) or stable (characteristic of chronic inflammation)?
- **Signature evolution** — which CpGs are shifting first? Does the pattern match a known disease trajectory?

With four readings over two years, the trajectory becomes diagnostic. Cardiovascular chronic inflammation shows a different rate-of-change pattern than early AML, which shows a different pattern than incipient breast cancer, which shows a different pattern than pre-symptomatic Alzheimer's.

### Why this is built into the framework

The GAPE framework produces explicit trajectory-based predictions. These are not aspirational — they are specific, dated, falsifiable research predictions already filed:

- **G-2026-P001 (cycling class):** In a prospective screening cohort of average-risk adults age 45-75 with archived serial blood samples and colonoscopy outcomes, the cycling-class combined A-score trajectory will identify CRC before colonoscopy.
- **G-2026-P002 (cycling class):** In longitudinal cohorts of Barrett's esophagus patients with known progression outcomes, the cycling-class combined A-score trajectory will distinguish progressors from non-progressors.
- **G-2026-P003 (secretory class):** Among PSA-screened prostate cancer patients with matched methylation + WPS + DELFI profiles at diagnosis and at 24 months, the combined A-score trajectory will distinguish indolent from aggressive disease.
- **G-2026-P004 (secretory class):** In asbestos-exposed occupational cohorts with archived serial blood samples, the secretory-class combined A-score will show elevation above 1.05 at least 2 years before mesothelioma diagnosis.
- **G-2026-P006 (terminal class):** In longitudinal cohorts with archived CSF samples and subsequent AD diagnosis, the terminal-class A-score will show elevation above 1.0 years before clinical onset.
- **G-2026-P010 (immune class):** In prospective cohorts of patients with known CHIP and archived serial blood samples, the immune-class A-score will show trajectory progression toward AML in subsequent years.
- **G-2026-P011 (immune class):** In patients receiving immune checkpoint inhibitor therapy, the immune-class A-score trajectory will distinguish responders from non-responders within 2-3 treatment cycles.
- **G-2026-P012 (stromal class):** In prospective IPF cohorts with archived serial blood samples and known progression outcomes, the stromal-class combined A-score trajectory slope will predict progression.
- **G-2026-P014 (progenitor class):** In MDS patients receiving hypomethylating agent therapy, the progenitor-class combined A-score trajectory at 3 months post-initiation will identify responders vs non-responders.
- **G-2026-P015 (terminal class):** In longitudinal cohorts of patients with subsequently-diagnosed ALS, the terminal-class A-score from CSF cfDNA will show trajectory progression before clinical onset.

Ten of the framework's dated research predictions are explicitly trajectory-based. They cannot be tested with single-timepoint data. Serial sampling is not a nice-to-have — it is the only way to validate half of the framework's clinical claims.

### The subscription deployment model

For EDEAR commercial deployment, trajectory-based diagnostics means a subscription service:

- **Initial baseline** — first blood draw establishes the patient's personal reference
- **Periodic re-sampling** — 6 months, 12 months, 24 months, then annual
- **Trajectory computation** — each new draw produces not just a tier call but a rate-of-change and acceleration relative to that patient's personal history
- **Card-specific trajectory interpretation** — each card interprets the trajectory against known disease progression patterns
- **Flag on trajectory, not just on magnitude** — a patient drifting from A = 0.98 to A = 1.04 to A = 1.08 over 12 months has crossed into DETECTABLE in the third reading but was already diagnostic by trajectory in the second

A patient in MARGINAL tier with accelerating trajectory may warrant earlier workup than a patient in DETECTABLE tier with stable trajectory. The framework provides the mathematical basis to say so.

### Per-card trajectory applications

All 15 Cookbook cards benefit from trajectory analysis, but some cards depend on it more critically than others. The table below indicates the trajectory relevance for each card at v2.1:

| Card | Trajectory criticality | Specific trajectory endpoint |
|---|---|---|
| breast-epic | HIGH | 10-year pre-dx drift detectable in serial blood (VAL-047 d = +1.36 to +1.78 at >10yr) |
| crc-epic | HIGH | Cycling-class pre-malignant drift, polyp → early cancer trajectory |
| ad-immune | HIGH | Pudas 2023 showed no single-timepoint predictive power; trajectory is where the signal lives |
| lung-epic | MODERATE | VAL-046 2-5 year pre-dx signal requires serial for trajectory confirmation |
| prostate-epic | CRITICAL | G-2026-P003 — indolent vs aggressive distinguishable ONLY by trajectory |
| hcc-epic | HIGH | Cirrhosis → early HCC separability 8.03× on combined-score trajectory |
| pancreatic-epic | HIGH | Rotterdam Study 5-year pre-dx signal requires serial sampling |
| gastric-epic | MODERATE | Similar to CRC — field cancerization detectable in trajectory |
| bladder-epic | MODERATE | Urine-based serial sampling for urothelial drift |
| cervical-epic | HIGH | CIN1 → CIN2 → CIN3 → SCC progression monotonic over years (VAL-042 WID-CIN) |
| kidney-epic | MODERATE | KIRC/KIRP pre-malignant drift not well characterized; trajectory may reveal |
| glioma-epic | HIGH (revised v0.2) | At-diagnosis blood detection viable (VAL-090 d=+1.96); pre-diagnostic CSF or cfMeDIP-seq still required for sub-clinical window; treatment-response trajectory monitoring also valuable |
| cardio-epic | CRITICAL | Chronic disease of decades; single-timepoint less informative than trajectory |
| heme-epic | CRITICAL | G-2026-P010 CHIP→AML and G-2026-P011 ICI response are both trajectory endpoints |
| immune-atlas | REFERENCE ONLY | Aggregates trajectory info from component cards |

Cards marked CRITICAL cannot achieve their full diagnostic value without serial sampling. Cards marked HIGH derive substantial additional signal from it. No card loses from serial sampling.

### Trajectory math — the core formulas

For a patient with readings A_1, A_2, ..., A_n taken at times t_1, t_2, ..., t_n:

**Rate of change (first derivative):**
dA/dt ≈ (A_n − A_{n-1}) / (t_n − t_{n-1}) — in A-score units per month or per year

**Acceleration (second derivative):**
d²A/dt² ≈ [(A_n − A_{n-1})/(t_n − t_{n-1})] − [(A_{n-1} − A_{n-2})/(t_{n-1} − t_{n-2})] — divided by the average time interval

**Signature drift similarity:**
Cosine similarity between (per-CpG Δβ at reading n) and (per-CpG Δβ at reading n-1) — is the same direction of drift persisting?

**Trajectory Z-score:**
(observed dA/dt − expected_healthy_dA/dt for patient's age decade) / SD(expected) — is the rate of change significantly above healthy aging drift?

All four quantities are computable from each new blood draw. Each card's report returns these quantities alongside the current A-score and tier call.

### What the subscription model requires

Technical:
- Serial β value storage per patient (patient-level database with IDAT archive)
- Per-patient baseline computation on first sample, updated as more samples arrive
- Age-adjusted expected-healthy drift rate per class (currently inferred from the 80-cell reference's age trajectories)
- Trajectory visualization in the customer report (line graph showing A-score trajectory against age-matched normal range)

Clinical:
- Defined re-sampling intervals per card (6, 12, 24 months, then annual)
- Defined acceleration tier thresholds (TBD — to be calibrated against clinical outcome data as subscription cohort grows)
- Trajectory-matches-disease-signature alerts when a patient's trajectory pattern matches a known disease progression signature

Commercial:
- Pricing model that makes annual re-sampling accessible
- Patient retention to enable trajectory accumulation over years
- Longitudinal data integrity across multiple sample dates

### Related lessons

- **CCL-022 (2026-04-24):** Single-timepoint EDEAR is a flag; serial-trajectory EDEAR is a diagnostic. See LESSONS_LEARNED.md.
- **Pudas / Hackenhaar 2023:** epigenetic age acceleration does NOT predict AD at single timepoints up to 16 years pre-onset, but longitudinal trajectories do. Same pattern expected for pre-clinical drift across other cards.

### Card author instructions

When building a new card, document:
1. **Trajectory criticality** (CRITICAL / HIGH / MODERATE / POST-DX ONLY / REFERENCE ONLY)
2. **Trajectory endpoint** (what the rate-of-change / acceleration / signature-drift reveals about this disease)
3. **Recommended sampling interval** per tier (more frequent for URGENT/FLOOR BREACH; baseline annual for NORMAL)
4. **Trajectory-specific clinical actions** (rising trajectory can trigger workup earlier than absolute tier)

---

## Coverage requirement — every Moss tissue needs a card

This is a blocking design constraint, not a future goal. When a real patient flags Stage 1 and Stage 2 localizes to a tissue, the report MUST have a disease-specific card for that tissue. A report that says "elevated immune class, localized to hepatocyte, no card defined" is unacceptable patient-facing output.

The 18-tissue Stage 2 output maps to the required card set as follows:

| Tissue (Stage 2) | Class | Required card | Status |
|---|---|---|---|
| breast_ductal | secretory | breast-epic | ✓ VALIDATED (VAL-047 cross-platform) |
| colon_epithelial | cycling | crc-epic | ✓ VALIDATED (VAL-047 Phase 12 inverted Stage 1 + VAL-061/062 tissue arm TCGA-COAD paired d = +0.724 cycling, +1.066 immune TIL) |
| (no tissue — Stage 2 null expected) | — | ad-immune | ✓ VALIDATED (VAL-051 directional, AIBL + AddNeuroMed) |
| lung_epithelial | cycling | lung-epic | ✓ VALIDATED (VAL-056 multi-modal: Kadota + Moss + TCGA-LUAD/LUSC + VAL-063 tissue arm TCGA-LUAD paired d = +1.020) |
| prostate_epithelial | secretory | prostate-epic | ✓ VALIDATED (VAL-058 GSE269244 paired d = +0.497) |
| hepatocyte | secretory | hcc-epic | ✓ VALIDATED (VAL-059 ccfDNA d = +0.634 + VAL-064 tissue arm TCGA-LIHC paired d = +0.498 pooled, +0.664 non-viral) |
| pancreatic_exocrine | secretory | pancreatic-epic | TO BUILD |
| gastric_epithelial | cycling | gastric-epic | TO BUILD |
| bladder_epithelial | cycling | bladder-epic | TO BUILD |
| cervical_epithelial | cycling | cervical-epic | TO BUILD |
| kidney_epithelial | cycling | kidney-epic | TO BUILD (added v2.1) |
| neuron + oligodendrocyte | terminal | glioma-epic | v0.2 single_cohort_validated (VAL-088 + VAL-089 + VAL-090); both blood and tissue arms validated, multi-pathway expansion ongoing |
| vascular_endothelial, fibroblast | stromal | (no card yet — route to safe-handling) | Research question |
| neutrophil, lymphocyte, monocyte | immune | (Stage 3 handles) | — |
| hsc | stem_adult | (Stage 3 handles; off-atlas hematologic disease flag) | Research question |

The remaining cards in the TO BUILD set are: pancreatic, gastric, bladder, cervical, kidney, and glioma. Each card is built at its best-achievable tier against available data. See EXPANSION_ROADMAP_8_MISSING_CARDS.md for per-card build plans. (Prostate-epic and HCC-epic have been validated since the original v2.1 expansion plan was written — see card descriptions below.)

**The v2.1 addition of kidney-epic.** v2.0 had an 11-card expansion table that omitted kidney despite kidney_epithelial being in the Stage 2 output space and KIRC+KIRP being validated cycling-class TCGA types in the Issue 002 build. A patient whose Stage 2 localizes to kidney_epithelial deserves a kidney-specific report. Kidney-epic is therefore added as the 12th required card. Its build plan is in EXPANSION_ROADMAP (v2.1 addition).

---

## The ten validated cards

1. `breast-epic/` — Breast cancer pre-diagnostic detection. Stage 1 Xu-538 pooled A_immune positive direction (d = +0.45 to +0.71 on GSE51057 and GSE51032), Stage 2 localizes to breast_ductal. **VAL-060 tissue arm added in v2.2 (2026-04-24):** TCGA-BRCA HM450 matched tumor-vs-adjacent-normal (n=89/89, 86 complete pairs) paired d = +0.676 tumor vs adjacent-normal, p = 0.0001, unpaired d = +0.745 [+0.451, +1.075]. Breast-epic is now validated across three substrates (blood pre-dx, adjacent-normal tissue field effect, tumor tissue) with the same Xu-538 panel and same H_min(immune). First retroactive per-card tissue re-validation under CCL-011.
2. `crc-epic/` — Colorectal cancer pre-diagnostic detection. Stage 1 Xu-538 pooled A_immune INVERTED (d = −0.33 on GSE51032), Stage 2 localizes to colon_epithelial. **VAL-061/062 tissue arm added 2026-04-24** on TCGA-COAD HM450 matched tumor/normal (n=26 paired): VAL-062 cycling-class primary paired d = +0.724 [+0.292, +1.156] p = 2.2e-04; VAL-061 TIL compartment supplementary on Xu-538 immune in tumor tissue paired d = +1.066 [+0.585, +1.547] p < 1e-05. Three-compartment CRC picture: peripheral blood immune negative direction (d = -0.33) + tumor cycling positive (d = +0.72) + tumor TIL immune positive (d = +1.07).
3. `ad-immune/` — Alzheimer's disease cross-sectional detection. Stage 1 Xu-538 pooled-entropy NULL expected (d = +0.08, VAL-050); Stage 1 AD 7-CpG Rule A directional panel positive (d = +0.624 on AIBL holdout, VAL-051; d = +0.33 raw / +0.12 age-regressed on AddNeuroMed cross-platform, VAL-052). Stage 2 NULL localization expected (brain not in buffy coat). Stage 3 EpiDISH sub-composition is descriptive. **VAL-057 external test on GSE53740 (n=15 AD) produced pooled null (d=+0.013); post-hoc sex-stratified recovered male AD d=+0.415 consistent with AIBL male d=+0.512; female AD non-replicated. PSP/CBD preserved 5/7 frozen directions suggesting tauopathy-associated drift co-detection. GSE53740 HC +2.306 SD cohort batch offset vs 80-cell baseline.** **VAL-091 added 2026-04-26 (card v2.1 → v2.2):** layered-atlas Stage 2 NNLS deconvolution (Moss 2018 + Loyfer/Moss array atlas) applied to AIBL n=161 AD vs 471 HC (d = −0.026 [−0.21, +0.17]), AddNeuroMed n=93 AD vs 96 HC (d = −0.083 [−0.36, +0.19]), and GIFT n=15 AD vs 193 HC (d = +0.96 outlier-driven, AD median 0.9% vs HC 0.0%). Outcome `O4_AD_NEURO_NULL` per pre-reg. **AD does not elevate cortical-neuron cfDNA at array-NNLS resolution — confirms the v2.0/v2.1 prediction that Stage 2 for AD is expected NULL.** GIFT specificity arm: FTD vs HC d=+0.19 (null), PSP/CBD vs HC d=−0.51 (PSP/CBD reads BELOW HC). Glioma-vs-AD Stage 2 differential added: Stage 1 immune positive AND Stage 2 cortical-neuron > 0.5% triggers `DIFFERENTIAL_DIAGNOSIS_REQUIRED` flag (consistent with glioma per VAL-090 anchor 1.09% d=+1.96, not AD per VAL-091 anchor 0.25%). AddNeuroMed cortical-neuron HC mean read 7.4% vs AIBL/GIFT/GSE51057 HC at ~0.3% — diagnosed as a 450K cross-platform NNLS routing artifact (8% Loyfer-CpG coverage gap, NNLS routes mass to Cortical_neurons by default); within-cohort comparisons remain valid, cross-cohort absolute fractions require platform-stratified thresholds. Tier remains cross_platform_validated. Stage 1 directional 7-CpG panel still frozen since VAL-051 SEAL 2026-04-23 07:23:53 UTC. Card-internal lessons ad-LL-006 (VAL-091 confirmation + AddNeuroMed routing artifact) and ad-LL-007 (BELOW_NORMAL added to universal tier vocabulary per Heath's flag during VAL-091 review) extend the cross-card lessons catalog.
4. `lung-epic/` — Lung cancer (NSCLC) multi-modal-validated flag. Stage 1 Xu-538 pooled A_immune expected positive per VAL-046 cohort-level support; per-patient blood pre-dx validation pending. Stage 2 localizes to lung_epithelial with very high confidence (VAL-041 / VAL-056 Part 2: top-1/top-2 ratio 60.87×). Tissue-level field effect confirmed by VAL-039 / Kadota 2014 (monotonic tumor → near → far → healthy, extends past 5 cm). TCGA-LUAD/LUSC crystallized tumor ΔA = +0.165 / +0.161 (FLOOR_BREACH). Smoking status is a mandatory covariate with per-stratum clinical action paths; current smokers get a mandatory smoking-adjustment sentence in every report. **VAL-063 tissue arm added 2026-04-24** on TCGA-LUAD HM450 matched tumor/normal (n=29, all passed QC): paired d = +1.020 [+0.571, +1.469] p = 3.9e-08 — largest paired tissue effect to date in any Cookbook card. Smoking-stratified per CCL-009: ever-smoker (n=22) d = +1.283 [+0.719, +1.847] p = 1.8e-09; lifelong non-smoker (n=2) d = +0.567 [-0.926, +2.061] underpowered, direction-only. TCGA-LUAD is 76% ever-smoker; VAL-063b on East Asian never-smoker-enriched LUAD cohort (Shanghai, KNUH, Taiwan Biobank) is the candidate next step for proper never-smoker arm resolution.
5. `prostate-epic/` — Prostate cancer stage_2_only_validated flag. **VAL-058 is the first card with its own tumor-vs-adjacent-normal tissue validation run** (GSE269244 n=238 African-American men, Berglund/Yamoah/Kresovich 2024, EPIC 850K): unpaired d = +0.400 [+0.146, +0.659] p = 0.003; paired d = +0.497 on 118 matched pairs p = 0.0001; 481/538 Xu-538 EPIC coverage. Fills the clinical gap for patients whose Stage 1 flags and Stage 2 Moss NNLS returns prostate_epithelial. Does NOT claim pre-diagnostic blood screening for prostate — early-stage localized prostate cancer sheds minimally into plasma ccfDNA; urine specimen pathway not yet integrated (v0.2+ upgrade). African-American cohort only; European/Asian ancestry validation pending. No Stage 1 blood immune-class per-patient validation exists for prostate; Health ABC and Rotterdam Study remain dbGaP-gated. **VAL-065 urine arm attempted 2026-04-25 on GSE119260 (Brikun 2018), the only public EPIC 850K urine methylation prostate cohort on GEO — n=4 advanced-disease (all bone metastatic, Gleason 4+4 to 5+5, PSA 10.9 to 1400 ng/mL).** Tumor vs benign paired d = −0.016 at n=4 (the expected positive tumor signal from VAL-058's n=238 is not recoverable at n=4); urine vs benign paired d = −2.39 in unexpected direction (urine A-score lower than benign tissue in all 4 patients); plasma vs benign paired d ≈ 0; classified O5_UNEXPECTED per pre-reg ("urine vs benign d > 0.3 but in NEGATIVE direction... convene with Heath before deciding card update direction"). Treated as exploratory open question — no card direction taken on urine substrate. Card validation tier UNCHANGED (remains stage_2_only_validated, anchored by VAL-058). Priority-1 unmet data need: larger urine methylation prostate cohort with healthy controls and mixed disease stages (CCL-026).
6. `hcc-epic/` — Hepatocellular carcinoma `multi_modal_validated` flag (promoted from `cohort_screening_validated` in v0.2 after tissue arm addition), **substrate-restricted to ccfDNA plasma for the blood arm**. VAL-059 cross-cohort run tested both substrates: GSE298812 Nigerian HIV+ HCC ccfDNA (n=245, HCC-Pos vs HCC-Neg) d = +0.634 [+0.175, +1.121] p = 0.002 with monotonic dose-response (healthy 0 → fibrosis +0.44 → cirrhosis +0.45 → HCC +0.63); GSE281691 Metabolic HCC multicenter whole-blood leukocyte (n=481) d = −0.156 NULL. **Xu-538 + ccfDNA captures HCC; Xu-538 + whole-blood leukocyte does not.** Primary HIV-HCC interaction confound: we cannot distinguish pure HCC signal from HIV-HCC interaction with GSE298812 alone; non-HIV HCC ccfDNA replication is priority 1 next step. Cannot discriminate HCC from advanced cirrhosis at moderate signal — AASLD surveillance required regardless of EDEAR output in cirrhotic patients. **VAL-064 tissue arm added 2026-04-24** on TCGA-LIHC HM450 matched tumor/normal (50 candidates, 46 passed QC) — secretory-class scoring (H_min = 0.843264, hepatocyte ref β = 0.742): paired d = +0.498 [+0.191, +0.804] p = 7.4e-04 PASS pooled. **Risk-factor stratified: non-viral HCC (alcohol/NAFLD/none, n=34) d = +0.664 [+0.293, +1.035] p = 1.1e-04 — comparable to VAL-060 breast secretory; viral hepatitis (HBV+HCV combined, n=12) d = +0.023 NULL.** Mechanism (Villanueva 2015 anchor): chronic viral infection drives methylation drift in adjacent-normal liver, raising the adjacent-normal A-score baseline above true-healthy and shrinking paired tumor-vs-normal contrast even when tumor architecture is genuinely disrupted. Does NOT mean EDEAR can't detect viral HCC — VAL-059 ccfDNA arm DID detect viral HCC in GSE298812 HIV+ HBV cohort. The blunting is specific to paired tissue contrast, not overall detectability. Surprisingly little fibrosis-vs-no-fibrosis difference (d = +0.58 vs +0.58) — viral-vs-non-viral distinction does more analytical work than fibrosis grade.
7. `pancreatic-epic/` — Pancreatic ductal adenocarcinoma (PDAC) `cohort_screening_validated` flag with `tissue_arm_exploratory_with_directional_recovery_partial` modifier (v0.1, 2026-04-25). **Stage 1 anchored by VAL-046 Rotterdam Study pre-diagnostic blood (n=182, 2-5 yr pre-dx detection at cohort level).** Pancreatic_exocrine is in the secretory class (H_min = 0.843264, ref β = 0.745) shared with breast ductal, prostate epithelium, and hepatocyte. **PDAC is the second confirmed bidirectional-cancellation disease at the Xu-538 panel level (after AD), per CCL-028.** Per-CpG positive-direction percentages cluster at 50% across three independent tissue cohorts (VAL-066 TCGA-PAAD 46.9% n=5 paired d=+1.18; VAL-067 GSE49149 50.4% n=196 unpaired d=+0.25; VAL-068 GSE74071 52.9% n=7 paired d=−0.31). Pooled-entropy CI straddles zero in all three. **VAL-069 directional fallback panel (324 CpGs, GSE49149-trained, z-score normalized so H_min-independent) recovers signal cleanly on TCGA-PAAD holdout (n=7 paired d=+1.51 [+0.43, +2.59] p=6.4e-05 PASS) and partially on GSE74071 holdout (n=7 paired d=+0.22, partial-fail driven by single-pair PH64 outlier ΔA_dir=−1.17, possibly mucinous PDAC sub-type).** A_dir is the recommended primary Stage 1 metric for PDAC; pooled-entropy reported alongside as backup. Card supports 7 IDAT specimen pathways (plasma cfDNA primary; tissue biopsy alternative-high-fidelity; pancreatic juice ERCP exploratory n=4; urine, saliva, FNA exploratory-unvalidated; CSF not applicable to PDAC) with per-pathway Stage 1/2/3 guidance and per-pathway confound documentation. **Mandatory covariates expanded to 20 fields including new-onset T2D status (Pannala 2008 paraneoplastic-PDAC trigger), recent ERCP/stent <30d, recent pancreatitis <3mo, BMI, smoking, alcohol, family history of PDAC (~10% familial: BRCA2/PALB2/ATM/CDKN2A/STK11), pregnancy (decline scoring), chimerism (decline scoring), active chemo/radiation (decline scoring).** Special clinical action rule: new-onset T2D age ≥50 with any DETECTABLE-or-above directional A-score → paraneoplastic-PDAC workup REGARDLESS of Stage 2 localization. Tier thresholds calibrated from TCGA-PAAD A_dir distribution (NORMAL <+0.5z, MARGINAL +0.5 to +1.0, DETECTABLE +1.0 to +1.5, URGENT +1.5 to +2.0, FLOOR_BREACH ≥+2.0z) — tissue-tumor-equivalent magnitudes; blood deployment expected to require lower thresholds, will recalibrate when blood-PDAC HM450/EPIC cohort becomes available. Per-patient pre-diagnostic Stage 1 sensitivity at the 2-5 yr window NOT validated at v0.1 — that is the priority next-step (Rotterdam individual-level β data not in public domain; alternative paths are dbGaP application for Sister Study or UK Biobank pancreatic subset). 11 known limitations + 9 open questions documented in the card README. **Card-internal lesson panc-LL-007: Stage 1 ALWAYS scores Xu-538 against H_min(immune) = 0.838889 regardless of disease — earlier drafts of all four VAL studies erroneously used H_min(secretory); Cohen's d unchanged scale-invariant but absolute A-scores were 0.5% off. Panel-class governs H_min in Stage 1; tissue-class is a Stage 2 concept only. Universal pipeline rule.**

8. `cervical-epic/` — Cervical squamous cell carcinoma and high-grade CIN detection. `exploratory_with_cohort_heterogeneity` tier (v0.1, 2026-04-25). Cervical_epithelial is in the cycling class (H_min = 0.856100). Stage 1 universal Xu-538 immune scoring across six VAL studies surfaces real cohort heterogeneity that single-cohort validation would have missed. **Tissue arm: VAL-073 GSE99511 Verlaat Amsterdam population-normal (n=68) Normal vs CIN3 d = +0.73 [+0.22, +1.24] p=0.004 monotonic Normal<CIN3<SCC POSITIVE anchor; VAL-074 GSE46306 Farkas Stockholm HPV-negative healthy normal (n=43) Normal vs CIN3 d = −0.61 [−1.27, +0.05] NEGATIVE-direction; VAL-081 GSE68339 Lando Oslo cancer-only (n=270) tumors d = −0.43 [−0.82, −0.04] vs VAL-073 normals NEGATIVE-direction confirmation at large n.** Two of three tissue cohorts read tumors at or below VAL-073's normal baseline; VAL-073 is the outlier rather than VAL-074/081 being the artifact. Most likely explanation: HPV-stratification of normals matters — VAL-074 selected HPV-negative healthy as normal which sits at depressed immune-class baseline relative to mixed/unspecified-HPV population normal. **LBC primary pathway: VAL-076 GSE143752 El-Zein 2020 EPIC 850K LBC (n=186, 54 Healthy + 50 CIN1 + 40 CIN2 + 42 CIN3) Healthy vs all-lesion d = −0.114 [−0.43, +0.20] PANEL TRANSFERABILITY FLAG — Xu-538 was buffy-coat trained, LBC is exfoliated cervical epithelium + mucosal-resident lymphocytes (different cell mixture). VAL-077 GSE287994 Bowden 2025 EPIC 850K LBC (n=247) deferred to v0.2+ DATA INTEGRITY FLAG — supplementary file `GSE287994_ewas_betas_2.txt.gz` is batch+chip+age+HPV-corrected residual M-values per Bowden 2025 Methods, NOT raw β; β distribution 50% in [0.4, 0.6] vs 12% extremes confirms file format issue per CHK-3.1; raw IDAT processing through minfi/sesame required for v0.2+.** VAL-075 GSE38266 EXCLUDED (landscape error: HNSCC not cervical, caught at runtime sample-title verification). VAL-078 CINCS Bukowski 2023 (5-yr LBC pre-dx, n=148-289) and VAL-079 Sundström CIN2 2026 deferred to v0.2+ contact list (data not GEO-deposited; available "from corresponding author upon request"). **VAL-073 stays as the tissue-arm anchor; the card cannot claim CIN3 detection at single_cohort_validated tier with VAL-074 + VAL-081 disagreeing at total n=313.** Path to v0.2+: build cervical-LBC-specific Stage 1 panel trained on LBC β (addresses VAL-076 transferability), or substitute published clinical-grade cervical methylation panels (FAM19A4/miR124-2 [QIAsure], ZNF671/SOX17/DLX1 [GynTect], PAX1/NREP-AS1 [Bowden 2025 AUC 0.92], EPB41L3) with dedicated H_min calibration as card-specific Stage 1 deviation; reprocess GSE287994 from raw IDATs; HPV-stratified re-run of all tissue cohorts; Test 2 lymphoid/myeloid sub-panel split when OQ-2026-01 immune-atlas staging operationalizes. **Card-internal lessons cerv-LL-008 through cerv-LL-016 catalog the cervical-specific landscape errors (HNSCC at VAL-075), supplementary file format pitfalls (residual M-values look like raw β until distribution check at VAL-077), healthy baseline cross-cohort heterogeneity (VAL-073 vs VAL-074 vs VAL-081), LBC panel transferability (Xu-538 buffy-coat vs LBC mucosal-immune cell mixture at VAL-076), and the diagnostic-order rule formalized as CCL-032 (data integrity → biology → framework, never the reverse).**

All eight use the same universal pipeline. Only the Stage 1 expected direction, Stage 1 panel-of-interest (Xu-538 for all; AD adds the directional 7-CpG panel), mandatory covariates (breast: sex, menopause; CRC: age; AD: age-regression + sex stratification; lung: smoking status; prostate: none [all male]; HCC: substrate verification + cirrhosis background flag; cervical: HPV status + specimen-pathway transferability), expected Stage 2 localization, and tier-specific firing conditions differ.

9. `heme-epic/` — Hematologic malignancy detection. **Myeloid arm `single_cohort_validated` (VAL-082); lymphoid arms `framework_calibrated_pending_per_patient_validation`** (v0.1, 2026-04-25). Three-arm card structure: lymphoid B-cell arm (CLL, DLBCL, MM, B-ALL), lymphoid T-cell arm (thymoma, T-ALL), myeloid arm (AML, MDS, MPN). **The immune compartment IS the diseased tissue — Stage 2 Moss NULL on solid organs is the diagnostic feature for heme cancer (inverted interpretation vs other Cookbook cards).** Stage 3 EpiDISH RPC lineage breakdown is load-bearing: neutrophil-dominant shift → myeloid arm; B-cell-dominant shift → lymphoid B arm; T-cell-dominant shift → lymphoid T arm; uniform elevation → NOT cancer (route to immune-atlas for inflammaging/autoimmune differential); uniform suppression → SUPPRESSED tier (immunocompromised state). **Brain-cancer differential gap documented (heme-LL-010):** Moss 2018 reference does NOT include brain/CNS tissue because the blood-brain barrier limits cfDNA fraction from primary CNS tumors. "Moss NULL on solid organs" rules out the 18 peripheral solid tissues but does NOT rule out CNS disease — glioma-epic (TBD) handles the CNS pathway separately, and v1 patient reports must surface "uniform Stage 3 + Moss NULL on peripherals" as a pattern warranting neurological evaluation alongside other differentials, NOT as confirmation of heme cancer. **Heme-epic introduces the SUPPRESSED tier to the framework-wide tier vocabulary** (A_immune > 1 SD below age-decade healthy reference is a real signal — post-chemo, post-transplant, HIV, primary immunodeficiency, cachexia, late-stage marrow infiltration). Other cards inherit SUPPRESSED. The patient-facing four-bin set (SUPPRESSED / NORMAL / ELEVATED / FLOOR_BREACH with MARGINAL flanks) replaces the previously-stated three-bin set across all cards as of heme-epic v0.1. **Per-disease A-score targets per Issue 002 framework calibration (5-substrate cfDNA combined, future L2/L3 platform target):** CLL ≈ 1.07, thymoma ≈ 1.09, AML ≈ 1.10, DLBCL ≈ 1.13. Per-disease ΔA spread (CLL +0.098 to DLBCL +0.203) reflects programmed B-cell methylation perturbation (class-switching, somatic hypermutation) that lymphoid cancers exploit further; Cancer Amplifier g for immune class is 5-10× rather than infinite because healthy immune cells are not at H_min floor. **VAL-082 GSE62298 Glass 2017 AML HM450 (n=68) vs GSE51057 EPIC-Italy menarche cohort buffy-coat healthy controls (n=115 cancer-free QC): ΔA = +0.1039 above Italian healthy baseline; Cohen's d = +3.71 [+3.23, +4.20] p < 1e-50; 98.5% of AML samples score above the Italian healthy 95th percentile; 91.2% above the 99th percentile.** This is the strongest single-cohort effect size measured anywhere in the Cookbook to date. The structural reason: AML is a myeloid-lineage cancer where the cancer cells ARE neutrophil/monocyte progenitors, and the Xu-538 panel was trained on whole-blood buffy coat where ~50-75% of cells are neutrophils. AML in blood is the case where the universal Stage 1 panel reads the disease cells directly rather than reading a small contaminant fraction against an immune-cell background. **VAL-082 also resets framework number expectations (heme-LL-009 ABSOLUTE):** Issue 002's prediction of A_AML ≈ 1.10 with ΔA = +0.168 is a **5-substrate combined cfDNA prediction** (future L2/L3 platform target), NOT directly comparable to v1 single-substrate methyl-only buffy-coat scoring. At v1 with 450K/EPIC arrays, AML reads ΔA = +0.10 absolute with d = +3.71 — this is the v1-deployment effect size and is sufficient for clinical-grade detection. All cards must distinguish substrate scopes when comparing VAL results to Issue 002 predictions; v1 single-substrate is what EDEAR launches with, multi-substrate cfDNA is post-launch capability expansion. **The pre-diagnostic CLL evidence is unusually strong but biobank-gated:** EnviroGenomarkers (Georgiadis 2017 BMC Genomics, Florence + Umeå joint cohort, n=347 with 28 future-CLL cases 2.0-15.7 yr pre-dx HM450 PMID 28903739) is the long-window pre-dx CLL cohort the framework needs — same evidence tier as the EPIC-Italy breast cohort that anchors VAL-047 — but data sits at EPIC-Italy + NSHDS biobanks requiring formal data-access applications, NOT GEO-deposited. MCCS (Wong/Severi, n=82 CLL up to 18 yr pre-dx HM450) provides cross-platform replication, also biobank-gated. Same access pattern as VAL-046 Rotterdam pre-dx pancreatic and Bukowski CINCS pre-dx cervical (heme-LL-011: Italian/biobank-gating is a recurring pattern for long-window pre-dx methylation cohorts). MARLIN reference (Capper 2025 Nat Genet, n=2,540 acute leukemia 450k/EPIC including 1,461 AML / 686 B-ALL / 266 T-ALL) is the framework-equivalent reference for myeloid arm cross-cohort replication. **Heme-epic is the highest signal-to-noise card in the catalog at v1 launch** — 70% of plasma cfDNA is immune-derived and the disease lives in that 70%, structurally better SNR than any solid-organ card. v0.2 priority validation queue: VAL-083 EnviroGenomarkers (biobank application required), VAL-084 MARLIN myeloid cross-cohort replication, VAL-085 CHIP→AML serial trajectory (G-2026-P010), VAL-086 ICI response (G-2026-P011). Card-internal lessons heme-LL-001 through heme-LL-011 catalog the inverted Moss interpretation, three-arm structure rationale, SUPPRESSED tier definition, EnviroGenomarkers cohort discovery, MARLIN reference identification, substrate-scope translation rule (heme-LL-009), brain-cancer Moss-gap (heme-LL-010), and biobank-gating recurring pattern (heme-LL-011).

10. `glioma-epic/` — Adult diffuse glioma (LGG and GBM, IDH-mutant and IDH-wildtype) detection across multiple specimen pathways. **Blood arm `single_cohort_validated` (VAL-088 + VAL-090); tissue arm `single_cohort_validated_with_substrate_scope_caveat` (VAL-089); CSF/cfMeDIP-seq plasma/lymphatic arms `pre-validation_skeleton`** (v0.2, 2026-04-25). Pediatric brain tumors and primary CNS lymphoma not in v0.2 scope. **Glioma-epic v0.1 was tier-labeled `exploratory_pending_replication` based on the assumption that terminal-class cfDNA contributes ~0.5% to plasma at healthy baseline, below the Moss 4% detection floor. VAL-090 demonstrated this assumption was wrong: brain-derived cfDNA is directly detectable in standard EPIC peripheral blood at array resolution when the reference atlas includes a sorted-cell `Cortical_neurons` entry.** Moss 2018's "brain (cortex)" entry is bulk-tissue mixture and does not separate cortical-neuron signal at array CpG resolution. The Loyfer/Moss array atlas (`nloyfer/meth_atlas/reference_atlas.csv`, 26 cell types, 7,890 array-indexed CpGs, distributed alongside Loyfer 2023 *Nature* 613:355) includes a sorted-cell `Cortical_neurons` reference. Applied directly via NNLS deconvolution to the same cohorts VAL-088/089 used, with no parameter tuning, no panel selection, no post-hoc adjustment. **VAL-088 GSE180683 Salas/Wiencke 2022 EPIC peripheral blood (n=76 glioma) Stage 1 immune A-score: Cohen's d = +0.91 [+0.61, +1.22] vs Italian healthy buffy coat reference; pre-surgery treatment-naive subset (n=37) d = +0.94. VAL-090 same cohort Stage 2 cortical-neuron cfDNA fraction: glioma mean = 1.092% vs healthy mean = 0.276%, Cohen's d = +1.96 [+1.62, +2.31]; pre-surgery treatment-naive subset d = +1.97; pre-surgery LGG (n=12) mean 1.292% LARGER than pre-surgery GBM (n=19) mean 0.858% — same LGG-louder-than-GBM ordering as VAL-088 under a completely different metric. 89% of glioma plasma samples cross the 0.5% threshold; 63% cross 1%; in healthy reference, only 7% cross 1% (NNLS noise floor activity, median sample = 0%).** Outcome label O1_PASS. **VAL-089 GSE60274 Lai 2015 brain tissue 450K (n=64 GBM primary + 4 recurrent + 4 spheres + 5 NTB controls): GBM primary mean A_terminal = 0.7013 (Stage 1 metric, d=+0.24 wide CI from small NTB); GBM cultured spheres d=-1.81 NEGATIVE confirming heterogeneity-not-tumorness biology cross-check; under VAL-090 Loyfer-atlas deconvolution, NTB controls read 62.4% cortical-neuron fraction, GBM primary 39.3% (d=-2.81 vs NTB), GBM recurrent 35.2%, spheres 42.9%. Tumor displaces normal cortical-neuron architecture in the tissue, in proportion to disease progression. The pipeline reads non-tumor brain as 62% neurons and healthy peripheral blood as 0.3% neurons — the expected biological gradient.** **CCL-023 revision after VAL-090 (glioma-LL-001 revised):** the v0.1 outcome label `O5_POSITIVE_INVERTED` for VAL-088 was based on the (incorrect) interpretation that the Bracci 2022 NLR-style cell-fraction prior (lymphocytes-down, neutrophils-up) had been refuted. VAL-090's Loyfer-atlas deconvolution shows the Bracci prior was actually CORRECT in its direction (neutrophils +16% [52% → 68%], CD8+ T-cells -9%, CD4+ T-cells -3%, B-cells -2%). The Shannon-entropy A-score is a different lens on the same disease state, not an opposite signal. **Cell-fraction direction and A-score direction are ORTHOGONAL (different facets of the same biology), not INVERTED (opposites).** VAL-088 outcome label revised to `O1_PASS_ORTHOGONAL_PRIORS_BOTH_CONFIRMED`. Three independent positive signals on a single cohort: VAL-088 Stage 1 A-score d=+0.91, VAL-090 Stage 2 cortical-neuron cfDNA fraction d=+1.96, VAL-090 Stage 3 NLR cell-fraction shift consistent with Bracci 2022. **commercial.web.py decision tree documented per CHK-5.5 with seven routing arms (A whole blood / B plasma standard / C plasma EPIC restoration kit Sabedot GeLB / D plasma cfMeDIP-seq / E CSF / F tumor tissue / G cervical lymph aspirate); v0.2 update: arm A and B can now produce a positive Stage 2 result via Loyfer-array deconvolution.** **What we'd need access to (priority-ordered):** dbGaP phs001497.v2.p1 UCSF AGS Bracci 2022 (139 pre-surgery glioma + 454 controls EPIC, the primary VAL-090 replication target with on-study controls — eliminates cross-platform reference confound); dbGaP phs002998.v1.p1 UCSF Immune Profiles Study; dbGaP phs001319.v1.p1 GICC (7,566 international biospecimens); Mayo Clinic Glioma cohort; Nassiri 2020 cfMeDIP-seq cohort; LP-CSF EPIC array glioma cohort (gold-standard specimen, no public cohort); pre-diagnostic glioma blood (UK Biobank, EPIC-Italy NSHDS, Sister Study, MCCS); deep cervical lymph aspirate methylation cohort; Sabedot 2021 GeLB Mendeley deposit cgrz6zztfg (already accessible Tier 1, v0.2 external-classifier integration). **Honest weaknesses summary:** single-cohort blood-arm validation (VAL-090 cohort matches VAL-088 cohort); no pre-diagnostic data; no CSF data validated; no cfMeDIP-seq integration; no Pathway 2 or 4 cohorts; cross-platform cross-cohort baseline confound on blood arm (HM450 Italian healthy vs EPIC glioma); treatment heterogeneity in test cohort (37/76 pre-surgery treatment-naive is cleanest signal, d=+1.97); glial cell-type separation (oligodendrocyte/astrocyte/microglia not separately resolved at array resolution; v0.3 task with Caggiano 2021 references). **Card-internal lessons glioma-LL-001 (revised — orthogonal not inverted), glioma-LL-002 (heterogeneity-not-tumorness biology cross-check from VAL-089 spheres-negative), glioma-LL-003 (the card IS the multi-pathway reference document), glioma-LL-004 (substrate-scope translation per heme-LL-009), glioma-LL-005 (defer-without-reason failure mode — Heath caught Walther deferring Loyfer integration to "v0.2 future task with 3-month timeline"; actual integration took 4 hours and produced d=+1.96; rule now CHK-7.7), glioma-LL-006 (direct cortical-neuron cfDNA detection at array resolution — VAL-090 headline finding), glioma-LL-007 (layered-atlas architecture applies cookbook-wide — Moss 2018 stays primary for cells it covers; Loyfer-array supplements for sorted-cell cortical neurons, vascular endothelial cells, left atrium, etc.).**

All ten use the same universal pipeline. **Heme-epic remains the structural exception** in that Stage 2 NULL on solid organs IS the diagnostic feature for hematologic malignancy. **Glioma-epic v0.2 is no longer a structural exception** — VAL-090 demonstrates Stage 2 deconvolution on glioma plasma works at standard array resolution when supplemented with the Loyfer/Moss array atlas. The card retains its multi-specimen-pathway structure for future expansion (LP-CSF, cervical lymph, cfMeDIP-seq enrichment) and pre-diagnostic windows. **The Stage 2 cell-of-origin reference architecture is now a layered atlas (Moss 2018 primary + Loyfer/Moss array supplementary) — see GAPE Reproduction Paper §5.2 and CHK-2.6 in TESTING_CHECKLIST.md.**

11. `psp-epic/` — Progressive Supranuclear Palsy / Corticobasal Degeneration detection. **`exploratory_pending_replication` tier (v0.1, 2026-04-26).** Stub card capturing the replicable PSP-specific signal that surfaced under run-everything. Three independent VAL studies on the same Munich GIFT GSE53740 cohort produce a consistent below-normal signature: **VAL-057** (Stage 1 directional 7-CpG Rule A panel) — PSP/CBD preserved 5/7 frozen directions vs only 4/7 on AD samples in the same cohort; **VAL-091** (Stage 2 cortical-neuron *fraction* via Loyfer-atlas NNLS deconvolution) — PSP/CBD vs HC d=−0.51 (PSP/CBD reads BELOW HC); **VAL-092** (Stage 2 per-class A_terminal at top-100 cortical-neuron-discriminating CpGs vs H_min(terminal)=0.7728) — PSP vs HC d=−0.433 [−0.747, −0.098] p=0.010 BELOW_NORMAL. **The signal is PSP-specific, not generic tauopathy** — FTD vs HC d=+0.19 (VAL-091) and d=−0.004 (VAL-092) confirm FTD reads at HC baseline, ruling out a "any tauopathy" mechanism. **The signal has the opposite sign from AD** — AD reads at HC baseline on cortical-neuron pathways (VAL-091 fraction d=−0.026 to −0.083, VAL-092 per-CpG drift d=−0.030 within-cohort AddNeuroMed) while PSP reads BELOW HC at d=−0.43 to −0.51. PSP is therefore not "AD plus more" or "AD-adjacent." It's a different signature entirely, and a same-card-as-AD framing would obscure both diseases. **Mechanism (working hypothesis pending replication):** PSP's tau pathology produces architectural homogenization at cortical-neuron-discriminating positions detectable in peripheral plasma cfDNA at array resolution. The homogenization-not-elevation direction parallels VAL-047 Phase 6 Deep Audit's secretory-class variance reduction at >10yr breast pre-dx (d=−1.226) and heme-epic's SUPPRESSED tier — three independent below-normal mechanisms now documented across the cookbook. **This card is a stub at v0.1**, not a full card structure. It exists to (1) capture the replicable signal at the right tier so it doesn't get buried inside ad-immune where it doesn't structurally fit, (2) anchor the priority replication cohort list, and (3) demonstrate that run-everything-architecture surfaces below-normal tile patterns that gating would have hidden. **Honest weaknesses at v0.1:** single-cohort evidence (only GIFT GSE53740, n=43 PSP / 1 CBD); no pre-diagnostic data; no cross-platform replication; PSP-vs-AD-vs-FTD-vs-HC three-way clinical decision boundary not yet calibrated; Stage 1 Xu-538 pooled A_immune behavior on PSP not separately characterized; the cohort had Ferrari 2014 ComBat preprocessing producing +2.306 SD HC offset vs the 80-cell baseline (VAL-057) which did not block within-cohort case-vs-control inference but does block direct cross-cohort A-score comparison until a second PSP cohort lands. **Priority replication cohorts:** PROGRESS-PSP biobank (Boxer/Cure-PSP), Allen et al. Mayo PSP cohort (n~40 per arm, 450K, public), Tang 2014 PSP/MSA blood methylation (n=68 PSP, GSE GSE/IDAT availability TBD). At least one of these must replicate the BELOW_NORMAL signal at d ≤ −0.3 within-cohort before promoting to `single_cohort_validated`. **What the card delivers at v0.1:** a documented BELOW_NORMAL pattern on cortical-neuron-discriminating CpGs, replicable across two metrics, distinct from AD and FTD, with an explicit replication-cohort list. Card-internal lessons psp-LL-001 (the run-everything architecture surfaced this finding — under elevation-gated Stage 2 the PSP BELOW_NORMAL pattern would never have been computed because Stage 1 immune A-score in PSP is not above HC), psp-LL-002 (PSP is not "AD plus more" — opposite sign on Stage 2 cortical-neuron tile vs AD's null; same-card framing obscures both diseases), psp-LL-003 (single-cohort cross-platform-batch-effect-flagged data is sufficient for an `exploratory_pending_replication` tier finding when within-cohort effect is replicable across two independent metrics; the +2.306 SD HC offset blocks cross-cohort comparison but does not block the within-cohort case-vs-control finding).

All eleven use the same universal pipeline. **Heme-epic remains the structural exception** in that Stage 2 NULL on solid organs IS the diagnostic feature for hematologic malignancy. **Glioma-epic v0.2 is no longer a structural exception.** **Psp-epic v0.1 introduces the first card whose primary signal is direction-negative (BELOW_NORMAL) on the disease-of-interest tile** — it joins heme-epic's SUPPRESSED interpretation as a documented case where below-normal A-scores carry primary diagnostic information. **The Stage 2 cell-of-origin reference architecture is the layered atlas (Moss 2018 primary + Loyfer/Moss array supplementary) — see GAPE Reproduction Paper §5.2 and CHK-2.6 in TESTING_CHECKLIST.md. Queue-1 atlas integration approved 2026-04-26 for v0.3** adds Tanaka 2025 6-cell neural / Konigsberg 2023 cardiac / EpiSCORE 42-cell pan-tissue / Caggiano 2021 / MARLIN leukemia / Sabedot 2021 GeLB to the Stage 2 reference layer; spec doc `EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md`.

---

## Cards are fully self-contained (v2.1)

Every card JSON at v2.1 contains a full-inline `universal_reference` block. This means the card includes — embedded in the JSON itself — the complete universal pipeline spec: all 8 architecture-class H_min constants, the 18-tissue Moss 2018 healthy reference β table, the 80-cell immune-class age-decade baseline (with Hannum/Horvath/Roadmap/Moss/Lister/Alisch sources), EpiDISH Salas QC bounds, universal tier thresholds, sex-stratification rule, language discipline, and the cross-cohort batch-offset warning established by VAL-057.

**The design goal is pure:** a new analyst loading ONLY `<disease>-card_v2.1.json` and `GAPE_WEB_v13.py` can run the full Stage 1 + Stage 2 + Stage 3 pipeline for that disease, without reading this master README, without consulting any other file. This is deliberately anti-DRY — universal constants are duplicated in every card — because the duplication cost is near-zero at Cookbook-vault scale and the deployment robustness gain is substantial.

**When a universal constant updates** (e.g. an H_min value re-estimated after a larger G-003b MCMC run, or the 80-cell baseline extended to a 10-cell-class table), regenerate all card JSONs from the updated `universal_reference_block.py` via `update_all_cards_v2.1.py`. Bump all card versions. Record the update in LESSONS_LEARNED.md.

---

## Per-card lessons_learned + cross-card catalog

Every card at v2.1 includes a `lessons_learned` key in the JSON. Each lesson entry records: source validation (e.g. VAL-051 or VAL-057), context (cohort, test), quirk observed, interpretation, and how the card was updated to handle it. Lessons are labeled with the card prefix (ad-LL-### for ad-immune, lung-LL-### for lung-epic, etc.).

Cross-card lessons aggregate in `LESSONS_LEARNED.md` at the Cookbook root. Categories:
- `CCL-###` — cross-card lesson (pattern seen across multiple cards, e.g. Directional-Score Principle from VAL-050/051; sex stratification from VAL-051/057)
- `PL-###` — process lesson (e.g. pre-registration discipline after VAL-057 gap-filling)

This documentation is deliberately for collaborators, acquirers, and referees. Every surprise is disclosed. Every failure mode is documented. No retrospective smoothing. The honest record IS the competitive moat.

---

## Per-card authoring checklist — for future AI and for Heath

Every card in this Cookbook — existing or new — is built against this checklist. A future AI reading this README plus the GAPE_WEB_v13.py engine should be able to author a new disease card for any Moss-atlas tissue by following these eleven steps in order. No step is skipped. Each step produces a named artifact that is archived.

**1. Landscape survey.** Identify every public GEO cohort, every gated cohort (EGA, dbGaP, UK Biobank), and every published cohort lacking a public deposit that could support per-patient pre-diagnostic validation. Minimum specification: n ≥ 100 cases on Illumina 450K or EPIC 850K blood methylation with time-to-diagnosis metadata. Record each candidate's access state (public / gated with application pathway / direct PI contact required / ruled out with reason). The landscape survey is itself a VAL entry (see VAL-056 for lung as the template).

**2. Decide the card's validation tier honestly from the landscape.** If a public cohort is available: author at `cross_platform_validated` or `single_cohort_validated` after the test runs. If cohorts are gated: author at `stage_2_only_validated` and define the upgrade path. If cohorts are at-diagnosis only: author at `exploratory` with explicit limitation. Never force a tier the data does not support. A `null_documented` entry is better than an unsupported `cross_platform_validated` claim.

**3. Determine the disease-specific covariates.** Read the EWAS literature for the disease and identify what stratifies the signal. Sex, age, and smoking are the standard three; some diseases add more (histology for cancers with distinct types, menopausal status for breast, hormonal status for prostate, alcohol and hepatitis for HCC, BMI for pancreatic). Document which are mandatory report fields, which are mandatory analysis stratifications, and which are observation-only. Record the rationale for each with citation.

**4. Determine Stage 1 expected direction at per-patient level.** Positive (elevation), negative (inversion), or bidirectional (directional panel required, AD template). Cite the source paper or prior VAL entry establishing the direction. If direction is expected but not validated per-patient, state that explicitly in the card and in the Evidence Report section.

**4b. MANDATORY bidirectional-cancellation guard — applies to every card without exception.** The Directional-Score Principle (CCL from VAL-050/051, also stated in line 118-120 above) is not optional and is not specific to AD or to neurodegenerative diseases. It applies to every card at the Stage 1 design step. Every card must explicitly answer all four of the following questions in its v0.1 build, and the answers must be embedded in both the card JSON `stage_1_immune_flag` block AND the card README "Why Stage 1 uses immune class" section:

  - **(i) Pooled-entropy expected direction.** What direction does the pooled Xu-538 A-score go for this disease — positive, negative, null-expected, or unknown? Cite source.
  - **(ii) Bidirectional-cancellation risk.** Is there a literature signal suggesting this disease may drive immune CpGs bidirectionally (some up, some down) rather than uniformly? Diseases with documented immune-population shifts in opposite directions (lymphocytes down + neutrophils up, or T-cell-up + monocyte-down) are at risk. Examples flagged so far: AD (VAL-050/051), glioma (Bracci 2022, lymphocytes down + neutrophils up), cardiovascular (CCL-021 Pathway 3, monocyte + Treg shift). Cite the literature source for this disease.
  - **(iii) Directional-panel fallback specification.** If pooled-entropy may null due to bidirectional cancellation, specify what directional panel runs as the fallback. The current Cookbook directional panels are the AD 7-CpG Rule A panel (validated) and the per-disease frozen-direction subset of Xu-538 (research-grade, generated from training cohort Δβ signs at card build). State which is used and why.
  - **(iv) Lymphoid-vs-myeloid expected pattern.** What does the literature say about whether this disease drives a lymphoid-arm shift, a myeloid-arm shift, both, or neither? This question cannot yet be operationally answered at score level (the immune-atlas card builds the split-A-score later — OQ-2026-01) but the *expected pattern from literature* must be documented in the card now so the immune-atlas card has the cross-card reference table populated when it builds. State the expected lymphoid/myeloid pattern with citation, even if the operational metric is "pending immune-atlas".

If any of these four questions cannot be answered from the literature for the disease in question, the card must state that explicitly and document the gap as a v0.x+ next-validation-step. **A card that does not answer all four questions cannot pass to the v0.1 publish step.** This is the AD-cancellation lesson generalized: every card carries the same risk and every card must show its work on the guard.

**5. Determine Stage 2 expected localization.** Name the Moss 2018 tissue, its architecture class, the class H_min value (pull from GAPE_WEB_v13.py line 87-96), the healthy reference β (pull from Moss 2018 Table S1 or VAL-041), and the expected case β and ΔA. Document how Stage 2 discriminates this tissue from other tissues in the same architecture class (the lung vs. gastric vs. bladder ambiguity within cycling-class is the canonical example).

**6. Define tier thresholds.** Start from the 80-cell healthy baseline reference used by breast-epic and crc-epic (NORMAL < 1.01, MARGINAL ≥ 1.01, DETECTABLE ≥ 1.05, URGENT ≥ 1.07, FLOOR BREACH ≥ 1.10). If the disease-specific literature justifies different thresholds, deviate and cite. Never deviate silently.

**7. Build the clinical action matrix.** For every combination of Stage 1 tier × covariate state × Stage 2 localization confidence, specify the clinical action. No result pattern returns a generic "see your doctor" — every pattern has a named action: LDCT, colonoscopy, MRI, CBC with differential, pulmonology consult, oncology referral, serial-sample at 6 months, etc. Cite the clinical guideline governing each recommendation (USPSTF, NCCN, ACR, EASL). The ambiguous-localization case (top-1 ΔA not 2× top-2 ΔA) and the sub-DETECTABLE flag case each need their own action path.

**8. Author the card JSON as a self-contained machine-executable spec.** Full panel embedded (CpG list), panel SHA-256, H_min values, tier thresholds with exact numeric cutoffs, mandatory covariate rules, expected direction, Stage 2 target (tissue + class + class H_min + healthy β + expected case β + expected ΔA), Stage 3 trigger, clinical action matrix, validation anchors with inline citations, known limitations, next-validation-steps. A new AI loading ONLY this JSON + GAPE_WEB_v13.py must be able to run the full pipeline and generate a patient report without consulting any other file.

**9. Author the card README as the partner-facing clinical spec.** Eleven sections: clinical claim, workflow-in-one-patient, why-stage-1-uses-immune, validation summary with DOI links, tier thresholds, mandatory covariates, known limitations, next validation steps, file pointers, language discipline, clinical action summary. Clickable DOIs everywhere. No reference to VAL numbers without the citation inline (future reader should not need to look elsewhere).

**10. Write the validation script as a parameterized Phase 9/12-equivalent pipeline.** Function structure: sha256_file, shannon_entropy, a_score, age_decade, tier_call, cohens_d, permutation_p, bootstrap_ci_d, load_xu538_panel (with SHA verification), load_series_matrix, compute_stage1_scores, analyze_windows, analyze_by_{covariate}, main with argparse. RNG seed 20260420. Output is a SHA-locked results JSON. Script runs standalone given the three inputs (matrix, metadata CSV, panel JSON).

**11. Append the Evidence Report section and update the references.** New VAL-### section in GAPE_Evidence_Report_CURRENT.html with five elements: hypothesis or scope statement, landscape survey table with clickable cohort links, result or landscape conclusion, reproducibility anchors (script SHA, panel SHA, matrix SHA if test ran, RNG seed), clinical action matrix if the card fires. Add the new citations to the reference list with [VAL-###] tags. Count lines before and after.

**What gets pushed where (the deployment boundary).** GitHub `github.com/hmahaffeyges/IAM-Validation/validation_runs/`: the validation script, the results JSON, the panel JSON, any derived figures. **Not** the Evidence Report HTML. **Not** the card JSON. **Not** the card README. The Evidence Report stays on Heath's machine for direct delivery to chosen researchers. The card and its README live in the physical vault for NDA delivery to commercial partners. The public reproducibility layer is the GitHub-deposited script + the Evidence Report section that Heath distributes; the commercial deployment layer is the card held in the vault.

**Line counts and SHAs — mandatory on every edit.** Record line count before, line count after, delta. Record SHA-256 of every static artifact (panel, card, README, script, results JSON). Verify SHAs match expected values before writing any VAL entry that references them. SHA mismatch is never silent — it blocks the commit.

**Language discipline — automatic on every artifact.** Use "consistent with," "tested against," "data are consistent with," "architectural signal detected," "elevated above age-matched baseline." Never use "confirms," "validates," "proves," "first derivation," "154 years no one has." The discipline applies to card JSON, card README, validation script comments, Evidence Report section, and GitHub commit messages.

**No-fabrication rule — absolute.** Read source files before writing anything referencing their content. Never invent CpG IDs, H_min values, cohort sizes, DOIs, or validation numbers. If a source doesn't have a detail, leave it out or flag it as needed. When GAPE_WEB_v13.py has the canonical H_min values, pull them from there rather than approximating.

---

## Stage 1 decision logic — safe-handling rules for all result patterns

These rules are what the report-generation layer consults once Stage 1 and Stage 2 have produced outputs. They must cover EVERY result pattern. No combination of Stage 1 result × Stage 2 result should return "no finding" if an architectural departure is present.

1. **Stage 1 pooled-entropy A_immune elevated above DETECTABLE tier + Stage 2 localizes to breast_ductal** → breast card fires, report follows breast-epic specification.
2. **Stage 1 pooled-entropy A_immune depressed below DETECTABLE tier (negative direction) + Stage 2 localizes to colon_epithelial** → CRC card fires, report follows crc-epic specification.
3. **Stage 1 pooled-entropy A_immune elevated + Stage 2 localizes to lung_epithelial with top-1/top-2 ratio ≥ 2×** → lung-epic card fires, report follows lung-epic specification with mandatory smoking-status-stratified clinical action matrix. Current smokers get the mandatory smoking-adjustment sentence per the ad-immune-style age-disclosure rule; lung-epic fires only when Stage 2 confidence ratio exceeds 2× (tighter than the generic rule) to control smoking-driven false positives.
4. **Stage 1 Xu-538 pooled-entropy null + AD 7-CpG Rule A A_dir elevated above AD DETECTABLE threshold** → AD card fires, report follows ad-immune specification.
5. **Stage 1 pooled-entropy elevated or depressed + Stage 2 localizes to a tissue with a matching card not already named above** (prostate, HCC, pancreatic, gastric, bladder, cervical, kidney, glioma) → that disease card fires. Each card specifies its own tier thresholds, expected direction, and clinical workup recommendation.
6. **Stage 1 flagged + Stage 2 localizes to a tissue WITHOUT a matching card** (stromal, HSC off-atlas) → Report: "ARCHITECTURAL FLAG. Immune-class A-score [direction] from age-matched baseline at [magnitude] tier. Stage 2 localization suggests [tissue]. EDEAR does not have a validated disease card for this tissue in this version; recommend clinician review and standard-of-care workup for [tissue type] conditions." Explicit list of standard workup per Moss 2018 tissue category included in the report.
7. **Stage 1 flagged + Stage 2 returns no solid-organ localization + AD-directional null** → Report: "UNEXPLAINED ARCHITECTURAL FLAG. Immune-class A-score [direction] from baseline. No solid-organ localization at Stage 2. AD-directional panel does not exceed tier threshold. Possible patterns: chronic infection, autoimmune condition, hematologic malignancy (off-atlas), pre-symptomatic condition below Stage 2 resolution floor. Recommend CBC with differential, CRP, ANA panel, and repeat EDEAR in 3 months."
8. **Stage 1 null across pooled-entropy AND all disease-directional panels** → NORMAL. Serial-sample next interval per age-based cadence.

Items 6 and 7 are the safe-handling rules for edge cases. They are NOT substitutes for having a card for every Moss-localizable tissue; they handle tissues outside the Moss atlas (stromal, off-atlas hematologic) and the non-solid-organ immune drift patterns. The 12-card required set above closes the major solid-organ coverage gap.

---

## Validation tiers per card

- **cross_platform_validated** — two independent cohorts, same panel, same H_min, same pipeline, direction preserved, effect size within cohort variation. Current: breast-epic, ad-immune.
- **single_cohort_validated_with_consistent_published_reference** — one cohort at per-patient level plus an independent published finding on the same dataset corroborating that the disease signal IS present (different methodology is OK; confirms signal exists). Current: crc-epic (GSE51032 + Zhao 2020).
- **cross_platform_validated_two_cohorts** — same as cross_platform_validated but with the specific condition that both are at per-patient level with SHA-locked re-runs. Current: breast-epic.
- **stage_2_only_validated** — Stage 2 Moss NNLS localization correct per VAL-041, Stage 1 per-patient pre-dx data not yet available. Card fires only if Stage 2 localization is high-confidence (top-1 ΔA > 2× top-2 ΔA). No Stage 1 per-patient claim. Current: prostate-epic (VAL-058 GSE269244 tissue case-control, paired d = +0.497).
- **multi_modal_validated** — Stage 2 Moss NNLS localization validated AND tissue-level ΔA confirmed on an independent published distance-annotated or tumor-normal cohort (VAL-039-class or TCGA-class anchor) AND cohort-level Stage 1 support from VAL-046 or equivalent. Three independent published datasets anchor the card. Stage 1 per-patient pre-dx remains pending. Current: lung-epic (VAL-041 Moss + VAL-039 Kadota + TCGA-LUAD/LUSC + VAL-046 UK Biobank cohort-level + VAL-063 TCGA-LUAD tissue arm paired d = +1.020), hcc-epic v0.2 (VAL-059 GSE298812 ccfDNA d = +0.634 + VAL-064 TCGA-LIHC tissue arm paired d = +0.498 pooled / +0.664 non-viral arm — tier promoted from cohort_screening_validated 2026-04-24).
- **cohort_screening_validated** — Stage 1 signal at cohort level (per VAL-046 or equivalent) but NOT per-patient validated. Card may fire at higher tier thresholds to control false-positive rate. Current: (no cards currently at this tier — hcc-epic was at this tier in v0.1 and is now multi_modal_validated as of v0.2).
- **substrate_restricted** — modifier flag combined with any validation tier above. Indicates the card is validated only on a specific specimen type (ccfDNA plasma vs whole-blood leukocyte vs urine vs CSF) and explicitly documents which substrate(s) were tested and which were NULL. Current: hcc-epic (blood arm: ccfDNA plasma validated; whole-blood leukocyte NULL at d = −0.156).
- **tissue_arm_validated** — modifier flag combined with any primary validation tier above. Indicates the card has its own per-card tumor-vs-adjacent-normal tissue validation run on a public tissue cohort (TCGA or equivalent), beyond the framework-level tissue evidence of VAL-001/VAL-009/VAL-039. The card's Xu-538 panel has been tested directly on tumor tissue and separation demonstrated. Required element for all new cards going forward per CCL-011; retroactive upgrade in progress. Current: prostate-epic (VAL-058 GSE269244, paired d = +0.497 — tier built directly from tissue because no blood cohort available); breast-epic v2.2 (VAL-060 TCGA-BRCA HM450, paired d = +0.676); crc-epic (VAL-061/062 TCGA-COAD HM450 paired d = +0.724 cycling, +1.066 immune TIL); lung-epic (VAL-063 TCGA-LUAD HM450 paired d = +1.020); hcc-epic (VAL-064 TCGA-LIHC HM450 paired d = +0.498 pooled, +0.664 non-viral arm).
- **post_dx_only** — card validated for post-diagnosis treatment-response monitoring (VAL-044), not pre-diagnostic screening. Glioma-epic is the archetype.
- **exploratory** — directionally positive but below confirmation threshold. Card flagged as exploratory in every patient report.
- **exploratory_with_cohort_heterogeneity** — Stage 1 immune-class score produces opposite-sign Cohen's d across independent cohorts of the same disease, with pattern not explained by single artifact (data-format issue, batch correction, sample mislabeling). Card holds at this tier when the cohort-direction-flip is real biological/cohort-design heterogeneity that cannot be resolved by data-integrity checks alone. Path to higher tier requires either (a) a card-specific panel substitution with dedicated H_min calibration, (b) Test 2 lymphoid/myeloid sub-panel split when OQ-2026-01 immune-atlas staging operationalizes, or (c) HPV-stratified or covariate-adjusted re-run that resolves the cohort-direction-flip. Current: cervical-epic (VAL-073 Verlaat Amsterdam d=+0.73 anchor positive; VAL-074 Farkas Stockholm d=−0.61 negative-direction with HPV-negative normals; VAL-081 Lando Oslo d=−0.43 negative-direction at n=270 vs external comparator).
- **framework_calibrated_pending_per_patient_validation** — disease-specific A-score signatures characterized in framework calibration source (Issue 002 Immune class chapter or equivalent), per-patient pre-diagnostic blood validation cohort identified but not yet run. Card is operational at the patient-facing tier-bin level (SUPPRESSED / NORMAL / ELEVATED / FLOOR_BREACH) with framework-derived thresholds; per-patient sensitivity at pre-diagnostic windows pending. Path to higher tier requires pre-locked Phase 9/12-equivalent VAL run on the identified cohort. Current: heme-epic (Issue 002 calibrated; EnviroGenomarkers VAL-082 priority).
- **null_documented** — honest negative on a tested panel/cohort combination. Documented so it doesn't get re-tried.

**Patient-facing tier vocabulary (defined here at framework level, applies across all cards as of heme-epic v0.1):**

The four-bin patient-facing tier set is **SUPPRESSED / NORMAL / ELEVATED / FLOOR_BREACH**, with MARGINAL bands flanking NORMAL on both sides. SUPPRESSED indicates A-immune > 1 SD below age-decade healthy reference (immunocompromised state — post-chemo, post-transplant, HIV, primary immunodeficiency, cachexia, late-stage marrow infiltration). MARGINAL indicates A-immune within 1-2 SD of healthy reference in either direction. ELEVATED indicates A-immune above the DETECTABLE threshold for a card. FLOOR_BREACH indicates A-immune at or above the URGENT/FLOOR_BREACH threshold for a card. Heme-epic introduced SUPPRESSED in v0.1; other cards (cardio-epic, immune-atlas, ad-immune for older patients) inherit it.

**Card-internal tier vocabulary (defined here at framework level as of ad-immune v2.2 / 2026-04-26):**

The card-internal A-score tier set used in card JSONs and per-disease scoring is **BELOW_NORMAL / NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH** (6 bins, including the negative side). `BELOW_NORMAL` covers A-score (or per-class architectural fraction) ≤ −1.0 SD below the within-cohort or 80-cell HC reference. It indicates non-disease-of-the-card differentials — immunosuppression, treatment effect, post-chemo/post-transplant state, primary immunodeficiency, or class-specific suppression patterns (e.g. VAL-091 GIFT PSP/CBD reads cortical-neuron Stage 2 fraction at d=−0.51 vs HC). **Below-normal is a real signal, not a missing one.** Card-internal `BELOW_NORMAL` corresponds to patient-facing `SUPPRESSED` (the names differ because card-internal vocabulary describes the score band; patient-facing vocabulary describes the clinical interpretation, and patient interpretation may vary by class). Heme-epic v0.1 introduced this bin under the name `SUPPRESSED`; ad-immune v2.2 standardized the card-internal name to `BELOW_NORMAL` per ad-LL-007 after Heath flagged the gap during VAL-091 review. Other cards inherit at next version bump.

---

## What this Cookbook v2.1 changes from v2.0

1. **Universal pipeline rule promoted to top-level section.** The "every patient's first test is immune-class on Xu-538" principle was previously stated inside the Stage 1 subsection. v2.1 makes it the first substantive section after the security boundary so it cannot be missed.

2. **Kidney-epic added to the required card set.** v2.0 had an 11-card expansion table that omitted kidney despite kidney_epithelial being in the Stage 2 output space. v2.1 makes it 12 cards. Kidney-epic to be built per the expansion roadmap.

3. **Coverage requirement made explicit.** v2.0 described the expansion as "commercial deployment needs every Moss tissue covered." v2.1 makes coverage a design constraint, not a future goal. A Cookbook with unfilled Moss-localizable tissues is not deployable.

4. **Stage 1 decision logic tightened.** Items 1-7 now cover every combination of Stage 1 × Stage 2 output explicitly. Items 5 and 6 are for tissues OUTSIDE the Moss atlas coverage, not for tissues that should have cards.

5. **Affiliation corrected.** "IAMPerformance Inter-Domain Research Institute" in the header (was "Research Initiative" in v2.0 header).

---

## What this Cookbook v2 fixed from v1

**Corrections made after re-reading the Evidence Report 2026-04-23:**

1. **Stage 1 class is IMMUNE for all three diseases, not per-disease.** v1 had breast on secretory at Stage 1 — wrong. Buffy coat contains immune cells. The Stage 1 measurement IS immune-class A-score, regardless of which disease is being flagged. Secretory only enters at Stage 2, where the Moss NNLS-deconvolved breast_ductal β is scored against secretory H_min.

2. **CRC signal is inverted, not positive, on Stage 1 immune panel.** v1 called CRC "immune inversion + secretory positive" but built the card numbers incorrectly. Phase 12 (re-run 2026-04-23, SHA-verified) confirms: CRC on Xu-538 immune is d = −0.33 all-pre-dx (p = 0.009), direction is NEGATIVE.

3. **Directional scoring (AD Rule A) is a first-class metric, not pooled entropy alone.** VAL-051 Directional-Score Principle: pooled entropy misses bidirectional disease signals. AD Rule A d = +0.624 holdout, pooled entropy on same panel same cohort d = +0.077 null.

4. **VAL-041 is the Stage 2 anchor, not a preprocessing step.** v1 described Moss NNLS as part of "Step 1 deconvolution" upstream of class-level scoring. That conflates two distinct stages. Stage 1 is class-level immune flag on bulk blood. Stage 2 is Moss NNLS deconvolution producing 18 per-tissue β → 18 per-tissue A-scores → max-ΔA localization.

5. **All headline numbers are now from SHA-verified re-runs of Phase 9 and Phase 12 on raw GEO matrices.**

---

## Per-card v0.1 build expectations — exhaustive line-by-line checklist

Every card v0.1 published into this Cookbook must contain every block listed below. A card missing any of these blocks cannot pass to v0.1 publish. This list is the canonical reference for future AI sessions building cards — read it before opening a card README, before writing the card JSON, and before running the first VAL study. Adopted 2026-04-25 after the pancreatic-epic build revealed that an earlier per-card authoring checklist (§ above, Steps 1-11) covered the validation methodology but did NOT explicitly require the saturation level architecture from the GAPE Reproduction Paper Part 2.4. This section closes that gap and makes every block explicit.

### Block 1 — Card header

Six fields required: card_id, card_version (vX.Y), card_date (ISO YYYY-MM-DD), supersedes (or null for v0.1), card_status (validation_tier + modifier), card_scope (one-paragraph description of what the card covers).

### Block 2 — Disease description

Required: disease.name, common_synonyms list, full_name, global_burden statement, primary_site, tissue_class (one of the 8 GAPE classes), h_min(class) value (six decimal places, pulled from G-002 or G-003b posterior, not approximated), healthy_reference_beta for the target tissue, biology_note describing any disease-specific architectural feature relevant to detection (PDAC stromal density, HCC viral hepatitis blunting, glioma blood-brain barrier, etc.).

### Block 3 — ICD-10 scope

Required: included codes list, excluded codes list with reason for exclusion. Pancreatic neuroendocrine PNETs (C25.4) excluded from pancreatic-epic is the canonical example.

### Block 4 — Specimen pathways supported

For every supported IDAT input pathway, document: specimen description, primary_pathway boolean, stage_1_role, stage_2_role, stage_3_role, deployment_anchor (which VAL anchors validation), validated_at tier, full list of pre-analytical confounds specific to that specimen, when_to_use guidance. Specimens to consider for every card (mark applicable / not_applicable / exploratory / unvalidated, never silently omit): plasma cfDNA, tissue biopsy, urine cfDNA, saliva, CSF, sputum, stool, cervical swab, pap smear cytology, FNA cytology, ascites, pleural effusion, BAL, semen, sweat, breast milk, vaginal discharge, nasal swab, ear cerumen, ERCP juice. The card must explicitly classify each candidate specimen rather than leaving it implicit. PDAC's 7-pathway documentation (plasma, tissue, juice, urine, saliva, CSF-not-applicable, FNA) is the v0.1 template.

### Block 5 — Universal Stage 1 / Stage 2 / Stage 3 pipeline applied to this card

Stage 1 must explicitly state: panel_id, panel_sha256 (file-bytes), n_cpgs, source_paper, scoring_methods (primary + secondary, with rationale for which is primary), H_min(immune) = 0.838889 verbatim. Stage 2 must explicitly state: moss_atlas_reference, target_tissue (Moss-atlas tissue name), target_class, h_min(class), healthy_reference_beta, score_formula, localization_criterion (top-1 + 2× second-place rule), ambiguous_action. Stage 3 must explicitly state: method (EpiDISH RPC + Salas IDOL-Ext), when_applied trigger, literature_predicted_pattern_for_this_disease, operational_status (often "pending immune-atlas" — that is acceptable; silent omission is not).

### Block 6 — CCL-027 four-question bidirectional cancellation guard (with CCL-030 Test 1 / Test 2 distinction)

Mandatory. Every card answers all four questions inline in both card JSON and card README. Per CCL-030 (formalized 2026-04-25), Stage 1 immune-class scoring has TWO distinct tests that must be reported separately:

**Test 1 — pooled A_immune on the full Xu-538 panel.** Standard scoring, direction-agnostic at the per-patient level due to Shannon symmetry. Operational on every disease in the record. This is what every Stage 1 validation actually runs.

**Test 2 — lymphoid-marker vs myeloid-marker sub-panel split.** Run pooled A_immune separately on the lymphoid-assigned subset of Xu-538 CpGs and on the myeloid-assigned subset; compare directions. Opposite directions with comparable magnitudes confirms AD-style lineage-level bidirectional cancellation. Same direction in both arms rules out lineage cancellation. **Test 2 is currently NOT runnable on any disease** — it requires per-CpG lineage assignment from an immune-cell-type methylation atlas (Salas IDOL-Ext or equivalent), which is OQ-2026-01 immune-atlas staging and not yet operational.

The four questions, answered with this distinction in mind:

1. **Pooled-entropy expected direction with citation.** What does Test 1 (pooled A_immune) read on this disease in this compartment? Cite the validation cohort or precedent.

2. **Bidirectional-cancellation risk with citation.** Has Test 1 produced a pooled null cross-cohort where a pooled pass was expected? If yes, the pattern is "pooled-null + directional-pass" (which AD and PDAC exhibit). The mechanism for that pattern is NOT confirmed as lineage cancellation until Test 2 is operational — possible non-lineage causes include z-scoring sensitivity gain and cohort/batch structure. Cite the cohorts.

3. **Directional-panel fallback specification.** If Test 1 nulls cross-cohort, document whether a per-CpG ±1 z-scored directional panel has been built. If yes, document the training cohort, the holdout cohort, the panel CpG count, and the per-patient holdout d. If no, document "none needed at current evidence" with rationale.

4. **Lymphoid-vs-myeloid expected pattern from literature (Test 2 placeholder).** Document the literature-anchored expected pattern for this disease: do reviews predict lymphoid suppression + myeloid expansion (PDAC, AD), uniform inflammation (breast), or some other pattern? Flag this answer explicitly as "Test 2 pending OQ-2026-01 immune-atlas; literature-anchored expected pattern only at v0.1; not directly measured at the Xu-538 panel level." This question CANNOT be answered as a confirmed measurement at v0.1 — only as a literature-predicted hypothesis.

**Per-CpG cohort Δβ direction percentage is descriptive only.** Cards may report it for completeness. Cards must NOT use it as a mechanism diagnostic. Per-CpG cohort-level Δβ direction does not predict per-patient A-score direction (Shannon symmetry) and does not measure lineage assignment.

A card with any of the four questions unanswered cannot publish. The cervical-epic build (April 25, 2026) is the canonical example of a card whose four-question guard answers all four cleanly without triggering directional fallback construction or flagging bidirectional-cancellation risk — Test 1 pooled A_immune passes (VAL-073 d = +0.73, monotonic Normal < CIN3 < SCC), Test 2 deferred to OQ-2026-01.

### Block 7 — Saturation level architecture (added 2026-04-25, mandatory)

Pulled from GAPE Reproduction Paper Part 2.4A and Part 2.4B. Every card must contain:

**7a. The 5-substrate A_ceiling row for the disease's tissue class.** A_ceiling(c, s) = 1 / H_min(c, s), pre-computed per the 40-cell grid. For secretory-class diseases (breast-ductal, prostate-epithelial, hepatocyte, pancreatic-exocrine): A_ceiling values are methyl 1.1859, nucl 1.0177, fuzz 1.1793, wps 1.5760, frag 1.4332. For cycling-class diseases (lung-epithelial, colon-epithelial, gastric-epithelial, bladder-epithelial, cervical-epithelial, skin): methyl 1.1681, nucl 1.0203, fuzz 1.2210, wps 1.5938, frag 1.4536. For immune-class (heme malignancies): methyl 1.1921, nucl 1.0102, fuzz 1.2043, wps 1.6959, frag 1.4054. For terminal-class (brain, cardiomyocyte): methyl 1.2939, nucl 1.0080, fuzz 1.3569, wps 1.0429, frag 1.6002. For stromal: methyl 1.1588, nucl 1.0145, fuzz 1.2014, wps 1.6322, frag 1.3799. For stem_pluri: methyl 1.0182, nucl 1.2503, fuzz 1.0385, wps 1.1050, frag 1.0271. For stem_adult: methyl 1.1445, nucl 1.0407, fuzz 1.0196, wps 1.0112, frag 1.1886. For progenitor: methyl 1.1734, nucl 1.0280, fuzz 1.0396, wps 1.0121, frag 1.2361.

**7b. Structural saturation flag per substrate.** A class-substrate pair is structurally saturated when A_ceiling < 1.10 (BREACH threshold). The 15 of 40 structurally saturated cells per the Reproduction Paper Part 2.4A. The card must list which substrates are structurally saturated for the disease's class and explicitly state which substrates carry BREACH-tier discrimination signal vs which substrates can only carry NORMAL/MARGINAL/DETECTABLE signal. For example: secretory class has nucl structurally saturated (A_ceiling 1.0177); a secretory-class card cannot use nucl for FLOOR_BREACH discrimination, only for sub-DETECTABLE signal.

**7c. Runtime saturation flag thresholds.** A_ceiling − 0.005 for each substrate per Part 2.4B. The card must specify for each substrate the exact A-value at which the runtime saturation flag fires. When the flag fires, the substrate is excluded from A_active aggregation per the Reproduction Paper Part 3.3 and the patient report carries a saturation alert for that substrate.

**7d. Per-substrate detection strategy for this disease.** Given the structural and runtime saturation flags, document which substrates carry the primary signal at each tier (NORMAL, MARGINAL, DETECTABLE, URGENT, FLOOR_BREACH). For PDAC (secretory class), methyl is primary at all tiers, fuzz/wps/frag carry confirmatory signal at DETECTABLE-and-above, nucl is restricted to sub-DETECTABLE drift detection only. For terminal-class diseases like glioma where wps is also structurally saturated (1.0429), only methyl, fuzz, and frag carry BREACH-tier signal — wps and nucl are sub-DETECTABLE-only.

**7e. The seminoma-style inversion check (for stem_pluri-class cards only).** Three of five stem_pluri ceilings sit below BREACH (methyl 1.0182, fuzz 1.0385, frag 1.0271), so A_combined elevation does not work for pluripotent-class cancers — the discrimination signal is multi-substrate divergence (one substrate up, another down). Stem_pluri cards must specify the divergence detection rule explicitly.

### Block 8 — Mandatory covariates

Every card publishes a covariate table with three columns: stratify_analysis (Yes/No/Observation), report_field (Yes/No), rationale (with citation when relevant). The standard universal covariates: sex, age, smoking, BMI, alcohol, recent acute infection (<2 wk → defer), pregnancy (decline scoring or flag), recent transplant/transfusion/chimerism (decline scoring), active or recent chemo/radiation (decline scoring), hormonal contraception/HRT, autoimmune disease history, hemolysis at draw, family history of the disease, race/ethnicity. Disease-specific covariates added per disease (PDAC: diabetes T2D status, recent ERCP/stent, recent pancreatitis, chronic pancreatitis history, occupational exposures; HCC: viral hepatitis status, fibrosis grade, etiology; lung: smoking dose-stratified; cervical: HPV genotype, parity, hormonal contraceptive duration; glioma: corticosteroid use, recent seizure, anti-epileptic use; AD: APOE genotype). Diurnal collection time and fasting status as observation-only fields.

### Block 9 — Tier thresholds

Universal Cookbook tier structure: NORMAL < 1.01, MARGINAL ≥ 1.01, DETECTABLE ≥ 1.05, URGENT ≥ 1.07, FLOOR BREACH ≥ 1.10. If the disease-specific literature justifies different thresholds, the card may deviate but must cite the deviation rationale. Tier thresholds may be expressed in z-score units when the directional fallback is the primary metric (PDAC v0.1 example: NORMAL < +0.5z, MARGINAL +0.5 to +1.0, etc.). Tier thresholds calibrated on tissue cohorts must explicitly state that blood-deployment thresholds may need re-calibration.

### Block 10 — Clinical action matrix

Every combination of Stage 1 tier × Stage 2 localization confidence × covariate state has a named action. No generic "see your doctor" — every cell specifies LDCT, colonoscopy, MRI, EUS, CBC with differential, pulmonology consult, oncology referral, serial-sample at 6 months, repeat at 3 months, etc. Cite the clinical guideline (USPSTF, NCCN, ACR, EASL, AGA) governing each recommendation. Disease-specific special rules embedded explicitly (PDAC: new-onset T2D ≥50 yr triggers paraneoplastic workup regardless of Stage 2; HCC: cirrhotic patients require AASLD surveillance regardless of EDEAR output).

### Block 11 — Trajectory monitoring guidance

Every card specifies the recommended cadence for serial sampling: high-risk patient cadence (typically every 6 months), average-risk patient cadence (typically every 1-3 years depending on disease incidence by age), trajectory slope diagnostic threshold (PDAC v0.1 example: > +0.3 z-units per year), and the rule for two-consecutive-MARGINAL escalation (consecutive MARGINAL readings 6 months apart triggers DETECTABLE-tier action even if individual readings stay MARGINAL).

### Block 12 — Validation summary

Every VAL study covering this card listed in a table with VAL ID, specimen, cohort, n, primary result with Cohen's d and 95% CI and p, status (anchor / exploratory / null / partial-fail). Stage 1 anchor study clearly identified. Tissue arm modifier clearly identified. The validation tier and modifier statement at the bottom of the table.

### Block 13 — Known limitations

Numbered list. Every limitation stated honestly with no smoothing. The PDAC v0.1 list of 11 limitations is the v0.1 floor — no card should publish with fewer than ~8 limitations, because every card has limitations and silent omission is dishonest. Bias of training cohort, missing covariates in metadata, single-cohort risk, non-validated specimen pathways, all explicitly itemized.

### Block 14 — Open questions for v0.2+

Numbered list with three columns: open_question, source (which limitation or VAL outcome generated it), action_needed (specific next step). The PDAC v0.1 list of 9 open questions is the template. dbGaP applications, partner outreach, additional cohort searches, lab partnership tier escalations all itemized.

### Block 15 — Sources and citations

Every cited paper has full DOI link or PMID link. Every cited cohort has GEO accession link. Every cited guideline has organization + year. The card README must be readable as a standalone document — no implicit references that require knowing what "VAL-046" or "Phase 12" mean without explanation.

### Block 16 — Pre-registration chain

Every VAL study covering the card listed with its pre-registration SHA-256 (sealed before any β-value access) and any amendment SHA-256. The Xu-538 panel SHA-256 listed verbatim. RNG seed for all VAL scripts listed verbatim. The chain must be reproducible from public-access data using Python 3 stdlib only — no proprietary dependencies.

### Block 17 — Reproduction bundle

Two file lists: (a) Heath-only Cookbook IP files in this card's directory: card README, card JSON, directional panel JSON if applicable, card-specific reference files. (b) GitHub-pushed reproducibility files in IAM-Validation/Biological_Physics/validation_runs/: VAL scripts, prereg files, amendment files, seal files, outcome files, results JSONs, manifests, clinical metadata, deferred-VAL notes. The two lists must not overlap. Per memory rule #14 the deployment boundary is non-negotiable.

### Block 18 — Lessons learned

Numbered list of card-specific lessons (panc-LL-### for pancreatic, breast-LL-### for breast, etc.). Each lesson: title, observation, implication, date_recorded, source_val (list). Cross-card lessons promoted to LESSONS_LEARNED.md as CCL-### entries when a pattern appears across multiple cards. The PDAC v0.1 set of 7 panc-LL entries is the template.

### Block 19 — Card validation tier statement

Final block. Validation tier (cohort_screening_validated, single_cohort_validated, cross_platform_validated, multi_modal_validated, stage_2_only_validated, exploratory, or null_documented). Modifier if applicable (tissue_arm_exploratory, ccfDNA_substrate_restricted, etc.). Path-to-next-tier statement: what specific next study moves this card to the next tier. Three or four sentences maximum. The PDAC v0.1 closing statement is the template.

### Block 20 — What we discovered (plain-language section)

Mandatory. Every card README contains a plain-language summary section that any clinician, partner, or non-specialist can read end-to-end and understand exactly what was found, what we are sure of, what we are not sure of, and how well the framework can detect this disease right now. This section is the partner-facing trust artifact — it is what allows a physician handed the card to verify that the framework is not fitting data, is honest about its limitations, and is operating on real evidence. Adopted 2026-04-25 after the pancreatic-epic build revealed that the technical card content was complete but the plain-language synthesis was missing.

The Block 20 section must contain six subsections, in this order:

**20a. Why this disease is hard to detect.** Two or three paragraphs in non-technical language describing the clinical detection problem. PDAC's dense stroma and silent presentation, glioma's blood-brain barrier, cervical cancer's HPV-driven slow progression and the existing pap smear screening infrastructure, AD's pre-symptomatic decade, etc. Avoid jargon. A patient or family member should be able to read this and understand why the standard-of-care has the gaps it has.

**20b. What we tested.** One paragraph naming every cohort, every specimen type, every n. The PDAC v0.1 example: "We tested every accessible HM450 cohort for PDAC. TCGA-PAAD matched tumor/normal at n=5 effective paired patients after QC. GSE49149 large unpaired cohort at n=196 (the largest publicly available PDAC tissue methylation dataset). GSE74071 multi-substrate at 28 samples covering tumor, adjacent normal, pancreatic juice, and CAFs. Plus VAL-046 Rotterdam blood pre-diagnostic cohort at n=182 for cohort-level support." Plain enumeration.

**20c. The headline finding.** One to three paragraphs stating the single most important methodological or clinical finding from the validation work. The PDAC v0.1 example: PDAC is the second confirmed bidirectional-cancellation disease at the Xu-538 panel level after AD. What "bidirectional cancellation" means in plain language. The 50/50 per-CpG split confirmed in three independent cohorts. The recovery method (directional fallback panel). Comparison to other cancers' 62-70% per-CpG positive direction. This is where the card argues that something genuine was discovered.

**20d. What we can be sure of, in order of confidence.** Numbered list. The PDAC v0.1 template has six items: (1) firmest finding, (2) per-patient validation level achieved, (3) framework rule generalized, (4) cohort-level pre-dx detection unchanged, then negative items: (5) what we cannot be sure of yet, (6) honest unresolved outliers. The list orders items from most-confident to least-confident. Honesty in the negative items is non-negotiable — these are what allow a physician to trust the positive items.

**20e. How well we can detect this disease right now, by specimen type.** One paragraph per supported specimen pathway from Block 4. State plainly what is anchored, what is exploratory, what is unvalidated, what is not applicable. The PDAC v0.1 example: plasma cfDNA cohort-level supported but per-patient pending; tissue biopsy validated as Stage 2 ceiling reference; pancreatic juice exploratory at n=4; urine/saliva/FNA documented but no validation cohorts; CSF not applicable. Use plain "this works" / "this might work" / "this hasn't been tested" / "this doesn't apply" language. Avoid technical hedging.

**20f. The honest picture (closing paragraph).** One paragraph synthesizing the card's clinical trust profile. Where it is anchored, what its primary limitation is, what the priority next-step is. The PDAC v0.1 closing example: "The card is now anchored by Rotterdam at the cohort level and equipped with a directional panel for tissue-arm per-patient detection, with the blood-deployment validation gap explicitly logged as priority #1 next-step." Three sentences.

**Why Block 20 exists.** Cards are technical documents containing pre-registration SHAs, MCMC posteriors, Cohen's d values with 95% CIs, A_ceiling tables, and the dimensional bookkeeping that allows another scientist to reproduce the work. A physician handed the card needs to know whether to trust it, and the technical bookkeeping does not by itself answer that question. Block 20 is the trust bridge — a plain-language synthesis that any reader can use to verify the card's claims and limitations match the evidence the rest of the card documents. A card without Block 20 is incomplete because the partner-facing trust artifact is missing.

**The order matters.** Block 20 is the second-to-last section in every card README, sitting between Block 19 (Card validation tier statement) and the reproduction bundle. The reader who flips to the back of the document encounters: tier statement → plain-language synthesis → reproduction bundle. This sequencing means the synthesis is the last thing read before the bundle and stays in working memory.

---

**Cohort-completeness rule (CCL-029, mandatory).** Every card v0.1 build runs every publicly-accessible 450K/EPIC methylation cohort for the disease (GEO + TCGA + ArrayExpress + curated references). Even if 20 cohorts. Partial coverage of public data creates a card that cannot honestly state its boundaries. dbGaP-gated and partner-collected cohorts are next-validation-steps, not in-scope for v0.1.

**Universal Stage 1 H_min rule (panc-LL-007 generalized).** Stage 1 ALWAYS scores Xu-538 against H_min(immune) = 0.838889 regardless of which disease card is being run. The Xu-538 panel is the immune-class panel. The disease-tissue class is a Stage 2 concept only. Future card authors must verify H_min(immune) = 0.838889 in their Stage 1 scripts before publishing.

**No-fabrication rule (memory #29, absolute).** Read source files before writing anything referencing their content. Never invent CpG IDs, H_min values, cohort sizes, DOIs, or validation numbers. If a source doesn't have a detail, leave it out or flag it. When GAPE_WEB_v13.py and the GAPE Reproduction Paper have the canonical values, pull them from there rather than approximating.

**File counts and SHAs — mandatory on every edit.** Record line count before, line count after, delta. Record SHA-256 of every static artifact (panel, card, README, script, results JSON). Verify SHAs match expected values before writing any reference to them. SHA mismatch is never silent — it blocks the commit.

**Language discipline — automatic on every artifact.** Use "consistent with," "tested against," "data are consistent with," "architectural signal detected," "elevated above age-matched baseline." Never use "confirms," "validates," "proves," "first derivation," "154 years no one has." Never use "it matters" or "why it matters" in any Heath-facing output.

---

## Physical vault, not cloud

This Cookbook lives on the physical machine at IAMPerformance HQ. The master README and each card are distributed to authorized partners under NDA, never committed to public GitHub or cloud storage. Per the patent disclosure boundary: operational constants (panels, H_min values, tier thresholds) are disclosable; derivations, class-assignment rule, and MCMC protocol are NOT.

The Evidence Report (`GAPE_Evidence_Report_UPDATED.html`) IS the public-facing reproducibility document. It contains every panel SHA, every cohort SHA, every script URL, and every JSON result needed for an independent party to replicate any numerical claim. The Cookbook adds the clinical deployment layer that the Evidence Report deliberately does not publish.
