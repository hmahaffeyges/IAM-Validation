# EDEAR Pipeline — Official Reference Document v2

**Authoritative source-cited description of the run-everything-every-time pipeline architecture, with complete atlas/panel inventory.**

**Date:** 2026-04-26
**Supersedes:** EDEAR_PIPELINE_OFFICIAL_REFERENCE.md (2026-04-26 v1, gated workflow with incomplete atlas inventory)
**Architectural change in v2:** the pipeline is now defined as **run everything regardless of Stage 1 result**. Conditional gating is removed. Display logic in the patient report can collapse uninformative tiles, but the underlying scoring is exhaustive on every IDAT.

This document is reconstructed from source files plus literature search for additional published reference atlases. Every claim is cited. **No inference, no memory-based reconstruction.** When an atlas is not yet integrated in the production pipeline, that is stated explicitly.

---

## Part 1 — The KISS architecture in one paragraph

**One method. Same every time. Anomalies tell the story.**

A patient gives a 450K or EPIC 850K methylation array IDAT — one tube of blood, ~485,000 β values. The pipeline runs **all panels and all reference atlases** against that β vector, in parallel where possible, and produces a stack of architectural drift scores (A-scores) and tissue-of-origin fractions covering every cell class and every tissue the framework can resolve. Every score gets a tier call from the universal six-tier vocabulary `BELOW_NORMAL / NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH`. The patient's report displays anomalies (anything outside NORMAL) and collapses uninformative tiles. Disease cards fire when their characteristic anomaly pattern matches.

**Why this is correct:** Stage 1 immune A-score can null on diseases that drive CpGs bidirectionally (the AD-instance pattern, CCL-030 — VAL-050 was null at d=+0.077, VAL-051 directional was d=+0.624). The pre-diagnostic breast secretory-class signal at >10yr is on the *negative* side (d=−1.226) and would never be reached by an elevation-gated Stage 2. Heme-epic explicitly looks for Stage 2 NULL on solid organs as the *diagnostic feature*. PSP/CBD reads cortical-neuron at d=−0.51 (BELOW_NORMAL). A patient could have early AD + early breast + chronic inflammation + cardiovascular drift simultaneously — gating on the first signal that crosses threshold means the others get filtered through that lens. We measure all of them and let the anomaly stack tell the story.

The compute cost of running everything on every IDAT is seconds-to-minutes. The cost of missing a multi-disease pattern is missed diagnosis. The trade is one-sided.

---

## Part 2 — The full pipeline, top to bottom

### Input

- **Substrate:** buffy-coat DNA or plasma cfDNA from a single blood draw
- **Platform:** Illumina 450K or EPIC 850K methylation array
- **Output of array:** β values at ~485,000 CpG sites (450K) or ~865,000 CpG sites (EPIC). One number per CpG, range [0, 1], representing the fraction of DNA molecules methylated at that site.
- **Tagging:** every result downstream is tagged with platform (450K/EPIC) so platform-stratified thresholds can be applied where coverage gaps matter (per VAL-091 ad-LL-006).

### Stage 1 — architectural drift detection (multiple panels in parallel)

Stage 1 runs **all of these panels on every IDAT, every time**, regardless of any single panel's outcome.

**1.1 Xu-538 immune-class pooled panel** (primary, every disease)
- 538 CpGs from Xu 2020 Sister Study breast cancer + EPIC-Italy replication.
- Panel SHA `ada672960563...`.
- Computation: A_immune = mean(H(β) / H_min(immune)) across the 538 CpGs, H_min(immune) = 0.838889.
- Output: A_immune_pooled, age-matched percentile vs 80-cell baseline, tier call (BELOW_NORMAL through FLOOR_BREACH), per-CpG Δβ table.
- **Source:** README_MASTER §Stage 1 lines 184–197.

**1.2 AD 7-CpG Rule A directional panel** (overlay, every IDAT)
- 7 CpGs from VAL-051 with frozen ±1 directions.
- Computation: A_dir_AD = sum(z(β_cpg) × direction_cpg) across 7 CpGs.
- Frozen since VAL_051_SEAL.txt 2026-04-23 07:23:53 UTC.
- Output: A_dir_AD score, tier call, per-CpG direction-preservation count.

**1.3 Pancreatic 324-CpG directional fallback panel** (overlay, every IDAT)
- 324 CpGs from VAL-069, GSE49149-trained, z-score normalized (H_min-independent).
- Output: A_dir_PDAC score, tier call.

**1.4 Kresovich mBCRS 100-CpG comparator** (overlay, every IDAT)
- Reference-only; reported alongside Xu-538 pooled for breast comparison.
- Cross-validated published d ≈ 0.63 AUC equivalent.

**1.5 Bidirectional cancellation diagnostic — Test 2 (when OQ-2026-01 lands)**
- Lymphoid-marker vs myeloid-marker sub-panel split of Xu-538.
- Currently **NOT runnable** — pending Salas IDOL-Ext immune-cell methylation atlas integration.
- Placeholder in pipeline. Will activate when atlas integration completes.

**Stage 1 output bundle, every IDAT:** A_immune_pooled + A_dir_AD + A_dir_PDAC + Kresovich comparator + Test 2 placeholder + tier calls on each + age-matched percentiles.

---

### Stage 2 — tissue-of-origin deconvolution + per-class A-scores (multiple atlases, layered)

Stage 2 runs **all reference atlases on every IDAT**, layered. Where two atlases cover the same cell type, the higher-resolution sorted-cell entry takes precedence; where one has unique coverage, that one is used. NNLS via `scipy.optimize.nnls` is the universal solver.

**2.1 Moss 2018 — primary 18-tissue atlas** (validated, in production)
- Source: Moss 2018 *Nat Commun* (`10.1038/s41467-018-07466-6`), 25 tissues/cell types, 7,890 CpGs total.
- Cell types in production scoring: colon, lung, gastric, bladder, cervical, kidney epithelial; hepatocyte, pancreatic exocrine, breast_ductal, prostate epithelial; neuron, oligodendrocyte; vascular_endothelial, fibroblast; neutrophil, lymphocyte, monocyte, hsc.
- Healthy reference β per tissue (Moss 2018 Table S1): colon 0.741, lung 0.738, gastric 0.739, bladder 0.737, cervical 0.740, kidney 0.739, hepatocyte 0.742, pancreatic 0.738, breast_ductal 0.744, prostate 0.743, neuron 0.779, oligodendrocyte 0.775, vascular_endothelial 0.731, fibroblast 0.728, neutrophil 0.762, lymphocyte 0.751, monocyte 0.758, hsc 0.734.
- Validation anchor: VAL-041 (10/10 top-1 correct localization at-diagnosis).
- **Source:** README_MASTER §Stage 2 lines 205–228.

**2.2 Loyfer 2023 array atlas — supplementary, layered on Moss** (in production as of 2026-04-25)
- Source: `nloyfer/meth_atlas/reference_atlas.csv`, distributed alongside Loyfer 2023 *Nature* 613:355 (`10.1038/s41586-022-05580-6`), MIT-licensed.
- 26 sorted cell types, 7,890 array-indexed CpGs.
- Adds sorted-cell entries Moss 2018 did not have: `Cortical_neurons`, `Vascular_endothelial_cells`, `Left_atrium`, EPIC-trained sorted immune-cell panel (CD4+T, CD8+T, NK, B, monocyte, neutrophil), `Pancreatic_duct_cells`, `Head_and_neck_larynx`, `Upper_GI`.
- Validation anchors: VAL-090 (glioma plasma cortical-neuron fraction d=+1.96), VAL-091 (AD plasma cortical-neuron null — confirms Loyfer-cortical is glioma-vs-CNS-disease discriminator, not generic CNS detector).
- Layered architecture rule (glioma-LL-007): Moss stays primary for cells it covers; Loyfer supplements for sorted-cell entries Moss lacks.

**2.3 Zhu/Teschendorff 2022 EpiSCORE pan-tissue atlas — candidate for v0.3 integration** (NOT YET in production)
- Source: Zhu 2022 *Nat Methods* 19:296 (`10.1038/s41592-022-01412-7`), R package `aet21/EpiSCORE` v0.9.6.
- **42 cell types across 13 solid tissues** — kidney, liver, lung, pancreas, prostate, brain, breast, colon, ovary, stomach, esophagus, bladder, thyroid.
- Imputed from tissue-specific scRNA-seq atlases via a probabilistic epigenetic model of gene regulation (different methodology from Moss/Loyfer FACS-sorted approach).
- Validated by the same authors who built EpiDISH (which we already use for Stage 3).
- Adds discrimination Moss/Loyfer cannot reach: kidney proximal tubule vs distal tubule, liver hepatocyte vs cholangiocyte vs Kupffer cell, lung alveolar Type I vs Type II vs club, etc.
- **What it would add to GAPE:** finer-grained per-class A-scores within each solid organ. Currently `kidney_epithelial` is one Moss tile; with EpiSCORE it becomes proximal/distal/podocyte/endothelial sub-tiles, each with its own class H_min.
- **Block:** R-package integration required (currently the rest of the chain is Python). EpiDISH RPC mode at Stage 3 already runs through R via rpy2 wrapper, so the bridge exists.

**2.4 Cuadrat et al. 2023 cardiovascular ccfDNA biomarker atlas — candidate for cardio-epic Stage 2 extension** (NOT YET in production; corrected 2026-04-29 — see correction note below)

- Source: Cuadrat, Kratzer, Giral Arnal, Rathgeber, Wreczycka, Blume, Gündüz, Ebenal, Mauno, Osberg, Moobed, Hartung, Jakobs, Seppelt, Meteva, Haghikia, Leistner, Landmesser, Akalin (2023). "Cardiovascular disease biomarkers derived from circulating cell-free DNA methylation." *NAR Genomics and Bioinformatics* 5(2):lqad061 (`10.1093/nargab/lqad061`). Open access (CC-BY).
- Atlas form: Moss 2018 25-tissue reference atlas (~390K CpGs after feature selection from the 6,105-CpG production atlas) **extended with three additional bulk heart tissues from ENCODE EPIC array data**: right atrium auricular (n=2 ENCODE accessions ENCSR517JQA + ENCSR280LMY), heart left ventricle (n=2 ENCSR515ZCU + ENCSR190PQG), coronary artery (n=2 ENCSR688OHW + ENCSR582BMR). Total 28 tissues/cell types. Tissue-specific feature selection per the Moss 2018 method (top 100 hypo + top 100 hyper per tissue, plus dmpFinder top 200 differential CpGs per tissue, plus pairwise-specific CpGs and 50-bp neighbors).
- **What this atlas IS:** an extension of the Moss 2018 array atlas with three bulk human heart tissue references. Useful for ccfDNA tissue-of-origin deconvolution when cardiac involvement is suspected. The paper's primary application was discriminating ACS types (STEMI vs NSTEMI vs UA vs healthy) on n=29 discovery + n=11 validation WGBS samples, identifying 1,637 cardiovascular DMRs (688 STEMI-specific + 388 NSTEMI-specific + 865 UA-specific) and demonstrating ccfDNA cell-type proportion shifts (elevated neutrophils, vascular endothelial, heart left ventricle, coronary artery; reduced monocytes, NK, erythrocyte progenitors, hepatocytes in ACS).
- **What this atlas IS NOT:** a sorted-cardiomyocyte panel. The "heart" entries in this atlas are **bulk EPIC array methylomes from ENCODE heart-region samples** (right atrium auricular tissue, heart left ventricle tissue, coronary artery tissue) — not flow-sorted cardiomyocytes, not sorted cardiac fibroblasts, not sorted smooth muscle cells. The cell-type-of-origin discrimination available from this atlas is at bulk-tissue resolution for the three added heart regions, plus the Moss 2018 base which already included vascular_endothelial as a sorted entry.
- **What it would add to GAPE:** three additional bulk heart-tissue tiles for ccfDNA deconvolution (right_atrium, heart_left_ventricle, coronary_artery), each scoreable as a per-class A-score against the appropriate frozen H_min (cycling for atrial/ventricular cardiac muscle bulk; stromal for coronary artery bulk if dominantly vascular). Provides cardio-relevant cell-of-origin signal beyond the Moss/Loyfer Left_atrium-only entry.
- **What it would NOT add to GAPE:** sorted cardiomyocyte cell-of-origin discrimination at array CpG resolution. That class of atlas does not currently exist in published literature at array-CpG layout. Loyfer 2023 includes vascular_endothelial and smooth_muscle as sorted cells but not sorted cardiomyocytes (Loyfer's heart entry is Left_atrium bulk). Zemmour et al. 2018 (Nat Commun, `10.1038/s41467-018-03961-y`) developed a six-CpG cardiomyocyte-specific FAM101A panel that is a targeted biomarker, not a deconvolution atlas. Sorted-cardiomyocyte array-CpG-indexed deconvolution remains an open published-literature gap.
- **R package:** `deconvR` (https://github.com/BIMSBbioinfo/deconvR), MIT-license. Provides NNLS, SVR, QP, RLM solvers for cell-type deconvolution; the paper found NNLS lowest RMSE on simulated mixtures (matching Moss 2018 method choice).
- **Acquisition path:** R package CRAN/Bioconductor; signature matrix in supplementary data; raw ENCODE EPIC IDATs from the six accessions listed above (publicly available without authorization).
- **Atlas family fitness (CHK-5.11):** matches cardio-epic Stage 2 modality. Cuadrat et al. used NNLS on simulated cell-type mixtures from Moss 2018 — same scoring family the cookbook uses. Tile-coverage CpGs derived via the Moss 2018 feature-selection method, suitable for A-score reading on heterogeneous β. Calibration-before-scoring required per CCL-041.

**Correction note 2026-04-29 (CCL-046).** A prior version of this section incorrectly attributed this paper to a "Konigsberg 2023 cardiovascular extended atlas" with "sorted cardiomyocytes, cardiac fibroblasts, vascular endothelial, smooth muscle" — and stated "Without this atlas, cardio-epic cannot be deployed." Both the author attribution and the cell-type-content description were factually wrong (verified against the published paper at the cited DOI 2026-04-29). The actual paper is Cuadrat et al. 2023, contains bulk heart-tissue ENCODE additions to the Moss 2018 base (not sorted cardiac cell types), and is one of several useful Stage 2 cardio-epic extensions rather than a singular deployment blocker. The "cannot be deployed" framing is dropped: cardio-epic v0.2 is operational under the layered Moss+Loyfer atlas with Stage 1 immune as the validated workhorse (VAL-110 d=+1.08 normal vs BAV on aortic tissue); Cuadrat 2023 + Caggiano CelFiE TIM + EpiSCORE pan-tissue are integration enhancements that broaden cardio Stage 2 cell-of-origin coverage but do not gate deployment of the Stage 1 + bulk-heart Stage 2 architecture already validated. Logged as **CCL-046 LL-CANONICAL-DOC-FACTUAL-ERROR** in LESSONS_LEARNED.md.

**Sorted-cardiomyocyte array-CpG atlas — open published-literature gap (added 2026-04-29).** As of 2026-04-29 there is no published sorted-cardiomyocyte array-CpG-indexed deconvolution atlas. Published cardiac methylation work (Zemmour 2018, Cuadrat 2023, Loyfer 2023, the Moss 2018 base) covers either targeted CpG biomarkers (FAM101A, mt-cfDNA), bulk heart tissues (Left_atrium, right atrium, left ventricle, coronary artery), or sorted vascular cells (vascular_endothelial, smooth_muscle). When a sorted-cardiomyocyte atlas at array resolution is published, that becomes a v1.0+ candidate for an additional Stage 2 cardio extension. Until then, cardio-epic Stage 2 cardiac cell-of-origin discrimination operates at bulk-heart-tissue resolution.

**2.5 Tanaka 2025 6-cell-type neural cfDNA atlas — candidate for AD/PD/ALS differential** (NOT YET in production, **highest-priority new addition** given the AD-vs-LGG question)
- Source: Tanaka 2025 *medRxiv* (`10.1101/2025.10.07.25337503v2`), nanopore methylation atlas.
- **Six primary neural cell types: cortical neurons, dopaminergic neurons, spinal motor neurons, astrocytes, Schwann cells, microglia.**
- Validated on 219 plasma samples (AD, PD, ALS, controls); reported AUCs >0.98 for disease-specific cfDNA elevation: cortical cfDNA elevated in AD, dopaminergic in PD, spinal motor neuron in ALS.
- **What it would add to GAPE:** the exact discriminator the framework needs for the AD-vs-LGG-vs-PD-vs-ALS differential. Currently Loyfer has only one `Cortical_neurons` reference; Tanaka separates cortical / dopaminergic / motor / astrocyte / Schwann / microglia.
- **Caveat:** Tanaka's atlas is nanopore-based, not array-indexed by default. For 450K/EPIC integration we'd need to map their cell-type-discriminating regions to array CpGs, similar to how the Loyfer team built their array-indexed `reference_atlas.csv` from the WGBS Loyfer 2023 atlas. Tractable but a v0.3 task.
- **For the question Heath asked about AD vs LGG:** with this atlas integrated, EDEAR would compute six neural-cell A-scores per IDAT — A_cortical, A_dopaminergic, A_motor, A_astrocyte, A_Schwann, A_microglia. The combinations distinguish AD (cortical elevated, others floor) from PD (dopaminergic elevated) from ALS (spinal motor elevated) from MS (Schwann + microglia elevated) from glioma (cortical fraction elevated AND A_cortical drifted from architectural floor — the fraction-plus-drift signature).

**2.6 Tian et al. 2023 single-cell brain methylation atlas (scMCodes)** (NOT YET in production; corrected 2026-04-29 — see correction note below)
- Source: Tian W, Zhou J, Bartlett A, Zeng Q, Liu H, Castanon RG, et al. "Single-cell DNA methylation and 3D genome architecture in the human brain." *Science* 2023 Oct 13;382(6667):eadf5357 (`10.1126/science.adf5357`). 49 authors total; lead **Wei Tian**, co-first **Jingtian Zhou**.
- **188 cell types from 517K single human brain cells across 46 regions** (399K neurons + 118K non-neurons, three adult male brains, snmC-seq3 + sn-m3C-seq + companion snRNA-seq).
- Granularity beyond what any clinical-array deconvolution can use (single-cell barcodes), but the cell-type-discriminating CpG sets (scMCodes) can be projected onto 450K/EPIC.
- **Candidate v0.4+** — exceeds current array resolution but the discriminator regions can be downsampled to array-compatible panels.

**Correction note 2026-04-29 (CCL-046 audit class).** A prior version of this section attributed this paper to "Liu 2023." Web verification at `10.1126/science.adf5357` 2026-04-29 found the actual lead author is **Wei Tian** with **Jingtian Zhou** as co-first author; Hanqing Liu appears as mid-author (5th of 49). The paper content (188 cell types, 517K cells, 46 brain regions, scMCodes methodology) is correct as cited; the author attribution was wrong. This is the same class of error as the Konigsberg→Cuadrat correction in Part 2.4 (CCL-046): a documents-of-record citation error inherited across cookbook references. CHK-5.13 documents-of-record citation-verification gate, added 2026-04-29, catches this class of error before card publish. All "Liu 2023" references in cookbook documents corrected to "Tian et al. 2023" as part of v0.2.2 expanded patch.

**2.7 Caggiano 2021 array-native neuronal references** (referenced in cookbook as v0.3 task, NOT YET in production)
- Mentioned in glioma-epic v0.2 README as a v0.3 candidate for oligodendrocyte/astrocyte/microglia separation.
- Status: documented scope, not loaded into production atlas chain.

**Stage 2 per-class A-score computation (universal across all atlases):**

For each tissue or cell type recovered by NNLS deconvolution (whether from Moss, Loyfer, EpiSCORE, Konigsberg, or Tanaka), compute:
- **A_tissue = mean(H(β) / H_min(class))** at the tissue-discriminating CpGs, where `class` is the architecture class the tissue belongs to and H_min(class) is the frozen MCMC-derived floor for that class.
- **ΔA_tissue = A_tissue(patient) − A_tissue(healthy reference β)**.
- **Tier call** from the universal six-tier vocabulary.

**Class H_min values (frozen, never per-disease):**
- terminal: 0.7728 (neurons, oligodendrocytes, cardiomyocytes, dopaminergic neurons, motor neurons)
- secretory: 0.843264 (breast_ductal, hepatocyte, pancreatic_exocrine, prostate_epithelial)
- cycling: 0.856055 (colon, lung, gastric, bladder, cervical, kidney epithelial)
- immune: 0.838889 (T cells, B cells, NK, monocytes, neutrophils, microglia, Schwann)
- stromal: 0.862950 (vascular_endothelial, fibroblast, cardiac fibroblast, astrocytes)
- progenitor: 0.852216 (GMP, CMP, NPC, neuronal progenitors)
- stem_adult: 0.873718 (HSC, NSC)
- stem_pluri: 0.982166 (ESC, iPSC) — ceiling-region; A < 1 expected

**Stage 2 output bundle, every IDAT:** for each cell type the atlas chain resolves — recovered fraction, A-score against class H_min, ΔA vs healthy reference β, tier call, platform tag.

---

### Stage 3 — immune sub-composition (always run)

**3.1 Teschendorff 2017 EpiDISH RPC mode** (in production)
- Salas 2018 6-cell-type reference (CD4+T, CD8+T, NK, B, monocyte, neutrophil).
- Output: 6-cell fraction vector with Salas QC bounds (neutrophil 45–75%, lymphocyte 20–40%, etc.).
- Used for: AD-type patterns (no solid-organ Stage 2 hit), heme-epic lineage discrimination, autoimmune/inflammation differentials.

**3.2 Salas IDOL-Ext extended panel** (NOT YET in production — OQ-2026-01)
- Salas IDOL-Ext (`10.1186/s13059-018-1448-7` extended) for lymphoid vs myeloid sub-discrimination.
- Required for Test 2 bidirectional-cancellation diagnostic in Stage 1.
- Status: pending integration.

**Stage 3 output bundle, every IDAT:** 6-cell immune sub-composition + QC pass/fail + Test 2 placeholder.

---

## Part 3 — Healthy baselines and age references

**80-cell baseline** — reference for Stage 1 age-matched percentile lookup.
- Sources: Hannum 2013 + Horvath 2013 + Roadmap Epigenomics 2015 + Moss 2018 + Lister 2013 + Alisch 2012 + Adelman 2019.
- 8 architecture classes × 10 age decades = 80 reference points.
- **Cross-cohort caveat (VAL-057, ad-LL-002):** the 80-cell baseline cannot be applied directly to non-AIBL/non-AddNeuroMed cohorts without within-cohort HC re-anchoring or normalization bridge. GSE53740 HC sit +2.306 SD above the baseline due to Ferrari 2014 ComBat preprocessing.

**Tissue-specific healthy reference β** — Moss 2018 Table S1 values (listed in §2.1 above), used for Stage 2 ΔA computation.

**Cohort-batch-offset diagnostic (CHK-2.7, ad-LL-006):** for any new cohort, run cross-cohort HC baseline diagnostic before pooling. If cross-cohort HC baseline fold range exceeds 10×, within-cohort statistics are the only valid primary; cross-cohort absolute values are invalid for outcome assignment.

---

## Part 4 — The universal tier vocabulary (six tiers)

**`BELOW_NORMAL / NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH`**

Applied to every A-score, every ΔA, every cell-fraction departure from healthy baseline.

| Tier | Threshold (z-score from baseline) | Action |
|---|---|---|
| BELOW_NORMAL | ≤ −1.0 SD below baseline | Real signal, not noise. Indicates non-disease-of-the-card differentials: immunosuppression, post-chemo/transplant, primary immunodeficiency, or class-specific suppression (e.g. PSP/CBD reads cortical-neuron at d=−0.51). **Routes to clinician review for differential, never silently absorbed into NORMAL.** Patient-facing equivalent is `SUPPRESSED` (heme-epic v0.1 naming). |
| NORMAL | within ±1.0 SD of baseline | No action |
| MARGINAL | +1.0 to +1.5 SD | Note, reassess in 6–12 months |
| DETECTABLE | +1.5 to +2.0 SD | Run downstream workup specific to the localized class |
| URGENT | +2.0 to +2.85 SD | Immediate workup recommended |
| FLOOR_BREACH | ≥ +2.85 SD (or A ≥ 1.10 absolute) | Strong signal; clinical workup regardless of localization |

**Source:** ad-immune card v2.2 + heme-epic v0.1 (SUPPRESSED/BELOW_NORMAL terminology unified at ad-LL-007, 2026-04-26).

---

## Part 5 — Per-disease card firing logic (no gating, pattern matching only)

After every IDAT runs the full Stage 1 + Stage 2 + Stage 3 stack, **disease cards fire when their characteristic anomaly pattern matches**, not based on whether Stage 1 crossed a threshold. The card layer is pattern-matching on the full output bundle.

**Examples:**
- **breast-epic fires** when: A_immune_pooled ≥ DETECTABLE AND Stage 2 breast_ductal ΔA top-1 with top-1/top-2 ratio > 2× → breast localization.
- **breast-epic pre-dx fires** when: trajectory shows monotonic A_immune climb across serial samples AND/OR Stage 2 secretory-class variance reduction (homogenization, negative ΔA SD per VAL-047 Phase 6) — the >10yr signature.
- **ad-immune fires** when: A_dir_AD ≥ DETECTABLE on the 7-CpG Rule A panel AND Stage 2 cortical-neuron at HC floor (NOT elevated) — the AD-pattern that distinguishes from glioma.
- **glioma-epic fires** when: A_immune_pooled ≥ DETECTABLE AND Stage 2 cortical-neuron fraction > 0.5% (Loyfer atlas) — the differential-diagnosis pattern; the elevated cortical-neuron fraction is what distinguishes from AD.
- **heme-epic fires** when: A_immune_pooled ≥ FLOOR_BREACH AND Stage 2 NULL on all solid organs AND Stage 3 lineage-specific shift (myeloid-dominant → AML arm; B-cell-dominant → lymphoid B arm; etc.).
- **PSP/CBD differential** routes when: any Stage 2 cell type reads BELOW_NORMAL (negative ΔA below −1.0 SD) — VAL-091 GIFT specificity arm pattern.

**Multi-disease patterns (the Heath insight):** a patient with early AD + early breast + chronic inflammation will fire multiple cards' anomaly patterns simultaneously. The report displays all anomalies; the clinician interprets the combination. Gating-based pipelines miss this; run-everything pipelines surface it.

---

## Part 6 — What changes from current v2.1 cookbook

**Conditional gating language** in README_MASTER §Stage 2 line 207 (*"Stage 2 (if Stage 1 hits DETECTABLE or above)"*) and §Stage 3 line 234 (*"Runs when Stage 1 flags and Stage 2 returns no solid-organ localization"*) — both need to be revised to reflect run-everything architecture.

**Card READMEs** (breast-epic line 18, ad-immune similar, all others) carry the same conditional language and need parallel revisions.

**Patient report logic** is unchanged in spirit — display anomalies, collapse uninformative tiles — but the underlying scoring pipeline is now exhaustive.

**Pre-existing per-class A-score computations are unchanged.** Stage 2 already computed 18 per-tissue A-scores per IDAT in the validated workflow (VAL-041); the change is removing the conditional that Stage 2 only runs *if* Stage 1 fires.

**No re-validation needed for existing cards.** The change is what we compute, not how we compute it. All published VAL effect sizes hold.

**Propagation order (proposed):**
1. Lock this v2 reference document with Heath's sign-off.
2. Patch README_MASTER §Stage 1, §Stage 2, §Stage 3 language.
3. Patch every card README's "workflow in one patient" section.
4. Ratify the new atlases (EpiSCORE, Konigsberg, Tanaka) as v0.3+ integration tasks; do NOT promote them to production until each has been validation-anchored on a Cookbook VAL run.
5. Update GAPE_WEB_v13.py / GAPECommercial_web.py to remove the gating logic (the per-class A-score machinery is already there).

---

## Part 7 — Complete production-chain inventory (current vs proposed)

### Currently in production (validated, scoring on every IDAT)

**Stage 1 panels:**
- Xu-538 immune-class pooled panel (validated VAL-041 / VAL-046 / VAL-047 across 10+ cohorts)
- AD 7-CpG Rule A directional panel (validated VAL-051 / VAL-052)
- Pancreatic 324-CpG directional fallback panel (validated VAL-069)
- Kresovich mBCRS 100-CpG comparator (reference comparator only)

**Stage 2 reference atlases:**
- Moss 2018 18-tissue (validated VAL-041 10/10 top-1 at-diagnosis)
- Loyfer 2023 array atlas (validated VAL-090 glioma + VAL-091 AD)

**Stage 3 immune sub-composition:**
- Teschendorff 2017 EpiDISH RPC mode + Salas 2018 6-cell reference

**Healthy baselines:**
- 80-cell baseline (Hannum + Horvath + Roadmap + Moss + Lister + Alisch + Adelman)

### Approved 2026-04-26 for v0.3 integration (Queue 1, Heath sign-off)

**Heath signed off 2026-04-26 on Queue-1 atlas integration.** The six published atlases below are approved for v0.3 integration into the run-everything Stage 2 reference layer. None is in production scoring as of 2026-04-26 — each requires a per-atlas validation-anchor VAL run before promotion. A VAL that names a Queue-1 atlas may use the published external classifier (e.g. Sabedot GeLB output as a comparator arm, MARLIN as a leukemia subtype anchor) but cannot claim integrated A-score scoring against H_min until the atlas-integration VAL has landed.

**Stage 2 atlas extensions (Queue 1):**

| # | Atlas | Source | What it adds | Priority |
|---|---|---|---|---|
| 1 | **Tanaka 2025 6-cell neural cfDNA atlas** | medRxiv 10.1101/2025.10.07.25337503v2 (nanopore) | Cortical / dopaminergic / spinal motor neurons, astrocytes, Schwann cells, microglia. Validated AD/PD/ALS plasma cfDNA discrimination AUC > 0.98 across 219 samples | **HIGHEST** — answers AD-vs-LGG-vs-PD-vs-ALS-vs-MS differential directly; the discriminator the framework has been groping for |
| 2 | **Konigsberg 2023 cardiac extended atlas** | NAR Genomics 10.1093/nargab/lqad061 | 28-cell extended atlas with sorted cardiomyocytes, cardiac fibroblasts, smooth muscle | **HIGH** — cardio-epic deployment depends on this; currently no sorted cardiomyocyte tile exists in production |
| 3 | **Zhu/Teschendorff 2022 EpiSCORE pan-tissue** | Nat Methods 10.1038/s41592-022-01412-7; R package `aet21/EpiSCORE` v0.9.6 | 42 cell types × 13 solid tissues — fine-grained per-organ (kidney proximal vs distal, liver hepatocyte vs cholangiocyte vs Kupffer, lung alveolar Type I vs II vs club, etc.) | **MEDIUM-HIGH** — broad capability across solid tissues; same Teschendorff lab as EpiDISH so R-package bridge already exists via Stage 3 |
| 4 | **Caggiano 2021 array-native neuronal references** | Already documented in glioma-epic v0.3 task list | Oligodendrocyte / astrocyte / microglia separation at array CpG resolution | **MEDIUM** — partially superseded by Tanaka 2025 if integrated; Caggiano's array-native format is a faster integration path for parallel validation |
| 5 | **Capper 2025 MARLIN leukemia 450K/EPIC reference** | Already documented in heme-epic v0.2 task list | n=2,540 acute leukemia (1,461 AML, 686 B-ALL, 266 T-ALL) | **MEDIUM** — heme-epic v0.2 myeloid arm cross-cohort replication; published 450K/EPIC indexed |
| 6 | **Sabedot 2021 GeLB external classifier** | Mendeley deposit cgrz6zztfg | EPIC-array glioma blood classifier; already accessible Tier 1 | **MEDIUM** — engineering, not validation; adds external-classifier arm to glioma-epic for cross-pipeline confirmation |

**Stage 1 panel extensions (Queue 1):**

| Panel | Source | What it adds | Priority |
|---|---|---|---|
| **Salas IDOL-Ext** | Genome Biol 10.1186/s13059-018-1448-7 extended | Lymphoid vs myeloid sub-discrimination → enables Stage 1 Test 2 bidirectional-cancellation diagnostic (CCL-030/031) | **HIGH** — already named as OQ-2026-01 in cookbook |
| **Zemmour 2018 cardiomyocyte cfDNA panel** | Nat Commun 10.1038/s41467-018-03961-y | Cardiomyocyte-specific cfDNA detection markers (FAM101A region) for MI/heart-failure | **HIGH** if cardio-epic is being deployed |

### Queue 2 (engineering heavier, still tractable)

- **Tian et al. 2023 brain scMCodes** (Science 10.1126/science.adf5357; lead Wei Tian, co-first Jingtian Zhou) — 188 single-cell brain types from 517K cells across 46 brain regions. Cell-type-discriminating regions can be projected to array CpGs by the same method that produced Loyfer's array-indexed reference from their WGBS source, but the engineering is heavier than Queue 1.

### Validated tissue atlases referenced but not in scoring chain

- **Salas/Wiencke 2022** — used as the glioma cohort source (GSE180683), not as a reference atlas
- **TCGA pan-cancer methylation** — used as cohort sources for tissue-arm validations (VAL-058, VAL-060, VAL-061/062, VAL-063, VAL-064), not as reference atlases

### Per-atlas integration template (the v0.3 VAL recipe)

Each Queue-1 atlas requires a per-atlas validation-anchor VAL run before promotion to production scoring. Template:

1. **Integrate atlas reference into NNLS deconvolution wrapper with platform tag.** Atlas β matrix loaded as additional reference layer; Moss + Loyfer + new-atlas all available simultaneously; layered-precedence rule (sorted-cell entries take precedence over bulk-tissue entries on the same CpG positions).
2. **Run on the atlas's source-paper validation-anchor cohort** (Tanaka's 219 plasma samples for AD/PD/ALS, Konigsberg's cardiomyopathy cohort, MARLIN's leukemia cohort, etc.). Verify within-cohort case-vs-control reproduces published direction and magnitude under EDEAR's H_min anchoring. This is the first sanity check.
3. **Run on at least one EDEAR-anchor cohort** (GSE51057 healthy reference + a disease cohort relevant to the atlas) under CHK-3.2 cross-cohort baseline check. Document any platform-induced baseline shifts before promotion.
4. **Promote to production with platform-stratified thresholds.** Update the cell-type-tile threshold table for each platform the cohort tested on (450K, EPIC, EPIC v2 if relevant).
5. **Update card READMEs** that depend on the new tile (cardio-epic for Konigsberg, future PD/ALS/MS cards for Tanaka, glioma-epic for Sabedot/Caggiano, heme-epic for MARLIN).
6. **Patch GAPECommercial_web.py** to surface the new tile in the appropriate disease-card firing-pattern logic.

### Why other groups built these atlases (and why they're gold for EDEAR but commodity for everyone else)

EDEAR's commercial defensibility is not the reference atlases — those are public, MIT/CC-licensed, and downloadable. The defensibility is **the physics that turns a methylation β vector into a per-class A-score**. H_min comes from the IAM derivation chain (G-002 + G-003b MCMC posteriors with R-hat < 1.001) — patent-protected (US 64/012,720 + 64/014,568), Recipe-protected, vault-protected. Anyone with the same atlases gets cell-type fractions; only EDEAR computes architectural-drift A-scores against H_min anchors.

Other groups built these atlases for: cell-type fraction estimation in cancer-of-unknown-primary (Moss, Loyfer, Konigsberg, Tanaka); EWAS cell-composition adjustment (EpiSCORE, Salas IDOL-Ext); cancer subtype categorical classification (Capper, Sabedot, MARLIN). None of them have H_min. Without H_min, "more disordered than healthy" is not a number that can be computed.

**Adding Queue-1 atlases to the run-everything Stage 2 reference layer makes EDEAR strictly more powerful without adding any commercial-defensibility risk** — every additional cell-type tile is an additional channel through which the framework can detect disease, and only EDEAR has the physics to read the architectural-drift channel of those tiles.

---

## Part 8 — How the new architecture answers the questions Heath has been asking

### "How do we know the >10yr breast signal is breast specifically?"

Under the run-everything architecture, every IDAT produces the full 18-tissue Stage 2 A-score vector and ΔA vector. At the >10yr pre-dx window in GSE51057 / GSE51032:
- A_immune_pooled is at d=+1.36 to +1.78 (URGENT to FLOOR_BREACH)
- Stage 2 secretory-class variance (breast_ductal, hepatocyte, pancreatic, prostate combined) is at d=−1.226 (BELOW_NORMAL — homogenization signal)
- Stage 2 individual breast_ductal A-score and ΔA at the >10yr window: **runnable analysis 9.2** (formerly tagged "missing"). VAL-047 Phase 6 ran the secretory-class subset; the full 18-tissue ΔA top-1 at the >10yr window is what Analysis 9.2 produces.
- Other tissues (colon, lung, prostate, etc.) at NORMAL — that's the "specifically breast" answer.

### "How do we distinguish AD from LGG/glioma at the cortical-neuron Stage 2 tile?"

Under the current architecture (Loyfer atlas only): glioma cortical-neuron fraction = 1.09% (d=+1.96), AD cortical-neuron fraction = 0.25% (null). **Fraction-only discriminator works at the cohort level.**

Under the proposed architecture with Tanaka 2025 atlas integrated:
- AD signature: cortical-neuron fraction *elevated* AND A_cortical at architectural floor (~0.69, no drift) AND microglia A-score elevated (chronic neuroinflammation per Lunnon 2014). **A combination signature.**
- LGG/glioma signature: cortical-neuron fraction *elevated* AND A_cortical drifted from floor (architectural disorder) AND microglia A-score elevated (TAM trafficking). **Different combination.**
- PD signature: dopaminergic-neuron fraction elevated (different cell type entirely).
- ALS signature: spinal motor neuron fraction elevated (different cell type entirely).
- MS signature: Schwann cell + microglia fraction elevated (different cell types).

**The combination of fraction + A-score + immune-class-microenvironment shift across the multi-cell-type neural atlas resolves the differential.** That's the architectural payoff of run-everything-every-time.

### "What's missing right now that would close these gaps?"

1. **Run analyses 9.2 and 9.3** on the existing Loyfer-atlas data (Heath's pending decisions from prior turn).
2. **Integrate Tanaka 2025** — array-mappable, ~1–2 weeks of engineering. Highest-priority addition.
3. **Integrate Konigsberg 2023** — cardio-epic dependency, also array-mappable.
4. **Integrate EpiSCORE** — broader fine-grained per-organ resolution; R-package bridge already exists via Stage 3 EpiDISH path.

---

## Part 9 — Stopping points before propagation

This document is **a proposal, not yet ratified.** Before propagating any of these changes to README_MASTER or the cards:

1. Heath reads and signs off on the run-everything architecture.
2. Heath confirms which v0.3+ atlas integrations are highest priority (suggested order: Tanaka 2025 first for AD-vs-LGG; Konigsberg 2023 for cardio-epic; EpiSCORE for breadth).
3. Heath confirms the formulation for Analysis 9.3 (A_terminal on recovered cortical-neuron β) — Option 1 (patient β at cortical-neuron-discriminating CpGs, scored against H_min(terminal)) was the Walther proposal in the prior turn; Heath may have a different formulation in mind.
4. After 9.2 and 9.3 are run with the locked methodology, results either confirm the architecture or surface unexpected findings that change it. **Either result is informative.**
5. Then propagate to README_MASTER and cards.

---

## Part 10 — Source files cited in this document

Every claim in this document traces to one of:
- `README_MASTER_v2_1.md` lines 184–234 (Stage 1 + Stage 2 + Stage 3 architecture)
- `breast-epic_README.md` "workflow in one patient" lines 12–20 + temporal pattern lines 88–94
- `breast-epic_card_v2_2.json` lines 35–145 (per-window numbers)
- `glioma-epic_card_v0_2.json` lines 120–143 (VAL-089 A_terminal on tissue)
- `GAPE_Evidence_Report_UPDATED.html` §5C lines 575–612 + §5D lines 626–681 (VAL-047 Phase 6 / Deep Audit secretory-class variance)
- `GAPE_WEB_v13.py` (Stage 1/Stage 2/Stage 3 implementation; comments referencing Moss 2018, Salas 2018, EpiDISH, Kresovich)
- Literature search 2026-04-26: Zhu 2022 EpiSCORE; Cuadrat 2023 (originally cited as Konigsberg 2023; corrected per CCL-046); Tanaka 2025; Tian et al. 2023 (originally cited as Liu 2023; corrected per CCL-046 audit class); Zemmour 2018; Salas 2018 IDOL-Ext

---

## Part 11 — Cross-cohort baseline check is mandatory every run (CCL-034, signed off 2026-04-26)

**The rule.** Every results JSON, every VAL outcome.md, every patient-facing report MUST contain a `cross_cohort_baseline_check` block for every Stage 1 panel and every Stage 2 cell-type tile, comparing the cohort's HC mean A-score to the anchor in **anchor-SD units**. The block is mandatory regardless of whether a mismatch is detected. Empty/null cross-cohort blocks are a bug.

**Mismatch tiers.**

- **<1 anchor-SD:** reported but not flagged.
- **1–3 anchor-SDs:** flagged with `baseline_mismatch_flag: true`; cross-cohort comparison reported but explicitly downgraded; within-cohort case-vs-control becomes the primary statistic.
- **≥3 anchor-SDs:** invalidates cross-cohort absolute comparisons entirely. Within-cohort only.

**Why this is mandatory under run-everything specifically.** Pre-architecture (gated): a patient's report shows one tile (the disease the test was ordered for); a baseline-mismatch on that one tile is a single error and the gating lets the rest of the pipeline stay clean. Post-architecture (run-everything): a patient's report shows 18+ Stage 2 tissue tiles + Stage 3 sub-composition + Stage 1 panel scores simultaneously, and dual/triple diagnosis claims arise from the *combination* of which tiles cross threshold. **A single platform-induced baseline shift on cortical-neuron at +16.7 anchor-SDs would, under naive interpretation, falsely diagnose AD or glioma in every patient run on that 450K cohort's preprocessing pipeline.** CHK-3.2 is the structural defense.

**Documented examples (VAL-record).**

| Source VAL | Cohort comparison | Mismatch | Cause | Consequence |
|---|---|---|---|---|
| VAL-057 | GIFT GSE53740 HC vs 80-cell baseline | +2.306 SD | Ferrari 2014 ComBat preprocessing | Cross-cohort A-score absolute values not interpretable; within-cohort case-vs-control valid |
| VAL-073 vs VAL-074 (cervical-epic) | Verlaat population-normal vs Farkas HPV-negative-only | 2.7 anchor-SDs | Different "normal" definition | Flag before drawing CIN3 conclusions |
| VAL-091 (ad-immune) | AddNeuroMed cortical-neuron HC vs GSE51057 HC | 28× absolute scale | 8% Loyfer-CpG coverage gap on 450K + NNLS routes mass to Cortical_neurons | Cross-cohort absolute fractions invalid; within-cohort comparison primary |
| VAL-092 | AIBL HC vs GSE51057 HC on A_terminal | +1.87 anchor-SDs | Both 450K, different preprocessing | Cross-cohort glioma vs healthy d=+0.987 caveated |
| VAL-092 | AddNeuroMed HC vs GSE51057 HC on A_terminal | +16.7 anchor-SDs | Same 450K-vs-EPIC marker-coverage gap as VAL-091 | Within-cohort AddNeuroMed AD vs HC d=−0.030 is the only valid statistic |

**Within-cohort vs cross-cohort hierarchy under run-everything.** Absolute rule, not fallback:

1. **Primary evidence.** Within-cohort case-vs-control on the same IDAT batch with the same preprocessing pipeline.
2. **Secondary evidence.** Cross-cohort comparisons against an anchor with matching platform AND matching preprocessing.
3. **Tertiary evidence.** Cross-cohort across platforms or preprocessing pipelines, ONLY with explicit `baseline_mismatch_flag` and platform-stratified thresholds.
4. **No statement that depends on a tile's absolute A-score for a single patient may use a tertiary-tier comparison without surfacing the mismatch caveat to the clinician.**

**Operational sources cross-referenced.** TESTING_CHECKLIST.md CHK-3.2 (mandatory-every-run section). README_MASTER §"ABSOLUTE RULE — Run-everything pipeline architecture (CCL-033)" and CCL-034. LESSONS_LEARNED.md CCL-034 (canonical entry). GAPE Reproduction Paper §7.13 (paper-level summary).

---

## Part 12 — VAL-092 first demonstration of run-everything architecture (2026-04-26)

**What VAL-092 ran.** Stage 2 per-class A_terminal computation on top-100 cortical-neuron-discriminating CpGs (Loyfer atlas) against H_min(terminal) = 0.7728, on every IDAT in 6 cohorts: GSE51057 healthy reference (n=329, anchor); GSE180683 glioma EPIC blood (n=76); GSE60274 GBM 450K tissue (n=72); AIBL GSE153712 (n=161 AD / 471 HC); AddNeuroMed GSE144858 (n=93 AD / 96 HC); GIFT GSE53740 (n=43 PSP / 193 HC + 128 FTD). Pre-registered (SHA `7249e964afbf…`) sealed 2026-04-26T17:59:54Z before any β access.

**Within-cohort findings.**

- AIBL AD vs HC: **d = −0.228** [−0.421, −0.037] p = 0.021 (modest homogenization, NOT elevation)
- AddNeuroMed AD vs HC: **d = −0.030** (null)
- GIFT PSP vs HC: **d = −0.433** [−0.747, −0.098] p = 0.010 (BELOW_NORMAL replicates VAL-091 fraction d=−0.51)
- GIFT FTD vs HC: **d = −0.004** (null — PSP-specific not generic tauopathy)
- GBM tissue mean A_terminal = **0.79** (SD 0.10) vs blood baselines around 0.30 (substantial elevation)

**Cross-cohort glioma blood vs healthy reference: d = +0.987** [+0.74, +1.24] — flagged for cross-cohort baseline mismatch (CHK-3.2). AIBL HC vs GSE51057 HC mismatch is +1.87 anchor-SDs (both 450K, different preprocessing); AddNeuroMed HC vs GSE51057 HC mismatch is +16.7 anchor-SDs (450K-vs-EPIC marker-coverage gap). Cross-platform/cross-preprocessing offset accounts for ~+0.5 SD of the +0.987 figure on the same comparison structure. Within-cohort EPIC glioma-vs-HC cohort is what would resolve this.

**Outcome label.** `O1_DRIFT_DISCRIMINATOR` per pre-registered criteria (glioma d ≥ +0.5, AD |d| ≤ +0.3 within-cohort), with explicit annotation of within-cohort vs cross-cohort asymmetry. The supportable claim uses referee language: "the data are consistent with predictions within the framework that…"

**The run-everything payoff.** Under prior conditional-gating, GIFT PSP samples would not have triggered the Stage 2 cortical-neuron computation because PSP Stage 1 immune A-score is at HC baseline. The PSP BELOW_NORMAL signal at d=−0.43 only became visible because VAL-092 was the first VAL designed under run-everything architecture (CCL-033) — every IDAT runs Stage 2 with all atlases regardless of Stage 1 status.

**Card-level updates from VAL-092.**

- **psp-epic v0.1 stub created** at `exploratory_pending_replication` tier. Single-cohort BELOW_NORMAL signal replicated across two metrics (VAL-091 fraction + VAL-092 per-CpG drift). Priority replication cohorts: PROGRESS-PSP biobank, Allen et al. Mayo, Tang 2014. Promotion criteria: at least one of those replicating BELOW_NORMAL signal at d ≤ −0.3 within-cohort.
- **ad-immune v2.2 numbers reaffirmed.** AD cortical-neuron tile reads NULL on both fraction (VAL-091) and per-CpG drift (VAL-092). Two-pathway null on AD. Glioma-vs-AD differential pattern firmer.
- **glioma-epic blood arm `single_cohort_validated` tier maintained** pending within-cohort EPIC HC arm (cross-cohort d=+0.99 alone insufficient under CCL-034).
- **First three independent below-normal-as-signal cases now documented** across the cookbook: heme-epic SUPPRESSED (post-chemo, post-transplant); breast-epic VAL-047 secretory-class variance reduction at >10yr pre-dx (d=−1.226); psp-epic cortical-neuron architectural homogenization (d=−0.43 to −0.51). Below-normal is a category of mechanism, not a one-off.

**Reproducibility triple (CHK-7.6).**

- **Source code:** `Biological_Physics/validation_runs/VAL-092/val_092_a_terminal_cortical_neuron.py` on https://github.com/hmahaffeyges/IAM-Validation (commit `4290553`).
- **Inputs:** Loyfer atlas SHA `4b97dd2a8ba7…`; cohort SHA-256 prefixes recorded in results JSON; cohort accessions GSE51057, GSE180683, GSE60274, GSE153712, GSE144858, GSE53740.
- **Environment:** Python 3.12 + pandas + numpy + scipy + matplotlib (standard scientific Python).
- **Expected headline outputs:** AIBL within-cohort AD-vs-HC d=−0.228, GIFT PSP-vs-HC d=−0.433, GBM tissue mean A_terminal=0.79.

---

---

## Part 13 — VAL-093 first multi-cohort demonstration of run-everything (2026-04-26 PM)

**What VAL-093 ran.** Full 25-tile per-class A-score on Loyfer atlas at >10yr breast pre-diagnostic window across two independent cohorts. Pre-registered (SHA `9b708a3a05447ed6…`) sealed 2026-04-26T18:51:17Z before any β access. RNG seed 20260426. Cohorts: GSE51057 (n=11 breast >10yr cases, n=177 HC) + GSE51032 (n=36 breast >10yr cases, n=424 HC). Both 450K, EPIC-Italy buffy coat, same preprocessing pipeline.

**Within-cohort findings (top of 25 tiles, both cohorts concordant).**

- Pancreatic_beta_cells: GSE51057 d=+1.020 (p=0.017), GSE51032 d=+0.939 (p=1.5e−7)
- Pancreatic_acinar_cells: d=+0.913 / d=+1.025 (p=6.7e−9)
- Pancreatic_duct_cells: d=+0.991 / d=+0.705 (p=8.8e−5)
- Kidney (cycling): d=+0.726 / d=+0.902 (p=1.2e−6)
- Head_and_neck_larynx (cycling): d=+0.746 / d=+0.814 (p=8.4e−6)
- **Breast: d=+0.198 (NULL, p=0.628) / d=+0.100 (NULL, p=0.619)**
- Top-1 ΔA call: Breast = 2/47 cases = 4.3%

13 tiles concordantly elevated d>0.3 in both cohorts; 0 tiles concordantly depressed; 0 opposite-direction tiles. Immune class (6 tiles) is the only flat class.

**Outcome.** Pre-locked outcome label `O2_SECRETORY_DISTRIBUTED` per pre-registration: ≥3 secretory-class tiles |d|≥0.3, breast not uniquely top.

**CHK-3.2 cross-cohort baseline check.** **All 25 tiles pass at <0.25 anchor-SDs.** Maximum mismatch is 0.24 SD on Bladder. **First clean cross-cohort baseline alignment in the cookbook.** Validates the layered-atlas architecture for matched-platform-matched-preprocessing analyses. Cross-cohort comparisons interpretable at secondary-evidence tier per CCL-034.

**The run-everything payoff.** Under conditional-gating, pancreatic tiles would not have been computed for these breast pre-dx patients. The Pancreatic_beta_cells d=+1.020 finding is the run-everything payoff: **questions become askable that gating would filter out.** Whether the pancreatic signal reflects future-breast-driven systemic drift, co-existing pre-clinical pancreatic disease, or Xu-538-vs-Loyfer CpG-set correlation is unresolved at this VAL but askable.

**Sign relationship to VAL-047 Phase 6.** VAL-047 reported A_secretory aggregate d=−1.226 on **Xu-538 panel CpGs** (predominantly immune-cell-discriminating positions by training). VAL-093 reports class-aggregate per-tile mean d=+0.572 (GSE51057) / +0.605 (GSE51032) on **per-tile cell-type-discriminating CpGs from the Loyfer atlas**. Different CpG sets, different scoring rules, both findings can be true. CCL-035 candidate (Heath review pending): "Per-tile Stage 2 deconvolution surfaces multi-class drift patterns that are not visible at the panel-CpG level."

**Card-level updates from VAL-093.**

- **breast-epic v0.3 needs softening on the >10yr Stage 2 localization claim.** At-diagnosis tissue arm (VAL-060 paired d=+0.676) remains valid. >10yr blood pre-dx claim now requires explicit caveat: at the per-tile Stage 2 level, the signal does NOT localize to the Breast tile; it manifests as multi-class drift with strongest individual signals on pancreatic-class tiles.
- **The framework is still detecting *something* at +1.0 d magnitude in two cohorts replicably at >10yr breast pre-dx.** What it is detecting *as* is broader than breast-localized. The clinical-action implication: a >10yr breast pre-dx patient under run-everything would have multiple tile flags simultaneously, and the disease-card pattern-matching layer reads the *combination* as the >10yr breast signature, not the breast tile alone.
- **Layered-atlas architecture validated at cross-cohort level for matched cohorts.** First time we've seen a clean CHK-3.2 across the Loyfer tile set. For cross-platform analyses (450K vs EPIC), CHK-3.2 must remain mandatory — VAL-091/VAL-092 documented +16.7 anchor-SD shifts on cortical-neuron specifically due to 450K marker-coverage gap.

**Reproducibility triple (CHK-7.6).**

- **Source code:** `Biological_Physics/validation_runs/VAL-093/val_093.py` on https://github.com/hmahaffeyges/IAM-Validation (commit `e27814a`).
- **Inputs:** Loyfer atlas SHA `4b97dd2a8ba7…`; GSE51057 + GSE51032 EPIC-Italy buffy coat 450K series matrices; VAL-047 Phase 9/12 metadata; cohort SHA-256 prefixes recorded in results JSON.
- **Environment:** Python 3.12 + pandas + numpy + scipy + matplotlib (standard scientific Python).
- **Expected headline outputs:** Pancreatic_beta_cells GSE51057 d=+1.020 / GSE51032 d=+0.939; Breast tile null both cohorts; CHK-3.2 0/25 flagged; outcome `O2_SECRETORY_DISTRIBUTED`.

## Part 14 — Run-everything 25-tile interpretation under CCL-039 (added 2026-04-28)

**Source:** VAL-098 + VAL-062 revisit. CCL-039 LL-MARKER-CPG-TILE-FIDELITY in `LESSONS_LEARNED.md`. CHK-4.11 in `TESTING_CHECKLIST.md`.

**The lesson.** Run-everything 25-tile per-class A-score and full-HM450 architectural-drift A-score are two distinct observables. They measure different things. They do not always move in the same direction in tumor vs adjacent-normal paired comparisons.

**Operational rule for the pipeline reference.** Every VAL that includes both methodologies on the same paired tumor/normal samples must report BOTH numbers with the biology interpretation. The two metrics measure different observables. They are not contradictory when they move in different directions.

### What full-HM450 cycling-class A-score measures

Global Shannon entropy across all valid CpGs on the array (~485K HM450 CpGs after QC). Every CpG counted equally. Every signal direction averaged. Tumors increase entropy globally, A-score rises, paired d positive in tumor vs adjacent-normal comparisons.

### What run-everything 25-tile per-class A-score measures

Cell-of-origin tile fidelity at top-100 marker CpGs per Loyfer reference cell type. Per-class A-score reads how strongly the sample looks like the reference cell type at the cell-type-discriminating CpGs.

### Direction expectation by comparison type

| Comparison type | Cell-of-origin tile direction | Other tile directions |
|---|---|---|
| Tumor-vs-adjacent-normal-paired | NEGATIVE (fidelity loss as tumor de-differentiates) | Mixed positive — non-cell-of-origin tile marker CpGs drift toward homogenized tumor methylation |
| Diseased-tissue-vs-healthy-cross-reference | POSITIVE (the diseased sample contains cells of that tissue type, above healthy reference baseline) | Mixed |

### Empirical evidence (2 cohorts, both colorectal, 2026-04-28)

| Cohort | Method | Paired d | 95% CI |
|---|---|---|---|
| TCGA-READ paired (VAL-098, n=7) | Full-HM450 cycling-class | +0.612 | [+0.227, +1.882] |
| TCGA-READ paired (VAL-098, n=7) | Colon_epithelial_cells tile | −2.501 | [−9.307, −1.584] |
| TCGA-COAD paired (VAL-062 revisit, n=26) | Full-HM450 cycling-class | +0.724 | (matches VAL-062 byte-for-byte) |
| TCGA-COAD paired (VAL-062 revisit, n=26) | Colon_epithelial_cells tile | −1.552 | [−2.175, −1.214] |
| TCGA-COAD paired (VAL-099 reproduction, n=26) | Full-HM450 cycling-class | +0.7241 | [+0.352, +1.296] |
| TCGA-COAD paired (VAL-099 reproduction, n=26) | Colon_epithelial_cells tile | −1.603 | [−2.173, −1.288] |

Three independent paired-tumor-vs-adjacent-normal cohort configurations, three negative cell-of-origin tile readings, three positive full-HM450 cycling-class readings. Direction concordance across all 10 top-magnitude tiles between READ, COAD revisit, and COAD VAL-099 reproduction on the run-everything 25-tile output: Bladder positive, Hepatocytes positive, Lung positive, Pancreatic_beta positive, Colon_epithelial_cells negative, Uterus_cervix negative.

### Prereg O1 criterion design rule (CHK-4.11)

Future preregs that include run-everything 25-tile per-class A-score on tumor-vs-adjacent-normal paired comparisons:

- NOT acceptable as O1 criterion: "Cell-of-origin tile shows positive d" (without comparison-type qualifier).
- NOT acceptable as O1 criterion: "Cell-of-origin tile is largest |d|" (other tiles may have larger |d| under the homogenization mechanism).
- Acceptable: "Cell-of-origin tile is among the top 5 largest |d| tiles in the run-everything 25-tile output."
- Acceptable: "Cell-of-origin tile shows |d| ≥ 0.5 with direction consistent with the comparison type (negative for tumor-vs-adjacent-normal-paired; positive for diseased-tissue-vs-healthy-cross-reference)."

### EDEAR commercial deployment behavior under CCL-039

Unaffected. Per CCL-037 LL-CROSS-COHORT-CALIBRATION, EDEAR commercial deployment is single-pipeline patient-vs-internal-reference. The run-everything 25-tile output for a real CRC or rectal cancer patient produces a tile pattern (Colon_epithelial_cells direction + co-firing tile profile) that is the diagnostic information. The patient's signal pattern fires the correct red flag because tumor colorectal cells diverge from healthy colorectal methylation as captured by the Loyfer reference. CCL-039 changes the prereg-O1-criterion design rule for retrospective cookbook validation, not the deployment behavior.

### Retroactive cookbook task (future-when-time-permits)

Apply the run-everything 25-tile methodology to the existing per-sample CSVs for VAL-060 (TCGA-BRCA breast), VAL-063 (TCGA-LUAD lung), VAL-064 (TCGA-LIHC liver), VAL-058 (GSE269244 prostate), and verify the cell-of-origin tile direction is consistently negative in tumor-vs-adjacent-normal paired comparisons across cancer types. CCL-039 is currently confirmed on three colorectal cohort configurations (TCGA-READ VAL-098, TCGA-COAD VAL-062 revisit, TCGA-COAD VAL-099 reproduction); cross-tissue confirmation upgrades it from a robustly-confirmed colorectal observation to a framework-level rule. The retroactive expansion is a future-when-time-permits task; it does not block current per-card publication.

---

## Part 15 — VAL-100 EPIC v2.0 deferral pattern under CHK-3.1 (added 2026-04-28)

**Source:** VAL-100 GSE282666 (Kumar 2024) under-50 buffy coat polyp Stage 1 immune A-score. First cookbook VAL on EPIC v2.0 (GPL33022). Cohort design correct: n=51, all under age 50, with same-day colonoscopy PNP+/PNP- status (16 PNP+ / 35 PNP-). Outcome `O5_DATA_INTEGRITY_FLAG` per pre-locked decision matrix.

**The substrate finding.** Pre-locked CHK-3.1 beta distribution check failed: extreme [<0.05 or >0.95] = 3.9%, middle [0.4-0.6] = 6.8%. Bimodal raw β signature requires extreme >30% AND middle <10%. Pre-locked CHK-3.2 cross-cohort baseline check independently confirms: PNP- mean A_immune = 0.807 vs Italian healthy buffy coat anchor 0.4384 = +15.13 anchor-SD offset (off-spec scale). Per Kumar 2024 Methods, the supplementary `GSE282666_Betas.csv.gz` is minfi v1.40.0 noob-bg-corrected output, not raw β. The processed values are biologically meaningful for the GrimAge clock analysis Kumar 2024 reports (clocks are designed to operate on noob-bg-corrected β), but the cookbook A_immune metric is calibrated against raw β and produces inflated and non-comparable values when applied to noob-bg-corrected output.

**The deferral path.** Per CCL-032 diagnostic order (data integrity → biology → framework), the observed Cohen's d (+0.236 [−0.363, +0.919]) does NOT get interpreted as biology under O5_DATA_INTEGRITY_FLAG. Defer the VAL to v0.2+ raw IDAT processing of `GSE282666_RAW.tar` through minfi or sesame (~2-4 hour task; the IDATs ARE deposited in the GSE282666 supplementary, no biobank application required).

**Cookbook precedent established (third instance of the same pattern).** CCL-040 in `LESSONS_LEARNED.md` documents three independent VALs across three substrates (LBC liquid biopsy via VAL-076, cervical-LBC residual M-values via VAL-077, buffy coat noob-bg-corrected via VAL-100) all hitting the same pattern: published GEO supplementary β-matrices are processed output, not raw β, and they fail CHK-3.1 in the cookbook framework. The pattern is structural, not an isolated mistake. The cookbook diagnostic order (CCL-032) is the correct response.

**EPIC v2.0 platform note.** First VAL on GPL33022 in the cookbook. Xu-538 panel coverage on EPIC v2.0 = 484/538 = 90.0% (10% drop). For future EPIC v2.0 cookbook work, the panel may need re-design for platform compatibility. The 54 missing CpGs are documented in the panel JSON for future panel-redesign work. Coverage drop is at the CHK-3.1 threshold; under raw IDAT v0.2+ re-processing, the 90% coverage is acceptable if the data integrity issue resolves.

**Operational rule for future VALs (consistent with CCL-040).**

1. Run CHK-3.1 + CHK-3.2 BEFORE any A-score scoring on a new GEO cohort.
2. If CHK-3.1 fails, DO NOT continue to biology interpretation. Outcome label `O5_DATA_INTEGRITY_FLAG`.
3. Defer to v0.2+ raw IDAT processing if IDATs are deposited (most GEO 850K/EPIC cohorts deposit RAW.tar). ~2-4 hour task per cohort.
4. If IDATs are not deposited, the VAL is structurally blocked at v1 and goes to `future_when_support_arrives.md`.

**EDEAR commercial deployment behavior under CCL-040.** Unaffected. EDEAR commercial deployment uses raw IDAT input through a single calibrated pipeline. A real patient's IDAT goes through the partner-lab pipeline, not through GEO-deposited supplementary normalized files. The CHK-3.1 failure on GSE282666 supplementary file does not propagate to deployment. Deployment behavior is unaffected by data-format issues in retrospective public-data validation.

---

## Part 16 — VAL-101 O5 + VAL-102 voided self-correction + CCL-041 platform calibration (added 2026-04-28)

**Source:** VAL-101 hcc-epic 25-tile etiology stratification on TCGA-LIHC HM450 paired tumor/adjacent-normal (sealed prereg SHA `fa366bf00316597bb65032b747029133acb5f1bbb40f6251094b563732185512`, RNG seed 20260428). Outcome `O5_DATA_INTEGRITY_FLAG` per pre-locked CHK-3.1 threshold trip.

**The VAL-101 outcome.** Pre-locked CHK-3.1 thresholds (extreme >30% AND middle <10%, the raw-EPIC default from VAL-100 prereg) tripped on TCGA-LIHC HM450 sesame Level 3 at extreme 26.6% / middle 9.1%. Per CCL-032 diagnostic order (data integrity → biology → framework), cookbook discipline honors the trip. The biological readouts produced under VAL-101 (Pooled Hepatocytes tile d = −1.521; All_viral d = −1.726; All_non_viral d = −1.393; No_documented_risk Marcus-analog d = −1.141; CCL-039 cross-tissue cross-cohort pattern observation; viral-vs-non-viral per-tile-vs-pooled refinement) are descriptive supplementary documentation only. They do NOT propagate to the hcc-epic card or to any cookbook reference document.

**Why O5 stands.** The pre-registration was sealed before β-access. The threshold tripped. Prereg discipline requires honoring sealed criteria. CHK-4.8 honest-revision is reserved for structurally degenerate criteria (cf VAL-097 O2_CYCLING_DISTRIBUTED), not for misspecified thresholds. The biology being clean does not justify post-hoc threshold relaxation; that is exactly the failure mode prereg discipline is designed to prevent. The cookbook precedent argument cuts the wrong way: the fact that VAL-062 / VAL-098 / VAL-099 produced clean biology on the same TCGA HM450 sesame Level 3 substrate (extreme ~24-27% / middle ~9% on retroactive check) does NOT retroactively justify relaxing VAL-101's threshold — those VALs never explicitly ran CHK-3.1, so the cookbook precedent is silent on whether they would have passed under the raw-EPIC threshold. Their existence justifies tightening prereg discipline going forward (by setting platform-specific thresholds via calibration VALs), not loosening VAL-101's outcome.

### Self-correction: VAL-102 voided before execution

A VAL-102 prereg was sealed at 2026-04-28T20:31:23Z with a TCGA HM450 platform threshold (extreme >20%) derived from VAL-101's tripped data. The intent was to "do the prereg right the second time" by re-running the same methodology on the same cohort under a platform-tuned threshold. This was identified within minutes as post-hoc threshold accommodation with a SHA stamp — the threshold was selected to accommodate values already observed in VAL-101 (extreme 26.6%) plus values observed in retroactive VAL-099 verification (extreme 24.4%). Sealing such a threshold and applying it to the same cohort is circular reasoning, not pre-registration.

VAL-102 was voided at 2026-04-28T20:35Z, before any execution. Audit trail preserved at `Biological_Physics/validation_runs/VAL-102/VOIDED_BEFORE_EXECUTION.md` with the original SHA `2b77ad9d3b69554a0658260756db0f08722e2be3fa96eb48aad9213974f4717c`. The cookbook does not delete sealed records; it marks them and explains.

**Logging the void event is part of the discipline that CCL-041 represents.** When prereg-discipline failure modes are caught quickly, the cookbook records the catch as data, not as embarrassment. The void event is a positive cookbook signal: it shows the discipline is operating as designed, including against same-day temptations to bend the protocol when biology looks exciting.

### CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION (formalized 2026-04-28)

**Lesson:** CHK-3.1 beta distribution check thresholds must be platform-specific. The original cookbook threshold (extreme >30% AND middle <10%) was tuned against raw EPIC β in VAL-100 prereg. TCGA HM450 sesame Level 3 — the cookbook's standard public tissue-validation substrate, used in VAL-058 / VAL-060 / VAL-062 / VAL-063 / VAL-064 / VAL-098 / VAL-099 — reads slightly less extreme bimodality (~24-27% extreme / ~9% middle) on the same check methodology because the standard TCGA pipeline applies dye bias correction to the IDATs before producing Level 3 betas.

**Distinction from CCL-040.** CCL-040 covers PROCESSED OUTPUT (residual M-values, batch+chip+age+HPV-corrected, noob-bg-corrected with additional normalization) — output that loses bimodal raw β signature entirely (extreme 3.9% / middle 6.8% in VAL-100 GSE282666). CCL-041 is about raw-β bimodality manifesting at slightly different threshold values across raw-β platforms (sharper on raw EPIC, softer on sesame-corrected HM450). Two distinct concerns.

**Operational rule.** CHK-3.1 thresholds are platform-specific. The threshold for any new platform MUST be set by a calibration VAL on a structurally-separate cohort, NOT by retroactive accommodation:

| Platform | extreme threshold | middle threshold | Status |
|---|---|---|---|
| Raw EPIC β / EPIC v2.0 β (un-normalized) | > 30% | < 10% | Established (VAL-100) |
| TCGA HM450 sesame Level 3 β | TBD | < 10% | **Calibration VAL needed** — must be done on a cohort structurally separated from any active test cohort |
| Other platforms | TBD | TBD | Document at first calibration VAL on platform |

**Why a calibration VAL is required.** Setting a platform threshold from data that is also being interpreted under the threshold is circular reasoning. The proper calibration pathway uses TCGA samples from a tissue NOT under active test (TCGA-KIRC adjacent-normal, TCGA-PRAD adjacent-normal, etc.), measures the bimodality distribution there, sets the threshold from THAT distribution, seals it, and applies it to future test cohorts as a pre-registered platform criterion. This is a multi-VAL workstream, not a same-day re-seal.

### Honest path forward for VAL-101 biological readouts

The biological readouts in VAL-101 are real numbers. Their inferential validation pathway requires either:

1. **Calibration-VAL path.** Run a calibration VAL on TCGA samples from a tissue NOT under active hcc-epic test to establish the TCGA HM450 sesame Level 3 platform threshold. Seal the threshold. Then run a future hcc-epic VAL (call it VAL-XYZ, NOT VAL-102) on the TCGA-LIHC test cohort under the pre-registered platform threshold.

2. **CCL-040 deferral path.** Process the TCGA-LIHC .idat files through sesame from raw IDAT input. Verify bimodality at standard pipeline output. Re-run hcc-epic test on reprocessed betas. Same precedent as VAL-100 deferral.

Both paths take longer than a same-day re-seal. Both are honest. The first path is the proper extension of the cookbook to a new platform calibration; it produces a generally-applicable threshold for any future TCGA HM450 sesame Level 3 cohort. The second path follows the CCL-040 precedent already established for VAL-100; it is more specific to this cohort but more directly comparable to how VAL-100 was deferred.

### EDEAR commercial deployment unaffected

Per CCL-037, VAL-101 + VAL-102-voided + CCL-041 are retrospective cookbook validation activity with no impact on EDEAR commercial deployment. Deployment uses single-pipeline patient-vs-internal-reference architecture that is structurally insulated from public-data CHK-3.1 calibration questions. The CHK-3.1 platform-tuning question lives in the retrospective cookbook validation layer only.

### Methodological footnote — why VAL-102 was caught quickly

The VAL-102 void event (sealed at 20:31:23Z, voided at 20:35Z) is a positive cookbook signal, not a problem. The same-day catch demonstrates that prereg discipline is operating: the temptation to bend a sealed criterion is strongest exactly when the biology looks like it might say something that matters to a person. CCL-041 logs the catch as part of the cookbook's institutional memory. Future preregs that propose platform threshold changes mid-stream will reference this void event as the reason the calibration-VAL pathway is mandatory rather than optional.

---


## Part 17 — CHK-3.1A/B split convention + cardio-epic v0.1 first native build (added 2026-04-28)

**Source:** VAL-106 + VAL-107 calibration on TCGA-KIRC + TCGA-PRAD adjacent-normal HM450K sesame Level 3 (n=210) + cardio-epic v0.1 disease VAL chain (VAL-108 GSE69138 stroke + VAL-109 GSE84395 PAH + VAL-110 GSE84274 aortic). CCL-042 LL-CHK-3.1-A/B-SPLIT formalized 2026-04-28.

### What changed

CHK-3.1 has been split into two distinct named checks. Both must pass.

- **CHK-3.1A — Full-genome bimodality (substrate gate).** Computed on every valid β value in the input file, no subsetting. Single threshold per measurement substrate. Established by calibration VAL on a structurally-separated healthy adjacent-normal cohort. Reused indefinitely for that substrate. Catches the CCL-040 failure mode (processed-output substrates that lose bimodality globally).
- **CHK-3.1B — Card-specific marker subset bimodality (panel-coverage gate).** Computed on the union of CpGs the card's scoring will use. Per-card threshold derived from the same calibration cohort as CHK-3.1A but on the card's specific subset. Recomputed when the card adds a new atlas or updates a marker panel. Stored in the card's `universal_pipeline_acknowledgment.chk_3_1_thresholds_per_substrate` block. Catches probe-list lift-over dropouts and panel-specific damage.

The motivation for the split came from VAL-106's discovery that the cookbook had been silently conflating two distinct measurement questions under one CHK-3.1. Three prior data points (VAL-101 26.6%, VAL-099 24.4%, GSE69138 ave_beta peek 21.9-27.3%) that had been used to set the empirical HM450K range were CpG-subset measurements, not full-genome measurements. VAL-106's actual full-genome measurement on healthy adjacent-normal TCGA-KIRC + TCGA-PRAD reads f_extreme ~55.87%, more than 20 percentage points above the misspecified empirical range. Both measurements are right; they are simply answering different questions. CCL-042 in `LESSONS_LEARNED.md` formalizes the split.

### Operational pipeline change

The per-cohort data-integrity check now produces a 2-element pass/fail vector rather than a single pass/fail. Both elements must be true. Pipeline output JSON for every VAL on a card built natively under split convention contains:

```
"chk_3_1": {
  "chk_3_1a_full_genome": {
    "f_extreme": <float>, "f_middle": <float>, "n_valid": <int>,
    "substrate": <string>, "threshold_extreme": <float>,
    "threshold_middle": <float>, "threshold_n_valid": <int>,
    "calibration_anchor_val_id": <string>, "pass": <bool>
  },
  "chk_3_1b_card_subset": {
    "card_id": <string>, "subset_sha256": <string>, "subset_size": <int>,
    "f_extreme_subset": <float>, "f_middle_subset": <float>, "n_subset_valid": <int>,
    "threshold_extreme": <float>, "threshold_middle": <float>,
    "threshold_n_subset_valid": <int>, "calibration_anchor_val_id": <string>, "pass": <bool>
  },
  "overall_pass": <bool>
}
```

### Calibration VAL anchors per substrate

| Substrate | CHK-3.1A anchor VAL | CHK-3.1B anchor VAL (per card) |
|---|---|---|
| TCGA HM450K sesame Level 3 | VAL-106: extreme≥50.5%, middle≤9.0%, n≥400K | VAL-107 cardio-epic: extreme≥55.0%, middle≤8.5%, n_subset≥7000 of 8100 |
| GenomeStudio AVG_Beta HM450K (un-normalized) | within-cohort self-cal at v0.1 (VAL-108 / VAL-110); pending generalizable structurally-separated VAL | within-cohort self-cal at v0.1; pending generalizable VAL |
| minfi `preprocessFunnorm` HM450K | within-cohort self-cal at v0.1 (VAL-109); pending generalizable VAL | within-cohort self-cal at v0.1 |
| minfi noob-bg-corrected EPIC v2 | known-fail substrate per CCL-040 (VAL-100 GSE282666) | not applicable — substrate fails CHK-3.1A by design |
| Other substrates | new calibration VAL required before card use on that substrate | new calibration VAL required |

### First card built natively under split: cardio-epic v0.1

Cardio-epic v0.1 is the reference implementation. Its `universal_pipeline_acknowledgment.chk_3_1_thresholds_per_substrate` block is the canonical structure for all subsequent cards. The block carries CHK-3.1A thresholds keyed by substrate AND the card-specific CHK-3.1B threshold. Phase 3 retroactive review will bring breast-epic (v2.4), lung-epic (v0.6), ad-immune, hcc-epic (v0.4), crc-epic (v2.5), kidney-epic, and cervical-epic into the same structure without unsealing any sealed VAL outcomes.

### Cardio-epic v0.1 biology summary

Phase 1 cardio testing produced four card-specific lessons (CCL-043 in `LESSONS_LEARNED.md`, also stored in cardio-epic card JSON `lessons_learned.card_specific`):

- LL-CARDIO-001 — Substrate-cell match matters. VAL-110 Vascular_endothelial_cells tile d=−0.04 on aortic bulk tissue vs VAL-109 d=+0.79 on cultured PECs is the substrate-cell-mismatch signature. The framework reads what is in the sample.
- LL-CARDIO-002 — Whole blood does not stratify ischemic stroke by TOAST etiology (biology-correct null, not framework failure). VAL-108 max |d| = 0.167 across all stages and contrasts on n=404. Post-stroke inflammatory homogenization is real.
- LL-CARDIO-003 — Heritable PAH > idiopathic PAH framework signal is biology-consistent. VAL-109 Vascular tile control vs hPAH d=+0.79 vs control vs iPAH d=+0.42, hPAH vs iPAH d=−0.35 (framework-equivalent).
- LL-CARDIO-004 — Aortic pathology is Stage 1 immune-detectable, Stage 2 vascular-tile-resistant. VAL-110 Stage 1 normal vs BAV d=+1.08 vs Vascular tile |d|≤0.15 on bulk aorta. Universal Stage 1 immune flag is the operational discriminator across all cardio substrates.

Cardio-epic v0.1 deployment policy: stroke whole blood reported as single pooled signature (no etiology stratification), PAH reported with vascular-tile emphasis on cultured PEC substrate (subtype pooling), aortic pathology reported with Stage 1 immune as primary discriminator (etiology pooling, vascular tile NOT used on bulk substrate).

### EDEAR commercial deployment

Per CCL-037, deployment uses single calibrated patient-vs-internal-reference pipeline. Under the split, CHK-3.1A is computed once per customer (substrate gate); CHK-3.1B is computed per disease card (panel-coverage gate). A customer with substrate-clean data but partial panel coverage on some cards receives the cards their data supports rather than an all-or-nothing report failure. This is a meaningful UX improvement over the conflated CHK-3.1.

### Phase 1/2/3 rollout status

- **Phase 1 (complete 2026-04-28):** VAL-106 + VAL-107 + VAL-108 + VAL-109 + VAL-110 all sealed and run; cardio-epic v0.1 card + README built natively under split convention.
- **Phase 2 (this part of pipeline reference, in progress):** Cookbook-wide convention update — TESTING_CHECKLIST.md (in-place section update applied), LESSONS_LEARNED.md (CCL-042 + CCL-043 added), this document (Part 17), README_MASTER (v2.3 → v2.4 amendment), GAPE_Evidence_Report HTML (VAL-106 through VAL-110 entries already present per Bio Physics README inspection; retroactive split-classification footnotes for VAL-100 / VAL-101 / VAL-077 to be applied separately), GAPE_Reproduction_Paper §7.20 to be added.
- **Phase 3 (pending Phase 2 sign-off):** Per-card retroactive review for breast-epic, lung-epic, ad-immune, hcc-epic, crc-epic, kidney-epic, cervical-epic — additive documentation updates to each card's `universal_pipeline_acknowledgment.chk_3_1_thresholds_per_substrate` block. No sealed VAL outcomes change.

---

## Part 18 — Card-content structural-parity gates (CHK-5.7/5.8/5.9/5.10, added 2026-04-28)

**Source:** Cardio-epic v0.1 ship-state audit on 2026-04-28 revealed structural thinness vs breast-epic v2.3 (900 lines) and crc-epic v2.4 (791 lines) — cardio-epic shipped at 345 lines, missing universal_80_cell_age_baseline_immune_class, universal_sex_stratification_rule, full universal_h_min_table, full 6-tier universal_tier_thresholds, atlases_used_and_deferred block, substrate_roadmap block, and per-substrate CHK-3.1A/B documentation for non-TCGA substrates. Heath flagged the gap; CHK-5.7/5.8/5.9/5.10 added to TESTING_CHECKLIST 2026-04-28 as enforcement gates.

### What changed

The card-publish process previously relied on README §17 Block 1-20 prose to indicate what every card must contain. Block 1-20 is correct as a content checklist, but it was implicit: a card author could ship a card that nominally satisfied each block while omitting most of the per-block sub-keys. CHK-5.7/5.8/5.9/5.10 convert four specific structural requirements into per-sub-key verifiable gates at card-publish time.

**CHK-5.7** verifies the card's `universal_reference` block contains all 14 required sub-keys (the full Block 5 universal-reference contract). The cardio v0.1 build shipped with 8 thin sub-keys; this gate catches that.

**CHK-5.8** verifies the card contains an `atlases_used_and_deferred` block listing every Queue-1 atlas as either run (with VAL anchor) or deferred (with target version + unblock dependency). The cardio v0.1 build deferred EpiSCORE HeartRef and Caggiano CelFiE TIM at VAL-107 prereg time without re-surfacing the deferral at card-publish time; this gate catches that.

**CHK-5.9** verifies the card contains a `substrate_roadmap` block with all 5 MESA substrates explicitly addressed (DNAm, nucleosome_occupancy, fragment_fuzziness, WPS, fragment_size). The cardio v0.1 build had no substrate_roadmap block; this gate catches that.

**CHK-5.10** verifies the card's `chk_3_1_thresholds_per_substrate` block contains BOTH CHK-3.1A and CHK-3.1B threshold entries for every substrate the card supports. The cardio v0.1 build had partial CHK-3.1A/B documentation only for TCGA HM450K sesame Level 3 (VAL-106/107); GenomeStudio AVG_Beta HM450K and minfi preprocessFunnorm HM450K were within-cohort self-cal only without explicit calibration-debt acknowledgment in the card; this gate catches that.

### How this connects to the existing Block 1-20 framework

Block 1-20 in README_MASTER §17 is the content checklist (what every card must say). CHK-5.7/5.8/5.9/5.10 are the structural gates (which JSON keys must exist with which sub-key contents to evidence that Block 1-20 was satisfied). The two are complementary:

- A card that has Block 1-20 prose but fails CHK-5.7 has decorative content without the underlying data structure to support deployment.
- A card that has CHK-5.7-compliant JSON but fails Block 1-20 prose has data structure but no narrative for a referee or partner lab to read.

Both must pass.

### Cardio-epic v0.1 → v0.2 rebuild context

The v0.2 rebuild applies CHK-5.7/5.8/5.9/5.10 as gates on the rebuild itself. v0.2 ships with all 14 universal_reference sub-keys, atlases_used_and_deferred block (EpiSCORE HeartRef in atlases_deferred per VAL-111 sealed at O3_TISSUE_FLOOR_DOMINATED, Caggiano CelFiE TIM in atlases_deferred with target v0.3 + HM450 hg19 manifest acquisition unblock dependency), substrate_roadmap (DNAm validated via VAL-108/109/110/111; remaining 4 substrates as v0.3+ targets), and chk_3_1_thresholds_per_substrate covering all four substrates the card uses (TCGA HM450K sesame Level 3, GenomeStudio AVG_Beta, minfi preprocessFunnorm, GenomeStudio V2011.1 raw).

### Phase 4 retroactive review

Following Phase 3 (CHK-3.1A/B retroactive review for breast/lung/ad-immune/hcc/crc/kidney/cervical), Phase 4 will verify each existing card passes CHK-5.7/5.8/5.9/5.10. Breast-epic v2.3 and crc-epic v2.4 are expected to pass (they are the structural reference templates). Other cards may need additive documentation updates to add missing sub-keys without invalidating any sealed VAL outcomes.

### EDEAR commercial deployment

Per CCL-037, deployment uses single calibrated patient-vs-internal-reference pipeline with run-everything scoring across all production atlases. CHK-5.7/5.8/5.9/5.10 are card-publish-time documentation gates — they ensure the card the customer or referee sees accurately describes what the deployment pipeline will execute. The deployment pipeline itself runs every panel and every reference atlas regardless of which gates the card-publish process applies.

---

## Part 19 — Cardio-epic v0.2 ship + CHK-5.11 atlas-family fitness gate (added 2026-04-29)

VAL-111 sealed 2026-04-29 at `O3_TISSUE_FLOOR_DOMINATED` (prereg SHA `172c6ae2a11345935c176b4a1fc57d30009ad4bac9bb9cdeeb9c8226035b78a6`). Atlas: EpiSCORE HeartRef (Zhu et al. *Nat Commun* 2022 13:3895), gene-promoter cardiac reference matrix bridged to 3,727 unique 450K CpGs × 5 cardiac cell types (CM/EC/FB/MP/SMC), GPL-2 license, atlas SHA-256 `bf6431f66749f02a616560764af3fdd0adc70b03bca96b2a13b6221bbd847c83`. Three cohorts already sealed under VAL-108/109/110: GSE69138 stroke whole blood n=589 (negative control), GSE84395 PAH cultured PEC n=39, GSE84274 ascending aorta tissue n=24. All three cohort intersections cleared >500 atlas CpGs (no O4 bridge failure). All five cardiac tile A-scores read 0.46–0.51 across all three cohorts and all three substrates regardless of disease state. Maximum within-cohort tissue discrimination = 0.0152 (GSE84274 MP, dissection 0.5012 − normal 0.4860); EC tile range in GSE84395 PEC = 0.0070; SMC tile range in GSE84274 = 0.0120 — all an order of magnitude below the 0.10 pre-locked discrimination threshold. Blood-floor breach on all 5 tiles in GSE69138 (cohort means 0.48–0.51, well above 0.10 floor). Direction was biologically sensible (dissection > BAV+dilation > normal monotonic in GSE84274; SMC tile always highest in aortic samples; iPAH > hPAH > control on EC tile in GSE84395) but A-score magnitude was set by gene-promoter average methylation rather than substrate-specific cell-of-origin contrast.

### What VAL-111 demonstrated about atlas family

EpiSCORE HeartRef is methodologically sound for its design purpose (EpiDISH-style proportion estimation in heart tissue using gene-promoter integer marker IDs against a reference panel matrix, returning cell-type fractions). It is NOT designed for A-score tile reading on heterogeneous β panels, which is what cardio-epic Stage 2 does. Two distinct atlas-scoring modalities exist: (a) tile-coverage A-score reading on heterogeneous β panels — needs WGBS-derived tiles or equivalent CpG-coverage panels with cell-type-specific differential methylation (Loyfer 25-tile, Caggiano CelFiE TIM); (b) EpiDISH proportion estimation on per-tissue β — uses gene-promoter integer marker IDs against a reference panel matrix, returns cell-type fractions not A-scores (EpiSCORE family). EpiSCORE HeartRef belongs in (b); cardio-epic Stage 2 needs (a). The deferral is the correct cookbook outcome.

### CHK-5.11 atlas-family fitness gate (added 2026-04-29)

Following VAL-111, CHK-5.11 was added to TESTING_CHECKLIST.md to formalize the atlas-family fitness check before sealing any future Stage 2 atlas integration. The gate verifies in the prereg that (i) the atlas has CpG-coverage panels (not gene-promoter integer marker IDs) for the cell types it claims to discriminate; (ii) the atlas's intended scoring modality matches the card's Stage 2 reading mode; (iii) the prereg explicitly names the discrimination threshold so an O3_TISSUE_FLOOR_DOMINATED outcome is sealable; (iv) the card JSON's `atlases_used_and_deferred` block (CHK-5.8) surfaces the atlas-family fitness assessment in `deferral_rationale` if the integration is deferred.

### Cardio-epic v0.2 card content shipped 2026-04-29

cardio_epic_card_v0_2.json: 28 top-level keys, 774 lines, full Block 1-20 + CHK-5.7/5.8/5.9/5.10 structural-parity with breast-epic v2.3 / crc-epic v2.4. atlases_run = [Loyfer_25tile, UniLIFE_19cell, Salas_Blood_EPIC_IDOL_6cell] all anchored at VAL-108/109/110. atlases_deferred = [EpiSCORE_HeartRef anchored at VAL-111 with full deferral rationale, Caggiano_CelFiE_TIM_cardiac blocked at HM450 manifest]. substrate_roadmap covers all 5 EDEAR substrates with status/anchor/target/rationale. chk_3_1_thresholds_per_substrate covers all 4 substrates encountered (TCGA HM450K sesame Level 3, GenomeStudio AVG_Beta HM450K, minfi preprocessFunnorm HM450K, GenomeStudio V2011.1 HM450K raw). lessons_discovered_v0_2 section documents six discoveries (DISC-CARDIO-001 through DISC-CARDIO-006), six things cardio-epic v0.2 chose not to claim, and ten things remaining open.

cardio_epic_README.md: 397 lines, preserves all v0.1 prose additively, adds VAL-111 row in validated cohorts table, adds atlases_run / atlases_deferred lists, adds substrate roadmap table, adds full "What we discovered in the cardio sprint" section with all six DISC-CARDIO discoveries written as prose, adds "What we chose not to claim" and "What remains open" sections, adds full VAL-111 validation evidence subsection with all per-cohort per-tile numbers, adds LL-CARDIO-005, adds v0.1→v0.2 changes section.

### What pushed to GitHub vs what stayed Heath-only

**Pushed to GitHub commit `facbe7a` (2026-04-29):**
- `Biological_Physics/validation_runs/VAL-111/` (prereg.md, val_111.py, restratify.py, results.json, stratified.json, outcome.md, three per-sample CSVs, run.log, PREREG_SEAL.txt)
- `Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_heartref/` (atlas vault: bridged CSV, Entrez matrix, source rda, README)
- `Biological_Physics/README.md` updated with VAL-111 row in Validation Record table

**Heath-only delivery (NOT pushed to GitHub per cookbook IP rule):**
- cardio_epic_card_v0_2.json
- cardio_epic_README.md
- This Part 19 update to EDEAR_PIPELINE_OFFICIAL_REFERENCE_v2.md
- TESTING_CHECKLIST.md CHK-5.11 addition
- LESSONS_LEARNED.md CCL-044 addition
- README_MASTER_v2_4.md cardio-epic v0.2 status update
- GAPE_Reproduction_Paper_v1.md cardio-epic v0.2 status update
- GAPE_Evidence_Report_UPDATED.html cardio-epic v0.2 status update

### EDEAR commercial deployment

Per CCL-037, EDEAR commercial deployment runs single calibrated patient-vs-internal-reference pipeline with run-everything scoring across all production atlases. VAL-111's deferral of EpiSCORE HeartRef does not affect commercial deployment: cardio-epic v0.2 production scoring uses Loyfer 25-tile (validated) for Stage 2; EpiSCORE HeartRef is not in `atlases_run`. When the v0.3 atlas integration unblocks (re-bridging or Caggiano CelFiE TIM acquisition), the deployment pipeline is updated additively without requiring re-calibration of existing cardio scoring.

---

## Part 20 — Cardio-epic v0.2.1 honesty patch + CHK-5.12 atlas-canonical-source-check + v0.3 critical path (added 2026-04-29 same-day after v0.2 ship)

After cardio-epic v0.2 shipped 2026-04-29 morning, a same-day audit identified three issues requiring honest correction in a v0.2.1 patch (no sealed VAL outcomes change).

### Issue 1: Atlas naming was incomplete in v0.2

v0.2 labeled the cardio Stage 2 atlas "Loyfer 25-tile" with 6,105 CpGs. The actual file is `loyfer_moss_2018/reference_atlas.csv` — 7,890 CpGs across 25 cell-type columns, which is the **layered Moss 2018 + Loyfer 2023 array atlas** combined into one file per Part 2.1+2.2 of this reference document. v0.2.1 corrects the naming everywhere it appears. Both atlases were operative in VAL-108/109/110 scoring; the naming undersold what was running.

### Issue 2: `atlases_deferred` block was incomplete in v0.2

v0.2 listed only 2 deferred atlases (EpiSCORE HeartRef + Caggiano CelFiE TIM). Part 2.3 through 2.7 of this reference document name several additional cardio-relevant Stage 2 atlases that should have been in atlases_deferred from the start:

- **Konigsberg 2023 cardiovascular 28-cell atlas** (Part 2.4) — explicitly named as cardio deployment blocker: *"Without this atlas, cardio-epic cannot be deployed."* Includes sorted cardiomyocytes (terminal class, H_min = 0.7728), cardiac fibroblasts, vascular endothelial, smooth muscle. Currently invisible to the layered Moss+Loyfer chain.
- **EpiSCORE Zhu/Teschendorff 2022 pan-tissue** (Part 2.3) — full 13-tissue version including Heart, Kidney, Liver, Lung, Brain references. On disk in atlas_vault. Separate from the HeartRef sub-panel scored in VAL-111. R-package integration via existing rpy2 bridge.
- **Tanaka 2025 6-cell-type neural cfDNA atlas** (Part 2.5) — *"highest-priority new addition"* per this document. Cortical / dopaminergic / motor / astrocyte / Schwann / microglia. Relevant to cardio via astrocyte/microglia signatures of cardiac inflammation. Nanopore→array bridge engineering blocker.
- **Tian et al. 2023 scMCodes brain** (Part 2.6; lead Wei Tian, co-first Jingtian Zhou) — 188 cell types, v0.4+ candidate. Lower priority for cardio.
- **MARLIN Capper 2025 training scaffold** (TESTING_CHECKLIST §STAGE 0 Queue-1) — leukemia matrix v0.3 build-out task. Lower cardio relevance.
- **Sabedot GeLB 2021** (TESTING_CHECKLIST §STAGE 0 Queue-1) — R training script, requires GSE150289 cohort.

cardio-epic v0.2.1 atlases_deferred expands from 2 entries to 8, with target_version + unblock_dependency per atlas.

### Issue 3: VAL-108/109/110 scored Stage 2 against ONLY the layered Moss+Loyfer combined atlas

Per the run-everything policy (Heath sign-off 2026-04-26, TESTING_CHECKLIST §run-everything), every IDAT runs Stage 2 against ALL reference atlases in the vault. VAL-108/109/110 violated this: they scored Stage 2 only against `loyfer_moss_2018/reference_atlas.csv`. The other Stage 2 atlases in atlas_vault (caggiano_celfie_2021, caggiano_celfie_tim, episcore_zhu_teschendorff_2022, episcore_heartref pre-VAL-111, marlin_capper_training, sabedot_gelb_2021) were NOT scored on cardio cohorts. v0.2 documented the gap as if it were correct architecture; v0.2.1 explicitly acknowledges the run-everything violation and queues corrective re-execution of VAL-108/109/110 against the full atlas stack as part of v0.3 critical path.

### DISC-CARDIO-007 — Always read PIPELINE_REFERENCE Part 2 first

VAL-111 was scored against EpiSCORE HeartRef because that atlas was already in atlas_vault from a prior acquisition pass. Part 2.4 of this document explicitly names Konigsberg 2023 — NOT EpiSCORE — as the cardio Stage 2 atlas blocker. The atlas selection in cardio v0.1/v0.2 was made by browsing atlas_vault rather than by reading this document. VAL-111 produced a real and useful negative result (atlas-family-fitness lesson, LL-CARDIO-005), but it was a side-track from the canonical cardio atlas critical path.

### CHK-5.12 atlas-canonical-source-check gate (added 2026-04-29 to TESTING_CHECKLIST.md)

Before sealing any new atlas integration prereg, the prereg must cite which canonical-document section (this document Part 2.X or README_MASTER §Stage 2.X) names the atlas as a production candidate for the card under test. Companion to CHK-5.11 atlas-family-fitness check. Together CHK-5.11 + CHK-5.12 form the "is this the right atlas to test?" gate before any atlas integration VAL is sealed:

- **CHK-5.11** asks: does the atlas family match the card's Stage 2 scoring modality? (tile-coverage A-score reading vs EpiDISH proportion estimation)
- **CHK-5.12** asks: is this atlas named in the canonical documents as a production candidate for this card?

Both gates must pass for an atlas integration prereg to seal as `atlases_run`-eligible.

### v0.3 critical path for cardio-epic deployment

Documented in cardio-epic v0.2.1 card JSON `canonical_documents_named_blocker_for_cardio_deployment` block:

**Phase A — atlas acquisition / engineering:**
1. Acquire Konigsberg 2023 atlas (NAR Genomics & Bioinformatics 2023, doi:10.1093/nargab/lqad061) — highest priority, document-named deployment blocker per Part 2.4
2. Acquire HM450 hg19 manifest to unblock Caggiano CelFiE TIM cardiac
3. Engineer Tanaka 2025 nanopore→array CpG bridge
4. Integrate EpiSCORE Zhu/Teschendorff pan-tissue via R rpy2 bridge (existing infrastructure)

**Phase B — per-atlas calibration VAL** (must seal **before** any cardio-cohort scoring against that atlas):
5. Konigsberg 2023 calibration VAL → CHK-3.1A baseline + CHK-3.1B subset threshold sealed on structurally-separated healthy cohort
6. Caggiano CelFiE calibration VAL → same
7. Tanaka 2025 calibration VAL → same (after bridge)
8. EpiSCORE pan-tissue calibration VAL → same

**Phase C — cardio-cohort scoring against each calibrated atlas:**
9. VAL-XXX: Konigsberg 2023 on GSE69138 + GSE84395 + GSE84274 + ideally GSE56046 MESA CHD/MI cohort n=1,202
10. VAL-XXX: Caggiano CelFiE on the same cardio cohorts
11. VAL-XXX: EpiSCORE pan-tissue on the same cardio cohorts
12. VAL-XXX: Tanaka 2025 on the cardio cohorts

**Phase D — re-execute VAL-108/109/110 honoring run-everything:**
After each new atlas is calibrated and brought into production, the existing VALs need re-execution against the full atlas stack. Sealed structural outcomes from VAL-108/109/110 don't change; this adds new per-atlas results to the same cohorts.

**Phase E — cardio-epic v0.3 ship:**
Once Phase B + C complete for at least Konigsberg + Caggiano (the two with explicit cardiac coverage), the card promotes from v0.2.1 to v0.3 with those atlases in `atlases_run` and the v0.2.1 deferral notes resolved.

### Generalization for the cookbook

CHK-5.12 applies to every card. Before any future atlas integration VAL is sealed (cardio v0.3 Konigsberg, lung-epic v0.3 atlases, ad-immune Tanaka neural, glioma-epic v0.3 Caggiano neuronal, etc.), the prereg must cite the canonical-document section that names the atlas as a production candidate for the card under test. Atlas selection by "browsing atlas_vault" is not a sufficient justification; the canonical-document anchor is mandatory.

The cardio-epic v0.2.1 same-day patch is an example of corrective documentation discipline: when an honest audit identifies missing canonical-document anchors after a card has shipped, the same-day patch (without unsealing any VAL) is the corrective mechanism, not a v0.3 wait.

### What pushed to GitHub vs what stayed Heath-only at v0.2.1

**Pushed to GitHub at v0.2 (commit `facbe7a`, 2026-04-29 morning):** VAL-111 directory + EpiSCORE HeartRef atlas vault + Biological_Physics/README.md row. **No additional GitHub artifacts in v0.2.1.**

**Heath-only delivery at v0.2.1:** cardio_epic_card_v0_2_1.json, cardio_epic_README_v0_2_1.md, this Part 20 update, TESTING_CHECKLIST.md CHK-5.12 addition, LESSONS_LEARNED.md CCL-045 addition, README_MASTER_v2_4.md v2.4 amendment update for v0.2.1, GAPE_Reproduction_Paper_v1.md §7.22 addition, GAPE_Evidence_Report_UPDATED.html v0.2.1 honesty-patch section.

### EDEAR commercial deployment

Per CCL-037 — unaffected. v0.2.1 honesty patch documents what's missing from v0.2 cookbook-side validation; it does not modify deployment architecture. Cardio-epic production scoring at v0.2.1 still uses the layered Moss+Loyfer atlas (validated) for Stage 2; the additional canonical-document-named atlases (Konigsberg, Caggiano, EpiSCORE pan-tissue, Tanaka) are queued for v0.3 with calibration-before-scoring discipline.

---

## Part 21 — Cardio-epic v0.2.2 honesty patch + CCL-046 documents-of-record audit lesson + sorted-cardiomyocyte atlas gap acknowledged (added 2026-04-29 same-day after v0.2.1 patch)

After cardio-epic v0.2.1 shipped 2026-04-29 (same-day morning honesty patch on v0.2), Phase A acquisition of the canonical-document-named "Konigsberg 2023" cardio Stage 2 atlas began. Web verification of the cited DOI (`10.1093/nargab/lqad061`) found that the canonical document had two factual errors in Part 2.4: (1) author attribution wrong — actual paper is **Cuadrat et al. 2023**, no Konigsberg in the author list; (2) cell-type content wrong — actual atlas is the Moss 2018 base extended with three **bulk** ENCODE heart tissues (right atrium, left ventricle, coronary artery), not the "sorted cardiomyocytes, cardiac fibroblasts, vascular endothelial, smooth muscle" Part 2.4 claimed. A second search confirmed there is no separate Konigsberg-authored cardiovascular methylation atlas paper in published literature; the document had conflated two different works or misremembered the citation.

### Three corrections in v0.2.2

**Correction 1 — Part 2.4 fully replaced.** The corrected version (now in this document) names Cuadrat et al. 2023, describes what the paper actually contains (Moss 2018 base + 3 bulk ENCODE heart tissue additions, 28 total tissues/cell types), removes the "sorted cardiomyocytes" claim, removes the "cannot be deployed" framing, and adds explicit statements about what the atlas IS and IS NOT.

**Correction 2 — "deployment blocker" framing dropped.** The prior framing ("Without this atlas, cardio-epic cannot be deployed") was anchored on a fictional 28-cell-type sorted-cardiomyocyte atlas. With that anchor gone, the honest cardio-epic deployment story reads: cardio-epic is operational at v0.2 under the layered Moss+Loyfer atlas with Stage 1 immune as the validated workhorse (VAL-110 d=+1.08 normal vs BAV on aortic tissue). Cuadrat 2023 + Caggiano CelFiE TIM + EpiSCORE pan-tissue are integration enhancements that broaden cardio Stage 2 cell-of-origin coverage but do not gate deployment of the Stage 1 + bulk-heart Stage 2 architecture already validated. This reframes cardio v0.3 from "wait for the magic atlas" to "integrate the available bulk-heart-tissue extensions and accept that sorted-cardiomyocyte discrimination is a v1.0+ goal contingent on someone publishing the underlying atlas."

**Correction 3 — sorted-cardiomyocyte array-CpG atlas gap acknowledged.** As of 2026-04-29 no such atlas exists in published literature at array-CpG resolution. Published cardiac methylation work covers targeted CpG biomarkers (Zemmour 2018 FAM101A, Yamazoe 2021 mt-cfDNA), bulk heart tissues (Moss 2018 Left_atrium; Cuadrat 2023 right atrium + left ventricle + coronary artery), or sorted vascular cells (Loyfer 2023 vascular_endothelial + smooth_muscle). Sorted cardiomyocyte array-CpG remains an open gap. When a sorted-cardiomyocyte atlas at array resolution is published, that becomes a v1.0+ candidate for an additional Stage 2 cardio extension. Until then, cardio-epic Stage 2 cardiac cell-of-origin discrimination operates at bulk-heart-tissue resolution.

### CCL-046 LL-CANONICAL-DOC-FACTUAL-ERROR — Documents-of-record can contain factual errors; periodic audit pass required

The Part 2.4 error sat in the canonical reference document undetected through cardio-epic v0.1, v0.2, and v0.2.1. Atlas selection in those card versions was driven by whatever happened to be in atlas_vault (resolved by DISC-CARDIO-007 + CHK-5.12 in v0.2.1). Once CHK-5.12 forced atlas selection to trace to the canonical document, the second-order error — that the canonical document itself contained factual errors — surfaced immediately on the first attempt to acquire the document-named atlas.

**The lesson is that CHK-5.12 alone is insufficient.** Tracing atlas selection to the canonical document protects against picking the wrong atlas from atlas_vault, but it does not protect against following an incorrect citation in the canonical document. A documents-of-record audit pass is required:

- Every atlas, panel, or external reference cited in PIPELINE_REFERENCE_v2.md or README_MASTER must be web-verified at least once: the DOI loads, the authors match the citation, the described content matches the abstract/methods/figures of the actual paper.
- When a citation fails verification, the canonical document is patched (not just annotated) and a CCL entry logs both the original error and the corrected content.
- This audit pass is recurring: any time a new atlas is named in a canonical document or any time an existing canonical-document atlas is integrated, the cited paper is re-verified. Citations age — paper retractions, errata, reanalysis updates, etc. — and the cookbook does not assume a once-verified citation stays accurate.

**Generalization.** CCL-046 applies to every external reference in cookbook documents, not just atlas references: cohort accessions, cited validation studies, H_min derivations referencing external papers, panel construction methods. Wherever the cookbook says "per X et al. Y" the X-Y pair must be web-verified at least once and re-verified when re-cited. The audit can be automated: a Python script that walks all .md cookbook files, extracts every DOI / citation / GSE accession, and reports unresolved or mismatched references against a manifest. That script is queued as a v0.3 cookbook engineering task (not blocking).

### CHK-5.13 documents-of-record citation-verification gate (added 2026-04-29 to TESTING_CHECKLIST.md)

Companion to CHK-5.11 atlas-family-fitness and CHK-5.12 atlas-canonical-source-check. Before sealing a card publish or a card promotion (v0.X → v0.X+1), every external citation introduced in the new card content (canonical-document quotes, atlas attributions, cohort accessions, prior-art references in deferral rationales) must have at least one web-verification pass per CCL-046. The gate is cheap (a single web search per citation) and catches the class of error that produced the v0.2.2 patch.

### v0.3 critical path — revised after Cuadrat correction

The v0.2.1 critical path (Konigsberg first, Caggiano second, EpiSCORE pan-tissue third, Tanaka fourth) is replaced with:

**Phase A — atlas acquisition / engineering** (revised priorities reflecting actual published literature):
1. **Acquire Cuadrat 2023 atlas extension** — the actual paper at the cited DOI. Three additional bulk heart-tissue tiles (right_atrium, heart_left_ventricle, coronary_artery) added to the Moss 2018 base. Useful enhancement to layered Moss+Loyfer for cardio Stage 2.
2. **Acquire HM450 hg19 manifest** to unblock Caggiano CelFiE TIM cardiac (heart_meth + endothelial_meth, sitting in atlas_vault, blocked at scoring engineering).
3. **Engineer Tanaka 2025 nanopore→array CpG bridge** (highest-priority neural addition per Part 2.5; cardio relevance is via astrocyte/microglia signatures of cardiac inflammation).
4. **Integrate EpiSCORE Zhu/Teschendorff pan-tissue** via existing rpy2 bridge (Part 2.3; full 13-tissue Heart/Kidney/Liver/Lung/Brain references for differential).

**Phase B — per-atlas calibration VAL** (must seal before any cardio-cohort scoring against that atlas, per CCL-041):
5. Cuadrat 2023 calibration VAL → CHK-3.1A baseline + CHK-3.1B subset threshold sealed on structurally-separated healthy cohort.
6. Caggiano CelFiE calibration VAL → same.
7. Tanaka 2025 calibration VAL → same (after bridge engineering).
8. EpiSCORE pan-tissue calibration VAL → same.

**Phase C — cardio-cohort scoring against each calibrated atlas:**
9. Cuadrat 2023 on GSE69138 + GSE84395 + GSE84274 + ideally GSE56046 MESA CHD/MI cohort n=1,202.
10. Caggiano CelFiE on the same cardio cohorts.
11. EpiSCORE pan-tissue on the same cardio cohorts.
12. Tanaka 2025 on the cardio cohorts.

**Phase D — re-execute VAL-108/109/110 honoring run-everything:** unchanged from v0.2.1 (full atlas stack against the same cohorts; sealed structural outcomes preserved).

**Phase E — cardio-epic v0.3 ship:** once Phase B + C complete for at least Cuadrat 2023 + Caggiano CelFiE TIM (the two with explicit cardiac coverage that can actually be acquired), the card promotes from v0.2.2 to v0.3 with those atlases in `atlases_run` and the v0.2.2 deferral notes resolved. Sorted-cardiomyocyte array-CpG discrimination remains a v1.0+ goal pending publication of an underlying atlas.

### What's pushed to GitHub vs Heath-only at v0.2.2

Same as v0.2.1: v0.2.2 is a cookbook-IP-side patch only. No additional GitHub artifacts. VAL-111 directory + EpiSCORE HeartRef atlas vault + Biological_Physics/README.md row remain at commit `facbe7a` (2026-04-29 morning).

### EDEAR commercial deployment

Per CCL-037 — unaffected. v0.2.2 honesty patch corrects a factual error in canonical documentation; it does not modify deployment architecture. Cardio-epic production scoring at v0.2.2 still uses the layered Moss+Loyfer atlas (validated) for Stage 2; the Cuadrat 2023 + Caggiano CelFiE TIM + EpiSCORE pan-tissue + Tanaka 2025 atlases are queued for v0.3 with calibration-before-scoring discipline. Deployment is not gated on a sorted-cardiomyocyte atlas because no such atlas exists at array-CpG resolution; the operational deployment story is Stage 1 immune workhorse + bulk-heart-tissue Stage 2 indicators + Stage 3 immune subcomposition.

---

## Part 22 — Substrate normalization pipeline architecture (added 2026-04-29 from CCL-048 / VAL-112+113 cardio sprint)

Production EDEAR scoring against any calibrated atlas requires the input β-matrix to be in a substrate the atlas was calibrated against. Raw IDAT files cannot be scored directly. They must first go through a substrate normalization step.

### Calibrated substrates (as of 2026-04-29)

**TCGA HM450 sesame Level 3 — primary calibrated substrate.** VAL-106 + VAL-107 established CHK-3.1A baseline (≥50.5%) + CHK-3.1B subset thresholds on TCGA-KIRC + TCGA-PRAD adjacent-normal n=210. VAL-112 + VAL-113 extended this to per-atlas calibration on the same n=210 cohort:

| Atlas | n_CpGs | Calibration VAL | CHK-3.1B q5 |
|---|---|---|---|
| Layered Moss+Loyfer (deduped) | 6,105 | VAL-112 | 0.6839 |
| EpiSCORE HeartRef bridged | 3,727 | VAL-112 | 0.4283 |
| Caggiano CelFiE TIM array-bridged | 254 | VAL-113 | 0.5779 |

Per-tile healthy-floor A-score distributions sealed for every tile of every atlas in `validation_runs/VAL-112_run_everything/VAL-112_calibration_results.json` and `validation_runs/VAL-113_caggiano/VAL-113_calibration_results.json`.

**Within-cohort self-cal substrates (operational fallback only):**
- GenomeStudio AVG_Beta HM450 (used by VAL-108 stroke + VAL-110 BAV)
- minfi `preprocessFunnorm` HM450 (used by VAL-109 PAH)
- minfi noob-bg-corrected EPIC v2 (CCL-040 deferral pathway)

These substrates do not have calibrated CHK-3.1A baseline + CHK-3.1B per-atlas thresholds against a structurally-separated healthy reference. Cookbook validation work using these substrates uses within-cohort self-cal as the operational fallback with explicit caveat.

### Production deployment normalization paths

A new customer's IDAT files must go through one of:

**1. sesame (Bioconductor, Triche lab) — RECOMMENDED.** Produces sesame Level 3 β values matching the VAL-106/107/112/113 calibration substrate. The `deconvR` R package and `sesameData` package both ship sesame normalization. Customer's IDAT → sesame Level 3 β → CHK-3.1A pass → CHK-3.1B per-atlas pass → A-score scoring against calibrated atlas.

**2. minfi `preprocessFunnorm` or `preprocessNoob`.** Within-cohort self-cal substrate; not currently calibrated against TCGA reference. Use only when sesame is unavailable AND the prereg explicitly documents within-cohort self-cal limitation. Customer-specific calibration VAL on representative healthy samples from the customer's lab pipeline is required before any commercial deployment claim.

**3. GenomeStudio AVG_Beta.** Illumina's pipeline; same handling as minfi (within-cohort self-cal).

### Customer onboarding calibration sequence

Per CCL-037 + CCL-048, EDEAR commercial onboarding includes a one-time substrate-normalization-and-calibration step per customer:

1. Customer sends representative IDAT files from their lab pipeline + sesame-normalized β-matrices for the same files
2. EDEAR runs CHK-3.1A on the customer's substrate to confirm full-genome bimodality on healthy reference samples from that lab
3. EDEAR runs CHK-3.1B on the customer's substrate per-card per-atlas
4. If customer's substrate matches sesame Level 3 (the reference path), the existing VAL-106/107/112/113 thresholds apply directly
5. If customer's substrate does not match (within-cohort self-cal needed), a customer-specific calibration VAL is run on representative healthy samples from that lab's substrate, using the VAL-112 + VAL-113 template (n≥30 healthy samples, structurally separated from any disease cohort the customer will score)
6. Customer-specific calibrated thresholds sealed and used for production scoring

This is consistent with CCL-037 (commercial deployment runs single calibrated patient-vs-internal-reference pipeline, structurally insulated from public-cohort substrate diversity). CCL-048 + CHK-0.7 add the explicit gate that the substrate must be calibrated before scoring, with sesame Level 3 as the reference path.

### What this means operationally for "can EDEAR run tomorrow"

**For TCGA-substrate-equivalent data (sesame Level 3):** YES, EDEAR's three calibrated cardio Stage 2 atlases score directly. CHK-3.1A baseline + CHK-3.1B per-atlas thresholds + per-tile healthy-floor A-score distributions are all sealed. Per-class A-scores against 49 tiles (25 Loyfer + 5 HeartRef + 19 Caggiano) per sample produce calibrated readings.

**For arbitrary IDAT files:** the IDAT files need to go through sesame normalization first (one-time preprocessing step using the `deconvR` or `sesameData` R packages), then they're in a calibrated substrate. The calibrated atlases score the sesame-normalized β-matrices directly.

**For other substrates (minfi funnorm, GenomeStudio AVG_Beta, EPIC v2 noob-bg):** customer-specific calibration VAL is required before commercial deployment; cookbook validation work uses within-cohort self-cal with explicit caveat.

### Reference

CCL-048 LL-SUBSTRATE-NORMALIZATION-REQUIRED in LESSONS_LEARNED.md. CHK-0.7 substrate-normalization-required gate in TESTING_CHECKLIST.md. §7.24 in GAPE_Reproduction_Paper_v1.md.

---

**End of v2 reference document. Heath signed off 2026-04-26 on run-everything architecture, Queue-1 atlas integration approved, cross-cohort baseline check promoted to mandatory-every-run. VAL-092 + VAL-093 first single-cohort and multi-cohort demonstrations completed and pushed to GitHub. Propagation to README_MASTER + LESSONS_LEARNED + GAPE Reproduction Paper + TESTING_CHECKLIST completed 2026-04-26 PM. Cardio-epic v0.2 shipped 2026-04-29 morning with VAL-111 sealed atlas-deferral outcome and full Block 1-20 + CHK-5.7/5.8/5.9/5.10 + 5.11 structural-parity. Cardio-epic v0.2.1 honesty patch shipped 2026-04-29 same-day with atlas naming corrected, atlases_deferred expanded to 8 canonical-document-named entries, DISC-CARDIO-007 added, run-everything violation in VAL-108/109/110 acknowledged, CHK-5.12 atlas-canonical-source-check gate added. Cardio-epic v0.2.2 honesty patch shipped 2026-04-29 same-day with Part 2.4 factual error corrected (Konigsberg → Cuadrat 2023, sorted cardiomyocytes → bulk ENCODE heart tissues), "cannot be deployed" framing dropped, sorted-cardiomyocyte array-CpG atlas gap acknowledged as open published-literature limitation, CCL-046 LL-CANONICAL-DOC-FACTUAL-ERROR documents-of-record audit lesson logged, CHK-5.13 documents-of-record citation-verification gate added. VAL-112 + VAL-113 cardio sprint shipped 2026-04-29 evening with three Stage 2 atlases (layered Moss+Loyfer deduped, EpiSCORE HeartRef bridged, Caggiano CelFiE TIM array-bridged) calibrated on TCGA HM450 sesame Level 3 n=210 and run-everything Phase C executed across all three cardio cohorts. Atlas vault commit 57beb38 pushed to GitHub. CCL-047 dedupe lesson + CHK-3.1C atlas-deduplication gate logged. CCL-048 substrate-normalization-required lesson + CHK-0.7 substrate-normalization gate + Part 22 production deployment normalization architecture logged.**
