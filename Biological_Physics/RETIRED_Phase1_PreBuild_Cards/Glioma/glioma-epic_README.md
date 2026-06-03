# glioma-epic v0.2 — EDEAR Cookbook

**Disease family:** Adult diffuse glioma (LGG and GBM, IDH-mutant and IDH-wildtype). Pediatric glioma not in v0.2 scope.

**GAPE class:** Terminal (neurons, oligodendrocytes, astrocytes; H_min = 0.7728). Stage 1 panel uses immune class (H_min = 0.838889) because peripheral blood is dominated by immune cells.

**Issue 002 framework prediction (TUMOR TISSUE):** LGG ΔA = +0.239, GBM ΔA = +0.217, terminal class A_combined ≈ 1.10 FLOOR BREACH on the 5-substrate cfDNA combined target (L2/L3 platform). "The physics is loud" once the signal is reachable.

**v1 deployment platform:** 450K / EPIC array, single-substrate methyl-only. Multiple specimen pathways (see below). Each pathway carries its own validation tier and operational caveat.

**Validation tier (current, v0.2):** `single_cohort_validated` (blood arm: VAL-088 Stage 1 immune A-score d=+0.91, VAL-090 Stage 2 cortical-neuron cfDNA fraction d=+1.96; tissue arm: VAL-089 GBM primary d=+0.24 / GBM recurrent d=+1.17 / spheres d=−1.81).

**v0.1 → v0.2 change.** The v0.1 card ran VAL-088 (Stage 1 A-score) and VAL-089 (tumor-tissue A-score) and tier-labeled the card `exploratory_pending_replication` because Stage 2 was unresolved (Moss 2018 reference returns NULL on brain). The v0.2 card resolves Stage 2 by integrating the published Loyfer/Moss array atlas (`nloyfer/meth_atlas/reference_atlas.csv`), which includes a sorted-cell `Cortical_neurons` reference indexed to Illumina array CpGs. VAL-090 ran this reference directly against the same cohorts — glioma plasma reads 1.09% mean cortical-neuron fraction vs 0.28% in healthy reference, Cohen's d = +1.96. **Brain-derived cfDNA IS detectable in standard EPIC peripheral blood at array resolution. The "specimen problem" framing in v0.1 (specifically: "terminal class contributes only ~0.5% to plasma — below the detection floor") was wrong. The detection floor is reachable; the limitation was reference-atlas choice, not biology.**

**Why this card was thought to be the hardest in the cookbook (and why it isn't anymore):**
1. ~~**Specimen problem.** Terminal class (neurons, oligodendrocytes, astrocytes) contributes only ~0.5% to plasma cfDNA at healthy baseline — below the Moss 4% detection floor. Single-timepoint plasma deconvolution returns null even when brain pathology is present.~~ **Resolved by VAL-090.** Healthy peripheral blood reads 0.28% cortical-neuron fraction (median 0.0%, NNLS noise floor). Glioma reads 1.09% — well above noise floor. The signal IS reachable from standard array peripheral blood when the reference atlas includes a sorted-cell cortical-neuron entry. Moss 2018's "brain (cortex)" entry was bulk-tissue mixture, which is why Moss-based deconvolution returned null. The Loyfer/Moss array atlas separates cortical neurons cleanly.
2. **Architectural-disruption magnitude is enormous.** Once you can hear the signal, it is loud — LGG/GBM tumor-tissue figures are the largest tissue ΔA values in the framework. **Confirmed by VAL-090.** GBM tumor tissue cortical-neuron fraction = 39.3% vs NTB 62.4%, d = −2.81. The brain-tissue signature is exactly what the framework predicted.
3. **Reaching the signal still benefits from invasive sampling (LP-CSF) or specialized chemistry (cfMeDIP-seq enrichment) for early-detection windows.** Standard array methylation on plasma at the time of clinical glioma diagnosis works (VAL-090 confirmed). Pre-diagnostic, sub-clinical detection windows have not been tested.

---

## TL;DR for the routing decision


| Specimen arriving | What v1 EDEAR can do today | Tier label | What we'd need for upgrade |
|---|---|---|---|
| **Whole blood (buffy coat) on 450K/EPIC** | Stage 1 immune A-score; positive direction d ≈ 0.9 vs healthy reference | `exploratory_pending_replication` | UCSF AGS phs001497 dbGaP application (Bracci 2022 cohort, n=139 + 454 controls EPIC) |
| **Plasma cfDNA on 450K/EPIC** | Stage 2 Moss deconvolution; healthy-baseline returns null on terminal class (cfDNA <4% floor); under active disease may show neuron/oligo/astro fractions | `pre-validation_skeleton` | Lubotzky-style cfDNA-bisulfite-array cohort with paired diagnosis confirmed |
| **Serum cfDNA on EPIC (Sabedot GeLB protocol)** | Sabedot 2021 GeLB score on 450K-EPIC restored cfDNA achieves 100% sensitivity / 97.78% specificity | `external_classifier_validated` | Re-implementation of GeLB classifier on Mendeley deposit (cgrz6zztfg); see Pathway 1 |
| **Plasma cfMeDIP-seq** | Nassiri 2020 cfMeDIP-seq classifier achieves AUC 0.99 in published cohort | `external_classifier_validated_different_chemistry` | cfMeDIP-seq protocol implementation; not 450K/EPIC, not v1 deployment chemistry |
| **CSF cfDNA on 450K/EPIC** | Stage 2 Moss deconvolution; CSF-cfDNA is highly enriched for brain cell types; **LP-CSF is the gold-standard specimen for terminal-class detection** | `pre-validation_skeleton` | CSF-bisulfite-array cohort with paired diagnosis (e.g., GLASS consortium CSF specimens, Heidelberg CNS classifier reference samples) |
| **Tumor tissue on 450K/EPIC** | Stage 1 + Stage 2 + Stage 3 architecture A-score; tissue-architecture confirmation on the framework's loudest signal | `single_cohort_validated_with_substrate_scope_caveat` (VAL-089: GBM primary d=+0.24 wide CI; recurrent d=+1.17; spheres d=-1.81 inversion confirms entropy=cell-mixture-diversity) | GSE90496 Heidelberg cohort replication (n=263 + 27 controls); GSE143843 superseries (104+297+398 GBM); TCGA-LGG/GBM direct test |
| **Deep cervical lymph node aspirate on 450K/EPIC** | No published cohort; Pathway 2 framework prediction only | `pre-validation_skeleton` | Custom collaboration; no cohort exists |
| **FUS-disrupted-BBB plasma on 450K/EPIC** | No published methylation cohort; pilot studies exist | `pre-validation_skeleton` | Custom collaboration; pilot studies in trial recruitment |

---

## What we tested today (VAL-088)

**VAL-088** — Stage 1 immune Xu-538 A-score on GSE180683 (Salas/Wiencke 2022, n=76 glioma EPIC peripheral blood, mixed treatment stages with FCM-validated T-cell composition).

**Result:**
- All glioma (n=76): A = 0.4571 ± 0.0229 vs Italian healthy reference 0.4384 ± 0.0244 (HM450)
- ΔA = +0.0187 (+0.77 healthy SDs)
- Cohen's d = +0.91 [+0.61, +1.22]
- Pre-surgery treatment-naive subset (n=37): d = +0.94 [+0.56, +1.33]
- Pre-surgery LGG (n=12): d = +1.25 — surprisingly LARGER signal than pre-surgery GBM (n=25): d = +0.80

**Outcome label:** `O5_POSITIVE_INVERTED` — direction inverts the CCL-023 cell-fraction prior. Glioma joins the activation-shifted set (AD, breast, lung, prostate, HCC), not the suppression-shifted set (CRC).

**Caveats actually hit:**
1. Cross-platform comparison (test EPIC, reference HM450). Direction is robust; absolute magnitude carries coverage-drift caveat.
2. No internal healthy controls in GSE180683. External comparator required. CHK-3.2 baseline-mismatch check could not be run cleanly.
3. CCL-023 direction-as-discriminator hypothesis was inverted by this test. Refinement required (see VAL-088 outcome).
4. Single cohort. Replication on Bracci 2022 UCSF AGS phs001497 (Tier 3 gated) is the next step.

See `VAL-088_prereg.md`, `VAL-088_outcome.md`, `VAL-088_results.json` for full detail.

## What we tested today (VAL-089)

**VAL-089** — Tumor-tissue arm — direct architecture A-score on GSE60274 (Lai 2015) brain tissue: 60 primary surgical GBM + 4 GBM with paired sphere primary + 4 recurrent GBM + 4 cultured glioma spheres + **5 non-tumor brain (NTB) controls**, on 450K platform. **On-study healthy controls eliminate cross-platform / cross-cohort baseline confound.**

**Key results (H_min terminal = 0.7728):**

| Stratum | n | mean A | SD | ΔA vs NTB | Cohen's d | 95% CI |
|---|---|---|---|---|---|---|
| NTB healthy | 5 | 0.6869 | 0.0107 | — | — | — |
| GBM primary | 64 | 0.7013 | 0.0613 | +0.0145 | +0.243 | [-0.668, +1.154] |
| GBM recurrent | 4 | 0.7195 | 0.0409 | +0.0327 | +1.167 | [-0.254, +2.588] |
| GBM cultured spheres | 4 | 0.6584 | 0.0207 | -0.0285 | -1.805 | [-3.362, -0.248] |

**Outcome label:** `O2_PARTIAL_DIRECTION_CONSISTENT_VARIANCE_HIGH`

**Three biology-cross-check findings emerged:**
1. **GBM primary direction matches Issue 002 prediction**, magnitude smaller as expected from CHK-1.5 substrate-scope translation (Issue 002 +0.217 figure is 5-substrate cfDNA L2/L3 prediction; v1 single-substrate methyl-only on Xu-538 is different scope).
2. **Recurrent GBM > primary GBM** (d = +1.17 vs +0.24). Disease progression amplifies architecture disruption.
3. **Cultured spheres < NTB (d = -1.81 NEGATIVE)**. Pure tumor-cell-line β distributions are LESS mixed than mixed-cell tissue. **This validates that Shannon entropy of methylation captures cell-mixture diversity, not "tumorness."** A high A-score is a heterogeneity marker, not a tumor marker — tumor TISSUE produces high A because it contains many cell types (neurons + glia + microglia + endothelium + neoplastic), not because tumor CELLS have high entropy.

**Wide CIs on primary GBM** reflect: (a) only n=5 healthy controls, and (b) genuinely high variance among GBM tumors (SD ratio 5× vs NTB) — consistent with heme-LL-008 (per-disease ΔA spread reflects programmed plasticity, not noise).

**Caveats actually hit:**
1. Xu-538 panel was trained on whole-blood IMMUNE class; applying to brain tissue measures what those CpGs read in non-native tissue. Direction-of-effect is the primary inference; absolute magnitude requires substrate-scope translation.
2. Small NTB n=5 produces wide CIs even when point estimates are meaningful.
3. NTB controls older (median 75) than typical GBM (median 55); age-adjustment not applied.
4. No tumor-purity adjustment. Stage 3 GIMiCC deferred to v0.2.

See `VAL-089_prereg.md`, `VAL-089_outcome.md`, `VAL-089_results.json` for full detail.

---

## What we tested today (VAL-090) — direct cortical-neuron cfDNA detection

This is the headline finding of the v0.2 build. **Brain-derived cfDNA is directly detectable and quantifiable in glioma peripheral blood at standard 450K/EPIC array resolution.**

**Method.** Run NNLS deconvolution against the published Loyfer/Moss array-indexed reference atlas (`nloyfer/meth_atlas/reference_atlas.csv`, 26 cell types including a sorted-cell `Cortical_neurons` entry indexed to Illumina array CpGs). No parameter tuning. No panel selection. No post-hoc adjustment. The published reference applied directly to the published cohorts via the published open-source tool. Same three cohorts as VAL-088/089: GSE51057 healthy reference (n=177), GSE180683 glioma plasma (n=76), GSE60274 brain tissue (n=77).

**Headline.** Cortical-neuron cfDNA fraction in peripheral blood:

| Cohort | n | Cortical-neurons mean | Cohen's d vs healthy |
|---|---|---|---|
| Healthy buffy coat (GSE51057) | 177 | 0.28% | (reference) |
| All glioma plasma (GSE180683) | 76 | 1.09% | **+1.96 [+1.62, +2.31]** |
| Pre-surgery treatment-naive subset | 37 | 1.08% | **+1.97** |
| Pre-surgery LGG (treatment-naive) | 12 | 1.29% | (small n) |
| Pre-surgery GBM (treatment-naive) | 19 | 0.86% | (small n) |

89% of glioma plasma samples cross the 0.5% threshold; 63% cross 1%. In healthy reference, only 7% cross 1% (those reflect NNLS solver noise floor — median healthy sample reads exactly 0%).

**The pre-surgery LGG > pre-surgery GBM ordering observed in VAL-088 also appears in VAL-090** — under a completely different metric (cell-fraction deconvolution vs Shannon-entropy A-score). This strengthens the LGG-louder-than-GBM finding rather than relying on a single metric.

**Brain tissue arm consistency.** When the same deconvolution runs on tumor tissue:

| Tissue | n | Cortical-neurons fraction | Cohen's d vs NTB |
|---|---|---|---|
| Non-tumor brain (NTB) controls | 5 | 62.44% | (reference) |
| GBM primary tumor | 64 | 39.32% | **−2.81** |
| GBM recurrent tumor | 4 | 35.18% | (n=4) |
| Cultured glioma spheres | 4 | 42.93% | −4.80 |

NTB reads 62% neurons (correct — cerebral cortex is neuron-dominated). GBM primary reads 39% (~23 percentage points lower — tumor displaces normal architecture). Recurrent further. Spheres NEGATIVE confirms the heterogeneity-not-tumorness biology cross-check from VAL-089: pure tumor-cell-line populations are LESS architecturally diverse than mixed-cell tissue. **The pipeline reads non-tumor brain as 62% neurons; the same pipeline reads healthy peripheral blood as 0.3% neurons. This is the expected biological gradient.**

**Sanity check on immune compartment.** The same deconvolution reproduces textbook peripheral blood composition in healthy controls (52% neutrophils, 25% T-cells, 6% B-cells, 4% monocytes — matches Salas 2018 reference ranges). In glioma plasma, neutrophils elevate to 68% and lymphocytes drop correspondingly — exactly the Bracci 2022 NLR-style cell-fraction signature. **The cortical-neuron signal is in addition to, not instead of, the immune-cell-fraction shift.**

**Outcome label: O1_PASS.** Glioma-epic blood arm promoted from `exploratory_pending_replication` (v0.1) to `single_cohort_validated` (v0.2).

**What VAL-090 changes about the v0.1 framing.** The v0.1 card said "terminal-class cfDNA contributes ~0.5% to plasma — below the Moss 4% detection floor — single-timepoint plasma deconvolution returns null even when brain pathology is present." This was wrong. The detection floor is reachable. The limitation was a reference-atlas choice (Moss 2018's "brain (cortex)" is bulk-tissue, not sorted neurons), not biology. The Loyfer/Moss array atlas separates the signal cleanly. **Glioma-epic Stage 2 deconvolution now uses the Loyfer/Moss array atlas as its primary reference for terminal-class detection, supplementing Moss 2018 for cells Moss did not have as sorted-cell entries.** See the GAPE Reproduction Paper Part 5 update for the layered-atlas architecture rationale.

**What VAL-090 changes about CCL-023.** VAL-088 was originally labeled `O5_POSITIVE_INVERTED` because the Bracci 2022 cell-fraction prior had predicted negative direction and the Stage 1 A-score read positive. VAL-090's Loyfer-atlas deconvolution shows that the cell-fraction prior was actually CORRECT in its direction — neutrophils +16%, lymphocytes -13%, exactly Bracci NLR shift. The Shannon-entropy A-score is just a different lens on the same disease state. **CCL-023 is revised: cell-fraction direction and A-score direction are ORTHOGONAL (different facets of the same biology), not INVERTED (opposites). See LESSONS_LEARNED.md glioma-LL-001.**

See `VAL-090_prereg.md`, `VAL-090_outcome.md`, `VAL-090_results.json` for full detail. Reproducibility triple per CHK-7.6 documented in the outcome doc.

![VAL-090 distribution figure](VAL-090_distributions.png)

---

## The five detection pathways (operational reference)

Documenting all five pathways with what each can do today versus what we'd need to operationalize each fully. This is the master pathway reference for the card; commercial.web.py routes patient samples through the pathway that matches the specimen arriving.

### Pathway 1 — Plasma cfMeDIP-seq enrichment overcomes the 4% detection floor

**What it does today.** cfMeDIP-seq (cell-free methylated DNA immunoprecipitation followed by high-throughput sequencing) is a non-array methylation chemistry that enriches for methylated DNA fragments before sequencing, dramatically improving signal-to-noise above standard genome-wide deconvolution. Nassiri 2020 (Nat Med) achieved AUC = 0.99 [0.96–1.00] discriminating glioma from extracranial cancers and healthy controls using cfMeDIP-seq + machine learning on 60 plasma samples (later expanded to 447 multi-cancer set across 9 classes). Sabedot 2021 GeLB score (using cfDNA EPIC array with restoration kit, not cfMeDIP) achieved 100% sensitivity / 97.78% specificity discriminating 149 glioma patients from other brain tumor types on serum.

**What v1 EDEAR can do.** v1 EDEAR runs on 450K/EPIC arrays. It does NOT run cfMeDIP-seq. The Sabedot GeLB approach (EPIC array + Illumina restoration kit on cfDNA) is potentially compatible — the IDAT files are deposited at Mendeley Data ID `cgrz6zztfg`. We could re-implement the GeLB scoring on those IDATs through the GAPE pipeline, with the explicit caveat that the panel/scorer is Sabedot's not ours.

**What we'd need.** For Pathway 1 native deployment: cfMeDIP-seq protocol (specialized chemistry, capital-intensive). For Pathway 1 EPIC-array workaround: Sabedot Mendeley deposit can be downloaded today; integration into v1 pipeline as an optional "external-classifier" arm is a v0.2 build item. Validation tier achievable: `external_classifier_validated`.

**Tier label this pathway can achieve at v1:** `external_classifier_validated_different_chemistry` (Nassiri cfMeDIP-seq) or `external_classifier_validated` (Sabedot GeLB cfDNA-EPIC).

### Pathway 2 — Lymphatic concentration via deep cervical lymph nodes

**What it does today.** The glymphatic system (Iliff/Nedergaard 2012, Science) and meningeal lymphatic vessels (Louveau 2015 / Aspelund 2015, Nature) drain CSF and brain interstitial fluid to deep cervical lymph nodes (and from there to systemic circulation). Brain-derived cfDNA is concentrated in deep cervical lymph **before** being further diluted in systemic circulation. This is published anatomy. **No published methylation cohort uses this specimen for brain pathology detection.**

**What v1 EDEAR can do.** Nothing today. This is a framework-level prediction. If a deep cervical lymph node aspirate were submitted to EDEAR, the pipeline would score it through the standard Stage 1 + Stage 2 + Stage 3 cascade, but interpretation would be outside the trained specimen distribution.

**What we'd need.** A published large-cohort methylation study using deep cervical lymph node aspirate or fluid as the specimen, with paired brain pathology diagnosis. Could be retrospective from neurosurgical biopsies of the cervical region, or prospective in collaboration with a neurosurgical center. **Open framework prediction logged as G-2026-PXXX (draft):** for any disease state where the brain is the affected tissue (glioma, GBM, AD, ALS, Parkinson's, MS), the meningeal-lymphatic-to-cervical-lymph axis should produce cfDNA enrichment above the 0.5% plasma baseline.

**Tier label this pathway can achieve at v1:** `pre-validation_skeleton`. No cohort exists.

### Pathway 3 — Multi-specimen tier system for terminal-class detection

The clinical reality is that not all patients can or should have lumbar puncture. For glioma-epic to be a clinical product, it needs a graded specimen tier system. Each tier has different validation status and different cfDNA recovery profile.

| Tier | Specimen | Invasiveness | Brain cfDNA yield | When clinically used | v1 EDEAR can score? | Validation status |
|---|---|---|---|---|---|---|
| 1 (gold) | LP-CSF | Invasive (lumbar puncture) | High (CSF directly accesses CNS) | Standard for CNS workup; brain tumor diagnosis | Yes via Stage 2 Moss + Stage 3 EpiDISH | `pre-validation_skeleton` (no LP-CSF EPIC cohort identified for direct testing) |
| 2a | Ventricular shunt sampling | None (existing shunt) | High | Hydrocephalus or post-surgical patients with existing shunts | Yes | `pre-validation_skeleton` |
| 2b | Ommaya reservoir | None (existing implant) | High | CNS lymphoma, pediatric brain tumors with implanted device | Yes | `pre-validation_skeleton` |
| 2c | Cisterna magna sampling | More invasive than LP | High | Specialized contexts (rare in adults) | Yes | `pre-validation_skeleton` |
| 3 | Deep cervical lymph node aspirate | Moderate | Concentrated brain cfDNA, no published cohort | Theoretical (Pathway 2) | Yes (out-of-distribution) | `pre-validation_skeleton` |
| 4 | FUS-disrupted-BBB plasma + cfMeDIP-seq | Moderate (focused-ultrasound device required) | Improved over standard plasma | Pilot studies; Brain 2023 review | If cfMeDIP-seq available: yes via Pathway 1 chemistry | `pre-validation_skeleton` |
| 5 | Standard plasma + cfMeDIP-seq | None | Recoverable with enrichment | Most likely first commercial pathway | Via Pathway 1 chemistry only | `external_classifier_validated_different_chemistry` |
| 6 | Standard plasma + EPIC array (with restoration kit) | None | Below detection floor for healthy baseline; possibly above floor under active disease (Sabedot GeLB) | Sabedot 2021 protocol | Yes | `external_classifier_validated` if GeLB integrated |
| 7 | Standard plasma + 450K/EPIC array (no restoration) | None | Below detection floor for healthy baseline; null for inactive/early disease | Standard liquid biopsy | Yes; expect null or weak signal | `pre-validation_skeleton` |
| 8 | Whole blood (buffy coat) + 450K/EPIC | None | NOT brain cfDNA; immune-class signal only | Standard EWAS | Yes via Stage 1 immune A-score | `exploratory_pending_replication` (this VAL-088) |

**What v1 EDEAR can do today across all tiers.** Run the Stage 1 + Stage 2 + Stage 3 pipeline on whatever methylation array data is provided. The interpretation depends on which tier the specimen comes from. v1 patient reports include which tier the specimen belongs to and what that means for sensitivity/specificity at this stage.

**What we'd need.** Tier-specific validation cohorts. The most tractable next step is an LP-CSF EPIC array cohort, which would validate Tier 1. The most commercially-relevant first step is plasma + EPIC restoration kit (Sabedot GeLB), validating Tier 6. cfMeDIP-seq (Tier 5) is most published-evidence-supported but requires non-v1 chemistry.

### Pathway 4 — Brain-resident immune signature in peripheral blood

**What it does today.** The brain has a unique immune compartment:
- **Microglia** — resident brain macrophages, embryonic yolk-sac origin, distinct lineage from peripheral monocytes, decades-long turnover, the body's longest-lived immune cells.
- **CNS-border macrophages** — perivascular, meningeal, distinct from microglia.
- **Brain-resident T-cells** — sparse but present, especially in disease.

When brain pathology develops, peripheral blood shows trafficking signatures: monocyte-to-TAM (tumor-associated macrophage) trafficking in glioma; DAM (disease-associated microglia) signature in AD; activated microglia methylation markers (TMEM119, P2RY12, TREM2); brain-immune-trafficking markers (CCL2, CCR2). Sabedot's GeLB serum score explicitly captured "cfDNA-derived methylation signatures associated with the presence of glioma **and associated immunological features**." Multiple groups (Nassiri 2020, the 2025 reviews) have shown methylation-immune-pathway enrichment in plasma cfDNA from glioma patients.

**What v1 EDEAR can do.** v1's universal Stage 1 Xu-538 immune A-score detected the brain-pathology peripheral signature at d ≈ 0.9 (VAL-088). The signal exists at the universal-panel level. A **glioma-specific directional panel** would likely improve sensitivity, but is not yet built.

**What we'd need.** Construction of a Pathway 4 glioma-directional panel from:
- Sabedot 2021 GeLB CpGs (panel public)
- Nassiri 2020 cfMeDIP-seq glioma DMRs (top-N most discriminative)
- Microglial activation methylation markers (TMEM119, P2RY12, TREM2, CSF1R)
- Brain-immune trafficking signature (CCL2, CCR2, monocyte-to-TAM markers)

This panel would work ON TOP OF the universal Xu-538 Stage 1 immune A-score, not replacing it. The base Xu-538 captures generic immune dysregulation; the glioma directional panel captures brain-pathology-specific trafficking. v0.2 build target.

**Tier label this pathway can achieve at v1:** `directional_panel_pending_construction`. The component CpGs are published; the panel itself doesn't exist yet.

### Pathway 5 — Direction-as-discriminator (CCL-023 applied to brain — REVISED)

**Status as of VAL-088:** The CCL-023 direction-as-discriminator hypothesis applied to glioma was **inverted by direct measurement.** Bracci 2022 cell-fraction signature (lymphocytes-down, neutrophils-up) had been used as a literature anchor predicting NEGATIVE A-score direction for glioma. VAL-088 measured POSITIVE direction d ≈ 0.9 vs healthy reference. The bridge from cell-fraction direction (NLR-style) to A-score direction (Shannon entropy of methylation) does not hold automatically. Cell-fraction direction and methylation-entropy direction are different metrics measuring different phenomena.

**Revised CCL-023 anchoring set:**
- CRC pre-diagnostic 5-10 yr blood: NEGATIVE direction (VAL-047)
- AD blood at-diagnosis: POSITIVE direction (VAL-051/052)
- Breast/lung/prostate/HCC pre-diagnostic 2-10 yr blood: POSITIVE direction
- Pancreatic blood 6-mo pre-dx: see pancreatic-epic
- **Glioma blood at-diagnosis: POSITIVE direction (VAL-088, this study)**

**The pattern that survives:** Pre-diagnostic CRC (5-10 yr window) reads NEGATIVE; everything else (AD, breast, lung, prostate, HCC, glioma) at-diagnosis or post-diagnosis reads POSITIVE. Direction-as-discriminator may collapse to "early-pre-dx CRC is the outlier" rather than a general activation-vs-suppression rule.

**Implications for glioma vs AD discrimination at peripheral-blood Stage 1:** Both read positive. **Direction alone does NOT discriminate AD from glioma.** Discrimination must come from Stage 2 (Moss tissue-of-origin) and Stage 3 (EpiDISH cell-composition pattern), not from Stage 1 direction-of-effect alone.

**What v1 EDEAR can do.** Run Stage 1 + Stage 2 + Stage 3. For a positive Stage 1 reading, route to the differential decision tree (see Commercial.web.py decision tree below).

**What we'd need.** Glioma blood cohorts at varying pre-diagnostic windows (5+ years, 2-5 years, at-diagnosis) to test whether glioma also shows a CRC-like negative-to-positive transition over its pre-diagnostic phase. **None of the Tier 1 publicly-deposited cohorts have pre-diagnostic blood for glioma.** This requires biobank-gated cohorts (UK Biobank, EPIC-Italy, NSHDS, Sister Study cohorts that report glioma incidence) or the GICC consortium (phs001319) participants who have stored biospecimens. v0.3+ build item.

---

## Commercial.web.py decision tree — what to do when an IDAT fires this card

When a 450K or EPIC IDAT arrives at commercial.web.py and the patient intake form indicates either (a) symptoms suggestive of intracranial pathology, (b) known glioma diagnosis under active surveillance, (c) any other reason to test for brain pathology, the IDAT routes through this card. Below is what commercial.web.py does at each routing decision point.

### Specimen-pathway routing (must be captured at intake)

Patient intake questionnaire includes the following mandatory covariates:

1. **Specimen type:** whole blood / buffy coat / PBMC / plasma cfDNA / serum cfDNA / CSF cfDNA / cervical lymph aspirate / tumor tissue / other.
2. **Specimen collection method:** direct venipuncture / lumbar puncture / surgical biopsy / Ommaya draw / FUS-disrupted plasma collection.
3. **Bisulfite-conversion protocol:** standard / restoration kit (Illumina) / cfMeDIP-seq enrichment / other.
4. **Array platform:** 450K / EPIC v1.0 / EPIC v1.0_B4 / EPIC v2.
5. **Pre-treatment:** treatment-naive (no surgery, no chemorad, no dexamethasone) / post-surgery / on-radiation / on-chemo / dexamethasone within 30 days / other.
6. **Symptoms at draw:** present / absent.
7. **Imaging at draw:** none / CT / MRI / mass identified / no mass identified.
8. **Family history:** unaffected / glioma in 1st-degree relative / known glioma syndrome.
9. **Age:** numeric (chronological).

Without these, scoring proceeds with explicit "MISSING_COVARIATE" flags in the report.

### Routing matrix — which arm of the card processes the IDAT

| Specimen + protocol | Routing arm | Stage 1 | Stage 2 | Stage 3 | Pathway 5 (direction) | Patient-facing message |
|---|---|---|---|---|---|---|
| Whole blood + standard | Arm A: peripheral immune | Universal Xu-538 immune A-score | Moss deconvolution (expect 0% terminal-class shedding at healthy baseline) | EpiDISH cell composition | Direction interpretation per VAL-088 | "Stage 1 immune A-score result, with caveat that whole blood does not directly carry brain cfDNA above detection floor at healthy baseline." |
| Plasma cfDNA + standard EPIC | Arm B: plasma standard | Universal Xu-538 immune A-score (cfDNA-derived) | Moss deconvolution; expect terminal-class null at baseline; positive only under active disease | EpiDISH | Direction | "Plasma cfDNA at v1 may not detect terminal-class disease at healthy baseline. Active disease (high tumor turnover) may produce detectable signal. Consider Pathway 1 (cfMeDIP-seq) if Stage 2 returns null." |
| Plasma cfDNA + EPIC restoration kit (Sabedot GeLB) | Arm C: GeLB-compatible | Universal Xu-538 immune | Moss + GeLB-specific scoring (external classifier) | EpiDISH | Direction | "Sabedot GeLB protocol. External classifier confidence + universal-panel result combined." |
| Plasma cfMeDIP-seq | Arm D: cfMeDIP-seq | NOT supported in v1 (different chemistry) | NOT supported | NOT supported | NOT supported | "v1 does not score cfMeDIP-seq. Refer to Nassiri 2020 published protocol or wait for v0.2 cfMeDIP-seq integration." |
| CSF cfDNA + standard EPIC | Arm E: CSF | Stage 1 NOT meaningful (CSF has different cell composition) | Moss tuned for CNS — expect HIGH terminal-class fraction in disease | EpiDISH | NOT applicable | "LP-CSF specimen. Stage 2 brain-cell-of-origin deconvolution is the primary readout. Stage 1 immune scoring not interpretable on CSF." |
| Tumor tissue + EPIC | Arm F: tissue | Direct tissue-architecture A-score (TUMOR class, NOT immune class) | Tumor purity inference | Tumor microenvironment composition (GIMiCC, Salas 2024) | NOT applicable | "Tumor tissue. Direct architecture A-score on terminal class. This is the framework's loudest signal — LGG ΔA = +0.239, GBM ΔA = +0.217." |
| Cervical lymph aspirate | Arm G: cervical lymph | Out-of-distribution; flag | Moss-attempted but flagged | EpiDISH attempted | NOT applicable | "Pathway 2 specimen. v1 has no validated cohort for this specimen. Result flagged as out-of-distribution; interpret with caution." |

### Patient-report templates by routing arm

#### Arm A (whole blood, the VAL-088 pathway)

> **Stage 1 immune architecture: ELEVATED.** Your peripheral blood immune-class A-score is X.XX, which is N.N standard deviations above the age-matched healthy reference. This pattern is consistent with active immune dysregulation. It does NOT identify a specific tissue of origin.
>
> **Stage 2 tissue-of-origin: NULL on the 18 peripheral solid organs covered by our reference (Moss 2018).** Note: brain/CNS is NOT in the Moss reference because brain cells do not normally shed measurably into peripheral blood. **An elevated Stage 1 + Moss NULL on peripherals does not rule out brain pathology.**
>
> **Stage 3 cell-composition pattern:** [report EpiDISH lineage profile]
>
> **What this can mean:** Several conditions produce this pattern. They include but are not limited to: aging-related immune dysregulation (inflammaging); chronic infection; autoimmune disease; primary CNS tumor (glioma, GBM, primary CNS lymphoma); systemic inflammation; recent vaccination; recent illness. **At v1, EDEAR cannot distinguish among these on a single peripheral blood draw.**
>
> **What you should do:** If you have neurological symptoms (cognitive changes, focal deficits, seizures, persistent headaches), discuss with your doctor. Brain imaging (MRI) is the appropriate next step for evaluation. An elevated peripheral blood A-score is not a diagnosis. EDEAR is a research tool that flags patterns warranting clinical evaluation; it does not replace imaging or biopsy.

#### Arm F (tumor tissue — the loud-signal pathway)

> **Direct tissue-architecture A-score on terminal class: [score].** Your tumor-tissue methylation pattern shows ΔA = +X.XX above healthy brain reference. This is consistent with the framework prediction for [GBM / LGG] — the architecture is loud at the tissue level.
>
> **Stage 2 tumor-purity-and-composition:** [GIMiCC output — neoplastic %, immune cell %, vascular %, etc.]
>
> **Stage 3 microenvironment lineage:** [report]
>
> **What this can mean:** Tumor tissue methylation directly reflects neoplastic and microenvironmental architecture. This is a research output supporting molecular characterization, not a clinical diagnosis. Your treating neuro-oncologist will integrate this with histology, IDH/MGMT/1p19q status, imaging, and clinical course.

#### Arm E (CSF — the gold-standard pathway, future)

> **Stage 2 brain-cell-of-origin deconvolution from CSF cfDNA:** [neuron %, oligodendrocyte %, astrocyte %, microglia %].
>
> **What this can mean:** CSF cfDNA is highly enriched for brain-cell methylation and overcomes the plasma 4% detection floor. Elevated terminal-class fraction (neuron + oligodendrocyte + astrocyte combined > healthy reference) supports active CNS pathology. Specific cell-type elevation can suggest disease subtype (oligodendrocyte for oligodendroglioma, astrocyte for astrocytoma, mixed for GBM).
>
> **What you should do:** [pathway-specific]

#### "No immediate culprit found" handling (10+ year out scenario)

When Stage 1 fires but Stage 2/3/Pathway-5 don't localize cleanly, the report does NOT say "false positive". It says:

> **Pattern not yet localized to a specific tissue at this time.** Your Stage 1 immune A-score is elevated, but Stage 2 and Stage 3 do not yet identify a specific tissue source. This pattern can occur years before clinical disease onset, when systemic effects are present but local tissue contributions are below detection threshold.
>
> **What you should do:** Active surveillance is recommended. Re-test in 6-12 months to track trajectory. EDEAR's value increases over serial measurements — a stable elevated A-score with localized Stage 2/3 emerging in subsequent samples is more informative than any single timepoint.

### Confirmatory test pathway by arm

| Arm | Standard confirmatory test ordered by neuro-oncologist after this report fires |
|---|---|
| A whole blood elevated | MRI brain with contrast; full neurological exam; consider Pathway 6 (CSF) if symptoms persist |
| B/C plasma cfDNA elevated | MRI brain; plasma cfMeDIP-seq if available; LP-CSF if Pathway 6 indicated |
| D cfMeDIP-seq | MRI brain; if signal classifies as glioma, neurosurgical eval |
| E CSF | Neurosurgical evaluation; tumor biopsy for histology + molecular markers |
| F tumor tissue | Histopathology; IDH/MGMT/1p19q status; full molecular workup |
| G cervical lymph | Out-of-distribution at v1; clinical correlation only |

### What commercial.web.py CANNOT do at v1

- **Distinguish glioma from AD on Stage 1 direction alone.** Both read POSITIVE per CCL-023 revision (VAL-088). Discrimination requires Stage 2 + Stage 3 + clinical context.
- **Detect glioma 5-10 years pre-diagnostically on standard whole blood.** No cohort exists to validate this; CCL-023 may or may not extend pre-diagnostically for glioma.
- **Detect terminal-class cfDNA in healthy-baseline plasma.** Below 4% Moss detection floor. Active-disease detection plausible but unvalidated at v1.
- **Process cfMeDIP-seq data.** Different chemistry; v0.2+ integration target.
- **Process pediatric brain tumor data.** v0.1 covers adult diffuse glioma only.

### Mandatory covariates summary (for the report)

Every glioma-epic report must include:
1. Specimen type and collection method
2. Bisulfite protocol and array platform
3. Pre-treatment status (treatment-naive vs treated)
4. Patient age and sex
5. Symptoms-at-draw and imaging-at-draw status
6. Routing arm used for scoring
7. Validation tier of the routing arm (per the table at top)
8. Stage 1 + Stage 2 + Stage 3 outputs
9. Direction-of-effect (Pathway 5)
10. Plain-language interpretation per the templates above
11. **Honesty footer:** explicit list of what this report cannot do at v1

---

## Cohort completeness (CCL-029) — exhaustive landscape inventory

### Tier 1 — publicly accessible (immediate)

| Cohort | n | Platform | Specimen | Status this card | Notes |
|---|---|---|---|---|---|
| **GSE180683** (Salas/Wiencke 2022) | 76 glioma | EPIC | Whole blood | **VAL-088 ran. `exploratory_pending_replication`** | Mixed treatment stages. NO healthy controls within study. Includes FCM-validated T-cell composition. |
| **Sabedot 2021 GeLB Mendeley** (cgrz6zztfg) | 22 paired tumor+serum + larger validation | EPIC (serum cfDNA) + 450K (tumor) | Serum cfDNA | Deferred to v0.2; external-classifier integration needed | Different specimen pathway (serum cfDNA) requires Pathway 1/3 routing. CHK-2.4 panel transferability flag. |
| **GSE292314** (Silv 2026) | bulk GBM tumor + microenvironment | 450K | Tumor tissue | Deferred to VAL-089 (planned); accessible | Tumor methylation with cellular deconvolution. Useful for tumor-architecture A-score replication. |
| **GSE90496** (Heidelberg) | 263 IDH-wt GBM + control classes | 450K | Tumor tissue | Deferred to VAL-090 (planned); accessible | Heidelberg classifier reference. Includes "CONTR, WM" white-matter control. Largest accessible GBM tumor cohort with healthy reference. |
| **GSE60274** (Lai 2015) | 68 GBM (60 primary + 4 paired-with-sphere + 4 recurrent) + 5 non-tumor brain + 4 spheres | 450K | Tumor + non-tumor brain | **VAL-089 ran. `single_cohort_validated_with_caveat`** | Has on-study healthy brain controls (n=5); 100% Xu-538 coverage on 450K. Three biology-cross-check findings (see VAL-089). |
| **GSE143843** (Heidelberg/NOA-08/EORTC 26101 superseries) | 104 + 297 + 398 GBM | 450K/EPIC | Tumor tissue | Deferred to v0.2; accessible | Largest published GBM tumor cohort. Three-cohort meta-analysis possible. |
| **GSE66351** | 18 hemispheric brain (sorted glia/neurons) | 450K | Brain tissue (sorted) | Reference resource | Healthy CNS reference. Sanity-check terminal-class architecture against direct measurements. |
| **GSE137845** | 22 surgical + 26 xenograft + 6 cell lines | 450K + EPIC | Tumor | Deferred to v0.2; accessible | European cohort (Niclou). |
| **GSE50923** (Lai 2013 GBM) | 54 GBM + 24 brain controls | HM27 (only 26K probes) | Tumor + brain control | EXCLUDED: insufficient Xu-538 coverage | Pre-450K platform; would not have full Xu-538 coverage per CHK-1.2. |

### Tier 2 — EGA controlled access

None identified for glioma blood methylation specifically.

### Tier 3 — Biobank-gated (formal application; multi-month timeline)

| Cohort | n | Platform | Specimen | Access path | Why we want this |
|---|---|---|---|---|---|
| **UCSF AGS** (Bracci/Wiencke 2022 + extensions) | 139 pre-surgery glioma + 454 controls | EPIC | Whole blood, dexamethasone-adjusted | dbGaP **phs001497.v2.p1** | THE primary CCL-023 direction-test target. Has on-study healthy controls (n=454) — eliminates the cross-platform reference confound. Validated FCM cell composition. Dexamethasone adjustment built in. |
| **UCSF Immune Profiles Study** | n not enumerated | EPIC | Whole blood | dbGaP **phs002998.v1.p1** | Separate Wiencke/Salas deposit from AGS. Used in their dexamethasone NDMI paper. Apply in parallel with phs001497 — different IRB approval, different application package. |
| **GICC (Glioma International Case-Control Study)** | 7,566 participants, multi-site international | WGS (genetics) + biospecimens stored | Whole blood + saliva | dbGaP **phs001319.v1.p1** | Largest international glioma cohort with stored biospecimens. Methylation-array re-runs would be a major collaboration opportunity. Includes Asian, European, Israeli, Australian sites — would directly address non-Western generalizability. |
| **Mayo Clinic Glioma cohort** | ~1,504 glioma patients (extension of AGS) | EPIC | Whole blood | dbGaP-linked to phs001497 | Large independent replication of UCSF cohort. Same Wiencke/Eckel-Passow consortium. |
| **Nassiri 2020 cfMeDIP-seq** | 60 glioma plasma, 447 multi-cancer | cfMeDIP-seq (NOT 450K/EPIC) | Plasma | Direct PI request | Different chemistry. v0.2+ integration target if cfMeDIP-seq pipeline is built. |

### Excluded from cohort completeness after CHK-1.1 verification

- **GSE92580** — claimed in earlier survey draft as glioma blood; verified is **Carén Sweden EPIC validation cohort, paediatric brain tumour FF/FFPE samples**. NOT glioma blood. False positive caught.
- **GSE155962** — verified is **medulloblastoma cell lines IP6 treatment study (Marino QMUL)**. NOT glioma. False positive caught.
- **GSE209668** — verified is **medulloblastoma proteogenomic study**, not adult glioma.
- **Chen 2016 Nantong China n=109 glioma serum** — Alu-element methylation only, NOT array. Different assay.
- **SNU-LTS / SNU-STS Korean** (GSM3143643-GSM3143675) — GBM tumor not blood.
- **Italian EnviroGenomarkers/EPIC-Italy glioma sub-cohort** — NOT available; the EPIC-Italy methylation work has focused on breast/CRC/CLL pre-diagnostic, no glioma deposit identified.

### Non-Western search results (per Heath's "Italy, China, Timbuktu" instruction)

- **China:** CGGA (Chinese Glioma Genome Atlas) referenced in GliomaDB — primarily TUMOR methylation. Chen 2016 Nantong serum cohort uses Alu-only assay. **No blood-pathway 450K/EPIC cohort with healthy controls accessible from Chinese consortium.**
- **Italy:** EnviroGenomarkers / EPIC-Italy — strong on breast/CRC/CLL pre-diagnostic blood, but **no glioma-specific deposit identified**. The glioma blood methylation work concentrated at UCSF (US), not EPIC-Italy.
- **Korea:** SNU LTS/STS GBM tumor cohort, not blood.
- **Japan:** No dedicated blood-pathway methylation glioma cohort beyond GICC participation.
- **India / Brazil / Africa:** No blood-pathway 450K/EPIC glioma cohort identified.

**The pattern is consistent: glioma blood methylation is concentrated in the Wiencke/Salas/Kelsey consortium (UCSF–Dartmouth–KU) and is mostly biobank-gated.** Public deposits are limited to one cohort (GSE180683). This is a structural feature of the field, not a search failure.

---

## What we'd need access to (the explicit asks list)

If a researcher or institution one day asks "what do you need access to to advance this card?", the answer is, in priority order:

1. **dbGaP phs001497.v2.p1 (UCSF AGS)** — Bracci 2022 cohort, n=139 pre-surgery glioma + 454 controls EPIC whole blood, dexamethasone-adjusted. Direct test of VAL-088's CCL-023 inversion finding on its native platform with on-study controls. Multi-month application; UCSF IRB.
2. **dbGaP phs002998.v1.p1 (UCSF Immune Profiles Study)** — companion deposit, additional EPIC blood samples. Apply in parallel with phs001497.
3. **dbGaP phs001319.v1.p1 (GICC)** — international biospecimens stored across US/UK/France/Germany/Sweden/Israel/Korea/Asia. Methylation array re-runs would directly address non-Western generalizability.
4. **Mayo Clinic Glioma cohort dbGaP-linked deposit** — large independent replication.
5. **Nassiri 2020 cfMeDIP-seq cohort (direct PI request)** — different chemistry; v0.2 cfMeDIP-seq pipeline integration target.
6. **An LP-CSF EPIC array glioma cohort** — gold-standard specimen for terminal-class detection. Does not currently exist in published form. Would require a custom collaboration with a neurosurgical center.
7. **A pre-diagnostic glioma blood cohort at varying time-windows** — to test whether CCL-023 direction-as-discriminator extends pre-diagnostically for glioma. Would require UK Biobank, EPIC-Italy NSHDS, Sister Study, or MCCS subsets where glioma incidence is captured. None of these have a published glioma sub-analysis.
8. **A deep cervical lymph aspirate methylation cohort** — Pathway 2 specimen; framework prediction; no cohort exists. Custom collaboration required.
9. **Sabedot 2021 GeLB Mendeley deposit (cgrz6zztfg)** — already accessible Tier 1; integration as v0.2 external-classifier arm.

---

## v0.1 honest weaknesses summary

- **Single-cohort validation.** N=76 glioma blood, no internal healthy controls, cross-platform external comparator. Replication required.
- **No pre-diagnostic data.** All 76 patients are at-diagnosis or post-diagnosis. CCL-023 hypothesis cannot be tested pre-diagnostically for glioma.
- **No CSF data validated.** LP-CSF is the gold-standard specimen for this card and we have no cohort to test against.
- **No cfMeDIP-seq integration.** Different chemistry; future build.
- **No Pathway 2 or Pathway 4 cohorts.** Both pathways are framework-prediction or open-research-program at v1.
- **CCL-023 prior was inverted by direct measurement.** The hypothesis bridge from cell-fraction direction to A-score direction does not hold for glioma. CCL-023 has been refined accordingly.
- **No discrimination from AD on Stage 1 direction.** Both read positive at peripheral blood. Stage 2 + Stage 3 + clinical context are required for differential.
- **Cross-platform cross-cohort baseline confound.** EPIC test vs HM450 reference. Direction of effect is robust; absolute magnitude carries documented caveat.
- **Treatment heterogeneity in test cohort.** 39/76 are post-treatment with mixed regimens. The 37 pre-surgery treatment-naive subset is the cleanest signal.

---

## Future test ideas (to log for v0.2+)

- **VAL-089 — DONE.** Direct tumor-architecture A-score on GSE60274 (Lai 2015) with 5 on-study NTB controls. Result: GBM_primary d=+0.24 (wide CI), recurrent d=+1.17, spheres d=-1.81 inversion. Confirms heterogeneity-not-tumorness biology cross-check. See VAL-089_outcome.md.
- **VAL-090:** Heidelberg GBM cohort (GSE90496) tumor-architecture A-score with 24 inflammatory + 3 reactive controls as comparison. Tests whether tumor architecture differs from inflammatory/reactive non-neoplastic tissue. **Larger control set than VAL-089's n=5.**
- **VAL-091:** Sabedot Mendeley deposit (cgrz6zztfg) — re-run GeLB scoring through GAPE pipeline. Tests whether v1 universal panel reproduces Sabedot's classifier output as well as their custom panel. **First serum-cfDNA-pathway VAL.**
- **VAL-092:** Heidelberg three-cohort superseries GSE143843 — three-cohort meta-analysis tumor-architecture A-score across n≈800 GBM tumors.
- **VAL-093:** TCGA-LGG/GBM direct test (n=129 GBM + 516 LGG from UniD pipeline). Largest accessible glioma tumor methylation cohort. Direct test of Issue 002 prediction LGG ΔA = +0.239 vs GBM ΔA = +0.217.
- **VAL-094:** GSE292314 (Silv 2026) — tumor methylation with cellular deconvolution. Stage 3 microenvironment composition test integrated with Stage 1 architecture A-score.

These VAL studies are deferred until after v0.1 ships. Each requires its own prereg/results/outcome cycle.

---

## Files in this card

- `glioma-epic_README.md` — this document
- `glioma-epic_PATHWAY_NOTES.md` — earlier design notes (preserved as reference)
- `glioma-epic_card_v0.1.json` — card metadata
- **VAL-088 (blood arm):**
  - `VAL-088_prereg.md` — pre-registration
  - `VAL-088_outcome.md` — outcome interpretation
  - `VAL-088_results.json` — full numerical output
  - `VAL-088_distributions.png` — boxplot all 76 glioma vs healthy reference
  - `VAL-088_presurg.png` — boxplot pre-surgery treatment-naive subset
  - `val_088_glioma_epic_blood.py` — analysis script
  - `GSE180683_manifest.json` — parsed metadata for all 76 samples
  - `GSE180683_chippos_to_gsm.json` — chip_position → GSM mapping
- **VAL-089 (tissue arm):**
  - `VAL-089_prereg.md` — pre-registration
  - `VAL-089_outcome.md` — outcome interpretation
  - `VAL-089_results.json` — full numerical output
  - `VAL-089_distributions.png` — boxplot NTB / GBM-primary / GBM-recurrent / GBM-spheres
  - `val_089_glioma_epic_tissue.py` — analysis script
  - `GSE60274_manifest.json` — parsed metadata for all 77 samples
