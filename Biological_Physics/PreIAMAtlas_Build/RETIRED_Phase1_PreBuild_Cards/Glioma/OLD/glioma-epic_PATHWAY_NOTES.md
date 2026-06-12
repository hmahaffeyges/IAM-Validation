# Glioma-EPIC Pathway Design Notes

**Status:** DESIGN NOTES, not validated card
**Date:** 2026-04-24
**Origin:** Session conversation between Heath W. Mahaffey and Walther capturing the framework-level prediction structure for glioma detection

## Purpose

This document captures the five detection pathways for brain pathology (glioma specifically, with extensions to GBM and brain metastasis) that were identified during the session. The card itself should NOT be built until at least one of the validation cohorts referenced below becomes accessible. This document preserves the design work so that, when the time comes, the pathway structure is already laid out.

## Why glioma is the hardest card in the Cookbook

Three converging difficulties:

1. **Specimen problem.** Terminal class (neurons, oligodendrocytes, astrocytes) contributes only ~0.5% to plasma cfDNA at healthy baseline — below the Moss 4% detection floor. Single-timepoint plasma deconvolution returns null even when brain pathology is present.

2. **Architectural-disruption magnitude is enormous.** Once you can hear the signal, it is loud — LGG ΔA = +0.239, GBM ΔA = +0.217, both deep FLOOR BREACH, the largest tissue ΔA values in the entire framework. Issue 002 phrasing: "the physics is extremely loud."

3. **Reaching the signal requires either invasive sampling (LP) or specialized chemistry (cfMeDIP-seq enrichment).** Standard array methylation on plasma cannot recover the signal under most conditions. The card needs multiple specimen-level pathways and multiple analytical approaches stacked on top of each other.

## The five detection pathways

### Pathway 1 — Plasma cfMeDIP-seq enrichment overcomes the 4% detection floor under active disease

The healthy-baseline 4% Moss detection floor describes what bulk methylation array can resolve from healthy plasma. Under active aggressive brain disease, three things change:

1. **Tumor cell death increases brain cfDNA shedding above 0.5% baseline.** Lubotzky et al. detected brain-cell-type-specific cfDNA in plasma of 27/29 patients with brain metastases (neuron-derived), 25/29 (oligodendrocyte-derived), and 29/29 (astrocyte-derived) — versus not detectable in healthy controls or patients without brain metastases.

2. **cfMeDIP-seq enrichment dramatically improves signal-to-noise above genome-wide deconvolution.** Nassiri 2020 (cfMeDIP-seq + machine learning) achieved AUC = 0.99 [95% CI 0.96–1.00] discriminating glioma from extracranial cancers and healthy controls.

3. **Direct methylation-based serum panels work.** Sabedot's Glioma-Epigenetic Liquid Biopsy (GeLB) score discriminates 149 glioma patients from other brain tumor types with 98% accuracy on serum.

**Pathway 1 implementation requires:** cfMeDIP-seq protocol (specialized chemistry, not standard EPIC/HM450 array), trained machine-learning classifier on glioma-specific differentially methylated regions, validation against the Nassiri/Sabedot cohorts.

**Validation tier achievable with Pathway 1:** assay-specific classifier validation tier. Different from the architectural A-score the rest of EDEAR uses — this is a method-specific signature, not the universal pipeline output.

### Pathway 2 — Lymphatic concentration via deep cervical lymph nodes

The glymphatic system (Iliff/Nedergaard 2012, Science) and meningeal lymphatic vessels (Louveau 2015 / Aspelund 2015, Nature) drain CSF and brain interstitial fluid to deep cervical lymph nodes (and from there to systemic circulation). This means brain-derived cfDNA is concentrated in deep cervical lymph before being further diluted in systemic circulation.

Sampling deep cervical lymph node aspirate or fluid is:
- More invasive than peripheral blood draw
- Less invasive than lumbar puncture
- NOT standard clinical practice

No published large-cohort methylation study uses this specimen for brain pathology detection. **Pathway 2 is a framework-level prediction, not a current-validation candidate.** It would be a "tier 2" specimen in the multi-specimen pyramid: above blood, below LP.

**Open framework prediction (G-2026-PXXX, draft):** For any disease state where the brain is the affected tissue (glioma, GBM, AD, ALS, Parkinson's, MS), the meningeal-lymphatic-to-cervical-lymph axis should produce cfDNA enrichment above the 0.5% plasma baseline. Testable but no published cohort exists.

### Pathway 3 — Multi-specimen tier system for terminal-class detection

The clinical reality is that not all patients can or should have lumbar puncture. For glioma-epic to be a clinical product, it needs a graded specimen tier system:

| Tier | Specimen | Invasiveness | cfDNA yield | When clinically used |
|---|---|---|---|---|
| 1 (gold) | LP-CSF | Invasive (lumbar puncture) | High brain cfDNA | Standard for CNS workup |
| 2a | Ventricular shunt sampling | None (existing shunt) | High | Hydrocephalus or post-surgical patients with existing shunts |
| 2b | Ommaya reservoir | None (existing implant) | High | CNS lymphoma, pediatric brain tumors with implanted device |
| 2c | Cisterna magna sampling | More invasive than LP | High | Specialized contexts (rare in adults) |
| 3 | Deep cervical lymph node aspirate | Moderate | Concentrated brain cfDNA, no published cohort | Theoretical (Pathway 2) |
| 4 | Focused-ultrasound BBB-disrupted plasma | Moderate (FUS device required) | Improved over standard plasma | Pilot studies; Brain 2023 review describes |
| 5 | Standard plasma + cfMeDIP-seq | None | Low brain cfDNA, recoverable with enrichment | Pathway 1 |
| 6 | Standard plasma + array methylation | None | Below detection floor for healthy baseline | Below floor for healthy; possibly above floor under active disease |

Each tier carries different validation requirements and different commercial deployment models. Tier 1 (LP-CSF) is the gold standard but rarely used for screening. Tier 5 (cfMeDIP-seq plasma) is the most likely first commercial pathway because it is non-invasive and has the strongest published evidence. Tiers 2a/2b are opportunistic — they apply to patients who already have the device for other clinical reasons.

### Pathway 4 — Brain-resident immune signature in peripheral blood (the microglial/trafficking signature)

The brain has a unique immune compartment:
- **Microglia** — resident brain macrophages, embryonic yolk-sac origin, distinct lineage from peripheral monocytes, decades-long turnover, the body's longest-lived immune cells
- **CNS-border macrophages** — perivascular, meningeal, distinct from microglia
- **Brain-resident T-cells** — sparse but present, especially in disease

When brain pathology develops, peripheral blood shows trafficking signatures:
- Monocyte-to-TAM (tumor-associated macrophage) trafficking in glioma
- DAM (disease-associated microglia) signature in AD
- Activated microglia methylation markers: TMEM119, P2RY12, TREM2
- Brain-immune trafficking markers: CCL2, CCR2

Sabedot's GeLB serum score explicitly captured "cfDNA-derived methylation signatures associated with the presence of glioma **and associated immunological features**." Multiple groups (Nassiri 2020, the 2025 reviews) have shown methylation-immune-pathway enrichment in plasma cfDNA from glioma patients.

**Pathway 4 implementation:** A glioma-specific directional panel built on top of the universal Stage 1 immune A-score. Components:
- Sabedot 2021 GeLB CpGs (panel public)
- Nassiri 2020 cfMeDIP-seq glioma DMRs (top-N most discriminative)
- Microglial activation methylation markers (TMEM119, P2RY12, TREM2, CSF1R)
- Brain-immune trafficking signature (CCL2, CCR2, monocyte-to-TAM markers, FOXP3 if AD-discrimination is required)

This works ON TOP OF the universal Xu-538 Stage 1 immune A-score, not replacing it. The base Xu-538 captures generic immune activation; the glioma directional panel captures brain-pathology-specific trafficking.

### Pathway 5 — Direction-as-discriminator (CCL-023 applied to brain)

This is the newest insight from the session. From CCL-023:

- AD shows POSITIVE peripheral immune-class A-score direction (validated, VAL-051/052/Nabais 2021)
- CRC shows NEGATIVE peripheral immune-class A-score direction (validated, VAL-047)
- Glioma's published cell-fraction signature (Bracci/Wiencke 2022 n=139 pre-surgery + 454 controls, EPIC array, dexamethasone-adjusted) — significantly lower lymphocyte (CD4/CD8/B/NK) and monocyte fractions, significantly higher neutrophils, all p < 0.001 — **is consistent with NEGATIVE direction.**

**If this hypothesis (CCL-023) holds: AD vs glioma is discriminable at the SIGN of Stage 1 alone**, before any tissue-specific testing or specimen change.

The discriminator decision tree becomes:
1. Stage 1 immune A-score: NEGATIVE direction → suspect CRC OR glioma OR other suppression-phase cancers
2. Stage 2 deconvolution: try Moss + cfMeDIP-seq enrichment
   - Localizes to colon_epithelial → CRC
   - Localizes to neuron / oligodendrocyte / astrocyte → glioma
   - Returns null at standard sensitivity → enrich with cfMeDIP-seq, retry
3. Stage 1 immune A-score: POSITIVE direction → AD or activation-phase cancers (breast, lung, prostate, HCC)
4. Stage 3 EpiDISH: cell-composition pattern provides further differential

**This combined with Pathway 4 (microglial signature) gives glioma-epic two independent peripheral-blood discriminators**, even before cfMeDIP-seq is brought in.

## Validation candidate cohorts (when accessible)

Listed in priority order for glioma-epic build:

1. **Bracci/Wiencke 2022 cohort (UCSF AGS).** n=139 pre-surgery glioma + 454 controls, EPIC array, peripheral blood, dexamethasone-adjusted. THE primary validation target. Not in GEO; UCSF controlled-access. Multi-month application timeline. Direct test of CCL-023 direction hypothesis at β-value level.

2. **Nassiri 2020 cfMeDIP-seq cohort.** AUC 0.99 published; primary Pathway 1 validation if methodology is portable to our framework. May be accessible via direct PI contact.

3. **GSE180683.** 76 glioma patients EPIC peripheral blood, mixed treatment stages. Supplementary only — requires careful stratification of treated vs naive. Useful for treatment-effect characterization but not for primary direction-test.

4. **TCGA-GBM and TCGA-LGG tumor tissue.** Already characterized in framework (LGG ΔA = +0.239, GBM ΔA = +0.217, terminal class A_combined ≈ 1.10 FLOOR BREACH). Not blood. Useful for tumor-architecture confirmation only.

5. **CSF cohorts.** Various pediatric and adult glioma CSF-cfDNA studies for Pathway 1 / cfMeDIP-seq method validation. Smaller cohorts, more specialized.

## Build readiness assessment

Glioma-epic should NOT be built (at v0.1 skeleton level or beyond) until at least one of Bracci 2022 or Nassiri 2020 data is accessible. Building a card without per-patient data on the right specimen at the right timepoint is premature.

**What CAN be built today:** these design notes, lessons-learned references (CCL-023, CCL-024), and the future-card placeholder structure. The card does NOT enter the Cookbook tier table at any validated tier until the data exists.

## Clinical question this card eventually answers

For a patient presenting with:
- Cognitive symptoms, seizures, focal neurological deficits, OR
- Imaging findings suggestive of intracranial mass, OR
- Family history of glioma + age 40+, OR
- Stage 1 EDEAR positive with characteristic per-CpG pattern matching brain-immune-trafficking signature

What does EDEAR add?
- **Risk stratification before imaging** (Stage 1 + Pathway 4 directional)
- **Tissue-of-origin confirmation** if Stage 1 fires (Pathway 1 cfMeDIP-seq + Pathway 5 direction match)
- **Differential vs AD and benign neuroinflammation** (CCL-023 direction principle)
- **Treatment response monitoring** (post-diagnosis serial, well-established for cfMeDIP-seq from Sabedot 2021)
- **Pseudoprogression vs true progression** (Sabedot 2021 GeLB showed this is achievable)

What EDEAR does NOT do:
- Replace MRI/CT for anatomic confirmation
- Replace tissue biopsy for histopathologic diagnosis
- Replace neurosurgical evaluation
- Provide treatment selection beyond what is already published for IDH-status and MGMT-promoter methylation

## Strategic notes (carried over from session)

The framework-level finding that emerged from this conversation is bigger than glioma-epic itself. CCL-023 (direction encodes immune-modulatory phenotype and possibly temporal stage) has implications for:

- **Cancer biology research** — peripheral-blood-detectable immunomodulation phenotyping decade pre-diagnostically
- **Treatment selection** — immune-restoration vs response-modulation approaches based on the direction
- **Immunotherapy response prediction** — G-2026-P011 trajectory test extended with direction encoding
- **A future paper** — "Peripheral immune-class methylation signature as decade-pre-diagnostic phenotyping of cancer immunomodulation"

These extend well beyond the Cookbook's detection mission. They are noted here as strategic captures from the session.

## File pointers

- `LESSONS_LEARNED.md` — CCL-023 (direction-as-phenotype hypothesis, glioma cell-fraction support from Bracci 2022) and CCL-024 (this pathway design summary)
- `immune-atlas/immune-atlas_README.md` — the differential-diagnosis engine that operationalizes CCL-023
- (future) `glioma-epic/glioma-epic_README.md` — TO BUILD when data accessible
- (future) `glioma-epic/glioma-epic_card_v0.1.json` — TO BUILD when data accessible

## Action items (logged in TODO_COOKBOOK_BUILDOUT.md)

- [ ] UCSF AGS data application for Bracci 2022 cohort (multi-month timeline)
- [ ] Nassiri 2020 cfMeDIP-seq direct PI contact
- [ ] GSE180683 download and treatment-stratified preliminary analysis (lower priority, supplementary)
- [ ] Pathway 4 directional panel CpG list compilation (literature-derived, can be done today)
- [ ] Pathway 5 (CCL-023) direction-as-discriminator validation as cohorts become available
