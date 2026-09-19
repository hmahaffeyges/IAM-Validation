# Cardio-EPIC Card — EDEAR Cardiovascular Disease Risk Flag

**Version 0.1 skeleton · 2026-04-24**
**Validation tier:** `skeleton_pending_validation`
**Card position:** #14 of 15 in Cookbook v2.1 expansion set
**Status:** Skeleton authored; full Phase 9/12 validation pending after current cancer cards complete

## What this card is

The cardio-epic card is EDEAR's first non-cancer card of the cardiovascular category. It operates on the same universal Stage 1 immune signature that every other card uses, but its expected Stage 2 result is **null for solid-organ localization** because atherosclerotic cardiovascular disease is systemic vascular inflammation, not a localized tumor. Cardio-epic reads the monocyte-specific, FOXP3-regulatory-T-cell, and inflammation-linked CpG signatures that distinguish cardiovascular risk from cancer-driven Stage 1 positives.

This card is the most important non-cancer addition to the Cookbook because cardiovascular disease is the #1 cause of death globally — more than all cancers combined — and the methylation literature has produced larger prospective cohorts for CVD than for any individual cancer.

## Why this card exists

Three forces made cardio-epic necessary:

1. **Stage 1 positive + Stage 2 null is the largest unexplained diagnostic gap.** Without a cardiovascular card, every patient whose Stage 1 immune signature fires for systemic inflammation gets routed to "architectural flag, see clinician for standard workup" with no specific interpretation. A cardiovascular card resolves the most common cause of that pattern.

2. **The cohorts are larger and more statistically mature than cancer cohorts.** ARIC+CHS+FHS+InCHIANTI+KORA+NAS+WHI+EPICOR meta-analysis totals n=11,461 with 1,895 incident CHD events over 11.2-year follow-up (Aslibekyan et al., Circulation 2019). That is substantially larger than any single cancer pre-diagnostic cohort validated to date in the Cookbook.

3. **The published CpG panels map directly onto EDEAR's framework.** The CVD-associated CpGs are immune-cell-derived (FOXP3 Treg, monocyte-specific modules, ZBTB12 15-CpG panel). Xu-538 already partially overlaps with this signature space. A CVD-specific directional panel on top of the universal Xu-538 Stage 1 is the natural implementation.

## Expected Stage 1 immune signature

**Expected direction:** POSITIVE pooled A-score with distinctive per-CpG pattern matching chronic vascular inflammation.

**Expected magnitude (to be validated):** d ≈ +0.2 to +0.4 at 5-10 year pre-event window based on published CHD methylation effect sizes. This is lower magnitude than pre-clinical cancer signatures (breast +0.45 to +0.71) because cardiovascular methylation changes are chronic rather than acute-progressive, and the signal accumulates over decades of traditional risk factor exposure.

**Distinguishing per-CpG pattern — the CVD fingerprint:**
- Elevated FOXP3 regulatory T-cell methylation (Zhu 2018 n=171 acute CHD, poor outcome correlation; Foxp3-TSDR conserved regulatory region)
- Monocyte-specific CpG module activation (WHI + FHS Offspring WGCNA, Clinical Epigenetics 2019)
- ZBTB12 15-CpG panel elevation (Guarrera 2015, AUC improvement +0.03-0.04 over Framingham risk score)
- CASZ1 differential methylation (NHLBI EWAS all-cause mortality, JAHA 2019)
- TNRC6C, SLC9A1, SLC1A5 regional methylation (WHI + FHS regional analysis)

## Expected Stage 2 result

**Stage 2 Moss NNLS deconvolution returns NULL for solid-organ tumor localization.** This is characteristic, not a failure. The disease is in the vascular bed systemically (coronary, carotid, peripheral) and the vascular cells contributing cfDNA to plasma (vascular_endothelial, smooth muscle) are stromal class — at the 4% detection floor, marginal for deconvolution.

A Stage 2 result showing elevated vascular_endothelial β or fibroblast β in the top-3 localization would be supportive of cardiovascular interpretation. Absence of any cycling/secretory localization with strong Stage 1 positive is the Cardio-epic trigger pattern combined with the CVD CpG signature above.

## Literature anchors (ready for Phase 9/12 validation when built)

### Primary cohort — 9-study methylation meta-analysis of incident CHD

**Aslibekyan S et al., Circulation 2019** — "Blood Leukocyte DNA Methylation Predicts Risk of Future Myocardial Infarction and Coronary Heart Disease" — doi:10.1161/CIRCULATIONAHA.118.039357

- **Cohorts:** ARIC (Atherosclerosis Risk in Communities), CHS (Cardiovascular Health Study), EPICOR, FHS (Framingham Heart Study), InCHIANTI, KORA, NAS (Normative Aging Study), WHI-EMPC, WHI-BAA23
- **N:** 11,461 CHD-free at baseline (mean age 64, 67% women, 35% African-American)
- **Events:** 1,895 incident CHD during mean 11.2-year follow-up
- **Platform:** Illumina Infinium 450K
- **Adjustments:** age, sex, smoking, education, BMI, blood cell type proportions, technical variables
- **Analysis:** fixed-effect meta-analysis + Mendelian randomization for causality

This is the cardiovascular equivalent of VAL-046 / VAL-047 for cancer. A Phase 9/12-equivalent run on these cohorts (or any accessible subset) is the primary validation pathway.

### Prognostic CAD adverse outcomes

**Nature Communications 2025 (no longer in open access check)** — DNA methylation predicts adverse outcomes of coronary artery disease — published prognostic models for clinical translation. CASZ1 differential methylation identified as all-cause CV mortality signal.

### FOXP3 regulatory T-cell methylation

**Zhu et al. 2018** — 171-patient acute CHD longitudinal cohort. Elevated FOXP3-TSDR methylation correlates with poor clinical outcomes and atherosclerosis severity. Independent study confirmed FOXP3 methylation associated with CHD after adjusting for age, smoking, BMI, blood pressure, blood glucose, lipid profiles.

### Monocyte-specific CpG modules

**Clinical Epigenetics 2019** — WGCNA of WHI (n=2,129) and replication in FHS Offspring (n=2,587). Two modules replicated across cohorts — one enriched for development/epigenetic aging, one with preliminary monocyte-specific effects linked to cumulative risk factor exposure. Three regional signals: SLC9A1, SLC1A5, TNRC6C.

### ZBTB12 15-CpG panel

**Guarrera et al. 2015** — 292 incident MI cases + 292 matched controls discovery; 317 cases + 262 controls replication. Both European ancestry. 5.6 to 6.9 year mean follow-up. ZBTB12 15-CpG panel plus LINE-1 improves prediction of MI over Framingham risk score alone (AUC 0.66→0.69 women, +0.70 men).

### All-cause mortality prediction

**Zhang et al. 2017, Nat Commun 8:14617** — Blood methylation panel predicts all-cause mortality, CpGs enriched in inflammation and cardiometabolic regulation genes. Independently replicated in large prospective cohorts.

### Short-term risk prediction

**Clinical Epigenetics 2022** — HRS + NICOLA n=4,018+8,504 — DNA methylation surrogate biomarkers predict short-term CV events better than self-reported or measured exposures.

### Recent addition

**Liu et al. Circulation 2025** — American Heart Association Life's Essential 8 cardiovascular health score correlates with DNA methylation changes across 5 cohorts. Provides the first cardiovascular-health-to-methylation prospective mapping.

## Planned validation approach (Phase D build)

The cardio-epic card will be validated using the same Phase 9/12 methodology as breast-epic and crc-epic:

1. **Data access:** ARIC and FHS are dbGaP-gated. Application timeline 4-12 weeks. WHI has public summary statistics plus individual-level data via dbGaP. CHS public genotype data; methylation via application. UK Biobank cardiovascular arm provides alternate pathway.
2. **Panel:** Xu-538 universal Stage 1 plus CVD-specific directional panel derived from the published CpGs above (FOXP3-TSDR, ZBTB12 15-CpG, monocyte module top-N, CASZ1).
3. **Frozen methodology:** cycling-class-style Phase 9 replication run on an independent cohort after panel lock.
4. **Primary endpoint:** per-patient Cohen's d at 5-year and 10-year pre-CHD windows with direction + magnitude locked to prediction.

## Tier thresholds (preliminary — to be locked at validation)

The same 80-cell healthy baseline reference used by every other card:
- **NORMAL:** A < 1.01
- **MARGINAL:** A ≥ 1.01 — annual serial sampling; document CVD risk factors
- **DETECTABLE:** A ≥ 1.05 with CVD-signature match — standard CVD workup
- **URGENT:** A ≥ 1.07 with CVD-signature match — expedited CVD workup, consider cardiac imaging
- **FLOOR BREACH:** A ≥ 1.10 with CVD-signature match — urgent cardiology consult, imaging, statin/antihypertensive optimization review

## Clinical action matrix (preliminary)

| Tier | Signature match | Clinical action |
|---|---|---|
| DETECTABLE | CVD signature | Lipid panel, hsCRP, BP check, HbA1c. If elevated risk → cardiology referral. |
| URGENT | CVD signature | Above + coronary calcium score OR CTA per clinical indication. Pharmacologic risk reduction discussion. |
| FLOOR BREACH | CVD signature | Urgent cardiology consult. Aggressive risk factor optimization. Serial EDEAR quarterly until A-score trajectory stabilizes. |
| MARGINAL | Any | Lifestyle intervention discussion. 6-12 month serial sampling. |
| DETECTABLE+ | No CVD signature | Route to immune-atlas Pathway 2 (hematologic) or Pathway 1 (terminal) instead. |

## Specimen

**Primary:** buffy-coat blood (same as every Cookbook card). No alternate specimen needed — systemic inflammation is best read in peripheral blood leukocytes.

## Known limitations (to be confirmed at validation)

- Cardiovascular methylation changes are chronic, not acute-progressive — expected magnitude smaller than cancer signatures
- Smoking as confounder: smoking drives independent immune methylation signatures (F2RL3, AHRR) that overlap with the cardiovascular pattern. Smoking status mandatory covariate per lung-epic precedent.
- Traditional CVD risk factors (statins, antihypertensives, BMI changes, physical activity) alter blood methylation dynamically. Longitudinal interpretation requires patient-level risk factor metadata.
- The FOXP3-TSDR assay and the ZBTB12 15-CpG panel are published but not in Xu-538. Cardio-epic validation will require a supplementary directional panel built from these published CpGs.
- Ethnicity effects published for the ARIC African-American sub-cohort suggest race-stratified analysis may be necessary.
- Age is the dominant CVD risk factor — age-matched baseline reference is mandatory (same as every other card, 80-cell reference provides this).

## Next steps to build

1. **Literature consolidation:** Pull all published CVD methylation CpGs from the anchor papers above. Build a `cardio_disease_panel_v0.json` analogous to the AD 7-CpG Rule A panel.
2. **dbGaP applications:** Submit for ARIC + FHS methylation access via standard dbGaP data-use application.
3. **Panel freeze:** After literature consolidation, freeze the supplementary cardio directional panel.
4. **Phase 9 equivalent on an available cohort:** WHI-EMPC public methylation subset may be accessible faster than ARIC/FHS. UK Biobank is the alternative.
5. **Phase 12 independent cohort replication:** Once Phase 9 passes on the first cohort, replicate on a second cohort with frozen panel.

## Relationship to other cards

- **immune-atlas (#13):** cardio-epic is Pathway 3 in the four-pathway differential for Stage-1-positive Stage-2-null cases
- **Every other Cookbook card:** cardiovascular signature is a key differential alternative when Stage 2 returns null for any cancer localization
- **ad-immune:** shares the directional-panel-on-top-of-pooled-Xu-538 architecture
- **README_MASTER specimen section:** vascular_endothelial and fibroblast (stromal class) sit at the 4% cfDNA detection floor — cardio-epic explicitly handles this

## Timeline

- **2026-04-24 v0.1 skeleton:** authored with literature anchors ready
- **Phase D (after all current cancer cards validated):** full card build
- **Pending:** dbGaP applications (4-12 weeks each), literature-derived panel freeze, Phase 9/12 validation runs

## File pointers

- **This README** — skeleton narrative
- **`cardio-epic_card_v0.1.json`** — TO BUILD (deferred until full authoring phase)
- **Parent cookbook** — `README_MASTER_v2.1.md` section on specimen selection + cfDNA detection floor
- **Differential reference** — `immune-atlas/immune-atlas_README.md` Pathway 3
