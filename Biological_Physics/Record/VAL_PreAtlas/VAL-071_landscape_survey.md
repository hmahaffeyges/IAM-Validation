# VAL-071 — cervical-epic landscape survey

**Status:** Pre-build landscape survey. No β-value access yet; this document is sealed BEFORE any prereg writing under the Block 1-20 expectations from README_MASTER §17.
**Date:** 2026-04-25.
**Card target:** cervical-epic v0.1.

---

## 1. Disease scope and class assignment

**Disease:** Cervical cancer, primarily squamous cell carcinoma (SCC) and adenocarcinoma (ADC), arising from the transformation zone (squamocolumnar junction) of the uterine cervix. ICD-10: C53.0 (endocervix), C53.1 (exocervix), C53.8 (overlapping), C53.9 (NOS). Excluded: D06 (carcinoma in situ — handled in the dysplasia/CIN3 stratification, not as the cancer endpoint).

**Tissue class:** **cycling** (cervical_epithelial). Same class as colon, lung, gastric, bladder, skin epithelium. H_min(cycling) = 0.856100 per GAPE_WEB_v13.py canonical constants. A_ceiling values for cycling class: methyl 1.1681, nucl 1.0203 ⚠ (structurally saturated), fuzz 1.2210, wps 1.5938, frag 1.4536. Same nucl-saturation pattern as PDAC and other non-pluripotent classes.

**Healthy reference β** for cervical_epithelial: not in Moss 2018 25-tissue reference panel. Closest reference is the broad squamous epithelium baseline. **This is a v0.1 data gap that the card must document explicitly** — Stage 2 NNLS deconvolution does not include cervical_epithelial as a target tissue. For cervical-epic, Stage 2 plays a different role than for PDAC: the primary specimen IS already cervical (LBC / pap smear / cervical brush), so deconvolution-from-blood is not the primary pathway.

**Critical biological feature:** cervical cancer is unique among Cookbook cancers because **the cervix sheds directly into liquid-based cytology samples that are already collected as standard-of-care during pap smear screening**. This makes LBC the primary specimen, not blood. The pathway hierarchy will be inverted from PDAC.

---

## 2. Specimen pathway hierarchy (preliminary, to be locked in card v0.1 §4)

Unlike PDAC where plasma cfDNA is primary, cervical-epic's primary specimen is direct cervical-cell shedding. Hierarchy:

| Priority | Pathway | Specimen | Rationale |
|---|---|---|---|
| 1 | LBC / pap smear cytology | Cervical exfoliated cells in ThinPrep or SurePath fluid | Already collected as standard-of-care; direct shedding from the disease tissue; no deconvolution needed |
| 2 | Self-sampled cervicovaginal swab | Cervicovaginal lavage, brush, or tampon-based self-collection | Lower clinical access barrier; methylation signal preserved (Doorbar/Bonde studies) |
| 3 | Tissue biopsy (cervical) | LEEP, conization, or punch biopsy from colposcopy | Highest-fidelity Stage 2 ceiling reference; tumor biopsy direct read |
| 4 | Plasma cfDNA | Blood draw | Stage 1 immune-class flag; pre-diagnostic detection at the long-window level (analogous to VAL-046 Rotterdam for PDAC); cervical_exocrine fraction expected in cfDNA via Moss-style deconvolution if added to the panel |
| 5 | Urine cfDNA | First-morning void | Exploratory; some HPV/cervical work exists but not the primary pathway |
| ~ | CSF | Not applicable | Cervical cancer does not preferentially shed into CSF |
| ~ | Saliva | Not applicable | No cervical signal in saliva expected |

**The cervical-epic card is the first Cookbook card where pap smear cytology is the primary specimen.** This means the Stage 1 / Stage 2 distinction operates differently than for PDAC. Stage 1 still scores the immune component of the cytology sample on Xu-538 — but the immune component of LBC is NOT buffy-coat leukocytes; it is the immune cells infiltrating the cervical mucosa plus mucus-trapped immune cells. This requires explicit documentation in the card and possibly a dedicated Xu-538 transferability check.

---

## 3. Public-access cohorts inventory

This is the cohort-completeness sweep per CCL-029. Every publicly-accessible 450K/EPIC cervical methylation cohort identified:

### 3.1 Tissue / biopsy cohorts (HM450 or EPIC)

| Cohort | Platform | n | Composition | Status | Use |
|---|---|---|---|---|---|
| **TCGA-CESC** | HM450 | ~307 | 307 tumor + 2 metastatic + 3 normal (very few normal — limits paired analysis) | Public (NIH GDC) | Tumor-architecture validation; very limited matched normal |
| **GSE99511** (Verlaat 2018) | HM450 | 68 | 28 normal + 36 CIN3 + 4 tumor | Public GEO | CIN3 progression validation (rare HM450 CIN3 cohort) |
| **GSE46306** | HM450 | 43 | 20 normal + 17 CIN3 + 6 cancer | Public GEO | Independent CIN3+cancer replication |
| **GSE38266** | HM450 | 42 | 21 HPV+ tumor + 21 HPV− tumor | Public GEO | HPV-stratified analysis |
| **GSE68339** (Farkas 2013) | HM450 | 270 | 149 cervical cancer (discovery) + 121 cervical cancer (validation) | Public GEO | Large-n cancer-only methylation; no normals |
| **GSE41384** | HM27 | 19 | 10 normal + 9 tumor | Public GEO | **EXCLUDED** — HM27 not array-compatible with Xu-538 panel |
| **GSE30759, GSE30760** | HM27 | 215 / 278 | normal + tumor | Public GEO | **EXCLUDED** — HM27 not compatible |

### 3.2 LBC / pap smear / cytology cohorts (the unique cervical specimen) — EPIC 850K

| Cohort | Platform | n | Composition | Status | Use |
|---|---|---|---|---|---|
| **GSE143752** (El-Zein 2020) | EPIC 850K | 96 | 42 CIN3 + 54 control LBC | Public GEO | The first EPIC cervical LBC EWAS; used as validation in Bowden 2025 |
| **GSE287994** (Bowden 2025) | EPIC 850K | 247 | 119 benign + 74 CIN3/CGIN + 54 cancer LBC | Public GEO | The largest LBC EWAS to date; primary candidate for VAL training |
| **CINCS** (Bukowski 2023, Duke/UNC) | HM450 (n=76) + EPIC 850K (n=213) | 289 | Prospective, normal/CIN1 enrollment, 5 yr FU, 15 progressed to CIN2+ | Published — **GEO accession not stated in abstract; confirm during VAL prereg** | Pre-diagnostic LBC-based progression cohort — direct analog to VAL-046 Rotterdam role |
| **WID-CIN** (Herzog 2022, Innsbruck/UCL) | EPIC 850K | 1254 | Multi-cohort: discovery 372 + diagnostic validation 454 + predictive validation cohort | EUTOPS/Karolinska — **NOT publicly deposited; access requires PI contact (Widschwendter group)** | The gold-standard cervical-LBC methylation reference; gated |
| **Karolinska WID-qCIN** (van der Graaf 2024) | qPCR (DPP6, RALYL, GSX1) on 28,017 women | n/a EPIC | Real-world Stockholm cohort | Published; method-only deposit | qPCR deployment study, not array-based; framework relevant but not directly Cookbook-scoreable |
| **Sundström CIN2 EPIC** (Bukowski-style) | EPIC 850K | 58 | 58 young women with CIN2, serial LBC, regressors vs non-regressors | Published 2026 | CIN2 active surveillance — small but unique regression vs progression design |

### 3.3 Self-sampled cervicovaginal swab cohorts

| Cohort | Platform | n | Composition | Status | Use |
|---|---|---|---|---|---|
| **POBASCAM trial** (Bonde 2020) | qPCR FAM19A4/miR124-2 | n=14k+ Dutch | Long-term FU | Published; method-only | Long-term prospective FAM19A4/miR124-2 reference; gated on EPIC array level |
| **Doorbar self-sample** | EPIC 850K | <100 estimated | Self-sampled cervicovaginal lavage | Published; check for GEO deposit | Self-sample feasibility validation |

---

## 4. Public access summary and v0.1 build plan

**Tier of cohort accessibility:**
- **Public, immediately usable** (HM450 + EPIC 850K, GEO-deposited): TCGA-CESC, GSE99511, GSE46306, GSE38266, GSE68339, GSE143752, GSE287994.
- **Probably public** (need to verify GEO deposit during prereg): CINCS Bukowski 2023, Sundström CIN2 2026.
- **Gated** (PI contact required): WID-CIN cohort (EUTOPS Innsbruck), POBASCAM Dutch.
- **Method-only / not array-deposited**: Karolinska WID-qCIN deployment, FAM19A4 qPCR studies.

**Cohort-completeness rule (CCL-029) compliance plan:**

The cervical-epic v0.1 card must run every publicly-accessible cohort meeting the platform criterion (HM450 or EPIC 850K with downloadable β values). HM27-only cohorts (GSE41384, GSE30759, GSE30760) are excluded by platform — Xu-538 is a 450K-derived panel and CpG coverage on HM27 is incomplete.

**Planned VAL studies for cervical-epic v0.1:**

| VAL | Cohort | Platform | n | Specimen | Hypothesis |
|---|---|---|---|---|---|
| **VAL-071 (this document)** | Landscape survey | n/a | n/a | n/a | Pre-build documentation; no β-value access |
| **VAL-072** | TCGA-CESC | HM450 | ~307 (tumor-heavy, ~3 normal) | Tissue biopsy | Tumor architecture A-score elevation; tumor-only baseline test (limited normal) |
| **VAL-073** | GSE99511 (Verlaat) | HM450 | 68 (28 nrm + 36 CIN3 + 4 tumor) | Tissue biopsy | Normal vs CIN3 vs tumor progression; primary tissue-arm anchor |
| **VAL-074** | GSE46306 | HM450 | 43 (20 nrm + 17 CIN3 + 6 cancer) | Tissue biopsy | Independent replication of VAL-073 progression pattern |
| **VAL-075** | GSE38266 | HM450 | 42 (21 HPV+ tumor + 21 HPV− tumor) | Tissue biopsy | HPV-stratified architectural elevation |
| **VAL-076** | GSE143752 (El-Zein) | EPIC 850K | 96 (42 CIN3 + 54 control LBC) | LBC pap smear | **First LBC-pathway validation; primary specimen anchor** |
| **VAL-077** | GSE287994 (Bowden 2025) | EPIC 850K | 247 (119 benign + 74 CIN3/CGIN + 54 cancer) | LBC pap smear | **Largest LBC cohort; primary pap-smear-pathway anchor** |
| **VAL-078** | CINCS (Bukowski 2023) | HM450 + EPIC 850K | 289 | LBC, prospective | **Pre-diagnostic LBC progression — direct VAL-046 analog for cervical** (CONDITIONAL on GEO deposit existing; if gated, document and skip) |
| **VAL-079** | Sundström CIN2 2026 | EPIC 850K | 58 | LBC, serial | CIN2 regressor vs non-regressor — directional discrimination at the borderline-disease level |
| **VAL-080** | Directional fallback panel build | — | — | — | If any of VAL-072 through VAL-079 show bidirectional cancellation per CCL-027, build cervical-specific directional panel using GSE143752 or GSE287994 as training |
| **VAL-081** | GSE68339 cancer-only | HM450 | 270 | Tissue biopsy | Cancer-only large-n confirmation; uses 113 random TCGA-non-cervical normals as controls (Cookbook reference standard) |

**Total planned: 10 VAL studies + 1 landscape survey** (VAL-071 through VAL-081).

This is approximately twice the pancreatic-epic VAL count, justified by cervical's unique specimen advantage (LBC, the primary specimen) requiring its own validation track separate from the tissue-biopsy track.

---

## 5. Critical questions to answer in the cervical-epic v0.1 card

These are the questions that the VAL battery must address before publishing v0.1:

1. **Does the Xu-538 immune panel transfer to LBC samples?** LBC contains cervical mucosa immune cells, not buffy-coat leukocytes. The Xu-538 panel was selected from buffy-coat training data. If the panel does not transfer to LBC — i.e., if pooled-entropy A_immune on LBC samples produces a null or inverted signal regardless of disease state — then LBC-pathway Stage 1 requires a dedicated panel build. This is the analog of the pancreatic juice problem in VAL-068.

2. **Does cervical cancer drive Xu-538 unidirectionally or bidirectionally?** Cervical cancer is HPV-driven, with a chronic local inflammation pattern and a distinct progression signature (CIN1 → CIN2 → CIN3 → cancer over years to decades). The immune signature could plausibly be bidirectional like PDAC (Treg expansion + effector T-cell suppression in HPV-immune-evading tumors) or unidirectional like breast (uniform inflammation marker drift). VAL-076/077 will answer this empirically.

3. **What is the per-CpG positive-direction percentage in cervical methylation tissue?** Empirical determination across VAL-072/073/074/075. A 50/50 split would put cervical-epic in the bidirectional-cancellation category alongside PDAC and AD; a 62-70% split would put it in the standard unidirectional category.

4. **Is Stage 2 deconvolution applicable to cervical-epic at all?** Cervical_epithelial is not in the Moss 2018 25-tissue reference. Either the framework adds a cervical_epithelial reference (possible from the public data) or Stage 2 is documented as not-applicable for cervical-epic at v0.1 because the primary specimen IS the target tissue.

5. **HPV stratification — is it mandatory?** HPV status confounds cervical methylation profoundly. HPV+ vs HPV− cancer have different methylation signatures (GSE38266 demonstrates this). The card's mandatory covariates must include HPV status with stratification. VAL-075 quantifies the magnitude of the stratification effect.

6. **Pre-diagnostic LBC detection window.** CINCS (VAL-078) potentially provides 5-year LBC pre-diagnostic data, which would be the equivalent of VAL-046 Rotterdam for the cervical card. If accessible, this becomes the cohort_screening_validated tier anchor.

7. **CIN2 regressor vs non-regressor.** Sundström 2026 (VAL-079) is a unique opportunity to test whether the framework's A-score predicts regression versus progression in active surveillance — a direct clinical-decision-support deliverable.

---

## 6. Mandatory covariates (preliminary list for §7 of card)

Cervical-epic mandatory covariates differ from PDAC. Standard universal covariates plus disease-specific:

| Covariate | Stratify | Report | Cervical-specific rationale |
|---|---|---|---|
| Age | Yes (decade) | Yes | Cervical incidence peaks in 35-44 yr group; younger = more likely HPV-transient |
| HPV status (negative / 16+ / 18+ / other hr-HPV / multiple) | **Mandatory** | **Mandatory** | HPV genotype is the strongest stratifier of cervical methylation |
| HPV vaccination status | Yes | Yes | HPV vaccinated population has different progression risk |
| Parity | Observation | Yes | Hormone exposure modifies methylation |
| Long-term hormonal contraceptive use | Observation | Yes | OC duration >5 yr is documented PDAC and cervical risk modifier |
| Smoking | Yes | Yes | Strongly modifies cervical immune compartment |
| Immunosuppression (HIV, transplant, autoimmune therapy) | **Mandatory** | **Mandatory** | HIV+ women have 6× cervical cancer risk via HPV persistence |
| Prior CIN treatment (LEEP, cone) | **Mandatory** | **Mandatory** | Treatment alters baseline methylation and recurrence patterns |
| Pregnancy status | Yes (decline scoring if pregnant) | Yes | Hormonal pregnancy state alters cervical immune signal |
| Recent vaginal infection (BV, candidiasis, trich) | Yes (defer if active) | Yes | Acute inflammation contaminates Stage 1 |
| Time since last menstrual period | Observation | Yes | Cervical cell turnover varies through menstrual cycle |
| Race / ethnicity | Observation | Yes | African-American and Hispanic populations have higher incidence (access-driven, not pure biology) |
| Cytology result at time of LBC collection | **Mandatory** | **Mandatory** | NILM / ASCUS / LSIL / HSIL / AGC stratification — the contextualizing variable |

---

## 7. Card validation tier targets

**v0.1 anchor target:** `cohort_screening_validated` if VAL-078 CINCS is accessible (LBC 5-yr pre-dx). If CINCS gated, `single_cohort_validated` from VAL-077 GSE287994 (the largest LBC EWAS). Either way, **VAL-076 and VAL-077 establish the LBC primary-pathway anchor** that no other Cookbook card has.

**Tissue arm modifier:** `tissue_arm_validated` if VAL-073 + VAL-074 + VAL-075 produce concordant tumor-vs-normal architectural elevation. `tissue_arm_exploratory` if any cohort fails or shows bidirectional cancellation requiring directional fallback (VAL-080).

**Stage 2 status:** `stage_2_not_applicable` if cervical_epithelial is not added to Moss reference panel for v0.1. This is honest documentation, not a tier downgrade, because the primary specimen IS the target tissue and Stage 2 deconvolution is conceptually unnecessary for LBC samples.

**Path to cross_platform_validated:** add a non-EUTOPS, non-Karolinska, non-CINCS independent cohort. WID-CIN access via Innsbruck contact is the obvious priority-1 next-step.

---

## 8. Pre-build sealing

This landscape survey is sealed before any β-value access. SHA-256 of this document at submission to commit:

(SHA computed at GitHub push time)

VAL-072 through VAL-081 preregistrations will be written sequentially, each sealed before its respective β-value access. The full cohort-completeness rule applies: every public 450K/EPIC cervical cohort gets its own VAL prereg + outcome + results JSON + script + manifest.

**Go / no-go decision points:**
- After VAL-072 (TCGA-CESC), decide whether tumor-only architecture elevation supports a v0.1 Stage 2 ceiling claim
- After VAL-076 + VAL-077 (LBC primary pathway), decide whether Xu-538 transfers to LBC; if not, build a cervical-LBC dedicated immune panel as a v0.2 deliverable
- After VAL-079 (Sundström CIN2), decide whether the regression-prediction is in scope for v0.1 or deferred to v0.2

**Estimated build time:** 2-3 sessions. The 10 VAL studies cluster naturally into three batches: tissue-biopsy battery (VAL-072 through VAL-075 + VAL-081), LBC-primary battery (VAL-076 through VAL-079), and directional fallback if needed (VAL-080).

---

## 9. What would NOT be in v0.1

- WID-CIN cohort access (gated; v0.2+ priority)
- POBASCAM long-term FU (gated; v0.2+ priority)
- Self-sampled cervicovaginal swab dedicated validation (separate cohort hunt; v0.2+)
- Plasma cfDNA pre-diagnostic cervical detection (no cervical-specific blood pre-dx cohort located in public domain; v0.2+ via UK Biobank cervical subset application)
- Stage 2 cervical_epithelial added to Moss reference (requires reference cohort assembly; v0.3+ engineering)

These are explicit known limitations to document in the card v0.1 §10.
