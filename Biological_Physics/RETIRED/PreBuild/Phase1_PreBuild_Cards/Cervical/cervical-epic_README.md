# cervical-epic — IAMPerformance EDEAR Cookbook v0.1

**Card ID:** cervical-epic
**Card version:** v0.1
**Card date:** 2026-04-25
**Supersedes:** none (first version)
**Card status:** `exploratory_with_cohort_heterogeneity` — VAL-072 through VAL-081 complete (six VAL studies, total n=807 samples scored). Tissue arm: VAL-073 Verlaat positive anchor (d=+0.73 Normal vs CIN3) cannot be replicated by VAL-074 Farkas (d=−0.61) or VAL-081 Lando (d=−0.43 vs external normals at n=270). LBC primary pathway: VAL-076 panel-transferability flag (Xu-538 was buffy-coat trained, LBC is exfoliated cervical epithelium + mucosal-resident lymphocytes — different cell mixture); VAL-077 deferred to v0.2+ data-integrity flag (supplementary file is residual M-values, not raw β). Card cannot make clinical claim at v0.1; path to v0.2+ documented in §11.
**Card scope:** Cervical squamous cell carcinoma and adenocarcinoma detection at three tiers — pre-cancerous CIN3 detection (screening), invasive cancer detection (diagnostic), and post-treatment surveillance — across five active specimen pathways with pap smear liquid-based cytology as the primary clinical specimen.

---

## 1. What cervical cancer is

Cervical cancer is the cancer of the uterine cervix, the lower portion of the uterus that opens into the vagina. The cervix has two histological zones: the ectocervix (outer, stratified squamous epithelium) and the endocervix (inner, columnar glandular epithelium). The boundary between them — the squamocolumnar junction or transformation zone — is where most cervical cancers originate. Globally, cervical cancer is the fourth most common cancer in women, with approximately 660,000 new cases and 350,000 deaths annually as of 2022 (WHO/GLOBOCAN). It is one of the few cancers with a known infectious etiology: persistent infection with high-risk human papillomavirus (hrHPV) genotypes — primarily HPV16 and HPV18 — drives the vast majority of cases through a slow, multi-decade progression from initial infection to invasive disease.

The progression is staged histologically: normal cervical epithelium → low-grade squamous intraepithelial lesion (LSIL, also called CIN1) → high-grade squamous intraepithelial lesion (HSIL, comprising CIN2 and CIN3) → invasive carcinoma. CIN1 lesions usually regress spontaneously. CIN3 lesions are the immediate precursor to invasive cancer and are the conventional treatment threshold (LEEP / cone biopsy). Adenocarcinoma in situ (AIS) and adenocarcinoma (ADC) arise from glandular endocervical epithelium and follow a parallel progression that is harder to detect cytologically.

Cervical cancer is unique among Cookbook cancers in three ways. First, it has a long, well-characterized pre-cancerous progression spanning years to decades, which means screening can intervene at multiple stages. Second, it has an established population-level screening infrastructure — pap smear (Papanicolaou) cytology and HPV DNA testing — already collecting the exact specimen this card needs. Third, an effective prophylactic vaccine (against HPV16/18 and other genotypes) has begun shifting the disease epidemiology in vaccinated populations, requiring future-cohort calibration.

**Tissue class:** cycling (cervical_epithelial). Same architectural class as colon-epic, lung-epic, gastric-epic, bladder-epic. H_min(cycling) = 0.856100 from G-002 MCMC posterior. The cycling class includes any epithelium that turns over rapidly through proliferation/differentiation — cervical mucosa fits this profile by design.

**Healthy reference β** for cervical_epithelial: not currently in the Moss 2018 25-tissue reference panel. Stage 2 deconvolution targeting cervical_epithelial is documented as a v0.2+ engineering deliverable (see §13 limitations and §14 open questions). For v0.1, Stage 2 deconvolution does not target cervical_epithelial, because the primary specimens (pap smear LBC and cervical biopsy) IS the target tissue — no deconvolution conceptually needed.

**Critical biological feature for the framework.** HPV-driven carcinogenesis differs from breast/lung/colon cancer immunology. HPV integrates into the host genome, evades MHC-I presentation via E5/E7 oncoprotein activity, and recruits a tolerogenic immune microenvironment dominated by regulatory T cells, MDSCs, and M2 macrophages. The Stage 1 immune signal in cervical lesions reflects this immune dysregulation pattern. Per VAL-073 (n=68 Verlaat Amsterdam population-normal anchor), the pooled Test 1 A_immune signal is positive and monotonic (Normal < CIN3 < SCC) on cervical tissue — but VAL-074 Farkas Stockholm (n=43, HPV-negative healthy normals) reads NEGATIVE-direction (d=−0.61), and VAL-081 Lando Oslo (n=270 tumor-only) reads tumors d=−0.43 BELOW VAL-073 normals. Two of three tissue cohorts reading negative-direction at total n=313 means VAL-073 is the outlier, not the artifact. Most likely explanation: HPV-stratification of the normal cohort matters — HPV-negative healthy cervical tissue may sit at depressed immune-class baseline relative to mixed/unspecified-HPV population normal. This is real cohort heterogeneity, not measurement error, and it places cervical-epic at `exploratory_with_cohort_heterogeneity` tier rather than the inflated `cross_platform_validated` that single-cohort VAL-073 would have produced.

---

## 2. Clinical claim of cervical-epic v0.1

**No clinical claim at v0.1.** Card is `exploratory_with_cohort_heterogeneity` per the master README v2.1 tier definition. The VAL-073 Verlaat tissue anchor is real (d=+0.73, monotonic Normal<CIN3<SCC, p=0.004) but cannot carry the card alone against VAL-074 + VAL-081 reading negative-direction at total n=313.

The card does NOT claim:
- Pre-diagnostic blood detection of cervical cancer or CIN3 (no cohort tested).
- LBC pap smear detection of CIN3 or cervical cancer (VAL-076 transferability flag; VAL-077 data integrity flag).
- Cross-cohort tissue replication of CIN3 detection (VAL-074 negative direction).
- Concordance with the published clinical-grade cervical methylation panels (FAM19A4/miR124-2 [QIAsure AUC 0.77 for CIN3], ZNF671 [GynTect], EPB41L3, PAX1/NREP-AS1 [Bowden 2025 AUC 0.92 on the same GSE287994 cohort that VAL-077 nulled]). Those panels detect strong cervical signal in the same data products where the universal Stage 1 panel reads null — that is itself a panel-transferability finding per CCL-032.

The card DOES document, as the v0.1 record:
- Cervical_epithelial sits in the cycling class, H_min = 0.856100, mathematically equivalent to cycling-class scoring with Cohen's d and CI invariant under H_min rescaling (immune vs cycling H_min differs by 2% on Xu-538 panel; the difference is a constant rescale of A-score magnitude, not a separate biological reading).
- Six VAL studies tested across all publicly accessible cohorts per CCL-029.
- VAL-078 (CINCS Bukowski 2023, 5-yr LBC pre-dx) and VAL-079 (Sundström CIN2 2026) deferred to v0.2+ contact-list pathway (data not GEO-deposited; available "from corresponding author upon request").
- The path to v0.2+ is documented in §11 of this README and includes panel substitution with published clinical-grade cervical methylation markers, raw IDAT reprocessing of GSE287994, HPV-stratified re-run of all tissue cohorts, and Test 2 lymphoid/myeloid sub-panel split when OQ-2026-01 immune-atlas staging operationalizes.

---

## 3. The universal pipeline applied to any cervical-epic IDAT

Every cervical-epic specimen follows the same universal pipeline (Master README §"Universal Stage 1/2/3 pipeline"). Specimen-specific guidance per §4 below, but the core scoring is identical:

### 3.1 Stage 1 — Xu-538 immune-class A-score (Test 1 per CCL-030)

- **Panel:** Xu-538 (538 CpGs, Xu et al. 2020 JNCI Sister Study).
- **Panel SHA-256:** `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6` (file-bytes verified at runtime).
- **n_cpgs:** 538.
- **H_min:** 0.838889 (immune class, G-003b MCMC posterior, frozen). Stage 1 ALWAYS uses H_min(immune) regardless of cervical disease tissue class. This is the universal Stage 1 rule (panc-LL-007 generalized).
- **Score:** `A_pooled = mean over Xu-538 CpGs present of [ H(β) / H_min(immune) ]` where H(β) = −β·log₂(β) − (1−β)·log₂(1−β).
- **Direction expected for cervical-epic:** POSITIVE (per VAL-073 Normal vs CIN3 d = +0.73, monotonic Normal < CIN3 < SCC).
- **QC threshold:** ≥400 valid Xu-538 CpGs per sample.
- **Per CCL-031:** cervical-epic is NOT bidirectional cancellation. Pooled Test 1 is the operational metric. No directional fallback panel needed at v0.1 evidence level.

### 3.2 Stage 2 — Tissue-of-origin localization (mostly N/A for v0.1 cervical-epic)

For most Cookbook cards, Stage 2 runs Moss 2018 NNLS deconvolution against the 25-tissue reference and produces per-tissue β values that are scored against the disease-specific tissue class H_min. For cervical-epic v0.1, Stage 2 plays a different role:

- **Pap smear LBC (primary specimen):** the specimen IS cervical tissue. Stage 2 deconvolution is conceptually unnecessary — score the bulk LBC β directly against H_min(cycling) = 0.856100.
- **Cervical tissue biopsy:** same — biopsy IS cervical tissue. Direct read.
- **Plasma cfDNA:** Stage 2 Moss NNLS deconvolution would extract the cervical_epithelial fraction. **PROBLEM: cervical_epithelial is not in the Moss 2018 25-tissue reference panel.** Stage 2 cervical-epithelial fraction extraction from blood is documented as a v0.2+ engineering deliverable. Adding cervical_epithelial to the reference requires a dedicated normal-cervical reference cohort (~20 healthy cervical biopsies on EPIC) that does not exist in the public domain at sufficient size.
- **Self-sampled swab / urine:** same Stage 2 limitation as plasma — no cervical_epithelial reference available for v0.1.

**Stage 2 reading for v0.1.** Direct read against H_min(cycling) = 0.856100. The bulk specimen IS the tissue, so no deconvolution noise; this is the highest-fidelity Stage 2 read available for the LBC and tissue pathways. Tissue Stage 2 magnitude is the ceiling against which any future blood-arm Stage 2 will be calibrated when the cervical_epithelial Moss reference is added.

### 3.3 Stage 3 — Sub-composition / lineage stratification

EpiDISH RPC + Salas IDOL-Ext per the universal pipeline. For cervical-epic specifically, Stage 3 is where HPV-driven immune compartment stratification would surface — once OQ-2026-01 immune-atlas staging completes, Test 2 (lymphoid-marker vs myeloid-marker sub-panel split per CCL-030) becomes operational. Until then, Stage 3 reports general lymphoid/myeloid fractions per the universal pipeline; Test 2 lineage assignment remains a v0.2+ deliverable.

---

## 4. Specimen pathways — every IDAT input route this card supports

cervical-epic v0.1 supports five active specimen pathways and explicitly documents two as not-applicable. **Pap smear LBC is the primary clinical specimen, distinguishing cervical-epic from every other Cookbook card** — no other tracked cancer in the Cookbook has a screening-grade specimen this directly accessing the disease tissue.

### 4.1 Pap smear / liquid-based cytology (LBC) — PRIMARY

| Field | Value |
|---|---|
| Specimen | Cervical exfoliated cells in ThinPrep or SurePath cytology fluid |
| Primary pathway | YES — cervical-epic primary specimen |
| Stage 1 role | Direct A_immune scoring on bulk LBC β (immune cells are mucosa-resident lymphocytes + infiltrating leukocytes + circulating cells trapped in mucus) |
| Stage 2 role | Direct read against H_min(cycling) = 0.856100 — no deconvolution; the LBC IS cervical epithelium plus immune infiltrate |
| Stage 3 role | EpiDISH RPC for immune-compartment fractionation; Test 2 deferred to OQ-2026-01 |
| Deployment anchor | VAL-076 GSE143752 El-Zein (n=96 LBC, 42 CIN3 + 54 control) — pending; VAL-077 GSE287994 Bowden 2025 (n=247 LBC, 119 benign + 74 CIN3/CGIN + 54 cancer) — pending |
| Validated at tier | PENDING (VAL-076 VAL-077 results) |
| Pre-analytical confounds | (1) Vaginal lubricant, douche, or recent intercourse contamination; (2) cycle phase β-fluctuation (cervical mucus thickness varies); (3) postpartum collection (different cervical immune state); (4) recent IUD insertion / removal; (5) recent colposcopy or biopsy within 4 weeks; (6) acute cervicitis or BV or candidiasis (defer if symptomatic); (7) post-LEEP / post-cone collection requires modified threshold (cervical anatomy altered) |
| When to use | Standard cervical cancer screening visit; 3-5 yr cadence per ACOG/USPSTF for HPV-negative women ≥30 yr |

### 4.2 Cervical tissue biopsy — STAGE 2 CEILING REFERENCE

| Field | Value |
|---|---|
| Specimen | Cervical punch biopsy, LEEP excision, cone biopsy, or post-hysterectomy specimen |
| Primary pathway | NO — secondary, used for Stage 2 ceiling and confirmation |
| Stage 1 role | Direct A_immune scoring on bulk biopsy β; immune compartment is tumor-infiltrating + adjacent normal mucosa |
| Stage 2 role | Direct read against H_min(cycling) — highest-fidelity Stage 2 ceiling reference (no deconvolution noise) |
| Stage 3 role | EpiDISH on bulk-tumor immune fraction; gives compartment-specific lineage estimates that may differ from blood per CCL-019 |
| Deployment anchor | **VAL-073 GSE99511 Verlaat (n=68; 28 normal + 36 CIN3 + 4 SCC) — Test 1 PASSES** with d = +0.73 Normal vs CIN3, monotonic Normal < CIN3 < SCC. **Tissue-arm anchor confirmed.** |
| Cross-cohort confirmation | VAL-072 TCGA-CESC (n=3 paired) — exploratory only at n=3; CI straddled zero; superseded by VAL-073 |
| Replication | VAL-074 GSE46306 (n=43; 20 normal + 17 CIN3 + 6 cancer) — pending NCBI access; VAL-075 GSE38266 (n=42; 21 HPV+ vs 21 HPV− tumor) — pending; VAL-081 GSE68339 (n=270 cancer-only with TCGA-non-cervical normal control) — pending |
| Validated at tier | `single_cohort_validated` per VAL-073 anchor; path to `cross_platform_validated` via VAL-074/075 replication |
| Pre-analytical confounds | (1) Tumor heterogeneity within biopsy; (2) LEEP versus punch versus cone — different tissue volumes capture different compartments; (3) post-LEEP residual disease tissue is depleted in tumor, enriched in inflammatory granulation; (4) FFPE versus fresh-frozen processing β-shift documented; (5) cone-biopsy adjacent-normal contamination from transformation zone |
| When to use | Post-colposcopy histological confirmation; surgical pathology specimen integration into framework reporting |

### 4.3 Self-sampled cervicovaginal swab — EXPLORATORY for v0.1

| Field | Value |
|---|---|
| Specimen | Cervicovaginal lavage, brush, or tampon-based self-collection in preservation fluid |
| Primary pathway | NO — exploratory at v0.1; v0.2+ priority pathway for screening access expansion |
| Stage 1 role | Same as LBC pap smear; immune signal captures mucosal infiltrate + sloughed immune cells |
| Stage 2 role | Direct read against H_min(cycling); contaminating vaginal epithelium dilutes cervical signal at ratio depending on collection device |
| Stage 3 role | EpiDISH RPC; lineage identical to LBC |
| Deployment anchor | NONE in v0.1 — no public 450K/EPIC cervicovaginal self-sample cohort located at n≥30 |
| Validated at tier | EXPLORATORY (no v0.1 cohort) |
| Pre-analytical confounds | (1) Contaminating vaginal epithelium dilution (ratio device-dependent); (2) menstrual blood contamination; (3) recent coitus (sperm DNA contamination); (4) diurnal variation in cervical mucosa shedding; (5) self-collection technique variation (depth of insertion, rotation count); (6) shipping ambient-temperature stability of preservation fluid |
| When to use | Screening-access expansion in low-resource settings; remote-access populations; patients declining clinician-collected cytology |
| Path to validation | v0.2+: identify or generate self-sample EPIC cohort with paired clinician-LBC ground truth; may require partner lab generation rather than public-data analysis |

### 4.4 Plasma cfDNA — Stage 1 immune-class flag, exploratory at v0.1 (no Stage 2 cervical_exocrine)

| Field | Value |
|---|---|
| Specimen | Peripheral blood plasma cfDNA (Streck or PAXgene tube, processed within stability window) |
| Primary pathway | NO — secondary at v0.1; would become primary if pre-diagnostic blood cervical detection were validated |
| Stage 1 role | Standard A_immune on Xu-538; expected positive direction by analogy to other cycling-class cancers; magnitude unknown without dedicated cohort |
| Stage 2 role | NOT AVAILABLE at v0.1 — cervical_epithelial absent from Moss 2018 reference; v0.2+ engineering deliverable |
| Stage 3 role | Standard EpiDISH; cervical-cancer-specific blood immune signature undocumented at the panel level |
| Deployment anchor | NONE in v0.1 — no public per-patient blood cervical pre-dx β cohort located |
| Literature support | Lindroth 2024 (Acta Obstet Gynecol Scand) detected FAM19A4/miR124-2 hypermethylation in LBC samples up to 8 years pre-AIS/ADC by qPCR — supports long-window detectability hypothesis but is qPCR not array-based |
| Validated at tier | EXPLORATORY (no v0.1 cohort); listed in pathway documentation for completeness |
| Pre-analytical confounds | Standard cfDNA confounds (hemolysis, time-to-spin, freeze-thaw); plus cycle-phase variation in cervical cell shedding into circulation; postpartum elevated cfDNA baseline |
| When to use | Adjunct to LBC primary pathway; future v0.2+ pre-diagnostic monitoring cohort |
| Path to validation | UK Biobank cervical subset application; partner-generated longitudinal blood cohort with paired LBC ground truth |

### 4.5 Urine cfDNA — EXPLORATORY (cell-free urine signal)

| Field | Value |
|---|---|
| Specimen | First-morning void urine, processed for cfDNA isolation |
| Primary pathway | NO — exploratory at v0.1; HPV DNA in urine has been validated in screening but methylation array work is sparse |
| Stage 1 role | Same Xu-538 A_immune scoring; immune signal source is uroepithelial-resident immune cells + circulating cell-free immune DNA |
| Stage 2 role | NOT AVAILABLE at v0.1 |
| Stage 3 role | Standard pipeline; cervical-specific signal magnitude in urine undocumented at array level |
| Deployment anchor | NONE in v0.1 |
| Literature support | Multiple HPV-DNA-in-urine screening studies; methylation work limited to single-CpG qPCR (FAM19A4 etc.) |
| Validated at tier | EXPLORATORY |
| Pre-analytical confounds | (1) Urine cfDNA fragmentation kinetics differ from plasma; (2) hydration status modulates concentration; (3) menstrual contamination; (4) UTI / bacteriuria contributes bacterial DNA contamination |
| When to use | Adjunct to clinician-LBC; population screening in regions with low colposcopy access |
| Path to validation | v0.3+: dedicated urine-cfDNA EPIC cohort with paired LBC ground truth |

### 4.6 CSF — NOT APPLICABLE

Cervical cancer does not preferentially shed methylation signal into cerebrospinal fluid. CSF sampling is invasive (lumbar puncture), provides no expected cervical-cancer diagnostic information, and is documented here only so a future operator does not waste a sample attempting this pathway. NOT APPLICABLE.

### 4.7 Saliva — NOT APPLICABLE

No mechanism exists for cervical-tissue-derived methylation signal to appear preferentially in saliva. Buccal cell methylation is its own substrate but has no cervical-disease-specific signature documented. NOT APPLICABLE.

### 4.8 Other candidate specimens evaluated and classified

Per Block 4 Master README expectation (every candidate specimen must be explicitly classified, never silently omitted):

| Specimen | Status for cervical-epic | Reason |
|---|---|---|
| Sputum | NOT APPLICABLE | No cervical signal in respiratory tract specimens |
| Stool | NOT APPLICABLE | No cervical signal in gastrointestinal tract specimens |
| Cervical swab (clinician-collected, dry) | EXPLORATORY | Differs from LBC by preservation medium; signal expected similar to LBC but not validated independently |
| FNA cytology (cervical) | EXPLORATORY | Used in evaluation of bulky cervical mass; same conceptual specimen as biopsy at smaller volume |
| Ascites | NOT APPLICABLE | Late-stage cervical metastasis presentation; not a screening or diagnostic pathway |
| Pleural effusion | NOT APPLICABLE | Same — late metastasis only |
| BAL (bronchoalveolar lavage) | NOT APPLICABLE | No cervical signal in respiratory specimens |
| Semen | NOT APPLICABLE | Cross-partner HPV transmission research uses semen; not a cervical-cancer-detection specimen |
| Sweat | NOT APPLICABLE | No cervical signal mechanism |
| Breast milk | NOT APPLICABLE | No cervical signal mechanism |
| Vaginal discharge | EXPLORATORY | Adjacent to cervical mucosa; signal contamination from vaginal epithelium expected |
| Nasal swab | NOT APPLICABLE | No cervical signal mechanism |
| Ear cerumen | NOT APPLICABLE | No cervical signal mechanism |
| ERCP juice | NOT APPLICABLE | Pancreatobiliary specimen, not cervical |

---

## 5. CCL-027 four-question bidirectional cancellation guard (clean pass per VAL-073)

Per CCL-027 + CCL-030 + CCL-031, every card v0.1 answers four questions in both README and JSON. cervical-epic answers all four cleanly. **This is the first Cookbook card whose four-question guard passes without triggering directional fallback construction or flagging bidirectional-cancellation risk** (per VAL-073 outcome §"Card consequences").

### Question 1 — Pooled-entropy expected direction (Test 1 per CCL-030)

**Answer:** POSITIVE.
**Citation:** VAL-073 GSE99511 Verlaat: Normal vs CIN3 d = +0.7253, 95% CI [+0.216, +1.235], p = 0.004; monotonic Normal < CIN3 < SCC; CIN3 magnitude 79% of SCC magnitude.
**Cross-cohort consistency check:** VAL-072 TCGA-CESC (n=3 paired exploratory): paired d = +1.26, CI straddled zero — direction consistent with VAL-073 even at n=3.

### Question 2 — Bidirectional-cancellation risk per CCL-031

**Answer:** LOW. Cervical-epic is NOT in the bidirectional-cancellation category. **Test 1 (pooled A_immune) PASSES cleanly with d = +0.73 Normal vs CIN3 and lower CI > 0.** There is no observed pooled-vs-directional discrepancy. The cohort-mean Δβ direction percentage is dominantly negative (37% positive at the cohort level per VAL-073), but per CCL-030 this is descriptive only — Shannon entropy is symmetric around β = 0.5, so per-patient entropy elevation is positive regardless of cohort-mean Δβ direction. **Per CCL-031 explicitly: cervical-epic does NOT exhibit the AD-instance pooled-null + directional-pass pattern; the observation that cohort-mean β values move dominantly downward is NOT bidirectional cancellation.**

### Question 3 — Directional-panel fallback specification

**Answer:** NONE NEEDED at v0.1 evidence level. Test 1 pooled A_immune is the operational Stage 1 metric for cervical-epic in every supported specimen pathway. If a future cohort produces a Test 1 null where a Test 1 pass was expected, a per-CpG ±1 z-scored panel could be built; this is not currently triggered.

### Question 4 — Lymphoid-vs-myeloid expected pattern (Test 2 per CCL-030, pending OQ-2026-01)

**Answer (literature-anchored expected pattern only at v0.1):** HPV-driven cervical lesions involve LYMPHOID SUPPRESSION + MYELOID EXPANSION. Specifically: MHC-I downregulation by HPV E7 oncoprotein blocks effector CD8+ T-cell recognition (Zhou 2020); chronic HPV-E7 antigen exposure drives effector T-cell exhaustion via PD-1/Tim-3/Lag-3 upregulation (de Vos van Steenwijk 2013); regulatory T-cell expansion in cervical lesions documented (Kobayashi 2008, Visser 2007); myeloid compartment shifts toward MDSCs and M2-polarized tumor-associated macrophages (Chen 2018, Galliverti 2020). Reviews: Stanley 2010 (Vaccine), Clarke 2020 (Cancer Letters).

**Operational status:** Test 2 (lymphoid-marker vs myeloid-marker sub-panel split on Xu-538) is PENDING OQ-2026-01 immune-atlas staging — Salas IDOL-Ext per-CpG lineage assignment is not currently runnable on any disease in the record. The cervical-specific lineage prediction above is a literature-anchored hypothesis only at v0.1; it is NOT directly measured at the Xu-538 panel level. When OQ-2026-01 becomes operational, cervical-epic will run Test 2 on the same VAL-073 GSE99511 cohort to test the lymphoid-down + myeloid-up prediction.

---

## 6. Validation summary (VAL studies in this card)

| VAL | Cohort | Specimen | n | Primary result | Status |
|---|---|---|---|---|---|
| VAL-071 | (Landscape survey) | n/a | n/a | 11 candidate cohorts identified | COMPLETE |
| VAL-072 | TCGA-CESC HM450 | Tissue biopsy | 3 paired | Paired d = +1.26 [−0.26, +2.78] CI straddles zero | EXPLORATORY (n=3) |
| **VAL-073** | **GSE99511 Verlaat 2018 (Amsterdam) HM450** | **Cervical tissue biopsy** | **28 N + 36 CIN3 + 4 SCC** | **Normal vs CIN3 d = +0.73 [+0.22, +1.24] p=0.004; monotonic Normal<CIN3<SCC** | **TISSUE ARM ANCHOR (positive)** |
| VAL-074 | GSE46306 Farkas 2013 (Stockholm) HM450 | Cervical tissue biopsy | 20 HPV-neg N + 17 CIN3 + 6 cancer | Normal vs CIN3 d = −0.61 [−1.27, +0.05]; cancer d = +0.89 [−0.05, +1.83] | O5_NEGATIVE_DIRECTION (HPV-negative normal baseline) |
| VAL-075 | GSE38266 | n/a | n/a | EXCLUDED — cohort is HNSCC (head/neck), NOT cervical (landscape error caught at runtime) | EXCLUDED |
| VAL-076 | GSE143752 El-Zein 2020 (Quebec) EPIC 850K | LBC pap smear | 54 H + 50 CIN1 + 40 CIN2 + 42 CIN3 | Healthy vs lesion d = −0.11 [−0.43, +0.20] flat | O6_UNEXPECTED — panel-transferability flag (Xu-538 buffy-coat trained, LBC = different cell mixture) |
| VAL-077 | GSE287994 Bowden 2025 (Imperial London) EPIC 850K | LBC pap smear | 115 benign + 126 disease | Benign vs disease d = −0.03 flat | O6_UNEXPECTED — data integrity flag, deferred to v0.2+ (supplementary file is residual M-values per Bowden 2025 Methods, not raw β; CHK-3.1 distribution check failed: 50% in [0.4, 0.6] vs 12% extremes) |
| VAL-078 | CINCS Bukowski 2023 (UNC) | LBC 5-yr pre-dx | 148-289 | Data not GEO-deposited; "available from corresponding author upon request" | DEFERRED to v0.2+ contact list |
| VAL-079 | Sundström CIN2 2026 | LBC active surveillance | 58 | Data access pending direct PI contact | DEFERRED to v0.2+ contact list |
| VAL-081 | GSE68339 Lando 2015 (Oslo) HM450 | Cervical tumor biopsy | 270 SCC tumors (no internal normals) | External vs VAL-073 normals d = −0.43 [−0.82, −0.04]; only 6.7% of tumors above VAL-073 normal p95 | O5_NEGATIVE_DIRECTION (confirms VAL-074 cohort-direction-flip pattern at large n) |

**Tissue arm summary at total n=313:**

| Cohort | Country | Normal definition | Disease vs normal d |
|---|---|---|---|
| VAL-073 Verlaat (Amsterdam) | NL | population-normal cervical tissue, no CIN history | **+0.73 POSITIVE** (anchor) |
| VAL-074 Farkas (Stockholm) | SE | HPV-negative healthy cervical | **−0.61 NEGATIVE** |
| VAL-081 Lando (Oslo) | NO | n/a (cancer-only; external comparator vs VAL-073 normals) | **−0.43 NEGATIVE** |

Two of three tissue cohorts read disease at or below VAL-073's normal baseline. **VAL-073 is the outlier**, not the artifact. Most likely explanation: HPV-stratification of normals matters — VAL-074's HPV-negative healthy selection sits at depressed immune-class baseline relative to VAL-073's mixed/unspecified-HPV population normal. Without HPV-stratification of normals across all three cohorts, the cohort-direction-flip cannot be cleanly resolved.

**LBC primary pathway summary:**

VAL-076 and VAL-077 cannot establish or refute LBC detection at v0.1. VAL-076 is a real raw-β cohort but the Xu-538 panel was trained on buffy-coat (whole blood) and LBC is a fundamentally different cell mixture (~80% exfoliated cervical epithelium + ~10-20% mucosal-resident lymphocytes + variable mucus); the flat A-score across CIN grades is most likely a panel-transferability issue rather than absence of biology. VAL-077's supplementary file is residual M-values per Bowden 2025 Methods, not raw β; raw IDAT processing through minfi/sesame from `GSE287994_RAW.tar` is required for v0.2+. Notably, the same Bowden 2025 paper achieved AUC 0.92 for cervical disease detection on this cohort using PAX1/NREP-AS1 methylation — the cervical immune signal IS in the raw data, just not accessible through the framework's universal Stage 1 panel on this data product.

**Cohort-completeness rule (CCL-029):** This v0.1 has run every publicly accessible 450K/EPIC cervical methylation cohort. Gated/contact-required cohorts (VAL-078 CINCS, VAL-079 Sundström) are documented as v0.2+ next-validation-steps per CCL-029, not omissions.

**Diagnostic-order rule (CCL-032):** Every null/negative outcome in this card cites the data-integrity check, biology-consistency check, and framework-finding interpretation separately. VAL-076 and VAL-077 were initially drafted as O3_NULL framework findings; both were reclassified to O6_UNEXPECTED after CCL-032 was applied — VAL-076 as panel-transferability flag (clinical-grade cervical methylation panels detect strong signal in LBC, framework's universal Stage 1 panel does not transfer), VAL-077 as data integrity flag (residual M-values, not raw β).

## 7. Mandatory covariates and confounds — every report field

Cervical-epic has the most extensive mandatory covariate list of any Cookbook card to date. HPV stratification alone adds 4 categorical levels (negative, 16+, 18+, other hr-HPV, multiple) above the standard universal covariates.

| Covariate | Stratify analysis | Report field | Rationale |
|---|---|---|---|
| Age | Yes (decade) | Yes | Cervical incidence peaks 35-44 yr; younger patients more likely transient HPV |
| Sex | n/a (cervical-epic is female-only) | Yes | Card explicitly female-only; male-anatomy patients excluded from scoring |
| **HPV status (negative / 16+ / 18+ / other hr-HPV / multiple)** | **MANDATORY** | **MANDATORY** | HPV genotype is the strongest stratifier of cervical methylation; VAL-075 quantifies the magnitude |
| HPV vaccination status | Yes | Yes | HPV-vaccinated populations have different progression risk profile and different baseline methylation |
| Cytology result at LBC collection (NILM / ASCUS / LSIL / HSIL / AGC) | **MANDATORY** | **MANDATORY** | Contextualizing variable for the methylation read; orthogonal information |
| **Immunosuppression (HIV+, transplant, autoimmune Rx)** | **MANDATORY** | **MANDATORY** | HIV+ women have ~6× cervical cancer risk via HPV persistence; immune compartment baseline shifted |
| **Prior CIN treatment (LEEP, cone, ablation, hysterectomy)** | **MANDATORY** | **MANDATORY** | Treatment alters cervical anatomy and baseline methylation; recurrence detection requires modified threshold |
| Pregnancy status | Yes (decline scoring if pregnant) | Yes | Pregnancy-associated hormonal state alters cervical immune signal; defer scoring |
| Recent vaginal infection (BV, candidiasis, trichomoniasis) | Yes (defer if active) | Yes | Acute inflammation contaminates Stage 1 immune signal |
| Time since last menstrual period | Observation | Yes | Cervical cell turnover varies through menstrual cycle |
| Hormonal contraceptive use (duration) | Yes | Yes | OC duration ≥5 yr is documented cervical risk modifier |
| Parity | Observation | Yes | High parity (≥4) is a cervical risk factor |
| Smoking status (current / former / never; pack-years) | Yes | Yes | Modifies cervical immune compartment; documented cervical co-factor |
| Race / ethnicity | Observation | Yes | African-American and Hispanic populations have higher cervical incidence (access-driven, not pure biology); documented for completeness |
| Recent acute infection (<2 wk) | Yes (defer scoring) | Yes | Standard universal covariate per Master README §8 |
| Hemolysis at draw (cfDNA pathway only) | Yes (decline scoring) | Yes | Standard cfDNA confound |
| Recent transplant / transfusion / chimerism | Yes (decline scoring) | Yes | Standard universal covariate |
| Active or recent chemo / radiation | Yes (decline scoring) | Yes | Standard universal covariate; additionally, prior cervical radiation alters all downstream readings |
| Family history of cervical cancer | Observation | Yes | Genetic susceptibility not strongly established for cervical; reported for completeness |
| BMI | Observation | Yes | Standard universal covariate |
| Alcohol | Observation | Yes | Standard universal covariate |
| Diurnal collection time | Observation | Yes | Standard universal covariate (cervical cell shedding diurnal variation) |
| Fasting status | Observation | Yes (cfDNA pathway only) | Standard universal covariate |
| **Last pap smear date** | **MANDATORY** | **MANDATORY** | Determines screening interval and risk stratification |
| **Last colposcopy date** | **MANDATORY** | **MANDATORY** | Recent colposcopy / biopsy alters baseline methylation |

---

## 8. Tier thresholds and clinical action matrix

Universal Cookbook tier structure: NORMAL < 1.01, MARGINAL ≥ 1.01, DETECTABLE ≥ 1.05, URGENT ≥ 1.07, FLOOR BREACH ≥ 1.10. Cervical-epic uses the universal thresholds at v0.1 — VAL-073 magnitudes (Normal A = 0.681, CIN3 A = 0.699, SCC A = 0.708) sit well below the universal thresholds because the tissue-arm anchor is Stage 1 immune scoring, not Stage 2 cycling-class scoring. Tier interpretation for cervical-epic Stage 1 differs from the universal scale: the SCC reading at A = 0.708 is positive elevation relative to the n=28 healthy reference baseline at A = 0.681, with d = +1.27 effect size. **Cervical-epic Stage 1 tier thresholds will be expressed in z-units relative to the cohort-specific healthy baseline rather than absolute A-values, calibrated against VAL-073 norms.**

| Tier | Stage 1 z-score (relative to healthy baseline) | Stage 2 cycling A-score | Action |
|---|---|---|---|
| NORMAL | z < +0.5 | A < 1.01 | Continue routine screening per ACOG/USPSTF guideline cadence |
| MARGINAL | +0.5 ≤ z < +1.0 | 1.01 ≤ A < 1.05 | Repeat in 6-12 months; ensure HPV co-test current; reflex to colposcopy if cytology ASCUS+ |
| DETECTABLE | +1.0 ≤ z < +1.5 | 1.05 ≤ A < 1.07 | Colposcopy referral within 4 weeks; HPV genotyping if not current; cytology review |
| URGENT | +1.5 ≤ z < +2.0 | 1.07 ≤ A < 1.10 | Expedited colposcopy with biopsy; HPV genotyping mandatory; gyn-oncology consult if biopsy CIN3+ |
| FLOOR BREACH | z ≥ +2.0 | A ≥ 1.10 | Same-week colposcopy with directed biopsy + ECC; gyn-oncology referral; staging workup if biopsy invasive |

Disease-specific special rules:
- **HPV-negative + DETECTABLE+ Stage 1:** same colposcopy referral but flag for non-HPV cervical pathology (including rare HPV-independent cervical adenocarcinoma).
- **HIV+ women:** lower threshold for colposcopy referral (MARGINAL → colposcopy at next visit).
- **Post-LEEP / post-cone:** modified baseline; first 6 months use serial sampling regardless of tier.
- **Pregnancy:** defer scoring; resume at 6 weeks postpartum.

---

## 9. Trajectory monitoring guidance

Cervical-epic is unique among Cookbook cards in having a slow pre-clinical progression measured in years rather than months. Trajectory cadence:

- **High-risk patient cadence** (HPV+ AND prior CIN, OR immunosuppressed, OR DETECTABLE prior reading): every 6 months
- **Average-risk patient cadence** (HPV-negative, no prior CIN, NORMAL reading): every 3-5 years per ACOG/USPSTF
- **Active-surveillance CIN2** (Sundström 2026 cohort relevant): every 6 months for 24 months minimum
- **Trajectory slope diagnostic threshold:** > +0.3 z-units per year suggests active progression — consider escalation regardless of absolute tier
- **Two-consecutive-MARGINAL escalation rule:** consecutive MARGINAL readings 6 months apart triggers DETECTABLE-tier action even if individual readings stay MARGINAL

---

## 10. Known limitations of cervical-epic v0.1

1. **Healthy reference β for cervical_epithelial not in Moss 2018.** Stage 2 deconvolution from blood / urine to cervical_epithelial fraction is not available at v0.1. The bulk-LBC and bulk-tissue pathways do NOT need this; pathways requiring deconvolution (plasma, urine) cannot operationalize Stage 2 in v0.1.

2. **Tier thresholds calibrated on Stage 1 immune signal at the tissue level (VAL-073).** LBC-pathway calibration awaits VAL-076/077; if LBC magnitudes differ substantially from tissue-biopsy magnitudes, tier thresholds will need re-calibration before LBC-pathway clinical deployment.

3. **HPV-stratified threshold differences not yet quantified.** VAL-075 (GSE38266 HPV+ vs HPV− tumors) will quantify this; until completion, a single-threshold scoring is used with explicit HPV stratification in the report metadata.

4. **No public per-patient pre-diagnostic blood cervical cohort located.** Plasma cfDNA pathway is exploratory at v0.1. Long-window pre-dx detection in LBC is conditional on VAL-078 CINCS Bukowski accessibility (5-yr FU prospective).

5. **Self-sampled cervicovaginal swab pathway has no v0.1 validation cohort.** Listed as exploratory; v0.2+ deliverable.

6. **Test 2 lymphoid-vs-myeloid lineage assignment is not operational.** OQ-2026-01 immune-atlas staging is required and not yet runnable. Question (iv) of CCL-027 is answered by literature only at v0.1.

7. **HM27-platform cohorts excluded.** GSE41384, GSE30759, GSE30760 cannot be evaluated by Xu-538 panel scoring; coverage is incomplete on HM27.

8. **WID-CIN (EUTOPS Innsbruck) and POBASCAM (Dutch) are gated.** v0.2+ priority access via PI contact.

9. **TCGA-CESC matched-pair pool limited to n=3.** VAL-072 is exploratory at the smallest tissue-arm pool of any Cookbook card; tissue-arm anchor relies on VAL-073 GSE99511 instead.

10. **VAL-073 does not include CIN1 or CIN2 samples.** Pre-CIN3 detection magnitude is extrapolated from the Normal-vs-CIN3 anchor; VAL-079 Sundström 2026 will add CIN2 active-surveillance detection if cohort is accessible.

11. **No validation in HPV-vaccinated cohorts.** All v0.1 validation cohorts pre-date widespread HPV vaccination or do not stratify by vaccination status. Vaccinated-population calibration is a v0.2+ deliverable as long-term-vaccinated cohorts age into screening years.

12. **LBC pre-analytical confound landscape larger than other Cookbook specimens.** Cycle phase, lubricant contamination, recent intercourse, IUD insertion, and acute cervicitis all affect LBC β values — quantification of each confound's magnitude is a v0.2+ engineering deliverable.

---

## 11. Open questions for v0.2+

| OQ | Open question | Source | Action needed |
|---|---|---|---|
| OQ-CRV-01 | Does Xu-538 transfer to LBC samples, where the immune compartment is mucosa-resident rather than buffy-coat circulating? | VAL-076/077 will answer empirically | Run VAL-076 GSE143752 + VAL-077 GSE287994 |
| OQ-CRV-02 | What is the magnitude of HPV-stratified A_immune signal difference (HPV+ vs HPV− cervical cancer)? | VAL-075 | Run VAL-075 GSE38266 |
| OQ-CRV-03 | What is the pre-diagnostic LBC detection window for the framework? | VAL-078 CINCS conditional on access | Run VAL-078 if GEO-accessible; if gated, contact Smith J / Bukowski A at UNC for collaboration |
| OQ-CRV-04 | Does the framework predict CIN2 regression vs non-regression in active surveillance? | VAL-079 | Run VAL-079 Sundström 2026 |
| OQ-CRV-05 | Add cervical_epithelial to Moss 2018 reference panel for plasma/urine Stage 2 deconvolution. | Stage 2 limitation §10.1 | Assemble healthy normal cervical EPIC reference (~20 samples); v0.3+ engineering |
| OQ-CRV-06 | Validate self-sampled cervicovaginal swab equivalence to clinician-collected LBC. | §4.3 exploratory pathway | v0.2+ cohort hunt or partner-lab generation |
| OQ-CRV-07 | Validate plasma cfDNA pre-diagnostic cervical detection window. | §4.4 exploratory pathway | UK Biobank cervical subset application; partner-lab longitudinal cohort |
| OQ-CRV-08 | Calibrate tier thresholds in HPV-vaccinated populations as they age into screening years. | §10.11 limitation | Long-term study; v0.2+ priority |
| OQ-CRV-09 | Operationalize Test 2 lymphoid-vs-myeloid sub-panel split for cervical-epic via OQ-2026-01 immune-atlas staging. | §5 question 4, CCL-030 | Cross-card priority blocked on OQ-2026-01 |
| OQ-CRV-10 | Quantify LBC pre-analytical confound magnitudes (cycle phase, lubricant, IUD, recent intercourse). | §10.12 | v0.2+ engineering; partner-lab paired-LBC studies |

---

## 12. Sources and citations

**Cervical cancer epidemiology and biology:**
- Sung H, Ferlay J, Siegel RL, et al. (2022). Global Cancer Statistics 2020: GLOBOCAN. CA Cancer J Clin. DOI: 10.3322/caac.21660
- Crow JM (2012). HPV: The global burden. Nature 488, S2-S3.
- Stanley M (2010). Pathology and epidemiology of HPV infection in females. Gynecol Oncol 117(2 Suppl):S5-S10.
- Clarke MA, Wentzensen N (2020). Strategies for screening and early detection of anal cancers. Cancer Cytopathol 128(7):447-460.

**HPV-driven immunology:**
- Zhou C, Tuong ZK, Frazer IH (2019). Papillomavirus immune evasion strategies target the infected cell and the local immune system. Front Oncol 9:682.
- Visser J, Nijman HW, Hoogenboom BN, et al. (2007). Frequencies and role of regulatory T cells in patients with HPV-related cervical (pre)neoplasia. Clin Exp Immunol 150(2):199-209.
- de Vos van Steenwijk PJ, Heusinkveld M, Ramwadhdoebe TH, et al. (2010). An unexpectedly large polyclonal repertoire of HPV-specific T cells is poised for action in patients with cervical cancer. Cancer Res 70(7):2707-2717.

**Methylation array cohorts (cervical):**
- Verlaat W, Snijders PJF, Novianti PW, et al. (2018). Genome-wide DNA methylation profiling reveals methylation markers associated with 3rd generation cervical cancer screening. Oncotarget 9:35064-35076. **VAL-073 cohort (GSE99511).** DOI: 10.18632/oncotarget.20454
- Farkas SA, Milutin-Gašperov N, Grce M, Nilsson TK (2013). Genome-wide DNA methylation assay reveals novel candidate biomarker genes in cervical cancer. Epigenetics 8:1213-1225. **VAL-081 cohort (GSE68339).**
- Bowden SJ, Ellis LB, Doulgeraki T, et al. (2025). DNA methylation signatures of cervical pre-invasive and invasive disease: An EWAS. Int J Cancer 157(2):305-316. **VAL-077 cohort (GSE287994).** DOI: 10.1002/ijc.35406
- El-Zein M, Cheishvili D, Gotlieb W, et al. (2020). Genome-wide DNA methylation EWAS in cervical cytology samples. **VAL-076 cohort (GSE143752).**
- Bukowski A, Hoyo C, Vielot NA, et al. (2023). Epigenome-wide methylation and progression to high-grade cervical intraepithelial neoplasia (CIN2+): a prospective cohort study in the United States. BMC Cancer 23:1056. **VAL-078 cohort (CINCS).** DOI: 10.1186/s12885-023-11518-6

**FAM19A4/miR124-2 LBC literature:**
- Lindroth Y, Borgfeldt C, Thorn G, Bjelkenkrantz K, Forslund O (2024). Cervix cytology samples revealed increased methylation of the human markers FAM19A4/miR124-2 up to 8 years before adenocarcinoma. Acta Obstet Gynecol Scand. DOI: 10.1111/aogs.14707
- van der Graaf Y, Lehtinen M, Suehnel C, et al. (2024). Cervical cancer screening using DNA methylation triage in a real-world population. Nat Med 30:1689-1696.
- Bonde J, Floore A, Ejegod D, et al. (2020). Methylation markers FAM19A4 and miR124-2 as triage strategy for primary HPV screen positive women. Int J Cancer 148(2):396-405.

**Clinical guidelines:**
- ACOG Practice Bulletin No. 168: Cervical Cancer Screening and Prevention (2016, reaffirmed 2020).
- USPSTF Recommendation Statement: Screening for Cervical Cancer (2018).
- ASCCP Risk-Based Management Consensus Guidelines (2019).

**Framework references:**
- Xu Z, Sandler DP, Taylor JA (2020). Blood DNA methylation and breast cancer: a prospective case-cohort analysis in the Sister Study. JNCI 112(1):87-94. **Xu-538 panel source.**
- Moss J, Magenheim J, Neiman D, et al. (2018). Comprehensive human cell-type methylation atlas reveals origins of circulating cell-free DNA in health and disease. Nat Commun 9:5068. **Moss 2018 reference panel.**

---

## 13. Pre-registration chain (full reproducibility)

| VAL | Pre-reg SHA-256 | Manifest / Series Matrix SHA-256 | Results JSON SHA-256 | Status |
|---|---|---|---|---|
| VAL-072 | 5a72e1ec4f3379f1406c747457b00a74952e27c57c598622612ddb43c35a5aaf | manifest 434c9f2b10570bfc1d92ae2ea0b83cce3218ed9b82898909d7b3f0625d0dd6d9 | b2ea81a380f38284a7809ed65d200c9b854b496d08ba32508a257d7a959a4476 | SEALED + COMPLETE |
| VAL-073 | f4f637c313c2b6250ce62887bf151640f8ef80dd54cae2dda4c743a063f42d0b | matrix 1bde17e6a236c78d18370fe8a98a5c4a21de3c32e8d4447076f1dad239074339 | d401f40d89bbf88031ab9537008b65c0af1acd5b413fe2f0d41b1ac68dcb65b8 | SEALED + COMPLETE |
| VAL-074 | (pending NCBI access) | (pending) | (pending) | PENDING SEAL |
| VAL-075 | (pending) | (pending) | (pending) | PENDING SEAL |
| VAL-076 | (pending) | (pending) | (pending) | PENDING SEAL |
| VAL-077 | (pending) | (pending) | (pending) | PENDING SEAL |
| VAL-078 | (pending GEO accessibility check) | (pending) | (pending) | PENDING ACCESS |
| VAL-079 | (pending) | (pending) | (pending) | PENDING SEAL |
| VAL-081 | (pending) | (pending) | (pending) | PENDING SEAL |

**Xu-538 panel SHA-256 (constant across all cervical-epic VALs):** ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
**RNG seed (constant across all cervical-epic VALs):** 20260425

---

## 14. Reproduction bundle

**GitHub-pushed (IAM-Validation/Biological_Physics/validation_runs/cervical-epic/):**
- VAL-072 / VAL-073 / VAL-074 .. VAL-081 Python scripts
- VAL-072 / VAL-073 / VAL-074 .. VAL-081 prereg.md files
- VAL-072 / VAL-073 / VAL-074 .. VAL-081 PREREG_SEAL.txt files
- VAL-072 / VAL-073 / VAL-074 .. VAL-081 outcome.md files
- VAL-072 / VAL-073 / VAL-074 .. VAL-081 results.json files
- VAL-072 manifest JSON; VAL-073 manifest JSON
- Updated Biological_Physics/README.md

**Heath-only Cookbook IP (NEVER GitHub):**
- cervical-epic_README.md (this file)
- cervical-epic_card_v0.1.json
- cervical-epic_directional_panel.json — NOT BUILT (not needed per CCL-031)
- VAL-071_landscape_survey.md

---

## 15. Lessons learned (cervical-epic-specific)

The full text of cerv-LL-001 through cerv-LL-016 is in `cervical-epic_LESSONS_LEARNED.md` and the master `LESSONS_LEARNED.md`. Summary catalog below.

### v0.1 design lessons (cerv-LL-001 through cerv-LL-007)

**cerv-LL-001.** Cycling-class tissue arm at proper power (n=68 in VAL-073) produces clean monotonic Normal < CIN3 < SCC progression with d = +0.73 Normal vs CIN3 and d = +1.27 Normal vs SCC. CIN3 magnitude reaches 79% of SCC magnitude. **Caveat added at v0.1 final review:** this VAL-073 anchor reading does NOT replicate at VAL-074 (Farkas Stockholm, n=43, d=−0.61) or VAL-081 (Lando Oslo, n=270 vs external normals, d=−0.43). The cross-cohort failure is documented in cerv-LL-010 below; cerv-LL-001 stands as the within-cohort observation but cannot be elevated to a card-level claim.

**cerv-LL-002.** Originally read: "cervical-epic passes CCL-027 four-question guard cleanly." Revised after VAL-074 + VAL-081: cervical-epic exhibits a cohort-direction-flip pattern (CCL-019 category, like crc-epic blood-vs-tumor), NOT bidirectional cancellation per CCL-031. The 37% per-CpG positive cohort-Δβ split at VAL-073 is descriptive only per CCL-030. The card does not need a directional fallback panel (Test 1 pooled passes in the anchor cohort), but it does need either HPV-stratified re-runs or panel substitution to resolve the cohort-direction-flip.

**cerv-LL-003.** TCGA-CESC at n=3 (VAL-072) produced 47.9% per-CpG positive Δβ; VAL-073 at n=68 produced 37.3% — 10-point divergence at low-vs-proper sample size. Per-CpG percentages are noisy small-sample statistics, not mechanism diagnostics. CCL-030 reaffirmed.

**cerv-LL-004.** LBC pap smear was originally framed as the structural advantage of cervical-epic — the screening specimen IS the disease tissue, no Stage 2 NNLS deconvolution needed. **Revised at v0.1 final review:** the structural advantage is real but it does NOT translate to immediate panel transferability. Xu-538 was buffy-coat trained; LBC is exfoliated epithelium + mucosal-resident lymphocytes (different cell mixture). The "screening-relevant primary specimen" advantage is contingent on a cervical-LBC-specific panel that does not yet exist (v0.2+ engineering). See cerv-LL-011.

**cerv-LL-005.** HPV stratification adds 4 categorical levels (negative / 16+ / 18+ / other hr / multiple) — the largest mandatory-stratification axis in any Cookbook card. **Originally cited VAL-075 to quantify the magnitude** (revised: VAL-075 was excluded as HNSCC, see cerv-LL-008). VAL-074's HPV-negative-healthy normals reading 0.06 below VAL-073's mixed/unspecified-HPV population normals (cerv-LL-010) is the v0.1 evidence that HPV-stratification of normals matters at a magnitude comparable to or larger than the disease signal itself.

**cerv-LL-006.** Cervical_epithelial is not in Moss 2018 reference. Stage 2 deconvolution from blood/urine to cervical_epithelial is not operational at v0.1; v0.2+ engineering deliverable. Bulk-LBC and bulk-tissue pathways do NOT depend on this — they read the disease tissue directly.

**cerv-LL-007.** Stage 1 ALWAYS uses H_min(immune) = 0.838889 — panel-class-governed, not tissue-class-governed (panc-LL-007 generalized). Cervical_epithelial sits in cycling class (H_min(cycling) = 0.856100), but rescaling H_min between immune and cycling produces an identical Cohen's d — the difference is a constant rescale of A-score magnitude, not a separate biological reading. To genuinely score the cervical epithelial cells separately requires a different PANEL (cervical-LBC-trained), not a different H_min.

### v0.1 build lessons (cerv-LL-008 through cerv-LL-016, this session)

**cerv-LL-008.** Landscape survey errors must be caught at landscape stage, not runtime. VAL-075 GSE38266 was claimed as cervical HPV-stratified; runtime Sample_title check revealed HNSCC (head/neck), not cervical. Now codified as CHK-1.1 in TESTING_CHECKLIST.md.

**cerv-LL-009.** Supplementary β files are NOT necessarily β values. VAL-077 GSE287994 supplementary file `_ewas_betas_2.txt.gz` was batch+chip+age+HPV-corrected residual M-values per Bowden 2025 Methods — the "_2" suffix and M-value distribution centered at 0 were warning signs. β distribution sanity check (CHK-3.1: real raw β >30% at extremes, <10% in [0.4, 0.6]) is now mandatory before scoring any supplementary β file.

**cerv-LL-010.** Healthy reference baseline shifts across cohorts are diagnostic, not invisible. VAL-073 healthy A=0.681 vs VAL-074 healthy A=0.621 — same panel, same platform, same disease, 0.06 A-units = 2.7 anchor-SDs apart. Reading Farkas 2013 paper resolved: VAL-074 selected HPV-negative healthy as normal (stricter than VAL-073's population-normal). **HPV exposure shifts the cervical immune compartment in ways the Xu-538 panel detects.** CHK-3.2 now requires healthy-vs-healthy cross-cohort comparison as the first check after a new cohort completes scoring.

**cerv-LL-011.** LBC is not buffy-coat. Specimen mixture matters more than platform. Xu-538 was selected from buffy-coat training data (Xu et al. 2020 Sister Study). LBC samples are ~80% exfoliated cervical epithelium + ~10-20% mucosal-resident lymphocytes + variable mucus and inflammatory infiltrate. Different cell mixture, different signal. CHK-0.5 now requires explicit "panel transferability not yet established" caveat in any prereg for a novel specimen pathway (LBC, urine, saliva, stool, CSF). A null on a new specimen pathway is a transferability finding, not a framework finding.

**cerv-LL-012.** Saturation flag check is mandatory before ANY null-finding outcome. VAL-077 mean A=1.011 was at 84.8% of the immune ceiling — under flag, but the saturation check should have been the first interpretation step for any near-ceiling reading. Now codified as CHK-3.5.

**cerv-LL-013.** Per-CpG cohort-mean Δβ direction percentage is descriptive only (CCL-030 reaffirmed). VAL-072 at n=3 produced 47.9% positive (looked like bidirectional cancellation). VAL-073 at n=68 produced 37.3%. 10-point swing at the same disease just from sample size. Per-CpG percentages are NEVER cited as evidence of bidirectional cancellation. CCL-030 is permanent.

**cerv-LL-014.** Common sense biology is the first check, not the last. Walther initially treated null framework readings (VAL-076, VAL-077) as biology before checking whether the readings were consistent with the published clinical-grade panels. The cervical immunology literature is overwhelming — HPV-driven inflammation, T-cell infiltration, MHC-I downregulation, Treg expansion. Clinical-grade LBC panels (FAM19A4/miR124-2, ZNF671, EPB41L3, PAX1/NREP-AS1) all detect strong signal. **If clinical-grade panels detect signal where the framework reads null, the framework's panel does not transfer — that is the finding.** Now codified as CHK-4.1 and as CCL-032.

**cerv-LL-015.** Compaction amnesia is a structural failure mode. Heath's exact words: "Every time you compact the chat you forget all this stuff and keep doing it." Memory edits survive compaction; conversation context does not. Memory #9 was rewritten this session to require `view` on TESTING_CHECKLIST.md as the FIRST tool call on any new card or new VAL session.

**cerv-LL-016.** Diagnostic order is fixed: data integrity → biology → framework. Never the reverse. The cervical-epic build burned ~4 hours on VAL-076/077 because Walther defaulted to "the data says X, therefore X" instead of "the data probably has a problem, find it." Codified as CCL-032 (master rule) and as STAGE 3 / STAGE 4 of TESTING_CHECKLIST.md.

### How these lessons connect to EDEAR commercial deployment

Cervical-epic v0.1 is not a deployable EDEAR card — that's the honest finding, and the lessons above explain why and what fixes it. For the commercial trajectory specifically:

1. **EDEAR's "one IDAT, many tests" model still holds.** Cervical-epic v0.1 nulls are NOT evidence against the architecture. They're evidence that some diseases need card-specific Stage 1 panels (the framework already accepts this — AD has a 7-CpG directional Rule A panel; PDAC has a 324-CpG directional fallback). Cervical-epic v0.2+ likely substitutes a clinical-grade cervical methylation panel (FAM19A4/miR124-2, ZNF671, PAX1/NREP-AS1, EPB41L3) with dedicated H_min calibration. The card structure stays; the panel adapts.

2. **The cohort-completeness rule (CCL-029) is the EDEAR moat.** Single-cohort validation publishing at single_cohort_validated tier is what conventional clinical methylation startups do. CCL-029 — running every publicly accessible cohort before publishing — is what surfaces real cohort heterogeneity that a competitor would publish over. cervical-epic v0.1 is the proof-of-value: VAL-073 alone would have looked like a clean tissue-arm anchor; full breadth showed it isn't.

3. **The diagnostic-order rule (CCL-032) protects against measurement-pipeline overclaim.** VAL-077 would have published as O3_NULL ("framework cannot detect cervical disease in LBC at n=247") if Walther had drafted the outcome before checking whether the supplementary file was raw β. The same paper achieved AUC 0.92 on the same cohort using PAX1/NREP-AS1. CCL-032 is what prevents the next card from publishing a measurement-pipeline artifact as a framework finding.

4. **The dual-axis honesty in EDEAR reports applies here.** Per the EDEAR product spec, every per-class tile shows A-score (architecture) plus fraction-vs-age-baseline (shedding) with a concordance indicator. Cervical-epic v0.1 in its current state would surface a low-confidence or N/A reading for cervical_epithelial in any patient report — exactly the transparency the EDEAR spec was designed to support. **Honest "we can't tell yet" is the competitive moat against competitors who would smooth over the heterogeneity.** The card's existence in this state, rather than being suppressed or inflated, is what makes EDEAR's transparency claim defensible.

5. **The HPV-stratification finding may extend to other HPV-driven cancers.** HNSCC (oropharyngeal), anal SCC, penile SCC, vulvar SCC are all HPV-driven and may show the same cohort-direction-flip pattern when normal cohorts are HPV-stratified. If a third HPV-driven disease confirms the pattern, it becomes a framework principle — like the smoking-stratification rule for lung-epic and the viral-hepatitis-stratification rule for hcc-epic. Three HPV-driven cancers would be enough to promote it from card-specific to framework-level.

6. **The path to EDEAR clinical deployment for cervical screening is clearer than the v0.1 numbers suggest.** Substituting the published clinical-grade cervical methylation panels (which already have AUC 0.77 to 0.92 on LBC pap smears) into the framework's universal Stage 1/2/3 pipeline with dedicated H_min calibration gives clinically deployable cervical screening immediately. The trade-off: cervical-epic stops being a "universal panel works for cervical too" story and becomes a "framework architecture supports clinical-grade panel substitution" story — which is the right story given what the data shows.

---

## 16. Saturation levels — cycling class A_ceiling architecture

Pulled from GAPE Reproduction Paper Part 2.4A and Part 2.4B. Cervical_epithelial sits in the cycling class. Same nucl-saturation pattern as PDAC (secretory class) and other non-pluripotent classes — only nucl saturated structurally; methyl, fuzz, wps, frag all active to BREACH.

### 16.1 Cycling class A_ceiling values (from Part 2.4A)

| Substrate | H_min(cycling) | A_ceiling = 1/H_min | Structural status | Active to BREACH? |
|---|---|---|---|---|
| methyl | 0.856100 | 1.1681 | Active | Yes |
| nucl | 0.980101 | 1.0203 | **Structurally saturated** ⚠ | No (ceiling < 1.10) |
| fuzz | 0.818993 | 1.2210 | Active | Yes |
| wps | 0.627427 | 1.5938 | Active | Yes |
| frag | 0.687948 | 1.4536 | Active | Yes |

**Structurally saturated substrate = nucl** (same as secretory-class PDAC). Nucleosome occupancy A-score for any cycling-class disease (including cervical-epic) cannot reach FLOOR_BREACH (≥ 1.10) on the nucl substrate alone. Physical feature of nucleosome occupancy in healthy cells (positioning fluctuates around 50% by design); not specific to cervical disease. Nucl is restricted to NORMAL/MARGINAL/DETECTABLE drift detection only for cycling-class diseases.

**Active substrates for BREACH-tier discrimination = methyl, fuzz, wps, frag.** Four of five substrates carry signal across the full tier range. Methylation is the primary substrate in v0.1 — every VAL study in this card uses 450K/EPIC methylation arrays. The fragmentomics substrates (wps, frag) and chromatin accessibility (fuzz) are framework-validated secondary substrates per the Reproduction Paper, operational when EDEAR multi-substrate platform reaches L2/L3 lab partnership tier.

### 16.2 Runtime saturation flag thresholds (from Part 2.4B)

| Substrate | A_ceiling | Runtime flag fires at | Interpretation when fired |
|---|---|---|---|
| methyl | 1.1681 | A ≥ 1.1631 | β has moved from healthy reference (~0.745) toward 0.5 (coin-flip state); methylation has saturated |
| nucl | 1.0203 | A ≥ 1.0153 | nucleosome occupancy saturated; structural — fires easily for cycling class |
| fuzz | 1.2210 | A ≥ 1.2160 | chromatin fuzziness saturated; total chromatin accessibility hit ceiling |
| wps | 1.5938 | A ≥ 1.5888 | windowed protection score saturated; cfDNA has lost positional protection signal |
| frag | 1.4536 | A ≥ 1.4486 | fragment-size distribution saturated; fragment heterogeneity hit ceiling |

When a runtime saturation flag fires for a substrate, that substrate is excluded from `A_active` aggregation per Reproduction Paper Part 3.3, and the patient EDEAR report carries a saturation alert. The flag does NOT indicate disease severity — it indicates measurement-channel exhaustion.

### 16.3 Cervical-epic-specific detection strategy by tier

| Tier | Primary substrate | Confirmatory substrates | Excluded substrate | Notes |
|---|---|---|---|---|
| NORMAL (A < 1.01) | methyl | nucl, fuzz, wps, frag | none | All five substrates carry healthy-baseline drift signal |
| MARGINAL (1.01 ≤ A < 1.05) | methyl | nucl, fuzz, wps, frag | none | All five substrates active in this band |
| DETECTABLE (1.05 ≤ A < 1.07) | methyl | fuzz, wps, frag | nucl approaches ceiling | Nucl A approaches 1.0203; weight nucl < 0.5 in A_combined |
| URGENT (1.07 ≤ A < 1.10) | methyl | fuzz, wps, frag | nucl saturated | Drop nucl from A_active per runtime flag |
| FLOOR_BREACH (A ≥ 1.10) | methyl, fuzz, wps, frag | (cross-substrate confirmation) | nucl (cannot reach BREACH) | Require ≥2 unsaturated substrates above 1.10 for BREACH confirmation |

For v0.1 deployment running on 450K/EPIC methylation arrays only, methyl is the operational substrate at all tiers. Non-methylation substrate guidance applies once L2/L3 multi-assay platform is operational.

### 16.4 Methylation substrate ceiling crossing — clinical interpretation

A cervical-epic patient whose methylation A-score climbs from 1.05 (DETECTABLE) toward 1.16 over serial sampling has crossed FLOOR_BREACH at 1.10 and is now within 0.005 of methylation channel saturation (1.1681 ceiling). At that point the methylation channel is exhausted as a progression metric. Continued progression must be tracked via Stage 2 cycling-class direct read on tissue/LBC, or — when L2 platform is operational — via the unsaturated chromatin substrates (fuzz, wps, frag). Saturation on methylation does NOT mean the patient is stable; it means the framework can no longer measure further deterioration through that channel.

### 16.5 Why cycling-class nucl saturation matters for cervical-epic

Cervical-epic is NOT bidirectional cancellation per CCL-031 (Test 1 pooled passes cleanly). The nucl substrate, structurally saturated for the cycling class, would normally serve as an independent cross-check on the methylation finding — and for cervical-epic v0.1, this cross-check is also unavailable for the same structural reason as PDAC. Until L2/L3 multi-substrate platform is operational, the cross-check role is filled by cross-cohort replication (VAL-073 → VAL-074 → VAL-076 → VAL-077) rather than by orthogonal substrate confirmation within a single sample. **Cervical-epic v0.1's robustness comes from cohort-completeness (CCL-029) rather than within-sample multi-substrate confirmation.**

---

## 17. Card validation tier statement

**Validation tier:** `exploratory_with_cohort_heterogeneity`.

VAL-073 GSE99511 Verlaat (Amsterdam, n=68 HM450) provides a clean tissue-arm anchor at d = +0.73 lower CI > 0 with monotonic Normal < CIN3 < SCC. That finding is preserved.

VAL-074 GSE46306 Farkas (Stockholm, n=43 HM450) reads NEGATIVE-direction at d = −0.61 [−1.27, +0.05]; reading Farkas 2013 paper confirms the cohort selected HPV-negative healthy as normal, a stricter selection than VAL-073's population-normal that sits at depressed immune-class baseline.

VAL-081 GSE68339 Lando (Oslo, n=270 HM450) reads NEGATIVE-direction at d = −0.43 [−0.82, −0.04] vs VAL-073 normals as external comparator; only 6.7% of the 270 SCC tumors fall above the VAL-073 normal 95th percentile.

At total tissue-arm n=313 across three cohorts, two read negative-direction. **VAL-073 Verlaat is the outlier**, not the artifact. The cohort-direction-flip is real biological/cohort-design heterogeneity that single-cohort validation would have missed.

VAL-076 GSE143752 El-Zein (Quebec, n=186 LBC EPIC 850K) and VAL-077 GSE287994 Bowden (Imperial London, n=247 LBC EPIC 850K) cannot establish or refute LBC detection at v0.1 — VAL-076 reads flat across CIN grades on a panel transferability flag (Xu-538 was buffy-coat trained, LBC is a different cell mixture); VAL-077 supplementary file is residual M-values per Bowden 2025 Methods, not raw β, and is deferred to v0.2+ raw IDAT processing.

The card cannot make a clinical claim at single_cohort_validated tier with the VAL-073 anchor alone against VAL-074 + VAL-081 disagreeing, and it cannot make a claim at cross_platform_validated tier with the LBC pathway unresolved. `exploratory_with_cohort_heterogeneity` honestly reflects the current evidence state.

**Path to v0.2+ (in priority order):**

1. **Reprocess GSE287994 from raw IDATs** through minfi/sesame to extract proper raw β values; rerun VAL-077 against the framework's universal Stage 1 panel. The same Bowden 2025 paper achieved AUC 0.92 using PAX1/NREP-AS1 on this cohort, so the cervical signal is in the raw data — the question is whether the universal Stage 1 panel can detect it from real β.

2. **HPV-stratify all tissue cohorts.** Re-run VAL-073, VAL-074, VAL-081 with explicit HPV+ vs HPV− subgroup analysis to determine whether the cohort-direction-flip is HPV-stratification driven. If yes, cervical-epic v0.2 publishes at single_cohort_validated tier with HPV+ and HPV− as separate sub-cards.

3. **Build cervical-LBC-specific Stage 1 panel** trained on LBC β values rather than buffy-coat. This is the principled path to clinical deployment via standard-of-care LBC pap smear collection.

4. **Substitute published clinical-grade cervical methylation panels** (FAM19A4/miR124-2 [QIAsure], ZNF671/SOX17/DLX1 [GynTect], EPB41L3, PAX1/NREP-AS1 [Bowden 2025 AUC 0.92]) as a card-specific Stage 1 deviation with dedicated H_min calibration. Produces clinically-deployable results immediately, but breaks the universal-pipeline rule for this specific card.

5. **Run Test 2 (lymphoid vs myeloid sub-panel split)** on Xu-538 when OQ-2026-01 immune-atlas staging operationalizes. Determines whether the cohort-direction-flip has compartment-specific sub-pattern.

6. **Pursue gated cohorts** VAL-078 CINCS (Bukowski 2023, 5-yr LBC pre-diagnostic, n=148-289) and VAL-079 Sundström CIN2 (2026, n=58 active surveillance) via direct corresponding-author contact. These are the closest existing cervical analogs to long-window prospective methylation screening.

## 18. What we discovered

### 18.1 Cohort heterogeneity surfaces only at full-breadth validation

cervical-epic v0.1 is the first Cookbook card where the cohort-completeness rule (CCL-029) caught a finding that single-cohort validation would have missed. With VAL-073 alone, the card would have published at `single_cohort_validated` with d=+0.73 monotonic CIN3 detection — a clean tissue-arm anchor, exactly the kind of finding that motivates clinical deployment. Only at VAL-074 (n=43) did the cohort-direction-flip become visible; only at VAL-081 (n=270) did the heterogeneity become undeniable. **At full tissue-arm n=313 across three cohorts, the framework's universal Stage 1 panel cannot replicate the VAL-073 anchor pattern.**

This is exactly the protective function CCL-029 was written to provide: full-breadth validation surfaces real heterogeneity that partial-coverage publishing would smooth over. The cervical-epic v0.1 record is the proof-of-value for the rule.

### 18.2 HPV-stratification of normals matters

The most likely explanation for the cohort-direction-flip is HPV-stratification of the normal cohort. VAL-073 Verlaat used population-normal cervical tissue (women without CIN history attending colposcopy with normal histology) without HPV-stratification of the normals. VAL-074 Farkas explicitly selected HPV-negative healthy as normal. VAL-074 healthy mean A is 0.06 below VAL-073 healthy mean — 2.7 anchor-SDs apart. HPV-negative healthy cervical tissue sits at depressed immune-class baseline relative to mixed/unspecified-HPV population normal.

Biologically: HPV exposure (even subclinical or transient) shifts the cervical immune compartment in ways the Xu-538 panel detects. Once you remove HPV+ samples from the normal cohort, you remove the HPV-driven baseline immune activation, and the contrast against HPV-positive disease tissue inverts.

This is consistent with the established cervical immunology literature. HPV-driven inflammation, T-cell infiltration, MHC-I downregulation by E7, Treg expansion, MDSC accumulation — all are well-documented features of HPV+ cervical tissue. The framework IS detecting them; the question is what to compare against.

### 18.3 Panel transferability: LBC is not buffy-coat

The framework's universal Stage 1 panel (Xu-538) was selected from buffy-coat training data. It transfers cleanly to plasma cfDNA, blood-derived signals, and tissue with high immune infiltrate (because tissue immune compartments contain similar cell types). It does NOT automatically transfer to LBC pap smear samples, which are ~80% exfoliated cervical epithelium + ~10-20% mucosal-resident lymphocytes + variable mucus and inflammatory infiltrate — a fundamentally different cell mixture.

VAL-076 produced a flat A-score across CIN grades on raw GenomeStudio AVG_Beta data. This is most likely a panel-transferability finding, not a "no signal" finding. Clinical-grade LBC methylation panels (FAM19A4/miR124-2, ZNF671, EPB41L3, PAX1/NREP-AS1) were trained specifically on cervical LBC samples and detect strong methylation signal in the same data — including the Bowden 2025 paper's AUC 0.92 on the GSE287994 cohort that VAL-077 nulled.

The framework needs either a cervical-LBC-specific panel (v0.2+ engineering) or a card-specific deviation to use the published clinical-grade panels with dedicated H_min calibration.

### 18.4 Why the universal pipeline produced a null where clinical panels detect AUC 0.92

VAL-077 attempted Bowden 2025's GSE287994 cohort (n=247 LBC EPIC 850K). Walther downloaded the supplementary file `GSE287994_ewas_betas_2.txt.gz` (1.7 GB), parsed it as raw M-values, converted to β via β = 2^M / (1+2^M), and scored Xu-538. Result: mean A = 1.011 in benign vs 1.010 in disease, d = −0.029 NULL.

Diagnostic per CCL-032: β distribution check across the 538-CpG panel showed 50% of values clustered in [0.4, 0.6] and only 12% at extremes. Real raw β is bimodal with >30% at extremes. The "_2" suffix in the filename, the M-value distribution centered at 0, and the literature confirmation in Bowden 2025 Methods ("logistic regression... with adjustment for batch, chip, age, and human papillomavirus status") together indicate the file contains **batch+chip+age+HPV-corrected residual M-values from the EWAS regression model**, NOT raw array M-values.

Residual M-values centered at 0 map to β ≈ 0.5 across the panel under standard conversion. Shannon entropy at β = 0.5 is at maximum (H = 1.0 bit). H_min(immune) = 0.838889. So A = 1.0/0.838889 = 1.192 — at the immune ceiling. VAL-077's observed mean A of 1.011 is consistent with the panel reading residualized data near β=0.5, producing A near the ceiling regardless of true disease state. Both benign and disease samples produce ~the same A because both are residuals around zero.

The "null" between benign and disease is a measurement-pipeline artifact. The actual cervical immune biology in this cohort is detectable — Bowden's analysis found 409 CpGs significant at p < 5×10⁻⁸, and PAX1/NREP-AS1 achieved AUC 0.92 — but it isn't accessible through framework scoring of the residual file. v0.2+ requires raw IDAT processing through minfi/sesame from `GSE287994_RAW.tar`.

### 18.5 What the v0.1 record actually proves

cervical-epic v0.1 proves:
1. **The cohort-completeness rule (CCL-029) works as intended.** Three independent tissue cohorts at total n=313 surface real heterogeneity that single-cohort validation would have missed. The rule earned its keep.
2. **The diagnostic-order rule (CCL-032) is necessary.** Walther initially drafted VAL-076/077 outcomes as O3_NULL framework findings before checking data integrity or biology consistency. CCL-032 was formalized to prevent the next card from repeating that mistake.
3. **The framework's universal Stage 1 panel does not transfer to LBC out of the box.** Cervical-LBC-specific engineering is required for clinical deployment at the screening-relevant specimen.
4. **HPV-stratification of normal cohorts matters in cervical methylation analysis** at a magnitude comparable to or larger than the disease signal itself. This may extend to other HPV-driven cancers (HNSCC, anal SCC, penile SCC).
5. **The published clinical-grade panels (FAM19A4/miR124-2, ZNF671, PAX1/NREP-AS1, EPB41L3) detect signal where the framework's universal panel does not.** This is a transferability finding at the panel level, not a "the disease has no signal" finding at the framework level. Substituting these panels into the framework with dedicated H_min calibration is the v0.2+ path to clinical deployment.

cervical-epic is the honest record of what happens when a framework's universal pipeline meets a disease whose primary specimen, normal-cohort selection, and cell mixture all differ from the panel's training data. The path to v0.2+ is documented, the failure modes are catalogued, and the lessons (cerv-LL-008 through cerv-LL-016, plus CCL-032) are now structural protections for every future card build.

### 18.6 What this means for EDEAR commercial deployment

cervical-epic v0.1 is not a deployable EDEAR card at the screening-relevant LBC pap smear specimen — that is the honest finding. It does not mean cervical screening is out of EDEAR's reach; it means the path to clinical deployment is different from the universal-panel-everywhere story the framework started with.

**The EDEAR architecture absorbs this finding without disruption.** The "one IDAT, many tests" model is intact. The universal Stage 1/2/3 pipeline is intact. The card-specific deviation that cervical-epic v0.2+ requires — substituting a clinical-grade cervical methylation panel (FAM19A4/miR124-2 [QIAsure AUC 0.77], ZNF671 [GynTect], EPB41L3, PAX1/NREP-AS1 [Bowden 2025 AUC 0.92]) into the Stage 1 slot with dedicated H_min calibration — has framework precedent. AD has its 7-CpG directional Rule A panel. PDAC has its 324-CpG directional fallback. Cervical-epic v0.2+ becomes a third case where the universal Stage 1 panel is replaced or supplemented by a card-specific panel chosen for the disease's specimen and immunology.

**What changes in the EDEAR product story.** The original frame was "every disease test runs off the same Xu-538 universal Stage 1." The post-cervical-epic frame is "every disease test runs off the same universal pipeline; for diseases where the universal Stage 1 panel does not transfer to the screening-relevant specimen, the card substitutes the clinical-grade panel for that disease and re-calibrates H_min on training data." This is a more honest and more durable claim. It also makes EDEAR more, not less, attractive to clinical partners — because it means the platform supports panel substitution as a first-class operation, not a workaround.

**What stays the same in the EDEAR report.** The dual-axis honesty (A-score plus fraction-vs-age-baseline plus concordance indicator) still applies. For cervical_epithelial in a v0.1 patient report, the report would surface "insufficient signal from this assay generation" in the same neutral N/A pattern that brain/CNS gets in blood — the EDEAR spec was designed for exactly this. Honest "we can't tell yet, here's why, here's what changes when we can" is the competitive moat against competitors who would publish VAL-073 alone and call it a screening test.

**What the cohort-direction-flip suggests for related diseases.** If HPV-stratification of normals drives cervical's cohort-direction-flip, the same effect may appear in other HPV-driven cancers — HNSCC (oropharyngeal SCC), anal SCC, penile SCC, vulvar SCC. Each of those is a candidate EDEAR card with the same structural pattern: HPV-driven inflammation, MHC-I downregulation, Treg expansion, mucosal-resident lymphocyte mixture in the screening-relevant specimen. If two more HPV-driven cancers confirm the pattern, "HPV stratification is mandatory for HPV-driven disease cards" becomes a framework-level rule comparable to lung-epic's smoking-stratification rule and hcc-epic's viral-hepatitis-stratification rule. The cervical-epic record positions EDEAR to recognize that pattern early in the next two builds rather than discover it after the fact.

**What a reviewer would see.** cervical-epic v0.1 published at `exploratory_with_cohort_heterogeneity` tier with full disclosure of the VAL-073 anchor, the VAL-074 + VAL-081 negative-direction cohorts, the VAL-076 panel-transferability flag, and the VAL-077 data-integrity flag — and with a documented v0.2+ path to clinical deployment via published clinical-grade panel substitution — is the kind of card that builds reviewer trust, not the kind that erodes it. Single-cohort papers claiming AUC 0.95 on n=68 without cross-cohort replication are what reviewers reject. The EDEAR record being trustworthy in this state is what makes the AD card and the PDAC card and the breast card more credible by association.

---

**End of cervical-epic v0.1 README.**
