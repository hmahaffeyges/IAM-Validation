# Phase 0 Cohort Survey — Gastric + Esophageal Sprint

**Date:** 2026-05-02
**Scope:** Gastric AND esophageal tested side-by-side per Heath sign-off; card-vs-cards decision deferred until data speaks
**Sprint deliverable target:** Combined gastric+esophageal-epic v0.1 OR separate gastric-epic v0.1 + esophageal-epic v0.1 (decision after Phase C)
**Compaction-safe:** This document records the landscape locked before prereg drafting

---

## Atlas inventory (Phase A acquisition complete)

### Boccellato gastric mucosoid atlas — DOWNLOADED + BUILT

- **Source:** Fritsche K, Boccellato F, Schlaermann P, et al. *DNA methylation in human gastric epithelial cells defines regional identity without restricting lineage plasticity.* Clin Epigenetics 2022;14:193. DOI 10.1186/s13148-022-01406-4. PMID 36585699.
- **GEO accession:** GSE141660 (SuperSeries with GPL21145 EPIC 850K + GPL23572 HM450 + GPL16304 Agilent expression sub-platforms)
- **Atlas substrate file (raw):** `GSE141660_EPIC_matrix.txt.gz` (71,243,285 bytes, SHA-256 `d43bd068645c9f9d2e63fb704d1f7caa4b02c137b0e007721d3f973738b25b04`)
- **Atlas substrate file (raw):** `GSE141660_HM450_matrix.txt.gz` (5,134,996 bytes, SHA-256 `4d2a6d4664b03df4cead7d0d08fadb91fccdff4065529cf758fd40da4cc88900`)
- **Sample structure (EPIC platform):** 18 healthy mucosoid samples = 3 donors (F-55, M-47, F-69) × 3 regions (antrum / corpus / fundus) × 2 differentiation states (undifferentiated stem-cell-enriched +W/R, differentiated pit-cell-like −W/R)
- **Cell type:** Purified primary gastric epithelial cells from sleeve resections, cultivated as plane mucosoids (cell-type-pure, NOT bulk biopsy)
- **Preprocessing applied (per author Methods):** SWAN normalization + ChAMP filtering (detection p>0.01, bead count <3, SNP-overlapping CpGs per Zhou 2016, multi-mapping probes per Nordlund 2013, sex-chromosome probes excluded). 738,115 CpGs survive filtering.

### BoccellatoStomachRef v1 — built atlas reference matrix (Type 1 atlas adaptation, SEALED)

- **Output file:** `/home/claude/gastric_esophageal_sprint/atlas_acquisition/boccellato_stomachref_v1.csv`
- **Size:** 48,715,676 bytes
- **SHA-256:** `fbe1dbfdeceb87a1f28c5737f0c3d8b6f86614dee5b9dfeb525741d3e4ef4d11`
- **Structure:** 738,115 CpGs × 6 tiles (Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff, Fundus_undiff, Fundus_diff)
- **Tile β = mean across 3 donor replicates per (region, state)** — this is the canonical reference-construction approach and matches how Loyfer 25-tile and EpiSCORE references are constructed
- **CHK-3.1C dedupe gate:** PASS (zero duplicate CpG IDs)
- **CHK-3.1A full-genome substrate gate:** PASS (37.0% extreme, 9.3% middle on raw 18 samples)

### Atlas-family-fitness diagnostic (per-CpG tile-range distribution)

| Statistic | Value |
|-----------|-------|
| Median tile-range across 6 tiles | 0.0385 |
| 90th percentile | 0.0997 |
| 95th percentile | 0.1265 |
| 99th percentile | 0.1907 |
| Maximum | 0.9005 |
| Fraction with range >0.1 | 9.92% |
| Fraction with range >0.2 | 0.81% (5,977 CpGs) |
| Fraction with range >0.4 | 0.06% (459 CpGs) |

The atlas's discriminating power lives in the long tail. Most CpGs (90%) do not distinguish between gastric regions/states (range <0.1). The 5,977 CpGs with range >0.2 are the candidate inter-regional discriminating CpGs — consistent with Boccellato 2022's reported 3,703 FDR<5%-significant inter-regional DMs (we report the larger superset before FDR; ratio 1.6× expected).

**Implication for prereg:** Within-cohort tile range will be moderate, not large. Phase B calibration will measure how well the 6 tiles separate on the VAL-106 healthy cohort. Outcome tiers must allow for the possibility that some tile pairs (e.g. Antrum_undiff vs Antrum_diff) read very similarly while others (e.g. Antrum_diff vs Fundus_undiff) read distinctly.

### Atlases ALREADY in vault for run-everything (Heath sign-off 2026-04-26 mandate)

- **Xu-538 panel** (Stage 1): SHA `ada6729605...`, 538 CpGs, immune-class architectural drift detector
- **Layered Moss+Loyfer 25-tile** (Stage 2): includes `Upper_GI` bulk tile (Loyfer-array, applies to gastric mucosa per CHK-2.6), `esophagus`, `small intestine`, `stomach` Moss-only entries, `Hepatocytes`, all 25 tissue tiles
- **Caggiano CelFiE TIM** (Stage 2 microenvironment): 254 CpGs × 19 cell types, calibrated VAL-113
- **Salas IDOL** (Stage 3): 6-cell immune fine-tune
- **UniLIFE Guo 2025** (Stage 3): 1,906 CpGs × 19 immune cell types, calibrated VAL-082+
- **EpiSCORE LiverRef** (relevant for hcc-epic v0.3 cross-check): on disk, NOT bridged to array yet
- **EpiSCORE EsophagusRef** (per `walther_mayer_lessons` reference list): in distribution, not yet bridged

### Atlases identified during landscape but DEFERRED to v0.2+

- **Hao/Berman 2023 esophageal WGBS** (45 ESCC + EAC samples, Genome Biology DOI 10.1186/s13059-023-03035-3) — would require WGBS→array bridging like Caggiano. **Defer to v0.2** unless trivial; not required for v0.1 since TCGA-ESCA HM450 is array-native and Loyfer skin_keratinocyte + head_and_neck_larynx tiles already cover the ESCC squamous lineage.
- **EpiSCORE EsophagusRef** — bridge engineering ~4 hours (template per ProstateRef bridge VAL); **defer to v0.2** unless squamous lineage scoring requires it.
- **EsccAtlas / TCGA-ESCA published methylation panels** — informational reference for clinical-grade panel CHK-1.4 comparison, not a deconvolution reference.

---

## Disease cohort inventory

### Tier 1 — Publicly accessible, GEO/TCGA

#### Gastric

| Cohort | Platform | Samples | Disease coverage | Use |
|--------|----------|---------|------------------|-----|
| TCGA-STAD | HM450 sesame Level 3 | n≈395 tumor + n≈2 paired adjacent-normal + ~25 unpaired normal (HM27 + HM450 mix) | Gastric adenocarcinoma; Lauren classification (intestinal/diffuse), TCGA molecular subtypes (CIN/MSI/EBV+/GS), H. pylori serology partial | **Phase C primary disease cohort** — Welch tumor-vs-pooled-normals (paired n=2 too small per CCL-029) |
| GSE30601 | HM27 (GPL8490) | n=297 (203 GC + 94 normal, Zouridis 2012) | Singapore gastric cancer cohort | **Phase C secondary** — platform caveat per CHK-1.2 (HM27 has ~30% Xu-538 coverage); cross-cohort triangulation only |
| GSE39600 | HM27 + HM450 mix | n≈30 H. pylori-infected gastric biopsies | H. pylori-mediated epigenetic dysregulation (Tahara 2014 etc.) | **Phase C exploratory** — H. pylori-driven adjacent-normal field defect demonstration |
| GSE141660 GPL16304 (organoids) | Agilent **gene expression**, NOT methylation | 6 organoid samples (Normal/Atrophy/IM × 2 donors F-73, M-68) | Correa cascade gene expression — NOT methylation | NOT scored; documented because authors compared to methylation in published paper figures |

#### Esophageal

| Cohort | Platform | Samples | Disease coverage | Use |
|--------|----------|---------|------------------|-----|
| TCGA-ESCA | HM450 sesame Level 3 | n≈185 (mix ESCC + EAC) | ESCC + EAC subtype split critical (different cells of origin) | **Phase C primary disease cohort** — Welch tumor-vs-pooled-normals; subtype-stratified |

### Tier 2 — EGA-controlled access

| Cohort | Platform | Samples | Disease coverage | Use |
|--------|----------|---------|------------------|-----|
| **MCCS gastric pre-dx** (Hodge 2021, Cancer Prev Res) | HM450 (Illumina HumanMethylation450K Beadchip) | n=168 cases + n=163 controls, blood collected median 12 years pre-diagnosis, **adjusted for H. pylori status** | Gastric cancer pre-diagnostic blood; matched on sex/year of birth/country/blood sample type; conditional logistic regression | **Italian-style pre-dx analog** Heath asked for. Tier 2 EGA application required (turnaround weeks-to-months). DEFER for biobank application; document in known_limitations as the priority tier-2 cohort. EGA archive: phs003213. |
| **EnviroGenomarkers gastric subset** | HM450 | n=variable (subset of ~600 multi-cancer pre-dx) | EPIC-Italy + NSHDS Italian/Swedish nested case-control; gastric not the primary focus but possibly a subset | Tier 2 EGA application; lower priority than MCCS for v0.1 |

### Tier 3 — Biobank / commercial

| Cohort | Platform | Samples | Disease coverage | Use |
|--------|----------|---------|------------------|-----|
| **Taizhou Longitudinal Study (TZL) PanSeer** | Targeted bisulfite NGS (Singlera proprietary 12,000 methylation sites × 600 regions) | n=605 samples (191 pre-diagnostic + 414 controls/post-diagnostic), up to 4 years pre-dx for stomach + esophageal + CRC + lung + liver cancers | Asian (Chinese) population, multi-cancer pre-diagnostic. **191 cancers detected up to 4 years before diagnosis at 91% sensitivity**. Singlera Genomics commercial. | Tier 3 commercial restricted access. NOT a 450K/EPIC platform — different substrate, would require separate substrate calibration even if accessible. **Document as the existence proof that pre-diagnostic gastric methylation detection works**, not a v0.1 cohort. |
| **Shanghai Women's Health Study** (Yang 2012) | Pyrosequencing only | n=192 cases + n=384 controls | Gastric pre-dx | NOT 450K/EPIC; pyrosequencing only (Alu/LINE-1 + targeted RNF180/RUNX3); different platform-class entirely. Document only. |

### Crohn's-disease-blood-methylation cohorts (per Q3 sign-off mandate)

| Cohort | Platform | Samples | Disease coverage | Use |
|--------|----------|---------|------------------|-----|
| GSE87650 (Adams 2014) | HM450 | n=240 IBD blood (Crohn's + UC + controls) | UK pediatric IBD methylation, baseline blood samples | **Phase C exploratory** for Crohn's-aware blood baseline + immune-class drift; tests inflammaging signature |
| GSE99788 (Ventham 2016) | HM450 | n=549 (240 CD + 190 UC + 119 controls) | Treatment-naïve adult IBD blood methylation | **Phase C exploratory** — larger; differentiates CD vs UC vs HC |
| GSE32148 (Harris 2012) | HM450 | n=43 (24 CD + 12 UC + 5 controls) | Pediatric IBD whole-blood methylation | Cross-validation; small n |
| **No public cohort with Crohn's diagnosis + long follow-up + HCC outcome on 450K/EPIC blood** | | | This pathway requires biobank acquisition through EDEAR's clinical pilot (when IBD subspecialty patients enrolled) | **Documented as the gap** that motivates the Crohn's-aware blood baseline acquisition plan. Per CCL-025, the mechanism is established (PSC-IBD HR=21 for HCC, HR=28 for CCA per Trivedi 2020 Gastroenterology); the cohort to validate is what's missing. |

---

## CCL-025 chronic-driver field-defect application matrix

Per Heath sign-off Q3, document the Crohn's pathway language on BOTH gastric-epic v0.1 AND amend hcc-epic v0.3 → v0.3.1.

### Gastric chronic drivers (mandate H. pylori stratification per CCL-025 line 392)

| Driver | Mechanism | Cohort coverage | Stratification status |
|--------|-----------|-----------------|------------------------|
| **H. pylori chronic infection** | Drives extensive methylation drift in adjacent-normal gastric mucosa via DNMT-upregulation and CpG island methylator phenotype (CIMP) (Maekita 2006, Niwa 2010) | TCGA-STAD has H. pylori serology (partial), MCCS pre-dx adjusted for H. pylori | **MANDATORY** per CCL-025 — gastric-epic prereg must specify H. pylori-status stratification |
| **EBV chronic infection** | EBV+ STAD has high-CIMP methylation phenotype (Cancer Genome Atlas Network 2014, Nature) | TCGA-STAD molecular subtype classification (CIN/MSI/**EBV+**/GS) | **MANDATORY** stratification — EBV+ subtype distinct methylation epigenotype |
| **Chronic atrophic gastritis** | Pre-cancerous lesion in Correa cascade; methylation drift accumulates from gastritis → atrophy → IM → dysplasia → adenocarcinoma | Boccellato GSE141659 IM organoid samples (small n); GSE39600 H. pylori biopsies | **DOCUMENT** as pathway in v0.1 known_limitations |

### Esophageal chronic drivers

| Driver | Mechanism | Cohort coverage | Stratification status |
|--------|-----------|-----------------|------------------------|
| **Chronic GERD / Barrett's metaplasia** | Columnar epithelium replaces squamous epithelium → EAC pathway. Boccellato authors compared their IM signature against published Barrett's methylation. | TCGA-ESCA has subtype split EAC vs ESCC; Barrett's not directly in TCGA-ESCA | **MANDATORY** subtype stratification (ESCC vs EAC); Barrett's is the pre-EAC pathway |
| **Tobacco + alcohol (ESCC primarily)** | Chronic inflammation + carcinogen exposure on squamous epithelium | TCGA-ESCA has tobacco + alcohol metadata partial | **MANDATORY** stratification per CCL-009 (smoking) for ESCC arm |

### HCC chronic drivers (already in card; ADD Crohn's pathway language per Q3 sign-off)

Per existing CCL-025 line 384 + VAL-064:
- HBV+ (n=7 in TCGA-LIHC): paired d = +0.0482 NULL (chronic infection field defect blunts paired contrast)
- HCV+ (n=5): paired d = −0.0464 NULL (same mechanism)
- alcohol_only (n=10): paired d = +0.8667 (clean secretory-class)
- NAFLD_only (n=3): paired d = +0.932 (clean, underpowered)
- **no_documented_risk (n=19, "Marcus-analog"): paired d = +0.6166 [+0.1261, +1.1071], p=0.0072 — clean secretory-class signal**

**Crohn's-disease pathway addition (proposed for hcc-epic v0.3.1 amendment):**

> Long-duration Crohn's disease (CD) is associated with hepatobiliary malignancies through two distinct mechanisms:
>
> 1. **Primary sclerosing cholangitis (PSC) co-morbidity** (1-3% of CD patients per literature; ~2.5% of UC patients; PSC-IBD HR for HCC = 21.00, HR for CCA = 28.46, both p<0.001 per Trivedi 2020 *Gastroenterology* UK population-level cohort). PSC-driven chronic biliary inflammation produces hepatic methylation drift mechanistically identical to chronic HBV/HCV at adjacent-normal liver tissue.
>
> 2. **CD-driven HCC without PSC** (rare; ~10 published case reports per Aleksandrova-Yankulovska 2014 + subsequent literature). 8/10 published cases received azathioprine therapy; mechanism implicated includes (a) immunosuppression-mediated reduced tumor surveillance, (b) chronic systemic inflammation methylation footprint, (c) bile acid dysregulation, (d) familial IBD susceptibility loci (NOD2, IL23R, ATG16L1) carrying their own immune-dysregulation phenotypes.
>
> **Operational consequence for hcc-epic interpretation:** A CD patient with HCC presenting clinically should be expected to behave like the viral-hepatitis subgroup at the paired tissue level (NULL paired d due to adjacent-normal field defect from chronic systemic inflammation) but POSITIVE on the ccfDNA plasma arm (per VAL-059 GSE298812 d=+0.634 in chronic-infection HIV+HBV+ patients). The hcc-epic v0.3.1 known_limitations adds: "Long-duration Crohn's disease is a chronic-driver risk factor analogous to HBV/HCV; tissue arm paired-d expected blunted, ccfDNA arm expected detectable. The cookbook does not yet have a CD-with-HCC cohort to confirm this prediction; flag for future biobank acquisition."

---

## Key access decisions locked

1. **TCGA-STAD Phase C disease scoring uses Welch tumor-vs-pooled-normals** (n=395 vs n≈27 normals). Paired-d on n=2 paired HM450 normals is documented as structurally underpowered. CHK-1.6 cohort access is Tier 1.

2. **TCGA-ESCA Phase C disease scoring is subtype-stratified ESCC vs EAC** with separate Welch d per subtype, given different cells-of-origin. Pooled-cohort paired-d is biologically uninterpretable.

3. **MCCS gastric pre-dx (n=168/163 HM450) is the priority Tier 2 acquisition** — analogous to how heme-LL-006 EnviroGenomarkers was queued. Document as "v1.0+ promotion path: EGA application required."

4. **PanSeer Taizhou is documented as existence proof of pre-diagnostic gastric methylation detection feasibility** (191 cancers detected up to 4 years pre-dx at 91% sensitivity), but is Tier 3 commercial-restricted on Singlera proprietary platform. NOT a v0.1 cohort.

5. **MCCS expanded scope:** Hodge group has nested case-control studies on EIGHT cancer types (breast, prostate, colorectal, lung, kidney, urothelial, **gastric**, mature B-cell). The gastric arm is the priority for this card; the bladder and kidney arms are next-card priorities.

6. **Crohn's pathway:** documented in BOTH gastric-epic and (via hcc-epic v0.3.1 amendment) liver/HCC cards. Standalone Crohn's-card or autoimmune-card decision deferred per Heath: pursue if more data emerges during this sprint.

---

## Sprint plan summary

### Phase A — Atlas acquisition + bridge engineering (DONE)
- ✅ GSE141660 EPIC + HM450 matrices downloaded, SHA-256 sealed
- ✅ BoccellatoStomachRef v1 atlas built (738,115 CpGs × 6 tiles), CHK-3.1C dedupe gate PASS
- ⏳ Push to GitHub atlas_vault/stage2_cell_of_origin/boccellato_stomachref_v1/ after VAL preregs sealed

### Phase B — Calibration VAL chain (PREREGS PENDING)
- **VAL-12X (Boccellato calibration on VAL-106 cohort):** Run BoccellatoStomachRef v1 against TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 HM450 sesame Level 3 (the standing healthy substrate). Pre-lock CHK-3.1B coverage threshold ≥80% per CHK-2.8 substrate floor. Per-tile A-score distributions sealed as healthy-floor thresholds (Type 2 calibration per the doc).
- **VAL-12X+1 (atlas-family-fitness):** Cross-tile separation verification — does each tile separate from the others in the VAL-106 cohort under non-stomach tissue? If all 6 tiles collapse on non-stomach tissue, the atlas is a clean stomach-class detector.

### Phase C — Run-everything Phase C (PREREGS PENDING)
Per Heath's reminder: **Stage 1 + Stage 2 + Stage 3 ALL run on every IDAT**, no gating. Magnitude-based |d| with direction labels per CHK-2.7. CHK-3.2 cross-cohort baseline check mandatory.

- **VAL-12X+2 (Stage 1 Xu-538 red-flag, GASTRIC arm):** TCGA-STAD HM450 Welch tumor-vs-pooled-normals + EBV+/MSI/CIN/GS-stratified + Lauren-stratified. CHK-2.17 cohort-substrate-coverage pre-flight check (mean coverage ≥90%, q5 ≥80%) BEFORE prereg seal. Run all atlases (Boccellato + Loyfer 25-tile + Caggiano TIM + Salas IDOL + UniLIFE + Moss).

- **VAL-12X+3 (Stage 1 Xu-538 red-flag, ESOPHAGEAL arm):** TCGA-ESCA HM450 Welch tumor-vs-pooled-normals + ESCC-vs-EAC subtype stratified + smoking-stratified ESCC subset. Same pre-flight + run-everything atlas stack.

- **VAL-12X+4 (Crohn's-blood baseline exploratory):** GSE87650 + GSE99788 + GSE32148 IBD blood methylation cohorts, run-everything atlas stack. Tests: (a) CD vs UC vs HC immune-class drift via Xu-538; (b) inflammaging trajectory; (c) any tissue-tile leakage suggestive of subclinical organ involvement; (d) cross-cohort baseline alignment via CHK-3.2.

### Phase D — Card construction (DELIVERABLE PENDING)
After Phase C runs complete, decide gastric-vs-esophageal-vs-combined card structure based on observed signal patterns:
- If both organs read with similar magnitudes and similar atlas patterns → single combined gastric+esophageal-epic v0.1 card
- If organs diverge meaningfully → separate gastric-epic v0.1 + esophageal-epic v0.1 cards
- Either way: CCL-025 chronic-driver stratification documented, Crohn's pathway in known_limitations + open-question pathway for v1.0+ biobank-acquired Crohn's-with-HCC validation

---

## Marcus's wife — what this report can honestly say

The card v0.1 will be an exploratory framework finding, not a clinical screening tool ready for deployment. What it can honestly tell Marcus's wife and her children:

1. **Methylation-based blood detection of stomach + esophageal cancer at 1-4 years pre-diagnosis IS feasible** — PanSeer Taizhou demonstrated 91% sensitivity for stomach + esophageal + CRC + lung + liver up to 4 years pre-dx. The science is real.

2. **Long-duration Crohn's disease IS associated with hepatobiliary cancer risk** at very large hazard ratios (HR=21 for HCC, HR=28 for CCA in PSC-IBD) per population-level epidemiology. Marcus's case fits this profile a priori, even if formal PSC was never diagnosed.

3. **The mechanism the EDEAR framework reads (immune-class methylation entropy drift via Xu-538 panel + tissue-of-origin tile shifts via Boccellato + Loyfer atlases)** is the same biological signal both PanSeer and the published gastric-cancer pre-dx literature exploit. EDEAR's contribution is making the framework available across many cancer types simultaneously rather than as separate per-disease tests.

4. **The clinically actionable recommendation for Marcus's children** if they inherit Crohn's susceptibility: standard CD/IBD surveillance (gastroenterology) + AASLD HCC surveillance protocols (AFP + abdominal imaging q6mo) IF CD develops + heightened vigilance for PSC overlap (annual MRCP + CA 19-9 monitoring). EDEAR's role would be supplementary blood-based methylation surveillance once the v1.0 commercial deployment is ready, not replacement of standard care.

5. **What we cannot say:** That EDEAR specifically would have detected Marcus's tumor in time. Card v0.1 has no cohort with Crohn's-disease-with-HCC outcome at 450K/EPIC platform. That cohort is the priority biobank acquisition for the next phase.

---

## Awaiting Heath sign-off before sealing prereg #1

The Phase 0 plan above is locked-in pending one final confirmation: **proceed with prereg drafting using this scope?** If yes, prereg #1 (VAL-12X Boccellato calibration on VAL-106 cohort) will be the first sealed file, then the three Phase C preregs follow.
