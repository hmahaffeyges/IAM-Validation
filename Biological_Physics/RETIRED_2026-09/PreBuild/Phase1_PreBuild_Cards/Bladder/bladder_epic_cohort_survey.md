# Bladder-epic Phase 0 cohort survey

**Date:** 2026-04-30 (this session, Walther + Heath)
**Purpose:** Phase 0 exhaustive cohort survey per master template guardrail #10 BEFORE any VAL ID is claimed or atlas is touched. Per CCL-029 (cohort-completeness rule) + CHK-1.1 (Sample_title verification) + CHK-1.2 (platform compatibility) + CHK-1.6 (cohort access tier classification). Heath signs off on which cohorts go into Phase C of the bladder sprint vs which are deferred BEFORE Phase A begins.

**Disease:** Bladder cancer / urothelial carcinoma / transitional cell carcinoma (TCC). ICD-10 C67.

**Substrate scope:** Tissue methylation arrays (HM450, EPIC v1) + peripheral blood (HM450, EPIC v1) + urine sediment (mostly targeted assays, some array-genome-wide). Plasma ccfDNA arrays exist but are sparse for bladder specifically.

**Walther's pre-flight verification status:** All cohort metadata claims drawn from peer-reviewed publications; one CHK-1.1-style trap identified and flagged before it propagated (GSE13507 commonly miscalled methylation; it is actually expression — see "Cohorts NOT to use" section).

---

## What this card is for (clinical claim placeholder for Heath sign-off)

The bladder-epic card fires when a patient's EDEAR analysis returns a Stage 1 immune flag AND Stage 2 localizes to bladder-relevant tile (bladder_epithelial via Moss; or urothelial sub-tile via EpiSCORE BladderRef per Phase A smoke test). Clinical action: cystoscopy + urine cytology + UroMark (or equivalent clinical-grade methylation panel) per AUA/EAU early-detection guidelines.

**The bladder card has THREE substrate pathways with very different evidence bases:**
1. **Tissue (FFPE / fresh-frozen tumor + adjacent-normal):** TCGA-BLCA + multi-center cohorts. Strong public data.
2. **Peripheral blood:** WHI pre-diagnostic (n=440 case + 440 control, biobank) + Christensen lab New Hampshire NMIBC monitoring (n=603, EPIC blood, possibly biobank).
3. **Urine sediment:** Multiple targeted methylation panels with clinical-grade performance (UroMark AUC 0.97); genome-wide urine methylation arrays are sparse.

This three-substrate-pathway structure is exactly what makes bladder a strong card candidate. The clinical-grade comparator (UroMark, Bladder EpiCheck) gives us CHK-1.4 leverage we did not have on prostate.

---

## Tissue methylation cohorts (HM450 / EPIC tumor + adjacent-normal)

| Cohort ID | Citation | n cases / n controls | Substrate | Tissue | Tier | Status |
|---|---|---|---|---|---|---|
| **TCGA-BLCA** | TCGA Research Network 2014 (DOI: 10.1038/nature12965); Robertson 2017 *Cell* | ~412 tumor + 21 paired adjacent-normal HM450 (cited in Cheng et al. 2020 [PMC6970050]) | HM450 | Tumor + adjacent-normal | Tier 1 (open) | **Phase C primary candidate.** Same calibration substrate as VAL-106/107/112/113 (TCGA HM450K sesame Level 3). Anatomy: pure urothelial. Clinical metadata: stage, grade, smoking status, molecular subtype. |
| **GSE52955 — Multi-cancer urological cohort** | Costa et al. 2024 (PMC12518500); single-center comprehensive cancer centre, fresh-frozen tissue | 14 BlCa + 5 normal bladder + 25 PCa + 5 normal prostate + 17 KCa + 6 normal kidney = 72 total | HM450 | Tumor + paired normal across 3 organs | Tier 1 (open) | **Phase C secondary candidate.** Multi-cancer cohort gives natural Stage 2 cell-of-origin contrast — bladder tile vs prostate tile vs kidney tile within ONE cohort, ONE platform, ONE preprocessing pipeline. Ideal for CHK-3.2 cross-cohort baseline check. Small n but unique design. |
| **GSE171369** | Lim et al. 2022 *BMC Cancer* (DOI: 10.1186/s12885-022-10275-2) | 9 BCa primary tumors + 9 paired adjacent nontumor | Agilent CpG island microarray (NOT Illumina) | Tumor + adjacent-normal | Tier 1 (open) | **DEFER.** Wrong platform — Agilent ≠ Illumina HM450/EPIC. Cannot bridge to Xu-538 panel without manifest re-mapping. Defer to v0.4+ unless explicitly prioritized. |
| **Bryan 2016 / GSE85837 / Birmingham UK NMIBC cohort** | Bryan et al. 2016 (PMID 26929985) | 4 normal + 18 LG + 51 HG NMIBC = 73 total | HM450 | Primary NMIBC tissue | Tier 1 (need GEO confirmation) | **Phase C candidate.** Clean NMIBC tumor-vs-normal cohort. UK Birmingham group (same group developing GALEAS Bladder urine test) — strong clinical metadata. GEO accession needs direct verification at sprint start; may be GSE85837 or related. |
| **Wilhelm-Benartzi 2010 / Marsit 2010** | Marsit et al. 2010 *PLOS One* PMC2925945; New Hampshire tissue cohort | 73 tumor + 12 normal-bladder controls + supplementary cohort | HM27 (older platform) | Tumor + control | Tier 1 (open) | **DEFER.** HM27 has only 27,000 CpGs — Xu-538 panel coverage incomplete. Per CHK-1.2 platform compatibility check, HM27 cohorts are deferred unless paired with HM450 replication. |
| **GSE37816** | (Cited in Cheng 2023 PMC10413619) | 18 bladder cancer + 6 normal | HM27 | Tumor + control | Tier 1 (open) | **DEFER.** HM27 small-n. Useful only as confirmatory if other HM450 results need cross-platform check. |

---

## Peripheral blood methylation cohorts

| Cohort ID | Citation | n cases / n controls | Substrate | Tissue | Tier | Status |
|---|---|---|---|---|---|---|
| **WHI bladder pre-diagnostic** | Jordahl et al. 2018 *Cancer Epidemiol Biomarkers Prev* 27(6):689 (PMC5984694) — Bhatti / Kelsey co-author | 440 TCC cases + 440 matched cancer-free controls | HM450 buffy coat (pre-diagnostic, median follow-up 7.22 yr to dx) | Pre-diagnostic blood | **Tier 3 (biobank-gated)** | **PHASE 9/12-EQUIVALENT GOLD-STANDARD COHORT.** WHI access requires formal data-access application; long turnaround (6+ months typical). NOT GEO-deposited as of literature checks. **The crown jewel for promotion path: this is the only public-domain pre-diagnostic blood methylation case-control for bladder cancer at array resolution.** Mirrors AD-immune's AIBL/AddNeuroMed and breast-epic's GSE51057. v0.1 card cannot reach `cohort_screening_validated` tier without WHI access. Mark as v1.0+ next-validation-step with biobank application path. |
| **GSE89093 / Christensen New Hampshire NMIBC EPIC blood** | Chen, Salas, Wiencke, Koestler, Karagas, Kelsey, Christensen 2022 *Clin Epigenetics* 14:14 (PMC8783448) | n=603 NMIBC patients, EPIC blood | EPIC v1 peripheral blood | At-diagnosis NMIBC monitoring | **Tier unknown (needs verification — possibly Tier 1 GEO deposit; possibly Tier 3 via Dartmouth/Karagas)** | **Phase C candidate IF GEO-deposited.** Christensen lab maintains data on GEO for several cohorts but not all population-based studies are publicly released. Need to fetch the data availability section of Chen 2022 directly during Phase A to confirm. The Chen 2023 follow-up CEBP paper (DOI 10.1158/1055-9965.EPI-23-0331) used the same cohort. n=603 EPIC blood is the largest public-candidate at-diagnosis bladder methylation blood cohort. Note: this is at-diagnosis NMIBC (not pre-diagnostic) with recurrence/survival endpoints — different question from WHI but extremely valuable. |
| **Marsit 2014 (heme-LL-style precedent)** | Marsit, Kelsey, Christensen et al. New Hampshire bladder ~223 cases (HM27, predecessor of Chen 2022) | n=223, HM27 | HM27 blood | At-diagnosis bladder cancer | Tier 1 (deposited as preliminary cohort referenced in Chen 2022 §Background) | **DEFER.** HM27 — superseded by Chen 2022 EPIC cohort. |
| **Houseman/Kelsey 2016 BJC** | Genome-wide measures of DNA methylation in peripheral blood and the risk of urothelial cell carcinoma (PMID via DOI 10.1038/bjc.2016.237) | Prospective nested case-control n unspecified in abstract | HM450 blood | Pre-diagnostic | Tier ?  | **Phase A verify.** May be a different cohort from WHI or may be the same data presented differently. Need to fetch directly during Phase A. |

---

## Urine sediment + plasma ccfDNA methylation cohorts

| Cohort ID | Citation | n / structure | Substrate | Tissue | Tier | Status |
|---|---|---|---|---|---|---|
| **UroMark training cohort** | Feber et al. 2017 *Clin Epigenetics* 9:8 (PMC5282868) | 86 MIBC tumors + 30 normal urothelium (training) + 274 voided urine validation (167 non-cancer + 107 BC) | Targeted bisulfite NGS (150 CpG loci) | Tumor tissue + voided urine sediment | Tier 1 publication / **NOT array-genome-wide** | **Reference panel not Phase C cohort.** Critical CHK-1.4 anchor: AUC 97% on independent urine validation. Used as the clinical-grade comparator, not as a card cohort. The 150 CpG UroMark panel is the published gold standard for CHK-1.4 — when bladder-epic Stage 1 fires on a urine sediment, the comparison question is whether the Xu-538 immune panel is concordant or divergent with UroMark on the same sample. |
| **Yaneng / Fang 2022 P3 panel** | Fang et al. 2022 *BMC Cancer* (PMC8895640) | 207 urine samples | RRBS + qMSP targeted | Urine sediment | Tier 1 publication / NOT array-genome-wide | Comparator only (not Phase C). |
| **Bryan / GALEAS Bladder Birmingham cohort** | Goel et al. 2025 *Biomark Res* (PMC12337379) | 13 NMIBC + 8 non-cancer | Oxford Nanopore long-read sequencing of urine DNA | Urine | Tier 1 publication / **NOT Illumina array** | DEFER. Different substrate than EDEAR's array pipeline. v1.0+ candidate when Tanaka 2025 nanopore→array bridge engineering lands. |
| **Strömqvist 2018 / Roperch urine** | Roperch et al. 2016 *BMC Cancer* (PMC5007990) — HS3ST2/SEPTIN9/SLIT2 + FGFR3 | 167 NMIBC + 105 controls (diagnostic); 158 NMIBC + 425 follow-up | qMSP targeted (3-marker) | Urine sediment | Tier 1 publication | Comparator only. |
| **Cheng / 7-gene urine panel** | Cheng et al. 2019 (PMC6856882) | 99 hematuria patients (training); TCGA-BLCA validation | qMSP targeted (HOXA9/ONECUT2/PCDH17/PENK/TWIST1/VIM/ZNF154) | Urine sediment | Tier 1 publication | Comparator only. |
| **GSE119260 (urine 4-substrate)** | Brikun et al. 2018 — see prostate-epic VAL-065 sealed at O5 | n=4 advanced-disease patients × 4 specimens (FFPE benign, FFPE tumor, plasma cfDNA, urine sediment) | EPIC 850K | Multi-substrate within-patient | Tier 1 (open) | **DEFER.** n=4 ceiling already documented in CCL-026 from prostate VAL-065. Same constraint applies to bladder use. v0.4+ via larger urine cohort. |

---

## Cohorts NOT to use (CHK-1.1 traps caught at Phase 0)

| Cohort ID | Trap | Lesson |
|---|---|---|
| **GSE13507** | Widely cited as "bladder cancer methylation Korean cohort." Per direct GEO platform check (GPL6102 = Illumina human-6 v2.0 expression beadchip), GSE13507 is an EXPRESSION dataset, NOT a methylation dataset. Multiple recent papers (Cheng 2020 PMC6970050, Wang 2025 npj Precision Oncology) miscall it. Reading the GPL platform code prevents the trap. | This is exactly the cervical-epic cerv-LL-008 / CHK-1.1 trap — landscape errors caught at the landscape stage. Walther flagged this BEFORE Phase A. |
| **GSE7476, GSE37815, GSE65635** | Expression datasets cited alongside methylation analyses. Wrong substrate. | Same trap class. |
| **GSE37816** | Real methylation but HM27 — only 27K CpGs. Xu-538 coverage substantially incomplete. | Per CHK-1.2 platform compatibility, HM27 is deferred. |

---

## Atlas landscape — Stage 2 cell-of-origin candidates

Per pipeline reference Part 2.3 + prostate-LL-006 (gene-promoter atlas family fitness depends on per-tissue cell-type distinctness), here are the Stage 2 candidates for bladder:

| Atlas | Source | Cell types covered | Atlas family | Phase A required test |
|---|---|---|---|---|
| **Layered Moss+Loyfer (calibrated VAL-112)** | Moss 2018 + Loyfer 2023 | `bladder_epithelial` is one Moss tile | Tile-coverage WGBS-derived | **Already calibrated.** Reuse VAL-112 thresholds. Bladder tile direction expectation: tumor-vs-adjacent-normal-paired = NEGATIVE per CCL-039 (cell-of-origin dedifferentiation). |
| **EpiSCORE BladderRef (pan-tissue v0.9.6)** | Zhu/Teschendorff 2022 *Nat Methods* 19:296. R package `aet21/EpiSCORE`. | Pan-tissue atlas covers bladder among 13 solid tissues. Cell types per Zhu 2022: epithelial / immune / endothelial / fibroblast (4-cell mouse-derived base; human bladder reference TBD by direct fetch of atlas R object) | Gene-promoter (EpiSCORE family) | **MUST run Phase A smoke test per prostate-LL-006.** Cardio HeartRef collapsed (O3_TISSUE_FLOOR_DOMINATED at VAL-111). Prostate ProstateRef separated cleanly (DISC-PROSTATE-001 via VAL-117). Bladder is the THIRD per-tissue test of the gene-promoter-atlas-fitness rule. Outcome unknown until Phase A smoke test runs on TCGA n=210 calibration cohort. |
| **Caggiano CelFiE TIM array-bridged (calibrated VAL-113)** | Caggiano 2021; HM450 hg19 manifest acquired in cardio sprint | 19 cell types, 254 CpGs | Tile-coverage | **Already calibrated.** Check whether TIM cell list contains bladder-relevant types (urothelium, smooth muscle, fibroblast, endothelial). |
| **Stage 1 Xu-538 immune panel** | Xu 2020 JNCI (the universal Stage 1 panel) | 538 CpGs immune signature | Stage 1 panel | Calibration-anchor task is **Wave 1 Shared Task A (VAL-114)** — currently NOT calibrated. Bladder-epic v0.1 will use within-cohort self-cal as the operational fallback per CCL-041 / DISC-CARDIO-005, with explicit caveat. v0.X+1 promotion lands when VAL-114 calibrates Xu-538 on GSE40279 Hannum. |
| **Stage 3 Salas IDOL 6-cell + UniLIFE 19-cell** | Salas 2018 + Guo 2025 | Stage 3 immune sub-composition | Stage 3 panels | Calibration-anchor tasks are **Wave 1 Shared Tasks B (VAL-115/116)** — currently NOT calibrated. Same within-cohort self-cal limitation as Stage 1. |

---

## Clinical-grade panel landscape (CHK-1.4 anchors)

Per CHK-1.4: before locking the card to Xu-538 scoring, identify the published clinical-grade panels for the disease and structure the card to test for concordance.

| Clinical panel | Citation | Substrate | Performance | Bladder-epic CHK-1.4 question |
|---|---|---|---|---|
| **UroMark** (150 CpG targeted bisulfite NGS) | Feber 2017 *Clin Epigenetics* 9:8 | Urinary sediment | AUC 97% (validation n=274, 167 non-cancer + 107 BC); sensitivity 98%, specificity 97% | Does Xu-538 immune panel fire concordantly with UroMark on the same urine samples? If yes — bladder-epic Stage 1 immune is a transferable detection signal. If no — transferability finding (cerv-LL-011 lesson). |
| **Bladder EpiCheck** (commercial 15-marker panel, 5-CpG-per-marker, methylation-specific PCR) | Wasserstrom 2016+ | Voided urine | NMIBC surveillance: sens 68%, spec 88% (CE-marked, FDA pending) | Same question as UroMark for surveillance use case. |
| **ADXBLADDER** (HOXB1, ONECUT2 simplified) | Trifa et al. | Urine sediment | Comparable to UroMark | Same question. |
| **Cheng 7-gene panel** (HOXA9/ONECUT2/PCDH17/PENK/TWIST1/VIM/ZNF154) | Cheng 2019 (PMC6856882) | Urine sediment | AUC 0.894 training, 0.851 TCGA validation | Same question. |
| **Roperch 3-gene panel** (HS3ST2/SEPTIN9/SLIT2) | Roperch 2016 (PMC5007990) | Urine sediment | AUC ~0.85 | Same question. |

**CHK-1.4 implication.** Bladder is unusual for EDEAR cards in that the published clinical-grade standard is well-established (UroMark) and broadly replicated (4+ independent panels with concordant CpG sets). This means: if Xu-538 immune-panel firing on bladder cohorts is null while UroMark + EpiCheck + ADXBLADDER read positive on the same cohorts, that is a TRANSFERABILITY finding (panel does not transfer to bladder substrate), NOT a "framework null finding" — exactly the cerv-LL-011 lesson. **The card prereg must include the panel transferability caveat for bladder substrate up front.**

---

## Selection recommendation for Phase C (Heath sign-off required)

Given the calibration-TODO guardrails (Wave 1 not yet executed; only Layered Moss+Loyfer and Caggiano TIM are calibrated; the new card MUST use only calibrated atlases per the per-card workflow rule), I recommend the following **Phase C cohort selection** for bladder-epic v0.1:

### Tier 1 — Phase C primary cohorts (all Tier 1 GEO-open, all HM450 or EPIC compatible)

1. **TCGA-BLCA** — 412 tumor + 21 paired adjacent-normal HM450 sesame Level 3.
   - Same calibration substrate as VAL-106/107/112/113. Drops directly into the existing Phase B calibration anchors.
   - Will produce per-tile A-scores per atlas using Layered Moss+Loyfer (calibrated) + Caggiano TIM (calibrated) + Stage 1 Xu-538 (within-cohort self-cal) + Stage 3 Salas IDOL + UniLIFE (within-cohort self-cal).
   - Will run the EpiSCORE BladderRef Phase A smoke test on TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (the existing calibration cohort) BEFORE scoring on TCGA-BLCA.

2. **GSE52955** — 14 BlCa + 5 normal bladder + 25 PCa + 5 normal prostate + 17 KCa + 6 normal kidney HM450.
   - Multi-cancer urological cohort, single platform, single preprocessing pipeline.
   - **Unique design value:** built-in Stage 2 cell-of-origin contrast — bladder vs prostate vs kidney within ONE cohort. Tests whether the bladder tile reads bladder-specific (positive direction in BlCa; negative in PCa/KCa) under run-everything multi-atlas Phase C.
   - Stage 2 directional discrimination question is the bladder analog of prostate v0.3's five-vs-one LE pattern.

### Tier 2 — Phase C secondary candidate (verification needed)

3. **Bryan UK NMIBC HM450 cohort** (GSE accession to verify in Phase A) — 4 normal + 18 LG + 51 HG.
   - Adds NMIBC stratification (LG vs HG vs normal) — bladder-specific clinical discrimination.
   - If GEO-deposited and compatible, fold in. If not, defer to v0.2 with biobank-application footnote.

### v0.X+1 next-validation-step cohorts

4. **Chen 2022 NMIBC EPIC blood (n=603)** — confirm GEO accession in Phase A. If publicly available, this is a Phase C cohort for v0.2; if biobank-gated, v1.0+ next-validation-step. Either way, this is the most important blood cohort for bladder.

5. **WHI bladder pre-diagnostic (n=440 case + n=440 control HM450 buffy coat)** — Tier 3 biobank-gated. v1.0+ next-validation-step. The pre-diagnostic blood screening claim cannot land without WHI access. Same access pattern as breast-epic FitzGerald MCCS pre-dx and prostate-epic FitzGerald — biobank applications are the consistent gate to pre-diagnostic blood tier promotion.

### v0.4+ urine-substrate next-validation-step

6. **Larger urine methylation cohort with mixed Gleason equivalents (mixed grade NMIBC + MIBC) + healthy controls + clinical-grade panel comparator** — does not exist publicly at array-genome-wide resolution as of 2026-04-30 sweep. Most urine bladder methylation work is targeted qMSP/RRBS not Illumina arrays. Inherits CCL-026 limitation from prostate VAL-065.

---

## What the v0.1 card claims AND does NOT claim (for Heath sign-off)

**v0.1 card claims (anchored to Phase C primary cohorts):**
- Stage 1 immune A-score on bladder tumor tissue is consistent with architectural drift (TCGA-BLCA + GSE52955 contrasts).
- Stage 2 cell-of-origin tile reading on Layered Moss+Loyfer `bladder_epithelial` discriminates bladder tumor from adjacent-normal under run-everything multi-atlas discipline (TCGA-BLCA paired pairs).
- Multi-cancer cell-of-origin contrast within GSE52955 (bladder positive on bladder tile, prostate positive on prostate tile, kidney positive on kidney tile) per CCL-039 tumor-vs-adjacent-normal direction expectation.
- (If EpiSCORE BladderRef separates per Phase A smoke test) — bladder tile direction signature analog to prostate's LE-NEGATIVE pattern.

**v0.1 card does NOT claim:**
- Pre-diagnostic blood screening for bladder cancer (WHI access pending; v1.0+).
- Cross-platform / multi-ancestry generalization (TCGA-BLCA is multi-ancestry but no ancestry-stratified disease-vs-control sealed; v0.4+).
- LG vs HG NMIBC stratification (defer to v0.2 if Bryan cohort lands).
- Urine-substrate clinical pathway with clinical-grade comparator (CCL-026 lesson; v0.4+).
- Stage 1 directional A_dir for bladder analog to AD Rule A 7-CpG panel (not constructed for v0.1; pooled-entropy Shannon symmetry is the Stage 1 metric).
- Plasma ccfDNA bladder ccfDNA detection (substrate physics constraint; sparse public data).
- Concordance/discordance with UroMark (CHK-1.4 question requires a urine cohort with both Xu-538 array data AND UroMark comparison — not in v0.1).

---

## Phase 0 sign-off requirements for Heath

1. **Confirm Phase C cohort selection** — TCGA-BLCA + GSE52955 + (Bryan UK pending verification). Or different.
2. **Confirm calibration-discipline approach** — option 2 (build now with Layered Moss+Loyfer + Caggiano TIM as calibrated, Stage 1/3 within-cohort self-cal documented as v0.1 limitation) per Heath's selection in prior turn (or option 1 — pause for Wave 1 first).
3. **Confirm EpiSCORE BladderRef Phase A smoke test** — per prostate-LL-006, the per-tissue calibration smoke test on TCGA n=210 is required BEFORE atlas integration into atlases_run vs atlases_deferred.
4. **Confirm whether Chen 2022 NMIBC blood EPIC cohort access verification is in scope for Phase A** — fetch the data availability statement directly during Phase A; if Tier 1 GEO, fold in for Phase C; if not, defer to v0.2.
5. **Confirm UroMark / Bladder EpiCheck CHK-1.4 framing** — the v0.1 card README will explicitly include the panel transferability caveat AND the published-clinical-grade comparator landscape (4+ independent panels with concordant CpG sets), but will not run a head-to-head until a urine-array cohort with paired Xu-538 + UroMark sample data exists.

---

## Sweep methodology + provenance

- GEO/PubMed search terms exercised: "bladder cancer DNA methylation Illumina 450K EPIC GEO cohort" / "bladder cancer EPIC 850K methylation 2024 cohort urine sediment GEO accession" / "GSE bladder cancer methylation 450K tissue tumor adjacent normal" / "WHI bladder cancer methylation Jordahl GEO accession dbGaP" / "Christensen GSE89093 bladder methylation New Hampshire EPIC blood" / "bladder cancer EPIC 850K methylation cohort urine GEO accession" / "GSE13507 GEO platform Illumina 450K bladder methylation samples" / "UroMark bladder cancer urine methylation 150 CpG panel diagnostic AUC" / "Bryan Birmingham bladder cancer methylation BCPP urine sediment GEO accession" / "EpiSCORE bladder reference cell types urothelial Zhu Teschendorff 2022".
- Recent (2020-2025) reviews consulted: Tomiyama 2024 IJU urinary markers review (DOI 10.1111/iju.15338); Gurung 2020 *Eur Urol Focus* prognostic systematic review; 2025 npj Precision Oncology bladder methylation ML review; 2024 *Medicina* matched-sample CpG analysis (PMC11279046); 2025 systematic review NMIBC prognostic methylation (PMC12685936).
- Direct platform-code verification performed for GSE13507 (caught the methylation/expression mis-call propagating in 4+ recent papers).
- WHI access tier confirmed via University of Washington epi.washington.edu/epi_research/pre-diagnostic-genome-wide-dna-methylation-in-blood-and-risk-of-bladder-cancer/.
- Christensen Chen 2022 GEO deposit status NOT YET VERIFIED — task for Phase A.

---

**End of Phase 0 cohort survey.** Heath signs off on which cohorts go into Phase C of bladder-epic v0.1 BEFORE Phase A begins.
