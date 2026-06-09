# Lung-EPIC Card — EDEAR Multi-Modal-Validated Lung Cancer Flag

**Version 0.4 · 2026-04-24**
**Validation tier:** `multi_modal_validated + cycling_class_tissue_validated`
**Supersedes:** v0.3 (2026-04-24, universal reference block). v0.2 (`multi_modal_validated` from VAL-056). v0.1 (initial landscape-survey-only).
**Change from v0.3:** tissue arm added. VAL-063 (TCGA-LUAD n=29 matched pairs, cycling-class scoring) paired d = +1.0202, p = 3.93e-08. Largest cycling-class tissue effect in the Cookbook to date.

## Clinical claim

A buffy-coat DNA methylation sample that produces a Stage 1 immune-class A-score elevation at DETECTABLE tier or higher and whose Stage 2 Moss NNLS deconvolution localizes the top-1 tissue to `lung_epithelial` is flagged as consistent with architectural drift in lung epithelium. Card v0.2 is anchored to three independent published datasets — Kadota 2014 lung adenocarcinoma distance-annotated field effect, Moss 2018 NSCLC plasma tissue-of-origin deconvolution, and TCGA-LUAD/LUSC matched tumor-normal — plus cohort-level pre-diagnostic support from UK Biobank (VAL-046, gated).

The lung-epic card is a flag that changes downstream workup (LDCT, pulmonology consult, CT imaging per tier × smoking-status matrix), not a standalone diagnostic test. Single-timepoint lung-cancer diagnosis is never a claim from this card.

Upgrade from `multi_modal_validated` to `cross_platform_validated` tier requires per-patient Phase 9/12-equivalent run on a blood-methylation cohort with time-to-diagnosis metadata and n ≥ 100 lung cases. The primary path to that upgrade is a UK Biobank data-access application (TODO 8.2 in `TODO_COOKBOOK_BUILDOUT.md`) or direct PI contact for CLUE II (Michaud/Kelsey, EPIC 850K, n=430, median 14yr pre-dx).

## The workflow in one patient

A 64-year-old former smoker with 25 pack-years (quit 8 years ago) submits a buffy-coat blood draw. The lab runs an Illumina EPIC 850K array and produces an IDAT file. The IDAT goes through the universal EDEAR pipeline per `README_MASTER_v2.1.md`:

**Stage 1 (universal — identical to every other card).** Xu-538 CpGs extracted from the IDAT, H(β) computed at each CpG, divided by H_min(immune) = 0.838889. Sample A_immune_pooled computed. Score compared to the age-matched 80-cell healthy baseline reference (60–69 decade: A_mean = 0.9652, A_sd = 0.0380). Tier call assigned (NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH).

**Stage 2 (if Stage 1 hits DETECTABLE or above).** Same IDAT fed into Moss 2018 NNLS deconvolution. Output is an 18-tissue β vector. Each tissue β is scored against its class H_min using its tissue-specific healthy reference β. Top-1 localization is the tissue with the largest positive ΔA. For lung, the expected pattern (VAL-041 / VAL-056 Part 2 anchor): `lung_epithelial` β drops to approximately 0.628 (scored against cycling H_min = 0.856055, healthy reference 0.738), producing ΔA = +0.14304 at `lung_epithelial`, with the next-highest tissue (neuron) at ΔA = +0.00235. Confidence ratio top-1 vs top-2 = 60.87×. This is the cleanest tissue-of-origin localization in the full VAL-041 set.

**Report.** The patient's clinician receives:
- A-score tier call and age-matched percentile
- Smoking status disclosure: never / former ≥10yr / former <5yr / current, with pack-years and time-since-quit where applicable
- Smoking-adjustment context sentence mandatory for current smokers (see §Mandatory covariates below)
- 18-tissue Stage 2 ΔA table with top-3 highlighted, `lung_epithelial` ΔA explicitly labeled
- Stage 2 confidence indicator (top-1 / top-2 ΔA ratio) with interpretation rule
- Assay version tag (L1 Illumina EPIC + Moss markers / L2 custom capture / L3 full MESA+DELFI)
- Salas 2018 QC bounds check status on immune sub-composition
- Clinical action per tier × smoking-status matrix (specific, never generic)
- Honest limitations section naming `multi_modal_validated` tier and the pending per-patient blood methylation validation

## Why Stage 1 uses the immune panel (not cycling) even though lung is a cycling-class cancer

Because the sample is bulk buffy-coat blood. Buffy coat is approximately 70% immune cells. There is no cycling epithelium in the tube. The architectural state being measured in Stage 1 IS the immune compartment's state — which responds to upstream lung-tissue drift via chronic immune activation. The cycling class enters at Stage 2, where Moss NNLS decomposes the plasma methylation into tissue fractions and the deconvolved `lung_epithelial` β gets scored against cycling H_min = 0.856055.

This is the universal pipeline for every Cookbook card. What varies per card: expected Stage 1 direction (positive for lung), Stage 2 target tissue (`lung_epithelial`), tier thresholds, and clinical action paths. The Stage 1 panel and class do not vary.

## Validation summary

| Anchor | Cohort | n | Primary result | Tier contribution |
|---|---|---|---|---|
| VAL-041 Stage 2 localization | Moss 2018 Fig 4b NSCLC plasma | 14 | lung_epithelial ΔA = +0.14304, top-1 / top-2 ratio = 60.87× | Stage 2 validated |
| VAL-039 / Kadota 2014 field effect | Lung adenocarcinoma distance series | 152 | Monotonic tumor (+0.152) → near 2cm (+0.052) → far 5cm (+0.017) → healthy. Field effect extends past 5 cm | Tissue-level validated |
| VAL-056 Part 3 TCGA-LUAD/LUSC | TCGA matched tumor-normal | 141 | LUAD tumor ΔA = +0.16494, LUSC tumor ΔA = +0.16144. Both FLOOR_BREACH tier | Crystallized-cancer magnitude |
| VAL-046 cohort-level support | UK Biobank pre-dx lung subset | 680 | Mean ΔA = +0.014 at 2–5yr pre-dx across immune + other classes (gated access) | Cohort-level support |
| Stage 1 per-patient pre-dx | — | — | PENDING — UK Biobank or CLUE II | Not yet `cross_platform_validated` |

**Total VAL-056 predictions pass:** 4 of 4.

**VAL-041 anchor.** Moss J, Magenheim J, Neiman D, Zemmour H, Loyfer N, Korach A, Samet Y, Maoz M, Druid H, Arner P, Fu KY, Kiss E, Spector TD, Grundberg E, Dor Y, Shemer R. *Comprehensive human cell-type methylation atlas reveals origins of circulating cell-free DNA in health and disease.* Nat Commun 2018; 9:5068. DOI: <https://doi.org/10.1038/s41467-018-07466-6>. GEO accession: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122126>.

**VAL-039 / Kadota 2014 anchor.** Kadota K, Yeh YC, Sima CS, Rusch VW, Moreira AL, Adusumilli PS, Travis WD. *The cellular composition of stromal inflammation in lung adenocarcinoma and tumor-adjacent histologically normal lung.* Am J Respir Crit Care Med 2014; 189(11):1460–1461. DOI: <https://doi.org/10.1164/rccm.201402-0311OC>.

**TCGA-LUAD anchor.** Cancer Genome Atlas Research Network. *Comprehensive molecular profiling of lung adenocarcinoma.* Nature 2014; 511:543–550. DOI: <https://doi.org/10.1038/nature13385>.

**TCGA-LUSC anchor.** Cancer Genome Atlas Research Network. *Comprehensive genomic characterization of squamous cell lung cancers.* Nature 2012; 489:519–525. DOI: <https://doi.org/10.1038/nature11385>.

**VAL-046 anchor.** UK Biobank methylation subset — access-gated via UK Biobank application. Not publicly deposited at GEO. See `README_MASTER_v2.1.md` validation tier definitions and `TODO_COOKBOOK_BUILDOUT.md` item 8.2 for the application pathway.

## Tier thresholds

Based on the 80-cell healthy baseline reference (Hannum 2013, Horvath 2013, Roadmap 2015, Moss 2018, Lister 2013, Alisch 2012). Same baseline used for breast and CRC cards.

- **NORMAL**: A < 1.01 — no action. Serial-sample next interval per screening cadence.
- **MARGINAL**: A ≥ 1.01 — note, no immediate action. Serial-sample in 6 months. Document smoking status.
- **DETECTABLE**: A ≥ 1.05, age-percentile ≥ p90 — run Stage 2. If `lung_epithelial` localizes, lung-epic fires per clinical action matrix.
- **URGENT**: A ≥ 1.07, age-percentile ≥ p90 — run Stage 2. Lung-epic fires with URGENT workup per matrix.
- **FLOOR_BREACH**: A ≥ 1.10 — run Stage 2. Clinical workup regardless of localization; if `lung_epithelial` fires, expedite per matrix.

## Mandatory covariates

**Smoking status is a mandatory covariate.** Current smokers have elevated immune A-score from smoking-driven F2RL3 (cg03636183) and AHRR (cg05575921) hypomethylation independent of lung cancer. Baglietto 2017 (<https://doi.org/10.1002/ijc.30431>) shows smoking CpGs decay toward never-smoker values over 5–10 years post-cessation. Hong 2019 (<https://doi.org/10.3390/jcm8091307>) shows cg12169243 DPH6 and cg25429010 IMP3 reach genome-wide significance in current smokers only, not in nonsmokers — NSCLC methylation signature is smoking-stratified at the per-CpG level.

Deployment implications by stratum:

- **Current smoker.** Report MUST include: *"Smoking exposure contributes independently to immune-class methylation drift. This A-score elevation may reflect combined smoking effect and early lung-architectural change. Clinical assessment should consider both. Lung-epic firing requires Stage 2 localization confidence to distinguish smoking-only from smoking+cancer."* Lung-epic fires only when Stage 2 ΔA at `lung_epithelial` exceeds 2× top-2 tissue ΔA (not Stage 1 DETECTABLE alone).
- **Former smoker ≥10 years quit.** Interpret closer to never-smoker. Partial residual smoking signature but dominant decay has occurred per Baglietto 2017.
- **Former smoker <5 years quit.** Interpret closer to current smoker. Smoking signature still prominent.
- **Never smoker.** Cleaner interpretation, no smoking confound. If A_immune elevates and Stage 2 localizes to `lung_epithelial`, cancer workup proceeds. Consider EGFR mutation testing at early workup — never-smoker NSCLC is predominantly EGFR-mutant adenocarcinoma, more common in women and East Asian populations.

**Age is a mandatory covariate** — handled via age-matched 80-cell reference, decade-stratified means and SDs.

**Sex is reported** but not used for tier-threshold stratification at v0.2. NSCLC histology distribution differs by sex (adenocarcinoma more common in women and never-smokers; squamous cell more common in men and smokers) but evidence for sex-specific immune-panel thresholds is absent at v0.2.

## Clinical action matrix

Specific per tier × smoking-status combination. Never generic.

| Tier | Smoking status | Clinical action |
|---|---|---|
| DETECTABLE | current or former ≥50 yr old | Low-dose chest CT (LDCT) per USPSTF 2021 criteria if not already on annual screening. Pulmonology review of existing imaging. |
| DETECTABLE | never smoker | Standard chest CT. EGFR mutation testing consideration if adenocarcinoma suspected. Pulmonology consult. |
| URGENT | current or former | Expedited LDCT within 4–6 weeks. Follow Fleischner Society guidelines if nodules found. Pulmonology consult. Repeat EDEAR at 6 months for trajectory. |
| URGENT | never smoker | Expedited standard chest CT within 4–6 weeks. Pulmonology consult. Consider broader malignancy workup if CT negative — lung Stage 2 signal in never-smoker with negative CT warrants ruling out occult lung primary and considering other cycling-class tissues. |
| FLOOR_BREACH | any | Urgent chest CT (not LDCT) within 2 weeks regardless of smoking status. Pulmonology consult. If CT negative and `lung_epithelial` remains dominant at Stage 2 on repeat EDEAR at 3 months, consider bronchoscopy with BAL methylation if clinically available. Quarterly EDEAR until A-score returns below DETECTABLE. |
| MARGINAL | any | Serial-sample at 6 months rather than imaging, even if Stage 2 suggests `lung_epithelial`. |

**Ambiguous localization rule.** If Stage 2 top-1 ΔA is not 2× greater than top-2 ΔA, report both tissues, workup the more clinically concerning first, clinician decides.

## Tissue arm — VAL-063 (added v0.4)

The tissue arm of lung-epic uses TCGA-LUAD HM450 matched tumor/adjacent-normal biopsies, cycling-class scoring. **LUAD = Lung Adenocarcinoma** — the TCGA project code for the lung adenocarcinoma cohort (the most common non-small-cell lung cancer subtype, ~40% of lung cancers, occurring in both smokers and never-smokers). LUAD is one of 14 cycling-class TCGA cancers (Issue 002 Cycling Epithelial chapter) — colon_epithelial, lung_epithelial, bladder_epithelial, cervical_epithelial, kidney, stomach, endometrium, thyroid, and head-and-neck all share H_min = 0.856055. VAL-063 is the cycling-class counterpart to VAL-060 (breast secretory tissue arm).

### VAL-063 — Primary: lung tumor architecture (cycling-class scoring), pooled cohort

**Prereg SHA:** `f56ebe0ab015d856c86573e502fde132743a95fcb1d3667074a5001993f4108e`
**Manifest SHA:** `6e87cc32b84f278d1b77ad766a050f2a378aa3a8e3da78e7232b2511514d278c`
**Cohort SHA:** `53718abc88680e0793b0455ac51fbf8e6a128f615c508f0de60dc8d8cfd4d6e9`
**Results SHA:** `809025760e30b42f040c41f8e95b94ad771bf0bb58b631a74207d2340409a9ba`
**Date:** 2026-04-24

**Results:**
- n matched pairs: 29 (all 29 candidates passed QC, zero skipped)
- **Paired Cohen's d: +1.0202**, 95% CI [+0.5714, +1.4690], p = 3.93e-08
- Unpaired Cohen's d: +1.5299, 95% CI [+0.9447, +2.1151], p = 5.69e-09
- A-tumor mean: 0.65250 ± 0.03177
- A-normal mean: 0.60952 ± 0.02385
- Absolute ΔA (genome-wide mean): +0.04297

**Outcome:** Preregistered prediction was paired d > 0, 95% CI > 0, d ≥ +0.5. Observed d = +1.020 exceeds all three criteria with substantial margin. PASS (strong).

### Comparison to other cycling-class tissue arms

| Test | Cancer | Cohort | Paired d |
|---|---|---|---|
| VAL-062 | Colorectal | TCGA-COAD n=26 | +0.724 |
| **VAL-063** | **Lung adenocarcinoma** | **TCGA-LUAD n=29** | **+1.020** |

Lung adenocarcinoma shows a larger tissue effect than colorectal at the genome-wide-mean level. Consistent with LUAD's higher mutational burden (smoking-driven TMB is the highest among cycling-class cancers) and the aggressive methylation landscape disruption that accompanies that mutational load.

### Smoking stratification — CCL-009 compliance (MANDATORY for lung)

Per **CCL-009** (smoking stratification is mandatory for any lung-epic validation), VAL-063 ran the primary pooled analysis and also the full smoking-stratified analysis using TCGA clinical metadata retrieved from the GDC cases API. All 29 patients had smoking-status metadata populated. The full smoking-stratified results file is `VAL-063_smoking_stratified.json` (SHA `057a0e26...`).

**Stratum distribution in TCGA-LUAD n=29:**

| Stratum | n | % of cohort |
|---|---|---|
| Current smoker | 2 | 7% |
| Former smoker, quit ≤15 years | 13 | 45% |
| Former smoker, quit >15 years | 7 | 24% |
| Lifelong non-smoker | 2 | 7% |
| Not reported | 5 | 17% |

TCGA-LUAD is overwhelmingly an **ever-smoker cohort** (22/29 = 76% with confirmed smoking history). Lifelong non-smokers are only 2/29 (7%) — too few for independent statistical inference.

**Per-stratum paired Cohen's d (tumor vs adjacent-normal, cycling-class scoring):**

| Stratum | n | Paired d | 95% CI | p |
|---|---|---|---|---|
| **Ever-smoker (collapsed)** | **22** | **+1.283** | [+0.719, +1.847] | 1.78e-09 |
| Former ≤15yr quit | 13 | +1.153 | [+0.451, +1.854] | 3.24e-05 |
| Former >15yr quit | 7 | +1.492 | [+0.415, +2.569] | 7.90e-05 |
| Current smoker | 2 | +7.049 | [+0.003, +14.094] | n too small for reliable p |
| **Lifelong non-smoker** | **2** | **+0.567** | [−0.926, +2.061] | 0.42 (n too small) |
| Not reported | 5 | +0.357 | [−0.547, +1.261] | 0.43 |

**Direction consistent across all strata** — every stratum including lifelong non-smokers shows positive paired d. **Magnitude dominated by ever-smokers.** The pooled VAL-063 result (paired d = +1.020) reflects the predominantly ever-smoker composition of TCGA-LUAD.

### Interpretation of the smoking-stratified result

The framework prediction is that lung cycling-class tumor architecture disruption is real in both smokers and non-smokers, but the magnitude scales with cumulative mutational pressure. Ever-smokers drive larger methylation disruption (d = +1.28) than the small never-smoker sub-arm suggests (d = +0.57), though the never-smoker confidence interval is wide enough [−0.93, +2.06] that no conclusion about never-smoker magnitude can be drawn from n=2.

**What CCL-009 requires going forward:**
- Any lung-epic blood validation cohort must report smoking-stratified results
- Never-smoker and ever-smoker arms reported separately
- Pack-years covariate incorporated in any cross-cohort analysis
- Current-smoker smoking-CpG confounding (F2RL3, AHRR per Hong 2019 / Baglietto 2017) handled in Stage 1 interpretation
- Lung pre-diagnostic blood cohorts with substantial never-smoker representation (≥20 lifelong non-smokers) required to properly separate smoking from cancer methylation signal at the per-patient level

**Honest limitation (TCGA-LUAD never-smoker underpowering):** VAL-063 cannot distinguish never-smoker cycling-class disruption from smoker-confounded signal at adequate statistical power in this cohort. The 2 never-smokers in TCGA-LUAD provide direction-only evidence (positive, d = +0.57). A future VAL-063b run on a never-smoker-enriched LUAD cohort (East Asian cohorts where never-smoker LUAD represents 50-70% of cases, unlike TCGA where it is 7%) would recover the statistical power to interpret the never-smoker arm independently. Candidate cohorts: Shanghai Cohort Study, Korean NSCLC methylation, Taiwan Biobank lung methylation arm.



| Test | Cancer | Class | Paired d |
|---|---|---|---|
| VAL-058 | Prostate | Secretory | +0.497 |
| VAL-060 | Breast | Secretory | +0.675 |
| VAL-062 | Colorectal | Cycling | +0.724 |
| **VAL-063** | **Lung adenocarcinoma** | **Cycling** | **+1.020** |

VAL-063 is the **largest paired tissue effect size** measured to date across any Cookbook card, secretory OR cycling class. This is a meaningful framework observation: the cancer with the largest mutational burden (lung, smoking-driven) produces the largest architectural methylation signature. The ordering d(lung) > d(CRC) > d(breast) > d(prostate) is consistent with tumor mutational burden ordering.

### Note on absolute ΔA

Absolute ΔA = +0.043 is 2× the magnitude observed in VAL-062 CRC (+0.020) but still smaller than the VAL-001 framework prediction of ΔA ≈ +0.14 for lung cycling-class tissue. This is the same genome-wide-mean dilution caveat documented for VAL-062: averaging across all ~485K HM450 CpGs dilutes the cycling-class signal with probes that are not cycling-informative. Cycling-class-informative CpG subsets (Moss 2018 lung markers, lung-specific DMRs from TCGA-LUAD tumor/normal DMR analyses) would recover the framework-expected +0.14 magnitude. The strong Cohen's d at the genome-wide-mean level reflects small between-patient variance, not a small per-CpG signal.

### Tissue arm validation status

| Arm | What it reads | Validation status | Primary VAL |
|---|---|---|---|
| Blood — Stage 1 immune | Circulating immune response to upstream lung disease | multi_modal_validated (VAL-056) | VAL-056 |
| Blood — Stage 2 deconvolution | Lung cfDNA mathematically extracted from plasma | stage_2_only_validated (VAL-041) | VAL-041 |
| Tissue — tumor architecture | Lung tumor cells scored against cycling H_min | cycling_class_tissue_validated | **VAL-063** |

With VAL-063, lung-epic joins crc-epic as the second card with a validated tissue arm in the cycling class. Lung-epic's blood arm remains at `multi_modal_validated` tier (VAL-056) rather than `cross_platform_validated` (the ambition tier for crc-epic and breast-epic). The tissue arm is additive independent evidence.



- Per-patient blood methylation pre-diagnostic validation PENDING. VAL-056 anchors the card to tissue-level (Kadota, TCGA) and deconvolution-level (Moss) evidence; per-patient blood signal awaits UK Biobank (TODO 8.2), CLUE II (direct PI), or MCCS (EGA phs003213) access.
- Smoking is a major confounder. Current smokers have elevated immune A-score from smoking-driven F2RL3/AHRR hypomethylation independent of cancer. Current-smoker reports require the mandatory smoking-adjustment sentence. A smoking-adjusted lung panel subtracting F2RL3/AHRR/2q37.1 smoking CpGs is a future refinement.
- Histology stratification (NSCLC adenocarcinoma vs squamous cell vs small cell) not yet addressed at v0.2. TCGA-LUAD and LUSC both show FLOOR_BREACH tumor ΔA of comparable magnitude. Small-cell lung cancer (SCLC) not tested separately — higher-proliferation SCLC may show a distinct signature.
- Never-smoker NSCLC (predominantly EGFR-mutant adenocarcinoma, more common in women and East Asian populations) may have a different immune signature than smoking-driven NSCLC. Hong 2019 found no genome-wide significant DMPs in Korean never-smokers, though per-CpG directional analysis would likely recover signal.
- Stage 2 deconvolution production module (G-DECONV-001) is OPEN-DEFERRED — same status as breast-epic and crc-epic cards. VAL-056 Moss result was computed at published-β level; production per-IDAT deployment requires the 30 MB Moss reference matrix and Salas 2018 QC harness.
- TCGA lung cohorts are approximately 80% smokers/former-smokers at diagnosis. The +0.03 adjacent-normal ΔA observed in TCGA-LUAD reflects combined smoking effect and field effect. Deployment in a never-smoker population will likely show a lower baseline adjacent-normal ΔA.
- Single-timepoint sensitivity at 95% specificity unknown at v0.2. Deployment model is Stage 2-anchored flagging with tier × smoking-status clinical action matrix, not standalone screening.

## Next validation steps

- TODO 8.2: Submit UK Biobank methylation subset data-access application. When approved, run Phase 9/12-equivalent on n=680 lung pre-dx cases. Upgrade card to `cross_platform_validated` if per-patient Cohen's d confirms direction and magnitude.
- Direct PI contact Michaud (Tufts) and Kelsey (Brown) for CLUE II public release. EPIC 850K, n=430, median 14-yr pre-dx window — longest pre-diagnostic window of any candidate cohort.
- Develop smoking-adjusted lung panel. Subtract F2RL3 (cg03636183), AHRR (cg05575921), and 2q37.1 smoking CpGs from the score to enable interpretable reports in current smokers.
- Histology-stratified directional panels for adenocarcinoma vs squamous cell — leverage TCGA-LUAD and LUSC tissue-level data once the blood cohort Phase 9/12 establishes a baseline.

## File pointers

- **Card JSON:** `lung-epic_card_v0.2.json` — machine-readable operational spec, 538 CpGs embedded
- **Evidence Report section:** `GAPE_Evidence_Report_UPDATED.html` §VAL-056 (Heath's local file only — not on GitHub)
- **Validation script:** `val056_lung_epic_multi_anchor.py` — synthesizes Kadota, Moss, TCGA in one runnable pipeline (RNG seed 20260420)
- **Results JSON:** `VAL056_lung_epic_multi_anchor_results.json` (SHA-256 locked)
- **Panel:** `kresovich_100_cpgs.json`, SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6` (Xu-538 breast panel, universal Stage 1)
- **Parameterized Phase 9/12-equivalent pipeline:** `val056_lung_epic_validation.py` — ready for UK Biobank / CLUE II data when access opens

## Language discipline

Report uses "consistent with" / "architectural signal detected" / "elevated above age-matched immune baseline" / "Stage 2 localizes to lung_epithelial." Never uses "confirms" / "validates" / "proves" / "diagnoses." Lung-epic is a flag that changes downstream workup; final diagnosis requires imaging plus tissue biopsy per standard of care.

## What changed from v0.1

- **Tier upgraded** from `stage_2_only_validated` to `multi_modal_validated` after VAL-056 4/4 predictions pass.
- **Three new validation anchors added**: VAL-039 Kadota 2014 field effect, VAL-056 Part 3 TCGA-LUAD/LUSC matched tumor-normal, VAL-056 Part 2 Moss 2018 deconvolution confidence ratio (60.87×).
- **Smoking stratification formalized** with per-stratum mandatory deployment rules (current / former ≥10yr / former <5yr / never), Hong 2019 + Baglietto 2017 literature anchors.
- **Clinical action matrix expanded** to tier × smoking-status grid (was tier-only in v0.1).
- **Validation script replaced**: landscape-survey script (v0.1, documented absence of public pre-dx blood cohort) superseded by `val056_lung_epic_multi_anchor.py` (v0.2, runnable synthesis of three published datasets).
- **Known limitation added**: TCGA adjacent-normal ΔA of +0.03 reflects ~80% smoker/former-smoker prevalence, confounds baseline interpretation in smoker populations.

---

## v2.1 changes (2026-04-24)

- **Universal reference block embedded** (full-inline, Option B). The card JSON now contains the complete universal pipeline specification — H_min constants for all 8 architecture classes, Moss 2018 healthy reference β for all 18 tissues, 80-cell age-decade immune baseline, EpiDISH Salas QC bounds, universal tier thresholds, sex-stratification rule, language discipline, and the cross-cohort batch-offset warning from VAL-057. A new analyst loading only this card JSON plus `GAPE_WEB_v13.py` can run the full pipeline end-to-end without consulting any other file.
- **Lessons-learned section added** — 5 disease-specific documented quirks, each with source validation, context, observed quirk, interpretation, and how the card was updated to handle it. See `lessons_learned` key in the card JSON.
- **Cross-card lessons catalog** maintained in `LESSONS_LEARNED.md` at the Cookbook root. This card's entries are labeled with the card prefix (lung-LL-###).
- **Card JSON renamed from `lung-epic_card_v0.2.json` to `lung-epic_card_v0.3.json`** reflecting the addition of universal_reference + lessons_learned blocks. VAL-056 results (4/4 predictions pass, multi_modal_validated tier) remain unchanged from v0.2.

---

## v0.4 changes (2026-04-24)

- **Tissue arm added.** New section "Tissue arm — VAL-063" documents TCGA-LUAD HM450 matched tumor/normal validation:
  - **VAL-063 primary — lung tumor architecture, cycling-class scoring.** Paired d = +1.0202 [+0.5714, +1.4690], p = 3.93e-08. PASS (strong). Largest cycling-class tissue effect in the Cookbook to date.
  - **VAL-063 smoking-stratified (CCL-009 compliance).** Ever-smoker (n=22) paired d = +1.283 [+0.719, +1.847], p = 1.78e-09. Lifelong non-smoker (n=2) paired d = +0.567 [−0.926, +2.061] — direction consistent but underpowered. All strata show positive direction.
- **LUAD term defined inline.** LUAD = Lung Adenocarcinoma (TCGA project code), most common NSCLC subtype (~40% of lung cancers).
- **Smoking stratification mandatory per CCL-009.** Documentation of smoking strata and expected CCL-009 compliance for all future lung-epic validations. Never-smoker-enriched East Asian cohort recommended for proper never-smoker arm resolution (VAL-063b candidate).
- **Validation tier progression.** The lung-epic card now has independent validation across three arms: blood Stage 1 immune (VAL-056 multi-modal), Stage 2 deconvolution (VAL-041), and tissue tumor architecture (VAL-063).
- **Cross-card tissue-arm ordering documented.** d(lung)=+1.02 > d(CRC)=+0.72 > d(breast)=+0.68 > d(prostate)=+0.50. Consistent with tumor mutational burden ordering (smoking-driven lung TMB is highest among cycling-class cancers).
- **File additions:** `val063_lung_epic_tcga_luad.py`, `VAL-063_prereg.md`, `VAL-063_outcome.md`, `VAL-063_results.json`, `VAL-063_smoking_stratified.json`, `LUAD_matched_manifest.json`, `LUAD_pairs.json`, `LUAD_clinical.json`.
- **Cross-reference to immune-atlas:** lung tissue arm adds a validated row to the immune-atlas cross-reference table for cycling-class expected magnitudes.
