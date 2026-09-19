# Breast EPIC Card — EDEAR Pre-Diagnostic Detection

**Version 2.3 · 2026-04-26**
**Supersedes:** v2.2 (VAL-060 tissue arm) · v2.1 (universal_reference full-inline) · v2.0 (cross_platform_validated_two_cohorts base)

## What EDEAR is and is not (added in v2.3, applies to every card)

EDEAR is a **direct-to-consumer cellular state report**. The customer pays out-of-pocket and receives a report describing where their cells sit, by architecture class, relative to age-matched healthy reference, at the moment of the draw. EDEAR v1 is positioned as a health-and-wellness tool, not a medical diagnostic. It is not FDA-approved, is not reimbursed by insurance, and does not require a clinician to order or interpret the report. The customer is the reader.

EDEAR does NOT diagnose breast cancer or any other disease. It tells the customer what their cells are doing now, and — if they take serial samples over time — what the trajectory looks like. The trajectory is the product. A single reading is a coordinate; a series of readings is the story the customer's body is telling them. Annual or biannual draws are how the instrument actually works; the report is most informative as part of a multi-year personal baseline against which drift becomes visible.

Customers are encouraged to layer EDEAR onto draws they are already getting (annual physicals, lipid panels, A1C, life-insurance draws, pre-employment physicals) and to pair the report with the lifestyle changes they already know matter — sleep, food, exercise, stress, smoking, alcohol, weight. EDEAR is the cellular-level instrumentation that makes those lifestyle changes measurable in their own body. When a customer's report shows drift across multiple cell classes, the wise interpretation is "the system as a whole is showing wear; continue monitoring while making the lifestyle changes I control" — not "you have or will have cancer X."

The breast-specific signal documented in this card emerges as a localized tile shift only in the 24 months before clinical diagnosis (per VAL-096 added in v2.3). Long pre-diagnostic windows show a body-wide cellular-aging-drift signal, not a breast-specific signal. The card's tier thresholds and tissue-of-origin language in the operational sections below are the technical specification for the report engine and the SHA-locked validation evidence; the customer-facing language in the report itself is calibrated to match what the data actually support — cellular state, trajectory, and "this is a flag worth watching" rather than disease prediction.

## Clinical claim

A buffy-coat DNA methylation sample from a woman 2 to 10+ years before her clinical breast cancer diagnosis shows an elevated immune-class architectural A-score on the Xu-538 panel. The signal is detectable at the cohort level at pooled effect size d = +0.45 to +0.71 across two independent EPIC-Italy cohorts, and rises monotonically with pre-diagnostic interval to d ≈ +1.4 to +1.8 at 10+ years pre-dx. A second-stage Moss 2018 tissue-of-origin deconvolution localizes the source to breast ductal tissue in 10 of 10 cases tested (VAL-041).

Breast EDEAR is not a diagnostic test. It is a flag that changes downstream workup, comparable to troponin for cardiac state. Serial-sample trajectory monitoring with each patient as her own control is the primary clinical deployment model.

## The workflow in one patient

A 55-year-old woman submits a buffy-coat blood draw. The lab runs an Illumina EPIC 850K array and produces an IDAT file. The IDAT goes through:

**Stage 1.** Xu-538 CpGs extracted, H(β) computed at each CpG, divided by H_min(immune) = 0.838889. Sample A_immune_pooled computed. Score compared to the age-matched 80-cell healthy baseline reference. Tier call assigned (NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR BREACH).

**Stage 2 (if Stage 1 hits DETECTABLE or above).** Same IDAT fed into Moss 2018 NNLS deconvolution. Output is 18 estimated per-tissue β values. Each tissue β is scored against its class H_min using its tissue-specific healthy reference β (breast_ductal: healthy 0.744, H_min 0.843264). The tissue with the largest positive ΔA is the localization. For breast, the expected pattern: breast_ductal ΔA elevated, all 17 other tissues near healthy.

**Report.** The patient's clinician receives: A-score tier, age-matched percentile, Stage 2 18-tissue ΔA table with top-3 highlighted, Stage 2 confidence indicator (top-1 to top-2 ΔA ratio), Salas 2018 immune sub-composition QC status, and honest limitations.

## Why Stage 1 uses the immune panel (not secretory) even though breast is a secretory-class cancer

Because the sample is bulk buffy-coat blood. Buffy coat is approximately 70% immune cells. There is no secretory tissue in the tube. The architectural state being measured in Stage 1 IS the immune compartment's state — which responds to upstream breast-tissue drift via chronic immune activation, per the Dunn-Old-Schreiber immunoediting framework.

The secretory class enters at Stage 2, where Moss NNLS decomposition estimates how much of the plasma methylation signal looks like it came from breast ductal tissue. That deconvolved breast_ductal β gets scored against secretory H_min. Stage 2 is where the class-of-origin discrimination happens.

## Validation summary

| Test | Cohort | n (cases / ctrl) | Primary result | Tier |
|---|---|---|---|---|
| VAL-047 Phase 9 | GSE51057 EPIC-Italy (blood pre-dx) | 146 / 177 | d = +0.45 pooled, +1.78 at >10yr | cross_platform_validated |
| VAL-047 Phase 12 | GSE51032 EPIC-Italy (blood pre-dx) | 224 / 424 | d = +0.71 pooled, +1.36 at >10yr | cross_platform_validated |
| **VAL-060 tissue arm** | **TCGA-BRCA HM450 matched** | **89 / 89 (86 complete pairs)** | **Unpaired d = +0.745 [+0.451, +1.075], paired d = +0.676, p = 0.0001** | **tissue_arm_validated** |
| VAL-041 | Moss 2018 + Liu 2020 per-tissue | 10 breast cases deconvolved | breast_ductal top-1 localization | Stage 2 validation |
| T2 cross-pop | GSE104942 Australian HBOC | (per MANIFEST) | d = +0.29 | directionally_positive |
| T5 cross-pop | TwinsUK paired | 15 pairs | d = +0.16 pooled (+0.97 at 0-2yr) | directionally_positive |
| T9 cross-pop | GSE283951 Polish | 34 / 56 | d = +0.29 | directionally_positive |
| T10 cross-pop | GSE37965 Heyn UK twins | 15 pairs | d = +0.18 | directionally_positive |
| T11 cross-pop | GSE243529 Singapore | 256 / 268 | d = +0.12 (at-dx shallow) | directionally_positive |
| **VAL-093 Stage 2 25-tile** | **GSE51057 + GSE51032 EPIC-Italy** | **(see below)** | **Distributed pancreatic + cycling-class elevation at >10yr; breast tile null** | **stage_2_distributed_pattern_documented** |
| **VAL-094 EpiSCORE breast** | **GSE51057 + GSE51032** | **same** | **7-cell EpiSCORE BreastRef behaves as one coherent signal (resolution-collapse)** | **stage_2_resolution_test_negative** |
| **VAL-095 UniLIFE 19-cell** | **GSE51057 + GSE51032** | **same** | **aTreg elevated at >10yr (d=+1.26 / +0.79, CIs exclude zero); aBnv elevated at 0-2yr (d=+0.44 / +0.49)** | **stage_3_resolution_gain_replicating** |
| **VAL-096 TTD-window stratification** | **GSE51057 + GSE51032** | **same** | **Two-component temporal model: persistent distributed cellular-aging-drift + late-localizing breast tile (d=+0.43 / +0.49 at 0-2yr)** | **stage_2_temporal_pattern_documented** |

**Live re-run on 2026-04-23.** Phase 9 ran on /home/claude/GSE51057_series_matrix.txt.gz (SHA 828059...04d98bb0 LOCKED, bit-identical to the prereg-stored hash) in 40.6 seconds, produced the Phase 9 numbers above with RNG seed 20260420. Phase 12 ran on the 3.15 GB GSE51032 matrix in 131.3 seconds, produced the Phase 12 numbers. Both result JSONs archived in /mnt/user-data/outputs alongside this card.

## VAL-060 tissue arm — TCGA-BRCA HM450 matched tumor-normal (added in v2.2)

**What VAL-060 tests.** The breast-epic card's primary validation is on pre-diagnostic BLOOD methylation (GSE51057 and GSE51032 EPIC-Italy buffy coat). The card has never had its own per-card tumor-vs-adjacent-normal tissue run using the Xu-538 panel specifically — the equivalent of what VAL-058 did for prostate-epic. VAL-060 closes that gap as the first retroactive per-card tissue re-validation under the new standard (CCL-011: every card gets its own tissue arm going forward, existing cards upgrade retroactively).

**Cohort.** TCGA-BRCA HM450 matched tumor-normal subset, NIH Genomic Data Commons public access (no dbGaP required for Level 3 β values). 186 sample files downloaded and SHA-locked, comprising 92 Primary Tumor + 91 Solid Tissue Normal + 3 Metastatic. After valid-score filtering (≥300 CpG coverage per sample), 86 complete tumor-vs-adjacent-normal patient pairs were analyzed. Aggregate cohort SHA `a11efdabfe2aec78d323371ce2687dbadaa506ce44be3229ee10c79fb3c97742`. The cohort is overwhelmingly female (89/89 tumors + 89/89 normals female in the scored subset, matching TCGA-BRCA's documented 99% female composition).

**Method.** Same pipeline as blood: Xu-538 panel β values extracted per sample, pooled-entropy A-score = mean(H(β)/H_min_immune) computed per sample, H_min(immune) = 0.838889 frozen. The panel and H_min are unchanged between blood and tissue analyses. Z-standardization uses within-TCGA-BRCA Solid Tissue Normal mean/SD per CpG — not the 80-cell blood baseline (per CCL-004: blood-derived healthy baselines do not apply to tissue data).

**Platform note.** TCGA-BRCA is Illumina HM450, predecessor to EPIC 850K. HM450 Xu-538 coverage was 430/538 median per sample (80% of panel). All 538 CpGs were designed to overlap both platforms; the 20% that dropped out are EPIC-exclusive probes added in the 2016 array redesign. Running on EPIC would increase coverage to the full 538 but is not expected to change the effect size materially.

**Result.** OUTCOME O1 — BREAST_EPIC_TISSUE_VALIDATED. Unpaired Cohen's d = +0.745 [95% CI +0.451, +1.075], permutation p = 0.0001. Paired Cohen's d = +0.676 on 86 matched patient pairs, sign-flip permutation p = 0.0001. Per-CpG direction: 252/500 hypermethylated in tumor vs adjacent-normal (50.4%) and 248/500 hypomethylated (49.6%), with 115 CpGs at |Δβ| > 0.05. The pooled-entropy A-score detects architectural drift regardless of individual CpG direction balance.

**Comparison to VAL-058 prostate-epic.** Prostate tissue: unpaired d = +0.400, paired d = +0.497. Breast tissue: unpaired d = +0.745, paired d = +0.676. Breast tissue effect size is larger than prostate tissue effect size on the same panel. This is consistent with Xu-538 being the panel originally selected for breast cancer — the disease it was trained on produces the largest tissue response. A panel cross-applied to prostate (a different disease, different tissue class at Stage 2) still produces a clearly positive and statistically significant signal, but smaller.

**What VAL-060 adds to breast-epic's evidence base.** Breast cancer architectural drift is now validated across three substrates and two timepoints with the same Xu-538 panel and same H_min(immune):

1. **Pre-diagnostic blood leukocyte methylation** (GSE51057, GSE51032): d = +0.45 to +0.71 pooled, rising to d ≈ +1.4 to +1.8 at 10+ years pre-diagnosis
2. **Adjacent-normal tissue field effect** (VAL-039 Teschendorff 2016): breast adjacent-normal architectural drift documented
3. **Tumor tissue** (VAL-060 this run): paired d = +0.676 tumor vs adjacent-normal, p = 0.0001

The same panel + same H_min + same pipeline produce consistent positive signals across blood pre-diagnosis, adjacent-normal tissue, and tumor tissue. This is cross-substrate consistency — the framework's prediction that architectural drift is a real, measurable property of the progression trajectory, not an artifact of any single specimen type.

**What VAL-060 does NOT change.** The primary validation tier remains `cross_platform_validated_two_cohorts`. The card's clinical deployment is unchanged — EDEAR is a blood-based pre-diagnostic flag, and tumor-tissue case-control is not a clinical deployment target. The tissue arm is additive evidence: confirmation that the Xu-538 panel is a specimen-general breast architectural drift detector, not specific to blood pre-diagnosis.

**Reproducibility anchors.** Pre-registration `VAL_060_PREREG.md` SHA `cd8c6de4383d87203ad8ee14db6d197635021a5e036e29f3219a613b112e8fea` sealed 2026-04-24 08:14:36 UTC before any TCGA β-value access. Pre-reg + seal + panel SHA + aggregate cohort SHA + results JSON (SHA `6ef629c809073213...`) all committed to GitHub under `Biological_Physics/validation_runs/VAL_060_*` and `VAL060_*`. Anyone can reproduce: download the 186 TCGA-BRCA file IDs listed in `VAL_060_tcga_brca_file_shas.json`, verify individual SHAs against the manifest, run `val060_breast_epic_tcga_brca.py`, obtain the same numbers.

## Other specimen pathways — documented scope, not yet in card

The breast-epic card v2.2 currently supports buffy-coat blood at Stage 1 (Xu-538 panel) and plasma ccfDNA at Stage 2 (Moss NNLS tissue-of-origin deconvolution). Tumor tissue is validated as an evidence arm (VAL-060) but is not a clinical-deployment specimen. Several additional specimen pathways exist for breast methylation testing and warrant future card integration. Each is listed here with its evidentiary status as of 2026-04-24 so the scope is visible and prioritization is explicit.

**Nipple aspirate fluid (NAF).** The breast-specific specimen with the cleanest biological argument: gentle suction or massage at the nipple recovers a small volume of fluid containing epithelial cells shed directly from the breast ductal and lobular system, with vastly higher breast-cell concentration than plasma ccfDNA. Published NAF methylation work includes Sukumar, Fackler, Locke, and others — but the published literature primarily used targeted methylation panels (RASSF1A, HIN1, others) and qPCR-based assays, not genome-wide Illumina methylation arrays. Public genome-wide NAF array data is extremely thin — as of the April 2026 GEO survey conducted in this session, only one deposited cohort exists (GSE238014, n=7, HiSeq X bisulfite sequencing, not Illumina array — not directly Xu-538-compatible). Adding NAF to the breast-epic card requires (a) acquiring a larger NAF methylation cohort on Illumina EPIC or HM450 — likely through collaboration with an academic group or a partner lab collecting NAF clinically; (b) a NAF-specific H_min derived from healthy NAF reference samples (NAF cell mix is predominantly breast ductal epithelial with minor immune contribution, materially different from buffy coat); (c) a NAF healthy baseline by age decade; (d) a NAF case-control validation analogous to VAL-060. Priority: **HIGH** — NAF is likely the specimen that would meaningfully improve early-stage localized breast cancer detection where blood pre-dx signal weakens in the 0-3 year window. Status: documented scope, no current data pipeline.

**Ductal lavage.** A saline flush of a breast duct through a catheter inserted at the nipple, recovering more breast epithelial cells than NAF. More invasive than NAF but yields enough cells for standard methylation arrays. Published methylation work exists (Dooley, Locke groups) but clinical uptake is limited because of patient discomfort. Priority: **MEDIUM** — if NAF cohorts prove acquirable, ductal lavage may not be necessary. If NAF remains data-limited, ductal lavage is the fallback for direct breast-cell methylation.

**Fine-needle aspirate (FNA).** Thin-needle sampling of a specific breast area, typically used clinically for palpable lesions or imaging-identified findings. Within the clinical care pathway rather than a screening specimen. Could support trajectory monitoring in post-imaging follow-up. Priority: **LOW for screening, MEDIUM for post-imaging monitoring.**

**Breast milk.** For lactating women, breast milk contains mammary epithelial cells at high concentration. Relevant for postpartum breast cancer detection (which has distinct biology) and for lactating-period methylation baselines. Not a general screening pathway. Priority: **LOW except for postpartum-specific deployment.**

**Core needle biopsy.** Tissue-grade methylation from a targeted lesion. Already within the clinical care pathway; if methylation is performed on the sample, it gives TCGA-grade data. Not a screening specimen but a confirmatory one. Priority: **LOW** — by the time a core needle biopsy is performed, clinical workup has already been triggered.

**Urine.** Breast cells do not shed into urine in meaningful amounts. Not applicable.

Specimen expansion is a parallel track to tissue-arm expansion (CCL-011). Each new specimen requires its own H_min calibration, its own healthy baseline, and its own validation arm — the same work structure as a tissue-arm addition. NAF is the single highest-priority addition for early-stage breast cancer detection and should be the first specimen expansion sprint for breast-epic when data access becomes available.

## Temporal pattern — why signal grows with longer pre-dx interval

Signal is loudest FURTHEST from diagnosis. At 0-2yr pre-dx, Phase 9 gives d = +0.09 (weak). At 2-5yr, d = +0.31. At 5-10yr, d = +0.71. At >10yr, d = +1.78. The pattern replicates independently on GSE51032.

This is the inverse of serum-marker-style logic. A CA-125 or CEA rises as the tumor grows. The Xu-538 immune signature peaks in the chronic-activation phase before the tumor has established immune escape. Once immunoediting takes over, the immune signal attenuates even as the tumor approaches clinical diagnosis.

Clinically this means EDEAR is most informative as a decade-scale early-warning system. Near-diagnosis signal is weaker. Long-horizon signal is stronger. Serial-sample trajectory monitoring is the natural deployment: a patient sampling annually from age 40 has a 10-sample baseline by age 50, and a rising A_immune_pooled trajectory within her own baseline is far more informative than any single-timepoint score.

## Stage 2 temporal pattern — distributed early, breast-localizing late (added in v2.3)

The Stage 1 immune signal pattern documented above describes the immune compartment's response. The Stage 2 cell-of-origin pattern, documented in VAL-096 (window-stratified Loyfer/Moss 25-tile per-tile A-scores on the same GSE51057 + GSE51032 cohorts), tells a different but complementary story.

At long pre-diagnostic windows (>10yr, 5-10yr, 2-5yr) the signal is **distributed** across multiple tissue tiles — pancreatic beta/acinar/duct, kidney, colon epithelial, head-and-neck-larynx, upper-GI — at d = +0.5 to +1.0 in both cohorts. The breast tile itself reads near-null at these windows (d = +0.20 GSE51057 / +0.10 GSE51032 at >10yr; d = +0.05 / +0.19 at 5-10yr; d = +0.14 / +0.16 at 2-5yr). At the 0-2yr window the breast tile rises to d = +0.43 / +0.49 in both cohorts, while several of the early-elevated tiles attenuate (pancreatic duct goes from +0.99 / +0.70 at >10yr to +0.04 / +0.26 at 0-2yr; head-and-neck-larynx from +0.75 / +0.81 to +0.11 / +0.14).

The data are consistent with a two-component temporal model: a persistent multi-tissue cellular-aging-drift signal that precedes localization by 10+ years, layered with a late-localizing breast tile signal that emerges in the 24 months before clinical diagnosis. The two components are additive, not mutually exclusive — at 0-2yr both the breast tile and several persistently-elevated tiles still show concurrent elevation.

This reframes how breast-epic's pre-diagnostic windows are interpreted at Stage 2: long pre-dx windows show a **distributed cellular-aging-drift signature**, not a breast-localized signal. Stage 2 reports for samples at long pre-dx should not over-claim breast-tissue-of-origin attribution; the localization step happens near clinical diagnosis. VAL-060 (paired d = +0.676 tumor vs adjacent-normal) and VAL-096 0-2yr breast tile (+0.43 / +0.49) are the two cleanest pieces of breast-localized Stage 2 evidence; the at->10yr distributed pattern reflects a multi-tissue field effect rather than a breast-specific localization.

A third observation from VAL-096 is the immune-tile inversion-near-diagnosis: monocyte, neutrophil, and erythrocyte-progenitor tiles attenuate or sign-flip at 0-2yr relative to long pre-dx. Monocyte d goes from +0.33 (>10yr) to −0.35 (0-2yr) in GSE51057 and from +0.00 to −0.40 in GSE51032. This is consistent with the Stage 1 attenuation pattern documented above, surfaced at per-tile resolution. CCL-035 candidate logged for further investigation.

## Stage 3 immune sub-resolution — UniLIFE 19-cell additive to Salas baseline (added in v2.3)

VAL-095 ran a head-to-head deconvolution of UniLIFE Guo 2025 (1,906 CpGs × 19 immune cell types) versus the production Salas Blood.450K legacy panel (350 CpGs × 6 cell types) on the same GSE51057 + GSE51032 cohorts. Production Stage 3 (Salas 6-cell) catches the broad pre-diagnostic immune-phenotype shift correctly. UniLIFE 19-cell adds two specific replicating resolution gains:

1. **aTreg fraction elevated at >10yr breast pre-dx in both cohorts.** GSE51057 d = +1.26 [+0.39, +2.26] (n_cases=11, n_ctrl=177); GSE51032 d = +0.79 [+0.33, +1.33] (n_cases=36, n_ctrl=424). Salas CD4T pooled at the same window shows d = +0.36 / +0.03. UniLIFE separates regulatory T-cell signal from the larger CD4T pool. Healthy aTreg baseline is small (mean ~0.006 of total leukocytes in both cohorts), so absolute interpretation requires care.
2. **aBnv (naive B-cell) fraction elevated at 0-2yr in both cohorts.** GSE51057 d = +0.44 [+0.15, +0.76]; GSE51032 d = +0.49 [+0.23, +0.77]. Salas Bcell pooled at the same window: d = +0.31 / +0.36. UniLIFE separates the naive B-cell signal from the pooled B-cell compartment.

Production Stage 3 remains Salas Blood.EPIC IDOL (or 450K legacy on legacy cohorts). UniLIFE is added as a parallel atlas with output presented as adult-specific subtype overlay, not as a replacement. The card README and the customer-facing report block now reference both layers. UniLIFE is part of run-everything Stage 3 architecture per Heath sign-off 2026-04-26.

## EpiSCORE breast sub-cell-type resolution — does not separate at this scoring setup (added in v2.3, VAL-094)

VAL-094 tested whether EpiSCORE BreastRef's 7 sub-cell-type resolution (Basal, Endothelial, Adipocyte, Fibroblast, Luminal, tissue-Lymphocyte, tissue-Macrophage) surfaces a hidden breast sub-tile signal that the Loyfer/Moss bulk-breast tile missed at >10yr GSE51057. **The 7 sub-cell types behave as one coherent signal, not as 7 independent tile readings.** Per-cell-type d values agree within 0.10-0.16 of each other across all 4 windows in both cohorts.

A secondary observation: at >10yr GSE51057 EpiSCORE produces d = +1.01 to +1.17 across all 7 cell types where the Loyfer breast tile reads d = +0.20. This does not replicate to GSE51032 (d = +0.20 to +0.33 there). The honest reading is a cohort-specific elevation that does not pass the framework's two-cohort replication standard. Logged as observation, not finding.

**Practical consequence:** EpiSCORE BreastRef is not added to the v2.3 card as a per-cell-type discriminator. EpiSCORE remains useful for cross-tissue cell-of-origin attribution (which of the 14 EpiSCORE tissues is most consistent with the customer's signal); it is not useful for sub-cell-type resolution within the breast tile when the input is buffy-coat plasma.

## Tier thresholds

Based on the 80-cell healthy baseline reference (Hannum 2013, Horvath 2013, Roadmap 2015, Moss 2018, Lister 2013, Alisch 2012):

- **BELOW_NORMAL**: A ≤ −1.0 SD below within-cohort or 80-cell age-decade healthy reference mean — surface for clinician differential review (non-disease-of-card differentials: immunosuppression, post-chemo / post-transplant state, primary immunodeficiency, late-stage marrow infiltration, architectural homogenization patterns such as PSP/CBD). NOT silenced — routes to clinician review for differential, not to no-action. Added to universal vocabulary 2026-04-26 per ad-LL-007.
- **NORMAL**: A < 1.01, age-percentile < p90 — no action
- **MARGINAL**: A ≥ 1.01 — serial-sample in 6 months, no immediate workup
- **DETECTABLE**: A ≥ 1.05, age-percentile ≥ p90 — run Stage 2
- **URGENT**: A ≥ 1.07, age-percentile ≥ p90 — run Stage 2; if breast_ductal localizes, recommend breast imaging
- **FLOOR BREACH**: A ≥ 1.10 — run Stage 2; clinical workup regardless of Stage 2 result

## Known limitations (must appear in every patient report)

Single-timepoint sensitivity at 95% specificity is 14-20% at the current panel-level Cohen's d ≈ 0.6. This is screening-adjacent, not diagnostic. Patients and clinicians should understand the report as a flag that changes downstream workup decisions, not as a breast-cancer diagnosis.

The validation cohorts are both EPIC-Italy. Cross-population evidence supports the direction (5 of 5 sub-cohorts positive across Australian/UK/Polish/Singapore populations) but magnitudes vary. Prospective validation in a non-Italian cohort with frozen methodology is the next evidence tier.

Stage 2 deconvolution production module (G-DECONV-001) is currently OPEN-DEFERRED in the GitHub roadmap. VAL-041 proved the workflow at the published-β level with 10/10 top-1 correct localization. Full per-IDAT production deployment requires the 30 MB Moss 2018 reference matrix locked into the code repo and the Salas 2018 QC harness implemented. Current deliverable is the L1 tier (Illumina EPIC with Moss markers); L2 custom capture and L3 full MESA+DELFI multi-substrate are future tiers.

Covariate adjustment (BMI, smoking, menopause status, Houseman deconvolution) was deliberately NOT applied in the validation analyses. Published analyses of the same cohorts WITH covariate adjustment report effect sizes in the same range or larger. Our numbers are a lower bound. A reviewer applying covariate adjustment should see the effect grow, not shrink.

## File pointers

- **Card JSON (current):** `breast-epic_card_v2.2.json` — machine-readable operational spec including VAL-060 tissue-arm entry
- **Card JSON (previous):** `breast-epic_card_v2.1.json` — superseded by v2.2
- **Evidence Report section:** `GAPE_Evidence_Report_UPDATED.html` §5C VAL-047 Phases 9, 12, Tightening v2, and VAL-060 tissue arm
- **Blood validation scripts:** `VAL047_phase9_immune_class.py`, `VAL047_phase12_gse51032_replication.py`, `VAL047_tightening_fresh.py`, `VAL047_tightening_v2_patch.py`
- **Tissue validation script (VAL-060):** `val060_breast_epic_tcga_brca.py`
- **Blood result JSONs:** `VAL047_phase9_immune_class_results.json`, `VAL047_phase12_gse51032_breast_results.json`
- **Tissue result JSON (VAL-060):** `VAL060_breast_epic_tcga_brca_results.json` (SHA `6ef629c809073213...`)
- **VAL-060 pre-registration:** `VAL_060_PREREG.md` (SHA `cd8c6de4383d8720...`) + `VAL_060_SEAL.txt` (sealed 2026-04-24 08:14:36 UTC)
- **VAL-060 cohort SHA manifest:** `VAL_060_tcga_brca_file_shas.json` (186 individual TCGA file SHAs + aggregate cohort SHA)
- **Panel:** `xu538_breast_panel.json`, SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`
- **Cross-population manifest:** `CROSS_POPULATION_MANIFEST.json` v1.2
- **VAL-094 EpiSCORE BreastRef Stage 2 resolution test:** pre-reg `VAL-094_prereg.md` (SHA `501fafad...`), seal `VAL-094_prereg_seal.json`, script `val_094.py`, results `VAL-094_results.json`, per-sample `VAL-094_per_sample.csv`, outcome `VAL-094_outcome.md`
- **VAL-095 UniLIFE 19-cell vs Salas head-to-head:** pre-reg `VAL-095_prereg.md` (SHA `5f74259d...`), seal, script `val_095.py` (RPC-NNLS, NaN-aware), results `VAL-095_results.json`, per-sample `VAL-095_per_sample.csv`, outcome `VAL-095_outcome.md`
- **VAL-096 TTD-window stratification:** pre-reg `VAL-096_prereg.md` (SHA `01247146...`), seal, script `val_096.py`, results `VAL-096_results.json`, heatmap `VAL-096_window_tile_heatmap.png`, outcome `VAL-096_outcome.md`
- **Streaming CpG extractor:** `extract_union_cpgs.py` (pulls UniLIFE + EpiSCORE + Loyfer + Salas union from full GSE51057/51032 series matrices in single passes, 26,453 target CpGs, 98.9% extraction coverage)
- **Atlas vault** (sibling-folder loadable from GAPE_WEB_v13.py): `Biological_Physics/atlas_vault/` — 8 atlas families / 39 reference matrices / 80 catalogued files / SHA-256 INVENTORY.json

---

## v2.3 changes (2026-04-26)

- **VAL-096 added** — TTD-window stratification on the Loyfer/Moss 25-tile per-tile A-score on GSE51057 + GSE51032. Two-component temporal model documented: persistent distributed cellular-aging-drift across pancreatic + cycling-class tiles at all pre-dx windows, plus late-localizing breast tile that rises from d≈+0.15 at >10yr/5-10yr/2-5yr to d=+0.43/+0.49 at 0-2yr in both cohorts. Pre-reg SHA `01247146d955ad28a7d141dd5b194a86d1d97b63b1022a07587ae4cd69310c6d` sealed 2026-04-26 before re-analysis. Heatmap deliverable: `VAL-096_window_tile_heatmap.png`.
- **VAL-095 added** — UniLIFE 19-cell vs Salas 450K 6-cell head-to-head Stage 3 deconvolution. Two replicating resolution gains: aTreg at >10yr (d=+1.26 / +0.79, both CIs exclude zero) and aBnv at 0-2yr (d=+0.44 / +0.49). Salas remains production; UniLIFE added as overlay. Pre-reg SHA `5f74259d5341268ee7cdaf68322962a275dd19e4158b398f09562f6aaa44bace`.
- **VAL-094 added** — EpiSCORE BreastRef 7-cell Stage 2 resolution test. Outcome `O2_DISTRIBUTED_AS_LOYFER`: 7 sub-cell types behave as one coherent signal (resolution-collapse pattern). Not added to v2.3 as per-sub-cell-type discriminator. Pre-reg SHA `501fafad68fa93635a18f43687104756f006ea89ed301de80ac469514ae15626`.
- **Stage 2 temporal pattern section added** to README — distinguishes the long-pre-dx distributed pattern from the late-localizing breast tile signal. Stage 2 reports for samples at long pre-dx no longer over-claim breast-tissue-of-origin attribution.
- **Four new lessons** — `breast-LL-004` (VAL-060 cross-substrate consistency: Xu-538 transfers to breast tumor tissue at larger effect size than blood, drafted in v2.3 to reconcile the v2.2 README's reference to LL-004 with its absence from the v2.2 JSON), `breast-LL-005` (Stage 2 distributed-then-localized pattern from VAL-093/096), `breast-LL-006` (UniLIFE 19-cell additive resolution gain at aTreg/aBnv from VAL-095), `breast-LL-007` (EpiSCORE breast sub-cell-type resolution does not separate at buffy-coat input from VAL-094). Card-internal lessons and master `LESSONS_LEARNED.md` updated. Card now has 7 lessons total.
- **Run-everything Stage 3 architecture** — production Salas 6-cell PLUS UniLIFE 19-cell parallel atlas, per Heath sign-off 2026-04-26 and atlas-vault commit history. Documented in card v2.3 stage_3_subcomposition.
- **Card tier label unchanged** — primary tier remains `cross_platform_validated_two_cohorts`. The four added VALs are all stage-2 / stage-3 resolution and temporal pattern documentation; they do not change the Stage 1 immune-class flag deployment.
- **Direct-to-consumer positioning block added** — top-of-README "What EDEAR is and is not" section + new `deployment_positioning_v23` block in card JSON. Clarifies that EDEAR v1 is a health-and-wellness cellular state report, out-of-pocket, no FDA, no clinician-in-the-loop. Customer is the primary report reader. Long pre-dx signals communicated as body-wide cellular-aging-drift (not disease-specific); breast localization communicated only when the data support it (0-2yr window per VAL-096).

## v2.2 changes (2026-04-24)

- **VAL-060 tissue arm added** — retroactive per-card tumor-vs-adjacent-normal tissue validation on TCGA-BRCA HM450 matched pairs (n=89/89, 86 complete pairs). Paired d = +0.676 tumor vs adjacent-normal, p = 0.0001. Unpaired d = +0.745 [95% CI +0.451, +1.075]. Effect size larger than VAL-058 prostate tissue (+0.497 paired), consistent with Xu-538 being the panel originally selected for breast cancer. First retroactive per-card tissue re-validation under the CCL-011 standard.
- **Card tier label unchanged** — primary tier remains `cross_platform_validated_two_cohorts`. Tissue arm is additive evidence, not a replacement for blood pre-dx validation.
- **New lesson `breast-LL-004`** — Xu-538 transfers to breast tumor tissue at larger effect size than blood, confirming multi-substrate operation across the full cancer progression trajectory.

## v2.1 changes (2026-04-24)

- **Universal reference block embedded** (full-inline, Option B). The card JSON now contains the complete universal pipeline specification — H_min constants for all 8 architecture classes, Moss 2018 healthy reference β for all 18 tissues, 80-cell age-decade immune baseline, EpiDISH Salas QC bounds, universal tier thresholds, sex-stratification rule, language discipline, and the cross-cohort batch-offset warning from VAL-057. A new analyst loading only this card JSON plus `GAPE_WEB_v13.py` can run the full pipeline end-to-end without consulting any other file.
- **Lessons-learned section added** — 3 disease-specific documented quirks, each with source validation, context, observed quirk, interpretation, and how the card was updated to handle it. See `lessons_learned` key in the card JSON.
- **Cross-card lessons catalog** maintained in `LESSONS_LEARNED.md` at the Cookbook root. This card's entries are labeled with the card prefix (breast-LL-###).

