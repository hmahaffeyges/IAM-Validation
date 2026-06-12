# HCC-EPIC Card — EDEAR Cohort-Screening + Tissue-Validated Hepatocellular Carcinoma Flag

**Version 0.3 · 2026-04-28**
**Validation tier:** `multi_modal_validated` (UNCHANGED from v0.2 — VAL-101 sealed at O5_DATA_INTEGRITY_FLAG; biology does NOT propagate; card stays at the v0.2 tier)
**Supersedes:** v0.2 (2026-04-24, ccfDNA + tissue arm via VAL-059 + VAL-064, multi_modal_validated tier). v0.3 adds VAL-101 sealed-at-O5 event documentation + VAL-102 voided audit-trail + CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION + the honest-path-forward calibration-VAL or CCL-040 deferral pathways. **Validation tier UNCHANGED at multi_modal_validated.** VAL-101 biological readouts are descriptive supplementary documentation only and do NOT promote the card; the card is updated to record the cookbook state, not to claim new biology.

## What HCC is

Hepatocellular carcinoma — the most common form of primary liver cancer. Arises from hepatocytes. Major causes: chronic hepatitis B, chronic hepatitis C, alcoholic cirrhosis, metabolic liver disease (NAFLD/NASH). HCC is the sixth most common cancer globally and the third-leading cause of cancer mortality. Early detection is limited by low sensitivity of the standard biomarker (alpha-fetoprotein) and by cirrhosis itself producing methylation drift that can mask the cancer-specific signal.

## Clinical claim

A plasma ccfDNA sample that produces a Stage 1 immune-class A-score elevation at DETECTABLE tier or higher AND whose Stage 2 Moss NNLS deconvolution localizes the top-1 tissue to `hepatocyte` is flagged as consistent with architectural drift in hepatocytes. Card v0.1 is anchored to VAL-059 cross-cohort validation on GSE298812 (Nigerian HIV+ HCC ccfDNA, n=245) with a dose-response signal across fibrosis (d = +0.44), cirrhosis (d = +0.45), and HCC (d = +0.63).

## CRITICAL SUBSTRATE RESTRICTION

**This card is validated ONLY for ccfDNA plasma. Whole-blood leukocyte DNA is NOT a validated substrate for HCC.**

VAL-059 tested both substrates:

| Cohort | Substrate | n | Primary d | p | Outcome |
|---|---|---|---|---|---|
| GSE298812 (Nigerian HIV+ HCC) | ccfDNA plasma | 245 | **+0.634** [+0.175, +1.121] | 0.0024 | PASSED (HCC-Pos vs HCC-Neg) |
| GSE281691 (Metabolic HCC multicenter) | whole-blood leukocyte | 481 | **−0.156** [−0.337, +0.027] | 0.09 | NULL |

Xu-538 on ccfDNA separated HCC from disease-spectrum controls; Xu-538 on whole-blood leukocyte against metabolic-liver-disease controls did not. The substrate-specific divergence is the primary finding of v0.1. EDEAR reports using this card MUST specify ccfDNA plasma as the specimen.

## What this card covers, and what it does not

**What v0.1 covers.** On ccfDNA plasma EPIC 850K data, the Xu-538 pooled-entropy A-score separates HCC cases from HCC-free controls with a dose-response trend consistent with disease progression (healthy < fibrosis ≈ cirrhosis < HCC). The architectural drift is detectable at clinically meaningful effect sizes.

**What v0.1 does NOT cover — with the specific limits.**

1. **HIV-HCC interaction confound.** GSE298812 is 100% HIV-positive. Both HCC cases and HCC-free controls are HIV+, so HIV is not the case-control differentiator. BUT the Xu-538 panel was trained on non-HIV Sister Study breast cancer immune drift, and an HIV+ immune system at steady state has methylation signatures distinct from non-HIV. With GSE298812 alone we cannot distinguish (a) HCC-specific immune drift that Xu-538 happens to capture in HIV+ people, from (b) HIV-HCC interaction signal that moves Xu-538 in the case direction for reasons other than pure HCC. The dose-response across fibrosis / cirrhosis / HCC supports disease-specificity but does not rule out HIV-interaction modulation of effect size. **Until a non-HIV HCC ccfDNA cohort replicates the d ≈ +0.6 finding, this card should NOT be claimed as pre-diagnostic screening for HIV-negative populations.** This is priority 1 in `next_validation_steps`.

2. **Whole-blood leukocyte substrate does not work.** Clearly documented above. Any EDEAR workflow calling this card on a whole-blood-leukocyte sample will produce a false-negative or uninterpretable result.

3. **HCC cannot be discriminated from advanced chronic liver disease at moderate signal.** Fibrosis d = +0.44, cirrhosis d = +0.45, HCC d = +0.63. The distributions overlap. A Stage 1 elevation with Stage 2 hepatocyte localization in a patient with known cirrhosis does NOT specifically indicate HCC — it indicates hepatocyte architectural drift, which can also reflect cirrhosis progression. AASLD-standard HCC surveillance (AFP + abdominal ultrasound every 6 months) is required in cirrhotic patients regardless of EDEAR output.

4. **Sex imbalance in primary cohort.** GSE298812 is 161 Female / 84 Male — unusual for HCC (typically 3:1 male). Male-stratified d = +0.998, female-stratified d = +0.497. Card performance by sex is not robust at v0.1 given the n=31 HCC-Pos case pool.

5. **n=31 HCC-Pos cases.** The primary case arm is small. The wide CI [+0.175, +1.121] reflects this. A replication cohort would narrow uncertainty. No directional-panel variant derived yet (analogous to VAL-051 AD 7-CpG Rule A panel) — if the pooled-entropy null on GSE281691 leukocyte is a Directional-Score-Principle scenario rather than a pure substrate effect, a directional panel on leukocyte might rescue that specimen.

## The workflow in one patient

A 56-year-old man with chronic hepatitis B submits a plasma ccfDNA sample for EDEAR analysis. The lab runs an Illumina EPIC 850K array on the ccfDNA.

**Stage 1 (universal).** Xu-538 CpGs extracted, pooled-entropy A-score computed against H_min(immune) = 0.838889, compared to age-matched 80-cell healthy baseline (50–59 decade). Tier call assigned. Note: the 80-cell blood immune baseline is applicable to ccfDNA immune-class signal because plasma ccfDNA contains substantial leukocyte-turnover contribution in healthy subjects; HCC ccfDNA shifts immune-class signal through tumor and hepatocyte contributions layered on the leukocyte baseline.

**Stage 2 (if Stage 1 hits DETECTABLE or above).** Moss 2018 NNLS deconvolution produces an 18-tissue β vector. Top-1 localization is identified. If top-1 = `hepatocyte` (secretory class H_min = 0.843264, healthy reference β = 0.742, VAL-041 HCC case β ≈ 0.598), this hcc-epic card fires.

**Report.** The patient's clinician receives:
- A-score tier call and age-matched percentile
- Stage 2 top-3 tissue localization table with `hepatocyte` ΔA highlighted
- Stage 2 confidence indicator (top-1 / top-2 ΔA ratio)
- Explicit note: ccfDNA specimen verified; card does NOT fire on whole-blood leukocyte input
- Explicit note if patient has known cirrhosis: Stage 2 hepatocyte localization cannot discriminate HCC from cirrhosis progression at this signal level; standard AASLD surveillance (AFP + ultrasound q6mo) is the clinical action
- Assay version tag (L1 Illumina EPIC + Moss markers / L2 custom capture / L3 full MESA+DELFI)
- Clinical action: AFP + abdominal multi-phase contrast MRI or CT per AASLD HCC surveillance and diagnostic guidelines; hepatology consultation

## Validation summary

| Anchor | Cohort | Substrate | n | Primary result | Tier contribution |
|---|---|---|---|---|---|
| VAL-059 primary | GSE298812 Nigerian HIV+ HCC | ccfDNA plasma | 245 (31 HCC-Pos / 115 HCC-Neg) | d = +0.634 [+0.175, +1.121], p = 0.002 | PASSED ccfDNA substrate |
| VAL-059 replication | GSE281691 Metabolic HCC multicenter | whole-blood leukocyte | 481 (221 Case / 260 Control) | d = −0.156 [−0.337, +0.027], p = 0.09 | NULL on leukocyte substrate |
| GSE298812 spectrum | Nigerian HIV+ HCC | ccfDNA plasma | 245 (4 groups) | HCC-Neg d = 0 · Fib d = +0.44 · Cir d = +0.45 · HCC d = +0.63 | Dose-response across liver disease progression |
| VAL-041 Stage 2 localization | Moss 2018 Fig 4 + Liu 2020 hepatocyte | ccfDNA (aggregate) | — | β_hepatocyte ≈ 0.598 (Δβ = −0.144) | Stage 2 mechanism reference |

**Total VAL-059 primary pass:** O3 — SINGLE_COHORT_VALIDATED_CCFDNA (substrate-restricted).

## Sources

**VAL-059 primary cohort (GSE298812).** Soliman AS et al. *Circulating cell-free DNA methylation biomarkers for hepatocellular carcinoma risk prediction in HIV-positive Nigerian population.* GEO: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE298812>. Published ccfDNAmRF classifier AUC 92-97%.

**VAL-059 replication cohort (GSE281691).** Multicenter international metabolic HCC study. GEO: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE281691>. Published 55-CpG panel classifier AUC 0.79 on the source study's own panel (not Xu-538).

**VAL-041 Stage 2 reference.** Moss J et al. *Comprehensive human cell-type methylation atlas reveals origins of circulating cell-free DNA in health and disease.* Nat Commun 2018; 9:5068. DOI: <https://doi.org/10.1038/s41467-018-07466-6>. Liu et al 2020 Ann Oncol hepatocyte Table S3.

**Xu-538 panel origin.** Xu Z, Sandler DP, Taylor JA. *Blood DNA methylation and breast cancer: a prospective case-cohort analysis in the Sister Study.* J Natl Cancer Inst 2020; 112(1):87-94. DOI: <https://doi.org/10.1093/jnci/djz065>.

## Pre-registration chain

- `VAL_059_PREREG.md` SHA-256: `f06fcd3fc91ae0ca9f212f029577357dfc69abc31e9fd97cc1c67a6f5aae4c90` sealed 2026-04-24 06:50:36 UTC
- `VAL_059_PREREG_AMENDMENT.md` SHA-256: `b669a4c87db545b054e8f8aa87cdab30e22130a4cf072fc00ae767a9c89b3191`

**On the amendment and Moss per-CpG public vs proprietary status.** Same as VAL-058: Moss 2018 marker CpG list and reference matrix R are public (Moss 2018 Supplementary Table S4 and `nloyfer/meth_atlas`). The H_min calibration layer is proprietary (US Provisional Patents 64/012,720 and 64/014,568). A future VAL-059b run using the public Moss S4 marker CpGs with `scipy.optimize.nnls` would add per-tissue β estimates on top of the Xu-538 ccfDNA signal documented here. See `hcc-LL-005`.

- GSE298812 β matrix SHA: `4a586138987065a70f473f9d97d7e36646829371f662ffe052146a5232edd981`
- GSE281691 β matrix SHA: `5ce39843c1a2cdf20db0c73d64be976be7c3820b19a3d95cb9003245a2b6e11f`
- Xu-538 panel SHA: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`

## Known limitations

See `hcc-epic_card_v0.1.json` → `known_limitations`. Summary: ccfDNA-substrate-only, HIV-HCC interaction confound in primary cohort, small case pool (n=31) with wide CI, cannot discriminate HCC from cirrhosis at moderate signal, unusual sex ratio, no directional-panel variant derived yet.

## Lessons learned

See card JSON `lessons_learned` section — entries hcc-LL-001 through hcc-LL-005 covering substrate-specific panel transferability, dose-response across disease spectrum, HIV-confound epistemics, cross-substrate null interpretation, and the public-Moss-vs-proprietary-H_min boundary.

## Next validation steps (priority-ordered)

1. **Non-HIV HCC ccfDNA cohort replication is the single highest priority** — without it, the HCC-specific vs HIV-HCC-interaction signal cannot be separated.
2. Directional-panel variant derivation on GSE298812 (analogous to VAL-051 methodology).
3. Direct comparison of Xu-538 pooled entropy vs directional-panel approach.
4. Treatment-response monitoring validation (post-resection or post-ablation HCC patients with serial ccfDNA).
5. dbGaP applications: UK Biobank, Rotterdam Study, NIH HCC biorepository (12-week typical approval cycle).
6. Underlying liver disease stratification (HBV/HCV/alcoholic/metabolic HCC subtype effect sizes).

---

## Tissue arm — VAL-064 (added v0.2)

The tissue arm of hcc-epic uses TCGA-LIHC HM450 matched tumor/adjacent-normal biopsies, secretory-class scoring. **LIHC = Liver Hepatocellular Carcinoma** — the TCGA project code for the HCC cohort. Hepatocyte = secretory class (H_min = 0.843264, Moss 2018 healthy reference β = 0.742 from the universal_reference block). Hepatocyte is one of 6 secretory-class TCGA cancers in Issue 002 alongside breast_ductal, prostate_epithelial, pancreatic_exocrine.

### VAL-064 — Primary: HCC tumor architecture (secretory-class scoring), pooled cohort

- **n matched pairs:** 46 (50 candidates, 4 skipped at QC ≥400,000 valid β)
- **Paired Cohen's d:** +0.4975, 95% CI [+0.1911, +0.8038], p = 7.41e-04
- **Unpaired Cohen's d:** +0.6595, 95% CI [+0.2399, +1.0792], p = 1.56e-03
- **Absolute ΔA:** +0.01960
- **Outcome:** PASS at the prereg-sealed threshold

The pooled result PASSES but sits at the lower end of the secretory-class precedent range (VAL-058 prostate +0.50, VAL-060 breast +0.68). The reason becomes apparent in the etiology-stratified analysis below.

### Risk-factor stratification — the viral-hepatitis blunting finding

| Stratum | n | Paired d | 95% CI | p |
|---|---|---|---|---|
| **Non-viral HCC (alcohol/NAFLD/none combined)** | **34** | **+0.6640** | [+0.2927, +1.0354] | **1.08e-04** |
| **Viral hepatitis (HBV+HCV combined)** | **12** | **+0.0231** | [-0.5428, +0.5890] | **0.94 (NULL)** |
| Alcohol+ | 10 | +0.8667 | [+0.1398, +1.5937] | 6.13e-03 |
| HBV+ alone | 7 | +0.0482 | [-0.6930, +0.7894] | 0.90 |
| HCV+ alone | 5 | -0.0464 | [-0.9234, +0.8306] | 0.92 |
| NAFLD | 3 | +0.9320 | [-0.4232, +2.2873] | 0.11 (underpowered) |
| No documented risk | 19 | +0.6166 | [+0.1261, +1.1071] | 7.20e-03 |

**The non-viral arm (n=34, d=+0.66) sits squarely in the secretory-class tissue-arm range alongside VAL-060 breast (+0.68) — classical secretory-class behavior.**

**The viral hepatitis arm (n=12, d=+0.02) is essentially null at the paired-tissue level.**

### Mechanism — framework-consistent, literature-anchored

Chronic HBV/HCV infection drives extensive methylation drift in adjacent-normal liver tissue (the "field defect" documented by Villanueva 2015 Hepatology, already a v0.1 anchor). This raises the adjacent-normal A-score baseline above true-healthy, shrinking the paired tumor-vs-adjacent-normal contrast even though the tumor architecture is genuinely disrupted. Non-viral HCC adjacent-normal is closer to true-healthy baseline, so the paired contrast is preserved at full magnitude.

This is consistent with v0.1's already-documented limitation #3: "HCC cannot be discriminated from advanced chronic liver disease at moderate signal." Viral hepatitis is exactly the chronic liver disease scenario that limitation describes, now with quantitative confirmation at the paired-tissue level.

### Important nuance — what this DOES NOT mean

This finding does NOT mean EDEAR cannot detect viral HCC:

1. **Paired tumor-vs-adjacent-normal tissue contrast** is blunted in viral HCC because the adjacent-normal baseline is already shifted up
2. **Unpaired analysis vs healthy non-cirrhotic controls** would still show elevated A-score in viral HCC tissue (both tumor AND adjacent-normal are elevated above true-healthy)
3. **ccfDNA plasma analysis (VAL-059 primary)** detected viral HCC at d=+0.634 in GSE298812 (HIV+ HBV cohort) — the ccfDNA arm captures viral HCC successfully

The blunting is specific to paired tissue contrast, not to overall detectability.

### Fibrosis stratification

| Stratum | n | Paired d | 95% CI | p |
|---|---|---|---|---|
| No fibrosis (Ishak 0) | 21 | +0.5816 | [+0.1191, +1.0440] | 7.70e-03 |
| Any fibrosis (Ishak ≥1) | 33 | +0.5898 | [+0.2201, +0.9594] | 7.04e-04 |

Surprisingly little difference between fibrosis strata. The viral-vs-non-viral distinction does more analytical work in this cohort than the fibrosis-vs-no-fibrosis distinction — viral infection drives methylation drift through mechanisms partly independent of histologic fibrosis grade (HBx oncoprotein effects on DNMT3A/3B, integration-driven effects, immune-mediated methylation independent of fibrosis).

### Stage and gender stratification

| Stratum | n | Paired d | p |
|---|---|---|---|
| Stage I | 21 | +0.4362 | 4.56e-02 |
| Stage II+ | 17 | +0.5496 | 2.35e-02 |
| Male | 27 | +0.4134 | 3.17e-02 |
| Female | 19 | +0.6163 | 7.22e-03 |

Modest stage gradient (Stage II+ > Stage I, consistent with progressive disruption). Female stratum shows somewhat larger contrast than male — notable because TCGA-LIHC is 60% male and gender is a real biological variable in HCC. The universal_sex_stratification_rule from v0.1 already mandates sex stratification for deployment; VAL-064 confirms this guidance is appropriate for hcc-epic.

### Comparison to all Cookbook tissue arms

| Card | Class | Cohort | Paired d (full pooled) |
|---|---|---|---|
| VAL-058 prostate | secretory | n=238 | +0.497 |
| VAL-060 breast | secretory | n=86 | +0.675 |
| VAL-062 CRC | cycling | n=26 | +0.724 |
| VAL-063 lung | cycling | n=29 | +1.020 |
| **VAL-064 HCC (pooled)** | **secretory** | **n=46** | **+0.498** |
| **VAL-064 HCC (non-viral arm)** | **secretory** | **n=34** | **+0.664** |

VAL-064 non-viral HCC is comparable in magnitude to the prior secretory-class tissue arm (VAL-060 breast +0.68). The pooled result is dragged down by the viral-hepatitis-driven adjacent-normal field defect.

### Reproduction bundle

- **Pre-registration:** [`VAL-064_prereg.md`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-064_prereg.md) (SHA `a03f2c2c...`, sealed BEFORE any β-value access)
- **Outcome:** [`VAL-064_outcome.md`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-064_outcome.md)
- **Reproducible script:** [`val064_hcc_epic_tcga_lihc.py`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/val064_hcc_epic_tcga_lihc.py) (Python 3 stdlib only)
- **Primary results:** [`VAL-064_results.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-064_results.json)
- **Stratified results:** [`VAL-064_stratified.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-064_stratified.json)
- **Cohort manifest:** [`LIHC_matched_manifest.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/LIHC_matched_manifest.json)
- **Clinical metadata:** [`LIHC_clinical.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/LIHC_clinical.json)
- **Manifest SHA:** `760bf65a213da5a86cdd7ecde6ff6d46dad04777eafc00eb14f56356b0088371`
- **Cohort SHA:** `78ccc7fecc9a8995b95d4f7ab1ecaaa2d431427dfa05bbda808aaacf31b565e4`

### Tissue arm validation status

The hcc-epic card now has independent validation across two arms: ccfDNA plasma (VAL-059, v0.1) and tissue tumor architecture (VAL-064, v0.2). Both arms PASS. The tissue arm reveals a strong etiology-dependent pattern that complements rather than contradicts the v0.1 ccfDNA finding: ccfDNA detects viral HCC successfully (VAL-059 d=+0.63 on HIV+HBV cohort), but paired tumor-vs-adjacent-normal tissue contrast is blunted in viral HCC due to chronic-infection-driven adjacent-normal field defect. These are complementary findings, not a contradiction.

### Honest framing for clinical interpretation

The v0.1 limitation #3 ("HCC cannot be discriminated from advanced chronic liver disease at moderate signal") is now further qualified: in viral hepatitis-driven HCC, even the tissue tumor-vs-adjacent-normal paired contrast is blunted because viral infection itself disrupts adjacent-normal methylation. Clinical deployment must continue to defer to AASLD-standard HCC surveillance (AFP + abdominal ultrasound q6mo) in cirrhotic patients regardless of EDEAR output, and now also in patients with active viral hepatitis.

### v0.2 changes (2026-04-24)

- **Tissue arm added.** New section "Tissue arm — VAL-064" documents TCGA-LIHC HM450 matched tumor/normal validation:
  - **VAL-064 primary — HCC tumor architecture, secretory-class scoring.** Paired d = +0.4975 [+0.191, +0.804], p = 7.41e-04. PASS at threshold.
  - **VAL-064 risk-factor stratified.** Non-viral HCC (n=34) paired d = +0.664 [+0.293, +1.035], p = 1.08e-04. Viral hepatitis (n=12) paired d = +0.023 [-0.543, +0.589] NULL. Direction-magnitude pattern reveals the viral-hepatitis adjacent-normal field-defect blunting mechanism (Villanueva 2015).
- **LIHC term defined inline.** LIHC = Liver Hepatocellular Carcinoma (TCGA project code).
- **Validation tier progression.** The hcc-epic card now has independent validation across two arms: ccfDNA plasma (VAL-059, v0.1) and tissue tumor architecture (VAL-064, v0.2).
- **Tier promoted from `cohort_screening_validated` (substrate-restricted ccfDNA only) to `multi_modal_validated`** — both ccfDNA and tissue arms validated.
- **Cross-card tissue-arm ordering.** d(lung)=+1.02 > d(CRC)=+0.72 > d(breast)=+0.68 ≈ d(HCC non-viral)=+0.66 > d(prostate)=+0.50 > d(HCC pooled)=+0.50.
- **Cross-card observation.** Two cards now show chronic-driver-exposure adjacent-normal field defects: lung-epic (smoking) and hcc-epic (viral hepatitis). Both blunt paired tissue contrast in the affected stratum. Pattern may generalize to other chronic-driver pairings (HPV-cervical, H. pylori-gastric, UV-skin, schistosoma-bladder).

---

## Tissue arm — VAL-101 (added v0.3, sealed at O5_DATA_INTEGRITY_FLAG, biology does NOT propagate)

VAL-101 is the run-everything Loyfer 25-tile per-class A-score with full etiology stratification on the same TCGA-LIHC HM450 paired cohort that anchored VAL-064 (n=46 QC-passed). Methodology mirrors VAL-098 / VAL-099. Three pre-locked questions: (1) CCL-039 cross-tissue generalization to HCC at the per-tile level; (2) viral-vs-non-viral blunting at the per-tile level vs the pooled-cycling-class level (where VAL-064 documented it); (3) Marcus-analog stratum (No_documented_risk patients, n=10) characterization.

### Personal motivation logged in card

This VAL was scoped because Heath's stepbrother Marcus died of an aggressive HCC without a documented chronic-driver risk factor. Marcus had liver transplant, the tumor returned in the new liver within months, and he died sedated and never said goodbye to his wife and three kids. The Marcus-analog stratum (No_documented_risk patients, n=10) was the closest available public analog for the kind of cancer Marcus had — and the proximate reason VAL-101 was scoped as a 25-tile etiology stratification rather than waiting for additional cohort acquisition. The motivation is logged here to preserve institutional memory of why the analytical extension was scoped, not as a claim that VAL-101 answered the question.

### VAL-101 — Sealed at O5_DATA_INTEGRITY_FLAG (2026-04-28)

- **Pre-registration SHA:** `fa366bf00316597bb65032b747029133acb5f1bbb40f6251094b563732185512`
- **Sealed at:** 2026-04-28T19:53:19Z (before β-value access)
- **RNG seed:** 20260428
- **Cohort:** TCGA-LIHC HM450 paired tumor/adjacent-normal — 50 candidate paired pairs (mirrors VAL-064 manifest), n=46 QC-passed (4 dropouts at QC ≥400,000 valid β per sample, mirrors VAL-064 sealed cohort exactly)
- **Methodology:** Run-everything Loyfer 25-tile per-class A-score, top-100 marker CpGs per tile, paired Cohen's d on (tumor − normal) per tile, bootstrap 10,000-iteration CI
- **Outcome:** **`O5_DATA_INTEGRITY_FLAG`** — pre-locked CHK-3.1 thresholds (extreme >30%, the raw-EPIC default from VAL-100 prereg) tripped on TCGA-LIHC HM450 sesame Level 3 at extreme **26.6%** / middle **9.1%**. Per CCL-032 diagnostic order (data integrity → biology → framework), cookbook discipline honors the trip.

### Why O5 stands

The pre-registration was sealed before β-access. The threshold tripped. CHK-4.8 honest-revision is reserved for structurally degenerate criteria (cf VAL-097 O2_CYCLING_DISTRIBUTED), not for misspecified thresholds. The biology being clean does not justify post-hoc threshold relaxation; that is exactly the failure mode prereg discipline is designed to prevent.

### Biological readouts (descriptive supplementary documentation only — DO NOT propagate)

These numbers are real but DO NOT propagate to card claims, validation tier, or Stage 2 mechanism narrative. They are documented here for institutional memory and for future cross-validation against any cohort that becomes available under a properly pre-registered platform threshold.

| Stratum | n | Hepatocytes paired d (descriptive) | 95% CI (descriptive) |
|---|---|---|---|
| Pooled (all QC-passed) | 46 | −1.521 | [−2.192, −1.182] |
| All_viral (HBV+HCV+co-infection) | 24 | −1.726 | [−3.025, −1.117] |
| All_non_viral (Alcohol+NAFLD+Other+No_documented_risk) | 22 | −1.393 | [−2.301, −1.064] |
| HBV+ alone | 19 | −2.036 | [−3.185, −1.499] |
| Alcohol+ | 6 | −1.681 | [−7.273, −0.774] |
| **No_documented_risk (Marcus-analog)** | **10** | **−1.141** | **[−6.157, −0.847]** |

Strata with n<5 (HCV+ alone, HBV+HCV co-infection, NAFLD+, Other) omitted per CHK-2.7; full table available in `Biological_Physics/validation_runs/VAL-101/stratified.json`.

The Hepatocytes tile is the third-largest |d| in the pooled 25-tile output. The two largest-positive tiles are Colon_epithelial_cells (+1.807) and Head_and_neck_larynx (+1.585). This is the tile-pattern signature CCL-039 predicts for tumor-vs-adjacent-normal paired comparisons (cell-of-origin tile fidelity-loss in tumor; non-cell-of-origin marker CpGs drift toward homogenized tumor methylation). **But none of this propagates** because of the CHK-3.1 trip.

### Would-be findings if biology propagated (but does NOT)

If VAL-101 had passed CHK-3.1 under a properly pre-registered platform threshold, the descriptive readouts would have established three findings:

1. **CCL-039 cross-tissue generalization to HCC.** Cell-of-origin tile fidelity-loss pattern that is currently colorectal-only at three cohorts (TCGA-READ, TCGA-COAD revisit, TCGA-COAD VAL-099 reproduction) would extend to a second cancer type.
2. **Viral-vs-non-viral blunting refinement.** VAL-064 documented blunting at pooled cycling-class viral d = +0.023 NULL. Descriptively, viral d = −1.726 and non-viral d = −1.393 at the per-tile cell-of-origin level — the blunting in VAL-064 would be specific to global cycling-class methodology, not present at the per-tile level.
3. **Marcus-analog stratum tile signature.** No_documented_risk stratum descriptively shows the same architectural-disruption signature as the risk-factor strata (Hepatocytes tile most-negative in panel, comparable magnitude to alcohol+ stratum).

**ALL THREE OF THESE DO NOT PROPAGATE.** The proper inferential pathway is one of the two paths in the next section.

### Self-correction logged: VAL-102 voided before execution

A VAL-102 prereg was sealed at 2026-04-28T20:31:23Z with a TCGA HM450 platform threshold (extreme >20%) derived from VAL-101's tripped data. The intent was to "do the prereg right the second time" by re-running the same methodology on the same cohort under a platform-tuned threshold. This was identified within minutes as post-hoc threshold accommodation with a SHA stamp — the threshold was selected to accommodate values already observed in VAL-101 (extreme 26.6%) plus values observed in retroactive VAL-099 verification (extreme 24.4%). Sealing such a threshold and applying it to the same cohort is circular reasoning, not pre-registration.

VAL-102 was voided at 2026-04-28T20:35Z, before any execution. Audit trail preserved at `Biological_Physics/validation_runs/VAL-102/VOIDED_BEFORE_EXECUTION.md` with the original SHA `2b77ad9d3b69554a0658260756db0f08722e2be3fa96eb48aad9213974f4717c`. The cookbook does not delete sealed records; it marks them and explains. The void event is logged as a positive cookbook signal — institutional memory of prereg discipline operating against the same-day temptation to bend the protocol when the biology looks like it might say something that matters to a person.

### CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION

CCL-041 was formalized in the cookbook from VAL-101's prereg-trip + post-hoc verification on cached TCGA-COAD HM450 sesame Level 3 data (the VAL-099 cohort) which reads extreme 24.4% / middle 9.7% on the same check methodology.

**Lesson:** CHK-3.1 thresholds need platform-specific tuning. Raw EPIC β reads sharper bimodality than TCGA HM450 sesame Level 3 due to standard TCGA pipeline dye bias correction.

**Distinct from CCL-040:** CCL-040 covers PROCESSED OUTPUT that loses bimodal raw β signature entirely (e.g., minfi noob-bg-corrected, residual M-values — the kind that triggered VAL-100 deferral). CCL-041 is about raw-β bimodality manifesting at slightly different threshold values across raw-β platforms. Two distinct concerns.

**Operational rule:** The threshold for any new platform MUST be set by a calibration VAL on a structurally-separated cohort, NOT by retroactive accommodation. The TCGA HM450 sesame Level 3 platform threshold value itself remains TBD pending a future calibration VAL.

### Honest path forward to propagate the biology

The biological readouts in VAL-101 results.json are real numbers. Their inferential validation pathway requires either:

1. **Calibration-VAL path.** Run a calibration VAL on TCGA samples from a tissue NOT under active hcc-epic test (TCGA-KIRC adjacent-normal, TCGA-PRAD adjacent-normal, etc.) to establish the TCGA HM450 sesame Level 3 platform threshold. Measure the bimodality distribution. Set the threshold from THAT distribution. Seal it. Then run a future hcc-epic VAL on TCGA-LIHC under the pre-registered platform threshold.

2. **CCL-040 deferral path.** Process the TCGA-LIHC .idat files through sesame from raw IDAT input. Verify bimodality at standard pipeline output. Re-run hcc-epic test on reprocessed betas. Same precedent as VAL-100 deferral.

Both paths are multi-VAL workstreams. Neither is a same-day re-seal. Both are honest.

### Reproduction bundle

- **Pre-registration:** [`VAL-101/prereg.md`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-101/prereg.md) (SHA `fa366bf00316597b...`, sealed before β-value access)
- **Outcome:** [`VAL-101/outcome.md`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-101/outcome.md)
- **Reproducible script:** [`val_101.py`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-101/val_101.py) (Python 3 stdlib only)
- **Primary results (sealed at O5; biology descriptive supplementary):** [`results.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-101/results.json)
- **Stratified results:** [`stratified.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-101/stratified.json)
- **Per-sample data:** [`per_sample.csv`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-101/per_sample.csv)
- **VAL-102 voided audit trail:** [`VAL-102/VOIDED_BEFORE_EXECUTION.md`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-102/VOIDED_BEFORE_EXECUTION.md)

### EDEAR commercial deployment unaffected

Per CCL-037, VAL-101 + VAL-102-voided + CCL-041 are retrospective cookbook validation activity with no impact on EDEAR commercial deployment. Deployment uses single-pipeline patient-vs-internal-reference architecture that is structurally insulated from public-data CHK-3.1 calibration questions. The CHK-3.1 platform-tuning question lives in the retrospective cookbook validation layer only.

### v0.3 changes (2026-04-28)

- **Card promotion:** NONE. Validation tier remains `multi_modal_validated` (the v0.2 tier). VAL-101 does NOT promote the card. v0.3 is a documentation update recording the cookbook state, not a validation tier promotion.
- **Tissue arm — VAL-101 sealed at O5_DATA_INTEGRITY_FLAG (2026-04-28).** New section above documents the run-everything Loyfer 25-tile per-class A-score on the same TCGA-LIHC HM450 paired cohort that anchored VAL-064 (n=46 QC-passed). Pre-registered SHA `fa366bf00316597b...` sealed before β-access. Pre-locked CHK-3.1 raw-EPIC threshold (extreme >30%) tripped on TCGA-LIHC HM450 sesame Level 3 at extreme 26.6% / middle 9.1%. Outcome `O5_DATA_INTEGRITY_FLAG`. Biological readouts (Hepatocytes tile pooled d=−1.521; All_viral d=−1.726; All_non_viral d=−1.393; HBV+ alone d=−2.036; Marcus-analog d=−1.141) are descriptive supplementary documentation only and do NOT propagate.
- **Personal motivation logged in card.** Marcus's case is the documented motivation for why VAL-101 was scoped as a 25-tile etiology stratification rather than waiting for additional cohort acquisition.
- **VAL-102 voided before execution — self-correction logged.** A VAL-102 prereg was sealed at 2026-04-28T20:31:23Z with a TCGA HM450 platform threshold (extreme >20%) derived from VAL-101's tripped data. Identified within minutes as post-hoc threshold accommodation with a SHA stamp. Voided at 2026-04-28T20:35Z, before any execution. Audit trail preserved per cookbook discipline (the cookbook does not delete sealed records).
- **CCL-041 LL-CHK-3.1-PLATFORM-CALIBRATION formalized.** New cookbook lesson logged in master `LESSONS_LEARNED.md`. Distinct from CCL-040: CCL-040 covers processed/normalized output that loses bimodality entirely; CCL-041 is about raw-β bimodality at slightly different threshold values across raw-β platforms. The TCGA HM450 platform threshold value itself remains TBD pending a future calibration VAL.
- **Three new lessons added to card:** hcc-LL-006 (prereg discipline does not bend even when biology looks exciting), hcc-LL-007 (CHK-3.1 thresholds platform-specific per CCL-041), hcc-LL-008 (setting a platform threshold from data being interpreted under it is circular reasoning — VAL-102 voided as institutional memory).
- **Two new known limitations added to card:** hcc-LIM-008 (TCGA HM450 platform CHK-3.1 threshold TBD pending calibration VAL), hcc-LIM-009 (no documented per-tile cell-of-origin validation for hcc-epic at v0.3 — Stage 2 continues to use Moss NNLS hepatocyte localization).
- **Two new next_validation_steps inserted at priority 1.5 and 1.6:** hcc-NVS-008 (calibration VAL on structurally-separated cohort to establish TCGA HM450 platform threshold), hcc-NVS-009 (CCL-040 raw-IDAT deferral pathway). Original priority 1 (non-HIV HCC ccfDNA cohort replication) preserved.
- **universal_pipeline_acknowledgment block extended** with `platform_chk_3_1_calibration_per_CCL_041` sub-block in the card JSON — documents the platform-specific CHK-3.1 threshold rule as part of the universal pipeline contract this card commits to.
- **Stage 2 mechanism block UNCHANGED.** Moss NNLS hepatocyte localization (top-1 hepatocyte tissue-of-origin) remains the production Stage 2 method. The run-everything 25-tile per-class A-score has been computed on the hcc-epic tissue arm cohort (VAL-101) but the result is descriptive supplementary only pending the CHK-3.1 platform calibration. Stage 2 reporting in v0.3 should not yet describe the tile-level cell-of-origin pattern for HCC.
- **Validation tier UNCHANGED at `multi_modal_validated`.** Per the cookbook discipline, VAL-101 sealed at O5 cannot promote the card. The hcc-epic tier promotion pathway from `multi_modal_validated` requires the calibration-VAL or CCL-040 deferral pathway to land an inferential pass under a properly pre-registered platform threshold.
- **EDEAR commercial deployment unaffected** per CCL-037.

### What is NOT in v0.3

- No promotion of validation tier (stays at `multi_modal_validated`)
- No update to Stage 2 mechanism block (Moss NNLS hepatocyte localization remains production Stage 2 method)
- No update to clinical_action_matrix
- No update to clinical claims about per-tile cell-of-origin tile signals (the descriptive readouts are not allowed to propagate to claims)
- No update to validation_evidence_summary headline numbers (those remain anchored on VAL-059 + VAL-064)
- No update to viral_etiology_stratification_VAL_064 block (the VAL-101 descriptive viral-vs-non-viral refinement does not propagate to the existing VAL-064 stratification block at the pooled-cycling-class level)

The institutional memory of what was measured is preserved (in tissue_arm_VAL_101_25tile_sealed_at_O5 block of the card JSON, and in this README section). The no-propagation rule is preserved (validation tier unchanged, claims unchanged, Stage 2 mechanism unchanged). Both are required by the cookbook discipline that came out of this session.
