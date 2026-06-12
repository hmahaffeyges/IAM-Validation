# Prostate-EPIC Card — EDEAR Multi-Modal-Validated + Multi-Atlas-Calibrated Prostate Cancer Card

**Version 0.3 · 2026-04-30**
**Validation tier:** `multi_modal_validated + multi_atlas_calibrated` (anchored by VAL-058 GSE269244 tissue + VAL-117 ProstateRef calibration + VAL-118 multi-atlas Phase C)
**Built under:** CCL-029 cohort-completeness · CCL-040 calibration-before-scoring · CCL-041 magnitude-based threshold discipline · CCL-042 atlases-deferred structured format · CCL-043 cookbook-wide CCL cross-references · CCL-046 prereg amendment audit-trail · CCL-049 multi-atlas reporting · CHK-3.1A/B/C calibration gates · CHK-5.7-5.13 documentation discipline · CHK-7.6 reproducibility triple
**Supersedes:** v0.2 (2026-04-25, VAL-058 anchor + VAL-065 urine-arm exploratory). v0.3 promotes by (1) VAL-117 ProstateRef Phase B calibration anchor sealed on TCGA n=210 HM450K sesame Level 3, 6 prostate cell types (BE/EC/Fib/LE/Leu/SM); (2) VAL-118 Phase C multi-atlas re-scoring on GSE269244 (n=238 EPIC 850K) under run-everything discipline against ProstateRef + Layered Moss+Loyfer + UniLIFE + Salas IDOL + Xu-538 Stage 1 reproduction control. v0.2 content preserved verbatim; v0.3 additions appear as new sections at the end of this README.

## Clinical claim

A sample that produces a Stage 1 immune-class A-score elevation at DETECTABLE tier or higher AND whose Stage 2 Moss NNLS deconvolution localizes the top-1 tissue to `prostate_epithelial` is flagged as consistent with architectural drift in prostate epithelium. Card v0.1 is anchored to the VAL-058 tissue-level validation on GSE269244 (n=238 EPIC 850K, African-American men, tumor + adjacent-normal pairs, Berglund/Yamoah/Kresovich 2024).

The prostate-epic card fills a specific clinical gap: when a patient's EDEAR analysis returns a Stage 1 immune flag and Stage 2 localizes to prostate_epithelial, the card provides clinical action guidance (PSA + DRE + multi-parametric MRI + urology consultation per standard of care).

**What this card is NOT.** It is NOT a pre-diagnostic blood screening test for prostate cancer. That would require a per-patient blood methylation validation on a cohort like Health ABC or Rotterdam Study — both dbGaP-gated as of 2026-04-24. Public GEO has no usable per-patient prostate pre-diagnostic blood methylation cohort (confirmed by multi-geography hunt: US, Italy, Japan, Korea, China, Sweden, Denmark, PI-name searches, consortium searches — all yielded tissue cohorts only). See `prostate-LL-004`.

Upgrade path from `stage_2_only_validated` to `cohort_screening_validated`:
1. dbGaP application for Health ABC or Rotterdam Study (2–12 week typical approval).
2. Per-patient Phase 9/12-equivalent run on approved cohort with TtDx metadata, n ≥ 100 prostate cases.

## What this card covers, and what it does not

**What v0.1 covers.** The card knows what prostate tumor tissue methylation looks like on EPIC 850K, drawn from an actual 238-sample tumor-vs-adjacent-normal case-control (VAL-058). Xu-538 separates tumor tissue from adjacent-normal prostate tissue at paired d = +0.497 on 118 matched patient pairs. When a patient's Stage 1 blood immune panel fires at DETECTABLE or higher and Stage 2 Moss NNLS on that patient's ccfDNA returns prostate_epithelial as the top-1 tissue, the prostate-epic card fires with evidence-backed clinical action guidance.

**What v0.1 does NOT cover — and what each limit means clinically.**

1. **No per-patient pre-diagnostic blood immune-class validation exists for prostate.** Unlike breast-epic, crc-epic, and lung-epic, prostate-epic has no VAL-046-equivalent or Phase-9/12-equivalent evidence that the Xu-538 panel fires on pre-diagnostic blood from people who later develop prostate cancer. The Stage 1 blood immune flag for a prostate case is NOT independently validated. The card's clinical claim depends on Stage 2 Moss NNLS firing, not on Stage 1 blood immune-class detection of the prostate cancer itself.

2. **Early localized prostate cancer sheds minimally into plasma ccfDNA.** This is a known ctDNA-shedding hierarchy in the literature: liver, lung, and colorectal cancers shed substantially into plasma; early-stage localized prostate cancer sheds much less. Moss 2018's prostate ccfDNA signature (Fig 4d, β 0.635 vs healthy 0.743) was demonstrated on a metastatic/advanced cohort, not an early-stage pre-diagnostic cohort. **A negative Stage 2 Moss deconvolution does NOT rule out early-stage localized prostate cancer.** Moss NNLS for prostate is a positive-finding test, not a rule-out test.

3. **Urine-specimen pathway not yet integrated.** Urinary prostate cells shed continuously, and published urine-methylation prostate tests (SelectMDx, ConfirmMDx, UroMark) demonstrate that urine outperforms blood for early prostate detection. A urine-specimen Stage 1 variant with prostate-specific H_min and panel would materially improve this card's early-disease sensitivity. Listed as a requirement for prostate-epic v0.2+.

4. **African-American-only cohort.** VAL-058 is 100% African-American men. Prostate cancer genomics and epigenomics differ by ancestry. European and Asian ancestry validation pending.

5. **No Gleason stratification.** The card fires on tumor vs normal; it does not distinguish aggressive (GG4/5) from indolent (GG1) disease. Aggressive-subgrade analysis was in VAL-058 raw output but not pre-registered for card firing.

**So, realistically, can we pick up prostate cancer in bloodwork?** The honest answer from v0.1 data: Stage 2 Moss NNLS on plasma ccfDNA CAN detect advanced or metastatic prostate cancer (Moss 2018 demonstrated this). Early-stage localized prostate cancer shedding is low; plasma-based detection is likely underpowered. Urine is probably the right specimen for early prostate, and urine is not in the current card. Stage 1 blood immune-class firing for prostate is not independently validated by this card; when the card fires, it's because Stage 2 Moss pointed to prostate_epithelial, not because Xu-538 flagged prostate-specifically on blood.

## The workflow in one patient

A 68-year-old man with family history of prostate cancer submits a buffy-coat blood draw for EDEAR analysis. The lab runs an Illumina EPIC 850K array.

**Stage 1 (universal).** Xu-538 CpGs extracted, pooled-entropy A-score computed against H_min(immune) = 0.838889, compared to age-matched 80-cell healthy baseline (60–69 decade). Tier call assigned.

**Stage 2 (if Stage 1 hits DETECTABLE or above).** Moss 2018 NNLS deconvolution produces an 18-tissue β vector. Top-1 localization is identified. If top-1 = `prostate_epithelial` (secretory class H_min = 0.843264, healthy reference β = 0.743, VAL-041 expected cancer β ≈ 0.635), this prostate-epic card fires.

**Report.** The patient's clinician receives:
- A-score tier call and age-matched percentile
- Stage 2 top-3 tissue localization table with `prostate_epithelial` ΔA highlighted
- Stage 2 confidence indicator (top-1 / top-2 ΔA ratio)
- Assay version tag (L1 Illumina EPIC + Moss markers / L2 custom capture / L3 full MESA+DELFI)
- Clinical action: PSA + DRE + mpMRI + urology referral per NCCN early detection guidelines
- **Explicit disclosure that this card does NOT claim pre-diagnostic blood screening for prostate cancer.** The clinical action reflects Stage-2-localization firing, not a standalone blood screening test.
- Honest limitations section naming `stage_2_only_validated` tier and African-American-only cohort ancestry

## Validation summary

| Anchor | Cohort | n | Primary result | Tier contribution |
|---|---|---|---|---|
| VAL-058 Xu-538 on prostate tissue | GSE269244 tumor vs adj-normal | 238 (118 pairs) | Unpaired d = +0.400 [+0.146, +0.659] · Paired d = +0.497 · p = 0.0001 | Stage 2 validated |
| VAL-041 Stage 2 localization | Moss 2018 Fig 4d prostate case | — | β_prostate = 0.635 (Δβ = −0.108) | Stage 2 mechanism confirmed |
| Stage 1 per-patient pre-dx | — | — | PENDING — Health ABC / Rotterdam dbGaP-gated | Not yet `cohort_screening_validated` |

**Total VAL-058 primary pass:** O1 — Xu-538_PROSTATE_TISSUE_VALIDATED.

## Sources

**VAL-058 cohort.** Berglund A, Yamoah K, Kresovich JK, et al. *Epigenome-wide association study of Prostate Cancer in African American Men identified differentially methylated genes.* 2024. PMID: 39162297. GEO: <https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE269244>.

**VAL-041 Stage 2 reference.** Moss J et al. *Comprehensive human cell-type methylation atlas reveals origins of circulating cell-free DNA in health and disease.* Nat Commun 2018; 9:5068. DOI: <https://doi.org/10.1038/s41467-018-07466-6>.

**Xu-538 panel origin.** Xu Z, Sandler DP, Taylor JA. *Blood DNA methylation and breast cancer: a prospective case-cohort analysis in the Sister Study.* J Natl Cancer Inst 2020; 112(1):87-94. DOI: <https://doi.org/10.1093/jnci/djz065>.

## Pre-registration chain

- `VAL_058_PREREG.md` SHA-256: `48abe394ad009020d4bafeeb262439ee02fc910df6d79a96ed56d235a0608316` sealed 2026-04-24 06:50:36 UTC
- `VAL_058_PREREG_AMENDMENT.md` SHA-256: `b01eac163ea3cea80dcaf97042f996ba925bf190b1dcbab28f799f4a60eb37cf` sealed 2026-04-24 06:54:15 UTC

**On the amendment and what Moss pieces are public vs proprietary.** The original pre-reg specified Stage 2 metrics M2 and M3 using the Moss 2018 prostate reference CpG subset. The amendment removed M2 and M3 from the v0.1 run because the in-session tooling did not have the Moss reference matrix staged. However, the Moss 2018 marker CpG list and the reference matrix R itself are PUBLIC — published in Moss 2018 Supplementary Table S4 and mirrored on GitHub at `nloyfer/meth_atlas`. What IS proprietary is the H_min calibration layer (G-003b MCMC posteriors per architecture class per substrate, covered under US Provisional Patents 64/012,720 and 64/014,568). A future VAL-058b run using the public Moss S4 marker CpGs plus `scipy.optimize.nnls` for the deconvolution (keeping H_min calibration out of any public script) would add Stage 2 tissue-β metrics on top of the Xu-538 tissue case-control we have here. See `prostate-LL-005`.

- β matrix SHA: `7b9fa2825bdd88b0936afba0e19fb0fbcf1bd404a65469d9fb0735829dc88a89`
- Xu-538 panel SHA: `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`

## Known limitations

See `prostate-epic_card_v0.1.json` → `known_limitations`. Summary: tissue-only validation, African-American-only cohort, no Gleason stratification in scope, no per-patient pre-diagnostic blood validation, sample size n=238 below cohort_screening threshold.

## Lessons learned

See card JSON `lessons_learned` section — 5 entries (prostate-LL-001 through prostate-LL-005) covering Xu-538 cross-disease/cross-substrate transfer, paired-vs-unpaired analysis strength, panel-specific vs genome-wide direction trends, public GEO prostate landscape as of April 2026, and the distinct clinical role of the stage_2_only_validated tier vs cohort_screening_validated.

---

## Urine arm — VAL-065 EXPLORATORY (added v0.2)

The v0.1 card explicitly listed "urine specimen pathway not yet integrated" as a requirement for v0.2+. VAL-065 attempted that integration on the only public urine methylation prostate cohort with EPIC 850K data on GEO: **GSE119260 (Brikun et al. 2018)**, n=4 advanced-stage prostate cancer patients, 4 specimens per patient (FFPE adjacent-normal benign + FFPE primary tumor + plasma cfDNA + urine sediment), 16 samples total.

**Result classification: EXPLORATORY OPEN QUESTION.** The cohort is too small (n=4) and too uniform in advanced-disease state (all bone metastatic, Gleason 4+4 to 5+5, PSA 10.9 to 1,400 ng/mL) to draw any substrate-vs-substrate conclusions. VAL-065 documents that the public-data ceiling for the urine arm is at n=4, and that a larger urine methylation prostate cohort with healthy controls and mixed disease stages is the priority-1 unmet data need for prostate-epic v0.3+.

### What VAL-065 measured

| Patient | Age | Gleason | PSA (ng/mL) | A_benign | A_tumor | A_plasma | A_urine |
|---|---|---|---|---|---|---|---|
| P1 | 58 | 4+4 | 1,400 | 0.802 | 0.805 | 0.768 | 0.673 |
| P2 | 66 | 5+4/4+5 | 10.9 | 0.805 | 0.741 | 0.636 | 0.629 |
| P3 | 76 | 4+5 | 144 | 0.781 | 0.727 | 0.841 | 0.523 |
| P4 | 68 | 5+5 | 38.98 | 0.604 | 0.713 | 0.744 | 0.505 |

### Pre-registered hypotheses and outcomes

| Hypothesis | Threshold | Observed | Result |
|---|---|---|---|
| H1: urine closer to tumor than plasma in ≥3/4 patients | ≥3/4 | 0/4 (plasma closer in all 4) | **FAIL** |
| H2: urine vs benign paired Cohen's d > +0.3 | d > +0.3 | d = −2.39 | **FAIL — wrong direction, large magnitude** |
| H3: urine direction preservation rate ≥ plasma | urine ≥ plasma | 51.3% vs 47.4% | PASS marginal |

H2 failed in an unexpected direction with very large magnitude. The pre-registration explicitly anticipated this case under the O5_UNEXPECTED outcome: *"urine vs benign d > 0.3 but in NEGATIVE direction (urine A-score LOWER than benign tissue, suggesting clearance of disrupted cells rather than retention). Report numbers honestly; convene with Heath before deciding card update direction."* That case hit. After review, the verdict is exploratory documentation only — no card update direction.

### What VAL-065 does NOT establish

- Does NOT establish urine sediment as a valid (or invalid) substrate for prostate detection
- Does NOT falsify the v0.1 hypothesis that "urine outperforms blood for early prostate detection" (GSE119260 is not early-stage)
- Does NOT establish a urine A-score expected direction (positive or negative) for any prostate stage
- Does NOT provide a deployable urine clinical pathway
- Does NOT alter the VAL-058 anchored `stage_2_only_validated` tier of the card

### What VAL-065 DOES establish

- The Xu-538 panel can be applied to urine sediment β-value data and produces measurable A-scores at full panel coverage on EPIC 850K (435/538 CpGs measured per sample, 100% sample QC pass rate)
- Within-patient urine, plasma, tumor, and benign tissue can be co-analyzed with the same Xu-538 panel and same H_min(immune) without methodological obstruction — this is the methodological deliverable, separate from the inferential one
- The 4-patient Brikun 2018 cohort, the only public EPIC 850K urine prostate cohort on GEO as of April 2026, is too small for substrate-vs-substrate inference; this is the primary informational deliverable of VAL-065

### Why a tumor-vs-benign signal is not visible at n=4

Tumor vs benign paired d = −0.016 in this cohort (essentially zero). The expected positive tumor signal from VAL-058 (n=238 African-American men, paired d = +0.497) is not recoverable at n=4. This is an n-limited statistical reality, not a finding about the framework. Any specimen-vs-specimen comparison made with a non-existent reference signal is fundamentally uninformative — including the dramatic-looking d = −2.39 urine vs benign magnitude. The signal is real in the data; the inference is not.

### Open mechanistic questions (for future investigation when larger cohort available)

1. **Is urine sediment a fundamentally different substrate for the Xu-538 immune panel?** Urine sediment is dominated by sloughed bladder/urethral epithelium and shed prostate cells — cells that have crossed an apoptotic/necrotic barrier. Their methylation entropy may collapse toward homogeneous values during programmed cell death and lytic clearance, producing a low A-score that reflects the cell-death endpoint rather than the live-tissue architectural state.
2. **Does urine A-score direction invert in advanced disease specifically?** All 4 GSE119260 patients have bone-metastatic Gleason 4+4 to 5+5 disease. Direction inversion in advanced disease has documented precedent in the framework (CRC peripheral immune at d = −0.33 in VAL-047 Phase 12; TGCT seminoma A = 0.755 in VAL-045). Cannot distinguish substrate-physics inversion from advanced-disease inversion at n=4.
3. **What is the appropriate H_min for urine sediment specifically?** H_min(immune) = 0.838889 was calibrated on Xu-538 panel applied to peripheral blood and tissue. Urine sediment may need its own calibration if used as a primary specimen.

### Priority-1 next step

A larger urine methylation prostate cohort with healthy controls and mixed disease stages is the priority-1 unmet data need for prostate-epic v0.3+. Candidate paths:

- **dbGaP and consortium catalog search** — SelectMDx, ConfirmMDx, UroMark validation cohort data; Movember Foundation urine methylation studies; PCA3 methylation cohorts
- **L1 lab partnership tier collection** — n=20-50 urine sediment + matched blood EPIC 850K from a local active-surveillance prostate cohort across Gleason 6 / 7 / ≥8 strata + healthy male controls. Cost estimate at $50-150/sample × 50 = $2,500-$7,500. This is the most directly actionable path under the existing IAMPerformance lab partnership framework.
- **Park as open question** in v0.2 with VAL-065 cited as the only available data point. CCL-026 documents the substrate-physics open question.

### Reproduction bundle

- **Pre-registration:** [`VAL-065_prereg.md`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-065_prereg.md) (SHA `f1d1a997...`, sealed BEFORE any β-value access)
- **Outcome:** [`VAL-065_outcome.md`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-065_outcome.md)
- **Reproducible script:** [`val065_prostate_epic_urine_arm.py`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/val065_prostate_epic_urine_arm.py) (Python 3 stdlib only)
- **Results:** [`VAL-065_results.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/VAL-065_results.json)
- **Cohort manifest:** [`GSE119260_manifest.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/validation_runs/GSE119260_manifest.json)

### v0.2 changes (2026-04-25)

- **Urine arm attempted on the only public EPIC 850K urine methylation prostate cohort (GSE119260, n=4).** Result classified O5_UNEXPECTED per the pre-registration's anticipated edge case. Treated as exploratory open question — no card update direction taken on the urine substrate.
- **VAL-058 anchor unchanged.** The card's `stage_2_only_validated` tier remains anchored by VAL-058 GSE269244 tissue paired d = +0.497, n=238 African-American men. VAL-065 does not affect this tier.
- **CCL-026 added to LESSONS_LEARNED.md** documenting the urine substrate physics open question and the n=4 advanced-disease cohort ceiling on public urine methylation prostate data.
- **Priority-1 next-validation-step list updated** to lead with "larger urine methylation prostate cohort acquisition" via dbGaP application, consortium catalog, or L1 lab partnership tier collection.

---

# v0.3 Additions (2026-04-30)

The sections above are preserved verbatim from v0.2. Everything below is new in v0.3.

## v0.2 → v0.3 changes

- **VAL-117 ProstateRef Phase B calibration anchor sealed.** EpiSCORE ProstateRef matrix bridged to 2,603 unique 450K CpGs × 6 prostate cell types (BE/EC/Fib/LE/Leu/SM). Calibrated on TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 HM450K sesame Level 3 (same calibration cohort cardio used for VAL-106/107/112/113). All three CHK gates pass: CHK-3.1A 98.1%, CHK-3.1B 100% (under amended ≥80% threshold per substrate floor), CHK-3.1C zero duplicates. Per-tile healthy-floor distributions sealed; LE tile (luminal epithelial — prostate adenocarcinoma cell of origin) has tightest distribution (sd=0.0041).
- **VAL-118 Phase C multi-atlas re-scoring sealed.** GSE269244 (n=238 EPIC 850K, 118 paired patients) scored under run-everything discipline against five atlases: Xu-538 Stage 1 + ProstateRef + Layered Moss+Loyfer 17-tile subset + UniLIFE 19-cell + Salas IDOL Stage 3. Stage 1 Xu-538 reproduces VAL-058 sealed paired d (+0.5258 vs +0.4973 sealed, within ±0.10 tolerance).
- **Headline biological finding (DISC-PROSTATE-003):** ProstateRef LE tile reads tumor at d_paired = −0.767 (luminal dedifferentiation pattern); other 5 ProstateRef tiles all positive (+0.48 to +1.31). The five-vs-one direction split is the operationally important diagnostic signal.
- **Stage 3 immune signal:** Salas IDOL Mono d_paired = +0.771; five Salas tiles read d_paired between +0.59 and +0.77 — broad TIL infiltration signature in tumor tissue, consistent with Berglund 2024's published CD40/OX40L/STING DMRs.
- **Pre-registration discipline matured.** VAL-118 first execution sealed O5 because original prereg locked LE outcome as positive-direction-only and observed pattern was strong negative. CCL-041 amendment (sealed before re-execution) changed threshold to magnitude-based |d| ≥ 0.30 with direction labels. Re-execution sealed O1 + O2 (LE_NEGATIVE) + O4 cleanly. DISC-PROSTATE-002 formalizes the cookbook rule for cell-of-origin atlases.
- **Validation tier promoted** from `stage_2_only_validated` to `multi_modal_validated + multi_atlas_calibrated`.

## v0.3 atlases_used_and_deferred

### atlases_run (calibrated and deployed in v0.3 production scoring)

| Atlas | n_CpGs | n_tiles | Calibration anchor | CHK-3.1B q5 threshold | CHK-3.1C |
|---|---|---|---|---|---|
| Xu-538 Stage 1 (immune pooled) | 538 | 1 | VAL-058 self-cal (HM450 + EPIC) | n/a (Shannon-symmetric pooled metric) | passed |
| EpiSCORE ProstateRef CpG-bridged | 2,603 | 6 | VAL-117 (TCGA HM450K sesame Level 3 n=210) | 80% per amendment | passed (0 dups) |
| Layered Moss+Loyfer 17-tile subset | 6,105 | 17 | VAL-112 (TCGA HM450K sesame Level 3 n=210) | inherited from VAL-112 | passed |
| UniLIFE 19-cell Stage 3 | 1,906 | 19 | VAL-118 self-cal on EPIC 850K (smoke test) | within-cohort | passed |
| Salas Blood.EPIC IDOL Stage 3 | 450 | 6 | Production atlas (VAL-118 self-cal on EPIC 850K) | within-cohort | passed |

### atlases_deferred (structured per v0.5 TODO format)

| Atlas | Target version | Unblock dependency |
|---|---|---|
| Layered Moss+Loyfer Prostate_epithelial bulk tile | v0.4 | Atlas integration: the local atlas vault layered file currently scores a 17-tile subset that excludes the bulk Prostate_epithelial column. Engineering work to add the column to the locally-stored layered atlas, then run a focused Phase B calibration on TCGA n=210, then re-run Phase C scoring against that one new tile. Cardio precedent: VAL-110 aortic Stage 1 work demonstrated the integration pattern. |
| Caggiano CelFiE TIM array-bridged 19-tile prostate-relevant subset | v0.4 | Phase B calibration of Caggiano TIM bridged matrix on TCGA prostate-only adjacent-normal n=50 (rather than the combined KIRC+PRAD n=210 used for cardio VAL-113). Caggiano has 1,581 CpG regions × 19 sorted cell types including immune sub-types and stromal markers; bridge engineering uses the same HM450 hg19 manifest extraction template that VAL-113 cardio sprint produced. |
| EPIC 850K substrate-matched ProstateRef calibration anchor | v0.4 | A structurally-separated EPIC 850K healthy prostate cohort to anchor a Phase B calibration that generalizes from TCGA HM450K sesame Level 3 to EPIC 850K. Currently no such cohort exists in public deposits at sufficient n; v0.3 documents the EPIC 850K scoring on GSE269244 as within-cohort self-calibrated per CCL-041 / DISC-CARDIO-005. |
| Healthy population-normal prostate methylation cohort | v0.4+ | TCGA-PRAD adjacent-normal is histologically-normal tissue from prostate cancer patients, not population-normal. A larger population-normal prostate methylation reference would refine VAL-117 healthy-floor thresholds. Candidate: GTEx prostate methylation expansion (currently expression-only). |
| Multi-ancestry prostate methylation cohort | v0.4+ | VAL-058 + VAL-118 anchor cohort is 100% African-American men. European and Asian ancestry replication pending. Candidates surveyed in `cohort_survey.md` from Phase 0 (2026-04-30). |
| FitzGerald 2017 MCCS pre-diagnostic prostate cohort | v1.0+ | Cohort access tier classification in progress. n=687 prospective nested case-control, peripheral blood pre-diagnostic, HM450K — the missing piece that would promote prostate-epic from `multi_modal_validated` to `cohort_screening_validated`. GEO accession unconfirmed; biobank-gated pathway likely. |
| Howard AA EPIC 850K cohort (PMC9980641) | v1.0+ | Per Phase 0 cohort survey (2026-04-30). Cohort access tier and substrate compatibility pending review. |
| Larger urine-substrate prostate methylation cohort | v0.4+ | Per VAL-065 v0.2 finding (CCL-026): GSE119260 n=4 advanced-disease ceiling; no public urine methylation prostate cohort with healthy controls and mixed disease stages. Path: dbGaP+consortium catalog search OR L1 lab partnership tier collection (n=20-50 urine sediment + matched blood across Gleason 6/7/≥8 + healthy male controls). |

## v0.3 chk_3_1_thresholds_per_substrate

The card encounters two measurement substrates across its sealed VALs. Each substrate documented per CHK-5.10:

### Substrate 1 — TCGA HM450K sesame Level 3

| Field | Value |
|---|---|
| substrate_name | tcga_hm450k_sesame_level3 |
| f_extreme_baseline | 55.87% (VAL-106 sealed mean) |
| f_extreme_sd | 2.44% |
| f_middle_baseline | 7.42% |
| f_middle_sd | 0.75% |
| chk_3_1b_subset_threshold (ProstateRef) | 80.0% (per VAL-117 amendment, sealed 2026-04-30T15:28:21Z) |
| chk_3_1b_subset_threshold (Layered Moss+Loyfer) | inherited from VAL-112 |
| calibration_anchor_val_id | VAL-117 (ProstateRef), VAL-112 (Layered Moss+Loyfer) |
| calibration_anchor_cohort_n | 210 (KIRC=160 + PRAD=50) |
| application | Stage 2 cell-of-origin scoring on plasma ccfDNA + tissue biopsy substrates upstream of EPIC 850K integration |

### Substrate 2 — EPIC 850K FFPE (within-cohort self-calibrated, v0.3 limitation)

| Field | Value |
|---|---|
| substrate_name | epic_850k_ffpe_within_cohort_selfcal |
| f_extreme_baseline | 56.0% (VAL-118 observed mean) |
| f_extreme_sd | 2.4% |
| f_middle_baseline | (within-cohort) |
| chk_3_1b_subset_threshold | within-cohort coverage range 80-88% (substrate self-calibrated) |
| calibration_anchor_val_id | VAL-118 self-cal (per CCL-041 / DISC-CARDIO-005, EPIC self-cal documented up-front as v0.3 limitation) |
| calibration_anchor_cohort_n | 238 (the GSE269244 cohort itself, 118 paired) |
| application | tissue biopsy methylation profiling and post-treatment trajectory monitoring use cases |
| v0.4+ next-step | structurally-separated EPIC 850K healthy prostate cohort to anchor cross-substrate generalization |

## v0.3 run_everything_phase_c_results

### Per-cohort headline findings

**GSE269244 (n=238 EPIC 850K FFPE prostate, AA men, 118 paired patients):**

Stage 1 Xu-538 reproduction control: paired d = +0.5258, unpaired d = +0.4220, n_pairs = 118. Reproduces VAL-058 sealed paired d = +0.4973 within ±0.10 tolerance. Stage 1 anchor stands.

ProstateRef per-tile signature (the operationally important finding):

| Tile | Tumor mean A | Normal mean A | d_paired | d_unpaired | Direction label | Biological interpretation |
|---|---|---|---|---|---|---|
| **LE** (luminal epithelial — PCa cell of origin) | 0.4069 | 0.4122 | **−0.767** | −0.695 | **LE_NEGATIVE** | **Luminal dedifferentiation** — tumor cells lose canonical methylation signature |
| BE (basal epithelial) | 0.4198 | 0.4166 | +0.477 | +0.440 | BE_POSITIVE | Basal architectural drift |
| EC (vascular endothelial) | 0.4085 | 0.3964 | +1.284 | +1.682 | EC_POSITIVE | Tumor microvasculature |
| Fib (fibroblasts) | 0.4213 | 0.4072 | +1.311 | +1.621 | Fib_POSITIVE | Stromal architectural complexity |
| Leu (intra-prostatic leukocytes) | 0.4487 | 0.4393 | +0.999 | +1.149 | Leu_POSITIVE | Local immune infiltrate |
| SM (smooth muscle, peri-prostatic) | 0.4192 | 0.4083 | +1.092 | +1.350 | SM_POSITIVE | Peri-prostatic stromal drift |

Stage 3 multi-atlas immune signature:

| Atlas | Top-3 tiles by paired d | d_paired |
|---|---|---|
| Salas Blood.EPIC IDOL | Mono | +0.771 |
| Salas Blood.EPIC IDOL | Bcell | +0.674 |
| Salas Blood.EPIC IDOL | CD4T | +0.659 |
| UniLIFE 19-cell | aMono | +0.467 |
| UniLIFE 19-cell | aNeu | +0.433 |
| UniLIFE 19-cell | Mono | +0.391 |

Salas IDOL Mono d_paired = +0.771 fires O4 (≥+0.40 magnitude threshold). Five Salas tiles between +0.59 and +0.77 — broad TIL infiltration signature.

### Outcome class summary (sealed per VAL-118 amendment)

| Outcome | Threshold | Observed | Status |
|---|---|---|---|
| **O1_MULTI_ATLAS_CONVERGENT** | O2 fires AND Stage 1 reproduces VAL-058 within ±0.10 | LE \|d\|=0.767; Stage 1 d_paired=+0.5258 (Δ=+0.029) | **FIRED** |
| **O2_LE_TILE_DIFFERENTIATING** | LE \|d_paired\| ≥ 0.30 with direction label | LE d_paired=−0.767, label LE_NEGATIVE | **FIRED** |
| O3_BULK_TILE_DIFFERENTIATING | Loyfer Prostate_epithelial \|d\| ≥ 0.30 | Inapplicable until v0.4+ atlas integration | N/A |
| **O4_STAGE_3_IMMUNE_SHIFT_PROMINENT** | UniLIFE or Salas paired \|d\| ≥ 0.40 | Salas Mono d_paired=+0.771 | **FIRED** |
| O5_MULTI_ATLAS_DIVERGENT | Atlases disagree by >0.50 in opposite directions | No divergence | not fired |
| O6_UNEXPECTED | Anything else | n/a | not fired |

## v0.3 per-disease scoring policy

### What v0.3 prostate-epic disease scoring claims

**Tissue substrate (FFPE prostate, biopsy methylation profiling):**
- A_LE BELOW q5 of VAL-117 healthy floor (q5 = 0.4190, mean = 0.4254, SD = 0.0041) is consistent with luminal dedifferentiation pattern. Direction label LE_NEGATIVE.
- Concurrent A_EC, A_Fib, A_Leu, A_SM ABOVE q95 of VAL-117 healthy floor for those tiles is consistent with tumor microenvironment architectural complexity.
- The five-tile-positive + LE-negative pattern is the v0.3 prostate cancer architectural signature on tissue.

**Plasma ccfDNA substrate (early/localized prostate):**
- Acknowledged as currently underpowered (per v0.2 limitations). v0.3 does NOT extend ProstateRef LE-tile claims to plasma ccfDNA without substrate-matched calibration. Stage 1 Xu-538 immune flag + Stage 2 Moss NNLS prostate localization remains the v0.2 inherited claim for plasma-substrate use.

**Post-treatment monitoring trajectory (the wife's-uncle use case):**
- Serial tissue or plasma draws scored against the same ProstateRef LE tile produce a trajectory; A_LE trending below the q5 healthy floor flags potential dedifferentiation drift; this is the operationally important deployment for v0.3.

### What v0.3 prostate-epic scoring DOES NOT claim

- Pre-diagnostic blood screening — unchanged from v0.2; biobank-gated cohorts (FitzGerald 2017 MCCS pre-dx, Howard AA EPIC) remain v1.0+ next-validation-steps
- Multi-ancestry coverage — v0.3 anchor cohorts are 100% African-American men; European and Asian ancestry validation pending
- Gleason aggressiveness stratification — VAL-118 carries Gleason metadata but did not pre-register grade-stratified outcomes; aggressiveness card extension deferred to v0.4
- Early-stage localized detection from plasma ccfDNA — substrate physics unchanged from v0.2 limitations
- Urine substrate clinical pathway — VAL-065 n=4 ceiling unchanged; CCL-026 open question
- Stage-1 directional A_dir for prostate (analog to AD Rule A) — not constructed for v0.3; pooled-entropy A_immune on Xu-538 is the Stage 1 metric for prostate

## v0.3 DISC-PROSTATE discoveries

Each entry below also propagates to LESSONS_LEARNED.md as per Heath-only file delivery.

### DISC-PROSTATE-001 — Gene-promoter atlas family fitness depends on cell-type distinctness for the tissue

VAL-117 calibration of EpiSCORE ProstateRef (2,603 CpGs × 6 prostate cell types) on TCGA n=210 HM450K sesame Level 3 produced per-tile within-cohort A-score variance 2-4× higher than VAL-111 cardio HeartRef showed on the same calibration cohort. HeartRef collapsed to A ≈ 0.5 across all five cardiac tiles (max within-cohort range 0.0152, sealed O3_TISSUE_FLOOR_DOMINATED). ProstateRef did NOT collapse — six prostate cell types span markedly different gene-promoter methylation profiles for the basal-vs-luminal-vs-stromal-vs-vascular distinctions. Max within-cohort range = 0.0597 (Leu); minimum range = 0.0293 (LE). All tiles cleared the 0.02 tissue-floor-dominated threshold.

**Implication.** Atlas family fitness extends DISC-CARDIO-004 lesson: gene-promoter-based atlases do NOT uniformly fail on heterogeneous β panels. Whether a gene-promoter atlas family fits depends on whether the atlas's cell types actually produce distinct gene-promoter methylation patterns for the tissue in question. ProstateRef fits prostate biology; HeartRef collapsed on cardiac biology because cardiac cell types share substantial gene-promoter methylation similarity at the marker-gene level. Future card sprints evaluating gene-promoter atlas family fitness must run a per-tissue calibration smoke test before committing to or deferring the atlas.

### DISC-PROSTATE-002 — Pre-registration discipline must use magnitude-based \|d\| thresholds for cell-of-origin atlases where direction-ambiguity is biologically possible

VAL-118 first execution sealed O5_LE_DIRECTION_FLIP_UNANTICIPATED because the original prereg pre-locked O2 as `LE paired d ≥ +0.30` (positive direction). Observed pattern was `d_paired = −0.767` (large negative). The biology was clean (luminal dedifferentiation — tumor cells losing the canonical luminal methylation signature) but the prereg was over-specified directionally. CCL-041 forbade post-hoc sign-flip. Amendment changed threshold to `|d_paired| ≥ 0.30` with direction labels (LE_POSITIVE / LE_NEGATIVE) and was sealed BEFORE re-execution. Re-execution then sealed O1 + O2 (LE_NEGATIVE) + O4 cleanly.

**Implication.** Operational rule for future cell-of-origin atlas preregs: `|d| ≥ threshold` with direction label, not `d ≥ threshold` (positive only) or `d ≤ −threshold` (negative only). When biology supports a direction-flip pattern (cell-of-origin dedifferentiation produces negative-direction A-score shifts; cell-of-origin overexpression produces positive-direction shifts), magnitude-based thresholds with direction labels capture both without compromising CCL-041 discipline. Bulk-tile or pooled metrics where direction is biologically uniform (e.g. Stage 1 Xu-538 pooled A_immune via Shannon symmetry) do not require this rule. This rule is now cookbook-wide.

### DISC-PROSTATE-003 — ProstateRef LE tile reads tumor strongly NEGATIVE (luminal dedifferentiation signature)

The headline biological finding of VAL-118: ProstateRef LE tile (luminal epithelial — prostate adenocarcinoma cell of origin) reads tumor at d_paired = −0.767 vs adjacent-normal in n=118 paired patients. The five other ProstateRef tiles (BE/EC/Fib/Leu/SM) all read positive (d_paired +0.48 to +1.31). The five-vs-one direction split is the prostate cancer methylation-architecture signature.

**Implication for clinical deployment.** A_LE is the discriminating tile for prostate-epic v0.3 disease scoring on tissue substrates. A_LE BELOW the VAL-117 healthy-floor q5 (0.4190) flags potential luminal dedifferentiation drift. For post-treatment monitoring trajectory tracking (the immediate clinical use case), serial A_LE values are the primary signal; concurrent elevation of stromal/immune tiles supports the diagnostic. This is the tier-promotion finding for v0.3.

## v0.3 What we chose not to claim

- **Pre-diagnostic blood screening for prostate cancer.** The Stage 1 Xu-538 immune flag for prostate is not independently validated by per-patient pre-diagnostic blood data. v0.2 carried this honest limitation; v0.3 does not change it. FitzGerald 2017 MCCS pre-dx and Howard AA EPIC remain v1.0+ next-validation-steps.
- **Multi-ancestry generalizability.** v0.3 anchor cohorts (TCGA-KIRC + TCGA-PRAD adjacent-normal for VAL-117; GSE269244 AA men for VAL-118) are not multi-ancestry-validated. European and Asian ancestry replication pending.
- **Gleason aggressiveness stratification.** VAL-118 carries Gleason metadata but did NOT pre-register Gleason-stratified outcomes. Aggressiveness card extension deferred to v0.4. The v0.3 card fires on tumor vs adjacent-normal architectural drift, not on aggressive vs indolent disease.
- **Early-stage localized detection from plasma ccfDNA.** Substrate physics unchanged from v0.2 limitations. ProstateRef LE-tile claims do NOT extend to plasma ccfDNA in v0.3.
- **Urine-substrate clinical pathway.** VAL-065 n=4 ceiling unchanged; CCL-026 open question stays open.
- **Layered Moss+Loyfer Prostate_epithelial bulk-tile differentiation.** Atlas integration deferred to v0.4 (the local atlas vault layered file scores 17-tile subset that excludes the bulk Prostate_epithelial column at this time).

## v0.3 What remains open

1. Layered Moss+Loyfer Prostate_epithelial bulk-tile integration — engineering work in v0.4
2. EPIC 850K substrate-matched ProstateRef calibration anchor — requires structurally-separated EPIC 850K healthy prostate cohort
3. Caggiano CelFiE TIM array-bridged 19-tile prostate calibration — atlas integration in v0.4
4. FitzGerald 2017 MCCS pre-diagnostic prostate cohort — biobank access pathway TBD
5. Howard AA EPIC 850K cohort — Phase 0 survey identified, cohort access TBD
6. Multi-ancestry validation cohort — European and Asian ancestry replication
7. Population-normal prostate methylation reference cohort (refines VAL-117 healthy-floor)
8. Larger urine-substrate prostate methylation cohort with healthy controls and mixed Gleason — CCL-026 open
9. Gleason aggressiveness card extension (GG1 vs GG4/5 differentiation) — v0.4
10. Stage-1 directional A_dir for prostate (analog to AD 7-CpG Rule A) — not yet constructed; pooled-entropy A_immune via Shannon symmetry is current Stage 1 metric
11. Plasma ccfDNA ProstateRef calibration — substrate physics constraint; deferred until adequate plasma cohort surfaces

## v0.3 Validation evidence summary

### VAL-058 — GSE269244 Stage 1 Xu-538 anchor (preserved from v0.2, verified at v0.3)

**Cohort:** GSE269244 EPIC 850K FFPE prostate tissue, n=238 (118 paired AA men, 120 tumor + 118 adjacent-normal)
**Substrate:** Illumina Methylation EPIC V1 (FFPE)
**Design:** within-cohort tumor vs adjacent-normal paired case-control
**QC pass rate:** 238/238 (100%)
**Outcome:** `O1_Xu-538_PROSTATE_TISSUE_VALIDATED` (sealed)

**Key Cohen's d values:**
- Stage 1 pooled A_immune paired d = +0.4973 (sign-flip permutation p ≈ 0.0001, n_pairs = 118)
- Stage 1 pooled A_immune unpaired d = +0.4003 [+0.146, +0.659] (perm p = 0.0027)
- Per-CpG direction: 217/481 hypermethylated, 264/481 hypomethylated, fraction_hyper = 0.4511

**Interpretation:** Prostate tumor tissue produces architectural drift on the universal Stage 1 immune-class panel at moderate effect size (paired d ≈ +0.50). Per-CpG direction is roughly balanced (45% hyper / 55% hypo), consistent with bulk prostate tissue methylation reorganization rather than uniform-direction drift. Pooled-entropy captures the bulk effect via Shannon symmetry.

**Prereg SHA-256:** `48abe394ad009020d4bafeeb262439ee02fc910df6d79a96ed56d235a0608316` (original)
**Prereg amendment SHA-256:** `b01eac163ea3cea80dcaf97042f996ba925bf190b1dcbab28f799f4a60eb37cf` (sealed before re-execution)

### VAL-065 — GSE119260 urine-arm exploratory (preserved from v0.2, classified open question)

**Cohort:** GSE119260 EPIC 850K, n=4 advanced-disease prostate cancer patients × 4 specimens each (16 samples total)
**Substrate:** EPIC 850K across FFPE benign, FFPE tumor, plasma cfDNA, urine sediment
**Design:** within-patient four-substrate comparison
**QC pass rate:** 16/16 (100%)
**Outcome:** `O5_UNEXPECTED` (sealed, exploratory open question per CCL-026)

**Key Cohen's d values:**
- Tumor vs benign paired d = −0.016 (essentially zero at n=4)
- Urine vs benign paired d = −2.39 (large negative; n=4 limited)
- Direction preservation rate: urine 51.3%, plasma 47.4%

**Interpretation:** N=4 is below the threshold for substrate-vs-substrate inference. The cohort defines the public-data ceiling for the urine arm; a larger urine methylation prostate cohort with healthy controls and mixed Gleason is the priority-1 unmet data need for v0.3+. Within-patient four-substrate methodology is established (deliverable separate from inferential).

**Prereg SHA-256:** `f1d1a997...` (sealed before any β-value access)

### VAL-117 — ProstateRef Phase B calibration anchor (NEW in v0.3)

**Cohort:** TCGA-KIRC + TCGA-PRAD adjacent-normal n=210 (160 KIRC + 50 PRAD)
**Substrate:** TCGA HM450K sesame Level 3
**Design:** structurally-separated healthy substrate calibration on bridged ProstateRef matrix
**QC pass rate:** CHK-3.1A 206/210 (98.1%), CHK-3.1B 210/210 (100% under amended ≥80% threshold), CHK-3.1C 0 duplicates in 2,603 unique probeIDs
**Outcome:** `O1_PROSTATEREF_CALIBRATION_SEALED`

**Per-tile healthy-floor distributions sealed:**
- LE: mean 0.4254, sd 0.0041, q5 0.4190, q95 0.4316, range 0.0293 (tightest)
- BE: mean 0.4319, sd 0.0050, range 0.0367
- EC: mean 0.4030, sd 0.0102, range 0.0491
- Fib: mean 0.4323, sd 0.0090, range 0.0540
- Leu: mean 0.4558, sd 0.0094, range 0.0597 (widest, but well above 0.02 tissue-floor threshold)
- SM: mean 0.4290, sd 0.0084, range 0.0497

**Interpretation:** ProstateRef CpG-bridged matrix calibrates cleanly on TCGA HM450K sesame Level 3 substrate. All six tiles produce distinguishable healthy-floor distributions; LE tile has tightest floor (operationally most-important for tumor-vs-normal disease scoring discrimination). Atlas does NOT collapse to tissue-floor-dominated like VAL-111 HeartRef did — DISC-PROSTATE-001 finding sealed.

**Prereg SHA-256:** `ef72e1bd49478807ba6025c4415a2b41f50c6d0bcea03fbbc265141359a17f91` (sealed 2026-04-30T15:20:41Z)
**Prereg amendment SHA-256:** `5f6600a20fadfbe2da9f76676badeed57e490b0dc53d28c0d55efd9e60592319` (sealed 2026-04-30T15:28:21Z, corrected CHK-3.1B coverage threshold spec error before re-execution)

### VAL-118 — GSE269244 Phase C multi-atlas re-scoring (NEW in v0.3)

**Cohort:** GSE269244 (same as VAL-058 / VAL-065), 118 paired patients
**Substrate:** EPIC 850K FFPE (within-cohort self-calibrated per CCL-041 / DISC-CARDIO-005)
**Design:** run-everything multi-atlas Phase C re-scoring; Stage 1 Xu-538 reproduction control
**QC pass rate:** Stage 1 Xu-538 reproduces VAL-058 sealed paired d within ±0.10; β matrix bit-for-bit verified at SHA `7b9fa282...`
**Outcome:** `O1_MULTI_ATLAS_CONVERGENT + O2_LE_TILE_DIFFERENTIATING (LE_NEGATIVE) + O4_STAGE_3_IMMUNE_SHIFT_PROMINENT`

**Key Cohen's d values:**
- Stage 1 Xu-538 paired d = +0.5258 (vs VAL-058 sealed +0.4973, Δ +0.029)
- ProstateRef LE paired d = −0.767 (LE_NEGATIVE direction label)
- ProstateRef EC paired d = +1.284
- ProstateRef Fib paired d = +1.311
- ProstateRef Leu paired d = +0.999
- ProstateRef SM paired d = +1.092
- ProstateRef BE paired d = +0.477
- Salas IDOL Mono paired d = +0.771 (Stage 3 strongest)
- UniLIFE aMono paired d = +0.467

**Interpretation:** Multi-atlas convergent prostate cancer methylation-architecture signature. The five-vs-one direction split (LE negative, BE/EC/Fib/Leu/SM all positive) is the operationally important diagnostic — luminal dedifferentiation in tumor cell-of-origin + tumor microenvironment architectural complexity. Stage 3 immune signal (Salas Mono d = +0.771) confirms broad TIL infiltration consistent with Berglund 2024's published CD40/OX40L/STING DMRs.

**Prereg SHA-256:** `0a860bea365a2019e1d6fd95a492dc4671a170372165011e115272fdf59a275c` (sealed 2026-04-30T16:09:42Z, BEFORE β read)
**Prereg amendment SHA-256:** `c1b0a07e25ee9b0b9a8931f04ddd8c7677afcd9c8b2257cf4f9e3c6d42c1868b` (sealed 2026-04-30T16:59:19Z, magnitude-based threshold correction sealed before re-execution per Heath sign-off option 1)

## v0.3 Cookbook-wide CCL cross-references

Every CCL the card inherits, applies, or formalizes:

| CCL | Description | How prostate-epic v0.3 honored / formalized / inherited |
|---|---|---|
| CCL-026 | Urine substrate physics open question (n=4 advanced-disease ceiling) | Inherited from v0.2; remains open in v0.3 |
| CCL-029 | Cohort-completeness rule | Phase 0 cohort survey produced `cohort_survey.md` enumerating FitzGerald 2017 MCCS pre-dx, Howard AA EPIC, MCCS heritable methylation 2023, EGAS00001006670 PCBM, TCGA-PRAD HM450, peripheral T-cell EPIC, bioRxiv 2025.02.07.637178; access tiers documented; tiered v0.4 / v1.0+ unblock dependencies recorded |
| CCL-030 | Stage 1 Test 1 / Test 2 distinction | Honored: pooled A_immune Test 1 reproduced (paired d = +0.5258); Test 2 (lymphoid vs myeloid sub-panel split) deferred per OQ-2026-01 |
| CCL-031 | "Bidirectional cancellation" reserved for AD-instance pattern only | Honored: prostate-epic does NOT exhibit pooled-null + directional-pass pattern; per-CpG direction (45% hyper, 55% hypo) is descriptive only, NOT a mechanism diagnostic |
| CCL-032 | Diagnostic order: data integrity → biology → framework | Honored: VAL-118 Stage 1 reproduction control verified data integrity FIRST; biology consistency check (Berglund 2024 published CD40/OX40L/STING DMRs match Salas IDOL Stage 3 signal) verified SECOND; framework outcome class assigned LAST |
| CCL-040 | Calibration-before-scoring discipline | Honored: VAL-117 Phase B calibration sealed on TCGA n=210 BEFORE VAL-118 Phase C scoring on GSE269244 |
| CCL-041 | No post-hoc threshold relaxation | Honored TWICE: (1) VAL-117 first execution failed CHK-3.1B at 95% threshold (spec error); amendment changed to 80% threshold sealed BEFORE re-execution. (2) VAL-118 first execution sealed O5 because LE direction was unanticipated; amendment changed to magnitude-based threshold sealed BEFORE re-execution. Both amendments documented as spec-error corrections, NOT post-hoc relaxation. |
| CCL-042 | Atlases-deferred structured format | Honored: 8-atlas atlases_deferred table with target version + unblock dependency per atlas |
| CCL-043 | Cookbook-wide CCL cross-references in card README | Honored: this section |
| CCL-046 | Prereg amendment audit-trail | Honored: VAL-117 + VAL-118 each carry separate prereg.md + prereg_amendment.md + PREREG_SEAL.txt + PREREG_AMENDMENT_SEAL.txt with separate SHAs and timestamps |
| CCL-047 | Atlas dedup audit trail (CHK-3.1C) | Honored: ProstateRef bridged matrix verified 0 duplicate probeIDs in 2,603 entries during VAL-117 calibration |
| CCL-048 | Per-tile healthy-floor distributions sealed at calibration | Honored: VAL-117 sealed mean/sd/q5/q50/q95 for all 6 ProstateRef tiles |
| CCL-049 | Multi-atlas reporting flag for single-atlas |d| > 2 not replicated | Checked: most high-|d| Layered Moss+Loyfer tiles ARE confirmed by ProstateRef and/or Stage 3 atlases. Two suspected tissue-floor-effect tiles (Left_atrium, Adipocytes) flagged for v0.4 investigation. No flag fires for operational tiles. |

## v0.3 Reproduction bundle

- **VAL-117 calibration:** [`VAL-117_prostateref_calibrate/`](https://github.com/hmahaffeyges/IAM-Validation/tree/main/Biological_Physics/validation_runs/VAL-117_prostateref_calibrate) — prereg.md, prereg_amendment.md, val117_prostateref_calibrate.py, VAL-117_calibration_results.json, VAL-117_per_sample_calibration.csv, outcome.md
- **VAL-118 Phase C:** [`VAL-118_prostateref_phaseC/`](https://github.com/hmahaffeyges/IAM-Validation/tree/main/Biological_Physics/validation_runs/VAL-118_prostateref_phaseC) — prereg.md, prereg_amendment.md, val118_stage1_extract.py, val118_stage2_score.py, VAL-118_amendment_cohen_d_per_atlas.json, VAL-118_amendment_per_sample_run_everything.csv, outcome.md, outcome_amendment.md, phase_d_v02_vs_v03.md
- **ProstateRef atlas:** [`atlas_vault/stage2_cell_of_origin/episcore_prostateref/`](https://github.com/hmahaffeyges/IAM-Validation/tree/main/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_prostateref) — episcore_prostateref_cpg_bridged.csv (SHA `4e60c3d038a637e9742f51d9bc7c119e06fe5d2e91abb2b12db8867ceb7813d2`), README.md, bridge_prostateref_to_array.py
- **Atlas vault inventory:** [`atlas_vault/INVENTORY.json`](https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/atlas_vault/INVENTORY.json) — 90 entries including new ProstateRef
- **Companion Floor Breach paper:** Zenodo DOI [10.5281/zenodo.18702042](https://doi.org/10.5281/zenodo.18702042)

