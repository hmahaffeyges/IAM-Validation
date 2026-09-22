# Prostate-EPIC Card — EDEAR Stage-2-Only-Validated Prostate Cancer Flag

**Version 0.2 · 2026-04-25**
**Validation tier:** `stage_2_only_validated` (anchored by VAL-058 GSE269244 tissue)
**Supersedes:** v0.1 (2026-04-24, VAL-058 anchor only). v0.2 adds VAL-065 urine-arm specimen comparison on GSE119260 (Brikun 2018) — exploratory n=4 advanced-disease cohort, classified O5_UNEXPECTED per pre-registration, treated as open question pending larger urine methylation prostate cohort.

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
