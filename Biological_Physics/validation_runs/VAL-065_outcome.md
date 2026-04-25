# VAL-065 Outcome — prostate-epic Urine Arm Specimen Comparison

**Date:** 2026-04-25
**Cohort:** GSE119260 (Brikun 2018) — Illumina EPIC 850K, n=16 samples (4 patients × 4 specimens)
**Card:** prostate-epic v0.2 (urine-arm-attempted)
**Pre-registration SHA:** `f1d1a99770396f217d636dd4e04e9d2162b1ee186bcacb8841b96e95ffcf437d`
**Manifest SHA:** `1b0eafcf8b34ece8168a3c4d0bf02c4bb20266092fe392db6b3f5698de6ef0ee`
**Status:** **EXPLORATORY — n=4 advanced-disease cohort, larger urine cohort required for substrate-vs-substrate conclusions**

## TL;DR

The only public urine-methylation prostate cancer cohort on GEO with EPIC 850K platform is GSE119260 (Brikun 2018), n=4 advanced-stage metastatic patients. Sample size precludes statistical inference about specimen-to-specimen comparison. **VAL-065 is treated as exploratory and the urine arm of prostate-epic remains an open question pending a larger urine methylation cohort.**

The pre-registered hypothesis (H2: urine vs benign paired Cohen's d > +0.3 in positive direction) failed in an unexpected direction — observed urine vs benign paired d = −2.39, with urine A-score dramatically lower than benign tissue A-score in all 4 patients. Per the pre-registered outcome decision matrix, this falls under O5_UNEXPECTED, which the prereg explicitly listed as: *"urine vs benign d > 0.3 but in NEGATIVE direction (urine A-score LOWER than benign tissue, suggesting clearance of disrupted cells rather than retention). Report numbers honestly; convene with Heath before deciding card update direction."*

After reviewing the result with Heath, the verdict is: cohort is too small (n=4) and too advanced-disease-skewed to draw any framework-relevant interpretation. Document as exploratory open question, refresh prostate-epic to v0.2 capturing only the VAL-058 as-built tissue-validated state, and identify "larger urine methylation prostate cohort with mixed disease stages and healthy controls" as the priority-1 unmet data need for prostate-epic v0.3+.

## Cohort

GSE119260, "Evaluating Liquid Biopsies for Methylomic Profiling of Prostate Cancer," Brikun I et al. 2018 (PMID 32000564, University College Dublin / Antoinette Perry group). Public access on GEO since April 22, 2020. Series matrix downloaded from `ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE119nnn/GSE119260/matrix/`.

**Design.** 4 prostate cancer patients, all male, 4 specimens per patient: FFPE adjacent-normal benign + FFPE primary tumor + plasma cfDNA + urine sediment. 16 samples total on Illumina Infinium MethylationEPIC 850K (GPL21145).

**Patient characteristics — important provenance.**

| Patient | Age | Gleason | PSA (ng/mL) | Metastases |
|---|---|---|---|---|
| P1 | 58 | 4+4 (LHS+RHS) | **1,400** | Bone |
| P2 | 66 | 5+4 (LHS) / 4+5 (RHS) | 10.9 | Bone |
| P3 | 76 | 4+5 (RHS) | 144 | Bone |
| P4 | 68 | 5+5 (LHS+RHS) | 38.98 | Bone |

**All 4 patients have bone metastatic disease.** This is NOT a pre-diagnostic cohort. This is NOT a localized-prostate-cancer cohort. PSA range 10.9 to 1,400 ng/mL spans an enormous biological range — P1 with PSA 1,400 is in fulminant disease, P2 with PSA 10.9 is at the screening-detectable threshold. Gleason scores are uniformly aggressive (4+4 to 5+5).

## Methods

Xu-538 immune panel (canonical SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`, file-bytes verified at load). Pooled-entropy A-score `H(β)/H_min(immune)` with `H_min(immune) = 0.838889` (G-002 MCMC posterior). All four specimens from all four patients passed QC (≥400 valid Xu-538 CpGs per sample; 435 of 538 panel CpGs measured in all samples; 100% coverage). Paired Cohen's d with Hedges small-sample correction. Per-CpG direction preservation rate computed as fraction of Xu-538 CpGs whose 4-patient majority sign of (β_specimen − β_benign) matches the 4-patient majority sign of (β_tumor − β_benign).

## Results

### M1: Per-sample Xu-538 immune-class A-score

| Patient | Benign | Tumor | Plasma | Urine |
|---|---|---|---|---|
| P1 (PSA 1400) | 0.802 | 0.805 | 0.768 | 0.673 |
| P2 (PSA 10.9) | 0.805 | 0.741 | 0.636 | 0.629 |
| P3 (PSA 144) | 0.781 | 0.727 | 0.841 | 0.523 |
| P4 (PSA 38.98) | 0.604 | 0.713 | 0.744 | 0.505 |

**Observation 1: Tumor A-score does not consistently exceed benign A-score in this n=4 cohort.** P1 tumor (0.805) > benign (0.802) by +0.003. P2 tumor (0.741) < benign (0.805) by −0.064. P3 tumor (0.727) < benign (0.781) by −0.054. P4 tumor (0.713) > benign (0.604) by +0.109. Direction is mixed across 4 patients. Tumor vs benign mean ΔA = −0.001, paired d = −0.016. The expected positive tumor signal from VAL-058 (n=238, paired d = +0.497) is not visible at n=4.

**Observation 2: Urine A-score is dramatically lower than benign tissue A-score in all 4 patients.** Urine ranges 0.505 to 0.673; benign ranges 0.604 to 0.805. Mean ΔA(urine − benign) = −0.165, paired d = −2.39. This is one of the largest paired effect sizes measured across the entire validation record.

**Observation 3: Plasma cfDNA A-score is highly variable across patients.** Plasma ranges 0.636 to 0.841. P3 plasma (0.841) is HIGHER than P3 benign (0.781), opposite direction to P1/P2/P4. Plasma vs benign paired d ≈ 0.

### M2: Within-patient distance from tumor

| Patient | A_tumor | |A_urine − A_tumor| | |A_plasma − A_tumor| | Closer |
|---|---|---|---|---|
| P1 | 0.805 | 0.132 | 0.037 | Plasma |
| P2 | 0.741 | 0.112 | 0.105 | Plasma |
| P3 | 0.727 | 0.203 | 0.115 | Plasma |
| P4 | 0.713 | 0.208 | 0.031 | Plasma |

**Plasma is closer to tumor than urine is, in all 4 patients (4/4).** Mean |urine − tumor| = 0.164. Mean |plasma − tumor| = 0.072. The pre-registered H1 ("urine closer to tumor in ≥3/4 patients") is rejected with strong directional opposite.

### M3, M4, tumor reference: paired Cohen's d (n=4)

| Comparison | mean ΔA | sd | paired d | Hedges d |
|---|---|---|---|---|
| Tumor vs benign (reference) | −0.0013 | 0.0794 | **−0.016** | −0.012 |
| Urine vs benign | −0.165 | 0.069 | **−2.39** | −1.74 |
| Plasma vs benign | −0.0003 | 0.133 | **−0.002** | −0.001 |

### M5: Per-CpG direction preservation (435 Xu-538 CpGs measured in all samples; 306 with non-zero majority tumor direction)

- Urine direction preservation: **51.31%**
- Plasma direction preservation: **47.39%**
- Brikun 2018 reported (different metric, full ~860K probes): 78.63% urine, 62.21% plasma

The Brikun 2018 paper's headline 78.6% / 62.2% number measures hypermethylation overlap on the full EPIC ~860K probe set using their custom selection criterion. Our 51.3% / 47.4% measures Xu-538 panel + 4-patient majority sign-direction agreement. These are different metrics on different probe sets and are not directly comparable. Both metrics agree directionally (urine slightly outperforms plasma on direction-preservation).

## Pre-registered outcome decision

| Hypothesis | Threshold | Observed | Result |
|---|---|---|---|
| H1: urine closer to tumor than plasma in ≥3/4 patients | ≥3/4 | 0/4 | **FAIL (4/4 plasma closer)** |
| H2: urine vs benign paired d > +0.3 | d > +0.3 | d = −2.39 | **FAIL (wrong direction, |d| = 2.39)** |
| H3: urine per-CpG direction preservation ≥ plasma | urine ≥ plasma | 51.3% vs 47.4% | PASS (marginal) |

Initial classification was O4_URINE_NULL (because H2 failed), but on review this falls under O5_UNEXPECTED per the prereg's explicit O5 case: *"urine vs benign d > 0.3 but in NEGATIVE direction (urine A-score LOWER than benign tissue, suggesting clearance of disrupted cells rather than retention). Report numbers honestly; convene with Heath before deciding card update direction."* The observed |d| = 2.39 with negative sign is squarely in this O5 case. Reclassified.

## Decision (after convening with Heath)

**Cohort is too small (n=4) and too uniform in advanced-disease state (all bone metastatic, Gleason 4+4 to 5+5) to draw any substrate-vs-substrate conclusions.** The dramatic urine−benign signal at d = −2.39 is real in the data but uninterpretable at n=4 — could be specimen physics (urine sediment is sloughed/dying cells with collapsed methylation entropy), advanced-disease-specific phenomenology, FFPE-benign field effect, or noise amplified by the small sample. We cannot distinguish these from the data we have.

**Action:** Document VAL-065 as **exploratory** in prostate-epic v0.2. Do NOT conclude that urine sediment is the wrong substrate for prostate detection. Do NOT promote urine to a primary or secondary card specimen. Identify a larger urine methylation prostate cohort with healthy controls and mixed disease stages as the priority-1 unmet data need for prostate-epic v0.3+. Add the open question to the cookbook lessons-learned catalog as CCL-026.

## What VAL-065 does NOT establish

- Does not establish urine sediment as a valid (or invalid) substrate for prostate detection
- Does not falsify the v0.1 hypothesis that urine outperforms blood for early prostate detection (cohort is not early-stage)
- Does not establish a urine A-score expected direction (positive or negative) for any prostate stage
- Does not provide a deployable urine clinical pathway
- Does not constitute pre-diagnostic blood screening evidence (was never intended to)

## What VAL-065 DOES establish

- The Xu-538 panel can be applied to urine sediment β-value data and produces measurable A-scores at full panel coverage on EPIC 850K (435/538 CpGs measured per sample)
- Within-patient urine, plasma, tumor, and benign tissue can be co-analyzed with the same Xu-538 panel and same H_min(immune) without methodological obstruction
- The 4-patient Brikun 2018 cohort, the only public EPIC 850K urine prostate cohort on GEO, is too small for substrate-vs-substrate inference; this is the primary informational deliverable of VAL-065 — that the public-data ceiling for the urine arm is at n=4 advanced-stage patients

## What VAL-065 should trigger as next steps

1. **Search dbGaP and consortium catalogs for larger urine methylation prostate cohorts** (SelectMDx, ConfirmMDx, UroMark validation studies; PCA3 methylation cohorts; the Movember Foundation urine methylation studies if data is depositable). Many of these may be available under reasonable-use research applications even if they are not publicly downloadable.
2. **Consider partner-lab collection of urine sediment + matched blood EPIC 850K from a local prostate-active surveillance cohort** as part of the L1 lab partnership tier (per universal_reference) — n=20-50 urine samples with matched blood across Gleason 6 / Gleason 7 / Gleason ≥8 strata would be sufficient to draw a substrate verdict. Cost estimate at $50-150/sample × 50 = $2,500-$7,500.
3. **Park the urine arm of prostate-epic in v0.2 as exploratory open question** with VAL-065 cited as the only available data point and CCL-026 documenting the substrate-physics open question.
4. **Do not let VAL-065 affect the VAL-058-anchored stage_2_only_validated tier of prostate-epic.** The VAL-058 finding (tumor vs adjacent-normal tissue paired d = +0.497 on n=238 African-American men) is unaffected by VAL-065 and remains the card's primary anchor.

## Reproducibility

- Pre-registration: `VAL-065_prereg.md` SHA `f1d1a99770396f217d636dd4e04e9d2162b1ee186bcacb8841b96e95ffcf437d`
- Manifest: `GSE119260_manifest.json` SHA `1b0eafcf8b34ece8168a3c4d0bf02c4bb20266092fe392db6b3f5698de6ef0ee`
- Script: `val065_prostate_epic_urine_arm.py` (Python 3 stdlib only)
- Series matrix: GSE119260_series_matrix.txt downloaded from NCBI GEO public access
- Xu-538 panel: file-bytes SHA verified at runtime
- Results JSON: contains all 16 per-sample A-scores, all paired Cohen's d values with CIs, all M5 per-CpG counts, full pre-reg outcome trace
