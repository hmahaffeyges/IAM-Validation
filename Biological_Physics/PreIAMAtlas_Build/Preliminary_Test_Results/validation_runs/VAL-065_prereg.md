# VAL-065 Pre-Registration — prostate-epic Urine Specimen Arm

**Sealed:** 2026-04-25 UTC
**Card:** prostate-epic
**Card version target:** v0.2 (urine-arm-added)
**Cohort:** GSE119260 (Brikun et al. 2018 / 2019)
**Platform:** Illumina Infinium MethylationEPIC 850K (GPL21145)
**Sample composition:** 4 prostate cancer patients (advanced stage), 4 specimens per patient (FFPE adjacent-normal benign + FFPE tumor + plasma cfDNA + urine sediment), n=16 samples total, all male, ages 58/66/76/68
**Reference:** Brikun I et al. *Evaluating Liquid Biopsies for Methylomic Profiling of Prostate Cancer.* Submitted 2018; series GSE119260.

## Background and motivation

Prostate-epic v0.1 (anchored by VAL-058 GSE269244, paired d = +0.497 on tumor vs adjacent-normal tissue, n=238) was deliberately released without a urine specimen pathway. The v0.1 README explicitly stated:

> "Urinary prostate cells shed continuously, and published urine-methylation prostate tests (SelectMDx, ConfirmMDx, UroMark) demonstrate that urine outperforms blood for early prostate detection. A urine-specimen Stage 1 variant with prostate-specific H_min and panel would materially improve this card's early-disease sensitivity. Listed as a requirement for prostate-epic v0.2+."

VAL-065 is the v0.2 urine arm. The cohort is small (n=4 patients) but uniquely informative: **the same 4 patients each have urine sediment AND plasma cfDNA AND tumor tissue AND adjacent-normal benign tissue analyzed on the same EPIC 850K platform.** This within-patient design directly tests whether urine sediment recovers more of the tumor architectural signal than plasma cfDNA — the question Brikun 2018 was designed to answer.

We also confirm Brikun's key finding (78.6% urine-tumor overlap vs 62.2% plasma-tumor overlap) at the per-CpG level with the secretory-class A-score framework.

## Hypothesis (pre-stated, before β access)

**H1 (primary):** Urine sediment Xu-538 A-score is closer to tumor tissue Xu-538 A-score than plasma cfDNA Xu-538 A-score is to tumor tissue Xu-538 A-score, in the same 4 patients. This tests whether urine "looks more like the tumor" than plasma does, at the architectural-disruption level.

**H2 (secondary):** Urine sediment recovers a positive Xu-538 immune-class A-score elevation (urine vs benign tissue paired d > 0.3) in advanced prostate cancer. This tests whether the v0.1 prediction "urine outperforms blood for prostate detection" holds at the within-patient paired level.

**H3 (tertiary):** Per-CpG direction preservation rate between tumor and urine ≥ corresponding rate between tumor and plasma. This tests Brikun 2018's headline finding (78.6% vs 62.2% overlap) using our framework's per-CpG direction metric instead of their hypermethylation overlap metric.

## Pre-registered outcome decision matrix

### O1: URINE_VALIDATED_AS_PRIMARY_PROSTATE_SPECIMEN
H1 holds (|A_urine − A_tumor| < |A_plasma − A_tumor| in ≥3 of 4 patients) AND H2 holds (urine vs benign paired d > 0.3) AND H3 holds (urine per-CpG direction preservation rate ≥ plasma rate). Card v0.2 promotes urine to PRIMARY specimen for prostate detection; plasma cfDNA pathway retained as SECONDARY.

### O2: URINE_AND_PLASMA_BOTH_VALIDATED
H1 holds in ≥3 of 4 patients AND H2 holds AND H3 marginal (urine ≥ plasma minus 5 percentage points). Card v0.2 adds urine as ADDITIONAL primary specimen alongside plasma; clinical workflow allows either or both.

### O3: URINE_VALIDATED_AT_LOWER_TIER
H2 holds (urine vs benign paired d > 0.3) but H1 fails. Urine added to card as exploratory specimen with explicit "n=4 cohort, larger validation pending" disclaimer. Tier upgrade pending future cohort.

### O4: URINE_NULL
H2 fails (urine vs benign paired d ≤ 0.3). Urine pathway NOT added to v0.2. v0.1 limitation #3 stays in place; urine listed as continued unmet need.

### O5: UNEXPECTED
Any pattern not matching O1-O4 — for example, urine vs benign d > 0.3 but in NEGATIVE direction (urine A-score LOWER than benign tissue, suggesting clearance of disrupted cells rather than retention). Report numbers honestly; convene with Heath before deciding card update direction.

## Methods (frozen pre-data)

### Cohort
- GEO accession: GSE119260
- Sample IDs (from public GEO metadata):
  - Patient 1 (age 58): GSM3362390 (FFPE_benign), GSM3362394 (FFPE_tumour), GSM3362398 (Plasma_cfDNA), GSM3362402 (Urine_sediment)
  - Patient 2 (age 66): GSM3362391, GSM3362395, GSM3362399, GSM3362403
  - Patient 3 (age 76): GSM3362392, GSM3362396, GSM3362400, GSM3362404
  - Patient 4 (age 68): GSM3362393, GSM3362397, GSM3362401, GSM3362405

### Panel
- Xu-538 immune panel (canonical SHA: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6)
- Same panel as VAL-058. Same secretory-class scoring as VAL-058 was prostate-tissue secretory.

### Note on class assignment
Prostate-epic Stage 1 fires on the IMMUNE class via Xu-538 (peripheral immune drift). VAL-058 used Xu-538 on tissue and demonstrated the panel separates tumor from adjacent-normal at d = +0.497 on n=238 — i.e., Xu-538 has cross-substrate transfer. VAL-065 tests the same Xu-538 panel on three additional substrates within-patient: tumor tissue, benign adjacent tissue, plasma cfDNA, urine sediment.

### H_min for scoring
For urine sediment and plasma cfDNA: H_min(immune) = 0.838889 (per the universal_reference block, Xu-538 is an immune-class panel).
For tumor and benign tissue: also H_min(immune) = 0.838889 (same panel, same scoring; VAL-058 used this).

### Pre-registered metrics
- **M1: A-score per (patient, specimen)** — Xu-538 pooled-entropy A-score on each of 16 samples.
- **M2: Within-patient signal recovery distance** — for each patient, |A_urine − A_tumor| and |A_plasma − A_tumor|. Smaller distance = better tumor-mimicry.
- **M3: Urine vs benign paired d** — paired Cohen's d of A_urine − A_benign across 4 patients.
- **M4: Plasma vs benign paired d** — paired Cohen's d of A_plasma − A_benign across 4 patients (companion comparison).
- **M5: Per-CpG direction preservation** — for each Xu-538 CpG, sign of (β_tumor − β_benign) compared to sign of (β_urine − β_benign) and sign of (β_plasma − β_benign), averaged across patients. Direction-preservation rate = fraction of CpGs where signs match.

### QC threshold
- Minimum 400/538 valid Xu-538 CpGs per sample (≥74% panel coverage); skip sample if fewer.

### RNG seed
- 20260425 (today's date). Used only for any tie-breaking in CI calculations; analysis is deterministic.

### Statistical procedure
- All paired tests use Cohen's d with 95% CI (small-sample formula because n=4).
- p-values reported but not gated; with n=4 per arm, the discriminating signal is effect size and within-patient distance ranking, not p < 0.05.
- We will NOT compute large-sample t-tests on n=4. We WILL report exact within-patient ranks.

### What this run does not do
- No subsetting by Gleason grade (sample size precludes).
- No age regression (n=4, age range 58-76, regression not informative).
- No batch correction beyond what's already in the published β matrix.
- No claims about pre-diagnostic detection — all 4 patients have advanced disease at sampling.
- No PSA correlation — PSA values not provided in the public sample metadata.
- No claim that 4-patient sample size establishes a deployment-ready urine assay. The result, whatever direction it points, sets the v0.2 specimen pathway and identifies whether a larger urine cohort search/build is the right priority-1 next step.

## Reproducibility anchors

- Pre-registration SHA-256: (computed at seal)
- Cohort SHA: (computed from sorted GSM list at seal)
- Xu-538 panel SHA: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
- β matrix SHA: (computed at run)
- Results SHA: (computed at output)

## Deliverables

1. `val065_prostate_epic_urine_arm.py` — reproducible Python 3 stdlib script
2. `VAL-065_prereg.md` — this document, sealed
3. `VAL-065_outcome.md` — outcome doc with per-patient, per-specimen, per-CpG numbers
4. `VAL-065_results.json` — primary M1-M5 results
5. `GSE119260_manifest.json` — sample-to-specimen map

GitHub destination: `Biological_Physics/validation_runs/`
Cookbook destination: `prostate-epic_README.md` v0.2 + `prostate-epic_card_v0.2.json` + `GAPE_Evidence_Report_UPDATED.html` + `README_MASTER_v2.1.md`
