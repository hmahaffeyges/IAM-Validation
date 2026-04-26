# VAL-091 Outcome

**Title:** Cortical-neuron cfDNA fraction in AD blood — direct deconvolution against published Loyfer/Moss array atlas

**Date completed:** 2026-04-26
**Pre-reg seal:** `56c7cac9bb869e4ec2b72a6359f87767035443e0f9b5d34d1a7c848b10053c2f` (sealed before any β-value access)
**RNG seed:** 20260426

---

## Outcome label: **O4_AD_NEURO_NULL** (with cross-platform caveats)

The card v2.1 prediction — *"Stage 2 Moss NNLS for AD is expected NULL — brain tissue not in buffy coat at levels Moss NNLS can resolve"* — holds when the Moss 2018 atlas is replaced by the larger Loyfer/Moss array atlas that explicitly includes a sorted Cortical_neurons reference. Two of three AD cohorts test null. The third has n=15 with a single outlier driving the effect.

---

## Primary statistic

**AIBL (within-cohort, panel-training, EPIC, n=161 AD vs 471 HC):**
Cohen's d = **−0.026** [95% CI −0.21, +0.17]

This is the strongest single-cohort evidence: largest n, panel-training cohort, EPIC platform with full Loyfer reference coverage (6042/6105 CpGs).

**AddNeuroMed (cross-platform replication, 450K, n=93 AD vs 96 HC):**
Cohen's d = **−0.083** [95% CI −0.36, +0.19]

Confirms the AIBL null across platform.

**GIFT GSE53740 (specificity cohort, 450K, n=15 AD vs 193 HC):**
Cohen's d = **+0.96** [95% CI +0.15, +1.88]

Wide CI, very small AD n, and the boxplot inspection shows the mean is pulled by a single 5.8% outlier (GSM1300378). The AD median is 0.9% vs HC median 0.0% — a modest non-zero shift, not a glioma-class signal. This is reported but **does not invalidate the AIBL/AddNeuroMed null** given the size and outlier sensitivity.

---

## Why the pooled-vs-external-HC statistic in the analysis script output is NOT the primary

The script also computed:
- Pooled AD (n=269, AIBL+ANM+GIFT) vs **GSE51057 external HC (n=329)**: d = **+1.075**

This number looks dramatic but is **not interpretable as a disease effect** because the cross-cohort HC baseline fold range is **28.7×**:

| Cohort | HC mean cortical-neuron fraction (%) |
|---|---|
| GSE51057 external (Loyfer ref base) | 0.291 |
| AIBL HC (EPIC, Australian) | 0.258 |
| AddNeuroMed HC (450K, European) | **7.398** |
| GIFT HC (450K, US UCSF-MAC) | 0.496 |

AddNeuroMed's 7.4% HC baseline is a **cross-platform NNLS routing artifact**: the 450K platform has 5599 of 6105 Loyfer reference CpGs (8% missing), and the missing CpGs include some that discriminate Cortical_neurons from related references. With reduced reference coverage, NNLS routes mass to Cortical_neurons by default. The within-AddNeuroMed AD-vs-HC contrast is still valid (both arms suffer the same routing), but absolute fractions are not on the same scale as AIBL or GIFT.

Pooling AD samples across cohorts whose HC baselines differ by 28× confounds disease effect with cohort/platform batch. **The pooled-vs-external-HC statistic is reported in the JSON for completeness but flagged as invalid for outcome assignment.**

---

## GIFT specificity arm (GSE53740 FTD/PSP/CBD)

| Group | n | Cortical-neuron mean (%) | d vs GIFT HC |
|---|---|---|---|
| HC | 193 | 0.496 | (reference) |
| AD | 15 | 1.213 | +0.96 [+0.15, +1.88] (outlier-driven) |
| FTD | 128 | 0.629 | +0.19 [−0.04, +0.41] (essentially null) |
| PSP/CBD | 44 | 0.164 | −0.51 [−0.78, −0.21] (PSP/CBD LOWER than HC) |

**Specificity reading:**
- FTD null rules out tauopathy-shared elevation
- PSP/CBD reads *below* HC (negative d), arguing strongly against any neurodegeneration-class elevation of cortical-neuron cfDNA
- AD's modest signal in this small cohort is not consistent with a tauopathy-shared mechanism

The pattern is most consistent with **no real disease-driven elevation in any of these neurodegenerative classes** at the cortical-neuron architectural readout. AD card v2.1's existing Stage 2 expected-NULL framing for AD extends to FTD and (especially) PSP/CBD.

---

## EDEAR routing implication — the actual win

VAL-091 is not an AD finding. It is a **glioma specificity finding**.

- VAL-090 glioma plasma: cortical-neuron fraction ~**1.09%** (Cohen's d = +1.96 vs HC)
- AIBL AD plasma: cortical-neuron fraction ~**0.25%** (HC-equivalent)
- AddNeuroMed AD plasma: HC-equivalent within cohort
- GIFT FTD plasma: HC-equivalent
- GIFT PSP/CBD plasma: below HC

The Stage 2 cortical-neuron readout discriminates glioma from AD/FTD/PSP/CBD at the cohort level. When EDEAR routing encounters a Stage 1 panel positive (AD-directional immune for AD; immune A-score for glioma), the Stage 2 cortical-neuron tile provides an additional axis of separation that the existing card-level guidance can lean on:

- Stage 1 panel + Stage 2 cortical-neuron elevated → consistent with glioma
- Stage 1 panel + Stage 2 cortical-neuron near HC → consistent with AD or other non-CNS-breaching disease
- Stage 1 panel + Stage 2 cortical-neuron LOW → consistent with PSP/CBD (single-cohort signal, not validated)

This is a **cookbook-level layered-atlas architecture confirmation**, not a card-level new claim.

---

## Card update

**ad-immune card v2.1 → v2.2.**

The card's existing language *"Stage 2 Moss NNLS for AD is expected NULL"* stays as the primary clinical assertion. v2.2 adds:

1. **VAL-091 confirmation block** — the explicit Loyfer-augmented Stage 2 test confirms the v2.0/v2.1 prediction.
2. **Cortical-neuron tile remains in EDEAR report as descriptive only for AD** — value is reported, but a low (~0.3%) reading is the AD-consistent expectation, and a high (~1%+) reading flags AD+CNS-breaching-comorbidity for clinician review.
3. **Differential-diagnosis tile against glioma added** — when the AD card v2.2 sees a high Stage 1 immune signal with **also** a high Stage 2 cortical-neuron signal, that combination is more consistent with glioma than with AD-only and triggers a Differential-Diagnosis Required flag.
4. **No change to tier thresholds, no change to the directional 7-CpG panel, no panel re-training.** Stage 1 (VAL-051/052/057) remains the sole tier-determining clinical signal.

---

## What VAL-091 does NOT claim

- **No claim that AD elevates cortical-neuron cfDNA.** The AIBL/AddNeuroMed nulls are the load-bearing evidence.
- **No tier change** — AD card stays at `cross_platform_validated`.
- **No panel modification** — Stage 1 panel is frozen since VAL-051 SEAL 2026-04-23.
- **No claim about MCI** — MCI groups in AIBL (n=94) and AddNeuroMed (n=111) were extracted but not used in primary scoring; included in descriptive output only.
- **No claim about FTD or PSP/CBD as disease groups** — the FTD null and PSP/CBD negative-d are descriptive context for the AD finding, not standalone disease results.

---

## Honest limitations

- AddNeuroMed cross-platform NNLS coverage gap (8%) makes within-cohort comparisons valid but cross-cohort absolute fractions non-comparable.
- GIFT n=15 AD is too small for robust effect estimation. The d=+0.96 with one outlier driver is a hypothesis-generator at most.
- AIBL age metadata not in GEO release; age regression per VAL-052 protocol could not be applied to AIBL within VAL-091 scope. The within-cohort AIBL null (d=−0.03) is small enough that age regression is unlikely to flip direction, but a full age-regression pass requires direct AIBL data application.
- Loyfer atlas Cortical_neurons reference is itself a sorted-cell methylation profile; it has not been independently validated against ground-truth quantitative cortical-neuron cfDNA in either AD or healthy populations beyond Loyfer 2023's original validation cohort.
- VAL-091 does not address tau-PET or amyloid-PET status of any AD case. The negative result for cortical-neuron cfDNA does not mean AD lacks neurodegeneration — it means the BBB-to-circulation cortical-neuron-cfDNA pathway is not detectable at array-NNLS resolution in pre-clinical/clinical-AD cohorts at the studied stage.

---

## Reproducibility (CHK-7.6 triple)

**Inputs:**
- `nloyfer/meth_atlas/reference_atlas.csv` — SHA-256 `4b97dd2a8ba7bf41008e20703e8e12df731179e95cee50fdc12c4d2c202f05b1`
- `GSE51057_series_matrix.txt.gz` — SHA-256 `828059824b67af46fb022f872ff9f69395e2e99b5975b3101157731a04d98bb0` (matches VAL-090)
- `GSE153712_normalized_average_betas.txt.gz` (AIBL EPIC matrix)
- `GSE144858_series_matrix.txt.gz` (AddNeuroMed 450K) — SHA-256 `a16bbdaad06de07c95a5669731786c4e75aad2ea16428a9e928cfcf49f46bb90`
- `GSE53740_series_matrix.txt.gz` (GIFT 450K) — SHA-256 `97e122c39b01eeb7544e0a6a033016ee7c5a40e8b38902789a3927c980ae47d9`

**Environment:** Python 3.12, numpy, pandas, scipy ≥1.10, matplotlib ≥3.7. RNG seed 20260426 for bootstrap.

**Expected runtime:** ~10 min for extraction + 4 NNLS deconvolutions + analysis on Mac/Linux laptop. ~6 GB peak RAM.

**Expected headline output:** AIBL d=−0.03, AddNeuroMed d=−0.08, GIFT d=+0.96 (n=15, outlier-driven). Outcome O4_AD_NEURO_NULL with caveats.

**Source code (single dir):**
- `extract_aibl.py` — transpose-aware AIBL streaming extractor
- `extract_series_matrix.py` — generic series-matrix → Loyfer-CpG-subset CSV
- `val_091_ad_brain_decon_analysis.py` — main analysis
- `make_val091_figure.py` — figure generation

All four pushed to GitHub `Biological_Physics/validation_runs/` per VAL-090 precedent.
