# VAL-091 Pre-registration

**Title:** Cortical-neuron cfDNA fraction in Alzheimer's disease peripheral blood — direct deconvolution against published Loyfer/Moss array atlas

**Card:** ad-immune-epic v2.2 (pending VAL-091 outcome)
**Date sealed:** 2026-04-26
**Pre-reg author:** Heath W. Mahaffey (with Walther)
**Run trigger:** Per glioma-LL-007 (cookbook-wide layered-atlas implication of VAL-090) and Heath's direction on 2026-04-26 — "should we try and run this on some of the same AD runs just to see how the neuron cells are affected by AD? Wouldn't that be good to understand for trying to determine the culprit when we run blood work for EDEAR?" The Loyfer/Moss array atlas integration that produced VAL-090's d = +1.96 cortical-neuron cfDNA finding in glioma plasma applies cookbook-wide. AD is the highest-priority next test because (1) AD has neuronal involvement at the disease level (neurodegeneration), (2) AD and glioma both read positive on Stage 1 immune A-score and currently cannot be discriminated on Stage 1 alone, (3) a cortical-neuron cfDNA reading on AD blood would establish either AD-vs-glioma Stage 2 specificity or expose AD as also CNS-positive (still useful for EDEAR routing).

---

## Hypothesis

The published `nloyfer/meth_atlas/reference_atlas.csv` (Loyfer/Kaplan group, distributed alongside Loyfer 2023 *Nature* 613:355) contains 26 cell-type references including a sorted-cell `Cortical_neurons` reference indexed to Illumina 450K/EPIC CpG IDs. When applied directly to publicly-deposited AD peripheral blood methylation data via standard NNLS deconvolution, this reference will allow direct quantification of cortical-neuron cfDNA fraction in AD plasma vs healthy reference and vs glioma plasma (VAL-090 reference comparison).

**Three competing hypotheses (locked before scoring):**

- **Hypothesis A — AD elevates cortical-neuron cfDNA at magnitude similar to glioma.** AD-driven neurodegeneration releases neuron DNA into circulation; signal magnitude in AD blood is comparable to glioma blood (d ≥ +1.0 vs healthy reference). If true: cortical-neuron cfDNA is a CNS-disease marker, not glioma-specific. EDEAR Stage 2 cortical-neuron readout flags CNS pathology of any kind; differential glioma-vs-AD requires Stage 3 or clinical context.

- **Hypothesis B — AD does NOT elevate cortical-neuron cfDNA.** AD neurodegeneration kills neurons but the BBB stays sufficiently intact that neuron DNA does not reach systemic circulation at array-detectable levels (d < +0.3). Only BBB-breaching pathologies (glioma, tumor, severe stroke, late-stage neurodegeneration) elevate cortical-neuron cfDNA. If true: cortical-neuron cfDNA discriminates glioma from AD at Stage 2 directly. Major specificity win for EDEAR.

- **Hypothesis C — AD elevates cortical-neuron cfDNA at intermediate magnitude.** AD shows partial elevation (+0.3 ≤ d < +1.0), consistent with progressive BBB compromise correlating with disease stage. If true: cortical-neuron cfDNA fraction is a graded CNS-injury marker, with quantitative magnitude useful for AD-vs-glioma discrimination but with overlap.

**No directional prediction is favored.** The hypothesis is descriptive: characterize the cortical-neuron fraction distribution in AD blood, compare to healthy reference and to glioma plasma (VAL-090).

---

## Cohorts

| Role | Cohort | Accession | Platform | n | Source |
|---|---|---|---|---|---|
| Healthy reference | EPIC-Italy menarche cohort, cancer-free subset | GSE51057 | HM450 | 177 | Re-use VAL-090 cached data |
| AD primary cohort | AIBL (Australian Imaging, Biomarker & Lifestyle, Nabais 2021) | GSE153712 | EPIC 850K | 161 AD / 471 HC (n=726 incl. MCI) | Re-stream from GEO |
| AD cross-platform replication | AddNeuroMed (multi-center European) | GSE144858 | HM450 | 93 AD / 96 HC | GEO public |
| AD specificity vs FTD/PSP/CBD | GSE53740 GIFT (Ferrari 2014) | GSE53740 | HM450 | 193 HC / 15 AD / 128 FTD / 44 PSP/CBD | GEO public |
| Glioma plasma comparison | Salas/Wiencke 2022 (VAL-090 cohort) | GSE180683 | EPIC 850K | 76 | Re-use VAL-090 results |

Re-use of VAL-090 data is explicit: the comparison "AD plasma cortical-neuron fraction vs glioma plasma cortical-neuron fraction (VAL-090)" requires the same atlas, same NNLS solver, same healthy reference. Both runs use cached or freshly downloaded inputs with SHA verification.

---

## Method

1. Use the same `nloyfer/meth_atlas/reference_atlas.csv` from VAL-090 (SHA-256 already recorded).
2. Extract per-sample β matrices from the AD cohorts (AIBL, AddNeuroMed, GSE53740) using same one-pass extraction approach as VAL-090.
3. Run `nloyfer/meth_atlas/deconvolve.py` against each AD cohort with `reference_atlas.csv`.
4. Extract `Cortical_neurons` row per cohort.
5. Compute per-cohort mean, SD, range, percentiles. Compute Cohen's d for AD vs HC within each AD cohort.
6. **Cross-cohort comparison:** compare AD plasma cortical-neuron fraction (from AIBL HC, AddNeuroMed HC, GSE53740 HC pooled) to:
   - VAL-090 healthy reference (GSE51057 HC, n=177)
   - VAL-090 glioma plasma (GSE180683 n=76)
7. **Sub-stratification within AIBL:** AD cases split by sex (matching VAL-051 sex-stratified analysis).
8. **Sub-stratification within GSE53740:** stratified analysis of AD, FTD, PSP/CBD against same HC reference. This is the key specificity test.
9. **Age regression** per VAL-052 protocol: fit linear regression on cortical-neuron fraction vs age in HC, subtract fitted values, recompute Cohen's d on residuals. (Tests whether cortical-neuron elevation tracks chronological age vs disease per se.)
10. Generate distribution figure (multi-panel boxplot: AD cohorts side-by-side, healthy reference, glioma reference).

**No parameter tuning, no panel selection, no post-hoc adjustment.** The reference atlas is fixed external; the NNLS solver is `scipy.optimize.nnls`; the analysis pipeline is the published one from `nloyfer/meth_atlas`.

---

## Decision criteria

Per-cohort and pooled-AD outcome assignment:

- **O1_AD_NEURO_POSITIVE_HIGH** — Pooled AD (AIBL + AddNeuroMed, primary tested cohorts) Cohen's d ≥ +1.0 vs healthy reference. Hypothesis A confirmed: AD elevates cortical-neuron cfDNA at glioma-comparable magnitude. **Card update:** ad-immune card adds Stage 2 cortical-neuron readout as confirmatory secondary signal; report includes cortical-neuron-fraction-specific tier; differential glioma-vs-AD pushed to Stage 3 / clinical context.

- **O2_AD_NEURO_POSITIVE_MEDIUM** — Pooled AD d ∈ [+0.5, +1.0). Hypothesis C confirmed: AD elevates cortical-neuron cfDNA at intermediate magnitude. **Card update:** Stage 2 cortical-neuron readout added with quantitative tier (AD typical 1-2× healthy; glioma typical 4× healthy); differential discrimination noted as graded.

- **O3_AD_NEURO_POSITIVE_LOW** — Pooled AD d ∈ [+0.2, +0.5). Slight elevation, possibly age-confounded or BBB-progression-stage-dependent. **Card update:** Stage 2 cortical-neuron added as descriptive only; not used for tier assignment; future ADNI / Framingham replication required before clinical interpretation.

- **O4_AD_NEURO_NULL** — Pooled AD d < +0.2 (or negative). Hypothesis B confirmed: AD does NOT elevate cortical-neuron cfDNA at array-detectable levels. **Card update:** ad-immune card retains existing Stage 1 directional A-score as sole clinical signal; cortical-neuron NULL becomes a discriminating feature against glioma in EDEAR routing. **Major specificity win.**

- **O5_AD_NEURO_NEGATIVE** — Pooled AD d ≤ −0.3 (AD reads LOWER cortical-neuron fraction than HC). Biologically anomalous; would prompt re-examination of cohort metadata for confounds (treatment effect, sample handling, age confound).

- **O6_UNEXPECTED_PATTERN** — Anything not fitting O1–O5. E.g., AIBL positive but AddNeuroMed null (cohort-specific), or AD null but FTD positive (tauopathy-specific), or sex-asymmetric in unexpected direction.

**FTD and PSP/CBD specificity (GSE53740 sub-arm):** Report Cohen's d for FTD vs HC and PSP/CBD vs HC alongside AD vs HC. **No tier-changing claims** for FTD or PSP based on this single small cohort (FTD n=128 is large enough; PSP/CBD n=44 is modest); the result is descriptive specificity context for the AD finding.

**Age regression (per VAL-052 protocol):** Report both raw Cohen's d and age-regressed d. If age-regressed d drops below half of raw d, the cortical-neuron signal is age-confounded and the finding is reported as such.

---

## Reproducibility (CHK-7.6)

**Inputs:**
- `nloyfer/meth_atlas` reference: `https://github.com/nloyfer/meth_atlas/blob/master/reference_atlas.csv` (SHA-256: `4b97dd2a8ba7bf41008e20703e8e12df731179e95cee50fdc12c4d2c202f05b1`, ~16 MB)
- AIBL GSE153712: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153712/suppl/GSE153712_normalized_average_betas.txt.gz` (~4.8 GB)
- AddNeuroMed GSE144858: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE144nnn/GSE144858/matrix/GSE144858_series_matrix.txt.gz` (~580 MB est.)
- GSE53740: `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE53nnn/GSE53740/matrix/GSE53740_series_matrix.txt.gz` (~1.4 GB est.)
- Healthy reference GSE51057: cached from VAL-090

**Environment:**
- Python 3.12
- numpy ≥1.20, pandas ≥2.0, scipy ≥1.10, matplotlib ≥3.7
- RNG seed: 20260426 (none required for NNLS — deterministic)

**Expected runtime:** ~30 min including downloads. AIBL streaming will dominate (~10 min).
**Expected memory:** ~6 GB peak.
**Expected headline output:** four cortical-neuron fraction means per cohort; Cohen's d AD-vs-HC for AIBL and AddNeuroMed; specificity contrast (AD vs FTD vs PSP/CBD on GSE53740); cross-comparison to VAL-090 glioma cohort.

---

**Pre-registration seal SHA-256:** computed at commit time, stored in `VAL-091_PREREG_SEAL.txt`.
