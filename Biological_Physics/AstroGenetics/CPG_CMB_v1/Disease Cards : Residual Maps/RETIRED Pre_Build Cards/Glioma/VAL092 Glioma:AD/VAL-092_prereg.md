# VAL-092 — Pre-registration

**A_terminal on cortical-neuron-discriminating CpGs: glioma vs AD vs healthy reference**

**Date sealed:** 2026-04-26
**RNG seed:** 20260426

---

## Background

VAL-090 (glioma EPIC blood, GSE180683) reported elevated cortical-neuron *cell fraction* in glioma plasma cfDNA (1.092% vs 0.276% healthy reference, d=+1.96, n=76 glioma, n=177 healthy reference) using Loyfer 2023 array atlas NNLS deconvolution. VAL-091 (AD plasma, GSE153712 / GSE144858 / GSE53740) reported AD cortical-neuron fraction null using the same atlas.

Neither VAL-090 nor VAL-091 separately reported A_terminal on the patient β at the cortical-neuron-discriminating CpG positions. That measurement is a per-class A-score on the cell-of-interest signal, computed against H_min(terminal) = 0.7728. It is the natural Stage 2 per-class A-score for terminal class, applied to cortical-neuron specifically, and is required by README_MASTER §Stage 2 lines 209-217 ("for each tissue, compute per-tissue A-score using that tissue's class H_min") under the run-everything architecture.

This pre-registration locks the analysis prior to any β access.

---

## Hypotheses

**H_A — fraction-only signal (null on architectural drift).** Glioma plasma has more cortical-neuron-derived cfDNA fragments, but those fragments come from architecturally healthy cortical neurons and read at A_terminal close to the Loyfer Cortical_neurons reference baseline. The signal is purely "more brain DNA in the blood." AD shows no fraction elevation and also no architectural drift on cortical-neuron-discriminating CpGs.

**H_B — fraction-plus-architectural-drift signal in glioma, null in AD.** Glioma plasma has both elevated cortical-neuron fraction (already established VAL-090) AND patient β at cortical-neuron-discriminating CpGs that reads with elevated A_terminal vs healthy reference. AD remains null on both measures. This is the discriminator-pattern outcome — glioma carries an architectural fingerprint that AD does not.

**H_C — drift in both glioma and AD.** Both diseases show A_terminal elevation at cortical-neuron-discriminating CpGs, magnitude differing. Differential becomes magnitude-based, not categorical.

**H_D — drift in AD, not glioma (surprise).** Would force a re-examination of the AD cortical-neuron pathway under array resolution and would reverse current AD-vs-glioma framing.

---

## Method

**Reference atlas:** Loyfer 2023 array atlas (`reference_atlas.csv` from `nloyfer/meth_atlas`). 25 cell types, 7,890 array-indexed CpGs.

**Discriminating CpG identification:** For each CpG in the Loyfer atlas, compute |β(Cortical_neurons) − mean(β(other 24 cell types))|. Sort descending. Take top **N = 100** CpGs as the cortical-neuron discrimination panel. (Loyfer's own deconvolution uses 25 markers per cell type per their paper; 100 is chosen for denser entropy estimate. Sensitivity to N reported as a check.)

**A_terminal computation per sample:** For each patient β vector at the identified marker CpGs:
  A_terminal = mean(H(β) / H_min(terminal))
  where H(β) = −β·log₂(β) − (1−β)·log₂(1−β) and H_min(terminal) = 0.7728.

**Cohorts (sample-level metadata already on disk):**
1. **GSE51057 healthy reference** (EPIC-Italy buffy coat, n=329 cancer-free).
2. **GSE180683 glioma EPIC blood** (n=76 glioma + n=177 healthy reference). Sub-strata: GBM (new), LGG (new), recurrent GBM, post-therapy.
3. **GSE60274 glioma 450K tissue** (n=66 GBM + n=4 NTB control). Tissue, not blood.
4. **GSE153712 AIBL AD** (n=161 AD + n=471 HC).
5. **GSE144858 AddNeuroMed AD** (n=93 AD + n=96 HC).
6. **GSE53740 GIFT tauopathy** (n=15 AD + n=193 HC + n=176 FTD/PSP/CBD other tauopathies).

**Within-cohort contrasts:** for each cohort with case/control structure, compute Cohen's d (case vs HC) on A_terminal. Bootstrap 95% CI, 10,000 resamples, RNG seed 20260426.

**Cross-cohort baseline check (CHK-3.2):** healthy mean A_terminal in each cohort vs anchor (GSE51057 healthy reference). If cross-cohort baseline differs by >1 SD of either group's HC SD, flag as cross-cohort baseline mismatch and rely on within-cohort contrasts as primary evidence.

**β distribution sanity (CHK-3.1):** before scoring each cohort, dump β distribution shape for 3 sample rows. If <20% extremes AND >40% in [0.4, 0.6], halt and flag (residual / processed data, not raw β).

**Saturation flag (CHK-3.5):** for terminal-class A-scores, A_ceiling depends on the H_min — but there is no published terminal-class A_ceiling (only immune/cycling/secretory have published ceilings per TESTING_CHECKLIST.md). Report distance to the implicit ceiling 1/H_min(terminal) = 1.2941 for each per-sample A_terminal so the reader can see headroom. Flag if A_terminal ≥ 1.289.

**Tier vocabulary:** the universal six-tier vocabulary is used for tier calls — BELOW_NORMAL / NORMAL / MARGINAL / DETECTABLE / URGENT / FLOOR_BREACH. Threshold defined relative to within-cohort HC mean ± SD.

---

## Pre-locked decision criteria

| Outcome | Glioma blood A_terminal vs HC | AD blood A_terminal vs HC | Hypothesis supported |
|---|---|---|---|
| O1_DRIFT_DISCRIMINATOR | d ≥ +0.5, CI excludes 0 | \|d\| ≤ +0.3, CI includes 0 | H_B — clean discriminator |
| O2_BOTH_DRIFT | d ≥ +0.5 | d ≥ +0.5 | H_C — magnitude differential |
| O3_FRACTION_ONLY | \|d\| ≤ +0.3 | \|d\| ≤ +0.3 | H_A — fraction-only signal |
| O4_INVERSE_DRIFT | d < −0.3 | d ≤ +0.3 | Surprise — homogenization in glioma |
| O5_AD_DRIFT_ONLY | \|d\| ≤ +0.3 | d ≥ +0.5 | H_D — surprise; reversed framing |
| O6_UNEXPECTED | data integrity flag, baseline mismatch, or ambiguous | — | revisit data-integrity stage |

**Cross-cohort secondary outcome:** the Loyfer Cortical_neurons reference β at the marker CpGs scored against H_min(terminal) gives the *reference* A_terminal — call it A_ref. The patient ΔA = A_patient − A_ref tells how far the patient's measured methylation deviates from the healthy cortical-neuron baseline. Report glioma ΔA and AD ΔA as supplementary.

---

## Outputs (committed to GitHub)

1. **`val_092_a_terminal_cortical_neuron.py`** — full source.
2. **`VAL-092_results.json`** — per-cohort summary, within-cohort contrasts, cross-cohort baseline comparisons, saturation status, tier calls.
3. **`VAL-092_per_sample.csv`** — per-sample A_terminal for every patient in every cohort.
4. **`VAL-092_distributions.png`** — distribution plot of A_terminal per cohort.
5. **`VAL-092_outcome.md`** — outcome interpretation per CHK-4.x.
6. **`VAL-092_PREREG_SEAL.txt`** — SHA-256 of this file.

---

## Caveats declared in advance

- **Specimen pathway (CHK-0.5):** glioma blood = peripheral plasma (validated transferability). Glioma tissue = surgical resection (NOT a blood-derived specimen; A-score on tissue is a different physical measurement; report separately).
- **Panel transferability (CHK-2.4):** Loyfer atlas is EPIC-trained for sorted cells. Application to plasma cfDNA is the standard Loyfer 2023 use case. No transferability caveat needed.
- **Cross-platform (CHK-1.2):** GSE51057 healthy ref is 450K. GSE180683 glioma blood is EPIC. GSE60274 glioma tissue is 450K. GSE153712 / GSE53740 are 450K. GSE144858 is EPIC. Loyfer-CpG-subset coverage will differ by platform; report per-cohort marker availability.
- **Cross-cohort baseline (CHK-3.2):** different cohorts use different preprocessing pipelines; cross-cohort absolute A_terminal values are not directly comparable. **Within-cohort contrasts are the primary evidence.**
- **Test 2 placeholder (CHK-2.5):** This VAL is a Stage 2 per-class A-score, not a Stage 1 panel. Test 1/Test 2 distinction does not apply.
- **Substrate scope (CHK-1.5):** This VAL measures methyl-only single-substrate A_terminal. Issue 002 framework predictions for terminal class are 5-substrate combined. Cross-tier comparison requires translation, not assumption.

---

## Pre-registration locked

This pre-registration is sealed before any β-value access on the cohorts above. The SHA-256 of this file at seal-time is recorded in `VAL-092_PREREG_SEAL.txt`. Any post-seal modification voids the prereg and triggers re-registration.
