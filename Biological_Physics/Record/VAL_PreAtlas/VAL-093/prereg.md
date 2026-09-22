# VAL-093 — Pre-registration

**Full 18-tissue Stage 2 NNLS deconvolution at >10yr breast pre-diagnostic window**

**Date sealed:** 2026-04-26
**RNG seed:** 20260426
**Run-everything architecture:** This VAL is designed under CCL-033 — every IDAT is processed against ALL Stage 2 tissue tiles regardless of which disease the cohort is anchored to.

---

## Background

VAL-047 Phase 6 Deep Audit on GSE51057 reported secretory-class A-score variance reduction at >10yr breast pre-diagnostic window with d = −1.226 (p = 3e-4, the strongest single-window effect in the breast pre-diagnostic record). This was a class-aggregated finding combining four secretory-class tissues (breast_ductal, hepatocyte, pancreatic, prostate) without separating the individual tissue contributions.

Under the run-everything architecture (CCL-033, signed off 2026-04-26), every IDAT runs Stage 1 + Stage 2 + Stage 3 with all panels and reference atlases regardless of any single-stage result. Per-class A-scores are computed for every tissue tile every IDAT. VAL-093 applies this architecture to the >10yr breast pre-diagnostic window and asks: at the >10yr window, which Stage 2 tissue tile shows the strongest signal? Is the secretory-class signal localized to breast specifically, or distributed across multiple tissues?

The expected answer if the framework prediction holds: breast (Loyfer's `Breast` tile) shows the strongest individual-tissue signal at >10yr, because the disease being detected is breast cancer pre-clinical biology specifically. If the signal is distributed, the secretory-class aggregate may have been carrying multi-tissue noise that happened to align in one direction.

---

## Hypotheses

**H_A — breast-localized signal.** At the >10yr breast pre-dx window, the Loyfer `Breast` tile shows the strongest single-tissue d (case-vs-HC), with comparable or smaller effects on other secretory-class tissues (hepatocyte, prostate, pancreas). Top-1 ΔA call at the patient level returns `Breast` as the most-departed tile in the majority of >10yr breast pre-dx cases.

**H_B — secretory-class-distributed signal.** All four secretory-class tissues show comparable d magnitudes at the >10yr window. The class-aggregate d=−1.226 reflects a real class-level signal that does not localize to breast specifically. Top-1 ΔA calls are distributed across secretory tissues.

**H_C — non-secretory tissue signal.** A non-secretory-class tissue (e.g. cortical_neurons, vascular_endothelial, immune cells) shows the strongest d at >10yr, suggesting the secretory-class aggregate was capturing a different biology than disease-of-interest tissue.

**H_D — null at the per-tissue level.** No individual tissue tile shows |d| ≥ 0.5 at the >10yr window, despite the secretory-class aggregate showing d=−1.226. This would suggest the class-aggregate signal is a class-level statistical phenomenon not visible at the per-tissue level.

---

## Method

**Reference atlas:** Loyfer 2023 array atlas (`reference_atlas.csv`, SHA `4b97dd2a8ba7…`). 25 cell types, 7,890 array-indexed CpGs.

**NNLS deconvolution:** for each patient β vector at the Loyfer atlas CpGs, solve `f @ atlas ≈ patient_β` with `f ≥ 0`, `sum(f) = 1`. Output: per-patient cell-type fraction vector across 25 Loyfer cell types.

**Per-tissue per-class A-score:** for each cell type, identify the top-100 discriminating CpGs (max |β(target_cell) − mean(β(other_cells))|). For each patient β vector at those marker CpGs, compute A_class = mean(H(β) / H_min(class)) where H_min is the architecture-class H_min for that cell type:

| Loyfer cell type | Architecture class | H_min |
|---|---|---|
| Cortical_neurons | terminal | 0.7728 |
| Left_atrium | terminal | 0.7728 |
| Hepatocytes | secretory | 0.843264 |
| Breast | secretory | 0.843264 |
| Prostate | secretory | 0.843264 |
| Pancreatic_acinar_cells | secretory | 0.843264 |
| Pancreatic_duct_cells | secretory | 0.843264 |
| Pancreatic_beta_cells | secretory | 0.843264 |
| Thyroid | secretory | 0.843264 |
| Bladder | cycling | 0.856055 |
| Colon_epithelial_cells | cycling | 0.856055 |
| Lung_cells | cycling | 0.856055 |
| Head_and_neck_larynx | cycling | 0.856055 |
| Upper_GI | cycling | 0.856055 |
| Uterus_cervix | cycling | 0.856055 |
| Kidney | cycling | 0.856055 |
| Adipocytes | stromal | 0.862950 |
| Vascular_endothelial_cells | stromal | 0.862950 |
| Erythrocyte_progenitors | progenitor | 0.852216 |
| Monocytes_EPIC | immune | 0.838889 |
| B-cells_EPIC | immune | 0.838889 |
| CD4T-cells_EPIC | immune | 0.838889 |
| NK-cells_EPIC | immune | 0.838889 |
| CD8T-cells_EPIC | immune | 0.838889 |
| Neutrophils_EPIC | immune | 0.838889 |

H_min values frozen from G-002 + G-003b MCMC posteriors (R-hat < 1.001). Class assignments from GAPE_WEB_v13 architecture-class definitions.

**Stratification window:** breast cases (cancer_site = c50) with ttd_years > 10 in either cohort:
- GSE51057 (Phase 9): 11 breast cases at >10yr
- GSE51032 (Phase 12): 36 breast cases at >10yr
- Combined: n = 47 breast >10yr pre-dx cases
- HC reference: 177 (GSE51057 controls) + 424 (GSE51032 controls) = 601 healthy buffy coat samples

Within-cohort case-vs-control on each cohort separately (CCL-034 — within-cohort statistics primary). Cross-cohort pooled comparison reported as secondary.

**Top-1 ΔA call per patient:** for each patient, identify the tissue tile with the largest |A_patient − A_HC_mean| (max absolute departure from healthy reference, computed within the patient's own cohort). Report the distribution of top-1 calls across the >10yr breast pre-dx case group.

**Cross-cohort baseline check (CHK-3.2 mandatory):** healthy mean A per tile, both cohorts, anchor-SD units. Flag any tile with |delta| > 1 anchor-SD. Within-cohort statistics remain primary.

---

## Pre-locked decision criteria

| Outcome | Within-cohort case-vs-HC d criteria | Hypothesis supported |
|---|---|---|
| O1_BREAST_LOCALIZED | Loyfer `Breast` d (case vs HC) ≥ +0.5 OR ≤ −0.5; |d| on `Breast` is largest in absolute value among 25 tiles in either cohort | H_A |
| O2_SECRETORY_DISTRIBUTED | At least 3 of 4 secretory-class tissues (Breast, Hepatocyte, Prostate, Pancreas) show |d| ≥ 0.3, with `Breast` not uniquely largest | H_B |
| O3_NON_SECRETORY_DOMINANT | A non-secretory tissue tile shows |d| ≥ 0.5 with magnitude greater than `Breast` and other secretory tiles | H_C |
| O4_PER_TISSUE_NULL | All individual tile |d| < 0.5 | H_D |
| O5_BIDIRECTIONAL_PATTERN | `Breast` shows |d| ≥ 0.5 in one cohort but opposite-direction non-trivial signal in the other | unexpected — investigate cohort-specific or platform-specific factors |
| O6_UNEXPECTED | Data integrity flag, baseline mismatch ≥ 3 anchor-SDs, or unparseable result | revisit data-integrity stage |

**Top-1 distribution criterion (descriptive, not outcome-determining):** report fraction of >10yr breast pre-dx patients whose top-1 ΔA call is `Breast` vs other tiles. If majority is `Breast`, supports H_A; if distributed, supports H_B.

---

## Outputs

1. **`val_093_full_18tissue_stage2_breast_predx.py`** — full source.
2. **`VAL-093_results.json`** — per-cohort per-tile statistics, within-cohort contrasts, cross-cohort baseline checks, top-1 distribution.
3. **`VAL-093_per_sample.csv`** — per-patient: cohort, group, ttd_years, A-score per tile, top-1 call.
4. **`VAL-093_tile_heatmap.png`** — 25-tile by case/HC heatmap visualization.
5. **`VAL-093_outcome.md`** — outcome interpretation per CHK-4.x.
6. **`VAL-093_PREREG_SEAL.txt`** — SHA-256 of this file.

---

## Caveats declared in advance

- **Specimen pathway (CHK-0.5):** GSE51057 and GSE51032 are EPIC-Italy buffy coat blood. Validated transferability for Xu-538 and Loyfer atlas.
- **Platform:** both cohorts are 450K — same platform, removing cross-platform variability for this analysis.
- **Cross-cohort baseline (CHK-3.2):** both cohorts are EPIC-Italy nested-case-control samples, processed by the EPIC-Italy preprocessing pipeline. Cross-cohort baseline expected to be tight; if it isn't, flag.
- **Within-cohort vs cross-cohort hierarchy (CCL-034):** within-cohort case-vs-control is primary. Cross-cohort pooled is secondary.
- **Top-100 marker count for per-tissue A-score:** consistent with VAL-092 choice. Sensitivity to N is a recommended follow-up but not blocker.
- **Class-aggregate vs per-tissue:** class-aggregate A_secretory = mean of breast + hepatocyte + prostate + pancreas individual A's. The class-aggregate d=−1.226 from VAL-047 Phase 6 may average over heterogeneous per-tissue signals. VAL-093 separates them.

---

## Pre-registration locked

This pre-registration is sealed before any β-value access on the cohorts above. The SHA-256 of this file at seal-time is recorded in `VAL-093_PREREG_SEAL.txt`. Any post-seal modification voids the prereg and triggers re-registration.
