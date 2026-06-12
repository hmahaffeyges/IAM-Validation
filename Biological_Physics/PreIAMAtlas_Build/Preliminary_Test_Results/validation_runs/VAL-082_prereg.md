# VAL-082 pre-registration — heme-epic v0.1 myeloid arm

**Card:** heme-epic v0.1
**Arm:** myeloid (AML)
**Date sealed:** 2026-04-25
**Cohort:** GSE62298 — Glass et al. 2017, "Genome-scale profiling of the DNA methylation landscape in human AML patients" (n=68 primary AML patients, blood-derived methylation, HM450)
**External healthy comparator:** GSE51057 — EPIC-Italy menarche cohort, buffy coat methylation HM450, cancer-free subset (~177 women without subsequent cancer diagnosis at follow-up)
**Specimen:** blood-derived methylation (AML cohort), buffy coat (healthy comparator)
**n total:** 68 AML + ~177 healthy = ~245 (Italian healthy n subject to QC)

## Design

VAL-082 is the first per-patient myeloid arm validation for heme-epic v0.1. Score Stage 1 universal immune Xu-538 A-score on AML cohort using H_min(immune) = 0.838889 per panc-LL-007 universal pipeline rule. Compare AML A-score distribution to GSE51057 EPIC-Italy healthy women cancer-free subset by unpaired Cohen's d.

**Key clarifications per CCL-032 diagnostic order:**

1. **Data integrity check (CHK-3.1):** β distribution bimodal sanity check on AML cohort. Real raw β has >30% at extremes (<0.1 or >0.9) and <10% in [0.4, 0.6]. Verify before scoring.
2. **Cross-cohort baseline check (CHK-3.2):** healthy mean A across the two cohorts compared. Italian healthy is the comparator; if Hannum 2013 GSE40279 anchor data were available the EPIC-Italy cohort would be cross-checked against it. Italian baseline used as the operational healthy comparator.
3. **Biology consistency check (CHK-4.1):** AML expected to score ABOVE healthy at the immune-class panel because AML myeloid cells exhibit lineage-commitment locus reprogramming detectable by Xu-538 panel. Direction expected positive.

## Pre-locked decision criteria

- **O1_PASS_MYELOID_ARM_AT_BLOOD_LEVEL:** Cohen's d ≥ +0.5 with lower CI > 0, AND ≥30% of AML samples score above healthy 95th percentile.
- **O2_PARTIAL:** 0.2 < d < 0.5 with lower CI > 0.
- **O3_NULL:** CI crosses zero.
- **O5_NEGATIVE_DIRECTION:** d < 0 (AML reads below healthy).
- **O6_UNEXPECTED:** β distribution sanity check fails OR saturation flag fires (A > 1.187 immune ceiling).

## Constants (all sealed)

- Panel: Xu-538 immune
- Panel SHA-256: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
- H_min: 0.838889 (immune class, universal Stage 1 rule per panc-LL-007)
- RNG seed: 20260425
- QC threshold: ≥ 400 valid Xu-538 CpGs per sample
- Confidence interval: 95% normal-approximation
- Significance threshold: p < 0.05 two-sided

## Substrate clarification (important for interpretation)

Issue 002 framework prediction A_AML ≈ 1.10 with ΔA ≈ +0.168 refers to the **5-substrate combined cfDNA A-score** (methyl + nucl + fuzz + WPS + frag). VAL-082 measures **single-substrate methyl-only buffy-coat A-score** because that is what 450K/EPIC platforms produce. The Issue 002 1.10 figure is NOT the expected reading at v1 deployment — it is the expected reading once L2/L3 multi-substrate cfDNA platform comes online in future expansion. v1 launch operates on methyl-only buffy-coat, where the AML signal is expected at ΔA ≈ 0.05-0.15 absolute units above healthy.

## Reproduction
- Pre-reg SEAL file: VAL-082_PREREG_SEAL.txt
- Results JSON: VAL-082_results.json
- Outcome: VAL-082_outcome.md
- Python script: val_082_heme_epic_aml.py
- Source: GEO GSE62298 + GSE51057 public access
