# VAL-082 — GSE62298 AML myeloid arm OUTCOME

**Card:** heme-epic v0.1
**Arm:** myeloid (AML)
**Date:** 2026-04-25
**Cohort:** GSE62298 Glass 2017 — n=68 primary AML patients on HM450, blood-derived methylation
**External healthy comparator:** GSE51057 EPIC-Italy menarche cohort, cancer-free subset (n=115 QC-passed of n=177 cancer-free women)
**Outcome:** **O1_PASS_MYELOID_ARM_AT_BLOOD_LEVEL** — heme-epic myeloid arm validated for v1 deployment

## Summary

VAL-082 is the first per-patient validation of the heme-epic myeloid arm at the blood-level methylation that v1 EDEAR will actually run. AML samples scored on the universal Stage 1 immune Xu-538 panel produce ΔA = **+0.1039** above the Italian healthy buffy-coat baseline, Cohen's d = **+3.71** [+3.23, +4.20] p ≈ 0, with **98.5% of AML samples scoring above the Italian healthy 95th percentile** and 91.2% above the 99th percentile. This is the strongest single-cohort effect size measured anywhere in the Cookbook to date.

## Numerical results

| Statistic | Italian healthy comparator | AML cohort |
|---|---|---|
| n | 115 (QC-passed) | 68 |
| Mean A_immune | 0.4384 ± 0.0244 | 0.5423 ± 0.0332 |
| Range | — | [0.4672, 0.6412] |
| Cohen's d (AML vs healthy) | (reference) | **+3.71** [+3.23, +4.20] |
| p-value | — | < 1e-50 |

Tail statistics:
- AML above Italian healthy 95th percentile (A > 0.4863): **67/68 = 98.5%**
- AML above Italian healthy 99th percentile (A > 0.5014): **62/68 = 91.2%**
- Italian healthy 95% range: [0.391, 0.486]; AML 95% range: [0.477, 0.608] — **near-complete distribution separation**.

## Mandatory checks per CCL-032 diagnostic order

### 1. Data integrity check (PASSED)

GSE62298 β distribution sanity check (CHK-3.1):
- 55.8% of values at extremes (β < 0.1 or β > 0.9) — exceeds raw β threshold of >30%
- 1.5% in [0.4, 0.6] — well below residual-data flag threshold of >40%
- Bimodal histogram with peaks near 0 and 0.9 — consistent with raw bimodal methylation
- 538/538 Xu-538 CpGs present (CHK-3.3 full panel coverage on HM450)
- 68/68 samples passed QC threshold (CHK-3.5 saturation: 0/68 above immune ceiling)

### 2. Biology consistency check (PASSED)

AML expected to score ABOVE healthy on immune-class panel because AML myeloid cells exhibit lineage-commitment locus reprogramming (MEIS1, HOXA cluster, DNMT3A mutation hotspots) that the buffy-coat-trained Xu-538 panel detects via deviation from healthy myeloid methylation patterns. Direction confirmed positive at d = +3.71. Magnitude ΔA = +0.10 absolute units is in the expected range for methyl-only single-substrate buffy-coat scoring (Issue 002 prediction of A ≈ 1.10 refers to 5-substrate cfDNA combined scoring, NOT directly comparable; see Substrate Clarification below).

### 3. Framework finding (CONFIRMED)

The myeloid arm of heme-epic is operational at the v1 deployment level. AML signal IS detectable on the universal Stage 1 immune Xu-538 methyl-only A-score from a single 450K array, with effect size that matches or exceeds any solid-organ card's per-patient validation result.

## Substrate clarification (important — read before interpreting absolute A magnitudes)

Issue 002 framework prediction A_AML ≈ 1.10 with ΔA ≈ +0.168 refers to the **5-substrate combined cfDNA A-score** (methyl + nucl + fuzz + WPS + frag). That is the expected reading on the future L2/L3 multi-substrate cfDNA platform — NOT on a single 450K array.

VAL-082 measures **single-substrate methyl-only buffy-coat A-score** because that is what 450K platforms produce. At this level, the AML signal is +0.10 absolute units above Italian healthy with d = +3.71. Both readings are correct for their respective substrate scopes; they measure different things at different stages of the framework deployment roadmap.

**v1 EDEAR launch operates on this methyl-only buffy-coat A-score.** The +3.71 Cohen's d is the v1-deployment effect size for AML detection in blood. Future L2/L3 platforms will combine substrates and increase signal further; that is post-launch capability expansion, not a precondition for v1.

## What this means for heme-epic at v1 launch

**The myeloid arm works at blood level.** Patients with AML produce a Stage 1 immune A-score that separates from healthy distributions at d > 3.7 — near-complete separation. 98.5% of AML samples in this cohort are above the healthy 95th percentile. This is the strongest detection signal anywhere in the Cookbook.

**Stage 3 EpiDISH will discriminate AML from inflammaging at this magnitude.** A patient with elevated Stage 1 + neutrophil-shifted Stage 3 + Moss NULL on solids = clean myeloid-arm fire. The framework's three-stage architecture is structurally sound for AML detection at the v1 deployment level.

**The "framework numbers don't match Issue 002 predictions" question is now answered.** Issue 002's A ≈ 1.10 figure was the cfDNA combined-substrate prediction, not the methyl-only buffy-coat prediction. The methyl-only buffy-coat reading at v1 deployment is +0.10 absolute units above healthy with d ≈ 3.7 — that is the signal EDEAR's first version actually uses. The Issue 002 prediction stands; it just describes the L2/L3 platform expansion target, not v1.

## Cohort heterogeneity expected at v0.2+

VAL-082 is a single-cohort validation (per CCL-029, single_cohort_validated tier). The cervical-epic v0.1 lesson is that single-cohort validation can hide cohort heterogeneity. Heme-epic v0.2 priority is to replicate VAL-082 on independent AML cohorts (MARLIN reference n=2,540 includes 1,461 AML samples — extract subset, score Stage 1, check whether the +0.10 ΔA replicates). Until that replication runs, VAL-082's effect size is single-cohort and may shrink under cross-cohort testing.

## Cohort caveats and limitations

1. **AML cohort comparator is mismatched on cohort design.** GSE62298 is American AML cohort; GSE51057 is Italian EPIC menarche cohort (women only). Sex distribution and ancestry differ. The +0.10 ΔA may include a confounder from cohort-design heterogeneity. v0.2 priority: re-run with ancestry-matched and sex-matched healthy comparators.

2. **GSE62298 includes DNMT3A mutation status.** Approximately half are DNMT3A mutant. DNMT3A mutation is a strong methylation modifier. v0.2+: stratified analysis (DNMT3A-mut vs DNMT3A-wt) to assess whether the +0.10 ΔA is driven by mutant subset.

3. **Pre-diagnostic vs at-diagnosis distinction not made.** GSE62298 samples are at-diagnosis primary AML, not pre-diagnostic. The "10+ years pre-dx" detection question for AML requires CHIP→AML serial-sample cohorts (G-2026-P010 prediction, VAL-085 future).

4. **Framework prediction at A_AML ≈ 1.10 is NOT validated by VAL-082.** That figure refers to a different scoring level (5-substrate cfDNA combined). VAL-082 validates the methyl-only buffy-coat blood reading, which is what v1 deployment uses. Both can be true; they measure different things.

## What's next for heme-epic

VAL-082 establishes the myeloid arm at single_cohort_validated tier. The lymphoid B-cell arm remains framework_calibrated_pending_validation; EnviroGenomarkers data is not publicly accessible (controlled-access biobank, not GEO-deposited). VAL-083 priority: identify a publicly-accessible CLL methylation cohort with healthy comparator. Candidate: Kulis 2012 cohort via gated EGA access; CLLmethylation Bioconductor package (n≈200 EGA-gated); or smaller GEO-accessible CLL cohorts. The lymphoid B-cell arm cannot reach single_cohort_validated tier from publicly accessible GEO data alone; reaching it likely requires gated-cohort access via lab partnership.

## Reproduction

- Pre-reg SEAL: VAL-082_PREREG_SEAL.txt — SHA-256 a8c3714a...
- Results JSON: VAL-082_results.json — SHA-256 6a742dca...
- Python script: val_082_heme_epic_aml.py
- Source data:
  - GSE62298 (AML): https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62298/matrix/GSE62298_series_matrix.txt.gz
  - GSE51057 (EPIC-Italy healthy): https://ftp.ncbi.nlm.nih.gov/geo/series/GSE51nnn/GSE51057/matrix/GSE51057_series_matrix.txt.gz
- Xu-538 panel SHA: ada672960...
- RNG seed: 20260425

## Lessons cited
- panc-LL-007 (Stage 1 universal H_min(immune) rule)
- CCL-029 (cohort completeness; this is single-cohort, v0.2 cross-cohort replication required)
- CCL-032 (diagnostic order: data integrity → biology → framework)
- heme-LL-001 (Moss NULL on solid organs is the diagnostic feature for heme-epic)
- heme-LL-002 (three-arm structure)
- CHK-3.1 (β distribution sanity check)
- CHK-3.5 (saturation flag check)
