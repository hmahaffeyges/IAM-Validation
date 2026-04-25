# VAL-088 — glioma-epic Stage 1 — peripheral blood immune A-score on GSE180683

**Status:** Pre-registration document, sealed before scoring decision lock.
**Honesty disclosure (CHK-2 head):** because GSE180683 has no internal healthy controls and the framework prediction (CCL-023 negative direction) was specified before scoring, the d-criteria below were locked before the absolute magnitude was known. Direction-of-effect was the primary pre-locked test; magnitude is reported as observed.

---

## Cohort

- **Test cohort:** GSE180683 (Salas/Wiencke 2022, PMID 35140201). n=76 glioma patients, EPIC v1.0_B4 peripheral blood, mixed treatment stages with FCM-validated T-cell composition.
- **External healthy reference:** Italian healthy buffy coat from VAL-082 (GSE51057 EPIC-Italy HM450, cancer-free subset, n=115 QC-passed). Mean A-immune = 0.4384 ± 0.0244. Treated as a fixed reference distribution per CHK-3.2 cross-cohort baseline check; absolute-magnitude comparisons carry a documented cross-platform caveat.

## Panel

- Xu-538 immune (SHA `ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6`, n=538 CpGs).
- Universal H_min(immune) = 0.838889 per panc-LL-007.

## Stratifications (declared before scoring)

1. **Full cohort** (n=76): all glioma blood samples.
2. **Pre-surgery treatment-naive subset** (n=37): CCL-024-cleanest test free of treatment confounding.
3. **Histological group**: 1 new gbm (n=39), 2 new lgg (n=14), 3 rec lgg still lgg (n=13), 4 rec lgg now gbm (n=10).
4. **Treatment time point**: 7 distinct values per metadata field.

## Pre-locked decision criteria (CHK-2.1)

The framework hypothesis under test is **CCL-023 direction-as-discriminator applied to glioma**: per Bracci 2022 cell-fraction signature (lymphocytes-down, neutrophils-up), CCL-023 predicts NEGATIVE direction for glioma immune A-score versus healthy.

Possible outcomes:
- **O1_PASS_NEGATIVE_DIRECTION**: ΔA < 0 with d < -0.5 (95% CI upper bound < 0). CCL-023 hypothesis confirmed in the predicted direction.
- **O2_PARTIAL_NEGATIVE**: ΔA < 0 with |d| < 0.5. Weak signal in predicted direction.
- **O3_NULL**: |d| < 0.2 with CI crossing zero. No detectable shift.
- **O5_POSITIVE_INVERTED**: ΔA > 0 with d > +0.5 (95% CI lower bound > 0). Direction inverted from CCL-023 prediction; reframe as glioma joining the AD/breast/lung/prostate activation-shifted set, NOT the CRC suppression-shifted set.
- **O6_UNEXPECTED**: data-integrity flagged (CHK-3.1 fail) or cohort-baseline flagged (CHK-3.2 fail), or any other diagnostic-pending status.

## CHK requirements (declared)

- **CHK-1.5 substrate-scope:** No Issue 002 substrate-scope conflict expected — both test and reference are single-substrate methyl-only buffy coat. No translation required.
- **CHK-1.6 access tier:** Test cohort is Tier 1 (GEO public). Reference cohort is Tier 1 (GEO public).
- **CHK-2.4 panel transferability:** Xu-538 trained on whole buffy coat; both test and reference are native specimen. No transferability flag.
- **CHK-2.5 Test 2 placeholder:** Lymphoid-vs-myeloid split blocked on OQ-2026-01 immune-atlas staging; cancellation hypothesis cannot be tested.
- **CHK-3.1 β-distribution:** Required >20% extremes <0.1 or >0.9, <40% in [0.4, 0.6]. Hard halt if fails.
- **CHK-3.2 cross-cohort baseline:** Required ΔA between healthy references < 1 SD; otherwise flag.
- **CHK-3.3 panel coverage:** Required mean Xu-538 coverage ≥400 of 538 per QC-passed sample; EPIC platform expected ~80%.
- **CHK-3.5 saturation:** Per-sample distance to A_ceiling=1.1921 reported; flag any sample within 0.005 of ceiling.

## Cross-platform comparison caveat

Test cohort is EPIC v1.0_B4. Healthy reference is HM450. EPIC has ~80% coverage of Xu-538 vs HM450's full coverage. For absolute magnitude comparisons, this introduces a coverage-drift confound. Direction of effect (sign of ΔA) is robust to this drift; absolute magnitude is not. **Primary inference is direction; magnitude is reported with explicit caveat.**

## Pre-locked secondary analyses

- Per-sample distance to A_ceiling reported in results JSON.
- Per-stratum effect sizes (GBM vs LGG, pre-surgery vs treated) reported.
- FCM-derived cell composition (cd4t, cd4nv, cd4mem, cd8t, cd8mem, cd8nv, nontcell from metadata) noted for cross-validation.

## Files at lock

- `val_088_glioma_epic_blood.py` — analysis script
- `GSE180683_manifest.json` — parsed metadata for all 76 samples
- `GSE180683_chippos_to_gsm.json` — chip_position → GSM mapping

## Output (post-scoring)

- `VAL-088_results.json` — full numerical output
- `VAL-088_distributions.png` — boxplot full cohort
- `VAL-088_presurg.png` — boxplot pre-surgery subset
- `VAL-088_outcome.md` — outcome interpretation
