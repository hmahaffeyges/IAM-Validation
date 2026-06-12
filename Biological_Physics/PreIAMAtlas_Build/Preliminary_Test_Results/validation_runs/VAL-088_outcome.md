# VAL-088 — glioma-epic Stage 1 outcome

**Outcome label:** `O5_POSITIVE_INVERTED`
**Interpretation:** Glioma peripheral blood reads positive direction at Stage 1 immune Xu-538 — NOT negative as the CCL-023 cell-fraction-prior had suggested. Direction-of-effect inverts the hypothesis, magnitude is meaningful, and the finding refines CCL-023.

## Headline numbers

| Stratum | n | mean A | SD | ΔA vs healthy | d vs healthy | 95% CI |
|---|---|---|---|---|---|---|
| All glioma | 76 | 0.4571 | 0.0229 | +0.0187 | +0.91 | [+0.61, +1.22] |
| Pre-surgery treatment-naive | 37 | 0.4563 | 0.0193 | +0.0179 | +0.94 | [+0.56, +1.33] |
| Pre-surgery GBM (n=25) | 25 | 0.4535 | 0.0188 | +0.0151 | +0.80 | — |
| Pre-surgery LGG (n=12) | 12 | 0.4622 | 0.0197 | +0.0238 | +1.25 | — |
| GBM (current grade, all) | 49 | 0.4603 | 0.0234 | +0.0218 | — | — |
| LGG (current grade, all) | 27 | 0.4514 | 0.0211 | +0.0129 | — | — |

Healthy reference (GSE51057 EPIC-Italy HM450, n=115): mean A = 0.4384 ± 0.0244.

## CCL-032 diagnostic order

### Step 1 — Data integrity (PASSED)

- **CHK-3.1 β distribution:** PASS. Spot-check across 5 chips × every-1000th-probe sample (n=4325): 56.5% of β at extremes (<0.1 or >0.9), 6.2% in mid [0.4, 0.6], median 0.823. Bimodal distribution consistent with raw β values from EPIC array. NOT residualized M-values.
- **CHK-3.3 panel coverage:** PASS. After p<0.05 detection-p QC (Illumina/minfi standard for EPIC), 76/76 samples retained. Mean Xu-538 coverage = 488/538 (90.6%) — better than typical EPIC drift (~80%); Xu-538 is well-preserved on EPIC v1.0_B4.
- **CHK-3.5 saturation:** PASS. 0/76 samples within 0.005 of A_ceiling=1.1921. Maximum observed A = 0.519 (43.5% of ceiling). Substantial headroom.
- **CHK-3.2 cross-cohort baseline:** Reference (Italian HM450, mean A = 0.4384) vs test (US/UCSF EPIC). Baseline difference is the test signal — cannot directly check for healthy-vs-healthy mismatch because the test cohort is all-glioma. Cross-platform caveat retained.

### Step 2 — Biology consistency (REFINED HYPOTHESIS, NOT FAILED)

- **CHK-4.1 published clinical-grade panels:** Glioma blood-based methylation literature is dominated by Wiencke/Salas immunomethylomics (mdNLR, neutrophil-to-lymphocyte methylation index) and Sabedot GeLB (serum cfDNA). The mdNLR / NLR direction in glioma blood is well-established as elevated (lymphocytes down, neutrophils up). Our Stage 1 immune A-score reads positive direction, **higher entropy in immune-class CpG distribution**. This is consistent with active immune dysregulation regardless of which lineage shifts up or down. The cell-fraction direction (Bracci 2022) and the methylation-entropy direction (this VAL) are **different metrics measuring different phenomena**. The CCL-023 hypothesis bridge from cell-fraction direction to A-score direction does not hold for glioma. The biology is consistent with active disease; the prior was wrong about how it would project onto the A-score.
- **CHK-4.2 cancellation hypothesis:** Not the AD-instance pattern (AD: pooled-null + directional pass). Glioma reads pooled-positive directly. No bidirectional-cancellation hypothesis needed.

### Step 3 — Framework finding

**Glioma blood reads positive direction at Stage 1 immune A-score, with d ≈ 0.9 vs Italian healthy reference.** Pre-surgery treatment-naive subset confirms d ≈ 0.94, ruling out treatment confounding. **CCL-023 direction-as-discriminator hypothesis applied to glioma reads opposite of the cell-fraction prior** — glioma joins the activation-shifted set (AD, breast, lung, prostate, HCC, pancreatic), not the suppression-shifted set (CRC).

## What this changes about CCL-023

CCL-023 was an open hypothesis with two anchoring data points (CRC negative VAL-047, AD positive VAL-051/052) and one literature-supported additional point (glioma cell-fraction signature consistent with negative direction, Bracci 2022). **The literature-supported point did not survive direct measurement.** This is genuinely new information — the bridge from cell-fraction-direction (NLR-style) to A-score-direction (entropy-style) does not hold automatically. Cell-fraction direction and methylation-entropy direction are different metrics. CCL-023 should be revised to acknowledge this:

- A-score direction may NOT directly track NLR cell-fraction direction.
- The CRC negative-direction signal (VAL-047) is on pre-diagnostic blood, well before clinical symptoms; the glioma positive-direction signal here is at-diagnosis or post-diagnosis. **Temporal phase may matter more than cell-fraction direction.**
- Update the CCL-023 anchoring set: CRC pre-diagnostic 5-10 yr = negative; AD = positive; glioma at-diagnosis = positive (this VAL); breast/lung/prostate/HCC pre-diagnostic 2-10 yr = positive. Pattern: **negative direction observed only for CRC at long pre-dx window; everything else (including glioma at diagnosis) reads positive at present.** The direction-as-discriminator may collapse to "early-pre-dx CRC is the outlier" rather than a general activation-vs-suppression rule.

## What this does NOT prove

- This is N=1 cohort. Reproduction on second cohort (Bracci 2022 UCSF AGS phs001497, currently gated) is required for `single_cohort_validated` tier.
- We cannot rule out cross-platform drift fully accounting for the +0.0187 ΔA. The direction-of-effect (positive sign) is robust; the absolute magnitude carries a coverage-drift caveat.
- CCL-023 may still hold for **untested temporal phases**. Pre-diagnostic glioma blood (5-10 yr before diagnosis) may read negative even if at-diagnosis glioma reads positive. We have no pre-diagnostic methylation cohort for glioma.
- We cannot detect terminal-class cfDNA in this VAL — this is whole blood, not plasma cfDNA, and the immune class dominates the signal. Direct neuron-class detection requires Pathway 1 (cfMeDIP-seq plasma) or Pathway 3 specimens.

## Card tier impact

- **glioma-epic v0.1 stage:** Cohort-completeness pass (CCL-029) on accessible Tier 1 publicly-deposited blood-methylation glioma cohorts: GSE180683 is the only one. Tier-3 cohorts (UCSF AGS phs001497, UCSF Immune Profiles phs002998, GICC phs001319, Mayo) require dbGaP applications and remain pending.
- **Validation tier achieved:** `exploratory_pending_replication` — one cohort, one direction-of-effect test, no internal healthy controls, cross-platform reference. Honest tier label.
- **Required for upgrade to `single_cohort_validated`:** access to AGS phs001497 (Bracci 2022) for direct β-value testing of the same hypothesis, on its native platform, with on-study healthy controls (n=454).
