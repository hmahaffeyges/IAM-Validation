# VAL-059 Pre-Registration Amendment — 2026-04-24 06:56 UTC

## Amendment justification

Same issue as VAL-058: original VAL_059_PREREG.md specified "Moss 2018 hepatocyte reference CpG subset" for Stage 2 metrics M2 and M3, but the Moss per-CpG reference panels are part of the proprietary calibration layer (NDA-gated per US Provisional Patents 64/012,720 and 64/014,568). Cannot be used in publicly-exposable scripts.

**Protocol amendment: M2 and M3 removed from pre-reg. Validation proceeds through Xu-538 panel case-control on both cohorts, plus the pre-reg-specified sex stratification, per-CpG directional preservation, age anchor, and cohort batch-offset checks.**

Issued BEFORE any β-value access to GSE281691 or GSE298812. Matrices have NOT been downloaded or parsed as of this timestamp.

---

## Amended analytical protocol

**Metrics (amended):**
- **M1: pooled_Ximmune** — Xu-538 immune panel A-score on each cohort. Primary cross-cohort metric.
- **M1_direction: per-CpG Δβ direction preservation** — report direction-preserved rate per cohort against Xu 2020 published direction pattern.
- **M2 and M3 REMOVED.** Moss per-CpG hepatocyte reference is NDA-gated.

**Outcome decision matrix (amended):**

### O1: CROSS-PLATFORM VALIDATED
d(M1) > 0.3 on both GSE281691 (whole-blood leukocyte) AND GSE298812 (ccfDNA), with direction match and magnitude ratio within 2×. Card enters at `cross_platform_validated` tier. Substrate-specific thresholds documented.

### O2: SINGLE-COHORT VALIDATED  
d(M1) > 0.3 on GSE281691 only; GSE298812 fails (d < 0.3). Card at `cohort_screening_validated` tier on whole-blood leukocyte specimen only. ccfDNA substrate explicitly flagged as not-validated.

### O3: cfDNA-ONLY VALIDATED
d(M1) > 0.3 on GSE298812 only; GSE281691 fails. Card at `cohort_screening_validated` tier restricted to ccfDNA substrate. Whole-blood specimen flagged as not-validated.

### O4: NULL ON BOTH
d(M1) < 0.3 on both cohorts. Card NOT deployed. Enters Cookbook at `null_documented` tier with explicit analysis: whether Xu-538 (breast-derived) simply doesn't transfer to HCC, or whether the HCC immune signature is bidirectional and requires a directional panel (Directional-Score Principle per CCL-001). Future work: HCC-specific directional panel per VAL-051 methodology.

### O5: UNEXPECTED PATTERN
Report numbers honestly; card update deferred.

**All other elements of VAL_059_PREREG.md retained:** cohort independence, panel freeze, RNG seed, per-cohort separate analysis before cross-cohort synthesis, age regression, sex stratification (HCC is male-predominant), per-CpG directional check, 80-cell age anchor applicability (whole-blood leukocyte only, NOT ccfDNA per CCL-004), cohort batch offset check, no cherry-picking, no panel re-training.

---

## Amendment seal

SHA-256 locked; appended to VAL_059_SEAL.txt.
