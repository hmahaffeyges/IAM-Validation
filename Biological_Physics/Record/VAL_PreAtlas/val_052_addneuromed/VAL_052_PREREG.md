# VAL-052 Pre-Registration — AddNeuroMed Cross-Platform AD Replication

**Date:** 2026-04-23
**Parent:** VAL-051
**Cohort:** AddNeuroMed GSE144858 (n=300), Illumina 450K, European multi-center AD cohort
**Panel:** VAL-051 Rule A (7 CpGs, directions frozen from AIBL training)

---

## 1. Scope

True external cross-platform replication of the AD-directional immune panel. AIBL was EPIC 850K, Australian. AddNeuroMed is 450K, European (UK, Finland, Italy, France, Poland, Greece). Completely independent population and platform.

**Panel:** The 7-CpG Rule A panel from VAL-051 with directions frozen from AIBL 80% training.
**Standardization:** AIBL-training HC mean/SD (NOT AddNeuroMed HC), because frozen-panel replication means frozen standardization.
**Sensitivity analysis:** also report AddNeuroMed-own-HC-standardized version for robustness.

---

## 2. Frozen inputs

- VAL-051 Rule A panel (cpg, direction, mean_hc_train, sd_hc_train) per val051_panel_ruleA.json
- AddNeuroMed 450K β values for 15 IMM panel CpGs (all 7 Rule A CpGs present)
- AddNeuroMed metadata (gsm, age, sex, disease state)

---

## 3. Hypotheses (pre-locked)

**H1 (primary):** A_dir(AD) > A_dir(control) on AddNeuroMed, one-sided MWU, α = 0.05.

**H2 (secondary):** A_dir(MCI) is intermediate between AD and control.

**H3 (secondary):** Sex-stratified H1.

**H4 (primary — age-confounding test):** After regressing chronological age out of A_dir, residuals still show d(AD vs HC) > 0.3. This is the test that could not be run on AIBL due to missing age metadata.

**H5 (secondary):** Cohen's d on chronological age (AD vs HC). Tests whether AD cases are older than controls in AddNeuroMed (which would create age confounding).

---

## 4. Outcome matrix (from VAL-051 §7 outcome 1-5)

Expected given VAL-051 holdout d=+0.624:

| AddNeuroMed d | p | Interpretation |
|---|---|---|
| > 0.3 | < 0.10 | **FULL EXTERNAL REPLICATION** — AD panel generalizes cross-platform + cross-population. Card validation_tier upgrades from internal_holdout to cross_platform_validated. |
| 0.1 – 0.3 | any | Partial replication — direction preserved, effect size smaller. Expected if 450K preprocessing differs from EPIC. |
| < 0.1 | any | Cross-platform transfer fails. Card stays at internal_holdout; panel must be re-derived for 450K. |
| negative | any | Direction flip — major biological or technical difference between cohorts. Requires investigation. |

All outcomes publishable.

---

## 5. Age-confounding specifics

**H4 model:** A_dir ~ age (simple linear regression on all 300 samples, residuals taken per-sample).

**Decision rule:**
- R² < 30% AND residual d (AD vs HC) > 0.3: age is minor, AD signal dominant.
- R² 30-60%: age partial confound, AD signal survives.
- R² > 60%: age dominant driver; AD case must be qualified.

---

## 6. Seal + integrity

All inputs hash-sealed. Frozen panel + frozen standardization. No re-selection on AddNeuroMed. One-shot test.
