# VAL-054 Pre-Registration — Cellular-Age-Regressed AD Signal

**Date:** 2026-04-23
**Parent:** VAL-051
**Motivating problem:** AIBL AD cases are older than HC on average in the published literature. The VAL-051 directional signal could partly reflect AD cases being older, not having AD. But the AIBL GEO release (GSE153712) has no chronological age metadata.

---

## 1. Scope

Since chronological age is unavailable, compute per-sample **cellular age** from the GAPE 80-cell healthy-baseline immune-class β trajectory (per §E.5 of IAM_Hubble2GAPE_Alpha_Omega_v3.tex). Then regress cellular age out of the A_dir signal and check whether the VAL-051 holdout signal survives.

**Logic:** If AD samples have elevated A_dir solely because their cells are aged beyond the HC distribution (a "cellular age confound"), then regressing cellular age out should eliminate the signal. If A_dir elevation survives the regression, the signal is AD-specific, not age-driven.

---

## 2. Frozen constants (inherited)

- VAL-051 split_map (seed=42 stratified)
- VAL-051 Rule A panel (7 CpGs, directions)
- 80-cell baseline, immune class (from §E.5 Alpha-Omega, already in GAPE_WEB_v13.py as HEALTHY_BASELINE)
- Seed = 42, N_boot = 10,000

---

## 3. Cellular age computation

For each sample:

1. Compute β_mean over the 7 Rule A panel CpGs.
2. Invert the immune-class age-decade β trajectory to find the age at which the healthy immune-class β_mean equals the patient's β_mean.
3. Interpolate linearly between bounding decades.
4. This gives `cellular_age_immune` per sample (years).

Note: The 80-cell baseline β_mean per decade is the class-wide mean across all immune CpGs. Our 7-CpG panel β_mean is NOT the same thing — it's a panel subset that is AD-directional by design. So the "cellular age" derived from this panel is **panel-specific cellular age**, not the general IAM cellular-immune-age. This is still useful as an age proxy for regression.

To be methodologically clean, also compute `cellular_age_fullpanel` from the full 18-CpG IMM_CPGS panel β_mean using the same interpolation — this is closer to the IAM §E.5 concept.

---

## 4. Regression

On holdout (n=148, all statuses):

Model 1: A_dir_RuleA ~ 1 (baseline for comparison)
Model 2: A_dir_RuleA ~ cellular_age_immune_panel
Model 3: A_dir_RuleA ~ cellular_age_fullpanel

For each model, compute residuals. Then re-run H1 (AD > HC) on residuals.

---

## 5. Hypotheses (pre-locked)

**H1 (primary):** After regressing out cellular_age_fullpanel, A_dir residuals still show d(AD) > d(HC) with Cohen's d > 0.3.

**H2 (secondary):** Variance explained by cellular age in the holdout A_dir distribution is < 30% (most of VAL-051 signal is NOT age).

**H3 (exploratory):** Compare mean cellular_age between AD/MCI/HC. AD cases show cellular age elevated vs HC, consistent with AD being a disease of accelerated cellular aging.

---

## 6. Outcome matrix

| A_dir residual d on holdout | Variance explained by age | Interpretation |
|---|---|---|
| > 0.4 | < 30% | **Primary signal is AD-specific**, not age. Clean case. |
| 0.2–0.4 | 30-50% | Some age confound, but real AD signal survives. |
| < 0.2 | > 50% | Age is the dominant driver; VAL-051 signal is largely accelerated aging. Framework must fold age into the panel interpretation. |
| negative | any | Overshot age regression — numerical concern. |

All four are publishable; outcome determines how AD is framed in the Cookbook (age-confounded vs age-independent).

---

## 7. Seal + integrity

All inputs hash-sealed. No re-split. No post-hoc threshold. All four outcomes publishable.
