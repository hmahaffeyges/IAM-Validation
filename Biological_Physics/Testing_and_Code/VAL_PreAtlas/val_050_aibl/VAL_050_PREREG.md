# VAL-050 Pre-Registration — AD Immune-Class Cross-Sectional Replication

**Principal Investigator:** Heath W. Mahaffey  
**Framework:** IAM / GAPE Informational Actualization Model  
**Drafted:** 2026-04-23  
**Seal target:** pre-analysis SHA-256 freeze of this file + panel file + AIBL metadata  
**Audit informing this prereg:** `AD_Immune_Sensitivity_Audit.md` (2026-04-23)

---

## 1. Scope

Test whether the GAPE immune-class A-score, computed with frozen H_min and a frozen CpG panel, discriminates AD cases from matched healthy controls in two publicly deposited blood-methylation cohorts of independent origin and ethnicity.

**Not in scope for this VAL-050:**
- Pre-diagnostic Time-to-Onset stratification (age-at-dx metadata not in public GEO release for AIBL; requires direct AIBL/ADNI access)
- EpiDISH immune subcomposition (deferred to VAL-051)
- Fragmentomics substrate (deferred, no published AD frag cohort exists)
- Patient-level clinical utility claims (this is a cohort-level signal test)

---

## 2. Frozen constants

| Constant | Value | Source |
|---|---|---|
| H_min(immune, methyl) | 0.838889 | G-002 MCMC posterior, 5 chains, R-hat < 1.001 |
| A-score formula | A = H(β) / H_min(immune) | GAPE_WEB_v13.py lines 466, 477 |
| H formula | H(β) = −β log₂β − (1−β) log₂(1−β) | Shannon binary entropy |
| Tier thresholds | MARGINAL 1.01, DETECTABLE 1.05, URGENT 1.07, FLOOR BREACH 1.10 | GAPE_WEB_v13.py lines 55-58 |

---

## 3. Frozen panels (two head-to-head)

### Panel A: IMM_CPGS_EPIC_18

Derived from `IMM_CPGS_RAW` (29 CpGs, GAPE_Evidence_Report_CURRENT.html line 11785, Xu 2020 Sister Study breast-derived). Intersected with the AIBL EPIC 850K platform. **18 CpGs retained:**

```
cg00431549, cg01127300, cg02228185, cg02489552, cg04023335,
cg09809672, cg10632894, cg12554573, cg14614643, cg16867657,
cg17861230, cg18834029, cg22454769, cg22736354, cg23244761,
cg25432518, cg25809905, cg26614073
```

**Rationale:** This panel is NOT AD-specific. It is the canonical GAPE immune class panel used in VAL-047 Phase 12 (colorectal) and is frozen for that validation. Applying it to AD tests cross-disease-class transfer of the immune class panel. A positive result = the framework generalizes. A null result = panels must be disease-specific (supports purpose-built AD panel case).

### Panel B: IMM_CPGS_ADNATIVE_PUBLISHED

**Reserved — to be populated from the Zhang 2025 FHS+ADNI 151-CpG supplement if accessible, or from Nabais 2021 top AD-associated CpGs published in the GSE153712 companion paper.**

In the absence of published-CpG-list access in this session, VAL-050 proceeds on Panel A alone. Panel B is a VAL-051 deliverable.

---

## 4. Cohorts

| Cohort | Accession | Platform | n_total | AD | MCI | HC | Status |
|---|---|---|---|---|---|---|---|
| AIBL Nabais 2021 | GSE153712 | EPIC 850K | 726 | 161 | 94 | 471 | **Pulled — ready** |
| AddNeuroMed | GSE144858 | 450K | ~300 | 93 | 111 | 96 | Secondary, time-permitting |
| Li 2020 FTD controls | GSE53740 | 450K | 165 HC only | — | — | 165 | Reference only |

**Primary analysis: AIBL GSE153712.** All results reported.

**Secondary analysis: AddNeuroMed GSE144858.** Run only if AIBL produces a signal worth replicating and compute budget allows. Cross-platform replication (EPIC→450K) is itself a meaningful test.

---

## 5. Pre-specified analyses

### Primary

**Hypothesis H1:** Mean A_immune(AD) > Mean A_immune(HC) in AIBL, one-sided.

**Test:** Mann-Whitney U on per-sample A-score, HC vs AD.

**Decision threshold:** α = 0.05, one-sided.

**Effect-size reporting:** Cohen's d = (mean_AD − mean_HC) / pooled_SD with 10,000-resample bootstrap 95% CI.

### Secondary

**H2:** A_immune(MCI) is intermediate between HC and AD.  
**Test:** Jonckheere-Terpstra trend test across HC→MCI→AD groups.

**H3:** Signal is not driven by sex confounding.  
**Test:** Re-run H1 in males-only and females-only strata.

**H4:** Per-CpG analysis — which of the 18 panel CpGs individually discriminate HC vs AD?  
**Test:** Per-CpG Mann-Whitney U, BH-FDR correction across 18 tests.

### Permutation backbone

10,000 random label shuffles of disease-status labels (within-sex), seed = 42. Generates the null distribution for every tested statistic. Reported empirical p-value alongside the parametric p.

### Bootstrap CIs

10,000 resamples with replacement per group, seed = 42. CIs on means, differences in means, Cohen's d, and AUC.

---

## 6. Pre-locked decision rules (4 outcomes)

**Outcome 1 — Positive:** H1 p_one-sided < 0.05 AND Cohen's d > 0.3.  
**Interpretation:** Framework generalizes. Xu-breast-derived immune panel detects AD architectural drift at the per-patient cohort level. Publication-worthy as cross-disease-class validation.

**Outcome 2 — Direction-positive-weak:** H1 p_one-sided < 0.10 AND 0 < Cohen's d ≤ 0.3.  
**Interpretation:** Framework direction is correct but effect size is small, consistent with VAL-040 whole-blood age-matched ΔA ≈ +0.007. Expected outcome if VAL-040 extrapolates to per-patient scale. Publication-worthy as directional replication.

**Outcome 3 — Null:** |Cohen's d| < 0.1 OR p_one-sided > 0.10.  
**Interpretation:** Panel is class-specific not disease-general. Supports the case for purpose-built AD panel (Panel B, VAL-051). VAL-048 discipline applies: honest negative, pre-registered, publication-worthy.

**Outcome 4 — Negative:** Cohen's d < −0.1 AND p < 0.10 in the wrong direction.  
**Interpretation:** Framework requires revision for AD immune-class direction. High-stakes result. Would trigger re-examination of H_min(immune) applicability across disease classes.

All four outcomes are publishable. All four are hash-sealed before analysis.

---

## 7. Expected outcome (not used in decision, documented for honesty)

Based on the audit:

- VAL-040 whole-blood age-matched ΔA = **+0.0073** (immune class, methyl only)
- Individual A_sd(70-79) = **0.0387**
- Expected Cohen's d = 0.0073 / 0.0387 = **0.19**
- Expected p_one-sided on Mann-Whitney U at n=161 AD vs n=471 HC, d=0.19: **approximately 0.04-0.08**

So the most likely outcome is **Outcome 2 (direction-positive-weak)**, with a non-trivial chance of falling into Outcome 3 (Null) if panel non-transferability eats the signal.

---

## 8. QC gates

**Sample exclusion before analysis:**
- Missing disease status → exclude
- Missing sex → exclude (needed for stratification)
- Fewer than 12 of 18 panel CpGs with valid β in [0, 1] → exclude as QC-failed

**No post-hoc exclusion.** Outliers are not removed; reported separately as robustness checks using median + MAD if needed.

---

## 9. Analysis order (locked)

1. Load AIBL manifest (`aibl_manifest.json`), join to per-sample β matrix (`aibl_imm_betas.json`)
2. Apply QC gate (step 8). Record N excluded, reason, GSM IDs
3. Compute per-sample mean-β across 18 panel CpGs
4. Compute per-sample A_immune = H(mean-β) / H_min(immune)
5. Run H1 primary Mann-Whitney U, HC vs AD, one-sided
6. Run 10,000-permutation null
7. Bootstrap 10,000 × Cohen's d
8. Evaluate against 4 pre-locked outcome rules
9. Report

**No step 10 before step 9.** No peeking. No re-running with different panels, different H_min, different exclusion criteria.

---

## 10. Seal procedure

Before step 9 (analysis execution) runs:

- SHA-256 of this prereg document
- SHA-256 of `aibl_manifest.json`
- SHA-256 of `aibl_imm_betas.json`
- SHA-256 of `stream_aibl.py`
- SHA-256 of `run_val_050.py` (analysis script, step 9)

All five hashes recorded in `VAL_050_SEAL.txt`. That file is committed to GitHub before `run_val_050.py` is executed. Any change to any input requires re-seal.

---

## 11. Scientific integrity commitments

- No post-hoc panel change
- No post-hoc exclusion
- No post-hoc threshold adjustment
- No re-run with different α or different sidedness
- All four outcomes (Positive / Direction-positive-weak / Null / Negative) are publishable
- Null result is not a failure — it is a pre-registered answer to a pre-registered question
- Panel B (AD-native) is VAL-051, not a safety net for VAL-050

Signed for submission (Heath W. Mahaffey): ___________________________ Date: ___________
