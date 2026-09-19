# Step 7 — IAMAtlas v0.1 Validation Bundle

**Date packaged:** 2026-05-04
**Run when:** After production MCMC completes (8 per-class result.json files in `iamatlas_v0_1_output/`) AND after `merge_iamatlas_v0_1.py` produces `IAMAtlas_v0_1.csv`.

---

## What this bundle does

Six validation checks against your freshly-built IAMAtlas v0.1 matrix. Each one asks a different question:

| Check | Asks | Required inputs | Time |
|---|---|---|---|
| **A** | Did the production MCMC converge cleanly per class? | per-class result.json files | 30 sec |
| **B** | Does the matrix predict held-out CpGs well? | matrix + universe + inputs | 1 min |
| **C** | Does IAMAtlas-anchored AD scoring preserve tonight's earlier d=+0.61 / d=+0.37 result? | AIBL/AddNeuroMed β + manifests + val051 panel | 30 sec |
| **D** | Does IAMAtlas-anchored breast pre-dx scoring preserve VAL-047's d=+1.85 / d=+1.34? | matrix + Xu2020 panel + VAL047 sample CSVs | 1 min |
| **E** | Does GSE130748 (Mozhui longitudinal) show ΔA_immune trajectory predicting cancer incidence? | matrix + GSE130748 IDATs + (optional) cancer labels | 10–30 min |
| **F** | Are male/female/per-decade sub-cohorts robust under unified scoring? | per-sample CSVs from C and D | 30 sec |

---

## Bundle contents

```
step_7_bundle/
├── step_7_run_all.py                            # orchestrator (runs A B C D F)
├── step_7_check_a_exit_gates.py                 # MCMC convergence + diagnostics
├── step_7_check_b_predictive_validation.py      # Hannum + held-out CpGs
├── step_7_check_c_ad_cohort_scoring.py          # AIBL + AddNeuroMed re-score
├── step_7_check_d_breast_iam_scoring.py         # GSE51057/GSE51032 re-score
├── step_7_check_e_gse130748_trajectory.py       # Mozhui IDATs + trajectory
├── step_7_check_f_sex_age_stratified.py         # CHK-A11 + CHK-A12
└── cohort_artifacts/
    ├── aibl_imm_betas.json                      # AIBL per-sample β at panel CpGs
    ├── aibl_manifest.json                       # AIBL sample → AD/HC label
    ├── addneuromed_imm_betas.json               # AddNeuroMed per-sample β
    ├── addneuromed_manifest.json                # AddNeuroMed sample → label
    ├── val051_panel_ruleA.json                  # 7-CpG AD directional panel
    └── xu2020_breast_directional_RuleA.json     # 98-CpG breast directional panel (built tonight)
```

---

## Setup on your laptop

```bash
cd ~/IAMPerformance

# 1. Verify the matrix exists from the production run + merge step:
ls -la IAMAtlas_v0_1.csv

# 2. Copy the artifact files into the working directory (script defaults look here):
cp step_7_bundle/cohort_artifacts/* .

# 3. (For Check E only) Install methylprep if not already:
pip install methylprep

# 4. (For Check D) Place these files in working dir if you want to run that check:
#    VAL047_samples_GSE51057.csv  (from your VAL-047 Tightening fresh output)
#    VAL047_samples_GSE51032.csv

# 5. (For Check E) Have GSE130748 IDATs extracted into folder GSE130748_RAW/
#    Series matrix: GSE130748_series_matrix.txt.gz  (download from GEO)
```

---

## Run the suite

**Quick run (Checks A B C D F, ~3 min total):**
```bash
python3 step_7_run_all.py
```

**Run individual check:**
```bash
python3 step_7_check_a_exit_gates.py
python3 step_7_check_b_predictive_validation.py
python3 step_7_check_c_ad_cohort_scoring.py
python3 step_7_check_d_breast_iam_scoring.py
python3 step_7_check_f_sex_age_stratified.py
```

**Run Check E separately (slow, IDAT extraction):**
```bash
python3 step_7_check_e_gse130748_trajectory.py \
    --idat_dir GSE130748_RAW \
    --series_matrix GSE130748_series_matrix.txt.gz
```

---

## What success looks like

Each check produces a `step_7_chk_<x>_*.md` file. Open them. Pass criteria:

- **Check A:** All 8 classes show `R-hat < 1.05`, `ESS > 200`, `divergent = 0`, `pearson > 0.90`.
- **Check B:** Hannum r > 0.95 (pilot was 0.997). Held-out class-prior drift < 0.05.
- **Check C:** AIBL d > 0 (i.e., positive direction preserved), AddNeuroMed d > 0. PASS+ if IAM d > Flat d.
- **Check D:** GSE51057 breast >10yr d > +1.4 (target +1.85). GSE51032 breast >10yr d > +1.0 (target +1.34). GSE51032 colorectal d < 0 (sign-flip preserved).
- **Check E:** ΔA cancer-incident > ΔA cancer-free (positive direction). Magnitude not powered at n=20.
- **Check F:** Male/Female d differ by < 40%. No decade-cliff > 0.05 in adjacent decades.

---

## What failure means

- **Check A fail:** A class needs more sampling. Re-run that class only with `--tune 2000 --draws 2000 --chains 4`.
- **Check B fail:** Class hyperprior is mis-specified — investigate hyperparameter posteriors.
- **Check C/D sign-flip:** Matrix is mis-calibrated for that disease. STOP, don't deploy.
- **Check E zero or negative trajectory:** Either small-sample noise (likely at n=20) or atlas needs more longitudinal anchor cohorts. Document, don't block deployment.
- **Check F flag:** Document in card, plan per-stratum panel work for v0.2. Doesn't block v0.1 deployment but worth knowing.

---

## After all checks pass

You're done with Step 7. Next is Step 8 — atlas vault freeze:

1. Copy `IAMAtlas_v0_1.csv` into `Biological_Physics/atlas_vault/iamatlas_production/`
2. Update `INVENTORY.json` with SHA-256
3. Update vault README
4. Modify `commercial_web_scoring_engine_skeleton.py` to load IAMAtlas at startup
5. Push to GitHub `hmahaffeyges/IAM-Validation`

I'll handle Step 8 with you when we get there.

---

## Cohort label notes

**AIBL manifest** field `disease status` has values like:
- `"healthy control"` → HC
- `"alzheimer's disease"` → AD
- (any other) → skipped

**AddNeuroMed manifest** field `disease state` has:
- `"healthy control"` → HC
- `"alzheimer's disease"` → AD
- `"mild cognitive impairment"` → AD (per VAL-052 protocol — MCI is pre-clinical AD)

If labels look wrong in your run, inspect `aibl_manifest.json` directly.
