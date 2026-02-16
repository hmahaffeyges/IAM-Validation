# MCMC Chain Files

This directory contains the raw Planck MCMC chain outputs for all three runs.

## Required Files

Each run produces a set of files. Copy your local `iam_planck_chains/` contents here:

### Run A: IAM Fixed (mu0 = -0.13495)
- `iam_fixed_mu0.1.txt` — Chain samples (weight, -loglike, params...)
- `iam_fixed_mu0.paramnames` — Parameter names and LaTeX labels
- `iam_fixed_mu0.ranges` — Parameter prior ranges
- `iam_fixed_mu0.inputparams` — Input parameter values
- `iam_fixed_mu0.updated.yaml` — Cobaya updated configuration

### Run B: mu0 Floating
- `iam_float_mu0.1.txt`
- `iam_float_mu0.paramnames`
- `iam_float_mu0.ranges`
- `iam_float_mu0.inputparams`
- `iam_float_mu0.updated.yaml`

### Run C: LCDM Baseline
- `lcdm_baseline.1.txt`
- `lcdm_baseline.paramnames`
- `lcdm_baseline.ranges`
- `lcdm_baseline.inputparams`
- `lcdm_baseline.updated.yaml`

## Verification

After placing chain files here, run the GetDist extraction scripts:

```bash
cd ../getdist_scripts/
python extract_run_a.py
python extract_run_b.py
python extract_run_c.py
python three_way_comparison.py
```

Expected results:

| Run | Best chi2 | H0 | sigma8 |
|-----|-----------|-----|--------|
| A (IAM fixed) | 984.57 | 67.06 +/- 0.51 | 0.8014 +/- 0.0057 |
| B (mu0 float) | 981.24 | 67.16 +/- 0.51 | 0.8146 +/- 0.0157 |
| C (LCDM) | 983.14 | 67.19 +/- 0.54 | 0.8139 +/- 0.0062 |

Delta-chi2 (A vs C) = +1.43 (statistically indistinguishable at Planck precision)
Delta-chi2 (B vs C) = -1.90 (neutral after AIC penalty)

## Note on File Sizes

Chain files are typically 5-20 MB each. If GitHub LFS is needed, the `.txt` chain
files are the large ones. The `.paramnames`, `.ranges`, and `.inputparams` files
are small text files.
