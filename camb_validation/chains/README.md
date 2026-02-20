# Level 2 MCMC Chain Files

This directory contains raw Planck MCMC chain outputs for Level 2 validation runs.

## Required Files

Copy chain output directories from the gaming PC here.

### L2 Run A: IAM Dual-Sector (iam_dual_sector = .true., mu_0 = -0.13495)
- Chain samples, paramnames, ranges, updated YAML

### L2 Run C: LCDM Baseline (iam_dual_sector = .false., mu_0 = 0)
- Chain samples, paramnames, ranges, updated YAML

### L2 Run D: IAM Dual-Sector + RSD (pending)
- Chain samples, paramnames, ranges, updated YAML

## Verification

After placing chain files here:

```bash
cd ../getdist_scripts/
python level2_comparison.py
```

Expected results:

| Run | sigma_8 | H0 | Total chi2 | Delta-chi2 |
|-----|---------|-----|-----------|------------|
| A (IAM dual-sector) | 0.7994 +/- 0.006 | 67.162 +/- 0.469 | 10985.17 | +0.02 |
| C (LCDM baseline) | 0.8089 +/- 0.006 | 67.159 +/- 0.465 | 10985.15 | baseline |

## Note on Likelihood

Level 2 chains use Planck NPIPE CamSpec TTTEEE (not plik_lite used in Level 1).
This accounts for the difference in absolute chi2 values between Level 1 and Level 2.
