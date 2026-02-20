# Level 2 Validation: Dual-Sector Perturbation and Background Tests via Modified CAMB

This directory contains the Level 2 validation of IAM using vanilla CAMB v1.5.8 with direct Fortran modifications to `equations.f90`. Unlike Level 1 (which uses MGCAMB's built-in mu-Sigma parametrization), Level 2 implements the dual-sector mechanism explicitly: CDM and baryons experience a modified effective expansion rate (`adotoa_matter`) while photon perturbation equations remain standard.

## Tool Distinction

| Level | Tool | Modification | Scope |
|-------|------|-------------|-------|
| Level 1 | MGCAMB v1.5.2 | mu-Sigma parametrization (built-in) | Perturbation equations only |
| **Level 2** | **CAMB v1.5.8** | **Direct Fortran modification** | **Perturbation equations (2a), background Friedmann equation (2b/2c)** |

## Fortran Modification

**File modified:** `equations.f90` → renamed `equations_iam_level2.f90`

Three surgical changes to implement the dual-sector perturbation split:

1. **Module-level parameters** (lines 54-56): `iam_beta = 0.15765`, `iam_dual_sector = .true.`
2. **Matter-sector Hubble rate** (lines 2244-2252): `adotoa_matter = sqrt((grho + iam_beta * E(a) * grho_0) / 3)`
3. **CDM and baryon equations** (lines 2323-2337): use `adotoa_matter` instead of `adotoa`

The `dtauda` function (background) is **unmodified** in Level 2a. Photon equations are **untouched**. See `equations_iam_level2.f90` for the exact code.

## Level 2a Results (Dual-Sector Perturbations, LCDM Background)

**Likelihood:** Planck NPIPE CamSpec TTTEEE + lowl TT + lowl EE + lensing CMBMarged

| Chain | Description | mu_0 | Dual-sector | Status |
|-------|------------|-----|------------|--------|
| L2 Run A | IAM Level 2a, fixed | -0.13495 | ON | Converged |
| L2 Run C | LCDM baseline | 0 | OFF | Converged |
| L2 Run D | IAM Level 2a + RSD | -0.13495 | ON | Pending |

### Parameter Comparison (Run A vs Run C)

| Parameter | Run A (IAM) | Run C (LCDM) | Shift |
|-----------|------------|-------------|-------|
| H0 | 67.162 +/- 0.469 | 67.159 +/- 0.465 | +0.01 sigma |
| sigma_8 | 0.7994 +/- 0.006 | 0.8089 +/- 0.006 | -1.58 sigma |
| ombh2 | 0.02217 +/- 0.0001 | 0.02218 +/- 0.0001 | -0.04 sigma |
| omch2 | 0.11994 +/- 0.0011 | 0.11994 +/- 0.0010 | -0.00 sigma |
| tau | 0.0534 +/- 0.0074 | 0.0532 +/- 0.0073 | +0.03 sigma |
| ns | 0.9629 +/- 0.0039 | 0.9627 +/- 0.0039 | +0.03 sigma |
| logA | 3.0400 +/- 0.0149 | 3.0396 +/- 0.0144 | +0.03 sigma |
| Omega_m | 0.3166 +/- 0.0065 | 0.3166 +/- 0.0065 | -0.01 sigma |
| S8 | 0.821 +/- 0.011 | 0.831 +/- 0.012 | -0.85 sigma |

### Chi-Squared Comparison

| Likelihood | Run A (IAM) | Run C (LCDM) |
|-----------|------------|-------------|
| Total chi2 | 10985.17 | 10985.15 |
| planck_2018_lowl.TT | 23.47 | 23.61 |
| planck_2018_lowl.EE | 396.91 | 396.84 |
| planck_NPIPE_highl_CamSpec.TTTEEE | 10555.22 | 10555.26 |
| planck_2018_lensing.CMBMarged | 9.56 | 9.43 |
| **Delta-chi2 (IAM - LCDM)** | **+0.02** | **baseline** |

## Directory Contents

```
camb_validation/
├── README.md                          # This file
├── equations_iam_level2.f90           # Modified equations.f90 (3 surgical changes)
├── chains/                            # Raw MCMC chain files
│   ├── README.md                      # Chain file documentation
│   ├── iam_l2_run_a/                  # L2 Run A: IAM dual-sector
│   ├── iam_l2_run_c/                  # L2 Run C: LCDM baseline
│   └── iam_l2_run_d/                  # L2 Run D: IAM + RSD (pending)
├── yaml_configs/                      # Cobaya YAML files
│   ├── iam_level2a_fixed.yaml         # L2 Run A configuration
│   ├── iam_level2a_lcdm.yaml         # L2 Run C configuration
│   └── iam_level2a_rsd.yaml          # L2 Run D configuration
└── getdist_scripts/                   # Analysis scripts
    └── level2_comparison.py           # Run A vs Run C extraction
```

## Reproduction

To reproduce Level 2a results:

1. Clone CAMB v1.5.8: `git clone https://github.com/cmbant/CAMB.git`
2. Replace `fortran/equations.f90` with `equations_iam_level2.f90`
3. Compile: `cd fortran && make clean && make`
4. Run chains: `cobaya-run yaml_configs/iam_level2a_fixed.yaml`
5. Extract posteriors: `python getdist_scripts/level2_comparison.py`

## Level 2b/2c (Planned)

Level 2b adds the IAM term to `dtauda` in `results.f90` (background Friedmann equation modification). Level 2c adds late-universe likelihoods (BAO, Pantheon+). See the main README for details.
