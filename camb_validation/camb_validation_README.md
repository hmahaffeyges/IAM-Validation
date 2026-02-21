# Level 2 Validation: Dual-Sector Perturbation and Background Tests via Modified CAMB

This directory contains the Level 2 validation of IAM using vanilla CAMB v1.5.8 with direct Fortran modifications to `equations.f90`. Unlike Level 1 (which uses MGCAMB's built-in mu-Sigma parametrization), Level 2 implements the dual-sector mechanism explicitly: CDM and baryons experience a modified effective expansion rate (`adotoa_matter`) while photon perturbation equations remain standard.

## Tool Distinction

| Level | Tool | Modification | Scope |
|-------|------|-------------|-------|
| Level 1 | MGCAMB v1.5.2 | mu-Sigma parametrization (built-in) | Perturbation equations only |
| **Level 2** | **CAMB v1.5.8** | **Direct Fortran modification** | **Perturbation equations (2a), background Friedmann equation (2b/2c)** |

## Validation Summary

| Phase | What It Tests | Chains | Status | Key Result |
|-------|--------------|--------|--------|------------|
| **Level 2a** | Dual-sector perturbations, LCDM background | 3 | **COMPLETE** | Delta-chi2 = -0.01 (Planck); +2.00 apples-to-apples (Planck+RSD) |
| **Level 2b** | Modified dtauda background | 4 | **In progress** | -- |
| **Level 2c** | Modified background + late-universe data | 4-6 | Pending | -- |

## Fortran Modification

**File modified:** `equations.f90` -> renamed `equations_iam_level2.f90`

Three surgical changes to implement the dual-sector perturbation split:

1. **Module-level parameters** (lines 54-56): `iam_beta = 0.15765`, `iam_dual_sector = .true.`
2. **Matter-sector Hubble rate** (lines 2244-2252): `adotoa_matter = sqrt((grho + iam_beta * E(a) * grho_0) / 3)`
3. **CDM and baryon equations** (lines 2323-2337): use `adotoa_matter` instead of `adotoa`

The `dtauda` function (background) is **unmodified** in Level 2a. Photon equations are **untouched**. See `equations_iam_level2.f90` for the exact code.

## Level 2a Results -- COMPLETE (3 chains converged)

**Scope:** Background expansion is standard LCDM (unmodified). Only CDM and baryon perturbation equations use a matter-sector expansion rate computed from the IAM dual-sector prescription. Photon perturbation equations are untouched.

**Likelihood:** Planck NPIPE CamSpec TTTEEE + lowl TT + lowl EE + lensing CMBMarged

| Chain | Description | mu_0 | Dual-sector | Likelihood | R-1 | Status |
|-------|------------|-----|------------|------------|-----|--------|
| L2 Run A | IAM Level 2a, fixed | -0.13495 | ON | Planck CamSpec | 0.0099 | Converged |
| L2 Run C | LCDM baseline | 0 | OFF | Planck CamSpec | 0.0081 | Converged |
| L2 Run D | IAM Level 2a + RSD | -0.13495 | ON | Planck CamSpec + RSD | 0.0080 | Converged |

### Run A (IAM, Planck) vs Run C (LCDM baseline) -- Direct Comparison

| Parameter | Run A (IAM) | Run C (LCDM) | Shift |
|-----------|------------|-------------|-------|
| H0 | 67.161 +/- 0.467 | 67.188 +/- 0.465 | -0.04 sigma |
| sigma_8 | 0.7998 +/- 0.006 | 0.8087 +/- 0.006 | -1.08 sigma |
| ombh2 | 0.02217 +/- 0.00013 | 0.02218 +/- 0.00013 | -0.05 sigma |
| omch2 | 0.11994 +/- 0.00105 | 0.11989 +/- 0.00105 | +0.04 sigma |
| tau | 0.0537 +/- 0.0073 | 0.0532 +/- 0.0074 | +0.05 sigma |
| ns | 0.9630 +/- 0.0040 | 0.9630 +/- 0.0040 | -0.00 sigma |
| logA | 3.0407 +/- 0.0145 | 3.0393 +/- 0.0146 | +0.06 sigma |
| Omega_m | 0.3166 +/- 0.0065 | 0.3162 +/- 0.0065 | +0.04 sigma |
| S8 | 0.822 +/- 0.011 | 0.830 +/- 0.011 | -0.55 sigma |

**Delta-chi2 (mean posterior): -0.01. Delta-chi2 (best-fit): +0.54.**

All standard cosmological parameters shift by less than 0.06 sigma. sigma_8 shifts from 0.809 to 0.800, a 1.1 sigma downward shift consistent with the direction reported by weak lensing surveys (KiDS, DES, HSC). The sigma_8 suppression is confirmed as real growth physics (Possibility A): logA shift = 0.06 sigma, Omega_m shift = 0.04 sigma -- both negligible, ruling out parameter rebalancing as the source of suppression.

### Chi-Squared Comparison (Run A vs Run C)

| Likelihood | Run A (IAM) | Run C (LCDM) | Delta |
|-----------|------------|-------------|-------|
| Total chi2 | 10985.07 | 10985.08 | -0.01 |
| planck_2018_lowl.TT | 23.46 | 23.56 | -0.10 |
| planck_2018_lowl.EE | 396.92 | 396.85 | +0.06 |
| planck_NPIPE_highl_CamSpec.TTTEEE | 10555.17 | 10555.22 | -0.05 |
| planck_2018_lensing.CMBMarged | 9.52 | 9.45 | +0.08 |

### Run D (IAM + RSD) -- Apples-to-Apples Breakdown

**CRITICAL NOTE:** Run D includes RSD likelihood data that Run C (LCDM baseline) was not evaluated against. Raw Delta-chi2 = +6.26 is misleading -- it compares mismatched likelihood sets. The apples-to-apples comparison computes LCDM's RSD chi2 separately from Run C posterior parameters via CAMB.

| Component | IAM chi2 | LCDM chi2 | Delta-chi2 |
|-----------|---------|----------|-----------|
| CMB | 10984.93 | 10985.08 | -0.16 |
| RSD (7 points) | 6.42 | 4.27 | +2.15 |
| **Total (apples-to-apples)** | | | **+2.00** |

Run D parameters: sigma_8 = 0.7995 +/- 0.006, H0 = 67.189 +/- 0.460. All standard parameters shift by less than 0.06 sigma relative to Run C. The CMB component slightly prefers IAM (Delta-chi2 = -0.16). The RSD component shows Delta-chi2 = +2.15, driven primarily by the z = 0.850 outlier point (LCDM pull = 1.56 sigma) which both models struggle with.

**Companion script:** `getdist_scripts/rsd_apples_to_apples.py` computes the correct apples-to-apples comparison for any chain pair with mismatched likelihoods.

### Possibility A Confirmation

The sigma_8 suppression is driven by real growth physics (mu < 1), not by parameter rebalancing:

| Diagnostic | Run A | Run D | Threshold |
|-----------|-------|-------|-----------|
| logA shift | 0.06 sigma | 0.06 sigma | < 0.3 sigma |
| Omega_m shift | 0.04 sigma | 0.00 sigma | < 0.3 sigma |

Both diagnostics are far below the 0.3 sigma threshold. The suppression is genuine -- it comes from the dual-sector perturbation mechanism, not from A_s or Omega_m shifting along degeneracy directions.

## Level 2b: Modified Background (dtauda) -- IN PROGRESS

**Scope:** Adds the IAM term to the actual Friedmann equation in `dtauda` (in CAMB's `results.f90`), so that the background expansion history H(z) is modified. Because E(a) ~ 0 at recombination (z = 1100), CMB acoustic peaks (theta_s) are unaffected. The modification only matters at z < 2, affecting late-time distances.

**Key test:** Does theta_s shift? E(a) = exp(1 - 1/a) at a = 0.0009 (recombination) gives E ~ exp(-1110) ~ 0 to any numerical precision. The modification should be invisible to the CMB by construction.

**Chains (4 chains, apples-to-apples by construction):**

| Chain | Description | Likelihood | Compares Against |
|-------|------------|------------|-----------------|
| L2b Run A | IAM (background + perturbation) | Planck CamSpec | L2b Run C |
| L2b Run D | IAM + RSD | Planck CamSpec + RSD | L2b Run F |
| L2b Run C | LCDM baseline | Planck CamSpec | -- (baseline) |
| L2b Run F | LCDM + RSD baseline | Planck CamSpec + RSD | -- (baseline) |

## Directory Contents

```
camb_validation/
+-- README.md                          # This file
+-- equations_iam_level2.f90           # Modified equations.f90 (3 surgical changes)
+-- chains/                            # Raw MCMC chain files
|   +-- iam_level2_runA.*              # L2 Run A: IAM dual-sector (Planck)
|   +-- iam_level2_runC_lcdm.*         # L2 Run C: LCDM baseline (Planck)
|   +-- iam_level2_runD.*              # L2 Run D: IAM + RSD
+-- yaml_configs/                      # Cobaya YAML files
|   +-- iam_level2a_fixed.yaml         # L2 Run A configuration
|   +-- iam_level2a_lcdm.yaml         # L2 Run C configuration
|   +-- iam_level2a_rsd.yaml          # L2 Run D configuration
+-- getdist_scripts/                   # Analysis scripts
    +-- level2_extract_results.py      # Posterior extraction and comparison
    +-- rsd_apples_to_apples.py        # Apples-to-apples RSD comparison
```

## Reproduction

To reproduce Level 2a results:

1. Clone CAMB v1.5.8: `git clone https://github.com/cmbant/CAMB.git`
2. Replace `fortran/equations.f90` with `equations_iam_level2.f90`
3. Compile: `cd fortran && make clean && make`
4. Install Planck likelihoods via Cobaya
5. Run chains: `cobaya-run yaml_configs/iam_level2a_fixed.yaml`
6. Extract posteriors: `python getdist_scripts/level2_extract_results.py chains/`
7. For RSD comparisons: `python getdist_scripts/rsd_apples_to_apples.py chains/`

---

*Last updated: February 20, 2026*
*Status: Level 2a COMPLETE (3 chains, all passed). Level 2b in progress (4 chains, background modification).*
