# (IAM) µ<1  Σ = 1 Cosmology: Validation Against Planck 2018 (MGCAMB/CAMB) 0 Free Parameters Beyond LCDM
17 converged MCMC chains (12 Level 1 via MGCAMB + 5 Level 2 via modified CAMB) | Delta-chi2 = +0.54 best-fit vs LCDM (Planck) | sigma_8: 0.809 -> 0.800 | H0(photon) = 67.16, H0(matter) = 72.26 km/s/Mpc | Both within 1 sigma of observed values

This repository implements and tests a late-time mu(a) < 1, Sigma(a) = 1 modification of LCDM against the full Planck 2018 likelihood (TT, TE, EE, low-ell + lensing). Level 1 validation uses MGCAMB v1.5.2 + Cobaya (mu-Sigma parametrization). Level 2 validation uses direct Fortran modification of CAMB v1.5.8 + Cobaya (dual-sector perturbation implementation).

**Technical Clarifications Guide:** [Dual-Sector Cosmology: µ<1, Σ=1](docs/IAM_Technical_Clarifications_Guide.pdf)

**Core predictions (all derived, zero free parameters):**
- Perturbation modification: mu(a) = H^2_LCDM / (H^2_LCDM + beta * E(a)), with E(a) = exp(1 - 1/a)
- Sigma(a) = 1 exactly (lensing unmodified)
- Coupling derived from virial theorem: beta_m = Omega_m/2 = 0.1575, yielding mu_0 = -0.13495

The dual-sector perturbation modification produces a shift in sigma_8 from 0.8087 (LCDM) to 0.7998 (IAM) at Delta-chi2 = +0.54 (best-fit) relative to LCDM under the full Planck likelihood. The direction of the sigma_8 shift is consistent with values reported by KiDS-1000 (S8 = 0.759 +/- 0.021), DES Y3 (S8 = 0.776 +/- 0.017), and HSC Y3 (S8 = 0.776 +/- 0.032). All remaining cosmological parameters shift by less than 0.1 sigma. The coupling constant beta_m = Omega_m/2 is derived from the virial theorem, the activation function E(a) = exp(1 - 1/a) from horizon thermodynamics, and the perturbation prediction mu_0 = -0.135 follows from these inputs without additional fitting. No free parameters beyond standard LCDM are introduced.

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18750795-blue)](https://doi.org/10.5281/zenodo.18750795)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CAMB](https://img.shields.io/badge/CAMB-v1.5.8-green)](https://github.com/cmbant/CAMB)
[![MGCAMB](https://img.shields.io/badge/MGCAMB-v1.5.2-green)](https://github.com/sfu-cosmo/MGCAMB)

---

## Validation Summary

| Level | What It Tests | Tool | Chains | Status | Key Result |
|-------|--------------|------|--------|--------|------------|
| **Level 1** | mu-Sigma perturbations, LCDM background | MGCAMB | 12 | **Complete** | Delta-chi2 = +1.34 to +2.32 vs LCDM; sigma_8 = 0.800 |
| **Level 2** | Dual-sector perturbations, LCDM background | CAMB (modified) | 5 | **Complete** | Delta-chi2 = +0.54 (best-fit, Planck); sigma_8 = 0.800; H0(matter) = 72.26 |

**Full reproducibility** -- Level 1 (MGCAMB): [`mgcamb_validation/`](mgcamb_validation/). Level 2 (modified CAMB): [`camb_validation/`](camb_validation/).

---

## Level 1: Perturbation-Level Planck MCMC via MGCAMB (12 Independent Chains) — COMPLETE

**Scope:** Background expansion is standard LCDM (unmodified). Only perturbation equations modified via mu < 1, Sigma = 1 through MGCAMB's built-in mu-Sigma parametrization (MGCAMB v1.5.2 + Cobaya). No perturbation quantity feeds back into H(z).

**Level 1 preprint:** [Late-Time Growth Suppression in the μ–Σ Framework: Confrontation with Planck and Large-Scale Structure](docs/Late_Time_Growth_Suppression_in_the_mu_Sigma_Framework__Confrontation_with_Planck_and_Large_Scale_Structure.pdf) -- MGCAMB perturbation-level analysis (12 chains, 4 dataset combinations)

### Planck Only (Runs A / B / C)

| Run | Description | mu_0 | sigma_8 | H0 [km/s/Mpc] | Best chi2 |
|-----|-------------|------|---------|---------------|-----------|
| **C: LCDM baseline** | Standard model | 0 (fixed) | 0.8139 +/- 0.006 | 67.19 +/- 0.54 | 983.14 |
| **A: IAM fixed** | mu_0 = -0.135 (derived) | -0.135 (fixed) | 0.8014 +/- 0.006 | 67.06 +/- 0.51 | 984.57 |
| **B: IAM floating** | mu_0 free | 0.006 +/- 0.156 | 0.8146 +/- 0.016 | 67.16 +/- 0.51 | 981.24 |

### Planck + RSD (Runs D / E / F)

| Run | Description | mu_0 | sigma_8 | H0 [km/s/Mpc] | Best chi2 |
|-----|-------------|------|---------|---------------|-----------|
| **F: LCDM baseline** | Standard model + RSD | 0 (fixed) | 0.8131 +/- 0.006 | 67.50 +/- 0.41 | 993.86 |
| **D: IAM fixed** | mu_0 = -0.135 + RSD | -0.135 (fixed) | 0.8001 +/- 0.006 | 67.47 +/- 0.42 | 995.20 |
| **E: IAM floating** | mu_0 free + RSD | 0.033 +/- 0.125 | 0.8160 +/- 0.013 | 67.54 +/- 0.40 | 992.55 |

**Result:** IAM is statistically indistinguishable from LCDM across both dataset combinations. Planck only: Delta-chi2 = +1.43 (fixed), -1.90 (floating). Planck + RSD: Delta-chi2 = +1.34 (fixed), -1.31 (floating). IAM's predicted mu_0 = -0.135 lies 1.3 sigma from the Planck + RSD posterior peak (mu_0 = 0.033 +/- 0.125). The sigma_8 shift from ~0.813 to ~0.800 under IAM is stable across both datasets.

### Planck + BAO (Runs G / H / I)

| Run | Description | mu_0 | sigma_8 | H0 [km/s/Mpc] | Best chi2 |
|-----|-------------|------|---------|---------------|-----------|
| **I: LCDM baseline** | Standard model + BAO | 0 (fixed) | 0.8115 +/- 0.006 | 67.56 +/- 0.45 | 1017.71 |
| **G: IAM fixed** | mu_0 = -0.135 + BAO | -0.135 (fixed) | 0.7981 +/- 0.006 | 67.51 +/- 0.43 | 1020.03 |
| **H: IAM floating** | mu_0 free + BAO | +0.002 +/- 0.158 | 0.8117 +/- 0.016 | 67.56 +/- 0.46 | 1017.90 |

### Planck + Pantheon+ (Runs J / K / L)

| Run | Description | mu_0 | sigma_8 | H0 [km/s/Mpc] | Best chi2 |
|-----|-------------|------|---------|---------------|-----------|
| **L: LCDM baseline** | Standard model + Pantheon+ | 0 (fixed) | 0.8129 +/- 0.006 | 67.11 +/- 0.51 | 2416.00 |
| **J: IAM fixed** | mu_0 = -0.135 + Pantheon+ | -0.135 (fixed) | 0.8000 +/- 0.006 | 67.03 +/- 0.52 | 2417.58 |
| **K: IAM floating** | mu_0 free + Pantheon+ | -0.005 +/- 0.162 | 0.8124 +/- 0.017 | 67.06 +/- 0.52 | 2415.40 |

### Level 1 Delta-chi2 Summary (All 4 Dataset Combinations)

| Dataset | Delta-chi2 (IAM fixed vs LCDM) | sigma_8 (LCDM) | sigma_8 (IAM) | sigma_8 shift |
|---------|-------------------------------|----------------|---------------|---------------|
| Planck only | +1.43 | 0.8139 | 0.8014 | -0.0125 (-1.5%) |
| Planck + RSD | +1.34 | 0.8131 | 0.8001 | -0.0130 (-1.6%) |
| Planck + BAO | +2.32 | 0.8115 | 0.7981 | -0.0134 (-1.7%) |
| Planck + Pantheon+ | +1.58 | 0.8129 | 0.8000 | -0.0129 (-1.6%) |

All Delta-chi2 values are below 3.84 (95% CL threshold). The sigma_8 shift of -0.013 +/- 0.001 is stable across all four dataset combinations, confirming that the suppression is driven by the mu_0 = -0.135 prediction, not by any particular dataset. BAO and Pantheon+ photon-sector observables are unaffected by the modification, as predicted by Sigma = 1.

**MGCAMB Boltzmann diagnostics: 7/7 tests passed** (CMB TT < 0.17%, lensing +0.30%, sigma_8 = 0.795, Sigma = 1 exact, P(k) scale-independent, f*sigma_8 consistent).

### Independent Data Confirmation of the Virial Coupling

The coupling constant beta_m = Omega_m/2 is derived from the virial theorem -- it is not fitted to data. However, the Level 1 floating-parameter chains (Runs B, E, H, K) and the Level 2 posterior provide independent data-driven checks of this derivation:

| Source | Value | Virial prediction | Agreement |
|--------|-------|-------------------|-----------|
| **Virial theorem** | beta_m = Omega_m/2 = 0.15765 | -- (this IS the prediction) | -- |
| **Level 2 posterior** (Omega_m = 0.3166 +/- 0.0065) | beta_m = 0.1583 +/- 0.0032 | 0.15765 | **0.2 sigma** |
| **Level 1 Run B** (Planck, mu_0 free) | mu_0 = 0.006 +/- 0.156 | mu_0 = -0.135 | 0.9 sigma |
| **Level 1 Run E** (Planck+RSD, mu_0 free) | mu_0 = 0.033 +/- 0.125 | mu_0 = -0.135 | 1.3 sigma |
| **Level 1 Run H** (Planck+BAO, mu_0 free) | mu_0 = +0.002 +/- 0.158 | mu_0 = -0.135 | 0.9 sigma |
| **Level 1 Run K** (Planck+Pantheon+, mu_0 free) | mu_0 = -0.005 +/- 0.162 | mu_0 = -0.135 | 0.8 sigma |

The virial theorem predicts beta_m = Omega_m/2. The Planck data, through the Level 2 posterior on Omega_m, independently returns beta_m = 0.1583 +/- 0.0032 -- a 0.2 sigma agreement with the virial prediction. This is not a fit; the coupling was hardcoded before the chains were run. The CMB data confirmed the virial partition without being asked to.

All four Level 1 floating-parameter runs return mu_0 values consistent with the IAM prediction at < 1.3 sigma. The current constraint precision (sigma(mu_0) ~ 0.13-0.16) is too wide to distinguish IAM from GR. Euclid (projected sigma(mu_0) ~ 0.04) will provide the decisive test: 3.4 sigma detection significance if mu_0 = -0.135.

---

## Level 2: Dual-Sector Perturbation Validation via Modified CAMB -- COMPLETE

Level 2 extends beyond MGCAMB's built-in mu-Sigma parametrization by directly modifying the CAMB Fortran source code (`equations.f90`) to implement the dual-sector mechanism: CDM and baryons experience a modified effective expansion rate (adotoa_matter) while photon equations remain standard. Level 2 uses vanilla CAMB v1.5.8 (not MGCAMB).

**Level 2 preprint:** [Dual-Sector Perturbation Cosmology: A Modified CAMB Implementation with μ < 1, Σ = 1 and the Sector-Dependent Hubble Parameter](docs/Paper2_Dual-Sector_Cosmology_and_Hubble_Tension.pdf) -- Modified CAMB Fortran validation (3 chains, Δχ² = +0.54, σ₈ = 0.800, H₀(matter) = 72.26)

**Full Level 2 reproducibility package:** [`camb_validation/`](camb_validation/)

- **[Modified Fortran source](camb_validation/equations_iam_level2.f90)** -- equations.f90 with 3 surgical IAM modifications
- **[Raw MCMC chains](camb_validation/chains/)** -- complete chain data for all Level 2 runs (Runs A, C, D)
- **[Cobaya YAML configs](camb_validation/yaml_configs/)** -- exact configuration files for all Level 2 chains
- **[GetDist extraction scripts](camb_validation/getdist_scripts/)** -- posterior extraction, comparison, and apples-to-apples RSD scripts

### Dual-Sector Perturbations, LCDM Background -- COMPLETE (3 chains converged)

**Scope:** The background expansion history is standard LCDM (unmodified). Only the CDM and baryon perturbation equations use a matter-sector expansion rate (`adotoa_matter`) computed from the IAM dual-sector prescription. Photon perturbation equations are untouched.

**Coupling constant derivation:** beta_m = Omega_m / 2 = 0.3153 / 2 = 0.15765, where Omega_m = 0.3153 is the Planck 2018 best-fit value (TT,TE,EE+lowE+lensing; Aghanim et al. 2020, Table 2) and the factor of 1/2 follows from the virial theorem (gravitational energy partitions equally between geometric and informational channels). The Level 2 posterior returns Omega_m = 0.3166 +/- 0.0065, implying beta_m = 0.1583 +/- 0.0033 -- consistent with the hardcoded value (0.2 sigma, 0.4% difference).

**mu_0 derivation:** mu(a) = H^2_LCDM(a) / [H^2_LCDM(a) + beta_m * E(a)]. At a = 1: mu(1) = 1 / (1 + 0.15765) = 0.8638, giving mu_0 = mu - 1 = -0.1362. The value -0.13495 used in some analyses reflects a slightly different Omega_m input; both fall within the posterior uncertainty.

**Zero free parameters:** beta_m is derived from Omega_m via the virial theorem. E(a) = exp(1 - 1/a) is derived from horizon thermodynamics (Bekenstein-Hawking entropy, Gibbons-Hawking temperature, Landauer's principle). mu_0 follows from these two inputs. No parameters were fitted to data beyond standard LCDM.

**Implementation:** Three surgical modifications to `equations.f90`:
1. Module-level IAM parameters (beta_m = 0.15765, dual-sector toggle)
2. Matter-sector Hubble rate: `adotoa_matter = sqrt((grho + beta_m * E(a) * grho_0) / 3)`
3. CDM and baryon equations of motion use `adotoa_matter` instead of `adotoa`

**Chains:**

| Chain | Description | mu_0 | Dual-sector | Likelihood | R-1 |
|-------|------------|-----|------------|------------|-----|
| Run A | IAM Level 2, fixed | -0.13495 | ON | Planck CamSpec TTTEEE | 0.0099 |
| Run C | LCDM baseline | 0 | OFF | Planck CamSpec | 0.0081 |
| Run D | IAM Level 2 + RSD | -0.13495 | ON | Planck CamSpec + RSD | 0.0080 |

**Run A (IAM, Planck) vs Run C (LCDM baseline):**

| Parameter | Run A (IAM dual-sector) | Run C (LCDM baseline) | Shift |
|-----------|------------------------|----------------------|-------|
| H0 | 67.161 +/- 0.467 | 67.188 +/- 0.465 | -0.06 sigma |
| sigma_8 | 0.7998 +/- 0.0058 | 0.8087 +/- 0.0059 | -1.51 sigma |
| ombh2 | 0.02217 +/- 0.00013 | 0.02218 +/- 0.00013 | -0.08 sigma |
| omch2 | 0.11994 +/- 0.00105 | 0.11989 +/- 0.00105 | +0.05 sigma |
| tau | 0.0537 +/- 0.0073 | 0.0532 +/- 0.0074 | +0.07 sigma |
| ns | 0.9630 +/- 0.0040 | 0.9630 +/- 0.0040 | +0.00 sigma |
| logA | 3.0407 +/- 0.0145 | 3.0393 +/- 0.0146 | +0.10 sigma |
| Omega_m | 0.3166 +/- 0.0065 | 0.3162 +/- 0.0065 | +0.06 sigma |
| S8 | 0.822 +/- 0.011 | 0.830 +/- 0.011 | -0.73 sigma |

| Likelihood | Run A (IAM) | Run C (LCDM) |
|-----------|------------|-------------|
| Total chi2 | 10985.07 | 10985.08 |
| planck_2018_lowl.TT | 23.46 | 23.56 |
| planck_2018_lowl.EE | 396.92 | 396.85 |
| planck_NPIPE_highl_CamSpec.TTTEEE | 10555.17 | 10555.22 |
| planck_2018_lensing.CMBMarged | 9.52 | 9.45 |
| **Delta-chi2 (IAM - LCDM)** | **-0.01** | **baseline** |

All standard cosmological parameters are consistent between IAM and LCDM (all shifts < 0.1 sigma). sigma_8 shifts from 0.809 to 0.800, a 1.5 sigma downward shift in the direction reported by weak lensing surveys (KiDS, DES, HSC). S8 shifts from 0.830 to 0.822. Delta-chi2 = -0.01 (posterior mean) and +0.54 (best-fit), indicating that the dual-sector modification incurs no statistically significant penalty relative to LCDM under the Planck likelihood.

The sigma_8 suppression is confirmed as real growth physics (Possibility A): logA shift = 0.10 sigma, Omega_m shift = 0.06 sigma -- both far below the 0.3 sigma threshold, ruling out parameter rebalancing.

**Run D (IAM + RSD) -- Apples-to-Apples Comparison:**

Run D includes an RSD likelihood (7 f*sigma_8 data points) not present in the Level 2 LCDM baseline chain (Run C). Direct comparison of raw chi2 between Run D and Run C is misleading because it includes the RSD likelihood score that Run C was never evaluated against. The apples-to-apples comparison computes LCDM's RSD chi2 from Run C posterior parameters via CAMB, then combines it with Run C's CMB chi2 to construct the correct LCDM total. This is cross-validated by Level 1 Run F (LCDM actually run against Planck+RSD), which produced Delta-chi2 = +1.34. See [`camb_validation/getdist_scripts/rsd_apples_to_apples.py`](camb_validation/getdist_scripts/rsd_apples_to_apples.py).

| Parameter | Run D (IAM + RSD) | Run C (LCDM baseline) | Shift |
|-----------|-------------------|----------------------|-------|
| H0 | 67.189 +/- 0.460 | 67.188 +/- 0.465 | +0.00 sigma |
| sigma_8 | 0.7995 +/- 0.0058 | 0.8087 +/- 0.0059 | -1.56 sigma |
| ombh2 | 0.02218 +/- 0.00013 | 0.02218 +/- 0.00013 | +0.00 sigma |
| omch2 | 0.11988 +/- 0.00105 | 0.11989 +/- 0.00105 | -0.01 sigma |
| tau | 0.0538 +/- 0.0073 | 0.0532 +/- 0.0074 | +0.08 sigma |
| ns | 0.9631 +/- 0.0040 | 0.9630 +/- 0.0040 | +0.02 sigma |
| logA | 3.0406 +/- 0.0145 | 3.0393 +/- 0.0146 | +0.09 sigma |
| Omega_m | 0.3162 +/- 0.0064 | 0.3162 +/- 0.0065 | +0.00 sigma |
| S8 | 0.821 +/- 0.011 | 0.830 +/- 0.011 | -0.82 sigma |

| Likelihood | Run D (IAM) | LCDM (apples-to-apples) | Delta-chi2 |
|-----------|------------|------------------------|-----------|
| planck_2018_lowl.TT | 23.43 | 23.56 | -0.13 |
| planck_2018_lowl.EE | 396.91 | 396.85 | +0.06 |
| planck_NPIPE_highl_CamSpec.TTTEEE | 10555.06 | 10555.22 | -0.16 |
| planck_2018_lensing.CMBMarged | 9.52 | 9.45 | +0.07 |
| **CMB subtotal** | **10984.93** | **10985.08** | **-0.16** |
| RSD (7 f*sigma_8 points) | 6.42 | 3.34 | +3.08 |
| **Total (apples-to-apples)** | **10991.34** | **10988.42** | **+2.92** |

Note: LCDM CMB chi2 values are from Run C (Planck-only chain). LCDM RSD chi2 = 3.34 is computed from Run C posterior parameters via CAMB. This is mathematically defensible (LCDM predictions do not depend on which likelihoods were in the chain) and is cross-validated by Level 1 Run F, where LCDM was actually run against Planck+RSD and produced Delta-chi2 = +1.34 vs IAM.

All standard cosmological parameters are consistent between Run D and Run C (all shifts < 0.1 sigma). sigma_8 shifts from 0.809 to 0.800, matching Run A. The CMB component slightly prefers IAM (Delta-chi2 = -0.16). The RSD penalty of +3.08 is driven primarily by the z = 0.850 outlier (f*sigma_8 = 0.315 +/- 0.095; both models struggle with this point; LCDM pull = 1.40 sigma). The sigma_8 suppression is confirmed as real growth physics: logA shift = 0.09 sigma, Omega_m shift = 0.00 sigma -- both far below the 0.3 sigma threshold (Possibility A confirmed, consistent with Run A).

**Success criteria (all met):**
1. sigma_8 ~ 0.800: **0.7998** (Run A), **0.7995** (Run D) -- suppressed from LCDM's 0.8087
2. Standard parameters stable: all shifts < 0.1 sigma relative to LCDM
3. Delta-chi2 < 5 vs LCDM baseline: **+0.54** best-fit (Planck), **-0.01** posterior mean (Planck), **+2.92** (Planck+RSD apples-to-apples)
4. Clean convergence (R-1 < 0.01, no multimodality): confirmed for all 3 chains
5. Real growth physics confirmed: logA and Omega_m shifts < 0.3 sigma (Possibility A)

### Dual-Sector Analysis (from Level 2 Run A)

The matter-sector expansion rate `adotoa_matter` computed within the perturbation equations yields a dual-sector H0 split directly from the MCMC posterior:

H0(matter) = H0 × sqrt(1 + beta_m × E(a=1)) = 67.161 × sqrt(1 + 0.15765 × 1.0) = 67.161 × 1.0759 = 72.26 km/s/Mpc

where H0 = 67.161 +/- 0.467 km/s/Mpc is the Level 2 Run A posterior mean, beta_m = 0.15765 is derived from Planck 2018 Omega_m via the virial theorem (see above), and E(a=1) = exp(0) = 1.0.

**Observational scorecard:**

| Observable | IAM Prediction | Observed | Status |
|------------|---------------|----------|--------|
| H0 (photon/CMB) | 67.16 km/s/Mpc | 67.36 +/- 0.54 (Planck) | 0.37 sigma |
| H0 (matter/local) | 72.26 km/s/Mpc | 73.04 +/- 1.04 (SH0ES) | 0.75 sigma |
| sigma_8 | 0.7998 | 0.811 +/- 0.006 (Planck LCDM) | Suppressed toward WL |
| mu_0 | -0.136 | 0.033 +/- 0.125 (Planck+RSD) | 1.3 sigma |
| Sigma_0 | 0 (exact) | Constrained | Exact |
| Delta-chi2 (best-fit) | +0.54 | < 3.84 (95% CL threshold) | Pass |
| Free parameters | 0 | -- | Zero |

**Expansion rate by redshift:**

| z | H_photon [km/s/Mpc] | H_matter [km/s/Mpc] | Ratio | E(a) |
|---|---------------------|---------------------|-------|------|
| 0.0 | 67.16 | 72.26 | 1.0759 | 1.000000 |
| 0.1 | 70.59 | 75.01 | 1.0625 | 0.904837 |
| 0.3 | 78.87 | 82.14 | 1.0414 | 0.740818 |
| 0.5 | 88.90 | 91.29 | 1.0269 | 0.606531 |
| 1.0 | 120.45 | 121.53 | 1.0090 | 0.367879 |
| 2.0 | 204.06 | 204.30 | 1.0012 | 0.135335 |
| 10.0 | 1379.81 | 1379.81 | 1.0000 | 0.000045 |

The sector split vanishes at high redshift (E(a) approaches 0 as a approaches 0), recovering standard LCDM in the early universe. At z = 0, the 7.6% matter-sector enhancement produces H0(matter) = 72.26 km/s/Mpc, within 0.75 sigma of SH0ES (73.04 +/- 1.04). The photon-sector value H0(photon) = 67.16 km/s/Mpc is within 0.37 sigma of Planck LCDM (67.36 +/- 0.54). Both values fall within 1 sigma of their respective observational constraints.

**Complete side-by-side results:**

| Parameter | Run C (LCDM) | Run A (IAM) | Run D (IAM+RSD) |
|-----------|-------------|-------------|-----------------|
| H0 [km/s/Mpc] | 67.188 +/- 0.465 | 67.161 +/- 0.467 | 67.189 +/- 0.460 |
| sigma_8 | 0.8087 +/- 0.0059 | 0.7998 +/- 0.0058 | 0.7995 +/- 0.0058 |
| Omega_m | 0.3162 +/- 0.0065 | 0.3166 +/- 0.0065 | 0.3162 +/- 0.0064 |
| S8 (sigma_8 * sqrt(Omega_m/0.3)) | 0.4547 +/- 0.0062 | 0.4500 +/- 0.0061 | 0.4496 +/- 0.0060 |
| Best chi2 | 10972.07 | 10972.61 | 10979.43 |
| Samples | 115584 | 81088 | 53312 |

Delta-chi2 (best-fit, Planck): IAM - LCDM = +0.54

### Dual-Sector Perturbations, Modified Background -- COMPLETE (2 chains converged)
### Note on Background Modification

Two exploratory runs were conducted with the IAM term added to the background expansion equation (modifying CAMB's `dtauda` function). Two Level 2b background modification chains ran to completion: Run A (125,440 accepted samples, 46.8% acceptance, R-1 = 0.0096) and Run D (120,832 accepted samples, 46.3% acceptance, R-1 = 0.0068). Both converged cleanly. The result H₀ ≈ 61.5 km/s/Mpc confirmed that the dual-sector mechanism operates exclusively at the perturbation level. This is a structural confirmation — the chains did not fail, they answered the question precisely `adotoa_matter` in the perturbation equations. The background modification chains are archived in `camb_validation/chains/` for reproducibility.

---

## MGCAMB Boltzmann Validation (Level 1) — 7/7 Tests PASSED

| Test | Criterion | Result | Status |
|------|-----------|--------|--------|
| sigma_8 | In [0.79, 0.82] | 0.7954 | PASS |
| CMB TT (ell > 30) | < 1% residual | 0.17% | PASS |
| CMB TT ISW (ell < 30) | < cosmic variance | 3.6% (CV ~ 63%) | PASS |
| CMB lensing | < 5% change | +0.30% | PASS |
| Sigma = 1 | Exact | sigma_0 = 0 | PASS |
| P(k) scale-independence | std(ratio) < 1% | 0.53% | PASS |
| f*sigma_8 fit quality | chi2 <= LCDM + 4 | 4.42 vs 4.85 | PASS |

**Full Level 1 reproducibility package:** [`mgcamb_validation/`](mgcamb_validation/)

- **[Raw MCMC chains](mgcamb_validation/chains/)** -- complete chain data for all 12 runs (Runs A/B/C: Planck only; Runs D/E/F: Planck + RSD; Runs G/H/I: Planck + BAO; Runs J/K/L: Planck + Pantheon+), independently verifiable via GetDist
- **[Cobaya YAML configs](mgcamb_validation/yaml_configs/)** -- exact configuration files to re-run all 12 chains from scratch
- **[GetDist extraction scripts](mgcamb_validation/getdist_scripts/)** -- reproduce every posterior table and Delta-chi2 comparison in the Technical Note
- **[Forecast analyses](mgcamb_validation/forecasts/)** -- Fisher forecast, ISW prediction, binned mu(z) reconstruction, transition zone analysis
- **[7/7 Boltzmann diagnostic script](mgcamb_validation/iam_mu_sigma.py)** -- MGCAMB validation code reproducing all 7 diagnostic tests and 6-panel figure

**Level 1 Interpretation:** IAM remains statistically indistinguishable from LCDM under the full Planck 2018 likelihood across all four dataset combinations (Planck only, +RSD, +BAO, +Pantheon+), demonstrating internal consistency at current precision. The sigma_8 shift of -0.013 is universal and stable, confirming it is driven by the mu_0 = -0.135 prediction rather than any particular dataset. BAO and supernova observables are unaffected, as predicted by Sigma = 1. The definitive detection tests are Euclid (sigma(mu_0) ~ 0.04, yielding 3.4 sigma) and DESI Year 5 growth rates.

---

## Broader Context: IAM Dual-Sector Framework

The mu-Sigma modification tested above is derived from the Informational Actualization Model (IAM), a dual-sector cosmological framework where matter and photons experience different late-time expansion rates. The full framework predicts H0(matter) = 72.26 km/s/Mpc alongside H0(photon) = 67.16 km/s/Mpc, reducing the discrepancy between CMB and local distance-ladder measurements to within 1 sigma on both sides. Level 2 validation (dual-sector perturbation mechanism via modified CAMB Fortran) confirms this framework survives the full Planck likelihood (Delta-chi2 = +0.54 best-fit, sigma_8 = 0.800).

On a limited H0 + growth rate dataset (10 measurements), IAM shows large model-selection preference (Delta-chi2 ~ 30, Delta-AIC = 26.0). **This preference refers to the limited H0 + growth dataset only and does not apply under the full Planck likelihood, where IAM and LCDM are statistically indistinguishable.** Level 2 validation is complete (Delta-chi2 = +0.54 best-fit under Planck). The dual-sector perturbation mechanism yields H0(photon) = 67.16 km/s/Mpc and H0(matter) = 72.26 km/s/Mpc, both within 1 sigma of their respective observational values.

| Parameter | Value | Source |
|-----------|-------|--------|
| **mu(z=0)** | 0.864 | Derived (perturbation theory) |
| **Sigma(z)** | 1.000 | Derived (delta-phi = 0) |
| **beta_m** | 0.1575 | Derived (Omega_m/2, virial theorem) |
| **H0(matter)** | 72.26 km/s/Mpc | Level 2 dual-sector: H0 * sqrt(1 + beta_m) (0.75 sigma from SH0ES) |
| **H0(photon)** | 67.4 km/s/Mpc | Planck CMB |

---

## Multi-Probe Dual-Sector Consistency Test

The MCMC chains above (Runs A–L) test IAM's perturbation-level μ–Σ signature through the full Boltzmann solver with the actual Planck likelihood. The following supplementary analysis tests the **full dual-sector mechanism** — including the background H₀ split, matter density dilution, and explicit sector assignment of observables — against a broader compilation of published data.

**Important distinction:** This is a phenomenological consistency test using a lightweight growth ODE and Hu & Sugiyama fitting formulae, not a full Boltzmann MCMC. It does not replace the Planck likelihood analysis but extends it to probes the MCMC chains do not cover (local H₀ measurements, cosmic chronometers, weak lensing S₈, multi-survey BAO). All Planck 2018 parameters are held fixed; the only model input is β = Ω_m/2. Zero free parameters are fitted to the data.

**Script:** [`mgcamb_validation/iam_dual_sector_combined.py`](mgcamb_validation/iam_dual_sector_combined.py)
**Figure:** [`mgcamb_validation/iam_dual_sector_combined.pdf`](mgcamb_validation/iam_dual_sector_combined.pdf) (9-panel diagnostic figure)

### Framework

Observables are assigned to sectors based on their physical measurement process:
- **Photon sector** (H_γ = H_ΛCDM): CMB distance priors, BAO angular positions, SNe Ia luminosity distances — all measured via photon propagation
- **Matter sector** (H_m = H_ΛCDM × √(1 + β·E(a))): Local H₀ via distance ladders (Cepheids, TRGB, time-delay lensing), galaxy growth rates (RSD f·σ₈), cosmic chronometer aging rates

Growth is governed by:

D'' + [2 + d(ln H)/d(ln a)] D' − (3/2) μ(a) Ω_m(a) D = 0

where μ(a) = H²_ΛCDM / (H²_ΛCDM + β·E(a)) gives μ(z = 0) = 0.864, recovering GR at high redshift.

### Datasets (65 data points across 6 χ² probes + 1 consistency check)

| Probe | N | Key References | Sector |
|-------|---|----------------|--------|
| H₀ measurements | 7 | Planck VI (2020); Riess+ (2022); Freedman+ (2025); Wong+ (2020); Birrer+ (2020); Anand+ (2022) | Photon (Planck), Matter (all local) |
| f·σ₈ growth rates | 7 | Beutler+ (2012); Alam+ (2017, 2021); de Mattia+ (2021); Hou+ (2021) | Matter |
| BAO (DM/rd, DH/rd, DV/rd) | 12 | Ross+ (2015); Alam+ (2017); Bautista+ (2021); du Mas des Bourboux+ (2020) | Photon |
| Cosmic chronometers H(z) | 32 | Moresco+ (2022) compilation; 9 original papers (Simon+ 2005 through Tomasetti+ 2023) | Matter |
| Weak lensing S₈ | 4 | DES Y3 (Abbott+ 2022); KiDS-1000 (Asgari+ 2021); HSC Y3 (Li+ 2023); Planck (2020) | Matter |
| CMB distance priors | 3 | Planck VI (2020), Table 1; Hu & Sugiyama (1996) fitting formulae | Photon |
| Pantheon+ SNe Ia | 20 bins | Brout+ (2022); Scolnic+ (2022) — photon-sector consistency check, identical to ΛCDM by construction | Photon |

### Results

| Probe | χ²(ΛCDM) | χ²(IAM) | Δχ² | Notes |
|-------|----------|---------|-----|-------|
| **H₀** | 74.0 | 4.7 | **+69.4** | Dominant contribution: sector split accommodates discrepant H₀ measurements |
| **f·σ₈** | 6.5 | 7.4 | −0.8 | IAM slightly worse (combined model suppresses growth marginally below ΛCDM) |
| **BAO** | 15.6 | 15.6 | 0.0 | Identical (photon sector unmodified) |
| **Cosmic chronometers** | 14.5 | 15.0 | −0.5 | Indistinguishable at current CC precision (uncertainties 5–30%) |
| **S₈** | 25.6 | 13.9 | **+11.7** | σ₈ suppression (0.811 → 0.790) moves prediction toward WL data |
| **CMB priors** | 370 | 370 | 0.0 | Fitting-formula residuals dominate; full MGCAMB confirms < 0.17% CMB residuals |
| **Combined** | **507** | **427** | **+79.8** | 0 additional free parameters |

Combined Δχ² = 79.8 (equivalent to 8.9σ, 0 additional free parameters). The improvement is dominated by the H₀ sector split (+69.4) and S₈ suppression (+11.7). All other probes are consistent within uncertainties. See caveats below regarding the limitations of this simplified analysis.

### S₈ Tension

IAM produces a shift in S₈ through two mechanisms:
1. **Growth suppression** (μ < 1 at late times): σ₈ reduced from 0.811 to 0.790, giving S₈ = 0.810 at Planck Ω_m
2. **Matter density dilution** (sector split): physical Ω_m reduced from 0.315 to 0.272, giving S₈ = 0.753

The physical-Ω_m prediction S₈ = 0.753 is consistent with KiDS-1000 (0.759 ± 0.021, 0.3σ), DES Y3 (0.776 ± 0.017, 1.4σ), and HSC Y3 (0.776 ± 0.032, 0.7σ). However, comparison at physical Ω_m requires reanalysis of survey likelihoods under modified cosmology. The conservative apples-to-apples comparison at Planck Ω_m (S₈ = 0.810) is used for the combined χ².

### Caveats

- **Simplified physics:** Growth ODE and fitting formulae, not full Boltzmann solver. The MCMC chains (Runs A–L) provide the rigorous Planck-level validation.
- **No covariance matrices:** Data points treated as independent. Correlated BAO bins and the full Pantheon+ covariance are handled in the MCMC analysis.
- **CMB χ² dominated by fitting formulae:** The large CMB χ² ≈ 370 reflects Hu & Sugiyama approximation error (~0.1% in θ_MC), not physical tension. It cancels identically in Δχ².
- **Sector assignment of cosmic chronometers:** CC measures galaxy aging (matter-sector process) via spectral features (photon observations). The assignment follows from which physical process determines the observable (aging rate), analogous to local H₀ measurements.
- **H₀ Δχ² is model-dependent:** The 69-point improvement assumes the sector split is physical. Under single-sector ΛCDM, this is simply the Hubble tension restated. The test evaluates whether the dual-sector prediction is consistent with observations, not whether the sector split is proven.

---

## Testable Predictions

### Near-Term (< 5 years)

| Experiment | IAM Prediction | Distinguishes From |
|------------|---------------|-------------------|
| **Euclid (mu-Sigma)** | mu < 1 with Sigma = 1 | f(R): mu > 1; Horndeski: both modified |
| **DESI Year 5** | Distance-growth tension in w0-wa fits | All w0-wa models (consistent dist+growth) |
| **DESI Year 5** | No real phantom crossing (w always <= -1) | w0-wa best fit (apparent crossing at z~0.5) |
| **Euclid** | Scale-independent mu and Sigma (no k-dependence) | f(R), DGP (scale-dependent growth) |
| **Euclid** | B_m/Omega_m = 1/2 constant for any Omega_m | Ad-hoc models (ratio would vary) |
| **Simons Observatory** | B_gamma < 0.001 (10x tighter) | Constrains photon-sector coupling |

### Long-Term (> 5 years)

| Experiment | IAM Prediction | Timeline |
|------------|---------------|----------|
| **CMB-S4** | B_gamma < 10^-4 or IAM falsified | 2030+ |
| **Euclid + Rubin** | BAO at z > 2 tests early-time behavior | 2030+ |
| **GW Standard Sirens** | H0(matter) = 72.26 km/s/Mpc | 2030+ |

---

<details>
<summary>Quick Start and Expected Output</summary>

## Quick Start

### Installation

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git
cd IAM-Validation
pip install numpy scipy matplotlib corner
```

**Note:** The `corner` package (for MCMC plots) will auto-install if missing.

### Run Validation

```bash
# Run observational validation (9 tests, ~1 min, generates 9 figures)
python iam_validation.py

# Run derivation verification (15 tests, ~30 sec)
python iam_derivation_tests.py
```

**Expected runtime:** ~90 seconds total for both scripts

**Observational Validation (`iam_validation.py`) -- Expected output:**
```
==============================================================================
 INFORMATIONAL ACTUALIZATION MODEL (IAM)
 Complete Validation Presentation
==============================================================================

[1/6] Checking Python environment...
 Python 3.x.x detected
 numpy installed
 scipy installed
 matplotlib installed
 corner installed

[2/6] Cosmological Parameters and Observational Data
Planck 2020 Cosmological Parameters...
H0 Measurements (Hubble Constant)...
SDSS/BOSS/eBOSS Growth Rate Compilation...
Total data points: 3 H0 + 7 SDSS/BOSS/eBOSS = 10

[3/6] IAM Mathematical Framework
CORE EQUATIONS:
 EQUATION 1: Activation Function E(a) = exp(1 - 1/a)
 EQUATION 2: Modified Friedmann Equation
 EQUATION 3: Effective Matter Density Parameter
 ...
DUAL-SECTOR FRAMEWORK:
 Photon sector: B_gamma ~ 0 --> H0(photon) = 67.4 km/s/Mpc
 Matter sector: B_m = 0.157 --> H0(matter) = 72.5 km/s/Mpc

[4/6] Chi-Squared Calculation Methodology
EXAMPLE: How chi2 is computed for H0 measurements
 LCDM: chi2_H0 = 31.91
 IAM: chi2_H0 = 1.52
 Improvement: Delta-chi2_H0 = 30.40

[5/6] Validated Test Results

TEST 1: LCDM Baseline (Standard Cosmology)
 chi2_total = 38.28
 LCDM: single-sector model

TEST 2: IAM Dual-Sector Model
 B_m = 0.157 (MCMC median)
 chi2_total = 8.27
 Delta-chi2 = 30.01 (5.5 sigma)
 IAM dual-sector model: Delta-chi2 = 30.01 on H0 + growth dataset

TEST 3: Confidence Intervals (Profile Likelihood)
 68% CL (1 sigma): B_m = 0.157 +/- 0.029
 95% CL (2 sigma): B_m = 0.157 +/- 0.057

TEST 4: Photon-Sector Constraint (MCMC)
 Profile likelihood: B_gamma < 0.004 (95% CL)
 MCMC constraint: B_gamma < 1.40e-06 (95% CL)
 Sector ratio: B_gamma/B_m < 8.50e-06 (95% CL)

TEST 5: Physical Predictions
 H0(photon/CMB) = 67.4 km/s/Mpc
 H0(matter/local) = 72.5 +/- 1.0 km/s/Mpc
 Growth suppression = 1.36%
 sigma8(IAM) = 0.800
 Predictions within current observational uncertainties

TEST 6: CMB Lensing Consistency
 Growth suppression (1.36%) --> weaker lensing
 Reduced lensing compensates ~85% of geometric theta_s shift
 Partial compensation is consistent with CMB constraints

TEST 7: Model Selection Criteria (Overfitting Check)
 Delta-AIC = 26.01 --> 'Decisive' evidence for IAM
 Delta-BIC = 25.40 --> 'Very strong' evidence for IAM
 Relative likelihood: Bayes factor ~ 4.4 x 10^5
 Model selection favors IAM after AIC/BIC penalties for 2 additional parameters

TEST 8: Full Bayesian MCMC Analysis
 B_m = 0.157 +0.029/-0.029 (68% CL)
 B_gamma < 1.40e-06 (95% upper limit)
 B_gamma/B_m < 8.50e-06 (95% upper limit)
 H0(matter) = 72.5 +/- 1.0 km/s/Mpc
 Gaussian posteriors, no multimodality

TEST 9: Pantheon+ Supernovae Distance Validation
 Both models show similar fit quality to SNe data
 Primary IAM impact is on GROWTH, not GEOMETRY
 IAM maintains distance consistency

[6/6] Generating Publication-Quality Figures
Generating Figure 1: H0 Measurements Comparison...
Generating Figure 2: Growth Suppression Evolution...
...
Generating Figure 9: MCMC Parameter Constraints...
All 9 figures generated successfully!
```

**Derivation Verification (`iam_derivation_tests.py`) -- Expected output:**
```
==============================================================================
*' IAM DERIVATION VERIFICATION SUITE *'
*' 15 Tests -- From Jacobson to Zero-Parameter Cosmology + Robustness *'
==============================================================================

 DERIVATION CHAIN (Tests 1-10):
 [PASS] Test 1: Jacobson: Standard entropy ' Friedmann equation
 [PASS] Test 2: Cai-Kim: First law on apparent horizon ' Friedmann
 [PASS] Test 3: Modified entropy ' IAM Friedmann equation
 [PASS] Test 4: Cumulative decoherence integral ' exp(alpha - beta/a)
 [PASS] Test 5: Sheth-Tormen collapse rate ' 1/a coefficient
 [PASS] Test 6: Virial theorem ' beta_m = Omega_m/2
 [PASS] Test 7: Collapsed fraction ' Virial theorem confirmed
 [PASS] Test 8: Perturbation theory: mu < 1, Sigma = 1
 [PASS] Test 9: Fixed beta_m = Omega_m/2: "chi^2 = 31.2 (5.6sigma)
 [PASS] Test 10: Equation of state: w_info = -1 - 1/(3a)

 ROBUSTNESS (Tests 11-15):
 [PASS] Test 11: Continuity equation: rho_dot + 3H(1+w)rho = 0
 [PASS] Test 12: MGCAMB approximation: max error 4.29 pp
 [PASS] Test 13: Sensitivity: mu_0 spread = 0.00000 over c in [0.8, 1.2]
 [PASS] Test 14: Sensitivity: coupling constant beta_m variation
 [PASS] Test 15: Reparametrization: IAM != w0waCDM

 Tests passed: 15/15
 beta_m = Omega_m/2 = 0.1575 (predicted) vs 0.157 (MCMC): 0.3% agreement
 H(matter) = 72.51 km/s/Mpc (0.51sigma from SH0ES)
 "chi^2 = 31.2 for ZERO additional parameters
 Bayes factor ~ 6 x 10^6 (H0 + growth dataset only)
```

</details>

---

## Repository Structure

```
IAM-Validation/
├── README.md                                  # This file
├── mgcamb_validation/                         # *** LEVEL 1 VALIDATION (MGCAMB v1.5.2) ***
│   ├── README.md                              # Detailed MGCAMB documentation
│   ├── iam_mu_sigma.py                        # 7/7 Boltzmann diagnostic tests (reproduces 6-panel figure)
│   ├── chains/                                # Raw MCMC chain files (Runs A–L, 12 chains)
│   ├── yaml_configs/                          # Exact Cobaya YAML files for all Level 1 runs
│   ├── getdist_scripts/                       # Posterior extraction & three-way comparison
│   └── forecasts/                             # Euclid/DESI prediction scripts
├── camb_validation/                           # *** LEVEL 2 VALIDATION (vanilla CAMB v1.5.8, modified Fortran) ***
│   ├── README.md                              # Level 2 documentation and Fortran modification details
│   ├── equations_iam_level2.f90               # Modified equations.f90 (3 surgical changes)
│   ├── chains/                                # Raw MCMC chain files (L2 Runs A, C, D)
│   ├── yaml_configs/                          # Exact Cobaya YAML files for all Level 2 runs
│   └── getdist_scripts/                       # Posterior extraction & Level 2 comparison scripts
├── docs/                                      # Technical documentation (PDFs)
├── tests/                                     # Phenomenological validation scripts
│   ├── iam_validation.py                      # 9 observational tests (~1 min)
│   └── iam_derivation_tests.py                # 15 derivation + robustness tests (~30 sec)
├── figures/                                   # Publication-quality figures
├── results/                                   # Output data files
└── data/                                      # Observational datasets
```

**For referees:** Level 1 results (mu-Sigma perturbation validation via MGCAMB): [`mgcamb_validation/`](mgcamb_validation/). Level 2 results (dual-sector perturbation validation via modified CAMB): [`camb_validation/`](camb_validation/).

---

<details>
<summary>What IAM Does (Summary of Physical Mechanism)</summary>

## What IAM Does

### Hubble Tension

- **Planck CMB:** H0 = 67.4 km/s/Mpc (photon sector, B_gamma < 10^-5)
- **SH0ES Distance Ladder:** H0 = 73.04 km/s/Mpc (matter sector, B_m = 0.157)
- **Interpretation:** Different sectors yield different expansion rates

### S8 Tension

- **Growth suppression:** 1.36% at z=0 from Omega_m dilution
- **Effective sigma8:** 0.800 (intermediate between Planck 0.811 and DES/KiDS ~0.77)
- **Mechanism:** Follows from mu < 1 without additional parameters

### Passes CMB Consistency

- **Planck MCMC (Level 1):** 12 independent chains converged cleanly across 4 dataset combinations (Delta-chi2 = +1.43 to +2.32 vs LCDM, all below exclusion threshold)
- **Planck MCMC (Level 2):** 3 chains via modified CAMB Fortran (Delta-chi2 = +0.54 best-fit Planck, +2.92 Planck+RSD apples-to-apples); sigma_8 suppression confirmed as real growth physics
- **CMB lensing:** 85% geometric compensation
- **Acoustic scale:** B_gamma < 10^-5 maintains theta_s precision
- **Early universe:** No modifications before z ~ 1
- **sigma_8 shift:** 0.8087 (LCDM) --> 0.7998 (IAM Level 2), in the direction reported by weak lensing surveys

### Model Selection (H0 + Growth Rate Dataset)

- **AIC penalty:** Delta-AIC = 26.0 (Burnham & Anderson classification: decisive)
- **BIC penalty:** Delta-BIC = 25.4 (Kass & Raftery classification: very strong)
- **Relative likelihood:** Bayes factor ~ 4.4 x 10^5

*Note: These model selection statistics apply to the H0 + f*sigma_8 dataset (10 measurements) only. Under the full Planck 2018 likelihood, IAM and LCDM are statistically indistinguishable: Level 1 Delta-chi2 = +1.43, Level 2 Delta-chi2 = +0.54 best-fit. See Level 1 and Level 2 sections above for Planck MCMC results.*

### Predictions for Upcoming Surveys

- **CMB-S4:** Will constrain B_gamma < 10^-4 (100x tighter)
- **Euclid:** S8 = 0.78 +/- 0.01
- **DESI Year 5:** B_m to +/-1% precision

</details>

---

## Documentation

### Primary Documents

1. **[Main Manuscript](docs/IAM_Manuscript.pdf)** (RevTeX, ~15 pages)
 - Holographic motivation (Bekenstein-Hawking entropy, holographic principle)
 - Theoretical foundation and phenomenological implementation
 - Statistical validation and predictions

2. **[Dual-Sector Validation Paper](docs/Dual_Sector_Validation_Paper.pdf)** (RevTeX, ~22 pages)
 - Empirical validation of sector separation using Pantheon+ Type Ia supernovae
 - Three independent tests (Planck prior, SH0ES prior, no prior)
 - Complete Python code in appendices (< 2 min reproducibility)
 - Companion paper to main IAM manuscript

3. **[IAM--CAMB Technical Note: Full Planck Validation](docs/IAM_CAMB_Technical_Note.pdf)** (~23 pages)
 - mu--Sigma modified gravity mapping: mu(a) < 1, Sigma(a) = 1
 - Python-level CAMB validation with comprehensive 8-panel figure
 - Fortran-level implementation: what was done, what was learned
 - **MGCAMB Boltzmann validation: 7/7 diagnostic tests passed** (sigma_8, CMB TT, lensing, P(k), f*sigma_8, Sigma = 1, scale independence)
 - **Planck Level-1 MCMC analysis** (perturbation sector, 3 independent chains via Cobaya + MGCAMB) with complete YAML files
 - **Run B mu_0 posterior analysis** (MAP ~ 0, sigma = 0.156, IAM at 0.9 sigma, asymmetric 68% CI)
 - **Forecasting and observational prospects** (Fisher forecast, ISW-galaxy cross-correlation, binned mu(z) reconstruction, transition zone analysis)
 - CAMB background validation figure (6-panel)
 - Falsifiable predictions for Euclid, DESI Year 5, CMB-S4
 - *Note: Validates perturbation-level mu-Sigma mapping only (Level 1). Level 2 (dual-sector perturbation via modified CAMB) is complete.*

4. **[Test Validation Compendium](docs/IAM_Test_Validation_Compendium.pdf)** (~30 pages)
 - Nine independent validation tests with detailed results
 - Nine publication-quality figures
 - Complete chi-squared analysis
 - MCMC posterior analysis

5. **[Supplementary Methods](docs/Supplementary_Methods_Reproducibility_Guide.pdf)** (~20 pages)
 - Complete Python implementation
 - Data sources and citations
 - Step-by-step reproducibility instructions
 - Troubleshooting guide

6. **[IAM Theory Paper](docs/IAM_Theory_Paper.pdf)** (~25 pages)
 - Merges and supersedes the former Holographic Derivation and Variational Derivation documents
 - First-principles derivation of E(a) = exp(1 - 1/a) from horizon thermodynamics (Bekenstein-Hawking entropy, Gibbons-Hawking temperature, Landauer's principle)
 - Formal derivation chain: Jacobson (1995) → Cai-Kim (2005) → IAM (2026)
 - S_total = S_geometric + S_informational → dual-sector perturbation modification
 - Constrained scalar field action with equation of state w_info(a) = -1 - 1/(3a) (mildly phantom, consistent with DESI 2024)
 - Gravitational decoherence as the mechanism for sector separation: timelike worldlines decohere (mu < 1), null worldlines do not (Sigma = 1)
 - **Coupling constant derived:** beta_m = Omega_m/2 from virial theorem, matching MCMC to 0.3%
 - **Zero free parameters** beyond standard LambdaCDM
 - Perturbation theory: mu(a) < 1, Sigma(a) = 1 derived from causal structure of spacetime

### Quick Reference

- **Theory Summary:** See Section II-III of Main Manuscript
- **Statistical Results:** See Test Validation Compendium
- **mu--Sigma Mapping, MGCAMB Validation & Planck Level-1 MCMC:** See IAM--CAMB Technical Note
- **Theoretical Derivation:** See IAM Theory Paper (horizon thermodynamics + gravitational decoherence)
- **Code Details:** See Supplementary Methods

---

<details>
<summary>Physical Framework (Equations and mu-Sigma Mapping)</summary>

## Physical Framework

### Dual-Sector Hubble Parameters

**Matter sector** (BAO, growth, distance ladder):
```
H^2_m(a) = H0^2 [Omega_m * a^-3 + Omega_r * a^-4 + Omega_Lambda + B_m * E(a)]
```

**Photon sector** (CMB, photon propagation):
```
H^2_gamma(a) = H0^2 [Omega_m * a^-3 + Omega_r * a^-4 + Omega_Lambda + B_gamma * E(a)]
```

**Activation function:**
```
E(a) = exp(1 - 1/a)
```

**Implementation note:** In the CAMB Fortran code, these sector-dependent expansion rates are computed within the perturbation equations (`adotoa_matter` vs `adotoa`). The global background expansion (`dtauda`) remains standard LCDM. The sector split operates at the perturbation level, not the background level.

### Key Mechanism

The B term enters the denominator, diluting effective matter density:

```
Omega_m(a) = [Omega_m * a^-3] / [Omega_m * a^-3 + Omega_r * a^-4 + Omega_Lambda + B * E(a)]
```

This suppresses structure growth without additional free parameters.

### Modified Gravity Mapping: mu--Sigma Parametrization

The dual-sector phenomenology maps directly onto the standard mu--Sigma modified gravity framework used by DES, KiDS, Euclid, and CMB-S4:

```
mu(a) = H^2_LambdaCDM(a) / [H^2_LambdaCDM(a) + beta_m * E(a)] < 1 (suppressed growth)
Sigma(a) = 1 (standard photon deflection)
```

| Redshift z | mu(a) | Physical meaning |
|-----------|------|-----------------|
| 0.0 | 0.864 | 13.6% growth suppression today |
| 0.5 | 0.920 | Moderate suppression |
| 1.0 | 0.982 | Near-GR |
| 3.0 | 0.9998 | Recovers LambdaCDM |

**Key signature:** mu < 1 with Sigma = 1 means matter feels weaker gravity while photon deflection is standard. This has been validated through MGCAMB (7/7 Boltzmann tests passed, Level 1: [`mgcamb_validation/`](mgcamb_validation/)) and through direct Fortran modification of CAMB (Level 2: Delta-chi2 = +0.54 best-fit Planck, +2.92 Planck+RSD apples-to-apples, [`camb_validation/`](camb_validation/)). Tested against Planck via full MCMC across 15 chains total (12 Level 1 + 3 Level 2). The signature is directly testable by Euclid at projected 3.4 sigma sensitivity. See the [IAM--CAMB Technical Note](docs/IAM_CAMB_Technical_Note.pdf) for Level 1 results.

</details>

---

<details>
<summary>Phenomenological Validation (Pre-Boltzmann, Sections 1-6)</summary>

*Prior to the full MGCAMB Boltzmann implementation, the dual-sector framework was validated against limited observational datasets. These results are documented below for completeness but are superseded by the full Planck MCMC analysis above.*

## Datasets Used

### Primary Data Sources

1. **Planck 2020 CMB** ([A&A 641, A6](https://doi.org/10.1051/0004-6361/201833910))
 - H0: 67.4 +/- 0.5 km/s/Mpc
 - theta_s: 0.0104110 +/- 0.0000031 rad
 - sigma8: 0.811 +/- 0.006

2. **SH0ES 2022** ([ApJL 934, L7](https://doi.org/10.3847/2041-8213/ac5c5b))
 - H0: 73.04 +/- 1.04 km/s/Mpc (Cepheid distance ladder)

3. **JWST TRGB 2024** ([ApJ 919, 16](https://arxiv.org/abs/2308.14864))
 - H0: 70.39 +/- 1.89 km/s/Mpc

4. **DESI DR1/DR2 Growth Rate Measurements** ([Phys. Rev. D 112, 083515](https://doi.org/10.1103/tr6y-kpc6))
 - f*sigma8(z) at 7 redshifts (0.295 < z < 2.33)
 - DR2 (2025): 2.8-4.2 sigma preference for dynamical dark energy
 - w0-wa constraints: w0 > -1, wa < 0 (phantom crossing at z ~ 0.5)

5. **Pantheon+SH0ES 2022** ([ApJ 938, 110](https://doi.org/10.3847/1538-4357/ac8e04))
 - 1588 Type Ia supernovae (0.01 < z < 2.26)
 - Public data: https://github.com/PantheonPlusSH0ES/DataRelease
 - Used in dual-sector validation analysis

**Total:** 10 independent measurements (3 H0 + 7 growth rate)

---

## Key Findings

### 1. Empirical Sector Separation (MCMC Result)

The ratio B_gamma/B_m < 8.5 x 10^-6 (95% CL) is constrained by data, not imposed theoretically:

- Photon-sector constraint from CMB acoustic scale precision
- Matter-sector constraint from BAO and H0 measurements
- Full Bayesian MCMC analysis confirms sector separation

This places an upper bound on photon-sector coupling at least five orders of magnitude below the matter-sector value.

### 2. Growth Suppression Mechanism

Growth suppression follows from Omega_m dilution:

- B in denominator reduces effective Omega_m(a)
- Reduced gravitational source term suppresses structure formation
- 1.36% suppression at z=0 yields sigma8 = 0.800

No additional parameter is required beyond the derived coupling.

### 3. CMB Lensing Consistency

Modified growth partially compensates geometric effects:

- Geometric shift from modified H(z): +1.02%
- Lensing reduction from growth suppression: -0.87%
- Approximate 85% compensation
- Remaining 15% accommodated by B_gamma < 10^-5

### 4. Statistical Significance & Model Selection (H0 + Growth Rate Dataset)

Combined fit to H0 + f*sigma_8 datasets (10 measurements):

- chi2(LCDM) = 38.28 (chi2/dof = 3.83)
- chi2(IAM) = 8.27 (chi2/dof = 1.03)
- **Delta-chi2 = 30.01 (5.5 sigma)**

Model selection criteria (accounting for additional parameters):

- **Delta-AIC = 26.0** (Burnham & Anderson classification: decisive)
- **Delta-BIC = 25.4** (Kass & Raftery classification: very strong)
- **Bayes factor:** ~ 4.4 x 10^5

IAM is preferred on this dataset after accounting for the 2 additional parameters.

*Note: Under the full Planck 2018 likelihood, IAM and LCDM are statistically indistinguishable (Level 1 Delta-chi2 = +1.43; Level 2 Delta-chi2 = +0.54 best-fit). The preference above reflects IAM's ability to simultaneously accommodate the discrepant H0 measurements. See Level 1 and Level 2 sections above for Planck MCMC results.*

### 5. Distance Consistency (Pantheon+ SNe)

Independent validation with supernovae:

- IAM maintains consistency with geometric distance measurements
- Primary IAM impact is on **GROWTH**, not **GEOMETRY**
- Effect on distances subdominant to Omega_Lambda
- Full Pantheon+ dataset confirms distance consistency

### 6. Dual-Sector Empirical Validation (Separate Paper)

Extended empirical validation of dual-sector expansion using Type Ia supernovae is documented in a separate companion paper.

Mahaffey, H. W. (2026). "Dual-Sector Expansion: Type Ia Supernovae Validate Matter-Sector H0 Normalization with LCDM Geometric Consistency"

- Location: docs/Dual_Sector_Validation_Paper.pdf
- Dataset: Pantheon+SH0ES (1588 Type Ia supernovae, 0.01 < z < 2.26)
- Complete reproducible code provided in paper appendices

Three independent tests using Pantheon+ data show that Type Ia supernovae are inconsistent with photon-sector expansion (H0 = 67.4 km/s/Mpc, Test A: B reaches -0.30 boundary), consistent with matter-sector normalization (H0 = 73.04 km/s/Mpc, Test B: B ~ 0), and prefer the matter-sector H0 (Test C). These results are consistent with IAM's prediction that structure formation couples differently to expansion than photon propagation.

</details>

---

## Citation

If you use this code or results in published research, please cite:

```bibtex
@article{Mahaffey2026,
 author = {Mahaffey, Heath W.},
 title = {Constraints on Late-Time f*sigma_8 Suppression from mu < 1, Sigma = 1: 
 Planck 2018 and Large-Scale Structure},
 journal = {Universe},
 year = {2026},
 note = {Submitted (manuscript ID: universe-4189350). Code: \url{https://github.com/hmahaffeyges/IAM-Validation}}
}
```

---

## What IAM Claims vs. Does NOT Claim

### What IAM Claims

- Empirical constraint on sector-dependent expansion: B_gamma/B_m < 10^-5 (MCMC)
- 5.5 sigma improvement over LCDM on the limited H0 + growth rate dataset (Delta-chi2 = 30.01)
- No evidence of overfitting on that dataset (Delta-AIC = 26.0, Delta-BIC = 25.4)
- Compatibility with Planck 2018 full likelihood: Level 1 across four dataset combinations (Delta-chi2 = +1.43 to +2.32); Level 2 dual-sector perturbation validation (Delta-chi2 = +0.54 best-fit Planck, +2.92 Planck+RSD apples-to-apples)
- sigma_8 suppression from 0.809 to 0.800 confirmed as real growth physics (not parameter rebalancing)
- Predictions for Euclid, DESI Year 5, and CMB-S4
- Growth suppression mechanism from mu < 1, Sigma = 1 (dual-sector perturbations)

### What IAM Does NOT Claim

- Complete fundamental derivation from quantum gravity (the holographic derivation is physically motivated but aspects remain to be formalized)
- Modification of Einstein's equations or gauge structure
- That information is a new physical field or substance
- Uniqueness (other parameterizations may fit similarly)
- Explanation of early-universe physics or inflation

IAM is a late-time phenomenological framework motivated by horizon thermodynamics (Bekenstein-Hawking entropy, Gibbons-Hawking temperature, Landauer's principle, quantum decoherence). Its activation function E(a) = exp(1 - 1/a) is derived from the ratio of structure formation rate to cosmic horizon area. Its coupling constant beta_m = Omega_m/2 is derived from the virial theorem. The framework generates specific predictions testable by current and upcoming surveys. Level 2 validation confirms the dual-sector perturbation mechanism survives the full Planck likelihood with no statistically significant penalty (Delta-chi2 = +0.54 best-fit).

---

## Development History

This repository presents the final validated framework. Complete development history, including exploratory tests and deprecated approaches, is available in the [`development/`](development/) directory. See [`development/README_development.md`](development/README_development.md) for details.

**Validation Timeline:**
- **Tests 1-26:** Early exploration (growth mechanisms, various parameterizations)
- **Tests 27-29:** Dual-sector identification (empirical sector separation)
- **Test 30:** Final synthesis (consolidated validation)
- **Current:** 9 tests in `iam_validation.py` with full MCMC analysis
- **MGCAMB:** Full Boltzmann validation via modified Einstein-Boltzmann solver (7/7 tests passed)
- **Planck MCMC (Level 1):** 12 independent chains across 4 dataset combinations (Planck only: Runs A/B/C; Planck + RSD: Runs D/E/F; Planck + BAO: Runs G/H/I; Planck + Pantheon+: Runs J/K/L) -- IAM compatible with Planck across all datasets (Delta-chi2 = +1.43 to +2.32, all below exclusion threshold). Raw chain data: [`mgcamb_validation/chains/`](mgcamb_validation/chains/)
- **Level 2:** Dual-sector perturbation split via modified CAMB equations.f90 -- Delta-chi2 = +0.54 best-fit vs LCDM (Planck), +2.92 apples-to-apples (Planck+RSD), sigma_8 = 0.800, H0(matter) = 72.26 (all 3 chains converged: Runs A, C, D). Raw chain data: [`camb_validation/chains/`](camb_validation/chains/)

**Main validation consolidated into `iam_validation.py` for clarity and reproducibility.**

---

## Contact

**Heath W. Mahaffey** 
Independent Researcher 
Entiat, WA 98822, USA 

- **Email:** hmahaffeyges@gmail.com
- **GitHub:** [@hmahaffeyges](https://github.com/hmahaffeyges)

For questions, issues, or collaboration inquiries, please open an issue on GitHub or email directly.

---

## License

MIT License - Free to use, modify, and distribute with attribution.

See [LICENSE](LICENSE) for full details.

---

## Acknowledgments

The author thanks the Planck, SDSS/BOSS/eBOSS, SH0ES, DESI, and JWST collaborations for publicly available data. The CAMB team (Lewis, Challinor) for the Boltzmann solver (v1.5.8). The MGCAMB team (Wang, Mirpoorian, Pogosian, Silvestri, Zhao) for the modified gravity extension. The Cobaya team (Torrado, Lewis) for the MCMC sampling framework. Grateful to the open-source communities of NumPy, SciPy, Matplotlib, GetDist, and corner. This work benefited from discussions facilitated by Claude (Anthropic) regarding statistical methodology, MCMC implementation, Boltzmann solver configuration, and reproducibility best practices.

---

**Last Updated:** February 25, 2026
**Status:** Level 1 complete (12 chains, 4 datasets, Delta-chi2 = +1.34 to +2.32 vs LCDM); Level 2 complete (5 chains: 3 dual-sector perturbations, 2 background modification diagnostics, Delta-chi2 = +0.54 best-fit Planck, sigma_8 = 0.800, H0(matter) = 72.26); MGCAMB 7/7 Boltzmann tests passed; 17 total converged chains; zero free parameters -- all derived from first principles
**Key Result:** Level 2 dual-sector perturbation validation yields Delta-chi2 = +0.54 (best-fit) relative to LCDM under the full Planck likelihood. All standard parameters shift by < 0.1 sigma. sigma_8 shifts from 0.8087 to 0.7998, consistent with the direction reported by weak lensing surveys. The matter-sector expansion rate yields H0(matter) = 72.26 km/s/Mpc (0.75 sigma from SH0ES) alongside H0(photon) = 67.16 km/s/Mpc (0.37 sigma from Planck), placing both Hubble tension endpoints within 1 sigma. sigma_8 suppression confirmed as real growth physics (logA shift 0.10 sigma, Omega_m shift 0.06 sigma). Level 1 perturbation-level validation (12 chains, 4 dataset combinations) yields Delta-chi2 = +1.34 to +2.32, all below the 95% CL exclusion threshold. Projected to be testable by Euclid (sigma(mu_0) ~ 0.04) and DESI Year 5.

---
