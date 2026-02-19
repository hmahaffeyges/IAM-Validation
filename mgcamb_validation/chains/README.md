# IAM – Perturbation-Level μ–Σ  Validation via MGCAMB

This repository implements and tests a late-time mu(a) < 1, Sigma(a) = 1 modification of LCDM using the full Planck 2018 likelihood (TT, TE, EE, low-ell + lensing) through MGCAMB v1.5.2 + Cobaya.

**Level-1 validation scope:**
- Background expansion: standard LCDM (unmodified)
- Perturbation modification: mu(a) = H^2_LCDM / (H^2_LCDM + beta * E(a)), with E(a) = exp(1 - 1/a)
- Sigma(a) = 1 exactly (lensing unmodified)
- No perturbation quantity feeds back into H(z)
- Coupling derived from virial theorem: beta_m = Omega_m/2 = 0.1575, yielding mu_0 = -0.13495

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Planck MCMC Results (12 Independent Chains)

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

### Delta-chi2 Summary (All 4 Dataset Combinations)

| Dataset | Delta-chi2 (IAM fixed vs LCDM) | sigma_8 (LCDM) | sigma_8 (IAM) | sigma_8 shift |
|---------|-------------------------------|----------------|---------------|---------------|
| Planck only | +1.43 | 0.8139 | 0.8014 | -0.0125 (-1.5%) |
| Planck + RSD | +1.34 | 0.8131 | 0.8001 | -0.0130 (-1.6%) |
| Planck + BAO | +2.32 | 0.8115 | 0.7981 | -0.0134 (-1.7%) |
| Planck + Pantheon+ | +1.58 | 0.8129 | 0.8000 | -0.0129 (-1.6%) |

All Delta-chi2 values are below 3.84 (95% CL threshold). The sigma_8 shift of -0.013 +/- 0.001 is stable across all four dataset combinations, confirming that the suppression is driven by the mu_0 = -0.135 prediction, not by any particular dataset. BAO and Pantheon+ photon-sector observables are unaffected by the modification, as predicted by Sigma = 1.

**MGCAMB Boltzmann diagnostics: 7/7 tests passed** (CMB TT < 0.17%, lensing +0.30%, sigma_8 = 0.795, Sigma = 1 exact, P(k) scale-independent, f*sigma_8 consistent).

**Full reproducibility** -- chains, YAML configs, GetDist scripts, Boltzmann diagnostic script, forecast analyses: [`mgcamb_validation/`](mgcamb_validation/)

---

## Broader Context: IAM Dual-Sector Framework

The mu-Sigma modification tested above is derived from the Informational Actualization Model (IAM), a dual-sector cosmological framework where matter and photons experience different late-time expansion rates. The full framework predicts H0(matter) = 72.51 km/s/Mpc alongside H0(photon) = 67.4 km/s/Mpc, addressing the Hubble tension.

On a limited H0 + growth rate dataset (10 measurements), IAM shows strong model-selection preference (Delta-chi2 ~ 30, Delta-AIC = 26.0). **This strong preference refers to the limited H0 + growth dataset only and does not apply under the full Planck likelihood, where IAM and LCDM are statistically indistinguishable.** The background-level prediction (Level 2/3) remains to be implemented and tested.

| Parameter | Value | Source |
|-----------|-------|--------|
| **mu(z=0)** | 0.864 | Derived (perturbation theory) |
| **Sigma(z)** | 1.000 | Derived (delta-phi = 0) |
| **beta_m** | 0.1575 | Derived (Omega_m/2, virial theorem) |
| **H0(matter)** | 72.51 km/s/Mpc | IAM prediction (0.51 sigma from SH0ES) |
| **H0(photon)** | 67.4 km/s/Mpc | Planck CMB |

---

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
 LCDM fails to resolve Hubble tension

TEST 2: IAM Dual-Sector Model
 B_m = 0.157 (MCMC median)
 chi2_total = 8.27
 Delta-chi2 = 30.01 (5.5 sigma)
 IAM resolves Hubble tension with high significance

TEST 3: Confidence Intervals (Profile Likelihood)
 68% CL (1 sigma): B_m = 0.157 +/- 0.029
 95% CL (2 sigma): B_m = 0.157 +/- 0.057

TEST 4: Photon-Sector Constraint (MCMC)
 Profile likelihood: B_gamma < 0.004 (95% CL)
 MCMC constraint: B_gamma < 1.40e-06 (95% CL)
 Sector ratio: B_gamma/B_m < 8.50e-06 (95% CL)
 Photons couple at least 100,000x more weakly than matter

TEST 5: Physical Predictions
 H0(photon/CMB) = 67.4 km/s/Mpc
 H0(matter/local) = 72.5 +/- 1.0 km/s/Mpc
 Growth suppression = 1.36%
 sigma8(IAM) = 0.800
 All predictions consistent with observations

TEST 6: CMB Lensing Consistency
 Growth suppression (1.36%) --> weaker lensing
 Reduced lensing compensates ~85% of geometric theta_s shift
 Natural compensation maintains CMB consistency

TEST 7: Model Selection Criteria (Overfitting Check)
 Delta-AIC = 26.01 --> 'Decisive' evidence for IAM
 Delta-BIC = 25.40 --> 'Very strong' evidence for IAM
 Relative likelihood: LCDM is 444,000x less likely
 No evidence of overfitting despite 2 additional parameters

TEST 8: Full Bayesian MCMC Analysis
 B_m = 0.157 +0.029/-0.029 (68% CL)
 B_gamma < 1.40e-06 (95% upper limit)
 B_gamma/B_m < 8.50e-06 (95% upper limit)
 H0(matter) = 72.5 +/- 1.0 km/s/Mpc
 Well-behaved Gaussian posteriors with no degeneracies

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
 LambdaCDM is 6,038,848x less likely than IAM
```

---

## Repository Structure

```
IAM-Validation/
├── README.md                                  # This file
├── mgcamb_validation/                         # *** PRIMARY VALIDATION DIRECTORY ***
│   ├── README.md                              # Detailed MGCAMB documentation
│   ├── iam_mu_sigma.py                        # 7/7 Boltzmann diagnostic tests (reproduces 6-panel figure)
│   ├── chains/                                # Raw MCMC chain files (Runs A–L complete)
│   ├── yaml_configs/                          # Exact Cobaya YAML files for all runs
│   ├── getdist_scripts/                       # Posterior extraction & three-way comparison
│   └── forecasts/                             # Euclid/DESI prediction scripts
├── docs/                                      # Technical documentation (PDFs)
├── tests/                                     # Phenomenological validation scripts
│   ├── iam_validation.py                      # 9 observational tests (~1 min)
│   └── iam_derivation_tests.py                # 15 derivation + robustness tests (~30 sec)
├── figures/                                   # Publication-quality figures
├── results/                                   # Output data files
└── data/                                      # Observational datasets
```

**For referees:** Start with [`mgcamb_validation/`](mgcamb_validation/) — it contains everything needed to reproduce the MCMC results, Boltzmann diagnostics, and forecasts.

---

## What IAM Does

### Addresses Hubble Tension

- **Planck CMB:** H0 = 67.4 km/s/Mpc (photon sector, B_gamma < 10^-5)
- **SH0ES Distance Ladder:** H0 = 73.04 km/s/Mpc (matter sector, B_m = 0.157)
- **Both correct:** Different sectors, not conflicting measurements

### Addresses S8 Tension

- **Growth suppression:** 1.36% at z=0 from Omega_m dilution
- **Effective sigma8:** 0.800 (intermediate between Planck 0.811 and DES/KiDS ~0.77)
- **Natural mechanism:** No ad-hoc parameters

### Passes CMB Consistency

- **Planck MCMC:** 12 independent chains converged cleanly across 4 dataset combinations (Delta-chi2 = +1.43 to +2.32 vs LCDM, all below exclusion threshold)
- **CMB lensing:** 85% geometric compensation
- **Acoustic scale:** B_gamma < 10^-5 maintains theta_s precision
- **Early universe:** No modifications before z ~ 1
- **sigma_8 shift:** 0.8139 (LCDM) --> 0.8014 (IAM), direction favored by weak lensing surveys

### No Overfitting (H0 + Growth Rate Dataset)

- **AIC penalty:** Delta-AIC = 26.0 >> 10 (decisive preference)
- **BIC penalty:** Delta-BIC = 25.4 >> 10 (very strong preference)
- **Relative likelihood:** LCDM is 444,000x less likely than IAM

*Note: These model selection statistics apply to the H0 + f*sigma_8 dataset (10 measurements). Under full Planck likelihood, IAM and LCDM are statistically indistinguishable (Delta-chi2 = +1.43). See Section 7 for Planck MCMC results.*

### Makes Testable Predictions

- **CMB-S4:** Will constrain B_gamma < 10^-4 (100x tighter)
- **Euclid:** S8 = 0.78 +/- 0.01
- **DESI Year 5:** B_m to +/-1% precision

---

## Documentation

### Primary Documents

1. **[Main Manuscript](docs/IAM_Manuscript.pdf)** (RevTeX, ~15 pages)
 - Full holographic motivation (Bekenstein-Hawking entropy, holographic principle)
 - Theoretical foundation and phenomenological implementation
 - Statistical validation and testable predictions

2. **[Dual-Sector Validation Paper](docs/Dual_Sector_Validation_Paper.pdf)** (RevTeX, ~22 pages)
 - Empirical validation of sector separation using Pantheon+ Type Ia supernovae
 - Three independent tests (Planck prior, SH0ES prior, no prior)
 - Complete Python code in appendices (< 2 min reproducibility)
 - Companion paper to main IAM manuscript

3. **[IAM--CAMB Technical Note: Planck Level-1 Validation](docs/IAM_CAMB_Technical_Note.pdf)** (~23 pages)
 - mu--Sigma modified gravity mapping: mu(a) < 1, Sigma(a) = 1
 - Python-level CAMB validation with comprehensive 8-panel figure
 - Fortran-level implementation: what was done, what was learned
 - **MGCAMB Boltzmann validation: 7/7 diagnostic tests passed** (sigma_8, CMB TT, lensing, P(k), f*sigma_8, Sigma = 1, scale independence)
 - **Planck Level-1 MCMC analysis** (perturbation sector, 3 independent chains via Cobaya + MGCAMB) with complete YAML files
 - **Run B mu_0 posterior analysis** (MAP ~ 0, sigma = 0.156, IAM at 0.9 sigma, asymmetric 68% CI)
 - **Forecasting and observational prospects** (Fisher forecast, ISW-galaxy cross-correlation, binned mu(z) reconstruction, transition zone analysis)
 - CAMB background validation figure (6-panel)
 - Falsifiable predictions for Euclid, DESI Year 5, CMB-S4
 - *Note: Validates perturbation-level mu-Sigma mapping only (Level 1). Full background modification (Levels 2/3) remains to be implemented.*

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
 - S_total = S_geometric + S_informational → modified Friedmann equation
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

### Key Mechanism

The B term enters the denominator, diluting effective matter density:

```
Omega_m(a) = [Omega_m * a^-3] / [Omega_m * a^-3 + Omega_r * a^-4 + Omega_Lambda + B * E(a)]
```

This naturally suppresses structure growth without additional parameters.

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

**Key signature:** mu < 1 with Sigma = 1 means matter feels weaker gravity while photon deflection is standard. This has been validated through MGCAMB (7/7 Boltzmann tests passed) and tested against Planck via full MCMC (6 chains across 2 dataset combinations, Delta-chi2 = +1.43 and +1.34 vs LCDM, statistically indistinguishable). The signature uniquely distinguishes IAM from generic modified gravity theories and is directly testable by Euclid at 3.4 sigma. See the [IAM--CAMB Technical Note](docs/IAM_CAMB_Technical_Note.pdf) for complete results and [`mgcamb_validation/chains/`](mgcamb_validation/chains/) for raw MCMC chain data.

---

## Phenomenological Validation (Pre-Boltzmann)

*Prior to the full MGCAMB Boltzmann implementation, the dual-sector framework was validated against limited observational datasets. These results are documented below for completeness but are superseded by the full Planck MCMC analysis in Section 7.*

<details>
<summary>Click to expand phenomenological validation results (Sections 1-6, Datasets)</summary>

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

The ratio B_gamma/B_m < 8.5 x 10^-6 (95% CL) is **data-driven**, not theoretically imposed:

- Photon-sector constraint from CMB acoustic scale precision
- Matter-sector constraint from BAO and H0 measurements
- Full Bayesian MCMC analysis confirms sector separation

**This transforms "photon exemption" from assumption to empirical discovery: photons couple at least 100,000x more weakly than matter.**

### 2. Growth Suppression Mechanism

Growth suppression emerges naturally from Omega_m dilution:

- B in denominator --> reduced effective Omega_m(a)
- Weaker gravity --> suppressed structure formation
- 1.36% suppression at z=0 --> sigma8 = 0.800

**No ad-hoc "growth tax" parameter required.**

### 3. CMB Lensing Consistency

Modified growth naturally compensates geometric effects:

- Geometric shift from modified H(z): +1.02%
- Lensing reduction from growth suppression: -0.87%
- **85% compensation** without tuning
- Remaining 15% resolved by B_gamma < 10^-5

### 4. Statistical Significance & Model Selection (H0 + Growth Rate Dataset)

Combined fit to H0 + f*sigma_8 datasets (10 measurements):

- chi2(LCDM) = 38.28 --> poor fit (chi2/dof = 3.83)
- chi2(IAM) = 8.27 --> excellent fit (chi2/dof = 1.03)
- **Delta-chi2 = 30.01 (5.5 sigma improvement)**

Model selection criteria (addressing overfitting):

- **Delta-AIC = 26.0** --> "Decisive" evidence for IAM (Burnham & Anderson)
- **Delta-BIC = 25.4** --> "Very strong" evidence for IAM (Kass & Raftery)
- **Relative likelihood:** LCDM is 444,000x less likely

**Even with penalties for 2 additional parameters, IAM is strongly preferred on these datasets.**

*Note: Under full Planck 2018 likelihood, IAM and LCDM are statistically indistinguishable (Delta-chi2 = +1.43). The strong preference above reflects IAM's ability to simultaneously fit the discrepant H0 measurements that LCDM cannot. See Section 7 for Planck MCMC results.*

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

Three independent tests using Pantheon+ data demonstrate that Type Ia supernovae reject photon-sector expansion (H0 = 67.4 km/s/Mpc, Test A: B --> -0.30 boundary), accept matter-sector normalization (H0 = 73.04 km/s/Mpc, Test B: B ~ 0), and maintain LCDM geometric consistency (Test C: confirms matter preference). These results validate that dual-sector separation emerges from data, not theoretical assumption, confirming IAM's prediction that structure formation couples differently to expansion than photon propagation.

</details>

---

### 7. MGCAMB Boltzmann Validation & Planck Level-1 MCMC (in IAM--CAMB Technical Note)

IAM's mu < 1, Sigma = 1 prediction has been validated at the perturbation level through the MGCAMB modified Einstein-Boltzmann solver (v1.5.2; Wang et al. 2023) and twelve independent Planck MCMC chains via Cobaya across four dataset combinations. All results are documented in the **[IAM--CAMB Technical Note: Planck Level-1 Validation](docs/IAM_CAMB_Technical_Note.pdf)**.

**Full reproducibility package:** [`mgcamb_validation/`](mgcamb_validation/)

- **[Raw MCMC chains](mgcamb_validation/chains/)** -- complete chain data for all 12 runs (Runs A/B/C: Planck only; Runs D/E/F: Planck + RSD; Runs G/H/I: Planck + BAO; Runs J/K/L: Planck + Pantheon+), independently verifiable via GetDist
- **[Cobaya YAML configs](mgcamb_validation/yaml_configs/)** -- exact configuration files to re-run all 12 chains from scratch
- **[GetDist extraction scripts](mgcamb_validation/getdist_scripts/)** -- reproduce every posterior table and Delta-chi2 comparison in the Technical Note
- **[Forecast analyses](mgcamb_validation/forecasts/)** -- Fisher forecast, ISW prediction, binned mu(z) reconstruction, transition zone analysis
- **[7/7 Boltzmann diagnostic script](mgcamb_validation/iam_mu_sigma.py)** -- MGCAMB validation code reproducing all 7 diagnostic tests and 6-panel figure

**MGCAMB Diagnostic Results (7/7 tests PASSED):**

| Test | Criterion | Result | Status |
|------|-----------|--------|--------|
| sigma_8 | In [0.79, 0.82] | 0.7954 | PASS |
| CMB TT (ell > 30) | < 1% residual | 0.17% | PASS |
| CMB TT ISW (ell < 30) | < cosmic variance | 3.6% (CV ~ 63%) | PASS |
| CMB lensing | < 5% change | +0.30% | PASS |
| Sigma = 1 | Exact | sigma_0 = 0 | PASS |
| P(k) scale-independence | std(ratio) < 1% | 0.53% | PASS |
| f*sigma_8 fit quality | chi2 <= LCDM + 4 | 4.42 vs 4.85 | PASS |

**Planck MCMC Results (12 chains, full Planck 2018 likelihood +/- RSD/BAO/Pantheon+):**

Likelihoods: planck_2018_lowl.TT + lowl.EE + highl_plik.TTTEEE_lite_native + lensing.CMBMarged (+/- SDSS DR12/DR16 f*sigma_8, BAO, Pantheon+)

| Run | Dataset | mu_0 | sigma_8 | H0 [km/s/Mpc] | Best chi2 | Extra params |
|-----|---------|------|---------|---------------|-----------|--------------|
| **A: IAM fixed** | Planck | -0.135 (fixed) | 0.8014 +/- 0.006 | 67.06 +/- 0.51 | 984.57 | 0 |
| **B: mu_0 floating** | Planck | 0.006 +/- 0.156 | 0.8146 +/- 0.016 | 67.16 +/- 0.51 | 981.24 | 1 |
| **C: LCDM baseline** | Planck | 0 (fixed) | 0.8139 +/- 0.006 | 67.19 +/- 0.54 | 983.14 | 0 |
| **D: IAM fixed** | Planck + RSD | -0.135 (fixed) | 0.8001 +/- 0.006 | 67.07 +/- 0.51 | 991.21 | 0 |
| **E: mu_0 floating** | Planck + RSD | +0.024 +/- 0.123 | 0.8147 +/- 0.013 | 67.10 +/- 0.52 | 989.87 | 1 |
| **F: LCDM baseline** | Planck + RSD | 0 (fixed) | 0.8131 +/- 0.006 | 67.18 +/- 0.53 | 989.87 | 0 |
| **G: IAM fixed** | Planck + BAO | -0.135 (fixed) | 0.7981 +/- 0.006 | 67.51 +/- 0.43 | 1020.03 | 0 |
| **H: mu_0 floating** | Planck + BAO | +0.002 +/- 0.158 | 0.8117 +/- 0.016 | 67.56 +/- 0.46 | 1017.90 | 1 |
| **I: LCDM baseline** | Planck + BAO | 0 (fixed) | 0.8115 +/- 0.006 | 67.56 +/- 0.45 | 1017.71 | 0 |
| **J: IAM fixed** | Planck + Pantheon+ | -0.135 (fixed) | 0.8000 +/- 0.006 | 67.03 +/- 0.52 | 2417.58 | 0 |
| **K: mu_0 floating** | Planck + Pantheon+ | -0.005 +/- 0.162 | 0.8124 +/- 0.017 | 67.06 +/- 0.52 | 2415.40 | 1 |
| **L: LCDM baseline** | Planck + Pantheon+ | 0 (fixed) | 0.8129 +/- 0.006 | 67.11 +/- 0.51 | 2416.00 | 0 |

**Key results:**
- **Planck only:** Delta-chi2 = +1.43 between IAM fixed and LCDM (statistically indistinguishable)
- **Planck + RSD:** Delta-chi2 = +1.34 (stable across datasets)
- **Planck + BAO:** Delta-chi2 = +2.32 (maximum penalty, still well below 3.84 threshold)
- **Planck + Pantheon+:** Delta-chi2 = +1.58 (consistent)
- All Delta-chi2 values below 3.84 (95% CL threshold for 0 additional parameters)
- IAM's predicted mu_0 = -0.135 is **within 0.8--1.3 sigma** of all floating mu_0 posteriors
- **sigma_8 shifted down** by -0.013 +/- 0.001 across all four dataset combinations (universal -1.6% suppression)
- sigma_8 shift direction favored by weak lensing surveys (KiDS, DES, HSC)
- BAO and Pantheon+ photon-sector observables unaffected, confirming Sigma = 1
- All 12 runs converged with R-1 < 0.01; acceptance rates 50-77%
- All standard parameters stable across all twelve runs (no pathology)

**Interpretation:** IAM remains statistically indistinguishable from LCDM under the full Planck 2018 likelihood across all four dataset combinations (Planck only, +RSD, +BAO, +Pantheon+), demonstrating internal consistency at current precision. The sigma_8 shift of -0.013 is universal and stable, confirming it is driven by the mu_0 = -0.135 prediction rather than any particular dataset. BAO and supernova observables are unaffected, as predicted by Sigma = 1. The definitive detection tests are Euclid (sigma(mu_0) ~ 0.04, yielding 3.4 sigma) and DESI Year 5 growth rates. Level 2 validation (background-modified CAMB with dual-sector Friedmann equation) is in progress.

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
| **H₀** | 74.0 | 4.7 | **+69.4** | Primary discriminator: sector split resolves Hubble tension |
| **f·σ₈** | 6.5 | 7.4 | −0.8 | IAM slightly worse (combined model suppresses growth marginally below ΛCDM) |
| **BAO** | 15.6 | 15.6 | 0.0 | Identical (photon sector unmodified) |
| **Cosmic chronometers** | 14.5 | 15.0 | −0.5 | Indistinguishable at current CC precision (uncertainties 5–30%) |
| **S₈** | 25.6 | 13.9 | **+11.7** | σ₈ suppression (0.811 → 0.790) moves prediction toward WL data |
| **CMB priors** | 370 | 370 | 0.0 | Fitting-formula residuals dominate; full MGCAMB confirms < 0.17% CMB residuals |
| **Combined** | **507** | **427** | **+79.8** | 0 additional free parameters |

Combined Δχ² = 79.8 (equivalent to 8.9σ, 0 additional free parameters). The improvement is driven by the H₀ sector split (+69.4) and S₈ suppression (+11.7). All other probes are consistent within uncertainties.

### S₈ Tension

IAM addresses the S₈ tension through two mechanisms:
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
| **Euclid (mu-Sigma)** | mu < 1 with Sigma = 1 (unique signature) | f(R): mu > 1; Horndeski: both modified |
| **DESI Year 5** | Distance-growth tension in w0-wa fits | All w0-wa models (consistent dist+growth) |
| **DESI Year 5** | No real phantom crossing (w always <= -1) | w0-wa best fit (apparent crossing at z~0.5) |
| **Euclid** | Scale-independent mu and Sigma (no k-dependence) | f(R), DGP (scale-dependent growth) |
| **Euclid** | B_m/Omega_m = 1/2 constant for any Omega_m | Ad-hoc models (ratio would vary) |
| **Simons Observatory** | B_gamma < 0.001 (10x tighter) | Photon exemption falsifiable |

### Long-Term (> 5 years)

| Experiment | IAM Prediction | Timeline |
|------------|---------------|----------|
| **CMB-S4** | B_gamma < 10^-4 or IAM falsified | 2030+ |
| **Euclid + Rubin** | BAO at z > 2 tests early-time behavior | 2030+ |
| **GW Standard Sirens** | H0(matter) = 72.51 km/s/Mpc | 2030+ |

---

## Citation

If you use this code or results in published research, please cite:

```bibtex
@article{Mahaffey2026,
 author = {Mahaffey, Heath W.},
 title = {Dual-Sector Cosmology from Structure-Driven Expansion: 
 The Informational Actualization Model (IAM)},
 journal = {In preparation},
 year = {2026},
 note = {Code: \url{https://github.com/hmahaffeyges/IAM-Validation}}
}
```

---

## What IAM Claims vs. Does NOT Claim

### What IAM Claims

- Empirical evidence for sector-dependent expansion: B_gamma/B_m < 10^-5 (MCMC)
- 5.5 sigma statistical improvement over LCDM (Delta-chi2 = 30.01)
- No evidence of overfitting (Delta-AIC = 26.0, Delta-BIC = 25.4)
- Simultaneous compatibility with H0 measurements from both sectors and partial shift of sigma_8 toward weak lensing values
- Testable predictions for upcoming surveys (CMB-S4, Euclid, DESI Year 5)
- Natural growth suppression mechanism from Omega_m dilution

### What IAM Does NOT Claim

- Complete fundamental derivation from quantum gravity (the holographic derivation is physically motivated but aspects remain to be formalized)
- Modification of Einstein's equations or gauge structure
- That information is a new physical field or substance
- Uniqueness (other parameterizations may fit similarly)
- Explanation of early-universe physics or inflation

**IAM is a physically motivated late-time framework** grounded in horizon thermodynamics (Bekenstein-Hawking entropy, Gibbons-Hawking temperature, Landauer's principle, quantum decoherence). Its activation function E(a) = exp(1 - 1/a) is derived from the ratio of structure formation rate to cosmic horizon area. Its value lies in providing empirically testable predictions that unify multiple cosmological tensions.

---

## Development History

This repository presents the final validated framework. Complete development history, including exploratory tests and deprecated approaches, is available in the [`development/`](development/) directory. See [`development/README_development.md`](development/README_development.md) for scientific evolution and key breakthroughs.

**Validation Timeline:**
- **Tests 1-26:** Early exploration (growth mechanisms, various parameterizations)
- **Tests 27-29:** Dual-sector discovery (breakthrough: empirical sector separation)
- **Test 30:** Final synthesis (consolidated validation)
- **Current:** 9 tests in `iam_validation.py` with full MCMC analysis
- **MGCAMB:** Full Boltzmann validation via modified Einstein-Boltzmann solver (7/7 tests passed)
- **Planck MCMC:** 12 independent chains across 4 dataset combinations (Planck only: Runs A/B/C; Planck + RSD: Runs D/E/F; Planck + BAO: Runs G/H/I; Planck + Pantheon+: Runs J/K/L) -- IAM compatible with Planck across all datasets (Delta-chi2 = +1.43 to +2.32, all below exclusion threshold). Raw chain data: [`mgcamb_validation/chains/`](mgcamb_validation/chains/)
- **Level 2:** Background-modified CAMB (dual-sector Friedmann equation via modified equations.f90) -- quicktest passed (sigma_8 = 0.801), full chains in progress

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

The author thanks the Planck, SDSS/BOSS/eBOSS, SH0ES, DESI, and JWST collaborations for publicly available data. The MGCAMB team (Wang, Mirpoorian, Pogosian, Silvestri, Zhao) for the modified gravity Boltzmann solver. The Cobaya team (Torrado, Lewis) for the MCMC sampling framework. Grateful to the open-source communities of NumPy, SciPy, Matplotlib, GetDist, and corner. This work benefited from discussions facilitated by Claude (Anthropic) regarding statistical methodology, MCMC implementation, Boltzmann solver configuration, and reproducibility best practices.

---

**Last Updated:** February 19, 2026 
**Status:** MGCAMB 7/7 Boltzmann tests passed; Planck MCMC compatible (12 chains across 4 datasets, Delta-chi2 = +1.43 to +2.32 vs LCDM, all statistically indistinguishable); Level 2 background-modified CAMB chains in progress; zero free parameters -- all derived from first principles 
**Key Result:** IAM's mu < 1, Sigma = 1 signature survives full Planck + growth-rate + BAO + Pantheon+ likelihood. sigma_8 shifts from 0.813 to 0.800 under IAM (direction favored by weak lensing), universally stable across all 4 dataset combinations (-0.013 +/- 0.001). mu_0 = -0.135 lies 0.8--1.3 sigma from all floating posterior peaks. Uniquely testable by Euclid (3.4 sigma) and DESI Year 5.

---

<p align="center">
 <i>"The universe actualizes its potential through structure formation, and geometry responds."</i>
</p>
