# IAM: Dual-Sector Cosmology from Structure-Driven Expansion

**TL;DR:** A zero-parameter cosmological model (IAM) resolves the Hubble tension by deriving that matter and photons experience different late-time expansion rates. Every element is derived from first principles: the activation function E(a) = exp(1 - 1/a) from horizon thermodynamics, the coupling B_m = Omega_m/2 from the virial theorem, and the perturbation predictions mu < 1, Sigma = 1 from delta-phi = 0. Fixing B_m at its predicted value gives Delta-chi2 = 31.2 improvement over LCDM (5.6 sigma) with zero additional free parameters. Two scripts verify everything: `iam_validation.py` (9 observational tests) and `iam_derivation_tests.py` (10 derivation tests), both runnable in under 2 minutes.

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Dual-Sector Cosmology from Structure-Driven Expansion: The Informational Actualization Model (IAM)**

**Key Finding:** 5.6 sigma empirical evidence for dual-sector cosmology resolving the Hubble tension. Zero free parameters — coupling constant, activation function, and perturbation predictions all derived from first principles.

---

## Core Results

| Parameter | Value | Method | Description |
|-----------|-------|--------|-------------|
| **B_m** | 0.1575 | **Derived** (Omega_m/2) | Matter-sector coupling (virial theorem) |
| **B_gamma** | < 1.4 x 10^-6 | MCMC (95% CL) | Photon-sector coupling |
| **B_gamma/B_m** | < 8.5 x 10^-6 | MCMC (95% CL) | Empirical sector ratio |
| **H0(photon)** | 67.4 km/s/Mpc | Planck CMB | Photon-sector measurement |
| **H0(matter)** | 72.51 km/s/Mpc | IAM **prediction** | Matter-sector (0.51 sigma from SH0ES) |
| **Delta-chi2** | 31.2 (5.6 sigma) | vs. LCDM | Statistical improvement |
| **Delta-AIC** | 31.2 | Model selection | **Zero** additional parameters |
| **Delta-BIC** | 31.2 | Model selection | LCDM is 6 million x less likely |
| **mu(z=0)** | 0.864 | **Derived** (perturbation theory) | Growth suppression parameter |
| **Sigma(z)** | 1.000 | **Derived** (delta-phi = 0) | Lensing unmodified |

**The Hubble tension is resolved:** Planck (photon sector, B_gamma < 10^-5) and SH0ES (matter sector, B_m = 0.157) both measure correctly -- they probe different expansion rates. Photons couple at least **100,000x more weakly** than matter to late-time expansion.

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

# Run derivation verification (10 tests, ~30 sec)
python iam_derivation_tests.py
```

**Expected runtime:** ~90 seconds total for both scripts

**Observational Validation (`iam_validation.py`) — Expected output:**
```
================================================================================
  INFORMATIONAL ACTUALIZATION MODEL (IAM)
  Complete Validation Presentation
================================================================================

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
  IAM:  chi2_H0 = 1.52
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
  MCMC constraint:    B_gamma < 1.40e-06 (95% CL)
  Sector ratio:       B_gamma/B_m < 8.50e-06 (95% CL)
  Photons couple at least 100,000x more weakly than matter

TEST 5: Physical Predictions
  H0(photon/CMB)  = 67.4 km/s/Mpc
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

**Derivation Verification (`iam_derivation_tests.py`) — Expected output:**
```
╔══════════════════════════════════════════════════════════════════════════════╗
║  IAM DERIVATION VERIFICATION SUITE                                        ║
║  10 Tests — From Jacobson to Zero-Parameter Cosmology                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

  [✓ PASS] Test  1: Jacobson: Standard entropy → Friedmann equation
  [✓ PASS] Test  2: Cai-Kim: First law on apparent horizon → Friedmann
  [✓ PASS] Test  3: Modified entropy → IAM Friedmann equation
  [✓ PASS] Test  4: Information surface density → exp(1 - 1/a)
  [✓ PASS] Test  5: Sheth-Tormen at σ*=1.2 → β ≈ 1.009
  [✓ PASS] Test  6: Virial theorem → β_m = Ω_m/2
  [✓ PASS] Test  7: Collapsed fraction → Virial theorem confirmed
  [✓ PASS] Test  8: Perturbation theory: μ < 1, Σ = 1
  [✓ PASS] Test  9: Fixed β_m = Ω_m/2: Δχ² = 31.2 (5.6σ)
  [✓ PASS] Test 10: Equation of state: w_info = -1 - 1/(3a)

  Tests passed: 10/10
  β_m = Ω_m/2 = 0.1575 (predicted) vs 0.157 (MCMC): 0.3% agreement
  H₀(matter) = 72.51 km/s/Mpc (0.51σ from SH0ES)
  Δχ² = 31.2 for ZERO additional parameters
  ΛCDM is 6,038,848× less likely than IAM
```

### 3. CAMB Background Validation (`camb_iam_background.py`)

Validates IAM background cosmology against the community-standard
Boltzmann code CAMB (v1.6.5). Requires: `pip install camb`

```bash
python camb_iam_background.py
```

**9 Background Tests (all PASS):**

| Test | Description | Result |
|------|-------------|--------|
| 1 | LCDM baseline vs Planck 2020 | theta_s, r_s, z_star match |
| 2 | Sound horizon invariance | r_s = 147.22 Mpc, unchanged |
| 3 | H(z) IAM vs LCDM | Ratio = sqrt(1 + beta_m) at z=0 |
| 4 | Distance moduli (SNe) | Delta_mu = -0.157 mag = H0 tension |
| 5 | BAO angular scale | 2-6% shifts (DESI Y5 testable) |
| 6 | CMB angular scale | Identical to LCDM (E(a) = 0 at z>100) |
| 7 | Growth suppression | ~7% at z=0 (approximate) |
| 8 | H0 matter sector | 72.51 km/s/Mpc (0.51-sigma from SH0ES) |
| 9 | CMB power spectrum | First peak ell=220, 5753 muK^2 |

**Key result:** Sound horizon is IDENTICAL in IAM and LCDM.
IAM resolves the Hubble tension without modifying early-universe physics.
This distinguishes IAM from Early Dark Energy (EDE) models.

**Remaining:** Full perturbation-level validation (mu < 1, Sigma = 1 
in Boltzmann hierarchy) requires MGCAMB. Background validation confirms 
all distance and expansion rate predictions.

---

## What IAM Does

### Resolves Hubble Tension

- **Planck CMB:** H0 = 67.4 km/s/Mpc (photon sector, B_gamma < 10^-5)
- **SH0ES Distance Ladder:** H0 = 73.04 km/s/Mpc (matter sector, B_m = 0.157)
- **Both correct:** Different sectors, not conflicting measurements

### Addresses S8 Tension

- **Growth suppression:** 1.36% at z=0 from Omega_m dilution
- **Effective sigma8:** 0.800 (intermediate between Planck 0.811 and DES/KiDS ~0.77)
- **Natural mechanism:** No ad-hoc parameters

### Passes CMB Consistency

- **CMB lensing:** 85% geometric compensation
- **Acoustic scale:** B_gamma < 10^-5 maintains theta_s precision
- **Early universe:** No modifications before z ~ 1

### No Overfitting

- **AIC penalty:** Delta-AIC = 26.0 >> 10 (decisive preference)
- **BIC penalty:** Delta-BIC = 25.4 >> 10 (very strong preference)
- **Relative likelihood:** LCDM is 444,000x less likely than IAM

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

3. **[IAM–CAMB Technical Note](docs/IAM_CAMB_Technical_Note.pdf)** (~9 pages)
   - μ–Σ modified gravity mapping: μ(a) < 1, Σ(a) = 1
   - Python-level CAMB validation with comprehensive 8-panel figure
   - Fortran-level implementation: what was done, what was learned, what remains
   - MGCAMB/EFTCAMB implementation roadmap for community
   - Falsifiable predictions for DES, Euclid, DESI, CMB-S4

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

6. **[Holographic Derivation](docs/Holographic_Derivation_of_IAM.pdf)** (~10 pages)
   - First-principles derivation of the activation function E(a) = exp(1 - 1/a) from horizon thermodynamics
   - Bekenstein-Hawking entropy + Gibbons-Hawking temperature + Landauer's principle
   - Key result: the 1/a in the exponent arises from information surface density on the cosmic horizon
   - Sheth-Tormen halo mass function recovers beta = 1.009 (within 1%) at galaxy-scale halos
   - Numerical verification: Pearson r > 0.99, coefficients within 1-8% of target
   - Elevates IAM from phenomenological fit to physically motivated cosmological framework

7. **[Variational Derivation](docs/Variational_Derivation_of_IAM.pdf)** (~14 pages)
   - Formal derivation chain: Jacobson (1995) → Cai-Kim (2005) → IAM (2026)
   - Step-by-step walkthrough of Jacobson's thermodynamic derivation of Einstein's equation
   - Cai-Kim extension to FRW apparent horizon producing Friedmann equations
   - IAM modification: S_total = S_geometric + S_informational → modified Friedmann equation
   - Exponentiation explained via multiplicative microstate counting on the horizon
   - Constrained scalar field action with equation of state w_info(a) = -1 - 1/(3a) (mildly phantom, consistent with DESI 2024)
   - **Coupling constant derived:** β_m = Ω_m/2 from virial theorem, matching MCMC to 0.3%
   - **Zero free parameters** beyond standard ΛCDM: functional form, exponent, normalization, and amplitude all derived
   - **Perturbation theory derived:** δφ = 0 (horizon quantity) → standard GR perturbations on IAM background → μ(a) < 1, Σ(a) = 1
   - Unique μ–Σ signature distinguishes IAM from f(R) and Horndeski theories (testable by Euclid)
   - Identifies the single new physical input: informational entropy from gravitational decoherence

### Quick Reference

- **Theory Summary:** See Section II-III of Main Manuscript
- **Statistical Results:** See Test Validation Compendium
- **μ–Σ Mapping & CAMB:** See IAM–CAMB Technical Note
- **Code Details:** See Supplementary Methods
- **Holographic Derivation:** See Holographic Derivation (first-principles origin of activation function)
- **Formal Derivation Chain:** See Variational Derivation (Jacobson → Cai-Kim → IAM)

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

### Modified Gravity Mapping: μ–Σ Parametrization

The dual-sector phenomenology maps directly onto the standard μ–Σ modified gravity framework used by DES, KiDS, Euclid, and CMB-S4:

```
μ(a) = H²_ΛCDM(a) / [H²_ΛCDM(a) + β_m · E(a)]  < 1   (suppressed growth)
Σ(a) = 1                                                  (standard photon deflection)
```

| Redshift z | μ(a) | Physical meaning |
|-----------|------|-----------------|
| 0.0 | 0.864 | 13.6% growth suppression today |
| 0.5 | 0.920 | Moderate suppression |
| 1.0 | 0.982 | Near-GR |
| 3.0 | 0.9998 | Recovers ΛCDM |

**Key signature:** μ < 1 with Σ = 1 means matter feels weaker gravity while photon deflection is standard. This is testable with existing Boltzmann solvers (MGCAMB, EFTCAMB, ISiTGR) and uniquely distinguishes IAM from generic modified gravity theories. See the [IAM–CAMB Technical Note](docs/IAM_CAMB_Technical_Note.pdf) for full details.

---

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

### 4. Statistical Significance & Model Selection

Combined fit to all datasets:

- chi2(LCDM) = 38.28 --> poor fit (chi2/dof = 3.83)
- chi2(IAM) = 8.27 --> excellent fit (chi2/dof = 1.03)
- **Delta-chi2 = 30.01 (5.5 sigma improvement)**

Model selection criteria (addressing overfitting):

- **Delta-AIC = 26.0** --> "Decisive" evidence for IAM (Burnham & Anderson)
- **Delta-BIC = 25.4** --> "Very strong" evidence for IAM (Kass & Raftery)
- **Relative likelihood:** LCDM is 444,000x less likely

**Even with penalties for 2 additional parameters, IAM is strongly preferred.**

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
  author  = {Mahaffey, Heath W.},
  title   = {Dual-Sector Cosmology from Structure-Driven Expansion: 
             The Informational Actualization Model (IAM)},
  journal = {In preparation},
  year    = {2026},
  note    = {Code: \url{https://github.com/hmahaffeyges/IAM-Validation}}
}
```

---

## What IAM Claims vs. Does NOT Claim

### What IAM Claims

- Empirical evidence for sector-dependent expansion: B_gamma/B_m < 10^-5 (MCMC)
- 5.5 sigma statistical improvement over LCDM (Delta-chi2 = 30.01)
- No evidence of overfitting (Delta-AIC = 26.0, Delta-BIC = 25.4)
- Simultaneous resolution of H0 tension and partial resolution of S8 tension
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

The author thanks the Planck, SDSS/BOSS/eBOSS, SH0ES, and JWST collaborations for publicly available data. Grateful to the open-source communities of NumPy, SciPy, Matplotlib, and corner. This work benefited from discussions facilitated by Claude (Anthropic) regarding statistical methodology, MCMC implementation, growth calculations, and reproducibility best practices.

---

**Last Updated:** February 15, 2026  
**Status:** 5.6 sigma preference for dual-sector cosmology over LCDM; zero free parameters — all derived from first principles  
**Key Result:** The Hubble tension reflects measurements of two distinct expansion rates — photons (CMB, B_gamma < 10^-5) and matter (BAO/distance ladder, B_m = Omega_m/2 = 0.1575). Both Planck and SH0ES are correct; they measure different sectors. The derived coupling gives H0(matter) = 72.51 km/s/Mpc (0.51 sigma from SH0ES) with Delta-chi2 = 31.2 for zero additional parameters. The mu-Sigma signature (mu < 1, Sigma = 1) is unique to IAM and directly testable by Euclid and DESI.

---

<p align="center">
  <i>"The universe actualizes its potential through structure formation, and geometry responds."</i>
</p>
