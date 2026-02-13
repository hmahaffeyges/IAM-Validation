# IAM: Holographic Horizon Dynamics Resolve Hubble Tension

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**The Informational Actualization Model: Holographic Horizon Dynamics Couple Quantum Structure Formation to Cosmic Expansion**

**Key Finding:** 5.5Ïƒ empirical evidence for dual-sector cosmology resolving the Hubble tension through sector-specific late-time expansion rates.

---

## ðŸŽ¯ Core Results

| Parameter | Value | Method | Description |
|-----------|-------|--------|-------------|
| **Î²_m** | 0.157 Â± 0.029 | MCMC (68% CL) | Matter-sector coupling |
| **Î²_Î³** | < 1.4 Ã— 10â»â¶ | MCMC (95% CL) | Photon-sector coupling |
| **Î²_Î³/Î²_m** | < 8.5 Ã— 10â»â¶ | MCMC (95% CL) | Empirical sector ratio |
| **Hâ‚€(photon)** | 67.4 km/s/Mpc | Planck CMB | Photon-sector measurement |
| **Hâ‚€(matter)** | 72.5 Â± 1.0 km/s/Mpc | IAM prediction | Matter-sector prediction |
| **Î”Ï‡Â²** | 30.01 (5.5Ïƒ) | vs. Î›CDM | Statistical improvement |
| **Î”AIC** | 26.0 | Model selection | No overfitting |
| **Î”BIC** | 25.4 | Model selection | Strong preference |

**The Hubble tension is resolved:** Planck (photon sector, Î²_Î³ < 10â»âµ) and SH0ES (matter sector, Î²_m = 0.157) both measure correctlyâ€”they probe different expansion rates. Photons couple at least **100,000Ã— more weakly** than matter to late-time expansion.

---

## ðŸš€ Quick Start

### Installation

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git
cd IAM-Validation
pip install numpy scipy matplotlib corner
```

**Note:** The `corner` package (for MCMC plots) will auto-install if missing.

### Run Validation

```bash
python iam_validation.py
```

**Expected runtime:** ~1 minute on standard laptop (generates 9 figures)

**Expected output:**
```
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
  INFORMATIONAL ACTUALIZATION MODEL (IAM)
  Complete Validation Presentation
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

[1/6] Checking Python environment...
âœ“ Python 3.x.x detected
âœ“ numpy installed
âœ“ scipy installed
âœ“ matplotlib installed
âœ“ corner installed

[2/6] Cosmological Parameters and Observational Data
Planck 2020 Cosmological Parameters...
Hâ‚€ Measurements (Hubble Constant)...
SDSS/BOSS/eBOSS Growth Rate Compilation...
Total data points: 3 Hâ‚€ + 7 SDSS/BOSS/eBOSS = 10

[3/6] IAM Mathematical Framework
CORE EQUATIONS:
  EQUATION 1: Activation Function E(a) = exp(1 - 1/a)
  EQUATION 2: Modified Friedmann Equation
  EQUATION 3: Effective Matter Density Parameter
  ...
DUAL-SECTOR FRAMEWORK:
  Photon sector: Î²_Î³ â‰ˆ 0 â†’ Hâ‚€(photon) = 67.4 km/s/Mpc
  Matter sector: Î²_m = 0.157 â†’ Hâ‚€(matter) = 72.5 km/s/Mpc

[4/6] Chi-Squared Calculation Methodology
EXAMPLE: How Ï‡Â² is computed for Hâ‚€ measurements
  Î›CDM: Ï‡Â²_Hâ‚€ = 31.91
  IAM:  Ï‡Â²_Hâ‚€ = 1.52
  Improvement: Î”Ï‡Â²_Hâ‚€ = 30.40

[5/6] Validated Test Results

TEST 1: Î›CDM Baseline (Standard Cosmology)
  Ï‡Â²_total = 38.28
  âœ— Î›CDM fails to resolve Hubble tension

TEST 2: IAM Dual-Sector Model
  Î²_m = 0.157 (MCMC median)
  Ï‡Â²_total = 8.27
  Î”Ï‡Â² = 30.01 (5.5Ïƒ)
  âœ“ IAM resolves Hubble tension with high significance

TEST 3: Confidence Intervals (Profile Likelihood)
  68% CL (1Ïƒ): Î²_m = 0.157 Â± 0.029
  95% CL (2Ïƒ): Î²_m = 0.157 Â± 0.057

TEST 4: Photon-Sector Constraint (MCMC)
  Profile likelihood: Î²_Î³ < 0.004 (95% CL)
  MCMC constraint:    Î²_Î³ < 1.40e-06 (95% CL)
  Sector ratio:       Î²_Î³/Î²_m < 8.50e-06 (95% CL)
  âœ“ Photons couple at least 100,000Ã— more weakly than matter

TEST 5: Physical Predictions
  Hâ‚€(photon/CMB)  = 67.4 km/s/Mpc
  Hâ‚€(matter/local) = 72.5 Â± 1.0 km/s/Mpc
  Growth suppression = 1.36%
  Ïƒâ‚ˆ(IAM) = 0.800
  âœ“ All predictions consistent with observations

TEST 6: CMB Lensing Consistency
  Growth suppression (1.36%) â†’ weaker lensing
  Reduced lensing compensates ~85% of geometric Î¸_s shift
  âœ“ Natural compensation maintains CMB consistency

TEST 7: Model Selection Criteria (Overfitting Check)
  Î”AIC = 26.01 â†’ 'Decisive' evidence for IAM
  Î”BIC = 25.40 â†’ 'Very strong' evidence for IAM
  Relative likelihood: Î›CDM is 444,000Ã— less likely
  âœ“ No evidence of overfitting despite 2 additional parameters

TEST 8: Full Bayesian MCMC Analysis
  Î²_m = 0.157 +0.029/-0.029 (68% CL)
  Î²_Î³ < 1.40e-06 (95% upper limit)
  Î²_Î³/Î²_m < 8.50e-06 (95% upper limit)
  Hâ‚€(matter) = 72.5 Â± 1.0 km/s/Mpc
  âœ“ Well-behaved Gaussian posteriors with no degeneracies

TEST 9: Pantheon+ Supernovae Distance Validation
  Both models show similar fit quality to SNe data
  Primary IAM impact is on GROWTH, not GEOMETRY
  âœ“ IAM maintains distance consistency

[6/6] Generating Publication-Quality Figures
Generating Figure 1: Hâ‚€ Measurements Comparison...
Generating Figure 2: Growth Suppression Evolution...
...
Generating Figure 9: MCMC Parameter Constraints...
âœ“ All 9 figures generated successfully!
```

---

## ðŸ“Š What IAM Does

### âœ… Resolves Hubble Tension

- **Planck CMB:** Hâ‚€ = 67.4 km/s/Mpc (photon sector, Î²_Î³ < 10â»âµ)
- **SH0ES Distance Ladder:** Hâ‚€ = 73.04 km/s/Mpc (matter sector, Î²_m = 0.157)
- **Both correct:** Different sectors, not conflicting measurements

### âœ… Addresses Sâ‚ˆ Tension

- **Growth suppression:** 1.36% at z=0 from Î©_m dilution
- **Effective Ïƒâ‚ˆ:** 0.800 (intermediate between Planck 0.811 and DES/KiDS ~0.77)
- **Natural mechanism:** No ad-hoc parameters

### âœ… Passes CMB Consistency

- **CMB lensing:** 85% geometric compensation
- **Acoustic scale:** Î²_Î³ < 10â»âµ maintains Î¸_s precision
- **Early universe:** No modifications before z ~ 1

### âœ… No Overfitting

- **AIC penalty:** Î”AIC = 26.0 >> 10 (decisive preference)
- **BIC penalty:** Î”BIC = 25.4 >> 10 (very strong preference)
- **Relative likelihood:** Î›CDM is 444,000Ã— less likely than IAM

### âœ… Makes Testable Predictions

- **CMB-S4:** Will constrain Î²_Î³ < 10â»â´ (100Ã— tighter)
- **Euclid:** Sâ‚ˆ = 0.78 Â± 0.01
- **DESI Year 5:** Î²_m to Â±1% precision

---

## ðŸ“– Documentation

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

3. **[Test Validation Compendium](docs/IAM_Test_Validation_Compendium.pdf)** (~30 pages)
   - Nine independent validation tests with detailed results
   - Nine publication-quality figures
   - Complete chi-squared analysis
   - MCMC posterior analysis

4. **[Supplementary Methods](docs/Supplementary_Methods_Reproducibility_Guide.pdf)** (~20 pages)
   - Complete Python implementation
   - Data sources and citations
   - Step-by-step reproducibility instructions
   - Troubleshooting guide

### Quick Reference

- **Theory Summary:** See Section II-III of Main Manuscript
- **Statistical Results:** See Test Validation Compendium
- **Code Details:** See Supplementary Methods

---

## ðŸ”¬ Physical Framework

### Dual-Sector Hubble Parameters

**Matter sector** (BAO, growth, distance ladder):
```
HÂ²_m(a) = Hâ‚€Â²[Î©_mÂ·aâ»Â³ + Î©_rÂ·aâ»â´ + Î©_Î› + Î²_mÂ·E(a)]
```

**Photon sector** (CMB, photon propagation):
```
HÂ²_Î³(a) = Hâ‚€Â²[Î©_mÂ·aâ»Â³ + Î©_rÂ·aâ»â´ + Î©_Î› + Î²_Î³Â·E(a)]
```

**Activation function:**
```
E(a) = exp(1 - 1/a)
```

### Key Mechanism

The Î² term enters the denominator, diluting effective matter density:

```
Î©_m(a) = [Î©_mÂ·aâ»Â³] / [Î©_mÂ·aâ»Â³ + Î©_rÂ·aâ»â´ + Î©_Î› + Î²Â·E(a)]
```

This naturally suppresses structure growth without additional parameters.

---

## ðŸ“Š Datasets Used

### Primary Data Sources

1. **Planck 2020 CMB** ([A&A 641, A6](https://doi.org/10.1051/0004-6361/201833910))
   - Hâ‚€: 67.4 Â± 0.5 km/s/Mpc
   - Î¸_s: 0.0104110 Â± 0.0000031 rad
   - Ïƒâ‚ˆ: 0.811 Â± 0.006

2. **SH0ES 2022** ([ApJL 934, L7](https://doi.org/10.3847/2041-8213/ac5c5b))
   - Hâ‚€: 73.04 Â± 1.04 km/s/Mpc (Cepheid distance ladder)

3. **JWST TRGB 2024** ([ApJ 919, 16](https://arxiv.org/abs/2308.14864))
   - Hâ‚€: 70.39 Â± 1.89 km/s/Mpc

4. **SDSS/BOSS/eBOSS Consensus** ([Alam et al. 2021, PRD 103, 083533](https://doi.org/10.1103/PhysRevD.103.083533))
   - fÂ·Ïƒâ‚ˆ(z) at 7 redshifts (0.295 < z < 2.33)

5. **Pantheon+SH0ES 2022** ([ApJ 938, 110](https://doi.org/10.3847/1538-4357/ac8e04))
   - 1588 Type Ia supernovae (0.01 < z < 2.26)
   - Public data: https://github.com/PantheonPlusSH0ES/DataRelease
   - Used in dual-sector validation analysis

**Total:** 10 independent measurements (3 Hâ‚€ + 7 growth rate)

---

## ðŸŽ“ Key Findings

### 1. Empirical Sector Separation (MCMC Result)

The ratio Î²_Î³/Î²_m < 8.5 Ã— 10â»â¶ (95% CL) is **data-driven**, not theoretically imposed:

- Photon-sector constraint from CMB acoustic scale precision
- Matter-sector constraint from BAO and Hâ‚€ measurements
- Full Bayesian MCMC analysis confirms sector separation

**This transforms "photon exemption" from assumption to empirical discovery: photons couple at least 100,000Ã— more weakly than matter.**

### 2. Growth Suppression Mechanism

Growth suppression emerges naturally from Î©_m dilution:

- Î² in denominator â†’ reduced effective Î©_m(a)
- Weaker gravity â†’ suppressed structure formation
- 1.36% suppression at z=0 â†’ Ïƒâ‚ˆ = 0.800

**No ad-hoc "growth tax" parameter required.**

### 3. CMB Lensing Consistency

Modified growth naturally compensates geometric effects:

- Geometric shift from modified H(z): +1.02%
- Lensing reduction from growth suppression: -0.87%
- **85% compensation** without tuning
- Remaining 15% resolved by Î²_Î³ < 10â»âµ

### 4. Statistical Significance & Model Selection

Combined fit to all datasets:

- Ï‡Â²(Î›CDM) = 38.28 â†’ poor fit (Ï‡Â²/dof = 3.83)
- Ï‡Â²(IAM) = 8.27 â†’ excellent fit (Ï‡Â²/dof = 1.03)
- **Î”Ï‡Â² = 30.01 (5.5Ïƒ improvement)**

Model selection criteria (addressing overfitting):

- **Î”AIC = 26.0** â†’ "Decisive" evidence for IAM (Burnham & Anderson)
- **Î”BIC = 25.4** â†’ "Very strong" evidence for IAM (Kass & Raftery)
- **Relative likelihood:** Î›CDM is 444,000Ã— less likely

**Even with penalties for 2 additional parameters, IAM is strongly preferred.**

### 5. Distance Consistency (Pantheon+ SNe)

Independent validation with supernovae:

- IAM maintains consistency with geometric distance measurements
- Primary IAM impact is on **GROWTH**, not **GEOMETRY**
- Effect on distances subdominant to Î©_Î›
- Full Pantheon+ dataset confirms distance consistency

### 6. Dual-Sector Empirical Validation (Separate Paper)

Extended empirical validation of dual-sector expansion using Type Ia supernovae is documented in a separate companion paper.

Mahaffey, H. W. (2026). "Dual-Sector Expansion: Type Ia Supernovae Validate Matter-Sector Hâ‚€ Normalization with Î›CDM Geometric Consistency"

- Location: docs/Dual_Sector_Validation_Paper.pdf
- Dataset: Pantheon+SH0ES (1588 Type Ia supernovae, 0.01 < z < 2.26)
- Complete reproducible code provided in paper appendices

Three independent tests using Pantheon+ data demonstrate that Type Ia supernovae reject photon-sector expansion (Hâ‚€ = 67.4 km/s/Mpc, Test A: Î² â†’ -0.30 boundary), accept matter-sector normalization (Hâ‚€ = 73.04 km/s/Mpc, Test B: Î² â‰ˆ 0), and maintain Î›CDM geometric consistency (Test C: confirms matter preference). These results validate that dual-sector separation emerges from data, not theoretical assumption, confirming IAM's prediction that structure formation couples differently to expansion than photon propagation.

---

## ðŸ”® Testable Predictions

### Near-Term (< 5 years)

| Experiment | Prediction | Timeline |
|------------|------------|----------|
| **DESI Year 5 (future)** | Î²_m to Â±1% precision | 2029 |
| **Euclid** | Sâ‚ˆ = 0.78 Â± 0.01 | 2025-2030 |
| **Simons Observatory** | Î²_Î³ < 0.001 (10Ã— tighter) | 2025-2028 |
| **Rubin-LSST** | Minimal deviation in SNe distances | 2025-2030 |

### Long-Term (> 5 years)

| Experiment | Prediction | Timeline |
|------------|------------|----------|
| **CMB-S4** | Î²_Î³ < 10â»â´ or detect nonzero coupling | 2030+ |
| **Euclid + Rubin** | BAO at z > 2 tests early-time behavior | 2030+ |
| **GW Standard Sirens** | Hâ‚€(matter) consistent with distance ladder | 2030+ |

---

## ðŸ¤ Citation

If you use this code or results in published research, please cite:

```bibtex
@article{Mahaffey2026,
  author  = {Mahaffey, Heath W.},
  title   = {The Informational Actualization Model: Holographic Horizon 
             Dynamics Couple Quantum Structure Formation to Cosmic Expansion},
  journal = {In preparation},
  year    = {2026},
  note    = {Code: \url{https://github.com/hmahaffeyges/IAM-Validation}}
}
```

---

## ðŸ“œ What IAM Claims vs. Does NOT Claim

### âœ… What IAM Claims

- Empirical evidence for sector-dependent expansion: Î²_Î³/Î²_m < 10â»âµ (MCMC)
- 5.5Ïƒ statistical improvement over Î›CDM (Î”Ï‡Â² = 30.01)
- No evidence of overfitting (Î”AIC = 26.0, Î”BIC = 25.4)
- Simultaneous resolution of Hâ‚€ tension and partial resolution of Sâ‚ˆ tension
- Testable predictions for upcoming surveys (CMB-S4, Euclid, DESI Year 5)
- Natural growth suppression mechanism from Î©_m dilution

### âŒ What IAM Does NOT Claim

- Fundamental derivation from quantum gravity (this is phenomenology)
- Modification of Einstein's equations or gauge structure
- That information is a new physical field or substance
- Uniqueness (other parameterizations may fit similarly)
- Explanation of early-universe physics or inflation

**IAM is a phenomenological late-time framework** motivated by horizon thermodynamics (Bekenstein-Hawking entropy, holographic principle, quantum decoherence). Its value lies in providing empirically testable predictions that unify multiple cosmological tensions.

---

## ðŸ§ª Development History

This repository presents the final validated framework. Complete development history, including exploratory tests and deprecated approaches, is available in the [`development/`](development/) directory. See [`development/README_development.md`](development/README_development.md) for scientific evolution and key breakthroughs.

**Validation Timeline:**
- **Tests 1-26:** Early exploration (growth mechanisms, various parameterizations)
- **Tests 27-29:** Dual-sector discovery (breakthrough: empirical sector separation)
- **Test 30:** Final synthesis (consolidated validation)
- **Current:** 9 tests in `iam_validation.py` with full MCMC analysis

**Main validation consolidated into `iam_validation.py` for clarity and reproducibility.**

---

## ðŸ”§ Contact

**Heath W. Mahaffey**  
Independent Researcher  
Entiat, WA 98822, USA  

- **Email:** hmahaffeyges@gmail.com
- **GitHub:** [@hmahaffeyges](https://github.com/hmahaffeyges)

For questions, issues, or collaboration inquiries, please open an issue on GitHub or email directly.

---

## ðŸ“„ License

MIT License - Free to use, modify, and distribute with attribution.

See [LICENSE](LICENSE) for full details.

---

## ðŸ™ Acknowledgments

The author thanks the Planck, SDSS/BOSS/eBOSS, SH0ES, and JWST collaborations for publicly available data. Grateful to the open-source communities of NumPy, SciPy, Matplotlib, and corner. This work benefited from discussions facilitated by Claude (Anthropic) regarding statistical methodology, MCMC implementation, growth calculations, and reproducibility best practices.

---

**Last Updated:** February 11, 2026  
**Status:** 5.5Ïƒ preference for dual-sector cosmology over Î›CDM  
**Key Result:** The Hubble tension reflects measurements of two distinct expansion ratesâ€”photons (CMB, Î²_Î³ < 10â»âµ) and matter (BAO/distance ladder, Î²_m = 0.157). Both Planck and SH0ES are correct; they measure different sectors with empirically constrained ratio Î²_Î³/Î²_m < 10â»âµ (95% CL, MCMC). Photons couple at least **100,000Ã— more weakly** than matter to late-time expansion.

---

<p align="center">
  <i>"The universe actualizes its potential through structure formation, and geometry responds."</i>
</p>
