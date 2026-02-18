# IAM: Holographic Horizon Dynamics Resolve Hubble Tension

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**The Informational Actualization Model: Holographic Horizon Dynamics Couple Quantum Structure Formation to Cosmic Expansion**

**Key Finding:** 5.6σ empirical evidence for dual-sector cosmology resolving the Hubble tension through sector-specific late-time expansion rates.

---

## 🎯 Core Results

| Parameter | Value | Method | Description |
|-----------|-------|--------|-------------|
| **β_m** | 0.164 ± 0.029 | MCMC (68% CL) | Matter-sector coupling |
| **β_γ** | < 1.4 × 10⁻⁶ | MCMC (95% CL) | Photon-sector coupling |
| **β_γ/β_m** | < 8.5 × 10⁻⁶ | MCMC (95% CL) | Empirical sector ratio |
| **H₀(photon)** | 67.4 km/s/Mpc | Planck CMB | Photon-sector measurement |
| **H₀(matter)** | 72.7 ± 1.0 km/s/Mpc | IAM prediction | Matter-sector prediction |
| **Δχ²** | 31.25 (5.6σ) | vs. ΛCDM | Statistical improvement |
| **ΔAIC** | 27.2 | Model selection | No overfitting |
| **ΔBIC** | 26.6 | Model selection | Strong preference |

**The Hubble tension is resolved:** Planck (photon sector, β_γ < 10⁻⁵) and SH0ES (matter sector, β_m = 0.164) both measure correctly—they probe different expansion rates. Photons couple at least **100,000× more weakly** than matter to late-time expansion.

---

## 🚀 Quick Start

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
════════════════════════════════════════════════════════════════════════════════
  INFORMATIONAL ACTUALIZATION MODEL (IAM)
  Complete Validation Presentation
════════════════════════════════════════════════════════════════════════════════

[1/6] Checking Python environment...
✓ Python 3.x.x detected
✓ numpy installed
✓ scipy installed
✓ matplotlib installed
✓ corner installed

[2/6] Cosmological Parameters and Observational Data
Planck 2020 Cosmological Parameters...
H₀ Measurements (Hubble Constant)...
DESI DR2 Growth Rate Measurements...
Total data points: 3 H₀ + 7 DESI = 10

[3/6] IAM Mathematical Framework
CORE EQUATIONS:
  EQUATION 1: Activation Function E(a) = exp(1 - 1/a)
  EQUATION 2: Modified Friedmann Equation
  EQUATION 3: Effective Matter Density Parameter
  ...
DUAL-SECTOR FRAMEWORK:
  Photon sector: β_γ ≈ 0 → H₀(photon) = 67.4 km/s/Mpc
  Matter sector: β_m = 0.157 → H₀(matter) = 72.5 km/s/Mpc

[4/6] Chi-Squared Calculation Methodology
EXAMPLE: How χ² is computed for H₀ measurements
  ΛCDM: χ²_H₀ = 31.91
  IAM:  χ²_H₀ = 1.52
  Improvement: Δχ²_H₀ = 30.40

[5/6] Validated Test Results

TEST 1: ΛCDM Baseline (Standard Cosmology)
  χ²_total = 41.63
  ✗ ΛCDM fails to resolve Hubble tension

TEST 2: IAM Dual-Sector Model
  β_m = 0.164 (MCMC median)
  χ²_total = 10.38
  Δχ² = 31.25 (5.6σ)
  ✓ IAM resolves Hubble tension with high significance

TEST 3: Confidence Intervals (Profile Likelihood)
  68% CL (1σ): β_m = 0.164 ± 0.029
  95% CL (2σ): β_m = 0.164 ± 0.058

TEST 4: Photon-Sector Constraint (MCMC)
  Profile likelihood: β_γ < 0.004 (95% CL)
  MCMC constraint:    β_γ < 1.40e-06 (95% CL)
  Sector ratio:       β_γ/β_m < 8.50e-06 (95% CL)
  ✓ Photons couple at least 100,000× more weakly than matter

TEST 5: Physical Predictions
  H₀(photon/CMB)  = 67.4 km/s/Mpc
  H₀(matter/local) = 72.7 ± 1.0 km/s/Mpc
  Growth suppression = 1.36%
  σ₈(IAM) = 0.800
  ✓ All predictions consistent with observations

TEST 6: CMB Lensing Consistency
  Growth suppression (1.36%) → weaker lensing
  Reduced lensing compensates ~85% of geometric θ_s shift
  ✓ Natural compensation maintains CMB consistency

TEST 7: Model Selection Criteria (Overfitting Check)
  ΔAIC = 27.25 → 'Decisive' evidence for IAM
  ΔBIC = 26.64 → 'Very strong' evidence for IAM
  Relative likelihood: ΛCDM is 827,000× less likely
  ✓ No evidence of overfitting despite 2 additional parameters

TEST 8: Full Bayesian MCMC Analysis
  β_m = 0.164 +0.029/-0.028 (68% CL)
  β_γ < 1.40e-06 (95% upper limit)
  β_γ/β_m < 8.50e-06 (95% upper limit)
  H₀(matter) = 72.7 ± 1.0 km/s/Mpc
  ✓ Well-behaved Gaussian posteriors with no degeneracies

TEST 9: Pantheon+ Supernovae Distance Validation
  Both models show similar fit quality to SNe data
  Primary IAM impact is on GROWTH, not GEOMETRY
  ✓ IAM maintains distance consistency

[6/6] Generating Publication-Quality Figures
Generating Figure 1: H₀ Measurements Comparison...
Generating Figure 2: Growth Suppression Evolution...
...
Generating Figure 9: MCMC Parameter Constraints...
✓ All 9 figures generated successfully!
```

---

## 📊 What IAM Does

### ✅ Resolves Hubble Tension

- **Planck CMB:** H₀ = 67.4 km/s/Mpc (photon sector, β_γ < 10⁻⁵)
- **SH0ES Distance Ladder:** H₀ = 73.04 km/s/Mpc (matter sector, β_m = 0.164)
- **Both correct:** Different sectors, not conflicting measurements

### ✅ Addresses S₈ Tension

- **Growth suppression:** 1.36% at z=0 from Ω_m dilution
- **Effective σ₈:** 0.800 (intermediate between Planck 0.811 and DES/KiDS ~0.77)
- **Natural mechanism:** No ad-hoc parameters

### ✅ Passes CMB Consistency

- **CMB lensing:** 85% geometric compensation
- **Acoustic scale:** β_γ < 10⁻⁵ maintains θ_s precision
- **Early universe:** No modifications before z ~ 1

### ✅ No Overfitting

- **AIC penalty:** ΔAIC = 27.2 >> 10 (decisive preference)
- **BIC penalty:** ΔBIC = 26.6 >> 10 (very strong preference)
- **Relative likelihood:** ΛCDM is 827,000× less likely than IAM

### ✅ Makes Testable Predictions

- **CMB-S4:** Will constrain β_γ < 10⁻⁴ (100× tighter)
- **Euclid:** S₈ = 0.78 ± 0.01
- **DESI Year 5:** β_m to ±1% precision

---

## 📖 Documentation

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

## 🔬 Physical Framework

### Dual-Sector Hubble Parameters

**Matter sector** (BAO, growth, distance ladder):
```
H²_m(a) = H₀²[Ω_m·a⁻³ + Ω_r·a⁻⁴ + Ω_Λ + β_m·E(a)]
```

**Photon sector** (CMB, photon propagation):
```
H²_γ(a) = H₀²[Ω_m·a⁻³ + Ω_r·a⁻⁴ + Ω_Λ + β_γ·E(a)]
```

**Activation function:**
```
E(a) = exp(1 - 1/a)
```

### Key Mechanism

The β term enters the denominator, diluting effective matter density:

```
Ω_m(a) = [Ω_m·a⁻³] / [Ω_m·a⁻³ + Ω_r·a⁻⁴ + Ω_Λ + β·E(a)]
```

This naturally suppresses structure growth without additional parameters.

---

## 📊 Datasets Used

### Primary Data Sources

1. **Planck 2020 CMB** ([A&A 641, A6](https://doi.org/10.1051/0004-6361/201833910))
   - H₀: 67.4 ± 0.5 km/s/Mpc
   - θ_s: 0.0104110 ± 0.0000031 rad
   - σ₈: 0.811 ± 0.006

2. **SH0ES 2022** ([ApJL 934, L7](https://doi.org/10.3847/2041-8213/ac5c5b))
   - H₀: 73.04 ± 1.04 km/s/Mpc (Cepheid distance ladder)

3. **JWST TRGB 2024** ([ApJ 919, 16](https://arxiv.org/abs/2308.14864))
   - H₀: 70.39 ± 1.89 km/s/Mpc

4. **DESI DR2 2024** ([arXiv:2404.03002](https://arxiv.org/abs/2404.03002))
   - f·σ₈(z) at 7 redshifts (0.295 < z < 2.33)

5. **Pantheon+SH0ES 2022** ([ApJ 938, 110](https://doi.org/10.3847/1538-4357/ac8e04))
   - 1588 Type Ia supernovae (0.01 < z < 2.26)
   - Public data: https://github.com/PantheonPlusSH0ES/DataRelease
   - Used in dual-sector validation analysis

**Total:** 10 independent measurements (3 H₀ + 7 growth rate)

---

## 🎓 Key Findings

### 1. Empirical Sector Separation (MCMC Result)

The ratio β_γ/β_m < 8.5 × 10⁻⁶ (95% CL) is **data-driven**, not theoretically imposed:

- Photon-sector constraint from CMB acoustic scale precision
- Matter-sector constraint from BAO and H₀ measurements
- Full Bayesian MCMC analysis confirms sector separation

**This transforms "photon exemption" from assumption to empirical discovery: photons couple at least 100,000× more weakly than matter.**

### 2. Growth Suppression Mechanism

Growth suppression emerges naturally from Ω_m dilution:

- β in denominator → reduced effective Ω_m(a)
- Weaker gravity → suppressed structure formation
- 1.36% suppression at z=0 → σ₈ = 0.800

**No ad-hoc "growth tax" parameter required.**

### 3. CMB Lensing Consistency

Modified growth naturally compensates geometric effects:

- Geometric shift from modified H(z): +1.02%
- Lensing reduction from growth suppression: -0.87%
- **85% compensation** without tuning
- Remaining 15% resolved by β_γ < 10⁻⁵

### 4. Statistical Significance & Model Selection

Combined fit to all datasets:

- χ²(ΛCDM) = 41.63 → poor fit (χ²/dof = 4.16)
- χ²(IAM) = 10.38 → excellent fit (χ²/dof = 1.15)
- **Δχ² = 31.25 (5.6σ improvement)**

Model selection criteria (addressing overfitting):

- **ΔAIC = 27.2** → "Decisive" evidence for IAM (Burnham & Anderson)
- **ΔBIC = 26.6** → "Very strong" evidence for IAM (Kass & Raftery)
- **Relative likelihood:** ΛCDM is 827,000× less likely

**Even with penalties for 2 additional parameters, IAM is strongly preferred.**

### 5. Distance Consistency (Pantheon+ SNe)

Independent validation with supernovae:

- IAM maintains consistency with geometric distance measurements
- Primary IAM impact is on **GROWTH**, not **GEOMETRY**
- Effect on distances subdominant to Ω_Λ
- Full Pantheon+ dataset confirms distance consistency

### 6. Dual-Sector Empirical Validation (Separate Paper)

Extended empirical validation of dual-sector expansion using Type Ia supernovae is documented in a separate companion paper.

Mahaffey, H. W. (2026). "Dual-Sector Expansion: Type Ia Supernovae Validate Matter-Sector H₀ Normalization with ΛCDM Geometric Consistency"

- Location: docs/Dual_Sector_Validation_Paper.pdf
- Dataset: Pantheon+SH0ES (1588 Type Ia supernovae, 0.01 < z < 2.26)
- Complete reproducible code provided in paper appendices

Three independent tests using Pantheon+ data demonstrate that Type Ia supernovae reject photon-sector expansion (H₀ = 67.4 km/s/Mpc, Test A: β → -0.30 boundary), accept matter-sector normalization (H₀ = 73.04 km/s/Mpc, Test B: β ≈ 0), and maintain ΛCDM geometric consistency (Test C: confirms matter preference). These results validate that dual-sector separation emerges from data, not theoretical assumption, confirming IAM's prediction that structure formation couples differently to expansion than photon propagation.

---

## 🔮 Testable Predictions

### Near-Term (< 5 years)

| Experiment | Prediction | Timeline |
|------------|------------|----------|
| **DESI Year 5** | β_m to ±1% precision | 2029 |
| **Euclid** | S₈ = 0.78 ± 0.01 | 2025-2030 |
| **Simons Observatory** | β_γ < 0.001 (10× tighter) | 2025-2028 |
| **Rubin-LSST** | Minimal deviation in SNe distances | 2025-2030 |

### Long-Term (> 5 years)

| Experiment | Prediction | Timeline |
|------------|------------|----------|
| **CMB-S4** | β_γ < 10⁻⁴ or detect nonzero coupling | 2030+ |
| **Euclid + Rubin** | BAO at z > 2 tests early-time behavior | 2030+ |
| **GW Standard Sirens** | H₀(matter) consistent with distance ladder | 2030+ |

---

## 🤝 Citation

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

## 📜 What IAM Claims vs. Does NOT Claim

### ✅ What IAM Claims

- Empirical evidence for sector-dependent expansion: β_γ/β_m < 10⁻⁵ (MCMC)
- 5.6σ statistical improvement over ΛCDM (Δχ² = 31.25)
- No evidence of overfitting (ΔAIC = 27.2, ΔBIC = 26.6)
- Simultaneous resolution of H₀ tension and partial resolution of S₈ tension
- Testable predictions for upcoming surveys (CMB-S4, Euclid, DESI Year 5)
- Natural growth suppression mechanism from Ω_m dilution

### ❌ What IAM Does NOT Claim

- Fundamental derivation from quantum gravity (this is phenomenology)
- Modification of Einstein's equations or gauge structure
- That information is a new physical field or substance
- Uniqueness (other parameterizations may fit similarly)
- Explanation of early-universe physics or inflation

**IAM is a phenomenological late-time framework** motivated by horizon thermodynamics (Bekenstein-Hawking entropy, holographic principle, quantum decoherence). Its value lies in providing empirically testable predictions that unify multiple cosmological tensions.

---

## 🧪 Development History

This repository presents the final validated framework. Complete development history, including exploratory tests and deprecated approaches, is available in the [`development/`](development/) directory. See [`development/README_development.md`](development/README_development.md) for scientific evolution and key breakthroughs.

**Validation Timeline:**
- **Tests 1-26:** Early exploration (growth mechanisms, various parameterizations)
- **Tests 27-29:** Dual-sector discovery (breakthrough: empirical sector separation)
- **Test 30:** Final synthesis (consolidated validation)
- **Current:** 9 tests in `iam_validation.py` with full MCMC analysis

**Main validation consolidated into `iam_validation.py` for clarity and reproducibility.**

---

## 🔧 Contact

**Heath W. Mahaffey**  
Independent Researcher  
Entiat, WA 98822, USA  

- **Email:** hmahaffeyges@gmail.com
- **GitHub:** [@hmahaffeyges](https://github.com/hmahaffeyges)

For questions, issues, or collaboration inquiries, please open an issue on GitHub or email directly.

---

## 📄 License

MIT License - Free to use, modify, and distribute with attribution.

See [LICENSE](LICENSE) for full details.

---

## 🙏 Acknowledgments

The author thanks the Planck, DESI, SH0ES, and JWST collaborations for publicly available data. Grateful to the open-source communities of NumPy, SciPy, Matplotlib, and corner. This work benefited from discussions facilitated by Claude (Anthropic) regarding statistical methodology, MCMC implementation, growth calculations, and reproducibility best practices.

---

**Last Updated:** February 11, 2026  
**Status:** 5.6σ preference for dual-sector cosmology over ΛCDM  
**Key Result:** The Hubble tension reflects measurements of two distinct expansion rates—photons (CMB, β_γ < 10⁻⁵) and matter (BAO/distance ladder, β_m = 0.164). Both Planck and SH0ES are correct; they measure different sectors with empirically constrained ratio β_γ/β_m < 10⁻⁵ (95% CL, MCMC). Photons couple at least **100,000× more weakly** than matter to late-time expansion.

---

<p align="center">
  <i>"The universe actualizes its potential through structure formation, and geometry responds."</i>
</p>
