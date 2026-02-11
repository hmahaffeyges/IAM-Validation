# IAM: Holographic Horizon Dynamics Resolve Hubble Tension

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**The Informational Actualization Model: Holographic Horizon Dynamics Couple Quantum Structure Formation to Cosmic Expansion**

**Key Finding:** 5.6σ empirical evidence for dual-sector cosmology resolving the Hubble tension through sector-specific late-time expansion rates.

---

## 🎯 Core Results

| Parameter | Value | Description |
|-----------|-------|-------------|
| **β_m** | 0.157 ± 0.029 | Matter-sector coupling (68% CL) |
| **β_γ** | < 0.004 | Photon-sector coupling (95% CL) |
| **β_γ/β_m** | < 0.022 | Empirical sector ratio (95% CL) |
| **H₀(photon)** | 67.4 km/s/Mpc | CMB measurement (Planck) |
| **H₀(matter)** | 72.5 ± 0.9 km/s/Mpc | Local measurement (SH0ES) |
| **Δχ²** | 31.25 (5.6σ) | Improvement over ΛCDM |

**The Hubble tension is resolved:** Planck (photon sector) and SH0ES (matter sector) both measure correctly—they probe different expansion rates with empirically constrained ratio.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git
cd IAM-Validation
pip install numpy scipy matplotlib
```

### Run Validation

```bash
python iam_validation.py
```

**Expected runtime:** < 5 minutes on standard laptop

**Expected output:**
```
======================================================================
IAM VALIDATION - Profile Likelihood Analysis
======================================================================

[1/4] Computing LCDM baseline...
  LCDM: chi^2_total = 41.63

[2/4] Scanning beta_m parameter space...
  Scan complete!

[3/4] Analyzing likelihood...
  Best-fit parameter:
    beta_m = 0.157
    chi^2_min = 10.38
    Delta chi^2 = 31.25
    Significance = 5.6 sigma

[4/4] Computing physical predictions...
  H0(matter) = 72.5 km/s/Mpc
  Growth suppression = 1.36%
  sigma_8(IAM) = 0.800

======================================================================
VALIDATION COMPLETE!
======================================================================
```

---

## 📊 What IAM Does

### ✅ Resolves Hubble Tension

- **Planck CMB:** H₀ = 67.4 km/s/Mpc (photon sector, β_γ ≈ 0)
- **SH0ES Distance Ladder:** H₀ = 73.04 km/s/Mpc (matter sector, β_m = 0.157)
- **Both correct:** Different sectors, not conflicting measurements

### ✅ Addresses S₈ Tension

- **Growth suppression:** 1.36% at z=0 from Ω_m dilution
- **Effective σ₈:** 0.800 (intermediate between Planck 0.811 and DES/KiDS ~0.77)
- **Natural mechanism:** No ad-hoc parameters

### ✅ Passes CMB Consistency

- **CMB lensing:** 85% geometric compensation
- **Acoustic scale:** β_γ ≈ 0 maintains θ_s precision
- **Early universe:** No modifications before z ~ 1

### ✅ Makes Testable Predictions

- **CMB-S4:** Will constrain β_γ < 0.001 (10× tighter)
- **Euclid:** S₈ = 0.78 ± 0.01
- **DESI Year 5:** β_m to ±1% precision

---

## 📖 Documentation

### Primary Documents

1. **[Main Manuscript](docs/IAM_Manuscript.pdf)** (RevTeX, ~15 pages)
   - Full holographic motivation (Bekenstein-Hawking entropy, holographic principle)
   - Theoretical foundation and phenomenological implementation
   - Statistical validation and testable predictions

2. **[Test Validation Compendium](docs/IAM_Test_Validation_Compendium.pdf)** (~25 pages)
   - Six independent validation tests with detailed results
   - Eight publication-quality figures
   - Complete chi-squared analysis

3. **[Supplementary Methods](docs/Supplementary_Methods_Reproducibility_Guide.pdf)** (~18 pages)
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
H²_m(a) = H₀²[Ωₘa⁻³ + Ωᵣa⁻⁴ + Ω_Λ + β_m·E(a)]
```

**Photon sector** (CMB, photon propagation):
```
H²_γ(a) = H₀²[Ωₘa⁻³ + Ωᵣa⁻⁴ + Ω_Λ + β_γ·E(a)]
```

**Activation function:**
```
E(a) = exp(1 - 1/a)
```

### Key Mechanism

The β term enters the denominator, diluting effective matter density:

```
Ωₘ(a) = [Ωₘ·a⁻³] / [Ωₘ·a⁻³ + Ωᵣ·a⁻⁴ + Ω_Λ + β·E(a)]
```

This naturally suppresses structure growth without additional parameters.

---

## 📁 Repository Structure

```
IAM-Validation/
├── README.md                          # This file
├── iam_validation.py                  # Main validation script
├── data/
│   ├── desi_bao_data.txt             # DESI DR2 growth rates
│   ├── h0_measurements.txt            # Planck, SH0ES, JWST
│   └── README.md                      # Data sources & citations
├── figures/
│   ├── figure1_h0_comparison.pdf      # H₀ measurements
│   ├── figure2_growth_suppression.pdf # Growth evolution
│   ├── figure3_desi_growth.pdf        # DESI comparison
│   ├── figure4_beta_gamma.pdf         # Photon-sector constraint
│   ├── figure5_beta_m_profile.pdf     # Matter-sector likelihood
│   ├── figure6_h0_ladder.pdf          # Complete H₀ ladder
│   ├── figure7_chi2_breakdown.pdf     # Statistical analysis
│   └── figure8_summary.pdf            # Physical quantities
├── docs/
│   ├── IAM_Manuscript.pdf             # Main paper
│   ├── IAM_Test_Validation_Compendium.pdf  # Detailed tests
│   └── Supplementary_Methods_Reproducibility_Guide.pdf
├── development/
│   ├── README_development.md          # Development history
│   └── archive/                       # Historical test archive
└── LICENSE                            # MIT License
```

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

**Total:** 10 independent measurements (3 H₀ + 7 growth rate)

---

## 🎓 Key Findings

### 1. Empirical Sector Separation

The ratio β_γ/β_m < 0.022 (95% CL) is **data-driven**, not theoretically imposed:

- Photon-sector constraint from CMB acoustic scale precision
- Matter-sector constraint from BAO and H₀ measurements
- Independent analyses converge on sector separation

**This transforms "photon exemption" from assumption to empirical discovery.**

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
- Remaining 15% resolved by β_γ ≈ 0

### 4. Statistical Significance

Combined fit to all datasets:

- χ²(ΛCDM) = 41.63 → poor fit (χ²/dof = 4.16)
- χ²(IAM) = 10.38 → excellent fit (χ²/dof = 1.15)
- **Δχ² = 31.25 (5.6σ improvement)**

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
| **CMB-S4** | β_γ < 0.0001 or detect nonzero coupling | 2030+ |
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

- Empirical evidence for sector-dependent expansion: β_γ/β_m < 0.022
- 5.6σ statistical improvement over ΛCDM (Δχ² = 31.25)
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

**Main validation consolidated into `iam_validation.py` for clarity and reproducibility.**

---

## 📧 Contact

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

The author thanks the Planck, DESI, SH0ES, and JWST collaborations for publicly available data. Grateful to the open-source communities of NumPy, SciPy, and Matplotlib. This work benefited from discussions facilitated by Claude (Anthropic) regarding statistical methodology, growth calculations, and reproducibility best practices.

---

**Last Updated:** February 11, 2026  
**Status:** 5.6σ preference for dual-sector cosmology over ΛCDM  
**Key Result:** The Hubble tension reflects measurements of two distinct expansion rates—photons (CMB, β_γ ≈ 0) and matter (BAO/distance ladder, β_m = 0.157). Both Planck and SH0ES are correct; they measure different sectors with empirically constrained ratio β_γ/β_m < 0.022 (95% CL).

---

<p align="center">
  <i>"The universe actualizes its potential through structure formation, and geometry responds."</i>
</p>
