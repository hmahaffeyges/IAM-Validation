# IAM: Dual-Sector Cosmology from Structure-Driven Expansion

**TL;DR:** We introduce a simple, reproducible cosmological model (Informational Actualization Model, IAM) that empirically resolves the Hubble tension by showing that matter and photon observables probe different late-time cosmic expansion rates. The model’s key prediction maps directly to μ(a) < 1, Σ(a) = 1 in standard modified gravity codes (MGCAMB, EFTCAMB), is validated at 5.5σ significance against all current data (Planck, SH0ES, Pantheon+), and can be independently reproduced by running the code in this repository in under 2 minutes.

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Dual-Sector Cosmology from Structure-Driven Expansion: The Informational Actualization Model (IAM)**

**Key Finding:** 5.5 sigma empirical evidence for dual-sector cosmology resolving the Hubble tension through sector-specific late-time expansion rates.

---

## Core Results

| Parameter | Value | Method | Description |
|-----------|-------|--------|-------------|
| **B_m** | 0.157 +/- 0.029 | MCMC (68% CL) | Matter-sector coupling |
| **B_gamma** | < 1.4 x 10^-6 | MCMC (95% CL) | Photon-sector coupling |
| **B_gamma/B_m** | < 8.5 x 10^-6 | MCMC (95% CL) | Empirical sector ratio |
| **H0(photon)** | 67.4 km/s/Mpc | Planck CMB | Photon-sector measurement |
| **H0(matter)** | 72.5 +/- 1.0 km/s/Mpc | IAM prediction | Matter-sector prediction |
| **Delta-chi2** | 30.01 (5.5 sigma) | vs. LCDM | Statistical improvement |
| **Delta-AIC** | 26.0 | Model selection | No overfitting |
| **Delta-BIC** | 25.4 | Model selection | Strong preference |

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
python iam_validation.py
```

**Expected runtime:** ~1 minute on standard laptop (generates 9 figures)

**Expected output: