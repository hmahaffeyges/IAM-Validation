# IAM Validation Repository

## Perturbation-Level µ–Σ Constraints from Planck 2018 and Large-Scale Structure

This repository contains the complete data products, chain files, configuration files, and analysis scripts for twelve MCMC analyses testing a late-time µ(a) < 1, Σ(a) = 1 modification within the standard modified gravity framework.

**Model specification:**
- Background expansion: standard ΛCDM (unmodified)
- Perturbation modification: µ(a) = H²\_ΛCDM / (H²\_ΛCDM + β·E(a)), with E(a) = exp(1 - 1/a)
- Σ(a) = 1 exactly (lensing unmodified)
- Coupling: β = Ω\_m/2 = 0.1575, yielding µ₀ = -0.13495
- Implementation: MGCAMB v1.5.2 + Cobaya + Planck 2018 likelihoods

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FKCZD9-blue)](https://doi.org/10.17605/OSF.IO/KCZD9)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## MCMC Results: Twelve Chains Across Four Dataset Combinations

### Planck Only (Runs A / B / C)

| Run | µ₀ | σ₈ | H₀ [km/s/Mpc] | χ² | Δχ² vs ΛCDM |
|-----|----|----|---------------|-----|-------------|
| **C: ΛCDM** | 0 (fixed) | 0.8139 ± 0.006 | 67.19 ± 0.54 | 983.14 | — |
| **A: IAM fixed** | −0.135 (fixed) | 0.8014 ± 0.006 | 67.06 ± 0.51 | 984.57 | +1.43 |
| **B: µ₀ float** | 0.006 ± 0.156 | 0.8146 ± 0.016 | 67.16 ± 0.51 | 981.24 | −1.90 |

### Planck + RSD (Runs D / E / F)

| Run | µ₀ | σ₈ | H₀ [km/s/Mpc] | χ² | Δχ² vs ΛCDM |
|-----|----|----|---------------|-----|-------------|
| **F: ΛCDM** | 0 (fixed) | 0.8131 ± 0.006 | 67.50 ± 0.41 | 993.86 | — |
| **D: IAM fixed** | −0.135 (fixed) | 0.8001 ± 0.006 | 67.47 ± 0.42 | 995.20 | +1.34 |
| **E: µ₀ float** | 0.033 ± 0.125 | 0.8160 ± 0.013 | 67.54 ± 0.40 | 992.55 | −1.31 |

### Planck + BAO (Runs G / H / I) — In Progress

| Run | µ₀ | σ₈ | H₀ [km/s/Mpc] | χ² | Δχ² vs ΛCDM |
|-----|----|----|---------------|-----|-------------|
| **I: ΛCDM** | 0 (fixed) | — | — | — | — |
| **G: IAM fixed** | −0.135 (fixed) | — | — | — | — |
| **H: µ₀ float** | — | — | — | — | — |

### Planck + Pantheon+ (Runs J / K / L) — In Progress

| Run | µ₀ | σ₈ | H₀ [km/s/Mpc] | χ² | Δχ² vs ΛCDM |
|-----|----|----|---------------|-----|-------------|
| **L: ΛCDM** | 0 (fixed) | — | — | — | — |
| **J: IAM fixed** | −0.135 (fixed) | — | — | — | — |
| **K: µ₀ float** | — | — | — | — | — |

### Combined Summary

| Dataset | Δχ² (fixed) | Δχ² (float) | µ₀ (float) | σ₈ shift |
|---------|-------------|-------------|------------|----------|
| Planck only | +1.43 | −1.90 | 0.006 ± 0.156 | −0.013 |
| Planck + RSD | +1.34 | −1.31 | 0.033 ± 0.125 | −0.013 |
| Planck + BAO | — | — | — | — |
| Planck + Pantheon+ | — | — | — | — |

**Principal result:** A 13.5% suppression of the effective gravitational coupling (µ₀ = −0.135, Σ = 1) is statistically indistinguishable from ΛCDM across all completed dataset combinations. The predicted value lies within 1.3σ of the Planck + RSD posterior peak when µ₀ is allowed to float. The model shifts σ₈ from 0.813 to 0.800 without degrading the CMB fit.

---

## Convergence and Chain Quality

All chains satisfy R−1 < 0.01 with acceptance rates of 69–77%. ΛCDM baseline chains use identical MGCAMB infrastructure (µ₀ = 0, Σ₀ = 0) to ensure differences arise solely from the modified gravity parameter.

| Run | Accepted Samples | R−1 | Acceptance Rate |
|-----|-----------------|-----|-----------------|
| A | ~25,000 | < 0.01 | 77% |
| B | ~28,000 | < 0.01 | 75% |
| C | 17,920 | 0.0089 | 76% |
| D | 25,920 | 0.0078 | 70% |
| E | 28,840 | 0.0084 | 70% |
| F | 26,320 | 0.0087 | 69% |

---

## MGCAMB Boltzmann Diagnostics (7/7 Passed)

| # | Test | Criterion | Result | Status |
|---|------|-----------|--------|--------|
| 1 | CMB TT (ℓ > 30) | < 1% residual | 0.17% | **PASS** |
| 2 | CMB TT ISW (ℓ < 30) | < cosmic variance | 3.6% (CV ~ 63%) | **PASS** |
| 3 | CMB lensing | < 5% change | +0.30% | **PASS** |
| 4 | σ₈ | In [0.79, 0.82] | 0.7954 | **PASS** |
| 5 | f·σ₈ fit quality | χ² ≤ ΛCDM + 4 | 4.42 vs 4.85 | **PASS** |
| 6 | Σ = 1 preservation | Exact | σ₀ = 0 | **PASS** |
| 7 | P(k) scale-independence | std(ratio) < 1% | 0.53% | **PASS** |

---

## Repository Structure

```
IAM-Validation/
├── README.md                              # This file
├── mgcamb_validation/                     # MCMC chains & Boltzmann validation
│   ├── README.md                          # Detailed MGCAMB documentation
│   ├── chains/                            # Raw MCMC chain files (Runs A–L)
│   ├── yaml_configs/                      # Exact Cobaya YAML files
│   ├── getdist_scripts/                   # Posterior extraction & comparison
│   └── forecasts/                         # Euclid/DESI prediction scripts
├── docs/                                  # Technical documentation
├── tests/                                 # Phenomenological validation scripts
├── results/                               # Figures and output files
└── data/                                  # Observational datasets
```

---

## How to Reproduce

### Verify chain results (requires GetDist)

```bash
pip install getdist
cd mgcamb_validation/getdist_scripts/
python three_way_comparison.py
```

### Re-run MCMC from scratch (requires MGCAMB + Cobaya + Planck data)

```bash
pip install cobaya
cobaya-install planck_2018_lowl.TT planck_2018_lowl.EE \
  planck_2018_highl_plik.TTTEEE_lite_native planck_2018_lensing.CMBMarged

cd mgcamb_validation/yaml_configs/
cobaya-run run_a_iam_fixed.yaml       # ~8-16 hours per chain
```

### Run Boltzmann diagnostic tests

```bash
cd mgcamb_validation/
python iam_mu_sigma.py
```

---

## Phenomenological Validation (Pre-Boltzmann)

Prior to the MGCAMB implementation, the µ–Σ prediction was validated against a limited observational dataset (3 H₀ measurements + 7 SDSS/BOSS/eBOSS growth rates). These results are documented below for completeness but are superseded by the full Planck MCMC analysis above.

<details>
<summary>Click to expand phenomenological validation results</summary>

### Observational Validation (10 measurements)

On the limited H₀ + growth rate dataset:
- ΛCDM: χ² = 38.28 (poor fit, driven by H₀ tension)
- IAM: χ² = 8.27 (ΔAIC = 26.0, ΔBIC = 25.4)
- **This strong preference refers to the limited dataset only and does not apply under the full Planck likelihood.**

### Dual-Sector Framework Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| µ(z=0) | 0.864 | Derived (perturbation theory) |
| Σ(z) | 1.000 | Derived |
| β\_m | 0.1575 | Derived (Ω\_m/2, virial theorem) |
| H₀(matter) | 72.51 km/s/Mpc | IAM prediction |
| H₀(photon) | 67.4 km/s/Mpc | Planck CMB |

### Run phenomenological tests

```bash
python iam_validation.py           # 9 observational tests (~1 min)
python iam_derivation_tests.py     # 10 derivation tests (~30 sec)
```

</details>

---

## Documentation

| Document | Description |
|----------|-------------|
| [Main Manuscript](docs/IAM_Manuscript.pdf) | Full holographic motivation, theoretical foundation, statistical validation |
| [Dual-Sector Validation Paper](docs/Dual_Sector_Validation_Paper.pdf) | Pantheon+ SNe sector separation tests, complete Python code in appendices |
| [IAM–CAMB Technical Note](docs/IAM_CAMB_Technical_Note.pdf) | Full Planck Level-1 validation report, MGCAMB diagnostics, MCMC results |
| [Test Validation Compendium](docs/IAM_Test_Validation_Compendium.pdf) | Nine independent validation tests with detailed results |
| [Supplementary Methods](docs/Supplementary_Methods_Reproducibility_Guide.pdf) | Detailed reproducibility guide |
| [IAM Theory Paper](docs/IAM_Theory_Paper.pdf) | Horizon thermodynamics and gravitational decoherence as the origin of µ < 1, Σ = 1 (merges former Holographic and Variational Derivation documents) |

---

## Requirements

- **MGCAMB v1.5.2** — [github.com/sfu-cosmo/MGCAMB](https://github.com/sfu-cosmo/MGCAMB)
- **Cobaya** — [cobaya.readthedocs.io](https://cobaya.readthedocs.io)
- **Planck 2018 likelihoods** — installed via `cobaya-install`
- **GetDist** — `pip install getdist`
- **Python 3.8+** with numpy, scipy, matplotlib

## Citation

```
Mahaffey, H. W. (2026). "Constraints on Late-Time f*sigma_8 Suppression
from mu < 1, Sigma = 1: Planck 2018 and Large-Scale Structure."
```

## License

MIT License. See [LICENSE](LICENSE) for details.
