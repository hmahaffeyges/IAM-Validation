# IAM Full Boltzmann Validation via MGCAMB

**Informational Actualization Model: Full Boltzmann solver validation with zero free parameters**

[![Tests](https://img.shields.io/badge/validation_tests-7%2F7_passed-brightgreen)]()
[![Parameters](https://img.shields.io/badge/free_parameters-zero-blue)]()
[![sigma8](https://img.shields.io/badge/%CF%83%E2%82%88-0.7946-orange)]()

---

## Overview

This repository contains the complete, self-contained reproducibility package for validating the **Informational Actualization Model (IAM)** through the [MGCAMB](https://github.com/sfu-cosmo/MGCAMB) modified Boltzmann solver.

IAM proposes that dark energy arises from gravitational decoherence, modifying the Friedmann equation through an informational energy density:

$$H^2(a) = H^2_{\Lambda\text{CDM}}(a) + \beta_m \cdot \mathcal{E}(a) \cdot \frac{8\pi G}{3}$$

where $\mathcal{E}(a) = e^{1-1/a}$ is the activation function and $\beta_m = \Omega_m/2$ is derived from the virial theorem with **zero free parameters**.

This produces a unique observational signature in the gravitational coupling functions:

| Function | IAM Prediction | f(R) | Horndeski | w₀wₐCDM |
|----------|---------------|------|-----------|---------|
| μ(z=0)   | **0.865** (< 1) | > 1  | varies    | = 1     |
| Σ(z)     | **= 1 (exact)** | > 1  | ≠ 1       | = 1     |

**No other proposed model occupies the (μ < 1, Σ = 1) region.**

---

## Quick Start

### Prerequisites

```bash
# Clone MGCAMB
git clone --recursive https://github.com/sfu-cosmo/MGCAMB.git
cd MGCAMB

# Install (builds Fortran + Python wrapper)
pip install -e .

# Verify
python -c "import camb; print(camb.__version__)"
# Should print: 1.5.2
```

### Run the Full Validation

```bash
# Copy IAM scripts into MGCAMB directory
cp iam_scripts/*.py MGCAMB/

# Run the complete reproducibility package
cd MGCAMB
python mgcamb_reproducibility_package.py 2>&1 | tee iam_mgcamb_reproducibility_log.txt
```

**That's it.** One script, one command, full validation. Every parameter, every intermediate value, every pass/fail test is printed and logged.

### Generate the 6-Panel Figure

```bash
python plot_mgcamb_validation.py
# Output: iam_mgcamb_validation_6panel.pdf/.png
```

### Generate the LaTeX Table

```bash
python mgcamb_results_table.py
# Output: mgcamb_validation_table.tex
```

---

## Results Summary

All results computed with **Planck 2018 baseline** cosmological parameters and **zero additional free parameters**.

### Validation Tests: 7/7 PASSED

| Test | Criterion | Result | Status |
|------|-----------|--------|--------|
| σ₈ | In [0.79, 0.82] | 0.7946 | ✅ PASS |
| CMB TT (ℓ > 30) | < 1% residual | 0.165% | ✅ PASS |
| CMB TT ISW (ℓ < 30) | < cosmic variance | 3.6% (CV ~ 63%) | ✅ PASS |
| CMB lensing | < 5% change | +0.30% | ✅ PASS |
| Σ = 1 | Exact | σ₀ = 0 by construction | ✅ PASS |
| P(k) scale-independence | std(ratio) < 1% | 0.53% | ✅ PASS |
| f·σ₈ fit quality | χ² ≤ ΛCDM + 4 | 4.42 vs 4.85 | ✅ PASS |

### Key Numbers

```
sigma8:    0.8082 (LCDM) → 0.7946 (IAM)  [-1.7%]
S8:        0.8243 (LCDM) → 0.8104 (IAM)  [eases S8 tension]
CMB TT:    Sub-percent above ell=30 (max 0.165%)
Lensing:   +0.30% change (Sigma = 1 exact)
P(k):      0.6% uniform suppression across all scales
f·sigma8:  chi2 = 4.42 (IAM) vs 4.85 (LCDM)
```

### Growth Rate Suppression

IAM suppresses structure growth at low redshift, exactly as predicted by μ < 1:

| z | f·σ₈ (ΛCDM) | f·σ₈ (IAM) | Change |
|---|-------------|------------|--------|
| 2.00 | 0.3246 | 0.3228 | -0.55% |
| 1.00 | 0.4311 | 0.4241 | -1.62% |
| 0.61 | 0.4680 | 0.4554 | -2.69% |
| 0.51 | 0.4731 | 0.4586 | -3.08% |
| 0.38 | 0.4748 | 0.4573 | -3.69% |
| 0.00 | 0.4255 | 0.3994 | -6.14% |

---

## Model Parameters

### Planck 2018 Baseline (Fixed)

| Parameter | Value | Source |
|-----------|-------|--------|
| H₀ | 67.4 km/s/Mpc | Planck 2018 |
| Ω_b h² | 0.02242 | Planck 2018 |
| Ω_c h² | 0.11933 | Planck 2018 |
| τ | 0.0544 | Planck 2018 |
| A_s | 2.1 × 10⁻⁹ | Planck 2018 |
| n_s | 0.9649 | Planck 2018 |

### IAM Parameters (Derived, Zero Free Parameters)

| Parameter | Value | Derivation |
|-----------|-------|------------|
| β_m | Ω_m/2 = 0.1560 | Virial theorem |
| μ₀ | -0.1350 | = μ(z=0) - 1 |
| σ₀ | 0.0 | Σ = 1 exact |

### MGCAMB Settings

```
MG_flag      = 1    (pure MG mode)
pure_MG_flag = 2    (mu-Sigma parametrization)
musigma_par  = 1    
GRtrans      = 0.001
mu0          = -0.134950
sigma0       = 0.0
```

---

## μ(z) Table

The full gravitational coupling function at key redshifts:

| z | a | μ(z) | Σ(z) | E(a) | w_info |
|---|---|------|------|------|--------|
| 0.00 | 1.000 | 0.8651 | 1.0 | 1.000 | -1.333 |
| 0.10 | 0.909 | 0.8866 | 1.0 | 0.905 | -1.367 |
| 0.20 | 0.833 | 0.9057 | 1.0 | 0.819 | -1.400 |
| 0.50 | 0.667 | 0.9485 | 1.0 | 0.607 | -1.500 |
| 1.00 | 0.500 | 0.9823 | 1.0 | 0.368 | -1.667 |
| 2.00 | 0.333 | 0.9977 | 1.0 | 0.135 | -2.000 |
| 5.00 | 0.167 | 1.0000 | 1.0 | 0.007 | -3.000 |
| 1089 | 0.001 | 1.0000 | 1.0 | 0.000 | -364.3 |

---

## Repository Structure

```
├── README.md                              ← You are here
├── iam_scripts/
│   ├── mgcamb_reproducibility_package.py  ← MAIN: Full validation (run this)
│   ├── plot_mgcamb_validation.py          ← 6-panel validation figure
│   ├── mgcamb_results_table.py            ← LaTeX table generator
│   └── iam_mu_sigma.py                    ← mu/Sigma table generator
├── mgcamb_config/
│   └── params_MG_IAM.ini                  ← MGCAMB parameter file
├── results/
│   ├── iam_mgcamb_reproducibility_log.txt ← Pre-generated full log
│   ├── iam_mgcamb_validation_6panel.pdf   ← Pre-generated figure
│   ├── iam_mgcamb_validation_6panel.png   ← Pre-generated figure
│   ├── mgcamb_validation_table.tex        ← Pre-generated LaTeX table
│   ├── mgcamb_validation_summary.txt      ← Summary text
│   └── iam_mgcamb_full_results.npz        ← All numerical data (numpy)
└── paper/
    └── (manuscript files)
```

---

## Falsifiable Predictions

IAM makes six specific predictions testable with upcoming surveys:

1. **Distance-growth tension in w₀wₐ fits** — Best-fit (w₀, wₐ) from BAO distances alone will differ from growth data (DESI Year 5)
2. **μ < 1, Σ = 1 detection** — Euclid should detect at 2–4σ; combined Euclid + DESI + CMB-S4 at ~5σ
3. **No phantom crossing** — Apparent w = -1 crossing is artifact of wrong model class
4. **Scale-independent growth modification** — P(k) suppression uniform across all k
5. **GW standard sirens** — H₀(sirens) ≈ 72.5 km/s/Mpc (testable with LIGO O5, ~2027)
6. **S₈ tension resolution** — S₈ reduced from 0.824 to 0.810

---

## Citation

If you use this code or results, please cite:

```bibtex
@article{Mahaffey2026_IAM_MGCAMB,
  title={Full Boltzmann Validation of the Informational Actualization Model via MGCAMB},
  author={Mahaffey, Heath},
  year={2026},
  note={GitHub: [repository URL]}
}
```

---

## Software

- **MGCAMB**: [github.com/sfu-cosmo/MGCAMB](https://github.com/sfu-cosmo/MGCAMB) (Zhuangfei Wang, Levon Pogosian, et al.)
- **Python** ≥ 3.10
- **NumPy** ≥ 1.26
- **SciPy** ≥ 1.10
- **Matplotlib** ≥ 3.7

---

## License

MIT
