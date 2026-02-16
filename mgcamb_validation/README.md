# MGCAMB Validation — Planck Level-1 (Perturbation Sector)

**3 Planck MCMC chains + 7/7 Boltzmann diagnostic tests — zero free parameters**

Complete MGCAMB (v1.5.2) Planck Level-1 validation of the IAM µ–Σ prediction (µ₀ = −0.13495, Σ = 1) via Cobaya.

> **Important:** This validates the perturbation-level µ–Σ mapping only (Level 1). The full IAM prediction requires background modification (Levels 2/3), which remains to be implemented.

---

## Planck Level-1 MCMC Results (Runs A / B / C)

Three independent chains using full Planck 2018 likelihood (lowl.TT + lowl.EE + highl_plik.TTTEEE_lite_native + lensing.CMBMarged):

| Run | µ₀ | σ₈ | H₀ [km/s/Mpc] | Best χ² | Extra params |
|-----|----|----|---------------|---------|--------------|
| **A: IAM fixed** | −0.135 (fixed) | 0.8014 ± 0.006 | 67.06 ± 0.51 | 984.57 | 0 |
| **B: µ₀ floating** | 0.006 ± 0.156 | 0.8146 ± 0.016 | 67.16 ± 0.51 | 981.24 | 1 |
| **C: ΛCDM baseline** | 0 (fixed) | 0.8139 ± 0.006 | 67.19 ± 0.54 | 983.14 | 0 |

### Three-Way Δχ² Comparison

- **Δχ²(A vs C) = +1.43** — statistically indistinguishable at Planck precision
- **Δχ²(B vs C) = −1.90** — neutral after AIC penalty for one extra parameter
- Planck alone mildly prefers GR (µ₀ ≈ 0), though with insufficient precision to exclude IAM

### Run A: IAM Fixed (µ₀ = −0.13495)

Zero additional free parameters beyond ΛCDM. Tests whether IAM's derived gravity modification is compatible with Planck.

| Parameter | Run A (IAM fixed) | Planck 2018 ΛCDM |
|-----------|--------------------|------------------|
| H₀ [km/s/Mpc] | 67.06 ± 0.51 | 67.36 ± 0.54 |
| σ₈ | 0.8014 ± 0.0057 | 0.8111 ± 0.0060 |
| ω_b | 0.02230 ± 0.00010 | 0.02237 ± 0.00015 |
| ω_c | 0.1206 ± 0.0011 | 0.1200 ± 0.0012 |
| τ | 0.0561 ± 0.0071 | 0.0544 ± 0.0073 |
| n_s | 0.9635 ± 0.0039 | 0.9649 ± 0.0042 |
| A_planck | 1.0008 ± 0.0024 | 1.0000 ± 0.0025 |
| Best χ² | 984.57 | — |

σ₈ shifts downward by 0.013 relative to ΛCDM, in the direction favored by weak lensing surveys, while maintaining comparable likelihood (Δχ² = +1.43). Convergence: R−1 = 0.023, 8,304 accepted samples, 77% acceptance rate.

### Run B: µ₀ Floating

One additional free parameter. Lets Planck freely choose the preferred strength of gravity modification.

| Parameter | Run B (µ₀ floating) | IAM prediction |
|-----------|----------------------|----------------|
| µ₀ | 0.006 ± 0.156 | −0.135 |
| H₀ [km/s/Mpc] | 67.16 ± 0.51 | — |
| σ₈ | 0.8146 ± 0.0157 | — |
| Best χ² | 981.24 | — |

**µ₀ posterior details:** Mean = +0.006, σ = 0.156, MAP ≈ 0, 68% CI = [−0.048, +0.200] (asymmetric — prior wall at +0.2 clips upper tail). IAM's predicted µ₀ = −0.135 lies 0.9σ from the mean — compatible but not confirmed at current precision. Convergence: R−1 = 0.020, 9,184 accepted samples, 75% acceptance rate.

### Run C: ΛCDM Baseline

Standard GR (µ₀ = 0, σ₀ = 0). Provides exact same-pipeline χ² baseline.

| Parameter | Run C (ΛCDM) | Planck 2018 |
|-----------|--------------|-------------|
| H₀ [km/s/Mpc] | 67.19 ± 0.54 | 67.36 ± 0.54 |
| σ₈ | 0.8139 ± 0.0062 | 0.8111 ± 0.0060 |
| Ω_Λ | 0.6823 ± 0.0076 | 0.6847 ± 0.0073 |
| S₈ | 0.8355 ± 0.0133 | 0.832 ± 0.013 |
| Best χ² | 983.14 | — |

Convergence: R−1 = 0.010, 77–80% acceptance rate.

---

## MGCAMB Boltzmann Diagnostic Results (7/7 PASSED)

MGCAMB computed the full Boltzmann evolution with µ₀ = −0.135, Σ = 1 using Planck 2018 best-fit parameters (H₀ = 67.36, Ω_b h² = 0.02237, Ω_c h² = 0.1200, τ = 0.0544, ln(10¹⁰A_s) = 3.044, n_s = 0.9649).

| # | Test | Criterion | Result | Status |
|---|------|-----------|--------|--------|
| 1 | CMB TT (ℓ > 30) | < 1% residual | 0.17% | **PASS** |
| 2 | CMB TT ISW (ℓ < 30) | < cosmic variance | 3.6% (CV ~ 63%) | **PASS** |
| 3 | CMB lensing | < 5% change | +0.30% | **PASS** |
| 4 | σ₈ | In [0.79, 0.82] | 0.7954 | **PASS** |
| 5 | f·σ₈ fit quality | χ² ≤ ΛCDM + 4 | 4.42 vs 4.85 | **PASS** |
| 6 | Σ = 1 preservation | Exact | σ₀ = 0 | **PASS** |
| 7 | P(k) scale-independence | std(ratio) < 1% | 0.53% | **PASS** |

### Physical Interpretation

The σ₈ suppression from 0.812 to 0.795 shifts in the direction favored by weak lensing surveys. KiDS, DES, and HSC measure σ₈ ≈ 0.76–0.80, consistently below ΛCDM's prediction. IAM's µ < 1 naturally produces this suppression by weakening the gravitational source for density perturbations while leaving photon paths (Σ = 1) unaffected.

### MGCAMB Configuration

```
MG_flag = 1          # Modified gravity active
pure_MG_flag = 2     # mu-Sigma parametrization
musigma_par = 1      # Direct mu-Sigma (not mu-gamma)
GRtrans = 0.001      # GR enforced below this scale factor
mu0 = -0.13495       # IAM predicted value
sigma0 = 0.0         # Sigma = 1 exactly
```

---

## Repository Structure

```
mgcamb_validation/
├── README.md                          # This file
├── chains/                            # Raw MCMC chain outputs (Runs A/B/C)
│   └── README.md                      # Chain file inventory & verification
├── yaml_configs/                      # Exact Cobaya YAML files
│   ├── run_a_iam_fixed.yaml
│   ├── run_b_mu0_float.yaml
│   └── run_c_lcdm_baseline.yaml
├── getdist_scripts/                   # Posterior extraction & comparison
│   ├── extract_run_a.py               # Reproduces Table 5 of Technical Note
│   ├── extract_run_b.py               # Reproduces Table 6 & Section 9
│   ├── extract_run_c.py               # Reproduces Table 7
│   └── three_way_comparison.py        # Reproduces Table 8 (Delta-chi2)
├── forecasts/                         # Section 10 of Technical Note
│   ├── README.md
│   ├── iam_fisher_forecast.py
│   ├── iam_isw_prediction.py
│   ├── iam_binned_mu_reconstruction.py
│   └── iam_transition_zone.py
├── iam_mu_sigma.py                    # MGCAMB diagnostic validation (7 tests)
├── mgcamb_reproducibility_package.py  # Full reproducibility package
├── mgcamb_results_table.py            # Results table generator
├── plot_mgcamb_validation.py          # 6-panel validation figure
├── params_MG_IAM.ini                  # MGCAMB parameter file
├── iam_planck_mcmc.yaml               # Combined Planck MCMC config
├── mgcamb_validation_table.tex        # LaTeX results table
├── iam_mgcamb_validation_6panel.pdf   # Validation figure (PDF)
├── iam_mgcamb_validation_6panel.png   # Validation figure (PNG)
└── iam_mgcamb_reproducibility_log.txt # Reproducibility log
```

---

## How to Reproduce

### Verify chain results (requires GetDist)

```bash
pip install getdist
cd getdist_scripts/
python extract_run_a.py
python extract_run_b.py
python extract_run_c.py
python three_way_comparison.py
```

### Re-run MCMC from scratch (requires MGCAMB + Cobaya + Planck data)

```bash
pip install cobaya
cobaya-install planck_2018_lowl.TT planck_2018_lowl.EE \
  planck_2018_highl_plik.TTTEEE_lite_native planck_2018_lensing.CMBMarged

cobaya-run yaml_configs/run_a_iam_fixed.yaml       # ~3-4 hours
cobaya-run yaml_configs/run_b_mu0_float.yaml       # ~3-4 hours
cobaya-run yaml_configs/run_c_lcdm_baseline.yaml   # ~3-4 hours
```

### Run MGCAMB diagnostic tests

```bash
python iam_mu_sigma.py
python plot_mgcamb_validation.py
```

---

## Requirements

- **MGCAMB v1.5.2** — [github.com/sfu-cosmo/MGCAMB](https://github.com/sfu-cosmo/MGCAMB)
- **Cobaya** — [cobaya.readthedocs.io](https://cobaya.readthedocs.io)
- **Planck 2018 likelihoods** — installed via `cobaya-install`
- **GetDist** — `pip install getdist`
- **Python 3.8+** with numpy, scipy, matplotlib

---

## Reference

See the [IAM–CAMB Technical Note: Planck Level-1 Validation](../docs/IAM_CAMB_Technical_Note.pdf) for full documentation.
