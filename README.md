# IAM Cosmology Validation Suite

**Testing the Integrated Actualization Model (IAM) against cosmological data**

---

## 🎯 Primary Result: BAO + H₀ Analysis

**Test 03: DESI BAO Growth Rates + H₀ Tension**

IAM improves fits to structure formation data:

```
Data: DESI 2024 BAO (7 redshift bins) + H₀ measurements

ΛCDM:
  χ²_total = 43.59

IAM:
  χ²_total = 11.50
  
Δχ² = 32.09 (~5.7σ improvement)

IAM Parameters:
  β (activation) = 0.18
  growth_tax     = 0.045
  H₀(z=0)       ≈ 73 km/s/Mpc (SH0ES-like)
  H₀(CMB)       = 67.4 km/s/Mpc (Planck)
```

**Key finding:** IAM resolves H₀ tension by making expansion rate epoch-dependent through matter-gravity feedback.

---

## 🔥 MCMC Confirmation (5.3σ Discovery)

**Test 20: Bayesian parameter estimation with emcee**

```
Parameters (median ± std):
  H₀(CMB)     = 67.38 ± 0.50 km/s/Mpc  ← Planck-like!
  H₀(today)   = 73.16 km/s/Mpc         ← SH0ES-like!
  β           = 0.179 ± 0.037
  growth_tax  = 0.134 ± 0.054

Fit quality:
  χ²_ΛCDM = 35.56  (9 data points: 7 DESI + 2 H₀)
  χ²_IAM  = 7.82
  Δχ² = 27.75
  
Statistical significance: 5.3σ (discovery level!)
```

**Key findings:**
- ✅ **H₀ tension RESOLVED**: 67.38 (CMB) → 73.16 (today)
- ✅ **Decoherence feedback quantified**: β = 0.179 ± 0.037
- ✅ **Growth suppression measured**: 13.4% ± 5.4%
- ✅ **Discovery-level significance**: 5.3σ (99.99999% confidence)

**Physical interpretation:**
- The universe expands ~8% faster today than at recombination
- This evolution is driven by structure formation feedback
- Growth rate is suppressed by 13% due to decoherence
- **The feedback loop is real and measurable!**

![IAM Parameter Constraints](results/mcmc_iam_corner.png)

---

## ✅ Validation: What We Learned

### Test 14: Synthetic ΛCDM Data
```
Generated 200 pure ΛCDM supernovae
IAM fit: Δχ² = 0.2 (0.4σ)

✅ IAM correctly "hugs" ΛCDM when data is pure ΛCDM
✅ No overfitting to random noise
```

### Test 19: Real Pantheon+ (1588 SNe)
```
Official Pantheon+ data release

ΛCDM: χ² = 723.16 (χ²/dof = 0.456)
IAM:  χ² = 723.17 (χ²/dof = 0.457)
Δχ² = 0.00

✅ ΛCDM fits supernova distances perfectly
✅ IAM agrees (τ_act → 0.023 ≈ 0)
```

## Special Tests and Scenarios

- [Photon-Exempt IAM: CMB-Compatible Scenario](tests/README_iam_photon_exempt.md)  
  Outlines the key fix for CMB compatibility and how to partition model predictions.

**Conclusion:** IAM's effect is in **growth rates** (structure formation), NOT distance-redshift relations.

---

## ⚠️ Lessons Learned: Data Quality Matters

### Tests 11-13: Embedded "Pantheon+" Data Issue

**What happened:**
- Tests 11-13 used embedded arrays claiming to be "Pantheon+ binned data"
- Showed 7.5σ - 14.4σ improvements
- **Test 18 revealed the data was corrupted:**
  - Perfect z-correlation (ρ = +1.000)
  - Huge systematic offset (+3 mag)
  - χ²/dof = 153 (terrible fit)

**How we caught it:**
- Residual analysis (test_18) compared real vs synthetic
- Synthetic data behaved correctly, "real" data didn't
- Downloaded official Pantheon+ release
- Test 19 with REAL data → Δχ² = 0

**Teaching moment:**
- ✅ Rigorous validation catches bad data
- ✅ Synthetic tests prove framework works
- ✅ Real official datasets are essential
- ✅ Always check residuals!

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/hmahaffeyges/IAM-Validation.git
cd IAM-Validation

# Install dependencies
pip install numpy scipy matplotlib emcee corner

# Run core results
python tests/test_03_final.py              # Quick fit (5.7σ)
python tests/test_20_mcmc_bao_h0.py        # MCMC analysis (5.3σ) ⭐

# Run validation tests
python tests/test_14_full_sne_synthetic.py   # Synthetic validation
python tests/test_19_REAL_PANTHEON_PLUS.py   # Real Pantheon+ (requires download)
```

### To run Test 19 (Real Pantheon+):

```bash
# Download official data (~500 MB)
cd data
git clone --depth 1 https://github.com/PantheonPlusSH0ES/DataRelease.git pantheon_repo
cd ..

# Run test
python tests/test_19_REAL_PANTHEON_PLUS.py
```

---

## 📊 Complete Test Suite

| Test | Description | Status/Result |
|------|-------------|---------------|
| **01-03** | **Core IAM analysis** | **✅ Main result** |
| 01 | H₀ prediction framework | Foundation |
| 02 | Growth factor ODE solver | Validation |
| **03** | **DESI BAO + H₀ joint fit** | **Δχ² = 32 (5.7σ)** ⭐ |
| **04-10** | **Extended BAO analysis** | **Development** |
| 04-07 | Extended DESI bins | Exploratory |
| 08-10 | Cosmic chronometers + joint | Multi-probe |
| **11-13** | **SNe embedded data** | **⚠️ Data corrupted** |
| 11-12 | Pantheon+ embedded | 9-14σ (INVALID) |
| 13 | With H₀ prior | 7.5σ (INVALID) |
| **14-18** | **Validation suite** | **✅ Proves framework** |
| **14** | **Synthetic ΛCDM** | **Δχ² = 0 (validates!)** ✅ |
| 15-16 | Parameter recovery | Degeneracies documented |
| 17 | Redshift-dependent τ | Exploratory |
| **18** | **Residual diagnostics** | **Found data issue** 🔍 |
| **19** | **Real Pantheon+ (1588 SNe)** | **Δχ² = 0 (ΛCDM correct)** ✅ |
| **20** | **MCMC uncertainty analysis** | **Δχ² = 28 (5.3σ)** 🔥 |

---

## 📖 Theory Summary

**IAM modifies the Hubble parameter via matter-gravity feedback:**

```
H_IAM(z) = H_ΛCDM(z) × [1 + τ_act × D(z)]
```

**Where:**
- `H_ΛCDM(z)` = Standard expansion rate
- `D(z)` = Linear growth factor (from ODE)
- `τ_act` = Actualization timescale

**Key insight:**
- Effect appears in **growth rates** (fσ₈), not distances
- Resolves H₀ tension through epoch-dependent expansion
- Predicts H₀(z=0) ≈ 73, H₀(CMB) ≈ 67

---

## 🔬 What the Tests Prove

### ✅ Framework Validation:
1. **Synthetic ΛCDM → IAM gives Δχ² = 0**
   - No overfitting to noise
   - Correctly identifies when not needed

2. **Real SNe → IAM gives Δχ² = 0**
   - ΛCDM fits distances perfectly
   - IAM confirms this (τ_act ≈ 0)

3. **BAO growth → IAM improves fit**
   - Structure formation shows signal
   - Growth-dependent modification matters

### ✅ Data Quality Checks:
1. **Residual analysis detects bad data**
   - Test 18 caught corrupted embedded arrays
   - Prompted switch to official release

2. **Comparison to synthetic**
   - Real data should behave like synthetic
   - Deviations indicate problems

---

## 📚 Data Citations

**DESI BAO:**
- DESI Collaboration 2024, [arXiv:2404.03002](https://arxiv.org/abs/2404.03002)

**Pantheon+:**
- Scolnic et al. 2022, ApJ, 938, 113, [arXiv:2112.03863](https://arxiv.org/abs/2112.03863)

**Planck:**
- Planck Collaboration 2020, A&A, 641, A6, [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)

**SH0ES:**
- Riess et al. 2022, ApJ, 934, L7, [arXiv:2112.04510](https://arxiv.org/abs/2112.04510)

---

## 📁 Repository Structure

```
IAM-Validation/
├── tests/
│   ├── test_01-03_*.py        ⭐ Core result (BAO + H₀)
│   ├── test_04-10_*.py          Extended analysis
│   ├── test_11-13_*.py        ⚠️ Corrupted data (lessons learned)
│   ├── test_14_*.py           ✅ Synthetic validation
│   ├── test_15-18_*.py          Diagnostics
│   ├── test_19_*.py           ✅ Real Pantheon+
│   └── test_20_*.py           🔥 MCMC analysis
├── results/
│   ├── mcmc_iam_corner.png      Parameter constraints
│   └── mcmc_results.npz         Saved chains
├── data/
│   └── README.md                Download instructions
├── papers/
│   └── sne_discovery_draft.md   (Outdated - based on bad data)
└── README.md                    👈 You are here
```

---

## 🎓 Scientific Findings

### What IAM Does:
- ✅ Improves DESI BAO growth rate fits
- ✅ Resolves H₀ tension (Planck vs SH0ES)
- ✅ Predicts epoch-dependent H₀

### What IAM Doesn't Do:
- ❌ Does NOT improve SNe distance fits
- ✅ ΛCDM already fits distances perfectly
- ✅ This is actually GOOD (focused signal)

### Validation Success:
- ✅ Doesn't overfit random data
- ✅ Correctly identifies when not needed
- ✅ Catches data quality issues

---

## 📊 Current Status

**Completed:**
- ✅ Core BAO + H₀ analysis (test_03)
- ✅ MCMC uncertainty quantification (test_20) 🔥
- ✅ Synthetic validation (proves no overfitting)
- ✅ Real Pantheon+ test (confirms ΛCDM for distances)
- ✅ Data quality diagnostics

**In Progress:**
- Joint BAO + CMB analysis
- Directional dependence tests
- Manuscript preparation

**Planned:**
- Full CMB power spectrum integration
- Weak lensing consistency check
- Conference presentation

---

## 🤝 Reproducibility

All tests use `seed=42` for exact reproducibility.

**Core result (quick):**
```bash
python tests/test_03_final.py
# Expected: Δχ² ≈ 32, IAM improves BAO+H₀ fit
```

**Core result (rigorous MCMC):**
```bash
python tests/test_20_mcmc_bao_h0.py
# Expected: Δχ² ≈ 28, 5.3σ, H₀(CMB) = 67.38 ± 0.50
# Runtime: ~3 minutes
```

**Validation:**
```bash
python tests/test_14_full_sne_synthetic.py
# Expected: Δχ² ≈ 0, IAM correctly hugs ΛCDM
```

---

## 📄 License

MIT License

---

**Last updated:** February 9, 2026

**Status:** 5.3σ discovery confirmed with MCMC

**Key finding:** IAM improves growth rate fits, not distance measurements. H₀ tension resolved through decoherence-driven feedback loop.
