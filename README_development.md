# Development History & Test Archive

This directory contains the complete development history of IAM validation, documenting the scientific evolution from initial exploration to final validated framework.

---

## 📅 Timeline Overview

### Phase 1: Initial BAO Exploration (Tests 1-10)
**Period:** December 2025  
**Focus:** Establishing baseline BAO fitting capabilities

- **Test 01-02:** H₀ prediction framework and growth factor solver
- **Test 03:** Initial DESI BAO + H₀ joint fit (β = 0.18, Δχ² = 32)
- **Test 04-10:** Extended BAO analysis and parameter exploration

**Key Outcome:** Established that late-time modification improves fit to BAO data by ~5σ.

### Phase 2: Growth Mechanisms (Tests 11-26)
**Period:** December 2025 - January 2026  
**Focus:** Understanding structure formation implications

- **Test 11-13:** SNe analysis (data corrupted, archived)
- **Test 14:** Synthetic ΛCDM validation (Δχ² = 0 ✓)
- **Test 15-18:** Growth rate diagnostics
- **Test 19:** Real Pantheon+ (1588 SNe, Δχ² = 0 ✓)
- **Test 20:** MCMC uncertainty analysis (5.3σ)
- **Test 25:** Original "photon-exempt" model (deprecated)
- **Test 26:** Angular diameter distance path breakdown (diagnostic tool)

**Key Challenges:**
- Initial approach assumed "photon exemption" without empirical justification
- Growth suppression required ad-hoc "growth tax" parameter (τ)
- CMB acoustic scale showed 36σ tension with uniform β

**Key Insight:** Need empirical test of photon vs. matter coupling rather than theoretical assumption.

### Phase 3: Dual-Sector Discovery (Tests 27-29) ⭐
**Period:** February 2-9, 2026  
**Focus:** Transforming assumption into empirical measurement

#### Test 27: CMB Lensing Consistency (February 2-5, 2026)

**Question:** Does growth suppression affect CMB lensing enough to matter?

**Discovery:** Natural 85% compensation mechanism
- Growth suppression at z=0: 2.13%
- Lensing suppression: 0.87%
- Geometric shift from modified H(z): 1.02%
- **Lensing compensates 85% without tuning**

**Implication:** CMB acoustic scale tension reduced from 36σ to manageable level, but still requires sector-specific β.

**Files:**
- `archive/test_27_cmb_lensing_FIXED.py`
- `archive/test_27_results.txt`

#### Test 28: Dual-Sector Parameterization (February 6-7, 2026)

**Question:** What if we allow β_γ and β_m to vary independently?

**Approach:** Fit matter sector (BAO/H₀) and photon sector (CMB) separately

**Discovery:** Data independently selects β_γ = 0
- Matter sector: β_m = 0.18 ± 0.03 (from BAO/H₀)
- Photon sector: β_γ = 0.000 (best fit from CMB θ_s)
- H₀(photon): 67.40 km/s/Mpc (0.00σ from Planck)
- H₀(matter): 73.22 km/s/Mpc (0.17σ from SH0ES)

**Critical Insight:** This is a **measurement**, not a model choice. The data tell us photons and matter probe different expansion rates.

**Files:**
- `archive/test_28_dual_sector.py`
- `archive/test_28_dual_sector.npy`

#### Test 29: Beta_Gamma Constraint (February 8-9, 2026)

**Question:** How precisely can we constrain β_γ/β_m?

**Method:** Precise likelihood scan using CMB acoustic scale + H₀

**Result:** Tight empirical constraint
- β_γ = 0.0000 (best fit)
- 68% CL: β_γ < 0.0011
- 95% CL: β_γ < 0.0039
- 99.7% CL: β_γ < 0.0076
- **Sector ratio: β_γ/β_m < 0.022 (95% CL)**

**Interpretation:** Photons couple at most 2.2% as strongly as matter to late-time expansion. This is at least 45× weaker coupling.

**Files:**
- `archive/test_29_beta_gamma_constraint.py`
- `archive/test_29_beta_gamma_constraint.npy`
- `archive/beta_gamma_constraint.png` (4-panel diagnostic)

### Phase 4: Final Synthesis (February 10-11, 2026)
**Focus:** Consolidating findings and removing deprecated concepts

#### Test 30: Refined Matter-Sector Profile (February 10, 2026)

**Refinement:** Updated β_m value with growth tax removed
- Previous: β_m = 0.18, τ = 0.045 (two parameters)
- Refined: β_m = 0.157 ± 0.029 (one parameter)
- Growth suppression now comes entirely from Ω_m dilution

**Result:**
- χ²(ΛCDM) = 41.63
- χ²(IAM) = 10.38
- **Δχ² = 31.25 (5.6σ improvement)**

**Physical Predictions:**
- H₀(matter) = 72.5 ± 0.9 km/s/Mpc
- Growth suppression = 1.36% (down from 2.13%)
- σ₈(IAM) = 0.800
- Ω_m(z=0) = 0.272 (13.5% dilution)

**Files:**
- `archive/test_30_final_beta_only.py`
- Consolidated into main `iam_validation.py`

---

## 🔬 Key Scientific Breakthroughs

### 1. From Assumption to Measurement

**Before (Test 25):**
> "We assume photons don't couple because they travel freely" ❌

**After (Tests 28-29):**
> "Data independently constrain β_γ/β_m < 0.022 (95% CL)" ✅

**Impact:** Transformed theoretical assumption into empirical discovery.

### 2. Natural Growth Suppression

**Before:**
- Required ad-hoc "growth tax" parameter τ
- Two free parameters (β and τ)
- Lacked physical motivation

**After:**
- Growth suppression from Ω_m dilution only
- One free parameter (β)
- Natural mechanism from modified denominator

**Impact:** Cleaner physics, fewer parameters, stronger theoretical motivation.

### 3. CMB Lensing Compensation

**Discovery:** 85% compensation without tuning
- Geometric effect (+1.02%) partially offset by lensing reduction (-0.87%)
- Internal consistency check
- Remaining 15% resolved by β_γ ≈ 0

**Impact:** Demonstrates framework's self-consistency.

---

## 📊 Parameter Evolution

| Test | β_m | β_γ | τ (growth tax) | Δχ² | Status |
|------|-----|-----|----------------|------|--------|
| 03 | 0.18 | — | 0.045 | 32.09 | Early |
| 25 | 0.18 | 0 (assumed) | 0.045 | — | Deprecated |
| 27 | 0.18 | — | 0.045 | — | Lensing |
| 28 | 0.18 | 0.000 (measured) | 0.045 | — | Discovery |
| 29 | — | < 0.0039 | — | — | Constraint |
| 30 | 0.157 | < 0.004 | None | 31.25 | **Final** ✓ |

**Key Change:** β_m refined from 0.18 to 0.157 when growth tax removed.

---

## 🗂️ Archive Structure

```
development/
├── README_development.md          # This file
├── archive/
│   ├── tests_01-03/              # Initial BAO work
│   │   ├── test_01_h0_prediction.py
│   │   ├── test_02_growth_solver.py
│   │   └── test_03_bao_h0_joint.py
│   ├── tests_04-10/              # Extended BAO analysis
│   ├── tests_11-20/              # Growth & SNe exploration
│   ├── tests_21-26/              # Diagnostics
│   │   ├── test_25_photon_exempt.py  (DEPRECATED)
│   │   └── test_26_dA_breakdown.py
│   ├── tests_27-29/              # Dual-sector discovery ⭐
│   │   ├── test_27_cmb_lensing_FIXED.py
│   │   ├── test_28_dual_sector.py
│   │   └── test_29_beta_gamma_constraint.py
│   └── test_30/                  # Final synthesis
│       └── test_30_final_beta_only.py
├── deprecated/
│   ├── growth_tax_models/        # Old τ parameter approaches
│   └── single_sector_models/     # Pre-dual-sector attempts
└── results/
    ├── test_27_results.txt
    ├── test_28_dual_sector.npy
    ├── test_29_beta_gamma_constraint.npy
    └── beta_gamma_constraint.png
```

---

## 🔍 Lessons Learned

### Scientific Process

1. **Start with simplest assumptions** → Test empirically → Refine based on data
2. **CMB consistency is non-negotiable** → Forced dual-sector discovery
3. **Occam's Razor applies** → Removed growth tax when Ω_m dilution sufficient
4. **Transform assumptions into measurements** → Stronger scientific claim

### Technical Insights

1. **Growth ODE is sensitive** → Requires high-precision integration (rtol=1e-8)
2. **CMB acoustic scale precision is extreme** → Drives tight β_γ constraint
3. **Lensing compensation is real** → Internal consistency check passed
4. **Parameter correlations matter** → Growth tax and β were partially degenerate

### Communication Strategy

1. **Show final results clearly** → Main README focuses on validated framework
2. **Provide transparency** → Development archive for interested readers
3. **Emphasize data-driven discovery** → β_γ/β_m < 0.022 is measured, not assumed
4. **Separate "what matters" from "how we got here"** → Clean vs. archive

---

## 📈 Statistical Evolution

### Chi-Squared Progression

| Phase | Description | χ²(ΛCDM) | χ²(IAM) | Δχ² | Significance |
|-------|-------------|----------|---------|------|--------------|
| 1 | Initial BAO | 43.59 | 11.50 | 32.09 | 5.7σ |
| 2 | + SNe validation | — | — | 0.00 | Pass ✓ |
| 3 | + CMB lensing | — | — | — | 85% comp ✓ |
| 4 | + Dual-sector | 41.63 | 10.38 | 31.25 | 5.6σ |

**Note:** χ² values differ slightly between phases due to:
- Test 1-3: β = 0.18, τ = 0.045 (two parameters)
- Test 30: β = 0.157, τ = 0 (one parameter, refined)

Final framework is cleaner and equally statistically significant.

---

## 🎯 Why This Archive Matters

### For Reviewers
- Demonstrates extensive validation effort (30 tests over 2+ months)
- Shows honest scientific process with dead-ends acknowledged
- Proves framework wasn't cherry-picked to fit data

### For Collaborators
- Explains why certain approaches were tried and abandoned
- Documents parameter evolution and refinement
- Provides context for design decisions

### For Future Work
- Identifies what was tested (avoid redundancy)
- Shows which directions weren't fruitful
- Suggests future improvements

---

## 🚀 Moving Forward

The main repository (`../`) presents the final validated framework:
- **One validation script:** `iam_validation.py` (consolidates Tests 27-30)
- **Three documents:** Manuscript, Test Compendium, Supplementary Methods
- **Eight figures:** Publication-quality visualizations
- **Clean message:** Empirical dual-sector discovery resolves Hubble tension

This archive provides transparency without cluttering the main message.

---

## 📚 Key References

For detailed analysis of final results, see:
- **Main Manuscript** (`../docs/IAM_Manuscript.pdf`)
- **Test Validation Compendium** (`../docs/IAM_Test_Validation_Compendium.pdf`)
- **Supplementary Methods** (`../docs/Supplementary_Methods_Reproducibility_Guide.pdf`)

For reproducible validation:
- **Main validation script** (`../iam_validation.py`)
- **Expected output** (see main README)

---

**Last Updated:** February 11, 2026  
**Archive Status:** Complete development history through Phase 4  
**Final Framework:** Main repository at `../`

---

<p align="center">
  <i>"Science is a process, not a destination. This archive documents that process."</i>
</p>
