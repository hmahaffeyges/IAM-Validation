# IAM Validation Suite

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**5.7σ evidence for IAM over ΛCDM**

## 🏆 Results

| Metric | ΛCDM | IAM | Improvement |
|--------|------|-----|-------------|
| χ² (total) | 43.59 | 11.50 | **Δχ² = +32.09** |
| H₀ prediction | 67.4 km/s/Mpc | 73.22 km/s/Mpc | Matches SH0ES (73.04 ± 1.04) |
| Significance | — | — | **5.7σ** |

✅ Resolves Hubble Tension  
✅ Fits DESI growth data (Δχ² = +2.44)  
✅ Physical mechanism via holographic encoding  

---

## 🚀 Quick Start

```bash
git clone https://github.com/hmahaffeyges/IAM-Validation.git
cd IAM-Validation
pip install numpy scipy matplotlib astropy
python tests/test_03_final.py

## Running Validation Tests

### ⚡ Ultra-Fast Validation (<2 seconds)

Unlike traditional cosmology codes that require minutes to hours, IAM's validation runs in **under 2 seconds** on standard hardware. This enables rapid iteration, extensive parameter exploration, and real-time model testing.

Run the complete validation suite with one command:

```bash
./run_tests.sh

📊 Test Results

Test 1: Hubble Constant

H₀,IAM = 73.22 km/s/Mpc
Matches SH0ES (73.04 ± 1.04)
Status: ✅ PASS
Test 2: Growth Factor

Δχ² = +2.44 vs ΛCDM
Status: ✅ PASS
Test 3: Combined Fit

χ²_ΛCDM = 43.59
χ²_IAM = 11.50
Significance: 5.7σ
Status: ✅ PASS

Quick Validation (60 seconds)

Run the complete validation suite with one command:

```bash
./run_tests.sh
```

Or run the test directly:

```bash
python tests/test_03_final.py
```

### What the Test Does

The validation test compares IAM and ΛCDM across:

1. **H₀ predictions** - Tests against SH0ES and JWST measurements
2. **Growth rate (fσ₈)** - Tests against DESI measurements
3. **Combined statistical fit** - Overall model comparison

**Expected output:**
- ✅ All tests pass
- Δχ² ≈ 30-60 (depending on datasets included)
- Statistical significance: >5σ
- Runtime: <60 seconds

### Test Results

After running, you should see:

```
✅ IAM FITS SIGNIFICANTLY BETTER
   Δχ² = +32.09
   Statistical significance: 5.7σ
   → STRONG EVIDENCE for IAM over ΛCDM
```

Results are saved to `results/validation_results.npz` for further analysis.

### Testing Robustness

To test IAM's robustness yourself:

**Modify parameters:**
Edit `tests/test_03_final.py` and change values in the parameters section:
```python
beta = 0.18  # Try different values
growth_tax = 0.045  # Try different values
```

**Test on data subsets:**
Comment out specific datasets in the test file to see how IAM performs on different combinations.

**Compare to your own models:**
Import the IAM implementation and compare to your preferred cosmological model:
```python
from src.iam_cosmology import solve_growth_iam, compute_fsigma8_iam
# Use alongside your ΛCDM or alternative model
```

 We Welcome Critical Testing

If you find issues or have questions:
- Open a GitHub issue
- Email: hmaffeyges@gmail.com

We actively encourage attempts to falsify these results. Rigorous scrutiny strengthens science.

🔬 What is IAM?

The Informational Actualization Model links cosmic expansion to information encoding on the apparent horizon.

Key equation:

Code
H²(z) = H²_ΛCDM(z) + β · H(z) · D(z)² · f(z)
Where:

D(z) = linear growth factor
β = 0.18 (informational amplitude)
📁 Repository Structure

Code
IAM-Validation/
├── tests/
│   ├── test_01_H0_prediction.py
│   ├── test_02_growth_factor.py
│   └── test_03_final.py          ⭐ RUN THIS
├── results/
│   └── *.npz
└── README.md

📄 Published Preprints

Latest Version (February 2026):**
- OSF Preprints: [DOI: 10.17605/OSF.IO/KCZD9](https://doi.org/10.17605/OSF.IO/KCZD9)
- Direct Link**: [https://osf.io/kczd9](https://osf.io/kczd9)

Original Version (December 2025):
- viXra: [2512.0029](https://vixra.org/abs/2512.0029)

📚 How to Cite

bibtex
@misc{mahaffey2026iam,
  author = {Mahaffey, Heath W.},
  title = {Holographic Black-Hole Cosmology: Resolving the Hubble Tension via Information-Driven Expansion},
  year = {2026},
  publisher = {OSF Preprints},
  doi = {10.17605/OSF.IO/KCZD9},
  url = {https://doi.org/10.17605/OSF.IO/KCZD9},
  note = {Original version: viXra:2512.0029 (2024)}
}
📧 Contact

Heath W. Mahaffey
📧 hmaffeyges@gmail.com
🔗 @hmahaffeyges

🔄 Revision History

v2.0 (February 2026) - Current Version

Refined statistical methodology (χ² replacing AIC)
Updated with DESI DR2 data
Added reproducible validation code
Enhanced significance: Δχ² = 32.09 (5.7σ)

v1.0 (December 2025) - viXra:2512.0029

Initial IAM framework

📄 License

MIT License

"In science, reproducibility is everything. Run the tests yourself."

⭐ Star this repo if it helps your research!
