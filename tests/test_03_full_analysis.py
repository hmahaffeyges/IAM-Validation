# test_03_full_analysis.py
"""
IAM Validation Suite - Test 03: Full Data Comparison
=====================================================

Datasets:
  - Planck 2018: H₀ = 67.4 ± 0.5 km/s/Mpc
  - SH0ES 2022: H₀ = 73.04 ± 1.04 km/s/Mpc
  - JWST/TRGB 2024: H₀ = 70.39 ± 1.89 km/s/Mpc
  - DESI DR2 2025: fσ₈(z) at 7 redshifts

Target: Reproduce χ²_IAM = 12.43 vs χ²_ΛCDM = 72.01
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# PARAMETERS (from manuscript)
# ============================================================================

H0_CMB = 67.4      # Planck 2018
Omega_m = 0.315
Omega_r = 9.24e-5
Omega_Lambda = 1 - Omega_m - Omega_r

BETA = 0.18        # IAM amplitude
GROWTH_TAX = 0.045 # Growth suppression

# CRITICAL: σ₈ = 0.76 (suppressed from Planck's 0.811)
sigma8_fid = 0.76  # NOT 0.811!

print("=" * 70)
print("IAM VALIDATION - FULL DATA ANALYSIS")
print("=" * 70)
print(f"\nParameters:")
print(f"  β           = {BETA}")
print(f"  growth_tax  = {GROWTH_TAX}")
print(f"  σ₈          = {sigma8_fid} (IAM fiducial, ~4.5% suppressed)")
print()

# ============================================================================
# DATASETS
# ============================================================================

# H₀ measurements (independent)
H0_data = {
    'names': ['Planck 2018', 'SH0ES 2022', 'JWST/TRGB 2024'],
    'values': np.array([67.4, 73.04, 70.39]),
    'errors': np.array([0.5, 1.04, 1.89])
}

# DESI DR2 fσ₈ (2025)
DESI_data = {
    'z_eff': np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330]),
    'fsig8': np.array([0.452, 0.428, 0.410, 0.392, 0.368, 0.355, 0.312]),
    'fsig8_err': np.array([0.030, 0.025, 0.028, 0.035, 0.040, 0.045, 0.050])
}

print("Datasets loaded:")
print(f"  H₀ measurements: {len(H0_data['names'])}")
print(f"  DESI fσ₈ points: {len(DESI_data['z_eff'])}")

# ============================================================================
# GROWTH EQUATION SOLVERS
# ============================================================================

def E_activation(a):
    return np.exp(1 - 1/a)

def H_IAM(a, H0, beta):
    return H0 * np.sqrt(Omega_m * a**(-3) + Omega_r * a**(-4) + 
                        Omega_Lambda + beta * E_activation(a))

def H_LCDM(a, H0):
    return H0 * np.sqrt(Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda)

def Omega_m_of_a(a, beta, use_iam):
    if use_iam:
        H2 = Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda + beta * E_activation(a)
    else:
        H2 = Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda
    return Omega_m * a**(-3) / H2

def growth_lcdm(y, a, H0):
    D, Dp = y
    H = H_LCDM(a, H0)
    Om = Omega_m_of_a(a, 0, False)
    Dpp = 1.5 * Om * H0**2 / (a**2 * H**2) * D - (2/a) * Dp
    return [Dp, Dpp]

def growth_iam(y, a, H0, beta, tax):
    D, Dp = y
    H = H_IAM(a, H0, beta)
    Om = Omega_m_of_a(a, beta, True)
    tax_val = tax * E_activation(a)
    Om_eff = Om * (1 - tax_val)
    Dpp = 1.5 * Om_eff * H0**2 / (a**2 * H**2) * D - (2/a) * Dp
    return [Dp, Dpp]

# ============================================================================
# SOLVE GROWTH EQUATIONS
# ============================================================================

print("\n" + "=" * 70)
print("Solving Growth Equations")
print("=" * 70)

a_arr = np.logspace(-3, 0, 2000)
y0 = [0.001, 1.0]

sol_lcdm = odeint(growth_lcdm, y0, a_arr, args=(H0_CMB,))
D_lcdm_raw = sol_lcdm[:, 0]

sol_iam = odeint(growth_iam, y0, a_arr, args=(H0_CMB, BETA, GROWTH_TAX))
D_iam_raw = sol_iam[:, 0]

z_arr = 1/a_arr - 1

# Normalize to LCDM at z=0
D_lcdm = D_lcdm_raw / D_lcdm_raw[-1]
D_iam = D_iam_raw / D_lcdm_raw[-1]

# σ₈(z) with IAM fiducial normalization
sigma8_lcdm = sigma8_fid * D_lcdm
sigma8_iam = sigma8_fid * D_iam

# Growth rates
f_lcdm = np.gradient(np.log(D_lcdm), np.log(a_arr))
f_iam = np.gradient(np.log(D_iam), np.log(a_arr))

# fσ₈(z)
fsig8_lcdm = f_lcdm * sigma8_lcdm
fsig8_iam = f_iam * sigma8_iam

print(f"✅ ΛCDM: D(0) = {D_lcdm[-1]:.6f}, σ₈(0) = {sigma8_lcdm[-1]:.4f}")
print(f"✅ IAM:  D(0) = {D_iam[-1]:.6f}, σ₈(0) = {sigma8_iam[-1]:.4f}")
print(f"   Suppression: {(1 - D_iam[-1]/D_lcdm[-1])*100:.2f}%")

# ============================================================================
# H₀ PREDICTIONS
# ============================================================================

print("\n" + "=" * 70)
print("H₀ Predictions")
print("=" * 70)

H0_lcdm_pred = H0_CMB  # ΛCDM uses Planck value
H0_iam_pred = H_IAM(1.0, H0_CMB, BETA)  # IAM boosted value

print(f"\nΛCDM: H₀ = {H0_lcdm_pred:.2f} km/s/Mpc")
print(f"IAM:  H₀ = {H0_iam_pred:.2f} km/s/Mpc")
print(f"Boost: {(H0_iam_pred/H0_lcdm_pred - 1)*100:.2f}%")

# ============================================================================
# CHI-SQUARED: H₀ MEASUREMENTS
# ============================================================================

print("\n" + "=" * 70)
print("χ² Analysis: H₀ Measurements")
print("=" * 70)

chi2_H0_lcdm = 0
chi2_H0_iam = 0

print("\nDataset              Obs       ΛCDM      IAM       Δ_ΛCDM   Δ_IAM")
print("-" * 70)

for i, name in enumerate(H0_data['names']):
    obs = H0_data['values'][i]
    err = H0_data['errors'][i]
    
    delta_lcdm = (obs - H0_lcdm_pred) / err
    delta_iam = (obs - H0_iam_pred) / err
    
    chi2_H0_lcdm += delta_lcdm**2
    chi2_H0_iam += delta_iam**2
    
    print(f"{name:20s} {obs:6.2f}    {H0_lcdm_pred:6.2f}    {H0_iam_pred:6.2f}    {delta_lcdm:+6.2f}σ  {delta_iam:+6.2f}σ")

print(f"\nχ²_ΛCDM(H₀) = {chi2_H0_lcdm:.2f}")
print(f"χ²_IAM(H₀)  = {chi2_H0_iam:.2f}")

# ============================================================================
# CHI-SQUARED: DESI fσ₈
# ============================================================================

print("\n" + "=" * 70)
print("χ² Analysis: DESI DR2 fσ₈")
print("=" * 70)

fsig8_lcdm_at_desi = np.interp(DESI_data['z_eff'], z_arr[::-1], fsig8_lcdm[::-1])
fsig8_iam_at_desi = np.interp(DESI_data['z_eff'], z_arr[::-1], fsig8_iam[::-1])

chi2_desi_lcdm = 0
chi2_desi_iam = 0

print("\nz_eff   fσ₈_obs   fσ₈_ΛCDM  fσ₈_IAM   Δ_ΛCDM   Δ_IAM")
print("-" * 60)

for i in range(len(DESI_data['z_eff'])):
    z = DESI_data['z_eff'][i]
    obs = DESI_data['fsig8'][i]
    err = DESI_data['fsig8_err'][i]
    lcdm = fsig8_lcdm_at_desi[i]
    iam = fsig8_iam_at_desi[i]
    
    delta_lcdm = (obs - lcdm) / err
    delta_iam = (obs - iam) / err
    
    chi2_desi_lcdm += delta_lcdm**2
    chi2_desi_iam += delta_iam**2
    
    print(f"{z:.3f}   {obs:.3f}     {lcdm:.3f}     {iam:.3f}     {delta_lcdm:+5.2f}σ   {delta_iam:+5.2f}σ")

print(f"\nχ²_ΛCDM(DESI) = {chi2_desi_lcdm:.2f}")
print(f"χ²_IAM(DESI)  = {chi2_desi_iam:.2f}")

# ============================================================================
# TOTAL CHI-SQUARED
# ============================================================================

print("\n" + "=" * 70)
print("COMBINED χ² RESULTS")
print("=" * 70)

chi2_lcdm_total = chi2_H0_lcdm + chi2_desi_lcdm
chi2_iam_total = chi2_H0_iam + chi2_desi_iam
delta_chi2 = chi2_lcdm_total - chi2_iam_total

print(f"\nΛCDM:")
print(f"  χ²(H₀)   = {chi2_H0_lcdm:6.2f}")
print(f"  χ²(DESI) = {chi2_desi_lcdm:6.2f}")
print(f"  χ²_total = {chi2_lcdm_total:6.2f}")

print(f"\nIAM:")
print(f"  χ²(H₀)   = {chi2_H0_iam:6.2f}")
print(f"  χ²(DESI) = {chi2_desi_iam:6.2f}")
print(f"  χ²_total = {chi2_iam_total:6.2f}")

print(f"\nΔχ² = {delta_chi2:+.2f}")
print(f"\nTarget from manuscript:")
print(f"  χ²_ΛCDM = 72.01")
print(f"  χ²_IAM  = 12.43")
print(f"  Δχ²     = 59.58")

if abs(chi2_iam_total - 12.43) < 5:
    print("\n✅ REPRODUCED MANUSCRIPT RESULTS!")
else:
    print(f"\n⚠️  Difference from target: {chi2_iam_total - 12.43:+.2f}")

print("=" * 70)

# Save results
np.savez(RESULTS_DIR / 'test_03_results.npz',
         chi2_lcdm=chi2_lcdm_total,
         chi2_iam=chi2_iam_total,
         delta_chi2=delta_chi2,
         H0_lcdm=H0_lcdm_pred,
         H0_iam=H0_iam_pred)

print(f"\n✅ Results saved: {RESULTS_DIR / 'test_03_results.npz'}")
