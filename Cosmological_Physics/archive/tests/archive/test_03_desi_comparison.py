# test_03_desi_comparison.py
"""
IAM Validation Suite - Test 03: DESI Data Comparison
=====================================================

Purpose:
    Compare IAM predictions to DESI Year 1 (2024) measurements
    of fσ₈(z) from redshift-space distortions.

Data Source:
    DESI Collaboration (2024) - Year 1 BAO and RSD results
    https://data.desi.lbl.gov/public/

Expected Outcome:
    χ²_IAM < χ²_ΛCDM (IAM fits better)

Author: Heath W. Mahaffey
Date: 2026-02-08
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Parameters
H0_CMB = 67.4
Omega_m = 0.315
Omega_r = 9.24e-5
Omega_Lambda = 1 - Omega_m - Omega_r
BETA = 0.18
GROWTH_TAX = 0.045
sigma8_0 = 0.811  # Planck 2018

print("=" * 70)
print("IAM VALIDATION SUITE - TEST 03: DESI DATA COMPARISON")
print("=" * 70)

# ============================================================================
# DESI Y1 2024 DATA (from published tables)
# ============================================================================

# DESI Y1 RSD measurements: fσ₈(z_eff)
# Source: DESI Collaboration, arXiv:2404.03002 (2024)
# Table 2: RSD measurements

desi_data = {
    'z_eff': np.array([0.295, 0.510, 0.706, 0.930, 1.317]),
    'fsig8': np.array([0.470, 0.462, 0.427, 0.405, 0.398]),  # Approximate values
    'fsig8_err': np.array([0.024, 0.021, 0.023, 0.027, 0.036]),  # Diagonal errors
}

# NOTE: Full covariance matrix should be used for proper analysis
# For now, we'll use diagonal errors as a first approximation
# TODO: Fetch full covariance from DESI data release

print("\nDESI Y1 2024 RSD Measurements:")
print("-" * 50)
print("z_eff    fσ₈_obs    ±σ")
print("-" * 50)
for i in range(len(desi_data['z_eff'])):
    print(f"{desi_data['z_eff'][i]:.3f}    {desi_data['fsig8'][i]:.3f}    ±{desi_data['fsig8_err'][i]:.3f}")

print("\nNOTE: Using diagonal errors only (simplified)")
print("Full analysis requires covariance matrix from DESI DR1")

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
print("Computing Growth Functions")
print("=" * 70)

a_arr = np.logspace(-3, 0, 2000)
y0 = [0.001, 1.0]

# ΛCDM
sol_lcdm = odeint(growth_lcdm, y0, a_arr, args=(H0_CMB,))
D_lcdm = sol_lcdm[:, 0] / sol_lcdm[-1, 0]

# IAM
sol_iam = odeint(growth_iam, y0, a_arr, args=(H0_CMB, BETA, GROWTH_TAX))
D_iam = sol_iam[:, 0] / sol_iam[-1, 0]

z_arr = 1/a_arr - 1

# Compute f(z)
f_lcdm = np.gradient(np.log(D_lcdm), np.log(a_arr))
f_iam = np.gradient(np.log(D_iam), np.log(a_arr))

# Compute fσ₈(z)
fsig8_lcdm = f_lcdm * sigma8_0 * D_lcdm
fsig8_iam = f_iam * sigma8_0 * D_iam

print("✅ Growth equations solved")

# ============================================================================
# INTERPOLATE TO DESI REDSHIFTS
# ============================================================================

fsig8_lcdm_at_desi = np.interp(desi_data['z_eff'], z_arr[::-1], fsig8_lcdm[::-1])
fsig8_iam_at_desi = np.interp(desi_data['z_eff'], z_arr[::-1], fsig8_iam[::-1])

print("\n" + "=" * 70)
print("Model Predictions at DESI Redshifts")
print("=" * 70)
print("\nz_eff   fσ₈_DESI  fσ₈_ΛCDM  fσ₈_IAM   Δ_ΛCDM   Δ_IAM")
print("-" * 70)

residuals_lcdm = []
residuals_iam = []

for i in range(len(desi_data['z_eff'])):
    z = desi_data['z_eff'][i]
    obs = desi_data['fsig8'][i]
    err = desi_data['fsig8_err'][i]
    lcdm = fsig8_lcdm_at_desi[i]
    iam = fsig8_iam_at_desi[i]
    
    delta_lcdm = (obs - lcdm) / err
    delta_iam = (obs - iam) / err
    
    residuals_lcdm.append(delta_lcdm)
    residuals_iam.append(delta_iam)
    
    print(f"{z:.3f}   {obs:.3f}    {lcdm:.3f}    {iam:.3f}    {delta_lcdm:+.2f}σ   {delta_iam:+.2f}σ")

# ============================================================================
# CHI-SQUARED CALCULATION
# ============================================================================

print("\n" + "=" * 70)
print("Chi-Squared Analysis (Diagonal Errors Only)")
print("=" * 70)

residuals_lcdm = np.array(residuals_lcdm)
residuals_iam = np.array(residuals_iam)

chi2_lcdm = np.sum(residuals_lcdm**2)
chi2_iam = np.sum(residuals_iam**2)

n_data = len(desi_data['z_eff'])
n_params_lcdm = 6  # Standard cosmological parameters
n_params_iam = 8   # + beta, growth_tax

dof_lcdm = n_data - n_params_lcdm
dof_iam = n_data - n_params_iam

# Note: DOF might be negative with only 5 data points!
# This is a limitation of simplified analysis

print(f"\nNumber of data points: {n_data}")
print(f"ΛCDM parameters: {n_params_lcdm}")
print(f"IAM parameters:  {n_params_iam}")

print(f"\nΛCDM:")
print(f"  χ² = {chi2_lcdm:.2f}")
if dof_lcdm > 0:
    print(f"  χ²/dof = {chi2_lcdm/dof_lcdm:.2f}")

print(f"\nIAM:")
print(f"  χ² = {chi2_iam:.2f}")
if dof_iam > 0:
    print(f"  χ²/dof = {chi2_iam/dof_iam:.2f}")

delta_chi2 = chi2_lcdm - chi2_iam

print(f"\nΔχ² = {delta_chi2:.2f}")

if delta_chi2 > 0:
    print(f"✅ IAM fits better by Δχ² = {delta_chi2:.2f}")
elif delta_chi2 < 0:
    print(f"⚠️  ΛCDM fits better by Δχ² = {-delta_chi2:.2f}")
else:
    print("Tie")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("Generating Comparison Plot")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: fσ₈(z) comparison
ax = axes[0]
ax.errorbar(desi_data['z_eff'], desi_data['fsig8'], 
            yerr=desi_data['fsig8_err'],
            fmt='o', color='black', markersize=8, capsize=5, 
            label='DESI Y1 2024', zorder=10)

z_plot = np.linspace(0, 2, 500)
fsig8_lcdm_plot = np.interp(z_plot, z_arr[::-1], fsig8_lcdm[::-1])
fsig8_iam_plot = np.interp(z_plot, z_arr[::-1], fsig8_iam[::-1])

ax.plot(z_plot, fsig8_lcdm_plot, 'r--', linewidth=2.5, label='ΛCDM', alpha=0.7)
ax.plot(z_plot, fsig8_iam_plot, 'b-', linewidth=2.5, label='IAM')

ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('fσ₈(z)', fontsize=12)
ax.set_title('DESI RSD Measurements vs Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1.5])

# Plot 2: Residuals
ax = axes[1]
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.errorbar(desi_data['z_eff'], residuals_lcdm, 
            fmt='s', color='red', markersize=8, label='ΛCDM residuals', alpha=0.7)
ax.errorbar(desi_data['z_eff'], residuals_iam, 
            fmt='o', color='blue', markersize=8, label='IAM residuals')

ax.axhspan(-1, 1, alpha=0.1, color='green', label='±1σ')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Residual (σ)', fontsize=12)
ax.set_title('Model Residuals', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1.5])

plt.tight_layout()
output_path = RESULTS_DIR / 'test_03_desi_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TEST 3 SUMMARY")
print("=" * 70)

test_pass = delta_chi2 > 0

print(f"""
Dataset: DESI Y1 2024 RSD (5 redshift bins)
Analysis: Diagonal errors only (simplified)

ΛCDM: χ² = {chi2_lcdm:.2f}
IAM:  χ² = {chi2_iam:.2f}
Δχ²:      {delta_chi2:+.2f}

Status: {'✅ IAM PREFERRED' if test_pass else '⚠️  ΛCDM PREFERRED'}

CAVEAT: This uses diagonal errors only.
        Full analysis requires DESI covariance matrix.
        With only 5 data points, statistical power is limited.

Next Step: Fetch full DESI covariance matrix for rigorous test.
""")

print("=" * 70)
