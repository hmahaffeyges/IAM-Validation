#!/usr/bin/env python3
"""
Test 14: Full SNe Dataset (Synthetic but realistic)
Expanded to ~200 SNe to test scaling
Based on real Pantheon+ statistics
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

# Generate synthetic SNe matching Pantheon+ distribution
np.random.seed(42)

# Real Pantheon+ has:
# - 1701 SNe total
# - z range: 0.001 to 2.26
# - Most SNe between z=0.01 and z=1.0

# Create 200 synthetic SNe for faster testing
n_sne = 200

# Redshift distribution (matches Pantheon+ roughly)
z_low = np.random.uniform(0.01, 0.1, 40)
z_mid = np.random.uniform(0.1, 0.5, 80)
z_high = np.random.uniform(0.5, 1.5, 60)
z_veryhigh = np.random.uniform(1.5, 2.3, 20)
z_all = np.concatenate([z_low, z_mid, z_high, z_veryhigh])
z_all = np.sort(z_all)

# Generate synthetic distance moduli using ΛCDM + noise
# This simulates what real SNe data looks like
Om0_true = 0.30
H0_true = 70.0

def H_lcdm(z, Om0, H0):
    return H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))

def dL_lcdm(z, Om0, H0):
    if z == 0: return 1e-10
    z_arr = np.linspace(0, z, 200)
    integrand = 3e5 / H_lcdm(z_arr, Om0, H0)
    dC = np.trapz(integrand, z_arr)
    return (1 + z) * dC

# Generate true values
M_true = -19.3
mu_true = []
for z in z_all:
    dL = dL_lcdm(z, Om0_true, H0_true)
    mu = M_true + 5.0 * np.log10(dL) + 25.0
    mu_true.append(mu)

mu_true = np.array(mu_true)

# Add realistic noise
# Errors scale with redshift (worse at high-z)
sigma_base = 0.15
sigma_z = sigma_base * (1 + 0.5 * z_all)
mu_obs = mu_true + np.random.normal(0, sigma_z)

print("="*80)
print("FULL SNe DATASET TEST (Synthetic)")
print(f"Simulating {n_sne} Type Ia supernovae")
print("="*80)
print()
print(f"Redshift range: {z_all.min():.4f} to {z_all.max():.4f}")
print(f"Median uncertainty: {np.median(sigma_z):.3f} mag")
print()

# Models (same as before)
def integrate_dL_lcdm(z, Om0, H0):
    if z == 0: return 1e-10
    z_arr = np.linspace(0, z, 500)
    integrand = 3e5 / H_lcdm(z_arr, Om0, H0)
    dC = np.trapz(integrand, z_arr)
    return (1 + z) * dC

def mu_theory_lcdm(z, Om0, H0, M):
    dL = integrate_dL_lcdm(z, Om0, H0)
    return M + 5.0 * np.log10(dL) + 25.0

def solve_growth_ode(z_max, Om0, n_points=500):
    z_vals = np.linspace(0, max(z_max, 3.0), n_points)
    a_vals = 1 / (1 + z_vals)
    lna_vals = np.log(a_vals)
    sort_idx = np.argsort(lna_vals)
    lna_sorted = lna_vals[sort_idx]
    
    def growth_ode(lna, y):
        a = np.exp(lna)
        z = 1/a - 1
        D, Dp = y
        Om_z = Om0 * (1+z)**3 / (Om0*(1+z)**3 + (1-Om0))
        eta = -3 * Om_z
        Dpp = -((2 + eta) * Dp - 1.5 * Om_z * D)
        return [Dp, Dpp]
    
    D0 = np.exp(lna_sorted[0])
    Dp0 = D0
    sol = solve_ivp(growth_ode, (lna_sorted[0], lna_sorted[-1]), [D0, Dp0],
                    t_eval=lna_sorted, method='DOP853', rtol=1e-10, atol=1e-12)
    D_sorted = sol.y[0]
    unsort_idx = np.argsort(sort_idx)
    D_vals = D_sorted[unsort_idx]
    D_vals /= D_vals[0]
    return z_vals, D_vals

def H_iam(z, Om0, H0, tau_act):
    z_max = np.max(z) if hasattr(z, '__len__') else z
    z_arr, D_vals = solve_growth_ode(z_max * 1.1, Om0, n_points=500)
    D_interp = interp1d(z_arr, D_vals, kind='cubic', fill_value='extrapolate')
    D_z = D_interp(z)
    H_base = H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))
    return H_base * (1 + tau_act * D_z)

def integrate_dL_iam(z, Om0, H0, tau_act):
    if z == 0: return 1e-10
    z_arr = np.linspace(0, z, 500)
    H_vals = H_iam(z_arr, Om0, H0, tau_act)
    integrand = 3e5 / H_vals
    dC = np.trapz(integrand, z_arr)
    return (1 + z) * dC

def mu_theory_iam(z, Om0, H0, M, tau_act):
    dL = integrate_dL_iam(z, Om0, H0, tau_act)
    return M + 5.0 * np.log10(dL) + 25.0

# H0 prior
H0_planck = 67.4
H0_planck_err = 0.5

def chi2_lcdm_with_prior(params):
    Om0, H0, M = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0: return 1e10
    
    mu_model = np.array([mu_theory_lcdm(z, Om0, H0, M) for z in z_all])
    chi2 = np.sum(((mu_obs - mu_model) / sigma_z)**2)
    chi2 += ((H0 - H0_planck) / H0_planck_err)**2
    return chi2

def chi2_iam_with_prior(params):
    Om0, H0, M, tau_act = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0: return 1e10
    
    mu_model = np.array([mu_theory_iam(z, Om0, H0, M, tau_act) for z in z_all])
    chi2 = np.sum(((mu_obs - mu_model) / sigma_z)**2)
    chi2 += ((H0 - H0_planck) / H0_planck_err)**2
    return chi2

print("Fitting ΛCDM (with Planck H₀ prior)...")
print("(This may take 1-2 minutes with 200 SNe...)")
bounds_lcdm = [(0.15, 0.45), (50.0, 90.0), (-21.0, -17.0)]
result_lcdm = differential_evolution(chi2_lcdm_with_prior, bounds_lcdm,
                                     seed=42, maxiter=1000, polish=True, disp=False)
Om0_lcdm, H0_lcdm, M_lcdm = result_lcdm.x
chi2_lcdm = result_lcdm.fun

print(f"  Ωm      = {Om0_lcdm:.4f}")
print(f"  H₀      = {H0_lcdm:.2f} km/s/Mpc")
print(f"  M       = {M_lcdm:.3f}")
print(f"  χ²      = {chi2_lcdm:.2f}")
print(f"  χ²/dof  = {chi2_lcdm/(n_sne-3):.3f}")
print()

print("Fitting IAM (with Planck H₀ prior)...")
print("(This may take 2-3 minutes...)")
bounds_iam = [(0.15, 0.45), (50.0, 90.0), (-21.0, -17.0), (-0.30, 0.30)]
result_iam = differential_evolution(chi2_iam_with_prior, bounds_iam,
                                    seed=42, maxiter=1000, polish=True, disp=False)
Om0_iam, H0_iam, M_iam, tau_act = result_iam.x
chi2_iam = result_iam.fun

print(f"  Ωm      = {Om0_iam:.4f}")
print(f"  H₀      = {H0_iam:.2f} km/s/Mpc")
print(f"  M       = {M_iam:.3f}")
print(f"  τ_act   = {tau_act:+.4f}")
print(f"  χ²      = {chi2_iam:.2f}")
print(f"  χ²/dof  = {chi2_iam/(n_sne-4):.3f}")
print()

delta_chi2 = chi2_lcdm - chi2_iam
sigma = np.sqrt(delta_chi2) if delta_chi2 > 0 else 0

print("="*80)
print(f"RESULTS WITH {n_sne} SNe")
print("="*80)
print(f"  χ²_ΛCDM = {chi2_lcdm:.2f}")
print(f"  χ²_IAM  = {chi2_iam:.2f}")
print(f"  Δχ²     = {delta_chi2:.2f}")
print(f"  Significance: ~{sigma:.1f}σ")
print()

print("SCALING ANALYSIS:")
print(f"  50 SNe:   Δχ² = 56.5  (7.5σ)")
print(f"  {n_sne} SNe:  Δχ² = {delta_chi2:.1f}  ({sigma:.1f}σ)")
print()

if n_sne == 200:
    expected = 56.5 * (200/50)
    print(f"  Expected if linear scaling: Δχ² ≈ {expected:.0f}")
    print(f"  Actual: Δχ² = {delta_chi2:.1f}")
    if delta_chi2 > 0.8 * expected:
        print("  ✓ Scaling holds! Pattern is robust.")
    
    # Extrapolate to full Pantheon+
    full_expected = delta_chi2 * (1701/200)
    full_sigma = np.sqrt(full_expected)
    print()
    print(f"  EXTRAPOLATION TO FULL PANTHEON+ (1701 SNe):")
    print(f"    Expected Δχ² ≈ {full_expected:.0f}")
    print(f"    Expected significance ≈ {full_sigma:.0f}σ")
    if full_expected > 500:
        print("    🚀 REVOLUTIONARY IF THIS HOLDS!")

print()
print("="*80)
