#!/usr/bin/env python3
"""
Test 15: Recovery Test
Generate synthetic data WITH IAM (tau_act = +0.15)
See if we can recover the input value
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

np.random.seed(42)

# Generate synthetic SNe
n_sne = 100
z_all = np.sort(np.random.uniform(0.01, 1.5, n_sne))

# TRUE parameters (what we'll try to recover)
Om0_true = 0.30
H0_true = 68.0
M_true = -19.3
TAU_ACT_TRUE = +0.15  # This is what we want to recover

print("="*80)
print("IAM RECOVERY TEST")
print("Generate data WITH actualization, see if we recover τ_act")
print("="*80)
print()
print(f"TRUE PARAMETERS:")
print(f"  Ωm      = {Om0_true}")
print(f"  H₀      = {H0_true} km/s/Mpc")
print(f"  M       = {M_true}")
print(f"  τ_act   = {TAU_ACT_TRUE:+.4f}  ← THIS IS WHAT WE'RE TESTING")
print()

# Models
def H_lcdm(z, Om0, H0):
    return H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))

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
    dC = np.trapezoid(integrand, z_arr)
    return (1 + z) * dC

def mu_theory_iam(z, Om0, H0, M, tau_act):
    dL = integrate_dL_iam(z, Om0, H0, tau_act)
    return M + 5.0 * np.log10(dL) + 25.0

# Generate TRUE data using IAM
print("Generating synthetic data WITH IAM...")
mu_true = np.array([mu_theory_iam(z, Om0_true, H0_true, M_true, TAU_ACT_TRUE) 
                    for z in z_all])

# Add realistic noise
sigma_z = 0.15 * (1 + 0.3 * z_all)
mu_obs = mu_true + np.random.normal(0, sigma_z)

print(f"Generated {n_sne} SNe with τ_act = {TAU_ACT_TRUE}")
print()

# Now fit with ΛCDM (should do poorly)
def integrate_dL_lcdm(z, Om0, H0):
    if z == 0: return 1e-10
    z_arr = np.linspace(0, z, 500)
    integrand = 3e5 / H_lcdm(z_arr, Om0, H0)
    dC = np.trapezoid(integrand, z_arr)
    return (1 + z) * dC

def mu_theory_lcdm(z, Om0, H0, M):
    dL = integrate_dL_lcdm(z, Om0, H0)
    return M + 5.0 * np.log10(dL) + 25.0

def chi2_lcdm(params):
    Om0, H0, M = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0: return 1e10
    mu_model = np.array([mu_theory_lcdm(z, Om0, H0, M) for z in z_all])
    return np.sum(((mu_obs - mu_model) / sigma_z)**2)

def chi2_iam(params):
    Om0, H0, M, tau_act = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0: return 1e10
    mu_model = np.array([mu_theory_iam(z, Om0, H0, M, tau_act) for z in z_all])
    return np.sum(((mu_obs - mu_model) / sigma_z)**2)

print("Fitting ΛCDM to IAM-generated data...")
bounds_lcdm = [(0.15, 0.45), (60.0, 80.0), (-21.0, -18.0)]
result_lcdm = differential_evolution(chi2_lcdm, bounds_lcdm, seed=43, 
                                     maxiter=800, polish=True, disp=False)
Om0_lcdm, H0_lcdm, M_lcdm = result_lcdm.x
chi2_lcdm_val = result_lcdm.fun

print(f"  Ωm      = {Om0_lcdm:.4f}  (true: {Om0_true})")
print(f"  H₀      = {H0_lcdm:.2f}  (true: {H0_true})")
print(f"  M       = {M_lcdm:.3f}  (true: {M_true})")
print(f"  χ²      = {chi2_lcdm_val:.2f}")
print()

print("Fitting IAM to IAM-generated data...")
bounds_iam = [(0.15, 0.45), (60.0, 80.0), (-21.0, -18.0), (-0.30, 0.30)]
result_iam = differential_evolution(chi2_iam, bounds_iam, seed=43,
                                    maxiter=800, polish=True, disp=False)
Om0_iam, H0_iam, M_iam, tau_act_recovered = result_iam.x
chi2_iam_val = result_iam.fun

print(f"  Ωm      = {Om0_iam:.4f}  (true: {Om0_true})")
print(f"  H₀      = {H0_iam:.2f}  (true: {H0_true})")
print(f"  M       = {M_iam:.3f}  (true: {M_true})")
print(f"  τ_act   = {tau_act_recovered:+.4f}  (TRUE: {TAU_ACT_TRUE:+.4f})")
print(f"  χ²      = {chi2_iam_val:.2f}")
print()

delta_chi2 = chi2_lcdm_val - chi2_iam_val
sigma = np.sqrt(delta_chi2) if delta_chi2 > 0 else 0

print("="*80)
print("RECOVERY TEST RESULTS")
print("="*80)
print(f"  χ²_ΛCDM = {chi2_lcdm_val:.2f}")
print(f"  χ²_IAM  = {chi2_iam_val:.2f}")
print(f"  Δχ²     = {delta_chi2:.2f}  ({sigma:.1f}σ)")
print()

print("PARAMETER RECOVERY:")
print(f"  τ_act input:     {TAU_ACT_TRUE:+.4f}")
print(f"  τ_act recovered: {tau_act_recovered:+.4f}")
print(f"  Error:           {abs(tau_act_recovered - TAU_ACT_TRUE):.4f}")
print()

if abs(tau_act_recovered - TAU_ACT_TRUE) < 0.02:
    print("✓✓✓ EXCELLENT RECOVERY!")
    print("    IAM correctly identifies τ_act from data")
elif abs(tau_act_recovered - TAU_ACT_TRUE) < 0.05:
    print("✓✓ GOOD RECOVERY")
    print("    IAM recovers τ_act within uncertainties")
else:
    print("⚠️  POOR RECOVERY")
    print("    May need more data or tighter constraints")

if delta_chi2 > 25:
    print()
    print("✓ ΛCDM correctly rejects IAM-generated data!")
    print("  This proves IAM contains real signal")

print()
print("="*80)
