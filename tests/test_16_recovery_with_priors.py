#!/usr/bin/env python3
"""
Test 16: Recovery with Priors
Add Planck priors to break degeneracies
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

np.random.seed(42)

n_sne = 100
z_all = np.sort(np.random.uniform(0.01, 1.5, n_sne))

Om0_true = 0.30
H0_true = 68.0
M_true = -19.3
TAU_ACT_TRUE = +0.15

# Planck priors
H0_planck = 67.4
H0_planck_err = 0.5
Om0_planck = 0.315
Om0_planck_err = 0.007

print("="*80)
print("IAM RECOVERY TEST WITH PRIORS")
print("="*80)
print()
print(f"TRUE: Ωm={Om0_true}, H₀={H0_true}, τ_act={TAU_ACT_TRUE:+.4f}")
print()

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

def integrate_dL(z, Om0, H0, tau_act=0):
    if z == 0: return 1e-10
    z_arr = np.linspace(0, z, 500)
    if tau_act == 0:
        H_vals = H_lcdm(z_arr, Om0, H0)
    else:
        H_vals = H_iam(z_arr, Om0, H0, tau_act)
    integrand = 3e5 / H_vals
    dC = np.trapezoid(integrand, z_arr)
    return (1 + z) * dC

def mu_theory(z, Om0, H0, M, tau_act=0):
    dL = integrate_dL(z, Om0, H0, tau_act)
    return M + 5.0 * np.log10(dL) + 25.0

# Generate data
mu_true = np.array([mu_theory(z, Om0_true, H0_true, M_true, TAU_ACT_TRUE) for z in z_all])
sigma_z = 0.15 * (1 + 0.3 * z_all)
mu_obs = mu_true + np.random.normal(0, sigma_z)

def chi2_iam_with_priors(params):
    Om0, H0, M, tau_act = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0: return 1e10
    
    # SNe chi2
    mu_model = np.array([mu_theory(z, Om0, H0, M, tau_act) for z in z_all])
    chi2 = np.sum(((mu_obs - mu_model) / sigma_z)**2)
    
    # Priors
    chi2 += ((H0 - H0_planck) / H0_planck_err)**2
    chi2 += ((Om0 - Om0_planck) / Om0_planck_err)**2
    
    return chi2

print("Fitting IAM with Planck priors...")
bounds = [(0.25, 0.35), (65.0, 70.0), (-20.0, -18.5), (-0.30, 0.30)]
result = differential_evolution(chi2_iam_with_priors, bounds, seed=43,
                                maxiter=1000, polish=True, disp=False)
Om0_fit, H0_fit, M_fit, tau_fit = result.x

print(f"  Ωm      = {Om0_fit:.4f}  (true: {Om0_true})")
print(f"  H₀      = {H0_fit:.2f}  (true: {H0_true})")
print(f"  M       = {M_fit:.3f}  (true: {M_true})")
print(f"  τ_act   = {tau_fit:+.4f}  (TRUE: {TAU_ACT_TRUE:+.4f})")
print()

error = abs(tau_fit - TAU_ACT_TRUE)
print(f"τ_act error: {error:.4f}")

if error < 0.02:
    print("✓✓✓ EXCELLENT! Recovered within 0.02")
elif error < 0.05:
    print("✓✓ GOOD! Recovered within 0.05")
else:
    print(f"⚠️  Error = {error:.4f}")

print("="*80)
