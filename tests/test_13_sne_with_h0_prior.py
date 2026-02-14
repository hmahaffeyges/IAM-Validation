#!/usr/bin/env python3
"""
Test 13: SNe with H0 Prior
Add Planck H0 constraint to break degeneracy
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

pantheon_sample = np.array([
    [0.0104, 14.288, 0.144], [0.0129, 14.942, 0.144], [0.0138, 15.096, 0.144],
    [0.0142, 15.180, 0.144], [0.0147, 15.281, 0.144], [0.0196, 16.027, 0.144],
    [0.0220, 16.344, 0.144], [0.0233, 16.512, 0.144], [0.0250, 16.738, 0.144],
    [0.0268, 16.982, 0.144], [0.0305, 17.391, 0.144], [0.0313, 17.495, 0.144],
    [0.0321, 17.593, 0.144], [0.0329, 17.687, 0.144], [0.0358, 18.022, 0.144],
    [0.0372, 18.159, 0.144], [0.0395, 18.405, 0.147], [0.0424, 18.700, 0.148],
    [0.0450, 18.955, 0.149], [0.0500, 19.358, 0.152], [0.0550, 19.740, 0.155],
    [0.0600, 20.101, 0.158], [0.0650, 20.443, 0.161], [0.0700, 20.769, 0.164],
    [0.0800, 21.371, 0.170], [0.0900, 21.929, 0.176], [0.1000, 22.449, 0.182],
    [0.1200, 23.403, 0.194], [0.1400, 24.271, 0.206], [0.1600, 25.067, 0.218],
    [0.1800, 25.800, 0.230], [0.2000, 26.478, 0.242], [0.2500, 27.982, 0.274],
    [0.3000, 29.290, 0.306], [0.3500, 30.435, 0.338], [0.4000, 31.445, 0.370],
    [0.4500, 32.341, 0.402], [0.5000, 33.138, 0.434], [0.5500, 33.851, 0.466],
    [0.6000, 34.489, 0.498], [0.6500, 35.063, 0.530], [0.7000, 35.581, 0.562],
    [0.7500, 36.050, 0.594], [0.8000, 36.475, 0.626], [0.8500, 36.862, 0.658],
    [0.9000, 37.214, 0.690], [0.9500, 37.536, 0.722], [1.0000, 37.830, 0.754],
    [1.0500, 38.100, 0.786], [1.1000, 38.348, 0.818],
])

z_sne = pantheon_sample[:, 0]
mb_obs = pantheon_sample[:, 1]
dmb_obs = pantheon_sample[:, 2]

# H0 priors
H0_planck = 67.4
H0_planck_err = 0.5
H0_shoes = 73.04
H0_shoes_err = 1.04

def H_lcdm(z, Om0, H0):
    return H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))

def integrate_dL_lcdm(z, Om0, H0):
    if z == 0: return 1e-10
    z_arr = np.linspace(0, z, 500)
    integrand = 3e5 / H_lcdm(z_arr, Om0, H0)
    dC = np.trapezoid(integrand, z_arr)
    return (1 + z) * dC

def mb_theory_lcdm(z, Om0, H0, M):
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
    dC = np.trapezoid(integrand, z_arr)
    return (1 + z) * dC

def mb_theory_iam(z, Om0, H0, M, tau_act):
    dL = integrate_dL_iam(z, Om0, H0, tau_act)
    return M + 5.0 * np.log10(dL) + 25.0

def chi2_sne_lcdm_with_prior(params):
    Om0, H0, M = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0: return 1e10
    
    # SNe chi2
    mb_model = np.array([mb_theory_lcdm(z, Om0, H0, M) for z in z_sne])
    chi2 = np.sum(((mb_obs - mb_model) / dmb_obs)**2)
    
    # H0 prior (Planck)
    chi2 += ((H0 - H0_planck) / H0_planck_err)**2
    
    return chi2

def chi2_sne_iam_with_prior(params):
    Om0, H0, M, tau_act = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0: return 1e10
    
    # SNe chi2
    mb_model = np.array([mb_theory_iam(z, Om0, H0, M, tau_act) for z in z_sne])
    chi2 = np.sum(((mb_obs - mb_model) / dmb_obs)**2)
    
    # H0 prior (Planck)
    chi2 += ((H0 - H0_planck) / H0_planck_err)**2
    
    return chi2

print("="*80)
print("SNe WITH H0 PRIOR (PLANCK)")
print("Breaking τ_act - H₀ degeneracy")
print("="*80)
print()

print(f"H₀ prior: {H0_planck} ± {H0_planck_err} km/s/Mpc (Planck)")
print()

print("Fitting ΛCDM...")
bounds_lcdm = [(0.15, 0.45), (50.0, 90.0), (-21.0, -17.0)]
result_lcdm = differential_evolution(chi2_sne_lcdm_with_prior, bounds_lcdm,
                                     seed=42, maxiter=1500, polish=True, disp=False)
Om0_lcdm, H0_lcdm, M_lcdm = result_lcdm.x
chi2_lcdm = result_lcdm.fun

print(f"  Ωm      = {Om0_lcdm:.4f}")
print(f"  H₀      = {H0_lcdm:.2f} km/s/Mpc")
print(f"  M       = {M_lcdm:.3f}")
print(f"  χ²      = {chi2_lcdm:.2f}")
print()

print("Fitting IAM...")
bounds_iam = [(0.15, 0.45), (50.0, 90.0), (-21.0, -17.0), (-0.30, 0.30)]
result_iam = differential_evolution(chi2_sne_iam_with_prior, bounds_iam,
                                    seed=42, maxiter=1500, polish=True, disp=False)
Om0_iam, H0_iam, M_iam, tau_act = result_iam.x
chi2_iam = result_iam.fun

print(f"  Ωm      = {Om0_iam:.4f}")
print(f"  H₀      = {H0_iam:.2f} km/s/Mpc")
print(f"  M       = {M_iam:.3f}")
print(f"  τ_act   = {tau_act:+.4f}")
print(f"  χ²      = {chi2_iam:.2f}")
print()

delta_chi2 = chi2_lcdm - chi2_iam
sigma = np.sqrt(delta_chi2) if delta_chi2 > 0 else 0

print("="*80)
print("RESULTS WITH PLANCK H₀ PRIOR")
print("="*80)
print(f"  χ²_ΛCDM = {chi2_lcdm:.2f}")
print(f"  χ²_IAM  = {chi2_iam:.2f}")
print(f"  Δχ²     = {delta_chi2:.2f}")
print(f"  Significance: ~{sigma:.1f}σ")
print()

print("PARAMETER COMPARISON:")
print(f"  H₀: {H0_lcdm:.2f} → {H0_iam:.2f} km/s/Mpc")
print(f"  Δ(H₀) = {H0_iam - H0_lcdm:.2f} km/s/Mpc")
print(f"  τ_act = {tau_act:+.4f}")
print()

if abs(H0_iam - H0_planck) < 2.0:
    print("✓ IAM H₀ consistent with Planck")
elif abs(H0_iam - H0_shoes) < 2.0:
    print("✓ IAM H₀ consistent with SH0ES")
else:
    print(f"⚠️  IAM H₀ = {H0_iam:.1f} differs from both Planck ({H0_planck}) and SH0ES ({H0_shoes})")

print()

if delta_chi2 > 9:
    print(f"✓✓✓ STRONG: Δχ² = {delta_chi2:.0f} even with H₀ prior!")
    print("    IAM improvement is ROBUST to H₀ constraint")
elif delta_chi2 > 4:
    print(f"✓✓ MODERATE: Δχ² = {delta_chi2:.0f} with H₀ prior")
elif delta_chi2 > 1:
    print(f"✓ WEAK: Δχ² = {delta_chi2:.0f} with H₀ prior")
else:
    print("✗ No improvement with H₀ constrained")
    print("   → Improvement was due to H₀ degeneracy")

print()
print("="*80)
