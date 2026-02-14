#!/usr/bin/env python3
"""
Test 07: Fixed Cosmology, Fit Only τ_act(z)
Fix Om0, H0, rd to best-fit ΛCDM values
Only optimize τ_0 and τ_1
Tests if degeneracies are preventing good fit
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

# ============================================================================
# DATA
# ============================================================================

desi_extended = {
    'z': np.array([0.295, 0.510, 0.706, 0.930, 1.317, 1.491, 2.330]),
    'DM_over_rd': np.array([7.93, 13.62, 18.33, 23.38, 30.21, 33.41, 39.71]),
    'DM_over_rd_err': np.array([0.14, 0.25, 0.17, 0.24, 0.52, 0.58, 1.08]),
    'DH_over_rd': np.array([25.50, 22.33, 20.78, 17.88, 13.82, 12.84, 8.52]),
    'DH_over_rd_err': np.array([0.50, 0.48, 0.38, 0.34, 0.42, 0.40, 0.28])
}

planck_H0 = 67.4
planck_H0_err = 0.5
shoes_H0 = 73.04
shoes_H0_err = 1.04

# FIXED to best-fit ΛCDM values from test_06
Om0_FIXED = 0.3096
H0_FIXED = 68.46
rd_FIXED = 141.21

# ============================================================================
# MODELS
# ============================================================================

def H_lcdm(z, Om0, H0):
    return H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))

def integrate_comoving_lcdm(z, Om0, H0):
    if z == 0:
        return 0.0
    z_arr = np.linspace(0, z, 500)
    integrand = 3e5 / H_lcdm(z_arr, Om0, H0)
    return np.trapezoid(integrand, z_arr)

def solve_growth_ode(z_max, Om0, n_points=500):
    z_vals = np.linspace(0, z_max, n_points)
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
    
    sol = solve_ivp(
        growth_ode,
        (lna_sorted[0], lna_sorted[-1]),
        [D0, Dp0],
        t_eval=lna_sorted,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12
    )
    
    D_sorted = sol.y[0]
    unsort_idx = np.argsort(sort_idx)
    D_vals = D_sorted[unsort_idx]
    D_vals /= D_vals[0]
    
    return z_vals, D_vals

def tau_act_evolving(z, tau_0, tau_1):
    return tau_0 + tau_1 * z

def H_iam_evolving(z, Om0, H0, tau_0, tau_1):
    z_max = np.max(z) if hasattr(z, '__len__') else z
    z_arr, D_vals = solve_growth_ode(z_max * 1.1, Om0, n_points=500)
    D_interp = interp1d(z_arr, D_vals, kind='cubic', fill_value='extrapolate')
    
    D_z = D_interp(z)
    tau_z = tau_act_evolving(z, tau_0, tau_1)
    
    H_base = H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))
    return H_base * (1 + tau_z * D_z)

def integrate_comoving_iam_evolving(z, Om0, H0, tau_0, tau_1):
    if z == 0:
        return 0.0
    z_arr = np.linspace(0, z, 500)
    H_vals = H_iam_evolving(z_arr, Om0, H0, tau_0, tau_1)
    integrand = 3e5 / H_vals
    return np.trapezoid(integrand, z_arr)

# ============================================================================
# CHI-SQUARED
# ============================================================================

def chi2_lcdm_fixed():
    """ΛCDM with FIXED parameters"""
    chi2 = 0.0
    chi2 += ((H0_FIXED - planck_H0) / planck_H0_err)**2
    chi2 += ((H0_FIXED - shoes_H0) / shoes_H0_err)**2
    
    for i, z in enumerate(desi_extended['z']):
        DM_theory = integrate_comoving_lcdm(z, Om0_FIXED, H0_FIXED)
        DM_obs = desi_extended['DM_over_rd'][i] * rd_FIXED
        DM_err = desi_extended['DM_over_rd_err'][i] * rd_FIXED
        chi2 += ((DM_theory - DM_obs) / DM_err)**2
        
        DH_theory = 3e5 / H_lcdm(z, Om0_FIXED, H0_FIXED)
        DH_obs = desi_extended['DH_over_rd'][i] * rd_FIXED
        DH_err = desi_extended['DH_over_rd_err'][i] * rd_FIXED
        chi2 += ((DH_theory - DH_obs) / DH_err)**2
    
    return chi2

def chi2_iam_tau_only(params):
    """IAM with FIXED Om0, H0, rd - only fit τ_0, τ_1"""
    tau_0, tau_1 = params
    
    chi2 = 0.0
    chi2 += ((H0_FIXED - planck_H0) / planck_H0_err)**2
    chi2 += ((H0_FIXED - shoes_H0) / shoes_H0_err)**2
    
    for i, z in enumerate(desi_extended['z']):
        DM_theory = integrate_comoving_iam_evolving(z, Om0_FIXED, H0_FIXED, tau_0, tau_1)
        DM_obs = desi_extended['DM_over_rd'][i] * rd_FIXED
        DM_err = desi_extended['DM_over_rd_err'][i] * rd_FIXED
        chi2 += ((DM_theory - DM_obs) / DM_err)**2
        
        DH_theory = 3e5 / H_iam_evolving(z, Om0_FIXED, H0_FIXED, tau_0, tau_1)
        DH_obs = desi_extended['DH_over_rd'][i] * rd_FIXED
        DH_err = desi_extended['DH_over_rd_err'][i] * rd_FIXED
        chi2 += ((DH_theory - DH_obs) / DH_err)**2
    
    return chi2

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FIXED COSMOLOGY TEST")
    print("Fix Om0, H0, rd to best-fit ΛCDM values")
    print("Only optimize τ_0 and τ_1")
    print("="*80)
    print()
    
    print("Fixed parameters:")
    print(f"  Ωm = {Om0_FIXED:.4f}")
    print(f"  H₀ = {H0_FIXED:.2f} km/s/Mpc")
    print(f"  rd = {rd_FIXED:.2f} Mpc")
    print()
    
    # ΛCDM chi-squared (already fixed)
    chi2_lcdm = chi2_lcdm_fixed()
    print(f"ΛCDM χ² (fixed cosmology): {chi2_lcdm:.2f}")
    print()
    
    # Fit only τ_0, τ_1
    print("Fitting IAM: only τ_0 and τ_1 (cosmology fixed)...")
    bounds_tau = [
        (-0.2, 0.2),   # tau_0
        (-0.2, 0.1)    # tau_1
    ]
    
    result = differential_evolution(
        chi2_iam_tau_only,
        bounds_tau,
        seed=42,
        maxiter=2000,
        workers=1,
        polish=True
    )
    
    tau_0, tau_1 = result.x
    chi2_iam = result.fun
    
    print(f"  τ_0 = {tau_0:+.4f}")
    print(f"  τ_1 = {tau_1:+.4f}")
    print(f"  χ² = {chi2_iam:.2f}")
    print()
    
    print("τ_act at key redshifts:")
    for z in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        tau_z = tau_act_evolving(z, tau_0, tau_1)
        print(f"  z = {z:.1f}: τ_act = {tau_z:+.4f}")
    print()
    
    # Compare
    delta_chi2 = chi2_lcdm - chi2_iam
    
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print(f"  χ²_ΛCDM = {chi2_lcdm:.2f}")
    print(f"  χ²_IAM  = {chi2_iam:.2f}")
    print(f"  Δχ²     = {delta_chi2:.2f}")
    print()
    
    if delta_chi2 > 5:
        print("✓✓ SIGNIFICANT IMPROVEMENT!")
        print("   → Degeneracies were the problem")
        print("   → τ_act effect is real when cosmology fixed")
    elif delta_chi2 > 1:
        print("✓ Modest improvement")
        print("   → Some degeneracy effect, but not complete explanation")
    else:
        print("✗ No improvement")
        print("   → Problem is NOT just degeneracies")
        print("   → Missing physics (backreaction?)")
    
    print()
    print("="*80)
