#!/usr/bin/env python3
"""
Test 05: Debug Individual DESI Bins
Find which redshift bins work vs don't work for IAM
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

# ============================================================================
# EXTENDED DESI BAO DATA
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

def H_iam(z, Om0, H0, tau_act):
    z_max = np.max(z) if hasattr(z, '__len__') else z
    z_arr, D_vals = solve_growth_ode(z_max * 1.1, Om0, n_points=500)
    D_interp = interp1d(z_arr, D_vals, kind='cubic', fill_value='extrapolate')
    D_z = D_interp(z)
    H_base = H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))
    return H_base * (1 + tau_act * D_z)

def integrate_comoving_iam(z, Om0, H0, tau_act):
    if z == 0:
        return 0.0
    z_arr = np.linspace(0, z, 500)
    H_vals = H_iam(z_arr, Om0, H0, tau_act)
    integrand = 3e5 / H_vals
    return np.trapezoid(integrand, z_arr)

# ============================================================================
# TEST INDIVIDUAL BINS
# ============================================================================

def test_single_bin(bin_index):
    """Test IAM vs ΛCDM for a single DESI bin plus H0 constraints"""
    
    z = desi_extended['z'][bin_index]
    DM_obs = desi_extended['DM_over_rd'][bin_index]
    DM_err = desi_extended['DM_over_rd_err'][bin_index]
    DH_obs = desi_extended['DH_over_rd'][bin_index]
    DH_err = desi_extended['DH_over_rd_err'][bin_index]
    
    def chi2_lcdm(params):
        Om0, H0, rd = params
        if Om0 <= 0 or Om0 >= 1 or H0 <= 0 or rd <= 0:
            return 1e10
        
        chi2 = 0.0
        chi2 += ((H0 - planck_H0) / planck_H0_err)**2
        chi2 += ((H0 - shoes_H0) / shoes_H0_err)**2
        
        DM_theory = integrate_comoving_lcdm(z, Om0, H0)
        DH_theory = 3e5 / H_lcdm(z, Om0, H0)
        
        chi2 += ((DM_theory - DM_obs * rd) / (DM_err * rd))**2
        chi2 += ((DH_theory - DH_obs * rd) / (DH_err * rd))**2
        
        return chi2
    
    def chi2_iam(params):
        Om0, H0, rd, tau_act = params
        if Om0 <= 0 or Om0 >= 1 or H0 <= 0 or rd <= 0:
            return 1e10
        
        chi2 = 0.0
        chi2 += ((H0 - planck_H0) / planck_H0_err)**2
        chi2 += ((H0 - shoes_H0) / shoes_H0_err)**2
        
        DM_theory = integrate_comoving_iam(z, Om0, H0, tau_act)
        DH_theory = 3e5 / H_iam(z, Om0, H0, tau_act)
        
        chi2 += ((DM_theory - DM_obs * rd) / (DM_err * rd))**2
        chi2 += ((DH_theory - DH_obs * rd) / (DH_err * rd))**2
        
        return chi2
    
    # Fit ΛCDM
    bounds_lcdm = [(0.25, 0.35), (67.0, 73.5), (140.0, 155.0)]
    result_lcdm = differential_evolution(chi2_lcdm, bounds_lcdm, seed=42, maxiter=1000)
    
    # Fit IAM
    bounds_iam = [(0.25, 0.35), (67.0, 73.5), (140.0, 155.0), (-0.1, 0.2)]
    result_iam = differential_evolution(chi2_iam, bounds_iam, seed=42, maxiter=1000)
    
    return {
        'z': z,
        'chi2_lcdm': result_lcdm.fun,
        'chi2_iam': result_iam.fun,
        'delta_chi2': result_lcdm.fun - result_iam.fun,
        'Om0_lcdm': result_lcdm.x[0],
        'H0_lcdm': result_lcdm.x[1],
        'rd_lcdm': result_lcdm.x[2],
        'Om0_iam': result_iam.x[0],
        'H0_iam': result_iam.x[1],
        'rd_iam': result_iam.x[2],
        'tau_act': result_iam.x[3]
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("INDIVIDUAL BIN DEBUGGING")
    print("Testing each DESI redshift bin separately")
    print("="*80)
    print()
    
    results = []
    
    for i in range(len(desi_extended['z'])):
        print(f"Testing bin {i+1}/7 (z = {desi_extended['z'][i]:.3f})...", end=" ")
        result = test_single_bin(i)
        results.append(result)
        
        if result['delta_chi2'] > 0:
            status = "✓ IAM BETTER"
        else:
            status = "✗ ΛCDM BETTER"
        
        print(f"{status} (Δχ² = {result['delta_chi2']:+.2f})")
    
    print()
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    print(f"{'z':<8} {'χ²_ΛCDM':<10} {'χ²_IAM':<10} {'Δχ²':<10} {'τ_act':<10} {'Status':<15}")
    print("-"*80)
    
    for r in results:
        status = "✓ IAM wins" if r['delta_chi2'] > 0 else "✗ ΛCDM wins"
        print(f"{r['z']:<8.3f} {r['chi2_lcdm']:<10.2f} {r['chi2_iam']:<10.2f} "
              f"{r['delta_chi2']:<+10.2f} {r['tau_act']:<+10.4f} {status:<15}")
    
    print()
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Count wins
    iam_wins = sum(1 for r in results if r['delta_chi2'] > 0)
    lcdm_wins = len(results) - iam_wins
    
    print(f"IAM wins:  {iam_wins}/{len(results)} bins")
    print(f"ΛCDM wins: {lcdm_wins}/{len(results)} bins")
    print()
    
    # Find problem bins
    problem_bins = [r for r in results if r['delta_chi2'] < 0]
    if problem_bins:
        print("PROBLEM BINS (IAM performs worse):")
        for r in problem_bins:
            print(f"  z = {r['z']:.3f}: Δχ² = {r['delta_chi2']:+.2f}, τ_act = {r['tau_act']:+.4f}")
    
    print()
    
    # Find strong bins
    strong_bins = [r for r in results if r['delta_chi2'] > 5.0]
    if strong_bins:
        print("STRONG BINS (IAM performs much better):")
        for r in strong_bins:
            print(f"  z = {r['z']:.3f}: Δχ² = {r['delta_chi2']:+.2f}, τ_act = {r['tau_act']:+.4f}")
    
    print()
    print("="*80)
