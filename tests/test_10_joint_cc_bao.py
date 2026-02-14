#!/usr/bin/env python3
"""
Test 10: Joint CC + BAO Fit
Combine Cosmic Chronometers + DESI BAO
Tests if datasets agree on τ_act or show tension
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

# ============================================================================
# COSMIC CHRONOMETER DATA
# ============================================================================

cc_data = {
    'z': np.array([
        0.070, 0.090, 0.120, 0.170, 0.179, 0.199, 0.200, 0.270, 0.280,
        0.352, 0.380, 0.400, 0.424, 0.445, 0.480, 0.593, 0.680, 0.781,
        0.875, 0.880, 0.900, 1.037, 1.300, 1.363, 1.430, 1.530, 1.750, 1.965
    ]),
    'H': np.array([
        69.0, 69.0, 68.6, 83.0, 75.0, 75.0, 72.9, 77.0, 88.8,
        83.0, 83.0, 95.0, 87.1, 92.8, 97.0, 104.0, 92.0, 105.0,
        125.0, 90.0, 117.0, 154.0, 168.0, 160.0, 177.0, 140.0, 202.0, 186.5
    ]),
    'H_err': np.array([
        19.6, 12.0, 26.2, 8.0, 4.0, 5.0, 29.6, 14.0, 36.6,
        14.0, 13.5, 17.0, 11.2, 12.9, 62.0, 13.0, 8.0, 12.0,
        17.0, 40.0, 23.0, 20.0, 17.0, 33.6, 18.0, 14.0, 40.0, 50.4
    ])
}

# ============================================================================
# DESI BAO DATA
# ============================================================================

desi_bao = {
    'z': np.array([0.295, 0.510, 0.706, 0.930, 1.317, 1.491, 2.330]),
    'DM_over_rd': np.array([7.93, 13.62, 18.33, 23.38, 30.21, 33.41, 39.71]),
    'DM_over_rd_err': np.array([0.14, 0.25, 0.17, 0.24, 0.52, 0.58, 1.08]),
    'DH_over_rd': np.array([25.50, 22.33, 20.78, 17.88, 13.82, 12.84, 8.52]),
    'DH_over_rd_err': np.array([0.50, 0.48, 0.38, 0.34, 0.42, 0.40, 0.28])
}

# H0 constraints
planck_H0 = 67.4
planck_H0_err = 0.5
shoes_H0 = 73.04
shoes_H0_err = 1.04

# ============================================================================
# MODELS
# ============================================================================

def H_lcdm(z, Om0, H0):
    """ΛCDM Hubble parameter"""
    return H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))

def integrate_comoving_lcdm(z, Om0, H0):
    """Comoving distance in ΛCDM"""
    if z == 0:
        return 0.0
    z_arr = np.linspace(0, z, 500)
    integrand = 3e5 / H_lcdm(z_arr, Om0, H0)
    return np.trapezoid(integrand, z_arr)

def solve_growth_ode(z_max, Om0, n_points=500):
    """Solve growth factor ODE"""
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
    """IAM Hubble parameter"""
    z_max = np.max(z) if hasattr(z, '__len__') else z
    z_arr, D_vals = solve_growth_ode(z_max * 1.1, Om0, n_points=500)
    D_interp = interp1d(z_arr, D_vals, kind='cubic', fill_value='extrapolate')
    
    D_z = D_interp(z)
    H_base = H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))
    return H_base * (1 + tau_act * D_z)

def integrate_comoving_iam(z, Om0, H0, tau_act):
    """Comoving distance in IAM"""
    if z == 0:
        return 0.0
    z_arr = np.linspace(0, z, 500)
    H_vals = H_iam(z_arr, Om0, H0, tau_act)
    integrand = 3e5 / H_vals
    return np.trapezoid(integrand, z_arr)

# ============================================================================
# CHI-SQUARED FUNCTIONS
# ============================================================================

def chi2_cc_only_lcdm(params):
    """CC data only - ΛCDM"""
    Om0, H0 = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0:
        return 1e10
    
    chi2 = 0.0
    for i, z in enumerate(cc_data['z']):
        H_theory = H_lcdm(z, Om0, H0)
        chi2 += ((H_theory - cc_data['H'][i]) / cc_data['H_err'][i])**2
    return chi2

def chi2_cc_only_iam(params):
    """CC data only - IAM"""
    Om0, H0, tau_act = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0:
        return 1e10
    
    chi2 = 0.0
    for i, z in enumerate(cc_data['z']):
        H_theory = H_iam(z, Om0, H0, tau_act)
        chi2 += ((H_theory - cc_data['H'][i]) / cc_data['H_err'][i])**2
    return chi2

def chi2_bao_only_lcdm(params):
    """BAO data only - ΛCDM"""
    Om0, H0, rd = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0 or rd <= 0:
        return 1e10
    
    chi2 = 0.0
    chi2 += ((H0 - planck_H0) / planck_H0_err)**2
    chi2 += ((H0 - shoes_H0) / shoes_H0_err)**2
    
    for i, z in enumerate(desi_bao['z']):
        DM_theory = integrate_comoving_lcdm(z, Om0, H0)
        DM_obs = desi_bao['DM_over_rd'][i] * rd
        DM_err = desi_bao['DM_over_rd_err'][i] * rd
        chi2 += ((DM_theory - DM_obs) / DM_err)**2
        
        DH_theory = 3e5 / H_lcdm(z, Om0, H0)
        DH_obs = desi_bao['DH_over_rd'][i] * rd
        DH_err = desi_bao['DH_over_rd_err'][i] * rd
        chi2 += ((DH_theory - DH_obs) / DH_err)**2
    
    return chi2

def chi2_bao_only_iam(params):
    """BAO data only - IAM"""
    Om0, H0, rd, tau_act = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0 or rd <= 0:
        return 1e10
    
    chi2 = 0.0
    chi2 += ((H0 - planck_H0) / planck_H0_err)**2
    chi2 += ((H0 - shoes_H0) / shoes_H0_err)**2
    
    for i, z in enumerate(desi_bao['z']):
        DM_theory = integrate_comoving_iam(z, Om0, H0, tau_act)
        DM_obs = desi_bao['DM_over_rd'][i] * rd
        DM_err = desi_bao['DM_over_rd_err'][i] * rd
        chi2 += ((DM_theory - DM_obs) / DM_err)**2
        
        DH_theory = 3e5 / H_iam(z, Om0, H0, tau_act)
        DH_obs = desi_bao['DH_over_rd'][i] * rd
        DH_err = desi_bao['DH_over_rd_err'][i] * rd
        chi2 += ((DH_theory - DH_obs) / DH_err)**2
    
    return chi2

def chi2_joint_lcdm(params):
    """Joint CC + BAO - ΛCDM"""
    Om0, H0, rd = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0 or rd <= 0:
        return 1e10
    
    chi2 = 0.0
    
    # CC contribution
    for i, z in enumerate(cc_data['z']):
        H_theory = H_lcdm(z, Om0, H0)
        chi2 += ((H_theory - cc_data['H'][i]) / cc_data['H_err'][i])**2
    
    # H0 priors
    chi2 += ((H0 - planck_H0) / planck_H0_err)**2
    chi2 += ((H0 - shoes_H0) / shoes_H0_err)**2
    
    # BAO contribution
    for i, z in enumerate(desi_bao['z']):
        DM_theory = integrate_comoving_lcdm(z, Om0, H0)
        DM_obs = desi_bao['DM_over_rd'][i] * rd
        DM_err = desi_bao['DM_over_rd_err'][i] * rd
        chi2 += ((DM_theory - DM_obs) / DM_err)**2
        
        DH_theory = 3e5 / H_lcdm(z, Om0, H0)
        DH_obs = desi_bao['DH_over_rd'][i] * rd
        DH_err = desi_bao['DH_over_rd_err'][i] * rd
        chi2 += ((DH_theory - DH_obs) / DH_err)**2
    
    return chi2

def chi2_joint_iam(params):
    """Joint CC + BAO - IAM"""
    Om0, H0, rd, tau_act = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0 or rd <= 0:
        return 1e10
    
    chi2 = 0.0
    
    # CC contribution
    for i, z in enumerate(cc_data['z']):
        H_theory = H_iam(z, Om0, H0, tau_act)
        chi2 += ((H_theory - cc_data['H'][i]) / cc_data['H_err'][i])**2
    
    # H0 priors
    chi2 += ((H0 - planck_H0) / planck_H0_err)**2
    chi2 += ((H0 - shoes_H0) / shoes_H0_err)**2
    
    # BAO contribution
    for i, z in enumerate(desi_bao['z']):
        DM_theory = integrate_comoving_iam(z, Om0, H0, tau_act)
        DM_obs = desi_bao['DM_over_rd'][i] * rd
        DM_err = desi_bao['DM_over_rd_err'][i] * rd
        chi2 += ((DM_theory - DM_obs) / DM_err)**2
        
        DH_theory = 3e5 / H_iam(z, Om0, H0, tau_act)
        DH_obs = desi_bao['DH_over_rd'][i] * rd
        DH_err = desi_bao['DH_over_rd_err'][i] * rd
        chi2 += ((DH_theory - DH_obs) / DH_err)**2
    
    return chi2

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("JOINT CC + BAO FIT")
    print("Do different datasets agree on τ_act?")
    print("="*80)
    print()
    
    n_cc = len(cc_data['z'])
    n_bao = 2 * len(desi_bao['z'])  # DM and DH
    n_H0 = 2
    n_total = n_cc + n_bao + n_H0
    
    print(f"Dataset breakdown:")
    print(f"  CC measurements:  {n_cc}")
    print(f"  BAO measurements: {n_bao}")
    print(f"  H0 constraints:   {n_H0}")
    print(f"  Total:            {n_total}")
    print()
    
    # ========================================================================
    # CC ONLY
    # ========================================================================
    print("="*80)
    print("CC ONLY")
    print("="*80)
    
    print("Fitting ΛCDM to CC only...")
    bounds_cc_lcdm = [(0.20, 0.40), (60.0, 80.0)]
    result_cc_lcdm = differential_evolution(
        chi2_cc_only_lcdm, bounds_cc_lcdm, seed=42, maxiter=1000, polish=True
    )
    print(f"  χ²_ΛCDM = {result_cc_lcdm.fun:.2f}")
    
    print("Fitting IAM to CC only...")
    bounds_cc_iam = [(0.20, 0.40), (60.0, 80.0), (-0.15, 0.15)]
    result_cc_iam = differential_evolution(
        chi2_cc_only_iam, bounds_cc_iam, seed=42, maxiter=1000, polish=True
    )
    tau_cc = result_cc_iam.x[2]
    print(f"  χ²_IAM  = {result_cc_iam.fun:.2f}")
    print(f"  τ_act   = {tau_cc:+.4f}")
    print(f"  Δχ²     = {result_cc_lcdm.fun - result_cc_iam.fun:.2f}")
    print()
    
    # ========================================================================
    # BAO ONLY
    # ========================================================================
    print("="*80)
    print("BAO ONLY")
    print("="*80)
    
    print("Fitting ΛCDM to BAO only...")
    bounds_bao_lcdm = [(0.25, 0.35), (67.0, 73.5), (140.0, 155.0)]
    result_bao_lcdm = differential_evolution(
        chi2_bao_only_lcdm, bounds_bao_lcdm, seed=42, maxiter=1000, polish=True
    )
    print(f"  χ²_ΛCDM = {result_bao_lcdm.fun:.2f}")
    
    print("Fitting IAM to BAO only...")
    bounds_bao_iam = [(0.25, 0.35), (67.0, 73.5), (140.0, 155.0), (-0.15, 0.15)]
    result_bao_iam = differential_evolution(
        chi2_bao_only_iam, bounds_bao_iam, seed=42, maxiter=1000, polish=True
    )
    tau_bao = result_bao_iam.x[3]
    print(f"  χ²_IAM  = {result_bao_iam.fun:.2f}")
    print(f"  τ_act   = {tau_bao:+.4f}")
    print(f"  Δχ²     = {result_bao_lcdm.fun - result_bao_iam.fun:.2f}")
    print()
    
    # ========================================================================
    # JOINT FIT
    # ========================================================================
    print("="*80)
    print("JOINT CC + BAO")
    print("="*80)
    
    print("Fitting ΛCDM to joint dataset...")
    bounds_joint_lcdm = [(0.25, 0.35), (65.0, 75.0), (140.0, 155.0)]
    result_joint_lcdm = differential_evolution(
        chi2_joint_lcdm, bounds_joint_lcdm, seed=42, maxiter=2000, polish=True
    )
    Om0_lcdm, H0_lcdm, rd_lcdm = result_joint_lcdm.x
    chi2_joint_lcdm = result_joint_lcdm.fun
    
    print(f"  Ωm = {Om0_lcdm:.4f}")
    print(f"  H₀ = {H0_lcdm:.2f} km/s/Mpc")
    print(f"  rd = {rd_lcdm:.2f} Mpc")
    print(f"  χ² = {chi2_joint_lcdm:.2f}")
    print()
    
    print("Fitting IAM to joint dataset...")
    bounds_joint_iam = [(0.25, 0.35), (65.0, 75.0), (140.0, 155.0), (-0.15, 0.15)]
    result_joint_iam = differential_evolution(
        chi2_joint_iam, bounds_joint_iam, seed=42, maxiter=2000, polish=True
    )
    Om0_iam, H0_iam, rd_iam, tau_joint = result_joint_iam.x
    chi2_joint_iam = result_joint_iam.fun
    
    print(f"  Ωm      = {Om0_iam:.4f}")
    print(f"  H₀      = {H0_iam:.2f} km/s/Mpc")
    print(f"  rd      = {rd_iam:.2f} Mpc")
    print(f"  τ_act   = {tau_joint:+.4f}")
    print(f"  χ²      = {chi2_joint_iam:.2f}")
    print()
    
    delta_chi2_joint = chi2_joint_lcdm - chi2_joint_iam
    sigma_joint = np.sqrt(delta_chi2_joint) if delta_chi2_joint > 0 else 0
    
    print("="*80)
    print("JOINT FIT RESULTS")
    print("="*80)
    print(f"  χ²_ΛCDM = {chi2_joint_lcdm:.2f}")
    print(f"  χ²_IAM  = {chi2_joint_iam:.2f}")
    print(f"  Δχ²     = {delta_chi2_joint:.2f}")
    print(f"  Significance: ~{sigma_joint:.1f}σ")
    print()
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("="*80)
    print("τ_act COMPARISON ACROSS FITS")
    print("="*80)
    print(f"  CC only:        τ_act = {tau_cc:+.4f}")
    print(f"  BAO only:       τ_act = {tau_bao:+.4f}")
    print(f"  Joint CC+BAO:   τ_act = {tau_joint:+.4f}")
    print()
    
    if abs(tau_cc - tau_bao) > 0.05:
        print("⚠️  TENSION between CC and BAO!")
        print(f"   Difference: {abs(tau_cc - tau_bao):.4f}")
        print("   → Datasets prefer different actualization rates")
        print("   → Evidence for spatial backreaction or systematic differences")
    else:
        print("✓ CC and BAO agree on τ_act")
        print("   → Consistent actualization across datasets")
    
    print()
    
    if delta_chi2_joint > 5:
        print("✓✓ Joint fit shows SIGNIFICANT improvement with IAM")
    elif delta_chi2_joint > 1:
        print("✓ Joint fit shows modest improvement with IAM")
    else:
        print("✗ Joint fit shows no improvement with IAM")
    
    print()
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    
    expected_sum = (result_cc_lcdm.fun - result_cc_iam.fun) + \
                   (result_bao_lcdm.fun - result_bao_iam.fun)
    
    print(f"If datasets were independent:")
    print(f"  Expected Δχ² ≈ {expected_sum:.2f}")
    print(f"  Actual Δχ²   = {delta_chi2_joint:.2f}")
    print()
    
    if delta_chi2_joint < 0.5 * expected_sum:
        print("⚠️  Joint fit Δχ² MUCH LESS than sum of individual fits!")
        print("   → Strong degeneracies or dataset tension")
        print("   → Datasets prefer DIFFERENT physics")
    elif delta_chi2_joint > 1.5 * expected_sum:
        print("✓ Joint fit Δχ² GREATER than sum!")
        print("   → Datasets reinforce each other")
        print("   → Consistent IAM signal across observables")
    else:
        print("✓ Joint fit Δχ² consistent with sum of individual fits")
        print("   → Datasets provide independent constraints")
    
    print()
    print("="*80)
    
    # Save results
    np.savez('results/test_10_joint_results.npz',
             tau_cc=tau_cc,
             tau_bao=tau_bao,
             tau_joint=tau_joint,
             chi2_cc_lcdm=result_cc_lcdm.fun,
             chi2_cc_iam=result_cc_iam.fun,
             chi2_bao_lcdm=result_bao_lcdm.fun,
             chi2_bao_iam=result_bao_iam.fun,
             chi2_joint_lcdm=chi2_joint_lcdm,
             chi2_joint_iam=chi2_joint_iam)
    
    print("Results saved to results/test_10_joint_results.npz")
    print()
    print("="*80)
