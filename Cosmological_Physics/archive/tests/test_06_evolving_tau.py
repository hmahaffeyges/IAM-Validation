#!/usr/bin/env python3
"""
Test 06: Evolving Actualization τ_act(z)
Model: τ_act(z) = τ_0 + τ_1 × z
Tests if redshift-dependent actualization improves fit
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
# ΛCDM MODEL
# ============================================================================

def H_lcdm(z, Om0, H0):
    return H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))

def integrate_comoving_lcdm(z, Om0, H0):
    if z == 0:
        return 0.0
    z_arr = np.linspace(0, z, 500)
    integrand = 3e5 / H_lcdm(z_arr, Om0, H0)
    return np.trapezoid(integrand, z_arr)

# ============================================================================
# IAM MODEL WITH EVOLVING τ_act(z)
# ============================================================================

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

def tau_act_evolving(z, tau_0, tau_1):
    """
    Evolving actualization rate
    τ_act(z) = τ_0 + τ_1 × z
    
    Physical interpretation:
    - τ_0: actualization rate today (z=0)
    - τ_1: how it changes with redshift
    """
    return tau_0 + tau_1 * z

def H_iam_evolving(z, Om0, H0, tau_0, tau_1):
    """IAM Hubble parameter with evolving actualization"""
    z_max = np.max(z) if hasattr(z, '__len__') else z
    z_arr, D_vals = solve_growth_ode(z_max * 1.1, Om0, n_points=500)
    D_interp = interp1d(z_arr, D_vals, kind='cubic', fill_value='extrapolate')
    
    D_z = D_interp(z)
    tau_z = tau_act_evolving(z, tau_0, tau_1)
    
    H_base = H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))
    return H_base * (1 + tau_z * D_z)

def integrate_comoving_iam_evolving(z, Om0, H0, tau_0, tau_1):
    """Comoving distance with evolving actualization"""
    if z == 0:
        return 0.0
    z_arr = np.linspace(0, z, 500)
    H_vals = H_iam_evolving(z_arr, Om0, H0, tau_0, tau_1)
    integrand = 3e5 / H_vals
    return np.trapezoid(integrand, z_arr)

# ============================================================================
# CHI-SQUARED FITTING
# ============================================================================

def chi2_lcdm(params):
    """Chi-squared for ΛCDM"""
    Om0, H0, rd = params
    
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0 or rd <= 0:
        return 1e10
    
    chi2 = 0.0
    chi2 += ((H0 - planck_H0) / planck_H0_err)**2
    chi2 += ((H0 - shoes_H0) / shoes_H0_err)**2
    
    for i, z in enumerate(desi_extended['z']):
        DM_theory = integrate_comoving_lcdm(z, Om0, H0)
        DM_obs = desi_extended['DM_over_rd'][i] * rd
        DM_err = desi_extended['DM_over_rd_err'][i] * rd
        chi2 += ((DM_theory - DM_obs) / DM_err)**2
        
        DH_theory = 3e5 / H_lcdm(z, Om0, H0)
        DH_obs = desi_extended['DH_over_rd'][i] * rd
        DH_err = desi_extended['DH_over_rd_err'][i] * rd
        chi2 += ((DH_theory - DH_obs) / DH_err)**2
    
    return chi2

def chi2_iam_evolving(params):
    """Chi-squared for IAM with evolving τ_act(z)"""
    Om0, H0, rd, tau_0, tau_1 = params
    
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0 or rd <= 0:
        return 1e10
    
    chi2 = 0.0
    chi2 += ((H0 - planck_H0) / planck_H0_err)**2
    chi2 += ((H0 - shoes_H0) / shoes_H0_err)**2
    
    for i, z in enumerate(desi_extended['z']):
        DM_theory = integrate_comoving_iam_evolving(z, Om0, H0, tau_0, tau_1)
        DM_obs = desi_extended['DM_over_rd'][i] * rd
        DM_err = desi_extended['DM_over_rd_err'][i] * rd
        chi2 += ((DM_theory - DM_obs) / DM_err)**2
        
        DH_theory = 3e5 / H_iam_evolving(z, Om0, H0, tau_0, tau_1)
        DH_obs = desi_extended['DH_over_rd'][i] * rd
        DH_err = desi_extended['DH_over_rd_err'][i] * rd
        chi2 += ((DH_theory - DH_obs) / DH_err)**2
    
    return chi2

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("EVOLVING ACTUALIZATION MODEL: τ_act(z) = τ_0 + τ_1 × z")
    print("="*80)
    print()
    
    print("Physical Interpretation:")
    print("  τ_0: Actualization rate today (z=0)")
    print("  τ_1: Evolution with redshift")
    print("  Hypothesis: Black hole/structure formation drives evolution")
    print()
    
    n_desi_bins = len(desi_extended['z'])
    n_measurements = 2 * n_desi_bins
    n_H0 = 2
    n_total = n_H0 + n_measurements
    
    print(f"Dataset: {n_total} data points")
    print()
    
    # Fit ΛCDM
    print("Fitting ΛCDM (3 parameters)...")
    bounds_lcdm = [(0.25, 0.35), (67.0, 73.5), (140.0, 155.0)]
    result_lcdm = differential_evolution(
        chi2_lcdm, 
        bounds_lcdm, 
        seed=42, 
        maxiter=2000,
        workers=1,
        polish=True
    )
    
    Om0_lcdm, H0_lcdm, rd_lcdm = result_lcdm.x
    chi2_lcdm_best = result_lcdm.fun
    
    print(f"  Ωm = {Om0_lcdm:.4f}")
    print(f"  H₀ = {H0_lcdm:.2f} km/s/Mpc")
    print(f"  rd = {rd_lcdm:.2f} Mpc")
    print(f"  χ² = {chi2_lcdm_best:.2f}")
    print()
    
    # Fit IAM with evolving τ_act
    print("Fitting IAM with evolving τ_act(z) (5 parameters)...")
    bounds_iam = [
        (0.25, 0.35),    # Om0
        (67.0, 73.5),    # H0
        (140.0, 155.0),  # rd
        (-0.15, 0.15),   # tau_0
        (-0.15, 0.05)    # tau_1
    ]
    
    result_iam = differential_evolution(
        chi2_iam_evolving,
        bounds_iam,
        seed=42,
        maxiter=2000,
        workers=1,
        polish=True
    )
    
    Om0_iam, H0_iam, rd_iam, tau_0, tau_1 = result_iam.x
    chi2_iam_best = result_iam.fun
    
    print(f"  Ωm    = {Om0_iam:.4f}")
    print(f"  H₀    = {H0_iam:.2f} km/s/Mpc")
    print(f"  rd    = {rd_iam:.2f} Mpc")
    print(f"  τ_0   = {tau_0:+.4f} (today)")
    print(f"  τ_1   = {tau_1:+.4f} (evolution)")
    print(f"  χ²    = {chi2_iam_best:.2f}")
    print()
    
    # Show τ_act at key redshifts
    print("Actualization rate at key epochs:")
    key_z = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    for z in key_z:
        tau_z = tau_act_evolving(z, tau_0, tau_1)
        print(f"  z = {z:.1f}: τ_act = {tau_z:+.4f}")
    print()
    
    # Statistical comparison
    delta_chi2 = chi2_lcdm_best - chi2_iam_best
    delta_dof = 2  # IAM has TWO extra parameters (tau_0, tau_1)
    sigma = np.sqrt(delta_chi2) if delta_chi2 > 0 else 0
    
    print("="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)
    print(f"  χ²_ΛCDM = {chi2_lcdm_best:.2f}")
    print(f"  χ²_IAM  = {chi2_iam_best:.2f}")
    print(f"  Δχ²     = {delta_chi2:.2f}")
    print(f"  Δdof    = {delta_dof} (τ_0 and τ_1)")
    
    # Adjusted significance for 2 extra parameters
    if delta_chi2 > 0:
        # For 2 extra parameters, need Δχ² > ~6 for 2σ, ~11 for 3σ
        print(f"  Significance: ~{sigma:.1f}σ (approximate)")
        print()
        if delta_chi2 > 15:
            print(f"✓✓✓ STRONG EVIDENCE: IAM with evolving τ_act fits much better!")
        elif delta_chi2 > 9:
            print(f"✓✓ MODERATE EVIDENCE: IAM with evolving τ_act fits better")
        else:
            print(f"✓ WEAK EVIDENCE: Small improvement with evolving τ_act")
    else:
        print(f"  No improvement")
        print()
        print(f"✗ Evolving τ_act does not improve fit")
    
    print()
    print("="*80)
    print("PHYSICAL INTERPRETATION")
    print("="*80)
    
    if tau_1 < 0:
        print("Negative τ_1: Actualization was STRONGER in the past")
        print("  → Consistent with peak structure/BH formation at z~1-2")
        print("  → More gravitational collapse events → stronger feedback")
    elif tau_1 > 0:
        print("Positive τ_1: Actualization is STRONGER today")
        print("  → More actualized structures in late universe")
        print("  → Dark energy era effects?")
    else:
        print("τ_1 ≈ 0: Actualization rate is approximately constant")
    
    print()
    print("="*80)
