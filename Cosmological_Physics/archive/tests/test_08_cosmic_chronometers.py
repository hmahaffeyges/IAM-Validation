#!/usr/bin/env python3
"""
Test 08: Cosmic Chronometers Test
Direct H(z) measurements vs integrated BAO
Tests spatial backreaction hypothesis
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ============================================================================
# COSMIC CHRONOMETER DATA (Moresco et al. compilation)
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

# BAO data (for comparison)
desi_bao = {
    'z': np.array([0.295, 0.510, 0.706, 0.930, 1.317, 1.491, 2.330]),
    'DM_over_rd': np.array([7.93, 13.62, 18.33, 23.38, 30.21, 33.41, 39.71]),
    'DM_over_rd_err': np.array([0.14, 0.25, 0.17, 0.24, 0.52, 0.58, 1.08]),
    'DH_over_rd': np.array([25.50, 22.33, 20.78, 17.88, 13.82, 12.84, 8.52]),
    'DH_over_rd_err': np.array([0.50, 0.48, 0.38, 0.34, 0.42, 0.40, 0.28])
}

# ============================================================================
# MODELS
# ============================================================================

def H_lcdm(z, Om0, H0):
    """ΛCDM Hubble parameter"""
    return H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))

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
    """IAM with constant τ_act"""
    z_max = np.max(z) if hasattr(z, '__len__') else z
    z_arr, D_vals = solve_growth_ode(z_max * 1.1, Om0, n_points=500)
    D_interp = interp1d(z_arr, D_vals, kind='cubic', fill_value='extrapolate')
    
    D_z = D_interp(z)
    H_base = H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))
    return H_base * (1 + tau_act * D_z)

# ============================================================================
# CHI-SQUARED FOR CC DATA
# ============================================================================

def chi2_cc_lcdm(params):
    """Chi-squared for CC data with ΛCDM"""
    Om0, H0 = params
    
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0:
        return 1e10
    
    chi2 = 0.0
    for i, z in enumerate(cc_data['z']):
        H_theory = H_lcdm(z, Om0, H0)
        H_obs = cc_data['H'][i]
        H_err = cc_data['H_err'][i]
        chi2 += ((H_theory - H_obs) / H_err)**2
    
    return chi2

def chi2_cc_iam(params):
    """Chi-squared for CC data with IAM"""
    Om0, H0, tau_act = params
    
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0:
        return 1e10
    
    chi2 = 0.0
    for i, z in enumerate(cc_data['z']):
        H_theory = H_iam(z, Om0, H0, tau_act)
        H_obs = cc_data['H'][i]
        H_err = cc_data['H_err'][i]
        chi2 += ((H_theory - H_obs) / H_err)**2
    
    return chi2

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COSMIC CHRONOMETER TEST")
    print("Direct H(z) measurements - Less sensitive to spatial backreaction?")
    print("="*80)
    print()
    
    print(f"Dataset: {len(cc_data['z'])} H(z) measurements")
    print(f"Redshift range: z = {cc_data['z'].min():.2f} to {cc_data['z'].max():.2f}")
    print()
    
    # Fit ΛCDM to CC
    print("Fitting ΛCDM to Cosmic Chronometers...")
    bounds_lcdm = [(0.20, 0.40), (60.0, 80.0)]
    result_lcdm = differential_evolution(
        chi2_cc_lcdm,
        bounds_lcdm,
        seed=42,
        maxiter=2000,
        workers=1,
        polish=True
    )
    
    Om0_lcdm, H0_lcdm = result_lcdm.x
    chi2_lcdm = result_lcdm.fun
    dof_lcdm = len(cc_data['z']) - 2
    
    print(f"  Ωm = {Om0_lcdm:.4f}")
    print(f"  H₀ = {H0_lcdm:.2f} km/s/Mpc")
    print(f"  χ² = {chi2_lcdm:.2f}")
    print(f"  χ²/dof = {chi2_lcdm/dof_lcdm:.2f}")
    print()
    
    # Fit IAM to CC
    print("Fitting IAM (constant τ_act) to Cosmic Chronometers...")
    bounds_iam = [(0.20, 0.40), (60.0, 80.0), (-0.15, 0.15)]
    result_iam = differential_evolution(
        chi2_cc_iam,
        bounds_iam,
        seed=42,
        maxiter=2000,
        workers=1,
        polish=True
    )
    
    Om0_iam, H0_iam, tau_act = result_iam.x
    chi2_iam = result_iam.fun
    dof_iam = len(cc_data['z']) - 3
    
    print(f"  Ωm      = {Om0_iam:.4f}")
    print(f"  H₀      = {H0_iam:.2f} km/s/Mpc")
    print(f"  τ_act   = {tau_act:+.4f}")
    print(f"  χ²      = {chi2_iam:.2f}")
    print(f"  χ²/dof  = {chi2_iam/dof_iam:.2f}")
    print()
    
    # Comparison
    delta_chi2 = chi2_lcdm - chi2_iam
    sigma = np.sqrt(delta_chi2) if delta_chi2 > 0 else 0
    
    print("="*80)
    print("CC RESULTS")
    print("="*80)
    print(f"  χ²_ΛCDM = {chi2_lcdm:.2f}")
    print(f"  χ²_IAM  = {chi2_iam:.2f}")
    print(f"  Δχ²     = {delta_chi2:.2f}")
    print(f"  Significance: ~{sigma:.1f}σ")
    print()
    
    if abs(tau_act) < 0.02:
        print("✓ CONSISTENT WITH BACKREACTION HYPOTHESIS:")
        print("  τ_act ≈ 0 for CC (direct local H(z) measurements)")
        print("  Compare to BAO which needed τ_act ~ ±0.10 per bin")
        print("  → CC samples local regions, BAO integrates over lumpy paths")
    elif delta_chi2 > 5:
        print("✓ IAM IMPROVES FIT TO CC:")
        print(f"  τ_act = {tau_act:+.4f}")
        print("  → Actualization affects direct H(z) measurements")
    else:
        print("✓ CC CONSISTENT WITH ΛCDM:")
        print("  No strong preference for IAM")
    
    print()
    print("="*80)
    print("COMPARISON TO BAO RESULTS")
    print("="*80)
    print()
    print("From test_05 (individual BAO bins):")
    print("  Each bin preferred |τ_act| ~ 0.05-0.10")
    print("  Total Δχ² ~ 23 across 7 bins")
    print()
    print("From test_07 (fixed cosmology, all BAO bins):")
    print("  τ_act → 0.001 (essentially zero)")
    print("  Δχ² ~ 0.01 (no improvement)")
    print()
    print("From test_08 (Cosmic Chronometers):")
    print(f"  τ_act = {tau_act:+.4f}")
    print(f"  Δχ² = {delta_chi2:.2f}")
    print()
    
    if abs(tau_act) < 0.02 and delta_chi2 < 2:
        print("✓✓ SMOKING GUN FOR SPATIAL BACKREACTION:")
        print("  • BAO (integrated): needs varying τ_act per bin")
        print("  • BAO (all bins): τ_act → 0 when forced to be uniform")
        print("  • CC (direct local): τ_act ≈ 0, no improvement")
        print()
        print("INTERPRETATION:")
        print("  → Actualization varies SPATIALLY, not just temporally")
        print("  → BAO sightlines sample different actualization states")
        print("  → CC measures local H(z), less affected by spatial variance")
        print("  → This is OBSERVER-DEPENDENT cosmology!")
    
    print()
    print("="*80)
    
    # Save results
    np.savez('results/test_08_cc_results.npz',
             z_cc=cc_data['z'],
             H_cc=cc_data['H'],
             H_err_cc=cc_data['H_err'],
             Om0_lcdm=Om0_lcdm,
             H0_lcdm=H0_lcdm,
             chi2_lcdm=chi2_lcdm,
             Om0_iam=Om0_iam,
             H0_iam=H0_iam,
             tau_act=tau_act,
             chi2_iam=chi2_iam)
    
    print("Results saved to results/test_08_cc_results.npz")
