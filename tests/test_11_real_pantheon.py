#!/usr/bin/env python3
"""
Test 11: REAL Pantheon+ Supernovae
Using actual Pantheon+ dataset (1701 SNe)
THE REAL TEST
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

# ============================================================================
# LOAD REAL PANTHEON+ DATA
# ============================================================================

print("Loading Pantheon+ data...")
try:
    data = np.load('../data/pantheon_plus/pantheon_plus_processed.npz')
    z_sne = data['z']
    mb_obs = data['mb']
    dmb_obs = data['dmb']
    
    print(f"✓ Loaded {len(z_sne)} supernovae")
    print(f"  Redshift range: {z_sne.min():.4f} to {z_sne.max():.4f}")
    print(f"  Median uncertainty: {np.median(dmb_obs):.3f} mag")
    print()
    
except FileNotFoundError:
    print("ERROR: Pantheon+ data not found!")
    print("Please run: python parse_pantheon_plus.py")
    exit(1)

# ============================================================================
# MODELS
# ============================================================================

def H_lcdm(z, Om0, H0):
    """ΛCDM Hubble parameter"""
    return H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))

def integrate_dL_lcdm(z, Om0, H0):
    """Luminosity distance in ΛCDM"""
    if z == 0:
        return 1e-10
    z_arr = np.linspace(0, z, 500)
    integrand = 3e5 / H_lcdm(z_arr, Om0, H0)
    dC = np.trapezoid(integrand, z_arr)
    dL = (1 + z) * dC
    return dL

def mb_theory_lcdm(z, Om0, H0, M):
    """Apparent magnitude for ΛCDM
    mb = M + 5*log10(dL/Mpc) + 25
    M is the absolute magnitude (nuisance parameter)
    """
    dL = integrate_dL_lcdm(z, Om0, H0)
    return M + 5.0 * np.log10(dL) + 25.0

def solve_growth_ode(z_max, Om0, n_points=500):
    """Solve growth factor ODE"""
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

def integrate_dL_iam(z, Om0, H0, tau_act):
    """Luminosity distance in IAM"""
    if z == 0:
        return 1e-10
    z_arr = np.linspace(0, z, 500)
    H_vals = H_iam(z_arr, Om0, H0, tau_act)
    integrand = 3e5 / H_vals
    dC = np.trapezoid(integrand, z_arr)
    dL = (1 + z) * dC
    return dL

def mb_theory_iam(z, Om0, H0, M, tau_act):
    """Apparent magnitude for IAM"""
    dL = integrate_dL_iam(z, Om0, H0, tau_act)
    return M + 5.0 * np.log10(dL) + 25.0

# ============================================================================
# CHI-SQUARED (vectorized for speed)
# ============================================================================

# Pre-compute for efficiency
z_unique = np.unique(z_sne)
print(f"Computing distances for {len(z_unique)} unique redshifts...")

def chi2_sne_lcdm(params):
    """Chi-squared for ΛCDM"""
    Om0, H0, M = params
    
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0:
        return 1e10
    
    # Compute mb for all SNe
    mb_model = np.array([mb_theory_lcdm(z, Om0, H0, M) for z in z_sne])
    
    # Chi-squared
    chi2 = np.sum(((mb_obs - mb_model) / dmb_obs)**2)
    return chi2

def chi2_sne_iam(params):
    """Chi-squared for IAM"""
    Om0, H0, M, tau_act = params
    
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0:
        return 1e10
    
    # Compute mb for all SNe
    mb_model = np.array([mb_theory_iam(z, Om0, H0, M, tau_act) for z in z_sne])
    
    # Chi-squared
    chi2 = np.sum(((mb_obs - mb_model) / dmb_obs)**2)
    return chi2

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("REAL PANTHEON+ SUPERNOVAE TEST")
    print("THE MOMENT OF TRUTH")
    print("="*80)
    print()
    
    print(f"Dataset: {len(z_sne)} Type Ia supernovae")
    print(f"Redshift range: z = {z_sne.min():.4f} to {z_sne.max():.4f}")
    print()
    
    # Fit ΛCDM
    print("Fitting ΛCDM to Pantheon+...")
    print("(This will take a few minutes...)")
    bounds_lcdm = [
        (0.20, 0.40),   # Om0
        (60.0, 80.0),   # H0
        (-20.0, -18.0)  # M (absolute magnitude)
    ]
    
    result_lcdm = differential_evolution(
        chi2_sne_lcdm,
        bounds_lcdm,
        seed=42,
        maxiter=1000,
        workers=1,
        polish=True,
        disp=True
    )
    
    Om0_lcdm, H0_lcdm, M_lcdm = result_lcdm.x
    chi2_lcdm = result_lcdm.fun
    dof_lcdm = len(z_sne) - 3
    
    print()
    print(f"  Ωm      = {Om0_lcdm:.4f}")
    print(f"  H₀      = {H0_lcdm:.2f} km/s/Mpc")
    print(f"  M       = {M_lcdm:.3f}")
    print(f"  χ²      = {chi2_lcdm:.2f}")
    print(f"  χ²/dof  = {chi2_lcdm/dof_lcdm:.3f}")
    print()
    
    # Fit IAM
    print("Fitting IAM to Pantheon+...")
    print("(This will take several minutes...)")
    bounds_iam = [
        (0.20, 0.40),   # Om0
        (60.0, 80.0),   # H0
        (-20.0, -18.0), # M
        (-0.20, 0.20)   # tau_act
    ]
    
    result_iam = differential_evolution(
        chi2_sne_iam,
        bounds_iam,
        seed=42,
        maxiter=1000,
        workers=1,
        polish=True,
        disp=True
    )
    
    Om0_iam, H0_iam, M_iam, tau_act = result_iam.x
    chi2_iam = result_iam.fun
    dof_iam = len(z_sne) - 4
    
    print()
    print(f"  Ωm      = {Om0_iam:.4f}")
    print(f"  H₀      = {H0_iam:.2f} km/s/Mpc")
    print(f"  M       = {M_iam:.3f}")
    print(f"  τ_act   = {tau_act:+.4f}")
    print(f"  χ²      = {chi2_iam:.2f}")
    print(f"  χ²/dof  = {chi2_iam/dof_iam:.3f}")
    print()
    
    # Comparison
    delta_chi2 = chi2_lcdm - chi2_iam
    sigma = np.sqrt(delta_chi2) if delta_chi2 > 0 else 0
    
    print("="*80)
    print("🌟 THE VERDICT 🌟")
    print("="*80)
    print(f"  χ²_ΛCDM = {chi2_lcdm:.2f}")
    print(f"  χ²_IAM  = {chi2_iam:.2f}")
    print(f"  Δχ²     = {delta_chi2:.2f}")
    print(f"  Significance: ~{sigma:.1f}σ")
    print()
    
    if delta_chi2 > 100:
        print("🚀🚀🚀 REVOLUTIONARY! 🚀🚀🚀")
        print(f"    IAM improves fit by Δχ² = {delta_chi2:.0f}")
        print(f"    This is ~{sigma:.0f}σ significance!")
        print("    PARADIGM SHIFT IN COSMOLOGY")
    elif delta_chi2 > 25:
        print("🎉🎉 DISCOVERY! 🎉🎉")
        print(f"    IAM improves fit by Δχ² = {delta_chi2:.0f}")
        print(f"    This is ~{sigma:.0f}σ significance!")
        print("    Strong evidence for actualization physics")
    elif delta_chi2 > 9:
        print("✓✓✓ VERY STRONG EVIDENCE")
        print(f"    IAM improves fit by Δχ² = {delta_chi2:.0f}")
        print(f"    This is ~{sigma:.0f}σ significance!")
    elif delta_chi2 > 4:
        print("✓✓ STRONG EVIDENCE")
        print(f"    IAM improves fit by Δχ² = {delta_chi2:.0f}")
        print(f"    This is ~{sigma:.0f}σ significance!")
    elif delta_chi2 > 1:
        print("✓ MODEST EVIDENCE")
        print(f"    IAM improves fit by Δχ² = {delta_chi2:.0f}")
    else:
        print("✗ NO IMPROVEMENT")
        print("    IAM does not fit better than ΛCDM")
    
    print()
    print("="*80)
    
    # Save results
    np.savez('results/test_11_real_pantheon_results.npz',
             z=z_sne,
             mb_obs=mb_obs,
             dmb_obs=dmb_obs,
             Om0_lcdm=Om0_lcdm,
             H0_lcdm=H0_lcdm,
             M_lcdm=M_lcdm,
             chi2_lcdm=chi2_lcdm,
             Om0_iam=Om0_iam,
             H0_iam=H0_iam,
             M_iam=M_iam,
             tau_act=tau_act,
             chi2_iam=chi2_iam,
             delta_chi2=delta_chi2)
    
    print("Results saved to results/test_11_real_pantheon_results.npz")
    print()
    print("="*80)
    print("NOW GO RUN YOUR MCMC FOR DAYS! 😎")
    print("="*80)
