#!/usr/bin/env python3
"""
Test 19: ACTUAL REAL PANTHEON+ DATA
Parse official Pantheon+SH0ES.dat and test IAM
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

# Parse the REAL Pantheon+ data
data_file = '../data/pantheon_repo/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'

print("="*80)
print("LOADING REAL PANTHEON+ DATA")
print("="*80)
print()

# Read data (skip header)
data = []
with open(data_file, 'r') as f:
    lines = f.readlines()[1:]  # Skip header
    for line in lines:
        parts = line.split()
        zCMB = float(parts[4])
        m_b = float(parts[8])
        m_b_err = float(parts[9])
        
        # Only use SNe in Hubble flow (z > 0.01, exclude calibrators)
        if zCMB > 0.01 and zCMB < 2.5:
            data.append([zCMB, m_b, m_b_err])

data = np.array(data)
z_sne = data[:, 0]
mb_obs = data[:, 1]
dmb_obs = data[:, 2]

print(f"Loaded {len(z_sne)} Type Ia supernovae")
print(f"Redshift range: {z_sne.min():.4f} to {z_sne.max():.4f}")
print(f"Median uncertainty: {np.median(dmb_obs):.3f} mag")
print()

# Models
def H_lcdm(z, Om0, H0):
    return H0 * np.sqrt(Om0 * (1+z)**3 + (1-Om0))

def dL_lcdm(z, Om0, H0):
    if z == 0: return 1e-10
    z_arr = np.linspace(0, z, 500)
    integrand = 3e5 / H_lcdm(z_arr, Om0, H0)
    dC = np.trapezoid(integrand, z_arr)
    return (1 + z) * dC

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

def dL_iam(z, Om0, H0, tau_act):
    if z == 0: return 1e-10
    z_arr = np.linspace(0, z, 500)
    H_vals = H_iam(z_arr, Om0, H0, tau_act)
    integrand = 3e5 / H_vals
    dC = np.trapezoid(integrand, z_arr)
    return (1 + z) * dC

# Planck priors
H0_planck = 67.4
H0_planck_err = 0.5

def chi2_lcdm(params):
    Om0, H0, M = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0: return 1e10
    
    mb_model = np.array([M + 5.0*np.log10(dL_lcdm(z, Om0, H0)) + 25.0 for z in z_sne])
    chi2 = np.sum(((mb_obs - mb_model) / dmb_obs)**2)
    chi2 += ((H0 - H0_planck) / H0_planck_err)**2
    return chi2

def chi2_iam(params):
    Om0, H0, M, tau_act = params
    if Om0 <= 0 or Om0 >= 1 or H0 <= 0: return 1e10
    
    mb_model = np.array([M + 5.0*np.log10(dL_iam(z, Om0, H0, tau_act)) + 25.0 for z in z_sne])
    chi2 = np.sum(((mb_obs - mb_model) / dmb_obs)**2)
    chi2 += ((H0 - H0_planck) / H0_planck_err)**2
    return chi2

print("Fitting ΛCDM (with Planck H₀ prior)...")
print("This will take ~5 minutes with full dataset...")
bounds_lcdm = [(0.20, 0.40), (60.0, 75.0), (-20.0, -18.0)]
result_lcdm = differential_evolution(chi2_lcdm, bounds_lcdm, seed=42,
                                     maxiter=500, polish=True, disp=True, workers=1)
Om0_lcdm, H0_lcdm, M_lcdm = result_lcdm.x
chi2_lcdm_val = result_lcdm.fun

print(f"\n  Ωm      = {Om0_lcdm:.4f}")
print(f"  H₀      = {H0_lcdm:.2f} km/s/Mpc")
print(f"  M       = {M_lcdm:.3f}")
print(f"  χ²      = {chi2_lcdm_val:.2f}")
print(f"  χ²/dof  = {chi2_lcdm_val/(len(z_sne)-3):.3f}")
print()

print("Fitting IAM (with Planck H₀ prior)...")
print("This will take ~10 minutes...")
bounds_iam = [(0.20, 0.40), (60.0, 75.0), (-20.0, -18.0), (-0.30, 0.30)]
result_iam = differential_evolution(chi2_iam, bounds_iam, seed=42,
                                    maxiter=500, polish=True, disp=True, workers=1)
Om0_iam, H0_iam, M_iam, tau_act = result_iam.x
chi2_iam_val = result_iam.fun

print(f"\n  Ωm      = {Om0_iam:.4f}")
print(f"  H₀      = {H0_iam:.2f} km/s/Mpc")
print(f"  M       = {M_iam:.3f}")
print(f"  τ_act   = {tau_act:+.4f}")
print(f"  χ²      = {chi2_iam_val:.2f}")
print(f"  χ²/dof  = {chi2_iam_val/(len(z_sne)-4):.3f}")
print()

delta_chi2 = chi2_lcdm_val - chi2_iam_val
sigma = np.sqrt(delta_chi2) if delta_chi2 > 0 else 0

print("="*80)
print("RESULTS WITH REAL PANTHEON+ DATA")
print("="*80)
print(f"  N_SNe   = {len(z_sne)}")
print(f"  χ²_ΛCDM = {chi2_lcdm_val:.2f}")
print(f"  χ²_IAM  = {chi2_iam_val:.2f}")
print(f"  Δχ²     = {delta_chi2:.2f}")
print(f"  Significance: ~{sigma:.1f}σ")
print()

if delta_chi2 > 25:
    print(f"🚀🚀🚀 DISCOVERY! IAM improves fit by {sigma:.1f}σ!")
elif delta_chi2 > 9:
    print(f"✓✓ STRONG: {sigma:.1f}σ improvement")
elif delta_chi2 > 4:
    print(f"✓ MODERATE: {sigma:.1f}σ")
else:
    print(f"No significant improvement (Δχ² = {delta_chi2:.1f})")

print()
print("="*80)
