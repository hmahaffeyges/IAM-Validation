#!/usr/bin/env python3
"""
Test 27: CMB Lensing - FULLY CORRECTED
Key fix: theta_s = r_s_comoving / chi_comoving (NOT d_A physical)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

c = 299792.458
H0_CMB = 67.4
Om0 = 0.315
BETA = 0.18
GROWTH_TAX = 0.045
Om_r = 9.24e-5
Om_L = 1 - Om0 - Om_r

print("="*70)
print("TEST 27: CMB LENSING - FULLY CORRECTED")
print("="*70)
print()

# ============================================================================
# HUBBLE FUNCTIONS
# ============================================================================

def E_activation(a):
    return np.exp(1 - 1/a)

def H_LCDM(a):
    return H0_CMB * np.sqrt(Om0*a**(-3) + Om_r*a**(-4) + Om_L)

def H_IAM(a):
    return H0_CMB * np.sqrt(Om0*a**(-3) + Om_r*a**(-4) + Om_L + BETA*E_activation(a))

print("Hubble functions:")
print(f"  H_LCDM(a=1) = {H_LCDM(1.0):.2f} km/s/Mpc")
print(f"  H_IAM(a=1)  = {H_IAM(1.0):.2f} km/s/Mpc")
print()

# ============================================================================
# COMOVING DISTANCE (this is what matters for theta_s!)
# ============================================================================

def comoving_distance(z_source, H_func):
    """
    chi = integral of c/H(z) dz from 0 to z_source
    This is the COMOVING distance.
    """
    z_vals = np.linspace(0, z_source, 50000)
    a_vals = 1 / (1 + z_vals)
    integrand = c / H_func(a_vals)
    chi = np.trapezoid(integrand, z_vals)
    return chi

print("Computing comoving distances to CMB (z=1090)...")
chi_lcdm = comoving_distance(1090, H_LCDM)
chi_iam = comoving_distance(1090, H_IAM)

print(f"  LCDM: chi = {chi_lcdm:.2f} Mpc")
print(f"  IAM:  chi = {chi_iam:.2f} Mpc")
print(f"  Difference = {100*(chi_iam/chi_lcdm - 1):.3f}%")
print()

# ============================================================================
# ACOUSTIC SCALE - CORRECT FORMULA
# theta_s = r_s_comoving / chi_comoving
# ============================================================================

r_s = 144.43  # Mpc comoving (Planck 2018)
theta_planck = 0.0104110
sigma_theta = 0.0000031

theta_lcdm = r_s / chi_lcdm
theta_iam = r_s / chi_iam

print("="*70)
print("ACOUSTIC SCALE (correct formula: r_s / chi)")
print("="*70)
print()
print(f"  r_s (comoving) = {r_s:.2f} Mpc")
print(f"  LCDM: theta_s = {theta_lcdm:.8f} rad")
print(f"  IAM:  theta_s = {theta_iam:.8f} rad")
print(f"  Planck obs:   = {theta_planck:.8f} rad")
print()

diff_lcdm = (theta_lcdm - theta_planck) / theta_planck * 100
diff_iam = (theta_iam - theta_planck) / theta_planck * 100
sig_lcdm = abs(theta_lcdm - theta_planck) / sigma_theta
sig_iam = abs(theta_iam - theta_planck) / sigma_theta

print(f"  LCDM: {diff_lcdm:+.3f}% ({sig_lcdm:.1f} sigma)")
print(f"  IAM:  {diff_iam:+.3f}% ({sig_iam:.1f} sigma)")
print()

# ============================================================================
# GROWTH FACTORS
# ============================================================================

def Omega_m_a(a, beta=0):
    if beta > 0:
        denom = Om0*a**(-3) + Om_r*a**(-4) + Om_L + beta*E_activation(a)
    else:
        denom = Om0*a**(-3) + Om_r*a**(-4) + Om_L
    return Om0 * a**(-3) / denom

def growth_ode_lna(lna, y, beta=0, tax=0):
    D, Dprime = y
    a = np.exp(lna)
    Om_a = Omega_m_a(a, beta)
    Q = 2 - 1.5 * Om_a
    Tax = tax * E_activation(a) if tax > 0 else 0
    D_double_prime = -Q * Dprime + 1.5 * Om_a * D * (1 - Tax)
    return [Dprime, D_double_prime]

def solve_growth_unnorm(beta=0, tax=0):
    """Unnormalized growth - shows physical amplitude"""
    lna_start = np.log(0.001)
    lna_end = 0.0
    a_start = np.exp(lna_start)
    y0 = [a_start, a_start]
    lna_eval = np.linspace(lna_start, lna_end, 2000)
    sol = solve_ivp(growth_ode_lna, (lna_start, lna_end), y0,
                    args=(beta, tax), t_eval=lna_eval,
                    method='DOP853', rtol=1e-8, atol=1e-10)
    D_raw = sol.y[0]
    D_interp = interp1d(lna_eval, D_raw,
                        kind='cubic', fill_value='extrapolate')
    return D_interp

print("Solving growth equations...")
D_lcdm_raw = solve_growth_unnorm(beta=0, tax=0)
D_iam_raw = solve_growth_unnorm(beta=BETA, tax=GROWTH_TAX)

# Growth suppression table
print()
print("="*70)
print("GROWTH FACTOR (unnormalized - physical amplitude)")
print("="*70)
print()
print(f"{'z':>6} {'D_LCDM':>10} {'D_IAM':>10} {'Ratio':>10} {'Suppression':>12}")
print("-"*52)

z_test = [0, 0.3, 0.5, 1.0, 2.0]
suppressions_by_z = {}
for z in z_test:
    a = 1 / (1 + z)
    lna = np.log(a)
    D_l = D_lcdm_raw(lna)
    D_i = D_iam_raw(lna)
    ratio = D_i / D_l
    supp = 100 * (1 - ratio)
    suppressions_by_z[z] = supp
    print(f"{z:>6.1f} {D_l:>10.6f} {D_i:>10.6f} {ratio:>10.6f} {supp:>11.2f}%")

print()

# ============================================================================
# LENSING CONVERGENCE ESTIMATE
# ============================================================================

print("="*70)
print("LENSING CONVERGENCE ESTIMATE")
print("="*70)
print()

# Lensing convergence: kappa ~ integral of D²(z) * W(z) dz
# W(z) = geometric weight (peaks around z~1-2)
# We compute ratio kappa_IAM / kappa_LCDM

z_int = np.linspace(0.01, 10, 500)
a_int = 1 / (1 + z_int)
lna_int = np.log(a_int)

# Geometric weight W(z) for CMB lensing
chi_cmb = chi_lcdm  # Use LCDM as reference
chi_z = np.array([comoving_distance(z, H_LCDM) for z in z_int])
W_z = (chi_cmb - chi_z) / chi_cmb * chi_z / (1 + z_int)
W_z = np.maximum(W_z, 0)

# Growth factors along line of sight
D_lcdm_los = np.array([D_lcdm_raw(lna) for lna in lna_int])
D_iam_los = np.array([D_iam_raw(lna) for lna in lna_int])

# Lensing power ∝ D²
kappa_integrand_lcdm = W_z * D_lcdm_los**2
kappa_integrand_iam = W_z * D_iam_los**2

kappa_lcdm = np.trapezoid(kappa_integrand_lcdm, z_int)
kappa_iam = np.trapezoid(kappa_integrand_iam, z_int)

kappa_ratio = kappa_iam / kappa_lcdm
lensing_suppression = 100 * (1 - kappa_ratio)

print(f"  Lensing integral (LCDM): {kappa_lcdm:.6e}")
print(f"  Lensing integral (IAM):  {kappa_iam:.6e}")
print(f"  Ratio kappa_IAM/kappa_LCDM = {kappa_ratio:.6f}")
print(f"  Lensing suppression = {lensing_suppression:.3f}%")
print()

# Show where lensing difference comes from
print("Lensing integrand by redshift bin:")
z_bins = [(0, 1), (1, 3), (3, 10)]
for z_lo, z_hi in z_bins:
    mask = (z_int >= z_lo) & (z_int < z_hi)
    k_l = np.trapezoid(kappa_integrand_lcdm[mask], z_int[mask])
    k_i = np.trapezoid(kappa_integrand_iam[mask], z_int[mask])
    frac_l = 100 * k_l / kappa_lcdm
    frac_i = 100 * k_i / kappa_iam
    print(f"  z={z_lo}-{z_hi}: LCDM {frac_l:.1f}%, IAM {frac_i:.1f}%")

print()

# ============================================================================
# FINAL ASSESSMENT
# ============================================================================

print("="*70)
print("FINAL ASSESSMENT")
print("="*70)
print()

print(f"chi change (IAM vs LCDM):         {100*(chi_iam/chi_lcdm - 1):+.3f}%")
print(f"theta_s change (IAM vs LCDM):     {100*(theta_iam/theta_lcdm - 1):+.3f}%")
print(f"Lensing suppression:              {lensing_suppression:+.3f}%")
print()
print(f"LCDM theta_s discrepancy:  {diff_lcdm:+.3f}% ({sig_lcdm:.1f} sigma)")
print(f"IAM theta_s discrepancy:   {diff_iam:+.3f}% ({sig_iam:.1f} sigma)")
print()

# Can lensing compensate?
theta_s_shift = abs(diff_iam - diff_lcdm)
print(f"theta_s shift due to IAM:  {theta_s_shift:.3f}%")
print(f"Lensing compensation:      {abs(lensing_suppression):.3f}%")
print()

if abs(lensing_suppression) >= theta_s_shift:
    print("RESULT: LENSING FULLY COMPENSATES!")
    print("        CMB naturally consistent with IAM!")
elif abs(lensing_suppression) >= 0.5 * theta_s_shift:
    print("RESULT: LENSING PARTIALLY COMPENSATES (>50%)")
    print("        Dual-sector parameterization completes the picture")
else:
    print(f"RESULT: LENSING PROVIDES {100*abs(lensing_suppression)/theta_s_shift:.0f}% compensation")
    print(f"        Remaining gap: {theta_s_shift - abs(lensing_suppression):.3f}%")
    print("        Photon-exempt scenario still needed for full consistency")

print()
print("="*70)
print("Test complete.")
print("="*70)
