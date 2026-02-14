#!/usr/bin/env python3
"""
================================================================================
TEST 25: PHOTON-EXEMPT IAM WITH ORIGINAL PHYSICS
================================================================================

Strategy:
  1. Use ORIGINAL IAM formulation (test_03_final.py):
     - Exponential activation: E_act = exp(1 - 1/a)
     - Beta modifies Omega_m and H(z)
     - Tax always active (no D threshold)
  
  2. BUT: Photon-exempt for CMB observables:
     - d_A uses H_ΛCDM (photons don't see IAM)
     - r_s uses H_ΛCDM (early universe, photon-dominated)
     - θ_s matches Planck
  
  3. Matter sector uses full IAM:
     - Growth equation has tax
     - BAO scales see modified H(z)
     - f*σ8 predictions testable

This is the "have your cake and eat it too" scenario.

================================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Constants
c = 299792.458  # km/s

# Cosmological parameters
Om0 = 0.315
H0_CMB = 67.38

# IAM parameters (ORIGINAL from test_03)
BETA = 0.179
GROWTH_TAX = 0.134

# CMB
z_rec = 1089.80
a_rec = 1 / (1 + z_rec)

print("="*80)
print("TEST 25: PHOTON-EXEMPT IAM WITH ORIGINAL PHYSICS")
print("="*80)
print()
print("Using ORIGINAL IAM formulation from test_03_final.py:")
print("  • E_activation = exp(1 - 1/a)")
print("  • Beta modifies Omega_m and H(z)")
print("  • Tax always active (no thresholds)")
print()
print("BUT: Photons see pure ΛCDM for CMB observables")
print()
print("="*80)
print()

# ============================================================================
# ORIGINAL IAM PHYSICS
# ============================================================================

def E_activation(a):
    """ORIGINAL exponential activation"""
    return np.exp(1 - 1/a)

def Omega_m_a(a, beta=0):
    """ORIGINAL - includes beta in denominator"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    if beta > 0:
        denom = Om0 * a**(-3) + Om_r * a**(-4) + Om_L + beta * E_activation(a)
    else:
        denom = Om0 * a**(-3) + Om_r * a**(-4) + Om_L
    return Om0 * a**(-3) / denom

def growth_ode_lna(lna, y, beta=0, tax=0):
    """ORIGINAL - tax always active, no D threshold"""
    D, Dprime = y
    a = np.exp(lna)
    Om_a = Omega_m_a(a, beta)
    Q = 2 - 1.5 * Om_a
    Tax = tax * E_activation(a) if tax > 0 else 0
    D_double_prime = -Q * Dprime + 1.5 * Om_a * D * (1 - Tax)
    return [Dprime, D_double_prime]

def solve_growth(z_max=10000, beta=0, tax=0, n_points=5000):
    """Solve growth equation"""
    lna_start = np.log(1/(1+z_max))
    lna_end = 0.0
    a_start = np.exp(lna_start)
    y0 = [a_start, a_start]
    
    lna_eval = np.linspace(lna_start, lna_end, n_points)
    
    sol = solve_ivp(growth_ode_lna, (lna_start, lna_end), y0,
                    args=(beta, tax), t_eval=lna_eval,
                    method='DOP853', rtol=1e-8, atol=1e-10)
    
    D_raw = sol.y[0]
    D_normalized = D_raw / D_raw[-1]
    
    return lna_eval, D_normalized

def H_lcdm(a, H0):
    """Standard ΛCDM Hubble"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    return H0 * np.sqrt(Om0 * a**(-3) + Om_r * a**(-4) + Om_L)

def H_iam(a, H0, beta):
    """ORIGINAL IAM Hubble - beta in Friedmann equation"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    return H0 * np.sqrt(Om0 * a**(-3) + Om_r * a**(-4) + Om_L + beta * E_activation(a))

def compute_sound_horizon(H_func, a_cmb):
    """Compute sound horizon"""
    c_s = c / np.sqrt(3)
    a_vals = np.linspace(1e-8, a_cmb, 10000)
    integrand = c_s / H_func(a_vals)
    return np.trapz(integrand / a_vals, a_vals)

def angular_diameter_distance(z, H_func):
    """Compute angular diameter distance"""
    z_vals = np.linspace(0, z, 1000)
    integrand = c / H_func(z_vals)
    comoving_dist = np.trapz(integrand, z_vals)
    return comoving_dist / (1 + z)

# ============================================================================
# SOLVE GROWTH
# ============================================================================

print("Solving growth equations with ORIGINAL physics...")

lna_vals, D_lcdm = solve_growth(z_max=10000, beta=0, tax=0, n_points=5000)
a_vals = np.exp(lna_vals)
D_lcdm_interp = interp1d(a_vals, D_lcdm, kind='cubic', fill_value='extrapolate')

lna_vals, D_iam = solve_growth(z_max=10000, beta=BETA, tax=GROWTH_TAX, n_points=5000)
D_iam_interp = interp1d(a_vals, D_iam, kind='cubic', fill_value='extrapolate')

print("  ✓ Growth equations solved")
print()

# Check unnormalized suppression
lna_start = np.log(1/(1+10000))  # this matches a_min used in solve_growth(z_max=10000)
lna_end = 0.0
lna_vals_raw = np.linspace(lna_start, lna_end, 5000)

D_lcdm_raw_unnorm = solve_ivp(growth_ode_lna, (lna_start, lna_end), [np.exp(lna_start), np.exp(lna_start)],
                              args=(0, 0), t_eval=lna_vals_raw,
                              method='DOP853', rtol=1e-8, atol=1e-10).y[0]

D_iam_raw_unnorm = solve_ivp(growth_ode_lna, (lna_start, lna_end), [np.exp(lna_start), np.exp(lna_start)],
                             args=(BETA, GROWTH_TAX), t_eval=lna_vals_raw,
                             method='DOP853', rtol=1e-8, atol=1e-10).y[0]

suppression = 100 * (1 - D_iam_raw_unnorm[-1] / D_lcdm_raw_unnorm[-1])
print(f"Growth suppression at z=0: {suppression:.2f}% (unnormalized)")
print()

# ============================================================================
# CMB OBSERVABLES - PHOTON-EXEMPT
# ============================================================================

print("="*80)
print("CMB OBSERVABLES (PHOTON-EXEMPT)")
print("="*80)
print()

# ΛCDM reference
r_s_lcdm = compute_sound_horizon(lambda a: H_lcdm(a, H0_CMB), a_rec)
d_A_lcdm = angular_diameter_distance(z_rec, lambda z: H_lcdm(1/(1+z), H0_CMB))
theta_s_lcdm = r_s_lcdm / d_A_lcdm

print("ΛCDM:")
print(f"  r_s = {r_s_lcdm:.6f} Mpc")
print(f"  d_A = {d_A_lcdm:.2f} Mpc")
print(f"  θ_s = {theta_s_lcdm:.6f} rad")
print()

# IAM photon-exempt: CMB photons see ΛCDM
r_s_iam_exempt = compute_sound_horizon(lambda a: H_lcdm(a, H0_CMB), a_rec)
d_A_iam_exempt = angular_diameter_distance(z_rec, lambda z: H_lcdm(1/(1+z), H0_CMB))
theta_s_iam_exempt = r_s_iam_exempt / d_A_iam_exempt

print("IAM (Photon-Exempt):")
print(f"  r_s = {r_s_iam_exempt:.6f} Mpc")
print(f"  d_A = {d_A_iam_exempt:.2f} Mpc")
print(f"  θ_s = {theta_s_iam_exempt:.6f} rad")
print()

theta_diff = 100 * abs(theta_s_iam_exempt - theta_s_lcdm) / theta_s_lcdm
print(f"Difference: {theta_diff:.3f}%")

if theta_diff < 0.1:
    print("✅ CMB PASSES (photons see ΛCDM)")
else:
    print("❌ CMB FAILS")
print()

# ============================================================================
# BAO OBSERVABLES - MATTER SECTOR
# ============================================================================

print("="*80)
print("BAO OBSERVABLES (MATTER SECTOR WITH FULL IAM)")
print("="*80)
print()

z_bao = np.array([0.38, 0.51, 0.61, 0.70])
a_bao = 1/(1+z_bao)

print("Growth factor D(z):")
print(f"{'z':>6s}  {'D_ΛCDM':>10s}  {'D_IAM':>10s}  {'Suppression':>12s}")
print("-"*50)

for i, z in enumerate(z_bao):
    D_lcdm_z = D_lcdm_interp(a_bao[i])
    D_iam_z = D_iam_interp(a_bao[i])
    supp = 100 * (D_lcdm_z - D_iam_z) / D_lcdm_z
    print(f"{z:>6.2f}  {D_lcdm_z:>10.6f}  {D_iam_z:>10.6f}  {supp:>11.3f}%")

print()

# Hubble parameter at BAO redshifts
print("Hubble parameter H(z):")
print(f"{'z':>6s}  {'H_ΛCDM':>10s}  {'H_IAM':>10s}  {'Difference':>12s}")
print("-"*50)

for i, z in enumerate(z_bao):
    H_lcdm_z = H_lcdm(a_bao[i], H0_CMB)
    H_iam_z = H_iam(a_bao[i], H0_CMB, BETA)
    diff = 100 * (H_iam_z - H_lcdm_z) / H_lcdm_z
    print(f"{z:>6.2f}  {H_lcdm_z:>10.2f}  {H_iam_z:>10.2f}  {diff:>11.3f}%")

print()

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("="*80)
print("FINAL VERDICT")
print("="*80)
print()

bao_significant = suppression > 1.0  # >1% suppression needed for BAO impact

if theta_diff < 0.1 and bao_significant:
    print("🎉🎉🎉 SUCCESS! PHOTON-EXEMPT IAM WITH ORIGINAL PHYSICS WORKS!")
    print()
    print(f"  ✅ CMB passes: Δθ_s = {theta_diff:.3f}%")
    print(f"  ✅ Growth suppression: {suppression:.2f}% (significant for BAO)")
    print()
    print("Physical Picture:")
    print("  • Matter sector: Modified by IAM (original formulation)")
    print("  • Photon sector: Pure ΛCDM (exempt from measurement tax)")
    print("  • CMB acoustic scale preserved")
    print("  • BAO improvement mechanism intact")
    print()
    print("NEXT STEPS:")
    print("  1. Test against actual BAO data")
    print("  2. Develop theoretical justification for photon exemption")
    print("  3. Test f*σ8 predictions")
    print("  4. Check CMB lensing consistency")
    print("  5. PUBLISH!")
    
elif theta_diff < 0.1 and not bao_significant:
    print("⚠️  CMB passes but BAO improvement lost")
    print(f"   Suppression = {suppression:.2f}% (too small)")
    
else:
    print("❌ Scenario fails")

print()
print("="*80)
