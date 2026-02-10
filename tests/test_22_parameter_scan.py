#!/usr/bin/env python3
"""
================================================================================
TEST 22: CMB-COMPATIBLE PARAMETER SCAN
================================================================================

Question: Is there a β > 0 that satisfies BOTH CMB and BAO?

Strategy:
  1. Binary search for maximum β that keeps |Δθ_s| < 0.1%
  2. Test that β's BAO performance
  3. Determine if IAM is viable or fundamentally incompatible

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
GROWTH_TAX = 0.134

# CMB recombination
z_rec = 1089.80
a_rec = 1 / (1 + z_rec)

print("="*80)
print("TEST 22: FINDING CMB-COMPATIBLE IAM PARAMETERS")
print("="*80)
print()
print("Goal: Find maximum β that satisfies CMB constraint |Δθ_s| < 0.1%")
print()

# ============================================================================
# COPY FUNCTIONS FROM TEST 21
# ============================================================================

def E_activation(a):
    """Hard cutoff activation at a=0.5"""
    a_cutoff = 0.5
    
    if np.isscalar(a):
        if a < a_cutoff:
            return 0.0
        else:
            a_transition = 0.75
            width = 0.1
            return 0.5 * (1 + np.tanh((a - a_transition) / width))
    else:
        result = np.zeros_like(a)
        mask = a >= a_cutoff
        a_transition = 0.75
        width = 0.1
        result[mask] = 0.5 * (1 + np.tanh((a[mask] - a_transition) / width))
        return result

def Omega_m_a(a):
    """Pure ΛCDM matter density"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    denom = Om0 * a**(-3) + Om_r * a**(-4) + Om_L
    return Om0 * a**(-3) / denom

def growth_ode_lna(lna, y, beta=0, tax=0):
    """Growth ODE with late-time tax"""
    D, Dprime = y
    a = np.exp(lna)
    Om_a = Omega_m_a(a)
    Q = 2 - 1.5 * Om_a
    
    if D > 0.15 and a > 0.5:
        Tax = tax * E_activation(a)
    else:
        Tax = 0
    
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
                    method='DOP853', rtol=1e-10, atol=1e-12)
    
    D_raw = sol.y[0]
    D_normalized = D_raw / D_raw[-1]
    
    return lna_eval, D_normalized

def H_lcdm(a, H0):
    """Standard ΛCDM Hubble"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    return H0 * np.sqrt(Om0 * a**(-3) + Om_r * a**(-4) + Om_L)

def H_iam(a, H0, beta, D):
    """IAM Hubble with hard cutoff"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    H_base = H0 * np.sqrt(Om0 * a**(-3) + Om_r * a**(-4) + Om_L)
    
    if np.isscalar(a):
        if a > 0.5 and D > 0.15:
            modification = 1 + beta * E_activation(a) * D
        else:
            modification = 1.0
    else:
        modification = np.ones_like(a)
        mask = (a > 0.5) & (D > 0.15)
        if np.any(mask):
            modification[mask] = 1 + beta * E_activation(a[mask]) * D[mask]
    
    return H_base * modification

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

def compute_theta_s(beta, D_interp):
    """Compute acoustic scale θ_s for given beta and growth function"""
    
    # Sound horizon
    r_s = compute_sound_horizon(
        lambda a: H_iam(a, H0_CMB, beta, D_interp(a)),
        a_cmb=a_rec
    )
    
    # Angular diameter distance
    d_A = angular_diameter_distance(
        z_rec,
        lambda z: H_iam(1/(1+z), H0_CMB, beta, D_interp(1/(1+z)))
    )
    
    return r_s / d_A

# ============================================================================
# SOLVE ΛCDM REFERENCE
# ============================================================================

print("Solving ΛCDM reference...")
lna_vals, D_lcdm = solve_growth(z_max=10000, beta=0, tax=0, n_points=5000)
a_vals = np.exp(lna_vals)
D_lcdm_interp = interp1d(a_vals, D_lcdm, kind='cubic', fill_value='extrapolate')

# Compute ΛCDM theta_s
r_s_lcdm = compute_sound_horizon(lambda a: H_lcdm(a, H0_CMB), a_rec)
d_A_lcdm = angular_diameter_distance(z_rec, lambda z: H_lcdm(1/(1+z), H0_CMB))
theta_s_lcdm = r_s_lcdm / d_A_lcdm

print(f"  θ_s(ΛCDM) = {theta_s_lcdm:.6f} rad")
print()

# ============================================================================
# BINARY SEARCH FOR MAXIMUM CMB-COMPATIBLE β
# ============================================================================

def test_beta(beta, verbose=False):
    """Test a specific beta value, return theta_s fractional error"""
    
    # Solve growth
    lna_vals, D_iam = solve_growth(z_max=10000, beta=beta, tax=GROWTH_TAX, n_points=5000)
    a_vals = np.exp(lna_vals)
    D_iam_interp = interp1d(a_vals, D_iam, kind='cubic', fill_value='extrapolate')
    
    # Compute theta_s
    theta_s_iam = compute_theta_s(beta, D_iam_interp)
    
    error = abs(theta_s_iam - theta_s_lcdm) / theta_s_lcdm
    
    if verbose:
        print(f"  β = {beta:.4f}")
        print(f"  θ_s(IAM) = {theta_s_iam:.6f} rad")
        print(f"  |Δθ_s| = {100*error:.3f}%")
    
    return error

print("="*80)
print("BINARY SEARCH FOR MAXIMUM CMB-COMPATIBLE β")
print("="*80)
print()
print("Target: |Δθ_s| < 0.1% (to be within Planck precision)")
print()

beta_low = 0.0
beta_high = 0.179  # Current MCMC value
target_error = 0.001  # 0.1%
tolerance = 0.001  # Search precision

iteration = 0
max_iterations = 20

print(f"Starting binary search: β ∈ [{beta_low:.4f}, {beta_high:.4f}]")
print()

# Test bounds first
print("Testing upper bound (β from BAO MCMC):")
error_high = test_beta(beta_high, verbose=True)
print()

if error_high < target_error:
    print("🎉 EXCELLENT! Current β already satisfies CMB!")
    print(f"   β = {beta_high:.4f} is CMB-compatible")
    beta_max = beta_high
else:
    print(f"❌ Current β = {beta_high:.4f} violates CMB (Δθ_s = {100*error_high:.3f}%)")
    print()
    print("Searching for maximum compatible β...")
    print()
    
    while (beta_high - beta_low > tolerance) and (iteration < max_iterations):
        beta_mid = (beta_low + beta_high) / 2
        error_mid = test_beta(beta_mid)
        
        iteration += 1
        print(f"Iteration {iteration}: β = {beta_mid:.4f}, |Δθ_s| = {100*error_mid:.3f}%", end="")
        
        if error_mid < target_error:
            # Can increase beta
            beta_low = beta_mid
            print(" → can increase β")
        else:
            # Must decrease beta
            beta_high = beta_mid
            print(" → must decrease β")
    
    beta_max = beta_low
    print()
    print("="*80)
    print(f"RESULT: Maximum CMB-compatible β = {beta_max:.4f}")
    print("="*80)

# ============================================================================
# TEST MAXIMUM β IN DETAIL
# ============================================================================

print()
print("="*80)
print("DETAILED TEST OF MAXIMUM CMB-COMPATIBLE β")
print("="*80)
print()

lna_vals, D_iam_max = solve_growth(z_max=10000, beta=beta_max, tax=GROWTH_TAX, n_points=5000)
a_vals = np.exp(lna_vals)
D_iam_max_interp = interp1d(a_vals, D_iam_max, kind='cubic', fill_value='extrapolate')

# CMB observables
D_lcdm_cmb = D_lcdm_interp(a_rec)
D_iam_max_cmb = D_iam_max_interp(a_rec)

r_s_iam_max = compute_sound_horizon(
    lambda a: H_iam(a, H0_CMB, beta_max, D_iam_max_interp(a)),
    a_rec
)

d_A_iam_max = angular_diameter_distance(
    z_rec,
    lambda z: H_iam(1/(1+z), H0_CMB, beta_max, D_iam_max_interp(1/(1+z)))
)

theta_s_iam_max = r_s_iam_max / d_A_iam_max

print(f"β_max = {beta_max:.4f} (vs β_BAO = 0.179)")
print()
print("CMB Observables:")
print(f"  D(z=1090):    ΛCDM = {D_lcdm_cmb:.6f}, IAM = {D_iam_max_cmb:.6f}, " +
      f"diff = {100*abs(D_iam_max_cmb-D_lcdm_cmb)/D_lcdm_cmb:.3f}%")
print(f"  r_s:          ΛCDM = {r_s_lcdm:.6f} Mpc, IAM = {r_s_iam_max:.6f} Mpc, " +
      f"diff = {100*abs(r_s_iam_max-r_s_lcdm)/r_s_lcdm:.3f}%")
print(f"  d_A:          ΛCDM = {d_A_lcdm:.2f} Mpc, IAM = {d_A_iam_max:.2f} Mpc, " +
      f"diff = {100*(d_A_iam_max-d_A_lcdm)/d_A_lcdm:.3f}%")
print(f"  θ_s:          ΛCDM = {theta_s_lcdm:.6f} rad, IAM = {theta_s_iam_max:.6f} rad, " +
      f"diff = {100*abs(theta_s_iam_max-theta_s_lcdm)/theta_s_lcdm:.3f}%")
print()

# ============================================================================
# ESTIMATE BAO PERFORMANCE
# ============================================================================

print("="*80)
print("ESTIMATED BAO PERFORMANCE WITH β_max")
print("="*80)
print()

# Compare late-time growth suppression
z_bao = np.array([0.38, 0.51, 0.61])  # Typical BAO redshifts
a_bao = 1/(1+z_bao)

print("Growth factor at BAO redshifts:")
print(f"{'z':>6s}  {'D_ΛCDM':>10s}  {'D_IAM(β=0.179)':>15s}  {'D_IAM(β_max)':>15s}")
print("-"*60)

for i, z in enumerate(z_bao):
    D_lcdm_z = D_lcdm_interp(a_bao[i])
    
    # Solve with original beta
    _, D_iam_orig = solve_growth(z_max=10000, beta=0.179, tax=GROWTH_TAX, n_points=5000)
    D_iam_orig_interp = interp1d(a_vals, D_iam_orig, kind='cubic', fill_value='extrapolate')
    D_iam_orig_z = D_iam_orig_interp(a_bao[i])
    
    D_iam_max_z = D_iam_max_interp(a_bao[i])
    
    print(f"{z:>6.2f}  {D_lcdm_z:>10.6f}  {D_iam_orig_z:>15.6f}  {D_iam_max_z:>15.6f}")

print()

# Rough BAO impact estimate
# BAO χ² improvement scales roughly with (growth suppression)²
growth_suppression_orig = 1 - D_iam_orig_interp(1.0) / D_lcdm_interp(1.0)
growth_suppression_max = 1 - D_iam_max_interp(1.0) / D_lcdm_interp(1.0)

bao_impact_ratio = (growth_suppression_max / growth_suppression_orig)**2 if growth_suppression_orig > 0 else 0

print(f"Growth suppression at z=0:")
print(f"  β = 0.179:  {100*growth_suppression_orig:.2f}%")
print(f"  β = {beta_max:.4f}: {100*growth_suppression_max:.2f}%")
print()
print(f"Estimated BAO improvement retention: {100*bao_impact_ratio:.1f}%")
print("(Rough estimate - actual MCMC needed for precise value)")
print()

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("="*80)
print("FINAL VERDICT")
print("="*80)
print()

if beta_max > 0.01:
    print(f"✅ CMB-COMPATIBLE IAM EXISTS!")
    print()
    print(f"   Maximum β = {beta_max:.4f}")
    print(f"   Original β = 0.179 (from BAO fit)")
    print(f"   Reduction factor: {beta_max/0.179:.2f}×")
    print()
    print("   This means:")
    print("   → IAM can satisfy CMB constraints")
    print("   → Some BAO improvement may be retained")
    print("   → Theory is NOT ruled out by Planck")
    print()
    print("   NEXT STEPS:")
    print("   1. Re-run full MCMC with CMB prior")
    print("   2. Test against actual BAO data with β_max")
    print("   3. Compute full CMB power spectrum (CAMB/CLASS)")
    print("   4. If BAO improvement survives, publish!")
else:
    print(f"❌ NO CMB-COMPATIBLE β FOUND")
    print()
    print(f"   Maximum β ≈ {beta_max:.4f} (essentially zero)")
    print()
    print("   This means:")
    print("   → IAM effects must be negligible to pass CMB")
    print("   → No BAO improvement possible")
    print("   → Theory is fundamentally incompatible with Planck")
    print()
    print("   OPTIONS:")
    print("   1. Revise theoretical framework")
    print("   2. Find alternative activation mechanism")
    print("   3. Exempt photon sector (needs justification)")

print()
print("="*80)
