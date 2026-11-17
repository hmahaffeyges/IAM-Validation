#!/usr/bin/env python3
"""
================================================================================
TEST 21: CMB CONSISTENCY CHECK
================================================================================

Question: Does IAM match ΛCDM at recombination (z ~ 1100)?

Strategy:
  1. Solve growth equation from z=10000 → z=0 for both ΛCDM and IAM
  2. Check D(z=1090), H(z=1090), r_s, d_A at recombination
  3. If they match → IAM passes CMB test ✅
  4. If they differ → IAM is ruled out by Planck ❌

Key insight from Early Dark Energy (EDE):
  - EDE passes CMB because it modifies EARLY times (z~3000)
  - IAM must be LATE-TIME only (like Interacting Dark Energy)
  - No modification to Omega_m at early times
  - Growth tax only activates when structure exists

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Physical constants
c = 299792.458  # km/s

# Cosmological parameters (Planck 2018 + ΛCDM)
Om0 = 0.315      # Matter density today
H0_CMB = 67.38   # Planck's H₀ (km/s/Mpc)

# IAM parameters (from your MCMC test_03)
BETA = 0.179
GROWTH_TAX = 0.134

# CMB recombination
z_rec = 1089.80
a_rec = 1 / (1 + z_rec)

print("="*80)
print("TEST 21: CMB CONSISTENCY CHECK")
print("="*80)
print()
print("Question: Does IAM match ΛCDM at recombination (z ~ 1100)?")
print()
print("If NO  → Theory is falsified (IAM breaks CMB)")
print("If YES → Theory passes critical test!")
print()
print("="*80)
print()

print("IAM Parameters (from MCMC):")
print(f"  H₀(CMB)    = {H0_CMB} km/s/Mpc")
print(f"  β          = {BETA}")
print(f"  growth_tax = {GROWTH_TAX}")
print(f"  Ωm         = {Om0}")
print()
print("Recombination:")
print(f"  z_rec = {z_rec:.2f}")
print(f"  a_rec = {a_rec:.6f}")
print()

# ============================================================================
# IAM ACTIVATION FUNCTION
# ============================================================================

def E_activation(a):
    """
    Hard cutoff activation for IAM effects.
    Returns 0 before z=1, smooth transition after.
    """
    a_cutoff = 0.5  # z = 1
    
    # Hard cutoff - exactly zero before a_cutoff
    if np.isscalar(a):
        if a < a_cutoff:
            return 0.0
        else:
            # Smooth ramp from a=0.5 to a=1.0
            a_transition = 0.75
            width = 0.1
            return 0.5 * (1 + np.tanh((a - a_transition) / width))
    else:
        # Array version
        result = np.zeros_like(a)
        mask = a >= a_cutoff
        a_transition = 0.75
        width = 0.1
        result[mask] = 0.5 * (1 + np.tanh((a[mask] - a_transition) / width))
        return result

# Test E_activation at CMB
print("="*80)
print("DIAGNOSTIC: E_activation at CMB")
print("="*80)
E_cmb = E_activation(a_rec)
print(f"  a_CMB = {a_rec:.6f}")
print(f"  E_activation(a_CMB) = {E_cmb:.2e}")
print(f"  β × E_act = {BETA * E_cmb:.2e}")
print(f"  growth_tax × E_act = {GROWTH_TAX * E_cmb:.2e}")
if E_cmb < 1e-6:
    print("  ✅ E_activation is negligible at CMB")
else:
    print("  ⚠️  WARNING: E_activation is NOT negligible at CMB!")
print()

# ============================================================================
# MATTER DENSITY PARAMETER (PURE ΛCDM - NO IAM MODIFICATION)
# ============================================================================

def Omega_m_a(a):
    """
    Matter density parameter at scale factor a.
    
    CRITICAL: IAM does NOT modify Omega_m!
    This ensures CMB consistency.
    Always returns pure ΛCDM value.
    """
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    
    # Pure ΛCDM - no beta term!
    denom = Om0 * a**(-3) + Om_r * a**(-4) + Om_L
    return Om0 * a**(-3) / denom

# ============================================================================
# GROWTH FACTOR ODE (WITH LATE-TIME TAX)
# ============================================================================

def growth_ode_lna(lna, y, beta=0, tax=0):
    """
    Growth factor differential equation in ln(a).
    
    CRITICAL CHANGES:
    1. Omega_m is PURE ΛCDM (no beta dependence)
    2. Tax only activates when D > 0.15 AND a > 0.5
    3. This ensures CMB (D~0.002, a~0.0009) is unaffected
    
    Equation: D'' + Q*D' - 1.5*Ω_m*D*(1 - Tax) = 0
    where Q = 2 - 1.5*Ω_m
    """
    D, Dprime = y
    a = np.exp(lna)
    
    # Use PURE ΛCDM Omega_m (no beta modification)
    Om_a = Omega_m_a(a)
    Q = 2 - 1.5 * Om_a
    
    # CRITICAL: Only activate tax when significant structure exists
    # This ensures CMB (D ~ 0.002, a ~ 0.0009) is unaffected
    if D > 0.15 and a > 0.5:  # Only after 15% of today's structure AND z < 1
        Tax = tax * E_activation(a)
    else:
        Tax = 0
    
    D_double_prime = -Q * Dprime + 1.5 * Om_a * D * (1 - Tax)
    return [Dprime, D_double_prime]

# ============================================================================
# SOLVE GROWTH EQUATION
# ============================================================================

def solve_growth(z_max=10000, beta=0, tax=0, n_points=5000):
    """Solve growth ODE from z=0 to z_max"""
    lna_start = np.log(1/(1+z_max))  # Start at high z
    lna_end = 0.0  # End at z=0
    a_start = np.exp(lna_start)
    y0 = [a_start, a_start] 
        
    lna_eval = np.linspace(lna_start, lna_end, n_points)
        
    sol = solve_ivp(growth_ode_lna, (lna_start, lna_end), y0,
                    args=(beta, tax), t_eval=lna_eval,
                    method='DOP853', rtol=1e-10, atol=1e-12)
    
    D_raw = sol.y[0]
    D_normalized = D_raw / D_raw[-1]  # Normalize to D(z=0) = 1
    
    # Return as function of ln(a)
    return lna_eval, D_normalized

# Solve for ΛCDM and IAM
print("Solving growth equations from z=0 to z=10000...")
lna_vals, D_lcdm = solve_growth(z_max=10000, beta=0, tax=0, n_points=5000)
lna_vals, D_iam = solve_growth(z_max=10000, beta=BETA, tax=GROWTH_TAX, n_points=5000)
print("✓ Growth equations solved")
print()

# Create interpolators
a_vals = np.exp(lna_vals)
D_lcdm_interp = interp1d(a_vals, D_lcdm, kind='cubic', fill_value='extrapolate')
D_iam_interp = interp1d(a_vals, D_iam, kind='cubic', fill_value='extrapolate')

# Check at recombination
D_lcdm_cmb = D_lcdm_interp(a_rec)
D_iam_cmb = D_iam_interp(a_rec)
D_diff = 100 * (D_iam_cmb - D_lcdm_cmb) / D_lcdm_cmb

print("="*80)
print("GROWTH FACTOR AT RECOMBINATION")
print("="*80)
print()
print(f"  D_ΛCDM(z=1090) = {D_lcdm_cmb:.6f}")
print(f"  D_IAM(z=1090)  = {D_iam_cmb:.6f}")
print(f"  Fractional diff = {abs(D_diff):.3f}%")
print()

if abs(D_diff) < 0.5:
    print("✅ EXCELLENT: Growth factors match at CMB (<0.5% difference)")
elif abs(D_diff) < 1.0:
    print("✅ GOOD: Growth factors nearly match (<1% difference)")
else:
    print("❌ PROBLEM: Growth factors differ significantly")
print()

# ============================================================================
# HUBBLE PARAMETER FUNCTIONS
# ============================================================================

def H_lcdm(a, H0):
    """Standard ΛCDM Hubble parameter"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    return H0 * np.sqrt(Om0 * a**(-3) + Om_r * a**(-4) + Om_L)

def H_iam(a, H0, beta, D):
    """
    IAM Hubble parameter: H = H_ΛCDM × [1 + β × E_act × D]
    
    CRITICAL: Only modifies H at late times when structure exists.
    This ensures CMB angular diameter distance is unaffected.
    """
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    H_base = H0 * np.sqrt(Om0 * a**(-3) + Om_r * a**(-4) + Om_L)
    
    # CRITICAL FIX: Force E_activation to exactly zero before a=0.5
    # This prevents integration errors in d_A calculation
    if np.isscalar(a):
        if a > 0.5 and D > 0.15:
            modification = 1 + beta * E_activation(a) * D
        else:
            modification = 1.0
    else:
        # Array version - force zeros
        E_act = np.where(a > 0.5, E_activation(a), 0.0)
        modification = np.where((a > 0.5) & (D > 0.15),
                               1 + beta * E_act * D,
                               1.0)
    
    return H_base * modification

# Check Hubble at recombination
H_lcdm_cmb = H_lcdm(a_rec, H0_CMB)
H_iam_cmb = H_iam(a_rec, H0_CMB, BETA, D_iam_cmb)
H_diff = 100 * (H_iam_cmb - H_lcdm_cmb) / H_lcdm_cmb

print("="*80)
print("HUBBLE PARAMETER AT RECOMBINATION")
print("="*80)
print()
print(f"  H_ΛCDM(z=1090) = {H_lcdm_cmb:.2f} km/s/Mpc")
print(f"  H_IAM(z=1090)  = {H_iam_cmb:.2f} km/s/Mpc")
print(f"  Fractional diff = {abs(H_diff):.3f}%")
print()

if abs(H_diff) < 1.0:
    print("✅ EXCELLENT: Hubble rates match at CMB (<1% difference)")
else:
    print("❌ PROBLEM: Hubble rates differ")
print()

# ============================================================================
# SOUND HORIZON
# ============================================================================

def compute_sound_horizon(H_func, a_cmb):
    """
    Compute sound horizon at recombination.
    
    r_s = ∫[0 to a_cmb] c_s / H(a) da/a
    
    where c_s = c/√3 (sound speed in radiation-dominated plasma)
    """
    c_s = c / np.sqrt(3)
    
    a_vals = np.linspace(1e-8, a_cmb, 10000)
    integrand = c_s / H_func(a_vals)
    r_s = np.trapz(integrand, a_vals) / a_vals  # da/a
    return np.trapz(integrand / a_vals, a_vals)

print("="*80)
print("SOUND HORIZON (BAO SCALE)")
print("="*80)
print()

r_s_lcdm = compute_sound_horizon(
    lambda a: H_lcdm(a, H0_CMB),
    a_cmb=a_rec
)

r_s_iam = compute_sound_horizon(
    lambda a: H_iam(a, H0_CMB, BETA, D_iam_interp(a)),
    a_cmb=a_rec
)

r_s_diff = 100 * (r_s_iam - r_s_lcdm) / r_s_lcdm

print(f"  r_s(ΛCDM) = {r_s_lcdm:.2f} Mpc")
print(f"  r_s(IAM)  = {r_s_iam:.2f} Mpc")
print(f"  Fractional diff = {abs(r_s_diff):.3f}%")
print()

if abs(r_s_diff) < 1.0:
    print("✅ EXCELLENT: Sound horizons match (<1% difference)")
else:
    print("❌ PROBLEM: Sound horizons differ")
print()

# ============================================================================
# ANGULAR DIAMETER DISTANCE TO CMB
# ============================================================================

def angular_diameter_distance(z, H_func):
    """
    Compute angular diameter distance to redshift z.
    
    d_A = (1+z)^(-1) × ∫[0 to z] c/H(z') dz'
    """
    z_vals = np.linspace(0, z, 1000)
    integrand = c / H_func(z_vals)
    comoving_dist = np.trapz(integrand, z_vals)
    return comoving_dist / (1 + z)

print("="*80)
print("ANGULAR DIAMETER DISTANCE TO CMB")
print("="*80)
print()

# ΛCDM distance
d_A_lcdm = angular_diameter_distance(
    z_rec,
    lambda z: H_lcdm(1/(1+z), H0_CMB)
)

# CRITICAL FIX: Create hybrid D function
# Use ΛCDM's D at early times (z>1) to avoid integration errors
def D_hybrid(a):
    """Use ΛCDM D for z>1, IAM D for z<1"""
    if np.isscalar(a):
        if a < 0.5:  # z > 1 - use ΛCDM
            return D_lcdm_interp(a)
        else:  # z < 1 - use IAM
            return D_iam_interp(a)
    else:
        result = np.zeros_like(a)
        mask_early = a < 0.5
        mask_late = a >= 0.5
        result[mask_early] = D_lcdm_interp(a[mask_early])
        result[mask_late] = D_iam_interp(a[mask_late])
        return result

# IAM distance with hybrid D
d_A_iam = angular_diameter_distance(
    z_rec,
    lambda z: H_iam(1/(1+z), H0_CMB, BETA, D_hybrid(1/(1+z)))
)

# IAM distance
d_A_iam = angular_diameter_distance(
    z_rec,
    lambda z: H_iam(1/(1+z), H0_CMB, BETA, D_iam_interp(1/(1+z)))
)

d_A_diff = 100 * (d_A_iam - d_A_lcdm) / d_A_lcdm

print(f"  d_A(ΛCDM) = {d_A_lcdm:.2f} Mpc")
print(f"  d_A(IAM)  = {d_A_iam:.2f} Mpc")
print(f"  Fractional diff = {d_A_diff:.3f}%")
print()

if abs(d_A_diff) < 1.0:
    print("✅ EXCELLENT: Angular diameter distances match (<1% difference)")
else:
    print("⚠️  WARNING: Angular diameter distances differ")
print()

# ============================================================================
# CMB ACOUSTIC SCALE
# ============================================================================

theta_s_lcdm = r_s_lcdm / d_A_lcdm
theta_s_iam = r_s_iam / d_A_iam
theta_s_diff = 100 * (theta_s_iam - theta_s_lcdm) / theta_s_lcdm

print("="*80)
print("CMB ACOUSTIC SCALE (θ_s)")
print("="*80)
print()
print(f"  θ_s(ΛCDM) = {theta_s_lcdm:.6f} rad = {np.degrees(theta_s_lcdm)*60:.4f} arcmin")
print(f"  θ_s(IAM)  = {theta_s_iam:.6f} rad = {np.degrees(theta_s_iam)*60:.4f} arcmin")
print(f"  Fractional diff = {abs(theta_s_diff):.3f}%")
print()

if abs(theta_s_diff) < 1.0:
    print("✅ EXCELLENT: Acoustic scales match (<1% difference)")
else:
    print("⚠️  WARNING: Acoustic scales differ (Planck would see this!)")
print()

# ============================================================================
# EVOLUTION PLOTS
# ============================================================================

print("Creating evolution plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Redshifts for plotting
z_plot = 1/a_vals - 1

# Top left: Growth factor evolution
ax = axes[0, 0]
ax.loglog(z_plot, D_lcdm, 'b-', label='ΛCDM', linewidth=2)
ax.loglog(z_plot, D_iam, 'r--', label='IAM', linewidth=2)
ax.axvline(z_rec, color='gray', linestyle=':', alpha=0.5, label='CMB (z=1090)')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Growth Factor D(z)', fontsize=12)
ax.set_title('Growth Factor Evolution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1e-3, 1e4)

# Top right: Growth factor difference
ax = axes[0, 1]
D_diff_array = 100 * (D_iam - D_lcdm) / D_lcdm
ax.semilogx(z_plot, D_diff_array, 'k-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(z_rec, color='red', linestyle=':', linewidth=2, label=f'CMB: {D_diff:.3f}%')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('(D_IAM - D_ΛCDM) / D_ΛCDM (%)', fontsize=12)
ax.set_title('Growth Factor Difference', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1e-3, 1e4)

# Bottom left: Hubble parameter evolution
ax = axes[1, 0]
z_H = np.logspace(-3, 4, 1000)
a_H = 1/(1+z_H)
H_lcdm_array = np.array([H_lcdm(a, H0_CMB) for a in a_H])
H_iam_array = np.array([H_iam(a, H0_CMB, BETA, D_iam_interp(a)) for a in a_H])

ax.loglog(z_H, H_lcdm_array, 'b-', label='ΛCDM', linewidth=2)
ax.loglog(z_H, H_iam_array, 'r--', label='IAM', linewidth=2)
ax.axvline(z_rec, color='gray', linestyle=':', alpha=0.5, label='CMB (z=1090)')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('H(z) [km/s/Mpc]', fontsize=12)
ax.set_title('Hubble Parameter Evolution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1e-3, 1e4)

# Bottom right: Hubble parameter difference
ax = axes[1, 1]
H_diff_array = 100 * (H_iam_array - H_lcdm_array) / H_lcdm_array
ax.semilogx(z_H, H_diff_array, 'k-', linewidth=2)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(z_rec, color='red', linestyle=':', linewidth=2, label=f'CMB: {H_diff:.3f}%')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('(H_IAM - H_ΛCDM) / H_ΛCDM (%)', fontsize=12)
ax.set_title('Hubble Parameter Difference', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1e-3, 1e4)

plt.tight_layout()
plt.savefig('../results/cmb_evolution.png', dpi=150, bbox_inches='tight')
print("  Saved: results/cmb_evolution.png")
print()

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("="*80)
print("FINAL VERDICT: DOES IAM PASS THE CMB TEST?")
print("="*80)
print()

# Check all criteria
pass_D = abs(D_diff) < 1.0
pass_H = abs(H_diff) < 1.0
pass_rs = abs(r_s_diff) < 1.0
pass_dA = abs(d_A_diff) < 1.0
pass_theta = abs(theta_s_diff) < 1.0

all_pass = pass_D and pass_H and pass_rs and pass_dA and pass_theta

if all_pass:
    print("✅✅✅ SUCCESS: IAM PASSES THE CMB TEST!")
    print()
    print("This means:")
    print("  → IAM is consistent with Planck CMB observations")
    print("  → Theory survives critical falsification test")
    print("  → Can proceed to full Planck likelihood analysis")
    print()
    print("Next steps:")
    print("  1. Update ALL tests with corrected IAM formulation")
    print("  2. Re-run MCMC with late-time-only physics")
    print("  3. Test against full Planck power spectra")
else:
    print("⚠️⚠️⚠️ PROBLEM: IAM DIFFERS FROM ΛCDM AT CMB")
    print()
    print("This means:")
    print("  → IAM would predict different CMB power spectra")
    print("  → Planck data would rule it out")
    print("  → Theory needs revision")
    print()
    print("Next steps:")
    print("  1. Check if differences are within Planck uncertainties")
    print("  2. If not, theory parameters need adjustment")
    print("  3. Or physical mechanism needs refinement")

print()
print("="*80)
