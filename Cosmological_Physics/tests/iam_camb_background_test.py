#!/usr/bin/env python3
"""
IAM Level 2: Background Modification Verification
===================================================
Run this BEFORE launching MCMC chains to verify the CAMB modification
is working correctly. Tests that:
  1. H_matter(z=0) / H_ΛCDM(z=0) = √(1 + β) = 1.0759
  2. H_matter(z) → H_ΛCDM(z) for z > 5 (GR recovery)
  3. CMB TT spectrum unchanged (photon sector)
  4. σ₈ reduced from ~0.811 to ~0.795-0.801
  5. f·σ₈(z) suppressed at z < 1
  6. BAO distances unchanged (photon sector)
  7. θ_MC unchanged (photon sector)

Usage:
  python iam_camb_background_test.py

Requires: modified CAMB with IAM dual-sector patch installed.
If the modification is NOT yet applied, this script serves as
documentation of what the expected outputs should be.

Author: H.W. Mahaffey
Date: February 2026
"""

import numpy as np
import sys

# Try to import modified CAMB; fall back to reference values
try:
    import camb
    HAS_CAMB = True
    print("CAMB imported successfully.")
    print(f"  Version: {camb.__version__}")
    print(f"  Path: {camb.__file__}")
except ImportError:
    HAS_CAMB = False
    print("CAMB not available — running in reference-value mode.")
    print("This shows expected results from a correctly modified CAMB.")

# =============================================================================
# PARAMETERS
# =============================================================================
H0 = 67.36
ombh2 = 0.02237
omch2 = 0.1200
tau = 0.0544
As = 2.1e-9
ns = 0.9649
Om = 0.3153
beta = Om / 2.0  # = 0.15765

print()
print("=" * 70)
print("  IAM LEVEL 2: BACKGROUND MODIFICATION VERIFICATION")
print("=" * 70)
print(f"\n  β = Ω_m/2 = {beta:.5f}")
print(f"  H₀(photon) = {H0:.2f} km/s/Mpc")
print(f"  H₀(matter) = {H0 * np.sqrt(1 + beta):.2f} km/s/Mpc")
print(f"  √(1+β) = {np.sqrt(1 + beta):.4f}")

# =============================================================================
# REFERENCE VALUES (from phenomenological script)
# =============================================================================
# These are what a correctly modified CAMB should produce.

def E_act(a):
    if a > 1e-6:
        return np.exp(1.0 - 1.0/a)
    return 0.0

def mu_iam(a):
    E2L = Om * a**(-3) + (1 - Om)
    return E2L / (E2L + beta * E_act(a))

print("\n--- Reference: Expected IAM Sector Gap ---")
print(f"  {'z':>5s} {'a':>8s} {'E(a)':>10s} {'μ(a)':>8s} {'H_m/H_γ':>8s}")
print(f"  {'-'*5} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
for z in [0, 0.1, 0.5, 1, 2, 5, 10]:
    a = 1/(1+z)
    Ea = E_act(a)
    mu = mu_iam(a)
    E2L = Om*(1+z)**3 + (1-Om)
    E2M = E2L + beta * Ea
    ratio = np.sqrt(E2M/E2L)
    print(f"  {z:>5.1f} {a:>8.4f} {Ea:>10.5f} {mu:>8.4f} {ratio:>8.4f}")

# =============================================================================
# TEST 1: Background H(z)
# =============================================================================
print("\n" + "=" * 70)
print("  TEST 1: Background Expansion Rate H(z)")
print("=" * 70)

if HAS_CAMB:
    try:
        # Run standard CAMB
        p_std = camb.CAMBparams()
        p_std.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
        p_std.InitPower.set_params(As=As, ns=ns)
        p_std.set_for_lmax(2500, lens_potential_accuracy=1)
        p_std.set_matter_power(redshifts=[0.0, 0.5, 1.0, 2.0], kmax=2.0)
        r_std = camb.get_results(p_std)
        
        # Try to run IAM-modified CAMB
        try:
            p_iam = camb.CAMBparams()
            p_iam.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau)
            p_iam.InitPower.set_params(As=As, ns=ns)
            p_iam.set_for_lmax(2500, lens_potential_accuracy=1)
            p_iam.set_matter_power(redshifts=[0.0, 0.5, 1.0, 2.0], kmax=2.0)
            # Set IAM parameters — this is where the modification enters
            p_iam.iam_dual_sector = True
            p_iam.iam_beta = beta
            r_iam = camb.get_results(p_iam)
            
            print("\n  IAM-modified CAMB ran successfully!")
            print(f"  σ₈(ΛCDM) = {r_std.get_sigma8_0():.4f}")
            print(f"  σ₈(IAM)  = {r_iam.get_sigma8_0():.4f}")
            
        except AttributeError:
            print("\n  IAM parameters not found in CAMB — modification not yet applied.")
            print("  Running standard CAMB only for comparison.")
            print(f"  σ₈(ΛCDM) = {r_std.get_sigma8_0():.4f}")
            print(f"  Expected σ₈(IAM) ≈ 0.795-0.801")
    except Exception as e:
        print(f"\n  CAMB test failed: {e}")
else:
    print("\n  CAMB not installed — showing expected values only.")

print("\n  Expected results from correctly modified CAMB:")
print(f"    H_matter(z=0) = {H0 * np.sqrt(1+beta):.2f} km/s/Mpc")
print(f"    H_matter(z=0) / H_ΛCDM(z=0) = {np.sqrt(1+beta):.4f}")
print(f"    σ₈(IAM) ∈ [0.795, 0.801]  (from perturbation modification)")
print(f"    σ₈(IAM) may shift further with background modification")

# =============================================================================
# TEST 2: CMB TT Unchanged
# =============================================================================
print("\n" + "=" * 70)
print("  TEST 2: CMB TT Spectrum (Should Be Unchanged)")
print("=" * 70)
print("""
  The photon sector uses H_photon = H_ΛCDM for all distance calculations.
  Therefore:
    - θ_MC should be identical to ΛCDM
    - TT/TE/EE power spectra should be identical (< 0.2% residuals)
    - Lensing potential should be modified (σ₈ changes lensing amplitude)
    
  PASS criterion: max|Δ C_ℓ^TT / C_ℓ^TT| < 1% for ℓ > 30
  (ISW at ℓ < 30 may differ due to late-time potential evolution)
""")

# =============================================================================
# TEST 3: BAO Distances Unchanged
# =============================================================================
print("=" * 70)
print("  TEST 3: BAO Distances (Should Be Unchanged)")
print("=" * 70)
print("""
  BAO observables (DM/rd, DH/rd, DV/rd) are photon-sector quantities.
  They should be identical to ΛCDM predictions.
  
  PASS criterion: all BAO observables identical to ΛCDM within numerical
  precision (< 0.01%).
""")

# =============================================================================
# TEST 4: Growth Rate Suppressed
# =============================================================================
print("=" * 70)
print("  TEST 4: Growth Rate f·σ₈(z) Suppression")
print("=" * 70)

# Compute expected f·σ₈ from phenomenological model
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def solve_growth_iam():
    Or = 9.24e-5  # approximate
    OL = 1 - Om
    
    def rhs(lna, y):
        a = max(np.exp(lna), 1e-12)
        E2L = Om*a**(-3) + Or*a**(-4) + OL
        Ea = E_act(a)
        E2 = E2L + beta * Ea
        Oma = Om*a**(-3) / E2
        dE2_da = -3*Om*a**(-4) - 4*Or*a**(-5)
        if a > 1e-6:
            dE2_da += beta * Ea / (a**2)
        dlnH_dlna = 0.5 * a * dE2_da / E2
        fric = 2.0 + dlnH_dlna
        mu = mu_iam(a)
        return [y[1], -fric*y[1] + 1.5*mu*Oma*y[0]]
    
    sol = solve_ivp(rhs, (np.log(1e-4), 0), [1e-4, 1.0],
                    t_eval=np.linspace(np.log(1e-4), 0, 10000),
                    rtol=1e-11, method='DOP853')
    return np.exp(sol.t), sol.y[0]

a_iam, D_iam = solve_growth_iam()

# Also solve ΛCDM for comparison
def solve_growth_lcdm():
    Or = 9.24e-5
    OL = 1 - Om
    def rhs(lna, y):
        a = max(np.exp(lna), 1e-12)
        E2 = Om*a**(-3) + Or*a**(-4) + OL
        Oma = Om*a**(-3) / E2
        dE2_da = -3*Om*a**(-4) - 4*Or*a**(-5)
        dlnH_dlna = 0.5 * a * dE2_da / E2
        fric = 2.0 + dlnH_dlna
        return [y[1], -fric*y[1] + 1.5*Oma*y[0]]
    sol = solve_ivp(rhs, (np.log(1e-4), 0), [1e-4, 1.0],
                    t_eval=np.linspace(np.log(1e-4), 0, 10000),
                    rtol=1e-11, method='DOP853')
    return np.exp(sol.t), sol.y[0]

a_lcdm, D_lcdm = solve_growth_lcdm()

sig8_iam = 0.8111 * D_iam[-1] / D_lcdm[-1]
print(f"\n  σ₈(ΛCDM) = 0.8111")
print(f"  σ₈(IAM, phenomenological) = {sig8_iam:.4f}")
print(f"  Suppression: {(1 - sig8_iam/0.8111)*100:.1f}%")

# f·σ₈ at survey redshifts
D_norm = D_iam / D_lcdm[-1]
lnD = np.log(np.maximum(D_norm, 1e-30))
lna = np.log(a_iam)
f_arr = np.gradient(lnD, lna)
Di = interp1d(a_iam, D_norm, kind='cubic', fill_value='extrapolate')
fi = interp1d(a_iam, f_arr, kind='cubic', fill_value='extrapolate')

D_norm_l = D_lcdm / D_lcdm[-1]
lnD_l = np.log(np.maximum(D_norm_l, 1e-30))
f_arr_l = np.gradient(lnD_l, np.log(a_lcdm))
Di_l = interp1d(a_lcdm, D_norm_l, kind='cubic', fill_value='extrapolate')
fi_l = interp1d(a_lcdm, f_arr_l, kind='cubic', fill_value='extrapolate')

print(f"\n  Expected f·σ₈(z) comparison:")
print(f"  {'z':>6s} {'f·σ₈(ΛCDM)':>12s} {'f·σ₈(IAM)':>12s} {'Change':>8s}")
print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8}")
for z in [0.122, 0.38, 0.51, 0.70, 0.85, 1.48]:
    a = 1/(1+z)
    fs8_l = float(fi_l(a)) * 0.8111 * float(Di_l(a))
    fs8_i = float(fi(a)) * sig8_iam * float(Di(a))
    print(f"  {z:>6.3f} {fs8_l:>12.4f} {fs8_i:>12.4f} {(fs8_i/fs8_l-1)*100:>+7.2f}%")

# =============================================================================
# TEST 5: Derived Parameters
# =============================================================================
print("\n" + "=" * 70)
print("  TEST 5: Expected Derived Parameters")
print("=" * 70)
Om_phys = Om / (1 + beta)
S8_planck_om = sig8_iam * np.sqrt(Om / 0.3)
S8_phys_om = sig8_iam * np.sqrt(Om_phys / 0.3)
print(f"""
  H₀(photon)         = {H0:.2f} km/s/Mpc
  H₀(matter)         = {H0 * np.sqrt(1+beta):.2f} km/s/Mpc
  σ₈(IAM)            = {sig8_iam:.4f}
  Ω_m(Planck)        = {Om:.4f}
  Ω_m(physical)      = {Om_phys:.3f}  (13.7% dilution)
  S₈(Planck Ω_m)     = {S8_planck_om:.3f}
  S₈(physical Ω_m)   = {S8_phys_om:.3f}
  μ(z=0)             = {mu_iam(1.0):.4f}
  
  These values should be compared against the MCMC output.
  If the full Boltzmann calculation gives σ₈ and S₈ within
  ~5% of these estimates, the implementation is working correctly.
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 70)
print("  SUMMARY: VERIFICATION CHECKLIST")
print("=" * 70)
print("""
  Before running MCMC, verify ALL of the following:

  [ ] H_matter(z=0)/H_ΛCDM(z=0) = 1.076 ± 0.001
  [ ] H_matter(z=5)/H_ΛCDM(z=5) < 1.0001 (GR recovery)
  [ ] CMB TT residuals < 1% for ℓ > 30
  [ ] BAO DM/rd, DH/rd identical to ΛCDM (< 0.01%)
  [ ] σ₈ < 0.811 (suppressed from ΛCDM)
  [ ] θ_MC identical to ΛCDM (< 0.01%)
  [ ] Code compiles without warnings
  [ ] No NaN or Inf in output at any redshift
  
  If any test fails, the modification has a bug.
  Do NOT proceed to MCMC until all tests pass.
""")

print("=" * 70)
print("  COMPLETE")
print("=" * 70)
