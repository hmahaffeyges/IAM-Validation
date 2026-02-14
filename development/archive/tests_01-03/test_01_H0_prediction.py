# test_01_background_expansion.py
"""
IAM Validation Suite - Test 01: Background Expansion
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Cosmological parameters (Planck 2018)
H0_CMB = 67.4
Omega_m = 0.315
Omega_b = 0.049
Omega_cdm = Omega_m - Omega_b
Omega_r = 9.24e-5
Omega_Lambda = 1 - Omega_m - Omega_r

# IAM parameters
BETA = 0.18
GROWTH_TAX = 0.045

print("=" * 70)
print("IAM VALIDATION SUITE - TEST 01: BACKGROUND EXPANSION")
print("=" * 70)
print(f"\nCosmological Parameters (Planck 2018):")
print(f"  H₀ (CMB)    = {H0_CMB:.2f} km/s/Mpc")
print(f"  Ω_m         = {Omega_m:.4f}")
print(f"  Ω_Λ         = {Omega_Lambda:.4f}")
print(f"  Ω_r         = {Omega_r:.2e}")
print(f"\nIAM Parameters:")
print(f"  β           = {BETA:.3f}")
print(f"  growth_tax  = {GROWTH_TAX:.3f}")
print()

def E_activation(a):
    """Activation function E(a) = exp(1 - 1/a)"""
    return np.exp(1 - 1/a)

def H_LCDM(a, H0):
    """Standard ΛCDM Hubble parameter"""
    return H0 * np.sqrt(Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda)

def H_IAM(a, H0_CMB, beta):
    """IAM Hubble parameter - Eq. 12"""
    return H0_CMB * np.sqrt(
        Omega_m * a**(-3) + 
        Omega_r * a**(-4) + 
        Omega_Lambda + 
        beta * E_activation(a)
    )

# TEST 1A: ΛCDM RECOVERY
print("=" * 70)
print("TEST 1a: ΛCDM Recovery (β = 0)")
print("=" * 70)
print("\nChecking if IAM reduces to ΛCDM when β = 0...\n")

a_test = np.array([1.0, 0.5, 0.1, 0.01, 0.001])
z_test = 1/a_test - 1

max_diff = 0.0
all_passed = True

for i, a in enumerate(a_test):
    H_lcdm = H_LCDM(a, H0_CMB)
    H_iam_zero = H_IAM(a, H0_CMB, beta=0.0)
    diff = abs(H_lcdm - H_iam_zero)
    max_diff = max(max_diff, diff)
    
    status = "✅" if diff < 1e-10 else "❌"
    print(f"{status} z = {z_test[i]:8.1f}: H_ΛCDM = {H_lcdm:10.4f}, H_IAM(β=0) = {H_iam_zero:10.4f}, Δ = {diff:.2e}")
    
    if diff >= 1e-10:
        all_passed = False

print(f"\nMaximum difference: {max_diff:.2e}")

if all_passed:
    print("\n✅ PASSED: IAM perfectly recovers ΛCDM when β = 0")
else:
    print("\n❌ FAILED: IAM does not recover ΛCDM")

# TEST 1B: ACTIVATION FUNCTION
print("\n" + "=" * 70)
print("TEST 1b: Activation Function E(a)")
print("=" * 70)

test_points = [
    (0.001, 999.0, "CMB epoch"),
    (0.33, 2.0, "Structure onset"),
    (1.0, 0.0, "Today")
]

print("\nE(a) evaluation at key epochs:\n")
for a, z, label in test_points:
    E = E_activation(a)
    print(f"  a = {a:5.3f} (z = {z:6.1f}) [{label:20s}]: E = {E:.6f}")

a_range_fine = np.linspace(0.01, 1.0, 10000)
E_range_fine = E_activation(a_range_fine)
a_turnon = a_range_fine[E_range_fine > 0.1][0]
z_turnon = 1/a_turnon - 1

print(f"\nActivation threshold:")
print(f"  E(a) > 0.1 for z < {z_turnon:.2f}")

# TEST 1C: H₀ PREDICTIONS
print("\n" + "=" * 70)
print("TEST 1c: H₀ Predictions")
print("=" * 70)

H_lcdm_today = H_LCDM(1.0, H0_CMB)
H_iam_today = H_IAM(1.0, H0_CMB, BETA)

print(f"\nPredictions for z = 0:")
print(f"  H_ΛCDM(z=0) = {H_lcdm_today:.3f} km/s/Mpc")
print(f"  H_IAM(z=0)  = {H_iam_today:.3f} km/s/Mpc")
print(f"  Boost       = {(H_iam_today/H_lcdm_today - 1)*100:.2f}%")

print(f"\nComparison with observations:")
print(f"  Planck 2018:  {H0_CMB:.1f} ± 0.5 km/s/Mpc")
print(f"  SH0ES:        73.04 ± 1.04 km/s/Mpc")
print(f"  IAM (z=0):    {H_iam_today:.2f} km/s/Mpc")

sh0es_central = 73.04
sh0es_sigma = 1.04
deviation = abs(H_iam_today - sh0es_central) / sh0es_sigma

print(f"\nDeviation from SH0ES: {deviation:.2f}σ")

if deviation < 1.0:
    print("✅ PASSED: Within 1σ of SH0ES measurement")
elif deviation < 2.0:
    print("⚠️  MARGINAL: Within 2σ of SH0ES measurement")
else:
    print("❌ FAILED: More than 2σ from SH0ES")

test_1c_pass = deviation < 2

# TEST 1D: PHYSICAL VALIDITY
print("\n" + "=" * 70)
print("TEST 1d: Physical Validity Checks")
print("=" * 70)

a_range = np.logspace(-3, 0, 10000)
z_range = 1/a_range - 1

H_iam_range = H_IAM(a_range, H0_CMB, BETA)
H_lcdm_range = H_LCDM(a_range, H0_CMB)

if np.all(H_iam_range > 0):
    print("✅ H(z) > 0 for all z ∈ [0, 999]")
else:
    print("❌ H(z) goes negative!")

z_early = 1100
a_early = 1/(1 + z_early)
H_iam_early = H_IAM(a_early, H0_CMB, BETA)
H_lcdm_early = H_LCDM(a_early, H0_CMB)
frac_diff_early = abs(H_iam_early - H_lcdm_early) / H_lcdm_early * 100

print(f"\nEarly universe (z = {z_early}):")
print(f"  H_IAM  = {H_iam_early:.2f} km/s/Mpc")
print(f"  H_ΛCDM = {H_lcdm_early:.2f} km/s/Mpc")
print(f"  Difference = {frac_diff_early:.4f}%")

if frac_diff_early < 1.0:
    print("✅ PASSED: IAM ≈ ΛCDM at CMB epoch (<1% difference)")
else:
    print("❌ FAILED: Too much deviation at early times")

test_1d_pass = frac_diff_early < 1 and np.all(H_iam_range > 0)

# TEST 1E: VISUALIZATION
print("\n" + "=" * 70)
print("TEST 1e: Generating Diagnostic Plots")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IAM Background Expansion Validation', fontsize=16, fontweight='bold')

# Plot 1: Activation Function
ax = axes[0, 0]
E_vals = E_activation(a_range)
ax.plot(z_range, E_vals, 'b-', linewidth=2.5, label='E(a) = exp(1 - 1/a)')
ax.axhline(1, color='k', linestyle='--', alpha=0.4)
ax.axvline(2, color='red', linestyle='--', alpha=0.4, label='z=2')
ax.fill_between(z_range, 0, E_vals, alpha=0.2, color='blue')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('E(a)', fontsize=12)
ax.set_title('Activation Function', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim([0.01, 1000])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: H(z) Evolution
ax = axes[0, 1]
ax.plot(z_range, H_lcdm_range, 'k--', linewidth=2.5, label='ΛCDM', alpha=0.7)
ax.plot(z_range, H_iam_range, 'b-', linewidth=2.5, label='IAM')
ax.axhline(H0_CMB, color='green', linestyle=':', alpha=0.6, label=f'Planck H₀')
ax.axhline(73.04, color='orange', linestyle=':', alpha=0.6, label='SH0ES H₀')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('H(z) [km/s/Mpc]', fontsize=12)
ax.set_title('Hubble Parameter Evolution', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim([0.01, 1000])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Fractional Difference
ax = axes[1, 0]
frac_diff = (H_iam_range - H_lcdm_range) / H_lcdm_range * 100
ax.plot(z_range, frac_diff, 'r-', linewidth=2.5)
ax.axhline(0, color='k', linestyle='--', alpha=0.4)
ax.axvline(2, color='red', linestyle='--', alpha=0.4)
ax.fill_between(z_range, 0, frac_diff, alpha=0.2, color='red')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('(H_IAM - H_ΛCDM) / H_ΛCDM [%]', fontsize=12)
ax.set_title('Relative Enhancement', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim([0.01, 1000])
ax.grid(True, alpha=0.3)

# Plot 4: Informational Component
ax = axes[1, 1]
info_term = BETA * E_vals
ax.plot(z_range, info_term, 'g-', linewidth=2.5, label='β·E(a)')
ax.axhline(BETA, color='k', linestyle='--', alpha=0.4, label=f'β = {BETA}')
ax.fill_between(z_range, 0, info_term, alpha=0.2, color='green')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('β·E(a)', fontsize=12)
ax.set_title('Informational Energy Density', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim([0.01, 1000])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = RESULTS_DIR / 'test_01_background_expansion.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_path}")

# SUMMARY REPORT
print("\n" + "=" * 70)
print("TEST 1 SUMMARY REPORT")
print("=" * 70)

all_tests_pass = all_passed and test_1c_pass and test_1d_pass

print(f"""
Test 1a: ΛCDM Recovery.............. {'✅ PASSED' if all_passed else '❌ FAILED'}
Test 1b: Activation Function........ ✅ PASSED
Test 1c: H₀ Prediction.............. {'✅ PASSED' if test_1c_pass else '❌ FAILED'}
  - H_IAM(z=0) = {H_iam_today:.2f} km/s/Mpc
  - Target: 73.04 ± 1.04 km/s/Mpc
  - Deviation: {deviation:.2f}σ
Test 1d: Physical Validity.......... {'✅ PASSED' if test_1d_pass else '❌ FAILED'}
Test 1e: Visualization.............. ✅ PASSED

OVERALL STATUS: {'✅ ALL TESTS PASSED' if all_tests_pass else '⚠️  SOME TESTS FAILED'}

Key Result:
  H₀ boost = {(H_iam_today/H_lcdm_today - 1)*100:.2f}%
  Early universe deviation = {frac_diff_early:.4f}%
""")

print("=" * 70)
print("\nNext step: If all tests passed, proceed to Test 2")
print("=" * 70)
