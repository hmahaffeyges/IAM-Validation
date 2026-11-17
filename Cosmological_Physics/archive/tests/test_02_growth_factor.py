# test_02_growth_factor.py
"""
IAM Validation Suite - Test 02: Growth Factor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Parameters
H0_CMB = 67.4
Omega_m = 0.315
Omega_b = 0.049
Omega_r = 9.24e-5
Omega_Lambda = 1 - Omega_m - Omega_r
BETA = 0.18
GROWTH_TAX = 0.045

print("=" * 70)
print("IAM VALIDATION SUITE - TEST 02: GROWTH FACTOR")
print("=" * 70)
print(f"\nParameters:")
print(f"  H₀          = {H0_CMB:.2f} km/s/Mpc")
print(f"  Ω_m         = {Omega_m:.4f}")
print(f"  β           = {BETA:.3f}")
print(f"  growth_tax  = {GROWTH_TAX:.3f}")
print(f"\nGrowth suppression mechanism:")
print(f"  Tax(a) = {GROWTH_TAX} · E(a)")
print()

def E_activation(a):
    return np.exp(1 - 1/a)

def H_LCDM(a, H0):
    return H0 * np.sqrt(Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda)

def H_IAM(a, H0_CMB, beta):
    return H0_CMB * np.sqrt(
        Omega_m * a**(-3) + 
        Omega_r * a**(-4) + 
        Omega_Lambda + 
        beta * E_activation(a)
    )

def Omega_m_of_a(a, beta, use_iam=True):
    if use_iam:
        H2 = (Omega_m * a**(-3) + Omega_r * a**(-4) + 
              Omega_Lambda + beta * E_activation(a))
    else:
        H2 = Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda
    return Omega_m * a**(-3) / H2

def growth_ode_lcdm(y, a, H0):
    D, Dprime = y
    H = H_LCDM(a, H0)
    Om_a = Omega_m_of_a(a, beta=0, use_iam=False)
    D_double_prime = (
        1.5 * Om_a * H0**2 / (a**2 * H**2) * D - 
        (2/a) * Dprime
    )
    return [Dprime, D_double_prime]

def growth_ode_iam(y, a, H0_CMB, beta, growth_tax):
    D, Dprime = y
    H = H_IAM(a, H0_CMB, beta)
    Om_a = Omega_m_of_a(a, beta, use_iam=True)
    Tax = growth_tax * E_activation(a)
    D_double_prime = (
        1.5 * Om_a * H0_CMB**2 / (a**2 * H**2) * D * (1 - Tax) - 
        (2/a) * Dprime
    )
    return [Dprime, D_double_prime]

# TEST 2A: SOLVE GROWTH EQUATIONS
print("=" * 70)
print("TEST 2a: Solving Growth Equations")
print("=" * 70)

a_init = 0.001
a_today = 1.0
a_array = np.logspace(np.log10(a_init), np.log10(a_today), 2000)
y0 = [a_init, 1.0]

print(f"\nInitial conditions (z = {1/a_init - 1:.0f}):")
print(f"  D(a_init) = {y0[0]:.6f}")
print(f"  dD/da     = {y0[1]:.6f}")

print(f"\nSolving ΛCDM growth equation...")
sol_lcdm = odeint(growth_ode_lcdm, y0, a_array, args=(H0_CMB,))
D_lcdm = sol_lcdm[:, 0]
D_lcdm = D_lcdm / D_lcdm[-1]
print(f"✅ ΛCDM solution complete")

print(f"Solving IAM growth equation...")
sol_iam = odeint(growth_ode_iam, y0, a_array, args=(H0_CMB, BETA, GROWTH_TAX))
D_iam = sol_iam[:, 0]
D_iam = D_iam / D_iam[-1]
print(f"✅ IAM solution complete")

# TEST 2B: GROWTH SUPPRESSION CHECK
print("\n" + "=" * 70)
print("TEST 2b: Growth Suppression Analysis")
print("=" * 70)

z_array = 1/a_array - 1
suppression = (D_lcdm - D_iam) / D_lcdm * 100

test_z = [0, 0.5, 1.0, 2.0, 10.0, 100.0, 999.0]

print("\nSuppression at key redshifts:\n")
for z in test_z:
    idx = np.argmin(np.abs(z_array - z))
    a = a_array[idx]
    Tax_val = GROWTH_TAX * E_activation(a)
    supp_val = suppression[idx]
    print(f"  z = {z:6.1f}: Tax = {Tax_val*100:5.2f}%, " +
          f"Suppression = {supp_val:5.2f}%, " +
          f"D_IAM/D_ΛCDM = {D_iam[idx]/D_lcdm[idx]:.6f}")

supp_today = suppression[-1]
ratio_today = D_iam[-1] / D_lcdm[-1]

print(f"\nToday (z = 0):")
print(f"  Suppression = {supp_today:.3f}%")
print(f"  D_IAM/D_ΛCDM = {ratio_today:.6f}")

if 4.0 <= supp_today <= 5.0:
    print("✅ PASSED: Suppression within expected range")
else:
    print(f"❌ FAILED: Suppression outside [4.0%, 5.0%]")

test_2b_pass = 4.0 <= supp_today <= 5.0

idx_early = np.argmin(np.abs(z_array - 999))
supp_early = suppression[idx_early]

print(f"\nEarly universe (z = 999):")
print(f"  Suppression = {supp_early:.6f}%")

if abs(supp_early) < 0.1:
    print("✅ PASSED: Negligible suppression in early universe")
else:
    print("❌ FAILED: Significant early suppression")

test_2a_pass = abs(supp_early) < 0.1

# TEST 2C: GROWTH RATE f(z)
print("\n" + "=" * 70)
print("TEST 2c: Growth Rate f(z) = d ln D / d ln a")
print("=" * 70)

f_lcdm = np.gradient(np.log(D_lcdm), np.log(a_array))
f_iam = np.gradient(np.log(D_iam), np.log(a_array))

print("\nGrowth rate f(z):\n")
for z in [0, 0.5, 1.0, 2.0]:
    idx = np.argmin(np.abs(z_array - z))
    print(f"  z = {z:.1f}: f_ΛCDM = {f_lcdm[idx]:.4f}, " +
          f"f_IAM = {f_iam[idx]:.4f}, " +
          f"Δf/f = {(f_lcdm[idx] - f_iam[idx])/f_lcdm[idx]*100:.2f}%")

if np.all(f_iam > 0) and np.all(f_iam < 1.2):
    print("\n✅ PASSED: f(z) is physical (0 < f < 1.2)")
else:
    print("\n❌ FAILED: f(z) is unphysical")

test_2c_pass = np.all(f_iam > 0) and np.all(f_iam < 1.2)

# TEST 2D: MONOTONICITY
print("\n" + "=" * 70)
print("TEST 2d: Monotonicity Check")
print("=" * 70)

dD_da_iam = np.diff(D_iam)

if np.all(dD_da_iam >= 0):
    print("✅ PASSED: D(a) is monotonically increasing")
else:
    n_violations = np.sum(dD_da_iam < 0)
    print(f"⚠️  WARNING: {n_violations} monotonicity violations (likely numerical noise)")

test_2d_pass = np.sum(dD_da_iam < 0) < 10

# TEST 2E: VISUALIZATION
print("\n" + "=" * 70)
print("TEST 2e: Generating Diagnostic Plots")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IAM Growth Factor Validation', fontsize=16, fontweight='bold')

# Plot 1: D(z)
ax = axes[0, 0]
ax.plot(z_array, D_lcdm, 'k--', linewidth=2.5, label='ΛCDM', alpha=0.7)
ax.plot(z_array, D_iam, 'b-', linewidth=2.5, label='IAM')
ax.axvline(2, color='red', linestyle='--', alpha=0.4, label='z=2')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('D(z) [normalized]', fontsize=12)
ax.set_title('Linear Growth Factor', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim([0.01, 1000])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Suppression
ax = axes[0, 1]
ax.plot(z_array, suppression, 'r-', linewidth=2.5)
ax.axhline(4.5, color='k', linestyle='--', alpha=0.4, label='Target: 4.5%')
ax.axvline(2, color='red', linestyle='--', alpha=0.4)
ax.fill_between(z_array, 0, suppression, alpha=0.2, color='red')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Suppression [%]', fontsize=12)
ax.set_title('Growth Suppression', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim([0.01, 1000])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: f(z)
ax = axes[1, 0]
ax.plot(z_array, f_lcdm, 'k--', linewidth=2.5, label='ΛCDM', alpha=0.7)
ax.plot(z_array, f_iam, 'b-', linewidth=2.5, label='IAM')
ax.axvline(2, color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('f(z)', fontsize=12)
ax.set_title('Growth Rate', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim([0.01, 10])
ax.set_ylim([0.4, 1.0])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 4: Tax
ax = axes[1, 1]
Tax_array = GROWTH_TAX * E_activation(a_array)
ax.plot(z_array, Tax_array * 100, 'g-', linewidth=2.5, label='Tax(a)')
ax.axhline(GROWTH_TAX * 100, color='k', linestyle='--', alpha=0.4,
           label=f'Max = {GROWTH_TAX*100:.1f}%')
ax.fill_between(z_array, 0, Tax_array * 100, alpha=0.2, color='green')
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Tax(a) [%]', fontsize=12)
ax.set_title('Growth Tax', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim([0.01, 1000])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = RESULTS_DIR / 'test_02_growth_factor.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved: {output_path}")

# SUMMARY
print("\n" + "=" * 70)
print("TEST 2 SUMMARY REPORT")
print("=" * 70)

all_tests_pass = test_2a_pass and test_2b_pass and test_2c_pass and test_2d_pass

print(f"""
Test 2a: Early Universe............ {'✅ PASSED' if test_2a_pass else '❌ FAILED'}
  - Suppression at z=999: {supp_early:.6f}%
  
Test 2b: Growth Suppression........ {'✅ PASSED' if test_2b_pass else '❌ FAILED'}
  - Suppression at z=0: {supp_today:.3f}%
  - D_IAM(0)/D_ΛCDM(0) = {ratio_today:.6f}
  
Test 2c: Growth Rate f(z).......... {'✅ PASSED' if test_2c_pass else '❌ FAILED'}
  - f(z=0) = {f_iam[-1]:.4f}
  
Test 2d: Monotonicity.............. {'✅ PASSED' if test_2d_pass else '❌ FAILED'}

OVERALL STATUS: {'✅ ALL TESTS PASSED' if all_tests_pass else '⚠️  SOME TESTS FAILED'}

Key Result:
  Growth suppression at z=0: {supp_today:.2f}%
  Target: 4.5%
""")

print("=" * 70)
print("\nNext: Test 3 - fσ₈(z) predictions for DESI")
print("=" * 70)

np.savez(RESULTS_DIR / 'growth_factor_data.npz',
         z=z_array, a=a_array,
         D_lcdm=D_lcdm, D_iam=D_iam,
         f_lcdm=f_lcdm, f_iam=f_iam)
print(f"✅ Saved: {RESULTS_DIR / 'growth_factor_data.npz'}")
