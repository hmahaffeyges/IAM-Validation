#!/usr/bin/env python3
"""Debug growth factor normalization"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

Om0 = 0.315
BETA = 0.179
GROWTH_TAX = 0.134

def E_activation(a):
    a_cutoff = 0.5
    if a < a_cutoff:
        return 0.0
    else:
        a_transition = 0.75
        width = 0.1
        return 0.5 * (1 + np.tanh((a - a_transition) / width))

def Omega_m_a(a):
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    denom = Om0 * a**(-3) + Om_r * a**(-4) + Om_L
    return Om0 * a**(-3) / denom

def growth_ode_lna(lna, y, beta=0, tax=0):
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
    lna_start = np.log(1/(1+z_max))
    lna_end = 0.0
    a_start = np.exp(lna_start)
    y0 = [a_start, a_start]
    
    lna_eval = np.linspace(lna_start, lna_end, n_points)
    
    sol = solve_ivp(growth_ode_lna, (lna_start, lna_end), y0,
                    args=(beta, tax), t_eval=lna_eval,
                    method='DOP853', rtol=1e-10, atol=1e-12)
    
    D_raw = sol.y[0]
    return lna_eval, D_raw  # Return UNNORMALIZED

print("="*80)
print("GROWTH FACTOR NORMALIZATION DEBUG")
print("="*80)
print()

lna_vals, D_lcdm_raw = solve_growth(z_max=10000, beta=0, tax=0)
lna_vals, D_iam_raw = solve_growth(z_max=10000, beta=BETA, tax=GROWTH_TAX)

a_vals = np.exp(lna_vals)
z_vals = 1/a_vals - 1

# Values at z=0
D_lcdm_z0 = D_lcdm_raw[-1]
D_iam_z0 = D_iam_raw[-1]

print(f"UNNORMALIZED values at z=0:")
print(f"  D_ΛCDM(z=0) = {D_lcdm_z0:.6f}")
print(f"  D_IAM(z=0)  = {D_iam_z0:.6f}")
print(f"  Ratio: D_IAM / D_ΛCDM = {D_iam_z0/D_lcdm_z0:.6f}")
print()

if D_iam_z0 < D_lcdm_z0:
    print("✅ IAM suppresses growth (as expected from tax)")
    suppression = 100 * (1 - D_iam_z0/D_lcdm_z0)
    print(f"   Total suppression: {suppression:.2f}%")
else:
    print("❌ IAM ENHANCES growth (WRONG!)")
    enhancement = 100 * (D_iam_z0/D_lcdm_z0 - 1)
    print(f"   Total enhancement: {enhancement:.2f}%")

print()

# Check at intermediate redshifts
z_test = [0.5, 1.0, 2.0, 5.0]
print("Unnormalized D at various redshifts:")
print(f"{'z':>6s}  {'D_ΛCDM':>12s}  {'D_IAM':>12s}  {'Ratio':>8s}")
print("-"*50)

for z in z_test:
    idx = np.argmin(np.abs(z_vals - z))
    D_lcdm = D_lcdm_raw[idx]
    D_iam = D_iam_raw[idx]
    ratio = D_iam / D_lcdm
    print(f"{z:>6.1f}  {D_lcdm:>12.6f}  {D_iam:>12.6f}  {ratio:>8.6f}")

print()
print("="*80)
