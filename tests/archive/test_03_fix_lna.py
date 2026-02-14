"""
Growth equation in ln(a) coordinates (matches other AI's method)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
H0 = 67.4
Om0 = 0.315
sigma8_0 = 0.811
BETA = 0.18
GROWTH_TAX = 0.045

def E_activation(a):
    return np.exp(1 - 1/a)

def Omega_m_a(a, beta=0):
    """Matter density parameter at scale factor a"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    if beta > 0:
        # IAM
        denom = Om0 * a**(-3) + Om_r * a**(-4) + Om_L + beta * E_activation(a)
    else:
        # ΛCDM
        denom = Om0 * a**(-3) + Om_r * a**(-4) + Om_L
    return Om0 * a**(-3) / denom

def growth_ode_lna(lna, y, beta=0, tax=0):
    """
    Growth equation in ln(a) coordinates:
    d²D/d(ln a)² + Q dD/d(ln a) = (3/2) Ωₘ(a) D (1 - Tax)
    
    where Q = 2 - (3/2) Ωₘ(a)
    """
    D, Dprime = y
    a = np.exp(lna)
    
    Om_a = Omega_m_a(a, beta)
    Q = 2 - 1.5 * Om_a
    
    # Tax depends on scale factor
    if tax > 0:
        Tax = tax * E_activation(a)
    else:
        Tax = 0
    
    # Second derivative
    D_double_prime = -Q * Dprime + 1.5 * Om_a * D * (1 - Tax)
    
    return [Dprime, D_double_prime]

def solve_growth(z_vals, beta=0, tax=0):
    """Solve growth equation from early times to today"""
    # Start deep in matter domination
    lna_start = np.log(0.001)  # z ~ 999
    lna_end = 0.0  # z = 0
    
    # Initial conditions in matter domination: D ∝ a
    a_start = np.exp(lna_start)
    y0 = [a_start, a_start]  # [D, dD/d(ln a)] = [a, a] in MD
    
    # Evaluation points
    a_vals = 1 / (1 + z_vals)
    lna_vals = np.log(a_vals)
    lna_eval = np.sort(np.append(lna_vals, [lna_start, lna_end]))
    
    # Solve ODE
    sol = solve_ivp(
        growth_ode_lna,
        (lna_start, lna_end),
        y0,
        args=(beta, tax),
        t_eval=lna_eval,
        method='DOP853',
        rtol=1e-8,
        atol=1e-10
    )
    
    D_raw = sol.y[0]
    D_normalized = D_raw / D_raw[-1]  # Normalize to D(z=0) = 1
    
    # Create dictionary
    lna_to_D = dict(zip(lna_eval, D_normalized))
    
    return lna_to_D, sol

def compute_f(z_vals, lna_to_D):
    """Compute f(z) = d ln D / d ln a"""
    a_vals = 1 / (1 + z_vals)
    lna_vals = np.log(a_vals)
    
    D_vals = np.array([lna_to_D[lna] for lna in lna_vals])
    
    # f = d ln D / d ln a
    f_vals = np.gradient(np.log(D_vals), lna_vals)
    
    return dict(zip(z_vals, f_vals))

# DESI data
desi_z = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
desi_fs8 = np.array([0.452, 0.428, 0.410, 0.392, 0.368, 0.355, 0.312])
desi_err = np.array([0.030, 0.025, 0.028, 0.035, 0.040, 0.045, 0.050])

print("="*70)
print("GROWTH CALCULATION IN ln(a) COORDINATES")
print("="*70)

# Solve for ΛCDM
print("\nSolving ΛCDM...")
D_lcdm, sol_lcdm = solve_growth(desi_z, beta=0, tax=0)
f_lcdm = compute_f(desi_z, D_lcdm)

# Solve for IAM
print("Solving IAM...")
D_iam, sol_iam = solve_growth(desi_z, beta=BETA, tax=GROWTH_TAX)
f_iam = compute_f(desi_z, D_iam)

# Compute fσ₈
print("\n" + "="*70)
print("fσ₈ PREDICTIONS")
print("="*70)

print("\nz_eff   fσ₈_DESI  f_ΛCDM  D_ΛCDM  fσ₈_ΛCDM  f_IAM   D_IAM   fσ₈_IAM")
print("-"*80)

fs8_lcdm = []
fs8_iam = []

for z in desi_z:
    a = 1 / (1 + z)
    lna = np.log(a)
    
    D_l = D_lcdm[lna]
    f_l = f_lcdm[z]
    fs8_l = f_l * sigma8_0 * D_l
    fs8_lcdm.append(fs8_l)
    
    D_i = D_iam[lna]
    f_i = f_iam[z]
    fs8_i = f_i * sigma8_0 * D_i
    fs8_iam.append(fs8_i)
    
    obs_idx = np.where(desi_z == z)[0][0]
    fs8_obs = desi_fs8[obs_idx]
    
    print(f"{z:.3f}   {fs8_obs:.3f}     {f_l:.3f}   {D_l:.3f}   {fs8_l:.3f}     {f_i:.3f}   {D_i:.3f}   {fs8_i:.3f}")

# Chi-squared
print("\n" + "="*70)
print("χ² ANALYSIS")
print("="*70)

fs8_lcdm = np.array(fs8_lcdm)
fs8_iam = np.array(fs8_iam)

chi2_lcdm = np.sum(((fs8_lcdm - desi_fs8) / desi_err)**2)
chi2_iam = np.sum(((fs8_iam - desi_fs8) / desi_err)**2)

print(f"\nχ²_ΛCDM(DESI) = {chi2_lcdm:.2f}")
print(f"χ²_IAM(DESI)  = {chi2_iam:.2f}")
print(f"Δχ²           = {chi2_lcdm - chi2_iam:+.2f}")

print("\nTarget from other AI:")
print("  χ²_ΛCDM(DESI) ~ 19.25")
print("  χ²_IAM(DESI)  ~ 14.76")

if abs(chi2_iam - 14.76) < 5:
    print("\n✅ MATCHED OTHER AI'S CALCULATION!")
else:
    print(f"\n⚠️  Still off by {chi2_iam - 14.76:.2f}")

print("="*70)
