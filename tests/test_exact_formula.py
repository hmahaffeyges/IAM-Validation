#!/usr/bin/env python3
"""
IAM EXACT FORMULA: H²(z) = H²_ΛCDM(z) + β·H(z)·D(z)²·f(z)
==========================================================
This implements your ORIGINAL physical insight directly.
No phenomenological approximations.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# Constants
H0_CMB = 67.4
Om0 = 0.315
Om_r = 9.24e-5
Om_L = 1 - Om0 - Om_r
SIGMA_8 = 0.811

print("="*70)
print("IAM EXACT FORMULA TEST")
print("="*70)
print()
print("Core equation:")
print("  H²(z) = H²_ΛCDM(z) + β·H(z)·D(z)²·f(z)")
print()

# ============================================================================
# SOLVE FOR H(z) SELF-CONSISTENTLY
# ============================================================================

def H_LCDM_sq(z):
    """ΛCDM Hubble parameter squared"""
    a = 1/(1+z)
    return H0_CMB**2 * (Om0*a**(-3) + Om_r*a**(-4) + Om_L)

def solve_H_exact(z, beta, D_interp, growth_from_beta):
    """Solve H²(z) = H²_ΛCDM(z) + β·H(z)·D(z)²·f(z) self-consistently"""
    
    a = 1/(1+z)
    lna = np.log(a)
    
    # Get D(z) and f(z) from pre-computed growth solution
    D_z = D_interp(lna)
    
    # Compute f(z)
    dlna = 0.001
    D_plus = D_interp(lna + dlna)
    D_minus = D_interp(lna - dlna)
    f_z = (np.log(D_plus) - np.log(D_minus)) / (2 * dlna)
    
    # Solve: H² = H²_ΛCDM + β·H·D²·f
    # Rearrange: H² - β·H·D²·f - H²_ΛCDM = 0
    H_lcdm_sq = H_LCDM_sq(z)
    
    # Quadratic in H: H² - β·D²·f·H - H²_ΛCDM = 0
    # Solution: H = (β·D²·f + √[(β·D²·f)² + 4·H²_ΛCDM]) / 2
    
    term = beta * D_z**2 * f_z
    H_exact = 0.5 * (term + np.sqrt(term**2 + 4*H_lcdm_sq))
    
    return H_exact

# ============================================================================
# GROWTH EQUATION
# ============================================================================

def Omega_m_exact(z, beta, D_interp):
    """Compute Ωₘ(z) using exact H(z)"""
    H_exact = solve_H_exact(z, beta, D_interp, None)
    a = 1/(1+z)
    
    numerator = Om0 * a**(-3)
    denominator = (H_exact / H0_CMB)**2
    
    return numerator / denominator

def growth_ode_exact(lna, y, beta, Omega_m_func):
    """Growth ODE using exact Ωₘ(z)"""
    D, Dprime = y
    a = np.exp(lna)
    z = 1/a - 1
    
    Om_z = Omega_m_func(z)
    Q = 2 - 1.5 * Om_z
    
    D_double_prime = -Q * Dprime + 1.5 * Om_z * D
    
    return [Dprime, D_double_prime]

# ============================================================================
# ITERATIVE SOLUTION
# ============================================================================

def solve_exact_model(beta, max_iter=10):
    """Solve the exact IAM model iteratively"""
    
    print(f"Solving exact model with β = {beta:.6f}")
    print()
    
    # Start with ΛCDM growth as initial guess
    print("  Iteration 1: Using ΛCDM growth as initial guess...")
    
    lna_eval = np.linspace(np.log(0.001), 0.0, 2000)
    
    # ΛCDM Ωₘ function
    def Om_lcdm(z):
        a = 1/(1+z)
        return Om0 * a**(-3) / (Om0*a**(-3) + Om_r*a**(-4) + Om_L)
    
    # Solve ΛCDM growth
    sol = solve_ivp(
        lambda lna, y: growth_ode_exact(lna, y, 0.0, Om_lcdm),
        (np.log(0.001), 0.0),
        [0.001, 0.001],
        t_eval=lna_eval,
        method='DOP853',
        rtol=1e-8
    )
    
    D_raw = sol.y[0]
    D_normalized = D_raw / D_raw[-1]
    D_interp = interp1d(lna_eval, D_normalized, kind='cubic', fill_value='extrapolate')
    
    # Now iterate to convergence
    for iteration in range(2, max_iter + 1):
        print(f"  Iteration {iteration}: Updating with β-modified growth...")
        
        # Create Ωₘ function using current D
        def Om_beta(z):
            return Omega_m_exact(z, beta, D_interp)
        
        # Solve growth with updated Ωₘ
        sol = solve_ivp(
            lambda lna, y: growth_ode_exact(lna, y, beta, Om_beta),
            (np.log(0.001), 0.0),
            [0.001, 0.001],
            t_eval=lna_eval,
            method='DOP853',
            rtol=1e-8
        )
        
        D_raw_new = sol.y[0]
        D_normalized_new = D_raw_new / D_raw_new[-1]
        
        # Check convergence
        max_change = np.max(np.abs(D_normalized_new - D_normalized))
        print(f"    Max change in D: {max_change:.2e}")
        
        if max_change < 1e-6:
            print(f"    ✓ Converged!")
            break
        
        # Update
        D_normalized = D_normalized_new
        D_interp = interp1d(lna_eval, D_normalized, kind='cubic', fill_value='extrapolate')
    
    return D_interp, D_raw_new[-1]

# ============================================================================
# TEST AT β = 0.001 (small value to verify)
# ============================================================================

print("="*70)
print("TESTING EXACT FORMULA")
print("="*70)
print()

beta_test = 0.001  # Start small

D_interp, D_final = solve_exact_model(beta_test)

print()
print(f"Testing at β = {beta_test}")
print()

# Compute H(z) at various redshifts
z_test = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

print("z      H_ΛCDM   H_exact   ΔH      D(z)")
print("-"*50)

for z in z_test:
    a = 1/(1+z)
    lna = np.log(a)
    
    H_lcdm = np.sqrt(H_LCDM_sq(z))
    H_exact = solve_H_exact(z, beta_test, D_interp, None)
    delta_H = H_exact - H_lcdm
    D_z = D_interp(lna)
    
    print(f"{z:.1f}    {H_lcdm:6.2f}   {H_exact:6.2f}   {delta_H:+.3f}   {D_z:.4f}")

print()
print("="*70)
print()
print("This is YOUR model - the exact formula!")
print("Now we need to find the β that fits the data.")
print()
print("Question: Should I proceed to fit this exact model?")
print("It will give a DIFFERENT β value than 0.157!")
