# Quick test: Growth with NO explicit tax, just H_IAM

import numpy as np
from scipy.integrate import odeint

H0_CMB = 67.4
Omega_m = 0.315
Omega_r = 9.24e-5
Omega_Lambda = 1 - Omega_m - Omega_r
BETA = 0.18

def E_activation(a):
    return np.exp(1 - 1/a)

def H_LCDM(a, H0):
    return H0 * np.sqrt(Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda)

def H_IAM(a, H0_CMB, beta):
    return H0_CMB * np.sqrt(
        Omega_m * a**(-3) + Omega_r * a**(-4) + 
        Omega_Lambda + beta * E_activation(a)
    )

def Omega_m_of_a(a, beta, use_iam=True):
    if use_iam:
        H2 = Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda + beta * E_activation(a)
    else:
        H2 = Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda
    return Omega_m * a**(-3) / H2

def growth_ode_lcdm(y, a, H0):
    D, Dprime = y
    H = H_LCDM(a, H0)
    Om_a = Omega_m_of_a(a, beta=0, use_iam=False)
    D_double_prime = 1.5 * Om_a * H0**2 / (a**2 * H**2) * D - (2/a) * Dprime
    return [Dprime, D_double_prime]

def growth_ode_iam_notax(y, a, H0_CMB, beta):
    """IAM growth with NO explicit tax - suppression from H_IAM only"""
    D, Dprime = y
    H = H_IAM(a, H0_CMB, beta)
    Om_a = Omega_m_of_a(a, beta, use_iam=True)
    # NO (1-Tax) term here - let larger H do the work
    D_double_prime = 1.5 * Om_a * H0_CMB**2 / (a**2 * H**2) * D - (2/a) * Dprime
    return [Dprime, D_double_prime]

a_array = np.logspace(-3, 0, 2000)
y0 = [0.001, 1.0]

sol_lcdm = odeint(growth_ode_lcdm, y0, a_array, args=(H0_CMB,))
D_lcdm = sol_lcdm[:, 0] / sol_lcdm[-1, 0]

sol_iam = odeint(growth_ode_iam_notax, y0, a_array, args=(H0_CMB, BETA))
D_iam = sol_iam[:, 0] / sol_iam[-1, 0]

z_array = 1/a_array - 1
suppression = (D_lcdm - D_iam) / D_lcdm * 100

print("Growth suppression WITHOUT explicit tax term:")
print("(Suppression from larger H_IAM alone)\n")

for z in [0, 0.5, 1.0, 2.0, 10.0]:
    idx = np.argmin(np.abs(z_array - z))
    print(f"z = {z:5.1f}: Suppression = {suppression[idx]:6.2f}%, D_IAM/D_ΛCDM = {D_iam[idx]/D_lcdm[idx]:.6f}")

print(f"\nSuppression at z=0: {suppression[-1]:.3f}%")
print(f"Target: 4.5%")
