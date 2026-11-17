import numpy as np
from scipy.integrate import odeint

H0_CMB = 67.4
Omega_m = 0.315
Omega_r = 9.24e-5
Omega_Lambda = 1 - Omega_m - Omega_r
BETA = 0.18
GROWTH_TAX = 0.045

def E_activation(a):
    return np.exp(1 - 1/a)

def H_IAM(a, H0, beta):
    return H0 * np.sqrt(Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda + beta * E_activation(a))

def H_LCDM(a, H0):
    return H0 * np.sqrt(Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda)

def Omega_m_of_a(a, beta, use_iam):
    if use_iam:
        H2 = Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda + beta * E_activation(a)
    else:
        H2 = Omega_m * a**(-3) + Omega_r * a**(-4) + Omega_Lambda
    return Omega_m * a**(-3) / H2

def growth_lcdm(y, a, H0):
    D, Dp = y
    H = H_LCDM(a, H0)
    Om = Omega_m_of_a(a, 0, False)
    Dpp = 1.5 * Om * H0**2 / (a**2 * H**2) * D - (2/a) * Dp
    return [Dp, Dpp]

# Version A: Tax on source term
def growth_iam_A(y, a, H0, beta, tax_amp):
    D, Dp = y
    H = H_IAM(a, H0, beta)
    Om = Omega_m_of_a(a, beta, True)
    tax = tax_amp * E_activation(a)
    Dpp = 1.5 * Om * H0**2 / (a**2 * H**2) * D * (1 - tax) - (2/a) * Dp
    return [Dp, Dpp]

# Version B: Tax as friction
def growth_iam_B(y, a, H0, beta, tax_amp):
    D, Dp = y
    H = H_IAM(a, H0, beta)
    Om = Omega_m_of_a(a, beta, True)
    tax = tax_amp * E_activation(a)
    Dpp = 1.5 * Om * H0**2 / (a**2 * H**2) * D - (2/a) * Dp - tax * H * Dp / a
    return [Dp, Dpp]

# Version C: Tax on Omega_m
def growth_iam_C(y, a, H0, beta, tax_amp):
    D, Dp = y
    H = H_IAM(a, H0, beta)
    Om = Omega_m_of_a(a, beta, True)
    tax = tax_amp * E_activation(a)
    Om_eff = Om * (1 - tax)
    Dpp = 1.5 * Om_eff * H0**2 / (a**2 * H**2) * D - (2/a) * Dp
    return [Dp, Dpp]

a_arr = np.logspace(-3, 0, 2000)
y0 = [0.001, 1.0]

sol_lcdm = odeint(growth_lcdm, y0, a_arr, args=(H0_CMB,))
D_lcdm = sol_lcdm[:, 0] / sol_lcdm[-1, 0]

sol_A = odeint(growth_iam_A, y0, a_arr, args=(H0_CMB, BETA, GROWTH_TAX))
D_A = sol_A[:, 0] / sol_A[-1, 0]

sol_B = odeint(growth_iam_B, y0, a_arr, args=(H0_CMB, BETA, GROWTH_TAX))
D_B = sol_B[:, 0] / sol_B[-1, 0]

sol_C = odeint(growth_iam_C, y0, a_arr, args=(H0_CMB, BETA, GROWTH_TAX))
D_C = sol_C[:, 0] / sol_C[-1, 0]

z = 1/a_arr - 1
idx0 = -1

supp_A = (D_lcdm[idx0] - D_A[idx0]) / D_lcdm[idx0] * 100
supp_B = (D_lcdm[idx0] - D_B[idx0]) / D_lcdm[idx0] * 100
supp_C = (D_lcdm[idx0] - D_C[idx0]) / D_lcdm[idx0] * 100

print("Testing three tax implementations:\n")
print(f"Version A (tax on source):     Suppression = {supp_A:6.2f}%")
print(f"Version B (tax as friction):   Suppression = {supp_B:6.2f}%")
print(f"Version C (tax on Omega_m):    Suppression = {supp_C:6.2f}%")
print(f"\nTarget: 4.5%")

print("\n" + "="*50)
if abs(supp_A - 4.5) < 0.5:
    print("✅ Version A is closest")
elif abs(supp_B - 4.5) < 0.5:
    print("✅ Version B is closest")
elif abs(supp_C - 4.5) < 0.5:
    print("✅ Version C is closest")
else:
    print("⚠️  None match target - may need different tax amplitude")
