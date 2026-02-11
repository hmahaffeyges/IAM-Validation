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

# Version C: Tax on Omega_m (seems most physical)
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
D_lcdm_raw = sol_lcdm[:, 0]

sol_C = odeint(growth_iam_C, y0, a_arr, args=(H0_CMB, BETA, GROWTH_TAX))
D_C_raw = sol_C[:, 0]

z_arr = 1/a_arr - 1

print("UNNORMALIZED growth factor comparison:\n")
print("z      D_ΛCDM      D_IAM      Ratio")
print("-" * 45)
for z in [0, 0.5, 1.0, 2.0, 10.0]:
    idx = np.argmin(np.abs(z_arr - z))
    ratio = D_C_raw[idx] / D_lcdm_raw[idx]
    print(f"{z:4.1f}  {D_lcdm_raw[idx]:9.6f}  {D_C_raw[idx]:9.6f}  {ratio:.6f}")

print("\n" + "="*50)
print("NORMALIZED growth factor (D(0)=1):\n")

D_lcdm_norm = D_lcdm_raw / D_lcdm_raw[-1]
D_C_norm = D_C_raw / D_C_raw[-1]

print("z      D_ΛCDM      D_IAM      Suppression")
print("-" * 50)
for z in [0, 0.5, 1.0, 2.0, 10.0]:
    idx = np.argmin(np.abs(z_arr - z))
    supp = (D_lcdm_norm[idx] - D_C_norm[idx]) / D_lcdm_norm[idx] * 100
    print(f"{z:4.1f}  {D_lcdm_norm[idx]:9.6f}  {D_C_norm[idx]:9.6f}  {supp:6.2f}%")

# The KEY metric: sigma8 suppression
# sigma8(z) = sigma8(0) * D(z)/D(0)
# So sigma8_IAM(z) / sigma8_LCDM(z) = [D_IAM(z)/D_IAM(0)] / [D_LCDM(z)/D_LCDM(0)]

print("\n" + "="*50)
print("SIGMA8 RATIO (what observations measure):\n")
print("z      σ8_IAM/σ8_ΛCDM")
print("-" * 30)
for z in [0, 0.5, 1.0, 2.0]:
    idx = np.argmin(np.abs(z_arr - z))
    sig8_ratio = D_C_norm[idx] / D_lcdm_norm[idx]
    print(f"{z:4.1f}  {sig8_ratio:.6f}  ({(1-sig8_ratio)*100:+.2f}%)")

print("\n" + "="*50)
print("At z=0, both normalized to 1, so no suppression appears.")
print("We need to look at GROWTH RATE f(z) instead!")
