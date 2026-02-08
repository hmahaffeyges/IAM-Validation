import numpy as np
from scipy.integrate import odeint

H0_CMB = 67.4
Omega_m = 0.315
Omega_r = 9.24e-5
Omega_Lambda = 1 - Omega_m - Omega_r
BETA = 0.18
GROWTH_TAX = 0.045
sigma8_0 = 0.811

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

def growth_iam(y, a, H0, beta, tax_amp):
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

sol_iam = odeint(growth_iam, y0, a_arr, args=(H0_CMB, BETA, GROWTH_TAX))
D_iam = sol_iam[:, 0] / sol_iam[-1, 0]

z_arr = 1/a_arr - 1

f_lcdm = np.gradient(np.log(D_lcdm), np.log(a_arr))
f_iam = np.gradient(np.log(D_iam), np.log(a_arr))

fsigma8_lcdm = f_lcdm * sigma8_0 * D_lcdm
fsigma8_iam = f_iam * sigma8_0 * D_iam

print("fσ₈(z) predictions for DESI comparison:\n")
print("z      fσ₈_ΛCDM   fσ₈_IAM    Suppression")
print("-" * 50)

desi_z = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

for z in desi_z:
    idx = np.argmin(np.abs(z_arr - z))
    supp = (fsigma8_lcdm[idx] - fsigma8_iam[idx]) / fsigma8_lcdm[idx] * 100
    print(f"{z:.1f}   {fsigma8_lcdm[idx]:.4f}    {fsigma8_iam[idx]:.4f}     {supp:+6.2f}%")

avg_supp = np.mean([(fsigma8_lcdm[np.argmin(np.abs(z_arr - z))] - 
                     fsigma8_iam[np.argmin(np.abs(z_arr - z))]) / 
                    fsigma8_lcdm[np.argmin(np.abs(z_arr - z))] * 100 
                    for z in desi_z])

print(f"\nAverage suppression across DESI range: {avg_supp:+.2f}%")
print(f"Target: ~4.5%")
