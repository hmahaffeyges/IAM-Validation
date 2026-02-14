import numpy as np
from scipy.integrate import odeint

H0_CMB = 67.4
Omega_m = 0.315
Omega_r = 9.24e-5
Omega_Lambda = 1 - Omega_m - Omega_r
BETA = 0.18
GROWTH_TAX = 0.045

# Planck 2018 normalization
# σ₈ is defined at z=0 for ΛCDM
sigma8_LCDM_fiducial = 0.811

def E_activation(a):
    return np.exp(1 - 1/a)

def H_IAM(a, H0, beta):
    return H0 * np.sqrt(Omega_m * a**(-3) + Omega_r * a**(-4) + 
                        Omega_Lambda + beta * E_activation(a))

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

def growth_iam(y, a, H0, beta, tax):
    D, Dp = y
    H = H_IAM(a, H0, beta)
    Om = Omega_m_of_a(a, beta, True)
    tax_val = tax * E_activation(a)
    Om_eff = Om * (1 - tax_val)
    Dpp = 1.5 * Om_eff * H0**2 / (a**2 * H**2) * D - (2/a) * Dp
    return [Dp, Dpp]

a_arr = np.logspace(-3, 0, 2000)
y0 = [0.001, 1.0]

# Solve WITHOUT normalization
sol_lcdm = odeint(growth_lcdm, y0, a_arr, args=(H0_CMB,))
D_lcdm_raw = sol_lcdm[:, 0]

sol_iam = odeint(growth_iam, y0, a_arr, args=(H0_CMB, BETA, GROWTH_TAX))
D_iam_raw = sol_iam[:, 0]

z_arr = 1/a_arr - 1

# Normalize LCDM to match Planck σ₈ = 0.811 at z=0
# σ₈ ∝ D(z=0), so we set D_LCDM(z=0) to give σ₈ = 0.811
D_lcdm_normalized = D_lcdm_raw / D_lcdm_raw[-1]  # D(0) = 1 for LCDM

# For IAM: keep the SAME absolute normalization
# This means σ₈_IAM(0) ≠ σ₈_LCDM(0) if growth differs
D_iam_normalized = D_iam_raw / D_lcdm_raw[-1]  # Normalize to LCDM at z=0

# Now σ₈(z) for each model:
sigma8_lcdm_z = sigma8_LCDM_fiducial * D_lcdm_normalized
sigma8_iam_z = sigma8_LCDM_fiducial * D_iam_normalized

# Growth rates
f_lcdm = np.gradient(np.log(D_lcdm_normalized), np.log(a_arr))
f_iam = np.gradient(np.log(D_iam_normalized), np.log(a_arr))

# fσ₈(z)
fsig8_lcdm = f_lcdm * sigma8_lcdm_z
fsig8_iam = f_iam * sigma8_iam_z

print("Fixed normalization test:\n")
print("At z=0:")
print(f"  D_LCDM(0) = {D_lcdm_normalized[-1]:.6f}")
print(f"  D_IAM(0)  = {D_iam_normalized[-1]:.6f}")
print(f"  σ₈_LCDM(0) = {sigma8_lcdm_z[-1]:.4f}")
print(f"  σ₈_IAM(0)  = {sigma8_iam_z[-1]:.4f}")
print(f"  Suppression = {(1 - sigma8_iam_z[-1]/sigma8_lcdm_z[-1])*100:.2f}%")

print("\nfσ₈ at representative redshifts:")
print("z      fσ₈_ΛCDM   fσ₈_IAM")
for z in [0.3, 0.5, 0.7, 0.9]:
    idx = np.argmin(np.abs(z_arr - z))
    print(f"{z:.1f}    {fsig8_lcdm[idx]:.4f}    {fsig8_iam[idx]:.4f}")

print("\nThis should give different fσ₈ predictions now!")
