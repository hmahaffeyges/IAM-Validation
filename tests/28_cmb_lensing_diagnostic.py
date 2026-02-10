import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt

# Cosmology params
Om0 = 0.315
Om_L = 0.685
H0 = 67.4
beta = 0.179
c = 299792.458

# ----- Activation for IAM -----
def E_activation(a):
    return np.exp(1 - 1/a)

# ----- Background H(z) -----
def H_lcdm(z):
    a = 1/(1+z)
    return H0 * np.sqrt(Om0 * a**-3 + Om_L)

def H_iam(z, beta=beta):
    a = 1/(1+z)
    return H0 * np.sqrt(Om0 * a**-3 + Om_L + beta * E_activation(a))

# ----- Growth factor D(z) -----
def growth_ode(z, y, beta_val):
    a = 1/(1+z)
    D, Dprime = y
    H = H_iam(z, beta_val)
    Om_a = Om0 * a**-3 / (Om0 * a**-3 + Om_L + beta_val * E_activation(a))
    Q = 2 - 1.5 * Om_a
    Ddbl = -Q * Dprime + 1.5 * Om_a * D
    return [Dprime/(1+z), Ddbl/(1+z)]

# Solve D(z) from z=0 to z=1100
z_start, z_end = 0, 1100
z_vals = np.linspace(z_start, z_end, 2000)
# LCDM
sol_lcdm = solve_ivp(growth_ode, (z_end, z_start), [1/(1+z_end), 1/(1+z_end)],
                     args=(0,), t_eval=z_vals[::-1], method='RK45')
D_lcdm = sol_lcdm.y[0][::-1]
# IAM
sol_iam = solve_ivp(growth_ode, (z_end, z_start), [1/(1+z_end), 1/(1+z_end)],
                    args=(beta,), t_eval=z_vals[::-1], method='RK45')
D_iam = sol_iam.y[0][::-1]

# Normalize so D(z=0) = 1
D_lcdm = D_lcdm / D_lcdm[0]
D_iam = D_iam / D_iam[0]

# ----- Lensing Kernel -----
def lensing_kernel_array(z_vals):
    z_rec = 1090.0
    kernel = []
    a_rec = 1/(1+z_rec)
    chi_rec = c * trapezoid(1/H_lcdm(np.linspace(0,z_rec,100)), np.linspace(0,z_rec,100))
    for z in z_vals:
        a = 1/(1+z)
        chi_z = c * trapezoid(1/H_lcdm(np.linspace(0,z,20)), np.linspace(0,z,20))
        kernel.append((chi_rec - chi_z)/(chi_rec * a))
    return np.array(kernel)

# ----- Lensing potential Phi(z) -----
# Simple: Phi(z) ~ D(z) * lensing_kernel(z)
Phi_lcdm = D_lcdm * lensing_kernel_array(z_vals)
Phi_iam  = D_iam  * lensing_kernel_array(z_vals)

# ----- Total lensing power -----
# Compute convergence-like quantity (integral of Phi(z))
lensing_lcdm = trapezoid(Phi_lcdm, z_vals)
lensing_iam  = trapezoid(Phi_iam, z_vals)

print("CMB Lensing Diagnostic (IAM vs LCDM)")
print("-------------------------------------")
print(f"Total lensing potential LCDM: {lensing_lcdm:.4e}")
print(f"Total lensing potential IAM  : {lensing_iam:.4e}")
print(f"Relative difference: {100*(lensing_iam-lensing_lcdm)/lensing_lcdm:.2f}%")

# ----- Plot -----
plt.figure(figsize=(8,5))
plt.plot(z_vals, D_lcdm, label="D_LCDM(z)")
plt.plot(z_vals, D_iam, label="D_IAM(z)")
plt.plot(z_vals, Phi_lcdm/lensing_lcdm, '--', label="Lensing LCDM (norm)")
plt.plot(z_vals, Phi_iam/lensing_lcdm, '--', label="Lensing IAM (norm)")
plt.xlabel("z")
plt.ylabel("Normalized")
plt.title("Growth Factor D(z) and CMB Lensing Potential")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("results/cmb_lensing_diagnostic.png")
plt.show()
