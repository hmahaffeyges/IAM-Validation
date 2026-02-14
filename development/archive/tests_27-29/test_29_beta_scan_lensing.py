import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt

Om0 = 0.315
Om_L = 0.685
H0 = 67.4
c = 299792.458
betas = np.linspace(0.01, 0.18, 10)  # 10 steps from 0.01 to 0.18

# ----- Activation for IAM -----
def E_activation(a):
    return np.exp(1 - 1/a)

# ----- Background H(z) -----
def H_lcdm(z):
    a = 1/(1+z)
    return H0 * np.sqrt(Om0 * a**-3 + Om_L)

def H_iam(z, beta):
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

# ----- Lensing kernel for array -----
def lensing_kernel_array(z_vals):
    z_rec = 1090.0
    kernel = []
    a_rec = 1/(1+z_rec)
    chi_rec = c * trapezoid(1/H_lcdm(np.linspace(0,z_rec,100)), np.linspace(0,z_rec,100))
    for z in z_vals:
        a = 1/(1+z)
        if z == 0:
            kernel.append(1.0)
            continue
        chi_z = c * trapezoid(1/H_lcdm(np.linspace(0,z,20)), np.linspace(0,z,20))
        kernel.append((chi_rec - chi_z)/(chi_rec * a))
    return np.array(kernel)

# Main scan
z_start, z_end = 0, 1100
z_vals = np.linspace(z_start, z_end, 400)
D_curves = []

# LCDM reference
sol_lcdm = solve_ivp(growth_ode, (z_end, z_start), [1/(1+z_end), 1/(1+z_end)],
                    args=(0,), t_eval=z_vals[::-1], method='RK45')
D_lcdm = sol_lcdm.y[0][::-1]
D_lcdm = D_lcdm / D_lcdm[0]  # Normalize

lens_kernel = lensing_kernel_array(z_vals)
Phi_lcdm = D_lcdm * lens_kernel
lensing_lcdm = trapezoid(Phi_lcdm, z_vals)

# Plot setup
plt.figure(figsize=(10,6))
plt.plot(z_vals, D_lcdm, 'k-', label="D_LCDM(z)")

print("Beta   Lensing_IAM    RelDiff(%)")
print("-----  -------------  -----------")

for beta in betas:
    # IAM growth
    sol_iam = solve_ivp(growth_ode, (z_end, z_start), [1/(1+z_end), 1/(1+z_end)],
                        args=(beta,), t_eval=z_vals[::-1], method='RK45')
    D_iam = sol_iam.y[0][::-1]
    D_iam = D_iam / D_iam[0]
    D_curves.append(D_iam)
    # IAM lensing
    Phi_iam = D_iam * lens_kernel
    lensing_iam = trapezoid(Phi_iam, z_vals)
    rel_diff = 100*(lensing_iam - lensing_lcdm)/lensing_lcdm
    print(f"{beta:0.3f}  {lensing_iam:12.3e}   {rel_diff:9.2f}")
    # Plot
    plt.plot(z_vals, D_iam, label=f"IAM β={beta:0.2f}")

plt.xlabel("z")
plt.ylabel("Normalized Growth D(z)")
plt.title("Growth Factor D(z) Scan for β Values (IAM vs LCDM)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/beta_scan_Dz.png")
plt.show()

