import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

H0_CMB = 67.4
Om0 = 0.315
Om_r = 9.24e-5
Om_L = 1 - Om0 - Om_r

def E_activation(a):
    return np.exp(1 - 1/a)

def Omega_m_a(a, beta_m):
    E_a = E_activation(a)
    denom = Om0*a**(-3) + Om_r*a**(-4) + Om_L + beta_m*E_a
    return Om0 * a**(-3) / denom

def growth_ode_lna(lna, y, beta_m, tau):
    D, Dprime = y
    a = np.exp(lna)
    Om_a = Omega_m_a(a, beta_m)
    Q = 2 - 1.5 * Om_a
    Tax = tau * E_activation(a)
    D_double_prime = -Q * Dprime + 1.5 * Om_a * D * (1 - Tax)
    return [Dprime, D_double_prime]

def solve_growth(beta_m, tau):
    lna_start = np.log(0.001)
    a_start = np.exp(lna_start)
    y0 = [a_start, a_start]
    lna_eval = np.linspace(lna_start, 0.0, 2000)
    
    sol = solve_ivp(growth_ode_lna, (lna_start, 0.0), y0,
                    args=(beta_m, tau), t_eval=lna_eval,
                    method='DOP853', rtol=1e-8, atol=1e-10)
    
    return sol.y[0]  # Raw D(a), NOT normalized

# Compute for ΛCDM and IAM
D_lcdm = solve_growth(0.0, 0.0)
D_iam_045 = solve_growth(0.157, 0.045)
D_iam_120 = solve_growth(0.157, 0.120)
D_iam_200 = solve_growth(0.157, 0.200)

print("Growth factor at z=0 (a=1, last element):")
print(f"ΛCDM:        D(0) = {D_lcdm[-1]:.6f}")
print(f"IAM τ=0.045: D(0) = {D_iam_045[-1]:.6f}")
print(f"IAM τ=0.120: D(0) = {D_iam_120[-1]:.6f}")
print(f"IAM τ=0.200: D(0) = {D_iam_200[-1]:.6f}")
print()

print("Suppression factor:")
print(f"τ=0.045: {D_iam_045[-1]/D_lcdm[-1]:.6f} ({100*(1-D_iam_045[-1]/D_lcdm[-1]):.2f}% suppression)")
print(f"τ=0.120: {D_iam_120[-1]/D_lcdm[-1]:.6f} ({100*(1-D_iam_120[-1]/D_lcdm[-1]):.2f}% suppression)")
print(f"τ=0.200: {D_iam_200[-1]/D_lcdm[-1]:.6f} ({100*(1-D_iam_200[-1]/D_lcdm[-1]):.2f}% suppression)")
print()

print("Effective σ₈ if Planck σ₈,0 = 0.811:")
for tau, D in [(0.045, D_iam_045), (0.120, D_iam_120), (0.200, D_iam_200)]:
    sig8_eff = 0.811 * (D[-1]/D_lcdm[-1])
    print(f"τ={tau:.3f}: σ₈(IAM) = {sig8_eff:.4f}")
