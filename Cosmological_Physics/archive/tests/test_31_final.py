#!/usr/bin/env python3
"""
Test 31 Final: 2D Likelihood with Extended τ Range
===================================================
Extended scan to τ_max = 0.30 to find true minimum and confidence regions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Constants
c = 299792.458
H0_CMB = 67.4
Om0 = 0.315
Om_r = 9.24e-5
Om_L = 1 - Om0 - Om_r
SIGMA_8 = 0.811

print("="*70)
print("TEST 31 FINAL: 2D LIKELIHOOD SCAN (β_m, τ) - EXTENDED")
print("="*70)
print()

# Data
h0_data = [
    ('Planck', 67.4, 0.5),
    ('SH0ES', 73.04, 1.04),
    ('JWST/TRGB', 70.39, 1.89),
]

desi_data = np.array([
    [0.295, 0.452, 0.030],
    [0.510, 0.428, 0.025],
    [0.706, 0.410, 0.028],
    [0.934, 0.392, 0.035],
    [1.321, 0.368, 0.040],
    [1.484, 0.355, 0.045],
    [2.330, 0.312, 0.050],
])

z_desi = desi_data[:, 0]
fsig8_desi = desi_data[:, 1]
sig_desi = desi_data[:, 2]

# Functions
def E_activation(a):
    return np.exp(1 - 1/a)

def H_IAM(a, beta_m):
    return H0_CMB * np.sqrt(Om0*a**(-3) + Om_r*a**(-4) + Om_L + beta_m*E_activation(a))

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
    
    if not sol.success:
        raise RuntimeError("Growth ODE failed")
    
    D_raw = sol.y[0]
    D_normalized = D_raw / D_raw[-1]
    return interp1d(lna_eval, D_normalized, kind='cubic', fill_value='extrapolate')

def compute_fsigma8(z_vals, beta_m, tau):
    D_interp = solve_growth(beta_m, tau)
    
    results = []
    for z in z_vals:
        a = 1 / (1 + z)
        lna = np.log(a)
        dlna = 0.001
        D_a = D_interp(lna)
        D_plus = D_interp(lna + dlna)
        D_minus = D_interp(lna - dlna)
        
        f = (np.log(D_plus) - np.log(D_minus)) / (2 * dlna)
        sigma8_z = SIGMA_8 * D_a
        fsig8 = f * sigma8_z
        results.append(fsig8)
    
    return np.array(results)

def chi2_total(beta_m, tau):
    H0_iam = H_IAM(1.0, beta_m)
    
    chi2_h0 = 0.0
    for name, h0_obs, sig in h0_data:
        if name == 'Planck':
            chi2_h0 += ((H0_CMB - h0_obs) / sig)**2
        else:
            chi2_h0 += ((H0_iam - h0_obs) / sig)**2
    
    fsig8_pred = compute_fsigma8(z_desi, beta_m, tau)
    chi2_desi = np.sum(((fsig8_pred - fsig8_desi) / sig_desi)**2)
    
    return chi2_h0 + chi2_desi

# Extended grid
print("Setting up extended 2D grid...")
print()

beta_m_min, beta_m_max = 0.08, 0.25
tau_min, tau_max = 0.00, 0.30  # Extended!
n_beta = 50
n_tau = 50

beta_m_grid = np.linspace(beta_m_min, beta_m_max, n_beta)
tau_grid = np.linspace(tau_min, tau_max, n_tau)

print(f"Grid: {n_beta} × {n_tau} = {n_beta*n_tau} points")
print(f"  β_m ∈ [{beta_m_min}, {beta_m_max}]")
print(f"  τ   ∈ [{tau_min}, {tau_max}]  ← EXTENDED")
print()
print("Computing χ² on grid...")
print()

chi2_grid = np.zeros((n_tau, n_beta))
total_points = n_beta * n_tau
completed = 0

for i, tau in enumerate(tau_grid):
    for j, beta_m in enumerate(beta_m_grid):
        try:
            chi2_grid[i, j] = chi2_total(beta_m, tau)
        except:
            chi2_grid[i, j] = np.nan
        
        completed += 1
        if completed % 250 == 0:
            pct = 100 * completed / total_points
            print(f"  Progress: {completed}/{total_points} ({pct:.1f}%)")

print()
print("Grid computation complete!")
print()

# Find minimum
mask = ~np.isnan(chi2_grid)
chi2_valid = chi2_grid[mask]

idx_2d = np.unravel_index(np.argmin(chi2_grid, axis=None), chi2_grid.shape)
tau_best = tau_grid[idx_2d[0]]
beta_m_best = beta_m_grid[idx_2d[1]]
chi2_min = chi2_grid[idx_2d]

print("="*70)
print("BEST-FIT RESULTS")
print("="*70)
print()
print(f"Best-fit parameters:")
print(f"  β_m = {beta_m_best:.4f}")
print(f"  τ   = {tau_best:.4f}")
print(f"  χ²  = {chi2_min:.4f}")
print()

H0_matter = H_IAM(1.0, beta_m_best)
print(f"Predicted H₀(matter) = {H0_matter:.2f} km/s/Mpc")
print()

# Compute effective σ₈ (using diagnostic info)
# Approximate suppression: ~2% at τ=0.045, ~3% at τ=0.12, ~4% at τ=0.20
# Linear interpolation: suppression ≈ 0.02 + (τ - 0.045) * (0.04 - 0.02) / (0.20 - 0.045)
if tau_best < 0.30:
    suppression_pct = 0.02 + (tau_best - 0.045) * (0.04 - 0.02) / (0.20 - 0.045)
else:
    suppression_pct = 0.05  # extrapolation
    
sigma8_eff = 0.811 * (1 - suppression_pct)
print(f"Effective σ₈(IAM) ≈ {sigma8_eff:.4f}")
print(f"  (Weak lensing: 0.76-0.78)")
print()

# Check if at boundary
delta_chi2 = chi2_grid - chi2_min
delta_chi2_tau = np.nanmin(delta_chi2, axis=1)

if delta_chi2_tau[-1] < 1.0:
    print("⚠️  WARNING: Minimum still at boundary (τ=0.30)")
    print("    Consider extending to τ_max = 0.40")
elif delta_chi2_tau[-1] > 9.0:
    print("✅ OK: Minimum well within grid")
else:
    print("✅ OK: Minimum found, boundary at >3σ")
print()

# Visualization
print("="*70)
print("GENERATING PLOTS")
print("="*70)
print()

# Main 2D contour
fig, ax = plt.subplots(figsize=(10, 8))

levels = [1.0, 4.0, 9.0]
colors = ['orange', 'red', 'purple']
labels = ['68% CL (1σ)', '95% CL (2σ)', '99.7% CL (3σ)']

contourf = ax.contourf(beta_m_grid, tau_grid, delta_chi2, 
                        levels=[0, 1, 4, 9, 100],
                        colors=['darkgreen', 'orange', 'red', 'purple'],
                        alpha=0.3)

contours = ax.contour(beta_m_grid, tau_grid, delta_chi2,
                      levels=levels, colors=colors, linewidths=2)

# Manual legend
for level, color, label in zip(levels, colors, labels):
    ax.plot([], [], color=color, linewidth=2, label=label)

ax.plot(beta_m_best, tau_best, 'g*', markersize=20, 
        label=f'Best fit: β_m={beta_m_best:.3f}, τ={tau_best:.3f}',
        markeredgecolor='black', markeredgewidth=1.5)

ax.axvline(beta_m_best, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(tau_best, color='green', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel(r'$\beta_m$ (matter sector coupling)', fontsize=14)
ax.set_ylabel(r'$\tau$ (growth tax)', fontsize=14)
ax.set_title(r'Joint Likelihood: $(\beta_m, \tau)$ - Extended Scan', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/beta_m_tau_2d_extended.png', dpi=150, bbox_inches='tight')
print("  Saved: results/beta_m_tau_2d_extended.png")

# Marginals
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# β_m marginal
delta_chi2_beta_m = np.nanmin(delta_chi2, axis=0)
ax1.plot(beta_m_grid, delta_chi2_beta_m, 'b-', linewidth=2)
ax1.axhline(1.0, color='orange', linestyle='--', label='1σ')
ax1.axhline(4.0, color='red', linestyle='--', label='2σ')
ax1.axhline(9.0, color='purple', linestyle='--', label='3σ')
ax1.axvline(beta_m_best, color='green', linestyle=':', linewidth=2)
ax1.set_xlabel(r'$\beta_m$', fontsize=12)
ax1.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
ax1.set_title('Marginalized over τ', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 15)

# τ marginal
ax2.plot(tau_grid, delta_chi2_tau, 'b-', linewidth=2)
ax2.axhline(1.0, color='orange', linestyle='--', label='1σ')
ax2.axhline(4.0, color='red', linestyle='--', label='2σ')
ax2.axhline(9.0, color='purple', linestyle='--', label='3σ')
ax2.axvline(tau_best, color='green', linestyle=':', linewidth=2)
ax2.set_xlabel(r'$\tau$', fontsize=12)
ax2.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
ax2.set_title('Marginalized over β_m', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 15)

plt.tight_layout()
plt.savefig('../results/beta_m_tau_marginals_extended.png', dpi=150, bbox_inches='tight')
print("  Saved: results/beta_m_tau_marginals_extended.png")
print()

# Correlation
mask_1sig = delta_chi2 < 1.0
if np.any(mask_1sig):
    beta_m_1sig = beta_m_grid[None, :].repeat(n_tau, axis=0)[mask_1sig]
    tau_1sig = tau_grid[:, None].repeat(n_beta, axis=1)[mask_1sig]
    corr = np.corrcoef(beta_m_1sig, tau_1sig)[0, 1]
    
    print("="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    print()
    print(f"Correlation coefficient (within 1σ): {corr:.3f}")
    print()
    
    if abs(corr) < 0.3:
        print("  → Parameters are weakly correlated")
        print("  → β_m and τ constrain independently")
    elif abs(corr) < 0.7:
        print("  → Parameters show moderate correlation")
        print("  → Some degeneracy but manageable")
    else:
        print("  → Parameters are strongly correlated")
        print("  → Significant degeneracy")
    print()

# Save
results = {
    'beta_m_grid': beta_m_grid,
    'tau_grid': tau_grid,
    'chi2_grid': chi2_grid,
    'delta_chi2': delta_chi2,
    'beta_m_best': beta_m_best,
    'tau_best': tau_best,
    'chi2_min': chi2_min,
    'sigma8_eff': sigma8_eff,
}

np.save('../results/test_31_final.npy', results)
print("Results saved to: results/test_31_final.npy")
print()

print("="*70)
print("TEST 31 FINAL COMPLETE")
print("="*70)
print()
print(f"Best fit: β_m = {beta_m_best:.3f}, τ = {tau_best:.3f}")
print(f"χ²_min = {chi2_min:.2f}")
print(f"σ₈(IAM) ≈ {sigma8_eff:.3f}")
