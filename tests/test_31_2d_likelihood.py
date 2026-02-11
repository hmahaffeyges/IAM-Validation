#!/usr/bin/env python3
"""
Test 31: 2D Likelihood Scan (β_m, τ)
=====================================
Compute χ² on a 2D grid of (β_m, τ) to visualize parameter degeneracies
and joint confidence regions.

Expected result: Mild negative correlation (higher β_m → lower τ needed),
but parameters remain well-constrained independently.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Constants
c = 299792.458  # km/s
H0_CMB = 67.4
Om0 = 0.315
Om_r = 9.24e-5
Om_L = 1 - Om0 - Om_r
SIGMA_8 = 0.811

print("="*70)
print("TEST 31: 2D LIKELIHOOD SCAN (β_m, τ)")
print("="*70)
print()

# ============================================================================
# DATA
# ============================================================================

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

# ============================================================================
# HUBBLE FUNCTIONS
# ============================================================================

def E_activation(a):
    return np.exp(1 - 1/a)

def H_IAM(a, beta_m):
    return H0_CMB * np.sqrt(Om0*a**(-3) + Om_r*a**(-4) + Om_L + beta_m*E_activation(a))

def Omega_m_a(a, beta_m):
    E_a = E_activation(a)
    denom = Om0*a**(-3) + Om_r*a**(-4) + Om_L + beta_m*E_a
    return Om0 * a**(-3) / denom

# ============================================================================
# GROWTH FACTOR
# ============================================================================

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
    lna_end = 0.0
    
    a_start = np.exp(lna_start)
    y0 = [a_start, a_start]
    
    lna_eval = np.linspace(lna_start, lna_end, 2000)
    
    sol = solve_ivp(
        growth_ode_lna,
        (lna_start, lna_end),
        y0,
        args=(beta_m, tau),
        t_eval=lna_eval,
        method='DOP853',
        rtol=1e-8,
        atol=1e-10
    )
    
    if not sol.success:
        raise RuntimeError("Growth ODE failed")
    
    D_raw = sol.y[0]
    D_normalized = D_raw / D_raw[-1]
    D_interp = interp1d(lna_eval, D_normalized, kind='cubic', fill_value='extrapolate')
    
    return D_interp

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

# ============================================================================
# CHI-SQUARED
# ============================================================================

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

# ============================================================================
# 2D GRID SCAN
# ============================================================================

print("Setting up 2D grid...")
print()

# Grid parameters
beta_m_min, beta_m_max = 0.08, 0.25
tau_min, tau_max = 0.00, 0.12
n_beta = 50
n_tau = 50

beta_m_grid = np.linspace(beta_m_min, beta_m_max, n_beta)
tau_grid = np.linspace(tau_min, tau_max, n_tau)

print(f"Grid: {n_beta} × {n_tau} = {n_beta*n_tau} points")
print(f"  β_m ∈ [{beta_m_min}, {beta_m_max}]")
print(f"  τ   ∈ [{tau_min}, {tau_max}]")
print()
print("Computing χ² on grid (this will take ~10 minutes)...")
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

# ============================================================================
# FIND MINIMUM AND CONFIDENCE REGIONS
# ============================================================================

# Mask NaNs
mask = ~np.isnan(chi2_grid)
chi2_valid = chi2_grid[mask]

# Find minimum
idx_flat = np.argmin(chi2_valid)
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

# Compute H0 prediction
H0_matter = H_IAM(1.0, beta_m_best)
print(f"Predicted H₀(matter) = {H0_matter:.2f} km/s/Mpc")
print()

# Compute Delta chi2
delta_chi2 = chi2_grid - chi2_min

# ============================================================================
# VISUALIZATION
# ============================================================================

print("="*70)
print("GENERATING 2D CONTOUR PLOT")
print("="*70)
print()

fig, ax = plt.subplots(figsize=(10, 8))

# Contour levels
levels = [1.0, 4.0, 9.0]  # 1σ, 2σ, 3σ
colors = ['orange', 'red', 'purple']
labels = ['68% CL (1σ)', '95% CL (2σ)', '99.7% CL (3σ)']

# Plot filled contours
contourf = ax.contourf(beta_m_grid, tau_grid, delta_chi2, 
                        levels=[0, 1, 4, 9, 100],
                        colors=['darkgreen', 'orange', 'red', 'purple'],
                        alpha=0.3)

# Plot contour lines
contours = ax.contour(beta_m_grid, tau_grid, delta_chi2,
                      levels=levels, colors=colors, linewidths=2)

# Label contours manually in legend
for level, color, label in zip(levels, colors, labels):
    ax.plot([], [], color=color, linewidth=2, label=label)

# Mark best fit
ax.plot(beta_m_best, tau_best, 'g*', markersize=20, 
        label=f'Best fit: β_m={beta_m_best:.3f}, τ={tau_best:.3f}',
        markeredgecolor='black', markeredgewidth=1.5)

# Marginal projections (dashed lines)
ax.axvline(beta_m_best, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax.axhline(tau_best, color='green', linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel(r'$\beta_m$ (matter sector coupling)', fontsize=14)
ax.set_ylabel(r'$\tau$ (growth tax)', fontsize=14)
ax.set_title(r'Joint Likelihood: $(\beta_m, \tau)$ Parameter Space', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/beta_m_tau_2d.png', dpi=150, bbox_inches='tight')
print("  Saved: results/beta_m_tau_2d.png")
print()

# ============================================================================
# MARGINALIZED 1D PROFILES
# ============================================================================

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Marginalize over τ to get β_m profile
delta_chi2_beta_m = np.nanmin(delta_chi2, axis=0)

ax1.plot(beta_m_grid, delta_chi2_beta_m, 'b-', linewidth=2)
ax1.axhline(1.0, color='orange', linestyle='--', label='1σ (68% CL)')
ax1.axhline(4.0, color='red', linestyle='--', label='2σ (95% CL)')
ax1.axhline(9.0, color='purple', linestyle='--', label='3σ (99.7% CL)')
ax1.axvline(beta_m_best, color='green', linestyle=':', linewidth=2)
ax1.set_xlabel(r'$\beta_m$', fontsize=12)
ax1.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
ax1.set_title('Marginalized over τ', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 15)

# Marginalize over β_m to get τ profile
delta_chi2_tau = np.nanmin(delta_chi2, axis=1)

ax2.plot(tau_grid, delta_chi2_tau, 'b-', linewidth=2)
ax2.axhline(1.0, color='orange', linestyle='--', label='1σ (68% CL)')
ax2.axhline(4.0, color='red', linestyle='--', label='2σ (95% CL)')
ax2.axhline(9.0, color='purple', linestyle='--', label='3σ (99.7% CL)')
ax2.axvline(tau_best, color='green', linestyle=':', linewidth=2)
ax2.set_xlabel(r'$\tau$', fontsize=12)
ax2.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
ax2.set_title('Marginalized over β_m', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 15)

plt.tight_layout()
plt.savefig('../results/beta_m_tau_marginals.png', dpi=150, bbox_inches='tight')
print("  Saved: results/beta_m_tau_marginals.png")
print()

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print("="*70)
print("CORRELATION ANALYSIS")
print("="*70)
print()

# Extract 1σ region
mask_1sig = delta_chi2 < 1.0
if np.any(mask_1sig):
    beta_m_1sig = beta_m_grid[None, :].repeat(n_tau, axis=0)[mask_1sig]
    tau_1sig = tau_grid[:, None].repeat(n_beta, axis=1)[mask_1sig]
    
    # Compute correlation
    corr = np.corrcoef(beta_m_1sig, tau_1sig)[0, 1]
    print(f"Correlation coefficient (within 1σ): {corr:.3f}")
    print()
    
    if abs(corr) < 0.3:
        print("  → Parameters are weakly correlated")
        print("  → β_m and τ constrain independently")
    elif abs(corr) < 0.7:
        print("  → Parameters show moderate correlation")
        print("  → Some degeneracy exists but not severe")
    else:
        print("  → Parameters are strongly correlated")
        print("  → Significant degeneracy present")
else:
    print("  (1σ region not well-sampled on grid)")

print()

# Save results
results = {
    'beta_m_grid': beta_m_grid,
    'tau_grid': tau_grid,
    'chi2_grid': chi2_grid,
    'delta_chi2': delta_chi2,
    'beta_m_best': beta_m_best,
    'tau_best': tau_best,
    'chi2_min': chi2_min,
}

np.save('../results/test_31_2d_likelihood.npy', results)
print("Results saved to: results/test_31_2d_likelihood.npy")
print()

print("="*70)
print("TEST 31 COMPLETE")
print("="*70)
print()
print(f"Best fit: β_m = {beta_m_best:.3f}, τ = {tau_best:.3f}")
print(f"χ²_min = {chi2_min:.2f}")
