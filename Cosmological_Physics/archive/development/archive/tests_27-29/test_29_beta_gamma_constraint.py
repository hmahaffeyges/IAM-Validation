#!/usr/bin/env python3
"""
Test 29: Precise Constraint on beta_gamma
==========================================
1D likelihood scan to determine confidence limits on photon coupling

Goal: Report beta_gamma = 0.000 +X.XXX (95% upper limit)

Method:
------
1. Compute chi2(beta_gamma) for photon-sector observables
2. Find Delta chi2 = 1 (1 sigma) and Delta chi2 = 4 (2 sigma, 95% CL)
3. Report constraint: beta_gamma < X.XXX (95% CL)

Observables used:
-----------------
- Planck CMB: theta_s = 0.0104110 +/- 0.0000031
- Planck H0:  67.4 +/- 0.5 km/s/Mpc

These are PHOTON-SECTOR measurements (CMB photons)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq

c = 299792.458
H0_CMB = 67.4
Om0 = 0.315
Om_r = 9.24e-5
Om_L = 1 - Om0 - Om_r

# Observational data (photon sector)
theta_obs = 0.0104110
sigma_theta = 0.0000031
H0_planck_obs = 67.4
sigma_H0_planck = 0.5

r_s = 144.43  # Mpc

print("="*70)
print("TEST 29: BETA_GAMMA LIKELIHOOD CONSTRAINT")
print("="*70)
print()
print("Observables (photon sector):")
print(f"  theta_s (Planck 2018): {theta_obs:.8f} +/- {sigma_theta:.8f} rad")
print(f"  H0 (Planck 2018):      {H0_planck_obs:.2f} +/- {sigma_H0_planck:.2f} km/s/Mpc")
print()

# ============================================================================
# PHOTON SECTOR FUNCTIONS
# ============================================================================

def E_activation(a):
    return np.exp(1 - 1/a)

def H_photon(a, beta_gamma):
    return H0_CMB * np.sqrt(
        Om0 * a**(-3) +
        Om_r * a**(-4) +
        Om_L +
        beta_gamma * E_activation(a)
    )

def comoving_distance(z_source, beta_gamma):
    z_vals = np.linspace(0, z_source, 50000)
    a_vals = 1 / (1 + z_vals)
    integrand = c / H_photon(a_vals, beta_gamma)
    chi = np.trapezoid(integrand, z_vals)
    return chi

def theta_s_pred(beta_gamma):
    chi = comoving_distance(1090, beta_gamma)
    return r_s / chi

def H0_pred(beta_gamma):
    return H_photon(1.0, beta_gamma)

# ============================================================================
# CHI-SQUARED CALCULATION
# ============================================================================

def chi2_photon(beta_gamma):
    """
    Chi-squared for photon sector
    
    Includes:
    1. theta_s from CMB acoustic peak
    2. H0 inferred from Planck CMB
    """
    # theta_s
    theta_p = theta_s_pred(beta_gamma)
    chi2_theta = ((theta_p - theta_obs) / sigma_theta)**2
    
    # H0
    H0_p = H0_pred(beta_gamma)
    chi2_H0 = ((H0_p - H0_planck_obs) / sigma_H0_planck)**2
    
    chi2_total = chi2_theta + chi2_H0
    
    return chi2_total, chi2_theta, chi2_H0

# ============================================================================
# LIKELIHOOD SCAN
# ============================================================================

print("="*70)
print("LIKELIHOOD SCAN")
print("="*70)
print()

# Fine grid around beta_gamma = 0
beta_gamma_grid = np.linspace(0.0, 0.10, 1000)

chi2_vals = []
chi2_theta_vals = []
chi2_H0_vals = []

print(f"Scanning {len(beta_gamma_grid)} values from 0.0 to 0.10...")

for i, bg in enumerate(beta_gamma_grid):
    chi2_tot, chi2_th, chi2_h0 = chi2_photon(bg)
    chi2_vals.append(chi2_tot)
    chi2_theta_vals.append(chi2_th)
    chi2_H0_vals.append(chi2_h0)
    
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(beta_gamma_grid)} ({100*i/len(beta_gamma_grid):.0f}%)")

chi2_vals = np.array(chi2_vals)
chi2_theta_vals = np.array(chi2_theta_vals)
chi2_H0_vals = np.array(chi2_H0_vals)

print("  Scan complete!")
print()

# Find minimum
idx_min = np.argmin(chi2_vals)
beta_gamma_best = beta_gamma_grid[idx_min]
chi2_min = chi2_vals[idx_min]

print("="*70)
print("BEST-FIT RESULTS")
print("="*70)
print()
print(f"Best-fit beta_gamma: {beta_gamma_best:.6f}")
print(f"Minimum chi2:        {chi2_min:.4f}")
print(f"  from theta_s:      {chi2_theta_vals[idx_min]:.4f}")
print(f"  from H0:           {chi2_H0_vals[idx_min]:.4f}")
print()

# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================

print("="*70)
print("CONFIDENCE INTERVALS")
print("="*70)
print()

# Delta chi2 thresholds
# For 1 parameter:
#   1 sigma (68.3% CL): Delta chi2 = 1.00
#   2 sigma (95.4% CL): Delta chi2 = 4.00
#   3 sigma (99.7% CL): Delta chi2 = 9.00

delta_chi2_vals = chi2_vals - chi2_min

# Create interpolator
interp_delta_chi2 = interp1d(beta_gamma_grid, delta_chi2_vals, 
                             kind='cubic', bounds_error=False, fill_value=np.inf)

# Find confidence limits
def find_upper_limit(delta_chi2_target):
    """Find beta_gamma where Delta chi2 = target"""
    # We know chi2 increases with beta_gamma
    # Find where it crosses the threshold
    try:
        # Search from best-fit to end of grid
        mask = beta_gamma_grid >= beta_gamma_best
        bg_search = beta_gamma_grid[mask]
        delta_search = delta_chi2_vals[mask]
        
        # Find crossing
        idx_cross = np.where(delta_search >= delta_chi2_target)[0]
        if len(idx_cross) > 0:
            # Linear interpolation around crossing
            i = idx_cross[0]
            if i > 0:
                bg1, bg2 = bg_search[i-1], bg_search[i]
                d1, d2 = delta_search[i-1], delta_search[i]
                # Linear interp
                bg_limit = bg1 + (delta_chi2_target - d1) * (bg2 - bg1) / (d2 - d1)
                return bg_limit
            else:
                return bg_search[i]
        else:
            return None
    except:
        return None

# 1 sigma (68.3% CL)
limit_1sig = find_upper_limit(1.00)

# 2 sigma (95.4% CL)
limit_2sig = find_upper_limit(4.00)

# 3 sigma (99.7% CL)
limit_3sig = find_upper_limit(9.00)

print(f"Delta chi2 = 1.00 (1 sigma, 68.3% CL):")
if limit_1sig is not None:
    print(f"  beta_gamma < {limit_1sig:.6f}")
else:
    print(f"  beta_gamma < {beta_gamma_grid[-1]:.4f} (unconstrained)")
print()

print(f"Delta chi2 = 4.00 (2 sigma, 95.4% CL):")
if limit_2sig is not None:
    print(f"  beta_gamma < {limit_2sig:.6f}")
    limit_95 = limit_2sig
else:
    print(f"  beta_gamma < {beta_gamma_grid[-1]:.4f} (unconstrained)")
    limit_95 = beta_gamma_grid[-1]
print()

print(f"Delta chi2 = 9.00 (3 sigma, 99.7% CL):")
if limit_3sig is not None:
    print(f"  beta_gamma < {limit_3sig:.6f}")
else:
    print(f"  beta_gamma < {beta_gamma_grid[-1]:.4f} (unconstrained)")
print()

# ============================================================================
# PUBLISHABLE CONSTRAINT
# ============================================================================

print("="*70)
print("PUBLISHABLE CONSTRAINT")
print("="*70)
print()

if limit_2sig is not None:
    print(f"RESULT:")
    print(f"  beta_gamma = {beta_gamma_best:.4f} + {limit_2sig - beta_gamma_best:.4f}")
    print(f"             = {beta_gamma_best:.4f} +{limit_2sig - beta_gamma_best:.4f}/-{beta_gamma_best:.4f}")
    print()
    print(f"95% Confidence Upper Limit:")
    print(f"  beta_gamma < {limit_2sig:.4f}")
    print()
    print(f"Comparison to matter sector:")
    beta_m = 0.18
    print(f"  beta_m     = {beta_m:.4f} (matter coupling)")
    print(f"  beta_gamma < {limit_2sig:.4f} (photon coupling, 95% CL)")
    print(f"  Ratio:       beta_gamma/beta_m < {limit_2sig/beta_m:.3f}")
    print()
    
    if limit_2sig < 0.01:
        print("INTERPRETATION:")
        print("  Photons couple < 5% as strongly as matter")
        print("  Strong empirical support for sector separation")
    elif limit_2sig < 0.05:
        print("INTERPRETATION:")
        print("  Photons couple < 30% as strongly as matter")
        print("  Moderate empirical support for sector separation")
    else:
        print("INTERPRETATION:")
        print("  Photon coupling not tightly constrained")
        print("  Data consistent with partial sector separation")
else:
    print("Constraint not tight with current data")

print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("="*70)
print("GENERATING PLOTS")
print("="*70)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Chi2 vs beta_gamma
ax = axes[0, 0]
ax.plot(beta_gamma_grid, chi2_vals, 'b-', linewidth=2)
ax.axhline(chi2_min + 1.0, color='orange', linestyle='--', 
           label=r'$\Delta\chi^2=1$ (1σ)')
ax.axhline(chi2_min + 4.0, color='red', linestyle='--', 
           label=r'$\Delta\chi^2=4$ (2σ)')
ax.axvline(beta_gamma_best, color='green', linestyle=':', 
           label=f'Best fit: {beta_gamma_best:.4f}')
if limit_2sig is not None:
    ax.axvline(limit_2sig, color='red', linestyle=':', alpha=0.7,
               label=f'95% limit: {limit_2sig:.4f}')
ax.set_xlabel(r'$\beta_\gamma$ (photon coupling)', fontsize=12)
ax.set_ylabel(r'$\chi^2$', fontsize=12)
ax.set_title('Likelihood Scan: Photon Sector', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.05)

# Plot 2: Delta Chi2 (zoomed)
ax = axes[0, 1]
ax.plot(beta_gamma_grid, delta_chi2_vals, 'b-', linewidth=2)
ax.axhline(1.0, color='orange', linestyle='--', label='1σ (68% CL)')
ax.axhline(4.0, color='red', linestyle='--', label='2σ (95% CL)')
ax.axhline(9.0, color='purple', linestyle='--', label='3σ (99.7% CL)')
ax.set_xlabel(r'$\beta_\gamma$', fontsize=12)
ax.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
ax.set_title(r'$\Delta\chi^2$ from Minimum', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.05)
ax.set_ylim(0, 10)

# Plot 3: theta_s vs beta_gamma
ax = axes[1, 0]
theta_pred_vals = [theta_s_pred(bg) for bg in beta_gamma_grid]
ax.plot(beta_gamma_grid, theta_pred_vals, 'b-', linewidth=2, label='IAM prediction')
ax.axhline(theta_obs, color='red', linestyle='-', linewidth=2, label='Planck')
ax.fill_between([0, 0.1], [theta_obs - sigma_theta]*2, [theta_obs + sigma_theta]*2,
                alpha=0.3, color='red', label='±1σ')
ax.set_xlabel(r'$\beta_\gamma$', fontsize=12)
ax.set_ylabel(r'$\theta_s$ (rad)', fontsize=12)
ax.set_title('CMB Acoustic Scale vs Photon Coupling', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.05)

# Plot 4: H0 vs beta_gamma
ax = axes[1, 1]
H0_pred_vals = [H0_pred(bg) for bg in beta_gamma_grid]
ax.plot(beta_gamma_grid, H0_pred_vals, 'b-', linewidth=2, label='IAM prediction')
ax.axhline(H0_planck_obs, color='red', linestyle='-', linewidth=2, label='Planck')
ax.fill_between([0, 0.1], [H0_planck_obs - sigma_H0_planck]*2, 
                [H0_planck_obs + sigma_H0_planck]*2,
                alpha=0.3, color='red', label='±1σ')
ax.axhline(73.04, color='orange', linestyle='--', linewidth=2, label='SH0ES (matter)')
ax.set_xlabel(r'$\beta_\gamma$', fontsize=12)
ax.set_ylabel(r'$H_0$ (km/s/Mpc)', fontsize=12)
ax.set_title('Hubble Constant vs Photon Coupling', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.05)
ax.set_ylim(66, 75)

plt.tight_layout()
plt.savefig('../results/beta_gamma_constraint.png', dpi=150, bbox_inches='tight')
print("  Saved: results/beta_gamma_constraint.png")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

results = {
    'beta_gamma_best': beta_gamma_best,
    'chi2_min': chi2_min,
    'limit_1sigma': limit_1sig,
    'limit_2sigma': limit_2sig,
    'limit_3sigma': limit_3sig,
    'beta_gamma_grid': beta_gamma_grid,
    'chi2_vals': chi2_vals,
    'delta_chi2_vals': delta_chi2_vals,
}

np.save('../results/test_29_beta_gamma_constraint.npy', results)
print("Results saved to: results/test_29_beta_gamma_constraint.npy")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*70)
print("SUMMARY FOR PUBLICATION")
print("="*70)
print()
print("Photon Sector Coupling to Informational Actualization:")
print()
if limit_2sig is not None:
    print(f"  beta_gamma = {beta_gamma_best:.4f} (+{limit_2sig:.4f}, -{beta_gamma_best:.4f}) [95% CL]")
    print()
    print(f"  95% Upper Limit: beta_gamma < {limit_2sig:.4f}")
    print()
    print(f"Matter Sector Coupling (established):")
    print(f"  beta_m = 0.1800 ± 0.0XXX (from BAO/H0 fits)")
    print()
    print(f"Empirical Sector Separation:")
    print(f"  beta_gamma / beta_m < {limit_2sig/0.18:.3f} (95% CL)")
    print()
    
    sigma_level = "strong" if limit_2sig < 0.01 else ("moderate" if limit_2sig < 0.05 else "weak")
    print(f"Evidence for sector separation: {sigma_level.upper()}")
    
print()
print("="*70)
print("Test complete.")
print("="*70)
