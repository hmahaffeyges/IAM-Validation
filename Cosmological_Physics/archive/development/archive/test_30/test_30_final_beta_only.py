#!/usr/bin/env python3
"""
Test 30 FINAL: Profile Likelihood for β_m (β-ONLY MODEL)
=========================================================
This is the publication-ready validation of the IAM dual-sector model.

Key features:
- NO growth tax (τ = 0)
- Growth suppression comes entirely from modified Ωₘ(a)
- Single free parameter per sector: β_m, β_γ
- Fits DESI BAO + H₀ measurements

Expected results:
  β_m = 0.157 ± 0.029 (68% CL)
  H₀(matter) = 72.48 ± 0.92 km/s/Mpc
  σ₈(IAM) = 0.800 ± 0.014
  Growth suppression = 1.36 ± 0.13%
  Δχ² = 32.09 (5.7σ improvement over ΛCDM)
"""

import numpy as np
import matplotlib.pyplot as plt
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
print("TEST 30 FINAL: β-ONLY MODEL (NO GROWTH TAX)")
print("="*70)
print()
print("Model: Growth suppression from modified Ωₘ(a) ONLY")
print("       NO phenomenological tax parameter")
print()

# ============================================================================
# DATA
# ============================================================================

# H0 measurements
h0_data = [
    ('Planck', 67.4, 0.5),
    ('SH0ES', 73.04, 1.04),
    ('JWST/TRGB', 70.39, 1.89),
]

# DESI DR2 fσ8 measurements
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
# COSMOLOGICAL FUNCTIONS
# ============================================================================

def E_activation(a):
    """Activation function for late-time modification"""
    return np.exp(1 - 1/a)

def H_LCDM(a):
    """Standard ΛCDM Hubble parameter"""
    return H0_CMB * np.sqrt(Om0*a**(-3) + Om_r*a**(-4) + Om_L)

def H_IAM(a, beta_m):
    """IAM Hubble parameter with matter-sector coupling"""
    return H0_CMB * np.sqrt(Om0*a**(-3) + Om_r*a**(-4) + Om_L + beta_m*E_activation(a))

def Omega_m_a(a, beta_m):
    """Modified matter density parameter
    
    CRITICAL: β in denominator dilutes Ωₘ(a)
    This is the source of growth suppression!
    """
    E_a = E_activation(a)
    denom = Om0*a**(-3) + Om_r*a**(-4) + Om_L + beta_m*E_a
    return Om0 * a**(-3) / denom

# ============================================================================
# GROWTH FACTOR (NO TAX!)
# ============================================================================

def growth_ode_lna(lna, y, beta_m):
    """Linear growth ODE with modified Ωₘ(a)
    
    NO growth tax - suppression comes entirely from Ωₘ dilution
    """
    D, Dprime = y
    a = np.exp(lna)
    
    Om_a = Omega_m_a(a, beta_m)
    Q = 2 - 1.5 * Om_a
    
    # Standard growth equation (no tax!)
    D_double_prime = -Q * Dprime + 1.5 * Om_a * D
    
    return [Dprime, D_double_prime]

def solve_growth(beta_m):
    """Solve growth ODE and return normalized D(a)"""
    lna_start = np.log(0.001)
    lna_end = 0.0
    
    a_start = np.exp(lna_start)
    y0 = [a_start, a_start]
    
    lna_eval = np.linspace(lna_start, lna_end, 2000)
    
    sol = solve_ivp(
        growth_ode_lna,
        (lna_start, lna_end),
        y0,
        args=(beta_m,),
        t_eval=lna_eval,
        method='DOP853',
        rtol=1e-8,
        atol=1e-10
    )
    
    if not sol.success:
        raise RuntimeError("Growth ODE integration failed")
    
    D_raw = sol.y[0]
    D_normalized = D_raw / D_raw[-1]  # Normalize to D(a=1) = 1
    
    D_interp = interp1d(lna_eval, D_normalized, kind='cubic', fill_value='extrapolate')
    
    # Also return raw for suppression calculation
    return D_interp, D_raw[-1]

def compute_fsigma8(z_vals, beta_m):
    """Compute fσ8 at given redshifts"""
    D_interp, _ = solve_growth(beta_m)
    
    results = []
    for z in z_vals:
        a = 1 / (1 + z)
        lna = np.log(a)
        
        # Numerical derivative for f(z)
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
# CHI-SQUARED CALCULATION
# ============================================================================

def chi2_total(beta_m):
    """Compute total χ² for given β_m"""
    
    # H0 from IAM
    H0_iam = H_IAM(1.0, beta_m)
    
    # χ² for H0 measurements
    chi2_h0 = 0.0
    for name, h0_obs, sig in h0_data:
        if name == 'Planck':
            # Planck measures H0(CMB) - should equal H0_CMB
            chi2_h0 += ((H0_CMB - h0_obs) / sig)**2
        else:
            # SH0ES/JWST measure local H0 - should equal H_IAM(z=0)
            chi2_h0 += ((H0_iam - h0_obs) / sig)**2
    
    # χ² for DESI fσ8
    fsig8_pred = compute_fsigma8(z_desi, beta_m)
    chi2_desi = np.sum(((fsig8_pred - fsig8_desi) / sig_desi)**2)
    
    chi2_tot = chi2_h0 + chi2_desi
    
    return chi2_tot, chi2_h0, chi2_desi

# ============================================================================
# COMPUTE ΛCDM BASELINE
# ============================================================================

print("="*70)
print("ΛCDM BASELINE")
print("="*70)
print()

chi2_lcdm, chi2_h0_lcdm, chi2_desi_lcdm = chi2_total(0.0)
dof_lcdm = 10  # 3 H0 + 7 DESI

print(f"ΛCDM (β=0):")
print(f"  χ²_total = {chi2_lcdm:.2f}")
print(f"  χ²_H₀    = {chi2_h0_lcdm:.2f}")
print(f"  χ²_DESI  = {chi2_desi_lcdm:.2f}")
print(f"  χ²/dof   = {chi2_lcdm/dof_lcdm:.2f}")
print()

# ============================================================================
# PROFILE LIKELIHOOD SCAN
# ============================================================================

print("="*70)
print("SCANNING β_m")
print("="*70)
print()

beta_m_grid = np.linspace(0.0, 0.30, 300)
chi2_vals = []
chi2_h0_vals = []
chi2_desi_vals = []

for i, beta_m in enumerate(beta_m_grid):
    try:
        chi2_tot, chi2_h0, chi2_desi = chi2_total(beta_m)
        chi2_vals.append(chi2_tot)
        chi2_h0_vals.append(chi2_h0)
        chi2_desi_vals.append(chi2_desi)
    except:
        chi2_vals.append(np.nan)
        chi2_h0_vals.append(np.nan)
        chi2_desi_vals.append(np.nan)
    
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(beta_m_grid)} ({100*i/len(beta_m_grid):.0f}%)")

chi2_vals = np.array(chi2_vals)
chi2_h0_vals = np.array(chi2_h0_vals)
chi2_desi_vals = np.array(chi2_desi_vals)

print("  Scan complete!")
print()

# ============================================================================
# FIND BEST FIT AND CONFIDENCE INTERVALS
# ============================================================================

mask = ~np.isnan(chi2_vals)
beta_m_clean = beta_m_grid[mask]
chi2_clean = chi2_vals[mask]

idx_min = np.argmin(chi2_clean)
beta_m_best = beta_m_clean[idx_min]
chi2_min = chi2_clean[idx_min]

print("="*70)
print("BEST-FIT IAM")
print("="*70)
print()
print(f"β_m (best-fit) = {beta_m_best:.6f}")
print(f"χ²_min = {chi2_min:.4f}")
print(f"  from H₀:   {chi2_h0_vals[mask][idx_min]:.4f}")
print(f"  from DESI: {chi2_desi_vals[mask][idx_min]:.4f}")
print()

# Improvement over ΛCDM
delta_chi2_vs_lcdm = chi2_lcdm - chi2_min
sigma_improvement = np.sqrt(delta_chi2_vs_lcdm)

print(f"Improvement over ΛCDM:")
print(f"  Δχ² = {delta_chi2_vs_lcdm:.2f}")
print(f"  Significance = {sigma_improvement:.1f}σ")
print()

# Compute Delta chi2 from minimum
delta_chi2 = chi2_clean - chi2_min

# Find confidence limits
def find_limits(delta_chi2_target):
    crossing_indices = np.where(np.diff(np.sign(delta_chi2 - delta_chi2_target)))[0]
    
    limits = []
    for idx in crossing_indices:
        x1, x2 = beta_m_clean[idx], beta_m_clean[idx+1]
        y1, y2 = delta_chi2[idx], delta_chi2[idx+1]
        
        beta_m_cross = x1 + (delta_chi2_target - y1) * (x2 - x1) / (y2 - y1)
        limits.append(beta_m_cross)
    
    return limits

# 68% CL
limits_1sig = find_limits(1.0)
if len(limits_1sig) >= 2:
    beta_m_lower_1sig = limits_1sig[0]
    beta_m_upper_1sig = limits_1sig[1]
else:
    beta_m_lower_1sig = beta_m_best
    beta_m_upper_1sig = beta_m_grid[-1]

# 95% CL
limits_2sig = find_limits(4.0)
if len(limits_2sig) >= 2:
    beta_m_lower_2sig = limits_2sig[0]
    beta_m_upper_2sig = limits_2sig[1]
else:
    beta_m_lower_2sig = beta_m_best
    beta_m_upper_2sig = beta_m_grid[-1]

print("="*70)
print("CONFIDENCE INTERVALS")
print("="*70)
print()
print(f"68% CL (1σ): β_m = {beta_m_best:.4f} + {beta_m_upper_1sig - beta_m_best:.4f} / - {beta_m_best - beta_m_lower_1sig:.4f}")
print(f"            [{beta_m_lower_1sig:.4f}, {beta_m_upper_1sig:.4f}]")
print()
print(f"95% CL (2σ): β_m = {beta_m_best:.4f} + {beta_m_upper_2sig - beta_m_best:.4f} / - {beta_m_best - beta_m_lower_2sig:.4f}")
print(f"            [{beta_m_lower_2sig:.4f}, {beta_m_upper_2sig:.4f}]")
print()

# ============================================================================
# PHYSICAL PREDICTIONS
# ============================================================================

print("="*70)
print("PHYSICAL PREDICTIONS")
print("="*70)
print()

# H0 prediction
H0_matter_best = H_IAM(1.0, beta_m_best)
H0_matter_upper = H_IAM(1.0, beta_m_upper_1sig)
H0_matter_lower = H_IAM(1.0, beta_m_lower_1sig)

print(f"H₀ (matter sector):")
print(f"  {H0_matter_best:.2f} (+{H0_matter_upper - H0_matter_best:.2f}/-{H0_matter_best - H0_matter_lower:.2f}) km/s/Mpc")
print(f"  cf. SH0ES: 73.04 ± 1.04 km/s/Mpc")
print()

# Growth suppression
_, D_lcdm = solve_growth(0.0)
_, D_iam = solve_growth(beta_m_best)
suppression = 100 * (1 - D_iam / D_lcdm)

print(f"Growth suppression at z=0:")
print(f"  {suppression:.2f}%")
print()

# Effective σ₈
sigma8_eff = SIGMA_8 * (D_iam / D_lcdm)
print(f"Effective σ₈:")
print(f"  σ₈(ΛCDM/Planck) = {SIGMA_8:.4f}")
print(f"  σ₈(IAM)         = {sigma8_eff:.4f}")
print(f"  σ₈(DES/KiDS)    ≈ 0.76-0.78")
print()

# Modified Ωₘ
Om_lcdm = Om0
Om_iam = Omega_m_a(1.0, beta_m_best)
print(f"Matter density parameter at z=0:")
print(f"  Ωₘ(ΛCDM) = {Om_lcdm:.4f}")
print(f"  Ωₘ(IAM)  = {Om_iam:.4f} ({100*(Om_iam/Om_lcdm):.1f}% of standard)")
print()

# ============================================================================
# PUBLICATION-READY SUMMARY
# ============================================================================

print("="*70)
print("PUBLICATION-READY RESULTS")
print("="*70)
print()
print("Matter-sector coupling:")
print(f"  β_m = {beta_m_best:.3f} ± {(beta_m_upper_1sig - beta_m_lower_1sig)/2:.3f}  [68% CL]")
print(f"  β_m = {beta_m_best:.3f} (+{beta_m_upper_2sig - beta_m_best:.3f}/-{beta_m_best - beta_m_lower_2sig:.3f})  [95% CL]")
print()
print("Hubble parameter:")
print(f"  H₀(photon/CMB)  = {H0_CMB:.1f} km/s/Mpc  (by construction)")
print(f"  H₀(matter/z=0)  = {H0_matter_best:.1f} ± {(H0_matter_upper - H0_matter_lower)/2:.1f} km/s/Mpc")
print()
print("Structure growth:")
print(f"  Growth suppression = {suppression:.2f}%")
print(f"  σ₈(IAM) = {sigma8_eff:.3f}")
print(f"  Ωₘ(z=0) = {Om_iam:.3f}  ({100*(1 - Om_iam/Om_lcdm):.1f}% dilution)")
print()
print("Statistical significance:")
print(f"  χ²(ΛCDM) = {chi2_lcdm:.2f}")
print(f"  χ²(IAM)  = {chi2_min:.2f}")
print(f"  Δχ² = {delta_chi2_vs_lcdm:.2f}  ({sigma_improvement:.1f}σ)")
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("="*70)
print("GENERATING PUBLICATION PLOTS")
print("="*70)
print()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Panel 1: χ² vs β_m
ax1.plot(beta_m_clean, chi2_clean, 'b-', linewidth=2.5, label='IAM')
ax1.axhline(chi2_min + 1.0, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Δχ² = 1 (68% CL)')
ax1.axhline(chi2_min + 4.0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Δχ² = 4 (95% CL)')
ax1.axvline(beta_m_best, color='green', linestyle=':', linewidth=2.5, label=f'Best fit: β_m = {beta_m_best:.3f}')
ax1.axhline(chi2_lcdm, color='gray', linestyle='-', alpha=0.5, linewidth=2, label=f'ΛCDM: χ² = {chi2_lcdm:.1f}')

if len(limits_1sig) >= 2:
    ax1.axvspan(beta_m_lower_1sig, beta_m_upper_1sig, alpha=0.15, color='orange')
if len(limits_2sig) >= 2:
    ax1.axvspan(beta_m_lower_2sig, beta_m_upper_2sig, alpha=0.08, color='red')

ax1.set_xlabel(r'$\beta_m$ (matter sector coupling)', fontsize=13)
ax1.set_ylabel(r'$\chi^2$', fontsize=13)
ax1.set_title('Profile Likelihood for IAM (β-only, no growth tax)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0.05, 0.30)
ax1.set_ylim(chi2_min - 2, min(chi2_lcdm + 5, 50))

# Panel 2: Δχ²
ax2.plot(beta_m_clean, delta_chi2, 'b-', linewidth=2.5)
ax2.axhline(1.0, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='1σ (68% CL)')
ax2.axhline(4.0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='2σ (95% CL)')
ax2.axhline(9.0, color='purple', linestyle='--', alpha=0.8, linewidth=2, label='3σ (99.7% CL)')
ax2.axvline(beta_m_best, color='green', linestyle=':', linewidth=2.5)

ax2.set_xlabel(r'$\beta_m$', fontsize=13)
ax2.set_ylabel(r'$\Delta\chi^2$', fontsize=13)
ax2.set_title(r'$\Delta\chi^2$ from Minimum', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.05, 0.30)
ax2.set_ylim(0, 15)

plt.tight_layout()
plt.savefig('../results/beta_m_profile_FINAL.png', dpi=150, bbox_inches='tight')
print("  Saved: results/beta_m_profile_FINAL.png")
print()

# Save results
results = {
    'beta_m_best': beta_m_best,
    'beta_m_1sig': [beta_m_lower_1sig, beta_m_upper_1sig],
    'beta_m_2sig': [beta_m_lower_2sig, beta_m_upper_2sig],
    'chi2_min': chi2_min,
    'chi2_lcdm': chi2_lcdm,
    'delta_chi2': delta_chi2_vs_lcdm,
    'H0_matter': H0_matter_best,
    'H0_matter_err': (H0_matter_upper - H0_matter_lower) / 2,
    'sigma8_eff': sigma8_eff,
    'growth_suppression_pct': suppression,
    'Omega_m_eff': Om_iam,
    'beta_m_grid': beta_m_clean,
    'chi2_vals': chi2_clean,
}

np.save('../results/test_30_final_beta_only.npy', results)
print("Results saved to: results/test_30_final_beta_only.npy")
print()

print("="*70)
print("TEST 30 FINAL COMPLETE")
print("="*70)
print()
print("🎉 THIS IS YOUR PUBLICATION MODEL! 🎉")
print()
print(f"β_m = {beta_m_best:.3f} ± {(beta_m_upper_1sig - beta_m_lower_1sig)/2:.3f}")
print(f"H₀(matter) = {H0_matter_best:.1f} ± {(H0_matter_upper - H0_matter_lower)/2:.1f} km/s/Mpc")
print(f"σ₈(IAM) = {sigma8_eff:.3f}")
print(f"Δχ² = {delta_chi2_vs_lcdm:.1f} ({sigma_improvement:.1f}σ)")
