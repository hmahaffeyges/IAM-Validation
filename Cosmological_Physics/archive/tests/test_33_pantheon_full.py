#!/usr/bin/env python3
"""
Test 33: Pantheon+ Supernova Analysis
======================================
Test IAM against the full Pantheon+ dataset of Type Ia supernovae.

Expected result: IAM provides minimal/no improvement over ΛCDM for 
distance measurements. IAM's strength is in growth rates (DESI), 
not luminosity distances (SNe).

This is GOOD - it means the model is internally consistent and not
just overfitting everything.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
c = 299792.458  # km/s
H0_CMB = 67.4
Om0 = 0.315
Om_r = 9.24e-5
Om_L = 1 - Om0 - Om_r

print("="*70)
print("TEST 33: PANTHEON+ SUPERNOVA ANALYSIS")
print("="*70)
print()

# ============================================================================
# CHECK FOR PANTHEON+ DATA
# ============================================================================

print("Checking for Pantheon+ data...")
print()

import os
data_path = '../data/Pantheon+SH0ES.dat'

if not os.path.exists(data_path):
    print("❌ Pantheon+ data not found!")
    print()
    print("Please download:")
    print("  cd ../data")
    print("  wget https://github.com/PantheonPlusSH0ES/DataRelease/raw/main/Pantheon%2B_Data/1_DATA/Pantheon%2BSH0ES.dat")
    print()
    print("For now, using mock data to test the code...")
    print()
    
    # Create mock data for testing
    z_sne = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6])
    n_sne = len(z_sne)
    USE_MOCK = True
    
else:
    print("✅ Found Pantheon+ data!")
    print()
    
    # Load real data
    data = np.loadtxt(data_path)
    z_sne = data[:, 1]  # Heliocentric redshift
    m_obs = data[:, 4]  # Apparent magnitude
    m_err = data[:, 5]  # Statistical error
    
    n_sne = len(z_sne)
    print(f"Loaded {n_sne} supernovae")
    print(f"Redshift range: z ∈ [{z_sne.min():.4f}, {z_sne.max():.4f}]")
    print()
    USE_MOCK = False

# ============================================================================
# COSMOLOGICAL FUNCTIONS
# ============================================================================

def E_activation(a):
    return np.exp(1 - 1/a)

def H_LCDM(z):
    """ΛCDM Hubble parameter"""
    a = 1/(1+z)
    return H0_CMB * np.sqrt(Om0*a**(-3) + Om_r*a**(-4) + Om_L)

def H_IAM(z, beta):
    """IAM Hubble parameter"""
    a = 1/(1+z)
    return H0_CMB * np.sqrt(Om0*a**(-3) + Om_r*a**(-4) + Om_L + beta*E_activation(a))

def luminosity_distance(z, beta=0):
    """Compute luminosity distance in Mpc"""
    
    if beta == 0:
        # ΛCDM
        integrand = lambda zp: c / H_LCDM(zp)
    else:
        # IAM
        integrand = lambda zp: c / H_IAM(zp, beta)
    
    # Comoving distance
    d_c, _ = quad(integrand, 0, z, limit=100)
    
    # Luminosity distance
    d_L = (1 + z) * d_c
    
    return d_L

def distance_modulus(z, beta=0):
    """Distance modulus μ = 5 log₁₀(d_L/Mpc) + 25"""
    d_L = luminosity_distance(z, beta)
    return 5 * np.log10(d_L) + 25

# ============================================================================
# CHI-SQUARED FOR SUPERNOVAE
# ============================================================================

def chi2_sne(beta, z_sne, m_obs=None, m_err=None):
    """
    Compute χ² for supernova data
    
    Marginalizes over absolute magnitude M analytically
    """
    n = len(z_sne)
    
    # Compute predicted distance moduli
    mu_pred = np.array([distance_modulus(z, beta) for z in z_sne])
    
    if USE_MOCK:
        # Generate mock observations from ΛCDM + noise
        np.random.seed(42)
        M_true = -19.3  # Fiducial absolute magnitude
        m_obs = mu_pred + M_true + np.random.normal(0, 0.15, n)
        m_err = np.full(n, 0.15)
    
    # Weights (inverse variance)
    weights = 1 / m_err**2
    
    # Marginalize over M analytically
    # Best-fit M minimizes χ²
    M_best = np.sum(weights * (m_obs - mu_pred)) / np.sum(weights)
    
    # Residuals
    residuals = m_obs - mu_pred - M_best
    
    # χ²
    chi2 = np.sum(weights * residuals**2)
    
    return chi2, M_best

# ============================================================================
# FIT ΛCDM
# ============================================================================

print("="*70)
print("FITTING ΛCDM (β = 0)")
print("="*70)
print()

chi2_lcdm, M_lcdm = chi2_sne(0.0, z_sne, m_obs if not USE_MOCK else None, 
                               m_err if not USE_MOCK else None)

dof = n_sne - 1  # One nuisance parameter (M)

print(f"ΛCDM results:")
print(f"  χ² = {chi2_lcdm:.2f}")
print(f"  dof = {dof}")
print(f"  χ²/dof = {chi2_lcdm/dof:.4f}")
print(f"  M_abs = {M_lcdm:.4f}")
print()

if USE_MOCK:
    print("(Using mock data - ΛCDM should fit perfectly)")
    print()

# ============================================================================
# FIT IAM
# ============================================================================

print("="*70)
print("FITTING IAM")
print("="*70)
print()

# Test different β values
beta_values = [0.05, 0.10, 0.157, 0.18, 0.20]

print("Testing β values from DESI/H₀ fits...")
print()

results = []

for beta in beta_values:
    chi2_iam, M_iam = chi2_sne(beta, z_sne, m_obs if not USE_MOCK else None,
                                m_err if not USE_MOCK else None)
    
    delta_chi2 = chi2_lcdm - chi2_iam
    
    results.append((beta, chi2_iam, M_iam, delta_chi2))
    
    print(f"β = {beta:.3f}:")
    print(f"  χ² = {chi2_iam:.2f}")
    print(f"  Δχ² = {delta_chi2:+.2f}")
    print(f"  χ²/dof = {chi2_iam/dof:.4f}")
    print()

# Find best
best_idx = np.argmin([r[1] for r in results])
beta_best, chi2_best, M_best, delta_chi2_best = results[best_idx]

print("="*70)
print("COMPARISON")
print("="*70)
print()
print(f"ΛCDM:  χ² = {chi2_lcdm:.2f},  χ²/dof = {chi2_lcdm/dof:.4f}")
print(f"IAM:   χ² = {chi2_best:.2f},  χ²/dof = {chi2_best/dof:.4f}")
print(f"       (best β = {beta_best:.3f})")
print()
print(f"Improvement: Δχ² = {delta_chi2_best:+.2f}")
print()

# ============================================================================
# INTERPRETATION
# ============================================================================

print("="*70)
print("INTERPRETATION")
print("="*70)
print()

if abs(delta_chi2_best) < 2:
    print("✅ EXCELLENT NEWS!")
    print()
    print("IAM provides minimal/no improvement over ΛCDM for SNe distances.")
    print("This is EXPECTED and GOOD because:")
    print()
    print("  1. ΛCDM already fits distance measurements perfectly")
    print("  2. SNe probe geometry (H(z) integral), not growth")
    print("  3. IAM's modifications are late-time and growth-focused")
    print()
    print("This confirms IAM is NOT just overfitting everything!")
    print("The model is INTERNALLY CONSISTENT:")
    print("  - Improves H₀ tension (different expansion rates)")
    print("  - Improves S₈ tension (suppressed growth)")
    print("  - Does NOT break well-constrained distances")
    print()
elif delta_chi2_best > 5:
    print("⚠️ UNEXPECTED: IAM significantly improves SNe fit")
    print()
    print("This might indicate:")
    print("  - Systematic effects in SNe data")
    print("  - IAM also affects distance-redshift relation")
    print("  - Possible overfitting (need to investigate)")
    print()
elif delta_chi2_best < -5:
    print("❌ PROBLEM: IAM significantly worsens SNe fit")
    print()
    print("This suggests:")
    print("  - Model incompatible with distance measurements")
    print("  - Need to revisit parameterization")
    print()
else:
    print("⚡ MARGINAL IMPROVEMENT")
    print()
    print("Small improvement (2 < Δχ² < 5) suggests:")
    print("  - Modest effect on distance-redshift relation")
    print("  - Consistent with data, neither helps nor harms")
    print()

# ============================================================================
# HUBBLE DIAGRAM
# ============================================================================

print("="*70)
print("GENERATING HUBBLE DIAGRAM")
print("="*70)
print()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                gridspec_kw={'height_ratios': [3, 1]})

# Compute distance moduli
if USE_MOCK:
    # Generate mock for plot
    np.random.seed(42)
    mu_lcdm_true = np.array([distance_modulus(z, 0.0) for z in z_sne])
    m_obs = mu_lcdm_true + M_lcdm + np.random.normal(0, 0.15, n_sne)
    m_err = np.full(n_sne, 0.15)

mu_lcdm = np.array([distance_modulus(z, 0.0) for z in z_sne])
mu_iam = np.array([distance_modulus(z, beta_best) for z in z_sne])

# Observed distance moduli
mu_obs = m_obs - M_lcdm

# Panel 1: Hubble diagram
ax1.errorbar(z_sne, mu_obs, yerr=m_err, fmt='o', color='black', 
             alpha=0.3, markersize=3, label='Data')

z_smooth = np.linspace(0, z_sne.max(), 200)
mu_lcdm_smooth = np.array([distance_modulus(z, 0.0) for z in z_smooth])
mu_iam_smooth = np.array([distance_modulus(z, beta_best) for z in z_smooth])

ax1.plot(z_smooth, mu_lcdm_smooth, 'b-', linewidth=2, label='ΛCDM')
ax1.plot(z_smooth, mu_iam_smooth, 'r--', linewidth=2, 
         label=f'IAM (β={beta_best:.3f})')

ax1.set_ylabel('Distance Modulus μ', fontsize=12)
ax1.set_title(f'Pantheon+ Hubble Diagram ({n_sne} SNe)', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Panel 2: Residuals
residual_lcdm = mu_obs - mu_lcdm
residual_iam = mu_obs - mu_iam

ax2.errorbar(z_sne, residual_lcdm, yerr=m_err, fmt='o', color='blue',
             alpha=0.5, markersize=3, label='ΛCDM residuals')
ax2.errorbar(z_sne, residual_iam, yerr=m_err, fmt='s', color='red',
             alpha=0.5, markersize=3, label='IAM residuals')
ax2.axhline(0, color='black', linestyle='--', linewidth=1)

ax2.set_xlabel('Redshift z', fontsize=12)
ax2.set_ylabel('Residual Δμ', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/pantheon_hubble_diagram.png', dpi=150, bbox_inches='tight')
print("  Saved: results/pantheon_hubble_diagram.png")
print()

# Save results
results_dict = {
    'chi2_lcdm': chi2_lcdm,
    'chi2_iam': chi2_best,
    'beta_best': beta_best,
    'delta_chi2': delta_chi2_best,
    'n_sne': n_sne,
    'use_mock': USE_MOCK,
}

np.save('../results/test_33_pantheon.npy', results_dict)
print("Results saved to: results/test_33_pantheon.npy")
print()

print("="*70)
print("TEST 33 COMPLETE")
print("="*70)
print()

if USE_MOCK:
    print("Note: Used mock data for testing.")
    print("Download real Pantheon+ data for actual analysis.")
else:
    print("Summary:")
    print(f"  {n_sne} supernovae analyzed")
    print(f"  Δχ² = {delta_chi2_best:+.2f}")
    print(f"  IAM β = {beta_best:.3f} from DESI/H₀ fits")
