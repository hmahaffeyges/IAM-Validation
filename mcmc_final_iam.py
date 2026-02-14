"""
MCMC Analysis for IAM Dual-Sector Model (Final Version)
========================================================
No growth tax parameter - just β_m and β_γ

Fits: SDSS/BOSS/eBOSS RSD + H₀ measurements + CMB θ_s
Uses: emcee for full Bayesian posterior
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import emcee
import corner

# ============================================================================
# COSMOLOGICAL PARAMETERS (Fixed)
# ============================================================================
h = 0.674
Om_m = 0.315
Om_Lambda = 1.0 - Om_m
Om_r = 9.4e-5
H0_CMB = 67.4  # Planck CMB value

# ============================================================================
# OBSERVATIONAL DATA
# ============================================================================

# H₀ measurements
H0_obs = np.array([67.4, 73.04, 70.39])  # Planck, SH0ES, JWST
H0_err = np.array([0.5, 1.04, 1.89])

# SDSS/BOSS/eBOSS consensus fσ₈ measurements
# Source: Alam et al. 2021, PRD 103, 083533
# https://www.sdss.org/science/final-bao-and-rsd-measurements/
# Individual references:
#   6dFGS:      Beutler et al. 2012, MNRAS 423, 3430
#   SDSS MGS:   Howlett et al. 2015, MNRAS 449, 848
#   BOSS DR12:  Alam et al. 2017, MNRAS 470, 2617
#   eBOSS LRG:  Bautista et al. 2021, MNRAS 500, 736
#   eBOSS ELG:  Tamone et al. 2020, MNRAS 499, 5527
#   eBOSS QSO:  Hou et al. 2021, MNRAS 500, 1201
z_growth = np.array([0.067, 0.150, 0.380, 0.510, 0.700, 0.850, 1.480])
fsig8_obs = np.array([0.423, 0.530, 0.497, 0.459, 0.473, 0.315, 0.462])
fsig8_err = np.array([0.055, 0.160, 0.045, 0.038, 0.041, 0.095, 0.045])

# CMB acoustic scale
theta_s_obs = 0.0104110
theta_s_err = 0.0000031
r_s = 144.43  # Mpc
z_star = 1090

# ============================================================================
# IAM MODEL FUNCTIONS
# ============================================================================

def activation(a):
    """Activation function E(a) = exp(1 - 1/a)"""
    return np.exp(1.0 - 1.0/a)

def H_IAM(a, beta_m):
    """Matter-sector Hubble parameter"""
    Om_r_over_a4 = Om_r / a**4
    Om_m_over_a3 = Om_m / a**3
    return H0_CMB * np.sqrt(Om_r_over_a4 + Om_m_over_a3 + Om_Lambda + beta_m * activation(a))

def H_photon(a, beta_gamma):
    """Photon-sector Hubble parameter"""
    Om_r_over_a4 = Om_r / a**4
    Om_m_over_a3 = Om_m / a**3
    return H0_CMB * np.sqrt(Om_r_over_a4 + Om_m_over_a3 + Om_Lambda + beta_gamma * activation(a))

def Omega_m_eff(a, beta_m):
    """Effective matter density (diluted by β term)"""
    Om_r_over_a4 = Om_r / a**4
    Om_m_over_a3 = Om_m / a**3
    denom = Om_r_over_a4 + Om_m_over_a3 + Om_Lambda + beta_m * activation(a)
    return Om_m_over_a3 / denom

def growth_ODE(D, lna, beta_m):
    """Growth factor ODE: d²D/dlna² + Q dD/dlna = (3/2)Ω_m D"""
    a = np.exp(lna)
    Om_eff = Omega_m_eff(a, beta_m)
    Q = 2.0 - 1.5 * Om_eff
    
    D_val, dDdlna = D
    d2Ddlna2 = 1.5 * Om_eff * D_val - Q * dDdlna
    return [dDdlna, d2Ddlna2]

def compute_growth(beta_m, z_array):
    """Compute growth factor D(z) for given β_m"""
    a_init = 0.001
    D_init = a_init  # D ∝ a during matter domination
    dDdlna_init = D_init
    
    lna_array = np.log(1.0 / (1.0 + z_array[::-1]))
    lna_init = np.log(a_init)
    lna_full = np.concatenate([[lna_init], lna_array])
    
    sol = odeint(growth_ODE, [D_init, dDdlna_init], lna_full, args=(beta_m,), rtol=1e-8, atol=1e-10)
    D_full = sol[:, 0]
    
    # Normalize to D(z=0) = 1
    D_z0 = D_full[-1]
    D_normalized = D_full[1:] / D_z0
    
    return D_normalized[::-1]

def compute_fsigma8(beta_m, z_array, sigma8_planck=0.811):
    """Compute fσ₈(z) for given β_m"""
    D_z = compute_growth(beta_m, z_array)
    
    # Growth rate f(z) = dlnD/dlna
    a_array = 1.0 / (1.0 + z_array)
    lna = np.log(a_array)
    lnD = np.log(D_z)
    
    # Numerical derivative
    f_z = np.gradient(lnD, lna)
    
    # Effective σ₈
    sigma8_eff = sigma8_planck * D_z[np.argmin(np.abs(z_array))]  # D(z=0)
    
    return f_z * sigma8_eff * D_z

def compute_theta_s(beta_gamma):
    """Compute CMB acoustic scale for given β_γ"""
    # Integrate comoving distance to last scattering
    z_array = np.linspace(0, z_star, 1000)
    a_array = 1.0 / (1.0 + z_array)
    
    integrand = 1.0 / np.array([H_photon(a, beta_gamma) for a in a_array])
    chi_star = np.trapz(integrand[::-1], z_array[::-1]) * 299792.458  # c in km/s
    
    return r_s / chi_star

# ============================================================================
# LOG-LIKELIHOOD
# ============================================================================

def log_likelihood(params):
    """
    Parameters:
    -----------
    params : [beta_m, beta_gamma]
    """
    beta_m, beta_gamma = params
    
    # Priors
    if beta_m < 0 or beta_m > 0.5:
        return -np.inf
    if beta_gamma < 0 or beta_gamma > 0.1:
        return -np.inf
    
    chi2 = 0.0
    
    # 1. H₀ measurements (matter sector)
    H0_matter = H0_CMB * np.sqrt(1.0 + beta_m)
    chi2_H0_planck = ((H0_obs[0] - H0_CMB) / H0_err[0])**2
    chi2_H0_shoes = ((H0_obs[1] - H0_matter) / H0_err[1])**2
    chi2_H0_jwst = ((H0_obs[2] - H0_matter) / H0_err[2])**2
    chi2 += chi2_H0_planck + chi2_H0_shoes + chi2_H0_jwst
    
    # 2. Growth rate fσ₈ (matter sector) - SDSS/BOSS/eBOSS
    fsig8_model = compute_fsigma8(beta_m, z_growth)
    chi2_growth = np.sum(((fsig8_obs - fsig8_model) / fsig8_err)**2)
    chi2 += chi2_growth
    
    # 3. CMB acoustic scale (photon sector)
    theta_s_model = compute_theta_s(beta_gamma)
    chi2_cmb = ((theta_s_obs - theta_s_model) / theta_s_err)**2
    chi2 += chi2_cmb
    
    return -0.5 * chi2

# ============================================================================
# RUN MCMC
# ============================================================================

print("=" * 70)
print("IAM DUAL-SECTOR MCMC ANALYSIS (Corrected Data - SDSS/BOSS/eBOSS)")
print("=" * 70)
print()

# Initial guess
beta_m_init = 0.157
beta_gamma_init = 0.002

# MCMC setup
ndim = 2
nwalkers = 32
nsteps = 5000
nburn = 1000

# Initialize walkers near initial guess
pos = [beta_m_init, beta_gamma_init] + 1e-3 * np.random.randn(nwalkers, ndim)

# Run MCMC
print("Running MCMC...")
print(f"  Parameters: β_m, β_γ")
print(f"  Walkers: {nwalkers}")
print(f"  Steps: {nsteps}")
print(f"  Burn-in: {nburn}")
print()

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
sampler.run_mcmc(pos, nsteps, progress=True)

# Get samples (after burn-in)
samples = sampler.get_chain(discard=nburn, flat=True)

print()
print("MCMC Complete!")
print()

# ============================================================================
# RESULTS
# ============================================================================

# Compute statistics
beta_m_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
beta_gamma_mcmc = np.percentile(samples[:, 1], [16, 50, 84])

print("=" * 70)
print("POSTERIOR RESULTS")
print("=" * 70)
print()
print(f"β_m (matter sector):")
print(f"  Median: {beta_m_mcmc[1]:.4f}")
print(f"  68% CL: {beta_m_mcmc[1]:.4f} +{beta_m_mcmc[2]-beta_m_mcmc[1]:.4f} -{beta_m_mcmc[1]-beta_m_mcmc[0]:.4f}")
print(f"  95% CL: [{np.percentile(samples[:, 0], 2.5):.4f}, {np.percentile(samples[:, 0], 97.5):.4f}]")
print()
print(f"β_γ (photon sector):")
print(f"  Median: {beta_gamma_mcmc[1]:.4f}")
print(f"  68% CL: {beta_gamma_mcmc[1]:.4f} +{beta_gamma_mcmc[2]-beta_gamma_mcmc[1]:.4f} -{beta_gamma_mcmc[1]-beta_gamma_mcmc[0]:.4f}")
print(f"  95% upper limit: {np.percentile(samples[:, 1], 95):.4f}")
print()
print(f"Sector ratio β_γ/β_m:")
ratio_samples = samples[:, 1] / samples[:, 0]
print(f"  Median: {np.median(ratio_samples):.4f}")
print(f"  95% upper limit: {np.percentile(ratio_samples, 95):.4f}")
print()

# Physical predictions
H0_matter_median = H0_CMB * np.sqrt(1.0 + beta_m_mcmc[1])
print("=" * 70)
print("PHYSICAL PREDICTIONS")
print("=" * 70)
print()
print(f"H₀(matter) = {H0_matter_median:.2f} km/s/Mpc")
print(f"H₀(photon) = {H0_CMB:.2f} km/s/Mpc (Planck)")
print()

# ============================================================================
# SAVE RESULTS
# ============================================================================

np.savez('mcmc_results_final.npz',
         samples=samples,
         beta_m=beta_m_mcmc,
         beta_gamma=beta_gamma_mcmc)

print("Results saved to: mcmc_results_final.npz")
print()

# ============================================================================
# CORNER PLOT
# ============================================================================

print("Creating corner plot...")

labels = [r'$\beta_m$', r'$\beta_\gamma$']
fig = corner.corner(samples, labels=labels,
                   quantiles=[0.16, 0.5, 0.84],
                   show_titles=True,
                   title_fmt='.4f',
                   title_kwargs={"fontsize": 12})

fig.suptitle('IAM Dual-Sector Parameter Constraints (SDSS/BOSS/eBOSS + H₀ + CMB)', fontsize=14, y=1.02)
plt.savefig('mcmc_iam_corner_final.pdf', bbox_inches='tight', dpi=300)
plt.savefig('mcmc_iam_corner_final.png', bbox_inches='tight', dpi=150)
print("Corner plot saved!")
print()

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
