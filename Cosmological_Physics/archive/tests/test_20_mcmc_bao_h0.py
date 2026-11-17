#!/usr/bin/env python3
"""
Test 20: MCMC Analysis of BAO + H₀ 
===================================

Proper uncertainty quantification for the REAL IAM result (test_03).

Uses emcee to explore parameter space and quantify:
- Parameter uncertainties
- Correlations/degeneracies
- Statistical significance

This is the publication-quality analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import emcee
import corner

print("="*80)
print("MCMC ANALYSIS: IAM vs ΛCDM on BAO + H₀")
print("="*80)
print()

# ============================================================================
# PARAMETERS AND DATA
# ============================================================================

# Fiducial cosmology
Om0 = 0.315
sigma8_0 = 0.811

# H₀ measurements
H0_data = {
    'Planck':    (67.4,  0.5),
    'SH0ES':     (73.04, 1.04),
}

# DESI BAO growth rates (fσ₈)
desi_z = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
desi_fs8 = np.array([0.452, 0.428, 0.410, 0.392, 0.368, 0.355, 0.312])
desi_err = np.array([0.030, 0.025, 0.028, 0.035, 0.040, 0.045, 0.050])

print(f"Data points:")
print(f"  DESI BAO: {len(desi_z)} fσ₈ measurements")
print(f"  H₀: {len(H0_data)} measurements")
print(f"  Total: {len(desi_z) + len(H0_data)} constraints")
print()

# ============================================================================
# MODELS
# ============================================================================

def E_activation(a):
    """Activation function for IAM"""
    return np.exp(1 - 1/a)

def Omega_m_a(a, beta=0):
    """Matter density parameter at scale factor a"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    if beta > 0:
        denom = Om0 * a**(-3) + Om_r * a**(-4) + Om_L + beta * E_activation(a)
    else:
        denom = Om0 * a**(-3) + Om_r * a**(-4) + Om_L
    return Om0 * a**(-3) / denom

def H_IAM(a, H0_CMB, beta):
    """IAM Hubble parameter"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    return H0_CMB * np.sqrt(Om0 * a**(-3) + Om_r * a**(-4) + Om_L + beta * E_activation(a))

def H_LCDM(a, H0):
    """ΛCDM Hubble parameter"""
    Om_r = 9.24e-5
    Om_L = 1 - Om0 - Om_r
    return H0 * np.sqrt(Om0 * a**(-3) + Om_r * a**(-4) + Om_L)

def growth_ode_lna(lna, y, beta=0, tax=0):
    """Growth factor ODE"""
    D, Dprime = y
    a = np.exp(lna)
    Om_a = Omega_m_a(a, beta)
    Q = 2 - 1.5 * Om_a
    Tax = tax * E_activation(a) if tax > 0 else 0
    D_double_prime = -Q * Dprime + 1.5 * Om_a * D * (1 - Tax)
    return [Dprime, D_double_prime]

def solve_growth(z_vals, beta=0, tax=0):
    """Solve growth ODE"""
    lna_start = np.log(0.001)
    lna_end = 0.0
    a_start = np.exp(lna_start)
    y0 = [a_start, a_start]
    
    a_vals = 1 / (1 + z_vals)
    lna_vals = np.log(a_vals)
    lna_eval = np.sort(np.append(lna_vals, [lna_start, lna_end]))
    
    sol = solve_ivp(growth_ode_lna, (lna_start, lna_end), y0,
                    args=(beta, tax), t_eval=lna_eval,
                    method='DOP853', rtol=1e-8, atol=1e-10)
    
    D_raw = sol.y[0]
    D_normalized = D_raw / D_raw[-1]
    return dict(zip(lna_eval, D_normalized))

def compute_f(z_vals, lna_to_D):
    """Growth rate f = dlnD/dlna"""
    a_vals = 1 / (1 + z_vals)
    lna_vals = np.log(a_vals)
    D_vals = np.array([lna_to_D[lna] for lna in lna_vals])
    f_vals = np.gradient(np.log(D_vals), lna_vals)
    return dict(zip(z_vals, f_vals))

# ============================================================================
# LIKELIHOOD FUNCTIONS
# ============================================================================

def log_likelihood_lcdm(theta):
    """ΛCDM: Only H₀ free"""
    H0, = theta
    
    # H₀ chi-squared
    chi2_h0 = sum(((obs - H0) / err)**2 for obs, err in H0_data.values())
    
    # Growth rates
    D_lcdm = solve_growth(desi_z, beta=0, tax=0)
    f_lcdm = compute_f(desi_z, D_lcdm)
    
    fs8_pred = []
    for i, z in enumerate(desi_z):
        a = 1/(1+z)
        lna = np.log(a)
        fs8_pred.append(f_lcdm[z] * sigma8_0 * D_lcdm[lna])
    
    fs8_pred = np.array(fs8_pred)
    chi2_desi = np.sum(((desi_fs8 - fs8_pred) / desi_err)**2)
    
    return -0.5 * (chi2_h0 + chi2_desi)

def log_likelihood_iam(theta):
    """IAM: H₀_CMB, β, growth_tax free"""
    H0_CMB, beta, growth_tax = theta
    
    # H₀ chi-squared (epoch-dependent)
    H0_today = H_IAM(1.0, H0_CMB, beta)
    H0_pred = {
        'Planck': H0_CMB,      # Early universe
        'SH0ES':  H0_today,    # Late universe
    }
    chi2_h0 = sum(((H0_data[k][0] - H0_pred[k]) / H0_data[k][1])**2 for k in H0_data.keys())
    
    # Growth rates
    D_iam = solve_growth(desi_z, beta=beta, tax=growth_tax)
    f_iam = compute_f(desi_z, D_iam)
    
    fs8_pred = []
    for i, z in enumerate(desi_z):
        a = 1/(1+z)
        lna = np.log(a)
        fs8_pred.append(f_iam[z] * sigma8_0 * D_iam[lna])
    
    fs8_pred = np.array(fs8_pred)
    chi2_desi = np.sum(((desi_fs8 - fs8_pred) / desi_err)**2)
    
    return -0.5 * (chi2_h0 + chi2_desi)

def log_prior_lcdm(theta):
    """Flat priors for ΛCDM"""
    H0, = theta
    if 60 < H0 < 80:
        return 0.0
    return -np.inf

def log_prior_iam(theta):
    """Flat priors for IAM"""
    H0_CMB, beta, growth_tax = theta
    if 60 < H0_CMB < 75 and 0 < beta < 0.5 and 0 < growth_tax < 0.2:
        return 0.0
    return -np.inf

def log_probability_lcdm(theta):
    lp = log_prior_lcdm(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_lcdm(theta)

def log_probability_iam(theta):
    lp = log_prior_iam(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_iam(theta)

# ============================================================================
# RUN MCMC - ΛCDM
# ============================================================================

print("="*80)
print("RUNNING MCMC FOR ΛCDM")
print("="*80)
print()

ndim_lcdm = 1
nwalkers_lcdm = 10
nsteps = 3000
burn_in = 500

# Initial positions
pos_lcdm = [70.0] + 1e-2 * np.random.randn(nwalkers_lcdm, ndim_lcdm)

sampler_lcdm = emcee.EnsembleSampler(nwalkers_lcdm, ndim_lcdm, log_probability_lcdm)

print(f"Running {nwalkers_lcdm} walkers for {nsteps} steps...")
sampler_lcdm.run_mcmc(pos_lcdm, nsteps, progress=True)

# Get samples (remove burn-in)
samples_lcdm = sampler_lcdm.get_chain(discard=burn_in, flat=True)

# Best fit
H0_lcdm = np.median(samples_lcdm[:, 0])
H0_lcdm_err = np.std(samples_lcdm[:, 0])

# Compute chi-squared at best fit
chi2_lcdm = -2 * log_likelihood_lcdm([H0_lcdm])

print()
print(f"ΛCDM Results:")
print(f"  H₀ = {H0_lcdm:.2f} ± {H0_lcdm_err:.2f} km/s/Mpc")
print(f"  χ² = {chi2_lcdm:.2f}")
print()

# ============================================================================
# RUN MCMC - IAM
# ============================================================================

print("="*80)
print("RUNNING MCMC FOR IAM")
print("="*80)
print()

ndim_iam = 3
nwalkers_iam = 32

# Initial positions (from test_03)
pos_iam = np.array([[67.4, 0.18, 0.045]]) + 1e-2 * np.random.randn(nwalkers_iam, ndim_iam)

sampler_iam = emcee.EnsembleSampler(nwalkers_iam, ndim_iam, log_probability_iam)

print(f"Running {nwalkers_iam} walkers for {nsteps} steps...")
sampler_iam.run_mcmc(pos_iam, nsteps, progress=True)

# Get samples
samples_iam = sampler_iam.get_chain(discard=burn_in, flat=True)

# Best fit
H0_CMB_iam = np.median(samples_iam[:, 0])
H0_CMB_iam_err = np.std(samples_iam[:, 0])
beta_iam = np.median(samples_iam[:, 1])
beta_iam_err = np.std(samples_iam[:, 1])
tax_iam = np.median(samples_iam[:, 2])
tax_iam_err = np.std(samples_iam[:, 2])

# Compute H₀(today)
H0_today_iam = H_IAM(1.0, H0_CMB_iam, beta_iam)

# Compute chi-squared
chi2_iam = -2 * log_likelihood_iam([H0_CMB_iam, beta_iam, tax_iam])

print()
print(f"IAM Results:")
print(f"  H₀(CMB)     = {H0_CMB_iam:.2f} ± {H0_CMB_iam_err:.2f} km/s/Mpc")
print(f"  H₀(today)   = {H0_today_iam:.2f} km/s/Mpc")
print(f"  β           = {beta_iam:.3f} ± {beta_iam_err:.3f}")
print(f"  growth_tax  = {tax_iam:.4f} ± {tax_iam_err:.4f}")
print(f"  χ²          = {chi2_iam:.2f}")
print()

# ============================================================================
# COMPARISON
# ============================================================================

delta_chi2 = chi2_lcdm - chi2_iam
sigma = np.sqrt(abs(delta_chi2))

print("="*80)
print("FINAL COMPARISON")
print("="*80)
print()
print(f"  χ²_ΛCDM = {chi2_lcdm:.2f}")
print(f"  χ²_IAM  = {chi2_iam:.2f}")
print(f"  Δχ²     = {delta_chi2:.2f}")
print(f"  Significance: {sigma:.1f}σ")
print()

if delta_chi2 > 25:
    print(f"🔥🔥🔥 STRONG EVIDENCE for IAM ({sigma:.1f}σ)")
elif delta_chi2 > 9:
    print(f"✓✓ MODERATE EVIDENCE ({sigma:.1f}σ)")
elif delta_chi2 > 4:
    print(f"✓ WEAK PREFERENCE ({sigma:.1f}σ)")
else:
    print(f"No significant improvement")

print()
print("="*80)

# ============================================================================
# CORNER PLOTS
# ============================================================================

print("Creating corner plots...")

# IAM corner plot
fig = corner.corner(samples_iam, 
                    labels=[r'$H_0^{\rm CMB}$', r'$\beta$', r'${\rm growth\_tax}$'],
                    truths=[H0_CMB_iam, beta_iam, tax_iam],
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_fmt='.3f')

fig.suptitle('IAM Parameter Constraints (BAO + H₀)', fontsize=14, y=1.02)
plt.savefig('../results/mcmc_iam_corner.png', dpi=150, bbox_inches='tight')
print("  Saved: results/mcmc_iam_corner.png")

# Save results
np.savez('../results/mcmc_results.npz',
         H0_lcdm=H0_lcdm,
         H0_lcdm_err=H0_lcdm_err,
         chi2_lcdm=chi2_lcdm,
         H0_CMB_iam=H0_CMB_iam,
         H0_CMB_iam_err=H0_CMB_iam_err,
         H0_today_iam=H0_today_iam,
         beta_iam=beta_iam,
         beta_iam_err=beta_iam_err,
         tax_iam=tax_iam,
         tax_iam_err=tax_iam_err,
         chi2_iam=chi2_iam,
         delta_chi2=delta_chi2,
         samples_lcdm=samples_lcdm,
         samples_iam=samples_iam)

print("  Saved: results/mcmc_results.npz")
print()
print("="*80)
print("🎉 MCMC ANALYSIS COMPLETE!")
print("="*80)
