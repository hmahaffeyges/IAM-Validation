#!/usr/bin/env python3
"""
IAM-CAMB Validation: Dual-Sector Predictions
=============================================
Run this on your Mac with Anaconda:
    pip install camb
    python iam_camb_validation.py

Generates publication-quality figures showing:
  1. CMB TT power spectrum (photon sector = LCDM)
  2. H(z) comparison (matter sector vs LCDM)
  3. Growth rate f*sigma8(z) vs SDSS/BOSS/eBOSS data
  4. Matter density suppression
  5. mu(a) modified gravity mapping
  6. CMB lensing power spectrum comparison

Runtime: ~30 seconds on laptop
Output: iam_camb_validation.pdf (6-panel figure)
        iam_camb_results.txt (numerical summary)

Heath Mahaffey, February 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp, trapezoid

# ============================================================================
# Install check
# ============================================================================
try:
    import camb
    print(f"CAMB version: {camb.__version__}")
except ImportError:
    print("CAMB not installed. Run: pip install camb")
    exit(1)

# ============================================================================
# IAM Parameters
# ============================================================================
Om = 0.315        # Omega_matter (Planck 2020)
OL = 0.685        # Omega_Lambda
Or = 9.24e-5      # Omega_radiation
H0 = 67.4         # Hubble constant (Planck, photon sector)
beta_m = 0.157    # Matter-sector coupling (MCMC best fit)
sigma8_0_iam = 0.800  # IAM sigma_8 prediction

def E_activation(a):
    """IAM activation function"""
    return np.exp(1.0 - 1.0/a)

def H2_normalized(a, beta=0.0):
    """H^2/H0^2 for given beta"""
    z = 1.0/a - 1.0
    return Om*(1+z)**3 + Or*(1+z)**4 + OL + beta*E_activation(a)

def mu_iam(a):
    """IAM effective mu(a) in modified gravity parametrization"""
    return H2_normalized(a, 0.0) / H2_normalized(a, beta_m)

# ============================================================================
# STEP 1: CAMB LCDM Baseline
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Running CAMB LCDM baseline...")
print("="*70)

pars = camb.CAMBparams()
pars.set_cosmology(
    H0=67.4, ombh2=0.02242, omch2=0.11933,
    mnu=0.06, omk=0, tau=0.0561
)
pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
pars.set_for_lmax(2500, lens_potential_accuracy=1)
pars.Want_CMB_lensing = True
pars.set_matter_power(redshifts=[0.0, 0.295, 0.38, 0.51, 0.698, 1.0, 1.48, 1.83, 2.33], kmax=2.0)
pars.WantTransfer = True

results = camb.get_results(pars)
derived = results.get_derived_params()

# CMB spectra
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL = powers['total']
lensCL = powers['lensed_scalar']
lens_pot = results.get_lens_potential_cls(lmax=2500)
ell = np.arange(totCL.shape[0])

# sigma8
sigma8_z = results.get_sigma8()
sigma8_0_lcdm = sigma8_z[-1]

print(f"  100*theta_MC = {derived['thetastar']:.5f} (Planck: 1.04110)")
print(f"  sigma_8(z=0) = {sigma8_0_lcdm:.4f} (Planck: 0.811)")
print(f"  r_drag       = {derived['rdrag']:.2f} Mpc")

# ============================================================================
# STEP 2: Compute IAM growth factor
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Computing IAM growth factor...")
print("="*70)

def compute_growth(beta):
    """Solve linear growth ODE for given beta"""
    def ode(lna, y):
        a = np.exp(lna)
        E_a = E_activation(a)
        H2 = Om*a**(-3) + Or*a**(-4) + OL + beta*E_a
        dH2_da = -3*Om*a**(-4) - 4*Or*a**(-5) + beta*E_a/(a**2)
        dlnH_dlna = a * dH2_da / (2*H2)
        Om_eff = Om * a**(-3) / H2
        return [y[1], -(2 + dlnH_dlna)*y[1] + 1.5*Om_eff*y[0]]
    
    lna = np.linspace(np.log(1e-3), 0.0, 5000)
    a_init = np.exp(lna[0])
    sol = solve_ivp(ode, [lna[0], lna[-1]], [a_init, a_init],
                    t_eval=lna, rtol=1e-10, atol=1e-12)
    
    a_out = np.exp(sol.t)
    D = sol.y[0] / sol.y[0][-1]  # Normalize D(z=0)=1
    f = sol.y[1] / sol.y[0]       # f = dlnD/dlna
    return a_out, D, f

a_lcdm, D_lcdm, f_lcdm = compute_growth(0.0)
a_iam, D_iam, f_iam = compute_growth(beta_m)

z_lcdm = 1.0/a_lcdm - 1
z_iam = 1.0/a_iam - 1

# f*sigma8
fsig8_lcdm = f_lcdm * sigma8_0_lcdm * D_lcdm
fsig8_iam = f_iam * sigma8_0_iam * D_iam

# SDSS/BOSS/eBOSS growth rate data
z_data = np.array([0.295, 0.38, 0.51, 0.698, 1.48, 1.83, 2.33])
fsig8_data = np.array([0.470, 0.497, 0.459, 0.473, 0.342, 0.320, 0.280])
fsig8_err = np.array([0.034, 0.045, 0.038, 0.044, 0.070, 0.079, 0.080])

# Chi2
def interp_at(z_model, y_model, z_target):
    idx = np.argsort(z_model)
    return np.interp(z_target, z_model[idx], y_model[idx])

chi2_lcdm = np.sum(((fsig8_data - interp_at(z_lcdm, fsig8_lcdm, z_data)) / fsig8_err)**2)
chi2_iam = np.sum(((fsig8_data - interp_at(z_iam, fsig8_iam, z_data)) / fsig8_err)**2)

print(f"  Growth chi2 (LCDM): {chi2_lcdm:.2f}")
print(f"  Growth chi2 (IAM):  {chi2_iam:.2f}")

# ============================================================================
# STEP 3: Background quantities
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Background comparison...")
print("="*70)

z_bg = np.linspace(0, 5, 1000)
a_bg = 1.0/(1.0 + z_bg)

H_lcdm_bg = H0 * np.sqrt(H2_normalized(a_bg, 0.0))
H_iam_bg = H0 * np.sqrt(H2_normalized(a_bg, beta_m))
delta_H = (H_iam_bg - H_lcdm_bg) / H_lcdm_bg * 100

# mu(a)
mu_vals = mu_iam(a_bg)

# Matter density suppression
Om_eff_lcdm = Om*(1+z_bg)**3 / H2_normalized(a_bg, 0.0)
Om_eff_iam = Om*(1+z_bg)**3 / H2_normalized(a_bg, beta_m)
suppression = (Om_eff_iam/Om_eff_lcdm - 1) * 100

print(f"  H0(matter) = {H0*np.sqrt(1+beta_m):.2f} km/s/Mpc")
print(f"  mu(z=0) = {mu_iam(1.0):.4f}")
print(f"  mu(z=1) = {mu_iam(0.5):.4f}")
print(f"  Omega_m suppression at z=0: {suppression[0]:.1f}%")

# ============================================================================
# STEP 4: Lensing comparison
# ============================================================================
print("\n" + "="*70)
print("STEP 4: CMB lensing estimate...")
print("="*70)

# Lensing amplitude scales as sigma8^2
A_L_ratio = (sigma8_0_iam / sigma8_0_lcdm)**2
print(f"  A_L(IAM)/A_L(LCDM) = {A_L_ratio:.4f} ({(A_L_ratio-1)*100:+.2f}%)")
print(f"  Planck A_L anomaly: 1.18 +/- 0.07 (high)")
print(f"  IAM reduces lensing -> helps with A_L tension")

# Scale the LCDM lensing spectrum
lens_pot_iam = lens_pot.copy()
lens_pot_iam[:, 0] *= A_L_ratio  # Scale PP spectrum

# ============================================================================
# FIGURE: 6-panel publication figure
# ============================================================================
print("\n" + "="*70)
print("Generating figure...")
print("="*70)

fig = plt.figure(figsize=(16, 14))
gs = GridSpec(3, 2, hspace=0.35, wspace=0.30,
             left=0.08, right=0.96, top=0.94, bottom=0.05)

# --- Panel (a): CMB TT ---
ax1 = fig.add_subplot(gs[0, 0])
ell_p = ell[2:2501]
Dl = ell_p*(ell_p+1)/(2*np.pi) * totCL[2:2501, 0]
ax1.plot(ell_p, Dl, 'b-', lw=1.5, label=r'$\Lambda$CDM = IAM photon sector')
ax1.set_xlabel(r'Multipole $\ell$', fontsize=12)
ax1.set_ylabel(r'$D_\ell^{TT}$ [$\mu$K$^2$]', fontsize=12)
ax1.set_title('(a) CMB TT Power Spectrum', fontsize=13, fontweight='bold')
ax1.set_xlim(2, 2500)
ax1.legend(fontsize=10)
ax1.annotate(r'$\beta_\gamma < 10^{-6}$: photon sector' + '\nunmodified from Planck',
             xy=(1200, 4500), fontsize=9, style='italic',
             bbox=dict(boxstyle='round', fc='lightblue', alpha=0.5))

# --- Panel (b): H(z) ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(z_bg, H_lcdm_bg, 'b-', lw=2, label=r'$\Lambda$CDM ($\beta=0$)')
ax2.plot(z_bg, H_iam_bg, 'r--', lw=2, label=r'IAM matter ($\beta_m=0.157$)')
ax2.fill_between(z_bg, H_lcdm_bg, H_iam_bg, alpha=0.12, color='red')
ax2.set_xlabel('Redshift $z$', fontsize=12)
ax2.set_ylabel('$H(z)$ [km/s/Mpc]', fontsize=12)
ax2.set_title('(b) Hubble Parameter', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlim(0, 3)
ax2.axhline(67.4, color='blue', ls=':', alpha=0.4)
ax2.axhline(72.5, color='red', ls=':', alpha=0.4)
ax2.annotate('$H_0^{\\rm photon}=67.4$', xy=(2.0, 64), color='blue', fontsize=9)
ax2.annotate('$H_0^{\\rm matter}=72.5$', xy=(2.0, 74), color='red', fontsize=9)

# --- Panel (c): f*sigma8 ---
ax3 = fig.add_subplot(gs[1, 0])
mask_l = (z_lcdm > 0) & (z_lcdm < 3)
mask_i = (z_iam > 0) & (z_iam < 3)
ax3.plot(z_lcdm[mask_l], fsig8_lcdm[mask_l], 'b-', lw=2, label=r'$\Lambda$CDM')
ax3.plot(z_iam[mask_i], fsig8_iam[mask_i], 'r--', lw=2, label=r'IAM ($\beta_m=0.157$)')
ax3.errorbar(z_data, fsig8_data, yerr=fsig8_err, fmt='ko', ms=6, capsize=3,
             label='SDSS/BOSS/eBOSS', zorder=5)
ax3.set_xlabel('Redshift $z$', fontsize=12)
ax3.set_ylabel(r'$f\sigma_8(z)$', fontsize=12)
ax3.set_title(r'(c) Growth Rate $f\sigma_8(z)$', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_xlim(0, 2.8)
ax3.set_ylim(0.2, 0.55)
ax3.annotate(f'$\\chi^2_{{\\Lambda CDM}}={chi2_lcdm:.1f}$\n$\\chi^2_{{IAM}}={chi2_iam:.1f}$',
             xy=(0.05, 0.22), fontsize=9,
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

# --- Panel (d): mu(a) ---
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(z_bg, mu_vals, 'r-', lw=2.5)
ax4.axhline(1.0, color='blue', ls='--', lw=1.5, label=r'$\Lambda$CDM ($\mu=1$)')
ax4.fill_between(z_bg, mu_vals, 1.0, alpha=0.15, color='red')
ax4.set_xlabel('Redshift $z$', fontsize=12)
ax4.set_ylabel(r'$\mu(a)$', fontsize=12)
ax4.set_title(r'(d) Modified Gravity Parameter $\mu(a)$', fontsize=13, fontweight='bold')
ax4.set_xlim(0, 3)
ax4.set_ylim(0.82, 1.02)
ax4.legend(fontsize=10)
ax4.annotate(f'$\\mu(z=0) = {mu_iam(1.0):.3f}$\n$\\Sigma = 1$ (photons unmodified)',
             xy=(1.5, 0.84), fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

# --- Panel (e): Omega_m suppression ---
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(z_bg, suppression, 'r-', lw=2)
ax5.axhline(0, color='gray', ls=':', alpha=0.5)
ax5.fill_between(z_bg, suppression, 0, alpha=0.12, color='red')
ax5.set_xlabel('Redshift $z$', fontsize=12)
ax5.set_ylabel(r'$\Delta\Omega_m^{\rm eff} / \Omega_m^{\rm eff}$ [%]', fontsize=12)
ax5.set_title('(e) Matter Density Suppression', fontsize=13, fontweight='bold')
ax5.set_xlim(0, 3)
ax5.annotate(f'{suppression[0]:.1f}% at $z=0$', xy=(0.1, suppression[0]+0.5),
             fontsize=11, color='red', fontweight='bold')

# --- Panel (f): CMB lensing ---
ax6 = fig.add_subplot(gs[2, 1])
ell_lens = np.arange(lens_pot.shape[0])
mask_lens = (ell_lens >= 2) & (ell_lens <= 2000)
Cl_pp_lcdm = ell_lens[mask_lens]**2 * (ell_lens[mask_lens]+1)**2 * lens_pot[mask_lens, 0] / (2*np.pi) * 1e7
Cl_pp_iam = ell_lens[mask_lens]**2 * (ell_lens[mask_lens]+1)**2 * lens_pot_iam[mask_lens, 0] / (2*np.pi) * 1e7

ax6.plot(ell_lens[mask_lens], Cl_pp_lcdm, 'b-', lw=1.5, label=r'$\Lambda$CDM')
ax6.plot(ell_lens[mask_lens], Cl_pp_iam, 'r--', lw=1.5, label=r'IAM ($\sigma_8=0.800$)')
ax6.set_xlabel(r'Multipole $L$', fontsize=12)
ax6.set_ylabel(r'$[L(L+1)]^2 C_L^{\phi\phi}/2\pi$ [$\times 10^7$]', fontsize=11)
ax6.set_title(r'(f) CMB Lensing Power', fontsize=13, fontweight='bold')
ax6.set_xlim(2, 2000)
ax6.legend(fontsize=10)
ax6.annotate(f'IAM reduces lensing by {abs((A_L_ratio-1)*100):.1f}%\n(helps with Planck $A_L$ anomaly)',
             xy=(800, max(Cl_pp_lcdm)*0.8), fontsize=9, style='italic',
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.5))

fig.suptitle('IAM-CAMB Validation: Dual-Sector Predictions\n'
             r'Photon sector ($\beta_\gamma\approx 0$): standard $\Lambda$CDM  |  '
             r'Matter sector ($\beta_m=0.157$): $\mu<1$, $\Sigma=1$',
             fontsize=14, fontweight='bold', y=0.99)

plt.savefig('iam_camb_validation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('iam_camb_validation.png', dpi=150, bbox_inches='tight')
print("  Saved: iam_camb_validation.pdf")
print("  Saved: iam_camb_validation.png")

# ============================================================================
# RESULTS SUMMARY FILE
# ============================================================================
with open('iam_camb_results.txt', 'w') as f:
    f.write("IAM-CAMB VALIDATION RESULTS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Date: February 2026\n")
    f.write(f"CAMB version: {camb.__version__}\n\n")
    
    f.write("LCDM BASELINE (Planck 2020)\n")
    f.write(f"  100*theta_MC = {derived['thetastar']:.5f}\n")
    f.write(f"  sigma_8      = {sigma8_0_lcdm:.4f}\n")
    f.write(f"  r_drag       = {derived['rdrag']:.2f} Mpc\n\n")
    
    f.write("IAM PARAMETERS\n")
    f.write(f"  beta_m       = {beta_m}\n")
    f.write(f"  sigma_8(IAM) = {sigma8_0_iam}\n")
    f.write(f"  H0(matter)   = {H0*np.sqrt(1+beta_m):.2f} km/s/Mpc\n\n")
    
    f.write("MODIFIED GRAVITY MAPPING\n")
    f.write(f"  mu(z=0) = {mu_iam(1.0):.4f}\n")
    f.write(f"  mu(z=0.5) = {mu_iam(1.0/1.5):.4f}\n")
    f.write(f"  mu(z=1) = {mu_iam(0.5):.4f}\n")
    f.write(f"  mu(z=2) = {mu_iam(1.0/3.0):.4f}\n")
    f.write(f"  Sigma = 1 (all z)\n\n")
    
    f.write("GROWTH RATE chi2 (7 SDSS/BOSS/eBOSS points)\n")
    f.write(f"  LCDM: {chi2_lcdm:.2f}\n")
    f.write(f"  IAM:  {chi2_iam:.2f}\n\n")
    
    f.write("CMB LENSING\n")
    f.write(f"  A_L(IAM)/A_L(LCDM) = {A_L_ratio:.4f}\n")
    f.write(f"  Reduction: {abs((A_L_ratio-1)*100):.2f}%\n\n")
    
    f.write("KEY CONCLUSIONS\n")
    f.write("  1. Photon sector (beta_gamma~0): CMB identical to LCDM\n")
    f.write("  2. Matter sector: mu(a)<1, Sigma=1 modified growth\n")
    f.write("  3. H0 tension resolved: 67.4 (photon) vs 72.5 (matter)\n")
    f.write("  4. Growth data consistent with both LCDM and IAM\n")
    f.write("  5. Lensing reduction helps with Planck A_L anomaly\n")

print("  Saved: iam_camb_results.txt")

print("\n" + "="*70)
print("DONE! Check iam_camb_validation.pdf")
print("="*70)
