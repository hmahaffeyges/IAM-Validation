#!/usr/bin/env python3
"""
IAM MGCAMB 6-Panel Validation Figure
Run from the directory containing iam_mgcamb_pureMG.npz
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import camb

# ============================================================
# Recompute everything fresh (avoids needing .npz file)
# ============================================================
Om, OL, beta = 0.315, 0.685, 0.1575
Or = 9.24e-5

def mu_iam(a):
    H2L = Om * a**(-3) + Or * a**(-4) + OL
    return H2L / (H2L + beta * np.exp(1 - 1.0/a))

def E_act(a):
    return np.exp(1 - 1.0/a)

def OmDE(a):
    H2 = Om*a**(-3) + Or*a**(-4) + OL
    return OL / H2

def make_pars(mu0=None):
    p = camb.CAMBparams()
    p.set_cosmology(H0=67.4, ombh2=0.02242, omch2=0.11933, omk=0, tau=0.0544)
    p.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)
    p.set_for_lmax(2500, lens_potential_accuracy=1)
    p.set_matter_power(redshifts=[0.0, 0.15, 0.38, 0.51, 0.61, 0.85, 1.0, 1.48, 2.0], kmax=2.0)
    p.NonLinear = camb.model.NonLinear_none
    if mu0 is not None:
        p.set_mgparams(MG_flag=1, pure_MG_flag=2, musigma_par=1,
            GRtrans=0.001, mu0=mu0, sigma0=0.0)
    return p

print("Computing LCDM baseline...")
r0 = camb.get_results(make_pars())
print("Computing IAM (mu0=-0.136)...")
r1 = camb.get_results(make_pars(mu0=-0.136))

cls0 = r0.get_cmb_power_spectra(raw_cl=False)
cls1 = r1.get_cmb_power_spectra(raw_cl=False)

tt0 = cls0['total'][:,0]
tt1 = cls1['total'][:,0]
ee0 = cls0['total'][:,1]
ee1 = cls1['total'][:,1]
lens0 = cls0['lens_potential'][:,0]
lens1 = cls1['lens_potential'][:,0]

s8_0 = r0.get_sigma8_0()
s8_1 = r1.get_sigma8_0()
fs8_0 = r0.get_fsigma8()
fs8_1 = r1.get_fsigma8()

ell = np.arange(len(tt0))

# redshifts sorted by CAMB (earliest=highest z first)
zz_camb = [2.0, 1.48, 1.0, 0.85, 0.61, 0.51, 0.38, 0.15, 0.0]

# Matter power spectrum at z=0
kh0, z0, pk0 = r0.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints=200)
kh1, z1, pk1 = r1.get_matter_power_spectrum(minkh=1e-4, maxkh=2, npoints=200)

print(f"sigma8 LCDM = {s8_0:.4f}")
print(f"sigma8 IAM  = {s8_1:.4f}")

# --- BOSS/eBOSS data ---
boss_z   = np.array([0.15, 0.38, 0.51, 0.61, 0.85, 1.48])
boss_fs8 = np.array([0.490, 0.497, 0.459, 0.436, 0.315, 0.282])
boss_err = np.array([0.145, 0.045, 0.038, 0.034, 0.095, 0.075])

# ============================================================
# 6-Panel Figure
# ============================================================
fig = plt.figure(figsize=(16, 10.5))
gs = GridSpec(2, 3, hspace=0.34, wspace=0.30,
              left=0.07, right=0.97, top=0.93, bottom=0.07)

c_lcdm = '#2166ac'
c_iam  = '#d6604d'
c_data = '#2ca02c'

# ------ (a) CMB TT ------
ax1 = fig.add_subplot(gs[0, 0])
m = (ell >= 2) & (ell <= 2500)
ax1.plot(ell[m], tt0[m], color=c_lcdm, lw=1.2, label=r'$\Lambda$CDM')
ax1.plot(ell[m], tt1[m], color=c_iam, lw=1.2, ls='--', label='IAM')
ax1.set_xscale('log')
ax1.set_xlabel(r'$\ell$', fontsize=11)
ax1.set_ylabel(r'$D_\ell^{TT}\ [\mu\mathrm{K}^2]$', fontsize=11)
ax1.set_title('(a) CMB Temperature Power Spectrum', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.set_xlim(2, 2500)

# ------ (b) TT Residual ------
ax2 = fig.add_subplot(gs[0, 1])
m2 = (ell >= 2) & (ell <= 2500) & (tt0 > 0)
residual = (tt1[m2] - tt0[m2]) / tt0[m2] * 100
ax2.plot(ell[m2], residual, color=c_iam, lw=0.8)
ax2.axhline(0, color='gray', lw=0.5, ls=':')
ax2.axhspan(-1, 1, color='#d4edda', alpha=0.5, label=r'$\pm 1\%$ (sub-Planck)')
ax2.axvspan(2, 30, color='#fff3cd', alpha=0.5, label=r'ISW region ($\ell<30$)')
ax2.set_xscale('log')
ax2.set_xlabel(r'$\ell$', fontsize=11)
ax2.set_ylabel(r'$\Delta C_\ell^{TT}/C_\ell^{TT}$ [%]', fontsize=11)
ax2.set_title('(b) TT Residual (IAM $-$ $\\Lambda$CDM)', fontsize=11, fontweight='bold')
ax2.set_xlim(2, 2500)
ax2.set_ylim(-5, 5)
ax2.legend(fontsize=8, loc='lower right')
# Annotate
ax2.annotate(r'$\ell=2$: 3.6%' + '\n(cosmic var ~63%)',
             xy=(2, 3.6), xytext=(15, 4.2),
             fontsize=8, color=c_iam,
             arrowprops=dict(arrowstyle='->', color=c_iam, lw=0.8))
ax2.annotate(r'$\ell>30$: $<0.17\%$',
             xy=(100, 0.1), xytext=(200, 2.5),
             fontsize=8, color=c_iam,
             arrowprops=dict(arrowstyle='->', color=c_iam, lw=0.8))

# ------ (c) Lensing ------
ax3 = fig.add_subplot(gs[0, 2])
ml = (ell >= 2) & (ell <= 2000) & (lens0 > 0)
ax3.plot(ell[ml], lens0[ml] * 1e7, color=c_lcdm, lw=1.2, label=r'$\Lambda$CDM')
ax3.plot(ell[ml], lens1[ml] * 1e7, color=c_iam, lw=1.2, ls='--', label=r'IAM ($\Sigma=1$)')
ax3.set_xscale('log')
ax3.set_xlabel(r'$\ell$', fontsize=11)
ax3.set_ylabel(r'$C_\ell^{\phi\phi}\ [\times 10^7]$', fontsize=11)
ax3.set_title(r'(c) CMB Lensing ($\Sigma=1$ exact)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.set_xlim(2, 2000)

# ------ (d) f*sigma8(z) ------
ax4 = fig.add_subplot(gs[1, 0])
# Build plotting arrays from CAMB output
z_theory = np.array(zz_camb)
fs8_0_arr = np.array(fs8_0)
fs8_1_arr = np.array(fs8_1)
# sort by increasing z for nice line plot
isort = np.argsort(z_theory)
z_sorted = z_theory[isort]
fs8_0_sorted = fs8_0_arr[isort]
fs8_1_sorted = fs8_1_arr[isort]

ax4.plot(z_sorted, fs8_0_sorted, color=c_lcdm, lw=2, label=r'$\Lambda$CDM')
ax4.plot(z_sorted, fs8_1_sorted, color=c_iam, lw=2, ls='--',
         label=r'IAM ($\mu_0=-0.136$)')
ax4.errorbar(boss_z, boss_fs8, yerr=boss_err, fmt='s', color=c_data,
             markersize=6, capsize=3, capthick=1.2, label='BOSS/eBOSS', zorder=5)
ax4.set_xlabel(r'Redshift $z$', fontsize=11)
ax4.set_ylabel(r'$f\sigma_8(z)$', fontsize=11)
ax4.set_title('(d) Growth Rate vs Data', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9, loc='upper right')
ax4.set_xlim(0, 2.2)
ax4.set_ylim(0.2, 0.6)

# ------ (e) Matter Power Spectrum ------
ax5 = fig.add_subplot(gs[1, 1])
# pk arrays: row 0 = highest z (last in sorted list = z=0 is last row)
# CAMB sorts redshifts, z=0 is the last entry
ax5.loglog(kh0, pk0[-1,:], color=c_lcdm, lw=1.5, label=r'$\Lambda$CDM')
ax5.loglog(kh1, pk1[-1,:], color=c_iam, lw=1.5, ls='--', label='IAM')
ax5.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$', fontsize=11)
ax5.set_ylabel(r'$P(k)\ [h^{-3}\mathrm{Mpc}^3]$', fontsize=11)
ax5.set_title('(e) Matter Power Spectrum ($z=0$)', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.set_xlim(1e-4, 2)

# Inset: P(k) ratio
ax5i = ax5.inset_axes([0.45, 0.08, 0.50, 0.35])
pk_ratio = pk1[-1,:] / pk0[-1,:]
ax5i.semilogx(kh0, pk_ratio, color=c_iam, lw=1.2)
ax5i.axhline(1.0, color='gray', lw=0.5, ls=':')
ax5i.set_ylabel(r'$P_{\rm IAM}/P_{\Lambda\rm CDM}$', fontsize=8)
ax5i.set_xlim(1e-4, 2)
ax5i.set_ylim(0.92, 1.02)
ax5i.tick_params(labelsize=7)

# ------ (f) mu(z) and activation ------
ax6 = fig.add_subplot(gs[1, 2])
z_arr = np.linspace(0.001, 3, 500)
a_arr = 1.0 / (1.0 + z_arr)
mu_arr = np.array([mu_iam(a) for a in a_arr])
E_arr = np.array([E_act(a) for a in a_arr])

# MGCAMB pure_MG parametrization for comparison
mu_mgcamb = np.array([1 + (-0.136) * OmDE(a)/OmDE(1.0) for a in a_arr])

ax6.plot(z_arr, mu_arr, color=c_iam, lw=2, label=r'$\mu_{\rm IAM}(z)$ [exact]')
ax6.plot(z_arr, mu_mgcamb, color=c_iam, lw=1.5, ls='--', alpha=0.6,
         label=r'$\mu_{\rm MGCAMB}(z)$ [approx]')
ax6.plot(z_arr, E_arr, color='#7570b3', lw=1.3, ls=':',
         label=r'$E(a)=e^{1-1/a}$')
ax6.axhline(1.0, color='gray', lw=0.5, ls=':')
ax6.fill_between(z_arr, mu_arr, mu_mgcamb, alpha=0.1, color=c_iam)
ax6.set_xlabel(r'Redshift $z$', fontsize=11)
ax6.set_ylabel(r'$\mu(z)$  /  $E(a)$', fontsize=11)
ax6.set_title(r'(f) Modified Gravity Coupling $\mu(z)$', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8, loc='center right')
ax6.set_xlim(0, 3)
ax6.set_ylim(0.0, 1.1)

# ---- Main title ----
fig.suptitle('Informational Actualization Model: Full Boltzmann Validation via MGCAMB\n'
             r'$\mu_0 = -0.136$,  $\Sigma = 1$ (exact),  $\beta_m = \Omega_m/2 = 0.1575$',
             fontsize=13, fontweight='bold', y=0.995)

plt.savefig('iam_mgcamb_validation_6panel.pdf', dpi=200, bbox_inches='tight')
plt.savefig('iam_mgcamb_validation_6panel.png', dpi=200, bbox_inches='tight')
print("\nSaved: iam_mgcamb_validation_6panel.pdf/.png")
