#!/usr/bin/env python3
"""
Generate all 5 figures for the IAM observational paper.
Filenames match the LaTeX \includegraphics references exactly.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.integrate import quad

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
    'font.family': 'serif',
})

# IAM parameters
Om = 0.315
OL = 0.685
Or = 9.24e-5
beta = Om / 2  # 0.1575
H0 = 67.36

def E_activation(a):
    """IAM activation function"""
    return np.exp(1.0 - 1.0/a)

def H2_LCDM(a):
    """Normalized H^2/H0^2 for LCDM"""
    return Om * a**(-3) + Or * a**(-4) + OL

def mu_exact(a):
    """IAM exact mu(a)"""
    h2 = H2_LCDM(a)
    return h2 / (h2 + beta * E_activation(a))

def mu_mgcamb(a):
    """MGCAMB approximation: mu = 1 + mu0 * Omega_DE(a)"""
    mu0 = -0.13495
    ODE_a = OL / (Om * a**(-3) + OL)  # neglecting radiation
    return 1.0 + mu0 * ODE_a


# ============================================================
# FIGURE 1: fig_mu_profile.pdf
# mu(z) exact vs MGCAMB + activation function E(a)
# ============================================================
def make_fig_mu_profile():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    z = np.linspace(0, 5, 500)
    a = 1.0 / (1.0 + z)

    # Left panel: mu(z)
    ax1.plot(z, mu_exact(a), 'b-', lw=2.0, label=r'IAM exact: $\mu = H^2_{\Lambda CDM}/(H^2_{\Lambda CDM} + \beta\mathcal{E})$')
    ax1.plot(z, mu_mgcamb(a), 'r--', lw=1.8, label=r'MGCAMB: $\mu = 1 + \mu_0 \Omega_{DE}(a)$')
    ax1.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax1.set_xlabel('Redshift $z$')
    ax1.set_ylabel(r'$\mu(z)$')
    ax1.set_ylim(0.84, 1.02)
    ax1.set_xlim(0, 5)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    # Right panel: E(a)
    a_arr = np.linspace(0.05, 1.0, 500)
    ax2.plot(a_arr, E_activation(a_arr), 'b-', lw=2.0)
    ax2.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax2.set_xlabel('Scale factor $a$')
    ax2.set_ylabel(r'$\mathcal{E}(a) = \exp(1 - 1/a)$')
    ax2.set_xlim(0.05, 1.0)
    ax2.set_ylim(0, 1.1)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())

    # Mark key redshifts
    for zmark, label in [(0, '$z=0$'), (1, '$z=1$'), (3, '$z=3$')]:
        amark = 1.0/(1.0+zmark)
        ax2.plot(amark, E_activation(amark), 'ko', ms=5)
        ax2.annotate(label, (amark, E_activation(amark)),
                     textcoords='offset points', xytext=(8, 5), fontsize=9)

    fig.tight_layout(w_pad=3)
    fig.savefig('/home/claude/fig_mu_profile.pdf')
    plt.close()
    print("  Created fig_mu_profile.pdf")


# ============================================================
# FIGURE 2: fig_posterior_comparison.pdf
# Parameter posteriors: LCDM vs IAM fixed (Planck+RSD)
# Using Gaussian approximations from chain results
# ============================================================
def make_fig_posterior_comparison():
    # Run F (LCDM+RSD) and Run D (IAM fixed+RSD) parameters
    params = {
        r'$H_0$': {'F': (67.18, 0.53), 'D': (67.07, 0.51), 'unit': r' [km s$^{-1}$ Mpc$^{-1}$]'},
        r'$\sigma_8$': {'F': (0.8131, 0.0060), 'D': (0.8001, 0.0058), 'unit': ''},
        r'$\Omega_b h^2$': {'F': (0.02240, 0.00014), 'D': (0.02240, 0.00014), 'unit': ''},
        r'$\Omega_c h^2$': {'F': (0.1196, 0.0009), 'D': (0.1197, 0.0009), 'unit': ''},
        r'$n_s$': {'F': (0.9658, 0.0036), 'D': (0.9657, 0.0036), 'unit': ''},
        r'$\ln(10^{10}A_s)$': {'F': (3.0517, 0.0143), 'D': (3.0530, 0.0149), 'unit': ''},
    }

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    axes = axes.flatten()

    for idx, (name, vals) in enumerate(params.items()):
        ax = axes[idx]
        mu_F, sig_F = vals['F']
        mu_D, sig_D = vals['D']

        # Range: 4 sigma around the wider distribution
        sig_max = max(sig_F, sig_D)
        center = (mu_F + mu_D) / 2
        x = np.linspace(center - 4.5*sig_max, center + 4.5*sig_max, 300)

        y_F = np.exp(-0.5*((x - mu_F)/sig_F)**2) / (sig_F * np.sqrt(2*np.pi))
        y_D = np.exp(-0.5*((x - mu_D)/sig_D)**2) / (sig_D * np.sqrt(2*np.pi))

        ax.fill_between(x, y_F, alpha=0.25, color='gray')
        ax.plot(x, y_F, 'k-', lw=1.5, label=r'$\Lambda$CDM (F)')
        ax.fill_between(x, y_D, alpha=0.25, color='steelblue')
        ax.plot(x, y_D, 'steelblue', lw=1.5, label=r'IAM fixed (D)')

        ax.set_xlabel(name + vals['unit'])
        ax.set_yticks([])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Planck + RSD: Parameter Posteriors', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig('/home/claude/fig_posterior_comparison.pdf')
    plt.close()
    print("  Created fig_posterior_comparison.pdf")


# ============================================================
# FIGURE 3: fig_fsigma8.pdf
# f*sigma8(z) predictions vs SDSS data
# ============================================================
def make_fig_fsigma8():
    # RSD data points
    rsd_data = [
        (0.067, 0.423, 0.055, '6dFGS'),
        (0.150, 0.530, 0.160, 'SDSS MGS'),
        (0.380, 0.497, 0.045, 'BOSS DR12'),
        (0.510, 0.459, 0.038, 'BOSS DR12'),
        (0.700, 0.473, 0.041, 'eBOSS LRG'),
        (0.850, 0.315, 0.095, 'eBOSS ELG'),
        (1.480, 0.462, 0.045, 'eBOSS QSO'),
    ]

    # Compute growth for LCDM and IAM using ODE integration
    from scipy.integrate import solve_ivp

    def growth_ode(lna, y, use_iam=False):
        """Growth ODE in ln(a): D'' + [2 + dlnH/dlna] D' - 3/2 mu Omega_m(a) D = 0"""
        D, dDdlna = y
        a = np.exp(lna)
        h2 = H2_LCDM(a)
        
        # dlnH/dlna
        dh2_da = -3*Om*a**(-4) - 4*Or*a**(-5)
        dlnH_dlna = a * dh2_da / (2 * h2)
        
        omega_m_a = Om * a**(-3) / h2
        
        mu_val = mu_exact(a) if use_iam else 1.0
        
        d2Ddlna2 = -(2.0 + dlnH_dlna) * dDdlna + 1.5 * mu_val * omega_m_a * D
        return [dDdlna, d2Ddlna2]

    lna_span = (np.log(0.01), np.log(1.0))
    lna_eval = np.linspace(lna_span[0], lna_span[1], 1000)

    sol_lcdm = solve_ivp(growth_ode, lna_span, [0.01, 0.01], t_eval=lna_eval,
                         args=(False,), rtol=1e-10, atol=1e-12)
    sol_iam = solve_ivp(growth_ode, lna_span, [0.01, 0.01], t_eval=lna_eval,
                        args=(True,), rtol=1e-10, atol=1e-12)

    a_arr = np.exp(lna_eval)
    z_arr = 1.0/a_arr - 1.0

    # Normalize D(a=1) = 1 for LCDM
    D_lcdm = sol_lcdm.y[0] / sol_lcdm.y[0][-1]
    dDdlna_lcdm = sol_lcdm.y[1] / sol_lcdm.y[0][-1]
    f_lcdm = dDdlna_lcdm / D_lcdm  # f = dlnD/dlna

    # For IAM, normalize to match sigma8 ratio
    sigma8_lcdm = 0.8131
    sigma8_iam = 0.8001
    D_iam = sol_iam.y[0] / sol_iam.y[0][-1]
    dDdlna_iam = sol_iam.y[1] / sol_iam.y[0][-1]
    f_iam = dDdlna_iam / D_iam

    # f*sigma8(z) = f(z) * sigma8 * D(z)/D(0)
    fsig8_lcdm = f_lcdm * sigma8_lcdm * D_lcdm
    fsig8_iam = f_iam * sigma8_iam * D_iam

    fig, ax = plt.subplots(figsize=(7, 4.8))

    # Theory curves
    mask = (z_arr > 0) & (z_arr < 2.0)
    ax.plot(z_arr[mask], fsig8_lcdm[mask], 'k-', lw=2.0, label=r'$\Lambda$CDM ($\mu = 1$)')
    ax.plot(z_arr[mask], fsig8_iam[mask], 'steelblue', lw=2.0, ls='-',
            label=r'IAM ($\mu_0 = -0.135$)')

    # Data points
    for z_d, fs_d, err_d, name_d in rsd_data:
        ax.errorbar(z_d, fs_d, yerr=err_d, fmt='o', color='firebrick',
                    ms=6, capsize=3, elinewidth=1.3, zorder=5)

    # Label one data point for legend
    ax.errorbar([], [], yerr=[], fmt='o', color='firebrick', ms=6, capsize=3,
                label='SDSS BOSS/eBOSS')

    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'$f\sigma_8(z)$')
    ax.set_xlim(0, 1.7)
    ax.set_ylim(0.25, 0.60)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.tight_layout()
    fig.savefig('/home/claude/fig_fsigma8.pdf')
    plt.close()
    print("  Created fig_fsigma8.pdf")


# ============================================================
# FIGURE 4: fig_deltachi2_summary.pdf
# Delta-chi2 bar chart across 4 datasets
# ============================================================
def make_fig_deltachi2_summary():
    datasets = ['Planck\nonly', 'Planck\n+ RSD', 'Planck\n+ BAO', 'Planck\n+ Pantheon+']
    dchi2 = [1.43, 1.34, 2.32, 1.58]

    fig, ax = plt.subplots(figsize=(6, 4.2))

    colors = ['steelblue'] * 4
    bars = ax.bar(datasets, dchi2, color=colors, edgecolor='navy', lw=1.2,
                  width=0.55, alpha=0.85)

    # Threshold line
    ax.axhline(3.84, color='firebrick', ls='--', lw=1.5,
               label=r'95% CL threshold ($\Delta\chi^2 = 3.84$)')

    # Value labels on bars
    for bar, val in zip(bars, dchi2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'+{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel(r'$\Delta\chi^2$ (IAM fixed vs $\Lambda$CDM)')
    ax.set_ylim(0, 5.0)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Add "consistent" annotation
    ax.text(0.97, 0.85, 'All consistent\n(below threshold)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontstyle='italic', color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

    fig.tight_layout()
    fig.savefig('/home/claude/fig_deltachi2_summary.pdf')
    plt.close()
    print("  Created fig_deltachi2_summary.pdf")


# ============================================================
# FIGURE 5: fig_mu0_posterior.pdf
# mu0 posterior from Runs B (Planck only) and E (Planck+RSD)
# ============================================================
def make_fig_mu0_posterior():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    mu0_pred = -0.135

    # Run B: Planck only, mu0 = 0.032 +/- 0.127
    mu_B, sig_B = 0.032, 0.127
    # Run E: Planck+RSD, mu0 = 0.024 +/- 0.123
    mu_E, sig_E = 0.024, 0.123

    x = np.linspace(-0.5, 0.3, 500)

    y_B = np.exp(-0.5*((x - mu_B)/sig_B)**2) / (sig_B * np.sqrt(2*np.pi))
    y_E = np.exp(-0.5*((x - mu_E)/sig_E)**2) / (sig_E * np.sqrt(2*np.pi))

    ax.fill_between(x, y_B, alpha=0.2, color='gray')
    ax.plot(x, y_B, 'k-', lw=1.8, label=f'Planck only (Run B): $\\mu_0 = {mu_B:+.3f} \\pm {sig_B:.3f}$')

    ax.fill_between(x, y_E, alpha=0.25, color='steelblue')
    ax.plot(x, y_E, 'steelblue', lw=1.8,
            label=f'Planck + RSD (Run E): $\\mu_0 = {mu_E:+.3f} \\pm {sig_E:.3f}$')

    # IAM prediction line
    ax.axvline(mu0_pred, color='firebrick', ls='--', lw=2.0,
               label=f'IAM prediction: $\\mu_0 = {mu0_pred}$')

    # GR line
    ax.axvline(0, color='green', ls=':', lw=1.5, label=r'GR: $\mu_0 = 0$')

    ax.set_xlabel(r'$\mu_0$')
    ax.set_ylabel('Posterior density')
    ax.set_xlim(-0.5, 0.3)
    ax.set_yticks([])
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Annotate sigma distances
    ax.annotate(f'{abs(mu0_pred - mu_E)/sig_E:.1f}$\\sigma$',
                xy=(mu0_pred, 0.3), xytext=(-0.3, 2.5),
                arrowprops=dict(arrowstyle='->', color='firebrick', lw=1.2),
                fontsize=11, color='firebrick', fontweight='bold')

    fig.tight_layout()
    fig.savefig('/home/claude/fig_mu0_posterior.pdf')
    plt.close()
    print("  Created fig_mu0_posterior.pdf")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print("Generating 5 paper figures...")
    make_fig_mu_profile()
    make_fig_posterior_comparison()
    make_fig_fsigma8()
    make_fig_deltachi2_summary()
    make_fig_mu0_posterior()
    print("\nAll 5 figures generated successfully.")
