#!/usr/bin/env python3
"""
IAM Theory Paper Figure: 3-Panel mu-Sigma Prediction
======================================================
Produces the main theory paper figure showing:
  (a) mu(z) prediction with current/projected constraints
  (b) mu-Sigma phase space showing IAM's unique signature
  (c) Activation function E(a) with physical interpretation

For inclusion in: "Horizon Thermodynamics and Gravitational Decoherence
                   as the Origin of mu < 1, Sigma = 1"

Author: H.W. Mahaffey
Date: February 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
from matplotlib.lines import Line2D

# ===========================================================================
# COSMOLOGICAL PARAMETERS (Planck 2018)
# ===========================================================================
H0 = 67.36
Om = 0.3153
OL = 0.6847
sigma8_fid = 0.8111

# IAM parameters
beta_m = Om / 2.0  # = 0.1575, derived from virial theorem
mu0_iam = -beta_m / (1.0 + beta_m)  # = -0.13495

# ===========================================================================
# IAM FUNCTIONS
# ===========================================================================
def E_activation(a):
    """IAM activation function E(a) = exp(1 - 1/a)"""
    return np.exp(1.0 - 1.0 / a)

def H2_LCDM(a):
    """LCDM H^2/H0^2"""
    return Om * a**(-3) + OL

def H2_IAM(a):
    """IAM H^2/H0^2"""
    return H2_LCDM(a) + beta_m * E_activation(a)

def mu_iam(z):
    """IAM mu(z) = E^2_LCDM / E^2_IAM"""
    a = 1.0 / (1.0 + z)
    return H2_LCDM(a) / H2_IAM(a)

def Omega_DE(a):
    """Dark energy density fraction"""
    return OL / (Om * a**(-3) + OL)

def mu_mgcamb(z, mu0):
    """MGCAMB parametrization: mu = 1 + mu0 * Omega_DE(a)"""
    a = 1.0 / (1.0 + z)
    return 1.0 + mu0 * Omega_DE(a)

# ===========================================================================
# FIGURE
# ===========================================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle(
    r'IAM Prediction: $\mu < 1$, $\Sigma = 1$ from Gravitational Decoherence',
    fontsize=15, fontweight='bold', y=1.02
)

# ---- Color scheme ----
iam_color = '#C62828'       # deep red
lcdm_color = '#424242'      # dark gray
euclid_color = '#1565C0'    # blue
desi_color = '#2E7D32'      # green
mgcamb_color = '#E65100'    # orange

# ===========================================================================
# PANEL (a): mu(z)
# ===========================================================================
z_fine = np.linspace(0, 3.0, 500)

# IAM exact prediction
mu_exact = mu_iam(z_fine)

# MGCAMB approximation
mu_approx = mu_mgcamb(z_fine, mu0_iam)

# LCDM
mu_lcdm = np.ones_like(z_fine)

# Plot
ax1.plot(z_fine, mu_lcdm, color=lcdm_color, linestyle='-', linewidth=2.0,
         label=r'$\Lambda$CDM ($\mu = 1$)')
ax1.plot(z_fine, mu_exact, color=iam_color, linestyle='-', linewidth=2.5,
         label=r'IAM: $\mu = E^2_{\Lambda\mathrm{CDM}} / E^2_\mathrm{IAM}$')
ax1.plot(z_fine, mu_approx, color=mgcamb_color, linestyle='--', linewidth=1.5,
         alpha=0.8, label=r'MGCAMB: $\mu = 1 + \mu_0\,\Omega_\mathrm{DE}(a)$')

# Current constraints
# DESI DR1: mu_0 = 0.11 +0.45/-0.54 (at z~0, plotted as mu(0) = 1 + mu_0)
ax1.errorbar(0.05, 1.0 + 0.11, yerr=[[0.54], [0.45]], fmt='D',
             color=desi_color, markersize=7, capsize=4, linewidth=1.5,
             label='DESI DR1', zorder=5)

# DES Y3: mu_0 = -0.4 +/- 0.4
ax1.errorbar(0.15, 1.0 - 0.4, yerr=0.4, fmt='s',
             color='#7B1FA2', markersize=7, capsize=4, linewidth=1.5,
             label='DES Y3', zorder=5)

# Andrade+2024 (ACT+WMAP+SDSS): mu_0 - 1 = 0.02 +/- 0.19
ax1.errorbar(0.25, 1.0 + 0.02, yerr=0.19, fmt='^',
             color='#00838F', markersize=7, capsize=4, linewidth=1.5,
             label='ACT+WMAP+SDSS', zorder=5)

# Euclid projected sensitivity band
sigma_euclid = 0.04
ax1.fill_between(z_fine, mu_exact - sigma_euclid, mu_exact + sigma_euclid,
                 alpha=0.12, color=euclid_color, label=r'Euclid $1\sigma$ (projected)')

# IAM specific values
for z_mark, mu_val_str in [(0, '0.864'), (0.5, '0.948'), (1.0, '0.982')]:
    mu_val = mu_iam(z_mark)
    ax1.plot(z_mark, mu_val, 'o', color=iam_color, markersize=5, zorder=6)
    ax1.annotate(f'{mu_val:.3f}', (z_mark, mu_val),
                 textcoords='offset points', xytext=(8, -12),
                 fontsize=8, color=iam_color)

ax1.set_xlabel('Redshift $z$', fontsize=13)
ax1.set_ylabel(r'$\mu(z)$', fontsize=14)
ax1.set_title(r'(a) Gravitational Coupling $\mu(z)$', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.05, 3.0)
ax1.set_ylim(0.45, 1.55)
ax1.legend(fontsize=8, loc='lower right', framealpha=0.9)
ax1.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax1.grid(alpha=0.15)

# ===========================================================================
# PANEL (b): mu-Sigma Phase Space
# ===========================================================================

# IAM prediction point
ax2.plot(0.0, mu0_iam, '*', color=iam_color, markersize=18, zorder=10,
         markeredgecolor='black', markeredgewidth=0.5)
ax2.annotate(r'IAM', (0.0, mu0_iam),
             textcoords='offset points', xytext=(14, -2),
             fontsize=11, fontweight='bold', color=iam_color)

# GR point
ax2.plot(0.0, 0.0, 'o', color=lcdm_color, markersize=10, zorder=9,
         markeredgecolor='black', markeredgewidth=0.5)
ax2.annotate(r'$\Lambda$CDM / GR', (0.0, 0.0),
             textcoords='offset points', xytext=(12, 8),
             fontsize=10, color=lcdm_color)

# f(R) gravity region (mu > 0, Sigma > 0, correlated)
fR_ellipse = Ellipse((0.15, 0.2), 0.25, 0.35, angle=-15,
                      facecolor='#42A5F5', alpha=0.2, edgecolor='#1565C0',
                      linewidth=1.5, linestyle='--')
ax2.add_patch(fR_ellipse)
ax2.annotate(r'$f(R)$', (0.22, 0.32), fontsize=10, color='#1565C0',
             fontweight='bold')

# Horndeski / scalar-tensor region (both can vary)
horn_ellipse = Ellipse((0.08, 0.1), 0.5, 0.6, angle=30,
                        facecolor='#66BB6A', alpha=0.12, edgecolor='#2E7D32',
                        linewidth=1.5, linestyle='--')
ax2.add_patch(horn_ellipse)
ax2.annotate('Horndeski', (0.25, 0.05), fontsize=10, color='#2E7D32',
             fontweight='bold')

# DGP region (mu > 0, Sigma < 0 typically)
dgp_ellipse = Ellipse((0.12, -0.06), 0.15, 0.12, angle=10,
                       facecolor='#FFA726', alpha=0.2, edgecolor='#E65100',
                       linewidth=1.5, linestyle='--')
ax2.add_patch(dgp_ellipse)
ax2.annotate('DGP', (0.16, -0.10), fontsize=10, color='#E65100',
             fontweight='bold')

# Highlight the unique IAM region: mu < 0, Sigma = 0
ax2.fill_between([-0.05, 0.05], -0.25, -0.01, alpha=0.08, color=iam_color)
ax2.annotate(r'$\mu < 1$, $\Sigma = 1$' + '\n(unique to IAM)',
             (-0.04, -0.22), fontsize=8, color=iam_color, style='italic',
             ha='left')

# Axes
ax2.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.axvline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.set_xlabel(r'$\Sigma_0 \equiv \Sigma(z{=}0) - 1$', fontsize=13)
ax2.set_ylabel(r'$\mu_0 \equiv \mu(z{=}0) - 1$', fontsize=14)
ax2.set_title(r'(b) $\mu$–$\Sigma$ Phase Space', fontsize=12, fontweight='bold')
ax2.set_xlim(-0.3, 0.45)
ax2.set_ylim(-0.3, 0.45)
ax2.set_aspect('equal')
ax2.grid(alpha=0.15)

# ===========================================================================
# PANEL (c): Activation Function E(a)
# ===========================================================================
a_fine = np.linspace(0.05, 2.5, 500)
z_for_a = 1.0/a_fine - 1.0
Ea = E_activation(a_fine)

ax3.plot(a_fine, Ea, color=iam_color, linewidth=2.5,
         label=r'$\mathcal{E}(a) = \exp(1 - 1/a)$')
ax3.axhline(np.e, color='gray', linestyle=':', linewidth=1.0, alpha=0.6)
ax3.annotate(r'$\mathcal{E} \to e \approx 2.718$ (saturation)',
             (1.8, np.e + 0.08), fontsize=9, color='gray')

# Mark key epochs
# Recombination: a ~ 1/1100
# But we start at a=0.05, so mark some physical epochs
epochs = [
    (1.0/(1+1100), r'Recombination', 'below'),  # too small to show
    (1.0/(1+10), r'$z=10$', 'above'),
    (1.0/(1+2), r'$z=2$', 'above'),
    (1.0/(1+1), r'$z=1$', 'below'),
    (1.0, r'Today ($a=1$)', 'above'),
]

for a_ep, label, pos in epochs:
    if a_ep >= 0.05:
        E_ep = E_activation(a_ep)
        ax3.plot(a_ep, E_ep, 'o', color=iam_color, markersize=5, zorder=6)
        offset = (5, 10) if pos == 'above' else (5, -15)
        ax3.annotate(f'{label}\n$\\mathcal{{E}}={E_ep:.3f}$',
                     (a_ep, E_ep), textcoords='offset points',
                     xytext=offset, fontsize=7.5, color=iam_color,
                     ha='left')

# Mark today
ax3.axvline(1.0, color=lcdm_color, linestyle='--', linewidth=1.0, alpha=0.4)

# Show mu(a) on right axis
ax3_right = ax3.twinx()
mu_of_a = H2_LCDM(a_fine) / H2_IAM(a_fine)
ax3_right.plot(a_fine, mu_of_a, color=euclid_color, linewidth=1.8,
               linestyle='--', alpha=0.8)
ax3_right.set_ylabel(r'$\mu(a)$ (dashed blue)', fontsize=11, color=euclid_color)
ax3_right.tick_params(axis='y', labelcolor=euclid_color)
ax3_right.set_ylim(0.5, 1.05)
ax3_right.axhline(1.0, color=euclid_color, linestyle=':', linewidth=0.5, alpha=0.3)

ax3.set_xlabel('Scale factor $a$', fontsize=13)
ax3.set_ylabel(r'$\mathcal{E}(a)$ (solid red)', fontsize=13, color=iam_color)
ax3.tick_params(axis='y', labelcolor=iam_color)
ax3.set_title(r'(c) Activation Function & $\mu(a)$', fontsize=12, fontweight='bold')
ax3.set_xlim(0.05, 2.5)
ax3.set_ylim(0, 3.0)
ax3.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax3.grid(alpha=0.15)

# Add secondary x-axis for redshift on panel (c)
ax3_top = ax3.twiny()
z_ticks = [10, 5, 2, 1, 0.5, 0]
a_ticks = [1.0/(1.0+z) for z in z_ticks]
ax3_top.set_xlim(ax3.get_xlim())
ax3_top.set_xticks(a_ticks)
ax3_top.set_xticklabels([str(z) for z in z_ticks], fontsize=8)
ax3_top.set_xlabel('Redshift $z$', fontsize=10, labelpad=8)

# ===========================================================================
# LAYOUT AND SAVE
# ===========================================================================
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.35)

output_path = '/mnt/user-data/outputs/iam_theory_paper_figure.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
output_png = '/mnt/user-data/outputs/iam_theory_paper_figure.png'
plt.savefig(output_png, dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"Saved: {output_path}")
print(f"Saved: {output_png}")

# Print key values for paper
print("\n=== Key Values for Paper ===")
for z_val in [0, 0.5, 1.0, 2.0, 3.0]:
    print(f"  mu(z={z_val}) = {mu_iam(z_val):.4f}")
print(f"  mu_0 = mu(0) - 1 = {mu_iam(0)-1:.5f}")
print(f"  beta_m = Omega_m/2 = {beta_m:.4f}")
print(f"  E(a=1) = {E_activation(1.0):.4f}")
print(f"  E(a->inf) = e = {np.e:.4f}")
