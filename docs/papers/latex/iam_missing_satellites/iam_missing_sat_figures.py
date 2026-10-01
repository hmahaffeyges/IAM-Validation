"""
IAM Missing Satellites Paper Figures
Two dedicated figures in IAM house style
Generated: March 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')

# ── IAM House Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.labelsize':   11,
    'axes.titlesize':   11,
    'legend.fontsize':  8.5,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'axes.linewidth':   1.1,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'figure.dpi':       150,
    'lines.linewidth':  2.0,
})

IAM_RED    = '#d62728'
IAM_BLUE   = '#1f77b4'
IAM_GREEN  = '#2ca02c'
IAM_GREY   = '#7f7f7f'
IAM_ORANGE = '#ff7f0e'
PANEL_BG   = '#f9f9f9'

# ── Constants ─────────────────────────────────────────────────────────────────
G       = 6.674e-11
c       = 3.0e8
H0      = 67.4e3 / 3.086e22
Omega_m = 0.3153
Omega_L = 1.0 - Omega_m
beta_m  = Omega_m / 2.0
Msun    = 1.989e30
kpc     = 3.086e19

def E(a):
    return np.exp(1.0 - 1.0/a)

def H2(a):
    return Omega_m/a**3 + Omega_L

def mu(a):
    h2 = H2(a)
    return h2 / (h2 + beta_m * E(a))

def M_min(sigma_kms):
    sigma = sigma_kms * 1e3
    return 4 * Omega_m * sigma**3 / (G * H0 * Msun)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Two mechanisms — mu(a) suppression + M_min scaling
# ─────────────────────────────────────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig1.patch.set_facecolor('white')
for ax in [ax1, ax2]:
    ax.set_facecolor(PANEL_BG)

# ── Panel (a): Mechanism A — growth suppression ──────────────────────────────
z = np.linspace(0, 3.0, 500)
a = 1.0/(1.0+z)
mu_z = mu(a)

ax1.plot(z, mu_z, color=IAM_RED, linewidth=2.5,
         label=r'IAM: $\mu(a) < 1$ (Mechanism A)')
ax1.axhline(1.0, color=IAM_GREY, linewidth=1.2, linestyle='--',
            label=r'$\Lambda$CDM: $\mu = 1$')

# Euclid sensitivity
ax1.axhspan(0.864 - 0.04, 0.864 + 0.04, alpha=0.15, color=IAM_GREEN,
            label=r'Euclid DR1 $\sigma(\mu_0) \approx 0.04$')

# mu_0 annotation — single, clean
ax1.annotate(r'$\mu_0 = 0.864$  (13.6% suppression)',
             xy=(0.05, 0.864), xytext=(0.6, 0.848),
             fontsize=8.5, color=IAM_RED,
             arrowprops=dict(arrowstyle='->', color=IAM_RED, lw=1.2))

# Suppression zone shading
ax1.fill_between(z, mu_z, 1.0, alpha=0.08, color=IAM_RED,
                 label='Suppression zone')

# High-z recovery — well inside axes
ax1.text(1.8, 0.988, r'$\mu \to 1$ at high $z$',
         fontsize=8, color=IAM_GREY, ha='center', style='italic')

ax1.set_xlabel('Redshift $z$', fontsize=11)
ax1.set_ylabel(r'Matter-sector coupling $\mu(a)$', fontsize=11)
ax1.set_xlim(0, 3.0)
ax1.set_ylim(0.835, 1.015)
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.2)
ax1.set_title('(a)  Mechanism A: Growth Suppression from $\\mu < 1$',
              fontweight='bold', pad=6)

# MCMC note as xlabel supplement
ax1.set_xlabel(
    'Redshift $z$\n'
    r'$\Delta\chi^2 = +0.54$ vs $\Lambda$CDM  |  17 MCMC chains  |  zero free parameters',
    fontsize=9)

# (mu_0 annotation already placed above)

# Suppression zone shading
ax1.fill_between(z, mu_z, 1.0, alpha=0.08, color=IAM_RED,
                 label='Suppression zone')

# (high-z and MCMC note now in xlabel above)

# ── Panel (b): Mechanism B — M_min sigma^3 scaling ───────────────────────────
sigma_arr = np.logspace(np.log10(1.5), np.log10(200), 300)
Mmin_arr = M_min(sigma_arr)

ax2.loglog(sigma_arr, Mmin_arr, color=IAM_RED, linewidth=2.5,
           label=r'$M_\mathrm{min} = 4\Omega_m\sigma^3/GH$' + '\n' +
                 r'($\sigma^3$ scaling derived)')

# sigma_crit line
sigma_crit = 4.0
Mmin_crit = M_min(sigma_crit)
ax2.axvline(sigma_crit, color=IAM_ORANGE, linewidth=1.8, linestyle='--',
            label=r'$\sigma_\mathrm{crit} \approx 4$ km s$^{-1}$')
ax2.axhline(10**8.4, color=IAM_BLUE, linewidth=1.5, linestyle=':',
            label=r'Observed floor $\sim 10^{8.4}\,M_\odot$')

# Dispersal zone shading
sigma_low = sigma_arr[sigma_arr <= sigma_crit]
ax2.fill_between([1.5, sigma_crit], [1e4, 1e4], [1e14, 1e14],
                 alpha=0.08, color=IAM_RED)
ax2.text(2.5, 2e5, 'Dispersal\nzone', ha='center', va='center',
         fontsize=8, color=IAM_RED, style='italic')

# sigma_crit annotation
ax2.annotate(r'$\sigma_\mathrm{crit} \approx 4$ km s$^{-1}$',
             xy=(sigma_crit, 1e7), xytext=(8, 3e5),
             fontsize=8.5, color=IAM_ORANGE,
             arrowprops=dict(arrowstyle='->', color=IAM_ORANGE, lw=1.2))

# Slope indicator
ax2.annotate('', xy=(30, M_min(30)), xytext=(10, M_min(10)),
             arrowprops=dict(arrowstyle='->', color=IAM_RED, lw=1.5))
ax2.text(40, M_min(25)*1.8, r'slope $= +3$', fontsize=8.5,
         color=IAM_RED, fontweight='bold', style='italic')

# Normalization note
ax2.text(50, 1e5,
         'Normalization offset\n~100x (open problem)',
         ha='center', fontsize=7.5, color=IAM_GREY, style='italic',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                   edgecolor=IAM_GREY, alpha=0.8))

ax2.set_xlabel(r'Velocity dispersion $\sigma$ (km s$^{-1}$)', fontsize=11)
ax2.set_ylabel(r'$M_\mathrm{min}$ ($M_\odot$)', fontsize=11)
ax2.set_xlim(1.5, 200)
ax2.set_ylim(1e4, 1e14)
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.2, which='both')
ax2.set_title(r'(b)  Mechanism B: Minimum Halo Mass from Virial Partition',
              fontweight='bold', pad=6)

fig1.suptitle('IAM: Two Mechanisms for the Missing Satellites Problem',
              fontsize=12, fontweight='bold', y=1.01)
fig1.text(0.5, -0.02,
    'Mahaffey (2026)  |  doi:10.5281/zenodo.18702042  |  '
    'github.com/hmahaffeyges/IAM-Validation  |  '
    'Timestamped predictions: March 16, 2026',
    ha='center', fontsize=7.5, style='italic', color='#666666')

plt.tight_layout()
plt.savefig('iam_missing_sat_fig1.pdf', bbox_inches='tight', dpi=180)
plt.savefig('iam_missing_sat_fig1.png', bbox_inches='tight', dpi=180)
plt.close()
print("Figure 1 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Unified mechanism — BH as mandatory vault
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('white')
ax.set_facecolor(PANEL_BG)
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Four columns
cols = [1.5, 4.0, 7.5, 10.5]
colors_col = [IAM_RED, IAM_ORANGE, IAM_GREEN, IAM_BLUE]
headers = [
    '$\\sigma < \\sigma_\\mathrm{crit}$\nNo BH forms',
    '$\\sigma \\approx \\sigma_\\mathrm{crit}$\nDispersing',
    '$\\sigma > \\sigma_\\mathrm{crit}$\nSmall galaxy',
    '$\\sigma \\gg \\sigma_\\mathrm{crit}$\nLarge galaxy'
]
mech_a_text = r'Mechanism A: $\mu < 1$' + '\n' + 'suppresses growth\nat late times'
mech_b = [
    'Mechanism B:\nVirial partition\ncannot close\n(no local vault)',
    'Mechanism B:\nNear threshold\n(marginally\ndisperses)',
    'Mechanism B:\nBH vault absorbs\ninformation half\n(core forms)',
    'Mechanism B:\nLarge BH vault\nabsorbs all local\ndecoherence'
]
outcomes = ['HALO\nDISPERSES', 'UNSTABLE', 'SATELLITE\nwith CORE', 'GALAXY\nwith SMBH']

for i, (x, col, hdr, mb, out) in enumerate(zip(
        cols, colors_col, headers, mech_b, outcomes)):

    # Header
    hbox = mpatches.FancyBboxPatch((x-1.1, 8.3), 2.2, 1.5,
        boxstyle="round,pad=0.12", facecolor=col, alpha=0.9,
        edgecolor=col, linewidth=1.5)
    ax.add_patch(hbox)
    ax.text(x, 9.05, hdr, ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='white', linespacing=1.4)

    # Arrow down
    ax.annotate('', xy=(x, 7.65), xytext=(x, 8.25),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    # Mechanism A box (same for all columns) — white bg, bold dark text
    abox = mpatches.FancyBboxPatch((x-1.1, 6.2), 2.2, 1.35,
        boxstyle="round,pad=0.12", facecolor='white',
        edgecolor=IAM_BLUE, linewidth=1.8)
    ax.add_patch(abox)
    ax.text(x, 6.87, mech_a_text, ha='center', va='center',
            fontsize=7.5, color='#111111', fontweight='bold', linespacing=1.3)

    # Arrow down
    ax.annotate('', xy=(x, 5.55), xytext=(x, 6.15),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    # Mechanism B box — white bg, bold dark text, colored border
    edge_col = IAM_RED if i < 2 else IAM_GREEN
    bbox = mpatches.FancyBboxPatch((x-1.1, 4.0), 2.2, 1.45,
        boxstyle="round,pad=0.12", facecolor='white',
        edgecolor=edge_col, linewidth=1.8)
    ax.add_patch(bbox)
    ax.text(x, 4.72, mb, ha='center', va='center',
            fontsize=7.5, color='#111111', fontweight='bold', linespacing=1.3)

    # Arrow down
    ax.annotate('', xy=(x, 3.35), xytext=(x, 3.95),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.5))

    # Outcome box
    obox = mpatches.FancyBboxPatch((x-1.1, 2.2), 2.2, 1.05,
        boxstyle="round,pad=0.12", facecolor=col, alpha=0.9,
        edgecolor=col, linewidth=1.5)
    ax.add_patch(obox)
    ax.text(x, 2.72, out, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='white', linespacing=1.3)

# Mechanism labels on left
ax.text(-0.1, 9.05, 'Halo\nproperties', ha='center', va='center',
        fontsize=8, fontweight='bold', color='#333333', linespacing=1.3)
ax.text(-0.1, 6.87, 'Mechanism\nA', ha='center', va='center',
        fontsize=8, fontweight='bold', color=IAM_BLUE, linespacing=1.3)
ax.text(-0.1, 4.72, 'Mechanism\nB', ha='center', va='center',
        fontsize=8, fontweight='bold', color='#333333', linespacing=1.3)
ax.text(-0.1, 2.72, 'Outcome', ha='center', va='center',
        fontsize=8, fontweight='bold', color='#333333', linespacing=1.3)

# Horizontal dividers
for y in [8.2, 7.55, 6.1, 3.9, 3.25]:
    ax.axhline(y, color=IAM_GREY, linewidth=0.5, alpha=0.3,
               xmin=0.05, xmax=0.98)

# Bottom note
ax.text(6.0, 1.2,
        r'Both mechanisms follow from the virial theorem: $2K + V = 0$ requires both halves to close simultaneously.',
        ha='center', va='center', fontsize=9, color='#333333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                  edgecolor=IAM_GREY, linewidth=1.0))

ax.text(6.0, 0.4,
        r'$\sigma_\mathrm{crit} \approx 4$ km s$^{-1}$ (Mechanism B)  $|$  '
        r'$\mu_0 - 1 = -0.136$ (Mechanism A, Euclid DR1 Oct 2026)  $|$  '
        r'Zero free parameters beyond $\Lambda$CDM',
        ha='center', va='center', fontsize=8, color=IAM_GREY, style='italic')

ax.set_title('The Two IAM Mechanisms for Missing Satellites',
             fontweight='bold', pad=8, fontsize=12)

fig2.text(0.5, -0.01,
    'Mahaffey (2026)  |  doi:10.5281/zenodo.18702042  |  '
    'github.com/hmahaffeyges/IAM-Validation  |  March 2026',
    ha='center', fontsize=7.5, style='italic', color='#666666')

plt.tight_layout()
plt.savefig('iam_missing_sat_fig2.pdf', bbox_inches='tight', dpi=180)
plt.savefig('iam_missing_sat_fig2.png', bbox_inches='tight', dpi=180)
plt.close()
print("Figure 2 saved.")
print("Done.")
