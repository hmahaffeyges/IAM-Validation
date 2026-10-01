"""
IAM Smolin Paper Figure
"Gravitational Decoherence, the Virial Partition, and the Emergence of Classical Structure"
Four-panel figure in IAM house style
Generated: March 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrowPatch
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ── IAM House Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'legend.fontsize':   8.5,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'axes.linewidth':    1.1,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.size':  4,
    'ytick.major.size':  4,
    'figure.dpi':        150,
})

IAM_RED    = '#d62728'
IAM_BLUE   = '#1f77b4'
IAM_GREEN  = '#2ca02c'
IAM_GREY   = '#7f7f7f'
IAM_PURPLE = '#9467bd'
IAM_ORANGE = '#ff7f0e'
PANEL_BG   = '#f9f9f9'

# ── Constants ─────────────────────────────────────────────────────────────────
G       = 6.674e-11
c       = 3.0e8
H0      = 67.4e3 / 3.086e22
Omega_m = 0.3153
beta_m  = Omega_m / 2.0
Msun    = 1.989e30

def E(a):
    return np.exp(1.0 - 1.0/a)

def mu(a):
    H2 = Omega_m/a**3 + (1-Omega_m)
    return H2 / (H2 + beta_m * E(a))

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 10))
fig.patch.set_facecolor('white')

gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.38,
                      left=0.08, right=0.97, top=0.92, bottom=0.07)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor(PANEL_BG)

# ═════════════════════════════════════════════════════════════════════════════
# Panel (a): The Virial Partition — Potency and Act
# ═════════════════════════════════════════════════════════════════════════════
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_facecolor(PANEL_BG)

# Central dividing line
ax1.axvline(5.0, color=IAM_GREY, linewidth=1.5, linestyle='--', alpha=0.5)

# Left side — Potential half → curvature (Potency)
ax1.text(2.5, 9.2, 'POTENTIAL HALF', ha='center', va='top',
         fontsize=9, fontweight='bold', color=IAM_BLUE)
ax1.text(2.5, 8.5, r'$\frac{1}{2}|V|$', ha='center', va='top',
         fontsize=14, color=IAM_BLUE)
ax1.text(2.5, 7.4, 'Geometric channel', ha='center', va='top',
         fontsize=8.5, color=IAM_BLUE, style='italic')

# Arrow down left
ax1.annotate('', xy=(2.5, 5.8), xytext=(2.5, 6.8),
             arrowprops=dict(arrowstyle='->', color=IAM_BLUE, lw=2.0))

# Spacetime curvature box
rect_l = mpatches.FancyBboxPatch((0.4, 4.0), 4.2, 1.6,
    boxstyle="round,pad=0.15", facecolor='#dce8f5',
    edgecolor=IAM_BLUE, linewidth=1.5)
ax1.add_patch(rect_l)
ax1.text(2.5, 4.8, 'Spacetime curvature', ha='center', va='center',
         fontsize=9, color=IAM_BLUE, fontweight='bold')
ax1.text(2.5, 4.25, '(dark matter geometry)', ha='center', va='center',
         fontsize=8, color=IAM_BLUE, style='italic')

# Arrow to "stable structure"
ax1.annotate('', xy=(2.5, 2.8), xytext=(2.5, 3.9),
             arrowprops=dict(arrowstyle='->', color=IAM_BLUE, lw=2.0))
ax1.text(2.5, 2.4, 'Stable structure', ha='center', va='center',
         fontsize=9, color=IAM_BLUE)

# Right side — Kinetic half → decoherence (Act)
ax1.text(7.5, 9.2, 'KINETIC HALF', ha='center', va='top',
         fontsize=9, fontweight='bold', color=IAM_RED)
ax1.text(7.5, 8.5, r'$K = \frac{1}{2}|V|$', ha='center', va='top',
         fontsize=14, color=IAM_RED)
ax1.text(7.5, 7.4, 'Decoherence channel', ha='center', va='top',
         fontsize=8.5, color=IAM_RED, style='italic')

# Arrow down right
ax1.annotate('', xy=(7.5, 5.8), xytext=(7.5, 6.8),
             arrowprops=dict(arrowstyle='->', color=IAM_RED, lw=2.0))

# Landauer cost box
rect_r = mpatches.FancyBboxPatch((5.4, 4.0), 4.2, 1.6,
    boxstyle="round,pad=0.15", facecolor='#fde8e8',
    edgecolor=IAM_RED, linewidth=1.5)
ax1.add_patch(rect_r)
ax1.text(7.5, 4.8, r'Landauer cost $k_BT\ln 2$', ha='center', va='center',
         fontsize=9, color=IAM_RED, fontweight='bold')
ax1.text(7.5, 4.25, '(written to local BH vault)', ha='center', va='center',
         fontsize=8, color=IAM_RED, style='italic')

# Arrow to "structure survives"
ax1.annotate('', xy=(7.5, 2.8), xytext=(7.5, 3.9),
             arrowprops=dict(arrowstyle='->', color=IAM_RED, lw=2.0))
ax1.text(7.5, 2.4, 'Structure survives', ha='center', va='center',
         fontsize=9, color=IAM_RED)

# Center label
ax1.text(5.0, 1.2, r'$2K + V = 0$  (virial theorem)',
         ha='center', va='center', fontsize=9.5, fontweight='bold',
         color='#333333',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                   edgecolor=IAM_GREY, linewidth=1.0))

# Top label — virial theorem
ax1.text(5.0, 10.0, 'Every 1/r potential, every scale',
         ha='center', va='top', fontsize=8, color=IAM_GREY, style='italic')

ax1.set_title('(a)  The Virial Partition', fontweight='bold', pad=6)

# ═════════════════════════════════════════════════════════════════════════════
# Panel (b): BH as mandatory vault — M_min dispersal condition
# ═════════════════════════════════════════════════════════════════════════════
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_facecolor(PANEL_BG)

# Three columns: no BH, M_min threshold, BH present
cols = [2.0, 5.0, 8.0]
colors = [IAM_RED, IAM_ORANGE, IAM_GREEN]
labels = ['$\\sigma < \\sigma_\\mathrm{crit}$\nNo BH forms',
          '$\\sigma \\approx \\sigma_\\mathrm{crit}$\nThreshold',
          '$\\sigma > \\sigma_\\mathrm{crit}$\nBH forms']
outcomes = ['Information half\nhas no local vault',
            'Caught between\ntwo regimes',
            'BH vault accepts\ninformation half']
results = ['HALO\nDISPERSES', 'UNSTABLE', 'HALO\nSURVIVES']
result_colors = [IAM_RED, IAM_ORANGE, IAM_GREEN]

for i, (x, col, lbl, out, res) in enumerate(zip(
        cols, colors, labels, outcomes, results)):
    # Header box
    hbox = mpatches.FancyBboxPatch((x-1.35, 8.1), 2.7, 1.7,
        boxstyle="round,pad=0.15", facecolor=col, alpha=0.9,
        edgecolor=col, linewidth=1.5)
    ax2.add_patch(hbox)
    ax2.text(x, 8.95, lbl, ha='center', va='center',
             fontsize=7.5, fontweight='bold', color='white',
             linespacing=1.5)

    # Arrow
    ax2.annotate('', xy=(x, 6.95), xytext=(x, 8.05),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.8))

    # Outcome box — white background, dark text, larger
    obox = mpatches.FancyBboxPatch((x-1.35, 5.5), 2.7, 1.35,
        boxstyle="round,pad=0.15", facecolor='white',
        edgecolor=col, linewidth=1.8)
    ax2.add_patch(obox)
    ax2.text(x, 6.17, out, ha='center', va='center',
             fontsize=7.5, color='#111111', fontweight='bold',
             linespacing=1.5)

    # Arrow
    ax2.annotate('', xy=(x, 4.4), xytext=(x, 5.45),
                 arrowprops=dict(arrowstyle='->', color=col, lw=1.8))

    # Result box
    rbox = mpatches.FancyBboxPatch((x-1.35, 3.15), 2.7, 1.15,
        boxstyle="round,pad=0.15", facecolor=col, alpha=0.9,
        edgecolor=col, linewidth=1.5)
    ax2.add_patch(rbox)
    ax2.text(x, 3.72, res, ha='center', va='center',
             fontsize=8.5, fontweight='bold', color='white',
             linespacing=1.3)

# sigma_crit annotation
ax2.annotate('', xy=(5.0, 1.8), xytext=(5.0, 3.0),
             arrowprops=dict(arrowstyle='->', color=IAM_ORANGE, lw=1.5,
                             linestyle='dashed'))
ax2.text(5.0, 1.4, r'$\sigma_\mathrm{crit} \approx 4$ km s$^{-1}$',
         ha='center', va='center', fontsize=9, color=IAM_ORANGE,
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff8e8',
                   edgecolor=IAM_ORANGE, linewidth=1.0))

ax2.text(5.0, 0.4, r'$M_\mathrm{min} = 4\Omega_m\sigma^3/GH$  (scaling derived)',
         ha='center', va='center', fontsize=8.5, color='#333333',
         style='italic')

ax2.set_title('(b)  Black Hole as Mandatory Decoherence Vault',
              fontweight='bold', pad=6)

# ═════════════════════════════════════════════════════════════════════════════
# Panel (c): E(a) — The accumulated ledger / arrow of time
# ═════════════════════════════════════════════════════════════════════════════
a_arr = np.linspace(0.01, 4.0, 1000)
Ea = E(a_arr)

ax3.plot(a_arr, Ea, color=IAM_RED, linewidth=2.5, label=r'$\mathcal{E}(a) = \exp(1-1/a)$')

# Asymptote
ax3.axhline(np.e, color=IAM_RED, linewidth=1.0, linestyle=':', alpha=0.6)
ax3.text(3.6, np.e + 0.04, r'$e$ (never reached)', fontsize=8,
         color=IAM_RED, va='bottom')

# Today marker
ax3.axvline(1.0, color=IAM_GREY, linewidth=1.0, linestyle='--', alpha=0.7)
ax3.text(1.02, 0.15, 'Today\n$a=1$', fontsize=8, color=IAM_GREY, va='bottom')

# Electroweak transition
ax3.axvline(0.05, color=IAM_BLUE, linewidth=1.0, linestyle='--', alpha=0.7)
ax3.text(0.07, 0.9, '$\mathcal{E}\\to 0$\nas $a\\to 0$', fontsize=7.5,
         color=IAM_BLUE, va='center')

# Shading under curve — accumulated duration
ax3.fill_between(a_arr[a_arr <= 1.0], 0, Ea[a_arr <= 1.0],
                 alpha=0.12, color=IAM_RED, label='Accumulated activation')

# Arrow showing direction
ax3.annotate('', xy=(2.5, 2.2), xytext=(1.5, 1.6),
             arrowprops=dict(arrowstyle='->', color=IAM_RED, lw=1.5))
ax3.text(2.0, 1.85, 'Monotonically\nincreasing', fontsize=8,
         color=IAM_RED, ha='center', style='italic')

ax3.set_xlabel('Scale factor $a$', fontsize=10)
ax3.set_ylabel(r'Activation $\mathcal{E}(a)$', fontsize=10)
ax3.set_xlim(0, 4.0)
ax3.set_ylim(0, 3.0)
ax3.set_title('(c)  Activation Function: The Accumulated Record',
              fontweight='bold', pad=6)
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.2)

# ═════════════════════════════════════════════════════════════════════════════
# Panel (d): mu(a) — observational fingerprint
# ═════════════════════════════════════════════════════════════════════════════
z_arr = np.linspace(0, 3.0, 500)
a_z = 1.0 / (1.0 + z_arr)
mu_z = mu(a_z)
suppression = (1.0 - mu_z) * 100

ax4.plot(z_arr, suppression, color=IAM_RED, linewidth=2.5,
         label=r'IAM: $\mu(a) < 1$')
ax4.axhline(0, color=IAM_GREY, linewidth=1.0, linestyle='--', alpha=0.7,
            label=r'$\Lambda$CDM: $\mu = 1$')

# Euclid sensitivity band
ax4.axhspan(13.6 - 4.0, 13.6 + 4.0, alpha=0.12, color=IAM_GREEN,
            label=r'Euclid DR1 sensitivity $\sigma(\mu_0)\approx 0.04$')

# Today marker
ax4.axvline(0, color=IAM_GREY, linewidth=0.8, linestyle=':', alpha=0.5)

# Peak suppression annotation
ax4.annotate('', xy=(0.0, 13.6), xytext=(0.5, 13.6),
             arrowprops=dict(arrowstyle='->', color=IAM_RED, lw=1.2))
ax4.text(0.55, 13.6, r'$\mu_0 - 1 = -0.136$', fontsize=8.5,
         color=IAM_RED, va='center')

# High-z recovery
ax4.text(2.2, 0.8, r'$\mathcal{E}(a) \to 0$', fontsize=8,
         color=IAM_GREY, ha='center')
ax4.text(2.2, 0.2, r'$\mu \to 1$ at high $z$', fontsize=8,
         color=IAM_GREY, ha='center', style='italic')

ax4.set_xlabel('Redshift $z$', fontsize=10)
ax4.set_ylabel(r'Growth suppression $(1-\mu)\times 100\%$', fontsize=10)
ax4.set_xlim(0, 3.0)
ax4.set_ylim(-1, 18)
ax4.set_title('(d)  Observational Consequence: Growth Suppression',
              fontweight='bold', pad=6)
ax4.legend(loc='upper right', fontsize=8)
ax4.grid(True, alpha=0.2)

# Euclid falsification note
ax4.text(1.5, -0.5, 'Euclid DR1 October 2026: 3.4$\\sigma$ sensitivity',
         ha='center', va='bottom', fontsize=7.5, color=IAM_GREEN, style='italic')

# ═════════════════════════════════════════════════════════════════════════════
# Overall title and caption
# ═════════════════════════════════════════════════════════════════════════════
fig.suptitle(
    'Gravitational Decoherence, the Virial Partition, and the Emergence of Classical Structure',
    fontsize=12, fontweight='bold', y=0.98
)

fig.text(0.5, 0.01,
    'Mahaffey (2026)  |  doi:10.5281/zenodo.18702042  |  '
    'github.com/hmahaffeyges/IAM-Validation  |  March 2026',
    ha='center', fontsize=7.5, style='italic', color='#666666')

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig('iam_smolin_figure.pdf', bbox_inches='tight', dpi=180)
plt.savefig('iam_smolin_figure.png', bbox_inches='tight', dpi=180)
plt.close()

print("Figure saved.")
print("Panels:")
print("  (a) Virial partition — potency/act split")
print("  (b) BH mandatory vault — dispersal condition")
print("  (c) E(a) — accumulated ledger / arrow of time")
print("  (d) mu(a) — observational fingerprint")
