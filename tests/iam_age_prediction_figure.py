"""
IAM Actualization Age Prediction Figure
----------------------------------------
Timestamped prediction figure for GitHub repository.
No paper — this is a dated prediction record.

Observed data sources (all cited in figure):
- Valcin et al. 2025, JCAP (arXiv:2503.19481):
  Oldest GCs: t_GC = 13.39 +/- 0.10 (stat.) +/- 0.23 (sys.) Gyr
  Formation redshift estimated z >= 3 from their analysis
  Universe age implied: t_U = 13.6 +/- 0.25 Gyr

- Valcin et al. 2021, JCAP 2021, 017 (arXiv:2102.04486):
  t_GC = 13.32 +/- 0.10 (stat.) +/- 0.47 (sys.) Gyr

- Gratton et al. 2003, A&A 408, 529:
  NGC 6397: 13.5 +/- 1.1 Gyr (z_form >= 2.5 per their analysis)
  NGC 6752: 13.4 +/- 1.1 Gyr (z_form >= 2.5 per their analysis)

- Curtis-Lake et al. 2023, Nature Astronomy (JADES):
  Spectroscopically confirmed galaxies at z = 10.38, 11.58, 12.63, 13.20
  These are existence confirmations, not age measurements.
  Plotted as formation redshift markers showing where IAM predicts
  excess — NOT as age data points (ages not precisely measured).

IAM prediction:
  t_act = t_coord * (f_act / f_coord)
  where f_act = [E(1) - E(a_form)] / E(1)
        f_coord = t_since_form / t_today
  Zero free parameters. Inputs: E(a) and LCDM Friedmann equation only.

Generated: March 16, 2026
Repository: https://github.com/hmahaffeyges/IAM-Validation
Zenodo DOI: 10.5281/zenodo.18702042
"""

import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.3,
    'lines.linewidth': 2.5,
    'figure.dpi': 150,
})

C_IAM   = '#2166AC'
C_GC    = '#D73027'
C_JWST  = '#762A83'
C_SHADE = '#FEE090'
C_ZERO  = '#4DAC26'

# ── Constants ─────────────────────────────────────────────────────────────────
H0_si   = 67.4e3 / 3.086e22
Omega_m = 0.3153
Omega_L = 0.6847
Omega_r = 9e-5
Gyr     = 3.156e16

def E_IAM(a):
    return np.exp(1.0 - 1.0/a)

def H_over_H0(a):
    return np.sqrt(Omega_r/a**4 + Omega_m/a**3 + Omega_L)

def dt_da(a):
    return 1.0 / (a * H_over_H0(a) * H0_si)

t_today, _ = quad(dt_da, 1e-5, 1.0, limit=500)
t_today_Gyr = t_today / Gyr
E_today = E_IAM(1.0)

# ── IAM prediction curve ──────────────────────────────────────────────────────
z_dense = np.linspace(1.0, 15.0, 300)
excess_dense = []

for z in z_dense:
    a = 1.0/(1.0+z)
    t_form, _ = quad(dt_da, 1e-5, a, limit=300)
    t_form_Gyr = t_form / Gyr
    t_since = t_today_Gyr - t_form_Gyr
    E_form = E_IAM(a)
    f_act   = (E_today - E_form) / E_today
    f_coord = t_since / t_today_Gyr
    excess_dense.append(t_since * (f_act/f_coord - 1.0))

excess_dense = np.array(excess_dense)

# ── Observed data points (fully cited) ───────────────────────────────────────
# Format: (label, z_form_estimate, apparent_excess_Gyr, err_Gyr, citation_key)
# apparent_excess = measured_age - LCDM_coordinate_age_since_formation
# For GCs: measured age - (13.787 - t_form(z_form))

def coord_age_since(z_form):
    a = 1.0/(1.0+z_form)
    t_form, _ = quad(dt_da, 1e-5, a, limit=300)
    return t_today_Gyr - t_form/Gyr

# Valcin et al. 2025: t_GC = 13.39 +/- 0.10 +/- 0.23 Gyr
# Formation redshift: Valcin et al. note oldest GCs formed within first 1-2 Gyr
# which corresponds to z >= 3-5. Use z=3.5 as representative midpoint,
# with horizontal error bar spanning z=3 to z=5.
# Apparent excess = 13.39 - coord_age_since(3.5)
z_v25 = 3.5
t_coord_v25 = coord_age_since(z_v25)
excess_v25 = 13.39 - t_coord_v25
err_v25_stat = 0.10
err_v25_sys  = 0.23
err_v25_tot  = np.sqrt(err_v25_stat**2 + err_v25_sys**2)  # combined in quadrature

# Valcin et al. 2021: t_GC = 13.32 +/- 0.10 +/- 0.47 Gyr
z_v21 = 3.5
t_coord_v21 = coord_age_since(z_v21)
excess_v21 = 13.32 - t_coord_v21
err_v21_tot = np.sqrt(0.10**2 + 0.47**2)

# Gratton et al. 2003 NGC 6397: 13.5 +/- 1.1 Gyr, z_form >= 2.5
z_n6397 = 2.8  # representative for z >= 2.5
excess_n6397 = 13.5 - coord_age_since(z_n6397)
err_n6397 = 1.1

# Gratton et al. 2003 NGC 6752: 13.4 +/- 1.1 Gyr, z_form >= 2.5
z_n6752 = 2.8
excess_n6752 = 13.4 - coord_age_since(z_n6752)
err_n6752 = 1.1

print("Computed apparent excesses:")
print(f"  Valcin+2025 (z~3.5): excess = {excess_v25:.3f} +/- {err_v25_tot:.3f} Gyr")
print(f"  Valcin+2021 (z~3.5): excess = {excess_v21:.3f} +/- {err_v21_tot:.3f} Gyr")
print(f"  NGC 6397 (z~2.8):    excess = {excess_n6397:.3f} +/- {err_n6397:.3f} Gyr")
print(f"  NGC 6752 (z~2.8):    excess = {excess_n6752:.3f} +/- {err_n6752:.3f} Gyr")
print()

# IAM prediction at those redshifts
for z_obs, label in [(3.5, "z=3.5"), (2.8, "z=2.8")]:
    a = 1.0/(1.0+z_obs)
    t_form, _ = quad(dt_da, 1e-5, a, limit=300)
    t_since = t_today_Gyr - t_form/Gyr
    f_act = (E_today - E_IAM(a))/E_today
    f_coord = t_since/t_today_Gyr
    pred = t_since*(f_act/f_coord - 1.0)
    print(f"IAM prediction at {label}: {pred:.3f} Gyr")

# JWST spectroscopic confirmations (Curtis-Lake et al. 2023, Nature Astronomy)
# These are NOT age measurements -- they are formation epoch markers.
# Plotted as vertical lines showing where IAM predicts excess,
# labeled as "spectroscopically confirmed" to show the epoch is real.
jwst_z = [10.38, 11.58, 12.63, 13.20]
jwst_labels = ['z=10.38', 'z=11.58', 'z=12.63', 'z=13.20']

# ── Build figure ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7.5))

# IAM prediction curve
ax.plot(z_dense, excess_dense, color=C_IAM, linewidth=3.0, zorder=5,
        label=r'IAM prediction: $\Delta t_\mathrm{act}(z_\mathrm{form})$'
              '\n(zero free parameters; $E(a)$ + Friedmann eq. only)')

# Shaded target range
ax.fill_between([1, 15], [1.0, 1.0], [2.0, 2.0],
                alpha=0.12, color=C_SHADE, zorder=1,
                label='Observed tension range cited in literature (1--2 Gyr)')

# Zero line
ax.axhline(y=0, color='gray', linewidth=1.0, linestyle='--', alpha=0.5, zorder=2)

# JWST formation epoch markers (vertical lines, not data points)
for z_j, lbl in zip(jwst_z, jwst_labels):
    ax.axvline(x=z_j, color=C_JWST, linewidth=1.2, linestyle=':', alpha=0.6, zorder=3)

# JWST label block -- bottom center, clear of info box
ax.text(9.0, -0.12,
        'JWST spectroscopic confirmations (Curtis-Lake et al. 2023, Nat. Astron.)\n'
        'Vertical lines = confirmed formation epochs'
        ' (stellar population ages not yet precisely measured)',
        fontsize=7.5, color=C_JWST, ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  alpha=0.85, edgecolor=C_JWST, linewidth=0.8))

# GC data points
gc_data = [
    (z_n6397, excess_n6397, err_n6397,
     'NGC 6397 (Gratton et al. 2003, A&A 408, 529)\n'
     r'$13.5 \pm 1.1$ Gyr; $z_\mathrm{form} \geq 2.5$'),
    (z_n6752, excess_n6752+0.05, err_n6752,
     'NGC 6752 (Gratton et al. 2003, A&A 408, 529)\n'
     r'$13.4 \pm 1.1$ Gyr; $z_\mathrm{form} \geq 2.5$'),
]

for z_obs, exc, err, lbl in gc_data:
    ax.errorbar(z_obs, exc, yerr=err,
                fmt='s', color=C_GC, markersize=8,
                capsize=5, capthick=1.8, elinewidth=1.5,
                zorder=6, label=lbl)

# Valcin et al. data points
valcin_data = [
    (z_v25+0.2, excess_v25, err_v25_tot,
     r'Valcin et al. 2025, JCAP (arXiv:2503.19481)'
     '\n'
     r'Oldest GCs: $13.39 \pm 0.10_\mathrm{stat} \pm 0.23_\mathrm{sys}$ Gyr'),
    (z_v21-0.2, excess_v21, err_v21_tot,
     r'Valcin et al. 2021, JCAP 2021, 017 (arXiv:2102.04486)'
     '\n'
     r'Oldest GCs: $13.32 \pm 0.10_\mathrm{stat} \pm 0.47_\mathrm{sys}$ Gyr'),
]

valcin_colors = ['#E66101', '#FDB863']
valcin_markers = ['D', 'D']
for (z_obs, exc, err, lbl), col, mk in zip(valcin_data,
                                             valcin_colors,
                                             valcin_markers):
    ax.errorbar(z_obs, exc, yerr=err,
                fmt=mk, color=col, markersize=9,
                capsize=5, capthick=1.8, elinewidth=1.5,
                zorder=6, label=lbl)

# Horizontal error bar for Valcin points showing z_form uncertainty
ax.annotate('', xy=(5.0, excess_v25), xytext=(3.0, excess_v25),
            arrowprops=dict(arrowstyle='<->', color='gray',
                           lw=1.2, mutation_scale=10))
ax.text(4.0, excess_v25+0.07, r'$z_\mathrm{form}$ range',
        fontsize=7.5, ha='center', color='gray')

# Key prediction annotations on curve
for z_mark, color_m in [(3, C_IAM), (5, C_IAM)]:
    idx = np.argmin(np.abs(z_dense - z_mark))
    pred_val = excess_dense[idx]
    ax.plot(z_dense[idx], pred_val, 'o', color=C_IAM,
            markersize=8, zorder=7, markerfacecolor='white',
            markeredgewidth=2)
    ax.annotate(f'IAM: +{pred_val:.2f} Gyr\nat z={z_mark}',
                xy=(z_dense[idx], pred_val),
                xytext=(z_dense[idx]+0.6, pred_val+0.18),
                fontsize=8.5, color=C_IAM, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C_IAM, lw=1.2))

# Title and labels
ax.set_xlabel('Estimated formation redshift $z_\\mathrm{form}$', fontsize=13)
ax.set_ylabel('Apparent age excess over $\\Lambda$CDM coordinate time (Gyr)',
              fontsize=12)
ax.set_title(
    'IAM Prediction: Apparent Stellar Age Excess as a Function of Formation Redshift\n'
    r'Stellar evolution runs on actualization clock $E(a)$; $\Lambda$CDM measures '
    r'coordinate time $t(a)$ — two distinct temporal rulers',
    fontsize=11.5, pad=10)

# Timestamp and info box
timestamp = datetime.now().strftime("%B %d, %Y")
info_text = (
    f'Prediction generated: {timestamp}\n'
    'IAM framework: Mahaffey (2026)\n'
    'doi:10.5281/zenodo.18702042\n'
    'github.com/hmahaffeyges/IAM-Validation\n\n'
    r'$E(a) = \exp(1 - 1/a)$; $\beta_m = \Omega_m/2$' + '\n'
    r'Planck 2018: $H_0=67.4$, $\Omega_m=0.3153$' + '\n'
    'Zero free parameters'
)
ax.text(0.985, 0.97, info_text,
        transform=ax.transAxes, fontsize=7.5,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#EFF3FF',
                  alpha=0.9, edgecolor=C_IAM, linewidth=0.8))

# Grid and limits
ax.grid(True, alpha=0.25, linestyle=':')
ax.set_xlim(1.5, 14.5)
ax.set_ylim(-0.4, 2.6)

# Legend
ax.legend(loc='upper left', fontsize=8, framealpha=0.92,
          edgecolor='gray', ncol=1)

plt.tight_layout()

outpath_pdf = '/mnt/user-data/outputs/IAM_age_prediction_timestamped_Mar2026.pdf'
outpath_png = '/mnt/user-data/outputs/IAM_age_prediction_timestamped_Mar2026.png'
plt.savefig(outpath_pdf, bbox_inches='tight')
plt.savefig(outpath_png, bbox_inches='tight', dpi=180)
plt.close()

print(f"\nFigure saved:")
print(f"  {outpath_pdf}")
print(f"  {outpath_png}")
print()
print("Citations in figure:")
print("  - Valcin et al. 2025, JCAP, arXiv:2503.19481")
print("  - Valcin et al. 2021, JCAP 2021, 017, arXiv:2102.04486")
print("  - Gratton et al. 2003, A&A 408, 529")
print("  - Curtis-Lake et al. 2023, Nature Astronomy (JADES)")
print("  - Planck Collaboration 2020, A&A 641, A6")
print()
print(f"Timestamp: {timestamp}")
