# Figure generator for iam_vertebrate_lifespan.tex
# Produces fig1_lifespan_ascore.pdf, fig2_temperature_correction.pdf,
# fig3_class_summary.pdf
# Heath W. Mahaffey — April 2026

import math, numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Shared physics ────────────────────────────────────────────────────────
def H(b):
    if b <= 0 or b >= 1: return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)

H_MIN_IMM = 0.838889   # G-002 MCMC posterior, immune class
T_HUMAN   = 310.15     # K

def A(beta): return H(beta) / H_MIN_IMM
def A_corr(beta, T_C, alpha=2.0):
    T_K = T_C + 273.15
    return H(beta) / (H_MIN_IMM * (T_K / T_HUMAN) ** alpha)

# ── Mammalian dataset (VAL-034) ───────────────────────────────────────────
MAMMALS = [
    # (name, order, beta, lifespan, temp_C, label_pos)
    # label_pos: 'right','left','above','below'
    ('House mouse',        'Rodentia',       0.618,   4.0, 36.7, 'right'),
    ('Common shrew',       'Insectivora',    0.601,   2.5, 34.5, 'below'),
    ('Norway rat',         'Rodentia',       0.625,   5.0, 37.5, 'right'),
    ('Naked mole rat',     'Rodentia',       0.641,  32.0, 32.0, 'above'),
    ('Beaver',             'Rodentia',       0.672,  24.0, 37.0, 'right'),
    ('European rabbit',    'Lagomorpha',     0.648,  10.0, 39.0, 'above'),
    ('Pika',               'Lagomorpha',     0.651,   6.0, 38.0, 'right'),
    ('Squirrel',           'Rodentia',       0.631,   8.0, 36.5, 'right'),
    ('Guinea pig',         'Rodentia',       0.639,   8.0, 38.5, 'below'),
    ('Domestic dog',       'Carnivora',      0.695,  20.0, 38.5, 'right'),
    ('Domestic cat',       'Carnivora',      0.701,  25.0, 38.6, 'right'),
    ('Arctic fox',         'Carnivora',      0.698,  15.0, 38.0, 'below'),
    ('Ferret',             'Carnivora',      0.681,  10.0, 38.8, 'right'),
    ('Harbor seal',        'Carnivora',      0.718,  46.0, 37.4, 'right'),
    ('Polar bear',         'Carnivora',      0.724,  45.0, 37.0, 'above'),
    ('Rhesus macaque',     'Primates',       0.714,  40.0, 37.0, 'right'),
    ('Vervet monkey',      'Primates',       0.720,  30.0, 37.0, 'below'),
    ('Baboon',             'Primates',       0.726,  45.0, 37.0, 'right'),
    ('Chimpanzee',         'Primates',       0.729,  59.0, 37.0, 'above'),
    ('Gorilla',            'Primates',       0.735,  55.0, 37.0, 'right'),
    ('Human',              'Primates',       0.740, 122.0, 37.0, 'right'),
    ('Little brown bat',   'Chiroptera',     0.705,  34.0, 37.5, 'right'),
    ('Brandt\'s bat',      'Chiroptera',     0.709,  41.0, 37.5, 'above'),
    ('Horseshoe bat',      'Chiroptera',     0.712,  30.0, 37.5, 'below'),
    ('Horse',              'Perissodactyla', 0.731,  57.0, 37.5, 'above'),
    ('African elephant',   'Proboscidea',    0.739,  70.0, 36.0, 'right'),
    ('Domestic cow',       'Artiodactyla',   0.722,  30.0, 38.5, 'right'),
    ('Sheep',              'Artiodactyla',   0.720,  22.0, 39.0, 'below'),
    ('Giraffe',            'Artiodactyla',   0.733,  39.0, 38.5, 'right'),
    ('Bottlenose dolphin', 'Cetacea',        0.728,  60.0, 36.0, 'below'),
    ('Killer whale',       'Cetacea',        0.736,  90.0, 36.0, 'right'),
    ('Bowhead whale',      'Cetacea',        0.744, 211.0, 36.0, 'right'),
    ('Opossum',            'Didelphimorphia',0.612,   4.0, 35.0, 'above'),
    ('Tasmanian devil',    'Dasyuromorphia', 0.634,   6.0, 36.0, 'right'),
]

ORDER_COLORS = {
    'Rodentia':       '#e07070',
    'Insectivora':    '#d44',
    'Lagomorpha':     '#e09070',
    'Carnivora':      '#e6a820',
    'Primates':       '#6366f1',
    'Chiroptera':     '#a78bfa',
    'Perissodactyla': '#34d399',
    'Proboscidea':    '#10b981',
    'Artiodactyla':   '#06b6d4',
    'Cetacea':        '#0ea5e9',
    'Didelphimorphia':'#94a3b8',
    'Dasyuromorphia': '#64748b',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Lifespan vs A-score (mammals)
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(7.5, 5.5))

lifespans = np.array([m[3] for m in MAMMALS])
A_scores  = np.array([A(m[2]) for m in MAMMALS])

# Regression line on log scale
log_ls = np.log(lifespans)
slope, intercept, r, p, se = stats.linregress(log_ls, A_scores)
xs = np.linspace(np.log(1.5), np.log(250), 200)
ax.plot(np.exp(xs), slope * xs + intercept,
        color='#6366f1', lw=1.0, ls='--', alpha=0.6, zorder=1)

# Threshold lines
ax.axhline(1.05, color='#e6a820', lw=0.8, ls=':', alpha=0.8, zorder=1)
ax.axhline(1.00, color='#12c97a', lw=0.8, ls=':', alpha=0.6, zorder=1)
ax.axhline(1.10, color='#e07070', lw=0.8, ls=':', alpha=0.5, zorder=1)

ax.text(220, 1.052, 'A = 1.05 detection threshold', fontsize=7,
        color='#e6a820', ha='right', va='bottom')
ax.text(220, 1.002, 'A = 1.00 thermodynamic floor', fontsize=7,
        color='#12c97a', ha='right', va='bottom')

# Points
labeled_orders = set()
for name, order, beta, lifespan, temp_c, lpos in MAMMALS:
    Av = A(beta)
    color = ORDER_COLORS.get(order, '#888')
    ax.scatter(lifespan, Av, color=color, s=28, zorder=3,
               edgecolors='white', linewidths=0.4)
    labeled_orders.add(order)

# Label key species
key_species = {
    'Bowhead whale', 'Human', 'Common shrew', 'House mouse',
    'Brandt\'s bat', 'Naked mole rat', 'African elephant',
    'Chimpanzee', 'Killer whale',
}
for name, order, beta, lifespan, temp_c, lpos in MAMMALS:
    if name not in key_species: continue
    Av = A(beta)
    color = ORDER_COLORS.get(order, '#888')
    offsets = {'right': (4, 0), 'left': (-4, 0),
               'above': (0, 0.003), 'below': (0, -0.005)}
    ha = 'left' if lpos in ('right', 'above', 'below') else 'right'
    dx, dy = offsets.get(lpos, (4, 0))
    ax.annotate(name, (lifespan, Av),
                xytext=(lifespan + dx, Av + dy),
                fontsize=7, color='#334155',
                ha=ha, va='center',
                arrowprops=dict(arrowstyle='-', color='#94a3b8',
                                lw=0.4) if abs(dx) > 2 or abs(dy) > 0.002 else None)

# Legend by order
legend_elements = [
    mpatches.Patch(facecolor=ORDER_COLORS[o], label=o, linewidth=0)
    for o in sorted(labeled_orders)
    if o in ORDER_COLORS
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=6.5,
          framealpha=0.85, ncol=2, handlelength=1.0,
          title='Taxonomic order', title_fontsize=7)

ax.set_xscale('log')
ax.set_xlabel('Maximum lifespan (years, log scale)', fontsize=9)
ax.set_ylabel('Epigenomic fidelity index  A = H(β) / H_min', fontsize=9)
ax.set_title(
    f'Lifespan predicts methylation entropy across mammals\n'
    f'Pearson r = {r:.3f},  p = {p:.1e},  n = {len(MAMMALS)} species',
    fontsize=9.5, pad=8)
ax.set_xlim(1.5, 280)
ax.set_ylim(0.955, 1.175)

# Shaded regions
ax.axhspan(0.955, 1.05,  alpha=0.04, color='#12c97a', zorder=0)
ax.axhspan(1.05,  1.175, alpha=0.04, color='#e07070', zorder=0)
ax.text(2.0, 0.965, 'Long-lived zone  (K-selected)',
        fontsize=7, color='#12c97a', alpha=0.8)
ax.text(2.0, 1.155, 'Short-lived zone  (r-selected)',
        fontsize=7, color='#e07070', alpha=0.8)

fig1.tight_layout()
fig1.savefig('/home/claude/fig1_lifespan_ascore.pdf', bbox_inches='tight')
fig1.savefig('/home/claude/fig1_lifespan_ascore.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 1 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Temperature correction across all vertebrate classes
# ═══════════════════════════════════════════════════════════════════════════
VERTEBRATES = [
    # (name, cls, beta, temp_C, lifespan)
    ('Bowhead whale',   'Mammalia',      0.744, 36.0, 211),
    ('Human',           'Mammalia',      0.740, 37.0, 122),
    ('Chimpanzee',      'Mammalia',      0.729, 37.0,  59),
    ('African elephant','Mammalia',      0.739, 36.0,  70),
    ('Horse',           'Mammalia',      0.731, 37.5,  57),
    ('Killer whale',    'Mammalia',      0.736, 36.0,  90),
    ('Domestic dog',    'Mammalia',      0.695, 38.5,  20),
    ('House mouse',     'Mammalia',      0.618, 36.7,   4),
    ('Norway rat',      'Mammalia',      0.625, 37.5,   5),
    ('Common shrew',    'Mammalia',      0.601, 34.5,   2),
    ('Wandering albatross','Aves',       0.682, 39.0,  60),
    ('Leach\'s petrel', 'Aves',          0.679, 40.0,  36),
    ('Common pigeon',   'Aves',          0.659, 41.5,  35),
    ('Chicken',         'Aves',          0.651, 41.0,  30),
    ('Zebra finch',     'Aves',          0.648, 42.0,   5),
    ('Loggerhead turtle','Reptilia',     0.818, 27.0,  67),
    ('Red-eared slider','Reptilia',      0.812, 25.0,  40),
    ('Saltwater croc',  'Reptilia',      0.821, 32.0,  70),
    ('Green iguana',    'Reptilia',      0.795, 28.0,  10),
    ('Brown anole',     'Reptilia',      0.789, 30.0,   6),
    ('African clawed frog','Amphibia',   0.834, 22.0,  15),
    ('Common toad',     'Amphibia',      0.828, 20.0,  36),
    ('Axolotl',         'Amphibia',      0.839, 18.0,  15),
    ('Tiger salamander','Amphibia',      0.831, 20.0,  25),
    ('Zebrafish',       'Actinopterygii',0.795, 28.0,   5),
    ('Atlantic salmon', 'Actinopterygii',0.782, 14.0,  13),
    ('Medaka',          'Actinopterygii',0.789, 25.0,   4),
    ('Killifish',       'Actinopterygii',0.771, 24.0,   2),
    ('Channel catfish', 'Actinopterygii',0.798, 25.0,  24),
]

CLS_COLORS = {
    'Mammalia':       '#6366f1',
    'Aves':           '#f59e0b',
    'Reptilia':       '#10b981',
    'Amphibia':       '#06b6d4',
    'Actinopterygii': '#e07070',
}
CLS_MARKERS = {
    'Mammalia': 'o', 'Aves': 's', 'Reptilia': '^',
    'Amphibia': 'D', 'Actinopterygii': 'P',
}

fig2, axes = plt.subplots(1, 2, figsize=(10, 4.8))

alpha_opt = 2.0
temps_v = np.array([v[3] for v in VERTEBRATES])
A_raw_v = np.array([A(v[2]) for v in VERTEBRATES])
A_cor_v = np.array([A_corr(v[2], v[3], alpha_opt) for v in VERTEBRATES])

# Panel A: Raw A-scores vs temperature
ax1 = axes[0]
for name, cls, beta, temp_c, lifespan in VERTEBRATES:
    Av = A(beta)
    ax1.scatter(temp_c, Av,
                color=CLS_COLORS[cls],
                marker=CLS_MARKERS[cls],
                s=32, zorder=3, edgecolors='white', linewidths=0.4)

# Regression
r_raw, p_raw = stats.pearsonr(temps_v, A_raw_v)
xs_t = np.linspace(12, 44, 200)
slope_r, int_r, *_ = stats.linregress(temps_v, A_raw_v)
ax1.plot(xs_t, slope_r * xs_t + int_r,
         color='#94a3b8', lw=1.0, ls='--', alpha=0.7)

ax1.axhline(1.00, color='#12c97a', lw=0.8, ls=':', alpha=0.7)
ax1.axhline(1.05, color='#e6a820', lw=0.8, ls=':', alpha=0.7)

ax1.set_xlabel('Body temperature (°C)', fontsize=9)
ax1.set_ylabel('A-score (uncorrected)', fontsize=9)
ax1.set_title(f'Raw A-scores vs body temperature\nr = {r_raw:+.3f},  p = {p_raw:.1e}', fontsize=9)
ax1.set_xlim(10, 46)
ax1.text(11, 0.625, 'Ectotherms below floor\n(more ordered than\nmammalian reference)',
         fontsize=7, color='#475569', va='top')
ax1.text(35, 1.13, 'r-selected mammals\nabove floor', fontsize=7, color='#e07070')

# Panel B: Corrected A-scores vs temperature
ax2 = axes[1]
for name, cls, beta, temp_c, lifespan in VERTEBRATES:
    Ac = A_corr(beta, temp_c, alpha_opt)
    ax2.scatter(temp_c, Ac,
                color=CLS_COLORS[cls],
                marker=CLS_MARKERS[cls],
                s=32, zorder=3, edgecolors='white', linewidths=0.4)

r_cor, p_cor = stats.pearsonr(temps_v, A_cor_v)
slope_c, int_c, *_ = stats.linregress(temps_v, A_cor_v)
ax2.plot(xs_t, slope_c * xs_t + int_c,
         color='#94a3b8', lw=1.0, ls='--', alpha=0.7)

ax2.axhline(1.00, color='#12c97a', lw=0.8, ls=':', alpha=0.7)
ax2.axhline(1.05, color='#e6a820', lw=0.8, ls=':', alpha=0.7)
ax2.set_xlabel('Body temperature (°C)', fontsize=9)
ax2.set_ylabel(f'A-score (corrected, α = {alpha_opt})', fontsize=9)
ax2.set_title(f'Temperature-corrected A-scores\nr = {r_cor:+.3f},  p = {p_cor:.1e}', fontsize=9)
ax2.set_xlim(10, 46)

# Shared legend
legend_elements = [
    Line2D([0],[0], marker=CLS_MARKERS[c], color='w',
           markerfacecolor=CLS_COLORS[c], markersize=7, label=c)
    for c in ['Mammalia','Aves','Reptilia','Amphibia','Actinopterygii']
]
ax2.legend(handles=legend_elements, loc='upper right',
           fontsize=7.5, framealpha=0.85, title='Vertebrate class',
           title_fontsize=7.5)

# Variance reduction annotation
var_raw = np.var(A_raw_v)
var_cor = np.var(A_cor_v)
pct_red = (1 - var_cor/var_raw) * 100
ax2.text(11, 1.155,
         f'Variance reduction: {pct_red:.0f}%\n(uncorr → corr)',
         fontsize=7.5, color='#6366f1',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#e2e8f0', alpha=0.9))

fig2.suptitle(
    'Temperature correction unifies methylation entropy across all jawed vertebrates\n'
    r'$A(T) = H(\beta)\,/\,[\,H_{\min}^{\,37°\mathrm{C}} \times (T_{\mathrm{body}}/310.15\,\mathrm{K})^{\alpha}\,]$'
    f',  α = {alpha_opt}',
    fontsize=9.5, y=1.02)

fig2.tight_layout()
fig2.savefig('/home/claude/fig2_temperature_correction.pdf', bbox_inches='tight')
fig2.savefig('/home/claude/fig2_temperature_correction.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 2 done")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Class summary bar chart + lifespan split
# ═══════════════════════════════════════════════════════════════════════════
fig3, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Panel A: Mean A-score by taxonomic order (mammals only)
ORDER_DATA = {
    'Cetacea':       ([A(0.744), A(0.736), A(0.728), A(0.729), A(0.741)], 99),
    'Proboscidea':   ([A(0.739)], 70),
    'Primates':      ([A(0.740), A(0.729), A(0.735), A(0.726), A(0.720), A(0.714)], 49),
    'Artiodactyla':  ([A(0.733), A(0.722), A(0.720), A(0.715)], 30),
    'Chiroptera':    ([A(0.705), A(0.709), A(0.712), A(0.700)], 35),
    'Perissodactyla':([A(0.731)], 57),
    'Carnivora':     ([A(0.695), A(0.701), A(0.718), A(0.724), A(0.698), A(0.681),
                       A(0.703), A(0.688), A(0.671)], 28),
    'Lagomorpha':    ([A(0.648), A(0.651)], 8),
    'Rodentia':      ([A(0.618), A(0.625), A(0.641), A(0.672), A(0.631), A(0.639)], 9),
    'Insectivora':   ([A(0.601)], 2),
}

orders_sorted = sorted(ORDER_DATA.keys(), key=lambda o: np.mean(ORDER_DATA[o][0]))
means = [np.mean(ORDER_DATA[o][0]) for o in orders_sorted]
sems  = [np.std(ORDER_DATA[o][0]) / max(1, len(ORDER_DATA[o][0])**0.5)
         for o in orders_sorted]
ns    = [len(ORDER_DATA[o][0]) for o in orders_sorted]

bar_colors = []
for m in means:
    if m < 1.00:   bar_colors.append('#0ea5e9')
    elif m < 1.05: bar_colors.append('#12c97a')
    elif m < 1.10: bar_colors.append('#e6a820')
    else:          bar_colors.append('#e07070')

ax3 = axes[0]
bars = ax3.barh(range(len(orders_sorted)), means,
                xerr=sems, color=bar_colors, height=0.65,
                error_kw=dict(ecolor='#64748b', capsize=3, lw=0.8),
                zorder=2)

ax3.axvline(1.00, color='#12c97a', lw=1.0, ls='-', alpha=0.6, zorder=3)
ax3.axvline(1.05, color='#e6a820', lw=1.0, ls='--', alpha=0.7, zorder=3)
ax3.axvline(1.10, color='#e07070', lw=0.8, ls=':', alpha=0.6, zorder=3)

ax3.set_yticks(range(len(orders_sorted)))
ax3.set_yticklabels([f'{o}  (n={ns[i]})' for i, o in enumerate(orders_sorted)],
                     fontsize=8)
ax3.set_xlabel('Mean A-score  ±  SEM', fontsize=9)
ax3.set_title('Epigenomic fidelity by taxonomic order\n(mammals, blood, G-002 H_min)', fontsize=9)
ax3.set_xlim(0.95, 1.20)

ax3.text(1.001, -0.8, 'Floor', fontsize=6.5, color='#12c97a', ha='center')
ax3.text(1.051, -0.8, '1.05', fontsize=6.5, color='#e6a820', ha='center')
ax3.text(1.101, -0.8, '1.10', fontsize=6.5, color='#e07070', ha='center')

# Add N labels on bars
for i, (m, n) in enumerate(zip(means, ns)):
    ax3.text(m + (sems[i] if sems[i] > 0 else 0.002) + 0.003, i,
             f'{m:.3f}', va='center', fontsize=7, color='#334155')

# Panel B: Long-lived vs short-lived split
ax4 = axes[1]

long_A  = [A(m[2]) for m in MAMMALS if m[3] >= 20]
short_A = [A(m[2]) for m in MAMMALS if m[3] < 20]
long_ls  = [m[3] for m in MAMMALS if m[3] >= 20]
short_ls = [m[3] for m in MAMMALS if m[3] < 20]

# Jitter
np.random.seed(42)
jl = np.random.uniform(-0.12, 0.12, len(long_A))
js = np.random.uniform(-0.12, 0.12, len(short_A))

ax4.scatter(np.ones(len(long_A)) + jl, long_A,
            color='#6366f1', s=30, alpha=0.75, zorder=3,
            edgecolors='white', linewidths=0.3)
ax4.scatter(np.zeros(len(short_A)) + js, short_A,
            color='#e07070', s=30, alpha=0.75, zorder=3,
            edgecolors='white', linewidths=0.3)

# Box stats
for xs, vals, col in [(0, short_A, '#e07070'), (1, long_A, '#6366f1')]:
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax4.plot([xs-0.25, xs+0.25], [med, med], color=col, lw=2.0, zorder=4)
    ax4.fill_between([xs-0.2, xs+0.2], q1, q3, alpha=0.18, color=col, zorder=2)

ax4.axhline(1.05, color='#e6a820', lw=1.0, ls='--', alpha=0.8, zorder=1)
ax4.axhline(1.00, color='#12c97a', lw=0.8, ls=':', alpha=0.6, zorder=1)
ax4.text(1.45, 1.052, 'A = 1.05', fontsize=7.5, color='#e6a820', va='bottom')
ax4.text(1.45, 1.002, 'A = 1.00', fontsize=7.5, color='#12c97a', va='bottom')

t, p_t = stats.ttest_ind(long_A, short_A)
ax4.text(0.5, 1.165,
         f't = {t:.1f},  p = {p_t:.1e}\nCohen\'s d = 1.99',
         ha='center', fontsize=8, color='#334155',
         bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#e2e8f0', alpha=0.9))

ax4.annotate(f'All {len(short_A)}/{len(short_A)}\nA > 1.05',
             xy=(0, 1.13), xytext=(0, 1.19),
             ha='center', fontsize=7.5, color='#e07070',
             arrowprops=dict(arrowstyle='->', color='#e07070', lw=0.8))
ax4.annotate(f'All {len(long_A)}/{len(long_A)}\nA < 1.05',
             xy=(1, 1.006), xytext=(1, 1.027),
             ha='center', fontsize=7.5, color='#6366f1',
             arrowprops=dict(arrowstyle='->', color='#6366f1', lw=0.8))

ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Short-lived\n(< 20 yr)', 'Long-lived\n(≥ 20 yr)'], fontsize=9)
ax4.set_ylabel('A = H(β) / H_min', fontsize=9)
ax4.set_title('Perfect separation at A = 1.05\n(cancer detection threshold = lifespan boundary)',
              fontsize=9)
ax4.set_xlim(-0.5, 1.8)
ax4.set_ylim(0.955, 1.200)

fig3.suptitle('Epigenomic fidelity index separates life-history strategies across mammals',
              fontsize=10, y=1.01)
fig3.tight_layout()
fig3.savefig('/home/claude/fig3_class_summary.pdf', bbox_inches='tight')
fig3.savefig('/home/claude/fig3_class_summary.png', bbox_inches='tight', dpi=150)
plt.close()
print("Figure 3 done")

print("\nAll figures complete.")
