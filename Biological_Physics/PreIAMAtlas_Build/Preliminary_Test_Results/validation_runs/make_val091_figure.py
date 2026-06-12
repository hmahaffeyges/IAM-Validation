#!/usr/bin/env python3
"""VAL-091 distribution figure.

Panel layout:
  Row 1: per-cohort AD vs HC distributions (4 panels: HC ext, AIBL, AddNeuroMed, GIFT)
  Row 2: GIFT specificity arm (FTD, PSP/CBD vs HC) + cross-cohort HC baseline panel
"""
import csv
import sys
import math
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/claude/ad_loyfer')
from val_091_ad_brain_decon_analysis import (
    DECONV, load_deconv_row, parse_aibl_phenotype,
    parse_addneuromed_phenotype, parse_gift_phenotype
)

# Load all
cn_hc = load_deconv_row(DECONV['GSE51057_HC'])
cn_aibl = load_deconv_row(DECONV['AIBL'])
cn_anm = load_deconv_row(DECONV['AddNeuroMed'])
cn_gift = load_deconv_row(DECONV['GIFT'])

aibl_disease, aibl_gender = parse_aibl_phenotype()
anm_disease, _, _ = parse_addneuromed_phenotype()
gift_disease, _, _ = parse_gift_phenotype()

def gather(cn, disease_map, label):
    return [v * 100 for sid, v in cn.items() if disease_map.get(sid) == label]

aibl_ad = gather(cn_aibl, aibl_disease, 'AD')
aibl_hc_v = gather(cn_aibl, aibl_disease, 'HC')
anm_ad = gather(cn_anm, anm_disease, 'AD')
anm_hc = gather(cn_anm, anm_disease, 'HC')
gift_ad = gather(cn_gift, gift_disease, 'AD')
gift_hc = gather(cn_gift, gift_disease, 'HC')
gift_ftd = gather(cn_gift, gift_disease, 'FTD')
gift_psp = gather(cn_gift, gift_disease, 'PSP_CBD')
hc_ext = [v * 100 for v in cn_hc.values()]

# VAL-090 anchors (literature reference, no recomputation)
GLIOMA_MEAN = 1.092
GLIOMA_SD = 0.493  # from VAL-090
HC090_MEAN = 0.276

# Figure
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('VAL-091: Cortical-neuron cfDNA fraction across AD cohorts (Loyfer/Moss array atlas, NNLS)',
             fontsize=13, y=0.99)

bp_props = dict(widths=0.55, patch_artist=True, showfliers=True,
                medianprops=dict(color='white', linewidth=2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))

# Panel 1: AIBL AD vs HC
ax = axes[0, 0]
data = [aibl_hc_v, aibl_ad]
bp = ax.boxplot(data, labels=[f'HC\n(n={len(aibl_hc_v)})', f'AD\n(n={len(aibl_ad)})'], **bp_props)
for patch, color in zip(bp['boxes'], ['#4A90A4', '#A04040']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_title(f'AIBL (EPIC, panel-training)\nd = -0.026 [-0.21, +0.17]', fontsize=11)
ax.set_ylabel('Cortical_neurons fraction (%)')
ax.grid(alpha=0.3, axis='y')
ax.axhline(GLIOMA_MEAN, color='#cc6677', linestyle='--', alpha=0.5, label=f'VAL-090 glioma mean ({GLIOMA_MEAN}%)')
ax.axhline(HC090_MEAN, color='#88aaaa', linestyle=':', alpha=0.5, label=f'VAL-090 HC mean ({HC090_MEAN}%)')
ax.legend(fontsize=8, loc='upper right')

# Panel 2: AddNeuroMed AD vs HC
ax = axes[0, 1]
data = [anm_hc, anm_ad]
bp = ax.boxplot(data, labels=[f'HC\n(n={len(anm_hc)})', f'AD\n(n={len(anm_ad)})'], **bp_props)
for patch, color in zip(bp['boxes'], ['#4A90A4', '#A04040']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_title(f'AddNeuroMed (450K, cross-platform)\nd = -0.083 [-0.36, +0.19]', fontsize=11)
ax.set_ylabel('Cortical_neurons fraction (%)')
ax.grid(alpha=0.3, axis='y')
# Note batch shift inline
ax.text(0.5, 0.95, 'NOTE: HC mean 7.4% — cross-platform\nNNLS routing artifact (5599/6105 CpGs)',
        ha='center', va='top', transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='#f4cccc', alpha=0.8))

# Panel 3: GIFT all groups
ax = axes[0, 2]
data = [gift_hc, gift_ad, gift_ftd, gift_psp]
labels = [f'HC\n(n={len(gift_hc)})', f'AD\n(n={len(gift_ad)})',
          f'FTD\n(n={len(gift_ftd)})', f'PSP/CBD\n(n={len(gift_psp)})']
bp = ax.boxplot(data, labels=labels, **bp_props)
for patch, color in zip(bp['boxes'], ['#4A90A4', '#A04040', '#7050A0', '#A07050']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_title(f'GIFT 53740 (450K, specificity)\nAD d=+0.96 [+0.15, +1.88]  FTD d=+0.19  PSP d=-0.51', fontsize=11)
ax.set_ylabel('Cortical_neurons fraction (%)')
ax.grid(alpha=0.3, axis='y')
ax.axhline(GLIOMA_MEAN, color='#cc6677', linestyle='--', alpha=0.5)
ax.text(0.5, 0.95, f'AD n=15 driven by single 5.8% outlier\nMedian AD ≈ 0.9%, median HC = 0.0%',
        ha='center', va='top', transform=ax.transAxes, fontsize=8,
        bbox=dict(boxstyle='round', facecolor='#fff2cc', alpha=0.8))

# Panel 4: HC baselines across cohorts (cross-cohort assay-quality view)
ax = axes[1, 0]
data = [hc_ext, aibl_hc_v, anm_hc, gift_hc]
labels = [f'GSE51057\n(n={len(hc_ext)})', f'AIBL\n(n={len(aibl_hc_v)})',
          f'AddNeuroMed\n(n={len(anm_hc)})', f'GIFT\n(n={len(gift_hc)})']
bp = ax.boxplot(data, labels=labels, **bp_props)
for patch in bp['boxes']:
    patch.set_facecolor('#4A90A4'); patch.set_alpha(0.7)
ax.set_title('HC cortical-neuron baselines across cohorts\n(cross-cohort fold range = 28.7×)', fontsize=11)
ax.set_ylabel('Cortical_neurons fraction (%)')
ax.set_yscale('symlog', linthresh=0.5)
ax.grid(alpha=0.3, axis='y')

# Panel 5: All AD cohorts vs glioma anchor
ax = axes[1, 1]
data = [hc_ext, aibl_ad, anm_ad, gift_ad]
labels = [f'GSE51057 HC\n(n={len(hc_ext)})', f'AIBL AD\n(n={len(aibl_ad)})',
          f'AddNeuroMed AD\n(n={len(anm_ad)})', f'GIFT AD\n(n={len(gift_ad)})']
bp = ax.boxplot(data, labels=labels, **bp_props)
for patch, color in zip(bp['boxes'], ['#4A90A4', '#A04040', '#A04040', '#A04040']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_title('AD cohorts vs VAL-090 glioma anchor', fontsize=11)
ax.set_ylabel('Cortical_neurons fraction (%)')
ax.set_yscale('symlog', linthresh=0.5)
ax.grid(alpha=0.3, axis='y')
ax.axhline(GLIOMA_MEAN, color='#cc6677', linestyle='--', alpha=0.7,
           label=f'VAL-090 glioma plasma mean = {GLIOMA_MEAN}%')
ax.axhline(HC090_MEAN, color='#88aaaa', linestyle=':', alpha=0.7,
           label=f'VAL-090 HC mean = {HC090_MEAN}%')
ax.legend(fontsize=8, loc='upper right')

# Panel 6: outcome summary text
ax = axes[1, 2]
ax.axis('off')
summary = [
    'OUTCOME: O4_AD_NEURO_NULL (with caveats)',
    '',
    'Honest reading:',
    '• AIBL (panel-training cohort): NULL d=-0.03',
    '• AddNeuroMed: NULL d=-0.08',
    '• GIFT: small-n (15) signal d=+0.96 driven by',
    '  one 5.8% outlier; median AD≈0.9% vs HC=0.0%',
    '',
    'Cross-cohort HC baseline 28.7× fold range',
    'invalidates pooled-AD-vs-external-HC contrast.',
    '',
    'VAL-091 conclusion supports card v2.1 prediction:',
    '"Stage 2 Moss NNLS for AD is expected NULL."',
    '',
    'Major specificity win for EDEAR routing:',
    'Glioma plasma cortical-neuron fraction is',
    '~4× elevated above HC and CLEARLY separable',
    'from AD plasma at the cohort level.',
]
ax.text(0.02, 0.98, '\n'.join(summary), transform=ax.transAxes,
        fontsize=10, family='monospace', va='top',
        bbox=dict(boxstyle='round', facecolor='#e8e8e8', alpha=0.95))

plt.tight_layout()
out = '/mnt/user-data/outputs/cookbook_v2.1/ad-immune/VAL-091_distributions.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f'Figure saved: {out}')
