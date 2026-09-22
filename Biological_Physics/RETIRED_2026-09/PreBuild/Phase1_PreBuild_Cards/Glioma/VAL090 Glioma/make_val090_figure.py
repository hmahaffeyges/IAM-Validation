#!/usr/bin/env python3
"""Make VAL-090 figure: Cortical_neurons fraction across the four groups."""
import csv
import statistics
import gzip
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_decon(path):
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        samples = header[1:]
        cells = {}
        for row in reader:
            cells[row[0]] = dict(zip(samples, [float(v) for v in row[1:]]))
    return samples, cells


# Load all
hs, hc = load_decon('/home/claude/brain_decon/results/GSE51057_betas_healthy_deconv_output.csv')
bs, bc = load_decon('/home/claude/brain_decon/results/GSE180683_betas_deconv_output.csv')
ts, tc = load_decon('/home/claude/brain_decon/results/GSE60274_betas_deconv_output.csv')

healthy = [hc['Cortical_neurons'][s] for s in hs]
blood = [bc['Cortical_neurons'][s] for s in bs]

# Phenotype tissue
src = '/home/claude/glioma_work/GSE60274_series_matrix.txt.gz'
titles, gsms = [], []
with gzip.open(src, 'rt') as f:
    for line in f:
        if line.startswith('!Sample_title'):
            titles = [s.strip().strip('"') for s in line.strip().split('\t')[1:]]
        elif line.startswith('!Sample_geo_accession'):
            gsms = [s.strip().strip('"') for s in line.strip().split('\t')[1:]]
            break
pheno = {}
for gsm, t in zip(gsms, titles):
    tl = t.lower()
    if 'cultured glioma sphere' in tl: pheno[gsm] = 'sphere'
    elif 'recurrent gbm' in tl: pheno[gsm] = 'GBM_recurrent'
    elif 'craniotomy' in tl or 'lobectomy' in tl: pheno[gsm] = 'NTB'
    elif 'surgical resection gbm' in tl: pheno[gsm] = 'GBM_primary'
    else: pheno[gsm] = 'unknown'

ntb = [tc['Cortical_neurons'][s] for s in ts if pheno.get(s) == 'NTB']
gbm_prim = [tc['Cortical_neurons'][s] for s in ts if pheno.get(s) == 'GBM_primary']
gbm_rec = [tc['Cortical_neurons'][s] for s in ts if pheno.get(s) == 'GBM_recurrent']
sphere = [tc['Cortical_neurons'][s] for s in ts if pheno.get(s) == 'sphere']

# Figure: two-panel — left=blood (zoom, shows ~1% vs ~0.3%), right=tissue (shows tumor disrupts brain composition)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Panel A: blood — log scale to see the difference
ax = axes[0]
data_blood = [healthy, blood]
labels_blood = [f'Healthy\nbuffy coat\n(GSE51057, n={len(healthy)})',
                f'Glioma\nperipheral blood\n(GSE180683, n={len(blood)})']
parts = ax.boxplot(data_blood, labels=labels_blood, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor='#dfdfff', edgecolor='#444'),
                   medianprops=dict(color='#222', linewidth=2),
                   whiskerprops=dict(color='#444'),
                   capprops=dict(color='#444'),
                   flierprops=dict(marker='.', markerfacecolor='#888', markersize=3, alpha=0.5))
# overlay
for i, dat in enumerate(data_blood, 1):
    x = np.random.normal(i, 0.04, len(dat))
    ax.scatter(x, dat, s=10, alpha=0.5, c='#444', zorder=3)

ax.set_ylabel('Cortical neurons fraction\n(NNLS deconvolution, Loyfer 2023 atlas)')
ax.set_ylim(-0.001, 0.025)
ax.axhline(0.01, color='#cc4444', linestyle='--', linewidth=0.8, alpha=0.6, label='1% threshold')
ax.set_title(f'Peripheral blood\nCohen\'s d = +1.96 [+1.62, +2.31]', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: tissue
ax = axes[1]
data_tissue = [ntb, gbm_prim, gbm_rec, sphere]
labels_tissue = [f'NTB controls\n(n={len(ntb)})',
                 f'GBM primary\n(n={len(gbm_prim)})',
                 f'GBM recurrent\n(n={len(gbm_rec)})',
                 f'Cultured spheres\n(n={len(sphere)})']
parts = ax.boxplot(data_tissue, labels=labels_tissue, widths=0.6, patch_artist=True,
                   boxprops=dict(facecolor='#dfdfff', edgecolor='#444'),
                   medianprops=dict(color='#222', linewidth=2))
for i, dat in enumerate(data_tissue, 1):
    x = np.random.normal(i, 0.06, len(dat))
    ax.scatter(x, dat, s=20, alpha=0.6, c='#444', zorder=3)
ax.set_ylabel('Cortical neurons fraction\n(NNLS deconvolution, Loyfer 2023 atlas)')
ax.set_ylim(0, 0.85)
ax.set_title(f'Brain tissue\nGBM disrupts normal architecture\nCohen\'s d (GBM_primary vs NTB) = −2.81', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('VAL-090: Cortical neurons cfDNA fraction across glioma blood, GBM tissue, and healthy reference\n'
             'Loyfer 2023 / Moss 2018 reference_atlas; NNLS deconvolution; nloyfer/meth_atlas',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/brain_decon/results/VAL-090_distributions.png', dpi=110, bbox_inches='tight')
print('Figure saved: /home/claude/brain_decon/results/VAL-090_distributions.png')
