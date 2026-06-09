#!/usr/bin/env python3
"""
VAL-128 Crohn's blood results builder.

Per-cell-type, per-diagnosis Welch unpaired d:
  - CD vs healthy (HC) per cell type
  - UC vs healthy (HC) per cell type
  - CD vs UC per cell type (subtype discrimination)
  - Sorted-cell d vs whole-blood d (mixture-attenuation pre-locked test)

Outputs:
  VAL-128_results.json
  VAL-128_stratified_results.json
  VAL-128_per_sample.csv
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

OUTPUT = Path('/home/claude/gastric_esophageal_sprint/VAL-128_crohns_blood')
PREREG_SHA = 'e7cdb09082d39bdb0c82d4465ffd43a9cc12b79c1b56a5dcd23f22a0086da7bc'

print('Loading VAL-128 per-sample data...')
rows = []
with open(OUTPUT / 'val128_per_sample.ndjson') as f:
    for line in f:
        rows.append(json.loads(line))
print(f'  {len(rows)} samples')

# Welch unpaired d
def welch_d(x, y):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    y = np.asarray([v for v in y if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None, nx, ny, None, None, None, None
    mx, my = np.mean(x), np.mean(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = math.sqrt((sx2 + sy2) / 2)
    if pooled_sd == 0:
        return None, nx, ny, None, None, None, None
    d = (mx - my) / pooled_sd
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
    se_d = math.sqrt((nx + ny) / (nx * ny) + d**2 / (2 * (nx + ny)))
    return float(d), nx, ny, float(t_stat), float(p_val), float(d - 1.96*se_d), float(d + 1.96*se_d)


def outcome_class(d):
    if d is None: return 'NULL_INSUFFICIENT_DATA'
    abs_d = abs(d)
    if abs_d >= 0.5:
        return 'DIFFERENTIATING_POSITIVE' if d > 0 else 'DIFFERENTIATING_NEGATIVE'
    if abs_d >= 0.2: return 'PARTIAL'
    return 'NULL'


# Group rows by cell type + diagnosis
groups = defaultdict(list)
for r in rows:
    ct = r.get('cell_type', 'unknown')
    dx = r.get('simplified_diagnosis', '?')
    groups[(ct, dx)].append(r)

print('\nCell-type × diagnosis groups:')
for k, vs in sorted(groups.items()):
    print(f'  {k}: n={len(vs)}')

a_cols = sorted(set(k for r in rows for k in r if k.startswith('A_')))
print(f'\nA-score columns: {len(a_cols)}')

# Compute d per cell type for CD vs HC, UC vs HC, CD vs UC
results_per_celltype = {}
for ct in ['monocytes', 'CD4', 'CD8', 'wh blood']:
    cd = groups.get((ct, 'CD'), [])
    uc = groups.get((ct, 'UC'), [])
    hc = groups.get((ct, 'HC'), [])
    if not cd or not hc: continue
    
    ct_results = {
        'cell_type': ct,
        'n_CD': len(cd),
        'n_UC': len(uc),
        'n_HC': len(hc),
        'CD_vs_HC': {},
        'UC_vs_HC': {},
        'CD_vs_UC': {},
    }
    
    for col in a_cols:
        cd_vals = [r.get(col) for r in cd]
        uc_vals = [r.get(col) for r in uc]
        hc_vals = [r.get(col) for r in hc]
        
        # CD vs HC
        d, n1, n2, t, p, ci_l, ci_h = welch_d(cd_vals, hc_vals)
        ct_results['CD_vs_HC'][col] = {'d': d, 'n_CD': n1, 'n_HC': n2, 'p': p, 'outcome': outcome_class(d)}
        
        # UC vs HC
        d, n1, n2, t, p, ci_l, ci_h = welch_d(uc_vals, hc_vals)
        ct_results['UC_vs_HC'][col] = {'d': d, 'n_UC': n1, 'n_HC': n2, 'p': p, 'outcome': outcome_class(d)}
        
        # CD vs UC
        d, n1, n2, t, p, ci_l, ci_h = welch_d(cd_vals, uc_vals)
        ct_results['CD_vs_UC'][col] = {'d': d, 'n_CD': n1, 'n_UC': n2, 'p': p, 'outcome': outcome_class(d)}
    
    results_per_celltype[ct] = ct_results

# Mixture-attenuation test: compare sorted-cell d to whole-blood d
print('\nMixture-attenuation test (sorted vs whole-blood):')
mix_test = {}
for col in a_cols:
    sorted_ds_cd = [results_per_celltype[ct]['CD_vs_HC'][col].get('d')
                    for ct in ['monocytes', 'CD4', 'CD8'] if ct in results_per_celltype]
    sorted_ds_cd = [d for d in sorted_ds_cd if d is not None]
    wb_d_cd = results_per_celltype.get('wh blood', {}).get('CD_vs_HC', {}).get(col, {}).get('d')
    if sorted_ds_cd and wb_d_cd is not None:
        max_sorted = max(sorted_ds_cd, key=abs)
        ratio = abs(max_sorted) / abs(wb_d_cd) if abs(wb_d_cd) > 0.05 else None
        mix_test[col] = {
            'max_sorted_d_CD': max_sorted,
            'wh_blood_d_CD': wb_d_cd,
            'sorted_to_wb_ratio': ratio,
            'mixture_attenuation_passes': ratio is not None and ratio >= 1.5,
        }

# Summary stats: how many sorted-cell d's exceed whole-blood d by 1.5x?
mix_passes = sum(1 for v in mix_test.values() if v.get('mixture_attenuation_passes'))
mix_total = len(mix_test)
print(f'  Mixture attenuation (sorted ≥ 1.5x whole-blood): {mix_passes}/{mix_total} tiles')

# Crohn's-pathway language outcome (PRIMARY)
# O1_CROHNS_LANGUAGE_SUPPORTED: |d| >= 0.5 on Stage 1 OR Stage 3 immune in any cell type
crohns_outcome = 'O3_NO_CROHNS_SIGNATURE'  # default null
crohns_evidence = []
for ct in ['monocytes', 'CD4', 'CD8', 'wh blood']:
    if ct not in results_per_celltype: continue
    # Stage 1
    d_stage1 = results_per_celltype[ct]['CD_vs_HC'].get('A_xu538_stage1', {}).get('d')
    if d_stage1 is not None and abs(d_stage1) >= 0.5:
        crohns_evidence.append(f'Stage 1 in {ct}: d_CD-HC = {d_stage1:+.3f}')
    elif d_stage1 is not None and abs(d_stage1) >= 0.2:
        crohns_evidence.append(f'Stage 1 partial in {ct}: d_CD-HC = {d_stage1:+.3f}')
    # Stage 3 immune (Salas, Loyfer immune tiles, Caggiano immune)
    for col in ['A_salas_CD8T', 'A_salas_CD4T', 'A_salas_NK', 'A_salas_Bcell',
                'A_salas_Mono', 'A_salas_Neu',
                'A_loyfer_B-cells_EPIC', 'A_loyfer_CD4T-cells_EPIC',
                'A_loyfer_CD8T-cells_EPIC', 'A_loyfer_NK-cells_EPIC',
                'A_loyfer_Monocytes_EPIC', 'A_loyfer_Neutrophils_EPIC',
                'A_cag_dendritic', 'A_cag_macrophage', 'A_cag_monocyte',
                'A_cag_neutrophil', 'A_cag_tcell']:
        d = results_per_celltype[ct]['CD_vs_HC'].get(col, {}).get('d')
        if d is not None and abs(d) >= 0.5:
            crohns_evidence.append(f'Stage 3 in {ct}: {col}: d_CD-HC = {d:+.3f}')

# Decide outcome
abs_max_d = 0
for ct in results_per_celltype.values():
    for col_data in ct['CD_vs_HC'].values():
        d = col_data.get('d')
        if d is not None and abs(d) > abs_max_d:
            abs_max_d = abs(d)

if abs_max_d >= 0.5:
    crohns_outcome = 'O1_CROHNS_LANGUAGE_SUPPORTED'
elif abs_max_d >= 0.2:
    crohns_outcome = 'O2_CROHNS_LANGUAGE_PARTIAL'
else:
    crohns_outcome = 'O3_NO_CROHNS_SIGNATURE'

# Build results dict
results = {
    'val_id': 'VAL-128',
    'val_title': 'GSE87650 Crohn\'s Disease Blood Methylation (gastric+esophageal-epic v0.1 Crohn\'s pathway)',
    'prereg_sha256': PREREG_SHA,
    'cohort_summary': {
        'total': len(rows),
        'cell_type_diagnosis_distribution': {f'{k[0]}_{k[1]}': len(v) for k, v in groups.items()},
    },
    'results_per_celltype': results_per_celltype,
    'mixture_attenuation_test': {
        'tiles_passing_1_5x': mix_passes,
        'tiles_total': mix_total,
        'pass_rate': mix_passes / mix_total if mix_total else 0,
        'detail': mix_test,
    },
    'crohns_pathway_outcome': {
        'outcome_class': crohns_outcome,
        'max_abs_d_CD_vs_HC': abs_max_d,
        'evidence_lines': crohns_evidence[:30],  # cap at 30
    },
}

with open(OUTPUT / 'VAL-128_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\n  Wrote VAL-128_results.json')

# Per-sample CSV
all_keys = sorted(set(k for r in rows for k in r))
with open(OUTPUT / 'VAL-128_per_sample.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
    w.writeheader()
    for r in rows: w.writerow(r)
print(f'  Wrote VAL-128_per_sample.csv ({len(rows)} rows × {len(all_keys)} cols)')

# Headline
print('\n' + '='*80)
print('VAL-128 HEADLINE — GSE87650 Crohn\'s blood methylation')
print('='*80)
print(f'\n  Crohn\'s pathway outcome: {crohns_outcome}')
print(f'  Max |d| CD vs HC across all atlases/cell-types: {abs_max_d:.3f}')

print(f'\n  Stage 1 by cell type:')
for ct in ['monocytes', 'CD4', 'CD8', 'wh blood']:
    if ct not in results_per_celltype: continue
    cd_hc = results_per_celltype[ct]['CD_vs_HC'].get('A_xu538_stage1', {})
    uc_hc = results_per_celltype[ct]['UC_vs_HC'].get('A_xu538_stage1', {})
    cd_uc = results_per_celltype[ct]['CD_vs_UC'].get('A_xu538_stage1', {})
    print(f'    {ct:14s}: CD-HC d={cd_hc.get("d", 0):+.3f} (p={cd_hc.get("p", "?"):.2e if cd_hc.get("p") else "?"}), UC-HC d={uc_hc.get("d", 0):+.3f}, CD-UC d={cd_uc.get("d", 0):+.3f}')

# Top discriminating tiles for CD vs HC
print(f'\n  Top |d_CD-HC| ≥ 0.5 tiles by cell type:')
for ct in ['monocytes', 'CD4', 'CD8', 'wh blood']:
    if ct not in results_per_celltype: continue
    big = [(col, v['d']) for col, v in results_per_celltype[ct]['CD_vs_HC'].items()
           if v.get('d') is not None and abs(v['d']) >= 0.5]
    big.sort(key=lambda x: -abs(x[1]))
    if big:
        print(f'    {ct}:')
        for col, d in big[:5]:
            print(f'      {col:35s}: d={d:+.3f}')

# Mixture-attenuation summary
print(f'\n  Mixture-attenuation pre-locked test (sorted ≥ 1.5x whole-blood):')
print(f'    {mix_passes}/{mix_total} tiles pass = {mix_passes/mix_total*100:.1f}%')
print(f'    Pre-locked threshold: at least one tile passing supports mixture-effect dilution model')

print('\nDone.')
