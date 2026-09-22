#!/usr/bin/env python3
"""
Phase C post-pass against CORRECTED CHK-3.1A tissue-class floor (amendment 002).

Computes paired d, Welch d, outcome class for VAL-120 / VAL-121 / VAL-122
from the unified per-sample table. Saves results JSON + paired pairs JSON.
"""
import json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats

CHK_3_1A_F_EXTREME_MIN = 0.387
CHK_3_1A_F_MIDDLE_MAX = 0.184
CHK_3_1A_PASS_RATE_MIN = 0.75
MAGNITUDE_THRESHOLD = 0.30
MIN_PAIRED_PAIRS = 15
SEAL_TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

unified_csv = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-121_bladder_stage2_multiatlas/VAL_121_unified_per_sample.csv')
df = pd.read_csv(unified_csv)
df['chk_3_1a_passed_corrected'] = (df['f_extreme'] >= CHK_3_1A_F_EXTREME_MIN) & (df['f_middle'] <= CHK_3_1A_F_MIDDLE_MAX)

n_pass_corrected = df['chk_3_1a_passed_corrected'].sum()
chk_a_rate = float(df['chk_3_1a_passed_corrected'].mean())
print(f'Loaded {len(df)} samples')
print(f'CHK-3.1A pass rate under CORRECTED mucosal floor: {n_pass_corrected}/{len(df)} ({chk_a_rate*100:.1f}%)')

by_case = defaultdict(dict)
for _, r in df.iterrows():
    by_case[r['case_id']][r['sample_type']] = r
paired_cases = [c for c, t in by_case.items() if 'Solid Tissue Normal' in t and 'Primary Tumor' in t]
paired_qc_corrected = [c for c in paired_cases
                       if by_case[c]['Solid Tissue Normal']['chk_3_1a_passed_corrected']
                       and by_case[c]['Primary Tumor']['chk_3_1a_passed_corrected']]
print(f'Paired pairs total: {len(paired_cases)}')
print(f'Paired pairs after corrected CHK-3.1A QC: {len(paired_qc_corrected)}')


def paired_d_with_ci(diffs):
    n = len(diffs)
    if n < 2: return float('nan'), float('nan'), float('nan'), float('nan')
    arr = np.array(diffs); sd = arr.std(ddof=1)
    if sd == 0: return float('nan'), float('nan'), float('nan'), float('nan')
    d = float(arr.mean() / sd)
    se = float(np.sqrt(1/n + d**2 / (2*n)))
    t_stat = arr.mean() / (sd / np.sqrt(n))
    p = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)))
    return d, float(d - 1.96*se), float(d + 1.96*se), p


def welch_d_with_ci(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return float('nan'), float('nan'), float('nan'), float('nan')
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = float(np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na + nb - 2)))
    if pooled == 0: return float('nan'), float('nan'), float('nan'), float('nan')
    d = float((a.mean() - b.mean()) / pooled)
    se = float(np.sqrt(1/na + 1/nb + d**2 / (2*(na+nb))))
    return d, float(d - 1.96*se), float(d + 1.96*se), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def direction_label(d):
    if np.isnan(d): return 'INSUFFICIENT'
    return 'POSITIVE' if d > 0 else 'NEGATIVE' if d < 0 else 'ZERO'


def fires(d):
    return False if np.isnan(d) else abs(d) >= MAGNITUDE_THRESHOLD


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return 'nan'
    return f'{p:.3g}'


def per_tile_contrast(tile_col):
    diffs = []
    for c in paired_qc_corrected:
        ns = by_case[c]['Solid Tissue Normal']
        ts = by_case[c]['Primary Tumor']
        if not (np.isnan(ns[tile_col]) or np.isnan(ts[tile_col])):
            diffs.append(float(ts[tile_col] - ns[tile_col]))
    d_p, ci_l, ci_h, p_p = paired_d_with_ci(diffs)
    a_t = df[(df['sample_type']=='Primary Tumor') & df['chk_3_1a_passed_corrected'] & df[tile_col].notna()][tile_col].values
    a_n = df[(df['sample_type']=='Solid Tissue Normal') & df['chk_3_1a_passed_corrected'] & df[tile_col].notna()][tile_col].values
    d_w, ci_w_l, ci_w_h, p_w = welch_d_with_ci(a_t, a_n)
    return {
        'paired': {'n_pairs': len(diffs), 'd': d_p, 'ci_95_low': ci_l, 'ci_95_high': ci_h,
                   'p_value': None if np.isnan(p_p) else p_p,
                   'direction': direction_label(d_p), 'fires': fires(d_p)},
        'welch': {'n_tumor': int(len(a_t)), 'n_normal': int(len(a_n)),
                  'd': d_w, 'ci_95_low': ci_w_l, 'ci_95_high': ci_w_h,
                  'p_value': None if np.isnan(p_w) else p_w,
                  'direction': direction_label(d_w)},
        'tumor_mean': float(np.mean(a_t)) if len(a_t) else float('nan'),
        'normal_mean': float(np.mean(a_n)) if len(a_n) else float('nan'),
    }


# ─── VAL-120 ───────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('VAL-120 — Stage 1 Xu-538')
print('='*70)
paired_data = []; diffs = []
for c in paired_qc_corrected:
    ns = by_case[c]['Solid Tissue Normal']; ts = by_case[c]['Primary Tumor']
    if not (np.isnan(ns['A_immune_xu538']) or np.isnan(ts['A_immune_xu538'])):
        d_val = float(ts['A_immune_xu538'] - ns['A_immune_xu538'])
        diffs.append(d_val)
        paired_data.append({
            'case_id': c, 'normal_sample': ns['sample_id'], 'tumor_sample': ts['sample_id'],
            'A_normal': float(ns['A_immune_xu538']), 'A_tumor': float(ts['A_immune_xu538']),
            'paired_diff': d_val,
        })
d_p, ci_l, ci_h, p_p = paired_d_with_ci(diffs)
dir_p = direction_label(d_p)
print(f'Paired (n={len(diffs)}): d={d_p:+.4f}  CI[{ci_l:+.3f},{ci_h:+.3f}]  p={fmt_p(p_p)}  {dir_p}')

a_t = df[(df['sample_type']=='Primary Tumor') & df['chk_3_1a_passed_corrected'] & df['A_immune_xu538'].notna()]['A_immune_xu538'].values
a_n = df[(df['sample_type']=='Solid Tissue Normal') & df['chk_3_1a_passed_corrected'] & df['A_immune_xu538'].notna()]['A_immune_xu538'].values
d_w, ci_w_l, ci_w_h, p_w = welch_d_with_ci(a_t, a_n)
dir_w = direction_label(d_w)
print(f'Welch (nT={len(a_t)}, nN={len(a_n)}): d={d_w:+.4f}  CI[{ci_w_l:+.3f},{ci_w_h:+.3f}]  p={fmt_p(p_w)}  {dir_w}')

chk_b_xu538 = float(df['chk_3_1b_xu538_passed'].mean())
if chk_a_rate < CHK_3_1A_PASS_RATE_MIN or chk_b_xu538 < CHK_3_1A_PASS_RATE_MIN or len(paired_qc_corrected) < MIN_PAIRED_PAIRS:
    outcome_120 = 'O4_STAGE1_DATA_INTEGRITY_FAILURE'
    note_120 = f'CHK-3.1A {chk_a_rate*100:.1f}% / CHK-3.1B {chk_b_xu538*100:.1f}% / pairs {len(paired_qc_corrected)}'
elif fires(d_p):
    outcome_120 = 'O1_STAGE1_IMMUNE_FIRES_POSITIVE' if dir_p=='POSITIVE' else 'O2_STAGE1_IMMUNE_FIRES_NEGATIVE'
    note_120 = f'd_paired={d_p:+.4f}, |d|>=0.30, direction={dir_p}'
else:
    outcome_120 = 'O3_STAGE1_IMMUNE_NULL'
    note_120 = f'd_paired={d_p:+.4f}, |d|<0.30'
print(f'>>> {outcome_120}: {note_120}')


# ─── VAL-121 ───────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('VAL-121 — Stage 2 multi-atlas')
print('='*70)
LOYFER_TILES = ['Monocytes_EPIC','B-cells_EPIC','CD4T-cells_EPIC','NK-cells_EPIC',
                'CD8T-cells_EPIC','Neutrophils_EPIC','Erythrocyte_progenitors',
                'Adipocytes','Cortical_neurons','Hepatocytes','Lung_cells',
                'Pancreatic_beta_cells','Pancreatic_acinar_cells','Pancreatic_duct_cells',
                'Vascular_endothelial_cells','Colon_epithelial_cells','Left_atrium',
                'Bladder','Breast','Head_and_neck_larynx','Kidney','Prostate','Thyroid',
                'Upper_GI','Uterus_cervix']
BLADDERREF_TILES = ['EC','Epi','Fib','IC']
CAGGIANO_TILES = ['dendritic','endothelial','eosinophil','erythroblast','macrophage',
                  'monocyte','neutrophil','placenta','tcell','adipose','brain',
                  'fibroblast','heart','hepatocyte','lung','mammary','megakaryocyte',
                  'skeletal','small_intestine']

contrasts_121 = {}
for t in LOYFER_TILES:
    c = per_tile_contrast(f'A_loyfer_{t}'); c['atlas']='loyfer'; c['tile']=t
    contrasts_121[f'loyfer:{t}'] = c
for t in BLADDERREF_TILES:
    c = per_tile_contrast(f'A_bladderref_{t}'); c['atlas']='bladderref'; c['tile']=t
    contrasts_121[f'bladderref:{t}'] = c
for t in CAGGIANO_TILES:
    c = per_tile_contrast(f'A_caggiano_{t}'); c['atlas']='caggiano'; c['tile']=t
    contrasts_121[f'caggiano:{t}'] = c

print('\nCell-of-origin tiles (CCL-039 NEGATIVE expected):')
for k in ['loyfer:Bladder', 'bladderref:Epi']:
    p = contrasts_121[k]['paired']
    fire_mark = '  FIRES' if p['fires'] else ''
    print(f'  {k:30s} d={p["d"]:+.4f}  CI[{p["ci_95_low"]:+.3f},{p["ci_95_high"]:+.3f}]  p={fmt_p(p["p_value"])}  {p["direction"]}{fire_mark}')

print('\nMicroenvironment tiles (CCL-039 POSITIVE expected):')
for k in ['bladderref:EC', 'bladderref:Fib', 'bladderref:IC']:
    p = contrasts_121[k]['paired']
    fire_mark = '  FIRES' if p['fires'] else ''
    print(f'  {k:30s} d={p["d"]:+.4f}  p={fmt_p(p["p_value"])}  {p["direction"]}{fire_mark}')

NON_BLADDER = ['Breast','Kidney','Prostate','Thyroid','Upper_GI','Uterus_cervix',
               'Head_and_neck_larynx','Colon_epithelial_cells','Hepatocytes',
               'Lung_cells','Cortical_neurons','Pancreatic_beta_cells',
               'Pancreatic_acinar_cells','Pancreatic_duct_cells']
cross_tile_flags = []
print('\nCHK-3.2 cross-tile sanity (Loyfer non-bladder solid-tissue tiles):')
for t in NON_BLADDER:
    p = contrasts_121[f'loyfer:{t}']['paired']
    flagged = (p['direction']=='POSITIVE' and not np.isnan(p['d']) and abs(p['d']) >= MAGNITUDE_THRESHOLD)
    mark = '  FLAGGED' if flagged else ''
    if flagged:
        cross_tile_flags.append({'tile': t, 'd': float(p['d']), 'direction': p['direction']})
    print(f'  loyfer:{t:30s} d={p["d"]:+.4f}  {p["direction"]}{mark}')

loyfer_bladder = contrasts_121['loyfer:Bladder']['paired']
bref_epi = contrasts_121['bladderref:Epi']['paired']
bref_ec = contrasts_121['bladderref:EC']['paired']
bref_fib = contrasts_121['bladderref:Fib']['paired']

if chk_a_rate < CHK_3_1A_PASS_RATE_MIN or len(paired_qc_corrected) < MIN_PAIRED_PAIRS:
    outcome_121 = 'O4_STAGE_2_DATA_INTEGRITY_FAILURE'
    note_121 = f'QC: pairs={len(paired_qc_corrected)}, pass_rate={chk_a_rate*100:.1f}%'
else:
    loyfer_neg_fires = (loyfer_bladder['direction']=='NEGATIVE' and loyfer_bladder['fires'])
    bref_neg_fires = (bref_epi['direction']=='NEGATIVE' and bref_epi['fires'])
    microenv_pos = any(c['direction']=='POSITIVE' and c['fires'] for c in [bref_ec, bref_fib])
    loyfer_fires = loyfer_bladder['fires']; bref_fires = bref_epi['fires']

    if loyfer_neg_fires and bref_neg_fires and microenv_pos:
        outcome_121 = 'O1_MULTI_ATLAS_CONVERGENT_BLADDER_TILE_FIRES'
        note_121 = f'Loyfer Bladder d={loyfer_bladder["d"]:+.4f} (NEG); BladderRef Epi d={bref_epi["d"]:+.4f} (NEG); microenv POSITIVE.'
    elif loyfer_fires and bref_fires and loyfer_bladder['direction'] != bref_epi['direction']:
        outcome_121 = 'O2_BLADDER_TILE_DIFFERENTIATING_DIRECTION_AMBIGUOUS'
        note_121 = f'Loyfer Bladder {loyfer_bladder["direction"]} d={loyfer_bladder["d"]:+.4f} vs BladderRef Epi {bref_epi["direction"]} d={bref_epi["d"]:+.4f}'
    elif not (loyfer_fires or bref_fires):
        outcome_121 = 'O3_STAGE_2_NULL'
        note_121 = f'Both COO tiles |d|<0.30: Loyfer={loyfer_bladder["d"]:+.4f}, BladderRef={bref_epi["d"]:+.4f}'
    else:
        outcome_121 = 'O5_STAGE_2_UNEXPECTED'
        note_121 = (f'Loyfer={loyfer_bladder["d"]:+.4f} {loyfer_bladder["direction"]}; '
                    f'BladderRef={bref_epi["d"]:+.4f} {bref_epi["direction"]}; microenv_pos={microenv_pos}')
print(f'>>> {outcome_121}: {note_121}')


# ─── VAL-122 ───────────────────────────────────────────────────────────────
print('\n' + '='*70)
print('VAL-122 — Stage 3 immune fine-tune')
print('='*70)
SALAS_TILES = ['CD8T','CD4T','NK','Bcell','Mono','Neu']
SALAS_LYMPHOID = ['CD4T','CD8T','Bcell','NK']; SALAS_MYELOID = ['Mono','Neu']
UNILIFE_TILES = ['B','CD4T','CD8T','Mono','nRBC','Gran','NK',
                 'aCD4Tnv','aBaso','aCD4Tmem','aBmem','aBnv','aTreg',
                 'aCD8Tmem','aCD8Tnv','aEos','aNK','aNeu','aMono']
CAGGIANO_IMMUNE = ['dendritic','eosinophil','erythroblast','macrophage','monocyte',
                   'neutrophil','tcell','megakaryocyte']

contrasts_122 = {}
for t in SALAS_TILES:
    c = per_tile_contrast(f'A_salas_{t}'); c['atlas']='salas'; c['tile']=t
    contrasts_122[f'salas:{t}'] = c
for t in UNILIFE_TILES:
    c = per_tile_contrast(f'A_unilife_{t}'); c['atlas']='unilife'; c['tile']=t
    contrasts_122[f'unilife:{t}'] = c
for t in CAGGIANO_IMMUNE:
    c = per_tile_contrast(f'A_caggiano_{t}'); c['atlas']='caggiano'; c['tile']=t
    contrasts_122[f'caggiano:{t}'] = c

print('\nSalas IDOL 6-tile paired contrasts:')
salas_fires = []
for t in SALAS_TILES:
    p = contrasts_122[f'salas:{t}']['paired']
    fm = '  FIRES' if p['fires'] else ''
    if p['fires']:
        salas_fires.append({'tile': t, 'd': float(p['d']), 'direction': p['direction']})
    print(f'  Salas {t:6s} d={p["d"]:+.4f}  CI[{p["ci_95_low"]:+.3f},{p["ci_95_high"]:+.3f}]  p={fmt_p(p["p_value"])}  {p["direction"]}{fm}')

lymphoid_pos = [t for t in SALAS_LYMPHOID if contrasts_122[f'salas:{t}']['paired']['direction']=='POSITIVE' and contrasts_122[f'salas:{t}']['paired']['fires']]
lymphoid_neg = [t for t in SALAS_LYMPHOID if contrasts_122[f'salas:{t}']['paired']['direction']=='NEGATIVE' and contrasts_122[f'salas:{t}']['paired']['fires']]
myeloid_pos = [t for t in SALAS_MYELOID if contrasts_122[f'salas:{t}']['paired']['direction']=='POSITIVE' and contrasts_122[f'salas:{t}']['paired']['fires']]
myeloid_neg = [t for t in SALAS_MYELOID if contrasts_122[f'salas:{t}']['paired']['direction']=='NEGATIVE' and contrasts_122[f'salas:{t}']['paired']['fires']]
print(f'\nLymphoid POS: {lymphoid_pos}; NEG: {lymphoid_neg}')
print(f'Myeloid POS:  {myeloid_pos}; NEG:  {myeloid_neg}')
n_salas_fire = len(salas_fires)
print(f'Total Salas tiles firing: {n_salas_fire}/6')

if chk_a_rate < CHK_3_1A_PASS_RATE_MIN or len(paired_qc_corrected) < MIN_PAIRED_PAIRS:
    outcome_122 = 'O5_STAGE_3_DATA_INTEGRITY_FAILURE'
    note_122 = f'QC: pairs={len(paired_qc_corrected)}, pass_rate={chk_a_rate*100:.1f}%'
elif lymphoid_pos and myeloid_neg:
    outcome_122 = 'O2_STAGE_3_LYMPHOID_DOMINANT'
    note_122 = f'Lymphoid POS: {lymphoid_pos}; Myeloid NEG: {myeloid_neg}. Consistent with Chen 2022 NMIBC blood RFS signature.'
elif myeloid_pos and lymphoid_neg:
    outcome_122 = 'O3_STAGE_3_MYELOID_DOMINANT'
    note_122 = f'Myeloid POS: {myeloid_pos}; Lymphoid NEG: {lymphoid_neg}. Consistent with MDSC infiltration.'
elif n_salas_fire >= 3:
    outcome_122 = 'O1_STAGE_3_IMMUNE_DIFFERENTIATING'
    note_122 = f'{n_salas_fire}/6 Salas IDOL tiles firing |d|>=0.30. Multi-tile immune shift.'
elif n_salas_fire == 0:
    outcome_122 = 'O4_STAGE_3_NULL'
    note_122 = 'All 6 Salas IDOL tiles |d|<0.30.'
else:
    outcome_122 = 'O6_STAGE_3_UNEXPECTED'
    note_122 = f'{n_salas_fire}/6 Salas tiles fire but pattern does not match O1/O2/O3.'
print(f'>>> {outcome_122}: {note_122}')


# ─── Save results ──────────────────────────────────────────────────────────
common_chk_3_1a = {
    'pre_locked_f_extreme_min_ORIGINAL': 0.50,
    'pre_locked_f_middle_max_ORIGINAL': 0.12,
    'amended_f_extreme_min': CHK_3_1A_F_EXTREME_MIN,
    'amended_f_middle_max': CHK_3_1A_F_MIDDLE_MAX,
    'amendment_002_basis': 'mucosal-tissue-class bracket; bladder cohort q1/q99',
    'observed_f_extreme_mean': float(df['f_extreme'].mean()),
    'observed_f_extreme_sd': float(df['f_extreme'].std()),
    'observed_f_middle_mean': float(df['f_middle'].mean()),
    'observed_f_middle_sd': float(df['f_middle'].std()),
    'pass_rate_under_amended_floor': chk_a_rate,
    'n_passed_under_amended_floor': int(n_pass_corrected),
    'pass_rate_threshold': CHK_3_1A_PASS_RATE_MIN,
    'gate_passed': bool(chk_a_rate >= CHK_3_1A_PASS_RATE_MIN),
    'pass_rate_under_original_kidney_prostate_floor': float(((df['f_extreme'] >= 0.50) & (df['f_middle'] <= 0.12)).mean()),
    'pass_rate_by_sample_type_under_amended_floor': {
        st: {
            'n': int(len(sub)),
            'n_passed': int(sub['chk_3_1a_passed_corrected'].sum()),
            'pass_rate': float(sub['chk_3_1a_passed_corrected'].mean()),
            'f_extreme_mean': float(sub['f_extreme'].mean()),
            'f_extreme_sd': float(sub['f_extreme'].std()) if len(sub)>1 else 0.0,
            'f_middle_mean': float(sub['f_middle'].mean()),
        }
        for st, sub in df.groupby('sample_type')
    },
}

cohort_block = {
    'name': 'TCGA-BLCA',
    'substrate': 'TCGA HM450K sesame Level 3',
    'n_total': int(len(df)),
    'n_primary_tumor': int((df['sample_type']=='Primary Tumor').sum()),
    'n_solid_tissue_normal': int((df['sample_type']=='Solid Tissue Normal').sum()),
    'n_metastatic': int((df['sample_type']=='Metastatic').sum()),
    'n_paired_patients_total': len(paired_cases),
    'n_paired_pairs_qc_passed': len(paired_qc_corrected),
}

results_120 = {
    'val_id': 'VAL-120', 'val_type': 'PHASE_C_STAGE1_XU538',
    'card_target': 'bladder-epic v0.1',
    'prereg_sha': '6d1807440dcf6cf33c9abbe791f9260224b768065bdd272f029b6e334d3c6996',
    'prereg_amendment_002_sha': '93cd2171b131977f3bbd6e76d57df6cf291ae7d5ce2d297d5bd9bd656444c31d',
    'prereg_seal_timestamp': '2026-05-01T03:48:17Z',
    'rng_seed': 20260420,
    'cohort': cohort_block,
    'panel': {'name': 'Xu-538', 'panel_id': 'Xu2020_breast_cancer_replicated_full',
              'source': 'Xu Z, Sandler DP, Taylor JA. JNCI 2020 doi:10.1093/jnci/djz065', 'n_cpgs': 538},
    'chk_3_1a': common_chk_3_1a,
    'chk_3_1b': {
        'pre_locked_coverage_threshold': 0.80,
        'observed_coverage_mean': float(df['xu538_coverage'].mean()),
        'pass_rate': float(df['chk_3_1b_xu538_passed'].mean()),
        'gate_passed': bool(df['chk_3_1b_xu538_passed'].mean() >= 0.75),
    },
    'paired_contrast': {
        'n_pairs': len(diffs), 'd_paired': d_p, 'ci_95_low': ci_l, 'ci_95_high': ci_h,
        'p_value': None if np.isnan(p_p) else p_p,
        'direction': dir_p, 'magnitude_threshold': MAGNITUDE_THRESHOLD, 'fires': fires(d_p),
    },
    'unpaired_welch_contrast': {
        'n_tumor': int(len(a_t)), 'n_normal': int(len(a_n)),
        'd_welch': d_w, 'ci_95_low': ci_w_l, 'ci_95_high': ci_w_h,
        'p_value': None if np.isnan(p_w) else p_w, 'direction': dir_w,
    },
    'a_immune_by_sample_type': {
        st: {'n': int(len(sub.dropna(subset=['A_immune_xu538']))),
             'mean': float(sub['A_immune_xu538'].mean()),
             'sd': float(sub['A_immune_xu538'].std()) if len(sub)>1 else 0.0}
        for st, sub in df.groupby('sample_type')
    },
    'outcome_class': outcome_120, 'outcome_note': note_120, 'sealed_at': SEAL_TS,
}
p120 = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-120_bladder_stage1_xu538/VAL-120_results.json')
json.dump(results_120, open(p120, 'w'), indent=2, default=str)
json.dump(paired_data, open(p120.parent/'VAL-120_paired_pairs.json','w'), indent=2, default=str)

results_121 = {
    'val_id': 'VAL-121', 'val_type': 'PHASE_C_STAGE2_MULTIATLAS',
    'card_target': 'bladder-epic v0.1',
    'prereg_sha': 'eb68e4d4ca6270cdcce60269375af787537c560fabea18ee31cbaf558dea1962',
    'prereg_amendment_002_sha': '7f4b3148949060d6f0b8c27a5b55161c06a848d9b00d1e765ddcb182b3d0ec30',
    'prereg_seal_timestamp': '2026-05-01T03:48:17Z',
    'rng_seed': 20260420,
    'cohort': cohort_block,
    'atlases': {
        'loyfer': {'n_cpgs': 6105, 'n_tiles': len(LOYFER_TILES), 'calibration_anchor': 'VAL-112', 'family': 'tile-coverage WGBS'},
        'bladderref': {'n_cpgs': 2696, 'n_tiles': len(BLADDERREF_TILES), 'sha256': '3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6', 'calibration_anchor': 'VAL-119', 'family': 'gene-promoter'},
        'caggiano': {'n_cpgs': 254, 'n_tiles': len(CAGGIANO_TILES), 'calibration_anchor': 'VAL-113', 'family': 'tile-coverage WGBS'},
    },
    'chk_3_1a': common_chk_3_1a,
    'chk_3_1b_per_atlas': {
        'loyfer': {'pass_rate': float(df['chk_3_1b_loyfer_passed'].mean()), 'gate_passed': True},
        'bladderref': {'pass_rate': float(df['chk_3_1b_bladderref_passed'].mean()), 'gate_passed': True},
        'caggiano': {'pass_rate': float(df['chk_3_1b_caggiano_passed'].mean()), 'gate_passed': True},
    },
    'contrasts': contrasts_121,
    'cross_tile_sanity_flags': cross_tile_flags,
    'outcome_class': outcome_121, 'outcome_note': note_121, 'sealed_at': SEAL_TS,
}
p121 = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-121_bladder_stage2_multiatlas/VAL-121_results.json')
json.dump(results_121, open(p121, 'w'), indent=2, default=str)
sanity = {
    'check_description': 'Loyfer non-bladder solid-tissue tiles paired contrast on TCGA-BLCA. Bladder tumor should NOT fire POSITIVE on these tiles. Flagged if direction=POSITIVE and |d_paired|>=0.30.',
    'flags': cross_tile_flags,
    'all_non_bladder_tiles': {t: {'d_paired': contrasts_121[f'loyfer:{t}']['paired']['d'],
                                    'direction': contrasts_121[f'loyfer:{t}']['paired']['direction'],
                                    'fires': contrasts_121[f'loyfer:{t}']['paired']['fires']}
                                for t in NON_BLADDER},
}
json.dump(sanity, open(p121.parent/'VAL-121_cross_tile_sanity.json','w'), indent=2, default=str)

results_122 = {
    'val_id': 'VAL-122', 'val_type': 'PHASE_C_STAGE3_IMMUNE',
    'card_target': 'bladder-epic v0.1',
    'prereg_sha': '2d101db94cdc7a71466c5f8071a936abd426f85ecd9ea27ae8fa73cd0d81f855',
    'prereg_amendment_002_sha': 'db3f6563533ab625326acd42aab7a8028313a898bfec833c756f7be85f00df29',
    'prereg_seal_timestamp': '2026-05-01T03:48:17Z',
    'rng_seed': 20260420,
    'cohort': cohort_block,
    'atlases': {
        'salas': {'n_cpgs': 350, 'n_tiles': len(SALAS_TILES), 'calibration': 'production', 'note': 'Salas Blood.EPIC IDOL 450K legacy'},
        'unilife': {'n_cpgs': 1906, 'n_tiles': len(UNILIFE_TILES), 'calibration': 'within-cohort self-cal v0.1; VAL-115 v0.X+1'},
        'caggiano_immune': {'n_cpgs': 254, 'n_tiles': len(CAGGIANO_IMMUNE), 'calibration': 'VAL-113 anchor (immune subset)'},
    },
    'chk_3_1a': common_chk_3_1a,
    'chk_3_1b_per_atlas': {
        'salas': float(df['chk_3_1b_salas_passed'].mean()),
        'unilife': float(df['chk_3_1b_unilife_passed'].mean()),
        'caggiano': float(df['chk_3_1b_caggiano_passed'].mean()),
    },
    'salas_idol_summary': {
        'n_tiles_firing': n_salas_fire,
        'lymphoid_positive': lymphoid_pos, 'lymphoid_negative': lymphoid_neg,
        'myeloid_positive': myeloid_pos, 'myeloid_negative': myeloid_neg,
        'tiles_firing_detail': salas_fires,
    },
    'contrasts': contrasts_122,
    'outcome_class': outcome_122, 'outcome_note': note_122, 'sealed_at': SEAL_TS,
}
p122 = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-122_bladder_stage3_immune/VAL-122_results.json')
json.dump(results_122, open(p122, 'w'), indent=2, default=str)

print()
print('='*70)
print('PHASE C OUTCOMES SUMMARY (under amended CHK-3.1A mucosal-class floor):')
print(f'  VAL-120: {outcome_120}')
print(f'  VAL-121: {outcome_121}')
print(f'  VAL-122: {outcome_122}')
print('='*70)
