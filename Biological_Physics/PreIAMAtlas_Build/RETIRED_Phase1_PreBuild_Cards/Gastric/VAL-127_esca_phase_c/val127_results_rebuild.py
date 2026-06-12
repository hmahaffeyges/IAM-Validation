#!/usr/bin/env python3
"""
VAL-127 ESCA results rebuilder with proper KIRC+PRAD anchor.

Reuses the same val106_anchor_per_sample.ndjson from VAL-126 anchor build —
identical pipeline, same atlases, no need to rescore.

ESCA-specific stratifications:
  - ESCC (n=96) vs EAC (n=89) — primary subtype discrimination test
  - Barrett's history (Yes-USA + Yes-UK = 28 vs No = 118)
  - Smoking (1=Lifelong-non, 2=Current, 3=Reformed-≥15yr, 4=Reformed-<15yr)
  - Sex
  - MSI status (continuous + threshold)
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

OUTPUT = Path('/home/claude/gastric_esophageal_sprint/VAL-127_esca_phase_c')
ANCHOR = Path('/home/claude/gastric_esophageal_sprint/VAL-126_stad_phase_c/val106_anchor_per_sample.ndjson')

PREREG_SHA = 'cb521d83afe8bee8136c73cf0e0526a9b5e60758df7a77ae51709000c4014b1e'

print('Loading ESCA per-sample data...')
esca_rows = []
with open(OUTPUT / 'val127_per_sample_progress.ndjson') as f:
    for line in f:
        esca_rows.append(json.loads(line))
print(f'  ESCA: {len(esca_rows)} samples')

print('Loading KIRC+PRAD anchor (shared with VAL-126)...')
anchor_rows = []
with open(ANCHOR) as f:
    for line in f:
        anchor_rows.append(json.loads(line))
print(f'  Anchor: {len(anchor_rows)} samples')

# Load manifest with clinical
with open(OUTPUT / 'tcga_esca_hm450_manifest_FINAL.json') as f:
    manifest = json.load(f)
fid_to_clin = {r['file_id']: r for r in manifest}

# Attach clinical
for row in esca_rows:
    fid = row.get('file_id')
    clin = fid_to_clin.get(fid, {})
    row['SUBTYPE'] = clin.get('SUBTYPE')
    row['HISTOLOGICAL_DIAGNOSIS'] = clin.get('HISTOLOGICAL_DIAGNOSIS')
    row['BARRETTS_ESOPHAGUS'] = clin.get('BARRETTS_ESOPHAGUS')
    row['TOBACCO_SMOKING_HISTORY'] = clin.get('TOBACCO_SMOKING_HISTORY')
    row['SMOKING_PACK_YEARS'] = clin.get('SMOKING_PACK_YEARS')
    row['ALCOHOL_CONSUMPTION_FREQUENCY'] = clin.get('ALCOHOL_CONSUMPTION_FREQUENCY')
    row['MSI_SENSOR_SCORE'] = clin.get('MSI_SENSOR_SCORE')
    row['gender'] = clin.get('gender')
    row['primary_diagnosis'] = clin.get('primary_diagnosis')
    row['ajcc_pathologic_stage'] = clin.get('ajcc_pathologic_stage')
    row['site_of_resection'] = clin.get('site_of_resection')
    row['age_at_index'] = clin.get('age_at_index')
    row['COLUMNAR_METAPLASIA_PRESENT'] = clin.get('COLUMNAR_METAPLASIA_PRESENT')

# === stratification mappers ===
def histology_class(diag):
    if diag is None: return 'NotReported'
    s = diag or ''
    if 'Squamous' in s: return 'ESCC'
    if 'Adenocarcinoma' in s: return 'EAC'
    return 'other'

def barrett_class(v):
    if v in ('Yes-USA', 'Yes-UK'): return 'Yes'
    if v == 'No': return 'No'
    return 'NotReported'

def smoking_class(v):
    """TCGA codes: 1=Lifelong-non-smoker, 2=Current, 3=Reformed-≥15yr, 4=Reformed-<15yr"""
    if v == '1': return 'Lifelong_non'
    if v == '2': return 'Current'
    if v == '3': return 'Reformed_ge15yr'
    if v == '4': return 'Reformed_lt15yr'
    return 'NotReported'

def msi_class(score):
    if score is None or score == '' or score == 'NA': return 'NotAvailable'
    try:
        return 'MSI-H' if float(score) >= 4.0 else 'MSS'
    except (ValueError, TypeError):
        return 'NotAvailable'

for row in esca_rows:
    row['histology'] = histology_class(row.get('HISTOLOGICAL_DIAGNOSIS'))
    row['barrett_status'] = barrett_class(row.get('BARRETTS_ESOPHAGUS'))
    row['smoking_status'] = smoking_class(row.get('TOBACCO_SMOKING_HISTORY'))
    row['msi_status'] = msi_class(row.get('MSI_SENSOR_SCORE'))

# === d computation ===
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


def outcome_class(d, expected_direction=None):
    if d is None: return 'NULL_INSUFFICIENT_DATA'
    abs_d = abs(d)
    if abs_d >= 0.5:
        label = 'DIFFERENTIATING_POSITIVE' if d > 0 else 'DIFFERENTIATING_NEGATIVE'
        if expected_direction == 'POSITIVE' and d < 0: label += '_UNEXPECTED'
        elif expected_direction == 'NEGATIVE' and d > 0: label += '_UNEXPECTED'
        elif expected_direction == 'NULL': label += '_CROSS_TISSUE_OVERREAD'
        return label
    if abs_d >= 0.2: return 'PARTIAL'
    return 'NULL'

EXPECTED = {
    'A_xu538_stage1': 'POSITIVE',
    'A_loyfer_Bladder': 'POSITIVE', 'A_loyfer_Lung_cells': 'POSITIVE',
    'A_loyfer_Hepatocytes': 'POSITIVE', 'A_loyfer_Pancreatic_beta_cells': 'POSITIVE',
    # ESCC-specific: squamous tiles SHOULD fire on ESCC (cell-of-origin POSITIVE for ESCC)
    # EAC: adenocarcinoma — squamous tiles should be lower or null
    # OEref Basal: oral squamous, may inform on ESCC
}

# === split tumor / normal ===
tumor_rows = [r for r in esca_rows if r.get('sample_type') == 'Primary Tumor']
normal_rows = [r for r in esca_rows if r.get('sample_type') == 'Solid Tissue Normal']
mets_rows = [r for r in esca_rows if r.get('sample_type') == 'Metastatic']
print(f'  Tumor n={len(tumor_rows)} (ESCC n={sum(1 for r in tumor_rows if r["histology"]=="ESCC")}, EAC n={sum(1 for r in tumor_rows if r["histology"]=="EAC")})')
print(f'  ESCA-paired-normal n={len(normal_rows)}, Metastatic n={len(mets_rows)}, anchor n={len(anchor_rows)}')

a_cols = sorted(set(k for r in esca_rows for k in r if k.startswith('A_')))
print(f'  A-score columns: {len(a_cols)}')

# === Primary: tumor vs anchor ===
print('\nComputing per-tile primary d-values...')
per_tile_primary = {}
for col in a_cols:
    tumor_vals = [r.get(col) for r in tumor_rows]
    anchor_vals = [r.get(col) for r in anchor_rows]
    norm_vals = [r.get(col) for r in normal_rows]
    
    d, n_t, n_a, t, p, ci_l, ci_h = welch_d(tumor_vals, anchor_vals)
    d_norm, _, _, _, _, _, _ = welch_d(tumor_vals, norm_vals)
    expected = EXPECTED.get(col)
    
    per_tile_primary[col] = {
        'd_vs_KIRC_PRAD_anchor': d,
        'n_tumor': n_t, 'n_anchor': n_a,
        'tumor_mean': float(np.mean([v for v in tumor_vals if v is not None])) if any(v is not None for v in tumor_vals) else None,
        'tumor_sd': float(np.std([v for v in tumor_vals if v is not None], ddof=1)) if sum(1 for v in tumor_vals if v is not None) >= 2 else None,
        'anchor_mean': float(np.mean([v for v in anchor_vals if v is not None])) if any(v is not None for v in anchor_vals) else None,
        'd_vs_ESCA_normal_n16': d_norm,
        'ci_95_low': ci_l, 'ci_95_high': ci_h,
        't_stat': t, 'p_value': p,
        'expected_direction': expected,
        'outcome_class': outcome_class(d, expected),
    }

# === ESCC vs EAC subtype discrimination — VAL-127's HEADLINE TEST ===
print('\nComputing ESCC vs EAC subtype discrimination...')
escc = [r for r in tumor_rows if r['histology'] == 'ESCC']
eac = [r for r in tumor_rows if r['histology'] == 'EAC']
print(f'  ESCC n={len(escc)}, EAC n={len(eac)}')

subtype_disc = {}
for col in a_cols:
    escc_vals = [r.get(col) for r in escc]
    eac_vals = [r.get(col) for r in eac]
    d, n_e, n_a, t, p, ci_l, ci_h = welch_d(escc_vals, eac_vals)
    # also vs anchor
    d_escc_anchor, _, _, _, p_escc_a, _, _ = welch_d(escc_vals, [r.get(col) for r in anchor_rows])
    d_eac_anchor, _, _, _, p_eac_a, _, _ = welch_d(eac_vals, [r.get(col) for r in anchor_rows])
    subtype_disc[col] = {
        'd_ESCC_minus_EAC': d,
        'p_value': p,
        'd_ESCC_vs_anchor': d_escc_anchor,
        'd_EAC_vs_anchor': d_eac_anchor,
        'escc_mean': float(np.mean([v for v in escc_vals if v is not None])) if any(v is not None for v in escc_vals) else None,
        'eac_mean': float(np.mean([v for v in eac_vals if v is not None])) if any(v is not None for v in eac_vals) else None,
        'discriminating': abs(d) >= 0.5 if d is not None else False,
    }

# === Stratified analyses ===
print('\nComputing stratifications...')
stratifications = {}

# By histology vs anchor
for hist in ['ESCC', 'EAC']:
    subset = [r for r in tumor_rows if r['histology'] == hist]
    stratifications[f'histology_{hist}'] = {'n': len(subset), 'tiles': {}}
    for col in a_cols:
        d, n_t, _, _, p, _, _ = welch_d([r.get(col) for r in subset], [r.get(col) for r in anchor_rows])
        stratifications[f'histology_{hist}']['tiles'][col] = {'d': d, 'n': n_t, 'p': p, 'outcome': outcome_class(d)}

# By Barrett's status (tumor only)
for status in ['Yes', 'No']:
    subset = [r for r in tumor_rows if r['barrett_status'] == status]
    if not subset: continue
    stratifications[f'barrett_{status}'] = {'n': len(subset), 'tiles': {}}
    for col in a_cols:
        d, n_t, _, _, p, _, _ = welch_d([r.get(col) for r in subset], [r.get(col) for r in anchor_rows])
        stratifications[f'barrett_{status}']['tiles'][col] = {'d': d, 'n': n_t, 'p': p, 'outcome': outcome_class(d)}

# By smoking status
for status in ['Current', 'Reformed_ge15yr', 'Reformed_lt15yr', 'Lifelong_non']:
    subset = [r for r in tumor_rows if r['smoking_status'] == status]
    if len(subset) < 5: continue
    stratifications[f'smoking_{status}'] = {'n': len(subset), 'tiles': {}}
    for col in a_cols:
        d, n_t, _, _, p, _, _ = welch_d([r.get(col) for r in subset], [r.get(col) for r in anchor_rows])
        stratifications[f'smoking_{status}']['tiles'][col] = {'d': d, 'n': n_t, 'p': p, 'outcome': outcome_class(d)}

# By sex
for sex in ['male', 'female']:
    subset = [r for r in tumor_rows if r.get('gender') == sex]
    if not subset: continue
    stratifications[f'sex_{sex}'] = {'n': len(subset), 'tiles': {}}
    for col in a_cols:
        d, n_t, _, _, p, _, _ = welch_d([r.get(col) for r in subset], [r.get(col) for r in anchor_rows])
        stratifications[f'sex_{sex}']['tiles'][col] = {'d': d, 'n': n_t, 'p': p, 'outcome': outcome_class(d)}

# === CHK-3.2 substrate baseline ===
esca_normal_fex = [r['f_extreme'] for r in normal_rows if 'f_extreme' in r]
esca_tumor_fex = [r['f_extreme'] for r in tumor_rows if 'f_extreme' in r]
anchor_fex = [r['f_extreme'] for r in anchor_rows if 'f_extreme' in r]
substrate = {
    'esca_tumor_f_extreme_mean': float(np.mean(esca_tumor_fex)) if esca_tumor_fex else None,
    'esca_tumor_f_extreme_sd': float(np.std(esca_tumor_fex, ddof=1)) if len(esca_tumor_fex) > 1 else None,
    'esca_normal_f_extreme_mean': float(np.mean(esca_normal_fex)) if esca_normal_fex else None,
    'esca_normal_f_extreme_sd': float(np.std(esca_normal_fex, ddof=1)) if len(esca_normal_fex) > 1 else None,
    'kirc_prad_anchor_f_extreme_mean': float(np.mean(anchor_fex)) if anchor_fex else None,
    'kirc_prad_anchor_f_extreme_sd': float(np.std(anchor_fex, ddof=1)) if len(anchor_fex) > 1 else None,
    'esca_tumor_minus_anchor_pp': (np.mean(esca_tumor_fex) - np.mean(anchor_fex)) * 100 if esca_tumor_fex and anchor_fex else None,
    'esca_normal_minus_anchor_pp': (np.mean(esca_normal_fex) - np.mean(anchor_fex)) * 100 if esca_normal_fex and anchor_fex else None,
}
if substrate['esca_tumor_f_extreme_mean'] and substrate['kirc_prad_anchor_f_extreme_sd']:
    diff_sd = (substrate['esca_tumor_f_extreme_mean'] - substrate['kirc_prad_anchor_f_extreme_mean']) / substrate['kirc_prad_anchor_f_extreme_sd']
    substrate['shift_in_anchor_sd_units'] = float(diff_sd)
    if abs(diff_sd) >= 3: substrate['baseline_tier'] = 'tier_3_invalidate_cross_cohort'
    elif abs(diff_sd) >= 1: substrate['baseline_tier'] = 'tier_2_baseline_mismatch_flag'
    else: substrate['baseline_tier'] = 'tier_1_report_only'

# === Build results ===
results = {
    'val_id': 'VAL-127',
    'val_title': 'TCGA-ESCA Phase C run-everything (gastric+esophageal-epic v0.1)',
    'prereg_sha256': PREREG_SHA,
    'cohort_summary': {
        'total': len(esca_rows),
        'tumor': len(tumor_rows),
        'normal_paired': len(normal_rows),
        'metastatic': len(mets_rows),
        'histology_distribution': {
            'ESCC': sum(1 for r in tumor_rows if r['histology'] == 'ESCC'),
            'EAC': sum(1 for r in tumor_rows if r['histology'] == 'EAC'),
        },
        'subtype_distribution': dict((s, sum(1 for r in tumor_rows if r.get('SUBTYPE') == s))
                                       for s in set(r.get('SUBTYPE') for r in tumor_rows) if s),
        'barrett_distribution': {
            'Yes': sum(1 for r in tumor_rows if r['barrett_status'] == 'Yes'),
            'No': sum(1 for r in tumor_rows if r['barrett_status'] == 'No'),
            'NotReported': sum(1 for r in tumor_rows if r['barrett_status'] == 'NotReported'),
        },
        'smoking_distribution': {
            s: sum(1 for r in tumor_rows if r['smoking_status'] == s)
            for s in ['Lifelong_non', 'Current', 'Reformed_ge15yr', 'Reformed_lt15yr', 'NotReported']
        },
        'msi_h_count': sum(1 for r in tumor_rows if r['msi_status'] == 'MSI-H'),
        'mss_count': sum(1 for r in tumor_rows if r['msi_status'] == 'MSS'),
        'sex_male': sum(1 for r in tumor_rows if r.get('gender') == 'male'),
        'sex_female': sum(1 for r in tumor_rows if r.get('gender') == 'female'),
    },
    'anchor_cohort': {
        'description': 'TCGA-KIRC + TCGA-PRAD adjacent-normal HM450 sesame Level 3 (shared with VAL-126)',
        'kirc_n': sum(1 for r in anchor_rows if r.get('project') == 'TCGA-KIRC'),
        'prad_n': sum(1 for r in anchor_rows if r.get('project') == 'TCGA-PRAD'),
        'total': len(anchor_rows),
    },
    'substrate_baseline_check_chk_3_2': substrate,
    'per_tile_results_primary': per_tile_primary,
    'subtype_discrimination_ESCC_vs_EAC': subtype_disc,
}

with open(OUTPUT / 'VAL-127_phase_c_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\n  Wrote VAL-127_phase_c_results.json ({len(per_tile_primary)} tiles + ESCC/EAC discrimination)')

with open(OUTPUT / 'VAL-127_stratified_results.json', 'w') as f:
    json.dump({'val_id': 'VAL-127', 'prereg_sha256': PREREG_SHA, 'stratifications': stratifications}, f, indent=2, default=str)
print(f'  Wrote VAL-127_stratified_results.json ({len(stratifications)} strata)')

# Per-sample CSV
all_keys = sorted(set(k for r in esca_rows for k in r))
with open(OUTPUT / 'VAL-127_per_sample.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
    w.writeheader()
    for r in esca_rows: w.writerow(r)
print(f'  Wrote VAL-127_per_sample.csv ({len(esca_rows)} rows × {len(all_keys)} cols)')

# === Headline ===
print('\n' + '='*80)
print('VAL-127 HEADLINE — TCGA-ESCA tumor (n=185) vs KIRC+PRAD anchor (n=210)')
print('='*80)
print(f'\n  Substrate baseline (CHK-3.2):')
print(f'    ESCA tumor f_extreme = {substrate["esca_tumor_f_extreme_mean"]:.4f} ± {substrate["esca_tumor_f_extreme_sd"]:.4f}')
print(f'    ESCA normal f_extreme = {substrate["esca_normal_f_extreme_mean"]:.4f}')
print(f'    KIRC+PRAD anchor     = {substrate["kirc_prad_anchor_f_extreme_mean"]:.4f}')
print(f'    Shift in anchor-SD   = {substrate.get("shift_in_anchor_sd_units"):.2f}')
print(f'    Tier                 = {substrate.get("baseline_tier")}')

print(f'\n  Stage 1 — Xu-538 architectural drift:')
v = per_tile_primary['A_xu538_stage1']
print(f'    All ESCA tumor: d={v["d_vs_KIRC_PRAD_anchor"]:+.3f}, p={v["p_value"]:.2e} → {v["outcome_class"]}')
escc_v = stratifications['histology_ESCC']['tiles']['A_xu538_stage1']
eac_v = stratifications['histology_EAC']['tiles']['A_xu538_stage1']
print(f'    ESCC (n={stratifications["histology_ESCC"]["n"]}): d={escc_v["d"]:+.3f}, p={escc_v["p"]:.2e}')
print(f'    EAC  (n={stratifications["histology_EAC"]["n"]}): d={eac_v["d"]:+.3f}, p={eac_v["p"]:.2e}')

print(f'\n  ESCC vs EAC subtype discrimination — HEADLINE TEST:')
disc_xu = subtype_disc['A_xu538_stage1']
print(f'    Stage 1: d_ESCC-EAC = {disc_xu["d_ESCC_minus_EAC"]:+.3f}, p={disc_xu["p_value"]:.2e}')
print(f'             ESCC vs anchor d = {disc_xu["d_ESCC_vs_anchor"]:+.3f}')
print(f'             EAC  vs anchor d = {disc_xu["d_EAC_vs_anchor"]:+.3f}')

# Discriminating tiles
disc_tiles = sorted([(k, v) for k, v in subtype_disc.items() if v.get('d_ESCC_minus_EAC') is not None and abs(v['d_ESCC_minus_EAC']) >= 0.5],
                    key=lambda x: -abs(x[1]['d_ESCC_minus_EAC']))
print(f'\n  Top discriminating tiles (|d_ESCC-EAC| ≥ 0.5):')
for tile, v in disc_tiles[:15]:
    print(f'    {tile:35s}: d_ESCC-EAC={v["d_ESCC_minus_EAC"]:+.3f} (ESCC vs anchor={v["d_ESCC_vs_anchor"]:+.3f}, EAC vs anchor={v["d_EAC_vs_anchor"]:+.3f})')

# EsoRef squamous tiles
print(f'\n  EsoRef squamous tiles — ESCC (squamous) vs EAC (adenocarcinoma):')
for tile in ['A_esoref_Epi_basal', 'A_esoref_Epi_stratified', 'A_esoref_Epi_suprabasal', 'A_esoref_Epi_upper']:
    v = subtype_disc.get(tile, {})
    d = v.get('d_ESCC_minus_EAC')
    if d is not None:
        print(f'    {tile:30s}: d_ESCC-EAC={d:+.3f}, ESCC vs anchor={v["d_ESCC_vs_anchor"]:+.3f}, EAC vs anchor={v["d_EAC_vs_anchor"]:+.3f}')

# Barrett's stratification
if 'barrett_Yes' in stratifications:
    print(f'\n  Barrett-positive vs Barrett-negative Stage 1:')
    bp = stratifications['barrett_Yes']['tiles']['A_xu538_stage1']
    bn = stratifications['barrett_No']['tiles']['A_xu538_stage1']
    print(f'    Barrett+ (n={stratifications["barrett_Yes"]["n"]}): d vs anchor = {bp["d"]:+.3f}')
    print(f'    Barrett− (n={stratifications["barrett_No"]["n"]}): d vs anchor = {bn["d"]:+.3f}')

# Smoking
print(f'\n  Smoking strata Stage 1:')
for status in ['Current', 'Reformed_ge15yr', 'Reformed_lt15yr', 'Lifelong_non']:
    if f'smoking_{status}' in stratifications:
        v = stratifications[f'smoking_{status}']['tiles']['A_xu538_stage1']
        n = stratifications[f'smoking_{status}']['n']
        print(f'    {status:18s} (n={n:3d}): d vs anchor = {v["d"]:+.3f}')

print('\nDone.')
