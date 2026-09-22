#!/usr/bin/env python3
"""
VAL-126 results rebuilder with proper KIRC+PRAD anchor.

Inputs:
  val126_per_sample_progress.ndjson  — 397 STAD samples, 93 A-scores each
  val106_anchor_per_sample.ndjson    — 210 KIRC+PRAD adj-normal samples, same atlas pipeline
  tcga_stad_hm450_manifest_FINAL.json — clinical metadata + subtype joins

Outputs:
  VAL-126_phase_c_results.json      — primary d-values + outcome classes
  VAL-126_stratified_results.json   — per-subtype, per-Lauren, per-MSI, per-sex, per-H_pylori d-values
  VAL-126_per_sample.csv            — flat tabular per-sample data (CHK-7.6)
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

OUTPUT = Path('/home/claude/gastric_esophageal_sprint/VAL-126_stad_phase_c')

PREREG_SHA = '8f47ba2e725319e116ce4fda24e49e1c3ba2fa3936142cce9d54c45584590cd3'

# Load STAD samples
print('Loading STAD per-sample data...')
stad_rows = []
with open(OUTPUT / 'val126_per_sample_progress.ndjson') as f:
    for line in f:
        stad_rows.append(json.loads(line))
print(f'  STAD: {len(stad_rows)} samples')

# Load anchor
print('Loading KIRC+PRAD anchor per-sample data...')
anchor_rows = []
with open(OUTPUT / 'val106_anchor_per_sample.ndjson') as f:
    for line in f:
        anchor_rows.append(json.loads(line))
print(f'  Anchor: {len(anchor_rows)} samples')

# Load STAD manifest with clinical
with open(OUTPUT / 'tcga_stad_hm450_manifest_FINAL.json') as f:
    manifest = json.load(f)
fid_to_clin = {r['file_id']: r for r in manifest}

# Attach clinical metadata to STAD rows
for row in stad_rows:
    fid = row.get('file_id')
    clin = fid_to_clin.get(fid, {})
    row['SUBTYPE'] = clin.get('SUBTYPE')
    row['MSI_SENSOR_SCORE'] = clin.get('MSI_SENSOR_SCORE')
    row['MSI_SCORE_MANTIS'] = clin.get('MSI_SCORE_MANTIS')
    row['H_PYLORI_INFECTION'] = clin.get('H_PYLORI_INFECTION')
    row['EBV_PRESENT'] = clin.get('EBV_PRESENT')
    row['gender'] = clin.get('gender')
    row['primary_diagnosis'] = clin.get('primary_diagnosis')
    row['ajcc_pathologic_stage'] = clin.get('ajcc_pathologic_stage')
    row['site_of_resection'] = clin.get('site_of_resection')
    row['age_at_index'] = clin.get('age_at_index')

# Compute Lauren category
def lauren_class(pd_diagnosis):
    if pd_diagnosis is None:
        return 'NotReported'
    s = pd_diagnosis or ''
    if s in ('Adenocarcinoma, intestinal type', 'Tubular adenocarcinoma', 'Papillary adenocarcinoma, NOS'):
        return 'intestinal_pooled'
    if s in ('Carcinoma, diffuse type', 'Signet ring cell carcinoma'):
        return 'diffuse_pooled'
    if s == 'Mucinous adenocarcinoma':
        return 'mucinous'
    if s == 'Adenocarcinoma, NOS':
        return 'adenoNOS'
    return 'other'

for row in stad_rows:
    row['lauren_class'] = lauren_class(row.get('primary_diagnosis'))

# MSI status
def msi_class(score):
    if score is None or score == '' or score == 'NA':
        return 'NotAvailable'
    try:
        return 'MSI-H' if float(score) >= 4.0 else 'MSS'
    except (ValueError, TypeError):
        return 'NotAvailable'

for row in stad_rows:
    row['msi_status'] = msi_class(row.get('MSI_SENSOR_SCORE'))

# H. pylori
def pylori_class(v):
    if v in ('Yes', 'No'):
        return v
    return 'NotReported'

for row in stad_rows:
    row['pylori_status'] = pylori_class(row.get('H_PYLORI_INFECTION'))

# === Cohen's d (Welch unpaired) ===
def welch_d(x, y):
    """Welch unpaired Cohen's d. Returns (d, n_x, n_y, t_stat, p_value, ci_low, ci_high)."""
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
    # Welch t
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
    # CI on d via approximate SE per Hedges & Olkin
    se_d = math.sqrt((nx + ny) / (nx * ny) + d**2 / (2 * (nx + ny)))
    ci_low = d - 1.96 * se_d
    ci_high = d + 1.96 * se_d
    return float(d), nx, ny, float(t_stat), float(p_val), float(ci_low), float(ci_high)


def outcome_class(d, ci_low, ci_high, expected_direction=None):
    """Assign outcome class per prereg pre-locked thresholds."""
    if d is None:
        return 'NULL_INSUFFICIENT_DATA'
    abs_d = abs(d)
    if abs_d >= 0.5:
        if d > 0:
            label = 'DIFFERENTIATING_POSITIVE'
        else:
            label = 'DIFFERENTIATING_NEGATIVE'
        # Add unexpected flag if direction conflicts with pre-locked expectation
        if expected_direction == 'POSITIVE' and d < 0:
            label += '_UNEXPECTED'
        elif expected_direction == 'NEGATIVE' and d > 0:
            label += '_UNEXPECTED'
        elif expected_direction == 'NULL':
            label += '_CROSS_TISSUE_OVERREAD'  # the EsoRef/OEref squamous-on-adeno test
        return label
    if abs_d >= 0.2:
        return 'PARTIAL'
    return 'NULL'


# Pre-locked direction expectations per VAL-126 prereg section 2
EXPECTED = {
    # Stage 1
    'A_xu538_stage1': 'POSITIVE',
    # Boccellato — cell-of-origin (NEGATIVE)
    'A_bocc_Antrum_undiff': 'NEGATIVE', 'A_bocc_Antrum_diff': 'NEGATIVE',
    'A_bocc_Corpus_undiff': 'NEGATIVE', 'A_bocc_Corpus_diff': 'NEGATIVE',
    'A_bocc_Fundus_undiff': 'NEGATIVE', 'A_bocc_Fundus_diff': 'NEGATIVE',
    # Loyfer cell-of-origin
    'A_loyfer_Upper_GI': 'NEGATIVE',
    'A_loyfer_Colon_epithelial_cells': 'NEGATIVE',
    # Loyfer homogenization-positive
    'A_loyfer_Bladder': 'POSITIVE', 'A_loyfer_Lung_cells': 'POSITIVE',
    'A_loyfer_Hepatocytes': 'POSITIVE', 'A_loyfer_Pancreatic_beta_cells': 'POSITIVE',
    # EsoRef squamous tiles — NULL (cross-tissue test)
    'A_esoref_Epi_basal': 'NULL', 'A_esoref_Epi_stratified': 'NULL',
    'A_esoref_Epi_suprabasal': 'NULL', 'A_esoref_Epi_upper': 'NULL',
    # OEref Basal — NULL (oral squamous on gastric)
    'A_oeref_Basal': 'NULL',
}

# === Per-tile primary d-values: STAD tumor (n=395) vs KIRC+PRAD anchor (n=210) ===
print('\nComputing per-tile primary d (tumor vs anchor)...')
tumor_rows = [r for r in stad_rows if r.get('sample_type') == 'Primary Tumor']
normal_rows = [r for r in stad_rows if r.get('sample_type') == 'Solid Tissue Normal']
print(f'  Tumor n={len(tumor_rows)}, STAD-normal n={len(normal_rows)}, anchor n={len(anchor_rows)}')

# All atlas A-score columns
a_cols = sorted(set(k for r in stad_rows for k in r if k.startswith('A_')))
print(f'  A-score columns: {len(a_cols)}')

per_tile_primary = {}
for col in a_cols:
    tumor_vals = [r.get(col) for r in tumor_rows]
    anchor_vals = [r.get(col) for r in anchor_rows]
    norm_vals = [r.get(col) for r in normal_rows]
    
    d, n_t, n_a, t, p, ci_l, ci_h = welch_d(tumor_vals, anchor_vals)
    
    # n=2 STAD normal descriptive
    d_norm = None
    if len(norm_vals) >= 2:
        d_norm, _, _, _, _, _, _ = welch_d(tumor_vals, norm_vals)
    
    expected = EXPECTED.get(col)
    oc = outcome_class(d, ci_l, ci_h, expected_direction=expected)
    
    per_tile_primary[col] = {
        'd_vs_KIRC_PRAD_anchor': d,
        'n_tumor': n_t,
        'n_anchor': n_a,
        'tumor_mean': float(np.mean([v for v in tumor_vals if v is not None])) if any(v is not None for v in tumor_vals) else None,
        'tumor_sd': float(np.std([v for v in tumor_vals if v is not None], ddof=1)) if sum(1 for v in tumor_vals if v is not None) >= 2 else None,
        'anchor_mean': float(np.mean([v for v in anchor_vals if v is not None])) if any(v is not None for v in anchor_vals) else None,
        'anchor_sd': float(np.std([v for v in anchor_vals if v is not None], ddof=1)) if sum(1 for v in anchor_vals if v is not None) >= 2 else None,
        'ci_95_low': ci_l,
        'ci_95_high': ci_h,
        't_stat': t,
        'p_value': p,
        'd_vs_STAD_normal_n2': d_norm,
        'expected_direction': expected,
        'outcome_class': oc,
    }

# === Stratified analyses ===
print('\nComputing stratified d-values...')

stratifications = {}

# By SUBTYPE
for subtype in ['STAD_CIN', 'STAD_MSI', 'STAD_GS', 'STAD_EBV', 'STAD_POLE']:
    subset = [r for r in tumor_rows if r.get('SUBTYPE') == subtype]
    if not subset:
        continue
    stratifications[f'subtype_{subtype}'] = {'n': len(subset), 'tiles': {}}
    for col in a_cols:
        sv = [r.get(col) for r in subset]
        av = [r.get(col) for r in anchor_rows]
        d, n_t, n_a, t, p, ci_l, ci_h = welch_d(sv, av)
        stratifications[f'subtype_{subtype}']['tiles'][col] = {'d': d, 'n': n_t, 'p': p, 'outcome': outcome_class(d, ci_l, ci_h)}

# By MSI status
for status in ['MSI-H', 'MSS']:
    subset = [r for r in tumor_rows if r.get('msi_status') == status]
    stratifications[f'msi_{status}'] = {'n': len(subset), 'tiles': {}}
    for col in a_cols:
        sv = [r.get(col) for r in subset]
        av = [r.get(col) for r in anchor_rows]
        d, n_t, n_a, t, p, ci_l, ci_h = welch_d(sv, av)
        stratifications[f'msi_{status}']['tiles'][col] = {'d': d, 'n': n_t, 'p': p, 'outcome': outcome_class(d, ci_l, ci_h)}

# By Lauren
for lauren in ['intestinal_pooled', 'diffuse_pooled', 'mucinous', 'adenoNOS']:
    subset = [r for r in tumor_rows if r.get('lauren_class') == lauren]
    if not subset:
        continue
    stratifications[f'lauren_{lauren}'] = {'n': len(subset), 'tiles': {}}
    for col in a_cols:
        sv = [r.get(col) for r in subset]
        av = [r.get(col) for r in anchor_rows]
        d, n_t, n_a, t, p, ci_l, ci_h = welch_d(sv, av)
        stratifications[f'lauren_{lauren}']['tiles'][col] = {'d': d, 'n': n_t, 'p': p, 'outcome': outcome_class(d, ci_l, ci_h)}

# By H. pylori
for pylori in ['Yes', 'No']:
    subset = [r for r in tumor_rows if r.get('pylori_status') == pylori]
    if not subset:
        continue
    stratifications[f'pylori_{pylori}'] = {'n': len(subset), 'tiles': {}}
    for col in a_cols:
        sv = [r.get(col) for r in subset]
        av = [r.get(col) for r in anchor_rows]
        d, n_t, n_a, t, p, ci_l, ci_h = welch_d(sv, av)
        stratifications[f'pylori_{pylori}']['tiles'][col] = {'d': d, 'n': n_t, 'p': p, 'outcome': outcome_class(d, ci_l, ci_h)}

# By sex
for sex in ['male', 'female']:
    subset = [r for r in tumor_rows if r.get('gender') == sex]
    if not subset:
        continue
    stratifications[f'sex_{sex}'] = {'n': len(subset), 'tiles': {}}
    for col in a_cols:
        sv = [r.get(col) for r in subset]
        av = [r.get(col) for r in anchor_rows]
        d, n_t, n_a, t, p, ci_l, ci_h = welch_d(sv, av)
        stratifications[f'sex_{sex}']['tiles'][col] = {'d': d, 'n': n_t, 'p': p, 'outcome': outcome_class(d, ci_l, ci_h)}

# Substrate baseline check (CHK-3.2)
stad_normal_fex = [r['f_extreme'] for r in normal_rows if 'f_extreme' in r]
stad_tumor_fex = [r['f_extreme'] for r in tumor_rows if 'f_extreme' in r]
anchor_fex = [r['f_extreme'] for r in anchor_rows if 'f_extreme' in r]
substrate_check = {
    'stad_tumor_f_extreme_mean': float(np.mean(stad_tumor_fex)) if stad_tumor_fex else None,
    'stad_tumor_f_extreme_sd': float(np.std(stad_tumor_fex, ddof=1)) if len(stad_tumor_fex) > 1 else None,
    'stad_normal_f_extreme_mean': float(np.mean(stad_normal_fex)) if stad_normal_fex else None,
    'kirc_prad_anchor_f_extreme_mean': float(np.mean(anchor_fex)) if anchor_fex else None,
    'kirc_prad_anchor_f_extreme_sd': float(np.std(anchor_fex, ddof=1)) if len(anchor_fex) > 1 else None,
    'stad_minus_anchor_pct_pts': (float(np.mean(stad_tumor_fex)) - float(np.mean(anchor_fex))) * 100 if stad_tumor_fex and anchor_fex else None,
    'stad_normal_minus_anchor_pct_pts': (float(np.mean(stad_normal_fex)) - float(np.mean(anchor_fex))) * 100 if stad_normal_fex and anchor_fex else None,
}

# Tier the baseline shift
if substrate_check['stad_tumor_f_extreme_mean'] and substrate_check['kirc_prad_anchor_f_extreme_sd']:
    diff_sd = (substrate_check['stad_tumor_f_extreme_mean'] - substrate_check['kirc_prad_anchor_f_extreme_mean']) / substrate_check['kirc_prad_anchor_f_extreme_sd']
    substrate_check['shift_in_anchor_sd_units'] = float(diff_sd)
    if abs(diff_sd) >= 3:
        substrate_check['baseline_tier'] = 'tier_3_invalidate_cross_cohort'
    elif abs(diff_sd) >= 1:
        substrate_check['baseline_tier'] = 'tier_2_baseline_mismatch_flag'
    else:
        substrate_check['baseline_tier'] = 'tier_1_report_only'

# === Summary ===
results = {
    'val_id': 'VAL-126',
    'val_title': 'TCGA-STAD Phase C run-everything (gastric+esophageal-epic v0.1)',
    'prereg_sha256': PREREG_SHA,
    'cohort_summary': {
        'total': len(stad_rows),
        'tumor': len(tumor_rows),
        'normal_paired': len(normal_rows),
        'subtype_distribution': {
            s: sum(1 for r in tumor_rows if r.get('SUBTYPE') == s)
            for s in ['STAD_CIN', 'STAD_MSI', 'STAD_GS', 'STAD_EBV', 'STAD_POLE']
        },
        'lauren_distribution': {
            l: sum(1 for r in tumor_rows if r.get('lauren_class') == l)
            for l in ['intestinal_pooled', 'diffuse_pooled', 'mucinous', 'adenoNOS', 'other']
        },
        'msi_h_count': sum(1 for r in tumor_rows if r.get('msi_status') == 'MSI-H'),
        'mss_count': sum(1 for r in tumor_rows if r.get('msi_status') == 'MSS'),
        'h_pylori_yes': sum(1 for r in tumor_rows if r.get('pylori_status') == 'Yes'),
        'h_pylori_no': sum(1 for r in tumor_rows if r.get('pylori_status') == 'No'),
        'sex_male': sum(1 for r in tumor_rows if r.get('gender') == 'male'),
        'sex_female': sum(1 for r in tumor_rows if r.get('gender') == 'female'),
    },
    'anchor_cohort': {
        'description': 'TCGA-KIRC + TCGA-PRAD adjacent-normal HM450 sesame Level 3, scored through full VAL-126 atlas pipeline',
        'kirc_n': sum(1 for r in anchor_rows if r.get('project') == 'TCGA-KIRC'),
        'prad_n': sum(1 for r in anchor_rows if r.get('project') == 'TCGA-PRAD'),
        'total': len(anchor_rows),
    },
    'substrate_baseline_check_chk_3_2': substrate_check,
    'per_tile_results_primary': per_tile_primary,
}

with open(OUTPUT / 'VAL-126_phase_c_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\n  Wrote VAL-126_phase_c_results.json ({len(per_tile_primary)} tiles)')

with open(OUTPUT / 'VAL-126_stratified_results.json', 'w') as f:
    json.dump({'val_id': 'VAL-126', 'prereg_sha256': PREREG_SHA, 'stratifications': stratifications}, f, indent=2, default=str)
print(f'  Wrote VAL-126_stratified_results.json ({len(stratifications)} strata)')

# Per-sample CSV (CHK-7.6)
print('\nWriting per-sample CSV...')
all_keys = sorted(set(k for r in stad_rows for k in r))
with open(OUTPUT / 'VAL-126_per_sample.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
    w.writeheader()
    for r in stad_rows:
        w.writerow(r)
print(f'  Wrote VAL-126_per_sample.csv ({len(stad_rows)} rows × {len(all_keys)} cols)')

# Print headline
print('\n' + '='*80)
print('VAL-126 HEADLINE RESULTS — STAD tumor (n=395) vs KIRC+PRAD anchor (n=210)')
print('='*80)
print(f'\n  Substrate baseline shift (CHK-3.2): STAD tumor f_extreme = {substrate_check["stad_tumor_f_extreme_mean"]:.4f}')
print(f'                                       KIRC+PRAD anchor      = {substrate_check["kirc_prad_anchor_f_extreme_mean"]:.4f}')
print(f'                                       difference            = {substrate_check["stad_minus_anchor_pct_pts"]:.2f} pp = {substrate_check["shift_in_anchor_sd_units"]:.2f} anchor-SD')
print(f'                                       tier                  = {substrate_check["baseline_tier"]}')

print(f'\n  Stage 1 — A_xu538_stage1:')
v = per_tile_primary['A_xu538_stage1']
print(f'    d={v["d_vs_KIRC_PRAD_anchor"]:+.4f}, 95% CI [{v["ci_95_low"]:+.3f}, {v["ci_95_high"]:+.3f}], p={v["p_value"]:.2e} → {v["outcome_class"]}')

# Per-subtype Stage 1
print(f'\n  Stage 1 by SUBTYPE:')
for s in ['STAD_EBV', 'STAD_MSI', 'STAD_POLE', 'STAD_CIN', 'STAD_GS']:
    v = stratifications.get(f'subtype_{s}', {}).get('tiles', {}).get('A_xu538_stage1', {})
    n = stratifications.get(f'subtype_{s}', {}).get('n', 0)
    if v.get('d') is not None:
        print(f'    {s:12s} (n={n:3d}): d={v["d"]:+.3f}, p={v["p"]:.2e} → {v["outcome"]}')

# Boccellato
print(f'\n  Boccellato (gastric cell-of-origin, expect NEGATIVE):')
for tile in ['A_bocc_Antrum_undiff', 'A_bocc_Antrum_diff', 'A_bocc_Corpus_undiff', 'A_bocc_Corpus_diff', 'A_bocc_Fundus_undiff', 'A_bocc_Fundus_diff']:
    v = per_tile_primary[tile]
    print(f'    {tile:30s}: d={v["d_vs_KIRC_PRAD_anchor"]:+.3f} → {v["outcome_class"]}')

# EsoRef squamous (cross-tissue test, expect NULL)
print(f'\n  EsoRef squamous tiles (cross-tissue overread test, expect NULL):')
for tile in ['A_esoref_Epi_basal', 'A_esoref_Epi_stratified', 'A_esoref_Epi_suprabasal', 'A_esoref_Epi_upper']:
    v = per_tile_primary[tile]
    print(f'    {tile:30s}: d={v["d_vs_KIRC_PRAD_anchor"]:+.3f} → {v["outcome_class"]}')

# Loyfer homogenization
print(f'\n  Loyfer homogenization-positive tiles (expect POSITIVE):')
for tile in ['A_loyfer_Bladder', 'A_loyfer_Lung_cells', 'A_loyfer_Hepatocytes', 'A_loyfer_Pancreatic_beta_cells']:
    v = per_tile_primary[tile]
    print(f'    {tile:30s}: d={v["d_vs_KIRC_PRAD_anchor"]:+.3f} → {v["outcome_class"]}')

# Loyfer cell-of-origin
print(f'\n  Loyfer cell-of-origin tiles (expect NEGATIVE):')
for tile in ['A_loyfer_Upper_GI', 'A_loyfer_Colon_epithelial_cells']:
    v = per_tile_primary[tile]
    print(f'    {tile:30s}: d={v["d_vs_KIRC_PRAD_anchor"]:+.3f} → {v["outcome_class"]}')

# Salas immune
print(f'\n  Salas IDOL Stage 3 (immune microenvironment):')
for tile in ['A_salas_CD8T', 'A_salas_CD4T', 'A_salas_NK', 'A_salas_Bcell', 'A_salas_Mono', 'A_salas_Neu']:
    v = per_tile_primary[tile]
    print(f'    {tile:30s}: d={v["d_vs_KIRC_PRAD_anchor"]:+.3f} → {v["outcome_class"]}')

print('\nDone.')
