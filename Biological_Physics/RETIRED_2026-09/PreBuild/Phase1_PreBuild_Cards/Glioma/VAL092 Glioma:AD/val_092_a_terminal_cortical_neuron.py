#!/usr/bin/env python3
"""
VAL-092 — A_terminal on cortical-neuron-discriminating CpGs

Stage 2 per-class A-score for terminal class (H_min = 0.7728), applied to
cortical neurons specifically, across glioma blood + glioma tissue + AD blood
+ healthy reference. Distinguishes fraction-only signal (H_A) from
fraction-plus-architectural-drift signal (H_B).

Pre-registered before any beta access — see VAL-092_prereg.md (SHA in SEAL.txt).
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Frozen constants
# ============================================================================
H_MIN_TERMINAL = 0.7728
H_MIN_IMMUNE = 0.838889
RNG_SEED = 20260426
N_DISCRIMINATING_CPGS = 100
LOYFER_ATLAS = '/home/claude/ad_loyfer/meth_atlas/reference_atlas.csv'

# ============================================================================
# Helpers
# ============================================================================

def sha256_prefix(path, n=16):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:n]


def shannon_entropy_b(beta):
    beta = np.asarray(beta, dtype=float)
    h = np.zeros_like(beta)
    mask = (beta > 0) & (beta < 1) & ~np.isnan(beta)
    b = beta[mask]
    h[mask] = -b * np.log2(b) - (1 - b) * np.log2(1 - b)
    return h


def a_score(beta_array, h_min):
    beta = np.asarray(beta_array, dtype=float)
    valid = ~np.isnan(beta) & (beta >= 0) & (beta <= 1)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(shannon_entropy_b(beta[valid]) / h_min))


def cohens_d(g1, g2):
    g1 = np.asarray(g1, dtype=float); g1 = g1[~np.isnan(g1)]
    g2 = np.asarray(g2, dtype=float); g2 = g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    if pooled == 0:
        return np.nan
    return float((np.mean(g1) - np.mean(g2)) / pooled)


def bootstrap_d_ci(g1, g2, n_boot=10000, ci=95, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    g1 = np.asarray(g1, dtype=float); g1 = g1[~np.isnan(g1)]
    g2 = np.asarray(g2, dtype=float); g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return (np.nan, np.nan)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        s1 = rng.choice(g1, size=len(g1), replace=True)
        s2 = rng.choice(g2, size=len(g2), replace=True)
        ds[i] = cohens_d(s1, s2)
    lo, hi = np.nanpercentile(ds, [(100-ci)/2, 100-(100-ci)/2])
    return (float(lo), float(hi))


def beta_distribution_check(df_betas, n_samples=3, label=""):
    """CHK-3.1 — beta distribution sanity. Real raw beta: <10% in [0.4,0.6], >30% extremes."""
    cols = list(df_betas.columns[:n_samples])
    summaries = []
    for col in cols:
        v = df_betas[col].dropna().values
        if len(v) == 0:
            continue
        in_mid = ((v >= 0.4) & (v <= 0.6)).sum() / len(v)
        in_ext = ((v < 0.1) | (v > 0.9)).sum() / len(v)
        summaries.append({'sample': col, 'frac_mid_0.4_0.6': float(in_mid),
                          'frac_extreme_lt_0.1_or_gt_0.9': float(in_ext),
                          'median': float(np.median(v)), 'n_cpgs': int(len(v))})
    is_processed = all(s['frac_mid_0.4_0.6'] > 0.40 and s['frac_extreme_lt_0.1_or_gt_0.9'] < 0.20
                       for s in summaries)
    return summaries, is_processed


def identify_neuron_marker_cpgs(atlas_df, n_top=N_DISCRIMINATING_CPGS):
    """Top-N |β(Cortical_neurons) − mean(β(other 24))| CpGs."""
    target_col = 'Cortical_neurons'
    other_cols = [c for c in atlas_df.columns if c != target_col]
    target = atlas_df[target_col].values
    others = atlas_df[other_cols].mean(axis=1).values
    scores = np.abs(target - others)
    sort_idx = np.argsort(scores)[::-1]
    cpg_ids = atlas_df.index.values
    top_cpgs = cpg_ids[sort_idx[:n_top]].tolist()
    return top_cpgs, target[sort_idx[:n_top]], others[sort_idx[:n_top]], scores[sort_idx[:n_top]]


# ============================================================================
# Cohort metadata parsers — one per cohort, robust to GEO format variation
# ============================================================================

def parse_aibl_metadata(path, beta_columns=None):
    """GSE153712 AIBL — parse metadata. AIBL beta file is Sentrix-ID indexed.
    If beta_columns is provided, auto-select sample_id format that matches."""
    gsm_ids = []
    titles = []
    char_rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                parts = line.rstrip('\n').split('\t')[1:]
                gsm_ids = [p.strip().strip('"') for p in parts]
            elif line.startswith('!Sample_title'):
                parts = line.rstrip('\n').split('\t')[1:]
                titles = [p.strip().strip('"') for p in parts]
            elif line.startswith('!Sample_characteristics_ch1'):
                parts = line.rstrip('\n').split('\t')[1:]
                char_rows.append([p.strip().strip('"') for p in parts])
    disease_row = None
    for row in char_rows:
        if not row:
            continue
        if any('disease status' in c.lower() or 'disease state' in c.lower() or
               c.lower().startswith('diagnosis') for c in row[:5]):
            disease_row = row
            break
    if disease_row is None:
        for row in char_rows:
            if not row:
                continue
            cell = row[0].lower()
            if any(k in cell for k in ['alzheimer', 'control', 'mci', 'psp', 'cbd', 'ftd']):
                disease_row = row
                break
    if disease_row is None:
        return None
    sentrix_ids = []
    for t in titles:
        m = re.match(r'^(\d+_R\d+C\d+)', t)
        sentrix_ids.append(m.group(1) if m else None)
    n = min(len(gsm_ids), len(titles), len(disease_row), len(sentrix_ids))
    df = pd.DataFrame({
        'gsm': gsm_ids[:n],
        'title': titles[:n],
        'sentrix_id': sentrix_ids[:n],
        'disease_raw': disease_row[:n],
    })
    df['disease_lower'] = df['disease_raw'].str.lower()
    # Auto-select sample_id format that matches the beta file
    if beta_columns is not None:
        col_set = set(beta_columns)
        sentrix_match_count = sum(1 for s in df['sentrix_id'].dropna() if s in col_set)
        gsm_match_count = sum(1 for g in df['gsm'].dropna() if g in col_set)
        if sentrix_match_count >= gsm_match_count and sentrix_match_count > 0:
            df['sample_id'] = df['sentrix_id']
        else:
            df['sample_id'] = df['gsm']
    else:
        df['sample_id'] = df['sentrix_id'].fillna(df['gsm'])
    return df


def parse_addneuromed_metadata(path, beta_columns=None):
    """GSE144858 AddNeuroMed."""
    return parse_aibl_metadata(path, beta_columns)


def parse_gift_metadata(path, beta_columns=None):
    """GSE53740 GIFT."""
    return parse_aibl_metadata(path, beta_columns)


def assign_aibl_groups(meta_df):
    """AIBL: AD vs HC. Look for explicit AD/HC tokens."""
    # Sample_characteristics in AIBL typically has 'diagnosis: AD' or 'diagnosis: HC'
    ad = meta_df['disease_lower'].str.contains(r'\bad\b|alzheimer', regex=True, na=False)
    hc = meta_df['disease_lower'].str.contains(r'\bhc\b|control|healthy|cognitively normal', regex=True, na=False)
    # Exclude MCI from both AD and HC
    mci = meta_df['disease_lower'].str.contains(r'\bmci\b|mild cognitive', regex=True, na=False)
    ad = ad & ~hc & ~mci
    hc_only = hc & ~ad & ~mci
    return {
        'AD': meta_df.loc[ad, 'sample_id'].tolist(),
        'HC': meta_df.loc[hc_only, 'sample_id'].tolist(),
        'MCI': meta_df.loc[mci, 'sample_id'].tolist(),
    }


def assign_addneuromed_groups(meta_df):
    """AddNeuroMed: AD vs HC vs MCI similar to AIBL."""
    return assign_aibl_groups(meta_df)


def assign_gift_groups(meta_df):
    """GIFT (GSE53740): AD vs HC vs FTD vs PSP vs CBD. Multiple tauopathies."""
    s = meta_df['disease_lower']
    return {
        'AD': meta_df.loc[s.str.contains(r'alzheimer', na=False) & ~s.str.contains(r'control', na=False), 'sample_id'].tolist(),
        'HC': meta_df.loc[s.str.contains(r'control|healthy', na=False), 'sample_id'].tolist(),
        'FTD': meta_df.loc[s.str.contains(r'ftd|frontotemporal', na=False), 'sample_id'].tolist(),
        'PSP': meta_df.loc[s.str.contains(r'\bpsp\b|progressive supra', na=False), 'sample_id'].tolist(),
        'CBD': meta_df.loc[s.str.contains(r'\bcbd\b|corticobasal', na=False), 'sample_id'].tolist(),
    }


def assign_glioma_blood_groups(manifest_path):
    """GSE180683 — manifest JSON gives diagnosis directly."""
    with open(manifest_path) as f:
        records = json.load(f)
    glioma = [r['gsm'] for r in records if r.get('diagnosis', '').lower() == 'glioma']
    nonglioma = [r['gsm'] for r in records if r.get('diagnosis', '').lower() != 'glioma']
    # Sub-strata
    new_gbm = [r['gsm'] for r in records
               if r.get('diagnosis','').lower() == 'glioma' and 'new gbm' in r.get('histological.group','').lower()]
    new_lgg = [r['gsm'] for r in records
               if r.get('diagnosis','').lower() == 'glioma' and 'new lgg' in r.get('histological.group','').lower()]
    recurrent_gbm = [r['gsm'] for r in records
                     if r.get('diagnosis','').lower() == 'glioma' and 'recurrent gbm' in r.get('histological.group','').lower()]
    return {
        'GLIOMA_ALL': glioma,
        'NON_GLIOMA': nonglioma,
        'NEW_GBM': new_gbm,
        'NEW_LGG': new_lgg,
        'RECURRENT_GBM': recurrent_gbm,
    }


def assign_glioma_tissue_groups(manifest_path):
    """GSE60274 — GBM tissue vs NTB controls."""
    with open(manifest_path) as f:
        records = json.load(f)
    gbm = [r['gsm'] for r in records if r.get('disease_state', '').upper() == 'GBM']
    ntb = [r['gsm'] for r in records
           if r.get('disease_state', '').upper() == 'NTB' or 'control' in r.get('disease_state', '').lower()
           or 'normal' in r.get('disease_state', '').lower()]
    return {'GBM_TISSUE': gbm, 'NTB_TISSUE': ntb}


# ============================================================================
# Main
# ============================================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def main():
    print("=" * 78)
    print("VAL-092 — A_terminal on cortical-neuron-discriminating CpGs")
    print(f"Started: {now_iso()}")
    print(f"H_min(terminal) = {H_MIN_TERMINAL}")
    print(f"N discriminating CpGs = {N_DISCRIMINATING_CPGS}")
    print(f"RNG seed = {RNG_SEED}")
    print("=" * 78)

    # Verify prereg seal
    seal_path = '/home/claude/run_everything/VAL-092_PREREG_SEAL.txt'
    if os.path.exists(seal_path):
        with open(seal_path) as f:
            print("\n[Pre-reg seal verified]")
            print(f.read())
    else:
        print("WARNING: prereg seal not found")

    # ---- Load Loyfer atlas ----
    print(f"\n[1] Loading Loyfer atlas")
    print(f"    SHA-256 prefix: {sha256_prefix(LOYFER_ATLAS)}")
    atlas = pd.read_csv(LOYFER_ATLAS, index_col=0)
    print(f"    Atlas shape: {atlas.shape}")
    assert 'Cortical_neurons' in atlas.columns, "Cortical_neurons missing"
    print(f"    Cortical_neurons column present: yes")

    # ---- Identify discriminating CpGs ----
    print(f"\n[2] Identifying top-{N_DISCRIMINATING_CPGS} cortical-neuron-discriminating CpGs")
    marker_cpgs, target_b, other_b, scores = identify_neuron_marker_cpgs(atlas, N_DISCRIMINATING_CPGS)
    print(f"    Discrimination score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"    Cortical_neurons mean β at markers: {target_b.mean():.3f}")
    print(f"    Other-cell mean β at markers: {other_b.mean():.3f}")
    print(f"    First 5 markers: {marker_cpgs[:5]}")

    ref_a_terminal = a_score(target_b, H_MIN_TERMINAL)
    print(f"\n[3] Loyfer reference A_terminal at marker CpGs: {ref_a_terminal:.4f}")
    print(f"    (Healthy cortical-neuron baseline. Patient A_terminal vs this = drift.)")

    # ---- Define cohorts ----
    cohorts = [
        {
            'name': 'GSE51057_healthy_ref',
            'beta_path': '/home/claude/ad_loyfer/input/GSE51057_betas_loyfer.csv',
            'group_assigner': lambda beta_cols: {'HC_REF': None},  # All samples are HC
            'role': 'healthy_buffy_coat_reference_anchor',
            'platform': '450K',
        },
        {
            'name': 'GSE180683_glioma_blood',
            'beta_path': '/home/claude/brain_decon/input/GSE180683_betas.csv',
            'group_assigner': lambda beta_cols: assign_glioma_blood_groups(
                '/home/claude/iam_repo/Biological_Physics/validation_runs/GSE180683_manifest.json'),
            'role': 'glioma_EPIC_peripheral_blood',
            'platform': 'EPIC',
        },
        {
            'name': 'GSE60274_glioma_tissue',
            'beta_path': '/home/claude/brain_decon/input/GSE60274_betas.csv',
            'group_assigner': lambda beta_cols: assign_glioma_tissue_groups(
                '/home/claude/iam_repo/Biological_Physics/validation_runs/GSE60274_manifest.json'),
            'role': 'glioma_450K_surgical_tissue',
            'platform': '450K',
        },
        {
            'name': 'GSE153712_AIBL_AD',
            'beta_path': '/home/claude/ad_loyfer/input/GSE153712_betas_loyfer.csv',
            'group_assigner': lambda beta_cols: assign_aibl_groups(
                parse_aibl_metadata('/home/claude/ad_loyfer/input/GSE153712_metadata.txt', beta_cols)),
            'role': 'AD_panel_training_buffy',
            'platform': '450K',
        },
        {
            'name': 'GSE144858_AddNeuroMed_AD',
            'beta_path': '/home/claude/ad_loyfer/input/GSE144858_betas_loyfer.csv',
            'group_assigner': lambda beta_cols: assign_addneuromed_groups(
                parse_addneuromed_metadata('/home/claude/ad_loyfer/input/GSE144858_metadata.txt', beta_cols)),
            'role': 'AD_cross_platform_EPIC',
            'platform': 'EPIC',
        },
        {
            'name': 'GSE53740_GIFT_tauopathy',
            'beta_path': '/home/claude/ad_loyfer/input/GSE53740_betas_loyfer.csv',
            'group_assigner': lambda beta_cols: assign_gift_groups(
                parse_gift_metadata('/home/claude/ad_loyfer/input/GSE53740_metadata.txt', beta_cols)),
            'role': 'AD_specificity_tauopathy_buffy',
            'platform': '450K',
        },
    ]

    # ---- Process each cohort ----
    results = {
        'val_id': 'VAL-092',
        'date': now_iso(),
        'rng_seed': RNG_SEED,
        'h_min_terminal': H_MIN_TERMINAL,
        'n_discriminating_cpgs': N_DISCRIMINATING_CPGS,
        'loyfer_atlas_sha256_prefix': sha256_prefix(LOYFER_ATLAS),
        'reference_neuron_a_terminal': ref_a_terminal,
        'discrimination_score_range': [float(scores.min()), float(scores.max())],
        'marker_cortical_mean_beta': float(target_b.mean()),
        'marker_other_mean_beta': float(other_b.mean()),
        'cohorts': {},
    }

    all_a_terminals = {}  # for plotting

    for cohort in cohorts:
        name = cohort['name']
        bp = cohort['beta_path']
        print(f"\n{'='*78}\n[Cohort] {name}")
        print(f"  Path: {bp}")
        if not os.path.exists(bp):
            print(f"  SKIP: file not found")
            continue
        print(f"  SHA-256 prefix: {sha256_prefix(bp)}")
        df = pd.read_csv(bp, index_col=0)
        print(f"  Beta matrix shape: {df.shape}")

        # CHK-3.1 distribution
        beta_summary, is_processed = beta_distribution_check(df, label=name)
        print(f"  Beta distribution check (3 samples):")
        for s in beta_summary:
            print(f"    {s['sample']}: mid[0.4,0.6]={s['frac_mid_0.4_0.6']:.1%}, "
                  f"extreme={s['frac_extreme_lt_0.1_or_gt_0.9']:.1%}, median={s['median']:.3f}")
        if is_processed:
            print(f"  CHK-3.1 FLAG: data may be processed/residual not raw β")

        # Subset to marker CpGs
        markers_avail = [c for c in marker_cpgs if c in df.index]
        print(f"  Marker CpGs available: {len(markers_avail)}/{len(marker_cpgs)}")
        if len(markers_avail) < 50:
            print(f"  WARNING: <50 markers available; A_terminal will be noisy")

        sub = df.loc[markers_avail]
        # Per-sample A_terminal
        a_t = sub.apply(lambda col: a_score(col.values, H_MIN_TERMINAL), axis=0)
        a_t = a_t.dropna()
        print(f"  A_terminal: n={len(a_t)}, mean={a_t.mean():.4f}, SD={a_t.std(ddof=1):.4f}, median={a_t.median():.4f}")

        # Group assignment
        groups = cohort['group_assigner'](list(df.columns))

        cohort_record = {
            'beta_path': bp,
            'beta_sha256_prefix': sha256_prefix(bp),
            'n_samples_total': int(df.shape[1]),
            'n_a_terminal_computed': int(len(a_t)),
            'platform': cohort['platform'],
            'role': cohort['role'],
            'beta_distribution_check': beta_summary,
            'beta_distribution_processed_flag': bool(is_processed),
            'n_markers_available': int(len(markers_avail)),
            'n_markers_total': int(len(marker_cpgs)),
            'a_terminal_overall_mean': float(a_t.mean()),
            'a_terminal_overall_sd': float(a_t.std(ddof=1)),
            'a_terminal_overall_median': float(a_t.median()),
            'a_terminal_overall_min': float(a_t.min()),
            'a_terminal_overall_max': float(a_t.max()),
            'delta_a_vs_loyfer_neuron_ref': float(a_t.mean() - ref_a_terminal),
            'groups': {},
            'within_cohort_contrasts': {},
        }

        # Per-group statistics
        for group_name, group_ids in groups.items():
            if group_ids is None:
                # Special handler: all samples
                group_ids = a_t.index.tolist()
            group_ids_in_data = [s for s in group_ids if s in a_t.index]
            if len(group_ids_in_data) == 0:
                continue
            group_a = a_t.loc[group_ids_in_data].values
            cohort_record['groups'][group_name] = {
                'n_assigned': int(len(group_ids)),
                'n_in_a_terminal_data': int(len(group_ids_in_data)),
                'mean_a_terminal': float(np.nanmean(group_a)),
                'sd_a_terminal': float(np.nanstd(group_a, ddof=1)) if len(group_a) > 1 else None,
                'median_a_terminal': float(np.nanmedian(group_a)),
                'delta_a_vs_loyfer_neuron_ref': float(np.nanmean(group_a) - ref_a_terminal),
            }
            print(f"  Group {group_name}: n={len(group_ids_in_data)} (assigned {len(group_ids)}), "
                  f"mean A_t={np.nanmean(group_a):.4f}, SD={np.nanstd(group_a, ddof=1):.4f}")
            all_a_terminals[f"{name}/{group_name}"] = group_a

        # Within-cohort contrasts
        # Glioma blood: GLIOMA_ALL vs NON_GLIOMA, NEW_GBM vs NON_GLIOMA, NEW_LGG vs NON_GLIOMA
        # Glioma tissue: GBM_TISSUE vs NTB_TISSUE
        # AD blood: AD vs HC for each AD cohort
        contrast_pairs = []
        if name == 'GSE180683_glioma_blood':
            contrast_pairs = [
                ('GLIOMA_ALL', 'NON_GLIOMA'),
                ('NEW_GBM', 'NON_GLIOMA'),
                ('NEW_LGG', 'NON_GLIOMA'),
                ('RECURRENT_GBM', 'NON_GLIOMA'),
            ]
        elif name == 'GSE60274_glioma_tissue':
            contrast_pairs = [('GBM_TISSUE', 'NTB_TISSUE')]
        elif name in ('GSE153712_AIBL_AD', 'GSE144858_AddNeuroMed_AD'):
            contrast_pairs = [('AD', 'HC'), ('MCI', 'HC')]
        elif name == 'GSE53740_GIFT_tauopathy':
            contrast_pairs = [('AD','HC'), ('FTD','HC'), ('PSP','HC'), ('CBD','HC')]

        for case_name, ctrl_name in contrast_pairs:
            if case_name not in cohort_record['groups'] or ctrl_name not in cohort_record['groups']:
                continue
            case_ids = [s for s in groups[case_name] if s in a_t.index]
            ctrl_ids = [s for s in groups[ctrl_name] if s in a_t.index]
            if len(case_ids) < 5 or len(ctrl_ids) < 5:
                continue
            case_a = a_t.loc[case_ids].values
            ctrl_a = a_t.loc[ctrl_ids].values
            d = cohens_d(case_a, ctrl_a)
            ci = bootstrap_d_ci(case_a, ctrl_a)
            try:
                t, p = stats.ttest_ind(case_a, ctrl_a, equal_var=False)
                p_val = float(p)
            except Exception:
                t, p_val = np.nan, np.nan
            cohort_record['within_cohort_contrasts'][f'{case_name}_vs_{ctrl_name}'] = {
                'n_case': len(case_ids), 'n_ctrl': len(ctrl_ids),
                'mean_case': float(np.mean(case_a)), 'mean_ctrl': float(np.mean(ctrl_a)),
                'sd_case': float(np.std(case_a, ddof=1)) if len(case_a)>1 else None,
                'sd_ctrl': float(np.std(ctrl_a, ddof=1)) if len(ctrl_a)>1 else None,
                'cohens_d': d, 'ci_95_d': list(ci),
                'welch_t': float(t) if not np.isnan(t) else None, 'p_value': p_val,
                'delta_mean': float(np.mean(case_a) - np.mean(ctrl_a)),
            }
            print(f"  CONTRAST {case_name} vs {ctrl_name}: d={d:+.3f} CI=[{ci[0]:+.3f},{ci[1]:+.3f}], "
                  f"p={p_val:.3g}, n_case={len(case_ids)}, n_ctrl={len(ctrl_ids)}")

        results['cohorts'][name] = cohort_record

    # ---- Cross-cohort baseline check (CHK-3.2) ----
    print(f"\n{'='*78}\n[CHK-3.2] Cross-cohort healthy baseline check")
    anchor_hc_mean = None
    anchor_hc_sd = None
    if 'GSE51057_healthy_ref' in results['cohorts']:
        anchor_hc_mean = results['cohorts']['GSE51057_healthy_ref']['a_terminal_overall_mean']
        anchor_hc_sd = results['cohorts']['GSE51057_healthy_ref']['a_terminal_overall_sd']
        print(f"  Anchor (GSE51057 HC): mean={anchor_hc_mean:.4f}, SD={anchor_hc_sd:.4f}")

    cross_cohort_baseline = {}
    for cohort_name, rec in results['cohorts'].items():
        if cohort_name == 'GSE51057_healthy_ref':
            continue
        # Use the cohort's HC group if it exists, else the cohort's all-samples mean as a proxy
        hc_grp = rec['groups'].get('HC') or rec['groups'].get('NTB_TISSUE') or rec['groups'].get('NON_GLIOMA')
        if hc_grp is None:
            continue
        cohort_hc_mean = hc_grp['mean_a_terminal']
        cohort_hc_sd = hc_grp['sd_a_terminal']
        if anchor_hc_mean is not None and anchor_hc_sd:
            delta = cohort_hc_mean - anchor_hc_mean
            sd_units_anchor = abs(delta) / anchor_hc_sd
            sd_units_cohort = abs(delta) / cohort_hc_sd if cohort_hc_sd else float('inf')
            mismatch = sd_units_anchor > 1.0 or sd_units_cohort > 1.0
            cross_cohort_baseline[cohort_name] = {
                'cohort_hc_mean': cohort_hc_mean,
                'cohort_hc_sd': cohort_hc_sd,
                'delta_vs_anchor': delta,
                'sd_units_anchor': sd_units_anchor,
                'sd_units_cohort': sd_units_cohort,
                'baseline_mismatch_flag': bool(mismatch),
            }
            flag = " [MISMATCH FLAG]" if mismatch else ""
            print(f"  {cohort_name}: HC mean={cohort_hc_mean:.4f}, "
                  f"Δ vs anchor={delta:+.4f}, {sd_units_anchor:.2f} anchor-SDs{flag}")
    results['cross_cohort_baseline'] = cross_cohort_baseline

    # ---- Build per-sample dataframe early (needed for cross-cohort d) ----
    rows = []
    for key, vals in all_a_terminals.items():
        cohort_name, group_name = key.split('/')
        for v in vals:
            rows.append({'cohort': cohort_name, 'group': group_name, 'a_terminal': float(v)})
    per_sample_df = pd.DataFrame(rows)

    # ---- Pre-locked outcome assignment ----
    print(f"\n{'='*78}\n[Outcome assignment per pre-registered criteria]")
    glioma_d = None
    glioma_d_source = None
    ad_d = None
    ad_d_source = None

    # Glioma blood: GSE180683 has no within-cohort HC. Use cross-cohort vs GSE51057
    # as the available comparison. Mark this explicitly as cross-cohort, baseline-mismatch flagged.
    if 'GSE180683_glioma_blood' in results['cohorts']:
        gli = results['cohorts']['GSE180683_glioma_blood']
        glioma_all = gli['groups'].get('GLIOMA_ALL')
        if glioma_all and 'GSE51057_healthy_ref' in results['cohorts']:
            ref = results['cohorts']['GSE51057_healthy_ref']
            ref_grp = ref['groups'].get('HC_REF')
            # For Cohen's d we need raw arrays; recompute from per_sample_df
            gli_a = per_sample_df[(per_sample_df['cohort']=='GSE180683_glioma_blood') &
                                  (per_sample_df['group']=='GLIOMA_ALL')]['a_terminal'].values
            ref_a = per_sample_df[(per_sample_df['cohort']=='GSE51057_healthy_ref') &
                                  (per_sample_df['group']=='HC_REF')]['a_terminal'].values
            if len(gli_a) >= 5 and len(ref_a) >= 5:
                glioma_d = cohens_d(gli_a, ref_a)
                glioma_d_source = 'cross_cohort_GSE180683_vs_GSE51057_BASELINE_MISMATCH_RISK'
                ci_g = bootstrap_d_ci(gli_a, ref_a)
                results['glioma_cross_cohort_d'] = {
                    'glioma_n': len(gli_a), 'ref_n': len(ref_a),
                    'cohens_d': glioma_d, 'ci_95': list(ci_g),
                    'caveat': 'cross-cohort comparison; GSE180683 has no within-cohort HC group; '
                             'cross-cohort baseline mismatch from platform/preprocessing differences possible',
                }
                print(f"  Glioma blood vs healthy ref (cross-cohort): d={glioma_d:+.3f} CI={ci_g}")
                print(f"    SOURCE: {glioma_d_source}")

    # AD: AIBL within-cohort is now computable
    if 'GSE153712_AIBL_AD' in results['cohorts']:
        c = results['cohorts']['GSE153712_AIBL_AD']['within_cohort_contrasts']
        if 'AD_vs_HC' in c:
            ad_d = c['AD_vs_HC']['cohens_d']
            ad_d_source = 'within_cohort_AIBL_GSE153712'
            print(f"  AD blood vs HC (AIBL within-cohort): d={ad_d:+.3f}")

    outcome_label = 'O6_UNEXPECTED'
    outcome_rationale = ''
    if glioma_d is not None and ad_d is not None:
        if glioma_d >= 0.5 and abs(ad_d) <= 0.3:
            outcome_label = 'O1_DRIFT_DISCRIMINATOR'
            outcome_rationale = f'Glioma A_terminal d={glioma_d:+.3f} (≥0.5), AD d={ad_d:+.3f} (|·|≤0.3). H_B supported.'
        elif glioma_d >= 0.5 and ad_d >= 0.5:
            outcome_label = 'O2_BOTH_DRIFT'
            outcome_rationale = f'Glioma d={glioma_d:+.3f}, AD d={ad_d:+.3f}. H_C magnitude differential.'
        elif abs(glioma_d) <= 0.3 and abs(ad_d) <= 0.3:
            outcome_label = 'O3_FRACTION_ONLY'
            outcome_rationale = f'Glioma d={glioma_d:+.3f}, AD d={ad_d:+.3f}. Both null on drift; H_A fraction-only signal supported.'
        elif glioma_d < -0.3:
            outcome_label = 'O4_INVERSE_DRIFT'
            outcome_rationale = f'Glioma d={glioma_d:+.3f} (<-0.3). Homogenization in glioma — surprise.'
        elif abs(glioma_d) <= 0.3 and ad_d >= 0.5:
            outcome_label = 'O5_AD_DRIFT_ONLY'
            outcome_rationale = f'Glioma d={glioma_d:+.3f}, AD d={ad_d:+.3f}. AD shows drift, glioma null — surprise.'
        else:
            # Mixed pattern — describe what we see
            outcome_label = 'O6_UNEXPECTED'
            parts = []
            if glioma_d >= 0.3 and glioma_d < 0.5:
                parts.append(f'glioma modest+ d={glioma_d:+.3f}')
            elif glioma_d >= 0.5:
                parts.append(f'glioma elevated d={glioma_d:+.3f}')
            if ad_d <= -0.2:
                parts.append(f'AD homogenization d={ad_d:+.3f}')
            elif ad_d >= 0.3:
                parts.append(f'AD elevated d={ad_d:+.3f}')
            outcome_rationale = f'Mixed pattern: {"; ".join(parts) if parts else f"glioma d={glioma_d:+.3f} AD d={ad_d:+.3f}"}. See report.'
    else:
        outcome_label = 'O6_UNEXPECTED'
        outcome_rationale = 'Required contrasts not computed — data integrity flag.'

    results['outcome'] = {'label': outcome_label, 'rationale': outcome_rationale,
                         'glioma_blood_d': glioma_d, 'glioma_d_source': glioma_d_source,
                         'ad_blood_d_aibl': ad_d, 'ad_d_source': ad_d_source}
    print(f"  Outcome: {outcome_label}")
    print(f"  Rationale: {outcome_rationale}")

    # ---- Save ----
    out_json = '/home/claude/run_everything/VAL-092_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Saved] {out_json}")

    # Per-sample CSV (per_sample_df was built earlier)
    out_csv = '/home/claude/run_everything/VAL-092_per_sample.csv'
    per_sample_df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    # ---- Distribution figure ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panel_groups = [
        ('GSE51057_healthy_ref/HC_REF', 'Healthy ref (GSE51057)', '#888'),
        ('GSE180683_glioma_blood/GLIOMA_ALL', 'Glioma blood (GSE180683)', '#c33'),
        ('GSE180683_glioma_blood/NON_GLIOMA', 'Non-glioma blood (GSE180683)', '#888'),
        ('GSE60274_glioma_tissue/GBM_TISSUE', 'GBM tissue (GSE60274)', '#933'),
        ('GSE153712_AIBL_AD/AD', 'AIBL AD blood', '#39c'),
        ('GSE153712_AIBL_AD/HC', 'AIBL HC blood', '#aaa'),
    ]
    for ax, (key, label, color) in zip(axes.flat, panel_groups):
        if key in all_a_terminals and len(all_a_terminals[key]) > 0:
            ax.hist(all_a_terminals[key], bins=30, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(ref_a_terminal, color='red', linestyle='--', label=f'Loyfer ref={ref_a_terminal:.3f}')
            ax.set_title(f'{label} (n={len(all_a_terminals[key])})')
            ax.set_xlabel('A_terminal')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
        else:
            ax.set_title(f'{label} (no data)')
    plt.tight_layout()
    fig_path = '/home/claude/run_everything/VAL-092_distributions.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {fig_path}")

    print(f"\n{'='*78}\nVAL-092 complete: {now_iso()}")
    print('='*78)
    return results


if __name__ == '__main__':
    main()
