#!/usr/bin/env python3
"""
VAL-093 — Full 25-tile per-class A-score on Loyfer atlas at >10yr breast pre-dx window

Run-everything architecture (CCL-033, signed off 2026-04-26): every IDAT runs Stage 2
with all reference atlas cell types regardless of single-tissue gating. Per-class A-score
computed for every cell type tile every IDAT.

This VAL focuses on the >10yr breast pre-diagnostic window (n=47 cases, n=601 HC across
GSE51057 + GSE51032) where VAL-047 Phase 6 reported secretory-class aggregate d=-1.226.
Asks: at the per-tile level, which tissue tile is responsible for the secretory aggregate?
Is the signal localized to breast specifically, or distributed across multiple tissues?

Pre-registered before any β access — see VAL-093_prereg.md (SHA in SEAL.txt).
"""

import hashlib
import json
import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Frozen H_min values per architecture class (G-002 + G-003b MCMC posteriors)
# ============================================================================
H_MIN = {
    'terminal':    0.7728,
    'immune':      0.838889,
    'secretory':   0.843264,
    'cycling':     0.856055,
    'progenitor':  0.852216,
    'stromal':     0.862950,
    'stem_adult':  0.873718,
    'stem_pluri':  0.982166,
}

# Loyfer cell type → architecture class assignments
CELL_TYPE_TO_CLASS = {
    'Cortical_neurons':            'terminal',
    'Left_atrium':                 'terminal',
    'Hepatocytes':                 'secretory',
    'Breast':                      'secretory',
    'Prostate':                    'secretory',
    'Pancreatic_acinar_cells':     'secretory',
    'Pancreatic_duct_cells':       'secretory',
    'Pancreatic_beta_cells':       'secretory',
    'Thyroid':                     'secretory',
    'Bladder':                     'cycling',
    'Colon_epithelial_cells':      'cycling',
    'Lung_cells':                  'cycling',
    'Head_and_neck_larynx':        'cycling',
    'Upper_GI':                    'cycling',
    'Uterus_cervix':               'cycling',
    'Kidney':                      'cycling',
    'Adipocytes':                  'stromal',
    'Vascular_endothelial_cells':  'stromal',
    'Erythrocyte_progenitors':     'progenitor',
    'Monocytes_EPIC':              'immune',
    'B-cells_EPIC':                'immune',
    'CD4T-cells_EPIC':             'immune',
    'NK-cells_EPIC':               'immune',
    'CD8T-cells_EPIC':             'immune',
    'Neutrophils_EPIC':            'immune',
}

LOYFER_ATLAS = '/home/claude/ad_loyfer/meth_atlas/reference_atlas.csv'
GSE51057_BETAS = '/home/claude/ad_loyfer/input/GSE51057_betas_loyfer.csv'
GSE51057_METADATA = '/home/claude/iam_repo/Biological_Physics/validation_runs/cross_population/results/T13_secretory_GSE51057/GSE51057_secretory_per_sample_A.csv'
GSE51032_METADATA = '/home/claude/iam_repo/Biological_Physics/validation_runs/cross_population/results/T14_secretory_GSE51032/GSE51032_secretory_per_sample_A.csv'

# GSE51032 beta path — extract from series matrix (need to derive Loyfer subset)
GSE51032_SERIES = '/home/claude/GSE51032_series_matrix.txt.gz'

N_DISCRIMINATING_CPGS = 100
RNG_SEED = 20260426
ANCHOR_PRE_DX_YEARS = 10  # >10yr breast pre-dx window

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


def identify_marker_cpgs(atlas_df, target_col, n_top=N_DISCRIMINATING_CPGS):
    """Top-N CpGs maximizing |β(target) − mean(β(other 24 cell types))|."""
    other_cols = [c for c in atlas_df.columns if c != target_col]
    target = atlas_df[target_col].values
    others = atlas_df[other_cols].mean(axis=1).values
    scores = np.abs(target - others)
    sort_idx = np.argsort(scores)[::-1]
    cpg_ids = atlas_df.index.values
    top_cpgs = cpg_ids[sort_idx[:n_top]].tolist()
    return top_cpgs, target[sort_idx[:n_top]], others[sort_idx[:n_top]], scores[sort_idx[:n_top]]


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def extract_gse51032_betas_loyfer_subset(series_path, loyfer_cpgs, output_path):
    """Extract β values from GSE51032 series matrix at Loyfer atlas CpGs only.
    The series matrix is ~3.1GB so we stream it line-by-line to avoid OOM."""
    print(f"Extracting GSE51032 betas at Loyfer CpGs from {series_path} ...")
    import gzip
    loyfer_set = set(loyfer_cpgs)
    rows_kept = []
    sample_ids = None
    in_matrix = False
    with gzip.open(series_path, 'rt') as f:
        for i, line in enumerate(f):
            line = line.rstrip('\n')
            if line.startswith('!series_matrix_table_begin'):
                in_matrix = True
                continue
            if line.startswith('!series_matrix_table_end'):
                break
            if not in_matrix:
                continue
            if sample_ids is None:
                # First line in matrix block is the header
                sample_ids = line.split('\t')[1:]
                # Strip quotes
                sample_ids = [s.strip().strip('"') for s in sample_ids]
                continue
            parts = line.split('\t')
            cpg = parts[0].strip().strip('"')
            if cpg in loyfer_set:
                rows_kept.append(parts)
    print(f"  Found {len(rows_kept)} Loyfer CpGs in GSE51032 series matrix")
    print(f"  GSE51032 sample count: {len(sample_ids)}")
    # Build dataframe
    cols = ['CpGs'] + sample_ids
    df = pd.DataFrame(rows_kept, columns=cols)
    df.set_index('CpGs', inplace=True)
    # Convert to numeric
    df = df.replace('null', np.nan).replace('NULL', np.nan).replace('', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.to_csv(output_path)
    print(f"  Saved: {output_path} ({df.shape})")
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 78)
    print("VAL-093 — Full 25-tile per-class A-score at >10yr breast pre-dx window")
    print(f"Started: {now_iso()}")
    print(f"H_min table: {H_MIN}")
    print(f"N_DISCRIMINATING_CPGS = {N_DISCRIMINATING_CPGS}")
    print(f"RNG seed = {RNG_SEED}")
    print("=" * 78)

    # Verify prereg seal
    seal_path = '/home/claude/run_everything/VAL-093_PREREG_SEAL.txt'
    if os.path.exists(seal_path):
        with open(seal_path) as f:
            print("\n[Pre-reg seal verified]")
            print(f.read())

    # ---- Load Loyfer atlas ----
    print(f"\n[1] Loading Loyfer atlas")
    print(f"    SHA-256: {sha256_prefix(LOYFER_ATLAS, 16)}")
    atlas = pd.read_csv(LOYFER_ATLAS, index_col=0)
    print(f"    Atlas shape: {atlas.shape}")
    cell_types = list(atlas.columns)
    print(f"    Cell types ({len(cell_types)}):")
    for ct in cell_types:
        cls = CELL_TYPE_TO_CLASS.get(ct)
        h = H_MIN.get(cls)
        print(f"      {ct:30s} → {cls or 'UNKNOWN':10s} → H_min = {h}")

    # ---- Identify per-tile marker CpGs ----
    print(f"\n[2] Identifying top-{N_DISCRIMINATING_CPGS} discriminating CpGs per cell type")
    tile_markers = {}
    for ct in cell_types:
        markers, target_b, other_b, scores = identify_marker_cpgs(atlas, ct, N_DISCRIMINATING_CPGS)
        tile_markers[ct] = markers
    print(f"    Done — {len(tile_markers)} tiles with markers identified")

    # ---- Load GSE51057 (Phase 9 cohort) ----
    print(f"\n[3] Loading GSE51057 (Phase 9, EPIC-Italy buffy coat 450K)")
    print(f"    Beta path: {GSE51057_BETAS}, SHA-256: {sha256_prefix(GSE51057_BETAS)}")
    g51057_betas = pd.read_csv(GSE51057_BETAS, index_col=0)
    print(f"    Beta matrix shape: {g51057_betas.shape}")
    print(f"    Metadata path: {GSE51057_METADATA}")
    g51057_meta = pd.read_csv(GSE51057_METADATA)
    print(f"    Metadata rows: {len(g51057_meta)}")

    # ---- Extract or load GSE51032 (Phase 12 cohort) ----
    print(f"\n[4] GSE51032 (Phase 12, EPIC-Italy nested-case-control 450K)")
    g51032_loyfer_path = '/home/claude/run_everything/GSE51032_betas_loyfer.csv'
    if os.path.exists(g51032_loyfer_path):
        print(f"    Loading cached Loyfer subset: {g51032_loyfer_path}")
        g51032_betas = pd.read_csv(g51032_loyfer_path, index_col=0)
        print(f"    Beta matrix shape: {g51032_betas.shape}")
    else:
        # Extract from series matrix
        loyfer_cpgs = atlas.index.tolist()
        g51032_betas = extract_gse51032_betas_loyfer_subset(
            GSE51032_SERIES, loyfer_cpgs, g51032_loyfer_path
        )
    print(f"    Metadata path: {GSE51032_METADATA}")
    g51032_meta = pd.read_csv(GSE51032_METADATA)
    print(f"    Metadata rows: {len(g51032_meta)}")

    # ---- Compute per-tile A-scores for every patient in both cohorts ----
    print(f"\n[5] Computing per-tile A-scores for every patient")

    def compute_per_tile_a_scores(beta_matrix, meta_df, cohort_label, cohort_meta_id_col='gsm'):
        """For each sample, compute A-score per Loyfer cell type tile."""
        results_rows = []
        sample_to_meta = {row[cohort_meta_id_col]: row for _, row in meta_df.iterrows()}
        for sample_id in beta_matrix.columns:
            meta = sample_to_meta.get(sample_id)
            if meta is None:
                continue
            row = {'sample_id': sample_id, 'cohort': cohort_label}
            for col in ['age', 'group', 'cancer_site', 'ttd_years']:
                row[col] = meta.get(col)
            for ct in cell_types:
                markers = tile_markers[ct]
                avail = [c for c in markers if c in beta_matrix.index]
                if len(avail) < 20:
                    row[f'A_{ct}'] = np.nan
                    continue
                cls = CELL_TYPE_TO_CLASS.get(ct)
                hmin = H_MIN.get(cls)
                if hmin is None:
                    row[f'A_{ct}'] = np.nan
                    continue
                vals = beta_matrix.loc[avail, sample_id].values
                row[f'A_{ct}'] = a_score(vals, hmin)
            results_rows.append(row)
        return pd.DataFrame(results_rows)

    print(f"    Computing GSE51057 (Phase 9) ...")
    g51057_a = compute_per_tile_a_scores(g51057_betas, g51057_meta, 'GSE51057')
    print(f"    Computing GSE51032 (Phase 12) ...")
    g51032_a = compute_per_tile_a_scores(g51032_betas, g51032_meta, 'GSE51032')
    print(f"    GSE51057 A-score rows: {len(g51057_a)}")
    print(f"    GSE51032 A-score rows: {len(g51032_a)}")

    combined = pd.concat([g51057_a, g51032_a], ignore_index=True)
    combined.to_csv('/home/claude/run_everything/VAL-093_per_sample.csv', index=False)
    print(f"    Saved: VAL-093_per_sample.csv")

    # ---- Stratify >10yr breast pre-dx window ----
    print(f"\n[6] Stratifying >10yr breast pre-dx window")

    def get_breast_predx_10yr_plus(df):
        breast_mask = (df['cancer_site'] == 'c50')
        ttd_mask = pd.to_numeric(df['ttd_years'], errors='coerce') > ANCHOR_PRE_DX_YEARS
        return df[breast_mask & ttd_mask]

    def get_hc(df, cohort_label):
        if cohort_label == 'GSE51057':
            return df[df['group'] == 'control']
        else:  # GSE51032
            return df[df['group'] == 'control']

    # Per-cohort breast >10yr cases vs HC
    g51057_breast_10yr = get_breast_predx_10yr_plus(g51057_a)
    g51057_hc = get_hc(g51057_a, 'GSE51057')
    g51032_breast_10yr = get_breast_predx_10yr_plus(g51032_a)
    g51032_hc = get_hc(g51032_a, 'GSE51032')

    print(f"    GSE51057: n={len(g51057_breast_10yr)} breast >10yr cases vs n={len(g51057_hc)} HC")
    print(f"    GSE51032: n={len(g51032_breast_10yr)} breast >10yr cases vs n={len(g51032_hc)} HC")

    # ---- Per-tile within-cohort case-vs-HC contrasts ----
    print(f"\n[7] Per-tile within-cohort case-vs-HC contrasts")
    tile_results = {}
    for ct in cell_types:
        col = f'A_{ct}'
        cls = CELL_TYPE_TO_CLASS.get(ct)
        hmin = H_MIN.get(cls)

        g51057_case_a = g51057_breast_10yr[col].dropna().values
        g51057_hc_a = g51057_hc[col].dropna().values
        g51032_case_a = g51032_breast_10yr[col].dropna().values
        g51032_hc_a = g51032_hc[col].dropna().values

        d_g51057 = cohens_d(g51057_case_a, g51057_hc_a) if len(g51057_case_a) >= 5 else np.nan
        ci_g51057 = bootstrap_d_ci(g51057_case_a, g51057_hc_a) if len(g51057_case_a) >= 5 else (np.nan, np.nan)
        d_g51032 = cohens_d(g51032_case_a, g51032_hc_a) if len(g51032_case_a) >= 5 else np.nan
        ci_g51032 = bootstrap_d_ci(g51032_case_a, g51032_hc_a) if len(g51032_case_a) >= 5 else (np.nan, np.nan)

        # Welch's t-test p-value
        try:
            _, p_g51057 = stats.ttest_ind(g51057_case_a, g51057_hc_a, equal_var=False) if len(g51057_case_a) >= 5 else (None, np.nan)
            p_g51057 = float(p_g51057) if not np.isnan(p_g51057) else np.nan
        except Exception:
            p_g51057 = np.nan
        try:
            _, p_g51032 = stats.ttest_ind(g51032_case_a, g51032_hc_a, equal_var=False) if len(g51032_case_a) >= 5 else (None, np.nan)
            p_g51032 = float(p_g51032) if not np.isnan(p_g51032) else np.nan
        except Exception:
            p_g51032 = np.nan

        tile_results[ct] = {
            'cell_type': ct,
            'class': cls,
            'h_min': hmin,
            'GSE51057': {
                'n_case': len(g51057_case_a), 'n_hc': len(g51057_hc_a),
                'mean_case': float(np.mean(g51057_case_a)) if len(g51057_case_a) else None,
                'mean_hc': float(np.mean(g51057_hc_a)) if len(g51057_hc_a) else None,
                'sd_case': float(np.std(g51057_case_a, ddof=1)) if len(g51057_case_a) > 1 else None,
                'sd_hc': float(np.std(g51057_hc_a, ddof=1)) if len(g51057_hc_a) > 1 else None,
                'd': d_g51057, 'ci_95': list(ci_g51057), 'p': p_g51057,
            },
            'GSE51032': {
                'n_case': len(g51032_case_a), 'n_hc': len(g51032_hc_a),
                'mean_case': float(np.mean(g51032_case_a)) if len(g51032_case_a) else None,
                'mean_hc': float(np.mean(g51032_hc_a)) if len(g51032_hc_a) else None,
                'sd_case': float(np.std(g51032_case_a, ddof=1)) if len(g51032_case_a) > 1 else None,
                'sd_hc': float(np.std(g51032_hc_a, ddof=1)) if len(g51032_hc_a) > 1 else None,
                'd': d_g51032, 'ci_95': list(ci_g51032), 'p': p_g51032,
            },
        }

    # Print results sorted by absolute Phase 9 d
    print(f"\n    Sorted by max(|d_g51057|, |d_g51032|):")
    sorted_tiles = sorted(tile_results.items(),
                          key=lambda kv: max(abs(kv[1]['GSE51057']['d'] or 0), abs(kv[1]['GSE51032']['d'] or 0)),
                          reverse=True)
    print(f"    {'cell_type':<30} {'class':<12} {'GSE51057_d':>12} {'GSE51032_d':>12}")
    for ct, r in sorted_tiles:
        d57 = r['GSE51057']['d']
        d32 = r['GSE51032']['d']
        d57_str = f"{d57:+.3f}" if d57 is not None and not np.isnan(d57) else 'na'
        d32_str = f"{d32:+.3f}" if d32 is not None and not np.isnan(d32) else 'na'
        print(f"    {ct:<30} {r['class']:<12} {d57_str:>12} {d32_str:>12}")

    # ---- CHK-3.2 cross-cohort baseline check ----
    print(f"\n[8] CHK-3.2 cross-cohort baseline check (GSE51057 HC vs GSE51032 HC)")
    cross_cohort_baseline = {}
    for ct in cell_types:
        col = f'A_{ct}'
        anchor_hc = g51057_hc[col].dropna().values
        cohort_hc = g51032_hc[col].dropna().values
        if len(anchor_hc) < 5 or len(cohort_hc) < 5:
            continue
        anchor_mean = float(np.mean(anchor_hc))
        anchor_sd = float(np.std(anchor_hc, ddof=1))
        cohort_mean = float(np.mean(cohort_hc))
        delta = cohort_mean - anchor_mean
        sd_units = abs(delta) / anchor_sd if anchor_sd > 0 else np.inf
        cross_cohort_baseline[ct] = {
            'anchor_mean_HC': anchor_mean, 'anchor_sd_HC': anchor_sd,
            'cohort_mean_HC': cohort_mean,
            'delta': delta, 'sd_units': sd_units,
            'baseline_mismatch_flag': bool(sd_units > 1.0),
        }
    flagged = [ct for ct, r in cross_cohort_baseline.items() if r['baseline_mismatch_flag']]
    print(f"    Tiles with baseline mismatch flag (>1 anchor-SD): {len(flagged)}/{len(cross_cohort_baseline)}")
    for ct in flagged[:10]:
        r = cross_cohort_baseline[ct]
        print(f"      {ct}: Δ={r['delta']:+.4f}, {r['sd_units']:.2f} anchor-SDs [FLAG]")

    # ---- Top-1 ΔA call per patient ----
    print(f"\n[9] Top-1 ΔA call per patient (>10yr breast pre-dx)")
    top1_calls = []
    for cohort_label, cohort_a, hc_df in [('GSE51057', g51057_breast_10yr, g51057_hc),
                                          ('GSE51032', g51032_breast_10yr, g51032_hc)]:
        for _, row in cohort_a.iterrows():
            sample_id = row['sample_id']
            best_tile = None
            best_abs_delta = -1
            for ct in cell_types:
                col = f'A_{ct}'
                a_patient = row.get(col)
                if pd.isna(a_patient):
                    continue
                hc_mean = hc_df[col].mean()
                hc_sd = hc_df[col].std(ddof=1)
                if hc_sd == 0 or pd.isna(hc_sd):
                    continue
                # Z-score the patient relative to HC
                z = (a_patient - hc_mean) / hc_sd
                if abs(z) > best_abs_delta:
                    best_abs_delta = abs(z)
                    best_tile = ct
                    best_z = z
            if best_tile:
                top1_calls.append({
                    'cohort': cohort_label, 'sample_id': sample_id,
                    'top1_tile': best_tile,
                    'top1_class': CELL_TYPE_TO_CLASS.get(best_tile),
                    'top1_z': best_z,
                    'top1_abs_z': best_abs_delta,
                    'ttd_years': row.get('ttd_years'),
                })
    top1_df = pd.DataFrame(top1_calls)
    if len(top1_df) > 0:
        top1_dist = top1_df['top1_tile'].value_counts().to_dict()
        top1_class_dist = top1_df['top1_class'].value_counts().to_dict()
        print(f"    Top-1 tile distribution across {len(top1_df)} >10yr breast pre-dx cases:")
        for tile, count in sorted(top1_dist.items(), key=lambda kv: -kv[1])[:8]:
            cls = CELL_TYPE_TO_CLASS.get(tile)
            print(f"      {tile} ({cls}): {count} ({100*count/len(top1_df):.1f}%)")
        print(f"    Top-1 class distribution: {top1_class_dist}")
        breast_top1_count = top1_dist.get('Breast', 0)
        breast_top1_pct = 100 * breast_top1_count / len(top1_df) if top1_df.shape[0] else 0
        print(f"    Breast as top-1: {breast_top1_count}/{len(top1_df)} = {breast_top1_pct:.1f}%")

    # ---- Save results JSON ----
    results = {
        'val_id': 'VAL-093',
        'date': now_iso(),
        'rng_seed': RNG_SEED,
        'h_min_table': H_MIN,
        'cell_type_to_class': CELL_TYPE_TO_CLASS,
        'n_discriminating_cpgs': N_DISCRIMINATING_CPGS,
        'loyfer_atlas_sha256_prefix': sha256_prefix(LOYFER_ATLAS),
        'pre_dx_window_years': ANCHOR_PRE_DX_YEARS,
        'cohort_summary': {
            'GSE51057': {'n_breast_10yr': int(len(g51057_breast_10yr)), 'n_hc': int(len(g51057_hc))},
            'GSE51032': {'n_breast_10yr': int(len(g51032_breast_10yr)), 'n_hc': int(len(g51032_hc))},
        },
        'tile_results': tile_results,
        'cross_cohort_baseline_check': cross_cohort_baseline,
        'top1_distribution': top1_dist if len(top1_df) > 0 else {},
        'top1_class_distribution': top1_class_dist if len(top1_df) > 0 else {},
        'top1_breast_count': int(breast_top1_count) if len(top1_df) > 0 else 0,
        'top1_breast_pct': float(breast_top1_pct) if len(top1_df) > 0 else 0,
        'top1_total_cases': int(len(top1_df)),
    }

    # ---- Outcome assignment per pre-locked criteria ----
    breast_d_g51057 = tile_results['Breast']['GSE51057']['d']
    breast_d_g51032 = tile_results['Breast']['GSE51032']['d']

    # Find max absolute d per cohort
    def max_abs_d(cohort_key):
        return max((abs(r[cohort_key]['d']) for r in tile_results.values()
                    if r[cohort_key]['d'] is not None and not np.isnan(r[cohort_key]['d'])), default=np.nan)

    max_abs_g51057 = max_abs_d('GSE51057')
    max_abs_g51032 = max_abs_d('GSE51032')

    breast_is_top_g51057 = (abs(breast_d_g51057 or 0) >= max_abs_g51057 - 0.01) if not np.isnan(max_abs_g51057) else False
    breast_is_top_g51032 = (abs(breast_d_g51032 or 0) >= max_abs_g51032 - 0.01) if not np.isnan(max_abs_g51032) else False
    breast_meets_threshold = (abs(breast_d_g51057 or 0) >= 0.5) or (abs(breast_d_g51032 or 0) >= 0.5)

    # Count secretory tiles with |d| >= 0.3 in either cohort
    secretory_tiles = [ct for ct in cell_types if CELL_TYPE_TO_CLASS.get(ct) == 'secretory']
    secretory_pass_count = 0
    for ct in secretory_tiles:
        d57 = tile_results[ct]['GSE51057']['d']
        d32 = tile_results[ct]['GSE51032']['d']
        if (abs(d57 or 0) >= 0.3) or (abs(d32 or 0) >= 0.3):
            secretory_pass_count += 1

    if breast_meets_threshold and (breast_is_top_g51057 or breast_is_top_g51032):
        outcome_label = 'O1_BREAST_LOCALIZED'
        outcome_rationale = f'Breast |d| ≥ 0.5 in at least one cohort and is the largest tile by absolute d.'
    elif secretory_pass_count >= 3 and not (breast_is_top_g51057 and breast_is_top_g51032):
        outcome_label = 'O2_SECRETORY_DISTRIBUTED'
        outcome_rationale = f'{secretory_pass_count}/4 secretory-class tiles with |d| ≥ 0.3; Breast not uniquely top.'
    else:
        # Check non-secretory dominance
        non_secretory_max = max(
            (abs(tile_results[ct]['GSE51057']['d'] or 0) for ct in cell_types
             if CELL_TYPE_TO_CLASS.get(ct) != 'secretory' and tile_results[ct]['GSE51057']['d'] is not None
             and not np.isnan(tile_results[ct]['GSE51057']['d'])), default=0)
        non_secretory_max = max(non_secretory_max,
            max((abs(tile_results[ct]['GSE51032']['d'] or 0) for ct in cell_types
                 if CELL_TYPE_TO_CLASS.get(ct) != 'secretory' and tile_results[ct]['GSE51032']['d'] is not None
                 and not np.isnan(tile_results[ct]['GSE51032']['d'])), default=0))
        breast_max = max(abs(breast_d_g51057 or 0), abs(breast_d_g51032 or 0))
        if non_secretory_max >= 0.5 and non_secretory_max > breast_max:
            outcome_label = 'O3_NON_SECRETORY_DOMINANT'
            outcome_rationale = f'Non-secretory tile |d| = {non_secretory_max:.3f} > Breast |d| = {breast_max:.3f}.'
        elif max_abs_g51057 < 0.5 and max_abs_g51032 < 0.5:
            outcome_label = 'O4_PER_TISSUE_NULL'
            outcome_rationale = f'All tile |d| < 0.5 in both cohorts (max GSE51057={max_abs_g51057:.3f}, max GSE51032={max_abs_g51032:.3f}).'
        elif (breast_d_g51057 or 0) * (breast_d_g51032 or 0) < 0 and abs(breast_d_g51057 or 0) >= 0.5 and abs(breast_d_g51032 or 0) >= 0.3:
            outcome_label = 'O5_BIDIRECTIONAL_PATTERN'
            outcome_rationale = f'Breast d directions diverge: GSE51057={breast_d_g51057:+.3f}, GSE51032={breast_d_g51032:+.3f}.'
        else:
            outcome_label = 'O6_UNEXPECTED'
            outcome_rationale = f'Mixed pattern. Breast d (51057={breast_d_g51057:+.3f}, 51032={breast_d_g51032:+.3f}); top1_breast_pct={breast_top1_pct:.1f}%; secretory pass count={secretory_pass_count}/4.'

    results['outcome'] = {
        'label': outcome_label, 'rationale': outcome_rationale,
        'breast_d_GSE51057': breast_d_g51057, 'breast_d_GSE51032': breast_d_g51032,
        'breast_is_top_tile_GSE51057': breast_is_top_g51057,
        'breast_is_top_tile_GSE51032': breast_is_top_g51032,
        'max_abs_d_GSE51057': max_abs_g51057, 'max_abs_d_GSE51032': max_abs_g51032,
        'secretory_pass_count': secretory_pass_count,
    }

    print(f"\n[10] Outcome: {outcome_label}")
    print(f"     {outcome_rationale}")

    out_json = '/home/claude/run_everything/VAL-093_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n     Saved: {out_json}")

    # ---- Heatmap figure ----
    print(f"\n[11] Generating tile heatmap")
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax_idx, cohort_key in enumerate(['GSE51057', 'GSE51032']):
        ax = axes[ax_idx]
        d_values = []
        labels = []
        colors = []
        cls_color = {'terminal': '#7B1FA2', 'immune': '#1976D2', 'secretory': '#D32F2F',
                     'cycling': '#F57C00', 'progenitor': '#388E3C', 'stromal': '#5D4037',
                     'stem_adult': '#00796B', 'stem_pluri': '#455A64'}
        for ct in cell_types:
            d = tile_results[ct][cohort_key]['d']
            if d is None or np.isnan(d):
                continue
            d_values.append(d)
            labels.append(ct)
            colors.append(cls_color.get(CELL_TYPE_TO_CLASS.get(ct), '#888'))
        # Sort by absolute d descending
        sort_idx = np.argsort(np.abs(d_values))[::-1]
        d_sorted = [d_values[i] for i in sort_idx]
        labels_sorted = [labels[i] for i in sort_idx]
        colors_sorted = [colors[i] for i in sort_idx]
        y_pos = np.arange(len(d_sorted))
        ax.barh(y_pos, d_sorted, color=colors_sorted, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_sorted, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.5)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(-0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel("Cohen's d (>10yr breast pre-dx vs HC)")
        n_case = results['cohort_summary'][cohort_key]['n_breast_10yr']
        n_hc = results['cohort_summary'][cohort_key]['n_hc']
        ax.set_title(f'{cohort_key} (n={n_case} cases, n={n_hc} HC)')
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig_path = '/home/claude/run_everything/VAL-093_tile_heatmap.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"     Saved: {fig_path}")

    print(f"\n{'=' * 78}")
    print(f"VAL-093 complete: {now_iso()}")
    print(f"{'=' * 78}")
    return results


if __name__ == '__main__':
    main()
