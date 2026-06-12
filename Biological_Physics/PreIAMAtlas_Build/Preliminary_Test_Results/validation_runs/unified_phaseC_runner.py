#!/usr/bin/env python3
"""
===============================================================================
Unified bladder-epic v0.1 Phase C runner

Single β-file pass over TCGA-BLCA n=440 produces per-sample outputs for
VAL-120 (Stage 1 Xu-538), VAL-121 (Stage 2 multi-atlas), and VAL-122
(Stage 3 immune fine-tune) simultaneously. This avoids 3× redundant I/O
on the cohort.

Each VAL's prereg-locked logic is preserved bit-for-bit; the only change
from the individual val12{0,1,2}_*.py scripts is that they share the β load.

Outputs (mirrored from the individual scripts):
  VAL-120/VAL-120_per_sample.csv
  VAL-121/VAL-121_per_sample_per_atlas.csv
  VAL-122/VAL-122_per_sample_per_atlas.csv

The per-sample CSVs are the foundation; the statistical contrasts (paired d,
Welch d, outcome class assignment) are then computed in a separate post-pass
step which is sub-second per VAL (no β load needed once per-sample data exists).

Pre-registered:
  VAL-120 prereg.md SHA: 6d1807440dcf6cf33c9abbe791f9260224b768065bdd272f029b6e334d3c6996
  VAL-121 prereg.md SHA: eb68e4d4ca6270cdcce60269375af787537c560fabea18ee31cbaf558dea1962
  VAL-122 prereg.md SHA: 2d101db94cdc7a71466c5f8071a936abd426f85ecd9ea27ae8fa73cd0d81f855
  All three sealed 2026-05-01T03:48:17Z BEFORE any β file read.

H_min anchors (G-002 MCMC frozen 2026-04-06; values from GAPE_WEB_v13.py _H_MIN registry):
  terminal=0.772837 immune=0.838889 secretory=0.843264 cycling=0.856055
  stromal=0.862950 stem_pluri=0.982166

RNG seed: 20260420
===============================================================================
"""

import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

_H_MIN = {
    'terminal': 0.772837,
    'immune': 0.838889,
    'secretory': 0.843264,
    'cycling': 0.856055,
    'stromal': 0.862950,
    'stem_pluri': 0.982166,
}

# Atlas paths
LOYFER_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv')
BLADDERREF_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_bladderref/episcore_bladderref_cpg_bridged.csv')
BLADDERREF_SHA = '3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6'
CAGGIANO_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv')
SALAS_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs450k_compTable.csv')
UNILIFE_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv')
XU538_PANEL = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/xu538_panel.json')

BLCA_DIR = Path('/home/claude/edear_working/bladder_epic/blca_betas')
BLCA_MANIFEST = Path('/home/claude/edear_working/bladder_epic/blca_manifest.json')

OUTPUT_DIR_120 = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-120_bladder_stage1_xu538')
OUTPUT_DIR_121 = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-121_bladder_stage2_multiatlas')
OUTPUT_DIR_122 = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-122_bladder_stage3_immune')

# Loyfer 25 tiles, class assignments
LOYFER_TILE_CLASS = {
    'Monocytes_EPIC':'immune','B-cells_EPIC':'immune','CD4T-cells_EPIC':'immune',
    'NK-cells_EPIC':'immune','CD8T-cells_EPIC':'immune','Neutrophils_EPIC':'immune',
    'Erythrocyte_progenitors':'cycling','Adipocytes':'stromal',
    'Cortical_neurons':'terminal','Hepatocytes':'terminal','Lung_cells':'secretory',
    'Pancreatic_beta_cells':'terminal','Pancreatic_acinar_cells':'secretory',
    'Pancreatic_duct_cells':'cycling','Vascular_endothelial_cells':'stromal',
    'Colon_epithelial_cells':'secretory','Left_atrium':'terminal',
    'Bladder':'secretory','Breast':'secretory','Head_and_neck_larynx':'secretory',
    'Kidney':'secretory','Prostate':'secretory','Thyroid':'secretory',
    'Upper_GI':'secretory','Uterus_cervix':'secretory',
}
LOYFER_TILES = list(LOYFER_TILE_CLASS.keys())

BLADDERREF_TILE_CLASS = {'EC':'stromal','Epi':'secretory','Fib':'stromal','IC':'immune'}
BLADDERREF_TILES = list(BLADDERREF_TILE_CLASS.keys())

CAGGIANO_TILE_CLASS = {
    'dendritic':'immune','endothelial':'stromal','eosinophil':'immune',
    'erythroblast':'cycling','macrophage':'immune','monocyte':'immune',
    'neutrophil':'immune','placenta':'secretory','tcell':'immune',
    'adipose':'stromal','brain':'terminal','fibroblast':'stromal',
    'heart':'terminal','hepatocyte':'terminal','lung':'secretory',
    'mammary':'secretory','megakaryocyte':'cycling','skeletal':'terminal',
    'small_intestine':'secretory',
}
CAGGIANO_TILES = list(CAGGIANO_TILE_CLASS.keys())
CAGGIANO_IMMUNE_TILES = ['dendritic','eosinophil','erythroblast','macrophage','monocyte','neutrophil','tcell','megakaryocyte']

SALAS_TILES = ['CD8T','CD4T','NK','Bcell','Mono','Neu']
UNILIFE_TILES = ['B','CD4T','CD8T','Mono','nRBC','Gran','NK',
                 'aCD4Tnv','aBaso','aCD4Tmem','aBmem','aBnv','aTreg',
                 'aCD8Tmem','aCD8Tnv','aEos','aNK','aNeu','aMono']


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def shannon_H_arr(beta_arr):
    """Vectorized binary Shannon entropy. Returns 0 for β outside (0,1)."""
    out = np.zeros_like(beta_arr, dtype=np.float64)
    valid = (beta_arr > 0) & (beta_arr < 1) & np.isfinite(beta_arr)
    b = beta_arr[valid]
    out[valid] = -b * np.log2(b) - (1 - b) * np.log2(1 - b)
    return out


def load_atlas_table(path, cpg_col, tile_cols):
    """Load atlas as DataFrame with cpg_id index and named tile columns."""
    df = pd.read_csv(path)
    df = df.rename(columns={cpg_col: 'cpg'})
    df['cpg'] = df['cpg'].astype(str).str.strip().str.strip('"')
    # Coerce tiles to numeric
    for t in tile_cols:
        df[t] = pd.to_numeric(df[t], errors='coerce')
    df = df.set_index('cpg')
    return df[tile_cols]


def compute_tile_a(sample_betas_aligned, tile_refs_aligned, h_min):
    """A-score = mean(|β_sample - β_ref|) / H_min, over CpGs with both present and non-NaN.
    
    sample_betas_aligned and tile_refs_aligned are numpy arrays aligned to the same
    CpG order (atlas order). NaNs in either are masked out.
    """
    mask = np.isfinite(sample_betas_aligned) & np.isfinite(tile_refs_aligned)
    if not mask.any():
        return float('nan'), 0
    deltas = np.abs(sample_betas_aligned[mask] - tile_refs_aligned[mask])
    return float(deltas.mean() / h_min), int(mask.sum())


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print('=' * 70)
    print('Unified bladder-epic v0.1 Phase C runner — VAL-120 + VAL-121 + VAL-122')
    print('=' * 70)
    print()

    # Verify BladderRef SHA (CCL-041)
    actual_sha = sha256_file(BLADDERREF_CSV)
    assert actual_sha == BLADDERREF_SHA, f'BladderRef SHA mismatch: {actual_sha}'
    print(f'BladderRef SHA verified: {BLADDERREF_SHA[:16]}…')

    # Load Xu-538 panel
    panel = json.load(open(XU538_PANEL))
    xu538_cpgs = set(panel['cpgs'])
    print(f'Xu-538 panel: {len(xu538_cpgs)} CpGs')

    # Load atlases
    print('\nLoading atlases...')
    loyfer_df = load_atlas_table(LOYFER_CSV, 'CpGs', LOYFER_TILES)
    print(f'  Loyfer:     {len(loyfer_df)} CpGs × {loyfer_df.shape[1]} tiles')
    bladderref_df = load_atlas_table(BLADDERREF_CSV, 'probeID', BLADDERREF_TILES)
    print(f'  BladderRef: {len(bladderref_df)} CpGs × {bladderref_df.shape[1]} tiles')
    caggiano_df = load_atlas_table(CAGGIANO_CSV, 'cpg_id', CAGGIANO_TILES)
    print(f'  Caggiano:   {len(caggiano_df)} CpGs × {caggiano_df.shape[1]} tiles')
    salas_df = load_atlas_table(SALAS_CSV, 'CpG_ID', SALAS_TILES)
    print(f'  Salas IDOL: {len(salas_df)} CpGs × {salas_df.shape[1]} tiles')
    unilife_df = load_atlas_table(UNILIFE_CSV, 'CpG_ID', UNILIFE_TILES)
    print(f'  UniLIFE:    {len(unilife_df)} CpGs × {unilife_df.shape[1]} tiles')

    # Pre-extract atlas-tile reference vectors as numpy arrays for speed
    atlas_specs = []  # (atlas_name, tile_name, h_min, cpg_index, ref_array)
    for t in LOYFER_TILES:
        atlas_specs.append(('loyfer', t, _H_MIN[LOYFER_TILE_CLASS[t]],
                            loyfer_df.index.values, loyfer_df[t].values))
    for t in BLADDERREF_TILES:
        atlas_specs.append(('bladderref', t, _H_MIN[BLADDERREF_TILE_CLASS[t]],
                            bladderref_df.index.values, bladderref_df[t].values))
    for t in CAGGIANO_TILES:
        atlas_specs.append(('caggiano', t, _H_MIN[CAGGIANO_TILE_CLASS[t]],
                            caggiano_df.index.values, caggiano_df[t].values))
    for t in SALAS_TILES:
        atlas_specs.append(('salas', t, _H_MIN['immune'],
                            salas_df.index.values, salas_df[t].values))
    for t in UNILIFE_TILES:
        atlas_specs.append(('unilife', t, _H_MIN['immune'],
                            unilife_df.index.values, unilife_df[t].values))
    print(f'  Total tile specs: {len(atlas_specs)} per-sample scoring tasks')

    # Build atlas CpG sets for coverage calculation
    atlas_cpg_sets = {
        'loyfer': set(loyfer_df.index),
        'bladderref': set(bladderref_df.index),
        'caggiano': set(caggiano_df.index),
        'salas': set(salas_df.index),
        'unilife': set(unilife_df.index),
    }
    atlas_n = {k: len(v) for k, v in atlas_cpg_sets.items()}

    # Load BLCA manifest
    manifest = json.load(open(BLCA_MANIFEST))
    by_filename = {m['file_name']: m for m in manifest}

    files_on_disk = sorted(BLCA_DIR.glob('*.txt'))
    print(f'\nBLCA files on disk: {len(files_on_disk)}')
    print()

    # Per-sample loop
    per_sample_rows = []
    t_io = 0.0
    t_score = 0.0

    for i, path in enumerate(files_on_disk, 1):
        meta = by_filename.get(path.name)
        if meta is None:
            continue

        # Load β with pandas (vectorized)
        t0 = time.time()
        df = pd.read_csv(path, sep='\t', header=None, names=['cpg','beta'],
                         engine='c', na_values=['NA','nan','NaN'])
        df = df.dropna(subset=['beta'])
        # Index by CpG
        df = df.set_index('cpg')
        beta_series = df['beta']
        t_io += time.time() - t0

        t0 = time.time()
        all_vals = beta_series.values
        n_cpgs_genome = len(all_vals)
        f_extreme = float(np.mean((all_vals < 0.1) | (all_vals > 0.9)))
        f_middle = float(np.mean((all_vals >= 0.4) & (all_vals <= 0.6)))
        median = float(np.median(all_vals))
        chk_3_1a_passed = (f_extreme >= 0.50) and (f_middle <= 0.12)

        # Stage 1 — Xu-538 A_immune (pooled Shannon-H entropy panel)
        # CHK-3.1B coverage on Xu-538 = fraction of 538 CpGs present in sample
        sample_cpgs = set(beta_series.index)
        xu538_present = xu538_cpgs & sample_cpgs
        n_xu538_present = len(xu538_present)
        xu538_coverage = n_xu538_present / len(xu538_cpgs)
        chk_3_1b_xu538_passed = xu538_coverage >= 0.80

        if xu538_present:
            xu538_betas = beta_series.reindex(list(xu538_present)).values
            H_vals = shannon_H_arr(xu538_betas)
            a_immune = float(H_vals.mean() / _H_MIN['immune'])
        else:
            a_immune = float('nan')

        # Stage 2/3 — per-(atlas, tile) A-scores
        # Reindex β by each atlas's CpG list, use vectorized |β-ref| mean
        per_atlas_coverage = {
            an: len(beta_series.index.intersection(cpgs)) / atlas_n[an]
            for an, cpgs in atlas_cpg_sets.items()
        }
        per_atlas_chk_3_1b_passed = {an: cov >= 0.80 for an, cov in per_atlas_coverage.items()}

        row = {
            'sample_id': meta['sample_id'],
            'case_id': meta['case_id'],
            'sample_type': meta['sample_type'],
            'file_name': path.name,
            'n_cpgs_genome': n_cpgs_genome,
            'f_extreme': f_extreme,
            'f_middle': f_middle,
            'median': median,
            'chk_3_1a_passed': chk_3_1a_passed,
            'n_xu538_present': n_xu538_present,
            'xu538_coverage': xu538_coverage,
            'chk_3_1b_xu538_passed': chk_3_1b_xu538_passed,
            'A_immune_xu538': a_immune,
        }
        for an in ['loyfer','bladderref','caggiano','salas','unilife']:
            row[f'cov_{an}'] = per_atlas_coverage[an]
            row[f'chk_3_1b_{an}_passed'] = per_atlas_chk_3_1b_passed[an]

        # Per-tile A-scores: reindex β to each atlas's CpG order, compute vectorized
        # Cache per-atlas reindexed β to avoid 73 reindex calls per sample
        atlas_reindexed_beta = {}
        for an, cpgs_idx in [('loyfer', loyfer_df.index),
                              ('bladderref', bladderref_df.index),
                              ('caggiano', caggiano_df.index),
                              ('salas', salas_df.index),
                              ('unilife', unilife_df.index)]:
            atlas_reindexed_beta[an] = beta_series.reindex(cpgs_idx).values

        for atlas_name, tile_name, h_min, cpg_index, ref_arr in atlas_specs:
            sample_aligned = atlas_reindexed_beta[atlas_name]
            a, n_used = compute_tile_a(sample_aligned, ref_arr, h_min)
            row[f'A_{atlas_name}_{tile_name}'] = a
            row[f'n_{atlas_name}_{tile_name}'] = n_used

        per_sample_rows.append(row)
        t_score += time.time() - t0

        if i % 20 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (len(files_on_disk) - i) / rate if rate > 0 else 0
            print(f'  {i:3d}/{len(files_on_disk)} samples  elapsed {elapsed:.0f}s  '
                  f'(I/O {t_io:.0f}s, score {t_score:.0f}s)  ETA {eta:.0f}s')

    elapsed = time.time() - t_start
    print(f'\nDone in {elapsed:.1f}s')
    print(f'  I/O time:   {t_io:.1f}s ({t_io/elapsed*100:.0f}%)')
    print(f'  Score time: {t_score:.1f}s ({t_score/elapsed*100:.0f}%)')

    # Save the unified per-sample table
    df_all = pd.DataFrame(per_sample_rows)
    unified_csv = OUTPUT_DIR_121 / 'VAL_121_unified_per_sample.csv'
    df_all.to_csv(unified_csv, index=False)
    print(f'\nUnified per-sample table: {unified_csv}')
    print(f'  Shape: {df_all.shape}')

    # Project per-VAL views from the unified table
    # VAL-120: subset of columns
    val120_cols = [
        'sample_id','case_id','sample_type','file_name',
        'n_cpgs_genome','f_extreme','f_middle','chk_3_1a_passed',
        'n_xu538_present','xu538_coverage','chk_3_1b_xu538_passed',
    ]
    # rename A_immune_xu538 → A_immune for consistency with VAL-120 prereg-named output
    df_120 = df_all[val120_cols + ['A_immune_xu538']].rename(columns={
        'xu538_coverage': 'xu538_coverage',
        'chk_3_1b_xu538_passed': 'chk_3_1b_passed',
        'A_immune_xu538': 'A_immune',
    })
    df_120.to_csv(OUTPUT_DIR_120 / 'VAL-120_per_sample.csv', index=False)
    print(f'VAL-120 per-sample CSV: {OUTPUT_DIR_120 / "VAL-120_per_sample.csv"}')

    # VAL-121: stage 2 atlas columns
    val121_cols = ['sample_id','case_id','sample_type','file_name',
                   'f_extreme','f_middle','chk_3_1a_passed',
                   'cov_loyfer','cov_bladderref','cov_caggiano',
                   'chk_3_1b_loyfer_passed','chk_3_1b_bladderref_passed','chk_3_1b_caggiano_passed']
    for atlas, tiles in [('loyfer', LOYFER_TILES),
                          ('bladderref', BLADDERREF_TILES),
                          ('caggiano', CAGGIANO_TILES)]:
        for t in tiles:
            val121_cols.append(f'A_{atlas}_{t}')
            val121_cols.append(f'n_{atlas}_{t}')
    df_all[val121_cols].to_csv(OUTPUT_DIR_121 / 'VAL-121_per_sample_per_atlas.csv', index=False)
    print(f'VAL-121 per-sample CSV: {OUTPUT_DIR_121 / "VAL-121_per_sample_per_atlas.csv"}')

    # VAL-122: stage 3 atlas columns (caggiano immune subset only)
    val122_cols = ['sample_id','case_id','sample_type','file_name',
                   'f_extreme','f_middle','chk_3_1a_passed',
                   'cov_salas','cov_unilife','cov_caggiano',
                   'chk_3_1b_salas_passed','chk_3_1b_unilife_passed','chk_3_1b_caggiano_passed']
    for atlas, tiles in [('salas', SALAS_TILES),
                          ('unilife', UNILIFE_TILES),
                          ('caggiano', CAGGIANO_IMMUNE_TILES)]:
        for t in tiles:
            val122_cols.append(f'A_{atlas}_{t}')
            val122_cols.append(f'n_{atlas}_{t}')
    df_all[val122_cols].to_csv(OUTPUT_DIR_122 / 'VAL-122_per_sample_per_atlas.csv', index=False)
    print(f'VAL-122 per-sample CSV: {OUTPUT_DIR_122 / "VAL-122_per_sample_per_atlas.csv"}')

    # QC quick summary
    print()
    print('=' * 70)
    print('QC summary across the unified pass:')
    print(f'  Total samples scored: {len(df_all)}')
    print(f'  Sample types: {dict(df_all["sample_type"].value_counts())}')
    print(f'  CHK-3.1A pass: {df_all["chk_3_1a_passed"].sum()}/{len(df_all)} ({df_all["chk_3_1a_passed"].mean()*100:.1f}%)')
    print(f'  Mean f_extreme: {df_all["f_extreme"].mean():.4f} ± {df_all["f_extreme"].std():.4f}')
    print(f'  Mean f_middle:  {df_all["f_middle"].mean():.4f} ± {df_all["f_middle"].std():.4f}')
    print('  CHK-3.1B per atlas:')
    for an in ['loyfer','bladderref','caggiano','salas','unilife']:
        npass = df_all[f'chk_3_1b_{an}_passed'].sum()
        cov = df_all[f'cov_{an}'].mean()
        print(f'    {an:10s}: pass {npass}/{len(df_all)}  mean coverage {cov*100:.1f}%')
    print(f'  CHK-3.1B Xu-538: pass {df_all["chk_3_1b_xu538_passed"].sum()}/{len(df_all)}  '
          f'mean coverage {df_all["xu538_coverage"].mean()*100:.1f}%')

    # Quick A_immune by sample type
    print()
    print('Stage 1 A_immune by sample type:')
    for st, sub in df_all.groupby('sample_type'):
        a = sub['A_immune_xu538'].dropna()
        print(f'  {st:25s}: n={len(a)} mean={a.mean():.4f} sd={a.std():.4f}')

    print()
    print(f'Total runtime: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
