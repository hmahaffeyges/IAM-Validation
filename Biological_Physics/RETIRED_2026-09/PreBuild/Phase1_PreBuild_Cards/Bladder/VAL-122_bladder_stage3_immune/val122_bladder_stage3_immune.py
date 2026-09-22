#!/usr/bin/env python3
"""
===============================================================================
VAL-122 — Stage 3 immune fine-tune on TCGA-BLCA n=440

Pre-registered: VAL-122_bladder_stage3_immune/prereg.md
                SHA-256 sealed in PREREG_SEAL.txt before any β file read.

Atlases:
  1. Salas Blood.EPIC IDOL 450K legacy — production calibrated; 350 CpGs × 6 tiles
                                          (CD8T, CD4T, NK, Bcell, Mono, Neu)
  2. UniLIFE Guo 2025 19-cell           — within-cohort self-cal at v0.1; 1,906 CpGs × 19 tiles
  3. Caggiano TIM immune subset         — VAL-113 anchor; immune cell types only
                                          (dendritic, eosinophil, macrophage, monocyte,
                                           neutrophil, tcell, erythroblast, megakaryocyte)

A_tile = mean(|sample_β - tile_ref_β|) / H_min(immune)
H_min(immune) = 0.838889

Outcomes:
  O1 — STAGE_3_IMMUNE_DIFFERENTIATING (≥3 of 6 Salas IDOL tiles fire)
  O2 — STAGE_3_LYMPHOID_DOMINANT (CD4T/CD8T POS + Mono/Neu NEG)
  O3 — STAGE_3_MYELOID_DOMINANT (Mono/Neu POS + CD4T/CD8T NEG)
  O4 — STAGE_3_NULL
  O5 — STAGE_3_DATA_INTEGRITY_FAILURE
  O6 — STAGE_3_UNEXPECTED

RNG seed: 20260420
===============================================================================
"""

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────────
# FROZEN CONSTANTS (from sealed prereg)
# ──────────────────────────────────────────────────────────────────────────────

VAL_ID = 'VAL-122'
PREREG_SHA = '2d101db94cdc7a71466c5f8071a936abd426f85ecd9ea27ae8fa73cd0d81f855'
SEAL_TIMESTAMP = '2026-05-01T03:48:17Z'

H_MIN_IMMUNE = 0.838889

# Salas Blood.EPIC IDOL 450K legacy tiles
SALAS_TILES = ['CD8T', 'CD4T', 'NK', 'Bcell', 'Mono', 'Neu']
SALAS_LYMPHOID = ['CD4T', 'CD8T', 'Bcell', 'NK']
SALAS_MYELOID = ['Mono', 'Neu']

# UniLIFE 19 cell types (from atlas header)
UNILIFE_TILES = ['B', 'CD4T', 'CD8T', 'Mono', 'nRBC', 'Gran', 'NK',
                 'aCD4Tnv', 'aBaso', 'aCD4Tmem', 'aBmem', 'aBnv', 'aTreg',
                 'aCD8Tmem', 'aCD8Tnv', 'aEos', 'aNK', 'aNeu', 'aMono']

# Caggiano TIM immune subset (from prereg)
CAGGIANO_IMMUNE_TILES = ['dendritic', 'eosinophil', 'erythroblast', 'macrophage',
                         'monocyte', 'neutrophil', 'tcell', 'megakaryocyte']

# Atlas paths
SALAS_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage3_immune_fraction/salas_blood_epic_idol/IDOLOptimizedCpGs450k_compTable.csv')
UNILIFE_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage3_immune_fraction/unilife_guo_2025/centUniLIFE_reference_matrix.csv')
CAGGIANO_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv')

BLCA_DIR = Path('/home/claude/edear_working/bladder_epic/blca_betas')
BLCA_MANIFEST = Path('/home/claude/edear_working/bladder_epic/blca_manifest.json')
OUTPUT_DIR = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-122_bladder_stage3_immune')
OUTPUT_DIR.mkdir(exist_ok=True)

MAGNITUDE_THRESHOLD = 0.30
MIN_PAIRED_PAIRS = 15
CHK_3_1A_PASS_RATE_MIN = 0.75
CHK_3_1B_COVERAGE_MIN = 0.80

RNG_SEED = 20260420


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS (same patterns as VAL-120/121)
# ──────────────────────────────────────────────────────────────────────────────

def load_beta_file(path):
    betas = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            try:
                beta = float(parts[1])
                if not np.isnan(beta):
                    betas[parts[0]] = beta
            except ValueError:
                continue
    return betas


def load_atlas_generic(path, cpg_col, tile_cols):
    """Generic: cpg in cpg_col, tiles in tile_cols (list)."""
    atlas = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpg = row[cpg_col]
            tiles = {}
            for t in tile_cols:
                v = row.get(t, '')
                try:
                    tiles[t] = float(v)
                except (ValueError, TypeError):
                    tiles[t] = float('nan')
            atlas[cpg] = tiles
    return atlas


def chk_3_1a(betas):
    vals = np.array(list(betas.values()))
    f_extreme = float(np.mean((vals < 0.1) | (vals > 0.9)))
    f_middle = float(np.mean((vals >= 0.4) & (vals <= 0.6)))
    return {
        'f_extreme': f_extreme,
        'f_middle': f_middle,
        'passed': (f_extreme >= 0.50) and (f_middle <= 0.12),
    }


def chk_3_1b(betas, atlas):
    n_atlas = len(atlas)
    n_present = sum(1 for c in atlas if c in betas)
    coverage = n_present / n_atlas if n_atlas > 0 else 0.0
    return {'coverage': coverage, 'passed': coverage >= CHK_3_1B_COVERAGE_MIN}


def compute_tile_ascore(betas, atlas, tile, h_min):
    deltas = []
    for cpg, tile_refs in atlas.items():
        if cpg in betas:
            ref = tile_refs.get(tile, float('nan'))
            if not np.isnan(ref):
                deltas.append(abs(betas[cpg] - ref))
    if not deltas:
        return float('nan'), 0
    return float(np.mean(deltas) / h_min), len(deltas)


def paired_d_with_ci(diffs):
    n = len(diffs)
    if n < 2:
        return float('nan'), float('nan'), float('nan'), float('nan')
    arr = np.array(diffs)
    sd = arr.std(ddof=1)
    if sd == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    d = float(arr.mean() / sd)
    se = float(np.sqrt(1/n + d**2 / (2*n)))
    ci_l = float(d - 1.96 * se)
    ci_h = float(d + 1.96 * se)
    t_stat = arr.mean() / (sd / np.sqrt(n))
    p = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)))
    return d, ci_l, ci_h, p


def welch_d_with_ci(a, b):
    a, b = np.array(a), np.array(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan'), float('nan'), float('nan'), float('nan')
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = float(np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na + nb - 2)))
    if pooled == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    d = float((a.mean() - b.mean()) / pooled)
    se = float(np.sqrt(1/na + 1/nb + d**2 / (2*(na+nb))))
    ci_l = float(d - 1.96 * se)
    ci_h = float(d + 1.96 * se)
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return d, ci_l, ci_h, float(p)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f'VAL-122 — Stage 3 immune fine-tune on TCGA-BLCA')
    print(f'Prereg SHA: {PREREG_SHA}')
    print(f'Sealed:     {SEAL_TIMESTAMP}')
    print()

    # Load atlases
    print('Loading Stage 3 atlases...')
    salas = load_atlas_generic(SALAS_CSV, 'CpG_ID', SALAS_TILES)
    print(f'  Salas IDOL 450K:  {len(salas):>5d} CpGs × {len(SALAS_TILES)} tiles')
    unilife = load_atlas_generic(UNILIFE_CSV, 'CpG_ID', UNILIFE_TILES)
    print(f'  UniLIFE 19-cell:  {len(unilife):>5d} CpGs × {len(UNILIFE_TILES)} tiles')
    # Caggiano shares its CSV with VAL-121; we slice immune tiles
    caggiano = load_atlas_generic(CAGGIANO_CSV, 'cpg_id', CAGGIANO_IMMUNE_TILES)
    print(f'  Caggiano immune:  {len(caggiano):>5d} CpGs × {len(CAGGIANO_IMMUNE_TILES)} tiles (immune subset)')

    # Load BLCA
    with open(BLCA_MANIFEST) as f:
        manifest = json.load(f)
    by_filename = {m['file_name']: m for m in manifest}
    files_on_disk = sorted(BLCA_DIR.glob('*.txt'))
    print(f'BLCA files: {len(files_on_disk)}')
    print()

    # Per-sample loop
    per_sample = []
    for i, path in enumerate(files_on_disk, 1):
        if i % 50 == 0:
            elapsed = time.time() - t_start
            print(f'  {i}/{len(files_on_disk)} samples in {elapsed:.0f}s')
        meta = by_filename.get(path.name)
        if meta is None:
            continue

        betas = load_beta_file(path)
        chk_a = chk_3_1a(betas)
        chk_b_salas = chk_3_1b(betas, salas)
        chk_b_unilife = chk_3_1b(betas, unilife)
        chk_b_caggiano = chk_3_1b(betas, caggiano)

        row = {
            'sample_id': meta['sample_id'],
            'case_id': meta['case_id'],
            'sample_type': meta['sample_type'],
            'file_name': path.name,
            'f_extreme': chk_a['f_extreme'],
            'f_middle': chk_a['f_middle'],
            'chk_3_1a_passed': chk_a['passed'],
            'cov_salas': chk_b_salas['coverage'],
            'cov_unilife': chk_b_unilife['coverage'],
            'cov_caggiano': chk_b_caggiano['coverage'],
            'chk_3_1b_salas_passed': chk_b_salas['passed'],
            'chk_3_1b_unilife_passed': chk_b_unilife['passed'],
            'chk_3_1b_caggiano_passed': chk_b_caggiano['passed'],
        }
        for t in SALAS_TILES:
            a, n = compute_tile_ascore(betas, salas, t, H_MIN_IMMUNE)
            row[f'A_salas_{t}'] = a
            row[f'n_salas_{t}'] = n
        for t in UNILIFE_TILES:
            a, n = compute_tile_ascore(betas, unilife, t, H_MIN_IMMUNE)
            row[f'A_unilife_{t}'] = a
            row[f'n_unilife_{t}'] = n
        for t in CAGGIANO_IMMUNE_TILES:
            a, n = compute_tile_ascore(betas, caggiano, t, H_MIN_IMMUNE)
            row[f'A_caggiano_{t}'] = a
            row[f'n_caggiano_{t}'] = n
        per_sample.append(row)

    print(f'Done. {len(per_sample)} samples scored.')
    print()

    per_sample_csv = OUTPUT_DIR / 'VAL-122_per_sample_per_atlas.csv'
    with open(per_sample_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        for row in per_sample:
            w.writerow(row)
    print(f'Per-sample saved: {per_sample_csv}')

    # QC
    n_chk_a = sum(1 for s in per_sample if s['chk_3_1a_passed'])
    n_chk_b_salas = sum(1 for s in per_sample if s['chk_3_1b_salas_passed'])
    n_chk_b_unilife = sum(1 for s in per_sample if s['chk_3_1b_unilife_passed'])
    n_chk_b_caggiano = sum(1 for s in per_sample if s['chk_3_1b_caggiano_passed'])
    print(f'CHK-3.1A: {n_chk_a}/{len(per_sample)} ({n_chk_a/len(per_sample):.1%})')
    print(f'CHK-3.1B Salas:    {n_chk_b_salas}/{len(per_sample)} ({n_chk_b_salas/len(per_sample):.1%})')
    print(f'CHK-3.1B UniLIFE:  {n_chk_b_unilife}/{len(per_sample)} ({n_chk_b_unilife/len(per_sample):.1%})')
    print(f'CHK-3.1B Caggiano: {n_chk_b_caggiano}/{len(per_sample)} ({n_chk_b_caggiano/len(per_sample):.1%})')

    # Paired
    by_case = defaultdict(dict)
    for s in per_sample:
        by_case[s['case_id']][s['sample_type']] = s
    paired_cases = [c for c, types in by_case.items()
                    if 'Solid Tissue Normal' in types and 'Primary Tumor' in types]
    paired_qc = []
    for c in paired_cases:
        n_s = by_case[c]['Solid Tissue Normal']
        t_s = by_case[c]['Primary Tumor']
        if n_s['chk_3_1a_passed'] and t_s['chk_3_1a_passed']:
            paired_qc.append({'case_id': c, 'normal': n_s, 'tumor': t_s})
    print(f'\nPaired pairs after QC: {len(paired_qc)}')

    # Per-(atlas, tile) contrasts
    contrasts = {}

    def compute_contrasts(atlas_name, tile, key_prefix):
        diffs = []
        for p in paired_qc:
            t_v = p['tumor'][f'{key_prefix}_{tile}']
            n_v = p['normal'][f'{key_prefix}_{tile}']
            if not (np.isnan(t_v) or np.isnan(n_v)):
                diffs.append(t_v - n_v)
        d_p, cp_l, cp_h, p_p = paired_d_with_ci(diffs)
        dir_p = 'POSITIVE' if not np.isnan(d_p) and d_p > 0 else 'NEGATIVE' if not np.isnan(d_p) and d_p < 0 else 'INSUFFICIENT'

        a_t = [s[f'{key_prefix}_{tile}'] for s in per_sample
               if s['sample_type'] == 'Primary Tumor' and s['chk_3_1a_passed']
               and not np.isnan(s[f'{key_prefix}_{tile}'])]
        a_n = [s[f'{key_prefix}_{tile}'] for s in per_sample
               if s['sample_type'] == 'Solid Tissue Normal' and s['chk_3_1a_passed']
               and not np.isnan(s[f'{key_prefix}_{tile}'])]
        d_w, cw_l, cw_h, p_w = welch_d_with_ci(a_t, a_n) if a_t and a_n else (float('nan'),)*4
        dir_w = 'POSITIVE' if not np.isnan(d_w) and d_w > 0 else 'NEGATIVE' if not np.isnan(d_w) and d_w < 0 else 'INSUFFICIENT'

        return {
            'atlas': atlas_name, 'tile': tile,
            'paired': {'n_pairs': len(diffs),
                       'd': d_p, 'ci_95_low': cp_l, 'ci_95_high': cp_h, 'p_value': p_p,
                       'direction': dir_p,
                       'fires': abs(d_p) >= MAGNITUDE_THRESHOLD if not np.isnan(d_p) else False},
            'welch': {'n_tumor': len(a_t), 'n_normal': len(a_n),
                      'd': d_w, 'ci_95_low': cw_l, 'ci_95_high': cw_h, 'p_value': p_w,
                      'direction': dir_w},
            'tumor_mean': float(np.mean(a_t)) if a_t else float('nan'),
            'normal_mean': float(np.mean(a_n)) if a_n else float('nan'),
        }

    print('\nComputing per-(atlas, tile) contrasts...')
    for t in SALAS_TILES:
        contrasts[f'salas:{t}'] = compute_contrasts('salas', t, 'A_salas')
    for t in UNILIFE_TILES:
        contrasts[f'unilife:{t}'] = compute_contrasts('unilife', t, 'A_unilife')
    for t in CAGGIANO_IMMUNE_TILES:
        contrasts[f'caggiano:{t}'] = compute_contrasts('caggiano', t, 'A_caggiano')

    # Salas IDOL 6-tile summary
    print('\n=== Salas IDOL 6-tile paired contrasts ===')
    salas_fires = []
    for t in SALAS_TILES:
        c = contrasts[f'salas:{t}']
        marker = ' ✓ FIRES' if c['paired']['fires'] else ''
        if c['paired']['fires']:
            salas_fires.append({'tile': t, 'd': c['paired']['d'], 'direction': c['paired']['direction']})
        print(f'  Salas {t:6s} d_paired={c["paired"]["d"]:+.4f}  CI[{c["paired"]["ci_95_low"]:+.3f},{c["paired"]["ci_95_high"]:+.3f}]  p={c["paired"]["p_value"]:.3g}  {c["paired"]["direction"]}{marker}')

    # Lymphoid vs myeloid pattern
    lymphoid_pos = [t for t in SALAS_LYMPHOID
                    if contrasts[f'salas:{t}']['paired']['direction'] == 'POSITIVE'
                    and contrasts[f'salas:{t}']['paired']['fires']]
    lymphoid_neg = [t for t in SALAS_LYMPHOID
                    if contrasts[f'salas:{t}']['paired']['direction'] == 'NEGATIVE'
                    and contrasts[f'salas:{t}']['paired']['fires']]
    myeloid_pos = [t for t in SALAS_MYELOID
                   if contrasts[f'salas:{t}']['paired']['direction'] == 'POSITIVE'
                   and contrasts[f'salas:{t}']['paired']['fires']]
    myeloid_neg = [t for t in SALAS_MYELOID
                   if contrasts[f'salas:{t}']['paired']['direction'] == 'NEGATIVE'
                   and contrasts[f'salas:{t}']['paired']['fires']]

    print(f'\nLymphoid POS firing: {lymphoid_pos}; NEG firing: {lymphoid_neg}')
    print(f'Myeloid POS firing:  {myeloid_pos}; NEG firing:  {myeloid_neg}')

    n_salas_fire = len(salas_fires)
    print(f'Total Salas tiles firing: {n_salas_fire}/6')

    # Determine outcome
    qc_pass_rate_salas = n_chk_b_salas / len(per_sample)
    qc_pass_rate_unilife = n_chk_b_unilife / len(per_sample)
    qc_pass_rate_caggiano = n_chk_b_caggiano / len(per_sample)

    if (n_chk_a / len(per_sample) < CHK_3_1A_PASS_RATE_MIN
        or qc_pass_rate_salas < CHK_3_1A_PASS_RATE_MIN
        or len(paired_qc) < MIN_PAIRED_PAIRS):
        outcome_class = 'O5_STAGE_3_DATA_INTEGRITY_FAILURE'
        outcome_note = f'QC failure or insufficient pairs ({len(paired_qc)})'
    elif lymphoid_pos and myeloid_neg:
        outcome_class = 'O2_STAGE_3_LYMPHOID_DOMINANT'
        outcome_note = f'Lymphoid POS: {lymphoid_pos}; Myeloid NEG: {myeloid_neg}. Consistent with Chen 2022 NMIBC blood RFS signature.'
    elif myeloid_pos and lymphoid_neg:
        outcome_class = 'O3_STAGE_3_MYELOID_DOMINANT'
        outcome_note = f'Myeloid POS: {myeloid_pos}; Lymphoid NEG: {lymphoid_neg}. Consistent with MDSC infiltration in advanced/MIBC.'
    elif n_salas_fire >= 3:
        outcome_class = 'O1_STAGE_3_IMMUNE_DIFFERENTIATING'
        outcome_note = f'{n_salas_fire}/6 Salas IDOL tiles firing |d|≥0.30. Multi-tile immune shift.'
    elif n_salas_fire == 0:
        outcome_class = 'O4_STAGE_3_NULL'
        outcome_note = 'All 6 Salas IDOL tiles |d|<0.30.'
    else:
        outcome_class = 'O6_STAGE_3_UNEXPECTED'
        outcome_note = f'{n_salas_fire}/6 Salas tiles fire but pattern doesn\'t match O1/O2/O3.'

    runtime = time.time() - t_start

    results = {
        'val_id': VAL_ID,
        'val_type': 'PHASE_C_STAGE3_IMMUNE',
        'card_target': 'bladder-epic v0.1',
        'prereg_sha': PREREG_SHA,
        'seal_timestamp': SEAL_TIMESTAMP,
        'rng_seed': RNG_SEED,
        'runtime_seconds': runtime,
        'cohort': {
            'name': 'TCGA-BLCA',
            'substrate': 'TCGA HM450K sesame Level 3',
            'n_total': len(per_sample),
            'n_paired_pairs_qc_passed': len(paired_qc),
        },
        'atlases': {
            'salas': {'n_cpgs': len(salas), 'n_tiles': len(SALAS_TILES), 'calibration': 'production'},
            'unilife': {'n_cpgs': len(unilife), 'n_tiles': len(UNILIFE_TILES), 'calibration': 'within-cohort self-cal v0.1; VAL-115 v0.X+1'},
            'caggiano_immune': {'n_cpgs': len(caggiano), 'n_tiles': len(CAGGIANO_IMMUNE_TILES), 'calibration': 'VAL-113 anchor (immune subset)'},
        },
        'chk_3_1a': {
            'pass_rate': n_chk_a / len(per_sample),
            'gate_passed': n_chk_a / len(per_sample) >= CHK_3_1A_PASS_RATE_MIN,
        },
        'chk_3_1b_per_atlas': {
            'salas': qc_pass_rate_salas,
            'unilife': qc_pass_rate_unilife,
            'caggiano': qc_pass_rate_caggiano,
        },
        'salas_idol_summary': {
            'n_tiles_firing': n_salas_fire,
            'lymphoid_positive': lymphoid_pos,
            'lymphoid_negative': lymphoid_neg,
            'myeloid_positive': myeloid_pos,
            'myeloid_negative': myeloid_neg,
            'tiles_firing_detail': salas_fires,
        },
        'contrasts': contrasts,
        'outcome_class': outcome_class,
        'outcome_note': outcome_note,
        'sealed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    results_path = OUTPUT_DIR / f'{VAL_ID}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print('=' * 70)
    print(f'OUTCOME: {outcome_class}')
    print(f'  {outcome_note}')
    print('=' * 70)
    print(f'Runtime: {runtime:.1f} sec')
    print(f'Results: {results_path}')


if __name__ == '__main__':
    main()
