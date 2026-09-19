#!/usr/bin/env python3
"""
===============================================================================
VAL-121 — Stage 2 multi-atlas Phase C run-everything on TCGA-BLCA n=440

Pre-registered: VAL-121_bladder_stage2_multiatlas/prereg.md
                SHA-256 sealed in PREREG_SEAL.txt before any β file read.

Atlases (all calibrated against TCGA HM450K sesame Level 3):
  1. Layered Moss+Loyfer  — VAL-112 anchor; 6,105 CpGs × 25 cell types
  2. EpiSCORE BladderRef  — VAL-119 anchor; 2,696 CpGs × 4 cell types (EC/Epi/Fib/IC)
  3. Caggiano CelFiE TIM  — VAL-113 anchor; 254 CpGs × 19 cell types

Cohort: TCGA-BLCA HM450 sesame Level 3 — 418 tumor + 21 normal + 1 metastatic
        21 paired patients (have both adjacent-normal + tumor)

Per-tile A-score = mean(|sample_β - tile_ref_β|) / H_min(class), over CpGs
present in sample AND with non-NaN tile reference.

H_min anchors (G-003b MCMC 2026-04-06):
  secretory = 0.843264 (epithelial)
  stromal   = 0.862950 (fibroblast/endothelial/smooth muscle/adipocytes)
  immune    = 0.838889
  terminal  = 0.772800 (terminal differentiated: neurons, hepatocytes, beta cells)
  cycling   = 0.856100 (proliferative: Erythrocyte_progenitors, Pancreatic_duct_cells)
  stem_pluri = 0.982200

CCL-039 cell-of-origin direction expectation:
  - Bladder cell-of-origin tile (Loyfer Bladder; BladderRef Epi) NEGATIVE in tumor
  - Microenvironment tiles POSITIVE in tumor

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

VAL_ID = 'VAL-121'
PREREG_SHA = 'eb68e4d4ca6270cdcce60269375af787537c560fabea18ee31cbaf558dea1962'
SEAL_TIMESTAMP = '2026-05-01T03:48:17Z'

# H_min anchors (G-002 MCMC posteriors, frozen 2026-04-06; values from GAPE_WEB_v13.py _H_MIN registry)
H_MIN_TERMINAL = 0.772837
H_MIN_IMMUNE = 0.838889
H_MIN_SECRETORY = 0.843264
H_MIN_CYCLING = 0.856055
H_MIN_STROMAL = 0.862950
H_MIN_STEM_PLURI = 0.982166

# Loyfer 25-tile class assignments (from GAPE engine architecture-class catalog)
# Conservative: epithelial→secretory, immune→immune, stromal/vascular→stromal,
# terminal-differentiated→terminal, progenitor→cycling
LOYFER_TILE_CLASS = {
    'Monocytes_EPIC': 'immune',
    'B-cells_EPIC': 'immune',
    'CD4T-cells_EPIC': 'immune',
    'NK-cells_EPIC': 'immune',
    'CD8T-cells_EPIC': 'immune',
    'Neutrophils_EPIC': 'immune',
    'Erythrocyte_progenitors': 'cycling',
    'Adipocytes': 'stromal',
    'Cortical_neurons': 'terminal',
    'Hepatocytes': 'terminal',
    'Lung_cells': 'secretory',
    'Pancreatic_beta_cells': 'terminal',
    'Pancreatic_acinar_cells': 'secretory',
    'Pancreatic_duct_cells': 'cycling',
    'Vascular_endothelial_cells': 'stromal',
    'Colon_epithelial_cells': 'secretory',
    'Left_atrium': 'terminal',
    'Bladder': 'secretory',
    'Breast': 'secretory',
    'Head_and_neck_larynx': 'secretory',
    'Kidney': 'secretory',
    'Prostate': 'secretory',
    'Thyroid': 'secretory',
    'Upper_GI': 'secretory',
    'Uterus_cervix': 'secretory',
}

LOYFER_TILES = list(LOYFER_TILE_CLASS.keys())

# BladderRef 4-tile class assignments (from VAL-119 prereg)
BLADDERREF_TILE_CLASS = {
    'EC': 'stromal',
    'Epi': 'secretory',
    'Fib': 'stromal',
    'IC': 'immune',
}
BLADDERREF_TILES = list(BLADDERREF_TILE_CLASS.keys())

# Caggiano TIM 19-tile class assignments
CAGGIANO_TILE_CLASS = {
    'dendritic': 'immune',
    'endothelial': 'stromal',
    'eosinophil': 'immune',
    'erythroblast': 'cycling',
    'macrophage': 'immune',
    'monocyte': 'immune',
    'neutrophil': 'immune',
    'placenta': 'secretory',
    'tcell': 'immune',
    'adipose': 'stromal',
    'brain': 'terminal',
    'fibroblast': 'stromal',
    'heart': 'terminal',
    'hepatocyte': 'terminal',
    'lung': 'secretory',
    'mammary': 'secretory',
    'megakaryocyte': 'cycling',
    'skeletal': 'terminal',
    'small_intestine': 'secretory',
}
CAGGIANO_TILES = list(CAGGIANO_TILE_CLASS.keys())

CLASS_HMIN = {
    'terminal': H_MIN_TERMINAL,
    'immune': H_MIN_IMMUNE,
    'secretory': H_MIN_SECRETORY,
    'cycling': H_MIN_CYCLING,
    'stromal': H_MIN_STROMAL,
    'stem_pluri': H_MIN_STEM_PLURI,
}

# Atlas paths
LOYFER_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv')
BLADDERREF_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_bladderref/episcore_bladderref_cpg_bridged.csv')
BLADDERREF_SHA = '3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6'
CAGGIANO_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/caggiano_celfie_tim/caggiano_tim_cpg_bridged.csv')

BLCA_DIR = Path('/home/claude/edear_working/bladder_epic/blca_betas')
BLCA_MANIFEST = Path('/home/claude/edear_working/bladder_epic/blca_manifest.json')
OUTPUT_DIR = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-121_bladder_stage2_multiatlas')
OUTPUT_DIR.mkdir(exist_ok=True)

# Pre-locked thresholds
MAGNITUDE_THRESHOLD = 0.30
MIN_PAIRED_PAIRS = 15
CHK_3_1A_PASS_RATE_MIN = 0.75
CHK_3_1B_COVERAGE_MIN = 0.80

# Bladder cell-of-origin tiles (CCL-039 direction expectation = NEGATIVE)
BLADDER_COO_TILES = [
    ('loyfer', 'Bladder'),
    ('bladderref', 'Epi'),
]

# Loyfer non-bladder tissue tiles (CHK-3.2 sanity check — should NOT fire positive on bladder tumor)
LOYFER_NON_BLADDER_TISSUE_TILES = [
    'Breast', 'Kidney', 'Prostate', 'Thyroid', 'Upper_GI', 'Uterus_cervix',
    'Head_and_neck_larynx', 'Colon_epithelial_cells',
    'Hepatocytes', 'Lung_cells', 'Cortical_neurons',
    'Pancreatic_beta_cells', 'Pancreatic_acinar_cells', 'Pancreatic_duct_cells',
]

RNG_SEED = 20260420


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


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


def load_loyfer_atlas(path):
    """Load Loyfer reference_atlas.csv: CpGs in col0, 25 cell types in col1-25."""
    atlas = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpg = row['CpGs']
            tiles = {}
            for t in LOYFER_TILES:
                v = row.get(t, '')
                try:
                    tiles[t] = float(v)
                except (ValueError, TypeError):
                    tiles[t] = float('nan')
            atlas[cpg] = tiles
    return atlas


def load_bladderref_atlas(path):
    """Load BladderRef bridged: probeID, EID, EC, Epi, Fib, IC, weight."""
    atlas = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            probe = row['probeID']
            tiles = {}
            for t in BLADDERREF_TILES:
                v = row[t]
                try:
                    tiles[t] = float(v)
                except (ValueError, TypeError):
                    tiles[t] = float('nan')
            atlas[probe] = tiles
    return atlas


def load_caggiano_atlas(path):
    """Load Caggiano TIM bridged: cpg_id + 19 cell types."""
    atlas = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cpg = row['cpg_id']
            tiles = {}
            for t in CAGGIANO_TILES:
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
        'n_cpgs_genome': len(vals),
        'passed': (f_extreme >= 0.50) and (f_middle <= 0.12),
    }


def chk_3_1b_atlas(betas, atlas):
    n_atlas = len(atlas)
    n_present = sum(1 for c in atlas if c in betas)
    coverage = n_present / n_atlas if n_atlas > 0 else 0.0
    return {
        'n_atlas_cpgs': n_atlas,
        'n_present': n_present,
        'coverage': coverage,
        'passed': coverage >= CHK_3_1B_COVERAGE_MIN,
    }


def compute_tile_ascore(betas, atlas, tile, h_min):
    """Mean |sample_β - tile_ref_β| over present-non-NaN, normalized by H_min."""
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
    """Cohen's d for paired contrast + 95% CI + p."""
    n = len(diffs)
    if n < 2:
        return float('nan'), float('nan'), float('nan'), float('nan')
    arr = np.array(diffs)
    sd = arr.std(ddof=1)
    if sd == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    d = float(arr.mean() / sd)
    se = float(np.sqrt(1/n + d**2 / (2*n)))
    ci_low = float(d - 1.96 * se)
    ci_high = float(d + 1.96 * se)
    t_stat = arr.mean() / (sd / np.sqrt(n))
    p = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)))
    return d, ci_low, ci_high, p


def welch_d_with_ci(a, b):
    """Welch's d (unequal variance) + 95% CI + p."""
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
    ci_low = float(d - 1.96 * se)
    ci_high = float(d + 1.96 * se)
    _, p = stats.ttest_ind(a, b, equal_var=False)
    return d, ci_low, ci_high, float(p)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f'VAL-121 — Stage 2 multi-atlas Phase C on TCGA-BLCA')
    print(f'Prereg SHA: {PREREG_SHA}')
    print(f'Sealed:     {SEAL_TIMESTAMP}')
    print()

    # Verify BladderRef SHA
    actual_sha = sha256_file(BLADDERREF_CSV)
    assert actual_sha == BLADDERREF_SHA, f'BladderRef SHA mismatch: {actual_sha}'
    print(f'BladderRef SHA verified: {BLADDERREF_SHA}')

    # Load atlases
    print('Loading atlases...')
    loyfer = load_loyfer_atlas(LOYFER_CSV)
    print(f'  Loyfer:     {len(loyfer):>5d} CpGs × {len(LOYFER_TILES)} tiles')
    bladderref = load_bladderref_atlas(BLADDERREF_CSV)
    print(f'  BladderRef: {len(bladderref):>5d} CpGs × {len(BLADDERREF_TILES)} tiles')
    caggiano = load_caggiano_atlas(CAGGIANO_CSV)
    print(f'  Caggiano:   {len(caggiano):>5d} CpGs × {len(CAGGIANO_TILES)} tiles')

    # Load BLCA manifest
    with open(BLCA_MANIFEST) as f:
        manifest = json.load(f)
    by_filename = {m['file_name']: m for m in manifest}

    files_on_disk = sorted(BLCA_DIR.glob('*.txt'))
    print(f'BLCA files on disk: {len(files_on_disk)}')
    print()

    # Per-sample loop (the long one — 440 samples × 3 atlases × multi-tile)
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
        chk_b_loyfer = chk_3_1b_atlas(betas, loyfer)
        chk_b_bladderref = chk_3_1b_atlas(betas, bladderref)
        chk_b_caggiano = chk_3_1b_atlas(betas, caggiano)

        row = {
            'sample_id': meta['sample_id'],
            'case_id': meta['case_id'],
            'sample_type': meta['sample_type'],
            'file_name': path.name,
            'f_extreme': chk_a['f_extreme'],
            'f_middle': chk_a['f_middle'],
            'chk_3_1a_passed': chk_a['passed'],
            'cov_loyfer': chk_b_loyfer['coverage'],
            'cov_bladderref': chk_b_bladderref['coverage'],
            'cov_caggiano': chk_b_caggiano['coverage'],
            'chk_3_1b_loyfer_passed': chk_b_loyfer['passed'],
            'chk_3_1b_bladderref_passed': chk_b_bladderref['passed'],
            'chk_3_1b_caggiano_passed': chk_b_caggiano['passed'],
        }

        # Loyfer 25 tiles
        for t in LOYFER_TILES:
            a, n = compute_tile_ascore(betas, loyfer, t, CLASS_HMIN[LOYFER_TILE_CLASS[t]])
            row[f'A_loyfer_{t}'] = a
            row[f'n_loyfer_{t}'] = n
        # BladderRef 4 tiles
        for t in BLADDERREF_TILES:
            a, n = compute_tile_ascore(betas, bladderref, t, CLASS_HMIN[BLADDERREF_TILE_CLASS[t]])
            row[f'A_bladderref_{t}'] = a
            row[f'n_bladderref_{t}'] = n
        # Caggiano 19 tiles
        for t in CAGGIANO_TILES:
            a, n = compute_tile_ascore(betas, caggiano, t, CLASS_HMIN[CAGGIANO_TILE_CLASS[t]])
            row[f'A_caggiano_{t}'] = a
            row[f'n_caggiano_{t}'] = n

        per_sample.append(row)

    print(f'Done. {len(per_sample)} samples scored.')
    print()

    # Save per-sample CSV
    per_sample_csv = OUTPUT_DIR / 'VAL-121_per_sample_per_atlas.csv'
    with open(per_sample_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        for row in per_sample:
            w.writerow(row)
    print(f'Per-sample saved: {per_sample_csv}')

    # QC summary
    n_chk_a = sum(1 for s in per_sample if s['chk_3_1a_passed'])
    n_chk_b_loyfer = sum(1 for s in per_sample if s['chk_3_1b_loyfer_passed'])
    n_chk_b_bladderref = sum(1 for s in per_sample if s['chk_3_1b_bladderref_passed'])
    n_chk_b_caggiano = sum(1 for s in per_sample if s['chk_3_1b_caggiano_passed'])
    print(f'CHK-3.1A: {n_chk_a}/{len(per_sample)} ({n_chk_a/len(per_sample):.1%})')
    print(f'CHK-3.1B Loyfer:     {n_chk_b_loyfer}/{len(per_sample)} ({n_chk_b_loyfer/len(per_sample):.1%})')
    print(f'CHK-3.1B BladderRef: {n_chk_b_bladderref}/{len(per_sample)} ({n_chk_b_bladderref/len(per_sample):.1%})')
    print(f'CHK-3.1B Caggiano:   {n_chk_b_caggiano}/{len(per_sample)} ({n_chk_b_caggiano/len(per_sample):.1%})')

    # Build paired-pair manifest
    by_case = defaultdict(dict)
    for s in per_sample:
        by_case[s['case_id']][s['sample_type']] = s
    paired_cases = [
        c for c, types in by_case.items()
        if 'Solid Tissue Normal' in types and 'Primary Tumor' in types
    ]
    print(f'\nPaired patients (both adjacent-normal + tumor): {len(paired_cases)}')

    # QC-passed paired pairs (must pass CHK-3.1A on both samples)
    paired_qc = []
    for c in paired_cases:
        n_s = by_case[c]['Solid Tissue Normal']
        t_s = by_case[c]['Primary Tumor']
        if n_s['chk_3_1a_passed'] and t_s['chk_3_1a_passed']:
            paired_qc.append({'case_id': c, 'normal': n_s, 'tumor': t_s})
    print(f'Paired pairs after QC: {len(paired_qc)}')

    # Per-(atlas, tile) statistical contrasts
    contrasts = {}

    def compute_contrasts(atlas_name, tile, key_prefix):
        # Paired
        diffs = []
        for p in paired_qc:
            t_v = p['tumor'][f'{key_prefix}_{tile}']
            n_v = p['normal'][f'{key_prefix}_{tile}']
            if not (np.isnan(t_v) or np.isnan(n_v)):
                diffs.append(t_v - n_v)
        d_p, ci_p_l, ci_p_h, p_p = paired_d_with_ci(diffs)
        dir_p = 'POSITIVE' if not np.isnan(d_p) and d_p > 0 else 'NEGATIVE' if not np.isnan(d_p) and d_p < 0 else 'INSUFFICIENT'

        # Welch (unpaired)
        a_t = [s[f'{key_prefix}_{tile}'] for s in per_sample
               if s['sample_type'] == 'Primary Tumor' and s['chk_3_1a_passed']
               and not np.isnan(s[f'{key_prefix}_{tile}'])]
        a_n = [s[f'{key_prefix}_{tile}'] for s in per_sample
               if s['sample_type'] == 'Solid Tissue Normal' and s['chk_3_1a_passed']
               and not np.isnan(s[f'{key_prefix}_{tile}'])]
        d_w, ci_w_l, ci_w_h, p_w = welch_d_with_ci(a_t, a_n) if a_t and a_n else (float('nan'),)*4
        dir_w = 'POSITIVE' if not np.isnan(d_w) and d_w > 0 else 'NEGATIVE' if not np.isnan(d_w) and d_w < 0 else 'INSUFFICIENT'

        return {
            'atlas': atlas_name,
            'tile': tile,
            'paired': {
                'n_pairs': len(diffs),
                'd': d_p, 'ci_95_low': ci_p_l, 'ci_95_high': ci_p_h, 'p_value': p_p,
                'direction': dir_p,
                'fires': abs(d_p) >= MAGNITUDE_THRESHOLD if not np.isnan(d_p) else False,
            },
            'welch': {
                'n_tumor': len(a_t), 'n_normal': len(a_n),
                'd': d_w, 'ci_95_low': ci_w_l, 'ci_95_high': ci_w_h, 'p_value': p_w,
                'direction': dir_w,
            },
            'tumor_mean': float(np.mean(a_t)) if a_t else float('nan'),
            'tumor_sd': float(np.std(a_t, ddof=1)) if len(a_t) > 1 else 0.0,
            'normal_mean': float(np.mean(a_n)) if a_n else float('nan'),
            'normal_sd': float(np.std(a_n, ddof=1)) if len(a_n) > 1 else 0.0,
        }

    print('\nComputing per-(atlas, tile) contrasts...')
    for tile in LOYFER_TILES:
        contrasts[f'loyfer:{tile}'] = compute_contrasts('loyfer', tile, 'A_loyfer')
    for tile in BLADDERREF_TILES:
        contrasts[f'bladderref:{tile}'] = compute_contrasts('bladderref', tile, 'A_bladderref')
    for tile in CAGGIANO_TILES:
        contrasts[f'caggiano:{tile}'] = compute_contrasts('caggiano', tile, 'A_caggiano')

    # Headline tiles
    print('\n=== Headline cell-of-origin tiles (CCL-039 NEGATIVE expected) ===')
    for atlas, tile in BLADDER_COO_TILES:
        key = f'{atlas}:{tile}'
        c = contrasts[key]
        print(f'  {atlas:11s} {tile:30s} d_paired={c["paired"]["d"]:+.4f}  CI[{c["paired"]["ci_95_low"]:+.3f},{c["paired"]["ci_95_high"]:+.3f}]  p={c["paired"]["p_value"]:.3g}  {c["paired"]["direction"]}')

    print('\n=== BladderRef microenvironment tiles (CCL-039 POSITIVE expected) ===')
    for tile in ['EC', 'Fib', 'IC']:
        c = contrasts[f'bladderref:{tile}']
        print(f'  bladderref  {tile:30s} d_paired={c["paired"]["d"]:+.4f}  CI[{c["paired"]["ci_95_low"]:+.3f},{c["paired"]["ci_95_high"]:+.3f}]  p={c["paired"]["p_value"]:.3g}  {c["paired"]["direction"]}')

    # CHK-3.2 cross-tile sanity check
    print('\n=== CHK-3.2 cross-tile sanity (Loyfer non-bladder tissue tiles) ===')
    cross_tile_flags = []
    for tile in LOYFER_NON_BLADDER_TISSUE_TILES:
        c = contrasts[f'loyfer:{tile}']
        flag = (c['paired']['direction'] == 'POSITIVE'
                and not np.isnan(c['paired']['d'])
                and abs(c['paired']['d']) >= MAGNITUDE_THRESHOLD)
        marker = ' ⚠ FLAGGED' if flag else ''
        if flag:
            cross_tile_flags.append({'tile': tile, 'd': c['paired']['d']})
        print(f'  loyfer:{tile:30s} d_paired={c["paired"]["d"]:+.4f}  {c["paired"]["direction"]}{marker}')

    # Determine outcome
    coo_loyfer = contrasts['loyfer:Bladder']['paired']
    coo_bladderref = contrasts['bladderref:Epi']['paired']
    bladderref_ec = contrasts['bladderref:EC']['paired']
    bladderref_fib = contrasts['bladderref:Fib']['paired']

    qc_pass_rate_loyfer = n_chk_b_loyfer / len(per_sample)
    qc_pass_rate_bladderref = n_chk_b_bladderref / len(per_sample)
    qc_pass_rate_caggiano = n_chk_b_caggiano / len(per_sample)

    if (n_chk_a / len(per_sample) < CHK_3_1A_PASS_RATE_MIN
        or qc_pass_rate_loyfer < CHK_3_1A_PASS_RATE_MIN
        or qc_pass_rate_bladderref < CHK_3_1A_PASS_RATE_MIN
        or qc_pass_rate_caggiano < CHK_3_1A_PASS_RATE_MIN
        or len(paired_qc) < MIN_PAIRED_PAIRS):
        outcome_class = 'O4_STAGE_2_DATA_INTEGRITY_FAILURE'
        outcome_note = f'QC failure or insufficient pairs ({len(paired_qc)})'
    else:
        loyfer_fires_neg = (coo_loyfer['direction'] == 'NEGATIVE'
                            and not np.isnan(coo_loyfer['d'])
                            and abs(coo_loyfer['d']) >= MAGNITUDE_THRESHOLD)
        bref_fires_neg = (coo_bladderref['direction'] == 'NEGATIVE'
                          and not np.isnan(coo_bladderref['d'])
                          and abs(coo_bladderref['d']) >= MAGNITUDE_THRESHOLD)
        microenv_fires_pos = any(
            (c['direction'] == 'POSITIVE' and not np.isnan(c['d']) and abs(c['d']) >= MAGNITUDE_THRESHOLD)
            for c in [bladderref_ec, bladderref_fib]
        )

        loyfer_fires = abs(coo_loyfer['d']) >= MAGNITUDE_THRESHOLD if not np.isnan(coo_loyfer['d']) else False
        bref_fires = abs(coo_bladderref['d']) >= MAGNITUDE_THRESHOLD if not np.isnan(coo_bladderref['d']) else False

        if loyfer_fires_neg and bref_fires_neg and microenv_fires_pos:
            outcome_class = 'O1_MULTI_ATLAS_CONVERGENT_BLADDER_TILE_FIRES'
            outcome_note = (f'Loyfer Bladder d_paired={coo_loyfer["d"]:+.4f} (NEG); '
                            f'BladderRef Epi d_paired={coo_bladderref["d"]:+.4f} (NEG); '
                            f'microenvironment positive')
        elif (loyfer_fires and bref_fires
              and coo_loyfer['direction'] != coo_bladderref['direction']):
            outcome_class = 'O2_BLADDER_TILE_DIFFERENTIATING_DIRECTION_AMBIGUOUS'
            outcome_note = (f'Loyfer Bladder {coo_loyfer["direction"]} d={coo_loyfer["d"]:+.4f} vs '
                            f'BladderRef Epi {coo_bladderref["direction"]} d={coo_bladderref["d"]:+.4f}')
        elif not (loyfer_fires or bref_fires):
            outcome_class = 'O3_STAGE_2_NULL'
            outcome_note = (f'Both COO tiles below |d|=0.30: '
                            f'Loyfer={coo_loyfer["d"]:+.4f}, BladderRef={coo_bladderref["d"]:+.4f}')
        else:
            outcome_class = 'O5_STAGE_2_UNEXPECTED'
            outcome_note = (f'Unexpected pattern: Loyfer={coo_loyfer["d"]:+.4f} ({coo_loyfer["direction"]}), '
                            f'BladderRef={coo_bladderref["d"]:+.4f} ({coo_bladderref["direction"]}), '
                            f'microenv_pos={microenv_fires_pos}')

    runtime = time.time() - t_start

    # Save cross-tile sanity
    sanity_path = OUTPUT_DIR / 'VAL-121_cross_tile_sanity.json'
    with open(sanity_path, 'w') as f:
        json.dump({
            'check_description': 'Loyfer non-bladder solid-tissue tiles paired contrast on TCGA-BLCA. Bladder tumor should NOT fire POSITIVE on these tiles. Flagged if direction=POSITIVE and |d_paired|≥0.30.',
            'flags': cross_tile_flags,
            'all_non_bladder_tiles': {
                tile: {
                    'd_paired': contrasts[f'loyfer:{tile}']['paired']['d'],
                    'direction': contrasts[f'loyfer:{tile}']['paired']['direction'],
                }
                for tile in LOYFER_NON_BLADDER_TISSUE_TILES
            },
        }, f, indent=2)

    # Results JSON
    results = {
        'val_id': VAL_ID,
        'val_type': 'PHASE_C_STAGE2_MULTIATLAS',
        'card_target': 'bladder-epic v0.1',
        'prereg_sha': PREREG_SHA,
        'seal_timestamp': SEAL_TIMESTAMP,
        'rng_seed': RNG_SEED,
        'runtime_seconds': runtime,
        'cohort': {
            'name': 'TCGA-BLCA',
            'substrate': 'TCGA HM450K sesame Level 3',
            'n_total': len(per_sample),
            'n_primary_tumor': sum(1 for s in per_sample if s['sample_type'] == 'Primary Tumor'),
            'n_solid_tissue_normal': sum(1 for s in per_sample if s['sample_type'] == 'Solid Tissue Normal'),
            'n_metastatic': sum(1 for s in per_sample if s['sample_type'] == 'Metastatic'),
            'n_paired_patients_total': len(paired_cases),
            'n_paired_pairs_qc_passed': len(paired_qc),
        },
        'atlases': {
            'loyfer': {
                'n_cpgs': len(loyfer), 'n_tiles': len(LOYFER_TILES),
                'calibration_anchor': 'VAL-112',
                'family': 'tile-coverage WGBS',
            },
            'bladderref': {
                'n_cpgs': len(bladderref), 'n_tiles': len(BLADDERREF_TILES),
                'sha256': BLADDERREF_SHA,
                'calibration_anchor': 'VAL-119',
                'family': 'gene-promoter',
            },
            'caggiano': {
                'n_cpgs': len(caggiano), 'n_tiles': len(CAGGIANO_TILES),
                'calibration_anchor': 'VAL-113',
                'family': 'tile-coverage WGBS',
            },
        },
        'chk_3_1a': {
            'observed_f_extreme_mean': float(np.mean([s['f_extreme'] for s in per_sample])),
            'observed_f_middle_mean': float(np.mean([s['f_middle'] for s in per_sample])),
            'pass_rate': n_chk_a / len(per_sample),
            'gate_passed': n_chk_a / len(per_sample) >= CHK_3_1A_PASS_RATE_MIN,
        },
        'chk_3_1b_per_atlas': {
            'loyfer': {'pass_rate': qc_pass_rate_loyfer, 'gate_passed': qc_pass_rate_loyfer >= CHK_3_1A_PASS_RATE_MIN},
            'bladderref': {'pass_rate': qc_pass_rate_bladderref, 'gate_passed': qc_pass_rate_bladderref >= CHK_3_1A_PASS_RATE_MIN},
            'caggiano': {'pass_rate': qc_pass_rate_caggiano, 'gate_passed': qc_pass_rate_caggiano >= CHK_3_1A_PASS_RATE_MIN},
        },
        'contrasts': contrasts,
        'cross_tile_sanity_flags': cross_tile_flags,
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
