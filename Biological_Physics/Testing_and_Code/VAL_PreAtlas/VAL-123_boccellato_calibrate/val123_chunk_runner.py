#!/usr/bin/env python3
"""
VAL-123 chunked runner — process cohort in batches, persist per-sample
results to NDJSON. Final aggregation happens in a separate step.

This works around tool execution timeouts by allowing each chunk to
complete in <2 min and persist progress to disk.
"""

import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Reuse same constants as the main calibration script
H_MIN_SECRETORY = 0.843264
TILE_NAMES = [
    'Antrum_undiff', 'Antrum_diff',
    'Corpus_undiff', 'Corpus_diff',
    'Fundus_undiff', 'Fundus_diff',
]
TILE_HMIN = {t: H_MIN_SECRETORY for t in TILE_NAMES}

ATLAS_HM450_CSV = Path('/home/claude/gastric_esophageal_sprint/atlas_acquisition/boccellato_stomachref_HM450_v1.csv')
ATLAS_SHA = 'f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390'
KIRC_DIR = Path('/home/claude/edear_working/VAL-106/calibration_betas/KIRC')
PRAD_DIR = Path('/home/claude/edear_working/VAL-106/calibration_betas/PRAD')
OUTPUT_DIR = Path('/home/claude/gastric_esophageal_sprint/VAL-123_boccellato_calibrate')
PERSAMPLE_NDJSON = OUTPUT_DIR / 'val123_per_sample_progress.ndjson'

CHK_3_1A_F_EXTREME_MIN = 0.505
CHK_3_1A_F_MIDDLE_MAX = 0.09
CHK_3_1B_COVERAGE_PER_SAMPLE = 0.80


def load_atlas(path):
    atlas = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            probe = row['CpG_ID']
            tile_betas = {}
            for tile in TILE_NAMES:
                try:
                    tile_betas[tile] = float(row[tile])
                except (ValueError, KeyError):
                    tile_betas[tile] = float('nan')
            atlas[probe] = tile_betas
    return atlas


def load_beta_file_full_and_atlas(path, atlas_cpgs):
    full_betas = []
    atlas_betas = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            cpg = parts[0]
            try:
                beta = float(parts[1])
                if np.isnan(beta):
                    continue
                full_betas.append(beta)
                if cpg in atlas_cpgs:
                    atlas_betas[cpg] = beta
            except ValueError:
                continue
    return np.array(full_betas), atlas_betas


def chk_3_1a(full_beta_array):
    vals = full_beta_array
    f_extreme = float(np.mean((vals < 0.1) | (vals > 0.9)))
    f_middle = float(np.mean((vals >= 0.4) & (vals <= 0.6)))
    median = float(np.median(vals))
    passed = (f_extreme >= CHK_3_1A_F_EXTREME_MIN) and (f_middle <= CHK_3_1A_F_MIDDLE_MAX)
    return f_extreme, f_middle, median, len(vals), passed


def chk_3_1b(betas, atlas_size):
    coverage = len(betas) / atlas_size
    return len(betas), coverage, coverage >= CHK_3_1B_COVERAGE_PER_SAMPLE


def compute_tile_ascore(betas, atlas, tile_name, h_min):
    deltas = []
    for cpg, ref_betas in atlas.items():
        if cpg in betas:
            tile_ref_b = ref_betas[tile_name]
            if np.isnan(tile_ref_b):
                continue
            deltas.append(abs(betas[cpg] - tile_ref_b))
    if not deltas:
        return float('nan'), 0
    return float(np.mean(deltas) / h_min), len(deltas)


def main(start_idx, end_idx):
    """Process cohort samples [start_idx, end_idx) and append to NDJSON."""
    print(f'Loading atlas...')
    atlas = load_atlas(ATLAS_HM450_CSV)
    atlas_cpg_set = set(atlas.keys())
    print(f'  Atlas CpGs: {len(atlas)}')

    kirc_files = sorted(KIRC_DIR.glob('*.txt'))
    prad_files = sorted(PRAD_DIR.glob('*.txt'))
    cohort_files = [(f, 'KIRC') for f in kirc_files] + [(f, 'PRAD') for f in prad_files]
    
    end_idx = min(end_idx, len(cohort_files))
    chunk = cohort_files[start_idx:end_idx]
    print(f'Processing samples [{start_idx}, {end_idx}) of {len(cohort_files)} total')
    print(f'Chunk size: {len(chunk)} samples')

    # Open NDJSON in append mode — each sample written as one JSON line
    t0 = time.time()
    with open(PERSAMPLE_NDJSON, 'a') as out_f:
        for j, (path, project) in enumerate(chunk):
            i = start_idx + j
            sample_id = path.stem.replace('.methylation_array.sesame.level3betas', '')
            
            full_beta_array, betas = load_beta_file_full_and_atlas(path, atlas_cpg_set)
            
            f_extreme, f_middle, median, n_genome, chk_a_pass = chk_3_1a(full_beta_array)
            n_atlas_present, coverage, chk_b_pass = chk_3_1b(betas, len(atlas))
            
            tile_ascores = {}
            tile_n_cpgs = {}
            for tile in TILE_NAMES:
                a, n_cpgs = compute_tile_ascore(betas, atlas, tile, TILE_HMIN[tile])
                tile_ascores[tile] = a
                tile_n_cpgs[tile] = n_cpgs
            
            row = {
                'idx': i,
                'sample_id': sample_id,
                'project': project,
                'n_cpgs_genome': n_genome,
                'f_extreme': f_extreme,
                'f_middle': f_middle,
                'median': median,
                'chk_3_1a_passed': chk_a_pass,
                'n_atlas_cpgs_present': n_atlas_present,
                'coverage': coverage,
                'chk_3_1b_passed': chk_b_pass,
                **{f'A_{t}': tile_ascores[t] for t in TILE_NAMES},
                **{f'n_cpgs_{t}': tile_n_cpgs[t] for t in TILE_NAMES},
            }
            out_f.write(json.dumps(row) + '\n')
            out_f.flush()
            
            if (j+1) % 20 == 0 or (j+1) == len(chunk):
                elapsed = time.time() - t0
                rate = (j+1) / elapsed
                print(f'  [{i+1:3d}/{len(cohort_files)}] processed, rate={rate:.2f}/s')
    
    print(f'Chunk complete. Elapsed: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    main(start, end)
