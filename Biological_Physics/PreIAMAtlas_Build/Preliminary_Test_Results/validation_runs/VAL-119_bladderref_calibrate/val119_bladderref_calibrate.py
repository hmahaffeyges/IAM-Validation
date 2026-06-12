#!/usr/bin/env python3
"""
===============================================================================
VAL-119 — EpiSCORE BladderRef calibration on TCGA-KIRC+PRAD adjacent-normal n=210

Pre-registered: VAL-119_bladderref_calibrate/prereg.md
                SHA-256 sealed in PREREG_SEAL.txt before any β file read.

Question: Does BladderRef CpG-bridged matrix produce per-tile A-score readings
          on TCGA HM450K sesame Level 3 adjacent-normal that pass CHK-3.1A,
          CHK-3.1B, and CHK-3.1C? If yes, seal per-tile healthy-floor
          distributions as the calibration anchor for bladder-epic v0.1.

Atlas:    EpiSCORE BladderRef CpG-bridged
          - 2,696 unique 450K CpGs × 4 bladder cell types + weight
          - SHA-256: 26b7ee3cb7254e28c1dab5bb4bd2c405f35c46f856f429b40aeab087d7f2ca16
          - Cell types: EC, Epi, Fib, IC

Cohort:   TCGA-KIRC adjacent-normal n=160 + TCGA-PRAD adjacent-normal n=50 = 210
          - Substrate: TCGA HM450K sesame Level 3
          - Same calibration cohort as VAL-106/107/112/113 cardio + VAL-117 prostate

H_min by cell type class assignment (G-003b MCMC frozen 2026-04-06):
  - EC  (vascular endothelial)    → stromal     H_min = 0.862950
  - Epi (urothelial epithelium)   → secretory   H_min = 0.843264
  - Fib (fibroblasts)             → stromal     H_min = 0.862950
  - IC  (immune cells)            → immune      H_min = 0.838889

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

# ──────────────────────────────────────────────────────────────────────────────
# FROZEN CONSTANTS (from sealed prereg)
# ──────────────────────────────────────────────────────────────────────────────

VAL_ID = 'VAL-119'
PREREG_SHA = '04d9d0d36faaf3bc6051c5f852f3bcb9d2ab48c437982b714ff8573d4cad178a'
PREREG_AMENDMENT_SHA = 'c3015ca3ba25f6c13f4f93fec85edea8506f64472657d03b59ed9ccda8355787'
SEAL_TIMESTAMP = '2026-05-01T03:35:46Z'
AMENDMENT_TIMESTAMP = '2026-05-01T03:38:56Z'

# H_min anchors (G-003b MCMC posteriors, frozen 2026-04-06)
H_MIN_SECRETORY = 0.843264
H_MIN_STROMAL = 0.862950
H_MIN_IMMUNE = 0.838889

# Cell-type → class assignment for BladderRef
TILE_HMIN = {
    'EC':  H_MIN_STROMAL,
    'Epi': H_MIN_SECRETORY,
    'Fib': H_MIN_STROMAL,
    'IC':  H_MIN_IMMUNE,
}
CELL_TYPES = ['EC', 'Epi', 'Fib', 'IC']

# Atlas paths
ATLAS_BRIDGED_CSV = Path('/home/claude/IAM-Validation/Biological_Physics/atlas_vault/stage2_cell_of_origin/episcore_bladderref/episcore_bladderref_cpg_bridged.csv')
ATLAS_SHA = '3005663b4ede4b20199bacff641952390b1434764b8cf0915cdc9d6a6c1517c6'

# Calibration cohort
KIRC_DIR = Path('/home/claude/edear_working/VAL-106/calibration_betas/KIRC')
PRAD_DIR = Path('/home/claude/edear_working/VAL-106/calibration_betas/PRAD')

# Output
OUTPUT_DIR = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-119_bladderref_calibrate')
OUTPUT_DIR.mkdir(exist_ok=True)

# Pre-locked thresholds (per prereg)
CHK_3_1A_F_EXTREME_MIN = 0.50
CHK_3_1A_F_MIDDLE_MAX = 0.12
CHK_3_1A_PASS_RATE_MIN = 190 / 210  # ≥ 90%
CHK_3_1B_COVERAGE_PER_SAMPLE = 0.80  # CHK-2.8 substrate-floor for TCGA HM450K small atlas subsets
CHK_3_1B_PASS_RATE_MIN = 200 / 210  # ≥ 95%
TISSUE_FLOOR_DOMINATED_RANGE = 0.02

RNG_SEED = 20260420


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def shannon_H(beta):
    """Binary Shannon entropy of beta value."""
    if beta is None or np.isnan(beta) or beta <= 0 or beta >= 1:
        return 0.0
    return -beta * math.log2(beta) - (1 - beta) * math.log2(1 - beta)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def load_beta_file(path):
    """Load TCGA sesame Level 3 β file: tab-separated CpG_id β_value."""
    betas = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            cpg = parts[0]
            try:
                beta = float(parts[1])
                if not np.isnan(beta):
                    betas[cpg] = beta
            except ValueError:
                continue
    return betas


def load_atlas(path):
    """Load the bridged BladderRef matrix as dict[probeID] -> dict[cell_type] -> beta."""
    atlas = {}
    weights = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            probe = row['probeID']
            atlas[probe] = {ct: float(row[ct]) for ct in CELL_TYPES}
            weights[probe] = float(row['weight'])
    return atlas, weights


# ──────────────────────────────────────────────────────────────────────────────
# CHK-3.1C — atlas dedup audit
# ──────────────────────────────────────────────────────────────────────────────

def chk_3_1c_dedup_audit(atlas_path):
    seen = set()
    duplicates = []
    with open(atlas_path) as f:
        next(f)  # skip header
        for line in f:
            probe = line.split(',')[0].strip('"')
            if probe in seen:
                duplicates.append(probe)
            seen.add(probe)
    return {
        'n_total_rows': len(seen) + len(duplicates),
        'n_unique_probes': len(seen),
        'n_duplicates': len(duplicates),
        'passed': len(duplicates) == 0,
        'duplicate_probes_sample': duplicates[:10],
    }


# ──────────────────────────────────────────────────────────────────────────────
# CHK-3.1A — full-genome substrate baseline
# ──────────────────────────────────────────────────────────────────────────────

def chk_3_1a_full_genome(betas):
    vals = np.array(list(betas.values()))
    f_extreme = float(np.mean((vals < 0.1) | (vals > 0.9)))
    f_middle = float(np.mean((vals >= 0.4) & (vals <= 0.6)))
    median = float(np.median(vals))
    passed = (f_extreme >= CHK_3_1A_F_EXTREME_MIN) and (f_middle <= CHK_3_1A_F_MIDDLE_MAX)
    return {
        'f_extreme': f_extreme,
        'f_middle': f_middle,
        'median': median,
        'n_cpgs_genome': len(vals),
        'passed': passed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CHK-3.1B — atlas-subset coverage
# ──────────────────────────────────────────────────────────────────────────────

def chk_3_1b_atlas_subset(betas, atlas):
    n_atlas_cpgs = len(atlas)
    n_present = sum(1 for cpg in atlas if cpg in betas)
    coverage = n_present / n_atlas_cpgs if n_atlas_cpgs > 0 else 0.0
    return {
        'n_atlas_cpgs': n_atlas_cpgs,
        'n_present_in_sample': n_present,
        'coverage': coverage,
        'passed': coverage >= CHK_3_1B_COVERAGE_PER_SAMPLE,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-tile A-score
# ──────────────────────────────────────────────────────────────────────────────

def compute_tile_ascore(betas, atlas, cell_type, h_min):
    """Compute pooled-entropy A-score for one cell-type tile.

    For each atlas CpG, compute |sample_β - tile_ref_β|. Mean over CpGs
    present in the sample AND with non-NaN tile reference. Normalize by
    H_min for the cell type's class.

    Equivalent methodology to VAL-111 HeartRef + VAL-117 ProstateRef tile A-scoring.
    """
    deltas = []
    for cpg, ref_betas in atlas.items():
        if cpg in betas:
            sample_b = betas[cpg]
            tile_ref_b = ref_betas[cell_type]
            if np.isnan(tile_ref_b):
                continue
            deltas.append(abs(sample_b - tile_ref_b))
    if not deltas:
        return float('nan'), 0
    a_score = np.mean(deltas) / h_min
    return float(a_score), len(deltas)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f'VAL-119 — BladderRef calibration on TCGA-KIRC+PRAD n=210')
    print(f'Prereg SHA: {PREREG_SHA}')
    print(f'Sealed:     {SEAL_TIMESTAMP}')
    print()

    # Verify atlas SHA
    actual_sha = sha256_file(ATLAS_BRIDGED_CSV)
    assert actual_sha == ATLAS_SHA, f'Atlas SHA mismatch: {actual_sha} != {ATLAS_SHA}'
    print(f'Atlas SHA verified: {ATLAS_SHA}')

    # CHK-3.1C dedup
    print('CHK-3.1C dedup audit...')
    dedup = chk_3_1c_dedup_audit(ATLAS_BRIDGED_CSV)
    print(f'  Unique probes: {dedup["n_unique_probes"]}, duplicates: {dedup["n_duplicates"]}, passed: {dedup["passed"]}')
    if not dedup['passed']:
        print(f'  DUPLICATE PROBES FOUND — calibration aborts per CHK-3.1C')
        return

    # Load atlas
    print('Loading bridged atlas...')
    atlas, weights = load_atlas(ATLAS_BRIDGED_CSV)
    print(f'  Atlas CpGs: {len(atlas)}')

    # Enumerate cohort
    kirc_files = sorted(KIRC_DIR.glob('*.txt'))
    prad_files = sorted(PRAD_DIR.glob('*.txt'))
    cohort_files = [(f, 'KIRC') for f in kirc_files] + [(f, 'PRAD') for f in prad_files]
    print(f'Cohort: {len(kirc_files)} KIRC + {len(prad_files)} PRAD = {len(cohort_files)} total')
    print()

    # Per-sample loop
    per_sample = []
    per_tile_ascores = defaultdict(list)
    per_tile_n_cpgs = defaultdict(list)

    for i, (path, project) in enumerate(cohort_files, 1):
        if i % 30 == 0:
            print(f'  {i}/{len(cohort_files)} samples processed')
        sample_id = path.stem.replace('.methylation_array.sesame.level3betas', '')
        betas = load_beta_file(path)

        # CHK-3.1A
        chk_a = chk_3_1a_full_genome(betas)

        # CHK-3.1B
        chk_b = chk_3_1b_atlas_subset(betas, atlas)

        # Per-tile A-scores
        tile_ascores = {}
        tile_n_cpgs = {}
        for ct in CELL_TYPES:
            a, n_cpgs = compute_tile_ascore(betas, atlas, ct, TILE_HMIN[ct])
            tile_ascores[ct] = a
            tile_n_cpgs[ct] = n_cpgs
            if not np.isnan(a):
                per_tile_ascores[ct].append(a)
                per_tile_n_cpgs[ct].append(n_cpgs)

        per_sample.append({
            'sample_id': sample_id,
            'project': project,
            'n_cpgs_genome': chk_a['n_cpgs_genome'],
            'f_extreme': chk_a['f_extreme'],
            'f_middle': chk_a['f_middle'],
            'median': chk_a['median'],
            'chk_3_1a_passed': chk_a['passed'],
            'n_atlas_cpgs_present': chk_b['n_present_in_sample'],
            'coverage': chk_b['coverage'],
            'chk_3_1b_passed': chk_b['passed'],
            **{f'A_{ct}': tile_ascores[ct] for ct in CELL_TYPES},
            **{f'n_cpgs_{ct}': tile_n_cpgs[ct] for ct in CELL_TYPES},
        })

    # Pass rate computation
    n_chk_3_1a_pass = sum(1 for s in per_sample if s['chk_3_1a_passed'])
    n_chk_3_1b_pass = sum(1 for s in per_sample if s['chk_3_1b_passed'])

    # Per-tile distributions (from samples that passed both gates)
    qc_passed_samples = [s for s in per_sample if s['chk_3_1a_passed'] and s['chk_3_1b_passed']]
    print(f'\nQC summary:')
    print(f'  CHK-3.1A pass: {n_chk_3_1a_pass}/{len(per_sample)} ({n_chk_3_1a_pass/len(per_sample):.1%})')
    print(f'  CHK-3.1B pass: {n_chk_3_1b_pass}/{len(per_sample)} ({n_chk_3_1b_pass/len(per_sample):.1%})')
    print(f'  Both pass:     {len(qc_passed_samples)}/{len(per_sample)}')
    print()

    per_tile_distributions = {}
    print('Per-tile healthy-floor distributions (QC-passed samples):')
    for ct in CELL_TYPES:
        vals = [s[f'A_{ct}'] for s in qc_passed_samples if not np.isnan(s[f'A_{ct}'])]
        if not vals:
            per_tile_distributions[ct] = None
            continue
        arr = np.array(vals)
        per_tile_distributions[ct] = {
            'mean': float(arr.mean()),
            'sd': float(arr.std(ddof=1)),
            'n': len(arr),
            'q2_5': float(np.percentile(arr, 2.5)),
            'q5': float(np.percentile(arr, 5)),
            'q50': float(np.percentile(arr, 50)),
            'q95': float(np.percentile(arr, 95)),
            'q97_5': float(np.percentile(arr, 97.5)),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'within_cohort_range': float(arr.max() - arr.min()),
        }
        d = per_tile_distributions[ct]
        print(f'  {ct:5s} (H_min={TILE_HMIN[ct]:.6f}): mean={d["mean"]:.4f} sd={d["sd"]:.4f} n={d["n"]} q5={d["q5"]:.4f} q95={d["q95"]:.4f} range={d["within_cohort_range"]:.4f}')

    # Determine outcome
    chk_3_1a_pass_rate = n_chk_3_1a_pass / len(per_sample)
    chk_3_1b_pass_rate = n_chk_3_1b_pass / len(per_sample)

    # Tissue-floor-dominated check: max within-cohort range across all tiles
    max_range = max((d['within_cohort_range'] for d in per_tile_distributions.values() if d), default=0)
    mean_within_range = max_range  # use max as the diagnostic, same as VAL-117

    if not dedup['passed']:
        outcome_class = 'O4_BLADDERREF_BRIDGE_FAILURE'
        outcome_note = 'CHK-3.1C dedup failed'
    elif chk_3_1a_pass_rate >= CHK_3_1A_PASS_RATE_MIN and chk_3_1b_pass_rate >= CHK_3_1B_PASS_RATE_MIN:
        # Check for tissue-floor-dominated pattern
        if mean_within_range < TISSUE_FLOOR_DOMINATED_RANGE:
            outcome_class = 'O3_BLADDERREF_TISSUE_FLOOR_DOMINATED'
            outcome_note = f'Within-cohort tile range max = {mean_within_range:.4f} (< 0.02 threshold). Gene-promoter atlas family pattern.'
        else:
            outcome_class = 'O1_BLADDERREF_CALIBRATION_SEALED'
            outcome_note = f'CHK-3.1A {chk_3_1a_pass_rate:.1%}, CHK-3.1B {chk_3_1b_pass_rate:.1%}, max tile range {mean_within_range:.4f}'
    elif chk_3_1a_pass_rate >= 0.75 and chk_3_1b_pass_rate >= 0.85:
        outcome_class = 'O2_BLADDERREF_CALIBRATION_PARTIAL'
        outcome_note = f'CHK-3.1A {chk_3_1a_pass_rate:.1%}, CHK-3.1B {chk_3_1b_pass_rate:.1%}'
    else:
        outcome_class = 'O5_BLADDERREF_UNEXPECTED'
        outcome_note = f'CHK-3.1A {chk_3_1a_pass_rate:.1%}, CHK-3.1B {chk_3_1b_pass_rate:.1%}'

    # CHK-3.1B q5 threshold (5th percentile of QC-passed coverage)
    coverage_vals = [s['coverage'] for s in qc_passed_samples]
    chk_3_1b_q5 = float(np.percentile(coverage_vals, 5)) if coverage_vals else 0

    runtime = time.time() - t_start

    # Results JSON
    results = {
        'val_id': VAL_ID,
        'val_type': 'PHASE_B_CALIBRATION',
        'card_target': 'bladder-epic v0.1',
        'atlas_target': 'EpiSCORE BladderRef CpG-bridged',
        'atlas_sha256': ATLAS_SHA,
        'atlas_n_cpgs': len(atlas),
        'atlas_cell_types': CELL_TYPES,
        'prereg_sha': PREREG_SHA,
        'seal_timestamp': SEAL_TIMESTAMP,
        'rng_seed': RNG_SEED,
        'runtime_seconds': runtime,
        'cohort': {
            'name': 'TCGA-KIRC + TCGA-PRAD adjacent-normal',
            'substrate': 'TCGA HM450K sesame Level 3',
            'n_total': len(per_sample),
            'n_kirc': len(kirc_files),
            'n_prad': len(prad_files),
        },
        'chk_3_1a': {
            'pre_locked_f_extreme_min': CHK_3_1A_F_EXTREME_MIN,
            'pre_locked_f_middle_max': CHK_3_1A_F_MIDDLE_MAX,
            'observed_f_extreme_mean': float(np.mean([s['f_extreme'] for s in per_sample])),
            'observed_f_extreme_sd': float(np.std([s['f_extreme'] for s in per_sample], ddof=1)),
            'observed_f_middle_mean': float(np.mean([s['f_middle'] for s in per_sample])),
            'observed_f_middle_sd': float(np.std([s['f_middle'] for s in per_sample], ddof=1)),
            'pass_rate': chk_3_1a_pass_rate,
            'n_passed': n_chk_3_1a_pass,
            'pass_rate_threshold': CHK_3_1A_PASS_RATE_MIN,
            'gate_passed': chk_3_1a_pass_rate >= CHK_3_1A_PASS_RATE_MIN,
        },
        'chk_3_1b': {
            'pre_locked_coverage_threshold': CHK_3_1B_COVERAGE_PER_SAMPLE,
            'pass_rate': chk_3_1b_pass_rate,
            'n_passed': n_chk_3_1b_pass,
            'q5_coverage_threshold_observed': chk_3_1b_q5,
            'gate_passed': chk_3_1b_pass_rate >= CHK_3_1B_PASS_RATE_MIN,
        },
        'chk_3_1c': dedup,
        'per_tile_healthy_floor_distributions': per_tile_distributions,
        'per_tile_max_within_cohort_range': mean_within_range,
        'tissue_floor_dominated_threshold': TISSUE_FLOOR_DOMINATED_RANGE,
        'outcome_class': outcome_class,
        'outcome_note': outcome_note,
        'sealed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    results_path = OUTPUT_DIR / f'{VAL_ID}_calibration_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    results_sha = sha256_file(results_path)
    results['results_sha256'] = results_sha
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Per-sample CSV
    per_sample_csv = OUTPUT_DIR / f'{VAL_ID}_per_sample_calibration.csv'
    with open(per_sample_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        for row in per_sample:
            w.writerow(row)

    print()
    print('=' * 70)
    print(f'OUTCOME: {outcome_class}')
    print(f'  {outcome_note}')
    print('=' * 70)
    print(f'Runtime: {runtime:.1f} sec')
    print(f'Results: {results_path}')
    print(f'Per-sample: {per_sample_csv}')


if __name__ == '__main__':
    main()
