#!/usr/bin/env python3
"""
===============================================================================
VAL-123 — BoccellatoStomachRef_HM450 v1 calibration on TCGA-KIRC+PRAD
          adjacent-normal n=210

Pre-registered: VAL-123_boccellato_calibrate/prereg.md
                SHA-256 91a5f06f984ce64b747bd2800fdde28f18dbc3105008f2008730a1ee659d71d7
                Sealed in PREREG_SEAL.txt 2026-05-02T18:45:54Z BEFORE any β read.

Question: Does BoccellatoStomachRef_HM450 v1 produce per-tile A-score readings
          on TCGA HM450K sesame Level 3 adjacent-normal that pass CHK-3.1A,
          CHK-3.1B, and CHK-3.1C? If yes, seal per-tile healthy-floor
          distributions as the calibration anchor for gastric+esophageal-epic
          v0.1.

Atlas:    BoccellatoStomachRef_HM450 v1 (HM450-restricted derivative of EPIC build)
          - 380,467 unique HM450 CpGs × 6 gastric mucosoid tiles
          - SHA-256: f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390
          - Tiles: Antrum_undiff, Antrum_diff, Corpus_undiff, Corpus_diff,
                   Fundus_undiff, Fundus_diff
          - Source: GSE141660 (Boccellato 2022 Clin Epigenetics)
          - Restriction: EPIC 850K → HM450 platform CpG intersection
          - CHK-2.17 pre-flight: PASS at 95.56% mean / 94.62% min coverage

Cohort:   TCGA-KIRC adjacent-normal n=160 + TCGA-PRAD adjacent-normal n=50 = 210
          - Substrate: TCGA HM450K sesame Level 3
          - Same calibration cohort as VAL-106/107/112/113 cardio + VAL-117 prostate
            + VAL-119 bladder

H_min by cell type class assignment (G-003b MCMC frozen 2026-04-06):
  All 6 tiles are gastric mucosoid epithelium (purified primary cells from
  sleeve resections, cultivated as plane mucosoids). Lineage assignment per
  CCL-038 + parallel to BladderRef Epi:

  - Antrum_undiff   → secretory  H_min = 0.843264  (gastric epithelium, stem-enriched)
  - Antrum_diff     → secretory  H_min = 0.843264  (gastric epithelium, pit-cell-like)
  - Corpus_undiff   → secretory  H_min = 0.843264
  - Corpus_diff     → secretory  H_min = 0.843264
  - Fundus_undiff   → secretory  H_min = 0.843264
  - Fundus_diff     → secretory  H_min = 0.843264

RNG seed: 20260502
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

VAL_ID = 'VAL-123'
PREREG_SHA = '91a5f06f984ce64b747bd2800fdde28f18dbc3105008f2008730a1ee659d71d7'
SEAL_TIMESTAMP = '2026-05-02T18:45:54Z'

# H_min anchors (G-003b MCMC posteriors, frozen 2026-04-06)
H_MIN_SECRETORY = 0.843264

# Cell-type → class assignment for BoccellatoStomachRef
TILE_NAMES = [
    'Antrum_undiff', 'Antrum_diff',
    'Corpus_undiff', 'Corpus_diff',
    'Fundus_undiff', 'Fundus_diff',
]
TILE_HMIN = {t: H_MIN_SECRETORY for t in TILE_NAMES}

# Atlas paths
ATLAS_HM450_CSV = Path('/home/claude/gastric_esophageal_sprint/atlas_acquisition/boccellato_stomachref_HM450_v1.csv')
ATLAS_SHA = 'f5a620a93aba40d0567346d156ce7ea2861f8ed38ee1bd669a4ff52b261fa390'

# Calibration cohort
KIRC_DIR = Path('/home/claude/edear_working/VAL-106/calibration_betas/KIRC')
PRAD_DIR = Path('/home/claude/edear_working/VAL-106/calibration_betas/PRAD')

# Output
OUTPUT_DIR = Path('/home/claude/gastric_esophageal_sprint/VAL-123_boccellato_calibrate')
OUTPUT_DIR.mkdir(exist_ok=True)

# Pre-locked thresholds (per prereg + VAL-106 + VAL-119 precedent)
# TCGA HM450K sesame Level 3 substrate floor: f_extreme ≥ 0.505, f_middle ≤ 0.09
CHK_3_1A_F_EXTREME_MIN = 0.505
CHK_3_1A_F_MIDDLE_MAX = 0.09
CHK_3_1A_PASS_RATE_MIN = 190 / 210  # ≥ ~90%
CHK_3_1B_COVERAGE_PER_SAMPLE = 0.80  # CHK-2.8 substrate-floor for TCGA HM450K
CHK_3_1B_PASS_RATE_MIN = 200 / 210  # ≥ ~95%
TISSUE_FLOOR_DOMINATED_RANGE = 0.02  # max within-cohort tile range threshold

RNG_SEED = 20260502


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


def load_beta_file(path, cpgs_of_interest=None):
    """Load TCGA sesame Level 3 β file: tab-separated CpG_id β_value.

    If cpgs_of_interest is provided, only retain those CpGs (much faster
    when atlas has 380K of the platform's 485K probes; we still need them all
    for CHK-3.1A full-genome bimodality check, so this is a two-pass split).
    """
    betas = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            cpg = parts[0]
            if cpgs_of_interest is not None and cpg not in cpgs_of_interest:
                continue
            try:
                beta = float(parts[1])
                if not np.isnan(beta):
                    betas[cpg] = beta
            except ValueError:
                continue
    return betas


def load_beta_file_full_and_atlas(path, atlas_cpgs):
    """Single-pass load: returns (full_genome_beta_array, atlas_betas_dict).

    full_genome_beta_array is a numpy array of all valid β values (for CHK-3.1A).
    atlas_betas_dict is a dict[cpg→β] restricted to atlas CpGs (for CHK-3.1B + A-scoring).
    """
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


def load_atlas(path):
    """Load BoccellatoStomachRef_HM450 atlas: dict[probeID] → dict[tile_name] → β."""
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

def chk_3_1a_full_genome(full_beta_array):
    """Full-genome bimodality check on a numpy array of β values."""
    vals = full_beta_array
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

def compute_tile_ascore(betas, atlas, tile_name, h_min):
    """Compute pooled-mean-deviation A-score for one gastric tile.

    For each atlas CpG, compute |sample_β - tile_ref_β|. Mean over CpGs
    present in the sample AND with non-NaN tile reference. Normalize by H_min.

    Same methodology as VAL-117 ProstateRef + VAL-119 BladderRef tile A-scoring.
    """
    deltas = []
    for cpg, ref_betas in atlas.items():
        if cpg in betas:
            tile_ref_b = ref_betas[tile_name]
            if np.isnan(tile_ref_b):
                continue
            sample_b = betas[cpg]
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
    print(f'VAL-123 — BoccellatoStomachRef_HM450 v1 calibration on TCGA-KIRC+PRAD n=210')
    print(f'Prereg SHA: {PREREG_SHA}')
    print(f'Sealed:     {SEAL_TIMESTAMP}')
    print()

    # Verify atlas SHA
    actual_sha = sha256_file(ATLAS_HM450_CSV)
    assert actual_sha == ATLAS_SHA, f'Atlas SHA mismatch: {actual_sha} != {ATLAS_SHA}'
    print(f'Atlas SHA verified: {ATLAS_SHA}')

    # CHK-3.1C dedup audit
    print('CHK-3.1C dedup audit...')
    dedup = chk_3_1c_dedup_audit(ATLAS_HM450_CSV)
    print(f'  Unique probes: {dedup["n_unique_probes"]}, duplicates: {dedup["n_duplicates"]}, passed: {dedup["passed"]}')
    if not dedup['passed']:
        print(f'  DUPLICATE PROBES FOUND — calibration aborts per CHK-3.1C')
        return

    # Load atlas
    print('Loading HM450-restricted atlas...')
    atlas = load_atlas(ATLAS_HM450_CSV)
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

    atlas_cpg_set = set(atlas.keys())

    for i, (path, project) in enumerate(cohort_files, 1):
        if i % 30 == 0:
            print(f'  {i}/{len(cohort_files)} samples processed')
        sample_id = path.stem.replace('.methylation_array.sesame.level3betas', '')

        # Single-pass: full-genome β array + atlas-CpG-restricted β dict
        full_beta_array, betas = load_beta_file_full_and_atlas(path, atlas_cpg_set)

        # CHK-3.1A
        chk_a = chk_3_1a_full_genome(full_beta_array)

        # CHK-3.1B
        chk_b = chk_3_1b_atlas_subset(betas, atlas)

        # Per-tile A-scores
        tile_ascores = {}
        tile_n_cpgs = {}
        for tile in TILE_NAMES:
            a, n_cpgs = compute_tile_ascore(betas, atlas, tile, TILE_HMIN[tile])
            tile_ascores[tile] = a
            tile_n_cpgs[tile] = n_cpgs
            if not np.isnan(a):
                per_tile_ascores[tile].append(a)
                per_tile_n_cpgs[tile].append(n_cpgs)

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
            **{f'A_{t}': tile_ascores[t] for t in TILE_NAMES},
            **{f'n_cpgs_{t}': tile_n_cpgs[t] for t in TILE_NAMES},
        })

    # Pass rate computation
    n_chk_3_1a_pass = sum(1 for s in per_sample if s['chk_3_1a_passed'])
    n_chk_3_1b_pass = sum(1 for s in per_sample if s['chk_3_1b_passed'])
    qc_passed_samples = [s for s in per_sample if s['chk_3_1a_passed'] and s['chk_3_1b_passed']]
    print(f'\nQC summary:')
    print(f'  CHK-3.1A pass: {n_chk_3_1a_pass}/{len(per_sample)} ({n_chk_3_1a_pass/len(per_sample):.1%})')
    print(f'  CHK-3.1B pass: {n_chk_3_1b_pass}/{len(per_sample)} ({n_chk_3_1b_pass/len(per_sample):.1%})')
    print(f'  Both pass:     {len(qc_passed_samples)}/{len(per_sample)}')
    print()

    # Per-tile distributions
    per_tile_distributions = {}
    print('Per-tile healthy-floor distributions (QC-passed samples):')
    for tile in TILE_NAMES:
        vals = [s[f'A_{tile}'] for s in qc_passed_samples if not np.isnan(s[f'A_{tile}'])]
        if not vals:
            per_tile_distributions[tile] = None
            continue
        arr = np.array(vals)
        per_tile_distributions[tile] = {
            'mean': float(arr.mean()),
            'sd': float(arr.std(ddof=1)),
            'n': len(arr),
            'q2_5': float(np.percentile(arr, 2.5)),
            'q5': float(np.percentile(arr, 5)),
            'q25': float(np.percentile(arr, 25)),
            'q50': float(np.percentile(arr, 50)),
            'q75': float(np.percentile(arr, 75)),
            'q95': float(np.percentile(arr, 95)),
            'q97_5': float(np.percentile(arr, 97.5)),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'within_cohort_range': float(arr.max() - arr.min()),
        }
        d = per_tile_distributions[tile]
        print(f'  {tile:18s} (H_min={TILE_HMIN[tile]:.6f}, secretory): mean={d["mean"]:.4f} sd={d["sd"]:.4f} n={d["n"]} q5={d["q5"]:.4f} q95={d["q95"]:.4f} range={d["within_cohort_range"]:.4f}')

    # Per-tissue stratification (KIRC vs PRAD)
    per_tile_per_tissue = {}
    print('\nPer-tile per-tissue A-score means (QC-passed samples):')
    for tile in TILE_NAMES:
        kirc_vals = [s[f'A_{tile}'] for s in qc_passed_samples if s['project']=='KIRC' and not np.isnan(s[f'A_{tile}'])]
        prad_vals = [s[f'A_{tile}'] for s in qc_passed_samples if s['project']=='PRAD' and not np.isnan(s[f'A_{tile}'])]
        per_tile_per_tissue[tile] = {
            'KIRC': {'n': len(kirc_vals), 'mean': float(np.mean(kirc_vals)) if kirc_vals else None,
                     'sd': float(np.std(kirc_vals, ddof=1)) if len(kirc_vals)>1 else None},
            'PRAD': {'n': len(prad_vals), 'mean': float(np.mean(prad_vals)) if prad_vals else None,
                     'sd': float(np.std(prad_vals, ddof=1)) if len(prad_vals)>1 else None},
        }
        kk = per_tile_per_tissue[tile]['KIRC']
        pp = per_tile_per_tissue[tile]['PRAD']
        diff_in_kirc_sd = (pp['mean'] - kk['mean'])/kk['sd'] if (kk['sd'] and kk['mean'] is not None and pp['mean'] is not None) else None
        diff_str = f"{diff_in_kirc_sd:+.2f} KIRC-SD" if diff_in_kirc_sd is not None else "n/a"
        print(f'  {tile:18s}: KIRC mean={kk["mean"]:.4f} (n={kk["n"]}), PRAD mean={pp["mean"]:.4f} (n={pp["n"]}), diff={diff_str}')

    # Atlas-family-fitness: max within-cohort tile range
    valid_tile_stats = {t: d for t, d in per_tile_distributions.items() if d}
    max_range_per_tile = max((d['within_cohort_range'] for d in valid_tile_stats.values()), default=0)

    # Cross-tile separation: difference between max tile mean and min tile mean
    tile_means = [d['mean'] for d in valid_tile_stats.values()]
    cross_tile_separation = max(tile_means) - min(tile_means) if tile_means else 0
    cross_tile_min = min(tile_means) if tile_means else None
    cross_tile_max = max(tile_means) if tile_means else None
    print(f'\nCross-tile separation diagnostic:')
    print(f'  Min tile mean: {cross_tile_min:.4f}, Max tile mean: {cross_tile_max:.4f}')
    print(f'  Cross-tile separation: {cross_tile_separation:.4f}')
    print(f'  Max within-cohort tile range: {max_range_per_tile:.4f}')

    # Determine outcome
    chk_3_1a_pass_rate = n_chk_3_1a_pass / len(per_sample)
    chk_3_1b_pass_rate = n_chk_3_1b_pass / len(per_sample)

    if not dedup['passed']:
        outcome_class = 'O4_BRIDGE_FAILURE'
        outcome_note = 'CHK-3.1C dedup failed'
    elif chk_3_1a_pass_rate >= CHK_3_1A_PASS_RATE_MIN and chk_3_1b_pass_rate >= CHK_3_1B_PASS_RATE_MIN:
        if max_range_per_tile < TISSUE_FLOOR_DOMINATED_RANGE:
            outcome_class = 'O3_TISSUE_FLOOR_DOMINATED'
            outcome_note = (f'Within-cohort tile range max = {max_range_per_tile:.4f} '
                           f'(< {TISSUE_FLOOR_DOMINATED_RANGE} threshold). All 6 tiles collapse to '
                           f'substrate floor on non-stomach healthy tissue. Per-tile q5 thresholds '
                           f'still sealed; atlas is stomach-tissue-only detector (acceptable v0.1 outcome).')
        else:
            # Check whether 4-5 tiles separate or all 6 do
            tiles_with_sd_above_005 = sum(1 for d in valid_tile_stats.values() if d['sd'] >= 0.005)
            if tiles_with_sd_above_005 >= 6:
                outcome_class = 'O1_TILES_DIFFERENTIATING_HEALTHY_FLOORS_SEALED'
                outcome_note = (f'All 6 tiles produce non-degenerate distributions on healthy substrate. '
                               f'CHK-3.1A {chk_3_1a_pass_rate:.1%}, CHK-3.1B {chk_3_1b_pass_rate:.1%}, '
                               f'max tile range {max_range_per_tile:.4f}. Per-tile q5 thresholds sealed.')
            elif tiles_with_sd_above_005 >= 4:
                outcome_class = 'O2_PARTIAL_FLOORS'
                outcome_note = (f'{tiles_with_sd_above_005}/6 tiles produce non-degenerate distributions. '
                               f'Per-tile q5 thresholds sealed where calculable; collapsed tiles flagged.')
            else:
                outcome_class = 'O3_TISSUE_FLOOR_DOMINATED'
                outcome_note = (f'Only {tiles_with_sd_above_005}/6 tiles produce non-degenerate '
                               f'distributions. Atlas behaves as substrate-floor-dominated on non-stomach.')
    elif chk_3_1a_pass_rate >= 0.75 and chk_3_1b_pass_rate >= 0.85:
        outcome_class = 'O4_QC_PARTIAL'
        outcome_note = f'CHK-3.1A {chk_3_1a_pass_rate:.1%}, CHK-3.1B {chk_3_1b_pass_rate:.1%}'
    else:
        outcome_class = 'O5_BRIDGE_FAILURE'
        outcome_note = (f'CHK-3.1A {chk_3_1a_pass_rate:.1%} or CHK-3.1B {chk_3_1b_pass_rate:.1%} '
                       f'below acceptable. Atlas calibration cannot be sealed.')

    print(f'\nOUTCOME: {outcome_class}')
    print(f'  {outcome_note}')

    # Save results
    elapsed_total = time.time() - t_start
    results = {
        'val_id': VAL_ID,
        'prereg_sha256': PREREG_SHA,
        'seal_timestamp': SEAL_TIMESTAMP,
        'executed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'rng_seed': RNG_SEED,
        'atlas': {
            'name': 'BoccellatoStomachRef_HM450 v1',
            'path': str(ATLAS_HM450_CSV),
            'sha256': ATLAS_SHA,
            'n_cpgs': len(atlas),
            'tiles': TILE_NAMES,
            'tile_class_assignment': {t: 'secretory' for t in TILE_NAMES},
            'tile_h_min': {t: TILE_HMIN[t] for t in TILE_NAMES},
        },
        'cohort': {
            'KIRC_n': len(kirc_files),
            'PRAD_n': len(prad_files),
            'total_n': len(cohort_files),
        },
        'chk_3_1c_dedup': dedup,
        'qc_summary': {
            'chk_3_1a_pass_n': n_chk_3_1a_pass,
            'chk_3_1a_pass_rate': chk_3_1a_pass_rate,
            'chk_3_1b_pass_n': n_chk_3_1b_pass,
            'chk_3_1b_pass_rate': chk_3_1b_pass_rate,
            'both_pass_n': len(qc_passed_samples),
            'both_pass_rate': len(qc_passed_samples) / len(per_sample),
        },
        'per_tile_distributions': per_tile_distributions,
        'per_tile_per_tissue': per_tile_per_tissue,
        'cross_tile_separation': float(cross_tile_separation),
        'max_within_cohort_tile_range': float(max_range_per_tile),
        'outcome': outcome_class,
        'outcome_note': outcome_note,
        'elapsed_seconds': elapsed_total,
    }

    results_path = OUTPUT_DIR / 'VAL-123_calibration_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nResults saved: {results_path}')

    # Per-sample CSV
    per_sample_path = OUTPUT_DIR / 'VAL-123_per_sample_calibration.csv'
    with open(per_sample_path, 'w', newline='') as f:
        if per_sample:
            keys = list(per_sample[0].keys())
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_sample)
    print(f'Per-sample CSV saved: {per_sample_path}')

    print(f'\nElapsed: {elapsed_total:.1f}s')


if __name__ == '__main__':
    main()
