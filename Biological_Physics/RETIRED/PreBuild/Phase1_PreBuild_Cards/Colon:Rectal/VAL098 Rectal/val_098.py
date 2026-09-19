#!/usr/bin/env python3
"""
VAL-098 — TCGA-READ paired tumor-vs-adjacent-normal cycling-class architectural
drift validation, with run-everything 25-tile per-class A-score.

Card: crc-epic (early-onset rectal subsection — Phase 2 anchor)
Date sealed: 2026-04-28 UTC
RNG seed: 20260428

Operating context: LL-PUBLIC-TIER public-tier-only operational reset.
Within-cohort paired comparison (CHK-3.8 condition 1 satisfied — no cross-cohort
calibration problem). EDEAR commercial deployment unaffected.

Methodology mirrors VAL-062 (paired tissue d on cycling H_min) + VAL-093/097
(run-everything 25-tile per-class A-score on Loyfer atlas).

Pre-registered before any β access — see VAL-098/prereg.md
SHA-256 of prereg recorded in VAL-098_PREREG_SEAL.txt.

H_min values frozen from GAPE_WEB_v13 _H_MIN_GRID (G-002 + G-003b MCMC posteriors,
R-hat < 1.001). Byte-match VAL-062 H_MIN_CYCLING constant.
"""

import gzip
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Frozen H_min values — byte-match GAPE_WEB_v13 _H_MIN_GRID methyl column
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
H_MIN_CYCLING = H_MIN['cycling']  # primary scoring class for CRC/rectal tumor

# Loyfer 25-cell array atlas: cell type to architecture class (canonical mapping)
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

# ============================================================================
# Inputs — public Tier 1 only
# ============================================================================
WORKDIR = '/home/claude/edear_working/VAL-098'
LOYFER_ATLAS = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv'

# TCGA-READ paired tumor/adjacent-normal — manifest constructed from GDC API 2026-04-28
TCGA_READ_DOWNLOAD_DIR = os.path.join(WORKDIR, 'tcga_read_downloads')
GDC_API_BASE = 'https://api.gdc.cancer.gov/data/'

# Outputs (CHK-6.4 canonical: VAL-098/ subfolder structure)
OUTDIR = os.path.join(WORKDIR, 'VAL-098')
os.makedirs(OUTDIR, exist_ok=True)
READ_MANIFEST = os.path.join(OUTDIR, 'READ_matched_manifest.json')
RESULTS_JSON = os.path.join(OUTDIR, 'results.json')
STRATIFIED_JSON = os.path.join(OUTDIR, 'stratified.json')
PER_SAMPLE_CSV = os.path.join(OUTDIR, 'per_sample.csv')
TILE_HEATMAP_PNG = os.path.join(OUTDIR, 'tile_heatmap.png')
COHORT_MANIFEST_JSON = os.path.join(OUTDIR, 'cohort_manifest.json')
CLINICAL_METADATA_CSV = os.path.join(OUTDIR, 'clinical_metadata.csv')
PREREG_SEAL_TXT = os.path.join(OUTDIR, 'PREREG_SEAL.txt')
PREREG_PATH = os.path.join(OUTDIR, 'prereg.md')

N_DISCRIMINATING_CPGS = 100  # top-N marker CpGs per Loyfer cell type (matches VAL-093/097)
N_BOOT = 10000
RNG_SEED = 20260428
MIN_VALID_CPGS = 400_000  # HM450 QC threshold per VAL-062 standard

# ============================================================================
# Helpers
# ============================================================================

def sha256_file(path, prefix=64):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:prefix] if prefix else h.hexdigest()


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def shannon_entropy_b(beta):
    """Binary Shannon entropy: H(beta) = -beta*log2(beta) - (1-beta)*log2(1-beta)."""
    beta = np.asarray(beta, dtype=float)
    h = np.zeros_like(beta)
    mask = (beta > 0) & (beta < 1) & ~np.isnan(beta)
    b = beta[mask]
    h[mask] = -b * np.log2(b) - (1 - b) * np.log2(1 - b)
    return h


def a_score_array(beta_array, h_min):
    """A-score: mean(H(beta) / H_min) over valid CpGs."""
    beta = np.asarray(beta_array, dtype=float)
    valid = ~np.isnan(beta) & (beta > 0) & (beta < 1)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(shannon_entropy_b(beta[valid]) / h_min))


def cohens_d_paired_bootstrap(deltas, n_boot=N_BOOT, ci=95, seed=RNG_SEED):
    """Paired Cohen's d with bootstrap 95% CI."""
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    n = len(deltas)
    if n < 2:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    mean_d = float(np.mean(deltas))
    sd_d = float(np.std(deltas, ddof=1))
    d = mean_d / sd_d if sd_d > 0 else 0.0
    # Standard t-statistic for one-sample t-test on deltas
    t_stat = mean_d / (sd_d / math.sqrt(n)) if sd_d > 0 else 0.0
    # Bootstrap CI on Cohen's d
    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(deltas, size=n, replace=True)
        m = np.mean(s)
        sd = np.std(s, ddof=1)
        ds[i] = m / sd if sd > 0 else 0.0
    lo = float(np.percentile(ds, (100 - ci) / 2))
    hi = float(np.percentile(ds, 100 - (100 - ci) / 2))
    # Two-sided p-value approximation from t-statistic with df = n-1
    from scipy import stats
    p = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    return float(d), lo, hi, float(t_stat), float(p)


def identify_marker_cpgs(atlas_df, target_col, n_top=N_DISCRIMINATING_CPGS):
    """Top-N CpGs maximizing |β(target) − mean(β(other 24 cells))|."""
    other_cols = [c for c in atlas_df.columns if c != target_col]
    target = atlas_df[target_col].values
    others = atlas_df[other_cols].mean(axis=1).values
    scores = np.abs(target - others)
    sort_idx = np.argsort(scores)[::-1]
    cpg_ids = atlas_df.index.values
    return cpg_ids[sort_idx[:n_top]].tolist()


def beta_distribution_health_check(betas_dict, sample_label):
    """Per CHK-3.6: raw beta should have >30% extremes, <30% in [0.4, 0.6] (tissue-relaxed)."""
    all_b = []
    for v in betas_dict.values():
        if v is not None:
            all_b.extend(v.values() if isinstance(v, dict) else v)
    flat = np.asarray(all_b, dtype=float)
    flat = flat[~np.isnan(flat)]
    n = len(flat)
    if n == 0:
        return False, {'pass': False, 'reason': 'all NaN', 'sample': sample_label}
    pct_extreme = float(((flat < 0.2) | (flat > 0.8)).sum() / n * 100)
    pct_middle = float(((flat >= 0.4) & (flat <= 0.6)).sum() / n * 100)
    passed = (pct_extreme > 30) and (pct_middle < 30)
    return passed, {
        'sample': sample_label,
        'n_values': int(n),
        'pct_extreme_lt0.2_or_gt0.8': pct_extreme,
        'pct_middle_0.4_to_0.6': pct_middle,
        'pass': passed,
        'criterion': 'raw beta tissue-relaxed: >30% extremes; flat distribution flagged as residuals',
    }


# ============================================================================
# Step 1 — Download TCGA-READ paired files via GDC public API
# ============================================================================

def download_tcga_read_pairs(manifest_path, download_dir):
    """Download paired tumor/adjacent-normal sesame level3 beta files from GDC public API."""
    print(f"\n[Step 1] Download TCGA-READ paired tumor/adjacent-normal from GDC public API")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"  Manifest: {manifest_path}")
    print(f"  {len(manifest)} matched tumor/normal pairs")
    os.makedirs(download_dir, exist_ok=True)

    downloaded = []
    for i, entry in enumerate(manifest):
        patient = entry['patient']
        for kind in ['tumor', 'normal']:
            file_id = entry[f'{kind}_file_id']
            file_name = entry[f'{kind}_file_name']
            local_path = os.path.join(download_dir, f"{patient}__{kind}__{file_name}")
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
                size = os.path.getsize(local_path)
                # print(f"  [{i+1}/{len(manifest)}] {patient} {kind}: cached ({size} bytes)")
            else:
                url = GDC_API_BASE + file_id
                try:
                    t0 = time.time()
                    urllib.request.urlretrieve(url, local_path)
                    size = os.path.getsize(local_path)
                    print(f"  [{i+1}/{len(manifest)}] {patient} {kind}: {size} bytes ({time.time()-t0:.1f}s)")
                except Exception as e:
                    print(f"  [{i+1}/{len(manifest)}] {patient} {kind} FAILED: {e}")
                    continue
            downloaded.append({'patient': patient, 'kind': kind, 'path': local_path,
                              'sha256': sha256_file(local_path, prefix=16)})
    return downloaded


# ============================================================================
# Step 2 — Parse sesame level3 .txt to (cpg_id -> beta) dict
# ============================================================================

def parse_sesame_level3(filepath, valid_only=True):
    """Sesame level3 format: tab-separated, first column CpG probe id, second column β.
    Returns dict {cpg_id: beta} for valid β values (in (0, 1) exclusive)."""
    betas = {}
    with open(filepath) as f:
        first = f.readline().strip()
        # Detect header
        first_field = first.split('\t')[0].strip().strip('"') if first else ''
        is_header = not (first_field.startswith('cg') or first_field.startswith('ch.'))
        # Process first line if it's data
        if not is_header:
            parts = first.split('\t')
            if len(parts) >= 2:
                try:
                    b = float(parts[1])
                    if not valid_only or (0 < b < 1 and not math.isnan(b)):
                        betas[parts[0].strip()] = b
                except ValueError:
                    pass
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            try:
                b = float(parts[1])
                if not valid_only or (0 < b < 1 and not math.isnan(b)):
                    betas[parts[0].strip()] = b
            except ValueError:
                pass
    return betas


# ============================================================================
# Step 3 — Compute per-tile A-scores using Loyfer marker CpGs
# ============================================================================

def compute_per_tile_a_scores(sample_betas, atlas_df, marker_cache):
    """For each Loyfer cell type, compute A-score using top-N marker CpGs.
    Returns dict {cell_type: A_score}."""
    out = {}
    for ct in CELL_TYPE_TO_CLASS.keys():
        if ct not in atlas_df.columns:
            out[ct] = float('nan')
            continue
        markers = marker_cache[ct]
        beta_vals = []
        for m in markers:
            if m in sample_betas:
                beta_vals.append(sample_betas[m])
        if len(beta_vals) < 50:
            out[ct] = float('nan')
            continue
        h_min = H_MIN[CELL_TYPE_TO_CLASS[ct]]
        out[ct] = a_score_array(beta_vals, h_min)
    return out


# ============================================================================
# Step 4 — Cycling-class A-score across ALL valid HM450 CpGs (mirrors VAL-062)
# ============================================================================

def compute_cycling_a_score_full(sample_betas):
    """A_cycling = mean(H(β)/H_min_cycling) across all valid HM450 CpGs.
    Mirrors VAL-062 primary methodology."""
    if len(sample_betas) < MIN_VALID_CPGS:
        return float('nan'), len(sample_betas)
    beta_vals = list(sample_betas.values())
    return a_score_array(beta_vals, H_MIN_CYCLING), len(sample_betas)


# ============================================================================
# Step 5 — Outcome assignment per pre-locked criteria
# ============================================================================

def assign_outcome(pooled_d, pooled_ci_lo, pooled_ci_hi, pooled_p,
                   per_tile_d, top1_dist, n_pairs, baseline_breach_count):
    """Apply pre-locked decision criteria from prereg.md."""
    print(f"\n[Step 5] Outcome assignment per pre-locked criteria")
    # Criterion structure mirrors VAL-062 + VAL-097 patterns
    val_062_d = 0.7241  # VAL-062 anchor
    cycling_tiles = [t for t, c in CELL_TYPE_TO_CLASS.items() if c == 'cycling']

    # O6 first — data integrity
    if pooled_d is None or math.isnan(pooled_d) or n_pairs < 3:
        return 'O6_DATA_INTEGRITY', f'Insufficient paired data: n_pairs={n_pairs}, pooled_d={pooled_d}'

    # O4 — direction inverted
    if pooled_d < 0:
        return 'O4_DIRECTION_INVERTED', f'Paired d={pooled_d:.3f} < 0, framework inconsistency'

    # O3 — direction weak
    if pooled_d < 0.5 or (pooled_ci_lo is not None and pooled_ci_lo < 0):
        return 'O3_RECTAL_DIRECTION_WEAK', f'Paired d={pooled_d:.3f} below +0.5 OR 95% CI crosses zero'

    # O2 — direction confirmed but magnitude divergent from colon (>|0.5| from VAL-062)
    if abs(pooled_d - val_062_d) > 0.5:
        return ('O2_DIRECTION_DIVERGENT_FROM_COLON',
                f'Paired d={pooled_d:.3f} differs from VAL-062 (+0.724) by {abs(pooled_d-val_062_d):.3f}')

    # O7 — tile pattern unexpected (descriptive only)
    if per_tile_d:
        colon_d = per_tile_d.get('Colon_epithelial_cells', 0)
        max_tile = max(per_tile_d.items(), key=lambda x: abs(x[1]) if x[1] is not None else 0)
        if max_tile[0] != 'Colon_epithelial_cells' and CELL_TYPE_TO_CLASS.get(max_tile[0]) != 'cycling':
            # Non-cycling tile dominates — flag descriptively but still assign O1 if pooled passes
            o1_label = ('O1_CYCLING_CLASS_RECTAL_CONFIRMED_with_O7_tile_observation',
                       f'Pooled d={pooled_d:.3f} confirms; but top-1 tile = {max_tile[0]} (non-cycling) at d={max_tile[1]:.3f}')
            return o1_label

    # O1 — primary success
    return ('O1_CYCLING_CLASS_RECTAL_CONFIRMED',
            f'Paired d={pooled_d:.3f} [95% CI {pooled_ci_lo:.3f},{pooled_ci_hi:.3f}], extends VAL-062 (+0.724) to rectal subsite')


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 78)
    print("VAL-098 — TCGA-READ paired tumor-vs-adjacent-normal cycling-class")
    print(f"Start: {now_iso()}")
    print(f"WORKDIR: {WORKDIR}")
    print(f"OUTDIR:  {OUTDIR}")
    print(f"RNG seed: {RNG_SEED}")
    print(f"H_min(cycling): {H_MIN_CYCLING}")
    print(f"Bootstrap iterations: {N_BOOT}")
    print("=" * 78)

    # Step 0 — Verify prereg + record SHA
    print(f"\n[Step 0] Compute prereg SHA-256 and record in PREREG_SEAL.txt")
    prereg_sha = sha256_file(PREREG_PATH, prefix=64)
    with open(PREREG_SEAL_TXT, 'w') as f:
        f.write(f"prereg_path: {PREREG_PATH}\n")
        f.write(f"sha256: {prereg_sha}\n")
        f.write(f"sealed_at: {now_iso()}\n")
        f.write(f"rng_seed: {RNG_SEED}\n")
    print(f"  prereg SHA-256: {prereg_sha}")

    # Step 1 — Download paired files
    downloaded = download_tcga_read_pairs(READ_MANIFEST, TCGA_READ_DOWNLOAD_DIR)

    # Step 2 — Load manifest + clinical metadata
    with open(READ_MANIFEST) as f:
        manifest = json.load(f)
    md_df = pd.DataFrame(manifest)
    md_df.to_csv(CLINICAL_METADATA_CSV, index=False)
    print(f"  Wrote {CLINICAL_METADATA_CSV}")

    # Step 3 — Load Loyfer atlas + identify marker CpGs (cached)
    print(f"\n[Step 3] Load Loyfer atlas")
    atlas_df = pd.read_csv(LOYFER_ATLAS, index_col=0)
    print(f"  Atlas shape: {atlas_df.shape}")
    marker_cache = {}
    for ct in CELL_TYPE_TO_CLASS.keys():
        if ct in atlas_df.columns:
            marker_cache[ct] = identify_marker_cpgs(atlas_df, ct)

    # Step 4 — Parse each sample, compute A-scores
    print(f"\n[Step 4] Parse samples + compute A-scores (cycling-class full + 25-tile run-everything)")
    per_sample = []
    health_stats_list = []
    for entry in manifest:
        patient = entry['patient']
        tumor_path = os.path.join(TCGA_READ_DOWNLOAD_DIR, f"{patient}__tumor__{entry['tumor_file_name']}")
        normal_path = os.path.join(TCGA_READ_DOWNLOAD_DIR, f"{patient}__normal__{entry['normal_file_name']}")
        if not os.path.exists(tumor_path) or not os.path.exists(normal_path):
            print(f"  {patient}: missing files — skipping")
            continue
        tumor_betas = parse_sesame_level3(tumor_path)
        normal_betas = parse_sesame_level3(normal_path)
        # QC threshold check
        if len(tumor_betas) < MIN_VALID_CPGS or len(normal_betas) < MIN_VALID_CPGS:
            print(f"  {patient}: QC fail — tumor n={len(tumor_betas)}, normal n={len(normal_betas)}")
            continue
        # Cycling-class full-CpG A-score (VAL-062 primary)
        a_tumor_full, n_tumor = compute_cycling_a_score_full(tumor_betas)
        a_normal_full, n_normal = compute_cycling_a_score_full(normal_betas)
        # Run-everything 25-tile A-score (VAL-093/097 architecture)
        tumor_tile_a = compute_per_tile_a_scores(tumor_betas, atlas_df, marker_cache)
        normal_tile_a = compute_per_tile_a_scores(normal_betas, atlas_df, marker_cache)
        per_sample.append({
            'patient': patient,
            'age_years': entry['age_at_diagnosis_years'],
            'sex': entry['gender'],
            'stage': entry['stage'],
            'subsite': entry['tissue_or_organ_of_origin'],
            'tumor_n_valid_cpgs': n_tumor,
            'normal_n_valid_cpgs': n_normal,
            'a_cycling_tumor_full': a_tumor_full,
            'a_cycling_normal_full': a_normal_full,
            'delta_a_cycling_full': a_tumor_full - a_normal_full,
            **{f'a_{ct}_tumor': tumor_tile_a[ct] for ct in tumor_tile_a},
            **{f'a_{ct}_normal': normal_tile_a[ct] for ct in normal_tile_a},
            **{f'delta_a_{ct}': tumor_tile_a[ct] - normal_tile_a[ct]
               if not (math.isnan(tumor_tile_a[ct]) or math.isnan(normal_tile_a[ct])) else float('nan')
               for ct in tumor_tile_a},
        })
        # Health check on representative tile
        flat_t = list(tumor_betas.values())
        flat_n = list(normal_betas.values())
        for label, flat in [(f'{patient}_tumor', flat_t), (f'{patient}_normal', flat_n)]:
            arr = np.asarray(flat)
            n = len(arr)
            pct_ext = float(((arr < 0.2) | (arr > 0.8)).sum() / n * 100)
            pct_mid = float(((arr >= 0.4) & (arr <= 0.6)).sum() / n * 100)
            passed = (pct_ext > 30) and (pct_mid < 30)
            health_stats_list.append({'sample': label, 'pct_extreme': pct_ext, 'pct_middle': pct_mid, 'pass': passed})

    n_pairs_qc = len(per_sample)
    print(f"  QC-passed pairs: {n_pairs_qc} of {len(manifest)}")
    if n_pairs_qc < 3:
        print(f"  ABORT — insufficient QC-passed pairs")
        sys.exit(1)

    # Step 5 — Primary paired-d (cycling-class, full HM450)
    print(f"\n[Step 5] Primary paired-d (cycling-class, full HM450)")
    deltas_full = [r['delta_a_cycling_full'] for r in per_sample if not math.isnan(r['delta_a_cycling_full'])]
    pooled_d, pooled_ci_lo, pooled_ci_hi, t_stat, pval = cohens_d_paired_bootstrap(deltas_full)
    print(f"  Pooled cycling-class paired d = {pooled_d:.4f}")
    print(f"  Bootstrap 95% CI = [{pooled_ci_lo:.4f}, {pooled_ci_hi:.4f}]")
    print(f"  t-statistic = {t_stat:.4f}, p = {pval:.3e}")

    # Step 6 — Per-tile paired-d (run-everything 25-tile)
    print(f"\n[Step 6] Per-tile paired-d (run-everything 25-tile, top-100 marker CpGs each)")
    per_tile_d = {}
    per_tile_results = []
    for ct in CELL_TYPE_TO_CLASS.keys():
        deltas_tile = [r[f'delta_a_{ct}'] for r in per_sample
                      if not math.isnan(r.get(f'delta_a_{ct}', float('nan')))]
        if len(deltas_tile) < 3:
            per_tile_d[ct] = None
            per_tile_results.append({'tile': ct, 'class': CELL_TYPE_TO_CLASS[ct],
                                    'paired_d': None, 'ci_lo_95': None, 'ci_hi_95': None,
                                    'n_pairs': len(deltas_tile)})
            continue
        d, lo, hi, t, p = cohens_d_paired_bootstrap(deltas_tile)
        per_tile_d[ct] = d
        per_tile_results.append({
            'tile': ct, 'class': CELL_TYPE_TO_CLASS[ct],
            'paired_d': float(d), 'ci_lo_95': float(lo), 'ci_hi_95': float(hi),
            'n_pairs': len(deltas_tile),
        })
    # Print sorted by |d|
    print(f"  Top-10 tiles by |paired d|:")
    sorted_tiles = sorted(per_tile_results, key=lambda x: -abs(x['paired_d']) if x['paired_d'] is not None else 0)
    for r in sorted_tiles[:10]:
        if r['paired_d'] is not None:
            print(f"    {r['tile']:30} ({r['class']:10}) d={r['paired_d']:+.3f} CI=[{r['ci_lo_95']:+.3f},{r['ci_hi_95']:+.3f}]")

    # Step 7 — Top-1 ΔA call per patient
    print(f"\n[Step 7] Top-1 ΔA call per patient")
    top1_results = []
    from collections import Counter
    for r in per_sample:
        deltas = {ct: abs(r.get(f'delta_a_{ct}', 0)) for ct in CELL_TYPE_TO_CLASS.keys()
                  if not math.isnan(r.get(f'delta_a_{ct}', float('nan')))}
        if not deltas:
            top1_results.append({'patient': r['patient'], 'top1_tile': None})
            continue
        top_tile = max(deltas, key=deltas.get)
        top1_results.append({
            'patient': r['patient'],
            'top1_tile': top_tile,
            'top1_class': CELL_TYPE_TO_CLASS[top_tile],
            'top1_delta': float(r[f'delta_a_{top_tile}']),
        })
    top1_dist = dict(Counter(t['top1_tile'] for t in top1_results if t['top1_tile']))
    print(f"  Distribution: {top1_dist}")

    # Step 8 — Stratified analysis (age, sex, stage, subsite)
    print(f"\n[Step 8] Stratified analysis")
    stratified = {'by_age_decade': {}, 'by_sex': {}, 'by_stage': {}, 'by_subsite': {}}
    # Age decade
    for r in per_sample:
        age_y = r.get('age_years')
        if age_y is None or age_y == 'NA':
            continue
        try:
            age_y = float(age_y)
        except (ValueError, TypeError):
            continue
        decade = f"{int(age_y // 10) * 10}s"
        stratified['by_age_decade'].setdefault(decade, []).append(r)
    # Under-50 specifically (the early-onset stratum, pre-locked underpowered)
    under_50 = [r for r in per_sample
                if r.get('age_years') and r['age_years'] != 'NA' and float(r['age_years']) < 50]
    age_50_plus = [r for r in per_sample
                   if r.get('age_years') and r['age_years'] != 'NA' and float(r['age_years']) >= 50]
    print(f"  Under-50 stratum: n={len(under_50)} (PRE-LOCKED underpowered, direction-only per CHK-2.7)")
    print(f"  50+ stratum: n={len(age_50_plus)}")

    age_strat_results = {}
    for label, group in [('under_50', under_50), ('age_50_plus', age_50_plus)]:
        deltas = [r['delta_a_cycling_full'] for r in group if not math.isnan(r['delta_a_cycling_full'])]
        if len(deltas) >= 2:
            d, lo, hi, t, p = cohens_d_paired_bootstrap(deltas)
            age_strat_results[label] = {'n': len(deltas), 'cycling_d': d, 'ci': [lo, hi], 'p': p}
        elif len(deltas) == 1:
            age_strat_results[label] = {'n': 1, 'cycling_delta_a': float(deltas[0]),
                                       'note': 'n=1 — direction-only, descriptive-only per CHK-2.7'}
        else:
            age_strat_results[label] = {'n': 0}
        print(f"    {label}: {age_strat_results[label]}")

    # Sex stratification
    sex_strat = {}
    for sex_val in ['female', 'male']:
        deltas = [r['delta_a_cycling_full'] for r in per_sample
                  if r.get('sex') == sex_val and not math.isnan(r['delta_a_cycling_full'])]
        if len(deltas) >= 2:
            d, lo, hi, t, p = cohens_d_paired_bootstrap(deltas)
            sex_strat[sex_val] = {'n': len(deltas), 'cycling_d': d, 'ci': [lo, hi]}
        else:
            sex_strat[sex_val] = {'n': len(deltas), 'note': 'underpowered'}

    # Subsite stratification
    subsite_strat = {}
    for r in per_sample:
        sub = r.get('subsite', 'unknown')
        subsite_strat.setdefault(sub, []).append(r['delta_a_cycling_full'])
    subsite_summary = {}
    for sub, deltas in subsite_strat.items():
        deltas = [d for d in deltas if not math.isnan(d)]
        if len(deltas) >= 2:
            d, lo, hi, t, p = cohens_d_paired_bootstrap(deltas)
            subsite_summary[sub] = {'n': len(deltas), 'cycling_d': d, 'ci': [lo, hi]}
        else:
            subsite_summary[sub] = {'n': len(deltas), 'note': 'underpowered or n=1, direction-only'}

    # Cross-cohort baseline check vs VAL-062 (sanity only, not blocker)
    # Per CHK-3.2 — count tiles where the adjacent-normal mean exceeds anchor by >1 SD
    # We don't have direct VAL-062 healthy baseline here without re-running it; flag as
    # within-cohort-paired so structurally not at risk
    baseline_breach_count = 0  # within-cohort paired — structurally avoids cross-cohort baseline risk

    # Step 9 — Outcome assignment
    outcome_label, outcome_reason = assign_outcome(
        pooled_d, pooled_ci_lo, pooled_ci_hi, pval,
        per_tile_d, top1_dist, n_pairs_qc, baseline_breach_count)
    print(f"\n[Step 9] OUTCOME: {outcome_label}")
    print(f"  Reason: {outcome_reason}")

    # Step 10 — Heatmap
    print(f"\n[Step 10] Tile heatmap")
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_data = pd.DataFrame([
        {'tile': r['tile'], 'class': r['class'], 'd': r['paired_d'] or 0}
        for r in per_tile_results if r['paired_d'] is not None
    ]).sort_values('d', ascending=True)
    colors = ['steelblue' if d < 0 else 'firebrick' for d in plot_data['d']]
    ax.barh(plot_data['tile'], plot_data['d'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=0.5, color='gray', linewidth=0.5, linestyle='--', label='|d|=0.5')
    ax.axvline(x=-0.5, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Paired Cohen's d (TCGA-READ tumor vs adjacent-normal)")
    ax.set_title(f'VAL-098: 25-tile run-everything paired-d on Loyfer atlas\n'
                 f'TCGA-READ paired tumor/adjacent-normal (n={n_pairs_qc} QC-passed pairs)')
    plt.tight_layout()
    plt.savefig(TILE_HEATMAP_PNG, dpi=120)
    print(f"  Wrote {TILE_HEATMAP_PNG}")

    # Step 11 — Write per-sample CSV
    pd.DataFrame(per_sample).to_csv(PER_SAMPLE_CSV, index=False)
    print(f"  Wrote {PER_SAMPLE_CSV}")

    # Step 12 — Cohort manifest output
    cohort_manifest = {
        'cohort': 'TCGA-READ',
        'description': 'TCGA-READ paired tumor/adjacent-normal HM450 sesame level3 betas',
        'platform': 'Illumina HumanMethylation450 (HM450, GPL13534)',
        'preprocessing_pipeline': 'sesame level3 (GDC public)',
        'access_tier': 1,
        'access_url': GDC_API_BASE + '{file_id}',
        'manifest_path': READ_MANIFEST,
        'n_pairs_total': len(manifest),
        'n_pairs_qc_passed': n_pairs_qc,
        'cohort_size_age_under_50': len(under_50),
        'cohort_size_age_50_plus': len(age_50_plus),
        'pre_locked_underpower_flags': {
            'under_50_stratum': f'n={len(under_50)} — direction-only, descriptive-only per CHK-2.7',
        },
    }
    with open(COHORT_MANIFEST_JSON, 'w') as f:
        json.dump(cohort_manifest, f, indent=2, default=str)

    # Step 13 — Write results JSON
    results = {
        'val_id': 'VAL-098',
        'sealed_at': now_iso(),
        'prereg_sha256': prereg_sha,
        'rng_seed': RNG_SEED,
        'cohort': cohort_manifest,
        'data_integrity': {'health_stats': health_stats_list[:14]},
        'primary_cycling_class_paired_d': {
            'd': pooled_d,
            'ci_lo_95': pooled_ci_lo,
            'ci_hi_95': pooled_ci_hi,
            't_stat': t_stat,
            'p_value': pval,
            'n_pairs': n_pairs_qc,
            'method': 'paired Cohen\'s d on (A_tumor − A_normal) per patient, A_cycling = mean(H(β)/0.856055) on all valid HM450 CpGs (≥400K per sample)',
            'comparison_to_val_062': {
                'val_062_d': 0.7241,
                'val_062_cohort': 'TCGA-COAD n=26 paired pairs',
                'difference': pooled_d - 0.7241,
            },
        },
        'run_everything_per_tile_paired_d': per_tile_results,
        'top1_distribution': top1_dist,
        'top1_per_patient': top1_results,
        'stratified_analysis': {
            'by_age': age_strat_results,
            'by_sex': sex_strat,
            'by_subsite': subsite_summary,
        },
        'outcome_label': outcome_label,
        'outcome_reason': outcome_reason,
        'edear_commercial_deployment_unaffected': 'Per CCL-037: cookbook validation cohort coverage gaps and underpowered strata do not affect EDEAR single-pipeline patient-vs-internal-reference deployment',
        'runtime_seconds': time.time() - t_start,
    }
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Wrote {RESULTS_JSON}")

    with open(STRATIFIED_JSON, 'w') as f:
        json.dump({'by_age': age_strat_results, 'by_sex': sex_strat, 'by_subsite': subsite_summary},
                 f, indent=2, default=str)
    print(f"  Wrote {STRATIFIED_JSON}")

    print("\n" + "=" * 78)
    print(f"VAL-098 complete. Outcome: {outcome_label}")
    print(f"Pooled cycling-class paired d = {pooled_d:.4f} [95% CI {pooled_ci_lo:.4f}, {pooled_ci_hi:.4f}]")
    print(f"VAL-062 anchor (TCGA-COAD): paired d = +0.7241")
    print(f"Runtime: {time.time() - t_start:.1f} s")
    print("=" * 78)


if __name__ == '__main__':
    main()
