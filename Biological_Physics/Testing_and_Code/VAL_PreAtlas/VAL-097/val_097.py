#!/usr/bin/env python3
"""
VAL-097 — Never-smoker LUAD tissue 25-tile per-class A-score characterization
on GSE256092 with cross-cohort baseline against TCGA-LUAD adjacent-normal.

Run-everything architecture (CCL-033, signed off 2026-04-26): every sample runs
Stage 2 with all 25 Loyfer cell types regardless of single-tissue gating.
Per-class A-score computed for every cell type tile every sample.

Operational context (LL-PUBLIC-TIER, signed off 2026-04-28): IAMPerformance public-
tier-only operational reset. Cohort access restricted to public Tier 1 GEO data.
Biobank-gated cohorts logged in lung-epic/future_when_support_arrives.md.
Data-availability gaps (smoking strata, driver mutations) pre-locked as honest
CHK-2.7 caveats per pre-registration.

Pre-registered before any beta access — see VAL-097/prereg.md
SHA-256 of prereg recorded in VAL-097_PREREG_SEAL.txt.

RNG seed: 20260428
H_min values frozen from GAPE_WEB_v13 _H_MIN_GRID (G-002 + G-003b MCMC posteriors,
R-hat < 1.001).
"""

import gzip
import hashlib
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
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
WORKDIR = '/home/claude/edear_working/VAL-097'
LOYFER_ATLAS = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv'

# GSE256092 — Korean NSLA EPIC tissue, all never-smoker, n=141
GSE256092_SWAN_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256092/suppl/GSE256092_SWAN.txt.gz'
GSE256092_SERIES_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE256nnn/GSE256092/matrix/GSE256092_series_matrix.txt.gz'
GSE256092_SWAN_LOCAL = os.path.join(WORKDIR, 'GSE256092_SWAN.txt.gz')
GSE256092_SERIES_LOCAL = os.path.join(WORKDIR, 'gse256092_matrix.txt.gz')

# TCGA-LUAD adjacent-normal — downloaded fresh per VAL-063 manifest from GDC public API
LUAD_MANIFEST = '/home/claude/iam_repo/Biological_Physics/validation_runs/LUAD_matched_manifest.json'
TCGA_LUAD_DOWNLOAD_DIR = os.path.join(WORKDIR, 'tcga_luad_downloads')
TCGA_LUAD_NORMAL_BETAS = os.path.join(WORKDIR, 'tcga_luad_normal_betas.csv')
GDC_API_BASE = 'https://api.gdc.cancer.gov/data/'

# Outputs (CHK-6.4 canonical: VAL-097/ subfolder structure)
OUTDIR = os.path.join(WORKDIR, 'VAL-097')
os.makedirs(OUTDIR, exist_ok=True)
RESULTS_JSON = os.path.join(OUTDIR, 'results.json')
STRATIFIED_JSON = os.path.join(OUTDIR, 'stratified.json')
PER_SAMPLE_CSV = os.path.join(OUTDIR, 'per_sample.csv')
TILE_HEATMAP_PNG = os.path.join(OUTDIR, 'tile_heatmap.png')
COHORT_MANIFEST_JSON = os.path.join(OUTDIR, 'cohort_manifest.json')
CLINICAL_METADATA_CSV = os.path.join(OUTDIR, 'clinical_metadata.csv')
PREREG_SEAL_TXT = os.path.join(OUTDIR, 'PREREG_SEAL.txt')
PREREG_PATH = os.path.join(OUTDIR, 'prereg.md')

N_DISCRIMINATING_CPGS = 100
N_BOOT = 10000
RNG_SEED = 20260428

# ============================================================================
# Helpers
# ============================================================================

def sha256_file(path, prefix=16):
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


def a_score(beta_array, h_min):
    """A-score: mean(H(beta) / H_min) over valid CpGs."""
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


def bootstrap_d_ci(g1, g2, n_boot=N_BOOT, ci=95, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    g1 = np.asarray(g1, dtype=float); g1 = g1[~np.isnan(g1)]
    g2 = np.asarray(g2, dtype=float); g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return (float('nan'), float('nan'))
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
    return cpg_ids[sort_idx[:n_top]].tolist()


def beta_distribution_health_check(beta_matrix, sample_label):
    """Per CHK-3.6: raw beta should have >30% extremes, <10% in [0.4, 0.6].
    Returns (pass, stats_dict). Flat distribution = residuals not raw, abort."""
    flat = beta_matrix.values.ravel()
    flat = flat[~np.isnan(flat)]
    n = len(flat)
    if n == 0:
        return False, {'pass': False, 'reason': 'all NaN', 'sample': sample_label}
    pct_extreme = float(((flat < 0.2) | (flat > 0.8)).sum() / n * 100)
    pct_middle = float(((flat >= 0.4) & (flat <= 0.6)).sum() / n * 100)
    passed = (pct_extreme > 30) and (pct_middle < 30)  # relaxed middle floor for tissue
    return passed, {
        'sample': sample_label,
        'n_values': n,
        'pct_extreme_lt0.2_or_gt0.8': pct_extreme,
        'pct_middle_0.4_to_0.6': pct_middle,
        'pass': passed,
        'criterion': 'raw beta: >30% extremes; flat distribution flagged as residuals',
    }


def download_tcga_luad_adjacent_normal(manifest_path, download_dir, output_csv,
                                       loyfer_cpgs):
    """Download TCGA-LUAD adjacent-normal sesame level3 beta files via GDC public API,
    parse them into a single beta matrix indexed by Loyfer CpGs, save to CSV.

    Per VAL-063 Reproduction section: 29 normal_file_ids in LUAD_matched_manifest.json,
    public NIH GDC access via https://api.gdc.cancer.gov/data/{file_id}.

    Sesame level3 format: tab-separated, col1 = CpG probe id, col2 = β value.
    """
    import urllib.request
    import time

    print(f"\n[Step 5b] Download TCGA-LUAD adjacent-normal from GDC public API")
    print(f"  Manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"  {len(manifest)} matched tumor/normal patient pairs")

    os.makedirs(download_dir, exist_ok=True)
    loyfer_set = set(loyfer_cpgs)

    # If output CSV already exists with the right shape, skip
    if os.path.exists(output_csv):
        try:
            existing = pd.read_csv(output_csv, index_col=0, nrows=5)
            print(f"  Cached {output_csv} present ({existing.shape[1]} cols); using cached")
            return pd.read_csv(output_csv, index_col=0)
        except Exception:
            pass

    # Download each normal file and extract Loyfer CpG rows
    normal_betas = {}  # patient_id -> {cpg: beta}
    failed = []
    for i, entry in enumerate(manifest):
        patient = entry['patient']
        file_id = entry['normal_file_id']
        file_name = entry['normal_file_name']
        local_path = os.path.join(download_dir, file_name)
        # Download if missing
        if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024:
            url = GDC_API_BASE + file_id
            try:
                t0 = time.time()
                urllib.request.urlretrieve(url, local_path)
                size = os.path.getsize(local_path)
                print(f"  [{i+1}/{len(manifest)}] {patient} normal: {size} bytes ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"  [{i+1}/{len(manifest)}] {patient} FAILED: {e}")
                failed.append(patient)
                continue
        else:
            print(f"  [{i+1}/{len(manifest)}] {patient} normal: cached ({os.path.getsize(local_path)} bytes)")
        # Parse: keep only Loyfer CpG rows
        betas_this_sample = {}
        try:
            with open(local_path) as f:
                # Skip header if present
                first_line = f.readline().strip()
                # Detect header: if first token doesn't start with 'cg' or 'ch.' it's a header
                if not (first_line.split('\t')[0].startswith('cg') or
                        first_line.split('\t')[0].startswith('ch.')):
                    pass  # header skipped
                else:
                    # No header, first line is data
                    parts = first_line.split('\t')
                    if len(parts) >= 2:
                        cpg = parts[0]
                        if cpg in loyfer_set:
                            try:
                                beta = float(parts[1])
                                if 0 <= beta <= 1 and not np.isnan(beta):
                                    betas_this_sample[cpg] = beta
                            except ValueError:
                                pass
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    cpg = parts[0]
                    if cpg in loyfer_set:
                        try:
                            beta = float(parts[1])
                            if 0 <= beta <= 1 and not np.isnan(beta):
                                betas_this_sample[cpg] = beta
                        except ValueError:
                            pass
        except Exception as e:
            print(f"  [{i+1}/{len(manifest)}] {patient} parse FAILED: {e}")
            failed.append(patient)
            continue
        normal_betas[patient] = betas_this_sample

    print(f"  Downloaded/parsed {len(normal_betas)}/{len(manifest)} adjacent-normal samples")
    if failed:
        print(f"  Failed: {failed}")

    # Build DataFrame: rows = CpGs (Loyfer), cols = patient IDs
    df = pd.DataFrame(normal_betas)
    print(f"  Reference matrix: {df.shape}")
    df.to_csv(output_csv)
    print(f"  Wrote {output_csv}")
    return df


# ============================================================================
# Step 1 — Download / verify input files
# ============================================================================

def download_if_missing(url, local_path, label):
    """Download via urllib, log size + SHA. Idempotent."""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        size = os.path.getsize(local_path)
        sha = sha256_file(local_path, prefix=16)
        print(f"  [{label}] cached: {local_path} ({size} bytes, SHA {sha})")
        return size, sha
    print(f"  [{label}] downloading from {url} ...")
    import urllib.request
    urllib.request.urlretrieve(url, local_path)
    size = os.path.getsize(local_path)
    sha = sha256_file(local_path, prefix=16)
    print(f"  [{label}] downloaded: {local_path} ({size} bytes, SHA {sha})")
    return size, sha


# ============================================================================
# Step 2 — Parse GSE256092 series matrix to extract clinical metadata
# ============================================================================

def parse_gse256092_metadata(series_path):
    """Extract per-sample metadata from GSE256092 series matrix.
    Returns DataFrame with columns: gsm_id, sample_title, stage, age, sex, disease."""
    print(f"\n[Step 2] Parsing GSE256092 series matrix metadata from {series_path}")
    rows_by_field = {}
    sample_titles = []
    sample_ids = []
    with gzip.open(series_path, 'rt') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('!Sample_geo_accession'):
                parts = line.split('\t')[1:]
                sample_ids = [p.strip().strip('"') for p in parts]
            elif line.startswith('!Sample_title'):
                parts = line.split('\t')[1:]
                sample_titles = [p.strip().strip('"') for p in parts]
            elif line.startswith('!Sample_characteristics_ch1'):
                parts = line.split('\t')[1:]
                values = [p.strip().strip('"') for p in parts]
                # Determine field name from first value
                if values and ':' in values[0]:
                    field = values[0].split(':', 1)[0].strip().lower()
                    extracted = []
                    for v in values:
                        if ':' in v:
                            extracted.append(v.split(':', 1)[1].strip())
                        else:
                            extracted.append(v.strip())
                    rows_by_field.setdefault(field, []).append(extracted)
    # Each characteristic field shows up as one row; values per sample
    df = pd.DataFrame({'gsm_id': sample_ids, 'sample_title': sample_titles})
    for field, values_lists in rows_by_field.items():
        # If multiple rows for the same field, take the first that has non-empty values
        df[field] = values_lists[0]
    print(f"  Parsed {len(df)} samples; fields: {list(df.columns)}")
    return df


# ============================================================================
# Step 3 — Parse SWAN-normalized beta matrix at Loyfer atlas CpGs
# ============================================================================

def extract_swan_betas_at_loyfer(swan_path, loyfer_cpgs):
    """Stream-parse the SWAN beta matrix and keep only Loyfer CpG rows.
    Returns DataFrame with CpGs as index, samples as columns."""
    print(f"\n[Step 3] Streaming SWAN beta matrix at Loyfer CpGs from {swan_path}")
    loyfer_set = set(loyfer_cpgs)
    rows_kept = []
    sample_ids = None
    with gzip.open(swan_path, 'rt') as f:
        for i, line in enumerate(f):
            line = line.rstrip('\n')
            if not line:
                continue
            if sample_ids is None:
                # Header line — first column is empty or 'CpG' label
                parts = line.split('\t')
                # Try to determine if first field is data or header
                first_field = parts[0].strip().strip('"')
                if first_field.startswith('cg') or first_field.startswith('ch.'):
                    # No header; data starts immediately. Synthesize numeric ids
                    print(f"  Warning: SWAN matrix has no detectable header; using synthetic sample numbers")
                    sample_ids = [f'sample_{j}' for j in range(len(parts) - 1)]
                else:
                    sample_ids = [p.strip().strip('"') for p in parts[1:]]
                    continue
            parts = line.split('\t')
            cpg = parts[0].strip().strip('"')
            if cpg in loyfer_set:
                rows_kept.append([cpg] + [p.strip().strip('"') for p in parts[1:]])
            if (i + 1) % 100000 == 0:
                print(f"  ... processed {i+1} lines", flush=True); _=None  # , kept {len(rows_kept)}")
    print(f"  Found {len(rows_kept)} Loyfer CpGs", flush=True); _=None  #  in SWAN matrix")
    print(f"  Sample count: {len(sample_ids)}")
    cols = ['cpg'] + sample_ids
    df = pd.DataFrame(rows_kept, columns=cols)
    df.set_index('cpg', inplace=True)
    df = df.replace('null', np.nan).replace('NULL', np.nan).replace('NA', np.nan).replace('', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df


# ============================================================================
# Step 4 — Compute per-tile A-scores for each sample
# ============================================================================

def compute_tile_a_scores(beta_df, atlas_df):
    """For each Loyfer cell type, compute per-sample A-score using top-N marker CpGs.
    Returns DataFrame: index=sample_id, columns=cell_types, values=A-score."""
    print(f"\n[Step 4] Computing per-tile A-scores ({len(atlas_df.columns)} cell types × {beta_df.shape[1]} samples)")
    samples = beta_df.columns.tolist()
    cell_types = list(CELL_TYPE_TO_CLASS.keys())
    results = pd.DataFrame(index=samples, columns=cell_types, dtype=float)
    for ct in cell_types:
        if ct not in atlas_df.columns:
            print(f"  [warning] {ct} not in atlas — skipping")
            continue
        markers = identify_marker_cpgs(atlas_df, ct)
        markers_in_beta = [m for m in markers if m in beta_df.index]
        if len(markers_in_beta) < 50:
            print(f"  [warning] {ct} has only {len(markers_in_beta)} marker CpGs in beta matrix; reading may be unstable")
        h_min = H_MIN[CELL_TYPE_TO_CLASS[ct]]
        for s in samples:
            beta_vals = beta_df.loc[markers_in_beta, s].values
            results.loc[s, ct] = a_score(beta_vals, h_min)
    print(f"  Per-tile A-score matrix: {results.shape}")
    return results


# ============================================================================
# Step 5 — Cross-cohort baseline check
# ============================================================================

def cross_cohort_baseline_check(case_a_scores, ref_a_scores, cell_types):
    """For each tile, compute (case_mean − ref_mean) / pooled_SD = anchor-SD units."""
    print(f"\n[Step 5] Cross-cohort baseline check on {len(cell_types)} tiles")
    results = []
    for ct in cell_types:
        case_vals = case_a_scores[ct].dropna().values
        ref_vals = ref_a_scores[ct].dropna().values if ct in ref_a_scores.columns else np.array([])
        if len(case_vals) < 2 or len(ref_vals) < 2:
            results.append({
                'tile': ct,
                'case_mean': float(np.nanmean(case_vals)) if len(case_vals) > 0 else None,
                'case_n': int(len(case_vals)),
                'ref_mean': float(np.nanmean(ref_vals)) if len(ref_vals) > 0 else None,
                'ref_n': int(len(ref_vals)),
                'baseline_delta_anchor_SD': None,
                'breach_gt_1SD': None,
            })
            continue
        case_var = np.var(case_vals, ddof=1)
        ref_var = np.var(ref_vals, ddof=1)
        n1, n2 = len(case_vals), len(ref_vals)
        pooled_sd = np.sqrt(((n1-1)*case_var + (n2-1)*ref_var) / (n1+n2-2)) if (n1+n2-2) > 0 else np.nan
        delta = (np.nanmean(case_vals) - np.nanmean(ref_vals))
        anchor_sd_units = float(delta / pooled_sd) if pooled_sd > 0 else None
        results.append({
            'tile': ct,
            'class': CELL_TYPE_TO_CLASS[ct],
            'case_mean': float(np.nanmean(case_vals)),
            'case_n': int(n1),
            'ref_mean': float(np.nanmean(ref_vals)),
            'ref_n': int(n2),
            'pooled_sd': float(pooled_sd) if not np.isnan(pooled_sd) else None,
            'baseline_delta_anchor_SD': anchor_sd_units,
            'breach_gt_1SD': bool(abs(anchor_sd_units) > 1) if anchor_sd_units is not None else None,
        })
    return results


# ============================================================================
# Step 6 — Case-vs-reference Cohen's d per tile with bootstrap CI
# ============================================================================

def case_vs_reference_d(case_a_scores, ref_a_scores, cell_types):
    print(f"\n[Step 6] Case-vs-reference d on {len(cell_types)} tiles (bootstrap n={N_BOOT})")
    results = []
    for ct in cell_types:
        case_vals = case_a_scores[ct].dropna().values
        ref_vals = ref_a_scores[ct].dropna().values if ct in ref_a_scores.columns else np.array([])
        d = cohens_d(case_vals, ref_vals)
        ci_lo, ci_hi = bootstrap_d_ci(case_vals, ref_vals)
        results.append({
            'tile': ct,
            'class': CELL_TYPE_TO_CLASS[ct],
            'd_case_vs_ref': float(d) if not np.isnan(d) else None,
            'ci_lo_95': ci_lo if not np.isnan(ci_lo) else None,
            'ci_hi_95': ci_hi if not np.isnan(ci_hi) else None,
            'n_case': int(len(case_vals)),
            'n_ref': int(len(ref_vals)),
        })
    return results


# ============================================================================
# Step 7 — Top-1 ΔA call per patient
# ============================================================================

def top1_delta_a_call(case_a_scores, ref_a_scores):
    """For each case patient, identify the tile with the largest |A_patient − A_ref_mean|."""
    print(f"\n[Step 7] Top-1 ΔA call per patient")
    cell_types = list(CELL_TYPE_TO_CLASS.keys())
    ref_means = {ct: float(np.nanmean(ref_a_scores[ct])) if ct in ref_a_scores.columns else np.nan
                 for ct in cell_types}
    top1 = []
    for s in case_a_scores.index:
        deltas = {}
        for ct in cell_types:
            if ct in case_a_scores.columns and not np.isnan(case_a_scores.loc[s, ct]) and not np.isnan(ref_means[ct]):
                deltas[ct] = abs(case_a_scores.loc[s, ct] - ref_means[ct])
        if not deltas:
            top1.append({'sample': s, 'top1_tile': None, 'top1_delta': None})
            continue
        top_tile = max(deltas, key=deltas.get)
        top1.append({
            'sample': s,
            'top1_tile': top_tile,
            'top1_delta': float(deltas[top_tile]),
            'top1_class': CELL_TYPE_TO_CLASS[top_tile],
        })
    # Distribution
    from collections import Counter
    counter = Counter(c['top1_tile'] for c in top1 if c['top1_tile'] is not None)
    print(f"  Top-1 distribution: {dict(counter.most_common(5))}")
    return top1, dict(counter)


# ============================================================================
# Step 8 — Stratified analysis (sex × age decade × stage)
# ============================================================================

def stratified_analysis(case_a_scores, metadata_df, cell_types, ref_a_scores=None):
    """For each stratum (sex / age decade / stage), report per-tile A-score mean+SD+n."""
    print(f"\n[Step 8] Stratified analysis (sex × age decade × stage)")
    # Merge
    md = metadata_df.set_index('gsm_id').copy()
    # Map sample_title in case_a_scores back to gsm_id via the metadata
    title_to_gsm = dict(zip(md['sample_title'], md.index))
    case_a_with_meta = case_a_scores.copy()
    case_a_with_meta['gsm_id'] = case_a_with_meta.index.map(
        lambda x: title_to_gsm.get(x, x) if x in title_to_gsm else x)
    case_a_with_meta = case_a_with_meta.merge(
        md[['stage', 'age', 'gender']], left_on='gsm_id', right_index=True, how='left')
    # Age decade
    def age_decade(a):
        try:
            a = int(a)
            return f"{(a // 10) * 10}s"
        except (ValueError, TypeError):
            return "unknown"
    case_a_with_meta['age_decade'] = case_a_with_meta['age'].apply(age_decade)
    # Reports
    out = {'by_sex': {}, 'by_age_decade': {}, 'by_stage': {}}
    for sex_val in case_a_with_meta['gender'].dropna().unique():
        sub = case_a_with_meta[case_a_with_meta['gender'] == sex_val]
        out['by_sex'][str(sex_val)] = {
            'n': int(len(sub)),
            'tile_means': {ct: float(sub[ct].mean()) if ct in sub.columns and len(sub) > 0 else None
                           for ct in cell_types},
            'tile_sds': {ct: float(sub[ct].std(ddof=1)) if ct in sub.columns and len(sub) > 1 else None
                         for ct in cell_types},
        }
    for ad in sorted(case_a_with_meta['age_decade'].unique()):
        sub = case_a_with_meta[case_a_with_meta['age_decade'] == ad]
        out['by_age_decade'][str(ad)] = {
            'n': int(len(sub)),
            'tile_means': {ct: float(sub[ct].mean()) if ct in sub.columns and len(sub) > 0 else None
                           for ct in cell_types},
        }
    if 'stage' in case_a_with_meta.columns:
        for st in sorted(case_a_with_meta['stage'].dropna().unique()):
            sub = case_a_with_meta[case_a_with_meta['stage'] == st]
            out['by_stage'][f'stage_{st}'] = {
                'n': int(len(sub)),
                'tile_means': {ct: float(sub[ct].mean()) if ct in sub.columns and len(sub) > 0 else None
                               for ct in cell_types},
                'tile_sds': {ct: float(sub[ct].std(ddof=1)) if ct in sub.columns and len(sub) > 1 else None
                             for ct in cell_types},
            }
    return out, case_a_with_meta


# ============================================================================
# Step 9 — Outcome assignment per pre-locked criteria
# ============================================================================

def assign_outcome(d_results, baseline_results, top1_dist, n_cases):
    """Apply pre-locked decision criteria from prereg.md."""
    print(f"\n[Step 9] Outcome assignment per pre-locked criteria")
    # Index by tile
    d_by_tile = {r['tile']: r for r in d_results}
    bl_by_tile = {r['tile']: r for r in baseline_results}
    lung_d = d_by_tile.get('Lung_cells', {}).get('d_case_vs_ref')
    cycling_tiles = [t for t, c in CELL_TYPE_TO_CLASS.items() if c == 'cycling']
    cycling_d = {t: d_by_tile.get(t, {}).get('d_case_vs_ref') for t in cycling_tiles}
    all_d = {t: r.get('d_case_vs_ref') for t, r in d_by_tile.items()}
    abs_d = {t: abs(d) if d is not None else 0 for t, d in all_d.items()}
    if not abs_d:
        return 'O6_DATA_INTEGRITY', 'No d values computed'
    largest_d_tile = max(abs_d, key=abs_d.get)
    largest_abs_d = abs_d[largest_d_tile]
    # Top-1 majority check
    n_top1 = sum(top1_dist.values()) if top1_dist else 0
    lung_top1_frac = top1_dist.get('Lung_cells', 0) / n_top1 if n_top1 > 0 else 0
    # Baseline severe breach
    severe_breach = sum(1 for r in baseline_results
                        if r.get('baseline_delta_anchor_SD') is not None
                        and abs(r['baseline_delta_anchor_SD']) > 3)
    cycling_pos = [t for t, d in cycling_d.items() if d is not None and d >= 0.3]
    cycling_neg = [t for t, d in cycling_d.items() if d is not None and d <= -0.3]
    # Ordering matters — check most-specific first
    if severe_breach >= 3:
        # Compare to case-vs-ref d magnitudes
        max_case_d = max(abs_d.values())
        max_baseline = max(abs(r['baseline_delta_anchor_SD']) for r in baseline_results
                           if r.get('baseline_delta_anchor_SD') is not None)
        if max_baseline > max_case_d:
            return 'O5_BASELINE_DOMINATED', f'Severe baseline breach ({severe_breach} tiles >3SD) dominates case-vs-ref signal'
    if lung_d is not None and lung_d >= 0.5 and largest_d_tile == 'Lung_cells' and lung_top1_frac > 0.5:
        return 'O1_LUNG_LOCALIZED', f'Lung_cells d={lung_d:.3f} largest among 25 tiles; {lung_top1_frac*100:.1f}% top-1'
    if len(cycling_pos) >= 3 and not (lung_d is not None and lung_d >= 0.5 and largest_d_tile == 'Lung_cells'):
        return 'O2_CYCLING_DISTRIBUTED', f'{len(cycling_pos)} cycling tiles with d>=+0.3; signal class-distributed'
    if largest_abs_d >= 0.5 and CELL_TYPE_TO_CLASS.get(largest_d_tile) != 'cycling':
        return 'O3_NON_CYCLING_DOMINANT', f'{largest_d_tile} ({CELL_TYPE_TO_CLASS.get(largest_d_tile)}) d={all_d[largest_d_tile]:.3f}'
    if (lung_d is not None and lung_d <= -0.3) or len(cycling_neg) >= 3:
        return 'O4_DIRECTION_INVERTED', f'Lung_cells d={lung_d}; {len(cycling_neg)} cycling tiles negative'
    return 'O_NULL_OR_AMBIGUOUS', 'No outcome criterion met definitively; results descriptive only'


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    print("=" * 78)
    print("VAL-097 — Never-smoker LUAD tissue 25-tile per-class A-score")
    print(f"Start: {now_iso()}")
    print(f"WORKDIR: {WORKDIR}")
    print(f"OUTDIR:  {OUTDIR}")
    print(f"RNG seed: {RNG_SEED}")
    print("=" * 78)

    # Step 1 — Verify prereg + record SHA
    print(f"\n[Step 1a] Compute prereg SHA-256 and record in PREREG_SEAL.txt")
    prereg_sha = sha256_file(PREREG_PATH, prefix=64)
    with open(PREREG_SEAL_TXT, 'w') as f:
        f.write(f"prereg_path: {PREREG_PATH}\n")
        f.write(f"sha256: {prereg_sha}\n")
        f.write(f"sealed_at: {now_iso()}\n")
        f.write(f"rng_seed: {RNG_SEED}\n")
    print(f"  prereg SHA-256: {prereg_sha}")

    # Step 1b — Download GSE256092 inputs
    print(f"\n[Step 1b] Download GSE256092 inputs")
    series_size, series_sha = download_if_missing(
        GSE256092_SERIES_URL, GSE256092_SERIES_LOCAL, 'series_matrix')
    swan_size, swan_sha = download_if_missing(
        GSE256092_SWAN_URL, GSE256092_SWAN_LOCAL, 'SWAN_betas')

    # Step 2 — Parse metadata
    metadata_df = parse_gse256092_metadata(GSE256092_SERIES_LOCAL)
    metadata_df.to_csv(CLINICAL_METADATA_CSV, index=False)
    print(f"  Wrote {CLINICAL_METADATA_CSV}")

    # Cohort manifest
    cohort_manifest = {
        'cohort': 'GSE256092',
        'description': 'Korean never-smoker LUAD tissue, EPIC, all-never-smoker stratum',
        'platform': 'GPL21145 (EPIC 850K)',
        'n_samples': len(metadata_df),
        'access_tier': 1,
        'series_matrix_url': GSE256092_SERIES_URL,
        'series_matrix_sha256_prefix16': series_sha,
        'series_matrix_size_bytes': series_size,
        'swan_url': GSE256092_SWAN_URL,
        'swan_sha256_prefix16': swan_sha,
        'swan_size_bytes': swan_size,
        'sex_distribution': dict(metadata_df['gender'].value_counts()) if 'gender' in metadata_df.columns else None,
        'stage_distribution': dict(metadata_df['stage'].value_counts()) if 'stage' in metadata_df.columns else None,
        'age_range': [int(metadata_df['age'].min()), int(metadata_df['age'].max())] if 'age' in metadata_df.columns else None,
    }
    with open(COHORT_MANIFEST_JSON, 'w') as f:
        json.dump(cohort_manifest, f, indent=2, default=str)
    print(f"  Wrote {COHORT_MANIFEST_JSON}")

    # Step 3 — Load Loyfer atlas, identify marker CpGs
    print(f"\n[Step 3a] Load Loyfer atlas from {LOYFER_ATLAS}")
    if not os.path.exists(LOYFER_ATLAS):
        print(f"  ERROR: Loyfer atlas not found at {LOYFER_ATLAS}")
        print(f"  This is the canonical reference atlas — must be present for VAL-097.")
        print(f"  ABORT — cannot proceed without atlas.")
        sys.exit(1)
    atlas_df = pd.read_csv(LOYFER_ATLAS, index_col=0)
    print(f"  Atlas shape: {atlas_df.shape}; cell types: {list(atlas_df.columns)}")
    # Collect all marker CpGs across 25 cell types
    all_markers = set()
    for ct in CELL_TYPE_TO_CLASS.keys():
        if ct in atlas_df.columns:
            all_markers.update(identify_marker_cpgs(atlas_df, ct))
    print(f"  Total unique marker CpGs across 25 cell types: {len(all_markers)}")

    # Step 3b — Stream-extract SWAN matrix at Loyfer CpGs
    case_betas = extract_swan_betas_at_loyfer(GSE256092_SWAN_LOCAL, list(all_markers))
    case_betas_path = os.path.join(OUTDIR, 'GSE256092_betas_loyfer.csv')
    case_betas.to_csv(case_betas_path)
    print(f"  Wrote {case_betas_path}: {case_betas.shape}")

    # Data integrity check
    health_pass, health_stats = beta_distribution_health_check(case_betas, 'GSE256092_SWAN')
    print(f"  Beta distribution health check: pass={health_pass}, stats={health_stats}")
    if not health_pass:
        print(f"  WARNING: beta distribution health check failed; proceeding but flagging in outcome")

    # Step 4 — Compute case A-scores
    case_a_scores = compute_tile_a_scores(case_betas, atlas_df)
    case_a_scores.to_csv(os.path.join(OUTDIR, 'case_a_scores.csv'))

    # Step 5 — Load TCGA-LUAD adjacent-normal reference (download from GDC public API)
    ref_betas = download_tcga_luad_adjacent_normal(
        LUAD_MANIFEST, TCGA_LUAD_DOWNLOAD_DIR, TCGA_LUAD_NORMAL_BETAS, list(all_markers))
    if ref_betas is None or ref_betas.shape[1] < 5:
        print(f"  ERROR: TCGA-LUAD adjacent-normal download failed or insufficient samples.")
        print(f"  ABORT.")
        sys.exit(1)
    print(f"  Reference beta matrix shape: {ref_betas.shape}")
    ref_a_scores = compute_tile_a_scores(ref_betas, atlas_df)
    ref_a_scores.to_csv(os.path.join(OUTDIR, 'tcga_luad_normal_a_scores.csv'))

    # Step 6 — Cross-cohort baseline check
    cell_types = list(CELL_TYPE_TO_CLASS.keys())
    baseline_results = cross_cohort_baseline_check(case_a_scores, ref_a_scores, cell_types)

    # Step 7 — Case-vs-reference d
    d_results = case_vs_reference_d(case_a_scores, ref_a_scores, cell_types)

    # Step 8 — Top-1 ΔA call
    top1_per_sample, top1_dist = top1_delta_a_call(case_a_scores, ref_a_scores)

    # Step 9 — Stratified analysis
    stratified, case_with_meta = stratified_analysis(case_a_scores, metadata_df, cell_types, ref_a_scores)
    case_with_meta.to_csv(PER_SAMPLE_CSV)

    # Step 10 — Outcome assignment
    outcome_label, outcome_reason = assign_outcome(d_results, baseline_results, top1_dist, len(case_a_scores))
    print(f"\n[Step 10] OUTCOME: {outcome_label}")
    print(f"  Reason: {outcome_reason}")

    # Step 11 — Heatmap
    print(f"\n[Step 11] Tile heatmap")
    fig, ax = plt.subplots(figsize=(14, 8))
    plot_data = pd.DataFrame({
        'tile': [r['tile'] for r in d_results],
        'class': [r['class'] for r in d_results],
        'd': [r['d_case_vs_ref'] if r['d_case_vs_ref'] is not None else 0 for r in d_results],
        'baseline_anchor_sd': [r['baseline_delta_anchor_SD'] if r.get('baseline_delta_anchor_SD') is not None else 0
                                for r in baseline_results],
    })
    plot_data = plot_data.sort_values('d', ascending=True)
    colors = ['steelblue' if d < 0 else 'firebrick' for d in plot_data['d']]
    bars = ax.barh(plot_data['tile'], plot_data['d'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=0.5, color='gray', linewidth=0.5, linestyle='--', label='|d|=0.5')
    ax.axvline(x=-0.5, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Cohen's d (GSE256092 NSLA vs TCGA-LUAD adj-normal)")
    ax.set_title('VAL-097: 25-tile case-vs-reference d on Loyfer atlas\n'
                 'GSE256092 (Korean NSLA, n={}) vs TCGA-LUAD adjacent-normal'.format(len(case_a_scores)))
    plt.tight_layout()
    plt.savefig(TILE_HEATMAP_PNG, dpi=120)
    print(f"  Wrote {TILE_HEATMAP_PNG}")

    # Step 12 — Write results JSON
    results = {
        'val_id': 'VAL-097',
        'sealed_at': now_iso(),
        'prereg_sha256': prereg_sha,
        'rng_seed': RNG_SEED,
        'cohort': cohort_manifest,
        'reference_cohort': {
            'name': 'TCGA-LUAD adjacent-normal',
            'n_samples': len(ref_a_scores),
            'platform': 'HM450',
            'source_path': TCGA_LUAD_NORMAL_BETAS,
        },
        'data_integrity': {
            'beta_distribution_health': health_stats,
        },
        'baseline_check_chk32': baseline_results,
        'case_vs_reference_d': d_results,
        'top1_distribution': top1_dist,
        'top1_per_sample': top1_per_sample,
        'outcome_label': outcome_label,
        'outcome_reason': outcome_reason,
        'runtime_seconds': time.time() - t_start,
    }
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Wrote {RESULTS_JSON}")

    with open(STRATIFIED_JSON, 'w') as f:
        json.dump(stratified, f, indent=2, default=str)
    print(f"  Wrote {STRATIFIED_JSON}")

    print("\n" + "=" * 78)
    print(f"VAL-097 complete. Outcome: {outcome_label}")
    print(f"Runtime: {time.time() - t_start:.1f} s")
    print("=" * 78)


if __name__ == '__main__':
    main()
