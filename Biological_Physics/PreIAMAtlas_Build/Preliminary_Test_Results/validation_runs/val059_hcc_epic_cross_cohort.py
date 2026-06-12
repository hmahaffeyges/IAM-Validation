#!/usr/bin/env python3
"""
===============================================================================
VAL-059 — hcc-epic Xu-538 blood methylation cross-cohort validation
===============================================================================

Pre-registered: VAL_059_PREREG.md (SHA f06fcd3fc91ae0ca9f212f029577357dfc69abc31e9fd97cc1c67a6f5aae4c90)
Amendment:      VAL_059_PREREG_AMENDMENT.md (SHA b669a4c87db545b054e8f8aa87cdab30e22130a4cf072fc00ae767a9c89b3191)
Seals:          VAL_059_SEAL.txt (original 2026-04-24 06:50:36 UTC, amendment 06:55 UTC)

Questions:
  1. Does the Xu-538 immune panel produce elevated pooled-entropy A-score in
     HCC cases vs controls on peripheral blood methylation at d > 0.3?
  2. Does the finding replicate across TWO cohorts with DIFFERENT substrates
     (whole-blood leukocyte vs ccfDNA plasma)?

Cohorts (both public GEO, EPIC 850K, non-overlapping):

  PRIMARY — GSE281691: Metabolic HCC international multicenter
    - n=481: 221 HCC cases + 260 metabolic-liver-disease controls
    - Substrate: peripheral blood LEUKOCYTE DNA (whole-blood cells)
    - Published classifier: 55-CpG panel AUC 0.79
    - Sex: 305 M / 176 F (1.73:1, matches HCC male-predominance)

  REPLICATION — GSE298812: Nigerian HIV+ HCC (Soliman et al.)
    - n=245 spanning 4 disease-spectrum groups
      * HIV-Pos_HCC-Neg:  115 (HCC-free controls)
      * HIV-Pos_HCC-Fib:   68 (fibrosis)
      * HIV-Pos_HCC-Cir:   31 (cirrhosis)
      * HIV-Pos_HCC-Pos:   31 (HCC cases)
    - Substrate: ccfDNA from PLASMA (cell-free circulating DNA)
    - Published classifier: ccfDNAmRF random forest AUC 92-97%
    - Sex: 84 M / 161 F (unusual: HCC literature typically 3:1 M:F,
      but this cohort HIV-positive Nigerian setting may differ)
    - Primary comparison: HCC-Pos vs HCC-Neg
    - Secondary: HCC-Pos vs (HCC-Fib + HCC-Cir + HCC-Neg combined)
    - Disease spectrum: dose-response analysis across 4 groups

Panel (frozen):  Xu-538 immune panel
  SHA-256: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
  n_CpGs:  538 (Xu 2020 JNCI Sister Study + EPIC-Italy breast cancer)

Scoring (amended):
  M1 = pooled_entropy A-score = mean(H(β)/H_min_immune)
  H_min(immune) = 0.838889 (G-003b MCMC posterior, frozen)
  Per-CpG Δβ direction preservation check retained.
  Moss per-CpG metrics M2/M3 REMOVED (NDA-gated calibration layer).

RNG seed: 20260420 (matches VAL-047 / VAL-051 / VAL-052 / VAL-056 / VAL-057 / VAL-058)
===============================================================================
"""

import gzip
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# FROZEN CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

PREREG_SHA_ORIGINAL = 'f06fcd3fc91ae0ca9f212f029577357dfc69abc31e9fd97cc1c67a6f5aae4c90'
PREREG_SHA_AMENDMENT = 'b669a4c87db545b054e8f8aa87cdab30e22130a4cf072fc00ae767a9c89b3191'

H_MIN_IMMUNE = 0.838889
XU538_PANEL_SHA = 'ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6'

PANEL_FILE = '/home/claude/val058_data/xu538_panel.json'  # same panel for both

GSE281691_BETA = '/home/claude/val059_data/GSE281691_HCC_matrix_processed.txt.gz'
GSE281691_BETA_SHA = '5ce39843c1a2cdf20db0c73d64be976be7c3820b19a3d95cb9003245a2b6e11f'
GSE281691_MAP = '/home/claude/val059_data/gse281691_sample_map.json'

GSE298812_BETA = '/home/claude/val059_data/GSE298812_processed_data.txt.gz'
GSE298812_BETA_SHA = '4a586138987065a70f473f9d97d7e36646829371f662ffe052146a5232edd981'
GSE298812_MAP = '/home/claude/val059_data/gse298812_sample_map.json'

RNG_SEED = 20260420


def H(b):
    if b <= 0 or b >= 1:
        return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def cohens_d(a, b):
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = math.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    return (ma - mb) / pooled if pooled > 0 else float('nan')


def permutation_p(case, ctrl, n_perms=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    case = np.asarray(case, dtype=float); case = case[~np.isnan(case)]
    ctrl = np.asarray(ctrl, dtype=float); ctrl = ctrl[~np.isnan(ctrl)]
    if len(case) < 2 or len(ctrl) < 2:
        return float('nan')
    observed = np.mean(case) - np.mean(ctrl)
    pooled = np.concatenate([case, ctrl])
    n_case = len(case)
    hits = 0
    for _ in range(n_perms):
        rng.shuffle(pooled)
        if abs(np.mean(pooled[:n_case]) - np.mean(pooled[n_case:])) >= abs(observed):
            hits += 1
    return (hits + 1) / (n_perms + 1)


def bootstrap_ci_d(case, ctrl, n_boot=10000, ci=95, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED + 1)
    case = np.asarray(case, dtype=float); case = case[~np.isnan(case)]
    ctrl = np.asarray(ctrl, dtype=float); ctrl = ctrl[~np.isnan(ctrl)]
    if len(case) < 2 or len(ctrl) < 2:
        return float('nan'), float('nan')
    ds = []
    for _ in range(n_boot):
        b1 = rng.choice(case, size=len(case), replace=True)
        b2 = rng.choice(ctrl, size=len(ctrl), replace=True)
        d = cohens_d(b1, b2)
        if not math.isnan(d):
            ds.append(d)
    if not ds:
        return float('nan'), float('nan')
    return float(np.percentile(ds, (100-ci)/2)), float(np.percentile(ds, 100-(100-ci)/2))


def extract_xu538_betas(beta_file, panel_cpgs, skip_cols=None):
    """Stream β file, pull only Xu-538 CpGs. skip_cols indicates column indices to skip (Detection Pval)."""
    target = set(panel_cpgs)
    found = {}
    sample_ids = None
    rows = 0
    with gzip.open(beta_file, 'rt') as f:
        header = f.readline().rstrip('\n').split('\t')
        # Determine kept columns (excluding first ID and any Pval columns)
        if skip_cols is None:
            # Default: all columns except first
            keep_idx = list(range(1, len(header)))
        else:
            keep_idx = [i for i in range(1, len(header)) if i not in skip_cols]
        sample_ids = [header[i] for i in keep_idx]
        for line in f:
            rows += 1
            tab_idx = line.find('\t')
            if tab_idx == -1:
                continue
            cpg = line[:tab_idx].strip('"')
            if cpg not in target:
                continue
            parts = line.rstrip('\n').split('\t')
            vals = []
            for i in keep_idx:
                if i < len(parts):
                    p = parts[i].strip().strip('"')
                    if p == '' or p.lower() in ('na', 'nan'):
                        vals.append(np.nan)
                    else:
                        try:
                            vals.append(float(p))
                        except ValueError:
                            vals.append(np.nan)
                else:
                    vals.append(np.nan)
            found[cpg] = np.array(vals, dtype=float)
            if rows % 100000 == 0:
                print(f"      ...{rows:,} rows scanned, {len(found)} CpGs found")
            if len(found) == len(target):
                break
    return sample_ids, found, rows


def compute_per_sample_A_score(betas, sample_order):
    """Pooled-entropy A-score per sample across all Xu-538 CpGs available."""
    n = len(sample_order)
    A = np.full(n, np.nan)
    for i in range(n):
        h_vals = []
        for cpg, arr in betas.items():
            b = arr[i]
            if not np.isnan(b):
                h_vals.append(H(b) / H_MIN_IMMUNE)
        if len(h_vals) >= 300:  # ≥300 CpGs required
            A[i] = np.mean(h_vals)
    return A


def analyze_cohort(cohort_label, beta_file, beta_sha, sample_map, panel_cpgs, case_label, control_label,
                   skip_pval_cols=False, rng=None):
    """Run full per-cohort analysis. Returns dict with all results."""
    print(f"\n{'='*78}\n{cohort_label}\n{'='*78}")

    # Verify β SHA
    with open(beta_file, 'rb') as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == beta_sha, f"SHA mismatch for {cohort_label}: {actual_sha}"
    print(f"  β matrix SHA verified: {actual_sha[:16]}...")

    # Determine skip_cols for Pval-interleaved matrices (GSE298812)
    skip_cols = None
    if skip_pval_cols:
        with gzip.open(beta_file, 'rt') as f:
            header = f.readline().rstrip('\n').split('\t')
        skip_cols = {i for i, h in enumerate(header) if 'Detection Pval' in h}
        print(f"  Skipping {len(skip_cols)} Detection Pval columns")

    # Extract Xu-538 betas
    print(f"  Extracting Xu-538 from {beta_file}...")
    sample_ids, betas, rows_scanned = extract_xu538_betas(beta_file, panel_cpgs, skip_cols=skip_cols)
    print(f"    Rows scanned: {rows_scanned:,}")
    print(f"    CpGs found: {len(betas)}/538 ({100*len(betas)/538:.1f}%)")
    print(f"    Sample IDs in β matrix: {len(sample_ids)}")

    # Match to metadata
    sentrix_to_meta = {s['sentrix']: s for s in sample_map if s.get('sentrix')}
    matched = 0
    aligned_samples = []
    for sid in sample_ids:
        if sid in sentrix_to_meta:
            aligned_samples.append(sentrix_to_meta[sid])
            matched += 1
        else:
            aligned_samples.append(None)
    print(f"    Matched to metadata: {matched}/{len(sample_ids)}")

    # Compute A-score per sample
    print(f"  Computing M1 A-score per sample...")
    A = compute_per_sample_A_score(betas, sample_ids)
    n_valid = int(np.sum(~np.isnan(A)))
    print(f"    Valid A-scores: {n_valid}/{len(sample_ids)}")

    # Partition by group
    def get_group_label(meta):
        if meta is None:
            return 'UNMATCHED'
        # For GSE281691: diseasestate Case/Control
        if 'diseasestate' in meta and meta['diseasestate']:
            return meta['diseasestate']
        # For GSE298812: group field
        if 'group' in meta and meta['group']:
            return meta['group']
        return 'UNKNOWN'

    groups = np.array([get_group_label(m) for m in aligned_samples])
    group_counts = Counter(groups)
    print(f"  Group composition: {dict(group_counts)}")

    # Case vs control (primary comparison)
    case_mask = (groups == case_label) & ~np.isnan(A)
    ctrl_mask = (groups == control_label) & ~np.isnan(A)
    A_case = A[case_mask]
    A_ctrl = A[ctrl_mask]
    print(f"  {case_label} (case):    n={len(A_case)}  mean={np.mean(A_case):.5f}  sd={np.std(A_case, ddof=1):.5f}")
    print(f"  {control_label} (ctrl): n={len(A_ctrl)}  mean={np.mean(A_ctrl):.5f}  sd={np.std(A_ctrl, ddof=1):.5f}")

    # Primary Cohen's d
    d = cohens_d(A_case, A_ctrl)
    p = permutation_p(A_case, A_ctrl, n_perms=10000, rng=rng)
    ci_lo, ci_hi = bootstrap_ci_d(A_case, A_ctrl, n_boot=10000)
    print(f"  d = {d:+.4f}  [95% CI {ci_lo:+.3f}, {ci_hi:+.3f}]  p_perm = {p:.4f}")

    # Sex stratification (CCL-002)
    print(f"  Sex-stratified:")
    sex_results = {}
    for sex_label in ['Male', 'Female']:
        sex_arr = np.array([m.get('sex') if m else None for m in aligned_samples])
        mc = case_mask & (sex_arr == sex_label)
        mt = ctrl_mask & (sex_arr == sex_label)
        A_c = A[mc]; A_t = A[mt]
        if len(A_c) >= 3 and len(A_t) >= 3:
            d_sex = cohens_d(A_c, A_t)
            p_sex = permutation_p(A_c, A_t, n_perms=10000, rng=rng)
            sex_results[sex_label] = {
                'n_case': int(len(A_c)), 'n_ctrl': int(len(A_t)),
                'cohens_d': float(d_sex), 'perm_p': float(p_sex),
            }
            print(f"    {sex_label:7s}  n_case={len(A_c):3d}  n_ctrl={len(A_t):3d}  d={d_sex:+.4f}  p={p_sex:.4f}")

    # Age regression
    print(f"  Age-regressed (ctrl HC fit):")
    ages = np.array([m.get('age') if m else None for m in aligned_samples], dtype=float)
    valid_ctrl = ctrl_mask & ~np.isnan(ages)
    ctrl_ages = ages[valid_ctrl]
    ctrl_A = A[valid_ctrl]
    age_reg = None
    if len(ctrl_ages) >= 10:
        slope, intercept = np.polyfit(ctrl_ages, ctrl_A, 1)
        r2 = float(1 - np.sum((ctrl_A - (slope*ctrl_ages + intercept))**2) / np.sum((ctrl_A - np.mean(ctrl_A))**2))
        A_resid = np.full(len(A), np.nan)
        for i in range(len(A)):
            if not np.isnan(A[i]) and not np.isnan(ages[i]):
                A_resid[i] = A[i] - (slope*ages[i] + intercept)
        A_case_resid = A_resid[case_mask & ~np.isnan(A_resid)]
        A_ctrl_resid = A_resid[ctrl_mask & ~np.isnan(A_resid)]
        d_age = cohens_d(A_case_resid, A_ctrl_resid)
        p_age = permutation_p(A_case_resid, A_ctrl_resid, n_perms=10000, rng=rng)
        age_reg = {
            'slope': float(slope), 'intercept': float(intercept), 'r_squared': r2,
            'cohens_d_age_regressed': float(d_age),
            'perm_p_age_regressed': float(p_age),
        }
        print(f"    slope={slope:+.5f}, intercept={intercept:+.4f}, R²={r2:.4f}")
        print(f"    age-regressed d = {d_age:+.4f}  p = {p_age:.4f}")

    # Per-CpG direction preservation check
    print(f"  Per-CpG Δβ analysis...")
    case_idx = np.where(case_mask)[0]
    ctrl_idx = np.where(ctrl_mask)[0]
    n_hyper_case = 0
    n_hypo_case = 0
    significant = []
    for cpg, arr in betas.items():
        c_vals = arr[case_idx]; c_vals = c_vals[~np.isnan(c_vals)]
        t_vals = arr[ctrl_idx]; t_vals = t_vals[~np.isnan(t_vals)]
        if len(c_vals) < 5 or len(t_vals) < 5:
            continue
        delta = float(np.mean(c_vals) - np.mean(t_vals))
        if delta > 0: n_hyper_case += 1
        elif delta < 0: n_hypo_case += 1
        if abs(delta) > 0.02:
            significant.append((cpg, delta))
    total_signed = n_hyper_case + n_hypo_case
    hyper_rate = n_hyper_case / total_signed if total_signed > 0 else 0
    print(f"    Hypermethylated in case: {n_hyper_case}/{total_signed} ({100*hyper_rate:.1f}%)")
    print(f"    |Δβ|>0.02 CpGs: {len(significant)}")

    return {
        'cohort_label': cohort_label,
        'beta_sha': actual_sha,
        'n_cpgs_on_epic': len(betas),
        'epic_coverage': round(len(betas)/538, 4),
        'n_samples_total': len(sample_ids),
        'n_samples_matched': matched,
        'n_valid_A': n_valid,
        'group_counts': dict(group_counts),
        'case_label': case_label,
        'control_label': control_label,
        'case_stats': {
            'n': int(len(A_case)),
            'mean': float(np.mean(A_case)),
            'sd': float(np.std(A_case, ddof=1)),
        },
        'control_stats': {
            'n': int(len(A_ctrl)),
            'mean': float(np.mean(A_ctrl)),
            'sd': float(np.std(A_ctrl, ddof=1)),
        },
        'primary_cohens_d': float(d),
        'primary_perm_p': float(p),
        'primary_ci_95_lo': float(ci_lo),
        'primary_ci_95_hi': float(ci_hi),
        'sex_stratified': sex_results,
        'age_regression': age_reg,
        'per_cpg_direction': {
            'n_analyzed': total_signed,
            'n_hypermethylated_in_case': n_hyper_case,
            'n_hypomethylated_in_case': n_hypo_case,
            'hyper_rate': round(hyper_rate, 4),
            'n_significant_abs_delta_gt_0_02': len(significant),
        },
    }


def main():
    t0 = time.time()
    print("="*78)
    print("VAL-059 — hcc-epic cross-cohort Xu-538 validation")
    print("="*78)
    print(f"Prereg original:  {PREREG_SHA_ORIGINAL[:16]}...")
    print(f"Prereg amendment: {PREREG_SHA_AMENDMENT[:16]}...")

    # Load Xu-538 panel
    with open(PANEL_FILE) as f:
        panel = json.load(f)
    panel_cpgs = set(panel['cpgs'])
    with open(PANEL_FILE, 'rb') as f:
        panel_sha = hashlib.sha256(f.read()).hexdigest()
    assert panel_sha == XU538_PANEL_SHA
    print(f"Xu-538 panel: {len(panel_cpgs)} CpGs, SHA verified {panel_sha[:16]}...")

    # Load sample maps
    with open(GSE281691_MAP) as f:
        map_281 = json.load(f)
    with open(GSE298812_MAP) as f:
        map_298 = json.load(f)

    rng = np.random.default_rng(RNG_SEED)

    # PRIMARY cohort
    result_281 = analyze_cohort(
        'GSE281691 (PRIMARY) — Metabolic HCC multicenter, whole-blood leukocyte',
        GSE281691_BETA, GSE281691_BETA_SHA, map_281, panel_cpgs,
        case_label='Case', control_label='Control',
        skip_pval_cols=False, rng=rng,
    )

    # REPLICATION cohort (primary comparison: HCC-Pos vs HCC-Neg)
    result_298 = analyze_cohort(
        'GSE298812 (REPLICATION) — Nigerian HIV+ HCC, ccfDNA plasma',
        GSE298812_BETA, GSE298812_BETA_SHA, map_298, panel_cpgs,
        case_label='HIV-Pos_HCC-Pos', control_label='HIV-Pos_HCC-Neg',
        skip_pval_cols=True, rng=rng,
    )

    # Cross-cohort synthesis
    print(f"\n{'='*78}\nCROSS-COHORT SYNTHESIS\n{'='*78}")
    d1 = result_281['primary_cohens_d']
    d2 = result_298['primary_cohens_d']
    print(f"GSE281691 (leukocyte) d = {d1:+.4f}")
    print(f"GSE298812 (ccfDNA)    d = {d2:+.4f}")

    direction_match = (d1 > 0 and d2 > 0) or (d1 < 0 and d2 < 0)
    magnitude_ratio = min(abs(d1), abs(d2)) / max(abs(d1), abs(d2)) if max(abs(d1), abs(d2)) > 0 else 0
    print(f"Direction match: {direction_match}")
    print(f"Magnitude preservation ratio: {magnitude_ratio:.3f}")

    # Outcome assignment per amended decision matrix
    if d1 > 0.3 and d2 > 0.3 and direction_match and magnitude_ratio > 0.5:
        outcome_code = 'O1'
        outcome_name = 'CROSS_PLATFORM_VALIDATED'
        tier = 'cross_platform_validated'
    elif d1 > 0.3 and not (d2 > 0.3):
        outcome_code = 'O2'
        outcome_name = 'SINGLE_COHORT_VALIDATED_LEUKOCYTE'
        tier = 'cohort_screening_validated'
    elif d2 > 0.3 and not (d1 > 0.3):
        outcome_code = 'O3'
        outcome_name = 'SINGLE_COHORT_VALIDATED_CCFDNA'
        tier = 'cohort_screening_validated'
    elif abs(d1) < 0.3 and abs(d2) < 0.3:
        outcome_code = 'O4'
        outcome_name = 'NULL_ON_BOTH'
        tier = 'null_documented'
    else:
        outcome_code = 'O5'
        outcome_name = 'UNEXPECTED'
        tier = 'deferred'

    print(f"\n>>> OUTCOME: {outcome_code} — {outcome_name}")
    print(f">>> Card tier: {tier}")

    # GSE298812 disease-spectrum analysis (secondary comparison)
    print(f"\n{'='*78}\nGSE298812 DISEASE SPECTRUM (secondary)\n{'='*78}")
    # Re-extract for spectrum analysis
    with gzip.open(GSE298812_BETA, 'rt') as f:
        header = f.readline().rstrip('\n').split('\t')
    skip_cols = {i for i, h in enumerate(header) if 'Detection Pval' in h}
    sids_298, betas_298, _ = extract_xu538_betas(GSE298812_BETA, panel_cpgs, skip_cols=skip_cols)
    A_298 = compute_per_sample_A_score(betas_298, sids_298)

    sentrix_to_298 = {s['sentrix']: s for s in map_298 if s.get('sentrix')}
    aligned_298 = [sentrix_to_298.get(s) for s in sids_298]
    groups_298 = np.array([m.get('group') if m else None for m in aligned_298])

    spectrum_means = {}
    baseline_group = 'HIV-Pos_HCC-Neg'
    baseline_mean = float(np.mean(A_298[(groups_298 == baseline_group) & ~np.isnan(A_298)]))
    print(f"  Baseline {baseline_group}: A = {baseline_mean:.5f}")
    for g in ['HIV-Pos_HCC-Neg', 'HIV-Pos_HCC-Fib', 'HIV-Pos_HCC-Cir', 'HIV-Pos_HCC-Pos']:
        A_g = A_298[(groups_298 == g) & ~np.isnan(A_298)]
        mean_g = float(np.mean(A_g))
        delta = mean_g - baseline_mean
        d_g = cohens_d(A_g, A_298[(groups_298 == baseline_group) & ~np.isnan(A_298)])
        spectrum_means[g] = {
            'n': int(len(A_g)),
            'mean': mean_g,
            'delta_from_baseline': delta,
            'cohens_d_vs_baseline': float(d_g),
        }
        print(f"    {g:22s} n={len(A_g):3d}  A={mean_g:.5f}  Δ={delta:+.5f}  d_vs_baseline={d_g:+.4f}")

    runtime_s = time.time() - t0
    print(f"\nRuntime: {runtime_s:.1f}s")

    # Compile full results
    results = {
        'val_id': 'VAL-059',
        'val_type': 'cross_cohort_blood_methylation_validation',
        'card_target': 'hcc-epic v0.1',
        'prereg_sha_original': PREREG_SHA_ORIGINAL,
        'prereg_sha_amendment': PREREG_SHA_AMENDMENT,
        'run_date': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'rng_seed': RNG_SEED,
        'runtime_seconds': round(runtime_s, 2),

        'panel': {
            'panel_id': 'Xu2020_breast_cancer_replicated_full',
            'panel_sha256': XU538_PANEL_SHA,
        },

        'cohort_gse281691_primary': result_281,
        'cohort_gse298812_replication': result_298,

        'cross_cohort_synthesis': {
            'd_gse281691_leukocyte': float(d1),
            'd_gse298812_ccfdna': float(d2),
            'direction_match': bool(direction_match),
            'magnitude_preservation_ratio': float(magnitude_ratio),
            'outcome_code': outcome_code,
            'outcome_name': outcome_name,
            'card_tier': tier,
        },

        'gse298812_disease_spectrum': spectrum_means,
    }

    results_json_str = json.dumps(results, indent=2, sort_keys=True, default=str)
    results['results_sha256'] = hashlib.sha256(results_json_str.encode()).hexdigest()

    out_path = Path('/home/claude/cookbook_v2.1/hcc-epic/VAL059_hcc_epic_cross_cohort_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults JSON: {out_path}")
    print(f"Results SHA:  {results['results_sha256'][:16]}...")


if __name__ == '__main__':
    main()
