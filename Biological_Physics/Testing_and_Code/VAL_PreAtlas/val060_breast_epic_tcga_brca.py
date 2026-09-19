#!/usr/bin/env python3
"""
===============================================================================
VAL-060 — breast-epic retroactive tissue validation on TCGA-BRCA HM450
===============================================================================

Pre-registered: VAL_060_PREREG.md (SHA cd8c6de4383d87203ad8ee14db6d197635021a5e036e29f3219a613b112e8fea)
Seal:           VAL_060_SEAL.txt (sealed 2026-04-24 08:14:36 UTC)

Question: Does the Xu-538 immune panel produce a measurable pooled-entropy
          A-score elevation in breast tumor tissue vs adjacent-normal breast
          tissue on TCGA-BRCA HM450, and if yes, what is the tissue-level
          effect size? Comparison anchor: VAL-058 prostate paired d = +0.497.

Cohort:   TCGA-BRCA HM450 matched tumor-normal subset
          - 186 files: 92 Primary Tumor + 91 Solid Tissue Normal + 3 Metastatic
          - 91 matched tumor-normal pairs by case_submitter_id
          - Female-predominant (783/793 primary tumors female in full cohort)
          - Illumina HumanMethylation450 platform, Level 3 β values
          - GDC public access, no dbGaP required

Panel:    Xu-538 immune panel, FROZEN
          SHA: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
          538 CpGs from Xu 2020 JNCI Sister Study + EPIC-Italy breast cancer

Scoring:  M1 = pooled-entropy A-score = mean(H(β)/H_min_immune)
          H_min(immune) = 0.838889 (G-003b MCMC posterior, frozen)

z-standardization: within-TCGA-BRCA Solid Tissue Normal (per CCL-004:
          80-cell blood baseline does NOT apply to tissue data).

RNG seed: 20260420
===============================================================================
"""

import os
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# FROZEN CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

PREREG_SHA = 'cd8c6de4383d87203ad8ee14db6d197635021a5e036e29f3219a613b112e8fea'
SEAL_TIMESTAMP = '2026-04-24 08:14:36 UTC'

H_MIN_IMMUNE = 0.838889
XU538_PANEL_SHA = 'ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6'
AGGREGATE_COHORT_SHA = 'a11efdabfe2aec78d323371ce2687dbadaa506ce44be3229ee10c79fb3c97742'

PANEL_FILE = '/home/claude/val058_data/xu538_panel.json'
FILES_DIR = '/home/claude/val060_data/files'
SAMPLE_MAP = '/home/claude/val060_data/matched_sample_metadata.json'

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


def paired_cohens_d(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return float('nan')
    sd = np.std(diffs, ddof=1)
    return np.mean(diffs) / sd if sd > 0 else float('nan')


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


def assign_outcome(d_unpaired, d_paired, direction_preserved, n_cpgs_scored):
    """
    Pre-reg decision matrix:
      O1: d_unpaired > 0.3 AND d_paired > 0.3 AND positive direction → BREAST_TISSUE_VALIDATED
      O2: 0 < d < 0.3 but direction preserved > threshold → below-threshold-positive
      O3: null or opposite direction
      O4: unexpected
    """
    # Binomial threshold for direction preservation at n=538, p<0.05 two-sided → ≥292
    direction_threshold = int(n_cpgs_scored / 2 + 1.96 * math.sqrt(n_cpgs_scored * 0.25))

    if d_unpaired > 0.3 and d_paired > 0.3:
        return 'O1', 'BREAST_EPIC_TISSUE_VALIDATED', 'tissue_arm_validated'

    if (0 < d_unpaired < 0.3 or 0 < d_paired < 0.3) and direction_preserved >= direction_threshold:
        return 'O2', 'DIRECTION_POSITIVE_BELOW_THRESHOLD', 'tissue_arm_directional_only'

    if d_unpaired < 0 and d_paired < 0:
        return 'O3', 'TISSUE_OPPOSITE_DIRECTION', 'tissue_arm_null_opposite_direction'

    if abs(d_unpaired) < 0.3 and abs(d_paired) < 0.3 and direction_preserved < direction_threshold:
        return 'O3', 'TISSUE_NULL', 'tissue_arm_null'

    return 'O4', 'UNEXPECTED_PATTERN', 'deferred'


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("="*78)
    print("VAL-060 — breast-epic TCGA-BRCA tissue validation")
    print("="*78)
    print(f"Prereg SHA:            {PREREG_SHA[:16]}... (sealed {SEAL_TIMESTAMP})")
    print(f"Panel SHA:             {XU538_PANEL_SHA[:16]}...")
    print(f"Cohort aggregate SHA:  {AGGREGATE_COHORT_SHA[:16]}...")

    # Load panel
    with open(PANEL_FILE) as f:
        panel = json.load(f)
    panel_cpgs = set(panel['cpgs'])
    with open(PANEL_FILE, 'rb') as f:
        panel_actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert panel_actual_sha == XU538_PANEL_SHA
    print(f"  Xu-538 panel verified: {panel_actual_sha[:16]}...")
    print(f"  Panel CpGs: {len(panel_cpgs)}")

    # Load sample metadata
    with open(SAMPLE_MAP) as f:
        samples = json.load(f)
    print(f"  Sample metadata: {len(samples)} entries")

    # Build mapping file_id → metadata
    meta_by_fid = {s['file_id']: s for s in samples}

    # For each file, extract only Xu-538 CpGs
    print(f"\n[extract] Reading β values at Xu-538 CpGs from {len(samples)} files...")
    t_extract = time.time()

    # per_sample_betas[file_id] = {cpg: β}
    per_sample_betas = {}
    files_read = 0

    for s in samples:
        fid = s['file_id']
        path = f"{FILES_DIR}/{fid}.txt"
        if not os.path.exists(path):
            continue
        betas = {}
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip('\n').split('\t')
                if len(parts) < 2:
                    continue
                cpg = parts[0]
                if cpg not in panel_cpgs:
                    continue
                v = parts[1].strip()
                if v.lower() in ('', 'na', 'nan'):
                    continue
                try:
                    b = float(v)
                    if 0.0 <= b <= 1.0:
                        betas[cpg] = b
                except ValueError:
                    continue
        per_sample_betas[fid] = betas
        files_read += 1
        if files_read % 30 == 0:
            print(f"    ...{files_read}/{len(samples)} files read, elapsed {time.time()-t_extract:.1f}s", flush=True)

    print(f"  Files read: {files_read}")
    print(f"  HM450 Xu-538 coverage per sample (median): {int(np.median([len(b) for b in per_sample_betas.values()]))}")
    coverage_fractions = [len(b)/538 for b in per_sample_betas.values()]
    print(f"  Coverage range: {min(coverage_fractions):.3f} to {max(coverage_fractions):.3f}")

    # Compute M1 per sample
    print(f"\n[score] Computing M1 pooled-entropy A-score per sample...")
    M1 = {}
    n_cpgs_per = {}
    for fid, betas in per_sample_betas.items():
        if len(betas) >= 300:
            h_vals = [H(b)/H_MIN_IMMUNE for b in betas.values()]
            M1[fid] = np.mean(h_vals)
            n_cpgs_per[fid] = len(betas)

    print(f"  Valid M1 scores: {len(M1)}/{files_read}")

    # Partition by sample type
    M1_tumor = []; M1_normal = []
    meta_tumor = []; meta_normal = []
    for fid, m in M1.items():
        meta = meta_by_fid.get(fid, {})
        st = meta.get('sample_type', '?')
        if st == 'Primary Tumor':
            M1_tumor.append(m); meta_tumor.append(meta)
        elif st == 'Solid Tissue Normal':
            M1_normal.append(m); meta_normal.append(meta)

    M1_tumor = np.array(M1_tumor)
    M1_normal = np.array(M1_normal)

    print(f"\n  M1_tumor:  n={len(M1_tumor)}  mean={np.mean(M1_tumor):.5f}  sd={np.std(M1_tumor, ddof=1):.5f}")
    print(f"  M1_normal: n={len(M1_normal)}  mean={np.mean(M1_normal):.5f}  sd={np.std(M1_normal, ddof=1):.5f}")

    rng = np.random.default_rng(RNG_SEED)
    # Unpaired
    print(f"\n[stats] Unpaired Cohen's d (tumor vs adj-normal, all samples)...")
    d_unpaired = cohens_d(M1_tumor, M1_normal)
    p_unpaired = permutation_p(M1_tumor, M1_normal, n_perms=10000, rng=rng)
    ci_lo, ci_hi = bootstrap_ci_d(M1_tumor, M1_normal, n_boot=10000)
    print(f"  d(unpaired) = {d_unpaired:+.4f}  [95% CI {ci_lo:+.3f}, {ci_hi:+.3f}]  p_perm = {p_unpaired:.4f}")

    # Paired — match by case_submitter_id
    print(f"\n[stats] Paired tumor-vs-normal difference...")
    pair_dict = defaultdict(dict)
    for fid, m in M1.items():
        meta = meta_by_fid.get(fid, {})
        case = meta.get('case_submitter_id')
        st = meta.get('sample_type', '?')
        if case and st in ('Primary Tumor', 'Solid Tissue Normal'):
            # If a case has multiple primary tumors, use mean
            if st in pair_dict[case]:
                pair_dict[case][st] = (pair_dict[case][st] + m) / 2
            else:
                pair_dict[case][st] = m

    complete_pairs = [c for c, d in pair_dict.items() if 'Primary Tumor' in d and 'Solid Tissue Normal' in d]
    paired_diffs = np.array([pair_dict[c]['Primary Tumor'] - pair_dict[c]['Solid Tissue Normal'] for c in complete_pairs])
    print(f"  Complete pairs: {len(complete_pairs)}")
    print(f"  Mean paired diff: {np.mean(paired_diffs):+.5f}  sd: {np.std(paired_diffs, ddof=1):.5f}")
    d_paired = paired_cohens_d(paired_diffs)

    # Paired sign-flip permutation
    n_paired_perms = 10000
    observed_mean = np.mean(paired_diffs)
    hits = 0
    for _ in range(n_paired_perms):
        signs = rng.choice([-1, 1], size=len(paired_diffs))
        if abs(np.mean(paired_diffs * signs)) >= abs(observed_mean):
            hits += 1
    p_paired = (hits + 1) / (n_paired_perms + 1)
    print(f"  d(paired) = {d_paired:+.4f}  sign-flip perm p = {p_paired:.4f}")

    # Per-CpG direction
    print(f"\n[stats] Per-CpG Δβ direction preservation vs Xu 2020 breast blood trend...")
    tumor_fids = [fid for fid, m in M1.items() if meta_by_fid.get(fid, {}).get('sample_type') == 'Primary Tumor']
    normal_fids = [fid for fid, m in M1.items() if meta_by_fid.get(fid, {}).get('sample_type') == 'Solid Tissue Normal']
    
    # For each Xu-538 CpG, compute Δβ = tumor_mean − normal_mean
    per_cpg_direction = {}
    n_hypermeth_tumor = 0
    n_hypometh_tumor = 0
    n_signif = 0
    for cpg in panel_cpgs:
        tumor_vals = [per_sample_betas[fid].get(cpg) for fid in tumor_fids if cpg in per_sample_betas.get(fid, {})]
        normal_vals = [per_sample_betas[fid].get(cpg) for fid in normal_fids if cpg in per_sample_betas.get(fid, {})]
        if len(tumor_vals) < 10 or len(normal_vals) < 10:
            continue
        delta = float(np.mean(tumor_vals) - np.mean(normal_vals))
        per_cpg_direction[cpg] = delta
        if delta > 0: n_hypermeth_tumor += 1
        elif delta < 0: n_hypometh_tumor += 1
        if abs(delta) > 0.05: n_signif += 1

    total_signed = n_hypermeth_tumor + n_hypometh_tumor
    hyper_rate = n_hypermeth_tumor / total_signed if total_signed > 0 else 0
    print(f"  CpGs analyzed: {total_signed}/{len(panel_cpgs)}")
    print(f"  Hypermethylated in tumor: {n_hypermeth_tumor} ({100*hyper_rate:.1f}%)")
    print(f"  Hypomethylated in tumor:  {n_hypometh_tumor} ({100*(1-hyper_rate):.1f}%)")
    print(f"  |Δβ| > 0.05 significant CpGs: {n_signif}")

    # Sex stratification — female only
    print(f"\n[stats] Female-only stratification...")
    female_tumor = [m for fid, m in M1.items() 
                    if meta_by_fid.get(fid, {}).get('sample_type') == 'Primary Tumor'
                    and meta_by_fid.get(fid, {}).get('gender') == 'female']
    female_normal = [m for fid, m in M1.items()
                     if meta_by_fid.get(fid, {}).get('sample_type') == 'Solid Tissue Normal'
                     and meta_by_fid.get(fid, {}).get('gender') == 'female']
    d_female = cohens_d(female_tumor, female_normal)
    print(f"  Female-only: n_tumor={len(female_tumor)}  n_normal={len(female_normal)}  d={d_female:+.4f}")

    # Outcome
    outcome_code, outcome_name, tier = assign_outcome(d_unpaired, d_paired, n_hypermeth_tumor, total_signed)
    print(f"\n{'='*78}")
    print(f"OUTCOME: {outcome_code} — {outcome_name}")
    print(f"Tissue-arm tier: {tier}")
    print(f"{'='*78}")
    print(f"  d(unpaired) = {d_unpaired:+.4f}  (threshold 0.3 for O1)")
    print(f"  d(paired)   = {d_paired:+.4f}  (threshold 0.3 for O1)")
    print(f"  Direction:    {n_hypermeth_tumor}/{total_signed} hypermethylated ({100*hyper_rate:.1f}%)")

    # VAL-058 comparison
    print(f"\n=== Comparison to VAL-058 prostate-epic ===")
    print(f"  VAL-058 (prostate): unpaired d = +0.400  paired d = +0.497  hypermethylation fraction 45.1%")
    print(f"  VAL-060 (breast):   unpaired d = {d_unpaired:+.3f}  paired d = {d_paired:+.3f}  hypermethylation fraction {100*hyper_rate:.1f}%")

    runtime_s = time.time() - t0
    print(f"\nRuntime: {runtime_s:.1f}s")

    # Build full results
    results = {
        'val_id': 'VAL-060',
        'val_type': 'retroactive_per_card_tissue_validation',
        'card_target': 'breast-epic v2.2 (tissue-arm addition)',
        'prereg_sha': PREREG_SHA,
        'seal_timestamp': SEAL_TIMESTAMP,
        'panel_sha256': XU538_PANEL_SHA,
        'cohort_aggregate_sha': AGGREGATE_COHORT_SHA,
        'run_date': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'rng_seed': RNG_SEED,
        'runtime_seconds': round(runtime_s, 2),

        'cohort': {
            'source': 'TCGA-BRCA HM450 matched tumor-normal subset',
            'platform': 'Illumina HumanMethylation450',
            'n_files_downloaded': files_read,
            'n_primary_tumor': int(len(M1_tumor)),
            'n_solid_tissue_normal': int(len(M1_normal)),
            'n_complete_pairs': len(complete_pairs),
            'access': 'open (NIH GDC, no dbGaP required)',
        },

        'xu538_coverage': {
            'panel_n_cpgs': 538,
            'median_coverage_per_sample': int(np.median([len(b) for b in per_sample_betas.values()])),
            'min_coverage_fraction': round(min(coverage_fractions), 4),
            'max_coverage_fraction': round(max(coverage_fractions), 4),
        },

        'M1_stats': {
            'h_min_immune': H_MIN_IMMUNE,
            'tumor': {'n': int(len(M1_tumor)), 'mean': float(np.mean(M1_tumor)), 'sd': float(np.std(M1_tumor, ddof=1))},
            'normal': {'n': int(len(M1_normal)), 'mean': float(np.mean(M1_normal)), 'sd': float(np.std(M1_normal, ddof=1))},
            'unpaired': {
                'cohens_d': float(d_unpaired),
                'perm_p_10k': float(p_unpaired),
                'ci_95_lo': float(ci_lo),
                'ci_95_hi': float(ci_hi),
            },
            'paired': {
                'n_pairs': len(complete_pairs),
                'mean_diff': float(np.mean(paired_diffs)),
                'sd_diff': float(np.std(paired_diffs, ddof=1)),
                'cohens_d_paired': float(d_paired),
                'sign_flip_perm_p': float(p_paired),
            },
            'female_only': {
                'n_tumor': int(len(female_tumor)),
                'n_normal': int(len(female_normal)),
                'cohens_d': float(d_female) if not math.isnan(d_female) else None,
            },
        },

        'per_cpg_direction': {
            'n_cpgs_analyzed': total_signed,
            'n_hypermethylated_in_tumor': n_hypermeth_tumor,
            'n_hypomethylated_in_tumor': n_hypometh_tumor,
            'hyper_rate': round(hyper_rate, 4),
            'n_abs_delta_gt_0_05': n_signif,
            'comparison_to_prostate_VAL058': 'VAL-058 prostate fraction hypermeth = 45.1%',
        },

        'pre_registered_outcome': {
            'code': outcome_code,
            'name': outcome_name,
            'tissue_arm_tier': tier,
        },

        'comparison_to_VAL058': {
            'VAL058_prostate_unpaired_d': 0.400,
            'VAL058_prostate_paired_d': 0.497,
            'VAL060_breast_unpaired_d': float(d_unpaired),
            'VAL060_breast_paired_d': float(d_paired),
        },
    }

    results_json_str = json.dumps(results, indent=2, sort_keys=True, default=str)
    results['results_sha256'] = hashlib.sha256(results_json_str.encode()).hexdigest()

    out_path = Path('/home/claude/cookbook_v2.1/breast-epic/VAL060_breast_epic_tcga_brca_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults JSON: {out_path}")
    print(f"Results SHA:  {results['results_sha256'][:16]}...")


if __name__ == '__main__':
    main()
