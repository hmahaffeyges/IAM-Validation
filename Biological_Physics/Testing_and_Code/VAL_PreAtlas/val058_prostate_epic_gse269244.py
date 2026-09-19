#!/usr/bin/env python3
"""
===============================================================================
VAL-058 — prostate-epic Xu-538 validation on GSE269244 prostate tissue
===============================================================================

Pre-registered: VAL_058_PREREG.md (SHA 48abe394ad009020d4bafeeb262439ee02fc910df6d79a96ed56d235a0608316)
Amendment:      VAL_058_PREREG_AMENDMENT.md (SHA b01eac163ea3cea80dcaf97042f996ba925bf190b1dcbab28f799f4a60eb37cf)
Seals:          VAL_058_SEAL.txt (original 2026-04-24 06:50:36 UTC, amendment 06:54:15 UTC)

Question: Does the Xu-538 breast-derived immune panel produce a measurable
          pooled-entropy A-score elevation in prostate tumor tissue vs
          adjacent-normal prostate tissue? If yes at d > 0.3, prostate-epic
          card enters Cookbook at stage_2_only_validated tier, filling the
          clinical gap for patients whose Stage 1 flags and Stage 2 Moss
          NNLS returns prostate_epithelial localization.

Cohort:   GSE269244 — Epigenome-wide DNA methylation in Prostate Cancer in
          African American Men (Berglund / Yamoah / Kresovich et al., 2024)
          - Illumina HumanMethylationEPIC 850K
          - FFPE prostate tissue: 120 tumor + 118 adjacent-normal
          - 121 African-American men, paired by patient
          - Published in 2024 (PMID 39162297)

Panel:    Xu-538 immune panel, FROZEN
          - Panel SHA-256: ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6
          - 538 CpGs from Xu 2020 JNCI breast cancer Sister Study EPIC-Italy

Scoring:  M1 = pooled-entropy A-score on Xu-538 CpGs = mean(H(β)/H_min_immune)
          H_min(immune) = 0.838889 (G-003b MCMC posterior, frozen)

RNG seed: 20260420
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

PREREG_SHA_ORIGINAL = '48abe394ad009020d4bafeeb262439ee02fc910df6d79a96ed56d235a0608316'
PREREG_SHA_AMENDMENT = 'b01eac163ea3cea80dcaf97042f996ba925bf190b1dcbab28f799f4a60eb37cf'
SEAL_TIMESTAMP_ORIGINAL = '2026-04-24 06:50:36 UTC'
SEAL_TIMESTAMP_AMENDMENT = '2026-04-24 06:54:15 UTC'

H_MIN_IMMUNE = 0.838889
XU538_PANEL_SHA = 'ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6'

PANEL_FILE = '/home/claude/val058_data/xu538_panel.json'
BETA_FILE = '/home/claude/val058_data/GSE269244_BetaValues.txt.gz'
BETA_SHA = '7b9fa2825bdd88b0936afba0e19fb0fbcf1bd404a65469d9fb0735829dc88a89'
SAMPLE_MAP_FILE = '/home/claude/val058_data/gse269244_sample_map.json'

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
    return np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else float('nan')


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
    Amended decision matrix (M1 only, no Moss per-CpG):
      O1: d_unpaired > 0.3 AND d_paired > 0.3 → VALIDATED (stage_2_only_validated tier)
      O2: d < 0.3 but directional preservation > 4.5/7 threshold (binomial p<0.05 at n≈538) → directional-only
      O3: null (card not deployed)
      O4: unexpected
    """
    if d_unpaired > 0.3 and d_paired > 0.3:
        return 'O1', 'Xu-538_PROSTATE_TISSUE_VALIDATED', 'stage_2_only_validated'
    
    # Binomial threshold for direction preservation at n=538 CpGs, p<0.05 two-sided
    # Expected random: 269/538, SD ≈ sqrt(538*0.5*0.5) ≈ 11.6, so p<0.05 threshold ≈ 269 + 1.96*11.6 ≈ 292
    direction_threshold = 292
    if d_unpaired < 0.3 and direction_preserved >= direction_threshold:
        return 'O2', 'DIRECTIONAL_ONLY_VALIDATION', 'stage_2_exploratory'
    
    if abs(d_unpaired) < 0.3 and abs(d_paired) < 0.3 and direction_preserved < direction_threshold:
        return 'O3', 'NULL_STAGE_2_NOT_VALIDATED', 'null_documented'
    
    return 'O4', 'UNEXPECTED', 'deferred'


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("="*78)
    print("VAL-058 — prostate-epic Xu-538 validation on GSE269244")
    print("="*78)
    print(f"Prereg SHA (original):  {PREREG_SHA_ORIGINAL[:16]}... (sealed {SEAL_TIMESTAMP_ORIGINAL})")
    print(f"Prereg SHA (amendment): {PREREG_SHA_AMENDMENT[:16]}... (sealed {SEAL_TIMESTAMP_AMENDMENT})")
    print(f"Panel SHA:              {XU538_PANEL_SHA[:16]}...")

    # Verify β matrix SHA
    print("\n[audit] Verifying β matrix file SHA...")
    with open(BETA_FILE, 'rb') as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    assert actual == BETA_SHA, f"β matrix SHA mismatch! Expected {BETA_SHA}, got {actual}"
    print(f"  β matrix SHA verified: {actual[:16]}...")

    # Load panel
    with open(PANEL_FILE) as f:
        panel = json.load(f)
    panel_cpgs = set(panel['cpgs'])
    print(f"  Xu-538 panel loaded: {len(panel_cpgs)} CpGs")

    # Verify panel SHA
    with open(PANEL_FILE, 'rb') as f:
        panel_actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert panel_actual_sha == XU538_PANEL_SHA
    print(f"  Panel SHA verified: {panel_actual_sha[:16]}...")

    # Load sample map
    with open(SAMPLE_MAP_FILE) as f:
        sample_map = json.load(f)
    # Build sentrix_id → sample metadata
    sentrix_to_meta = {e['sentrix_id']: e for e in sample_map}
    print(f"  Sample map loaded: {len(sample_map)} samples")

    # Stream β matrix, pull only target CpGs
    print(f"\n[extract] Streaming {BETA_FILE} for {len(panel_cpgs)} target CpGs...")
    sample_order = None
    betas = {}  # cpg → array of β values aligned with sample_order
    rows_scanned = 0

    with gzip.open(BETA_FILE, 'rt') as f:
        header = f.readline().rstrip('\n').split('\t')
        sample_order = header[1:]  # first column is 'Id'
        print(f"  Sample columns: {len(sample_order)}  first 3: {sample_order[:3]}")

        for line in f:
            rows_scanned += 1
            tab_idx = line.find('\t')
            if tab_idx == -1:
                continue
            cpg = line[:tab_idx].strip('"')
            if cpg not in panel_cpgs:
                continue
            parts = line.rstrip('\n').split('\t')
            vals = []
            for p in parts[1:]:
                p = p.strip().strip('"')
                if p == '' or p.lower() == 'na' or p.lower() == 'nan':
                    vals.append(np.nan)
                else:
                    try:
                        vals.append(float(p))
                    except ValueError:
                        vals.append(np.nan)
            betas[cpg] = np.array(vals, dtype=float)
            if rows_scanned % 100000 == 0:
                print(f"  ...{rows_scanned:,} rows scanned, {len(betas)} CpGs found")
            if len(betas) == len(panel_cpgs):
                break

    print(f"  Total rows scanned: {rows_scanned:,}")
    print(f"  Xu-538 CpGs found on EPIC: {len(betas)}/538")
    epic_coverage = len(betas) / 538

    # Map sample_order (sentrix IDs) to metadata
    print(f"\n[map] Matching sentrix IDs to sample metadata...")
    n_samples = len(sample_order)
    sample_types = []
    gleason_scores = []
    geo_ids = []
    titles = []
    mismatches = 0
    for s in sample_order:
        if s in sentrix_to_meta:
            m = sentrix_to_meta[s]
            sample_types.append(m['sample_type'])
            gleason_scores.append(m['gleason'])
            geo_ids.append(m['geo_id'])
            titles.append(m['title'])
        else:
            sample_types.append('UNKNOWN')
            gleason_scores.append('NA')
            geo_ids.append('?')
            titles.append('?')
            mismatches += 1
    print(f"  Matched: {n_samples - mismatches}/{n_samples}")
    if mismatches > 0:
        print(f"  ⚠ {mismatches} unmatched samples")

    type_counts = Counter(sample_types)
    print(f"  Sample composition: {dict(type_counts)}")

    # Compute M1 (pooled-entropy A-score on Xu-538) per sample
    print(f"\n[compute] Computing M1 per sample (pooled-entropy Xu-538 A-score)...")
    M1_per_sample = np.full(n_samples, np.nan)
    n_cpgs_per_sample = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        h_vals = []
        for cpg, arr in betas.items():
            b = arr[i]
            if not np.isnan(b):
                h_vals.append(H(b) / H_MIN_IMMUNE)
        if len(h_vals) >= 300:  # require ≥300 CpGs for valid score (~60% coverage)
            M1_per_sample[i] = np.mean(h_vals)
            n_cpgs_per_sample[i] = len(h_vals)

    n_valid = int(np.sum(~np.isnan(M1_per_sample)))
    print(f"  Valid M1 scores: {n_valid}/{n_samples}")
    print(f"  Median CpGs per valid sample: {int(np.median(n_cpgs_per_sample[n_cpgs_per_sample > 0]))}")

    # Partition by sample type
    types_arr = np.array(sample_types)
    M1_tumor = M1_per_sample[(types_arr == 'Tumor') & ~np.isnan(M1_per_sample)]
    M1_normal = M1_per_sample[(types_arr == 'Normal') & ~np.isnan(M1_per_sample)]
    print(f"\n  M1_tumor:  n={len(M1_tumor)}  mean={np.mean(M1_tumor):.5f}  sd={np.std(M1_tumor, ddof=1):.5f}")
    print(f"  M1_normal: n={len(M1_normal)}  mean={np.mean(M1_normal):.5f}  sd={np.std(M1_normal, ddof=1):.5f}")

    # Unpaired Cohen's d
    print(f"\n[stats] Unpaired Cohen's d (tumor vs adjacent-normal)...")
    rng = np.random.default_rng(RNG_SEED)
    d_unpaired = cohens_d(M1_tumor, M1_normal)
    p_unpaired = permutation_p(M1_tumor, M1_normal, n_perms=10000, rng=rng)
    ci_lo, ci_hi = bootstrap_ci_d(M1_tumor, M1_normal, n_boot=10000)
    print(f"  d(unpaired) = {d_unpaired:+.4f}  [95% CI {ci_lo:+.3f}, {ci_hi:+.3f}]  p_perm = {p_unpaired:.4f}")

    # Paired analysis using title prefix (P-XXXX)
    print(f"\n[stats] Paired tumor-vs-normal difference (by patient P-XXXX prefix)...")
    # Extract patient ID from title (P-XXXX-T/N → P-XXXX)
    patient_ids = [t.rsplit('-', 1)[0] if '-' in t else '?' for t in titles]
    # Map patient_id to M1_tumor and M1_normal values
    pair_data = {}
    for i, pid in enumerate(patient_ids):
        if pid == '?' or np.isnan(M1_per_sample[i]):
            continue
        if pid not in pair_data:
            pair_data[pid] = {}
        pair_data[pid][sample_types[i]] = M1_per_sample[i]

    complete_pairs = [pid for pid, d in pair_data.items() if 'Tumor' in d and 'Normal' in d]
    paired_diffs = np.array([pair_data[pid]['Tumor'] - pair_data[pid]['Normal'] for pid in complete_pairs])
    print(f"  Complete pairs: {len(complete_pairs)}")
    print(f"  Paired differences:  mean={np.mean(paired_diffs):+.5f}  sd={np.std(paired_diffs, ddof=1):.5f}")
    d_paired = paired_cohens_d(paired_diffs)
    # Paired permutation: flip signs
    observed_mean = np.mean(paired_diffs)
    n_paired_perms = 10000
    hits = 0
    for _ in range(n_paired_perms):
        signs = rng.choice([-1, 1], size=len(paired_diffs))
        if abs(np.mean(paired_diffs * signs)) >= abs(observed_mean):
            hits += 1
    p_paired = (hits + 1) / (n_paired_perms + 1)
    print(f"  d(paired) = {d_paired:+.4f}  p(sign-flip perm) = {p_paired:.4f}")

    # Per-CpG direction check
    print(f"\n[stats] Per-CpG Δβ direction analysis...")
    tumor_idx = np.where((types_arr == 'Tumor') & ~np.isnan(M1_per_sample))[0]
    normal_idx = np.where((types_arr == 'Normal') & ~np.isnan(M1_per_sample))[0]

    per_cpg_direction = {}
    n_hypermeth_in_tumor = 0
    n_hypometh_in_tumor = 0
    significant_cpgs = []
    for cpg, arr in betas.items():
        t_vals = arr[tumor_idx]; t_vals = t_vals[~np.isnan(t_vals)]
        n_vals = arr[normal_idx]; n_vals = n_vals[~np.isnan(n_vals)]
        if len(t_vals) < 10 or len(n_vals) < 10:
            continue
        mean_t = float(np.mean(t_vals))
        mean_n = float(np.mean(n_vals))
        delta = mean_t - mean_n
        if delta > 0:
            n_hypermeth_in_tumor += 1
        elif delta < 0:
            n_hypometh_in_tumor += 1
        per_cpg_direction[cpg] = {
            'mean_tumor': mean_t, 'mean_normal': mean_n,
            'delta': round(delta, 5),
            'direction': '+' if delta > 0 else ('-' if delta < 0 else '0'),
        }
        if abs(delta) > 0.05:
            significant_cpgs.append((cpg, delta))

    total_signed = n_hypermeth_in_tumor + n_hypometh_in_tumor
    print(f"  CpGs analyzed: {total_signed}/{len(betas)}")
    print(f"  Hypermethylated in tumor: {n_hypermeth_in_tumor} ({100*n_hypermeth_in_tumor/total_signed:.1f}%)")
    print(f"  Hypomethylated in tumor: {n_hypometh_in_tumor} ({100*n_hypometh_in_tumor/total_signed:.1f}%)")
    print(f"  |Δβ| > 0.05 (meaningful shift): {len(significant_cpgs)} CpGs")
    # Publication (Berglund 2024) reports "overall trend of hypermethylation in prostate tumors"
    # Direction preservation rate vs this published direction (hypermeth in tumor = +1)
    # Each CpG's "preserved" status requires a known-direction prior per CpG.
    # Use the binary "published overall hypermethylation trend" as the prior.
    direction_preserved = n_hypermeth_in_tumor
    direction_preserve_rate = direction_preserved / total_signed
    print(f"  Direction preserved vs published hypermeth trend: {direction_preserved}/{total_signed} ({100*direction_preserve_rate:.1f}%)")

    # Outcome assignment
    outcome_code, outcome_name, tier = assign_outcome(
        d_unpaired, d_paired, direction_preserved, total_signed
    )

    print(f"\n{'='*78}")
    print(f"OUTCOME: {outcome_code} — {outcome_name}")
    print(f"Card tier: {tier}")
    print(f"{'='*78}")
    print(f"  d(unpaired) = {d_unpaired:+.4f}  (threshold 0.3 for O1)")
    print(f"  d(paired)   = {d_paired:+.4f}  (threshold 0.3 for O1)")
    print(f"  direction preserved: {direction_preserved}/{total_signed} ({100*direction_preserve_rate:.1f}%)")

    runtime_s = time.time() - t0
    print(f"\nRuntime: {runtime_s:.1f}s")

    # Build results
    results = {
        'val_id': 'VAL-058',
        'val_type': 'stage_2_only_tissue_validation',
        'card_target': 'prostate-epic v0.1',
        'prereg_sha_original': PREREG_SHA_ORIGINAL,
        'prereg_sha_amendment': PREREG_SHA_AMENDMENT,
        'seal_timestamp_original': SEAL_TIMESTAMP_ORIGINAL,
        'seal_timestamp_amendment': SEAL_TIMESTAMP_AMENDMENT,
        'beta_matrix_sha256': BETA_SHA,
        'panel_sha256': XU538_PANEL_SHA,
        'run_date': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'rng_seed': RNG_SEED,
        'runtime_seconds': round(runtime_s, 2),

        'cohort': {
            'geo_accession': 'GSE269244',
            'platform': 'Illumina HumanMethylationEPIC 850K',
            'specimen': 'FFPE prostate tissue',
            'n_total': n_samples,
            'composition': dict(type_counts),
            'population': 'African American men',
            'source_paper': 'Berglund/Yamoah/Kresovich et al 2024, PMID 39162297',
        },

        'panel': {
            'panel_id': 'Xu2020_breast_cancer_replicated_full',
            'panel_sha256': XU538_PANEL_SHA,
            'n_cpgs_in_panel': 538,
            'n_cpgs_on_epic': len(betas),
            'epic_coverage': round(epic_coverage, 4),
        },

        'stage_1_metric_M1_pooled_Ximmune': {
            'h_min_immune': H_MIN_IMMUNE,
            'tumor': {
                'n': int(len(M1_tumor)),
                'mean': float(np.mean(M1_tumor)),
                'sd': float(np.std(M1_tumor, ddof=1)),
            },
            'adjacent_normal': {
                'n': int(len(M1_normal)),
                'mean': float(np.mean(M1_normal)),
                'sd': float(np.std(M1_normal, ddof=1)),
            },
            'unpaired': {
                'cohens_d': float(d_unpaired),
                'perm_p': float(p_unpaired),
                'ci_95_lo': float(ci_lo),
                'ci_95_hi': float(ci_hi),
            },
            'paired': {
                'n_complete_pairs': len(complete_pairs),
                'mean_paired_diff': float(np.mean(paired_diffs)),
                'sd_paired_diff': float(np.std(paired_diffs, ddof=1)),
                'cohens_d_paired': float(d_paired),
                'sign_flip_perm_p': float(p_paired),
            },
        },

        'per_cpg_direction': {
            'n_cpgs_analyzed': total_signed,
            'n_hypermethylated_in_tumor': n_hypermeth_in_tumor,
            'n_hypomethylated_in_tumor': n_hypometh_in_tumor,
            'fraction_hypermethylated': round(direction_preserve_rate, 4),
            'n_significant_abs_delta_gt_0_05': len(significant_cpgs),
            'published_expectation': 'Berglund 2024 reports overall hypermethylation trend in prostate tumors',
            'preservation_threshold_binomial_0_05': 292,
        },

        'pre_registered_outcome': {
            'outcome_code': outcome_code,
            'outcome_name': outcome_name,
            'card_tier': tier,
        },

        'sex_stratification': 'N/A — all samples are male (prostate tissue)',
        'age_80cell_anchor': 'N/A — tissue data, 80-cell immune baseline is blood-derived (per CCL-004)',
    }

    results_json_str = json.dumps(results, indent=2, sort_keys=True, default=str)
    results['results_sha256'] = hashlib.sha256(results_json_str.encode()).hexdigest()

    out_path = Path('/home/claude/cookbook_v2.1/prostate-epic/VAL058_prostate_epic_gse269244_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults JSON: {out_path}")
    print(f"Results SHA:  {results['results_sha256'][:16]}...")


if __name__ == '__main__':
    main()
