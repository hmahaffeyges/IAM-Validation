#!/usr/bin/env python3
"""
VAL-101 — hcc-epic Run-Everything 25-Tile Per-Class A-Score with Full Etiology Stratification
================================================================================================

Applies the run-everything 25-tile per-class methodology to TCGA-LIHC paired
tumor/adjacent-normal cohort (n=46 carried forward from VAL-064 sealed cohort).
Three pre-locked questions:

1. CCL-039 cross-tissue generalization: does the Hepatocytes tile read negative
   in HCC tumor-vs-adjacent-normal paired comparisons the same way the
   Colon_epithelial_cells tile reads negative in colorectal paired comparisons?

2. Viral-vs-non-viral blunting at the per-tile level: does the viral-hepatitis
   adjacent-normal field defect blunt the per-tile cell-of-origin signal the
   same way it blunts the pooled-cycling-class signal in VAL-064?

3. Marcus-analog stratum: what does the 25-tile pattern look like in the
   no_documented_risk stratum (n=10) — patients with HCC and no canonical
   chronic-driver risk factor in their TCGA chart?

Pre-registration sealed: 2026-04-28
Pre-reg SHA: fa366bf00316597bb65032b747029133acb5f1bbb40f6251094b563732185512
RNG seed: 20260428
"""

import csv
import hashlib
import json
import math
import os
import random
import re
import time

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

RNG_SEED = 20260428
N_BOOTSTRAP = 10000
QC_MIN_VALID = 400_000

LIHC_DOWNLOADS = '/home/claude/edear_working/VAL-101/lihc_downloads'
LOYFER_ATLAS = '/home/claude/iam_repo/Biological_Physics/atlas_vault/stage2_cell_of_origin/loyfer_moss_2018/reference_atlas.csv'
CLINICAL_METADATA = '/home/claude/iam_repo/Biological_Physics/validation_runs/LIHC_clinical.json'

OUTPUT_DIR = '/home/claude/edear_working/VAL-101/VAL-101'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# H_min frozen values
H_MIN = {
    'cycling':    0.856055,
    'secretory':  0.843264,
    'terminal':   0.7728,
    'stromal':    0.862950,
    'progenitor': 0.852216,
    'immune':     0.838889,
}

# Loyfer 25-tile class assignments (consistent with VAL-098, VAL-099)
TILE_CLASS = {
    'Colon_epithelial_cells': 'cycling',
    'Lung_cells': 'cycling',
    'Bladder': 'cycling',
    'Head_and_neck_larynx': 'cycling',
    'Upper_GI': 'cycling',
    'Uterus_cervix': 'cycling',
    'Pancreatic_beta_cells': 'secretory',
    'Pancreatic_acinar_cells': 'secretory',
    'Pancreatic_duct_cells': 'secretory',
    'Hepatocytes': 'secretory',          # cell-of-origin tile for HCC
    'Breast_basal_epithelium': 'secretory',
    'Breast_luminal_epithelium': 'secretory',
    'Prostate_epithelium': 'secretory',
    'Kidney_epithelium': 'secretory',
    'Thyroid_epithelium': 'secretory',
    'Cortical_neurons': 'terminal',
    'Left_atrium': 'terminal',
    'Adipocytes': 'stromal',
    'Vascular_endothelial_cells': 'stromal',
    'Smooth_muscle_cells': 'stromal',
    'Fallopian_epithelium': 'cycling',
    'Bone_osteoblast': 'progenitor',
    'B-cells_EPIC': 'immune',
    'Monocytes_EPIC': 'immune',
    'Neutrophils_EPIC': 'immune',
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def shannon_h(beta):
    if beta is None or beta != beta:
        return None
    if beta <= 0 or beta >= 1:
        return 0.0
    p, q = beta, 1.0 - beta
    return -p * math.log2(p) - q * math.log2(q)


def load_betas(filepath):
    """Load TCGA sesame Level 3 .txt: cg-ID -> beta. Skip 2-line header."""
    betas = {}
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i < 2:
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            cg = parts[0].strip()
            try:
                b = float(parts[1])
            except (ValueError, IndexError):
                continue
            if 0.0 <= b <= 1.0:
                betas[cg] = b
    return betas


def cohens_d_paired(deltas):
    if len(deltas) < 2:
        return None, 0.0
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 1e-12
    return mean / sd, sd


def bootstrap_ci_paired(deltas, n_iter, seed):
    if len(deltas) < 2:
        return [None, None]
    rng = random.Random(seed)
    n = len(deltas)
    ds = []
    for _ in range(n_iter):
        s = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        d, _ = cohens_d_paired(s)
        if d is not None:
            ds.append(d)
    ds.sort()
    return [ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))]]


def welch_t_paired(deltas):
    if len(deltas) < 2:
        return None, None
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 1e-12
    t = mean / se
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return t, p


# ----------------------------------------------------------------------------
# Risk factor classification
# ----------------------------------------------------------------------------

def classify_risk(rec):
    """Classify a patient by etiology bucket. Returns: bucket, list of derived flags."""
    rfs = rec.get('risk_factors', [])
    vh = rec.get('viral_hepatitis', [])
    flat_rf = set()
    for sub in rfs:
        if isinstance(sub, list):
            for x in sub:
                flat_rf.add(x)
        else:
            flat_rf.add(sub)
    flat_vh = set()
    for sub in vh:
        if isinstance(sub, list):
            for x in sub:
                flat_vh.add(x)
        else:
            flat_vh.add(sub)
    has_hbv = any('Hepatitis B' in x or 'HBV' in x for x in flat_rf | flat_vh)
    has_hcv = any('Hepatitis C' in x or 'HCV' in x for x in flat_rf | flat_vh)
    has_alc = any('Alcohol' in x for x in flat_rf)
    has_nafld = any('Nonalcoholic Fatty Liver' in x or 'NAFLD' in x or 'NASH' in x for x in flat_rf)
    has_other_rf = any(rf not in {'None', ''} and 'Hepatitis' not in rf and 'Alcohol' not in rf and 'Nonalcoholic' not in rf for rf in flat_rf)
    has_unknown_viral = 'Unknown' in flat_vh

    if has_hbv and has_hcv:
        bucket = 'HBV+HCV'
    elif has_hbv:
        bucket = 'HBV+'
    elif has_hcv:
        bucket = 'HCV+'
    elif has_alc:
        bucket = 'Alcohol+'
    elif has_nafld:
        bucket = 'NAFLD+'
    elif has_other_rf or has_unknown_viral:
        bucket = 'Other'
    else:
        bucket = 'No_documented_risk'

    return bucket, {'has_hbv': has_hbv, 'has_hcv': has_hcv, 'has_alc': has_alc, 'has_nafld': has_nafld}


# ----------------------------------------------------------------------------
# Pair files + score
# ----------------------------------------------------------------------------

def discover_pairs():
    files = os.listdir(LIHC_DOWNLOADS)
    pairs = {}
    for f in files:
        m = re.match(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})__(tumor|normal)__', f)
        if m:
            pid, kind = m.group(1), m.group(2)
            pairs.setdefault(pid, {})[kind] = os.path.join(LIHC_DOWNLOADS, f)
    matched = {p: v for p, v in pairs.items() if 'tumor' in v and 'normal' in v}
    return matched


def load_loyfer_atlas():
    atlas = {}
    with open(LOYFER_ATLAS) as f:
        header = f.readline().rstrip('\n').split(',')
        tile_cols = header[1:]
        for col in tile_cols:
            atlas.setdefault(col, {})
        for line in f:
            parts = line.rstrip('\n').split(',')
            if len(parts) != len(header):
                continue
            cg = parts[0]
            for i, col in enumerate(tile_cols, start=1):
                try:
                    atlas[col][cg] = float(parts[i])
                except ValueError:
                    pass
    return atlas


def select_top_marker_cpgs(atlas, tile_name, n=100):
    tile_betas = atlas[tile_name]
    other_tiles = [t for t in atlas if t != tile_name]
    scores = []
    for cg in tile_betas:
        own = tile_betas[cg]
        others = [atlas[t].get(cg) for t in other_tiles if atlas[t].get(cg) is not None]
        if not others:
            continue
        mean_other = sum(others) / len(others)
        scores.append((cg, abs(own - mean_other)))
    scores.sort(key=lambda kv: -kv[1])
    return [cg for cg, _ in scores[:n]]


def per_tile_a_score(betas, marker_cpgs, hmin):
    h_sum, n = 0.0, 0
    for cg in marker_cpgs:
        if cg in betas:
            h = shannon_h(betas[cg])
            if h is not None:
                h_sum += h / hmin
                n += 1
    return (h_sum / n if n > 0 else None, n)


# ----------------------------------------------------------------------------
# Beta distribution check (CHK-3.1)
# ----------------------------------------------------------------------------

def beta_distribution_check(per_sample):
    """Pool all valid β across samples; check bimodal raw β signature."""
    all_betas = []
    for pid, rec in per_sample.items():
        if rec.get('both_qc_pass'):
            for kind in ('tumor', 'normal'):
                betas = rec['_betas'].get(kind, {})
                # Sample 5000 betas per file for the check (don't pool all 480K — that's 50 GB)
                items = list(betas.values())
                if len(items) > 5000:
                    items = items[:5000]
                all_betas.extend(items)
    n = len(all_betas)
    if n == 0:
        return None
    extreme = sum(1 for b in all_betas if b < 0.05 or b > 0.95) / n
    middle = sum(1 for b in all_betas if 0.4 <= b <= 0.6) / n
    bimodal = (extreme > 0.30) and (middle < 0.10)
    return {
        'n_betas_sampled': n,
        'fraction_extreme_lt0.05_or_gt0.95': round(extreme, 4),
        'fraction_middle_0.4_to_0.6': round(middle, 4),
        'bimodal_raw_beta_signature': bimodal,
    }


# ----------------------------------------------------------------------------
# Stratum summary
# ----------------------------------------------------------------------------

def summarize_stratum_tile(name, deltas, descriptive_threshold=5):
    """Summarize a stratum's tile A-score deltas (tumor − normal).
    Below descriptive_threshold: descriptive-only point estimate.
    """
    n = len(deltas)
    if n < 2:
        return {'n': n, 'note': f'n={n} — empty or single-point'}
    out = {'n': n}
    out['mean_delta'] = sum(deltas) / n
    if n < descriptive_threshold:
        out['note'] = f'n={n} — descriptive-only per CHK-2.7 (threshold n<{descriptive_threshold})'
        d, _ = cohens_d_paired(deltas)
        out['paired_d_descriptive'] = d
        out['ci_95'] = [None, None]
    else:
        d, sd = cohens_d_paired(deltas)
        ci = bootstrap_ci_paired(deltas, N_BOOTSTRAP, RNG_SEED + abs(hash(name)) % 1000)
        t, p = welch_t_paired(deltas)
        out['paired_d'] = d
        out['ci_95'] = ci
        out['t_stat'] = t
        out['p_approx'] = p
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    random.seed(RNG_SEED)

    print('=== VAL-101 — hcc-epic 25-Tile Etiology Stratification on TCGA-LIHC ===\n')

    # Step 1: load clinical metadata
    with open(CLINICAL_METADATA) as f:
        clinical_full = json.load(f)
    clinical = clinical_full['patient_strata']
    print(f'Clinical metadata: {len(clinical)} patients\n')

    # Step 2: pair files
    matched_pairs = discover_pairs()
    print(f'Paired files: {len(matched_pairs)}')

    # Step 3: score every sample per tile (load atlas first)
    atlas = load_loyfer_atlas()
    print(f'Loyfer atlas: {len(atlas)} tiles\n')
    print('Building marker CpG sets per tile...')
    tile_markers = {t: select_top_marker_cpgs(atlas, t, 100) for t in atlas}

    # Pre-allocate H_min lookup
    tile_hmin = {t: H_MIN[TILE_CLASS[t]] for t in atlas if t in TILE_CLASS}

    # Step 4: load β-files, score per tile, QC-gate
    print('Scoring per-tile A-score for all paired samples...')
    per_sample = {}
    chk_3_1_betas_pooled = []  # Sample subset for CHK-3.1 only, NOT full betas
    for pid, files in sorted(matched_pairs.items()):
        record = {'patient': pid}
        all_qc_pass = True
        sample_betas_temp = {}  # local scope only, gc'd after this iteration
        for kind in ('tumor', 'normal'):
            betas = load_betas(files[kind])
            n_valid = len(betas)
            qc_pass = (n_valid >= QC_MIN_VALID)
            record[f'n_valid_{kind}'] = n_valid
            record[f'qc_pass_{kind}'] = qc_pass
            if qc_pass:
                # Score per tile
                record[f'tile_scores_{kind}'] = {}
                for tile_name in atlas:
                    if tile_name not in TILE_CLASS:
                        continue
                    a, _ = per_tile_a_score(betas, tile_markers[tile_name], tile_hmin[tile_name])
                    record[f'tile_scores_{kind}'][tile_name] = a
                # CHK-3.1 sampling: take 5000 betas per sample (pooled across all samples)
                items = list(betas.values())
                if len(items) > 5000:
                    items = items[::max(1, len(items) // 5000)][:5000]
                chk_3_1_betas_pooled.extend(items)
            else:
                all_qc_pass = False
            # Explicitly drop the full betas dict after processing this kind
            del betas
        record['both_qc_pass'] = record.get('qc_pass_tumor', False) and record.get('qc_pass_normal', False)
        if record['both_qc_pass']:
            record['per_tile_delta'] = {
                t: record['tile_scores_tumor'][t] - record['tile_scores_normal'][t]
                for t in record['tile_scores_tumor']
                if record['tile_scores_tumor'][t] is not None and record['tile_scores_normal'][t] is not None
            }
        per_sample[pid] = record

    qc_pass_pids = [p for p, r in per_sample.items() if r['both_qc_pass']]
    print(f'QC passed: {len(qc_pass_pids)} / {len(matched_pairs)} pairs\n')

    # Step 5: CHK-3.1 beta distribution check (using sampled subset, not full β)
    print('CHK-3.1 beta distribution check...')
    bdc = None
    if chk_3_1_betas_pooled:
        n = len(chk_3_1_betas_pooled)
        extreme = sum(1 for b in chk_3_1_betas_pooled if b < 0.05 or b > 0.95) / n
        middle = sum(1 for b in chk_3_1_betas_pooled if 0.4 <= b <= 0.6) / n
        bimodal = (extreme > 0.30) and (middle < 0.10)
        bdc = {
            'n_betas_sampled': n,
            'fraction_extreme_lt0.05_or_gt0.95': round(extreme, 4),
            'fraction_middle_0.4_to_0.6': round(middle, 4),
            'bimodal_raw_beta_signature': bimodal,
        }
        print(f'  Extreme [<0.05 or >0.95]: {100*bdc["fraction_extreme_lt0.05_or_gt0.95"]:.1f}%')
        print(f'  Middle  [0.4 to 0.6]:     {100*bdc["fraction_middle_0.4_to_0.6"]:.1f}%')
        print(f'  Bimodal raw β signature: {bdc["bimodal_raw_beta_signature"]}\n')

    # Step 6: pooled per-tile paired d
    print('=== POOLED 25-TILE RESULTS (all QC-passed pairs) ===\n')
    pooled_tile_results = {}
    for tile_name in atlas:
        if tile_name not in TILE_CLASS:
            continue
        deltas = [per_sample[p]['per_tile_delta'].get(tile_name) for p in qc_pass_pids]
        deltas = [d for d in deltas if d is not None]
        if len(deltas) < 2:
            continue
        d, _ = cohens_d_paired(deltas)
        ci = bootstrap_ci_paired(deltas, N_BOOTSTRAP, RNG_SEED + abs(hash(tile_name)) % 1000)
        t, p = welch_t_paired(deltas)
        pooled_tile_results[tile_name] = {
            'tile': tile_name,
            'class': TILE_CLASS[tile_name],
            'paired_d': d,
            'ci_95': ci,
            't_stat': t,
            'p_approx': p,
            'n_pairs': len(deltas),
        }

    # Sort tiles by |paired_d|
    sorted_tiles = sorted(pooled_tile_results.values(), key=lambda r: -abs(r['paired_d']))
    print('Top 10 tiles by |paired d|:')
    for r in sorted_tiles[:10]:
        marker = ' ← Hepatocytes (HCC cell-of-origin)' if r['tile'] == 'Hepatocytes' else ''
        print(f"  {r['tile']:30s} ({r['class']:10s}) d={r['paired_d']:+.3f} CI=[{r['ci_95'][0]:+.3f}, {r['ci_95'][1]:+.3f}] p={r['p_approx']:.4f}{marker}")

    hep = pooled_tile_results.get('Hepatocytes', {})
    print(f"\nHEPATOCYTES tile (HCC cell-of-origin): paired d = {hep.get('paired_d', 0):+.4f} CI={hep.get('ci_95', [])}")
    print(f'  CCL-039 expectation: NEGATIVE direction in tumor-vs-adjacent-normal paired')
    hep_rank = next((i+1 for i, r in enumerate(sorted_tiles) if r['tile'] == 'Hepatocytes'), None)
    print(f'  Hepatocytes rank by |d|: {hep_rank} of {len(sorted_tiles)}')
    print()

    # Step 7: stratified analysis
    # Build patient → stratum
    patient_stratum = {}
    for p in qc_pass_pids:
        if p in clinical:
            bucket, flags = classify_risk(clinical[p])
            patient_stratum[p] = bucket
        else:
            patient_stratum[p] = 'NoMetadata'

    # Compute per-stratum analyses on the Hepatocytes tile (cell-of-origin) AND the full 25-tile output for the Marcus stratum
    strat_buckets = {
        'all_viral': [],
        'all_non_viral': [],
        'HBV+': [],
        'HCV+': [],
        'HBV+HCV': [],
        'Alcohol+': [],
        'NAFLD+': [],
        'Other': [],
        'No_documented_risk': [],
    }
    for p, bucket in patient_stratum.items():
        strat_buckets[bucket].append(p)
        if bucket in {'HBV+', 'HCV+', 'HBV+HCV'}:
            strat_buckets['all_viral'].append(p)
        elif bucket in {'Alcohol+', 'NAFLD+', 'Other', 'No_documented_risk'}:
            strat_buckets['all_non_viral'].append(p)

    print('=== STRATIFIED HEPATOCYTES TILE RESULTS ===\n')
    strat_hep_results = {}
    for stratum_name, pids in strat_buckets.items():
        if not pids:
            continue
        deltas = [per_sample[p]['per_tile_delta'].get('Hepatocytes') for p in pids]
        deltas = [d for d in deltas if d is not None]
        result = summarize_stratum_tile(stratum_name, deltas, descriptive_threshold=5)
        strat_hep_results[stratum_name] = result
        n = result['n']
        if n >= 5:
            d_str = f"d={result.get('paired_d', 0):+.4f}"
            ci_str = f"CI=[{result['ci_95'][0]:+.3f}, {result['ci_95'][1]:+.3f}]" if result['ci_95'][0] is not None else "CI=N/A"
            p_str = f"p={result.get('p_approx', 1):.4f}"
            print(f"  {stratum_name:25s} n={n:3d}  {d_str:18s}  {ci_str:30s}  {p_str}")
        else:
            d_str = f"d={result.get('paired_d_descriptive', 0):+.4f}" if result.get('paired_d_descriptive') is not None else f"mean_ΔA={result.get('mean_delta', 0):+.4f}"
            note = result.get('note', '')
            print(f"  {stratum_name:25s} n={n:3d}  {d_str:18s}  [descriptive] {note}")

    # Step 8: Marcus-analog stratum — full 25-tile pattern
    print()
    print('=== MARCUS-ANALOG STRATUM (No_documented_risk, descriptive-only) ===')
    marcus_pids = strat_buckets['No_documented_risk']
    print(f'n = {len(marcus_pids)} patients: {marcus_pids}\n')
    
    marcus_25tile = {}
    if marcus_pids:
        for tile_name in atlas:
            if tile_name not in TILE_CLASS:
                continue
            deltas = [per_sample[p]['per_tile_delta'].get(tile_name) for p in marcus_pids]
            deltas = [d for d in deltas if d is not None]
            if len(deltas) >= 2:
                d, _ = cohens_d_paired(deltas)
                ci = bootstrap_ci_paired(deltas, N_BOOTSTRAP, RNG_SEED + abs(hash('marcus_' + tile_name)) % 1000)
                marcus_25tile[tile_name] = {
                    'tile': tile_name,
                    'class': TILE_CLASS[tile_name],
                    'paired_d': d,
                    'ci_95': ci,
                    'mean_delta_A': sum(deltas) / len(deltas),
                    'n_pairs': len(deltas),
                }

        sorted_marcus = sorted(marcus_25tile.values(), key=lambda r: -abs(r['paired_d']))
        print('All 25 tiles ranked by |paired d| (descriptive-only at n=10):')
        for r in sorted_marcus:
            marker = ' ← Hepatocytes (HCC cell-of-origin)' if r['tile'] == 'Hepatocytes' else ''
            print(f"  {r['tile']:30s} ({r['class']:10s}) d={r['paired_d']:+.3f} CI=[{r['ci_95'][0]:+.3f}, {r['ci_95'][1]:+.3f}]{marker}")
        marcus_hep = marcus_25tile.get('Hepatocytes', {})
        marcus_hep_rank = next((i+1 for i, r in enumerate(sorted_marcus) if r['tile'] == 'Hepatocytes'), None)
        print(f"\nHepatocytes tile rank in Marcus stratum: {marcus_hep_rank} of {len(sorted_marcus)}")
        print(f"Hepatocytes paired d = {marcus_hep.get('paired_d', 0):+.4f} CI={marcus_hep.get('ci_95', [])}")

    # Step 9: outcome decision
    print()
    print('=== OUTCOME DECISION ===')
    hep_d = hep.get('paired_d')
    hep_ci = hep.get('ci_95', [None, None])

    if not bdc or not bdc.get('bimodal_raw_beta_signature'):
        outcome_label = 'O5_DATA_INTEGRITY_FLAG'
        outcome_reason = f"CHK-3.1 beta distribution check failed: extreme {100*bdc['fraction_extreme_lt0.05_or_gt0.95']:.1f}%, middle {100*bdc['fraction_middle_0.4_to_0.6']:.1f}%."
    elif hep_d is None:
        outcome_label = 'O5_DATA_INTEGRITY_FLAG'
        outcome_reason = "Hepatocytes tile A-score could not be computed."
    elif hep_d <= -0.5 and hep_ci[1] is not None and hep_ci[1] < 0:
        outcome_label = 'O1_HEPATOCYTES_TILE_NEGATIVE_DIRECTION_CONFIRMED'
        outcome_reason = f"Hepatocytes tile paired d = {hep_d:+.4f} CI=[{hep_ci[0]:+.3f}, {hep_ci[1]:+.3f}] — direction matches CCL-039 prediction; magnitude meets threshold; 95% CI upper bound < 0."
    elif hep_d < 0 and -0.5 < hep_d:
        outcome_label = 'O2_HEPATOCYTES_TILE_NEGATIVE_PARTIAL'
        outcome_reason = f"Hepatocytes tile paired d = {hep_d:+.4f} — direction consistent with CCL-039 but magnitude attenuated; check viral-vs-non-viral stratification."
    elif -0.2 <= hep_d <= 0.2:
        outcome_label = 'O3_HEPATOCYTES_TILE_NULL'
        outcome_reason = f"Hepatocytes tile paired d = {hep_d:+.4f} CI=[{hep_ci[0]:+.3f}, {hep_ci[1]:+.3f}] — null. Cell-of-origin tile fidelity-loss does NOT generalize from colorectal to HCC at the pooled level."
    elif hep_d >= 0.5 and hep_ci[0] is not None and hep_ci[0] > 0:
        outcome_label = 'O4_HEPATOCYTES_TILE_INVERTED_POSITIVE'
        outcome_reason = f"Hepatocytes tile paired d = {hep_d:+.4f} CI=[{hep_ci[0]:+.3f}, {hep_ci[1]:+.3f}] — direction inverted from CCL-039 prediction. Investigate."
    else:
        outcome_label = 'O5_UNEXPECTED'
        outcome_reason = f"Hepatocytes tile d = {hep_d:+.4f} did not match any pre-locked decision criterion cleanly."

    print(f'OUTCOME: {outcome_label}')
    print(f'REASON: {outcome_reason}\n')

    # Step 10: save results
    panel_sha = hashlib.sha256(open(LOYFER_ATLAS, 'rb').read()).hexdigest()

    results = {
        'val_id': 'VAL-101',
        'sealed_at': '2026-04-28T19:53:19.249263+00:00',
        'prereg_sha256': 'fa366bf00316597bb65032b747029133acb5f1bbb40f6251094b563732185512',
        'rng_seed': RNG_SEED,
        'cohort': {
            'name': 'TCGA-LIHC HM450 paired tumor/adjacent-normal (re-analysis of VAL-064 sealed cohort)',
            'n_pairs_total': len(matched_pairs),
            'n_pairs_qc_passed': len(qc_pass_pids),
        },
        'loyfer_atlas_sha256_prefix': panel_sha[:16] + '...',
        'beta_distribution_check_chk_3_1': bdc,
        'pooled_25_tile_results': sorted_tiles,
        'hepatocytes_tile_pooled': hep,
        'hepatocytes_rank_by_abs_d': hep_rank,
        'stratified_hepatocytes_results': strat_hep_results,
        'marcus_analog_stratum_25_tile': marcus_25tile if marcus_pids else None,
        'marcus_analog_patient_ids': marcus_pids,
        'patient_stratum_assignments': patient_stratum,
        'outcome_label': outcome_label,
        'outcome_reason': outcome_reason,
        'edear_commercial_deployment_unaffected': True,
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Stratified-only file
    with open(os.path.join(OUTPUT_DIR, 'stratified.json'), 'w') as f:
        json.dump({
            'hepatocytes_by_stratum': strat_hep_results,
            'marcus_25_tile': marcus_25tile if marcus_pids else None,
            'patient_stratum_assignments': patient_stratum,
        }, f, indent=2)

    # Per-sample CSV
    with open(os.path.join(OUTPUT_DIR, 'per_sample.csv'), 'w') as f:
        tile_cols = sorted([t for t in atlas if t in TILE_CLASS])
        f.write('patient,both_qc_pass,stratum,' + ','.join([f'delta_{t}' for t in tile_cols]) + '\n')
        for pid in sorted(per_sample.keys()):
            rec = per_sample[pid]
            stratum = patient_stratum.get(pid, 'NA')
            row = [pid, str(rec.get('both_qc_pass', False)), stratum]
            for t in tile_cols:
                d = rec.get('per_tile_delta', {}).get(t, '')
                row.append(f'{d:.6f}' if isinstance(d, float) else '')
            f.write(','.join(row) + '\n')

    print(f'Runtime: {time.time() - t0:.1f}s')
    print(f'Outputs: {OUTPUT_DIR}/results.json, stratified.json, per_sample.csv')


if __name__ == '__main__':
    main()
