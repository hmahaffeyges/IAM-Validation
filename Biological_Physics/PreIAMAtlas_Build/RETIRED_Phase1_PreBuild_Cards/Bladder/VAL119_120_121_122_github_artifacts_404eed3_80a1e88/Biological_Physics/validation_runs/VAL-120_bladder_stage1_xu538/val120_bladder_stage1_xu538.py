#!/usr/bin/env python3
"""
===============================================================================
VAL-120 — Stage 1 Xu-538 immune red flag on TCGA-BLCA n=440

Pre-registered: VAL-120_bladder_stage1_xu538/prereg.md
                SHA-256 sealed in PREREG_SEAL.txt before any β file read.

Question: Does Stage 1 Xu-538 immune red flag fire on TCGA-BLCA tumor tissue
          vs adjacent-normal? At what magnitude (paired n=21 + unpaired Welch)?

Cohort:   TCGA-BLCA HM450 sesame Level 3
          - 418 Primary Tumor + 21 Solid Tissue Normal + 1 Metastatic
          - 21 patients with both adjacent-normal and primary tumor (paired pairs)

Stage 1 panel: Xu 2020 djz065 — 538 CpGs, panel ID Xu2020_breast_cancer_replicated_full

A_immune = mean(Shannon_H(β_i) for i in panel ∩ sample) / H_min(immune)
H_min(immune) = 0.838889 (G-003b MCMC frozen 2026-04-06)

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

VAL_ID = 'VAL-120'
PREREG_SHA = '6d1807440dcf6cf33c9abbe791f9260224b768065bdd272f029b6e334d3c6996'
SEAL_TIMESTAMP = '2026-05-01T03:48:17Z'

H_MIN_IMMUNE = 0.838889

# Paths
XU538_PANEL = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/xu538_panel.json')
BLCA_DIR = Path('/home/claude/edear_working/bladder_epic/blca_betas')
BLCA_MANIFEST = Path('/home/claude/edear_working/bladder_epic/blca_manifest.json')
OUTPUT_DIR = Path('/home/claude/IAM-Validation/Biological_Physics/validation_runs/VAL-120_bladder_stage1_xu538')
OUTPUT_DIR.mkdir(exist_ok=True)

# Pre-locked thresholds
MAGNITUDE_THRESHOLD = 0.30
MIN_PAIRED_PAIRS = 15
CHK_3_1A_F_EXTREME_MIN = 0.50
CHK_3_1A_F_MIDDLE_MAX = 0.12
CHK_3_1A_PASS_RATE_MIN = 0.75
CHK_3_1B_COVERAGE_MIN = 0.80

RNG_SEED = 20260420


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def shannon_H(beta):
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
    """TCGA sesame Level 3 β file: tab-separated CpG_id β_value."""
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


def chk_3_1a(betas):
    vals = np.array(list(betas.values()))
    f_extreme = float(np.mean((vals < 0.1) | (vals > 0.9)))
    f_middle = float(np.mean((vals >= 0.4) & (vals <= 0.6)))
    return {
        'f_extreme': f_extreme,
        'f_middle': f_middle,
        'n_cpgs_genome': len(vals),
        'passed': (f_extreme >= CHK_3_1A_F_EXTREME_MIN) and (f_middle <= CHK_3_1A_F_MIDDLE_MAX),
    }


def compute_a_immune(betas, panel_cpgs):
    """Stage 1 pooled-entropy immune A-score."""
    H_vals = []
    for cpg in panel_cpgs:
        if cpg in betas:
            H_vals.append(shannon_H(betas[cpg]))
    if not H_vals:
        return float('nan'), 0
    a = np.mean(H_vals) / H_MIN_IMMUNE
    return float(a), len(H_vals)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print(f'VAL-120 — Stage 1 Xu-538 on TCGA-BLCA')
    print(f'Prereg SHA: {PREREG_SHA}')
    print(f'Sealed:     {SEAL_TIMESTAMP}')
    print()

    # Load panel
    with open(XU538_PANEL) as f:
        panel = json.load(f)
    panel_cpgs = panel['cpgs']
    n_panel = len(panel_cpgs)
    print(f'Xu-538 panel: {n_panel} CpGs')

    # Load manifest
    with open(BLCA_MANIFEST) as f:
        manifest = json.load(f)
    by_filename = {m['file_name']: m for m in manifest}

    # Find files on disk
    files_on_disk = sorted(BLCA_DIR.glob('*.txt'))
    print(f'BLCA files on disk: {len(files_on_disk)}')
    print()

    # Per-sample loop
    per_sample = []
    for i, path in enumerate(files_on_disk, 1):
        if i % 50 == 0:
            print(f'  {i}/{len(files_on_disk)} samples processed')
        meta = by_filename.get(path.name)
        if meta is None:
            print(f'  WARNING: no manifest entry for {path.name}; skipping')
            continue

        betas = load_beta_file(path)

        # CHK-3.1A
        chk_a = chk_3_1a(betas)

        # CHK-3.1B (panel coverage)
        n_panel_present = sum(1 for c in panel_cpgs if c in betas)
        coverage = n_panel_present / n_panel
        chk_b_passed = coverage >= CHK_3_1B_COVERAGE_MIN

        # A_immune
        a_immune, n_used = compute_a_immune(betas, panel_cpgs)

        per_sample.append({
            'sample_id': meta['sample_id'],
            'case_id': meta['case_id'],
            'sample_type': meta['sample_type'],
            'file_name': path.name,
            'n_cpgs_genome': chk_a['n_cpgs_genome'],
            'f_extreme': chk_a['f_extreme'],
            'f_middle': chk_a['f_middle'],
            'chk_3_1a_passed': chk_a['passed'],
            'n_xu538_present': n_panel_present,
            'xu538_coverage': coverage,
            'chk_3_1b_passed': chk_b_passed,
            'A_immune': a_immune,
            'n_xu538_used': n_used,
        })

    print(f'  Done. {len(per_sample)} samples scored.')
    print()

    # Save per-sample CSV
    per_sample_csv = OUTPUT_DIR / 'VAL-120_per_sample.csv'
    with open(per_sample_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_sample[0].keys()))
        w.writeheader()
        for row in per_sample:
            w.writerow(row)
    print(f'Per-sample saved: {per_sample_csv}')

    # QC summary
    n_chk_a_pass = sum(1 for s in per_sample if s['chk_3_1a_passed'])
    n_chk_b_pass = sum(1 for s in per_sample if s['chk_3_1b_passed'])
    print(f'CHK-3.1A pass: {n_chk_a_pass}/{len(per_sample)} ({n_chk_a_pass/len(per_sample):.1%})')
    print(f'CHK-3.1B pass: {n_chk_b_pass}/{len(per_sample)} ({n_chk_b_pass/len(per_sample):.1%})')

    # By-sample-type summary
    by_type = defaultdict(list)
    for s in per_sample:
        by_type[s['sample_type']].append(s['A_immune'])
    print()
    print('A_immune by sample type:')
    for st, vals in by_type.items():
        v = np.array(vals)
        print(f'  {st:25s}: n={len(v):3d}  mean={v.mean():.4f}  sd={v.std(ddof=1) if len(v)>1 else 0:.4f}')

    # Identify paired patients
    by_case = defaultdict(dict)
    for s in per_sample:
        by_case[s['case_id']][s['sample_type']] = s
    paired_cases = [
        c for c, types in by_case.items()
        if 'Solid Tissue Normal' in types and 'Primary Tumor' in types
    ]
    print(f'\nPaired patients (have both Solid Tissue Normal + Primary Tumor): {len(paired_cases)}')

    paired_data = []
    for c in paired_cases:
        n_s = by_case[c]['Solid Tissue Normal']
        t_s = by_case[c]['Primary Tumor']
        if (np.isnan(n_s['A_immune']) or np.isnan(t_s['A_immune'])
                or not n_s['chk_3_1a_passed'] or not n_s['chk_3_1b_passed']
                or not t_s['chk_3_1a_passed'] or not t_s['chk_3_1b_passed']):
            continue
        paired_data.append({
            'case_id': c,
            'normal_sample': n_s['sample_id'],
            'tumor_sample': t_s['sample_id'],
            'A_normal': n_s['A_immune'],
            'A_tumor': t_s['A_immune'],
            'paired_diff': t_s['A_immune'] - n_s['A_immune'],
        })

    print(f'Paired pairs after QC: {len(paired_data)}')
    paired_pairs_path = OUTPUT_DIR / 'VAL-120_paired_pairs.json'
    with open(paired_pairs_path, 'w') as f:
        json.dump(paired_data, f, indent=2)

    # Paired contrast
    if len(paired_data) >= MIN_PAIRED_PAIRS:
        diffs = np.array([p['paired_diff'] for p in paired_data])
        d_paired = float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else float('nan')
        # 95% CI on d (approximate, using sample-size adjustment)
        n = len(diffs)
        t_stat, p_paired = stats.ttest_rel(
            [p['A_tumor'] for p in paired_data],
            [p['A_normal'] for p in paired_data],
        )
        # CI on mean(diff) → CI on d via SE(d) ≈ sqrt(1/n + d²/(2n))
        se_d = float(np.sqrt(1/n + d_paired**2 / (2*n))) if not np.isnan(d_paired) else float('nan')
        ci_low = float(d_paired - 1.96 * se_d) if not np.isnan(d_paired) else float('nan')
        ci_high = float(d_paired + 1.96 * se_d) if not np.isnan(d_paired) else float('nan')
        direction = 'POSITIVE' if d_paired > 0 else 'NEGATIVE'
        print(f'\nPaired contrast (n={n}):')
        print(f'  d_paired = {d_paired:+.4f}  CI_95 [{ci_low:+.4f}, {ci_high:+.4f}]  p = {p_paired:.4g}  direction = {direction}')
    else:
        d_paired, ci_low, ci_high, p_paired, direction = float('nan'), float('nan'), float('nan'), float('nan'), 'INSUFFICIENT_PAIRS'

    # Unpaired Welch contrast
    qc_pass = [s for s in per_sample if s['chk_3_1a_passed'] and s['chk_3_1b_passed'] and not np.isnan(s['A_immune'])]
    a_tumor = [s['A_immune'] for s in qc_pass if s['sample_type'] == 'Primary Tumor']
    a_normal = [s['A_immune'] for s in qc_pass if s['sample_type'] == 'Solid Tissue Normal']
    a_met = [s['A_immune'] for s in qc_pass if s['sample_type'] == 'Metastatic']
    n_t, n_n = len(a_tumor), len(a_normal)
    if n_t >= 2 and n_n >= 2:
        m_t, m_n = np.mean(a_tumor), np.mean(a_normal)
        s_t, s_n = np.std(a_tumor, ddof=1), np.std(a_normal, ddof=1)
        pooled_sd = float(np.sqrt(((n_t-1)*s_t**2 + (n_n-1)*s_n**2) / (n_t + n_n - 2)))
        d_welch = float((m_t - m_n) / pooled_sd) if pooled_sd > 0 else float('nan')
        t_w, p_welch = stats.ttest_ind(a_tumor, a_normal, equal_var=False)
        # Welch d CI ≈ d ± 1.96·sqrt(1/n_t + 1/n_n + d²/(2(n_t+n_n)))
        se_w = float(np.sqrt(1/n_t + 1/n_n + d_welch**2/(2*(n_t+n_n)))) if not np.isnan(d_welch) else float('nan')
        ci_w_low = float(d_welch - 1.96*se_w) if not np.isnan(d_welch) else float('nan')
        ci_w_high = float(d_welch + 1.96*se_w) if not np.isnan(d_welch) else float('nan')
        d_welch_dir = 'POSITIVE' if d_welch > 0 else 'NEGATIVE'
        print(f'\nUnpaired Welch contrast (n_tumor={n_t}, n_normal={n_n}):')
        print(f'  d_welch = {d_welch:+.4f}  CI_95 [{ci_w_low:+.4f}, {ci_w_high:+.4f}]  p = {p_welch:.4g}  direction = {d_welch_dir}')
    else:
        d_welch, ci_w_low, ci_w_high, p_welch, d_welch_dir = float('nan'), float('nan'), float('nan'), float('nan'), 'INSUFFICIENT_DATA'

    if a_met:
        print(f'\nMetastatic exploratory: A_immune = {a_met[0]:.4f}')

    # Determine outcome
    if not (n_chk_a_pass / len(per_sample) >= CHK_3_1A_PASS_RATE_MIN
            and n_chk_b_pass / len(per_sample) >= CHK_3_1A_PASS_RATE_MIN):
        outcome_class = 'O4_STAGE1_DATA_INTEGRITY_FAILURE'
        outcome_note = f'CHK-3.1A {n_chk_a_pass/len(per_sample):.1%} or CHK-3.1B {n_chk_b_pass/len(per_sample):.1%} below 75% threshold'
    elif len(paired_data) < MIN_PAIRED_PAIRS:
        outcome_class = 'O4_STAGE1_DATA_INTEGRITY_FAILURE'
        outcome_note = f'Only {len(paired_data)} paired pairs after QC (< {MIN_PAIRED_PAIRS})'
    elif abs(d_paired) >= MAGNITUDE_THRESHOLD:
        if direction == 'POSITIVE':
            outcome_class = 'O1_STAGE1_IMMUNE_FIRES_POSITIVE'
        else:
            outcome_class = 'O2_STAGE1_IMMUNE_FIRES_NEGATIVE'
        outcome_note = f'd_paired = {d_paired:+.4f}, direction = {direction}, |d| ≥ 0.30'
    else:
        outcome_class = 'O3_STAGE1_IMMUNE_NULL'
        outcome_note = f'd_paired = {d_paired:+.4f}, |d| < 0.30, direction = {direction}'

    runtime = time.time() - t_start

    # Results JSON
    results = {
        'val_id': VAL_ID,
        'val_type': 'PHASE_C_STAGE1_XU538',
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
            'n_paired_pairs_qc_passed': len(paired_data),
        },
        'panel': {
            'name': 'Xu-538',
            'panel_id': panel.get('panel_id'),
            'source': panel.get('source'),
            'n_cpgs': n_panel,
        },
        'chk_3_1a': {
            'observed_f_extreme_mean': float(np.mean([s['f_extreme'] for s in per_sample])),
            'observed_f_extreme_sd': float(np.std([s['f_extreme'] for s in per_sample], ddof=1)),
            'observed_f_middle_mean': float(np.mean([s['f_middle'] for s in per_sample])),
            'observed_f_middle_sd': float(np.std([s['f_middle'] for s in per_sample], ddof=1)),
            'pass_rate': n_chk_a_pass / len(per_sample),
            'n_passed': n_chk_a_pass,
            'gate_passed': n_chk_a_pass / len(per_sample) >= CHK_3_1A_PASS_RATE_MIN,
        },
        'chk_3_1b': {
            'observed_coverage_mean': float(np.mean([s['xu538_coverage'] for s in per_sample])),
            'observed_coverage_sd': float(np.std([s['xu538_coverage'] for s in per_sample], ddof=1)),
            'pass_rate': n_chk_b_pass / len(per_sample),
            'n_passed': n_chk_b_pass,
            'gate_passed': n_chk_b_pass / len(per_sample) >= CHK_3_1A_PASS_RATE_MIN,
        },
        'paired_contrast': {
            'n_pairs': len(paired_data),
            'd_paired': d_paired,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'p_value': float(p_paired) if not np.isnan(p_paired) else None,
            'direction': direction,
            'magnitude_threshold': MAGNITUDE_THRESHOLD,
            'fires': abs(d_paired) >= MAGNITUDE_THRESHOLD if not np.isnan(d_paired) else False,
        },
        'unpaired_welch_contrast': {
            'n_tumor': n_t,
            'n_normal': n_n,
            'd_welch': d_welch,
            'ci_95_low': ci_w_low,
            'ci_95_high': ci_w_high,
            'p_value': float(p_welch) if not np.isnan(p_welch) else None,
            'direction': d_welch_dir,
        },
        'metastatic_exploratory': {
            'n': len(a_met),
            'A_immune': a_met[0] if a_met else None,
        },
        'a_immune_by_sample_type': {
            st: {
                'n': len(by_type[st]),
                'mean': float(np.mean(by_type[st])),
                'sd': float(np.std(by_type[st], ddof=1)) if len(by_type[st]) > 1 else 0.0,
            }
            for st in by_type
        },
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
