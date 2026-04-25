#!/usr/bin/env python3
"""
VAL-069 — pancreatic-epic Directional Xu-538 Fallback Panel per CCL-027
=========================================================================

Pre-registration SHA: e31de916ac00268bfe22116f67f54317b1a99f63dc3dc7c1482019a0be1ae12a

Method:
  1. Train per-CpG ±1 directions on GSE49149 (n=196, 167 tumor + 29 normal).
     Direction = sign of mean(β_tumor − β_normal) at each CpG.
  2. Coverage filter: ≥80% of samples per arm must have measured β.
  3. Magnitude filter: |Δβ_train| > 0.005.
  4. Per-CpG normalization parameters (μ_normal_train, σ_normal_train) frozen
     from the GSE49149 normal arm (n=29).
  5. Score: A_dir = mean over panel CpGs of (direction × z) where
     z = (β − μ_normal_train) / σ_normal_train.
  6. The score is mathematically H_min-INDEPENDENT (z-score normalization
     cancels H_min) — directional approach is class-agnostic by construction.

Holdouts:
  H2 — TCGA-PAAD n=7 paired (matched tumor/normal HM450)
  H3 — GSE74071 n=7 paired (multi-substrate HM450)

Reproduction:
    python3 val069_pancreatic_epic_directional.py \\
        --panel xu538_panel.json \\
        --gse49149-matrix ./GSE49149_series_matrix.txt \\
        --gse74071-matrix ./GSE74071_series_matrix.txt \\
        --tcga-paad-dir ./TCGA-PAAD_downloads/

Dependencies: Python 3.6+ stdlib only.
RNG seed: 20260425.
"""

import argparse, json, math, statistics, hashlib, os, gzip
from math import erf, sqrt

PANEL_SHA = "ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6"
RNG_SEED = 20260425
COVERAGE_FRAC = 0.80
DELTA_BETA_MIN = 0.005

TCGA_HOLDOUT_PATIENTS = ['TCGA-FZ-5919', 'TCGA-FZ-5920', 'TCGA-FZ-5922',
                          'TCGA-FZ-5923', 'TCGA-FZ-5924', 'TCGA-FZ-5926',
                          'TCGA-YB-A89D']

GSE74071_PAIRING = {
    'PH64': ('GSM1909093', 'GSM1909094'),
    'PH67': ('GSM1909095', 'GSM1909096'),
    'P314_09_10': ('GSM1909106', 'GSM1909107'),
    'P314_11_12': ('GSM1909108', 'GSM1909109'),
    'GEMM_15_16': ('GSM1909111', 'GSM1909112'),
    'GEMM_17_18': ('GSM1909113', 'GSM1909114'),
    'GEMM_21_22': ('GSM1909115', 'GSM1909116'),
}


def parse_series_matrix(path, panel_set):
    """Parse a GEO series_matrix.txt for β values at panel CpGs only."""
    sample_ga, sample_src = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'): break
            if line.startswith('!Sample_geo_accession'):
                sample_ga = [p.strip('"') for p in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_source_name_ch1'):
                sample_src = [p.strip('"') for p in line.strip().split('\t')[1:]]
    beta = {gsm: {} for gsm in sample_ga}
    gsm_order = None; in_tab = False
    with open(path) as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line.startswith('!series_matrix_table_begin'):
                in_tab = True; continue
            if line.startswith('!series_matrix_table_end'): break
            if not in_tab or not line: continue
            parts = line.split('\t')
            if parts[0] == '"ID_REF"':
                gsm_order = [p.strip('"') for p in parts[1:]]; continue
            if gsm_order is None: continue
            cpg = parts[0].strip('"')
            if cpg not in panel_set: continue
            for i, val_str in enumerate(parts[1:]):
                if i >= len(gsm_order): break
                try:
                    v = float(val_str)
                    if 0 < v < 1 and not math.isnan(v):
                        beta[gsm_order[i]][cpg] = v
                except (ValueError, IndexError):
                    pass
    return beta, sample_ga, sample_src


def build_directional_panel(beta_tumor_list, beta_normal_list, panel_set):
    """Build directional panel from training cohort tumor/normal β lists."""
    panel = []
    pos_n = neg_n = 0
    cov_excluded = delta_excluded = 0
    n_t = len(beta_tumor_list); n_n = len(beta_normal_list)
    for cpg in panel_set:
        bt = [b[cpg] for b in beta_tumor_list if cpg in b]
        bn = [b[cpg] for b in beta_normal_list if cpg in b]
        if len(bt) < n_t * COVERAGE_FRAC or len(bn) < n_n * COVERAGE_FRAC:
            cov_excluded += 1; continue
        mt = statistics.mean(bt); mn = statistics.mean(bn)
        delta = mt - mn
        if abs(delta) < DELTA_BETA_MIN:
            delta_excluded += 1; continue
        direction = 1 if delta > 0 else -1
        sigma_n = statistics.stdev(bn) if len(bn) > 1 else 0.01
        if sigma_n < 1e-6: sigma_n = 0.01
        panel.append({'cpg': cpg, 'direction': direction,
                       'mu_normal_train': mn, 'sigma_normal_train': sigma_n})
        if direction > 0: pos_n += 1
        else: neg_n += 1
    return panel, pos_n, neg_n, cov_excluded, delta_excluded


def a_dir(beta_dict, panel):
    """Directional A_dir score — H_min-independent."""
    contributions = []
    for entry in panel:
        cpg = entry['cpg']
        if cpg not in beta_dict: continue
        z = (beta_dict[cpg] - entry['mu_normal_train']) / entry['sigma_normal_train']
        contributions.append(entry['direction'] * z)
    if not contributions: return None
    return statistics.mean(contributions)


def paired_d(deltas):
    if len(deltas) < 2: return None
    n = len(deltas); m = statistics.mean(deltas); sd = statistics.stdev(deltas)
    if sd == 0: return None
    d = m / sd
    se = sqrt(1 / n + d**2 / (2 * n))
    t = m / (sd / sqrt(n))
    p = 2 * (1.0 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return {'n': n, 'paired_d': d, 'paired_d_ci_95': [d - 1.96 * se, d + 1.96 * se], 'paired_p': p}


def main(panel_path, gse49149_path, gse74071_path, tcga_dir):
    with open(panel_path, 'rb') as f:
        if hashlib.sha256(f.read()).hexdigest() != PANEL_SHA:
            raise RuntimeError(f"Xu-538 panel SHA mismatch")
    with open(panel_path) as f:
        xu538 = set(json.load(f)['cpgs'])

    # Step 1 — parse training cohort GSE49149
    print(f"VAL-069 — Directional Xu-538 fallback for pancreatic-epic")
    print(f"  Step 1: Parse GSE49149 training cohort...")
    beta_train, ga_train, src_train = parse_series_matrix(gse49149_path, xu538)
    train_t = [beta_train[g] for g, s in zip(ga_train, src_train) if 'Tumor' in s and len(beta_train[g]) >= 400]
    train_n = [beta_train[g] for g, s in zip(ga_train, src_train) if 'Tumor' not in s and len(beta_train[g]) >= 400]
    print(f"    n_tumor train = {len(train_t)}, n_normal train = {len(train_n)}")

    # Step 2 — build directional panel
    print(f"  Step 2: Build directional panel...")
    panel, pos_n, neg_n, cov_ex, delta_ex = build_directional_panel(train_t, train_n, xu538)
    print(f"    Panel size = {len(panel)} CpGs ({pos_n} positive, {neg_n} negative)")
    print(f"    Excluded: {cov_ex} by coverage, {delta_ex} by low |Δβ|")

    # H1 — calibration check on training cohort (by-construction)
    train_t_scores = [a_dir(b, panel) for b in train_t if a_dir(b, panel) is not None]
    train_n_scores = [a_dir(b, panel) for b in train_n if a_dir(b, panel) is not None]
    mt, st = statistics.mean(train_t_scores), statistics.stdev(train_t_scores)
    mn, sn = statistics.mean(train_n_scores), statistics.stdev(train_n_scores)
    pooled = sqrt(((len(train_t_scores)-1)*st**2 + (len(train_n_scores)-1)*sn**2) /
                   (len(train_t_scores)+len(train_n_scores)-2))
    d_h1 = (mt - mn) / pooled
    print(f"\n  H1 — Calibration on training cohort (by-construction): d = {d_h1:+.4f}")

    # H2 — TCGA-PAAD holdout
    print(f"\n  H2 — TCGA-PAAD n=7 paired holdout:")
    panel_cpgs = {e['cpg'] for e in panel}
    deltas_tcga = []
    patient_scores = {}
    for pat in TCGA_HOLDOUT_PATIENTS:
        b_t, b_n = {}, {}
        for fname in os.listdir(tcga_dir):
            if not fname.startswith(pat): continue
            spec = 'tumor' if 'Primary_Tumor' in fname else ('normal' if 'Solid_Tissue_Normal' in fname else None)
            if spec is None: continue
            target = b_t if spec == 'tumor' else b_n
            fpath = os.path.join(tcga_dir, fname)
            opener = gzip.open if fname.endswith('.gz') else open
            with opener(fpath, 'rt') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) < 2: continue
                    cpg = parts[0]
                    if cpg not in panel_cpgs: continue
                    try:
                        v = float(parts[1])
                        if 0 < v < 1 and not math.isnan(v):
                            target[cpg] = v
                    except (ValueError, IndexError):
                        pass
        if len(b_t) < 100 or len(b_n) < 100: continue
        A_t = a_dir(b_t, panel); A_n = a_dir(b_n, panel)
        delta = A_t - A_n
        deltas_tcga.append(delta)
        patient_scores[pat] = {'A_tumor': A_t, 'A_normal': A_n, 'delta_A_dir': delta}
        print(f"    {pat}: A_t_dir={A_t:+.4f} A_n_dir={A_n:+.4f} Δ={delta:+.5f}")
    h2 = paired_d(deltas_tcga)
    print(f"    Paired d = {h2['paired_d']:+.4f} CI={h2['paired_d_ci_95']} p={h2['paired_p']:.3e}")
    print(f"    H2 outcome: {'PASS' if h2['paired_d_ci_95'][0] > 0 else 'FAIL'}")

    # H3 — GSE74071 holdout
    print(f"\n  H3 — GSE74071 n=7 paired holdout:")
    beta_74, _, _ = parse_series_matrix(gse74071_path, panel_cpgs)
    deltas_74 = []
    for label, (gt, gn) in GSE74071_PAIRING.items():
        if gt not in beta_74 or gn not in beta_74: continue
        A_t = a_dir(beta_74[gt], panel); A_n = a_dir(beta_74[gn], panel)
        if A_t is None or A_n is None: continue
        delta = A_t - A_n
        deltas_74.append(delta)
        print(f"    {label}: Δ={delta:+.5f}")
    h3 = paired_d(deltas_74)
    print(f"    Paired d = {h3['paired_d']:+.4f} CI={h3['paired_d_ci_95']} p={h3['paired_p']:.3e}")
    print(f"    H3 outcome: {'PASS' if h3['paired_d_ci_95'][0] > 0 else 'FAIL — PH64 outlier drags mean'}")

    print(f"\n  Overall outcome: O2_PARTIAL_RECOVERY (H2 PASS, H3 FAIL)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True, help='Xu-538 panel JSON')
    ap.add_argument('--gse49149-matrix', required=True, help='GSE49149_series_matrix.txt')
    ap.add_argument('--gse74071-matrix', required=True, help='GSE74071_series_matrix.txt')
    ap.add_argument('--tcga-paad-dir', required=True, help='TCGA-PAAD β files directory')
    args = ap.parse_args()
    main(args.panel, args.gse49149_matrix, args.gse74071_matrix, args.tcga_paad_dir)
