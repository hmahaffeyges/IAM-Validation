#!/usr/bin/env python3
"""
VAL-066 — pancreatic-epic Tissue Arm on TCGA-PAAD HM450
=========================================================

Pre-registration SHA: 694206201d45c1e3cbced1ef17b565b99e5d7f86a96b29fd58f6ba6050ea887e
Amendment SHA:        9533d64cc98d361a168ee941bcb737156b8410f655a15d2f878297734f5c344b

Cohort: TCGA-PAAD HM450 matched tumor/normal — n=7 amended (per amendment).
QC threshold ≥400 valid Xu-538 panel CpGs per sample → n=5 effective.

Stage 1 ALWAYS scores Xu-538 against H_min(immune) = 0.838889, regardless of
which disease card is being run. The disease-tissue class (secretory for
pancreas) is a Stage 2 concept, not a Stage 1 concept.

Reproduction:
    1. Download 18 TCGA-PAAD HM450 β files for the 7 amended patients via
       NIH GDC public access. Patient IDs and file IDs in PAAD_matched_manifest.json.
    2. Provide Xu-538 panel JSON (file SHA must match ada672960...).
    3. python3 val066_pancreatic_epic_tcga_paad.py \
           --panel xu538_panel.json \
           --downloads ./downloads/

Dependencies: Python 3.6+ stdlib only.
RNG seed: 20260425 (deterministic; no random sampling in this analysis).
"""

import argparse, json, math, statistics, hashlib, os, gzip
from math import erf, sqrt

H_MIN_IMMUNE = 0.838889
PANEL_SHA = "ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6"
PATIENTS = ['TCGA-FZ-5919', 'TCGA-FZ-5920', 'TCGA-FZ-5922', 'TCGA-FZ-5923',
            'TCGA-FZ-5924', 'TCGA-FZ-5926', 'TCGA-YB-A89D']
QC_MIN = 400
RNG_SEED = 20260425


def shannon(b):
    if b <= 0 or b >= 1: return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def a_score_immune(beta_dict):
    if not beta_dict: return None
    return sum(shannon(b) / H_MIN_IMMUNE for b in beta_dict.values()) / len(beta_dict)


def paired_d(deltas):
    n = len(deltas)
    m = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    if sd == 0: return None
    d = m / sd
    correction = 1.0 - 3.0 / (4 * (n - 1) - 1) if n > 2 else 1.0
    se = sqrt(1 / n + d ** 2 / (2 * n))
    t = m / (sd / sqrt(n))
    p = 2 * (1.0 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return {
        'n': n, 'mean_delta': m, 'sd_delta': sd,
        'paired_d': d, 'paired_d_hedges': d * correction,
        'paired_d_ci_95': [d - 1.96 * se, d + 1.96 * se],
        'paired_t': t, 'paired_p': p,
    }


def main(panel_path, downloads_dir):
    # Verify panel SHA
    with open(panel_path, 'rb') as f:
        if hashlib.sha256(f.read()).hexdigest() != PANEL_SHA:
            raise RuntimeError(f"Xu-538 panel SHA mismatch (expected {PANEL_SHA})")
    with open(panel_path) as f:
        xu538 = set(json.load(f)['cpgs'])

    # Load β at panel positions
    beta = {p: {'tumor': {}, 'normal': {}} for p in PATIENTS}
    for fname in os.listdir(downloads_dir):
        pat = fname.split('__')[0]
        if pat not in PATIENTS: continue
        is_tumor = 'Primary_Tumor' in fname
        is_normal = 'Solid_Tissue_Normal' in fname
        if not (is_tumor or is_normal): continue
        spec = 'tumor' if is_tumor else 'normal'
        fpath = os.path.join(downloads_dir, fname)
        open_fn = gzip.open if fname.endswith('.gz') else open
        with open_fn(fpath, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                cpg = parts[0]
                if cpg not in xu538: continue
                try:
                    v = float(parts[1])
                    if 0 < v < 1 and not math.isnan(v):
                        beta[pat][spec][cpg] = v
                except (ValueError, IndexError):
                    pass

    # QC + scoring
    qc_pass = []
    a_scores = {}
    for pat in PATIENTS:
        nt = len(beta[pat]['tumor']); nn = len(beta[pat]['normal'])
        if nt >= QC_MIN and nn >= QC_MIN:
            qc_pass.append(pat)
            a_scores[(pat, 'tumor')] = a_score_immune(beta[pat]['tumor'])
            a_scores[(pat, 'normal')] = a_score_immune(beta[pat]['normal'])

    # Primary paired d
    deltas = [a_scores[(p, 'tumor')] - a_scores[(p, 'normal')] for p in qc_pass]
    primary = paired_d(deltas) if len(deltas) >= 2 else None

    # Per-CpG direction preservation
    common = set()
    if qc_pass:
        common = set(beta[qc_pass[0]]['tumor'].keys()) & set(beta[qc_pass[0]]['normal'].keys())
        for p in qc_pass:
            common &= set(beta[p]['tumor'].keys()) & set(beta[p]['normal'].keys())
    n_pos = n_neg = 0
    for cpg in common:
        signs = [1 if beta[p]['tumor'][cpg] > beta[p]['normal'][cpg]
                 else (-1 if beta[p]['tumor'][cpg] < beta[p]['normal'][cpg] else 0)
                 for p in qc_pass]
        s = sum(signs)
        if s > 0: n_pos += 1
        elif s < 0: n_neg += 1
    n_eval = n_pos + n_neg

    print(f"VAL-066: n={len(qc_pass)} QC-passed of {len(PATIENTS)}")
    if primary:
        print(f"  Paired d = {primary['paired_d']:+.4f} CI={primary['paired_d_ci_95']} p={primary['paired_p']:.3e}")
    print(f"  Per-CpG: {100*n_pos/n_eval:.1f}% positive ({n_pos}/{n_eval})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True, help='Xu-538 panel JSON path')
    ap.add_argument('--downloads', required=True, help='TCGA-PAAD β files directory')
    args = ap.parse_args()
    main(args.panel, args.downloads)
