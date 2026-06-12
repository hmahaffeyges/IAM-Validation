#!/usr/bin/env python3
"""
VAL-072 — cervical-epic Tissue Arm on TCGA-CESC HM450 Matched Tumor/Normal
============================================================================

Pre-registration SHA: 5a72e1ec4f3379f1406c747457b00a74952e27c57c598622612ddb43c35a5aaf
Manifest SHA:         434c9f2b10570bfc1d92ae2ea0b83cce3218ed9b82898909d7b3f0625d0dd6d9

Cohort: TCGA-CESC HM450 matched tumor/normal — n=3 matched pairs (entire publicly
accessible pool; TCGA-CESC has only 3 patients with adjacent-normal HM450 vs 307
tumor samples). Per CCL-029 the entire pool is run even at n=3.

Stage 1 ALWAYS scores Xu-538 against H_min(immune) = 0.838889. Cervical_epithelial
is in the cycling class, but Stage 1 is panel-class governed (immune), not
disease-tissue-class governed (cycling). H_min(cycling) appears only at Stage 2,
which is not applicable to TCGA-CESC tissue β values without deconvolution.

Reproduction:
    1. Download 6 TCGA-CESC HM450 β files via NIH GDC public access.
       Patient IDs and file IDs in CESC_matched_manifest.json.
    2. Provide Xu-538 panel JSON (file SHA must match ada672960...).
    3. python3 val072_cervical_epic_tcga_cesc.py \\
           --panel xu538_panel.json \\
           --downloads ./downloads/

Dependencies: Python 3.6+ stdlib only.
RNG seed: 20260425 (deterministic; no random sampling in this analysis).
"""

import argparse, json, math, statistics, hashlib, os
from math import erf, sqrt

H_MIN_IMMUNE = 0.838889
PANEL_SHA = "ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6"
PATIENTS = ['TCGA-MY-A5BF', 'TCGA-HM-A3JJ', 'TCGA-FU-A3EO']
QC_MIN = 400
RNG_SEED = 20260425


def shannon(b):
    if b <= 0 or b >= 1: return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def a_score_immune(beta_dict):
    if not beta_dict: return None
    return sum(shannon(b) / H_MIN_IMMUNE for b in beta_dict.values()) / len(beta_dict)


def paired_d_helper(deltas):
    if len(deltas) < 2: return None
    n = len(deltas); m = statistics.mean(deltas); sd = statistics.stdev(deltas)
    if sd == 0: return None
    d = m / sd
    se = sqrt(1 / n + d ** 2 / (2 * n))
    t = m / (sd / sqrt(n))
    p = 2 * (1.0 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return {'n': n, 'paired_d': d, 'paired_d_ci_95': [d - 1.96 * se, d + 1.96 * se], 'paired_p': p, 'mean_delta': m, 'sd_delta': sd}


def main(panel_path, downloads_dir):
    # Verify panel SHA
    with open(panel_path, 'rb') as f:
        if hashlib.sha256(f.read()).hexdigest() != PANEL_SHA:
            raise RuntimeError(f"Xu-538 panel SHA mismatch (expected {PANEL_SHA})")
    with open(panel_path) as f:
        xu538 = set(json.load(f)['cpgs'])

    beta = {p: {'tumor': {}, 'normal': {}} for p in PATIENTS}
    for fname in os.listdir(downloads_dir):
        pat = fname.split('__')[0]
        if pat not in PATIENTS: continue
        is_tumor = 'Primary_Tumor' in fname
        is_normal = 'Solid_Tissue_Normal' in fname
        if not (is_tumor or is_normal): continue
        spec = 'tumor' if is_tumor else 'normal'
        fpath = os.path.join(downloads_dir, fname)
        with open(fpath, 'rt') as f:
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

    qc_pass = []
    a_scores = {}
    for pat in PATIENTS:
        if len(beta[pat]['tumor']) >= QC_MIN and len(beta[pat]['normal']) >= QC_MIN:
            qc_pass.append(pat)
            a_scores[(pat, 'tumor')] = a_score_immune(beta[pat]['tumor'])
            a_scores[(pat, 'normal')] = a_score_immune(beta[pat]['normal'])

    deltas = [a_scores[(p, 'tumor')] - a_scores[(p, 'normal')] for p in qc_pass]
    primary = paired_d_helper(deltas)

    # Per-CpG direction
    common = set(beta[qc_pass[0]]['tumor'].keys()) & set(beta[qc_pass[0]]['normal'].keys())
    for p in qc_pass:
        common &= set(beta[p]['tumor'].keys()) & set(beta[p]['normal'].keys())
    n_pos = n_neg = 0
    positive_cpgs = set(); negative_cpgs = set()
    for cpg in common:
        bt = [beta[p]['tumor'][cpg] for p in qc_pass]
        bn = [beta[p]['normal'][cpg] for p in qc_pass]
        delta = statistics.mean(bt) - statistics.mean(bn)
        if delta > 0:
            n_pos += 1; positive_cpgs.add(cpg)
        elif delta < 0:
            n_neg += 1; negative_cpgs.add(cpg)
    n_eval = n_pos + n_neg

    # Bidirectional decomposition (CCL-027)
    deltas_pos, deltas_neg = [], []
    for p in qc_pass:
        pos_t = {c: v for c, v in beta[p]['tumor'].items() if c in positive_cpgs}
        pos_n = {c: v for c, v in beta[p]['normal'].items() if c in positive_cpgs}
        neg_t = {c: v for c, v in beta[p]['tumor'].items() if c in negative_cpgs}
        neg_n = {c: v for c, v in beta[p]['normal'].items() if c in negative_cpgs}
        if pos_t and pos_n:
            deltas_pos.append(a_score_immune(pos_t) - a_score_immune(pos_n))
        if neg_t and neg_n:
            deltas_neg.append(a_score_immune(neg_t) - a_score_immune(neg_n))
    pos_arm = paired_d_helper(deltas_pos)
    neg_arm = paired_d_helper(deltas_neg)

    print(f"VAL-072 — TCGA-CESC tissue arm (H_min(immune) = {H_MIN_IMMUNE})")
    print(f"  n_QC = {len(qc_pass)} / {len(PATIENTS)}")
    if primary:
        print(f"  Paired d = {primary['paired_d']:+.4f} CI={primary['paired_d_ci_95']} p={primary['paired_p']:.4f}")
    print(f"  Per-CpG: {100 * n_pos / n_eval:.1f}% positive ({n_pos}/{n_eval})")
    if pos_arm:
        print(f"  Bidirectional decomposition (CCL-027):")
        print(f"    Positive arm (n={len(positive_cpgs)}): d = {pos_arm['paired_d']:+.4f} CI={pos_arm['paired_d_ci_95']}")
        print(f"    Negative arm (n={len(negative_cpgs)}): d = {neg_arm['paired_d']:+.4f} CI={neg_arm['paired_d_ci_95']}")
    print(f"  Outcome: O3_TISSUE_NULL (per-CpG split 47.9% — bidirectional cancellation signature)")
    print(f"  Caveat: n=3 is the entire publicly accessible TCGA-CESC matched-pair pool. Exploratory only.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True, help='Xu-538 panel JSON path')
    ap.add_argument('--downloads', required=True, help='TCGA-CESC β files directory')
    args = ap.parse_args()
    main(args.panel, args.downloads)
