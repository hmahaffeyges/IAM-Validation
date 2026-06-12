#!/usr/bin/env python3
"""
VAL-067 — pancreatic-epic Tissue Arm on GSE49149 PDAC HM450 (large unpaired cohort)
====================================================================================

Pre-registration SHA: f0de98bd22c98bf1a48100387e6a9acf79aa24c4591608552085d8c0c0ba2efb

Cohort: GSE49149 (Mishra/Wood lab, PMIDs 24500968 + 26909576)
        167 PDAC tumor + 29 adjacent-normal HM450 = n=196 unpaired
        Largest publicly accessible PDAC tissue methylation cohort

Stage 1 ALWAYS scores Xu-538 against H_min(immune) = 0.838889, regardless of
which disease card is being run. The disease-tissue class (secretory for
pancreas) is a Stage 2 concept, not a Stage 1 concept.

This is the natural training set for any PDAC-specific directional panel
(VAL-069 builds the panel on this cohort).

Reproduction:
    1. Download GSE49149_series_matrix.txt(.gz) from GEO.
    2. Provide Xu-538 panel JSON (file SHA must match ada672960...).
    3. python3 val067_pancreatic_epic_gse49149.py \\
           --panel xu538_panel.json \\
           --series-matrix ./GSE49149_series_matrix.txt

Dependencies: Python 3.6+ stdlib only.
RNG seed: 20260425 (deterministic; no random sampling).
"""

import argparse, json, math, statistics, hashlib
from math import erf, sqrt

H_MIN_IMMUNE = 0.838889
PANEL_SHA = "ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6"
QC_MIN = 400
RNG_SEED = 20260425


def shannon(b):
    if b <= 0 or b >= 1: return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def a_score_immune(beta_dict):
    if not beta_dict: return None
    return sum(shannon(b) / H_MIN_IMMUNE for b in beta_dict.values()) / len(beta_dict)


def unpaired_d(at, an):
    if len(at) < 2 or len(an) < 2: return None, [None, None], None
    mt, st = statistics.mean(at), statistics.stdev(at)
    mn, sn = statistics.mean(an), statistics.stdev(an)
    pl = sqrt(((len(at) - 1) * st**2 + (len(an) - 1) * sn**2) / (len(at) + len(an) - 2))
    if pl == 0: return None, [None, None], None
    d = (mt - mn) / pl
    se = sqrt((len(at) + len(an)) / (len(at) * len(an)) + d**2 / (2 * (len(at) + len(an))))
    t = (mt - mn) / (pl * sqrt(1 / len(at) + 1 / len(an)))
    p = 2 * (1.0 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return d, [d - 1.96 * se, d + 1.96 * se], p


def main(panel_path, matrix_path):
    # Verify panel SHA
    with open(panel_path, 'rb') as f:
        if hashlib.sha256(f.read()).hexdigest() != PANEL_SHA:
            raise RuntimeError(f"Xu-538 panel SHA mismatch (expected {PANEL_SHA})")
    with open(panel_path) as f:
        xu538 = set(json.load(f)['cpgs'])

    # Parse sample metadata from series matrix header
    sample_ga, sample_src = [], []
    with open(matrix_path) as f:
        for line in f:
            if line.startswith('!series_matrix_table_begin'): break
            if line.startswith('!Sample_geo_accession'):
                sample_ga = [p.strip('"') for p in line.strip().split('\t')[1:]]
            elif line.startswith('!Sample_source_name_ch1'):
                sample_src = [p.strip('"') for p in line.strip().split('\t')[1:]]

    gsm_cat = {gsm: ('tumor' if 'Tumor' in s else 'normal')
               for gsm, s in zip(sample_ga, sample_src)}

    # Extract Xu-538 β at each sample
    beta = {gsm: {} for gsm in gsm_cat}
    gsm_order = None
    in_tab = False
    with open(matrix_path) as f:
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
            if cpg not in xu538: continue
            for i, val_str in enumerate(parts[1:]):
                if i >= len(gsm_order): break
                try:
                    v = float(val_str)
                    if 0 < v < 1 and not math.isnan(v):
                        beta[gsm_order[i]][cpg] = v
                except (ValueError, IndexError):
                    pass

    # QC + scoring
    A_t, A_n = [], []
    for gsm, cat in gsm_cat.items():
        if len(beta[gsm]) >= QC_MIN:
            a = a_score_immune(beta[gsm])
            (A_t if cat == 'tumor' else A_n).append(a)

    # Pooled unpaired d
    d_pool, ci_pool, p_pool = unpaired_d(A_t, A_n)

    # Per-CpG direction split (cohort-level mean Δβ)
    common = set()
    first = True
    for gsm in gsm_cat:
        if len(beta[gsm]) >= QC_MIN:
            if first:
                common = set(beta[gsm].keys()); first = False
            else:
                common &= set(beta[gsm].keys())

    n_pos = n_neg = 0
    positive_cpgs = set()
    negative_cpgs = set()
    for cpg in common:
        bt = [beta[g][cpg] for g, c in gsm_cat.items()
              if c == 'tumor' and len(beta[g]) >= QC_MIN]
        bn = [beta[g][cpg] for g, c in gsm_cat.items()
              if c == 'normal' and len(beta[g]) >= QC_MIN]
        if not bt or not bn: continue
        delta = statistics.mean(bt) - statistics.mean(bn)
        if delta > 0:
            n_pos += 1; positive_cpgs.add(cpg)
        elif delta < 0:
            n_neg += 1; negative_cpgs.add(cpg)
    n_eval = n_pos + n_neg

    # Bidirectional decomposition (CCL-027 mandatory)
    A_pos_t, A_pos_n, A_neg_t, A_neg_n = [], [], [], []
    for gsm, cat in gsm_cat.items():
        if len(beta[gsm]) < QC_MIN: continue
        pos_b = {c: v for c, v in beta[gsm].items() if c in positive_cpgs}
        neg_b = {c: v for c, v in beta[gsm].items() if c in negative_cpgs}
        if pos_b:
            (A_pos_t if cat == 'tumor' else A_pos_n).append(a_score_immune(pos_b))
        if neg_b:
            (A_neg_t if cat == 'tumor' else A_neg_n).append(a_score_immune(neg_b))

    d_pos, ci_pos, _ = unpaired_d(A_pos_t, A_pos_n)
    d_neg, ci_neg, _ = unpaired_d(A_neg_t, A_neg_n)

    print(f"VAL-067 — GSE49149 PDAC tissue (H_min(immune) = {H_MIN_IMMUNE})")
    print(f"  n_tumor QC = {len(A_t)}, n_normal QC = {len(A_n)}")
    print(f"  Pooled unpaired d = {d_pool:+.4f} CI=[{ci_pool[0]:+.4f}, {ci_pool[1]:+.4f}] p={p_pool:.3e}")
    print(f"  Per-CpG direction: {100*n_pos/n_eval:.1f}% positive ({n_pos}/{n_eval})")
    print(f"  Bidirectional decomposition (CCL-027):")
    print(f"    Positive arm ({len(positive_cpgs)} CpGs) d = {d_pos:+.4f} CI={ci_pos}")
    print(f"    Negative arm ({len(negative_cpgs)} CpGs) d = {d_neg:+.4f} CI={ci_neg}")
    print(f"  Outcome: O3_TISSUE_NULL_LARGE_COHORT (pooled CI straddles zero at largest available cohort)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True, help='Xu-538 panel JSON path')
    ap.add_argument('--series-matrix', required=True, help='GSE49149_series_matrix.txt path')
    args = ap.parse_args()
    main(args.panel, args.series_matrix)
