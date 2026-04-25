#!/usr/bin/env python3
"""
VAL-068 — pancreatic-epic Multi-Substrate Tissue Arm on GSE74071 PDAC HM450
============================================================================

Pre-registration SHA: 50c0c7e8afccc2a5dfc407bf95e29b846cb1f3effc1458484e28e88f3cbaedfc

Cohort: GSE74071 (Tjensvoll et al.)
        Multi-substrate PDAC HM450 — 28 samples total:
          14 tumor + 7 adjacent-normal + 4 pancreatic juice CCC +
          3 cancer-associated fibroblasts (CAFs) + 1 primary culture

7 paired tumor/normal pairs identified by sequential GSM ID convention:
  PH64A/B, PH67A/B, P314_09/10, P314_11/12, GEMM 15/16, 17/18, 21/22

Stage 1 ALWAYS scores Xu-538 against H_min(immune) = 0.838889. CAFs are
scored with H_min(stromal) = 0.862950 as supplementary analysis only.

Reproduction:
    1. Download GSE74071_series_matrix.txt(.gz) from GEO.
    2. Provide Xu-538 panel JSON (file SHA must match ada672960...).
    3. python3 val068_pancreatic_epic_gse74071.py \\
           --panel xu538_panel.json \\
           --series-matrix ./GSE74071_series_matrix.txt

Dependencies: Python 3.6+ stdlib only.
RNG seed: 20260425 (deterministic; no random sampling).
"""

import argparse, json, math, statistics, hashlib
from math import erf, sqrt

H_MIN_IMMUNE = 0.838889
H_MIN_STROMAL = 0.862950
PANEL_SHA = "ada6729605639138fb1d9128b5d708aea009b8ac98a49d0fd9e8d7343334a6d6"
QC_MIN = 400
RNG_SEED = 20260425

# Sample-to-category map from GSE74071 manifest
SAMPLE_MAP = {
    'GSM1909093': ('tumor', 'PH64'), 'GSM1909094': ('normal', 'PH64'),
    'GSM1909095': ('tumor', 'PH67'), 'GSM1909096': ('normal', 'PH67'),
    'GSM1909097': ('normal', 'PH70'),
    'GSM1909098': ('juice', '314001'), 'GSM1909099': ('juice', '314002'),
    'GSM1909100': ('juice', '314003'), 'GSM1909101': ('juice', '314004'),
    'GSM1909102': ('primary_culture', '314005'),
    'GSM1909103': ('caf', '314006'), 'GSM1909104': ('caf', '314007'),
    'GSM1909105': ('caf', '314008'),
    'GSM1909106': ('tumor', '314009'), 'GSM1909107': ('normal', '314010'),
    'GSM1909108': ('tumor', '314011'), 'GSM1909109': ('normal', '314012'),
    'GSM1909110': ('tumor', 'GEMM9'), 'GSM1909111': ('tumor', 'GEMM15'),
    'GSM1909112': ('normal', 'GEMM16'),
    'GSM1909113': ('tumor', 'GEMM17'), 'GSM1909114': ('normal', 'GEMM18'),
    'GSM1909115': ('tumor', 'GEMM21'), 'GSM1909116': ('normal', 'GEMM22'),
    'GSM1909117': ('tumor', 'GEMM23'), 'GSM1909118': ('tumor', 'GEMM25'),
    'GSM1909119': ('tumor', 'GEMM26'), 'GSM1909120': ('tumor', 'GEMM27'),
}

PAIRING_MAP = {
    'PH64': ('GSM1909093', 'GSM1909094'),
    'PH67': ('GSM1909095', 'GSM1909096'),
    'P314_09_10': ('GSM1909106', 'GSM1909107'),
    'P314_11_12': ('GSM1909108', 'GSM1909109'),
    'GEMM_15_16': ('GSM1909111', 'GSM1909112'),
    'GEMM_17_18': ('GSM1909113', 'GSM1909114'),
    'GEMM_21_22': ('GSM1909115', 'GSM1909116'),
}


def shannon(b):
    if b <= 0 or b >= 1: return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def a_score(beta_dict, hmin):
    if not beta_dict: return None
    return sum(shannon(b) / hmin for b in beta_dict.values()) / len(beta_dict)


def paired_d(deltas):
    if len(deltas) < 2: return None
    n = len(deltas); m = statistics.mean(deltas); sd = statistics.stdev(deltas)
    if sd == 0: return None
    d = m / sd
    se = sqrt(1 / n + d**2 / (2 * n))
    t = m / (sd / sqrt(n))
    p = 2 * (1.0 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return {'n': n, 'paired_d': d, 'paired_d_ci_95': [d - 1.96 * se, d + 1.96 * se], 'paired_p': p, 'mean_delta': m}


def unpaired_d(at, an):
    if len(at) < 2 or len(an) < 2: return None
    mt, st = statistics.mean(at), statistics.stdev(at)
    mn, sn = statistics.mean(an), statistics.stdev(an)
    pl = sqrt(((len(at) - 1) * st**2 + (len(an) - 1) * sn**2) / (len(at) + len(an) - 2))
    if pl == 0: return None
    d = (mt - mn) / pl
    se = sqrt((len(at) + len(an)) / (len(at) * len(an)) + d**2 / (2 * (len(at) + len(an))))
    return {'unpaired_d': d, 'unpaired_d_ci_95': [d - 1.96 * se, d + 1.96 * se]}


def main(panel_path, matrix_path):
    with open(panel_path, 'rb') as f:
        if hashlib.sha256(f.read()).hexdigest() != PANEL_SHA:
            raise RuntimeError(f"Xu-538 panel SHA mismatch (expected {PANEL_SHA})")
    with open(panel_path) as f:
        xu538 = set(json.load(f)['cpgs'])

    beta = {gsm: {} for gsm in SAMPLE_MAP}
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

    # Per-sample A_immune
    A_immune = {}
    for gsm in SAMPLE_MAP:
        if len(beta[gsm]) >= QC_MIN:
            A_immune[gsm] = a_score(beta[gsm], H_MIN_IMMUNE)

    # H1 — paired tumor vs normal
    deltas = []
    print(f"VAL-068 — GSE74071 multi-substrate PDAC (H_min(immune) = {H_MIN_IMMUNE})")
    print(f"\n  H1 — Paired tumor vs adjacent normal:")
    for label, (gt, gn) in PAIRING_MAP.items():
        if gt in A_immune and gn in A_immune:
            d = A_immune[gt] - A_immune[gn]
            deltas.append(d)
            print(f"    {label}: A_t={A_immune[gt]:.4f} A_n={A_immune[gn]:.4f} ΔA={d:+.5f}")
    h1 = paired_d(deltas)
    print(f"    Paired d = {h1['paired_d']:+.4f} CI={h1['paired_d_ci_95']} p={h1['paired_p']:.3e}")

    # H2 — juice vs normal (unpaired)
    A_juice = [A_immune[g] for g, (c, _) in SAMPLE_MAP.items() if c == 'juice' and g in A_immune]
    A_norm_all = [A_immune[g] for g, (c, _) in SAMPLE_MAP.items() if c == 'normal' and g in A_immune]
    h2 = unpaired_d(A_juice, A_norm_all)
    print(f"\n  H2 — Pancreatic juice CCC (n={len(A_juice)}) vs adjacent normal (n={len(A_norm_all)}):")
    print(f"    Unpaired d = {h2['unpaired_d']:+.4f} CI={h2['unpaired_d_ci_95']}")

    # H3 — CAFs scored with H_min(stromal) supplementary
    A_caf = []
    for g, (c, _) in SAMPLE_MAP.items():
        if c == 'caf' and len(beta[g]) >= QC_MIN:
            A_caf.append(a_score(beta[g], H_MIN_STROMAL))
    if A_caf:
        print(f"\n  H3 — CAFs supplementary at H_min(stromal) = {H_MIN_STROMAL}: n={len(A_caf)}, mean A = {statistics.mean(A_caf):.4f}")

    # H4 — per-CpG direction split
    gsm_t = [g for g, (c, _) in SAMPLE_MAP.items() if c == 'tumor' and g in A_immune]
    gsm_n = [g for g, (c, _) in SAMPLE_MAP.items() if c == 'normal' and g in A_immune]
    common = set(beta[gsm_t[0]].keys()) if gsm_t else set()
    for g in gsm_t + gsm_n: common &= set(beta[g].keys())
    n_pos = n_neg = 0
    positive_cpgs = set(); negative_cpgs = set()
    for cpg in common:
        bt = [beta[g][cpg] for g in gsm_t]
        bn = [beta[g][cpg] for g in gsm_n]
        delta = statistics.mean(bt) - statistics.mean(bn)
        if delta > 0:
            n_pos += 1; positive_cpgs.add(cpg)
        elif delta < 0:
            n_neg += 1; negative_cpgs.add(cpg)
    n_eval = n_pos + n_neg
    print(f"\n  H4 — Per-CpG direction split: {100*n_pos/n_eval:.1f}% positive ({n_pos}/{n_eval})")

    # Bidirectional decomposition (CCL-027 mandatory)
    deltas_pos, deltas_neg = [], []
    for label, (gt, gn) in PAIRING_MAP.items():
        if gt not in A_immune or gn not in A_immune: continue
        pos_t = {c: v for c, v in beta[gt].items() if c in positive_cpgs}
        pos_n = {c: v for c, v in beta[gn].items() if c in positive_cpgs}
        neg_t = {c: v for c, v in beta[gt].items() if c in negative_cpgs}
        neg_n = {c: v for c, v in beta[gn].items() if c in negative_cpgs}
        if pos_t and pos_n:
            deltas_pos.append(a_score(pos_t, H_MIN_IMMUNE) - a_score(pos_n, H_MIN_IMMUNE))
        if neg_t and neg_n:
            deltas_neg.append(a_score(neg_t, H_MIN_IMMUNE) - a_score(neg_n, H_MIN_IMMUNE))
    pos_arm = paired_d(deltas_pos)
    neg_arm = paired_d(deltas_neg)
    print(f"  Bidirectional decomposition (CCL-027):")
    print(f"    Positive arm ({len(positive_cpgs)} CpGs) paired d = {pos_arm['paired_d']:+.4f}")
    print(f"    Negative arm ({len(negative_cpgs)} CpGs) paired d = {neg_arm['paired_d']:+.4f}")
    print(f"\n  Outcome: O3_TUMOR_NULL — paired CI straddles zero; PH64 strong negative outlier")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', required=True, help='Xu-538 panel JSON path')
    ap.add_argument('--series-matrix', required=True, help='GSE74071_series_matrix.txt path')
    args = ap.parse_args()
    main(args.panel, args.series_matrix)
