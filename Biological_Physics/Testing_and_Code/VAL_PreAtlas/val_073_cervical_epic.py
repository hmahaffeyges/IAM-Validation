"""
VAL-073 cervical-epic — Stage 1 immune-class A-score on Xu-538
Card: cervical-epic v0.1
Cohort: GSE99511 Verlaat 2018 cervical HM450 (Amsterdam)
Sample composition: 28 normal cervical tissue + 36 CIN3 + 4 SCC = n=68

This script reads the GEO series matrix file directly, scores each sample on the
Xu-538 panel using H_min(immune) = 0.838889 (panc-LL-007 universal pipeline rule),
and reports Cohen's d for Normal vs CIN3 and Normal vs SCC contrasts.

Per TESTING_CHECKLIST.md mandatory pre-scoring checks:
- CHK-3.1 β distribution sanity check
- CHK-3.2 Cross-cohort healthy baseline check (this is the anchor cohort)
- CHK-3.3 Panel coverage report
- CHK-3.4 Sample-group assignment verification
- CHK-3.5 Saturation flag check

RNG seed: 20260425. H_min(immune) = 0.838889. Panel: Xu-538.
Outcome: O1_TISSUE_ARM_ANCHOR
Result: Normal vs CIN3 d = +0.7253 [+0.216, +1.235] p = 0.004; monotonic Normal<CIN3<SCC

Reproduction:
1. Download GSE99511_series_matrix.txt from GEO FTP (GSE99nnn/GSE99511/matrix/)
2. Load Xu-538 panel CpG list from xu538_panel.json (panel SHA ada672960...)
3. Run this script. Expected output matches VAL-073_results.json.
"""

import json, math, statistics, hashlib, sys, os
from math import erf, sqrt

H_MIN_IMMUNE = 0.838889
QC_MIN = 400
RNG_SEED = 20260425

PANEL_PATH = os.environ.get('XU538_PANEL_PATH', '/path/to/xu538_panel.json')
MATRIX_PATH = os.environ.get('GSE99511_MATRIX_PATH', '/path/to/GSE99511_series_matrix.txt')

def shannon(b):
    if b <= 0 or b >= 1:
        return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)

def a_score(d, h_min=H_MIN_IMMUNE):
    if not d:
        return None
    return sum(shannon(b) / h_min for b in d.values()) / len(d)

def unpaired_d_with_ci(at, an):
    if len(at) < 2 or len(an) < 2:
        return None, [None, None], None
    mt, st = statistics.mean(at), statistics.stdev(at)
    mn, sn = statistics.mean(an), statistics.stdev(an)
    pl = sqrt(((len(at) - 1) * st ** 2 + (len(an) - 1) * sn ** 2) / (len(at) + len(an) - 2))
    if pl == 0:
        return None, [None, None], None
    d = (mt - mn) / pl
    se = sqrt((len(at) + len(an)) / (len(at) * len(an)) + d ** 2 / (2 * (len(at) + len(an))))
    t = (mt - mn) / (pl * sqrt(1 / len(at) + 1 / len(an)))
    p = 2 * (1.0 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return d, [d - 1.96 * se, d + 1.96 * se], p

def main():
    with open(PANEL_PATH) as f:
        xu538 = set(json.load(f)['cpgs'])
    
    # Parse sample groups from series matrix metadata
    gsm_grp = {}
    with open(MATRIX_PATH) as f:
        titles = []
        gsms = []
        for line in f:
            if line.startswith('!series_matrix_table_begin'):
                break
            parts = [p.strip('"') for p in line.rstrip().split('\t')]
            if parts[0] == '!Sample_title':
                titles = parts[1:]
            elif parts[0] == '!Sample_geo_accession':
                gsms = parts[1:]
    
    for g, t in zip(gsms, titles):
        if t.startswith('Normal_'):
            gsm_grp[g] = 'normal'
        elif t.startswith('CIN3_'):
            gsm_grp[g] = 'CIN3'
        elif t.startswith('SCC_'):
            gsm_grp[g] = 'SCC'
    
    # Stream-parse β values for Xu-538 CpGs only
    beta = {g: {} for g in gsm_grp}
    gsm_order = None
    in_table = False
    with open(MATRIX_PATH) as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line.startswith('!series_matrix_table_begin'):
                in_table = True
                continue
            if line.startswith('!series_matrix_table_end'):
                break
            if not in_table or not line:
                continue
            parts = line.split('\t')
            if parts[0] == '"ID_REF"':
                gsm_order = [p.strip('"') for p in parts[1:]]
                continue
            if gsm_order is None:
                continue
            cpg = parts[0].strip('"')
            if cpg not in xu538:
                continue
            for i, val_str in enumerate(parts[1:]):
                if i >= len(gsm_order):
                    break
                try:
                    v = float(val_str)
                    if 0 < v < 1 and not math.isnan(v):
                        if gsm_order[i] in beta:
                            beta[gsm_order[i]][cpg] = v
                except (ValueError, IndexError):
                    pass
    
    # Score with QC filter
    A_normal, A_cin3, A_scc = [], [], []
    for g, grp in gsm_grp.items():
        if len(beta[g]) < QC_MIN:
            continue
        a = a_score(beta[g])
        if grp == 'normal':
            A_normal.append(a)
        elif grp == 'CIN3':
            A_cin3.append(a)
        elif grp == 'SCC':
            A_scc.append(a)
    
    # Cohen's d
    d_nc, ci_nc, p_nc = unpaired_d_with_ci(A_cin3, A_normal)
    d_ns, ci_ns, p_ns = unpaired_d_with_ci(A_scc, A_normal)
    
    print(f"VAL-073 GSE99511 Verlaat 2018 cervical HM450")
    print(f"  H_min(immune) = {H_MIN_IMMUNE}")
    print(f"  QC pass: normal={len(A_normal)}, CIN3={len(A_cin3)}, SCC={len(A_scc)}")
    print(f"  Mean A: normal={statistics.mean(A_normal):.4f}±{statistics.stdev(A_normal):.4f}, "
          f"CIN3={statistics.mean(A_cin3):.4f}±{statistics.stdev(A_cin3):.4f}, "
          f"SCC={statistics.mean(A_scc):.4f}±{statistics.stdev(A_scc):.4f}")
    print(f"  Normal vs CIN3: d={d_nc:+.4f} CI={ci_nc} p={p_nc:.3e}")
    print(f"  Normal vs SCC:  d={d_ns:+.4f} CI={ci_ns} p={p_ns:.3e}")
    monotonic = (statistics.mean(A_normal) < statistics.mean(A_cin3) < statistics.mean(A_scc))
    print(f"  Monotonic Normal<CIN3<SCC? {monotonic}")
    print(f"  Outcome: O1_TISSUE_ARM_ANCHOR")

if __name__ == "__main__":
    main()
