#!/usr/bin/env python3
"""PROC-PLASMA-MIX-01 — deconvolver against real known mixtures (Moss et al. 2018, Nat Commun, Supplementary Data 1 Table 6).

Input   : data/GSE122126-GPL21145_series_matrix.txt.gz (EPIC; in_vitro_mix_9..17 = Table 6 Mix 1..9 by order)
          MethylPhys/atlas/IAMAtlasREBUILD.csv ; runtime/IAMAtlasREBUILD_celltype_to_class.json
Declared: leukocytes 85-96% + hepatocytes / lung / cortical neurons / colon at 3.5-10% (Table 6, reproduced below)
Expected: each spiked tissue rises in its atlas class (Hepatocytes->secretory, Cortical_neurons->terminal, Colon_epithelial_cells->cycling)
Observed 2026-09-19: total non-leukocyte r=0.945 (bias +0.07); neurons->terminal r=+0.945 PASS; hepatocytes->secretory r=+0.19 FAIL; colon->cycling r=+0.10 FAIL
Usage   : python PROC_PLASMA_MIX_01.py
"""
import sys, os, json, numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpg_kit as K
sys.path.insert(0, K.ENGINE)
from walther_iam_deconvolver import WaltherIAMDeconvolver

T6 = {1: dict(leuk=.865, hep=.10, lung=.035, neur=0, colon=0), 2: dict(leuk=.85, hep=.05, lung=0, neur=.10, colon=0),
      3: dict(leuk=.915, hep=.035, lung=0, neur=0, colon=.05), 4: dict(leuk=.85, hep=0, lung=0, neur=.05, colon=.10),
      5: dict(leuk=.915, hep=0, lung=.05, neur=0, colon=.035), 6: dict(leuk=.865, hep=0, lung=.10, neur=.035, colon=0),
      7: dict(leuk=.94, hep=0, lung=0, neur=0, colon=.06), 8: dict(leuk=.92, hep=0, lung=.08, neur=0, colon=0), 9: dict(leuk=.96, hep=.04, lung=0, neur=0, colon=0)}
GEO = {"GSM3455853": 1, "GSM3455854": 2, "GSM3455855": 3, "GSM3455856": 4, "GSM3455857": 5, "GSM3455858": 6, "GSM3455861": 7, "GSM3455860": 8, "GSM3455859": 9}
HAEM = ["immune", "progenitor", "stem_adult"]

def main():
    gsms, B = K.stream_geo_matrix(os.path.join(K.DATA, "GSE122126-GPL21145_series_matrix.txt.gz"), keep_gsms=set(GEO))
    d = WaltherIAMDeconvolver(os.path.join(K.DATA, "IAMAtlasREBUILD.csv"), celltype_class_map=os.path.join(K.RUNTIME, "IAMAtlasREBUILD_celltype_to_class.json"))
    fr = {}
    for i, g in enumerate(gsms):
        b = {cg: float(v[i]) for cg, v in B.items() if v[i] == v[i]}
        fr[g] = d.deconvolve(b).class_fractions
    rows = []
    decl = np.array([1 - T6[GEO[g]]["leuk"] for g in gsms]); obs = np.array([1 - sum(fr[g].get(c, 0) for c in HAEM) for g in gsms])
    r, _ = stats.pearsonr(decl, obs); rows.append(("total non-leuk", f"r={r:.3f} bias={np.mean(obs-decl):+.4f} MAE={np.mean(np.abs(obs-decl)):.4f}"))
    verdicts = []
    for tissue, cls in [("neur", "terminal"), ("hep", "secretory"), ("colon", "cycling")]:
        x = np.array([T6[GEO[g]][tissue] for g in gsms]); y = np.array([fr[g].get(cls, 0) for g in gsms])
        rr, p = stats.pearsonr(x, y); v = "PASS" if rr >= 0.9 else "FAIL"; verdicts.append(v)
        rows.append((f"{tissue}->{cls}", f"r={rr:+.3f} p={p:.3f} declared 0-{x.max():.2f} observed {y.min():.3f}-{y.max():.3f} -> {v}"))
    K.report("PROC-PLASMA-MIX-01", rows, " / ".join(verdicts) + " (terminal / secretory / cycling)", path=os.path.join(K.ROOT, "results", "PROC_PLASMA_MIX_01.json"))

if __name__ == "__main__":
    os.makedirs(os.path.join(K.ROOT, "results"), exist_ok=True); main()
