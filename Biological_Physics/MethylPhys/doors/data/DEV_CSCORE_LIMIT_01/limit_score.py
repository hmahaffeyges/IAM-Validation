"""DEV-CSCORE-LIMIT-01 scorer (bar 2), written before it is run. Each healthy blood bundle (DEV-METAA-HEALTHY-01's five-laboratory set):
the neutrophil C-Score's raw statistic (met_a_cscore.clustering, untared) against limit(6,000 identity sites, or n_sites when given); each
cell's C (cscore_each_cell.cells, cells at a share of at least 0.01) and the whole genome's against limit(n_sites). Share at or below each.
Usage: python3 limit_score.py DIR OUT.txt"""
import sys, os, glob, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import limit as LM
R = {"neutrophil (commissioned sites)": [], "each cell": [], "whole genome": []}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "pass1*", "GSM*.json"), recursive=True)):
    if f.endswith(("_values.json", "_planted.json")): continue
    B = json.load(open(f)); m = B.get("met_a_cscore") or {}
    if m.get("clustering") is not None: R["neutrophil (commissioned sites)"].append((m["clustering"], LM.limit(m.get("n_sites") or 6000)))
    e = B.get("cscore_each_cell") or {}
    for c in e.get("cells") or []:
        if c.get("C") is not None and (c.get("share") or 0) >= 0.01: R["each cell"].append((c["C"], LM.limit(c["n_sites"])))
    w = e.get("whole_genome") or {}
    if w.get("C") is not None: R["whole genome"].append((w["C"], LM.limit(w["n_sites"])))
L = ["reading | values | median C | median limit | share at or below its limit"]; ok = True
for k, v in R.items():
    if not v: L.append(f"{k} | 0"); continue
    c = np.array([a for a, _ in v]); l = np.array([b for _, b in v]); s = float(np.mean(c <= l)); ok &= s >= 0.9
    L.append(f"{k} | {len(v)} | {np.median(c):.4f} | {np.median(l):.4f} | {s:.4f}")
L.append(f"bar 2 (at least 90 % of healthy values at or below the derived limit, each reading): {'met' if ok else 'not met'}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
