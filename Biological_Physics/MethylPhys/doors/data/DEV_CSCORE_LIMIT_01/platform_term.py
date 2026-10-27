"""DEV-CSCORE-LIMIT-01: the 450K platform term, measured once and frozen (2026-10-26, before any GSE67393 array is read). From the one command's
bundles of the spent healthy 450K bloods (the five-lab healthy line's GSE42861, GSE51032, GSE87571): the median correlation of neighboring
sites at lags 1-9 (neighbor_corr), and the factor it gives the whole-genome C, f = 1 + 2 sum_k (1 - k/10) r_k. Nothing is taken from the
sample being read. Usage: python3 platform_term.py BUNDLE_DIR OUT.json"""
import sys, json, glob, os, numpy as np
V = []
for f in glob.glob(os.path.join(sys.argv[1], "**", "GSM*.json"), recursive=True):
    if f.endswith(("_values.json", "_planted.json")): continue
    B = json.load(open(f)); w = (B.get("cscore_each_cell") or {}).get("whole_genome") or {}
    if B.get("array_type") == "HM450K" and w.get("neighbor_corr"): V.append(w["neighbor_corr"])
r = np.median(np.array(V), axis=0); fac = 1 + 2 * sum((1 - k / 10) * x for k, x in enumerate(r, 1))
J = dict(platform="HM450K", bloods=len(V), neighbor_corr_median=[round(float(x), 5) for x in r], factor=round(float(fac), 5),
         rule="limit(n) x factor for a 450K whole-genome C; EPIC unchanged", source="the spent healthy 450K bloods of DEV-METAA-HEALTHY-01 (five-lab line), read by the one command")
json.dump(J, open(sys.argv[2], "w"), indent=1); print(J)
