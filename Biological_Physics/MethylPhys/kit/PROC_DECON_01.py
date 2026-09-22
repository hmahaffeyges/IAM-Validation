#!/usr/bin/env python3
"""PROC-DECON-01 — deconvolver conformance against the project's documented outputs (TEST_DATA_MANIFEST.md),
plus PROC-WB-COMP-01 (whole blood: epithelial fraction ~0) and the presence-gated gauge read (PROC-WB-IMMUNE-01).

Input   : MethylPhys/atlas/IAMAtlasREBUILD.csv  (decompress from the repo's MethylPhys/atlas/IAMAtlasREBUILD.csv.xz; 605 MB; 483,092 rows)
          runtime/IAMAtlasREBUILD_celltype_to_class.json ; data/betas_cache.pkl
Expected: class-fraction MAE <= 0.001 vs manifest on GSM8772491, GSM5065990, GSM5065985; whole-blood epithelial sum < 0.02
Usage   : python PROC_DECON_01.py
"""
import sys, os, json, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpg_kit as K
sys.path.insert(0, K.ENGINE)
from walther_iam_deconvolver import WaltherIAMDeconvolver

EXPECTED = {  # TEST_DATA_MANIFEST.md 'Secretory readability' paragraph — only the classes the manifest states, at its 3-decimal precision
 "GSM8772491": {"cycling": .354, "immune": .253, "stem_pluri": .161, "secretory": .122, "terminal": .112},
 "GSM5065990": {"secretory": .129, "cycling": .357, "immune": .340},
 "GSM5065985": {"secretory": .248, "cycling": .481, "immune": .219},
}
WB = ["GSM2333901", "GSM2333905", "GSM2333950", "GSM1051525", "GSM1051526", "GSM1051533", "GSM1051534"]
AGE = {"GSM2333950": 43, "GSM2333901": 58, "GSM2333905": 67}
EPI = ["cycling", "secretory", "terminal", "stromal"]

def main():
    cache = pickle.load(open(os.path.join(K.DATA, "betas_cache.pkl"), "rb"))
    d = WaltherIAMDeconvolver(os.path.join(K.DATA, "IAMAtlasREBUILD.csv"), celltype_class_map=os.path.join(K.RUNTIME, "IAMAtlasREBUILD_celltype_to_class.json"))
    ident = K.load_identity(); arm = json.load(open(os.path.join(K.RUNTIME, "age_reference_matrix.json")))
    rows = []; ok_all = True
    for gsm, exp in EXPECTED.items():
        b = cache[gsm]; b = {k: float(x) for k, x in (b.items() if hasattr(b, "items") else enumerate(b)) if x == x}
        fr = d.deconvolve(b).class_fractions; mae = np.mean([abs(fr.get(c, 0) - e) for c, e in exp.items()])
        ok = mae <= 0.001; ok_all &= ok
        rows.append((gsm, f"class-fraction MAE vs manifest {mae:.4f} -> {'PASS' if ok else 'FAIL'}"))
    for gsm in WB:
        b = cache[gsm]; b = {k: float(x) for k, x in (b.items() if hasattr(b, "items") else enumerate(b)) if x == x}
        r = d.deconvolve(b); fr = r.class_fractions; epi = sum(fr.get(c, 0) for c in EPI)
        A, info = K.gauge_A(b, "immune", ident, presence=fr.get("immune", 0))
        band = ""
        if gsm in AGE and A is not None:
            e = min(arm["immune"], key=lambda e: abs(e["age_midpoint"] - AGE[gsm]))
            band = f" | age {AGE[gsm]} band [{e['A_p10']:.3f},{e['A_p90']:.3f}] -> {'BELOW' if A < e['A_p10'] else 'IN' if A <= e['A_p90'] else 'ABOVE'}"
        ok = epi < 0.02; ok_all &= ok
        rows.append((gsm, f"epithelial {epi:.4f} immune {fr.get('immune',0):.4f} resid {r.diagnostics['class_residual_mae']:.4f} -> {'PASS' if ok else 'FAIL'} | immune gauge A {A if A is None else round(A,4)}{band}"))
    K.report("PROC-DECON-01 + PROC-WB-COMP-01 + PROC-WB-IMMUNE-01", rows, "PASS (composition) — gauge placement reported, not judged", path=os.path.join(K.ROOT, "results", "PROC_DECON_01.json"))

if __name__ == "__main__":
    os.makedirs(os.path.join(K.ROOT, "results"), exist_ok=True); main()
