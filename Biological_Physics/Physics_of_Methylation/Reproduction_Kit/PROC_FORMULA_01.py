#!/usr/bin/env python3
"""PROC-FORMULA-01 — the two aggregations on the two loci sets, same samples (the measurement behind RULING A3).

Input   : data/betas_cache.pkl (11 Stage-1 samples: 7 whole blood, 4 EPIC tissue); runtime identity loci; runtime markers; age_reference_matrix.json
Operation: per class in {immune, cycling, secretory}, per sample: H(beta_mean)/H_min and mean_i H(beta_i)/H_min over
           (a) identity loci and (b) the class union of discriminative markers. Then Spearman and offset on whole blood
           vs tissue, and the check that age_reference_matrix A_mean == H(beta_mean)/H_min.
Expected (2026-09-19): whole blood immune: Spearman +1.000, offset +0.029; absent classes ~+0.16; tissue ~+0.24;
           age band == H(beta_mean)/H_min to 5 decimals in 80/80 cells.
Usage    : python PROC_FORMULA_01.py
"""
import sys, os, json, pickle, numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpg_kit as K

WB = ["GSM2333901","GSM2333905","GSM2333950","GSM1051525","GSM1051526","GSM1051533","GSM1051534"]
TI = ["GSM8772491","GSM8772492","GSM5065990","GSM5065985"]

def main():
    cache = pickle.load(open(os.path.join(K.DATA, "betas_cache.pkl"), "rb"))
    ident = K.load_identity(); markers, c2c = K.load_markers("chrX_removed")
    B = {g: {k: float(x) for k, x in (v.items() if hasattr(v, "items") else enumerate(v)) if x == x} for g, v in cache.items() if g in WB + TI}
    rows = []
    for cls in ["immune", "cycling", "secretory"]:
        ID = ident[cls]["loci"]; hm = ident[cls]["H_min"]
        DM = set(); [DM.update(l) for ct, l in markers.items() if c2c.get(ct) == cls]
        def both(g, loci):
            v = np.array([B[g][c] for c in loci if c in B[g]]); return K.H(v.mean())/hm, K.H(v).mean()/hm
        wb = np.array([both(g, ID) for g in WB]); ti = np.array([both(g, ID) for g in TI])
        rho = stats.spearmanr(wb[:, 0], wb[:, 1]).correlation
        rows.append((cls, f"identity loci | WB: Spearman(H(bmean),meanH)={rho:+.3f} offset {np.mean(wb[:,0]-wb[:,1]):+.4f}±{np.std(wb[:,0]-wb[:,1]):.4f} | tissue offset {np.mean(ti[:,0]-ti[:,1]):+.4f}±{np.std(ti[:,0]-ti[:,1]):.4f}"))
        dwb = np.array([both(g, DM) for g in WB]); dti = np.array([both(g, DM) for g in TI])
        rows.append((cls, f"discriminative | WB meanH {dwb[:,1].min():.3f}-{dwb[:,1].max():.3f} vs tissue {dti[:,1].min():.3f}-{dti[:,1].max():.3f} (presence proxy) | H(bmean) WB {dwb[:,0].min():.3f}-{dwb[:,0].max():.3f}"))
    arm = json.load(open(os.path.join(K.RUNTIME, "age_reference_matrix.json"))); mx = 0; n = 0
    for c, rs in arm.items():
        if c.startswith("_") or not isinstance(rs, list): continue
        for e in rs:
            if "beta_mean" in e: mx = max(mx, abs(e["A_mean"] - K.H(e["beta_mean"])/ident[c]["H_min"])); n += 1
    rows.append(("age band", f"max |A_mean - H(beta_mean)/H_min| = {mx:.5f} over {n} (class x decade) cells -> compiled as H(beta_mean)" if mx < 2e-3 else f"NOT H(beta_mean): max dev {mx:.4f}"))
    K.report("PROC-FORMULA-01", rows, "MEASURED — see RULING A3 (Issue 003 §1.5) for the decision this supports", path=os.path.join(K.ROOT, "results", "PROC_FORMULA_01.json"))

if __name__ == "__main__":
    os.makedirs(os.path.join(K.ROOT, "results"), exist_ok=True); main()
