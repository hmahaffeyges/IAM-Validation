"""DEV-FINGERPRINT-02: each normal array against its own atlas cell at that cell's identity sites (coverage, mean |beta - posterior mean|,
mean H of the array and of the atlas means). Usage: python3 normal_array_check.py WORKDIR   (WORKDIR/betas/<experiment>.parquet from calib_arrays.py)"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); W = sys.argv[1]
PAIR = {"ENCSR000ACD": "prostate_epithelium", "ENCSR955LKF": "hepatocyte", "ENCSR000ABW": "hepatocyte", "ENCSR422EPB": "lung_alveolar_epithelium",
        "ENCSR000ABH": "lung_alveolar_epithelium", "ENCSR000ACA": "lung_bronchus_epithelium", "ENCSR000ACY": "lung_bronchus_epithelium",
        "ENCSR148KKY": "breast_basal_epithelium", "ENCSR583ILE": "breast_basal_epithelium", "ENCSR000ABL": "breast_basal_epithelium"}
Hb = lambda b: (lambda c: -(c * np.log2(c) + (1 - c) * np.log2(1 - c)))(np.clip(b, 1e-6, 1 - 1e-6)); rows = []
for e, t in PAIR.items():
    S = json.load(open(os.path.join(HERE, f"identity_sites_{t}.json"))); s = pd.Series(S["posterior_mean"], index=S["sites"])
    b = pd.read_parquet(os.path.join(W, "betas", e + ".parquet")).beta; b.index = b.index.astype(str); bb = b.reindex(s.index); m = bb.notna()
    rows.append(dict(exp=e, tissue=t, sites=int(m.sum()), mae_vs_atlas=round(float((bb[m] - s[m]).abs().mean()), 3), H_array=round(float(Hb(bb[m]).mean()), 3), H_atlas=round(float(Hb(s[m]).mean()), 3)))
print(pd.DataFrame(rows).to_string(index=False))
