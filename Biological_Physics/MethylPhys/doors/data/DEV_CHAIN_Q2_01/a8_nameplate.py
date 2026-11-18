#!/usr/bin/env python3
"""A8 nameplate test, round 2 (register A8; author: 'it lies somewhere in one or both of the cell's identity pattern and its islands meter').
Runs ON A GITHUB MACHINE. Reads the chain's own per-type atlas tables (Runtime Matrices/Seq_Atlas: metaa_seq_reference_v1, held_reference_v1),
computes per-type identity features that disease should not move (how the type's identity sites are laid out), and asks which predict the
type's healthy position on the derived line (A8-position, passed in below from the 150 one-button readings). Spearman over types with a
permutation p; the islands meter (round 1) gave nothing. Exploration on spent tables; no reading of new data."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow", "numpy"], check=True)
import json, os, random, numpy as np, pandas as pd
D = "Biological_Physics/MethylPhys/chain/Runtime Matrices/Seq_Atlas"
POS = json.loads('{"erythrocyte progenitors": 0.8043, "cortical neurons": 0.8954, "naive cd8 t cells": 0.8979, "naive cd4 t cells": 0.9323, "lung alveolar macrophages": 0.9464, "pancreatic acinar": 0.951, "smooth muscle": 0.9583, "lung interstitial macrophages": 0.9597, "heart fibroblasts": 0.9713, "breast luminal epithelium": 0.9937, "hepatocyte": 0.9986, "breast basal epithelium": 1.0012, "effector memory cd8 t cells": 1.0014, "colon macrophages": 1.0087, "endometrium epithelium": 1.0199, "monocytes": 1.0207, "b cells": 1.0256, "oligodendrocytes": 1.0266, "colon epithelium": 1.0302, "colon enteroendocrine": 1.0314, "fallopien tubes epithelium": 1.035, "nk cells": 1.0445, "t effector memory cd4": 1.0451, "cd8 t cells": 1.0491, "cardiomyocyte": 1.0541, "gastric fundus epithelium": 1.0551, "cd4 t cells": 1.0563, "small intestine epithelium": 1.0621, "aorta endothelium": 1.0622, "kidney tubular epithelium": 1.0633, "gastric body epithelium": 1.0643, "lung alveolar epithelium": 1.0672, "pancreatic duct": 1.0698, "prostate epithelium": 1.0708, "gastric antrum epithelium": 1.0766, "lung alveolar endothelium": 1.0806, "tonsil pharyngeal epithelium": 1.0835, "tonsil palatine epithelium": 1.0935, "memory b cells": 1.0957, "colon fibroblasts": 1.0982, "kidney glomerular epithelium": 1.1052, "t effector cell cd8": 1.1056, "bladder epithelium": 1.1074, "thyroid epithelium": 1.1164, "lung bronchus epithelium": 1.1187, "adipocytes": 1.1201, "vascular saphenous endothelium": 1.1272, "pancreatic beta": 1.1281, "tongue epithelium": 1.1327, "kidney glomerular endothelium": 1.1414, "kidney tubular endothelium": 1.142, "pancreatic alpha": 1.1459, "pancreas endothelium": 1.1498, "pancreatic delta": 1.1651, "striated muscle": 1.1709, "t central memory cd4": 1.1713}')   # {type: mean A_derived}
def norm(s): return str(s).strip().lower().replace("_", " ")
out = {}
for name in ("metaa_seq_reference_v1", "held_reference_v1"):
    j = json.load(open(f"{D}/{name}.json")); df = pd.read_parquet(f"{D}/{name}.parquet")
    print(f"::notice::{name}: shape {df.shape}; columns {list(df.columns)[:14]}; meta {json.dumps(j)[:400]}")
    out[name] = df
def per_type_features(df):
    """per cell-type column of site levels: held share (>=0.8), unmarked share (<=0.2), intermediate share, mean, level entropy"""
    num = [c for c in df.columns if df[c].dtype.kind == "f"]
    F = {}
    for c in num:
        x = df[c].to_numpy(dtype="float64"); x = x[np.isfinite(x)]
        if len(x) < 1000: continue
        h = np.clip(x, 1e-6, 1 - 1e-6); H = -(h * np.log2(h) + (1 - h) * np.log2(1 - h))
        F[norm(c)] = dict(n=len(x), held=float(np.mean(x >= 0.8)), unmarked=float(np.mean(x <= 0.2)), mid=float(np.mean((x > 0.2) & (x < 0.8))),
                          mean=float(x.mean()), entropy=float(H.mean()))
    return F
def sp(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
res = {}
for name, df in out.items():
    F = per_type_features(df); common = sorted(set(F) & set(POS))
    print(f"::notice::{name}: {len(F)} type columns, {len(common)} matched to the 56 positions; unmatched positions {sorted(set(POS) - set(F))[:8]}")
    if len(common) < 15: continue
    a = [POS[t] for t in common]
    for k in ("held", "unmarked", "mid", "mean", "entropy"):
        b = [F[t][k] for t in common]; r = sp(a, b); random.seed(4)
        nul = [sp(a, random.sample(b, len(b))) for _ in range(5000)]; p = float(np.mean([abs(z) >= abs(r) for z in nul]))
        res[f"{name}:{k}"] = dict(rho=round(r, 3), p=round(p, 4), n=len(common))
        print(f"::notice::A8 NAMEPLATE {name} {k}: Spearman {r:+.3f} p {p:.4f} (n {len(common)} types)")
os.makedirs("xout", exist_ok=True); json.dump(res, open("xout/a8_nameplate.json", "w"), indent=1)
