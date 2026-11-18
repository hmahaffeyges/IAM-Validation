#!/usr/bin/env python3
"""A8 nameplate test, round 2b (register A8; author: 'it lies somewhere in one or both of the cell's identity pattern and its islands meter'). Runs ON A
GITHUB MACHINE. Round 2 read nothing (the tables are long, not one column per type). This reads the chain's own per-type atlas tables in their real layout:
Seq_Atlas metaa_seq_reference_v1 (one row per identity site and cell: level, spread, disorder) and held_reference_v1 (per-cell meta), and asks whether a
feature of a type's identity pattern predicts its position on the derived line (positions from the joint run; written before this run). Prediction (lead,
author): a type whose identity pattern is held less firmly (lower mean level, more disorder at its identity sites) sits higher. Spearman over types,
permutation p (5000). held_reference's own held error is reported only as a consistency check (it is the same quantity as the position, not a nameplate)."""
import os, sys, json, glob, subprocess, random
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow", "numpy"], check=True)
import numpy as np, pandas as pd
POS = json.loads('{"erythrocyte progenitors": 0.8043, "cortical neurons": 0.8954, "naive cd8 t cells": 0.8979, "naive cd4 t cells": 0.9323, "lung alveolar macrophages": 0.9464, "pancreatic acinar": 0.951, "smooth muscle": 0.9583, "lung interstitial macrophages": 0.9597, "heart fibroblasts": 0.9713, "breast luminal epithelium": 0.9937, "hepatocyte": 0.9986, "breast basal epithelium": 1.0012, "effector memory cd8 t cells": 1.0014, "colon macrophages": 1.0087, "endometrium epithelium": 1.0199, "monocytes": 1.0207, "b cells": 1.0256, "oligodendrocytes": 1.0266, "colon epithelium": 1.0302, "colon enteroendocrine": 1.0314, "fallopien tubes epithelium": 1.035, "nk cells": 1.0445, "t effector memory cd4": 1.0451, "cd8 t cells": 1.0491, "cardiomyocyte": 1.0541, "gastric fundus epithelium": 1.0551, "cd4 t cells": 1.0563, "small intestine epithelium": 1.0621, "aorta endothelium": 1.0622, "kidney tubular epithelium": 1.0633, "gastric body epithelium": 1.0643, "lung alveolar epithelium": 1.0672, "pancreatic duct": 1.0698, "prostate epithelium": 1.0708, "gastric antrum epithelium": 1.0766, "lung alveolar endothelium": 1.0806, "tonsil pharyngeal epithelium": 1.0835, "tonsil palatine epithelium": 1.0935, "memory b cells": 1.0957, "colon fibroblasts": 1.0982, "kidney glomerular epithelium": 1.1052, "t effector cell cd8": 1.1056, "bladder epithelium": 1.1074, "thyroid epithelium": 1.1164, "lung bronchus epithelium": 1.1187, "adipocytes": 1.1201, "vascular saphenous endothelium": 1.1272, "pancreatic beta": 1.1281, "tongue epithelium": 1.1327, "kidney glomerular endothelium": 1.1414, "kidney tubular endothelium": 1.142, "pancreatic alpha": 1.1459, "pancreas endothelium": 1.1498, "pancreatic delta": 1.1651, "striated muscle": 1.1709, "t central memory cd4": 1.1713}')
R = "Biological_Physics/MethylPhys/chain/Runtime Matrices/Seq_Atlas"
def find(name):
    f = [p for p in glob.glob(f"{R}/**/{name}*", recursive=True) if p.endswith((".parquet", ".json"))]
    return sorted(f, key=len)[0] if f else None
def sp(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])
def test(name, feat):
    common = sorted(set(feat) & set(POS)); a = [POS[t] for t in common]; b = [feat[t] for t in common]
    if len(common) < 8: print(f"::notice::A8 NAMEPLATE {name}: only {len(common)} types"); return None
    r = sp(a, b); random.seed(4); nul = [sp(a, random.sample(b, len(b))) for _ in range(5000)]
    p = float(np.mean([abs(z) >= abs(r) for z in nul])); return dict(rho=round(r, 3), p=round(p, 4), n=len(common))
res = {}
f = find("metaa_seq_reference_v1"); M = pd.read_parquet(f)
M["cell"] = M["cell"].astype(str).str.strip().str.lower()
g = M.groupby("cell")
feats = {"identity mean level": g["level"].mean(), "identity mean disorder": g["disorder"].mean(), "identity mean spread": g["spread"].mean(),
         "identity share held (level >= 0.8)": g["level"].apply(lambda x: float((x >= 0.8).mean())),
         "identity share unmarked (level <= 0.2)": g["level"].apply(lambda x: float((x <= 0.2).mean())),
         "identity disorder at held sites": M[M["level"] >= 0.8].groupby("cell")["disorder"].mean()}
for k, v in feats.items(): res[k] = test(k, v.to_dict())
h = find("held_reference_v1"); meta = None
mj = [p for p in glob.glob(f"{R}/**/held_reference_v1*.json", recursive=True)]
if mj: meta = json.load(open(mj[0]))
if meta and "cells" in meta:
    he = {c.strip().lower(): v.get("held_error") for c, v in meta["cells"].items() if isinstance(v, dict) and v.get("held_error") is not None}
    res["atlas held error (consistency, same quantity)"] = test("atlas held error", he)
    tm = {c.strip().lower(): v.get("tile_median") / v.get("held_error") for c, v in meta["cells"].items() if isinstance(v, dict) and v.get("held_error") and v.get("tile_median")}
    res["tile median over pooled held error"] = test("tile median ratio", tm)
lines = [f"{k}: Spearman {v['rho']:+.3f} p {v['p']:.4f} (n {v['n']})" for k, v in res.items() if v]
print("::notice::A8 NAMEPLATE 2b: " + " | ".join(lines[:4]))
print("::notice::A8 NAMEPLATE 2b cont: " + " | ".join(lines[4:]))
os.makedirs("xout", exist_ok=True); json.dump(res, open("xout/a8_nameplate.json", "w"), indent=1)
