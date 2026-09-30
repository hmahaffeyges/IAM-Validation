#!/usr/bin/env python3
"""Development sweep of the v2 solver on the 24 Salas known mixtures (author 2026-09-29: build, fix, iterate)."""
import json, glob, numpy as np, pandas as pd, concurrent.futures as cf, itertools, time
from deconv_v2 import DeconvV2
POOL1 = ["astrocytes", "microglia", "oligodendrocyte precursors", "vascular leptomeningeal cells"]
import os, json, urllib.request
P = "/home/ubuntu/data/IAMAtlas_v2.parquet"
if not os.path.exists(P): urllib.request.urlretrieve(json.load(open("atlas_url.json"))["atlas_get"], P)
ATL = pd.read_parquet(P)
T = pd.read_csv("salas_mixture_truth.csv"); T["bas"] = np.where(T.gse == "GSE167998", T.gran - T.neu - T.eos, np.nan)
R = "/home/ubuntu/data/atlas_sources/blood"
BETA = {}
for g, m in zip(T.gse, T.gsm):
    b = pd.read_parquet(glob.glob(f"{R}/{g}/shards/{m}_*.parquet")[0]).iloc[:, 0]; b.index = b.index.astype(str); BETA[m] = b
G = {"CD4": ["cd4 t cells", "naive cd4 t cells", "memory cd4 t cells", "t central memory cd4", "t effector memory cd4", "regulatory t cells"],
     "CD8": ["cd8 t cells", "naive cd8 t cells", "effector memory cd8 t cells", "t effector cell cd8"], "NK": ["nk cells"],
     "B": ["b cells", "naive b cells", "memory b cells"], "Mono": ["monocytes"], "Neu": ["neutrophils"], "Eos": ["eosinophils"], "Baso": ["basophils"]}
TR = {"CD4": "cd4t", "CD8": "cd8t", "NK": "nk", "B": "bcell", "Mono": "mono", "Neu": "neu", "Eos": "eos", "Baso": "bas"}
blood = set(sum(G.values(), []))
PARENTS = ["cd4 t cells", "cd8 t cells", "b cells"]
V = []
for drop in ([], PARENTS):
    for mk, mg, top in (("nearest", 0.20, 200), ("nearest", 0.15, 300), ("nearest", 0.10, 400), ("pairwise", 0.10, None), ("pairwise", 0.15, None)):
        for sg in (0.02, 0.05):
            V.append(dict(drop=drop, markers=mk, margin=mg if mk == "nearest" else 0.2, pair_margin=mg, top=top or 200, sigma=sg))
def run(cfg):
    t0 = time.time()
    D = DeconvV2(None, pooled_one_sample=POOL1, drop=cfg["drop"], markers=cfg["markers"], margin=cfg["margin"], pair_margin=cfg["pair_margin"], top=cfg["top"], sigma=cfg["sigma"], A=ATL)
    rows = []
    for r in T.itertuples():
        o = D.deconvolve(BETA[r.gsm], n_boot=0); fr = o["fractions"]; sc = 100.0 if r.gse == "GSE110554" else 1.0
        row = dict(gsm=r.gsm, gse=r.gse, non_blood=sum(v for c, v in fr.items() if c not in blood))
        for g, cs in G.items():
            tv = getattr(r, TR[g]); row[f"{g}_err"] = sum(fr.get(c, 0) for c in cs) - (tv / sc if pd.notna(tv) else np.nan)
        rows.append(row)
    X = pd.DataFrame(rows)
    return dict(cfg=cfg, n_markers=D.meta["n_markers"], mae={g: round(float(X[f"{g}_err"].abs().mean()), 4) for g in G},
                bias={g: round(float(X[f"{g}_err"].mean()), 4) for g in ("CD4", "CD8", "Neu")}, non_blood_med=round(float(X.non_blood.median()), 4), s=round(time.time() - t0))
with cf.ProcessPoolExecutor(10) as ex: out = list(ex.map(run, V))
out.sort(key=lambda o: max(o["mae"]["CD4"], o["mae"]["CD8"], o["mae"]["NK"]))
json.dump(out, open("sweep.json", "w"), indent=1)
for o in out: print(("drop" if o["cfg"]["drop"] else "keep"), o["cfg"]["markers"], o["cfg"]["pair_margin"] if o["cfg"]["markers"]=="pairwise" else o["cfg"]["margin"], o["cfg"]["top"], o["cfg"]["sigma"], "| markers", o["n_markers"], "| MAE", o["mae"], "| bias", o["bias"], "| nonblood", o["non_blood_med"], flush=True)
