#!/usr/bin/env python3
"""Do the six bone-marrow cells earn their place in healthy whole blood? Solve each of the 24 GSE87571 arrays with and without them;
report residual, neutrophil fraction and where the bone-marrow mass goes. Also the 24 Salas known mixtures for comparison."""
import glob, json, numpy as np, pandas as pd, concurrent.futures as cf, os
from deconv_v2 import DeconvV2
POOL1 = ["astrocytes", "microglia", "oligodendrocyte precursors", "vascular leptomeningeal cells"]
BM = ["CMP (bone marrow)", "GMP (bone marrow)", "HSC (bone marrow)", "L-MPP (bone marrow)", "MEP (bone marrow)", "MPP (bone marrow)"]
base = dict(markers="hybrid", margin=0.10, pair_margin=0.15, per_pair=60, k_near=8, top=600, sigma=0.02)
ATL = pd.read_parquet("/home/ubuntu/data/IAMAtlas_v2.parquet")
D1 = DeconvV2(None, pooled_one_sample=POOL1, drop=["vascular endothelium"], A=ATL, **base)
D0 = DeconvV2(None, pooled_one_sample=POOL1, drop=["vascular endothelium"] + BM, A=ATL, **base)
specs = [("GSE87571", p) for p in sorted(glob.glob("/home/ubuntu/data/v2run01/shards/*.parquet"))]
T = pd.read_csv("salas_mixture_truth.csv")
for g, m in zip(T.gse, T.gsm): specs += [(g + "_mix", glob.glob(f"/home/ubuntu/data/atlas_sources/blood/{g}/shards/{m}_*.parquet")[0])]
rows = []
G = {"CD4": ["cd4 t cells", "naive cd4 t cells", "memory cd4 t cells", "t central memory cd4", "t effector memory cd4", "regulatory t cells"],
     "CD8": ["cd8 t cells", "naive cd8 t cells", "effector memory cd8 t cells", "t effector cell cd8"], "NK": ["nk cells"], "Neu": ["neutrophils"]}
TR = {"CD4": "cd4t", "CD8": "cd8t", "NK": "nk", "Neu": "neu"}
Tt = {m: r for m, r in zip(T.gsm, T.itertuples())}
for s, p in specs:
    b = pd.read_parquet(p).iloc[:, 0]; b.index = b.index.astype(str); gsm = os.path.basename(p).split("_")[0].split(".")[0]
    for aff in (False, True):
        o = D1.deconvolve(b, n_boot=0, affine=aff); fr = o["fractions"]
        row = dict(set=s, gsm=gsm, affine=aff, bm_mass=sum(fr[c] for c in BM), neu=fr["neutrophils"], resid=o["residual_mae"], a=o["scale"]["a"], b=o["scale"]["b"],
                   nonblood=sum(v for c, v in fr.items() if not any(k in c for k in ("t cell", "cd4", "cd8", "b cell", "nk", "mono", "neutro", "eosino", "baso", "regulatory"))) - sum(fr[c] for c in BM))
        if gsm in Tt:
            r = Tt[gsm]; sc = 100.0 if r.gse == "GSE110554" else 1.0
            for g, cs in G.items(): row[g + "_err"] = sum(fr.get(c, 0) for c in cs) - getattr(r, TR[g]) / sc
        rows.append(row)
X = pd.DataFrame(rows); X.to_csv("affine_test.csv", index=False); pd.set_option("display.width", 250)
agg = X.groupby(["set", "affine"]).agg(bm_mass=("bm_mass", "median"), bm_max=("bm_mass", "max"), neu=("neu", "median"), resid=("resid", "median"), a=("a", "median"), b=("b", "median"), nonblood=("nonblood", "median"))
print(agg.round(4).to_string())
M = X[X.set.str.endswith("_mix")]
print(M.groupby("affine")[[c for c in X.columns if c.endswith("_err")]].apply(lambda d: d.abs().mean()).round(4).to_string())
