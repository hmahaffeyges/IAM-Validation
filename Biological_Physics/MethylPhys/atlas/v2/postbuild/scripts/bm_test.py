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
for s, p in specs:
    b = pd.read_parquet(p).iloc[:, 0]; b.index = b.index.astype(str)
    # compare residuals on the SAME loci: the union markers of the with-BM model, solved both ways
    o1 = D1.deconvolve(b, n_boot=0); o0 = D0.deconvolve(b, n_boot=0)
    y = b.reindex(D1.loci).to_numpy(float); m = np.isfinite(y)
    f1 = np.array([o1["fractions"][c] for c in D1.cells]); r1 = np.abs(y[m] - D1.mu[m] @ f1).mean()
    idx0 = [D1.cells.index(c) for c in D0.cells]; f0 = np.zeros(len(D1.cells)); f0[idx0] = [o0["fractions"][c] for c in D0.cells]
    r0 = np.abs(y[m] - D1.mu[m] @ f0).mean()
    bm = sum(o1["fractions"][c] for c in BM)
    rows.append(dict(set=s, gsm=os.path.basename(p)[:10], bm_mass=bm, gmp=o1["fractions"]["GMP (bone marrow)"],
                     neu_with=o1["fractions"]["neutrophils"], neu_without=o0["fractions"]["neutrophils"],
                     resid_with=r1, resid_without=r0, resid_increase_pct=100 * (r0 / r1 - 1),
                     platform_loci=int(m.sum())))
X = pd.DataFrame(rows); X.to_csv("bm_test.csv", index=False)
pd.set_option("display.width", 220)
print(X.groupby("set")[["bm_mass", "gmp", "neu_with", "neu_without", "resid_with", "resid_without", "resid_increase_pct", "platform_loci"]].median().round(4).to_string())
print(X[X.set == "GSE87571"][["gsm", "bm_mass", "neu_with", "neu_without", "resid_increase_pct"]].round(4).to_string(index=False))
