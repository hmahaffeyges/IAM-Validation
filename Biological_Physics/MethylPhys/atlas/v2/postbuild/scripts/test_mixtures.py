#!/usr/bin/env python3
import json, sys, numpy as np, pandas as pd, concurrent.futures as cf
from deconv_v2 import DeconvV2
POOL1 = ["astrocytes", "microglia", "oligodendrocyte precursors", "vascular leptomeningeal cells"]
D = DeconvV2("/home/ubuntu/data/atlas_pack/wide/IAMAtlas_v2.parquet", pooled_one_sample=POOL1)
print(json.dumps(D.meta), flush=True)
T = pd.read_csv("salas_mixture_truth.csv")
# Salas 2022 reports granulocytes (gran) = neutrophils + eosinophils + basophils; basophils are not listed separately, so bas = gran - neu - eos
T["bas"] = np.where(T.gse == "GSE167998", T.gran - T.neu - T.eos, np.nan)
chk = T[T.gse == "GSE167998"][["cd4t","cd8t","bcell","nk","mono","neu","eos","bas"]].sum(axis=1); print("GSE167998 truth row sums", round(chk.min(),3), round(chk.max(),3), flush=True)
R = "/home/ubuntu/data/atlas_sources/blood"
def run(a):
    gse, gsm = a
    import glob; p = glob.glob(f"{R}/{gse}/shards/{gsm}_*.parquet"); assert len(p) == 1, (gsm, p)
    b = pd.read_parquet(p[0]).iloc[:, 0]; b.index = b.index.astype(str)
    return gsm, D.deconvolve(b)
with cf.ProcessPoolExecutor(24) as ex: out = dict(ex.map(run, list(zip(T.gse, T.gsm))))
G = {"CD4": ["cd4 t cells", "naive cd4 t cells", "memory cd4 t cells", "t central memory cd4", "t effector memory cd4", "regulatory t cells"],
     "CD8": ["cd8 t cells", "naive cd8 t cells", "effector memory cd8 t cells", "t effector cell cd8"], "NK": ["nk cells"],
     "B": ["b cells", "naive b cells", "memory b cells"], "Mono": ["monocytes"], "Neu": ["neutrophils"], "Eos": ["eosinophils"], "Baso": ["basophils"],
     "CD4naive": ["naive cd4 t cells"], "CD4mem": ["memory cd4 t cells", "t central memory cd4", "t effector memory cd4"], "Treg": ["regulatory t cells"],
     "Bnaive": ["naive b cells"], "Bmem": ["memory b cells"]}
TR = {"CD4": "cd4t", "CD8": "cd8t", "NK": "nk", "B": "bcell", "Mono": "mono", "Neu": "neu", "Eos": "eos", "Baso": "bas",
      "CD4naive": "cd4nv", "CD4mem": "cd4mem", "Treg": "treg", "Bnaive": "bnv", "Bmem": "bmem"}
blood = set(sum([G[k] for k in ("CD4", "CD8", "NK", "B", "Mono", "Neu", "Eos", "Baso")], []))
rows = []
for r in T.itertuples():
    o = out[r.gsm]; fr = o["fractions"]; scale = 100.0 if r.gse == "GSE110554" else 1.0
    row = dict(gsm=r.gsm, gse=r.gse, residual_mae=o["residual_mae"], non_blood=sum(v for c, v in fr.items() if c not in blood),
               n_present=sum(o["present"].values()), present_not_blood=[c for c, p in o["present"].items() if p and c not in blood])
    for g, cs in G.items():
        tv = getattr(r, TR[g], np.nan) if TR[g] in T.columns else np.nan
        row[f"{g}_est"] = sum(fr[c] for c in cs); row[f"{g}_true"] = (tv / scale) if pd.notna(tv) else np.nan
    rows.append(row)
X = pd.DataFrame(rows)
err = {g: float((X[f"{g}_est"] - X[f"{g}_true"]).abs().mean()) for g in G if X[f"{g}_true"].notna().any()}
res = dict(meta=D.meta, A5_MAE={g: round(err[g], 4) for g in ("CD4", "CD8", "NK")},
           A5_PASS=bool(all(err[g] <= 0.015 for g in ("CD4", "CD8", "NK"))), MAE_all={g: round(v, 4) for g, v in err.items()},
           non_blood_fraction_median=float(X.non_blood.median()), non_blood_fraction_max=float(X.non_blood.max()),
           arrays_with_nonblood_present=int((X.present_not_blood.str.len() > 0).sum()),
           not_separable=[c for c, s in D.sep.items() if not s["separable"]], separability=D.sep,
           mae_by_study={s: {g: round(float((X[X.gse == s][f"{g}_est"] - X[X.gse == s][f"{g}_true"]).abs().mean()), 4) for g in ("CD4", "CD8", "NK", "B", "Mono", "Neu")} for s in X.gse.unique()})
X.to_csv("deconv_v2_mixtures.csv", index=False); json.dump(res, open("deconv_v2_result.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "separability"}, indent=1))
