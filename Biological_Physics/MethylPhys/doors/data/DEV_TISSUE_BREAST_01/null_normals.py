"""DEV-TISSUE-BREAST-01 synthetic/real-normal step 3 (2026-10-11; normal tissue only, no tumour array read). The two readings of
development/sims/breast_tissue_01.py on the 97 calibrated TCGA-BRCA normal-tissue arrays (calib_tcga.py), each read leave-one-out against the
other 96: PLAIN Met-A = mean H(beta) at the luminal identity sites / median over the others; COMP Met-A = (mean H(beta) / mean H(composition-
predicted beta)) / median of the same over the others. Panel and identity sites from the FULL atlas v2 posteriors (atlas/tools/extract_posterior.py;
fibroblast = mean of colon and heart fibroblasts; T = mean of CD4 and CD8). Also: each normal's fitted composition. Writes null_normals_output.csv
and prints the spread and the healthy 95th percentile (the per-tumour bar candidate). Usage: python3 null_normals.py POSTDIR BETADIR"""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import nnls
POST, BD = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n: pd.read_parquet(os.path.join(POST, f"atlas_{n.replace(' ', '_')}_posterior.parquet")).set_index("cpg")
names = {"luminal": ["breast luminal epithelium"], "basal": ["breast basal epithelium"], "adipocytes": ["adipocytes"], "fibroblast": ["colon fibroblasts", "heart fibroblasts"],
         "endothelium": ["vascular endothelium"], "macrophage": ["colon macrophages"], "T": ["cd4 t cells", "cd8 t cells"], "B": ["b cells"]}
P = {k: [R(n) for n in v] for k, v in names.items()}
idx = P["luminal"][0].index
for v in P.values():
    for x in v: idx = idx.intersection(x.index)
Mu = pd.DataFrame({k: np.mean([x.loc[idx, "mean"].values for x in v], 0) for k, v in P.items()}, index=idx)
L = P["luminal"][0].loc[idx]
files = sorted(f for f in os.listdir(BD) if f.endswith(".parquet")); Bt = {f[:-8]: pd.read_parquet(os.path.join(BD, f)).beta for f in files}
for k in Bt: Bt[k].index = Bt[k].index.astype(str)
common = idx
for b_ in Bt.values(): common = common.intersection(b_.index[b_.notna().values])   # measured on every normal array
Mu, L = Mu.loc[common], L.loc[common]
ok_ = Mu.notna().all(1) & L["mean"].notna() & L["sd"].notna(); common = common[ok_.values]; Mu, L = Mu.loc[common], L.loc[common]   # loci every panel cell measures
hi = L[(L["mean"].between(.75, .95)) & (L.sd <= .05)].sort_values("sd").index[:3000]; lo = L[(L["mean"].between(.05, .25)) & (L.sd <= .05)].sort_values("sd").index[:3000]
S = hi.union(lo); rest = common.difference(S); inf = Mu.loc[rest].std(1).sort_values(ascending=False).index[:8000]
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
rows = []
for k, b in Bt.items():
    b = b.loc[common]; w, _ = nnls(Mu.loc[inf].values, b.loc[inf].values); w = w / w.sum(); bp = Mu.values @ w
    rows.append(dict(sample=k, plain=float(H(b.loc[S].values).mean()), comp=float(H(b.loc[S].values).mean() / H(pd.Series(bp, index=common).loc[S].values).mean()), **{f"f_{c}": float(x) for c, x in zip(Mu.columns, w)}))
T = pd.DataFrame(rows)
for c in ("plain", "comp"): T[c + "_rel"] = [v / np.median(np.delete(T[c].values, i)) for i, v in enumerate(T[c].values)]
T.to_csv(os.path.join(HERE, "null_normals_output.csv"), index=False)
print(f"arrays {len(T)} | loci {len(common)} | identity sites {len(hi)} + {len(lo)}")
for c in ("plain_rel", "comp_rel"): print(f"{c}: SD {T[c].std(ddof=1):.4f}, range {T[c].min():.4f}-{T[c].max():.4f}, 95th percentile {T[c].quantile(.95):.4f}")
print("fitted composition, median:", T[[c for c in T.columns if c.startswith('f_')]].median().round(3).to_dict())
