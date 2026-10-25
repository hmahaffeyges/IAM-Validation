"""DEV-METAA-HEALTHY-01 step 3 version 7, selection diagnostic (added after version 7 was scored on the development donors; disclosed;
nothing scored). Met-A's identity sites were chosen on the purified neutrophil arrays themselves (mean 0.75-0.95 or 0.05-0.25, spread at most
0.05). Question: do sites chosen that way read above each array's own response curve, whatever the cell's molecules say? The same array rule
is applied to the atlas sites that are not identity sites (the six arrays' mean and spread), and each group's mean residual (array reading
minus the array's curve at the site's molecule level) is set beside the identity sites' and all sites'.  Usage: python3 v7_select.py DIR OUT"""
import sys, os, glob, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain")
F = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]; ID = pd.Index(F["sites"])
EDGES = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]   # version 7's
rd = lambda f: (lambda v: v.set_axis(v.index.astype(str)))(pd.read_parquet(f).iloc[:, 0].astype("float64"))
S = pd.concat({os.path.basename(f)[11:-11]: rd(f) for f in glob.glob(os.path.join(sys.argv[1], "**", "sequencing_*_v6.parquet"), recursive=True)}, axis=1)
A = pd.concat({os.path.basename(f)[6:-11]: rd(f) for f in glob.glob(os.path.join(sys.argv[1], "**", "array_*_v6.parquet"), recursive=True)}, axis=1)
m = S.dropna().mean(1); j = m.index.intersection(A.dropna().index); off = j.difference(ID)
R = pd.DataFrame(index=j)
for g in A.columns:
    P = pd.DataFrame({"m": m[off], "a": A[g][off], "b": pd.cut(m[off], EDGES, include_lowest=True)}).groupby("b", observed=True).mean().sort_values("m")
    R[g] = A[g][j] - np.interp(m[j].to_numpy(), P.m.to_numpy(), P.a.to_numpy())
r = R.mean(1); mu, sd = A.loc[j].mean(1), A.loc[j].std(1)
L = ["group | sites | molecules (mean) | arrays (mean) | residual above the curve (mean)"]
for nm, side in (("marked", (mu >= 0.5)), ("unmarked", (mu < 0.5))):
    rule = ((mu >= 0.75) & (mu <= 0.95) if nm == "marked" else (mu >= 0.05) & (mu <= 0.25)) & (sd <= 0.05)
    for lab, k in ((f"{nm} identity sites", j.intersection(ID)[side.reindex(j.intersection(ID)).to_numpy()]),
                   (f"{nm}: other sites passing Met-A's array rule", off[rule.reindex(off).to_numpy()]),
                   (f"{nm}: all other sites at the identity sites' molecule levels", off[(side.reindex(off) & ((m[off] >= 0.9) if nm == "marked" else (m[off] <= 0.02))).to_numpy()])):
        L.append(f"{lab} | {len(k)} | {m[k].mean():.4f} | {mu[k].mean():.4f} | {r[k].mean():+.4f}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
