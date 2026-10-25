"""DEV-METAA-HEALTHY-01 step 3 version 6, side diagnostic (added after version 6 was scored; disclosed; nothing scored). For each array:
its mean reading at the marked and the unmarked identity sites, beside what version 6's line predicts there, and the arrays' mean reading
binned by the molecules' level (all atlas sites read by every donor), to show the shape of the array's response.
Usage: python3 v6_side.py READINGS_DIR OUT.txt"""
import sys, os, glob, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain")
F = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]; ID = pd.Index(F["sites"])
rd = lambda f: (lambda v: v.set_axis(v.index.astype(str)))(pd.read_parquet(f).iloc[:, 0].astype("float64"))
S = pd.concat({os.path.basename(f)[11:-11]: rd(f) for f in glob.glob(os.path.join(sys.argv[1], "**", "sequencing_*_v6.parquet"), recursive=True)}, axis=1)
A = pd.concat({os.path.basename(f)[6:-11]: rd(f) for f in glob.glob(os.path.join(sys.argv[1], "**", "array_*_v6.parquet"), recursive=True)}, axis=1)
m = S.dropna().mean(1); off = m.index.difference(ID); bare, full = off[(m[off] <= 0.01).to_numpy()], off[(m[off] >= 0.99).to_numpy()]
mid = m.reindex(ID).dropna(); side = pd.Series(np.where(mid >= 0.5, "marked", "unmarked"), index=mid.index)
am = A.mean(1); f0, f1 = am.reindex(bare).mean(), am.reindex(full).mean(); m0, m1 = m[bare].mean(), m[full].mean()
L = [f"arrays averaged ({A.shape[1]}) | F0 {f0:.4f}, F1 {f1:.4f} | molecules at the anchors {m0:.4f}, {m1:.4f}", "side | identity sites | molecules | arrays read | version 6 line predicts"]
for s_ in ("marked", "unmarked"):
    k = side.index[side == s_]; L.append(f"{s_} | {len(k)} | {mid[k].mean():.4f} | {am.reindex(k).mean():.4f} | {(f0 + (f1 - f0) * (mid[k] - m0) / (m1 - m0)).mean():.4f}")
L.append("molecule level bin | sites | arrays read (mean)")
j = m.index.intersection(am.dropna().index); b = pd.cut(m[j], [0, 0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0], include_lowest=True)
for k, g in am[j].groupby(b, observed=True): L.append(f"{k} | {len(g)} | {g.mean():.4f}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
