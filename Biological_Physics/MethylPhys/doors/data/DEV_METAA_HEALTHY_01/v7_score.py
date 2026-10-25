"""DEV-METAA-HEALTHY-01 step 3 version 7, the scorer (written 2026-10-24 after version 6 and its side diagnostic were read, disclosed: the
diagnostic showed each array's response to the molecules is a curve, raised at the bottom and flat at the top, not the straight line
version 6 drew). Each array's own response curve, measured on that array against the cell's own molecules away from the identity sites:
at atlas sites that are not Met-A's neutrophil identity sites and are read by every donor, the sites are binned by the donors' mean molecule
level (fixed edges below), and in each bin the array's mean reading and the molecules' mean level make one point of the curve. At each
identity site the array should read the curve at that site's molecule level (straight segments between points; outside the first and last
points, the end values). The array's derived mark is the mean entropy over the identity sites; Met-A against it is the array's own mean
entropy there over the derived mark. Binned means only; nothing fitted; the identity sites take no part in the curve.
Bar 1 (version 1's): all six commissioned arrays read Normal (0.95-1.05).  Usage: python3 v7_score.py READINGS_DIR OUT.txt"""
import sys, os, glob, json, numpy as np, pandas as pd
EDGES = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0]   # stated before scoring
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain")
F = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]; ID = pd.Index(F["sites"])
rd = lambda f: (lambda v: v.set_axis(v.index.astype(str)))(pd.read_parquet(f).iloc[:, 0].astype("float64"))
Hb = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
S = pd.concat({os.path.basename(f)[11:-11]: rd(f) for f in glob.glob(os.path.join(sys.argv[1], "**", "sequencing_*_v6.parquet"), recursive=True)}, axis=1)
A = {os.path.basename(f)[6:-11]: rd(f) for f in glob.glob(os.path.join(sys.argv[1], "**", "array_*_v6.parquet"), recursive=True)}
m = S.dropna().mean(1); off = m.index.difference(ID); mid = m.reindex(ID).dropna(); b = pd.cut(m[off], EDGES, include_lowest=True)
L = [f"donors {', '.join(sorted(S.columns))} | arrays {len(A)} | curve sites (not identity sites, every donor read) {len(off)} | identity sites with molecules {len(mid)} of {len(ID)}",
     "array | curve at molecules 0.005 / 0.94 | mean entropy at identity sites | derived mark | Met-A against it"]; n = 0
for g, x in sorted(A.items()):
    j = off.intersection(x.dropna().index); P = pd.DataFrame({"m": m[j], "a": x[j], "b": b.reindex(j)}).groupby("b", observed=True).mean().sort_values("m")
    curve = lambda v: np.interp(v, P.m.to_numpy(), P.a.to_numpy()); k = mid.index.intersection(x.dropna().index)
    d = float(Hb(curve(mid[k].to_numpy())).mean()); h = float(Hb(x[k]).mean()); a = h / d; n += 0.95 <= a <= 1.05
    L.append(f"{g} | {float(curve(0.005)):.4f} / {float(curve(0.94)):.4f} | {h:.5f} | {d:.5f} | {a:.4f}")
L.append(f"bar 1 (all six Normal, 0.95-1.05, each against its own derived mark): {n} of {len(A)}: {'met' if n == len(A) == 6 else 'not met'}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
