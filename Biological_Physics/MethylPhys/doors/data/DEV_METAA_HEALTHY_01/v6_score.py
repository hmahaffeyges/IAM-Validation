"""DEV-METAA-HEALTHY-01 step 3 version 6, the scorer (written and committed 2026-10-24 before the readings exist). Built on version 5 (purified
neutrophils are nearly bare at Met-A's unmarked identity sites, about 0.005, where the arrays read about 0.07; blur and purity ruled out).
Each array's own two-point line, measured on that array against the cell's own molecules, away from the identity sites:
  F0 = the array's mean reading at sites the neutrophil's molecules hold bare (mean of the development donors at most 0.01),
  F1 = its mean reading at sites they hold fully marked (at least 0.99),
both over atlas sites that are not Met-A's neutrophil identity sites and have at least 10 reads in every donor. At each identity site the
level that array should read is the straight line through the two anchor points (m0, F0) and (m1, F1), m0 and m1 the
donors' mean molecule levels at those anchors (the first synthetic run used F0 + (F1 - F0) m, which ignored that bare anchors are not exactly
0; corrected before any real reading, disclosed): F0 + (F1 - F0) (m - m0) / (m1 - m0), m the donors' mean molecule level there; the array's derived mark is the mean entropy over
the identity sites; its Met-A against it is its own mean entropy at those sites (the frozen ones) over the derived mark. Nothing fitted: two
means on the array's own sites; the identity sites take no part in F0 and F1.
Bar 1 (version 1's): all six commissioned arrays read Normal (0.95-1.05).  Usage: python3 v6_score.py READINGS_DIR OUT.txt"""
import sys, os, glob, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "chain")
F = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]; ID = pd.Index(F["sites"])
rd = lambda f: (lambda v: v.set_axis(v.index.astype(str)))(pd.read_parquet(f).iloc[:, 0].astype("float64"))
Hb = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
S = pd.concat({os.path.basename(f)[11:-11]: rd(f) for f in glob.glob(os.path.join(sys.argv[1], "**", "sequencing_*_v6.parquet"), recursive=True)}, axis=1)
A = {os.path.basename(f)[6:-11]: rd(f) for f in glob.glob(os.path.join(sys.argv[1], "**", "array_*_v6.parquet"), recursive=True)}
m = S.dropna().mean(1); off = m.index.difference(ID); bare, full = off[(m[off] <= 0.01).to_numpy()], off[(m[off] >= 0.99).to_numpy()]; mid = m.reindex(ID).dropna()
L = [f"donors {', '.join(S.columns)} | arrays {len(A)} | anchor sites (not identity sites, every donor read): bare {len(bare)}, fully marked {len(full)} | identity sites with molecules {len(mid)} of {len(ID)}"]
L.append("array | F0 (bare) | F1 (fully marked) | mean entropy at identity sites | derived mark | Met-A against it"); n = 0
for g, x in sorted(A.items()):
    f0, f1 = float(x.reindex(bare).mean()), float(x.reindex(full).mean()); k = mid.index.intersection(x.dropna().index)
    m0, m1 = float(m[bare].mean()), float(m[full].mean()); d = float(Hb(f0 + (f1 - f0) * (mid[k] - m0) / (m1 - m0)).mean()); h = float(Hb(x[k]).mean()); a = h / d; n += 0.95 <= a <= 1.05
    L.append(f"{g} | {f0:.4f} | {f1:.4f} | {h:.5f} | {d:.5f} | {a:.4f}")
L.append(f"bar 1 (all six Normal, 0.95-1.05, each against its own derived mark): {n} of {len(A)}: {'met' if n == len(A) == 6 else 'not met'}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
