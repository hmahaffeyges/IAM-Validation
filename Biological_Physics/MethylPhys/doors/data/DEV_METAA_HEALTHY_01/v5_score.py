"""DEV-METAA-HEALTHY-01 step 3 version 5, the scorer (written and committed 2026-10-24 before any BLUEPRINT file is read).
1. The cause split (stated before reading): the purified neutrophils' mean sequencing level at the unmarked identity sites against the
   granulocytes' 0.0114 (version 3). At least 0.020: the granulocyte stand-in accounts for part of the gap. Below 0.015: it does not, and the
   rest is the probe offset being larger in neutrophils. Between: undecided.
2. Version 5's mark: at each identity site, level = the atlas's inverse transfer of the donors' mean sequencing level there + version 4's
   per-site difference measured on other blood cells (step3_04/v4_offsets.csv), clipped; the mark is the mean entropy over the sites. Per
   site, where version 4 used one level per side (disclosed). Nothing fitted; no neutrophil array used.
Bar 1 (version 1's): all six commissioned arrays read Normal (0.95-1.05) against the derived mark.  Usage: python3 v5_score.py READINGS_DIR OUT.txt"""
import sys, os, glob, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import v4_score as V
X = pd.concat([pd.read_csv(f, index_col=0) for f in glob.glob(os.path.join(sys.argv[1], "**", "*_v5.csv"), recursive=True)])
s = X.groupby(level=0).value.mean().reindex(V.side.index); O = pd.read_csv(os.path.join(H, "step3_04", "v4_offsets.csv"), index_col=0).d.reindex(V.side.index)
ok = s.notna(); lvl = pd.Series(V.inv(s[ok].to_numpy()), index=s[ok].index).add(O[ok]).clip(1e-6, 1 - 1e-6)
Hb = lambda b: -(b * np.log2(b) + (1 - b) * np.log2(1 - b)); mark = float(Hb(lvl).mean()); side = V.side[ok]
L = [f"donors read: {', '.join(sorted(X.donor.unique()))} | identity sites with a value {int(ok.sum())} of {len(V.side)}"]
for k in ("held", "absent"):
    m = side == k; L.append(f"{k}: sequencing level {s[ok][m].mean():.4f} (granulocytes {'0.9344' if k == 'held' else '0.0114'}; per donor {', '.join(f'{x.value[x.index.isin(side.index[m])].mean():.4f}' for _, x in X.groupby('donor'))}) | version 5 array-scale level {lvl[m].mean():.4f} (neutrophil arrays {V.st[V.side == k].mean():.4f}) | entropy {Hb(lvl[m]).mean():.4f}")
a = s[ok][side == "absent"].mean(); L.append(f"cause split: purified neutrophils at the unmarked sites {a:.4f}: " + ("the granulocyte stand-in accounts for part of the gap" if a >= 0.020 else "the stand-in does not account for it; the probe offset is larger in neutrophils" if a < 0.015 else "undecided"))
L += [f"derived mark {mark:.5f} bits | measured mark {V.F['floor']:.5f} bits (6 arrays) | derived / measured {mark / V.F['floor']:.4f}", "array | Met-A against the measured mark (frozen sites) | against the derived mark"]
LOO = pd.read_csv(os.path.join(H, "..", "..", "..", "chain", "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3_loo.csv")); n = 0
for r in LOO.itertuples():
    v = r.A_loo_frozen_sites * V.F["floor"] / mark; n += 0.95 <= v <= 1.05; L.append(f"{r.ref} | {r.A_loo_frozen_sites:.4f} | {v:.4f}")
L.append(f"bar 1 (all six Normal, 0.95-1.05, against the derived mark): {n} of {len(LOO)}: {'met' if n == len(LOO) == 6 else 'not met'}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
