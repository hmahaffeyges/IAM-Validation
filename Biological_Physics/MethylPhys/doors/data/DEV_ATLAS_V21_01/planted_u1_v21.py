"""Planted test of score_u1_v21.py (synthetic first): fake blocks with known draws, arrays drawn from the predictive distribution (must pass),
the same with a normal laboratory shift (must pass), arrays shifted by 0.10 (must fail), a cell with no arrays (not assessed), a withheld array and
a cell not in the atlas (listed). Everything runs through the scorer's own functions and chain Stage A21."""
import os, sys, json, tempfile, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import score_u1_v21 as U
rng = np.random.default_rng(7); W = tempfile.mkdtemp(); B = os.path.join(W, "blocks"); S = os.path.join(W, "shards"); os.makedirs(B); os.makedirs(S)
cells = ["alpha", "beta", "gamma"]; mu = {}; sd = 0.03
for b in range(2):
    cp = [f"cg{b}{i:05d}" for i in range(3000)]; m = rng.uniform(0.05, 0.95, (3, 3000))
    for k, c in enumerate(cells): mu.update({(c, x): m[k, j] for j, x in enumerate(cp)})
    d = np.clip(m[None] + rng.normal(0, 0.004, (100, 3, 3000)), 0, 1).astype(np.float16)
    np.savez(os.path.join(B, f"block_{b:05d}_draws.npz"), draws=d, sd_draws=np.full((100, 3, 3000), sd, np.float16), cells=np.array(cells), cpg=np.array(cp, dtype=object), chain=np.zeros(100, np.int8))
allcp = sorted({x for _, x in mu})
def arr(name, c, shift=0.0, lab=0.0):
    y = np.array([mu[(c, x)] for x in allcp]) + lab + shift + rng.normal(0, np.sqrt(sd**2 + 0.0065**2 + 1e-4), len(allcp))
    pd.DataFrame({name: np.clip(y, 0, 1)}, index=pd.Index(allcp, name="IlmnID")).to_parquet(os.path.join(S, name + ".parquet"))
rows = []
for i in range(40): arr(f"a{i}", "alpha"); rows.append((f"a{i}", "L1", "alpha cells", "450K"))
for i in range(40): arr(f"b{i}", "beta", lab=rng.normal(0, 0.0065)); rows.append((f"b{i}", "L2", "beta cells", "EPIC v1"))
for i in range(20): arr(f"g{i}", "gamma", shift=0.10); rows.append((f"g{i}", "L3", "gamma cells", "450K"))
rows += [("w1", "L1", "alpha cells", "450K"), ("x1", "L4", "sperm", "450K")]
T = pd.DataFrame(rows, columns=["sample", "series", "cell_type", "platform"])
cmap = {"alpha cells": "alpha", "beta cells": "beta", "gamma cells": "gamma", "sperm": None}
IV, lab = U.intervals(B, cells, 2); X, V = U.score(T, cmap, IV, S)
want = {"alpha | 450K": "COMMISSIONED", "beta | EPIC": "COMMISSIONED", "gamma | 450K": "FAILED"}
ok = all(V[k]["verdict"] == v for k, v in want.items()) and "alpha | EPIC" not in V
ok &= set(X[X.status != "scored"].status) == {"listed: withheld by Stage 1", "listed: cell not in v2.1"}
L = [f"{k}: {v['verdict']} (want {want[k]}), share {v['share']:.3f}, median coverage {v['median_coverage']:.3f}" for k, v in V.items()]
L += [f"listed: {r['sample']} - {r['status']}" for _, r in X[X.status != "scored"].iterrows()]
L += [f"lab spread used {lab:.4f}", f"PLANTED U1 TEST: {'PASSED' if ok else 'FAILED'}"]
open(os.path.join(H, "planted_u1_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
