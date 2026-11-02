"""The IAM Sequencing Atlas's noise maps against their shuffled null on the sphere: summary of noise_null.csv (run_sample.py --build
seq-noise-null). Usage: python3 noise_null_summary.py OUT.txt"""
import os, sys, ast, numpy as np, pandas as pd
D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_null.csv")); D["n"] = D.bands_above_null.apply(lambda x: len(ast.literal_eval(x)))
L = []
for m, g in D.groupby("map"):
    R = np.array([ast.literal_eval(x) for x in g.ratio]); L.append(f"{m}: {len(g)} cells; at most one band above its null: {(g.n <= 1).mean():.3f}; "
                                                                  f"median power over the null by band (multipoles 2-8 ... 192-255): {' '.join(f'{x:.3f}' for x in np.median(R, 0))}")
g = D[D["map"].str.startswith("map1")].copy(); g["low"] = [ast.literal_eval(x)[0] for x in g.ratio]; t = g.sort_values("low").tail(3)
L.append("map1, largest power at multipoles 2-8: " + "; ".join(f"{c} {v:.3f}" for c, v in zip(t.cell, t.low)))
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
