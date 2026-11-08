"""Scores the share synthetic (share_synthetic.py, written before it ran). For each cell and share: how far the cell's own reading moved with a
loss in that cell (own) and with the same loss in every other cell at the same sites (others), and their ratio; beside it the ratio the DNA
alone predicts if the reading is the sample's disorder at the cell's identity sites, share / (1 - share). Bar 1 as written: for each of the nine
cells, at 50 % its own loss moves its reading more than the others' loss does. Usage: python3 share_synthetic_score.py FOLDER OUT.txt"""
import sys, glob, os, json, pandas as pd
R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(sys.argv[1], "share_synthetic_c*_s*.json")))]
X = pd.DataFrame([dict(cell=x["cell"], share=x["share"], own=x["own_move"], others=x["others_move"]) for x in R]); X["ratio"] = X.own / X.others; X["dna"] = X.share / (1 - X.share)
L = [f"share synthetic: {len(X)} parts (9 cells x 8 shares), 3 arrays each, every array by the one command", "cell, share: own move | others' move | ratio | share/(1-share)"]
for (c, s), g in X.groupby(["cell", "share"]): r = g.iloc[0]; L.append(f"  {c[:28]:28} {s:6.3f}: {r.own:+.4f} | {r.others:+.4f} | {r.ratio:6.3f} | {r.dna:6.3f}")
h = X[X.share == 0.5]; met = int((h.own > h.others).sum())
L.append(f"at 50 %: ratio {h.ratio.min():.3f} to {h.ratio.max():.3f}; own loss larger for {met} of {len(h)} cells")
L.append(f"all shares: ratio over share/(1-share), median {(X.ratio / X.dna).median():.3f} (5 to 95 % {(X.ratio / X.dna).quantile(.05):.3f} to {(X.ratio / X.dna).quantile(.95):.3f})")
L.append(f"bar 1 (each of the nine cells, at 50 % own loss moves the reading more than the others' loss): {'met' if met == 9 else 'not met'} ({met} of 9)")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L[-4:]))
