"""Met-A against the IAM Sequencing Atlas, healthy test: the readings by fifths of depth and within each cell (added after the test was scored,
disclosed; nothing scored here). Reads metaa_seqatlas_readings.csv, writes metaa_seqatlas_by_depth.txt. Usage: python3 metaa_seqatlas_by_depth.py"""
import os, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); d = pd.read_csv(os.path.join(H, "metaa_seqatlas_readings.csv")); d = d[d.A.notna()].copy()
d["fifth"] = pd.qcut(d.depth, 5, labels=False); L = ["by fifths of depth (shallowest first): fifth | depth range | samples | median Met-A"]
for q, g in d.groupby("fifth"): L.append(f"{q + 1} | {int(g.depth.min()):,} to {int(g.depth.max()):,} | {len(g)} | {g.A.median():.3f}")
d["dev"] = d.A - d.groupby("cell").A.transform("median"); d["rel"] = d.depth / d.groupby("cell").depth.transform("median")
c = d.groupby("cell").agg(A=("A", "median"), depth=("depth", "median"))
L += [f"within a cell, Met-A minus the cell's median against depth over the cell's median depth: rho {spearmanr(d.dev, d.rel).correlation:+.3f} over {len(d)}",
      f"between cells, each cell's median Met-A against its median depth: rho {spearmanr(c.A, c.depth).correlation:+.3f} over {len(c)} cells"]
open(os.path.join(H, "metaa_seqatlas_by_depth.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
