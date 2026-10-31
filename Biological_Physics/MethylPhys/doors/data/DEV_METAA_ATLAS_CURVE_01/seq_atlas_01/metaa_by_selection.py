"""Met-A against the IAM Sequencing Atlas, healthy test: each cell's median Met-A (metaa_seqatlas_readings.csv, the one command's readings)
beside how closely its two donor halves agree at the marked identity sites over counting (noise_at_identity.csv, the one command's
reading of the atlas). Diagnostic after the test was scored, disclosed; nothing scored. Usage: python3 metaa_by_selection.py"""
import os, pandas as pd
from scipy.stats import spearmanr
H0 = os.path.dirname(os.path.abspath(__file__)); H = pd.read_csv(os.path.join(H0, "metaa_seqatlas_readings.csv"))
N = pd.read_csv(os.path.join(H0, "noise_at_identity.csv")); nm = N[N.side == "marked"].set_index("cell").map1_over_counting
H["c"] = H.cell.astype(str).str.replace("sorted ", "", regex=False).str.strip().str.lower()
g = H.groupby("c").A.median().rename("metaa").to_frame().join(nm.rename("agree"), how="inner")
L = [f"{len(g)} cells with both readings | Spearman(cell's median Met-A, marked halves over counting) {spearmanr(g.metaa, g.agree).correlation:+.3f}",
     "cells whose halves agree like counting (0.8 or more): " + ", ".join(f"{c} {v:.4f}" for c, v in g[g.agree >= 0.8].metaa.items()),
     f"median Met-A, halves at 0.8 or more of counting: {g[g.agree >= 0.8].metaa.median():.4f} over {int((g.agree >= 0.8).sum())} cells",
     f"median Met-A, halves below 0.3 of counting: {g[g.agree < 0.3].metaa.median():.4f} over {int((g.agree < 0.3).sum())} cells"]
g.to_csv(os.path.join(H0, "metaa_by_selection.csv")); open(os.path.join(H0, "metaa_by_selection.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
