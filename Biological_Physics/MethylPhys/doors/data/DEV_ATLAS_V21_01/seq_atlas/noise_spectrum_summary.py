"""Summary of the genome-distance spectrum of the IAM Sequencing Atlas's noise maps (noise_spectrum.csv, run_sample.py --build
seq-noise-spectrum). Usage: python3 noise_spectrum_summary.py OUT.txt"""
import os, sys, ast, numpy as np, pandas as pd
D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_spectrum.csv")); D["corr"] = D["corr"].apply(ast.literal_eval); L = []
for m, g in D.groupby("map"):
    A = np.array(g["corr"].tolist(), float); md = np.median(A, 0)
    L.append(f"{m}: {len(g)} cells; median correlation by band (under 1 kb, 1-10 kb, 10-100 kb, 0.1-1 Mb, 1-10 Mb, 10-100 Mb): {' '.join(f'{x:.4f}' for x in md)}; largest |median| {np.abs(md).max():.4f}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
