"""Summary of the Minkowski functionals of the IAM Sequencing Atlas's noise maps (noise_minkowski.csv, run_sample.py --build
seq-noise-minkowski). Usage: python3 noise_minkowski_summary.py OUT.txt"""
import os, sys, pandas as pd
D = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_minkowski.csv")); D["out"] = D.V1_outside_null.astype(int) + D.V2_outside_null.astype(int)
L = [f"{m}: {g.cell.nunique()} cells; tests outside their shuffles {int(g.out.sum())} of {2 * len(g)} ({g.out.sum() / (2 * len(g)):.3f})" for m, g in D.groupby("map")]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
