"""The study's third reading (57-cell atlas) against its second (56 cells): every numeric reading's group median both ways, and the
largest change in any sample (diagnostic; nothing scored). Usage: python3 hopkins_3_vs_2.py OUT.txt"""
import os, sys, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); A = pd.read_csv(os.path.join(H, "hopkins_fullchain2.csv")); B = pd.read_csv(os.path.join(H, "hopkins_fullchain3.csv"))
k = [c for c in B.columns if c in A.columns and A[c].dtype != object and B[c].dtype != object]
A[k] = A[k].astype(float); B[k] = B[k].astype(float); g = "genotype"
J = A.set_index("gsm")[k].sub(B.set_index("gsm")[k]).abs(); L = [f"{len(B)} samples, {len(k)} readings compared"]
for c in k: L.append(f"{c}: largest change in any sample {J[c].max():.4g}; medians by group, second -> third: " + ", ".join(f"{x} {A.groupby(g)[c].median()[x]:.4g} -> {v:.4g}" for x, v in B.groupby(g)[c].median().items()))
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L[:3]))
