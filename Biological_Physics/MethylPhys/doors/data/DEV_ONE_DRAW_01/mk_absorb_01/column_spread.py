"""DEV-ONE-DRAW-01: the median spread of each provisional megakaryocyte column by read depth (version 1 against version 2), from the saved
columns.  Usage: python3 column_spread.py OUT.txt"""
import sys, os, pandas as pd
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "chain", "Runtime Matrices", "Provisional_References")
L = ["reads | version 1: sites, median spread | version 2: sites, median spread"]
C = {v: pd.read_parquet(os.path.join(P, f"megakaryocytes_v{v}.parquet")) for v in (1, 2)}
for n in range(1, 11):
    r = [f"{int((C[v].n == n).sum())}, {C[v].sd[C[v].n == n].median():.3f}" if (C[v].n == n).any() else "none" for v in (1, 2)]
    L.append(f"{n} | {r[0]} | {r[1]}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
