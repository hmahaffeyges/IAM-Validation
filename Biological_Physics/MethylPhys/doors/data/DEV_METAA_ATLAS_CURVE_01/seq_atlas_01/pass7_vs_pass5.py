"""The 57-cell healthy pass against pass 5 on the same 137 inputs, reading by reading (diagnostic, added after scoring; nothing scored), and the
three granulocytes. Usage: python3 pass7_vs_pass5.py OUT.txt"""
import os, sys, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); k = ["gsm", "reading"]
A = pd.read_csv(os.path.join(H, "healthy_fullchain5_readings.csv")).drop_duplicates(k).set_index(k).A; B7 = pd.read_csv(os.path.join(H, "healthy_fullchain7_readings.csv")); B = B7.drop_duplicates(k).set_index(k).A
j = A.index.intersection(B.index); d = (B[j] - A[j]).abs(); L = [f"{len(j)} readings of the 137 inputs compared; largest change by reading: " + "; ".join(f"{r} {v:.5f}" for r, v in d.groupby(level=1).max().items())]
G = B7[B7.cell == "granulocytes"].pivot_table(index="gsm", columns="reading", values="A")
for g, r in G.iterrows(): L.append(f"{g}: " + "; ".join(f"{c} {v:.4f}" for c, v in r.items()))
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
