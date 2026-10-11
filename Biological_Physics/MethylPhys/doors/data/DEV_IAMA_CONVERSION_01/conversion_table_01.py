"""DEV-IAMA-CONVERSION-01: the prediction table and the bias factor, from conversion_01_rows.csv. Per TruSeq donor with two repeat libraries:
c rep1, c rep2, predicted eps ratio (and, for every donor and kit, the measured ratios of the outcome table) c2/c1, predicted IAM-A ratio H(eps1 * c2/c1) / H(eps1); and d ln H / d ln eps at eps 0.043.
Writes conversion_table_01_output.txt. Usage: python3 conversion_table_01.py"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = pd.read_csv(os.path.join(HERE, "conversion_01_rows.csv"))
H = lambda e: -e * np.log2(e) - (1 - e) * np.log2(1 - e); out = []
for d, g in R[R.kit == "TruSeq"].groupby("donor"):
    g = g.set_index("rep")
    if {"rep1", "rep2"} <= set(g.index):
        a, b = g.loc["rep1"], g.loc["rep2"]; r = b.c / a.c
        out.append(f"{d}: c {a.c:.5f} / {b.c:.5f} | eps ratio predicted {r:.4f} | IAM-A ratio predicted {H(a.eps * r) / H(a.eps):.4f}")
for (d, k), g in R.groupby(["donor", "kit"]):          # the outcome table: measured repeat ratios, raw and after eps / c
    g = g.set_index("rep")
    if {"rep1", "rep2"} <= set(g.index):
        a, b = g.loc["rep1"], g.loc["rep2"]
        out.append(f"{d} {k}: c {a.c:.4f} / {b.c:.4f} | eps rep2/rep1 measured {b.eps / a.eps:.4f}, predicted {b.c / a.c:.4f} | "
                   f"IAM-A rep2/rep1 raw {H(b.eps) / H(a.eps):.4f}, after eps/c {H(b.eps_corr) / H(a.eps_corr):.4f}")
e = 0.043; out.append(f"d ln H / d ln eps at eps {e}: {np.log2((1 - e) / e) * e / H(e):.3f}")
open(os.path.join(HERE, "conversion_table_01_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
