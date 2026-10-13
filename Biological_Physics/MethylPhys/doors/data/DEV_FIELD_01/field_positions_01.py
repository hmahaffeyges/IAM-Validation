"""DEV-FIELD-01: where on the slide do the paired arrays sit? (metadata only; written 2026-10-13 after the planted test showed F1 would
detect a 0.5 % difference, so a systematic position offset between the two arrays of a pair could pass F1 by itself.)

# NO CHAIN: metadata (gse287331_samples.csv, from field_meta_01.py).
For the one-slide patients: the row (R01-R08) of the other-quadrant array against the same woman's other-breast array.
Writes field_positions_01_output.txt.
"""
import os
import pandas as pd
from scipy.stats import binomtest
HERE = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(HERE, "gse287331_samples.csv"), dtype=str)
T["row"] = T.position.str[1:3].astype(int)
L = []; diff = []
for pid, g in T[T.tissue.isin(["CUB", "OQ", "TU"])].groupby("patient"):
    if {"CUB", "OQ", "TU"} <= set(g.tissue) and g.slide.nunique() == 1:
        r = g.set_index("tissue").row; diff.append(int(r["OQ"]) - int(r["CUB"]))
d = pd.Series(diff)
L.append(f"one-slide patients {len(d)} | OQ row - CUB row: below 0 in {int((d < 0).sum())}, 0 in {int((d == 0).sum())}, above 0 in {int((d > 0).sum())}")
L.append(f"counts by difference: {d.value_counts().sort_index().to_dict()}")
nz = d[d != 0]; L.append(f"sign test on the order (two-sided): p = {binomtest(int((nz > 0).sum()), len(nz)).pvalue:.3g}")
open(os.path.join(HERE, "field_positions_01_output.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
