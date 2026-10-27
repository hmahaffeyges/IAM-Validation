"""DEV-ALL-INFO-01: how the study's leukemias sit against its cord-blood B-cell controls on each reading (from hopkins_full_depth.csv;
nothing new read). Two-sided Mann-Whitney; and how many leukemias read above the highest control. Usage: python3 hopkins_separation.py OUT.txt"""
import sys, os, pandas as pd
from scipy.stats import mannwhitneyu
H = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(H, "hopkins_full_depth.csv"))
L_, C_ = T[T.group == "leukemia"], T[T.group == "cord-blood B cells"]
L = [f"leukemias {len(L_)} | cord-blood controls {len(C_)}", "reading | leukemia median | control median | leukemias above the highest control | Mann-Whitney p"]
for k in ("iam_a", "held_error", "met_a_whole", "islands_A", "domains_A"):
    a, b = L_[k].dropna(), C_[k].dropna()
    L.append(f"{k} | {a.median():.4f} | {b.median():.4f} | {int((a > b.max()).sum())} of {len(a)} | {mannwhitneyu(a, b, alternative='two-sided').pvalue:.3g}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
