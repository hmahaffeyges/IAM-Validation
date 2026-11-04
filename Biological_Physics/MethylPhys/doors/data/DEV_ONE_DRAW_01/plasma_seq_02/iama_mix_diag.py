"""IAM-A in the sample's own mix on the 30 tuning plasmas, against the healthy inputs (development; diagnostic, added after both were
read; nothing scored). From the A-score rows the one command's bundles gave: plasma_seq_02_rows.csv and healthy pass 8's rows
(DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/healthy_fullchain8_rows.csv). Printed: each reading's median, how many plasmas sit inside the healthy
inputs' 2.5th to 97.5th percentile, and how IAM-A in the mix follows the domains and the erythrocyte progenitor share (plasma_shares_diag.csv).
Usage: python3 iama_mix_diag.py OUT.txt"""
import os, sys, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__)); HP = os.path.join(H, "..", "..", "DEV_METAA_ATLAS_CURVE_01", "seq_atlas_01", "healthy_fullchain8_rows.csv")
P = pd.read_csv(os.path.join(H, "plasma_seq_02_rows.csv")); Hh = pd.read_csv(HP); S = pd.read_csv(os.path.join(H, "plasma_shares_diag.csv")).set_index("run")
L = []
for rd in ("IAM-A", "IAM-A in the mix"):
    h = Hh[Hh.reading == rd].A.dropna(); p = P[P.reading == rd].drop_duplicates("gsm").set_index("gsm").A.dropna(); lo, hi = h.quantile(.025), h.quantile(.975)
    L.append(f"{rd}: healthy inputs {len(h)}, median {h.median():.4f}, 2.5 to 97.5 % {lo:.4f} to {hi:.4f}; plasmas {len(p)}, median {p.median():.4f}, inside {int(((p >= lo) & (p <= hi)).sum())} of {len(p)}")
m = P[P.reading == "IAM-A in the mix"].drop_duplicates("gsm").set_index("gsm").A
L.append(f"IAM-A in the mix against the domains: rho {spearmanr(S.domains.reindex(m.index), m, nan_policy='omit').correlation:+.3f}; "
         f"against the erythrocyte progenitor share: rho {spearmanr(S['share|erythrocyte progenitors'].reindex(m.index), m, nan_policy='omit').correlation:+.3f}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
