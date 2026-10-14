"""DEV-ARCH-01 (exploration): rank correlations of the architecture numbers with the healthy held error. Writes arch_vs_held_summary.txt."""
# NO CHAIN: exploration on two committed tables, no chain stage involved
import os, pandas as pd
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__))
X = pd.read_csv(os.path.join(H, "arch_vs_held.csv"), index_col=0)
L = [f"cells {len(X)}"] + [f"{v}: rank correlation with the held error {spearmanr(X.held_err, X[v])[0]:+.2f}" for v in ("loose_share", "fully_meth_share", "identity_sites")]
P = pd.read_csv(os.path.join(H, "..", "DEV_POSITION_EXPR_01", "position_expr_01_points.csv"))
open(os.path.join(H, "arch_vs_held_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
