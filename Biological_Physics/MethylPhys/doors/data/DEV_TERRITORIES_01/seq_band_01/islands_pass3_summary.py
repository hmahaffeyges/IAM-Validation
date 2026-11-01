"""The islands in healthy pass 3 (diagnostic, added after pass 3 was scored): median, mean, and what the reading follows across the 130
samples. From committed outputs: healthy_fullchain3_readings.csv and metaa_depth_diag_donors.csv. Usage: python3 islands_pass3_summary.py OUT.txt"""
import os, sys, pandas as pd
from scipy.stats import spearmanr
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "DEV_METAA_ATLAS_CURVE_01", "seq_atlas_01")
R = pd.read_csv(os.path.join(H, "healthy_fullchain3_readings.csv")); D = pd.read_csv(os.path.join(H, "metaa_depth_diag_donors.csv"), index_col=0)
x = R[R.reading == "territory islands"].drop_duplicates("gsm").set_index("gsm").A.dropna(); Z = D.join(x.rename("isl"), how="inner")
rho = lambda a, b: spearmanr(a, b, nan_policy="omit").correlation; c = Z.groupby("cell").isl.median().sort_values()
L = [f"islands, healthy pass 3: {len(x)} read, median {x.median():.4f}, mean {x.mean():.4f}, inside 0.95 to 1.05 {((x >= .95) & (x <= 1.05)).mean() * 100:.1f} %",
     f"against: depth {rho(Z.depth, Z.isl):+.3f}; off-pattern reads, unmarked {rho(Z.off_unmarked, Z.isl):+.3f}; IAM-A {rho(Z.iama, Z.isl):+.3f}; own-cell share {rho(Z.own, Z.isl):+.3f}",
     f"cell medians from {c.iloc[0]:.3f} ({c.index[0]}) to {c.iloc[-1]:.3f} ({c.index[-1]})"]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
