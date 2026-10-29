"""DEV-METAA-ATLAS-CURVE-01: where the healthy Met-A ratio's fall with depth comes from (written and committed before it runs; reported, not
scored, nothing set from it). On the 150 healthy ratios (ratio_depth_readings.csv: Met-A over the median of the other healthy samples of the
same cell) and the same samples' IAM-A from the one command (DEV_ALL_INFO_01/coupling_02/healthy_readings.csv): Spearman of the ratio against
depth, against IAM-A, and against each with the other held (rank partial correlation). If the ratio follows IAM-A (the measured error rate
in held marks) with depth held, the excess is the library's own error rate, which IAM-A already reads; if it follows depth with IAM-A held,
it is depth itself. Usage: python3 trend_source.py OUT.txt"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
H = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(H, "ratio_depth_readings.csv"))
Q = pd.read_csv(os.path.join(H, *[".."] * 3, "DEV_ALL_INFO_01", "coupling_02", "healthy_readings.csv"))[["gsm", "iam_a"]]
d = R.merge(Q, on="gsm").dropna(subset=["ratio", "depth", "iam_a"])
def partial(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z)); bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    return float(np.corrcoef(rx - np.polyval(bx, rz), ry - np.polyval(by, rz))[0, 1])
L = [f"samples {len(d)}",
     f"ratio vs depth: rho {spearmanr(d.ratio, d.depth)[0]:+.3f} | ratio vs IAM-A: rho {spearmanr(d.ratio, d.iam_a)[0]:+.3f} | IAM-A vs depth: rho {spearmanr(d.iam_a, d.depth)[0]:+.3f}",
     f"ratio vs IAM-A, depth held: {partial(d.ratio, d.iam_a, d.depth):+.3f} | ratio vs depth, IAM-A held: {partial(d.ratio, d.depth, d.iam_a):+.3f}"]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
