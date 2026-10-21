"""DEV-SELFTARE-03, after the readings: how well an array's own anchor range (the span between its type II fixed-site anchors, U - L, the number
self-tare II divides by) can carry a loss at every site that self-tare II removes. Each replicate array of DEV-SELFTARE-02 against the median of
the other arrays on its slide; a 1 % loss of the neutrophils' pattern at every site narrows the range by f_neu x 1 %. Writes
anchor_range_spread_summary.txt here. Usage: python3 anchor_range_spread.py"""
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = pd.read_csv(os.path.join(H, "..", "DEV_SELFTARE_02", "selftare02_replicates.csv"))
R["range_II"] = R.U_II - R.L_II; R["slide"] = R.slide.astype(str); out = []
for _, x in R.iterrows():
    ref = R[(R.slide == x.slide) & (R.gsm != x.gsm)]
    if len(ref) >= 3: out.append(x.range_II / ref.range_II.median() - 1)
r = np.array(out); sd = r.std(ddof=1); f = float(R.f_neu.median())
L = [f"anchor range against the same slide's other arrays: {len(r)} arrays, SD {sd:.4f}",
     f"a 1 % loss of the neutrophils' pattern at every site narrows it by f_neu x 1 % = {f * 0.01:.4f} (median neutrophil share {f:.4f})",
     f"smallest loss at every site one array could show (2 SD): {100 * 2 * sd / f:.1f} % of the neutrophils' pattern",
     f"a group of 55 (Group-A's planted half): {100 * 2 * sd / np.sqrt(55) / f:.2f} %"]
open(os.path.join(H, "anchor_range_spread_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
