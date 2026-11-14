"""Canary rule for PHY-087 step 2 (design/PHY-087_s2.json): in a canary's artifact (one library recounted by the chain from saved molecules),
the tile file has at least 14,000 rows, a mean call for at least 95 % of them (the column added 2026-11-14), and its two halves correlate
above 0.7 over tiles with at least 200 opportunities per half (the first laboratory's canary: 14,106 tiles, 0.823). Exit 0 when met."""
import sys, glob, os, numpy as np, pandas as pd
F = glob.glob(os.path.join(sys.argv[1], "**", "*_tiles.csv.gz"), recursive=True); ok = bool(F)
for f in F:
    T = pd.read_csv(f); X = T[(T.opp_a >= 200) & (T.opp_b >= 200)]
    r = float(np.corrcoef(X.iso_a / X.opp_a, X.iso_b / X.opp_b)[0, 1]); mc = float(T.mean_call.notna().mean()) if "mean_call" in T else 0.0
    good = len(T) >= 14000 and r > 0.7 and mc >= 0.95; ok &= good
    print(os.path.basename(f), len(T), "rows, halves r", round(r, 3), "mean call present", round(mc, 3), "met" if good else "NOT MET")
print("canary rule:", "MET" if ok else "NOT MET"); sys.exit(0 if ok else 1)
