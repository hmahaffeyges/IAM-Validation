"""Canary rule for PHY-087 step 2 (design/PHY-087_s2.json, revised 2026-11-14 after the first canary): each canary library's chain tile
file, pooled to tiles of 50,000 CpGs (25 tiles), has at least 400 usable tiles (>= 200 opportunities per half), a mean call for at least 95 %
of its 2,000-CpG tiles, and halves correlating above 0.7. Exit 0 when met."""
import sys, glob, os, numpy as np, pandas as pd
F = glob.glob(os.path.join(sys.argv[1], "**", "*_tiles.csv.gz"), recursive=True); ok = bool(F)
for f in F:
    T = pd.read_csv(f, dtype={"chrom": str}); mc = float(T.mean_call.notna().mean()) if "mean_call" in T else 0.0
    X = T.assign(tile=T.tile // 25).groupby(["chrom", "tile"])[["iso_a", "opp_a", "iso_b", "opp_b"]].sum(); X = X[(X.opp_a >= 200) & (X.opp_b >= 200)]
    r = float(np.corrcoef(X.iso_a / X.opp_a, X.iso_b / X.opp_b)[0, 1]); good = len(X) >= 400 and r > 0.7 and mc >= 0.95; ok &= good
    print(os.path.basename(f), len(X), "usable pooled tiles, halves r", round(r, 3), "mean call present", round(mc, 3), "met" if good else "NOT MET")
print("canary rule:", "MET" if ok else "NOT MET"); sys.exit(0 if ok else 1)
