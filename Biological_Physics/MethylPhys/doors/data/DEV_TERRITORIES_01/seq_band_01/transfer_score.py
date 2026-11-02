"""Score the transfer-function synthetic (transfer_synthetic.py) from its rows files. Usage: python3 transfer_score.py OUT.txt ROWS.csv [ROWS.csv ...]"""
import sys, numpy as np, pandas as pd
D = pd.concat([pd.read_csv(f) for f in sys.argv[2:]], ignore_index=True).dropna(subset=["level_change"]).sort_values(["delta", "draw"])
x = D.delta.to_numpy(); y = D.level_change.to_numpy(); slope = float(np.polyfit(x, y, 1)[0]); med = D.groupby("delta")[["level_change", "islands", "islands_plugin", "metaa_whole"]].median()
rise = bool(np.all(np.diff(med.islands.to_numpy()) > 0)); b1 = 0.95 <= slope <= 1.05
L = [f"reads {len(D)} of 15; by delta, median level change, islands (counted), islands (plug-in), whole-sample Met-A:"] + \
    [f"  delta {d:.2f}: {r.level_change:+.5f}, {r.islands:.4f}, {r.islands_plugin:.4f}, {r.metaa_whole:.4f}" for d, r in med.iterrows()] + \
    [f"bar 1 slope of the level change on delta {slope:.4f} (within 0.95 to 1.05): {'met' if b1 else 'not met'}",
     f"bar 2 the counted islands reading rises at every step: {'met' if rise else 'not met'}", f"synthetic: {'met' if b1 and rise and len(D) == 15 else 'not met'}"]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
