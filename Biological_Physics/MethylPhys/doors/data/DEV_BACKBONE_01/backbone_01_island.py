"""DEV-BACKBONE-01, island arm (added 2026-10-15, before any tumor island count exists). KISS: per pair, the tumor's island gain error
over its own normal's, counted with PROC-CHANNEL-01's rule (development/island_eraser_2026-10-15/island_eraser.py count()). Reported
beside the healthy island range across 56 cell types (0.0094-0.0162, loose_end_line.json). The author's picture: the overdrive writer
raises it in the tumor; nothing changes if the islands hold.
# NO CHAIN: the island count is PROC-CHANNEL-01's de novo channel; no chain stage counts island gains
Usage: python backbone_01_island.py synthetic"""
import os, sys, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "..", "..", "development", "island_eraser_2026-10-15"))
sys.argv_saved = sys.argv; sys.argv = ["x", "lib"]
_f = os.path.join(sys.path[0], "island_eraser.py"); src = open(_f).read(); ns = {"__file__": _f}; exec(src.split("if sys.argv[1] ==")[0], ns)
count, planted = ns["count"], ns["planted"]; sys.argv = sys.argv_saved
def pair(gn, gt, seed): return count(planted(gn, 0.02, seed=seed))["gain"], count(planted(gt, 0.02, seed=seed + 100))["gain"]
if sys.argv[1] == "synthetic":
    L = ["DEV-BACKBONE-01 island arm, synthetic (planted pairs; no tumor data)"]
    for name, gt in (("tumor islands written over (2x)", 0.0248), ("tumor islands unchanged", 0.0124)):
        R = [pair(0.0124, gt, s) for s in range(6)]; ratios = [t / n for n, t in R]
        L.append(f"{name}: tumor over normal per pair {', '.join(f'{x:.2f}' for x in ratios)}; pairs above 1.2: {sum(x > 1.2 for x in ratios)} of 6")
    open(os.path.join(H, "backbone_01_island_synthetic.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
