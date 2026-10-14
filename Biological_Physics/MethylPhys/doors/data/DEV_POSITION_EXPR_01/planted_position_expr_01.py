"""Planted test of DEV-POSITION-EXPR-01's bar (written 2026-10-14 before it runs; synthetic only, no expression or copy error read).
# NO CHAIN: tests a rank-correlation bar on synthetic points, no chain stage involved
30 points. TRUE: eps = pred(r) + measurement noise of the measured scatter's size (SD 0.002), r log-normal (sd 0.5) -> must be MET in >= 95 of 100.
NULL: eps independent of r -> must be NOT MET in >= 95 of 100. Uses position_expr_01.perm_p (10,000 permutations here for speed)."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import position_expr_01 as PE
rng = np.random.default_rng(20261014); met = {"TRUE": 0, "NULL": 0}
for case in met:
    for _ in range(100):
        r = np.exp(rng.normal(0, 0.5, 30)); p_ = PE.pred(r)
        eps = (p_ if case == "TRUE" else rng.permutation(p_)) + rng.normal(0, 0.002, 30)
        met[case] += PE.perm_p(p_, eps, n=10000)[1] < 0.01
L = [f"TRUE relation: MET in {met['TRUE']} of 100 (bar >= 95)", f"NULL: MET in {met['NULL']} of 100 (bar <= 5)",
     f"PLANTED: {'PASSED' if met['TRUE'] >= 95 and met['NULL'] <= 5 else 'FAILED'}"]
open(os.path.join(HERE, "planted_position_expr_01.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
