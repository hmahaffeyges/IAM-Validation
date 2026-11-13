"""Combines the forward-model runs (forward_candidate.py on GitHub explore runs; forward_out/*.csv): the candidate's readings on both scales
against the counter's own clustering measure, read at the healthy samples' measured clustering (median 1.46, percall_out), set beside the
healthy readings (lone-loss intercept 0.04526, CANON eps_healthy_pool; per-call intercept 0.05048, PHY-050), and the per-site loss and work
per restore those healthy readings imply. Writes forward_candidate_summary.txt."""
import os, glob, math, numpy as np, pandas as pd
from scipy.optimize import brentq
H = os.path.dirname(os.path.abspath(__file__))
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(H, "forward_out", "*.csv"))]).groupby(["mode", "param"]).mean(numeric_only=True).reset_index()
L = ["The candidate pushed through the counter (per-site loss 0.04454; mean of the samples per setting)", R.drop(columns="replicate").round(5).to_string(index=False), ""]
S = R[R["mode"].isin(["ind", "seq"])].sort_values("coupling")
lc, pc = np.interp(1.46, S.coupling, S.held_err_length), np.interp(1.46, S.coupling, S.held_e_call)
L.append(f"at the healthy clustering 1.46 (losses following each other): lone-loss {lc:.5f} against healthy 0.04526 ({100 * (lc / 0.04526 - 1):+.1f} %); per-call {pc:.5f} against healthy 0.05048 ({100 * (pc / 0.05048 - 1):+.1f} %)")
L.append("molecules simply losing at different rates (het) lower both readings without raising the clustering measure, so they are not what the healthy clustering shows")
odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w); M = 20.94
for lab, r in (("lone-loss", 0.04526 / lc), ("per-call", 0.05048 / pc)):
    e = 0.04454 * r; w = brentq(lambda w: odds(w) - (1 / e - 1), 1, 200)
    L.append(f"{lab}: the healthy reading implies a per-site loss {e:.4f}, work per restore {w:.1f} kT = {w / M:.2f} coins (candidate 2.00)")
open(os.path.join(H, "forward_candidate_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
