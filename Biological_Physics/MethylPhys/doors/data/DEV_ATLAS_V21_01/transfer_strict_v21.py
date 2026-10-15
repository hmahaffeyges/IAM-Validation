"""Atlas v2.1 readiness L21 reopened (2026-10-15): the strictly increasing transfer. Written and committed BEFORE any test of it runs.

WHY. The adopted transfer (transfer_v21_final.json) is exactly flat over array 0.035-0.055 and 0.964-0.984 and held flat below 0.017 and
above 0.984. On real block 0 the v2.1 model did not converge (speed_diag_v21.json: R-hat p99 3.35, every draw at the maximum tree depth):
where the transfer is flat, sequencing readings carry no information about a cell's value, and most sites sit near 0 or 1.

THE RULE (no new parameter, nothing fitted):
 S1 ties: consecutive knots with equal y are one pool-adjacent-violators block, i.e. ONE estimate; each run becomes one knot at the run's
    mean x and its common y.
 S2 ends: anchors (0, 0) and (1, 1): a fully unmethylated site reads 0 on both instruments and a fully methylated site reads 1.
 Linear between knots, as before. The result must be strictly increasing in x and y (checked). The measured scatter s(m) is unchanged.

CHECKS (this file, laptop, before any fit):
 C1 strictly increasing everywhere; C2 inside [0.017, 0.984] the new curve differs from the adopted one by at most 0.01 (L22's bars were
    met on the adopted curve; a change this small cannot move them past 0.02), reported with the largest change.
Writes transfer_v21_strict.json (all other fields copied from transfer_v21_final.json) and transfer_strict_v21_summary.txt."""
import os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(H, "transfer_v21_final.json"))); kx = np.array(T["knots_x"], float); ky = np.array(T["knots_y"], float)
nx, ny, i = [], [], 0
while i < len(kx):
    j = i
    while j + 1 < len(kx) and ky[j + 1] == ky[i]: j += 1
    nx.append(float(kx[i:j + 1].mean())); ny.append(float(ky[i])); i = j + 1
nx = [0.0] + nx + [1.0]; ny = [0.0] + ny + [1.0]
C1 = bool(np.all(np.diff(nx) > 0) and np.all(np.diff(ny) > 0))
g = np.linspace(0.017, 0.984, 20001); d = np.abs(np.interp(g, nx, ny) - np.interp(g, kx, ky)); C2 = bool(d.max() <= 0.01)
S = dict(T); S["knots_x"] = nx; S["knots_y"] = ny
S["strict_rule"] = "transfer_strict_v21.py S1 (pool-adjacent-violators ties merged to one knot at the run's mean x) and S2 (anchors (0,0) and (1,1))"
S["strict_from"] = {"knots_x": T["knots_x"], "knots_y": T["knots_y"]}
json.dump(S, open(os.path.join(H, "transfer_v21_strict.json"), "w"), indent=1)
L = ["Atlas v2.1 strict transfer (transfer_strict_v21.py)", f"knots: adopted {len(kx)}, strict {len(nx)} (ties merged, two anchors)",
     f"C1 strictly increasing: {'met' if C1 else 'NOT met'} | smallest segment slope {np.min(np.diff(ny) / np.diff(nx)):.4f}",
     f"C2 largest change inside 0.017-0.984: {d.max():.5f} at array {g[d.argmax()]:.4f} -> {'met' if C2 else 'NOT met'}",
     f"STRICT TRANSFER: {'MET' if (C1 and C2) else 'NOT MET'}"]
open(os.path.join(H, "transfer_strict_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
