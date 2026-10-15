"""Atlas v2.1 readiness L21 reopened, DESIGN 2 (2026-10-15). Written and committed BEFORE any test of it runs.
Design 1 (transfer_strict_v21.py: ties merged and anchors at 0 and 1) failed its own check C2 before any fit (largest change 0.0159 at array
0.984 against 0.01; transfer_strict_v21_summary.txt) and is not used. What it showed: sequencing levels off near 0.959 while arrays still
separate 0.964 from 0.984, a real plateau of the measurement, so merging the top ties moved the curve.
DESIGN 2 (nothing fitted): the adopted knots EXACTLY as they are, plus anchors (0, 0) and (1, 1) outside them. Inside the measured range the
curve is unchanged; the only flat stretches left are the two bounded measured plateaus (array 0.035-0.055 and 0.964-0.984); nothing is flat
without bound, which is where the sampler loses its way (beyond the end knots a cell's value could drift without limit in logit space).
CHECKS: D1 non-decreasing everywhere and strictly increasing outside [0.017, 0.984]; D2 zero change inside [0.017, 0.984].
Writes transfer_v21_anchored.json and transfer_anchored_v21_summary.txt. Then planted_strict_v21.py runs with this file (its STRICT arm)."""
import os, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(H, "transfer_v21_final.json"))); kx = list(map(float, T["knots_x"])); ky = list(map(float, T["knots_y"]))
nx = [0.0] + kx + [1.0]; ny = [0.0] + ky + [1.0]
D1 = bool(np.all(np.diff(ny) >= 0) and ny[1] > ny[0] and ny[-1] > ny[-2] and np.all(np.diff(nx) > 0))
g = np.linspace(kx[0], kx[-1], 20001); D2 = bool(np.abs(np.interp(g, nx, ny) - np.interp(g, kx, ky)).max() == 0.0)
S = dict(T); S["knots_x"] = nx; S["knots_y"] = ny; S["anchored_rule"] = "transfer_anchored_v21.py: adopted knots unchanged, anchors (0,0) and (1,1)"
json.dump(S, open(os.path.join(H, "transfer_v21_anchored.json"), "w"), indent=1)
L = ["Atlas v2.1 anchored transfer, design 2 (transfer_anchored_v21.py)", f"knots: {len(nx)} (adopted {len(kx)} plus two anchors)",
     f"end slopes: low {ny[1] / nx[1]:.3f}, high {(1 - ny[-2]) / (1 - nx[-2]):.3f}",
     f"D1 bounded (no unbounded flat end): {'met' if D1 else 'NOT met'} | D2 unchanged inside the measured range: {'met' if D2 else 'NOT met'}",
     f"ANCHORED TRANSFER: {'MET' if (D1 and D2) else 'NOT MET'}"]
open(os.path.join(H, "transfer_anchored_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
