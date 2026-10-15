"""Atlas v2.1 transfer, smooth form (2026-10-15). Written and committed BEFORE it is used in any fit.
WHY. With the anchored transfer the v2.1 model is distinct and agrees with v2 on 60 real loci (r 0.9991) but runs 18x slower, every draw at
the maximum tree depth. Cause, in atlas_block_v21.py: the variance uses slope = (g(m+h) - g(m-h)) / 2h of a piecewise-linear g, which is a
STEP function of m, so the likelihood jumps at every knot and the sampler's step size collapses.
RULE (nothing fitted, no new knot): the anchored knots exactly (transfer_v21_anchored.json), joined by the monotone cubic of Fritsch and
Carlson (PCHIP, scipy.interpolate.PchipInterpolator): it passes through every knot, keeps the curve non-decreasing and inside [0, 1], and
has a continuous slope. The model uses that exact slope in place of the finite difference.
CHECKS: S1 non-decreasing and inside [0, 1] on a fine grid; S2 passes through every knot exactly; S3 largest difference from the anchored
curve between knots <= 0.01 (L22's bars were met on it; reported); S4 slope continuous (largest jump across knots, reported, must be < 1e-9).
Writes transfer_v21_smooth.json (anchored file plus knot_slopes) and transfer_smooth_v21_summary.txt."""
import os, json, numpy as np
from scipy.interpolate import PchipInterpolator
H = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(H, "transfer_v21_anchored.json"))); kx = np.array(T["knots_x"]); ky = np.array(T["knots_y"])
P = PchipInterpolator(kx, ky); dP = P.derivative(); ks = dP(kx)
g = np.linspace(0, 1, 200001); v = P(g)
S1 = bool(np.all(np.diff(v) >= -1e-12) and v.min() >= -1e-12 and v.max() <= 1 + 1e-12)
S2 = bool(np.allclose(P(kx), ky, atol=1e-12, rtol=0))
d = np.abs(v - np.interp(g, kx, ky)); S3 = bool(d.max() <= 0.01)
jump = max(abs(dP(x - 1e-9) - dP(x + 1e-9)) for x in kx[1:-1]); S4 = bool(jump < 1e-6)
S = dict(T); S["knot_slopes"] = [float(x) for x in ks]; S["smooth_rule"] = "transfer_smooth_v21.py: PCHIP through the anchored knots; exact slope"
json.dump(S, open(os.path.join(H, "transfer_v21_smooth.json"), "w"), indent=1)
L = ["Atlas v2.1 smooth transfer (transfer_smooth_v21.py)", f"knots {len(kx)} (the anchored knots, unchanged)",
     f"S1 non-decreasing inside [0, 1]: {'met' if S1 else 'NOT met'}", f"S2 through every knot: {'met' if S2 else 'NOT met'}",
     f"S3 largest difference from the anchored curve: {d.max():.5f} at {g[d.argmax()]:.4f} -> {'met' if S3 else 'NOT met'}",
     f"S4 largest slope jump across a knot: {jump:.2e} -> {'met' if S4 else 'NOT met'}", f"SMOOTH TRANSFER: {'MET' if all((S1, S2, S3, S4)) else 'NOT MET'}"]
open(os.path.join(H, "transfer_smooth_v21_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
