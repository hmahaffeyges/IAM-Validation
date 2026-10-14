"""Atlas v2.1 readiness L21: the bounded array-to-sequencing transfer. Written and committed 2026-10-14 BEFORE the synthetic test or the
real check is run.

WHY. L20 found v2's straight line y = a + b x bends (14 of 42 bins beyond +/-0.02). A 3-parameter logistic in logit space, tried on the
committed L20 bin table before this file was written, still missed 3 of 42 bins (worst 0.027 at x 0.58) with a smooth systematic wave, so
it is structurally wrong and is not used. Disclosed: the choice below was made after seeing L20's residuals, on build cells only.

THE TRANSFER (an empirical calibration of one instrument against another, like v2's line; not physics). Pairs (x = our array-scale cell
mean, y = Loyfer's cell mean at the same site) are binned with L20's edges (0.01 wide in [0, 0.15] and [0.85, 1], 0.05 between). Each bin
with >= 500 pairs gives a knot (median x, mean y). The knot values are made non-decreasing by pool-adjacent-violators, weighted by pairs.
g(x) is linear interpolation between knots, held at the end knots outside them. Every knot is a mean of readings in [0, 1], so g cannot
leave [0, 1] and is monotone. In the model it replaces a + b m for sequencing rows; its local slope g'(m) replaces b in the variance.

SYNTHETIC TEST (laptop, run once; MET only if all three hold):
 S1 recovery: on 17 synthetic cells built from a known S-shaped truth with array-side noise, g built on all cells is within +/-0.02 of
    the truth at every knot with >= 500 pairs.
 S2 cross-halves on synthetic data: the cross-halves check below passes.
 S3 bounds: g is in [0, 1] and non-decreasing.
REAL CHECK (box, run once on v2's 17 shared cells, the L20 pairs): the 17 cells are ordered by sha256("ATLAS_V21_L21|" + cell) and dealt
alternately into halves A and B. g built on A is checked on B's pairs, and g built on B on A's: in every bin with >= 500 pairs of the
checked half, |mean y - g(median x)| <= 0.02. ADOPTED only if both directions pass and the synthetic test is MET; g is then built on all
17 cells and written as knots (transfer_bounded_v21.json). Otherwise NOT ADOPTED and v2.1 does not launch on it.
"""
import hashlib, json, numpy as np, pandas as pd
EDGES = np.unique(np.round(np.concatenate([np.arange(0, 0.15, 0.01), np.arange(0.15, 0.85, 0.05), np.arange(0.85, 1.0001, 0.01)]), 4))
NMIN, TOL = 500, 0.02
def bins(X):
    X = X.assign(bin=pd.cut(X.x, EDGES, include_lowest=True))
    T = X.groupby("bin", observed=True).agg(n=("x", "size"), xm=("x", "median"), ym=("y", "mean")).reset_index()
    return T[T.n >= NMIN].reset_index(drop=True)
def pav(y, w):
    y, w = list(map(float, y)), list(map(float, w)); blocks = [[y[i], w[i], 1] for i in range(len(y))]; i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]; ww = a[1] + b[1]; blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / ww, ww, a[2] + b[2]]; del blocks[i + 1]; i = max(i - 1, 0)
        else: i += 1
    return np.concatenate([[v] * k for v, _, k in blocks])
def build(X):
    T = bins(X); return T.xm.values, pav(T.ym.values, T.n.values)
def g(x, kx, ky): return np.interp(x, kx, ky)
def check(build_X, test_X):
    kx, ky = build(build_X); T = bins(test_X); r = T.ym.values - g(T.xm.values, kx, ky)
    return dict(bins=int(len(T)), worst=float(np.abs(r).max()), beyond=int((np.abs(r) > TOL).sum()), ok=bool((np.abs(r) <= TOL).all()))
def halves(cells):
    o = sorted(cells, key=lambda c: hashlib.sha256(("ATLAS_V21_L21|" + c).encode()).hexdigest()); return o[0::2], o[1::2]
def cross(P):
    A, B = halves(sorted(P)); XA = pd.concat([P[c] for c in A]); XB = pd.concat([P[c] for c in B])
    return dict(halves={"A": A, "B": B}, A_on_B=check(XA, XB), B_on_A=check(XB, XA))

if __name__ == "__main__":     # the synthetic test (S1-S3)
    import os
    HERE = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(20261014)
    L20 = pd.read_csv(os.path.join(HERE, "transfer_shape_v21.csv")); L20 = L20[L20.n >= NMIN]
    tx = L20.x_median.values; ty = pav((L20.line_at_median + L20.mean_residual).values, L20.n.values)   # truth: L20's own S shape
    truth = lambda x: np.interp(x, tx, ty)
    P = {}
    for c in range(17):
        n = 45000; z = rng.random(n)
        xt = np.where(z < 0.35, rng.beta(1.2, 20, n), np.where(z < 0.85, rng.beta(25, 1.6, n), rng.random(n)))   # bimodal, like methylation
        xo = np.clip(xt + rng.normal(0, 0.01, n), 0, 1)                                                    # array-side noise on the cell mean
        mu = truth(xt); y = np.clip(mu + rng.normal(0, 0.04, n) * np.sqrt(np.maximum(mu * (1 - mu), 0.01)) * 2, 0, 1)
        P[f"cell{c:02d}"] = pd.DataFrame({"x": xo, "y": y})
    kx, ky = build(pd.concat(P.values()))
    s1 = float(np.abs(ky - truth(kx)).max()); S1 = s1 <= TOL
    cr = cross(P); S2 = cr["A_on_B"]["ok"] and cr["B_on_A"]["ok"]
    S3 = bool((ky >= 0).all() and (ky <= 1).all() and (np.diff(ky) >= 0).all())
    L = ["Atlas v2.1 L21 synthetic test, run once (transfer_bounded_v21.py)",
         f"S1 recovery: worst |g - truth| at knots {s1:.4f} over {len(kx)} knots -> {'met' if S1 else 'NOT met'}",
         f"S2 cross-halves: A on B worst {cr['A_on_B']['worst']:.4f} ({cr['A_on_B']['beyond']} beyond), B on A worst {cr['B_on_A']['worst']:.4f} ({cr['B_on_A']['beyond']} beyond) -> {'met' if S2 else 'NOT met'}",
         f"S3 bounds: g in [{ky.min():.4f}, {ky.max():.4f}], non-decreasing -> {'met' if S3 else 'NOT met'}",
         f"SYNTHETIC: {'MET' if (S1 and S2 and S3) else 'NOT MET'}"]
    open(os.path.join(HERE, "transfer_bounded_v21_synthetic.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
