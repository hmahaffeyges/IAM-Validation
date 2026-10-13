"""DEV-SPHERE-01 planted test 3 (written and committed 2026-10-13, BEFORE it was run): construction v2 on a CONTINUOUS synthetic methylome.

Why: test 2 passed P1-P4 but showed hard edges on the 12 HEALPix base faces, and its methylome had only three clean classes, which the
first splits handed whole faces. Here sites vary continuously, as real methylomes do: S = 40 cell types share 6 lineage programs; each
site has a baseline level (logit drawn from a broad three-part mixture, so held, island-like and in-between sites grade into each other)
and its own loadings on the programs (loading scale drawn continuously, from near zero to strong). b = sigmoid(base + loadings . programs
+ noise). N = 60,000 sites, as before. Island-like = baseline logit below -2 with loading scale below 0.5; held-methylated = above +2 with
loading scale below 0.5; variable = across-cell SD above 0.08. Plant: 1,200 variable sites given one shared profile plus noise (sd 0.05).

Bars, UNCHANGED from test 2 (P1-P4). One declared change for laptop time: P2 uses 20 null maps, bar at most 1 of 20 (the same 5 percent
rate as at most 3 of 50, rounded down).
Reported, not a bar: rank agreement globe vs full-space angle (test 2: 0.313); EDGE RATIO - mean jump of the mean-methylation map across
pairs of neighboring pixels on different base faces, divided by the same across neighbors on the same face (about 1 = no edges; test 2's
picture showed strong edges). If the edge ratio stays well above 1 on continuous data, the edges are the construction's, not the data's.

# NO CHAIN: synthetic methylome; the construction and the chain's sky statistics (through sphere_construction_02) are what is tested.
"""
import os, sys, numpy as np, pandas as pd, healpy as hp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sphere_construction_02 as C2
from scipy.stats import spearmanr
S, N, M, K = 40, 60000, 1200, 6
def methylome(seed, plant_sd=None):
    r = np.random.default_rng(seed)
    Z = r.normal(0, 1, (K, S))                                   # lineage programs across cell types
    comp = r.choice(3, N, p=[0.5, 0.25, 0.25])
    base = np.where(comp == 0, r.normal(2.8, 0.8, N), np.where(comp == 1, r.normal(-3.2, 0.8, N), r.normal(0, 1.2, N)))
    scale = r.exponential(0.6, N); W = r.normal(0, 1, (N, K)) * scale[:, None] / np.sqrt(K)
    B = 1 / (1 + np.exp(-(base[:, None] + W @ Z + r.normal(0, 0.15, (N, S)))))
    sd = B.std(1)
    cls = np.full(N, 2); cls[(base > 2) & (scale < 0.5)] = 0; cls[(base < -2) & (scale < 0.5)] = 1
    var = np.where(sd > 0.08)[0]; grp = var[:M]
    if plant_sd is not None:
        prof = r.beta(0.5, 0.5, S); B[grp] = np.clip(prof + r.normal(0, plant_sd, (M, S)), 0, 1)
    return B, cls, var, grp
def ref(X, var, rng, k): return np.array([C2.concentration(X, rng.choice(var, M, replace=False)) for _ in range(k)])
def edge_ratio(mb, ns):
    nest = np.arange(hp.nside2npix(ns)); face = hp.pix2xyf(ns, nest, nest=True)[2]
    m = mb[hp.nest2ring(ns, nest)]; nb = hp.get_all_neighbours(ns, nest, nest=True)
    same, diff = [], []
    for j in range(8):
        q = nb[j]; ok = (q >= 0) & np.isfinite(m) & np.isfinite(m[np.clip(q, 0, None)])
        d = np.abs(m[ok] - m[q[ok]])
        sf = face[ok] == face[q[ok]]; same.append(d[sf]); diff.append(d[~sf])
    return float(np.mean(np.concatenate(diff)) / np.mean(np.concatenate(same)))
if __name__ == "__main__":
    L, rows = [], []; rng = np.random.default_rng(20261013)
    B, cls, var, grp = methylome(1, plant_sd=0.05); U = C2.info_vectors(B); X, pr, ns = C2.globe(U)
    L.append(f"continuous methylome: {N} sites, {len(var)} variable, island-like {int((cls == 1).sum())}, held {int((cls == 0).sum())}; NSIDE {ns}")
    R = C2.concentration(X, grp); q = float(np.quantile(ref(X, var, rng, 500), 0.99)); P1 = R > q
    L.append(f"P1 planted group R {R:.3f} | 99th percentile of 500 random groups {q:.3f} -> {'MET' if P1 else 'NOT MET'}")
    a, b = X[cls == 1].mean(0), X[cls == 0].mean(0); ang = float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))); P3 = ang >= 90
    L.append(f"P3 island-like vs held-methylated mean directions {ang:.1f} deg -> {'MET' if P3 else 'NOT MET'}")
    cnt, mb, _ = C2.maps(pr, ns, B); cov = float((cnt > 0).mean()); P4 = cov >= 0.90
    L.append(f"P4 pixels holding a site {cov:.1%} (bar 90 %) -> {'MET' if P4 else 'NOT MET'}")
    i, j = rng.integers(0, N, 5000), rng.integers(0, N, 5000)
    rho = float(spearmanr(C2.fr_angle(U, i, j), np.arccos(np.clip(np.einsum("ij,ij->i", X[i], X[j]), -1, 1))).correlation)
    L.append(f"reported: rank agreement globe vs full-space angle rho {rho:.3f} (test 2: 0.313)")
    er = edge_ratio(mb, ns); L.append(f"reported: EDGE RATIO {er:.2f} (cross-face neighbor jump / same-face neighbor jump; about 1 = no edges)")
    hits = 0
    for k in range(20):
        Bn, _, varn, grpn = methylome(1000 + k); Xn, _, _ = C2.globe(C2.info_vectors(Bn))
        Rn = C2.concentration(Xn, grpn); qn = float(np.quantile(ref(Xn, varn, rng, 200), 0.99)); hits += Rn > qn; rows.append(dict(case=f"null {k}", R=Rn, q99=qn))
        print("null", k, flush=True)
    P2 = hits <= 1
    L.append(f"P2 nothing planted: exceeded in {hits} of 20 maps (bar at most 1) -> {'MET' if P2 else 'NOT MET'}")
    ok = P1 and P2 and P3 and P4
    L.append("PLANTED TEST 3 " + ("PASSED" if ok else "FAILED") + f" (bars P1-P4); edge ratio {er:.2f}, distance agreement {rho:.3f} reported")
    rows.insert(0, dict(case="planted sd 0.05", R=R, q99=q))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "planted_sphere_03_output.csv"), index=False)
    np.savez(os.environ.get("MAPOUT", "/tmp/synthetic_sphere_v3_maps.npz"), count=cnt, mean_b=mb)
    open(os.path.join(HERE, "planted_sphere_03_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
