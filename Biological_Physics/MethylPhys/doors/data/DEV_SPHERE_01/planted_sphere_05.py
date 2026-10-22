"""DEV-SPHERE-01 planted test 5 (written and committed 2026-10-22, BEFORE it was run): construction v4 on test 3's continuous methylome.
Test 3's methylome, plant, bars P1-P4 and reported measures, imported unchanged (planted_sphere_03.py), with the globe of construction v3.
Two new bars, written in DEV_SPHERE_01.md on 2026-10-13 before v3 existed:
  P5 edge ratio (cross-face neighbor jump / same-face neighbor jump of the mean-methylation map) below 1.5;
  P6 rank agreement between globe angles and full-space Fisher-Rao angles (5,000 random pairs) at least 0.7.
# NO CHAIN: synthetic methylome; the construction and the chain's sky statistics are what is tested.
Writes planted_sphere_05_summary.txt and planted_sphere_05_output.csv."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import planted_sphere_03 as T3, sphere_construction_04 as C3
from scipy.stats import spearmanr
N = T3.N
if __name__ == "__main__":
    L, rows = [], []; rng = np.random.default_rng(20261013)
    B, cls, var, grp = T3.methylome(1, plant_sd=0.05); U = C3.info_vectors(B); X, pr, ns = C3.globe(U)
    L.append(f"continuous methylome (test 3's): {N} sites, {len(var)} variable; construction v4; NSIDE {ns}")
    R = C3.concentration(X, grp); q = float(np.quantile(T3.ref(X, var, rng, 500), 0.99)); P1 = R > q
    L.append(f"P1 planted group R {R:.3f} | 99th percentile of 500 random groups {q:.3f} -> {'MET' if P1 else 'NOT MET'}")
    a, b = X[cls == 1].mean(0), X[cls == 0].mean(0); ang = float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))); P3 = ang >= 90
    L.append(f"P3 island-like vs held-methylated mean directions {ang:.1f} deg -> {'MET' if P3 else 'NOT MET'}")
    cnt, mb, _ = C3.maps(pr, ns, B); cov = float((cnt > 0).mean()); P4 = cov >= 0.90
    L.append(f"P4 pixels holding a site {cov:.1%} (bar 90 %; test 4: 46.9 %) -> {'MET' if P4 else 'NOT MET'}")
    er = T3.edge_ratio(mb, ns); P5 = er < 1.5
    L.append(f"P5 edge ratio {er:.2f} (bar below 1.5; test 3: 21.78) -> {'MET' if P5 else 'NOT MET'}")
    i, j = rng.integers(0, N, 5000), rng.integers(0, N, 5000)
    rho = float(spearmanr(C3.fr_angle(U, i, j), np.arccos(np.clip(np.einsum("ij,ij->i", X[i], X[j]), -1, 1))).correlation); P6 = rho >= 0.7
    L.append(f"P6 distance agreement rho {rho:.3f} (bar at least 0.7; test 4: 0.567) -> {'MET' if P6 else 'NOT MET'}")
    hits = 0
    for k in range(20):
        Bn, _, varn, grpn = T3.methylome(1000 + k); Xn, _, _ = C3.globe(C3.info_vectors(Bn))
        Rn = C3.concentration(Xn, grpn); qn = float(np.quantile(T3.ref(Xn, varn, rng, 200), 0.99)); hits += Rn > qn; rows.append(dict(case=f"null {k}", R=Rn, q99=qn))
    P2 = hits <= 1
    L.append(f"P2 nothing planted: exceeded in {hits} of 20 maps (bar at most 1) -> {'MET' if P2 else 'NOT MET'}")
    ok = P1 and P2 and P3 and P4 and P5 and P6
    L.append("PLANTED TEST 5 " + ("PASSED (bars P1-P6)" if ok else "FAILED: " + ", ".join(n for n, v in (("P1", P1), ("P2", P2), ("P3", P3), ("P4", P4), ("P5", P5), ("P6", P6)) if not v) + " not met"))
    rows.insert(0, dict(case="planted sd 0.05", R=R, q99=q))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "planted_sphere_05_output.csv"), index=False)
    np.savez(os.environ.get("MAPOUT", "/tmp/synthetic_sphere_v5_maps.npz"), count=cnt, mean_b=mb)
    open(os.path.join(HERE, "planted_sphere_05_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
