"""DEV-SPHERE-01 planted test 2 (written and committed 2026-10-13, BEFORE it was run): planted test 1, unchanged, on construction v2,
plus one new bar. Synthetic only; no real data is read.

Same synthetic methylome and plant as planted_sphere_01.py (S 40, N 60,000, 1,200 planted sites, noise sd 0.05).
Bars, fixed before running:
  P1 the planted group's concentration R is above the 99th percentile of 500 random groups of 1,200 variable sites.
  P2 in 50 maps with nothing planted, the same sites exceed their map's 99th percentile in at most 3 of 50.
  P3 the mean directions of island-like and held-methylated sites are at least 90 degrees apart.
  P4 NEW - the globe is full: at least 90 % of pixels hold a site.
Reported, no bar: rank agreement of globe angle with full-space Fisher-Rao angle (5,000 pairs); the weaker plant (sd 0.15); the
chain's band powers of the mean-methylation map, planted against not planted. Writes planted_sphere_02_summary.txt, _output.csv.
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sphere_construction_02 as C2
from planted_sphere_01 import methylome, M  # the identical synthetic methylome (importing runs nothing: guarded below)
from scipy.stats import spearmanr
def ref(X, var, rng, k):
    return np.array([C2.concentration(X, rng.choice(var, M, replace=False)) for _ in range(k)])
L, rows = [], []; rng = np.random.default_rng(20261013)
B, cls, var, grp = methylome(1, plant_sd=0.05); U = C2.info_vectors(B); X, pr, ns = C2.globe(U)
R = C2.concentration(X, grp); q = float(np.quantile(ref(X, var, rng, 500), 0.99)); P1 = R > q
L.append(f"NSIDE {ns} ({12 * ns * ns} pixels for {len(B)} sites)")
L.append(f"P1 planted group R {R:.3f} | 99th percentile of 500 random groups {q:.3f} -> {'MET' if P1 else 'NOT MET'}")
a, b = X[cls == 1].mean(0), X[cls == 0].mean(0); ang = float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))); P3 = ang >= 90
L.append(f"P3 island-like vs held-methylated mean directions {ang:.1f} deg -> {'MET' if P3 else 'NOT MET'}")
cnt, mb, _ = C2.maps(pr, ns, B); cov = float((cnt > 0).mean()); P4 = cov >= 0.90
L.append(f"P4 pixels holding a site {cov:.1%} (bar 90 %) -> {'MET' if P4 else 'NOT MET'}")
i, j = rng.integers(0, len(B), 5000), rng.integers(0, len(B), 5000)
rho = float(spearmanr(C2.fr_angle(U, i, j), np.arccos(np.clip(np.einsum("ij,ij->i", X[i], X[j]), -1, 1))).correlation)
L.append(f"reported: rank agreement globe vs full-space angle rho {rho:.3f} (v1: 0.892)")
Bw, _, varw, grpw = methylome(1, plant_sd=0.15); Xw, _, _ = C2.globe(C2.info_vectors(Bw)); Rw = C2.concentration(Xw, grpw); qw = float(np.quantile(ref(Xw, varw, rng, 500), 0.99))
L.append(f"reported: weaker plant (sd 0.15) R {Rw:.3f} | 99th percentile {qw:.3f} -> {'found' if Rw > qw else 'not found'}")
hits = 0
for k in range(50):
    Bn, _, varn, grpn = methylome(1000 + k, plant_sd=None); Xn, prn, _ = C2.globe(C2.info_vectors(Bn))
    Rn = C2.concentration(Xn, grpn); qn = float(np.quantile(ref(Xn, varn, rng, 200), 0.99)); hits += Rn > qn; rows.append(dict(case=f"null {k}", R=Rn, q99=qn))
    if k == 0: mb0 = C2.maps(prn, ns, Bn)[1]
P2 = hits <= 3
L.append(f"P2 nothing planted: exceeded in {hits} of 50 maps (bar at most 3) -> {'MET' if P2 else 'NOT MET'}")
def bp(m):
    d = m - np.nanmean(m); cl, _, _ = C2.SS.masked_spectrum(d); return C2.SS.bandpowers(cl)
bpa, bpb = bp(mb), bp(mb0)
L.append("reported: band power of the mean-methylation map, planted / not planted: " + ", ".join(f"l {x}-{y}: {p / r:.2f}" for (x, y), p, r in zip(C2.SS.BANDS, bpa, bpb)))
ok = P1 and P2 and P3 and P4
L.append("PLANTED TEST 2 " + ("PASSED: v2 keeps everything v1 found and fills the globe" if ok else "FAILED"))
rows.insert(0, dict(case="planted sd 0.05", R=R, q99=q)); rows.insert(1, dict(case="planted sd 0.15", R=Rw, q99=qw))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "planted_sphere_02_output.csv"), index=False)
open(os.path.join(HERE, "planted_sphere_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
