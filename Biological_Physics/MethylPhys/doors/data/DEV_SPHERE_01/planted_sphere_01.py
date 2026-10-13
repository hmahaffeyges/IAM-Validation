"""DEV-SPHERE-01 planted test 1 (written and committed 2026-10-13, BEFORE it was run). Synthetic data only; no real data is read.

Synthetic methylome: S = 40 cell types, N = 60,000 sites in three classes, like a real array:
  55 % held methylated (b about 0.88 in every cell type), 25 % CpG-island-like (b about 0.04 everywhere),
  20 % variable (b drawn independently in each cell type from Beta(0.5, 0.5)).
Planted: 1,200 of the variable sites made to move TOGETHER - one shared profile across the 40 cell types plus per-site noise
(sd 0.05) - a group of sites held in order together. Construction: sphere_construction_01.py, unchanged.

Bars, fixed before running:
  P1 finds what is planted: the planted group's concentration on the globe (mean resultant length R) is above the 99th percentile
     of R for 500 random groups of 1,200 variable sites in the same map.
  P2 finds nothing when nothing is planted: in 50 maps built without the plant (fresh seeds), the same 1,200 sites exceed that map's
     99th percentile in at most 3 of 50 (about 0.5 expected by chance).
  P3 the two big classes separate: the mean directions of island-like and held-methylated sites are at least 90 degrees apart.
Reported, no bar (what we are learning): the variance fraction the globe keeps; the rank agreement between full-space Fisher-Rao
angles and globe angles on 5,000 random pairs; a weaker plant (noise sd 0.15) against the same reference; and the chain's power
spectrum (sky_statistics.masked_spectrum) of the site-density map with and without the plant, by band.
Writes planted_sphere_01_summary.txt and planted_sphere_01_output.csv.
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import sphere_construction_01 as C
from scipy.stats import spearmanr
S, N, M = 40, 60000, 1200
def methylome(seed, plant_sd=None):
    r = np.random.default_rng(seed)
    n1, n2 = int(0.55 * N), int(0.25 * N); n3 = N - n1 - n2
    B = np.vstack([np.clip(r.normal(0.88, 0.04, (n1, S)), 0, 1), np.clip(r.normal(0.04, 0.02, (n2, S)), 0, 1), r.beta(0.5, 0.5, (n3, S))])
    cls = np.r_[np.zeros(n1, int), np.ones(n2, int), 2 * np.ones(n3, int)]
    var = np.where(cls == 2)[0]; grp = var[:M]
    if plant_sd is not None:
        prof = r.beta(0.5, 0.5, S); B[grp] = np.clip(prof + r.normal(0, plant_sd, (M, S)), 0, 1)
    return B, cls, var, grp
def reference(X, var, rng, k=500):
    return np.array([C.concentration(X, rng.choice(var, M, replace=False)) for _ in range(k)])
def density(X):
    cnt, _, _ = C.maps(X, np.zeros((len(X), 1))); d = cnt / cnt.mean() - 1.0; return d
L, rows = [], []
rng = np.random.default_rng(20261013)
# planted, strong
B, cls, var, grp = methylome(1, plant_sd=0.05); U = C.info_vectors(B); X, keep = C.globe(U)
R = C.concentration(X, grp); ref = reference(X, var, rng); q99 = float(np.quantile(ref, 0.99))
P1 = R > q99
L.append(f"P1 planted group R {R:.3f} | 99th percentile of 500 random groups {q99:.3f} -> {'MET' if P1 else 'NOT MET'}")
rows.append(dict(case="planted sd 0.05", R=R, q99=q99))
# P3
a = X[cls == 1].mean(0); b = X[cls == 0].mean(0)
ang = float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))); P3 = ang >= 90
L.append(f"P3 angle between island-like and held-methylated mean directions {ang:.1f} deg -> {'MET' if P3 else 'NOT MET'}")
# reported: variance kept, rank agreement
i, j = rng.integers(0, N, 5000), rng.integers(0, N, 5000)
fr = C.fr_angle(U, i, j); gl = np.arccos(np.clip(np.einsum("ij,ij->i", X[i], X[j]), -1, 1)); rho = float(spearmanr(fr, gl).correlation)
L.append(f"reported: the globe keeps {keep:.1%} of the full-space variance; rank agreement full-space vs globe angle rho {rho:.3f} (5,000 pairs)")
# reported: weak plant
Bw, _, varw, grpw = methylome(1, plant_sd=0.15); Xw, _ = C.globe(C.info_vectors(Bw)); Rw = C.concentration(Xw, grpw); qw = float(np.quantile(reference(Xw, varw, rng), 0.99))
L.append(f"reported: weaker plant (sd 0.15) R {Rw:.3f} | 99th percentile {qw:.3f} -> {'found' if Rw > qw else 'not found'}")
rows.append(dict(case="planted sd 0.15", R=Rw, q99=qw))
# P2: 50 null maps
hits = 0
for k in range(50):
    Bn, _, varn, grpn = methylome(1000 + k, plant_sd=None); Xn, _ = C.globe(C.info_vectors(Bn))
    Rn = C.concentration(Xn, grpn); qn = float(np.quantile(reference(Xn, varn, rng, 200), 0.99)); hits += Rn > qn
    rows.append(dict(case=f"null {k}", R=Rn, q99=qn))
P2 = hits <= 3
L.append(f"P2 nothing planted: the same 1,200 sites exceed their map's 99th percentile in {hits} of 50 maps (bar: at most 3) -> {'MET' if P2 else 'NOT MET'}")
# reported: the chain's power spectrum of the site-density map, planted vs not
Bn, _, _, _ = methylome(1000, plant_sd=None); Xn, _ = C.globe(C.info_vectors(Bn))
bp = {}
for name, XX in (("planted", X), ("not planted", Xn)):
    cl, fsky, _ = C.SS.masked_spectrum(density(XX)); bp[name] = C.SS.bandpowers(cl)
L.append("reported: band power of the site-density map, planted / not planted, by band " +
         ", ".join(f"l {a}-{b}: {p / q:.2f}" for (a, b), p, q in zip(C.SS.BANDS, bp["planted"], bp["not planted"])))
ok = P1 and P2 and P3
L.append("PLANTED TEST " + ("PASSED: the construction finds a planted group of sites held in order together, finds nothing when nothing is planted, and separates the two big classes" if ok else "FAILED"))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "planted_sphere_01_output.csv"), index=False)
open(os.path.join(HERE, "planted_sphere_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
