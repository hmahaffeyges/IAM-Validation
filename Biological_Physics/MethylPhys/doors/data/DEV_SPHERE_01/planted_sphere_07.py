"""DEV-SPHERE-01 planted test 7 (written and committed 2026-10-22 after test 6, BEFORE it is run; synthetic only; the chain's way).
Test 6 met C1 and C3 and missed C2 (2 of 20 null groups past a 99th-percentile line, both by hairs; chance gives 2 or more of 20 about
1.7 % of the time). Test 7 is test 6 unchanged except in size, so chance alone cannot decide C2: 100 methylomes with nothing planted, the
line from 1,000 random groups, and C2's bar set by the binomial at 1 % (at most 4 of 100; chance gives 5 or more 0.3 % of the time).
The instrument's job: the information sphere shows how far apart states are. Every globe here is built by the one command,
chain/MethylPhys_Interface/run_sample.py --build info-sphere (construction v1, chain/stage_info_sphere.py; the empty sky masked, not filled).
This file only writes the input matrices (test 3's continuous methylome, planted_sphere_03.methylome, unchanged: 60,000 sites, 40 cell types,
1,200 planted sites given one shared profile) and reads the bundles the one command writes.
Checks (at most three, KISS; fixed before running):
  C1 the planted group lands together: its concentration R above the 99th percentile of 1,000 random groups of variable sites;
  C2 nothing lands together when nothing is planted: R above that percentile in at most 4 of 100 methylomes with nothing planted;
  C3 distances kept: rank agreement of globe angles with full-space Fisher-Rao angles (5,000 seeded pairs) at least 0.7.
Reported, not scored: share of the sky the sites cover (the rest is masked), edge ratio, variance the three directions keep.
Usage: python3 planted_sphere_07.py OUTDIR   (writes OUTDIR/planted_sphere_07_summary.txt and the bundles)"""
import os, sys, json, subprocess, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__))
# test 3's continuous methylome, copied verbatim from planted_sphere_03.py (that file imports healpy, which the chain does not use)
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
class T3: methylome = staticmethod(methylome)
RS = os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"); out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def one(tag, seed, plant):
    B, cls, var, grp = T3.methylome(seed, plant_sd=0.05 if plant else None); ids = [f"s{i}" for i in range(len(B))]
    pd.DataFrame(B, index=ids, columns=[f"c{j}" for j in range(B.shape[1])]).to_parquet(os.path.join(out, tag + ".parquet"))
    json.dump({"planted": [ids[i] for i in grp], "pool": [ids[i] for i in var]}, open(os.path.join(out, tag + "_groups.json"), "w"))
    r = subprocess.run([sys.executable, RS, "--build", "info-sphere", "--build-input", os.path.join(out, tag + ".parquet"), "--build-groups", os.path.join(out, tag + "_groups.json"),
                        "--build-out", os.path.join(out, tag + ".npz"), "--build-random", "1000", "--bundle", os.path.join(out, tag + ".json")], capture_output=True, text=True)
    if r.returncode: sys.exit(f"{tag}: run_sample.py --build failed: {r.stdout[-500:]} {r.stderr[-800:]}")
    os.remove(os.path.join(out, tag + ".parquet")); [os.remove(os.path.join(out, tag + x)) for x in (".npz",) if tag != "planted"]; return json.load(open(os.path.join(out, tag + ".json")))
P = one("planted", 1, True); g = P["groups"]["planted"]; L = [f"DEV-SPHERE-01 planted test 7 (synthetic; every globe built by run_sample.py --build info-sphere, construction v1)",
     f"{P['sites']} sites x {P['cell_types']} cell types; variance kept {P['variance_kept']}; sky covered {P['sky_covered']:.1%} (masked {P['sky_masked']:.1%}); edge ratio {P['edge_ratio']} (reported)"]
b1 = g["together"]; L.append(f"C1 planted group R {g['R']} | 99th percentile of 1,000 random groups {g['random_q99']} -> {'MET' if b1 else 'NOT MET'}")
hits = 0
for k in range(100):
    Z = one(f"null_{k:02d}", 1000 + k, False); hits += Z["groups"]["planted"]["together"]; print("null", k, Z["groups"]["planted"], flush=True)   # 2026-10-22: was N, which overwrote the methylome's site count
b2 = hits <= 4; L.append(f"C2 nothing planted: together in {hits} of 100 (bar at most 4) -> {'MET' if b2 else 'NOT MET'}")
b3 = P["distance_rho"] >= 0.7; L.append(f"C3 distance agreement rho {P['distance_rho']} (bar at least 0.7) -> {'MET' if b3 else 'NOT MET'}")
L.append("PLANTED TEST 7 " + ("PASSED (C1-C3)" if b1 and b2 and b3 else "FAILED: " + ", ".join(n for n, v in (("C1", b1), ("C2", b2), ("C3", b3)) if not v) + " not met"))
open(os.path.join(out, "planted_sphere_07_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
