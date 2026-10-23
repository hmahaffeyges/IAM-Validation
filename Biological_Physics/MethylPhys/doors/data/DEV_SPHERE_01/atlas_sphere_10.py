"""DEV-SPHERE-01 test 10 (written and committed 2026-10-22 before it runs): test 8 unchanged in every respect except the construction, v5
(--build-construction v5: each site's square-root vector minus that of a flat profile at its own mean level, the level taken out), because
test 8 found a site's place on real cells set mostly by its level. Runs only after test 9 passed; the same groups, level-matched twins and bars as test 8.
"""
"""DEV-SPHERE-01 test 10: the information sphere on real healthy cells (written and committed 2026-10-22, BEFORE it is run; exploratory,
not sealed: "we explore without sealing until we are certain"). The first test on real data, after planted test 7 passed on synthetic data.
The instrument's job: on real healthy cells, the information sphere places together the sites that behave alike, and not the sites that
merely share a methylation level.
The globe is built by the one command, chain/MethylPhys_Interface/run_sample.py --build info-sphere (construction v5,
chain/stage_info_sphere.py, unchanged). This file only writes its input and reads the bundle it writes.
Input: IAM-Atlas v2.1's best value (mu) for all 79 cell types at every site measured in all 79, keeping the variable sites (spread across
the 79 above 0.08, test 7's rule): the pool.
Groups (fixed here, from the atlas alone, before the globe is built):
  marker groups (12): for each of the six blood cells of the Deconvolver's panel (neutrophils, monocytes, b cells, cd4 t cells, cd8 t cells,
     nk cells), the 100 pool sites where it sits furthest below every other of the six (its nearest one), and the 100 where it sits furthest
     above them: sites known to behave alike.
  level-matched groups (100): for each marker group in turn, a group of the same size drawn from the pool's other sites with the same
     distribution of mean methylation across the 79 cells (20 bins of 0.05; seeded): the same levels, no shared behavior.
Checks (at most three, KISS; fixed before running):
  C1 sites that behave alike land together: at least 10 of the 12 marker groups have concentration R above the 99th percentile of 1,000
     random pool groups of the same size;
  C2 a shared level alone does not: at most 4 of the 100 level-matched groups past that line (the binomial bar of test 7);
  C3 distances kept: rank agreement of globe angles with full-space Fisher-Rao angles (5,000 seeded pairs) at least 0.7.
Reported, not scored: sites, variance kept by the three directions, sky covered, each marker group's R and line.
Usage: python3 atlas_sphere_10.py OUTDIR   (writes OUTDIR/atlas_sphere_10_summary.txt and the bundle)"""
import os, sys, json, subprocess, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(H, "..", "..", ".."))
RS = os.path.join(MP, "chain", "MethylPhys_Interface", "run_sample.py"); AT = os.path.join(MP, "atlas", "v2.1")
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
BLOOD6 = ["neutrophils", "monocytes", "b cells", "cd4 t cells", "cd8 t cells", "nk cells"]; PER, NULL_N, BINS = 100, 100, 20
parts = []
for n in (1, 2, 3):
    T = pq.read_table(os.path.join(AT, f"mu_part{n}.parquet")).to_pandas()
    parts.append(T.astype("float32").where(T != 65535) / 10000.0)
B = pd.concat(parts, axis=1); assert B.shape[1] == 79, B.shape; B.index = [f"s{i}" for i in range(len(B))]
B = B[B.notna().all(1)]; pool = B[B.std(1) > 0.08]
print("sites measured in all 79:", len(B), "| pool (spread > 0.08):", len(pool), flush=True)
G = {}
for c in BLOOD6:
    d = pool[[o for o in BLOOD6 if o != c]].sub(pool[c], axis=0)
    G[f"{c}: below every other blood cell"] = list(d.min(1).nlargest(PER).index)
    G[f"{c}: above every other blood cell"] = list((-d).min(1).nlargest(PER).index)
mark = set(x for v in G.values() for x in v); mean = pool.mean(1); b = np.minimum((mean * BINS).astype(int), BINS - 1)
rest = b[~b.index.isin(mark)]; byb = {k: np.asarray(v.index) for k, v in rest.groupby(rest)}
names = list(G); rng = np.random.default_rng(20261022); N = {}
for k in range(NULL_N):
    src = G[names[k % len(names)]]; pick = []
    for s in src:
        cand = byb.get(int(b[s]), np.asarray([])); cand = cand[~np.isin(cand, pick)]
        if len(cand): pick.append(rng.choice(cand))
    N[f"level-matched {k:03d} (as {names[k % len(names)]})"] = pick
pool.to_parquet(os.path.join(out, "atlas_pool.parquet"))
json.dump({**G, **N, "pool": list(pool.index)}, open(os.path.join(out, "atlas_groups.json"), "w"))
r = subprocess.run([sys.executable, RS, "--build", "info-sphere", "--build-input", os.path.join(out, "atlas_pool.parquet"), "--build-groups",
                    os.path.join(out, "atlas_groups.json"), "--build-out", os.path.join(out, "atlas_sphere.npz"), "--build-random", "1000", "--build-construction", "v5",
                    "--bundle", os.path.join(out, "atlas_sphere.json")], capture_output=True, text=True)
if r.returncode: sys.exit(f"run_sample.py --build failed: {r.stdout[-800:]} {r.stderr[-1500:]}")
os.remove(os.path.join(out, "atlas_pool.parquet"))
P = json.load(open(os.path.join(out, "atlas_sphere.json"))); g = P["groups"]
L = ["DEV-SPHERE-01 test 10 (real data, IAM-Atlas v2.1, 79 healthy cell types; exploratory, not sealed; the globe built by run_sample.py --build info-sphere)",
     f"{P['sites']} pool sites x {P['cell_types']} cell types; variance kept {P['variance_kept']}; sky covered {P['sky_covered']:.1%} (masked {P['sky_masked']:.1%}) (reported)"]
L += [f"  {n}: R {g[n]['R']} | line {g[n]['random_q99']} | {'together' if g[n]['together'] else 'not together'}" for n in names]
m = sum(g[n]["together"] for n in names); b1 = m >= 10
L.append(f"C1 marker groups together: {m} of {len(names)} (bar at least 10) -> {'MET' if b1 else 'NOT MET'}")
h = sum(g[n]["together"] for n in N); b2 = h <= 4
L.append(f"C2 level-matched groups past the line: {h} of {NULL_N} (bar at most 4) -> {'MET' if b2 else 'NOT MET'}")
b3 = P["distance_rho"] >= 0.7; L.append(f"C3 distance agreement rho {P['distance_rho']} (bar at least 0.7) -> {'MET' if b3 else 'NOT MET'}")
L.append("TEST 10 " + ("PASSED (C1-C3)" if b1 and b2 and b3 else "FAILED: " + ", ".join(n for n, v in (("C1", b1), ("C2", b2), ("C3", b3)) if not v) + " not met"))
open(os.path.join(out, "atlas_sphere_10_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
