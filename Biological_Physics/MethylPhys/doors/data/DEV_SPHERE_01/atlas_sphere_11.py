"""DEV-SPHERE-01 test 11 (written and committed 2026-10-22 before it runs; exploratory, not sealed). Test 10 (construction v5, the level
taken out) met C1 and C3 on IAM-Atlas v2.1's 79 healthy cell types but not C2: 29 of 100 level-matched groups passed a line drawn from groups
of ANY level. That line answers the wrong question. The question is whether a group lands together beyond what its level alone gives
(author 2026-10-22: "YES! We are almost there!"). So here every line is drawn from level-matched groups: for each marker group, 1,000 groups
of the same size and the same distribution of mean methylation (20 bins of 0.05; drawn exactly as test 10 drew its 100 twins; seeded).
The same data, pool, marker groups, 100 level-matched twins and construction (v5) as test 10; the globe and every concentration R by the one
command (run_sample.py --build info-sphere --build-construction v5); this script only draws the groups and compares the chain's R values.
Bars (as test 10, with the lines level-matched):
  C1 sites that behave alike land together beyond their level: at least 10 of the 12 marker groups have R above the 99th percentile of their
     own 1,000 level-matched groups;
  C2 a shared level alone does not: at most 4 of the 100 level-matched twins above their marker group's level-matched line (the binomial bar
     of test 7: about 1 expected);
  C3 distance agreement rho at least 0.7 (unchanged).
Reported, not scored: each marker group's R, its level-matched line, and test 10's any-level line for comparison.
Usage: python3 atlas_sphere_11.py OUTDIR"""
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
LM_N = 1000; LM = {}; rng2 = np.random.default_rng(20261023)
def lm_draw(src, rg):
    pick = []
    for s in src:
        cand = byb.get(int(b[s]), np.asarray([])); cand = cand[~np.isin(cand, pick)]
        if len(cand): pick.append(rg.choice(cand))
    return pick
for k, n in enumerate(names):
    for j in range(LM_N): LM[f"lm {k:02d} {j:04d}"] = lm_draw(G[n], rng2)
    print("level-matched lines drawn for", n, flush=True)
pool.to_parquet(os.path.join(out, "atlas_pool.parquet"))
json.dump({**G, **N, **LM, "pool": list(pool.index)}, open(os.path.join(out, "atlas_groups.json"), "w"))
r = subprocess.run([sys.executable, RS, "--build", "info-sphere", "--build-input", os.path.join(out, "atlas_pool.parquet"), "--build-groups",
                    os.path.join(out, "atlas_groups.json"), "--build-out", os.path.join(out, "atlas_sphere.npz"), "--build-random", "1", "--build-construction", "v5",
                    "--bundle", os.path.join(out, "atlas_sphere.json")], capture_output=True, text=True)
if r.returncode: sys.exit(f"run_sample.py --build failed: {r.stdout[-800:]} {r.stderr[-1500:]}")
os.remove(os.path.join(out, "atlas_pool.parquet")); os.remove(os.path.join(out, "atlas_groups.json"))
P = json.load(open(os.path.join(out, "atlas_sphere.json"))); g = P["groups"]
line = {k: float(np.quantile([g[f"lm {k:02d} {j:04d}"]["R"] for j in range(LM_N)], 0.99)) for k in range(len(names))}
L = ["DEV-SPHERE-01 test 11 (real data, IAM-Atlas v2.1, 79 healthy cell types; exploratory, not sealed; the globe and every R by run_sample.py --build info-sphere, v5)",
     f"{P['sites']} pool sites x {P['cell_types']} cell types; sky covered {P['sky_covered']:.1%} (reported); lines: 99th percentile of {LM_N} level-matched groups per marker group"]
m = 0
for k, n in enumerate(names):
    t = g[n]["R"] > line[k]; m += t; L.append(f"  {n}: R {g[n]['R']} | level-matched line {line[k]:.4f} | {'together' if t else 'not together'}")
b1 = m >= 10; L.append(f"C1 marker groups together beyond their level: {m} of {len(names)} (bar at least 10) -> {'MET' if b1 else 'NOT MET'}")
h = sum(g[x]["R"] > line[int(x.split(" ")[1]) % len(names)] for x in N); b2 = h <= 4
L.append(f"C2 level-matched twins past their level-matched line: {h} of {NULL_N} (bar at most 4) -> {'MET' if b2 else 'NOT MET'}")
b3 = P["distance_rho"] >= 0.7; L.append(f"C3 distance agreement rho {P['distance_rho']} (bar at least 0.7) -> {'MET' if b3 else 'NOT MET'}")
L.append("TEST 11 " + ("PASSED (C1-C3)" if b1 and b2 and b3 else "FAILED: " + ", ".join(n for n, v in (("C1", b1), ("C2", b2), ("C3", b3)) if not v) + " not met"))
open(os.path.join(out, "atlas_sphere_11_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
