"""The sky null test on the sphere (run_sample.py --build seq-noise-null; stage_seq_atlas.sky_null; checklist D6 on the sphere, D10):
synthetic, read by the one command (rewritten 2026-11-01 so the reading comes from run_sample.py; the first version called the chain
function directly). The instrument's job: a map with no spatial structure stays inside its shuffled null; real large-scale structure does not.
Known answer (seed 20261114): constructed noise columns in the committed layout (halfA, halfB, readA, readB; IAM-Atlas v2.1's sites), ten
cells: "plain 1" to "plain 5", noise N(0, 0.06) at every site; "planted 1" to "planted 5", the same kind of noise plus a change of 0.02 that
rises and falls five times along the genome (sin, genome order). Map 1 of each (half A minus half B) carries the values. The masks of
sky_mask_sites, 20 shuffles each. Bars, on map 1: the plain maps at most one band above their null each; the planted maps' lowest band
(multipoles 2 to 8) above its null in all five.
Usage: python3 sky_null_synthetic.py WORKDIR OUT.txt   (needs the array manifests, as the sphere build does)"""
import os, sys, ast, subprocess, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "..", "chain"); sys.path[:0] = [os.path.join(CH, "MethylPhys_Interface"), os.path.join(CH, "..", "atlas", "tools"), CH]
import reference_globes as RG, sky_maps_v21 as SM
RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); AT = os.path.join(CH, "..", "atlas", "v2.1"); W = sys.argv[1]; os.makedirs(W, exist_ok=True)
ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist(); order, _ = SM.genome_order(RG.manifests(), ids)
rng = np.random.default_rng(20261114); n = len(ids); val = {}
for k in range(1, 6):
    val[f"plain {k}"] = rng.normal(0, 0.06, n); w = rng.normal(0, 0.06, n); w[order] += 0.02 * np.sin(2 * np.pi * 5 * np.arange(len(order)) / len(order)); val[f"planted {k}"] = w
enc = lambda x: np.rint(np.clip(x, 0, 1) * 1e4).astype(np.uint16); names = list(val); thirds = np.array_split(np.arange(len(names)), 3)
E = {c: rng.normal(0, 0.06, n) for c in names}
for st in ("halfA", "halfB", "readA", "readB"):
    for p, js in enumerate(thirds, 1):
        cols = {}
        for j in js:
            c = names[j]; v = val[c]; e = E[c]; cols[c] = enc({"halfA": 0.5 + v / 2, "halfB": 0.5 - v / 2, "readA": 0.5 + e / 2, "readB": 0.5 - e / 2}[st])
        pq.write_table(pa.table(cols), os.path.join(W, f"{st}_part{p}.parquet"))
r = subprocess.run([sys.executable, RS, "--build", "seq-noise-null", "--build-input", W, "--build-out", os.path.join(W, "null.csv")], capture_output=True, text=True)
if r.returncode: sys.exit(r.stdout[-800:] + r.stderr[-1500:])
D = pd.read_csv(os.path.join(W, "null.csv")); D = D[D["map"].str.startswith("map1")].set_index("cell"); ab = D.bands_above_null.apply(ast.literal_eval)
plain = [len(ab[f"plain {k}"]) for k in range(1, 6)]; planted = [0 in ab[f"planted {k}"] for k in range(1, 6)]
L = ["read by run_sample.py --build seq-noise-null on constructed noise columns (seed 20261114)"] + [f"{c}: bands above null {ab[c]}, ratio {D.ratio[c]}" for c in names]
b1 = all(x <= 1 for x in plain); b2 = all(planted)
L += [f"bar 1 plain, at most one band above its null in each: {plain}: {'met' if b1 else 'not met'}",
      f"bar 2 planted, lowest band above its null in all five: {planted}: {'met' if b2 else 'not met'}", f"synthetic: {'met' if b1 and b2 else 'not met'}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
