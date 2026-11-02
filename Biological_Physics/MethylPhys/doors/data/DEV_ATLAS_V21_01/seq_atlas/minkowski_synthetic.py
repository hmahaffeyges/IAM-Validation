"""Minkowski functionals on the sphere (run_sample.py --build seq-noise-minkowski; stage_seq_atlas.sky_minkowski; checklist D13):
synthetic, written and committed before it runs, read by the one command. The instrument's job: a map with no spatial structure keeps its
boundary length and genus inside the range of shuffles of its own pixels; a map with large-scale structure does not.
Known answer (seed 20261122): the sky-null synthetic's design (sky_null_synthetic.py): ten constructed cells, "plain 1" to "plain 5", noise
N(0, 0.06) at every site; "planted 1" to "planted 5", the same noise plus 0.02 rising and falling five times along the genome. Map 1 of each
carries the values; the masks of sky_mask_sites; 20 shuffles. Bars, on map 1: of the plain maps' 50 tests (5 thresholds, V1 and V2), at
most 10 outside their shuffles (with 20 shuffles a structureless map falls outside about 1 test in 10 by chance); every planted map's V1
at threshold 0 outside its shuffles.
Usage: python3 minkowski_synthetic.py WORKDIR OUT.txt   (needs the array manifests, as the sphere build does)"""
import os, sys, ast, subprocess, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "..", "chain"); sys.path[:0] = [os.path.join(CH, "MethylPhys_Interface"), os.path.join(CH, "..", "atlas", "tools"), CH]
import reference_globes as RG, sky_maps_v21 as SM
RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); AT = os.path.join(CH, "..", "atlas", "v2.1"); W = sys.argv[1]; os.makedirs(W, exist_ok=True)
ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist(); order, _ = SM.genome_order(RG.manifests(), ids)
rng = np.random.default_rng(20261122); n = len(ids); val = {}
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
r = subprocess.run([sys.executable, RS, "--build", "seq-noise-minkowski", "--build-input", W, "--build-out", os.path.join(W, "mink.csv")], capture_output=True, text=True)
if r.returncode: sys.exit(r.stdout[-800:] + r.stderr[-1500:])
D = pd.read_csv(os.path.join(W, "mink.csv")); D = D[D["map"].str.startswith("map1")]
pl = D[D.cell.str.startswith("plain")]; out_pl = int(pl.V1_outside_null.sum() + pl.V2_outside_null.sum())
pv = D[D.cell.str.startswith("planted") & (D.nu == 0)].set_index("cell").V1_outside_null
L = ["read by run_sample.py --build seq-noise-minkowski on constructed noise columns (seed 20261122)"] + \
    [f"{c} nu {r.nu}: V0 {r.V0} V1 {r.V1} ({r.V1_null_min} to {r.V1_null_max}) V2 {r.V2} ({r.V2_null_min:g} to {r.V2_null_max:g})" for c, g in D.groupby("cell") for r in g.itertuples()]
b1 = out_pl <= 10 and len(pl) == 25; b2 = len(pv) == 5 and bool(pv.all())
L += [f"bar 1 plain maps, tests outside their shuffles: {out_pl} of {2 * len(pl)} (at most 10): {'met' if b1 else 'not met'}",
      f"bar 2 planted maps, V1 at threshold 0 outside its shuffles: {pv.tolist()}: {'met' if b2 else 'not met'}", f"synthetic: {'met' if b1 and b2 else 'not met'}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L[-3:]))
