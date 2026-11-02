"""The genome-distance spectrum (run_sample.py --build seq-noise-spectrum; checklist D6): synthetic, written and committed before it runs,
read by the one command. The instrument's job: values with no regional structure read no correlation at any distance; values that share
1 Mb blocks read a correlation within a block that falls beyond it.
Known answer (seed 20261115): constructed noise columns in the committed layout (halfA, halfB, readA, readB; three parts; IAM-Atlas v2.1's
sites), two cells: "plain", noise N(0, 0.06) at every site; "blocks", the same kind of noise plus one value per 1 Mb block, N(0, 0.03) (a
quarter of the variance: a correlation of 0.2 within a block). Map 1 of each (half A minus half B) carries the value; map 2 carries fresh
noise. Bars, on map 1: plain at most 0.01 in absolute value in every band; blocks at least 0.1 in the bands under 100 kb, and in the 10 to
100 Mb band at most a quarter of the 10 to 100 kb band.
Usage: python3 genome_spectrum_synthetic.py WORKDIR OUT.txt"""
import os, sys, ast, subprocess, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(H, "..", "..", "..", "..", "chain"); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
AT = os.path.join(CH, "..", "atlas", "v2.1"); W = sys.argv[1]; os.makedirs(W, exist_ok=True)
ids = pq.read_table(os.path.join(AT, "sites.parquet")).column("cpg").to_pylist()
K = pq.read_table(os.path.join(AT, "sites_sequencing_key.parquet")).to_pandas().drop_duplicates("probe").set_index("probe").reindex(ids)
rng = np.random.default_rng(20261115); n = len(ids)
blk = pd.Series(list(zip(K.chrom_hg19.fillna("none"), (K.pos_hg19.fillna(0) // 1e6).astype(int)))); u = {b: x for b, x in zip(blk.unique(), rng.normal(0, 0.03, blk.nunique()))}
val = {"plain": rng.normal(0, 0.06, n), "blocks": rng.normal(0, 0.06, n) + blk.map(u).to_numpy()}
enc = lambda x: np.rint(np.clip(x, 0, 1) * 1e4).astype(np.uint16)
for st in ("halfA", "halfB", "readA", "readB"):
    for k in (1, 2, 3):
        cols = {}
        if k == 1:
            for c, v in val.items():
                e = rng.normal(0, 0.06, n)
                cols[c] = enc({"halfA": 0.5 + v / 2, "halfB": 0.5 - v / 2, "readA": 0.5 + e / 2, "readB": 0.5 - e / 2}[st])
        pq.write_table(pa.table(cols) if cols else pa.table({"_": pa.array([], pa.uint16())}).drop(["_"]), os.path.join(W, f"{st}_part{k}.parquet"))
r = subprocess.run([sys.executable, RS, "--build", "seq-noise-spectrum", "--build-input", W, "--build-out", os.path.join(W, "spec.csv")], capture_output=True, text=True)
if r.returncode: sys.exit(r.stdout[-800:] + r.stderr[-1500:])
D = pd.read_csv(os.path.join(W, "spec.csv")); D["corr"] = D["corr"].apply(ast.literal_eval); m1 = D[D["map"].str.startswith("map1")].set_index("cell")["corr"]
a, b = m1["plain"], m1["blocks"]
b1 = all(x is not None and abs(x) <= 0.01 for x in a); b2 = all(x >= 0.1 for x in b[:3]) and b[5] <= b[2] / 4
L = ["read by run_sample.py --build seq-noise-spectrum on constructed noise columns (seed 20261115)", f"plain, map 1: {a}", f"blocks, map 1: {b}",
     f"bar 1 plain, |correlation| at most 0.01 in every band: {'met' if b1 else 'not met'}",
     f"bar 2 blocks, at least 0.1 under 100 kb and the 10 to 100 Mb band at most a quarter of 10 to 100 kb: {'met' if b2 else 'not met'}", f"synthetic: {'met' if b1 and b2 else 'not met'}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
