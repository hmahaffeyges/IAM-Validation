"""Atlas v2 posterior summary for chosen cells: per CpG, the posterior mean and SD (population form, divided by 20, as the prostate file of DEV-FINGERPRINT-01) over the 20 stored draws (s3 atlas_v2/blocks_v2/
block_XXXXX_draws.npz: draws [20 x 74 cells x loci], cells, cpg). One parquet per cell: cpg, mean, sd. Streams the 700 blocks one at a time.
Usage: python3 extract_posterior.py OUTDIR "cell 1" ["cell 2" ...]     (the blocks fetched from the public Zenodo record by fetch_atlas.py: BLOCKS_DIR=<dir>, or, if unset, fetched here into
~/.cache/iam_atlas_v2 and verified file by file. 2026-10-20: the cloud-bucket route was removed; the published record is the only source)"""
import os, sys, re, io, numpy as np, pandas as pd
OUT = sys.argv[1]; WANT = sys.argv[2:]; os.makedirs(OUT, exist_ok=True)
LOCAL = os.environ.get("BLOCKS_DIR")   # the blocks as fetched from the public Zenodo record (atlas/tools/fetch_atlas.py), no cloud access
if not LOCAL:
    import subprocess
    _c = os.path.join(os.path.expanduser("~"), ".cache", "iam_atlas_v2")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_atlas.py"), _c, "23325865"], check=True)
    LOCAL = os.path.join(_c, "blocks_v2")
keys = sorted(os.path.join(d, f) for d, _, fs in os.walk(LOCAL) for f in fs if f.endswith("_draws.npz"))
load = lambda k: np.load(k, allow_pickle=True)
assert len(keys) == 700, len(keys)
acc = {c: [] for c in WANT}
for i, k in enumerate(keys):
    z = load(k)
    cells = z["cells"].tolist(); D = z["draws"]; cpg = z["cpg"]
    for c in WANT:
        j = cells.index(c); d = D[:, j, :].astype(np.float64)
        acc[c].append(pd.DataFrame({"cpg": cpg, "mean": d.mean(0), "sd": d.std(0)}))
    if i % 100 == 0: print(i, flush=True)
for c in WANT:
    f = os.path.join(OUT, "atlas_" + re.sub(r"\W+", "_", c).strip("_") + "_posterior.parquet")
    pd.concat(acc[c], ignore_index=True).to_parquet(f, index=False); print(c, "->", f)
