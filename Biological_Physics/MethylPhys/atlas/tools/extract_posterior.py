"""Atlas v2 posterior summary for chosen cells: per CpG, the posterior mean and SD over the 20 stored draws (s3 atlas_v2/blocks_v2/
block_XXXXX_draws.npz: draws [20 x 74 cells x loci], cells, cpg). One parquet per cell: cpg, mean, sd. Streams the 700 blocks one at a time.
Usage: python3 extract_posterior.py OUTDIR "cell 1" ["cell 2" ...]     (needs AWS read access to the bucket)"""
import os, sys, re, io, boto3, numpy as np, pandas as pd
B = "methylphys-data-945451304272-us-west-2-an"; OUT = sys.argv[1]; WANT = sys.argv[2:]; os.makedirs(OUT, exist_ok=True)
s3 = boto3.client("s3"); keys = sorted(o["Key"] for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix="atlas_v2/blocks_v2/")
                                    for o in pg.get("Contents", []) if o["Key"].endswith("_draws.npz"))
assert len(keys) == 700, len(keys)
acc = {c: [] for c in WANT}
for i, k in enumerate(keys):
    z = np.load(io.BytesIO(s3.get_object(Bucket=B, Key=k)["Body"].read()), allow_pickle=True)
    cells = z["cells"].tolist(); D = z["draws"]; cpg = z["cpg"]
    for c in WANT:
        j = cells.index(c); d = D[:, j, :].astype(np.float64)
        acc[c].append(pd.DataFrame({"cpg": cpg, "mean": d.mean(0), "sd": d.std(0, ddof=1)}))
    if i % 100 == 0: print(i, flush=True)
for c in WANT:
    f = os.path.join(OUT, "atlas_" + re.sub(r"\W+", "_", c).strip("_") + "_posterior.parquet")
    pd.concat(acc[c], ignore_index=True).to_parquet(f, index=False); print(c, "->", f)
