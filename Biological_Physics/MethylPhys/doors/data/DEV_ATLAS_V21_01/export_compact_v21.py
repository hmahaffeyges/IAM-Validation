"""Atlas v2.1 compact copy for the repository (author 2026-10-16: keep a copy of the atlas in the repo). From atlas_v21_summary.parquet (the
assembled fit, S3 results/ATLAS_V21/assembly/) writes, into <out>:
  sites.parquet   every site id (813,951), in the row order of the three tables
  mu_partN.parquet     best estimate (posterior mean) per site (rows) and cell (columns, the source atlases' exact names); the 79 cells are
                       split into 3 parts in alphabetical order so every file stays under GitHub's 100 MB limit
  mu_lo_partN.parquet  2.5th percentile;  mu_hi_partN.parquet  97.5th percentile
Values are stored as round(value x 10000) in uint16 (exact to 4 decimals); 65535 = not measured. Read: value = stored / 10000.
Nothing is computed beyond rounding to 4 decimals. Usage: python export_compact_v21.py <atlas_v21_summary.parquet> <out>"""
import os, sys, hashlib, json, numpy as np, pandas as pd
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
A = pd.read_parquet(src, columns=["cell", "cpg", "n_obs", "mu", "mu_lo", "mu_hi"]); A = A[A.n_obs > 0]
sites = np.sort(pd.read_parquet(src, columns=["cpg"]).cpg.unique()); cells = sorted(A.cell.unique()); pos = pd.Index(sites)
pd.DataFrame({"cpg": sites}).to_parquet(os.path.join(out, "sites.parquet"), index=False, compression="zstd")
r = pos.get_indexer(A.cpg); c = pd.Index(cells).get_indexer(A.cell)
for k in ("mu", "mu_lo", "mu_hi"):
    M = np.full((len(sites), len(cells)), 65535, np.uint16); M[r, c] = np.round(A[k].values * 10000).astype(np.uint16)
    for n, part in enumerate(np.array_split(np.arange(len(cells)), 3), 1):
        pd.DataFrame(M[:, part], columns=[cells[i] for i in part]).to_parquet(os.path.join(out, f"{k}_part{n}.parquet"), index=False, compression="zstd", compression_level=19)
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
J = {f: dict(bytes=os.path.getsize(os.path.join(out, f)), sha256=sha(os.path.join(out, f))) for f in sorted(os.listdir(out)) if f.endswith(".parquet")}
J["_source"] = dict(file=os.path.basename(src), sha256=sha(src) if os.path.getsize(src) < 4e9 else "see output_manifest_v21.json", cells=len(cells), sites=len(sites), measured_pairs=int(len(A)))
json.dump(J, open(os.path.join(out, "compact_manifest.json"), "w"), indent=1); print(json.dumps({k: (v["bytes"] if "bytes" in v else v) for k, v in J.items()}))
