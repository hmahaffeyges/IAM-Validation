#!/usr/bin/env python3
"""DEV-ATLAS-EPIC-01 setup: fetch atlas v2 wide parquet + 3 EPIC series (RAW tar) from S3 via presigned URLs, Stage 1 (chain) -> one beta parquet per array."""
import json, os, subprocess, tarfile, glob, gzip, shutil, sys, multiprocessing as mp
import pandas as pd
sys.path.insert(0, os.getcwd())
U = json.load(open("urls.json")); D = "/home/ubuntu/data/dev_atlas_epic_01"; os.makedirs(D, exist_ok=True)
def get(name, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000: return
    subprocess.run(["curl", "-sS", "-f", "-o", dst + ".part", U[name]], check=True); os.replace(dst + ".part", dst)
get("IAMAtlas_v2.parquet", f"{D}/IAMAtlas_v2.parquet")
import pyarrow.parquet as pq
pf = pq.ParquetFile(f"{D}/IAMAtlas_v2.parquet"); names = pf.schema_arrow.names
json.dump(dict(n_rows=pf.metadata.num_rows, columns=names), open("atlas_schema.json", "w"))
print("atlas rows", pf.metadata.num_rows, "cols", len(names), flush=True)
jobs = []
for gse in ["GSE112618", "GSE182379", "GSE110530"]:
    g = f"{D}/{gse}"; os.makedirs(f"{g}/idat", exist_ok=True); os.makedirs(f"{D}/betas", exist_ok=True)
    get(f"{gse}_series_matrix.txt.gz", f"{g}/{gse}_series_matrix.txt.gz"); shutil.copy(f"{g}/{gse}_series_matrix.txt.gz", ".")
    if not os.path.exists(f"{g}/idat/.ok"):
        get(f"{gse}_RAW.tar", f"{g}/{gse}_RAW.tar")
        with tarfile.open(f"{g}/{gse}_RAW.tar") as t: t.extractall(f"{g}/idat")
        open(f"{g}/idat/.ok", "w").write("ok"); os.remove(f"{g}/{gse}_RAW.tar")
    for f in glob.glob(f"{g}/idat/*_Grn.idat.gz"):
        subprocess.run(["gzip", "-t", f], check=True)
        jobs.append((gse, os.path.basename(f).split("_")[0], f, f.replace("_Grn.", "_Red.")))
get("GSE250556_series_matrix.txt.gz", "GSE250556_series_matrix.txt.gz")
print("arrays", len(jobs), flush=True)
def cal(j):
    gse, gsm, g, r = j; out = f"{D}/betas/{gsm}.parquet"
    if os.path.exists(out): return gse, gsm, "exists"
    try:
        tmp = f"/tmp/s1_{gsm}"; os.makedirs(tmp, exist_ok=True); gg, rr = [f"{tmp}/{os.path.basename(x)[:-3]}" for x in (g, r)]
        for s, d in ((g, gg), (r, rr)):
            with gzip.open(s) as a, open(d, "wb") as b: b.write(a.read())
        from stage_1_idat_calibration import calibrate_idat_to_beta
        beta, meta = calibrate_idat_to_beta(gg, rr, verbose=False); beta = beta.iloc[:, 0] if hasattr(beta, "columns") else beta
        beta.astype("float32").to_frame("beta").to_parquet(out); shutil.rmtree(tmp); return gse, gsm, "ok"
    except Exception as e: return gse, gsm, f"error {type(e).__name__}: {str(e)[:200]}"
first = cal(jobs[0]); print(first, flush=True)
with mp.get_context("fork").Pool(32) as p: rows = [first] + list(p.imap_unordered(cal, jobs[1:]))
pd.DataFrame(rows, columns=["gse", "gsm", "status"]).to_csv("calib_status.csv", index=False)
print(pd.DataFrame(rows)[2].str[:5].value_counts().to_dict())
