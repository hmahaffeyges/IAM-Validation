#!/usr/bin/env python3
"""Genotype (rs) probe values for atlas v2's own arrays, for the half-donor split (decision 17). Written 2026-10-14, committed before it runs.
v2's array sources carry no donor field in GEO (Moss GSE122126, Salas GSE110554 and GSE167998, GSE63409), so who-is-who is read from the
arrays' own genotype probes, exactly as identity_check_v21.py does for the v2.1 arrays. The chain's Stage 1 (calibrate_idat_to_beta,
return_snp=True) is run on each QC-passing v2 array's IDATs already on this disk; only the rs betas are kept (the betas themselves are v2's,
unchanged, and are not rewritten). Writes v2_rs.parquet (array x rs probe) and v2_rs_arrays.csv, uploaded to S3. Computes no split."""
import os, sys, glob, json, time, pandas as pd, boto3
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
OUTP = "results/ATLAS_V21/halfsplit/"; BUCKET = "methylphys-data-945451304272-us-west-2-an"
CODE = ["v21_v2rs_box.py", "publish_code.py", "stage_1_idat_calibration.py", "roster_samples.csv"]
publish_code(OUTP + "code/v21_v2rs/", CODE, job="atlas v2 genotype probes for the half split", commit=os.environ.get("REPO_COMMIT"))
from stage_1_idat_calibration import calibrate_idat_to_beta
R = "/home/ubuntu/data/atlas_sources"
DIRS = {"Moss2018": [f"{R}/moss2018/idats"], "Salas2018": [f"{R}/blood/GSE110554/idats"], "Salas2022": [f"{R}/blood/GSE167998/idats"], "GSE63409": [f"{R}/hsc_gse63409/idats"]}
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
S = pd.read_csv("roster_samples.csv"); S = S[S.qc & S.source.isin(DIRS) & (S.platform == "array")].drop_duplicates("sample")
def find(src, gsm):
    for d in DIRS[src]:
        g = sorted(glob.glob(f"{d}/{gsm}_*Grn.idat*")) or sorted(glob.glob(f"{d}/*{gsm}*Grn.idat*"))
        if g:
            r = g[0].replace("_Grn.idat", "_Red.idat")
            if os.path.exists(r): return g[0], r
    return None, None
def one(row):
    g, r = find(row.source, row["sample"])
    if g is None: return dict(sample=row["sample"], source=row.source, cell=row.cell, status="IDAT not found")
    try:
        beta, meta = calibrate_idat_to_beta(g, r, verbose=False, return_snp=True)
        rs = meta["_rs_beta"]; os.makedirs("rs", exist_ok=True); rs.to_frame(row["sample"]).to_parquet(f"rs/{row['sample']}.parquet")
        return dict(sample=row["sample"], source=row.source, cell=row.cell, status="ok", n_rs=int(rs.notna().sum()), idat=os.path.basename(g))
    except Exception as e:
        return dict(sample=row["sample"], source=row.source, cell=row.cell, status=f"error: {type(e).__name__}: {str(e)[:120]}")
log("arrays", len(S), S.source.value_counts().to_dict())
with ProcessPoolExecutor(24) as ex: rows = list(ex.map(one, [r for _, r in S.iterrows()]))
T = pd.DataFrame(rows); T.to_csv("v2_rs_arrays.csv", index=False); log("status", T.status.str.split(":").str[0].value_counts().to_dict())
fs = sorted(glob.glob("rs/*.parquet"))
if fs: pd.concat([pd.read_parquet(f) for f in fs], axis=1).T.to_parquet("v2_rs.parquet")
for f in ("v2_rs_arrays.csv", "v2_rs.parquet"):
    if os.path.exists(f): s3.upload_file(f, BUCKET, OUTP + f) if (s3 := boto3.client("s3")) else None
open("DONE", "w").write("ok\n")
