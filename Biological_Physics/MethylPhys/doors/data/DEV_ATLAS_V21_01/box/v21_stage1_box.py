#!/usr/bin/env python3
"""Atlas v2.1 Stage 1 on the box (readiness L4). Modeled on atlas v2's own scripts/05_salas_blood_stage1.py: raw IDATs -> the chain's
Stage 1 -> one parquet shard per array + one meta json, resumable, process pool. Added for v2.1, in this order:

 0. PINS: the chain's guarded Stage 1 (methylprep 1.7.1, Illumina manifests by sha256) refuses to run otherwise.
 1. REPRODUCTION GATE: recalibrate 3 of atlas v2's own Moss arrays and compare with v2's stored shards. Max |difference| must be 0 at
    every probe, or the job stops before any new array: v2.1's arrays and v2's are then provably one Stage 1.
 2. BACKUP: atlas v2's calibrated shards and meta exist only on this disk; copy them to S3 (atlas_v2/stage1_shards/) first.
 3. FETCH: each series once - the RAW.tar already in S3 for GSE103541, GSE118144, GSE122244; GEO's series RAW.tar otherwise (per-file
    GEO downloads are throttled from this box); E-MTAB-4957 per file from ArrayExpress. Only the listed arrays are extracted, each
    .idat.gz passes gzip -t, each IDAT is checksummed (sha256). The tar is deleted after extraction.
 4. CALIBRATE every IN and CONDITIONAL array of atlas_v21_build_samples.csv (sealed, checksum checked here), one serially first so
    methylprep's manifest is not raced, then a pool. Shard: beta indexed by probe name (a value-only table is refused).
 5. COPY AS PRODUCED: each shard and meta to S3 results/ATLAS_V21/stage1/ when written (spot-safe), and the manifest at the end.
 Writes stage1_v21_manifest.csv: sample, series, cell, state, base, array_type, pipeline, n_cpgs, call_rate, idat sha256s, shard sha256,
 status. Usage: ~/env/bin/python v21_stage1_box.py   (inputs in the working directory)
"""
import os, re, sys, json, gzip, glob, hashlib, shutil, subprocess, tarfile, time, multiprocessing as mp
import numpy as np, pandas as pd, boto3
HERE = os.getcwd(); sys.path.insert(0, HERE)
BUCKET = "methylphys-data-945451304272-us-west-2-an"; PFX = "results/ATLAS_V21/stage1/"
ROOT = "/home/ubuntu/data/atlas_sources/v21"; IDAT = f"{ROOT}/idats"; SH = f"{ROOT}/shards"; MD = f"{ROOT}/meta"
for d in (IDAT, SH, MD): os.makedirs(d, exist_ok=True)
s3 = boto3.client("s3", region_name="us-west-2")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
SEALED_BUILD_SHA = open("sealed_build_sha256.txt").read().strip()
assert sha("atlas_v21_build_samples.csv") == SEALED_BUILD_SHA, "build list differs from the sealed one"
S3_TARS = json.load(open("s3_tars.json"))
from stage_1_idat_calibration import calibrate_idat_to_beta, _check_pins
_check_pins(); import methylprep; log("pins ok: methylprep", methylprep.__version__, "numpy", np.__version__, "pandas", pd.__version__)

def shard_write(beta, base):
    if hasattr(beta, "columns"): beta = beta.iloc[:, 0]
    assert beta.index.astype(str).str.match(r"^(cg|ch|rs)").mean() > 0.99, "shard has no probe names"
    beta.to_frame(base).to_parquet(f"{SH}/{base}.parquet")

# ---- 1. reproduction gate on v2's own arrays
V2 = "/home/ubuntu/data/atlas_sources/moss2018"
v2s = sorted(glob.glob(f"{V2}/shards/*.parquet"))[:3]
assert len(v2s) == 3, "atlas v2 Moss shards not found on this disk; cannot run the reproduction gate"
gate = []
for p in v2s:
    b = os.path.basename(p)[:-8]; g, r = f"{V2}/idats/{b}_Grn.idat", f"{V2}/idats/{b}_Red.idat"
    for f in (g, r):
        if not os.path.exists(f) and os.path.exists(f + ".gz"):
            with gzip.open(f + ".gz") as a, open(f, "wb") as o: o.write(a.read())
    new, _ = calibrate_idat_to_beta(g, r, verbose=False); old = pd.read_parquet(p).iloc[:, 0]
    new = new.iloc[:, 0] if hasattr(new, "columns") else new
    common = old.index.intersection(new.index)
    d = float(np.nanmax(np.abs(old.loc[common].values - new.loc[common].values)))
    gate.append(dict(array=b, probes_v2=len(old), probes_now=len(new), common=len(common), max_abs_diff=d))
    log("gate", b, len(old), len(new), d)
json.dump(gate, open("reproduction_gate.json", "w"), indent=1)
if any(x["max_abs_diff"] != 0 or x["probes_v2"] != x["probes_now"] for x in gate):
    log("REPRODUCTION GATE FAILED: Stage 1 today does not reproduce atlas v2's shards exactly; stopping before any new array"); sys.exit(2)
log("REPRODUCTION GATE PASSED")

# ---- 2. back up v2's shards and meta (exist only on this disk)
n = 0
for src in sorted(glob.glob("/home/ubuntu/data/atlas_sources/*/shards/*.parquet") + glob.glob("/home/ubuntu/data/atlas_sources/*/*/shards/*.parquet")):
    if "/v21/" in src: continue
    key = "atlas_v2/stage1_shards/" + os.path.relpath(src, "/home/ubuntu/data/atlas_sources")
    try: s3.head_object(Bucket=BUCKET, Key=key)
    except Exception: s3.upload_file(src, BUCKET, key); n += 1
log("v2 shards backed up:", n, "new uploads")

# ---- 3. fetch
B = pd.read_csv("atlas_v21_build_samples.csv"); B = B[B.state != "OUT"].copy()
def fetch_series(ser, rows):
    need = {f for fs in rows.idat_files for f in str(fs).split(";") if f}
    have = {f for f in need if os.path.exists(f"{IDAT}/{f}")}
    if have == need: return
    if ser.startswith("E-MTAB"):
        for f in sorted(need - have):
            subprocess.run(["curl", "-s", "-f", "-L", "--retry", "5", "--retry-all-errors", "-o", f"{IDAT}/{f}", f"https://www.ebi.ac.uk/biostudies/files/{ser}/{f}"], check=True)
        return
    tar = f"{ROOT}/{ser}_RAW.tar"
    if ser in S3_TARS: s3.download_file(BUCKET, S3_TARS[ser], tar)
    else:
        sub = ser[:-3] + "nnn"
        subprocess.run(["curl", "-s", "-f", "-L", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", tar, f"https://ftp.ncbi.nlm.nih.gov/geo/series/{sub}/{ser}/suppl/{ser}_RAW.tar"], check=True)
    with tarfile.open(tar) as t:
        names = {os.path.basename(m.name): m for m in t.getmembers()}
        miss = sorted(need - set(names)); assert not miss, f"{ser}: {len(miss)} listed IDATs not in the tar, e.g. {miss[:2]}"
        for f in sorted(need - have):
            m = names[f]; m.name = f; t.extract(m, IDAT)
    os.remove(tar)
for ser, rows in B.groupby("series"):
    fetch_series(ser, rows); log("fetched", ser, len(rows), "arrays")
for f in glob.glob(f"{IDAT}/*.idat.gz"):
    subprocess.run(["gzip", "-t", f], check=True)

# ---- 4. calibrate
def pair(r):
    fs = [f for f in str(r.idat_files).split(";") if f]
    g = next(f for f in fs if re.search(r"(?i)_grn\.idat", f)); rd = next(f for f in fs if re.search(r"(?i)_red\.idat", f))
    return f"{IDAT}/{g}", f"{IDAT}/{rd}", re.sub(r"(?i)_grn\.idat(\.gz)?$", "", g)
def unz(p):
    if p.endswith(".gz"):
        o = p[:-3]
        if not os.path.exists(o):
            with gzip.open(p) as a, open(o, "wb") as b: b.write(a.read())
        return o
    return p
def calib(i):
    r = B.loc[i]; g, rd, base = pair(r)
    out = dict(sample=r["sample"], series=r.series, cell=r.cell, state=r.state, base=base, idat_grn_sha256=sha(g), idat_red_sha256=sha(rd))
    sh = f"{SH}/{base}.parquet"
    if os.path.exists(sh) and os.path.exists(f"{MD}/{base}.json"):
        m = json.load(open(f"{MD}/{base}.json")); out.update(m.get("_row", {})); out["status"] = "exists"; return out
    try:
        beta, meta = calibrate_idat_to_beta(unz(g), unz(rd), verbose=False)
        shard_write(beta, base); d = meta.get("detection") or {}
        row = dict(array_type=meta.get("array_type"), pipeline=meta.get("pipeline"), n_cpgs=meta.get("n_cpgs"),
                   call_rate=round(d.get("n_detected", 0) / max(d.get("n_probes", 1), 1), 5), shard_sha256=sha(sh))
        keep = {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool, type(None), dict, list))}; keep["_row"] = row
        json.dump(keep, open(f"{MD}/{base}.json", "w"), default=str)
        for f, k in ((sh, f"{PFX}shards/{base}.parquet"), (f"{MD}/{base}.json", f"{PFX}meta/{base}.json")): s3.upload_file(f, BUCKET, k)
        out.update(row); out["status"] = "ok"
    except Exception as e:
        out["status"] = f"error: {type(e).__name__}: {str(e)[:160]}"
    return out
idx = list(B.index); rows = [calib(idx[0])]; log("first array", rows[0]["status"])
with mp.get_context("fork").Pool(int(os.environ.get("PAR", "28"))) as p:
    for k, o in enumerate(p.imap_unordered(calib, idx[1:]), 1):
        rows.append(o)
        if k % 25 == 0: log("calibrated", k + 1, "of", len(idx))
M = pd.DataFrame(rows).sort_values(["cell", "series", "sample"])
M.to_csv("stage1_v21_manifest.csv", index=False); s3.upload_file("stage1_v21_manifest.csv", BUCKET, PFX + "stage1_v21_manifest.csv")
s3.upload_file("reproduction_gate.json", BUCKET, PFX + "reproduction_gate.json")
log("status:", M.status.str.split(":").str[0].value_counts().to_dict(), "| call rate < 0.93:", int((pd.to_numeric(M.call_rate, errors="coerce") < 0.93).sum()))
open("DONE", "w").write("ok\n")
