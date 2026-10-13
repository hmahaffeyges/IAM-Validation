#!/usr/bin/env python3
"""Atlas v2.1 identity job on the box (readiness L5 inputs, B2, and the same-donor genotype check). Committed before it runs.

 0. publish_code: this script and every input go to S3 (results/ATLAS_V21/identity/code/) before any work, or the job stops.
 1. BUILD arrays (472, IDATs already on this disk from the Stage 1 job): recalibrate with the chain's Stage 1 and return_snp=True. Every
    new beta is compared with the shard the Stage 1 job stored: the maximum difference must be 0 for every array, which proves the new
    option changes nothing. The SNP (rs) probe betas are kept.
 2. HELD-OUT arrays (atlas_v21_heldout_samples.csv, sealed in seal 2): fetched once per series (GEO RAW.tar, per-sample files if the tar is
    missing, ArrayExpress per file), calibrated, shard + meta + rs saved and copied to S3 (results/ATLAS_V21/heldout/). NOT SCORED: no
    reading of any held-out array is computed here.
 3. Per array: chrY probes detected / chrY probes on the chip, chrX probes' median beta (for sex), rs probe betas.
 Writes identity_v21_arrays.csv and identity_v21_rs.parquet (array x rs probe), uploaded to S3. Usage: ~/env/bin/python v21_identity_box.py
"""
import os, re, sys, json, gzip, glob, hashlib, subprocess, tarfile, time, multiprocessing as mp
import numpy as np, pandas as pd, boto3
HERE = os.getcwd(); sys.path.insert(0, HERE)
from publish_code import publish_code
CODE = ["v21_identity_box.py", "publish_code.py", "stage_1_idat_calibration.py", "atlas_v21_build_samples.csv", "atlas_v21_heldout_samples.csv",
        "sealed_split_v21_02.json", "E-MTAB-2145.sdrf.txt", "E-MTAB-4957.sdrf.txt"]
publish_code("results/ATLAS_V21/identity/code/", CODE, job="atlas v2.1 identity", commit=os.environ.get("REPO_COMMIT"))
BUCKET = "methylphys-data-945451304272-us-west-2-an"; HPFX = "results/ATLAS_V21/heldout/"; IPFX = "results/ATLAS_V21/identity/"
ROOT = "/home/ubuntu/data/atlas_sources/v21"; BID = f"{ROOT}/idats"; BSH = f"{ROOT}/shards"
HO = f"{ROOT}/heldout"; HID, HSH, HMD, RS = f"{HO}/idats", f"{HO}/shards", f"{HO}/meta", f"{ROOT}/rs"
for d in (HID, HSH, HMD, RS): os.makedirs(d, exist_ok=True)
s3 = boto3.client("s3", region_name="us-west-2"); sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
S = json.load(open("sealed_split_v21_02.json"))["sha256"]
for f in ("atlas_v21_build_samples.csv", "atlas_v21_heldout_samples.csv"):
    k = [x for x in S if x.endswith(f)][0]; assert sha(f) == S[k], f"{f} differs from seal 2"
from stage_1_idat_calibration import calibrate_idat_to_beta, _check_pins
_check_pins(); log("pins ok")
# chromosome of every probe, from the Illumina manifests Stage 1 itself uses (pinned by sha256)
def chrom(f):
    p = os.path.expanduser(f"~/.methylprep_manifest_files/{f}")
    hdr = next(i for i, l in enumerate(gzip.open(p, "rt")) if l.startswith("IlmnID"))
    M = pd.read_csv(p, skiprows=hdr, usecols=["IlmnID", "CHR"], dtype=str, low_memory=False).dropna()
    return M.set_index("IlmnID").CHR.str.replace("chr", "", regex=False)
CHR = {"450k": chrom("HumanMethylation450k_15017482_v3.csv.gz"), "epic": chrom("MethylationEPIC_v-1-0_B4.CoreColumns.csv.gz")}
def sexinfo(beta, at):
    c = CHR["epic" if "epic" in str(at).lower() else "450k"]; cg = c[c.index.str.startswith("cg")]
    y = cg.index[cg == "Y"]; x = cg.index[cg == "X"]
    return dict(chrY_detected=int(beta.index.isin(y).sum()), chrY_on_chip=len(y), chrX_median_beta=float(beta.reindex(x).median()))
def pair(fs, base_dir):
    fs = [f for f in str(fs).split(";") if f]
    g = next(f for f in fs if re.search(r"(?i)_grn\.idat", f)); r = next(f for f in fs if re.search(r"(?i)_red\.idat", f))
    return f"{base_dir}/{g}", f"{base_dir}/{r}", re.sub(r"(?i)_grn\.idat(\.gz)?$", "", g)
def unz(p):
    if p.endswith(".gz"):
        o = p[:-3]
        if not os.path.exists(o):
            with gzip.open(p) as a, open(o, "wb") as b: b.write(a.read())
        return o
    return p
def as_series(b): return b.iloc[:, 0] if hasattr(b, "columns") else b

def build_one(i):
    r = BUILD.loc[i]; g, rd, base = pair(r.idat_files, BID); out = dict(sample=r["sample"], set="build", series=r.series, cell=r.cell, state=r.state, base=base)
    try:
        beta, meta = calibrate_idat_to_beta(unz(g), unz(rd), verbose=False, return_snp=True); beta = as_series(beta)
        old = pd.read_parquet(f"{BSH}/{base}.parquet").iloc[:, 0]
        same_idx = old.index.equals(beta.index)
        d = float(np.nanmax(np.abs(old.values - beta.reindex(old.index).values))) if len(old) else float("nan")
        rs = meta["_rs_beta"]; rs.to_frame(base).to_parquet(f"{RS}/{base}.parquet")
        out.update(sexinfo(beta, meta.get("array_type")), array_type=meta.get("array_type"), n_rs=int(rs.notna().sum()),
                   same_probes=bool(same_idx), max_abs_diff_vs_stage1=d, status="ok")
    except Exception as e: out["status"] = f"error: {type(e).__name__}: {str(e)[:160]}"
    return out
def fetch_series(ser, rows):
    need = {f for fs in rows.idat_files for f in str(fs).split(";") if f}
    have = {f for f in need if os.path.exists(f"{HID}/{f}")}
    if have == need: return "have"
    get = lambda url, dst: subprocess.run(["curl", "-s", "-f", "-L", "--retry", "5", "--retry-all-errors", "-o", dst, url]).returncode == 0
    if ser.startswith("E-MTAB"):
        ok = [get(f"https://www.ebi.ac.uk/biostudies/files/{ser}/{f}", f"{HID}/{f}") for f in sorted(need - have)]; return f"files {sum(ok)}/{len(ok)}"
    tar = f"{HO}/{ser}_RAW.tar"
    if get(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{ser[:-3]}nnn/{ser}/suppl/{ser}_RAW.tar", tar):
        with tarfile.open(tar) as t:
            names = {os.path.basename(m.name): m for m in t.getmembers()}
            for f in sorted((need - have) & set(names)):
                m = names[f]; m.name = f; t.extract(m, HID)
        os.remove(tar)
    left = sorted(f for f in need if not os.path.exists(f"{HID}/{f}"))
    for f in left:   # per-sample fallback: the GSM is the file name's prefix
        gsm = f.split("_")[0]
        get(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{f}", f"{HID}/{f}")
    return f"missing after fetch {sum(not os.path.exists(f'{HID}/{f}') for f in need)}"
def held_one(i):
    r = HELD.loc[i]; out = dict(sample=r["sample"], set="heldout", series=r.series, cell=r.cell_type, state="held-out")
    try:
        g, rd, base = pair(r.idat_files, HID); out["base"] = base
        if not (os.path.exists(g) and os.path.exists(rd)): out["status"] = "missing IDAT"; return out
        beta, meta = calibrate_idat_to_beta(unz(g), unz(rd), verbose=False, return_snp=True); beta = as_series(beta)
        assert beta.index.astype(str).str.startswith("cg").mean() > 0.99
        beta.to_frame(base).to_parquet(f"{HSH}/{base}.parquet"); rs = meta["_rs_beta"]; rs.to_frame(base).to_parquet(f"{RS}/{base}.parquet")
        keep = {k: v for k, v in meta.items() if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None), dict, list))}
        json.dump(keep, open(f"{HMD}/{base}.json", "w"), default=str)
        for f, k in ((f"{HSH}/{base}.parquet", f"{HPFX}shards/{base}.parquet"), (f"{HMD}/{base}.json", f"{HPFX}meta/{base}.json")): s3.upload_file(f, BUCKET, k)
        d = meta.get("detection") or {}
        out.update(sexinfo(beta, meta.get("array_type")), array_type=meta.get("array_type"), n_rs=int(rs.notna().sum()),
                   call_rate=round(d.get("n_detected", 0) / max(d.get("n_probes", 1), 1), 5), idat_grn_sha256=sha(g), idat_red_sha256=sha(rd),
                   shard_sha256=sha(f"{HSH}/{base}.parquet"), status="ok")
    except Exception as e: out["status"] = f"error: {type(e).__name__}: {str(e)[:160]}"
    return out

BUILD = pd.read_csv("atlas_v21_build_samples.csv"); BUILD = BUILD[BUILD.state != "OUT"].copy()
HELD = pd.read_csv("atlas_v21_heldout_samples.csv"); HELD = HELD[HELD.idat_files.notna() & (HELD.idat_files != "")].copy()
PAR = int(os.environ.get("PAR", "28"))
rows = [build_one(BUILD.index[0])]; log("first build array", rows[0]["status"], rows[0].get("max_abs_diff_vs_stage1"))
with mp.get_context("fork").Pool(PAR) as p:
    for k, o in enumerate(p.imap_unordered(build_one, BUILD.index[1:]), 2):
        rows.append(o)
        if k % 50 == 0: log("build", k, "of", len(BUILD))
B = pd.DataFrame(rows); bad = B[(B.status != "ok") | (B.max_abs_diff_vs_stage1 != 0) | (~B.same_probes.astype(bool))]
log("build reproduction:", len(B) - len(bad), "of", len(B), "identical to Stage 1")
for ser, g in HELD.groupby("series"): log("fetch", ser, len(g), fetch_series(ser, g))
with mp.get_context("fork").Pool(PAR) as p:
    for k, o in enumerate(p.imap_unordered(held_one, HELD.index), 1):
        rows.append(o)
        if k % 100 == 0: log("held-out", k, "of", len(HELD))
A = pd.DataFrame(rows); A.to_csv("identity_v21_arrays.csv", index=False)
R = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{RS}/*.parquet"))], axis=1).T; R.index.name = "base"; R.to_parquet("identity_v21_rs.parquet")
for f in ("identity_v21_arrays.csv", "identity_v21_rs.parquet"): s3.upload_file(f, BUCKET, IPFX + f)
log("status:", A.groupby("set").status.apply(lambda s: s.str.split(":").str[0].value_counts().to_dict()).to_dict())
open("DONE", "w").write("ok\n")
