#!/usr/bin/env python3
"""Fetch and calibrate IAM-Atlas v2.1's held-out public arrays exactly as the atlas job did (doors/data/DEV_ATLAS_V21_01/box/v21_identity_box.py,
step 2): GEO series RAW.tar, per-sample GEO files when a file is missing, ArrayExpress files for E-MTAB series; the chain's Stage 1
(stage_1_idat_calibration.calibrate_idat_to_beta, return_snp=True); one beta file per array, <base>.parquet, in the same form.
Each downloaded IDAT is checked against the SHA-256 the atlas job recorded (identity_v21_arrays.csv), so the input is proved to be the same
file. Usage: python3 heldout_arrays.py SHARD_DIR SERIES [SERIES ...]   (or a comma list of sample names after --samples)"""
import os, sys, re, json, hashlib, subprocess, tarfile
import pandas as pd
MP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Biological_Physics/MethylPhys (reproduce/ moved here 2026-10-19; a counted repo root broke the deconv reproduction)
assert os.path.isdir(os.path.join(MP, "chain")), f"{MP}: not the MethylPhys folder"
sys.path.insert(0, os.path.join(MP, "chain")); D = os.path.join(MP, "doors", "data", "DEV_ATLAS_V21_01")
from stage_1_idat_calibration import calibrate_idat_to_beta, _check_pins
out = sys.argv[1]; arg = sys.argv[2:]; idd = os.path.join(out, "_idats"); os.makedirs(idd, exist_ok=True)
H = pd.read_csv(os.path.join(D, "atlas_v21_heldout_samples.csv")); I = pd.read_csv(os.path.join(D, "identity_v21_arrays.csv")).set_index("sample")
H = H[H["sample"].isin(arg[1].split(","))] if arg[0] == "--samples" else H[H.series.isin(arg)]
assert len(H), "no held-out arrays match"; _check_pins()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
get = lambda url, dst: subprocess.run(["curl", "-s", "-f", "-L", "--retry", "5", "--retry-all-errors", "-o", dst, url]).returncode == 0
def fetch(ser, rows):
    need = {f for fs in rows.idat_files for f in str(fs).split(";") if f}; have = {f for f in need if os.path.exists(f"{idd}/{f}")}
    if have == need: return
    if ser.startswith("E-MTAB"):
        for f in sorted(need - have): get(f"https://www.ebi.ac.uk/biostudies/files/{ser}/{f}", f"{idd}/{f}")
        return
    tar = f"{idd}/{ser}_RAW.tar"
    if get(f"https://ftp.ncbi.nlm.nih.gov/geo/series/{ser[:-3]}nnn/{ser}/suppl/{ser}_RAW.tar", tar):
        with tarfile.open(tar) as t:
            names = {os.path.basename(m.name): m for m in t.getmembers()}
            for f in sorted((need - have) & set(names)): m = names[f]; m.name = f; t.extract(m, idd)
        os.remove(tar)
    for f in sorted(f for f in need if not os.path.exists(f"{idd}/{f}")):
        g = f.split("_")[0]; get(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/{f}", f"{idd}/{f}")
bad, done = [], 0
for ser, rows in H.groupby("series"):
    fetch(ser, rows)
    for r in rows.itertuples():
        fs = [f for f in str(r.idat_files).split(";") if f]
        g = next(f for f in fs if re.search(r"(?i)_grn\.idat", f)); rd = next(f for f in fs if re.search(r"(?i)_red\.idat", f))
        base = re.sub(r"(?i)_grn\.idat(\.gz)?$", "", g); g, rd = f"{idd}/{g}", f"{idd}/{rd}"
        rec = I.loc[r.sample] if r.sample in I.index else None
        if not (os.path.exists(g) and os.path.exists(rd)): bad.append((r.sample, "missing IDAT")); continue
        if rec is not None and isinstance(rec.idat_grn_sha256, str) and (sha(g) != rec.idat_grn_sha256 or sha(rd) != rec.idat_red_sha256):
            bad.append((r.sample, "IDAT differs from the one the atlas read")); continue
        beta, meta = calibrate_idat_to_beta(g, rd, verbose=False, return_snp=True)
        beta = beta if isinstance(beta, pd.Series) else beta.iloc[:, 0]
        beta.to_frame(base).to_parquet(f"{out}/{base}.parquet"); done += 1
        for f in (g, rd): os.remove(f)
print(json.dumps(dict(arrays=len(H), calibrated=done, problems=bad)))
sys.exit(1 if bad else 0)
