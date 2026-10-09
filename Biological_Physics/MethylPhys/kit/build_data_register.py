#!/usr/bin/env python3
"""Regenerate doors/DATA_REGISTER.csv (never edit it by hand; STATUS.md document rules).
Sizes and file counts from the S3 listing of downloads/ (needs boto3 and read access); 'tests_done' from the LOG: every dated LOG heading
whose section names the dataset folder or a GSE accession under it. 'what', 'planned_tests' and 'status' are carried from the previous file
(status changes - e.g. 'candidate for deletion' - are made in STATUS.md section 7 and copied here by this script's --status DATASET=TEXT)."""
import argparse, collections, os, re, sys
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REG = os.path.join(ROOT, "doors", "DATA_REGISTER.csv"); LOG = os.path.join(ROOT, "..", "..", "development", "METHYLPHYS_DEVELOPMENT_LOG.md")
ap = argparse.ArgumentParser(); ap.add_argument("--bucket", default="methylphys-data-945451304272-us-west-2-an"); ap.add_argument("--status", action="append", default=[])
ap.add_argument("--no-s3", action="store_true", help="keep sizes from the previous file (offline)"); a = ap.parse_args()
old = pd.read_csv(REG)
if "earlier_tests" not in old.columns: old["earlier_tests"] = old.get("tests_done")   # notes written by hand before 2026-10-09, kept
keep = old.set_index("prefix")[["track", "dataset", "what", "planned_tests", "status", "earlier_tests"]].to_dict("index")
if a.no_s3:
    size = {r.prefix: (r.GB, r.files) for r in old.itertuples()}; acc = collections.defaultdict(set)
else:
    import boto3; s3 = boto3.client("s3", region_name="us-west-2"); size = collections.defaultdict(lambda: [0, 0]); acc = collections.defaultdict(set)
    for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix="downloads/"):
        for o in pg.get("Contents", []):
            k = o["Key"].split("/")
            if len(k) < 3: continue   # loose files directly under downloads/ are not datasets
            p = "/".join(k[:3]) + "/" if len(k) > 3 else "/".join(k[:2]) + "/"
            size[p][0] += o["Size"] / 1e9; size[p][1] += 1; acc[p] |= set(re.findall(r"GSE\d+", o["Key"]))
log = open(LOG, encoding="utf-8").read(); secs = re.split(r"\n(?=#{2,3} )", log)
rows = []
for p, (gb, n) in size.items():
    k = keep.get(p, {"track": p.split("/")[1], "dataset": p.rstrip("/").split("/")[-1], "what": "", "planned_tests": "", "status": "NEW: add to STATUS.md", "earlier_tests": ""})
    keys = {k["dataset"]} | acc.get(p, set())
    hits = [s.split("\n")[0].lstrip("# ").strip()[:90] for s in secs if any(re.search(rf"(?<![A-Za-z0-9]){re.escape(x)}(?![0-9])", s) for x in keys if len(x) > 3)]
    rows.append(dict(track=k["track"], dataset=k["dataset"], prefix=p, GB=round(float(gb), 2), files=int(n), what=k["what"], planned_tests=k["planned_tests"],
                     tests_done=" | ".join(hits[-6:]), n_log_sections=len(hits), earlier_tests=k.get("earlier_tests"), status=k["status"]))
D = pd.DataFrame(rows).sort_values(["track", "GB"], ascending=[True, False])
for s in a.status:
    ds, txt = s.split("=", 1); D.loc[D.dataset == ds, "status"] = txt
D.to_csv(REG, index=False); print(f"{len(D)} datasets, {D.GB.sum():.0f} GB; no test recorded (LOG or earlier notes): {int(((D.n_log_sections == 0) & D.earlier_tests.isna()).sum())}")
