"""READY_TO_TEST input check (2026-10-11): for every GEO series under downloads/C_tissue_tests, B_cfDNA_plasma, A_blood_immune in S3, the format
of its RAW tar from the archive member NAMES only (no sample content is read): IDAT (chain Stage 1 can read it) or other. Writes s3_format_check.csv.
Usage: python3 s3_format_check.py"""
import os, re, boto3, pandas as pd
s3 = boto3.client("s3"); B = "methylphys-data-945451304272-us-west-2-an"; HERE = os.path.dirname(os.path.abspath(__file__))
def members(key, n=6):
    off = 0; out = []
    for _ in range(n):
        h = s3.get_object(Bucket=B, Key=key, Range=f"bytes={off}-{off + 511}")["Body"].read(); nm = h[:100].split(b"\0")[0].decode(errors="replace")
        if not nm: break
        sz = int(h[124:136].split(b"\0")[0].strip() or b"0", 8); out.append(nm); off += 512 + ((sz + 511) // 512) * 512
    return out
rows = []
for pre in ("downloads/C_tissue_tests/", "downloads/B_cfDNA_plasma/", "downloads/A_blood_immune/"):
    ks = [o["Key"] for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=pre) for o in pg.get("Contents", [])]
    for g in sorted({k.split("/")[2] for k in ks}):
        tars = [k for k in ks if k.split("/")[2] == g and re.search(r"RAW\.tar(\.part_00)?$", k)]
        fmt = "no RAW tar"
        if tars: m = members(tars[0]); fmt = "IDAT" if any("idat" in x.lower() for x in m) else "other (" + ", ".join(sorted({x.split(".")[-2] if x.endswith(".gz") else x.split(".")[-1] for x in m})) + ")"
        rows.append(dict(track=pre.split("/")[1], series=g, format=fmt, prefix=pre + g + "/"))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "s3_format_check.csv"), index=False); print(pd.DataFrame(rows).format.value_counts().to_dict())
