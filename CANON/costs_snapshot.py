"""Costs for the public pages (author 2026-10-12). Writes CANON/costs_snapshot.json:
  aws: this month and last month, by category (machines, machine disks and snapshots, S3 storage, data transfer, other), from Cost Explorer;
  s3: bytes stored in the project bucket now (listing), and objects;
  fixed: costs AWS does not bill, entered here as the author states them, with the date stated (CANON/costs_fixed.json).
Usage: python3 CANON/costs_snapshot.py (needs AWS credentials with ce:GetCostAndUsage)"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "Biological_Physics", "MethylPhys", "boxruns")); from ce_cache import cost_and_usage as _ce   # cached: Cost Explorer bills per request (2026-10-20)
import os, json, datetime, boto3
HERE = os.path.dirname(os.path.abspath(__file__)); B = "methylphys-data-945451304272-us-west-2-an"
def cat(k):
    if "BoxUsage" in k or "SpotUsage" in k: return "machines"
    if "TimedStorage" in k or ("Requests" in k and "Tier" in k) or "Monitoring-Automation" in k: return "S3 storage"
    if "EBS" in k or "Snapshot" in k: return "machine disks and snapshots"
    if "DataTransfer" in k or "Bytes" in k: return "data transfer"
    return "other"
today = datetime.date.today(); first = today.replace(day=1); prev = (first - datetime.timedelta(days=1)).replace(day=1)
ce = boto3.client("ce", region_name="us-east-1")
r = _ce(TimePeriod={"Start": prev.isoformat(), "End": (today + datetime.timedelta(days=1)).isoformat()}, Granularity="MONTHLY",
                          Metrics=["UnblendedCost"], GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}])
aws = {}
for p in r["ResultsByTime"]:
    t = {}
    for g in p["Groups"]:
        c = cat(g["Keys"][0]); t[c] = round(t.get(c, 0) + float(g["Metrics"]["UnblendedCost"]["Amount"]), 2)
    aws[p["TimePeriod"]["Start"][:7]] = dict(sorted(t.items(), key=lambda x: -x[1]))
s3 = boto3.client("s3"); n = 0; b = 0
for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=B):
    for o in pg.get("Contents", []): n += 1; b += o["Size"]
out = {"as_of": today.isoformat(), "aws_by_month": aws, "s3": {"bytes": b, "objects": n},
       "fixed": json.load(open(os.path.join(HERE, "costs_fixed.json"), encoding="utf-8"))}
json.dump(out, open(os.path.join(HERE, "costs_snapshot.json"), "w", encoding="utf-8"), indent=1); print(json.dumps(out)[:600])
