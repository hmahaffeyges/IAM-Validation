#!/usr/bin/env python3
"""Computing and cost, day by day (author 2026-10-19: "add that requirement to the same code that demands a daily log after midnight ... the
numbers for the webpage section could auto update from there"). For every day log in CANON/daylogs/ (Pacific dates), writes its "compute"
block from AWS billing (Cost Explorer, DAILY): that day's bill by category, machine hours and processor-core hours (usage hours per machine
type x its vCPUs), and the running totals since the campaign began (2026-09-28). AWS bills about a day behind, so every run rewrites the last
three days' blocks; a block is marked final once the day is more than two days old. daylog.py refuses a closed day log without its block,
and the page's campaign numbers read the latest one. Cost Explorer dates are UTC days. Usage: python3 CANON/day_compute.py"""
import sys as _s, os as _o; _s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "Biological_Physics", "MethylPhys", "boxruns")); from ce_cache import cost_and_usage as _ce   # cached: Cost Explorer bills per request (2026-10-20)
import os, re, json, glob, datetime, collections, boto3
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); DL = os.path.join(ROOT, "CANON", "daylogs"); START = "2026-09-28"
def cat(k):
    if "BoxUsage" in k or "SpotUsage" in k: return "machines"
    if "TimedStorage" in k or ("Requests" in k and "Tier" in k) or "Monitoring-Automation" in k: return "S3 storage"
    if "EBS" in k or "Snapshot" in k: return "machine disks and snapshots"
    if "DataTransfer" in k or "Bytes" in k: return "data transfer"
    return "other"
def days():
    ce = boto3.client("ce", region_name="us-east-1"); end = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    D = {}; tok = None
    while True:
        kw = dict(TimePeriod={"Start": START, "End": end}, Granularity="DAILY", Metrics=["UsageQuantity", "UnblendedCost"],
                  GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}])
        if tok: kw["NextPageToken"] = tok
        r = _ce(**kw)
        for p in r["ResultsByTime"]:
            d = D.setdefault(p["TimePeriod"]["Start"], dict(cost=collections.Counter(), hours=collections.Counter()))
            for g in p["Groups"]:
                u = g["Keys"][0]; d["cost"][cat(u)] += float(g["Metrics"]["UnblendedCost"]["Amount"])
                if re.search(r"(BoxUsage|SpotUsage):", u): d["hours"][u.split(":", 1)[1]] += float(g["Metrics"]["UsageQuantity"]["Amount"])
        tok = r.get("NextPageToken")
        if not tok: break
    types = sorted({t for d in D.values() for t in d["hours"]})
    V = {}
    for i in range(0, len(types), 100):
        V.update({t["InstanceType"]: t["VCpuInfo"]["DefaultVCpus"] for t in boto3.client("ec2", region_name="us-west-2").describe_instance_types(InstanceTypes=types[i:i + 100])["InstanceTypes"]})
    out, run = {}, dict(billed_usd=0.0, machine_hours=0.0, core_hours=0.0)
    for day in sorted(D):
        d = D[day]; mh = sum(d["hours"].values()); chh = sum(h * V.get(t, 0) for t, h in d["hours"].items()); usd = sum(d["cost"].values())
        run = dict(billed_usd=run["billed_usd"] + usd, machine_hours=run["machine_hours"] + mh, core_hours=run["core_hours"] + chh)
        out[day] = dict(billed_usd=round(usd, 2), by_category={k: round(v, 2) for k, v in sorted(d["cost"].items(), key=lambda x: -x[1]) if round(v, 2)},
                        machine_hours=round(mh, 1), core_hours=round(chh), since=START,
                        total_billed_usd=round(run["billed_usd"], 2), total_machine_hours=round(run["machine_hours"], 1), total_core_hours=round(run["core_hours"]))
    return out
if __name__ == "__main__":
    C = days(); today = datetime.date.today(); n = 0
    for f in sorted(glob.glob(os.path.join(DL, "????-??-??.json"))):
        day = os.path.basename(f)[:10]
        if day not in C: continue
        d = json.load(open(f, encoding="utf-8")); old = d.get("compute")
        if old and old.get("final"): continue
        b = dict(C[day]); b["source"] = "AWS billing (Cost Explorer, daily, UTC day)"; b["final"] = (today - datetime.date.fromisoformat(day)).days > 2
        b["read"] = today.isoformat()
        if old != b: d["compute"] = b; json.dump(d, open(f, "w", encoding="utf-8"), indent=1, ensure_ascii=False); open(f, "a").write("\n"); n += 1
    print(f"day_compute: {n} day log(s) updated; {len(C)} billed days since {START}")
