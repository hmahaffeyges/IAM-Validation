"""Public run ledger (2026-10-11). Builds ../RUNS.md from the session logs every box run writes to S3 (results/**/session*.log):
start, end, job hours, vCPUs used, end status, and the purpose and record from RUNS_PURPOSE below. Job hours are the time the job's own log
covers; the machine also runs for setup and between jobs, so job hours are a LOWER bound on billed hours. Actual cost per run comes from AWS
billing (Cost Explorer, by the run-name tag on the machine) once that access is granted; until then the cost column says so.
Upcoming runs are listed with their estimate before they start. Usage: python3 make_runs_ledger.py   (needs read access to the bucket)"""
import os, re, boto3, pandas as pd
B = "methylphys-data-945451304272-us-west-2-an"; HERE = os.path.dirname(os.path.abspath(__file__)); s3 = boto3.client("s3")
RUNS_PURPOSE = {   # prefix -> (run, purpose, record)
 "results/BOXRUN10_EPIQC/cause01": ("Box Run 10, cause test", "IAM-A: why three EMSeq LAB02 libraries read low (three parts of each run)", "doors/DEV_IAMA_CAUSE_01.md"),
 "results/WORKBOX01": ("Work machine 01", "neuron reference (10 Loyfer files read whole) and colon replication normals (152 arrays, Stage 1); small spot machine, unattended", "boxruns/workbox/jobs_01.sh"),
 "results/BOXRUN1": ("Box Run 1", "array jobs A-E: commissioning check and blood-set readings (arrays only)", "boxruns/run1/JOBS.md"),
 "results/BOXRUN3": ("Box Run 3", "first tests on the commissioned Met-A (neutrophils, EPIC v1)", "boxruns/run3/JOBS.md"),
 "results/BOXRUN2/session1": ("Box Run 2, setup", "build the pinned sequencing pipeline and genome index", "boxruns/run2/JOBS.md"),
 "results/BOXRUN2/session2": ("Box Run 2, s2", "IAM-A on a second lab's healthy neutrophils (first attempt; stopped on a tool-version check)", "boxruns/run2/JOBS.md"),
 "results/BOXRUN2/session3": ("Box Run 2, s3", "IAM-A on a second lab's healthy neutrophils, and CD4 cells", "boxruns/run2/JOBS.md"),
 "results/BOXRUN2/session4": ("Box Run 2, s4", "IAM-A on whole blood, two library kits", "doors/DEV_IAMA_CONVERSION_01.md"),
 "results/BOXRUN2/session5": ("Box Run 2, s5", "decitabine dose series (IAM-A side)", "doors/DEV_LINK_IAMA_METAA_02_OUTCOME.md"),
 "results/BOXRUN6_SAM": ("Box Run 6", "mouse SAM knockout: IAM-A's first physics lever", "doors/DEV_SAM_LEVER_01_OUTCOME.md"),
 "results/BOXRUN7_XSP_REALIGN": ("Box Run 7", "cross-species reference check", "boxruns/xspecies/session7_realign.sh"),
 "results/BOXRUN8_FINGERPRINT": ("Box Run 8", "first cancer fingerprint: LNCaP against PrEC", "doors/DEV_FINGERPRINT_01_OUTCOME.md"),
 "results/BOXRUN9_FP2": ("Box Run 9", "cancer fingerprints: liver, lung, breast lines and normals", "doors/DEV_FINGERPRINT_02_OUTCOME.md"),
 "results/BOXRUN10_EPIQC": ("Box Run 10", "IAM-A commissioning on identical DNA (EpiQC): format check, Stage A", "doors/DEV_IAMA_COMMISSION_01.md"),
}
UPCOMING = [   # (run tag used by the launcher, run, purpose, machine, estimate); ledger_guard.require(tag) refuses any launch not listed here
    ("BOXRUN10_stageB", "Box Run 10, Stage B (Stage A met C1)", "IAM-A commissioning on identical DNA: the other 71 libraries, 12.2M read pairs each", "spot, 128 cores: m7a.32xlarge until AWS reclaimed it at 25 of 71 libraries (2026-10-12 02:02 Pacific); then c6a.32xlarge spot, reclaimed three times (25, 50 and 50 of 71); the last 21 on an on-demand c6a.32xlarge (author decision, about $23)", "about $50 in total"),
    ("BOXRUN10_cause01", "Box Run 10, cause test (DEV-IAMA-CAUSE-01)", "why three EMSeq LAB02 libraries read low: 5 runs read from 3 parts each, 12.2M read pairs per part", "spot c6a.32xlarge, 128 cores; shuts down at the end", "about 3 hours, $5-10"),
    ("ATLAS_V21_FIT", "Atlas v2.1 fit (one-block check, then 700 blocks)", "the v2.1 reference atlas: block 0 measured with 100 and 20 saved draws on the box, then the full Bayesian fit; two m7a.48xlarge spot machines were reclaimed by AWS on 2026-10-14 before any block finished (about 3 hours together, about $12; the first was launched before this row existed)", "on-demand m7a.32xlarge, 128 cores, 512 GB: the box resized (author decision 2026-10-14; spot capacity scores 1-3 of 10); finished blocks copied to S3 as they land", "about 20-23 hours, $150-170"),
    ("WORKBOX01", "Work machine 01", "neuron reference (10 Loyfer files) and colon replication: 152 normal arrays calibrated, then synthetic tests", "spot r7a.2xlarge, 8 cores, 64 GB", "about 5 hours, $1-2"),
]
rows = []
for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix="results/"):
    for o in pg.get("Contents", []):
        k = o["Key"]
        if not (k.endswith(".log") and re.match(r"(session|job_)", k.split("/")[-1])): continue
        pre = next((p for p in sorted(RUNS_PURPOSE, key=len, reverse=True) if k.startswith(p + "/") or k.startswith(p)), None)
        if pre is None: continue
        t = s3.get_object(Bucket=B, Key=k)["Body"].read().decode(errors="replace").split("\n")
        ts = [l[:20] for l in t if re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", l)]
        if not ts: continue
        vc = next((m.group(1) for l in t for m in [re.search(r"(?:vcpus|threads)=(\d+)", l)] if m), "")
        end = next((l.split(" END ", 1)[1].strip() for l in t if " END " in l), "not recorded")
        rows.append(dict(prefix=pre, log=k.split("/", 1)[1], start=ts[0], end=ts[-1], vcpu=vc, status=end))
for pre in RUNS_PURPOSE:                                   # runs that wrote no log to S3 are listed, not dropped
    if not any(r["prefix"] == pre for r in rows) and s3.list_objects_v2(Bucket=B, Prefix=pre + "/", MaxKeys=1)["KeyCount"]:
        rows.append(dict(prefix=pre, log="(no log in S3)", start="", end="", vcpu="", status="not recorded"))
# box jobs registered in CANON/box_jobs.json (2026-10-14): their logs carry HH:MM:SS stamps and the register carries the date
import json as _json, datetime as _dtj
for _j in _json.load(open(os.path.join(HERE, "..", "..", "..", "CANON", "box_jobs.json")))["jobs"]:
    _pre = "JOB:" + _j["job_id"][:8]; RUNS_PURPOSE[_pre] = (f"{_j['name']} (job {_j['job_id'][:8]})", _j["purpose"], _j["record"])
    try: _t = s3.get_object(Bucket=B, Key=_j["log"])["Body"].read().decode(errors="replace").split("\n")
    except Exception: rows.append(dict(prefix=_pre, log=_j["log"].split("/", 1)[1] + " (not in S3 yet)", start="", end="", vcpu="", status=_j["state"])); continue
    _ts = [l[:8] for l in _t if re.match(r"\d\d:\d\d:\d\d ", l)]
    if not _ts: rows.append(dict(prefix=_pre, log=_j["log"].split("/", 1)[1], start="", end="", vcpu="", status=_j["state"])); continue
    _d0 = _dtj.datetime.fromisoformat(_j["date"] + "T" + _ts[0]); _d1 = _dtj.datetime.fromisoformat(_j["date"] + "T" + _ts[-1])
    if _d1 < _d0: _d1 += _dtj.timedelta(days=1)
    rows.append(dict(prefix=_pre, log=_j["log"].split("/", 1)[1], start=_d0.strftime("%Y-%m-%dT%H:%M:%SZ"), end=_d1.strftime("%Y-%m-%dT%H:%M:%SZ"),
                     vcpu=_j["machine"].split("(")[-1].split(" ")[0], status=_j["state"]))
R = pd.DataFrame(rows).sort_values("start"); R["hours"] = (pd.to_datetime(R.end.replace("", None)) - pd.to_datetime(R.start.replace("", None))).dt.total_seconds() / 3600
L = ["# Run ledger", "", "Every rented-computer (AWS) run, generated by `boxruns/make_runs_ledger.py` from the logs each run writes. **Job hours are a lower bound**:",
     "the machine is also billed for setup and for time between jobs. Actual cost per run is filled from AWS billing once access is granted.",
     "Work done on the laptop is free and is not listed.", "", "## Completed", "",
     "| run | purpose | log | start (UTC) | job hours | vCPUs | end | actual cost | record |", "|---|---|---|---|---|---|---|---|---|"]
for r in R.itertuples():
    n, p, rec = RUNS_PURPOSE[r.prefix]
    L.append(f"| {n} | {p} | `{r.log}` | {r.start[:16].replace('T', ' ')} | {'' if pd.isna(r.hours) else f'{r.hours:.1f}'} | {r.vcpu} | {r.status} | pending billing access | [{rec}]({rec}) |")
L += ["", f"Total job hours: {R.hours.sum():.1f}.", "", "## Upcoming", "", "| run tag | run | purpose | machine | estimate |", "|---|---|---|---|---|"]
L += [f"| `{t}` | {a} | {b} | {c} | {d} |" for t, a, b, c, d in UPCOMING]
# Actual charges from AWS billing (Cost Explorer), by month and by type, and by day; per-run cost from the "run" tag once active.
ce = boto3.client("ce", region_name="us-east-1"); import datetime as _dt
_end = (_dt.date.today() + _dt.timedelta(days=1)).isoformat()
_m = ce.get_cost_and_usage(TimePeriod={"Start": "2026-09-01", "End": _end}, Granularity="MONTHLY", Metrics=["UnblendedCost"],
                           GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}])["ResultsByTime"]
L += ["", "## Actual AWS charges (from AWS billing)", "", "| month | total | largest items |", "|---|---|---|"]
for x in _m:
    g = sorted(((y["Keys"][0], float(y["Metrics"]["UnblendedCost"]["Amount"])) for y in x["Groups"]), key=lambda t: -t[1])
    L.append(f"| {x['TimePeriod']['Start'][:7]} | ${sum(v for _, v in g):,.2f} | " + "; ".join(f"{k.replace('USW2-', '')} ${v:,.2f}" for k, v in g[:4]) + " |")
_d = ce.get_cost_and_usage(TimePeriod={"Start": "2026-09-25", "End": _end}, Granularity="DAILY", Metrics=["UnblendedCost"])["ResultsByTime"]
L += ["", "Daily: " + ", ".join(f"{x['TimePeriod']['Start'][5:]} ${float(x['Total']['UnblendedCost']['Amount']):,.0f}" for x in _d if float(x['Total']['UnblendedCost']['Amount']) >= 1),
      "", "BoxUsage = on-demand machine hours; SpotUsage = spot machine hours. Runs before 2026-10-12 carry no run tag, so their cost is by day and type only."]
open(os.path.join(HERE, "..", "RUNS.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L[-12:]))
