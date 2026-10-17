#!/usr/bin/env python3
"""The Commissioning Campaign in numbers (author 2026-10-17: "all of those numbers should def be on the webpage, somewhere up top").
Writes CANON/compute_snapshot.json, which the page builder reads. Run by checkpoint.sh (needs AWS read access), like costs_snapshot.py.
  machine hours, processor-core hours : AWS billing (Cost Explorer usage hours per machine type x that type's vCPUs), from 2026-09-01
  billed                              : AWS billing, every cost
  CpG readings on single molecules    : held_opp of every chain count (CpG calls inside qualifying molecules), committed records and S3 outputs;
                                        a lower bound on all calls read
  CpG readings on arrays              : arrays stored in S3 (one green IDAT each) x 485,577 sites (450K size); a lower bound (EPIC arrays,
                                        arrays read from GEO archives and repeat reads are not counted)
Numbers only grow: a value lower than the committed one keeps the committed one."""
import os, re, json, glob, datetime, collections, boto3
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); OUT = os.path.join(ROOT, "CANON", "compute_snapshot.json")
B = "methylphys-data-945451304272-us-west-2-an"; START = "2026-09-28"
ce = boto3.client("ce", region_name="us-east-1"); ec2 = boto3.client("ec2", region_name="us-west-2"); s3 = boto3.client("s3")
end = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
H = collections.Counter(); cost = 0.0
for x in ce.get_cost_and_usage(TimePeriod={"Start": "2026-09-01", "End": end}, Granularity="MONTHLY", Metrics=["UsageQuantity", "UnblendedCost"],
                               GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}])["ResultsByTime"]:
    for g in x["Groups"]:
        u = g["Keys"][0]; cost += float(g["Metrics"]["UnblendedCost"]["Amount"])
        if re.search(r"(BoxUsage|SpotUsage):", u): H[u.split(":", 1)[1]] += float(g["Metrics"]["UsageQuantity"]["Amount"])
V = {t["InstanceType"]: t["VCpuInfo"]["DefaultVCpus"] for t in ec2.describe_instance_types(InstanceTypes=list(H))["InstanceTypes"]} if H else {}
seq = {}
def take(x, key):
    if isinstance(x, dict) and isinstance(x.get("held_opp"), (int, float)) and x["held_opp"] > 0: seq[key] = max(seq.get(key, 0), int(x["held_opp"]))
for f in glob.glob(os.path.join(ROOT, "Biological_Physics/MethylPhys/doors/data/**/*.json"), recursive=True) + glob.glob(os.path.join(ROOT, "development/**/*.json"), recursive=True):
    if f.endswith(".stamp.json"): continue
    try: o = json.load(open(f))
    except Exception: continue
    for x in (o if isinstance(o, list) else [o]):
        if isinstance(x, dict) and x.get("sample"): take(x, (x["sample"], x.get("reader"), x.get("assembly")))
for pre in ("results/PRED_TEMPERATURE_01/T3/out/", "results/DEV_IAMA_GRAND_01/"):
    for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=pre):
        for ob in pg.get("Contents", []):
            if ob["Key"].endswith(".json") and "stamp" not in ob["Key"] and ob["Size"] < 2e6:
                try: x = json.loads(s3.get_object(Bucket=B, Key=ob["Key"])["Body"].read())
                except Exception: continue
                take(x, (x.get("sample") or ob["Key"], x.get("reader"), "s3") if isinstance(x, dict) else ob["Key"])
arrays = set()
for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=B):
    for ob in pg.get("Contents", []):
        b = os.path.basename(ob["Key"])
        if re.search(r"_Grn\.idat(\.gz)?$", b) and not ob["Key"].startswith("laptop_backup"): arrays.add(re.sub(r"_Grn\.idat(\.gz)?$", "", b))
S = dict(as_of=datetime.date.today().isoformat(), since=START, name="The Commissioning Campaign",
         machine_hours=round(sum(H.values()), 1), core_hours=round(sum(h * V.get(t, 0) for t, h in H.items())), billed_usd=round(cost, 2),
         seq_samples=len(seq), seq_calls=sum(seq.values()), arrays=len(arrays), array_reads=len(arrays) * 485577)
if os.path.exists(OUT):
    P = json.load(open(OUT))
    for k in ("machine_hours", "core_hours", "billed_usd", "seq_samples", "seq_calls", "arrays", "array_reads"): S[k] = max(S[k], P.get(k, 0))
S["cpg_total"] = S["seq_calls"] + S["array_reads"]; S["core_years"] = round(S["core_hours"] / 8766, 1)
json.dump(S, open(OUT, "w"), indent=1); open(OUT, "a").write("\n"); print("compute_snapshot:", json.dumps(S))
