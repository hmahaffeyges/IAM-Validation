#!/usr/bin/env python3
"""A8C (2026-11-16, author: 'Our rule is to ALWAYS USE THE CHAIN'; 'working smarter not harder'). The chain's own IAM-A reading
(stage_q2_selfref.reading and shape_check) applied to the count records run_sample.py already made for the 11 same-donor atlas libraries
(GSE186458, spent), committed in DEV_CHAIN_Q2_01/percall_out. No download, nothing computed here: the chain's functions only."""
import os, sys, json, glob
CH = "Biological_Physics/MethylPhys/chain"; sys.path.insert(0, CH)
import stage_q2_selfref as SR
G = ["GSM5652296","GSM5652298","GSM5652286","GSM5652289","GSM5652292","GSM5652295","GSM5652297","GSM5652285","GSM5652288","GSM5652291","GSM5652294"]
D = "Biological_Physics/MethylPhys/doors/data/DEV_CHAIN_Q2_01/percall_out"
os.makedirs("xout", exist_ok=True); rows = []
for g in G:
    fs = [f for f in glob.glob(os.path.join(D, "*.json")) if g in os.path.basename(f)]
    if not fs: rows.append({"gsm": g, "missing": "no count record"}); print(f"::warning::{g}: no count record in percall_out"); continue
    rec = json.load(open(fs[0])); rec = rec.get("q2", rec.get("count", rec))
    try:
        r = SR.reading(rec); out = {"gsm": g, "file": os.path.basename(fs[0]), "A": round(r["A"], 4), "held_err": rec["held_err"], "loose_share": rec["loose_share"],
                                    "predicted": round(r["predicted"], 5), "line_version": r["line_version"]}
    except Exception as e:
        out = {"gsm": g, "file": os.path.basename(fs[0]), "error": str(e)[:200], "keys": sorted(rec)[:25]}
    rows.append(out); print("::notice::" + json.dumps(out)[:400])
json.dump(rows, open("xout/a8c_chain_reading.json", "w"), indent=1)
