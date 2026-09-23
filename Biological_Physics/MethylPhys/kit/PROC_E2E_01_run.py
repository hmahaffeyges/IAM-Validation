#!/usr/bin/env python3
"""PROC-E2E-01: run every array in the test package through run_sample.py, exactly as a reviewer would.

Per-array metadata comes from GEO's own sample characteristics (handoff/test11_meta.json), not from the test
manifest, whose sample descriptions disagree with GEO for the three Uppsala arrays.
"""
import json, os, re, subprocess, sys, time

CHAIN = "iamrepo/Biological_Physics/MethylPhys/chain"
RUN = f"{CHAIN}/MethylPhys_Interface/run_sample.py"
meta = json.load(open("handoff/test11_meta.json"))
ZERO = {"GSE87571": -0.0117, "GSE42861": 0.0084}          # commissioned; tissue cohorts have none
SUBSTRATE = {"GSE87571": "whole blood", "GSE42861": "whole blood",
             "GSE288652": "colon adenoma tissue", "GSE166212": "colorectal tumour tissue"}
rows = {}
for gsm in sorted(meta):
    m = meta[gsm]; gse = m["gse"]
    grn = f"{CHAIN}/TEST_DATA/idats/{gsm}_Grn.idat.gz"
    red = f"{CHAIN}/TEST_DATA/idats/{gsm}_Red.idat.gz"
    if not os.path.exists(grn):
        rows[gsm] = {"status": "IDAT ABSENT"}; continue
    age = m.get("age"); sex = (m.get("sex") or m.get("gender") or "")[:1].upper() or None
    cmd = [sys.executable, RUN, "--grn", grn, "--red", red, "--lab", gse,
           "--specimen", SUBSTRATE.get(gse, "unknown"), "--id", gsm,
           "--out", f"results/test9/{gsm}.html", "--bundle", f"results/test9/{gsm}_bundle.json",
           "--intake-log", f"results/test9/{gsm}_intake.jsonl",
           "--manifest-dir", "results/test9/manifests"]
    if age: cmd += ["--age", str(age)]
    if sex: cmd += ["--sex", sex]
    if gse in ZERO: cmd += ["--lab-zero", str(ZERO[gse])]
    # a fresh custody log per run: re-running the same pair against an existing log is correctly held as
    # RE_TRANSMISSION_DETECTED (SOP 13), which is not what this test is measuring
    for f in (f"results/test9/{gsm}_intake.jsonl",):
        if os.path.exists(f): os.remove(f)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "HOME": os.path.abspath("stage1/mp_home")})
    out = "\n".join(l for l in (r.stdout or "").split("\n")
                    if not re.match(r"^(INFO|Reading IDATs|\s*$)", l) and "it/s]" not in l)
    rows[gsm] = {"exit": r.returncode, "secs": round(time.time() - t0, 1), "age": age, "sex": sex,
                 "gse": gse, "stdout": out[-1800:], "stderr": (r.stderr or "")[-500:]}
    print(f"== {gsm} ({gse}, age {age}, sex {sex}) exit {r.returncode} in {rows[gsm]['secs']}s", flush=True)
    for l in out.split("\n")[-6:]:
        if l.strip(): print("   ", l[:200], flush=True)
    # an array whose cohort publishes no age is expected to quarantine; run the measurement path too
    if r.returncode == 2 and not age:
        cmd2 = [c for c in cmd if c not in ("--intake-log", f"results/test9/{gsm}_intake.jsonl")] + ["--no-intake"]
        r2 = subprocess.run(cmd2, capture_output=True, text=True,
                            env={**os.environ, "HOME": os.path.abspath("stage1/mp_home")})
        rows[gsm]["no_intake_exit"] = r2.returncode
        rows[gsm]["no_intake_stdout"] = "\n".join(
            l for l in (r2.stdout or "").split("\n")
            if not re.match(r"^(INFO|Reading IDATs|\s*$)", l) and "it/s]" not in l)[-1200:]
        print(f"   [--no-intake] exit {r2.returncode}", flush=True)
    json.dump(rows, open("handoff/e2e_runs.json", "w"), indent=1)
print("\nDONE", len(rows), "arrays", flush=True)
