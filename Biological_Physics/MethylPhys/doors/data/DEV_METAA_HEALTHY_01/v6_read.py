"""DEV-METAA-HEALTHY-01 step 3 version 6, the readings (written and committed 2026-10-24 before they exist). Whole-array values, every atlas
site, through the one command (run_sample.py --save-betas): (a) the six commissioned purified neutrophil arrays (GSE110554; spent), from their
IDATs; (b) the three development BLUEPRINT neutrophil donors (spent in version 5), from their GRCh38 bigWig pairs (--dev-seq-beta CALL,COV,
--dev-seq-values 10). Usage: python3 v6_read.py OUTDIR [development|confirmation]"""
import os, sys, csv, time, subprocess, urllib.request
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
sys.path.insert(0, os.path.join(CH, "MethylPhys_Interface")); import run_series as RSER
import pandas as pd
out = sys.argv[1]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "_work"); os.makedirs(W, exist_ok=True)
ARR = pd.read_csv(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3_loo.csv")).ref.tolist()
jobs = [("array", g, None) for g in ARR] + [("sequencing", r["donor"], (r["call_url"], r["cov_url"])) for r in csv.DictReader(open(os.path.join(H, "v5_samples.csv"))) if r["use"] == (sys.argv[2] if len(sys.argv) > 2 else "development")]   # the donors to read (2026-10-24: confirmation added for version 7)
for kind, g, u in jobs:
    t0 = time.time()
    try:
        if kind == "array":
            f = RSER.fetch(g, W); src = ["--grn", f["grn"], "--red", f["red"], "--specimen", "whole blood"]
        else:
            fs = []
            for x in u:
                p_ = os.path.join(W, os.path.basename(x)); urllib.request.urlretrieve(x, p_); fs.append(p_)
            src = ["--dev-seq-beta", ",".join(fs), "--dev-seq-values", "10", "--specimen", "whole blood"]
        sv = os.path.join(out, f"{kind}_{g}_v6.parquet")
        p = subprocess.run([sys.executable, RS] + src + ["--id", "v6_" + g, "--save-betas", sv, "--out", os.path.join(W, g + ".html"), "--bundle", os.path.join(out, f"{kind}_{g}_bundle.json")], capture_output=True, text=True)
        print(kind, g, f"exit {p.returncode}", os.path.exists(sv), f"{time.time() - t0:.0f} s", flush=True)
        if p.returncode: print(p.stdout[-400:], p.stderr[-400:], flush=True)
    except Exception as e: print(kind, g, "FAILED", type(e).__name__, str(e)[:200], flush=True)
    for x in os.listdir(W): os.remove(os.path.join(W, x))
