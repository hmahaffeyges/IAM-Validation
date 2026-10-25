"""DEV-METAA-HEALTHY-01 step 3 version 5, the readings (written and committed 2026-10-24 before any BLUEPRINT file is read). Each donor of
v5_samples.csv with the given use (purified mature neutrophils of healthy donors, whole-genome bisulfite sequencing, BLUEPRINT's public GRCh38
bigWig calls: level and coverage) is fetched and read by the one command (run_sample.py --dev-seq-beta CALL.bw,COV.bw --dev-seq-values 10),
its values saved (--save-betas) and cut to Met-A's neutrophil identity sites.  Usage: python3 v5_read.py OUTDIR USE"""
import os, sys, csv, json, time, subprocess, urllib.request, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
SITES = set(json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]["sites"])
out, use = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); W = os.path.join(out, "_work"); os.makedirs(W, exist_ok=True)
for r in csv.DictReader(open(os.path.join(H, "v5_samples.csv"))):
    if r["use"] != use: continue
    g, t0 = r["donor"], time.time(); fs = []
    for u in (r["call_url"], r["cov_url"]):
        f = os.path.join(W, os.path.basename(u))
        for a in range(5):
            try: urllib.request.urlretrieve(u, f); break
            except Exception as e: print(g, "download retry", type(e).__name__, flush=True); time.sleep(60 * (a + 1))
        fs.append(f)
    sv = os.path.join(W, g + "_values.parquet")
    p = subprocess.run([sys.executable, RS, "--dev-seq-beta", ",".join(fs), "--dev-seq-values", "10", "--specimen", "whole blood", "--id", "v5_" + g, "--save-betas", sv,
                        "--out", os.path.join(W, g + ".html"), "--bundle", os.path.join(out, g + "_bundle.json")], capture_output=True, text=True)
    try:
        v = pd.read_parquet(sv).iloc[:, 0]; v.index = v.index.astype(str); v = v[v.index.isin(SITES)]
        v.rename("value").to_frame().assign(donor=g).to_csv(os.path.join(out, g + "_v5.csv")); print(g, f"exit {p.returncode}", len(v), "identity sites", f"{time.time() - t0:.0f} s", flush=True)
    except Exception as e: print(g, "FAILED", type(e).__name__, p.stdout[-400:], p.stderr[-600:], flush=True)
    for x in os.listdir(W): os.remove(os.path.join(W, x))
