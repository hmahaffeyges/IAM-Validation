"""DEV-METAA-HEALTHY-01 step 3 version 4, the readings (written and committed 2026-10-24 before any file is read). Each sample of
v4_samples.csv (B, CD4 and NK cells, CD8 T cells and monocytes of one laboratory's purified-blood arrays, GSE110554, and of the published
sequencing atlas, GSE186458; no neutrophil or granulocyte; all spent as atlas sources) is fetched from GEO and read by the one command
(run_sample.py: arrays from their IDATs through Stage 1, sequencing from its .beta file with at least 10 calls a site), its per-site values
saved (--save-betas) and cut to Met-A's neutrophil identity sites. Usage: python3 v4_read.py OUTDIR SHARD NSHARDS"""
import os, sys, csv, json, time, subprocess, urllib.request, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
sys.path.insert(0, os.path.join(CH, "MethylPhys_Interface")); import run_series as RSER
SITES = set(json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]["sites"])
out, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True); W = os.path.join(out, "_work"); os.makedirs(W, exist_ok=True)
for i, r in enumerate(csv.DictReader(open(os.path.join(H, "v4_samples.csv")))):
    if i % n != k: continue
    g, t0 = r["gsm"], time.time(); sv = os.path.join(W, g + "_values.parquet")
    try:
        if r["platform"] == "array":
            f = RSER.fetch(g, W); src = ["--grn", f["grn"], "--red", f["red"]]
        else:
            f = os.path.join(W, r["file"])
            for a in range(5):
                try: urllib.request.urlretrieve(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{r['file']}", f); break
                except Exception as e: print(g, "download retry", type(e).__name__, flush=True); time.sleep(60 * (a + 1))
            src = ["--dev-seq-beta", f, "--dev-seq-values", "10"]
        p = subprocess.run([sys.executable, RS] + src + ["--specimen", "whole blood", "--id", "v4_" + g, "--save-betas", sv,
                            "--out", os.path.join(W, g + ".html"), "--bundle", os.path.join(W, g + ".json")], capture_output=True, text=True)
        v = pd.read_parquet(sv).iloc[:, 0]; v.index = v.index.astype(str); v = v[v.index.isin(SITES)]
        v.rename("value").to_frame().assign(gsm=g, platform=r["platform"], cell=r["cell"]).to_csv(os.path.join(out, g + "_v4.csv"))
        print(g, r["platform"], r["cell"], f"exit {p.returncode}", len(v), "identity sites", f"{time.time() - t0:.0f} s", flush=True)
    except Exception as e:
        print(g, "FAILED", type(e).__name__, str(e)[:200], flush=True)
    for x in os.listdir(W): os.remove(os.path.join(W, x))
