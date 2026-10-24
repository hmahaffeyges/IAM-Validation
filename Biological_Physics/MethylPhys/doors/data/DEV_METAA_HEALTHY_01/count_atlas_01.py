"""DEV-METAA-HEALTHY-01 layer 1: the loss-side and gain-side held errors of every healthy cell type the published sequencing atlas measured
(written and committed before any file is counted). For each sample in layer1_samples.csv (153 .pat files of GSE186458, 56 cell types;
spent): the file is streamed from GEO, read by run_sample.py --dev-count-only (Stage Q2's count, recorded even where IAM-A's intake has no
healthy shape reference for the cell type; flagged), its bundle kept and the file deleted. Usage: python3 count_atlas_01.py OUTDIR SHARD NSHARDS"""
import os, sys, csv, json, time, subprocess, urllib.request
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)
SL = sys.argv[4] if len(sys.argv) > 4 else "layer1_samples.csv"   # 2026-10-24: optional sample list (step 3); default unchanged
rows = [r for i, r in enumerate(csv.DictReader(open(os.path.join(H, SL)))) if i % n == k]
for r in rows:
    g = r["gsm"]; f = os.path.join(out, r["file"]); t0 = time.time()
    for a in range(5):
        try: urllib.request.urlretrieve(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{r['file']}", f); break
        except Exception as e: print(g, "download retry", type(e).__name__, flush=True); time.sleep(60 * (a + 1))
    else: print(g, "download failed", flush=True); continue
    b = os.path.join(out, g + "_bundle.json")
    p = subprocess.run([sys.executable, RS, "--pat", f, "--specimen", "purified cells", "--seq-cell", r["cell"], "--dev-count-only", "--id", g,
                        "--out", os.path.join(out, g + ".html"), "--bundle", b] + sys.argv[5:], capture_output=True, text=True)   # 2026-10-24: further run_sample.py flags (step 3 version 2)
    os.remove(f); c = (json.load(open(b)).get("iam_a_count_only") or {}).get("count") if os.path.exists(b) else None
    for x in os.listdir(out):
        if x.startswith(g) and not x.endswith("_bundle.json"): os.remove(os.path.join(out, x))
    print(g, r["cell"], f"exit {p.returncode}", c, f"{time.time() - t0:.0f} s", flush=True)
    if p.returncode and not c: print(p.stdout[-600:], p.stderr[-600:], flush=True)
