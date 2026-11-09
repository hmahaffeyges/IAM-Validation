"""Where the IAM-Deconvolver puts each stage of neutrophil maturation (2026-11-08, written and committed before it runs; development, a
diagnostic, nothing scored). One row of bm_neutrophil_stages_samples.csv (public healthy bone marrow: myelocyte, metamyelocyte, band and
segmented neutrophils of two donors, whole-genome bisulfite, GRCh38 level and coverage tracks) is read by the one command as a sequenced
sample whose cells are found against IAM-Atlas v2.1 (--dev-seq-beta CALL,COV; no --dev-seq-ref), specimen unknown so nothing is assumed.
Output in OUTDIR: the bundle and a one-line JSON of the cells found and their shares. Usage: bm_neutrophil_stages_read.py OUTDIR ROW (0-7)"""
import os, sys, json, subprocess, urllib.request, time, csv
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out, row = sys.argv[1], int(sys.argv[2]); os.makedirs(out, exist_ok=True)
r = list(csv.DictReader(open(os.path.join(H, "bm_neutrophil_stages_samples.csv"))))[row]
wd = os.path.join(out, "tmp"); os.makedirs(wd, exist_ok=True); fs = []
for u in (r["call_bw"], r["cov_bw"]):
    f = os.path.join(wd, os.path.basename(u))
    for t in range(5):
        try: urllib.request.urlretrieve(u, f); break
        except Exception as e: print("retry", t, e, flush=True); time.sleep(30)
    fs.append(f)
bf = os.path.join(out, r["id"] + "_bundle.json")
p = subprocess.run([sys.executable, RS, "--dev-seq-beta", ",".join(fs), "--dev-seq-values", "1", "--specimen", "unknown", "--id", r["id"], "--full-chain", "--cell-panel", "atlas",
                    "--out", os.path.join(wd, "r.html"), "--bundle", bf], capture_output=True, text=True)
print(p.stdout[-1500:], p.stderr[-1500:])
res = dict(r, rc=p.returncode)
if p.returncode == 0:
    B = json.load(open(bf)); res["cells"] = {c["cell"]: c.get("share") for c in (B.get("cells_found") or {}).get("cells") or []}
    res["ruler"] = B.get("sequencing_ruler")
json.dump(res, open(os.path.join(out, f"stage_{row}.json"), "w"), indent=1)
for f in fs + [os.path.join(wd, "r.html")]:
    if os.path.exists(f): os.remove(f)
