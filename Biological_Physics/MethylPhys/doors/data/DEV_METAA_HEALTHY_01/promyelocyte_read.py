"""Where the IAM-Deconvolver puts a healthy promyelocyte on arrays (2026-11-08, written and committed before it runs; development, a
diagnostic, nothing scored). GSE42118's two healthy-donor arrays (450K, processed values; no raw files published): GSM1032772, sorted
promyelocytes, and GSM1032771, CD34-positive cells, each read by the one command from its values (--betas), specimen unknown. Output in
OUTDIR: each bundle and promyelocyte_read.json (cells found and shares). Usage: promyelocyte_read.py OUTDIR"""
import os, sys, json, gzip, subprocess, urllib.request, time
import pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); wd = os.path.join(out, "tmp"); os.makedirs(wd, exist_ok=True)
u = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42118/matrix/GSE42118_series_matrix.txt.gz"; f = os.path.join(wd, "m.txt.gz")
for t in range(5):
    try: urllib.request.urlretrieve(u, f); break
    except Exception as e: print("retry", e, flush=True); time.sleep(30)
lines = gzip.open(f, "rt").read().splitlines(); i0 = lines.index("!series_matrix_table_begin") + 1; i1 = lines.index("!series_matrix_table_end")
from io import StringIO
M = pd.read_csv(StringIO("\n".join(lines[i0:i1])), sep="\t", index_col=0); M.columns = [c.strip('"') for c in M.columns]; M.index = [str(x).strip('"') for x in M.index]
res = {}
for g, name in (("GSM1032772", "promyelocyte"), ("GSM1032771", "CD34")):
    x = pd.to_numeric(M[g], errors="coerce").dropna(); bf = os.path.join(wd, g + ".csv"); pd.DataFrame({"cpg_id": x.index, "beta": x.round(4).values}).to_csv(bf, index=False)
    bj = os.path.join(out, g + "_bundle.json")
    p = subprocess.run([sys.executable, RS, "--betas", bf, "--specimen", "unknown", "--id", g, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide", "--out", os.path.join(wd, g + ".html"), "--bundle", bj], capture_output=True, text=True)
    print(g, p.returncode, p.stdout[-800:], p.stderr[-800:], flush=True)
    res[g] = dict(cell=name, sites=int(len(x)), rc=p.returncode)
    if p.returncode == 0:
        B = json.load(open(bj)); res[g]["cells"] = {c["cell"]: c.get("share") for c in (B.get("cells_found") or {}).get("cells") or []}
json.dump(res, open(os.path.join(out, "promyelocyte_read.json"), "w"), indent=1)
