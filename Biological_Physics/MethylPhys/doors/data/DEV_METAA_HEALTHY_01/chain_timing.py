"""How long each stage of the one command takes on one array (2026-11-08; a timing tool, nothing scored). One spent healthy whole blood
(GSE318173, the first row of realblood_GSE318173_samples.csv) is fetched from GEO and read by run_sample.py --full-chain --cell-panel atlas
--dev-line-wide, its printed lines timestamped as they arrive (each stage prints as it starts), then read again under cProfile. Output in
OUTDIR: chain_timing.txt (seconds between printed lines, the slowest first) and chain_profile.txt (the 40 functions with the most cumulative
time). Usage: chain_timing.py OUTDIR"""
import os, sys, time, subprocess, urllib.request, re, csv, pstats, io
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); wd = os.path.join(out, "tmp"); os.makedirs(wd, exist_ok=True)
g = next(csv.DictReader(open(os.path.join(H, "realblood_GSE318173_samples.csv"))))["gsm"]
u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"
names = sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode())))
fs = {}
for n in names:
    f = os.path.join(wd, n); urllib.request.urlretrieve(u + n, f); fs["grn" if "_Grn" in n else "red"] = f
base = [sys.executable, "-u", RS, "--grn", fs["grn"], "--red", fs["red"], "--specimen", "whole blood", "--id", g, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide",
        "--out", os.path.join(wd, "r.html"), "--bundle", os.path.join(wd, "b.json")]
t0 = time.time(); last = t0; prev = "start"; rows = []
p = subprocess.Popen(base, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    now = time.time(); rows.append((now - last, prev[:150])); last = now; prev = line.strip() or prev
p.wait(); rows.append((time.time() - last, prev[:150])); total = time.time() - t0
with open(os.path.join(out, "chain_timing.txt"), "w") as o:
    o.write(f"{g}: rc {p.returncode}, {total:.0f} s in all\nseconds after each printed line (the slowest 40):\n")
    for s, l in sorted(rows, reverse=True)[:40]: o.write(f"{s:8.1f}  {l}\n")
prof = os.path.join(wd, "p.prof")
subprocess.run([sys.executable, "-m", "cProfile", "-o", prof] + base[2:], capture_output=True, text=True)
sio = io.StringIO(); pstats.Stats(prof, stream=sio).sort_stats("cumulative").print_stats(40)
open(os.path.join(out, "chain_profile.txt"), "w").write(sio.getvalue())
print(open(os.path.join(out, "chain_timing.txt")).read())
