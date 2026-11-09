"""Identity check of the marker cache (2026-11-08; written and committed with the change, before it runs). One spent healthy whole blood
(GSE318173, the first row of realblood_GSE318173_samples.csv) is read by the one command twice with an empty cache folder: the first reading
computes the Deconvolver's marker sites (as before the cache) and stores them, the second reads them from the cache. Met: the two bundles are
identical once timestamps, run ids and paths are set aside, and the second is faster. Output: OUTDIR/marker_cache_identity.txt.
Usage: marker_cache_identity.py OUTDIR"""
import os, sys, time, subprocess, urllib.request, re, csv, json, tempfile
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); wd = os.path.join(out, "tmp"); os.makedirs(wd, exist_ok=True)
g = next(csv.DictReader(open(os.path.join(H, "realblood_GSE318173_samples.csv"))))["gsm"]
u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"
names = sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))); fs = {}
for n in names:
    f = os.path.join(wd, n); urllib.request.urlretrieve(u + n, f); fs["grn" if "_Grn" in n else "red"] = f
cache = tempfile.mkdtemp(); env = dict(os.environ, METHYLPHYS_CACHE=cache); res = []
for k in (1, 2):
    bj = os.path.join(wd, f"b{k}.json"); t0 = time.time()
    p = subprocess.run([sys.executable, RS, "--grn", fs["grn"], "--red", fs["red"], "--specimen", "whole blood", "--id", g, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide",
                        "--out", os.path.join(wd, f"r{k}.html"), "--bundle", bj], capture_output=True, text=True, env=env)
    res.append((p.returncode, time.time() - t0, bj, len(os.listdir(cache))))
SKIP = re.compile(r"(?i)time|stamp|run_id|^id$|path|ledger|elapsed|command|out$|bundle$")
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items() if not SKIP.search(k)}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, str): return re.sub(r"/[^ ]*/(tmp|xout)/[^ ]*", "<path>", o)
    return o
A, B = (clean(json.load(open(r[2]))) for r in res); same = json.dumps(A, sort_keys=True, default=str) == json.dumps(B, sort_keys=True, default=str)
diff = [k for k in sorted(set(A) | set(B)) if json.dumps(A.get(k), sort_keys=True, default=str) != json.dumps(B.get(k), sort_keys=True, default=str)]
L = [f"{g}: reading 1 rc {res[0][0]}, {res[0][1]:.0f} s, cache files after {res[0][3]}; reading 2 rc {res[1][0]}, {res[1][1]:.0f} s, cache files after {res[1][3]}",
     f"bundles identical apart from timestamps, run ids and paths: {same}" + ("" if same else f"; differing keys: {diff[:20]}"), "met" if same and res[1][1] < res[0][1] else "not met"]
open(os.path.join(out, "marker_cache_identity.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
