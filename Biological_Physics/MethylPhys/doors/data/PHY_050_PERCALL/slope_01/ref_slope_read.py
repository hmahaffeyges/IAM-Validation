"""PHY-087 step 2, the reading (exploration on spent libraries; 2026-11-14; runs on GitHub as the last pipeline item). The blind-test
laboratory's 54 healthy libraries (GSE173787; ref_tile_runs.csv) were recounted by the chain from the molecules saved for PHY-084 (pipeline
items p087s2-<GSM>: erase_test.yml with patfrom, run_sample.py --dev-count-only --dev-count-tiles, which since 2026-11-14 writes each tile's
mean call). This script reads only those chain outputs (each run's <GSM>_tiles.csv.gz), then:
  (1) slope_read.py's reading unchanged (null test, within-sample and across-sample slopes);
  (2) the derived hold with cooperative maintenance (slope_pump.py's form): each library's within-sample slope over the form's slope at its own
      looseness, and the across-type slope over the form's. Prediction written first (design/PHY-087_s2.json): within ratio 0.64-1.34.
Writes xout/ref_slope_read_summary.txt and xout/ref_slope_pump_summary.txt."""
import os, sys, io, csv, json, math, zipfile, subprocess, urllib.request, statistics as S
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.abspath("xout"); TD = os.path.join(OUT, "tiles"); os.makedirs(TD, exist_ok=True)
TOK = os.environ["GH_TOKEN"]; REPO = os.environ.get("GITHUB_REPOSITORY", "hmahaffeyges/IAM-Validation")
def get(u, raw=False):
    rq = urllib.request.Request(u, headers={"Authorization": "Bearer " + TOK, "Accept": "application/vnd.github+json"})
    b = urllib.request.urlopen(rq, timeout=600).read(); return b if raw else json.loads(b)
runs = {}
for page in range(1, 6):
    for w in get(f"https://api.github.com/repos/{REPO}/actions/runs?per_page=100&page={page}")["workflow_runs"]:
        t = w.get("display_title") or ""
        if t.startswith("pq:p087s2-") and w["conclusion"] == "success" and t[10:] not in runs: runs[t[10:]] = w["id"]
rows = list(csv.DictReader(open(os.path.join(H, "ref_tile_runs.csv")))); miss = []
for r in rows:
    g = r["gsm"]
    if g not in runs: miss.append(g); continue
    A = get(f"https://api.github.com/repos/{REPO}/actions/runs/{runs[g]}/artifacts")["artifacts"]
    z = zipfile.ZipFile(io.BytesIO(get(A[0]["archive_download_url"], raw=True)))
    open(os.path.join(TD, f"{g}_tiles.csv.gz"), "wb").write(z.read(f"eout/{g}_tiles.csv.gz"))
print(len(rows) - len(miss), "libraries read; missing:", miss, flush=True)
subprocess.run([sys.executable, os.path.join(H, "slope_read.py")], check=True, env=dict(os.environ, TILES_DIR=TD,
               SAMPLES=os.path.join(H, "ref_tile_runs.csv"), SUMMARY=os.path.join(OUT, "ref_slope_read_summary.txt"), SKIP_MISSING="1"))
M = 54000 / (8.314462618 * 310.15); odds = lambda w: (w / 2) / (1 - (1 - math.exp(-w)) / w)
e = lambda x: 1 / (1 + odds(2 * M * (1 - x))); sl = lambda x: (e(x + 0.005) - e(x - 0.005)) / 0.01
T = [l.split() for l in open(os.path.join(OUT, "ref_slope_read_summary.txt")) if l.strip().startswith("GSM")]
X = np.array([[float(t[-4]), float(t[-3]), float(t[-2])] for t in T]); cells = [" ".join(t[1:-7]) for t in T]; hr = [float(t[-5]) for t in T]   # columns: ... tiles, halves_r, x_mean, held, within_b, within_a
ok = np.array([h > 0.7 for h in hr]); rat = X[ok, 2] / np.array([0.915 * sl(x) for x in X[ok, 0]])
pr = np.array([0.915 * e(x) for x in X[:, 0]]); bm, bp = np.polyfit(X[ok, 0], X[ok, 1], 1)[0], np.polyfit(X[ok, 0], pr[ok], 1)[0]
L = [f"second laboratory, {ok.sum()} libraries passing the null test of {len(X)}:",
     f"within-sample slope over the derived form at each library's own looseness: median {np.median(rat):.2f} (range {rat.min():.2f}-{rat.max():.2f}); "
     f"prediction written first 0.64-1.34: {'INSIDE' if 0.64 <= np.median(rat) <= 1.34 else 'OUTSIDE'}",
     f"across libraries: held level rises {bm:.4f} per unit mean tile looseness; the form gives {bp:.4f} ({bm / bp:.1f} times; first laboratory 2.4)"]
for c in sorted(set(cells)):
    k = [i for i, x in enumerate(cells) if x == c and ok[i]]
    if k: L.append(f"  {c:14} n {len(k):2}  mean looseness {X[k, 0].mean():.3f}  held {X[k, 1].mean():.4f}  within ratio {np.median([X[i, 2] / (0.915 * sl(X[i, 0])) for i in k]):.2f}")
L.append("Exploration on spent libraries, chain outputs only; nothing sealed.")
open(os.path.join(OUT, "ref_slope_pump_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
