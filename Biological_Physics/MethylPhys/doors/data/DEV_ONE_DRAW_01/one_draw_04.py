"""DEV-ONE-DRAW-01, exploration 4 (written and committed 2026-10-22 before it runs; exploration, not sealed; to-do 40.10). One blood draw, both
halves, through the chain: for each of 3 healthy donors of GSE186888 (white blood cells and matched plasma cell-free DNA, whole-genome bisulfite
sequencing; the first three donors by number, declared spent before reading; the other 20 stay unread), the one-draw runner
(chain/MethylPhys_Interface/run_draw.py, which runs run_sample.py on each half on IAM-Atlas v2.1's full panel) reads the two .beta count files.
This file only downloads the public files and reads the draw records. Reported per donor: each half's cells and shares above 0.1 %, Met-A for each
cell (untared), and what the plasma holds that the person's own white cells do not explain. IAM-A is not read (count files, no molecules).
Usage: python3 one_draw_04.py OUTDIR"""
import os, sys, csv, json, subprocess, urllib.request, time
H = os.path.dirname(os.path.abspath(__file__)); RD = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_draw.py"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True); S = list(csv.DictReader(open(os.path.join(H, "gse186888_explore_samples.csv")))); L = ["DEV-ONE-DRAW-01 exploration 4: GSE186888, 3 healthy donors, white cells and plasma, through run_draw.py"]
def get(u):
    f = os.path.join(out, os.path.basename(u))
    for k in range(5):
        try: urllib.request.urlretrieve(u, f); return f
        except Exception as e: print("retry", type(e).__name__, flush=True); time.sleep(60 * (k + 1))
    sys.exit("download failed: " + u)
for d in sorted({r["donor"] for r in S}):
    h = {r["half"]: get(r["url"]) for r in S if r["donor"] == d}; o = os.path.join(out, "draw_" + d)
    r = subprocess.run([sys.executable, RD, "--id", "GSE186888_" + d, "--out", o, "--buffy-seqbeta", h["buffy"], "--plasma-seqbeta", h["plasma"]], capture_output=True, text=True)
    if r.returncode: L.append(f"donor {d}: run_draw.py exit {r.returncode}: {(r.stdout + r.stderr)[-600:]}"); continue
    s = open(os.path.join(o, "draw_summary.txt")).read(); L += [f"===== donor {d}", s.strip()]
    for f in h.values(): os.remove(f)
open(os.path.join(out, "one_draw_04_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
