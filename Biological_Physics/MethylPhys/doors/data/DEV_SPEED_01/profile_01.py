"""DEV-SPEED-01 profile 1 (2026-10-23; tuning): where the one command spends its time. Several jobs hit GitHub's time limit after the one
command gained the 79-cell Deconvolver, Met-A for each cell and the whole sample, the territories, four C-Scores, the sphere and the globe.
This runs run_sample.py once, unchanged, under Python's profiler on one healthy whole blood (GSE235717 GSM7508429, spent) with the series
flags (--cell-panel atlas --dev-metaa-allsites), and writes the wall time and the 40 functions with the most time inside them.
Usage: python3 profile_01.py OUTDIR"""
import os, sys, re, time, subprocess, urllib.request, pstats, io
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
out = sys.argv[1] if len(sys.argv) > 1 else "xout"; os.makedirs(out, exist_ok=True); g = "GSM7508429"
u = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"; got = {}
for n in sorted(set(re.findall(rf"{g}_[^\"<>]*_(?:Grn|Red)\.idat\.gz", urllib.request.urlopen(u, timeout=120).read().decode()))):
    urllib.request.urlretrieve(u + n, os.path.join(out, n)); got["grn" if "_Grn" in n else "red"] = os.path.join(out, n)
t0 = time.time(); prof = os.path.join(out, "prof.out")
r = subprocess.run([sys.executable, "-m", "cProfile", "-o", prof, RS, "--grn", got["grn"], "--red", got["red"], "--specimen", "whole blood", "--id", g,
                    "--out", os.path.join(out, g + ".html"), "--bundle", os.path.join(out, g + ".json"), "--cell-panel", "atlas", "--dev-metaa-allsites"],
                   capture_output=True, text=True)
wall = time.time() - t0; s = io.StringIO(); P = pstats.Stats(prof, stream=s); P.sort_stats("cumulative").print_stats(60)
t = io.StringIO(); P2 = pstats.Stats(prof, stream=t); P2.sort_stats("tottime").print_stats(30)
L = [f"DEV-SPEED-01 profile 1: run_sample.py on {g}, exit {r.returncode}, wall {wall:.0f} s", "--- by time including calls (cumulative) ---", s.getvalue()[-9000:], "--- by time inside the function itself ---", t.getvalue()[-6000:], "--- last output ---", r.stdout[-1500:], r.stderr[-1500:]]
open(os.path.join(out, "profile_01_summary.txt"), "w").write("\n".join(L)); print(L[0])
