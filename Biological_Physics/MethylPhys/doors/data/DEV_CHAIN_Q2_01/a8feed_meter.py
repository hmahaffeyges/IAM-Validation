import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', '-r', 'Biological_Physics/MethylPhys/chain/requirements.txt'], check=True, timeout=900)
"""A8FEED meter check (register '#### A8 FEEDER METER CHECK', written before this ran): the same true per-site loss (the two-coin level, independent
losses) with molecules carrying 0.5x, 0.75x, 1x, 1.5x the real libraries' CpGs; the counter's held_err."""
import os, sys, json, subprocess
D = os.path.dirname(os.path.abspath(__file__)); FC = os.path.join(D, "forward_candidate.py")
os.chdir(os.path.join(D, "..", "..", "..", "chain", "MethylPhys_Interface")); R = {}
for ls in (0.5, 0.75, 1.0, 1.5):
    out = f"/tmp/a8fm_{ls}"
    p = subprocess.run([sys.executable, FC, "ind", "0", "2", out, "1.0", "0.0", str(ls)], capture_output=True, text=True, timeout=7200)
    if p.returncode != 0: print("::error::" + p.stdout[-400:] + p.stderr[-400:]); raise SystemExit(1)
    rows = [l.split(",") for l in open(out + ".csv").read().splitlines()[1:]]; he = sum(float(x[6]) for x in rows) / len(rows)
    R[ls] = he; print(f"A8FM CpGs per molecule x{ls}: counter held_err {he:.5f} (true per-site 0.04454)", flush=True)
print("::notice::A8FM " + " | ".join(f"x{k}: {v:.5f}" for k, v in R.items()) + f" | sparse/dense ratio x0.5/x1.5 = {R[0.5] / R[1.5]:.3f}", flush=True)
