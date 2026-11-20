import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', '-r', 'Biological_Physics/MethylPhys/chain/requirements.txt'], check=True, timeout=900)
#!/usr/bin/env python3
"""A8D3b (register '#### A8D3b COUNTER BY PUSH', written before this ran): the counter's held_err for a constant push of several coins, through the
one command (forward_candidate.py, unchanged), seq 1.5, 2 replicates each."""
import os, sys, math, json, subprocess
D = os.path.dirname(os.path.abspath(__file__)); FC = os.path.join(D, "forward_candidate.py")
C = json.load(open(os.path.join(D, "..", "..", "..", "..", "..", "CANON", "iam_canon.json")))["constants"]; E0 = C["eps0_candidate"]["value"]
M = 54000.0 / (8.314462618 * 310.15)
def eps(coins):
    w = coins * M; o = (w / 2) / (1 - (1 - math.exp(-w)) / w); return 1 / (1 + o)
os.chdir(os.path.join(D, "..", "..", "..", "chain", "MethylPhys_Interface")); R = {}
for c in (0.9, 1.0, 1.1, 1.2, 1.3):
    e = eps(c); ratio = E0 / e; out = f"/tmp/a8d3_{c}"
    p = subprocess.run([sys.executable, FC, "seq", "1.5", "2", out, str(ratio)], capture_output=True, text=True, timeout=7200)
    if p.returncode != 0: print("::error::" + p.stdout[-400:] + p.stderr[-400:]); raise SystemExit(1)
    rows = [l.split(",") for l in open(out + ".csv").read().splitlines()[1:]]
    he = sum(float(x[6]) for x in rows) / len(rows); cp = sum(float(x[5]) for x in rows) / len(rows)
    R[c] = dict(per_site=e, counter=he, coupling=cp, transfer=he / e)
    print(f"A8D3b push {c} coins: per-site {e:.5f} counter held_err {he:.5f} transfer {he/e:.3f} coupling {cp:.3f}", flush=True)
print("::notice::A8D3b " + " | ".join(f"{c}:{v['counter']:.4f}" for c, v in R.items()), flush=True)
os.makedirs("/tmp/xo", exist_ok=True); json.dump(R, open(os.path.join(D, "a8d3b_out.json"), "w"), indent=1)
