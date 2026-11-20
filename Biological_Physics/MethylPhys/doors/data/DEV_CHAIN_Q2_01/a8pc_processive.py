import subprocess as _sp, sys as _sy
_sp.run([_sy.executable, '-m', 'pip', 'install', '-q', '-r', 'Biological_Physics/MethylPhys/chain/requirements.txt'], check=True, timeout=900)
"""A8 processive crew (register '#### A8 PROCESSIVE CREW, PREDICTION', written before this ran): forward_candidate.py mode 'proc', windows of
1, 2, 4, 8 CpGs, recruitment share rho 0, 0.5, 0.8; 1 replicate each; the counter's held_err and coupling."""
import os, sys, math, json, subprocess
D = os.path.dirname(os.path.abspath(__file__)); FC = os.path.join(D, "forward_candidate.py")
os.chdir(os.path.join(D, "..", "..", "..", "chain", "MethylPhys_Interface")); R = []
for rho in (0.0, 0.5, 0.8):
    for w in (1, 2, 4, 8):
        out = f"/tmp/a8pc_{rho}_{w}"
        p = subprocess.run([sys.executable, FC, "proc", str(w), "1", out, "1.0", str(rho)], capture_output=True, text=True, timeout=7200)
        if p.returncode != 0: print("::error::" + p.stdout[-400:] + p.stderr[-400:]); raise SystemExit(1)
        row = open(out + ".csv").read().splitlines()[1].split(","); he, cp = float(row[6]), float(row[5])
        true = (lambda fr: fr / (1 + fr))((0.04454 / (1 - 0.04454)) * (rho + (1 - rho) * w) / w)
        R.append((rho, w, true, he, cp)); print(f"A8PC rho {rho} w {w}: true per-site {true:.5f} counter held_err {he:.5f} coupling {cp:.3f}", flush=True)
for rho in (0.0, 0.5, 0.8):
    x = [math.log(r[3]) for r in R if r[0] == rho]; y = [r[4] for r in R if r[0] == rho]
    my, mx = sum(x) / len(x), sum(y) / len(y); b = sum((a - my) * (c - mx) for a, c in zip(x, y)) / sum((c - mx) ** 2 for c in y)
    print(f"::notice::A8PC rho {rho}: d ln(counter held_err) / d coupling {b:+.3f} over coupling {min(y):.2f}-{max(y):.2f} (observed between types -0.258)", flush=True)
json.dump(R, open(os.path.join(D, "a8pc_out.json"), "w"), indent=1)
