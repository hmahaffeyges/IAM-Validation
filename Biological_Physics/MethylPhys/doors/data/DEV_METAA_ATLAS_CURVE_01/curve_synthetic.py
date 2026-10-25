"""DEV-METAA-ATLAS-CURVE-01 synthetic test (written and committed 2026-10-25 before any monocyte array is read; plan 14.14). Constructed arrays
from IAM-Atlas v2.1's own monocyte column, read by the one command (run_sample.py --dev-metaa-atlas monocytes --dev-metaa-curve):
  ARR1: each atlas site's best value m read through a curved response 0.05 + 0.89 (1 - (1 - m)^1.8), plus scatter sd 0.02 (seed 20261025)
  ARR2: the same through 0.04 + 0.91 (1 - (1 - m)^1.3); a different array, the same cells
  ARR1_LOSS: ARR1 with a 5 % loss of the monocyte pattern at its identity sites (m moved 5 % of the way to 0.5 before the response)
Expected: ARR1 and ARR2 read A within 0.01 of 1; ARR1_LOSS reads the A the planted loss predicts (computed here from the same curve,
sites and scatter) within 0.005. Run 3 (runs 1 and 2 disclosed: run 1 left the array's scatter out of the mark and the prediction; run 2 took it from binned entropy, which also averages over the levels inside a bin).   Usage: python3 curve_synthetic.py WORKDIR OUT.txt"""
import sys, os, json, subprocess, numpy as np, pandas as pd
W, OUT = sys.argv[1], sys.argv[2]; os.makedirs(W, exist_ok=True); H = os.path.dirname(os.path.abspath(__file__))
RS = os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface"); sys.path.insert(0, os.path.join(RS, "..")); import stage_metaa_atlas as SA
mu = SA._col("mu", "monocytes").dropna(); ID = pd.Index(SA.identity_sites("monocytes")); g = np.random.default_rng(20261025)
Hb = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
r1 = lambda m: 0.05 + 0.89 * (1 - (1 - m) ** 1.8); r2 = lambda m: 0.04 + 0.91 * (1 - (1 - m) ** 1.3)
m_loss = mu.copy(); k = ID.intersection(mu.index); m_loss[k] = mu[k] + 0.05 * (0.5 - mu[k])
gp = np.random.default_rng(7)   # the predicted A with the same scatter (sd 0.02), 40 draws per site (run 2; run 1 predicted without scatter)
pred = float(np.mean([Hb(np.clip(r1(m_loss[k].to_numpy()) + gp.normal(0, 0.02, len(k)), 0.001, 0.999)).mean() for _ in range(40)]) /
             np.mean([Hb(np.clip(r1(mu[k].to_numpy()) + gp.normal(0, 0.02, len(k)), 0.001, 0.999)).mean() for _ in range(40)]))
L = [f"atlas monocyte sites {len(mu)} | identity sites {len(ID)} | ARR1_LOSS predicted A {pred:.4f}"]; res = {}
for nm, m, f in (("ARR1", mu, r1), ("ARR2", mu, r2), ("ARR1_LOSS", m_loss, r1)):
    b = np.clip(f(m) + g.normal(0, 0.02, len(m)), 0.001, 0.999); p = os.path.join(W, nm + "_betas.csv")
    pd.DataFrame({"beta": b}, index=pd.Index(m.index, name="probe")).to_csv(p)
    q = subprocess.run([sys.executable, "run_sample.py", "--betas", p, "--specimen", "isolated monocytes", "--id", nm, "--dev-metaa-atlas", "monocytes",
                        "--dev-metaa-curve", "--out", os.path.join(W, nm + ".html"), "--bundle", os.path.join(W, nm + ".json")], cwd=RS, capture_output=True, text=True)
    B = json.load(open(os.path.join(W, nm + ".json"))) if q.returncode == 0 else {}
    def find(o):
        if isinstance(o, dict):
            if o.get("stage") == "M-atlas-curve": return o
            for v in o.values():
                r = find(v)
                if r: return r
        return None
    R = find(B) or {}; res[nm] = R.get("A"); L.append(f"{nm} | exit {q.returncode} | A {R.get('A')} | above curve {R.get('above_curve')} | sites {R.get('n_sites')} | curve sites {R.get('n_curve_sites')} | {R.get('reason') or ''}" + ("" if q.returncode == 0 else " | " + q.stdout[-300:] + q.stderr[-300:]))
ok = all(res.get(n) is not None and abs(res[n] - 1) <= 0.01 for n in ("ARR1", "ARR2")) and res.get("ARR1_LOSS") is not None and abs(res["ARR1_LOSS"] - pred) <= 0.005
L.append(f"synthetic (ARR1, ARR2 within 0.01 of 1; ARR1_LOSS within 0.005 of {pred:.4f}): {'met' if ok else 'not met'}")
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
