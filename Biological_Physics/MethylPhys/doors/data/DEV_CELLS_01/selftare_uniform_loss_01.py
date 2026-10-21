"""DEV-CELLS-01, exploration (2026-10-21, not sealed, no bar): does the chain's whole-blood Met-A see a loss of the neutrophils' pattern that
happens at EVERY site, not only at its 6,000 identity sites? Synthetic whole bloods from IAM-Atlas v2.1's six blood cells (shares around a
typical blood count, each cell drawn from its atlas mean and donor spread, 0.02 measurement noise). Each is read by the chain's conductor
(conductor_v3.run_neutrophil, unchanged): healthy; a 1 % loss of the neutrophils' pattern at the identity sites only (the rule the
commissioning tests planted); the same loss at every site; and the last read again with Stage T step 1 (self-tare II) replaced by the
identity map for this test only. Writes selftare_uniform_loss_01_summary.txt here. Usage: python3 selftare_uniform_loss_01.py"""
import os, sys, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_deconv_v21 as D, conductor_v3 as C
R = D.reference(list(D.BLOOD6)); mu, sd = R["mu"], R["sd"]; ok = mu.notna().all(1) & sd.notna().all(1); mu, sd = mu[ok], sd[ok]
F = json.load(open(os.path.join(CH, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
NS = pd.Index(F["sites"]).intersection(mu.index); rng = np.random.default_rng(5); base = np.array([0.60, 0.08, 0.05, 0.15, 0.08, 0.04])
orig = C.stage_t_selftare_ii; off = lambda beta, *a, **k: (beta, {"status": "replaced by the identity map for this test"}); L = []
for k in range(4):
    f = dict(zip(D.BLOOD6, rng.dirichlet(base * 60))); P = {g: (mu[g] + rng.normal(0, 1, len(mu)) * sd[g]).clip(0, 1) for g in D.BLOOD6}
    b0 = (sum(f[g] * P[g] for g in D.BLOOD6) + rng.normal(0, 0.02, len(mu))).clip(0.001, 0.999); m = pd.Series(0.0, index=mu.index); m[NS] = 1.0
    plant = f["neutrophils"] * 0.01 * (0.5 - mu["neutrophils"]); bi = (b0 + m * plant).clip(0.001, 0.999); be = (b0 + plant).clip(0.001, 0.999)
    A = lambda b: C.run_neutrophil(b, specimen="whole blood")["met_a"]["A"]
    a0, ai, ae = A(b0), A(bi), A(be); C.stage_t_selftare_ii = off; o0, oe = A(b0), A(be); C.stage_t_selftare_ii = orig
    L.append(f"blood {k} (neutrophil share {f['neutrophils']:.3f}): 1 % loss at identity sites {ai - a0:+.4f}; at every site {ae - a0:+.4f}; "
             f"at every site with self-tare II off {oe - o0:+.4f}")
open(os.path.join(H, "selftare_uniform_loss_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
