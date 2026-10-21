"""DEV-SELFTARE-03, synthetic first (2026-10-21, exploration, not sealed): the anchor range (dev_stages.anchor_range) as the reading of a loss
at every site. Synthetic whole bloods from IAM-Atlas v2.1's six blood cells (shares around a typical blood count, each cell from its atlas mean
and donor spread, 0.02 measurement noise), read by the chain's conductor (conductor_v3.run_neutrophil, unchanged) for Met-A and its
neutrophil share; the anchor range of each is tared against the 12 healthy bloods. Planted on 6 donors: a loss of the neutrophils' pattern at
every site of 1, 2 and 5 %. Writes anchor_range_synthetic_01_summary.txt here. Usage: python3 anchor_range_synthetic_01.py"""
import os, sys, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain"))
import conductor_v3 as C, dev_stages as DV, stage_deconv_v21 as D
R = D.reference(list(D.BLOOD6)); mu, sd = R["mu"], R["sd"]; ok = mu.notna().all(1) & sd.notna().all(1); mu, sd = mu[ok], sd[ok]
rng = np.random.default_rng(20261021); base = np.array([0.60, 0.08, 0.05, 0.15, 0.08, 0.04]); rows = []
def blood():
    f = dict(zip(D.BLOOD6, rng.dirichlet(base * 60))); P = {g: (mu[g] + rng.normal(0, 1, len(mu)) * sd[g]).clip(0, 1) for g in D.BLOOD6}
    return f, (sum(f[g] * P[g] for g in D.BLOOD6) + rng.normal(0, 0.02, len(mu))).clip(0.001, 0.999)
healthy = [blood() for _ in range(12)]
Rref = [DV.anchor_range(b, None)["R"] for _, b in healthy]
L = ["12 healthy bloods: anchor range R " + ", ".join(f"{x:.4f}" for x in Rref)]
for k in range(6):
    f, b0 = blood()
    for p in (0.0, 0.01, 0.02, 0.05):
        b = (b0 + f["neutrophils"] * p * (0.5 - mu["neutrophils"])).clip(0.001, 0.999)
        o = C.run_neutrophil(b, specimen="whole blood"); fn = (o.get("composition") or {}).get("fractions", {}).get("NEU")
        a = DV.anchor_range(b, fn, Rref)
        rows.append(dict(donor=k, planted_pct=100 * p, true_share=f["neutrophils"], read_share=fn, met_a=o["met_a"].get("A"), R_rel=a.get("R_rel"),
                         loss_read_pct=a.get("loss_every_site_pct"), limit_pct=a.get("detection_limit_pct")))
T = pd.DataFrame(rows); T.to_csv(os.path.join(H, "anchor_range_synthetic_01_rows.csv"), index=False)
for p, g in T.groupby("planted_pct"):
    h = T[T.planted_pct == 0].set_index("donor")
    L.append(f"planted {p:.0f} % at every site: loss read {g.loss_read_pct.mean():+.2f} % (range {g.loss_read_pct.min():+.2f} to {g.loss_read_pct.max():+.2f}); "
             f"change from the donor's healthy reading {(g.set_index('donor').loss_read_pct - h.loss_read_pct).mean():+.2f} %; Met-A change {(g.set_index('donor').met_a - h.met_a).mean():+.4f}; limit {g.limit_pct.median():.2f} %")
open(os.path.join(H, "anchor_range_synthetic_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
