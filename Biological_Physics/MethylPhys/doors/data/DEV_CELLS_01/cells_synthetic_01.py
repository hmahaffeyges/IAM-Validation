"""DEV-CELLS-01, synthetic first (2026-10-21, exploratory, not sealed): every panel cell scored in whole blood (to-do step 21).
The instrument's job: a healthy cell reads no loss, and a loss of a cell's pattern is read as that loss.
Synthetic whole bloods are built from IAM-Atlas v2.1's six blood cells: shares drawn around a typical blood count, each cell's values drawn
from its atlas mean and donor spread, plus 0.02 of measurement noise. 8 healthy bloods; then on donors 1-6 a 25 % loss of one cell's pattern
each (beta + f_c x 0.25 x (0.5 - mu_c) at every site), and on donor 7 a 1 % loss of the neutrophils' pattern. Every blood is read by the one
command, run_sample.py --betas, unchanged. This tests the machinery on data that follows the atlas by construction; real arrays come next.
Needs the draft stage (stage_cells_draft.patch: git apply it from the repository root); the stage is kept out of the chain until decided.
Usage: python3 cells_synthetic_01.py OUTDIR   Writes OUTDIR/cells_synthetic_01_rows.csv and cells_synthetic_01_summary.txt here."""
import os, sys, json, subprocess, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_deconv_v21 as D
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
R = D.reference(list(D.BLOOD6)); mu, sd = R["mu"], R["sd"]; ok = mu.notna().all(1) & sd.notna().all(1); mu, sd = mu[ok], sd[ok]
rng = np.random.default_rng(20261021); base = np.array([0.60, 0.08, 0.05, 0.15, 0.08, 0.04])
runs = [(k, None, 0.0) for k in range(8)] + [(k + 1, c, 0.25) for k, c in enumerate(D.BLOOD6)] + [(7, "neutrophils", 0.01)]
donors = {}
for k in range(8):
    f = dict(zip(D.BLOOD6, rng.dirichlet(base * 60))); donors[k] = (f, {c: (mu[c] + rng.normal(0, 1, len(mu)) * sd[c]).clip(0, 1) for c in D.BLOOD6}, rng.normal(0, 0.02, len(mu)))
rows = []
for k, cell, loss in runs:
    f, P, noise = donors[k]; b = sum(f[c] * P[c] for c in D.BLOOD6) + noise
    if cell: b = b + f[cell] * loss * (0.5 - mu[cell])
    tag = f"d{k}" + (f"_{cell.replace(' ', '')}_{int(loss * 100)}" if cell else ""); bp = os.path.join(out, tag + ".csv")
    pd.DataFrame({"beta": b.clip(0.001, 0.999)}).to_csv(bp)
    rp = os.path.join(out, tag + ".html")
    subprocess.run([sys.executable, os.path.join(CH, "MethylPhys_Interface", "run_sample.py"), "--betas", bp, "--specimen", "whole blood", "--id", tag, "--out", rp],
                   check=True, capture_output=True)
    B = json.load(open(rp[:-5] + "_bundle.json")); os.remove(bp)
    for x in B["cell_scores"]["cells"]:
        rows.append(dict(run=tag, donor=k, planted_cell=cell or "", planted_loss=loss if x["cell"] == cell else 0.0, cell=x["cell"], true_share=round(f[x["cell"]], 4),
                         share=x["share"], pattern_lost=x.get("pattern_lost"), se=x.get("se"), se_share=x.get("se_share"), detection_limit_pct=x.get("detection_limit_pct"), state=x.get("state")))
    print(tag, flush=True)
T = pd.DataFrame(rows); T.to_csv(os.path.join(H, "cells_synthetic_01_rows.csv"), index=False)
h = T[T.planted_cell == ""]; L = []
L.append(f"healthy: {len(h)} cell readings on 8 bloods; inside 2 SE (no loss found): {int((h.state == 'no loss found').sum())} of {len(h)}")
for c, g in h.groupby("cell", sort=False):
    L.append(f"  {c}: share {g.true_share.mean():.3f}, pattern lost mean {g.pattern_lost.mean():+.3f} (SD {g.pattern_lost.std():.3f}), detection limit median {g.detection_limit_pct.median():.1f} %")
L.append("planted (the planted cell only; its reading against the same donor's healthy reading):")
for _, r in T[T.planted_loss > 0].iterrows():
    b0 = h[(h.donor == r.donor) & (h.cell == r.cell)].iloc[0]
    L.append(f"  {r.cell} {100 * r.planted_loss:.0f} % loss at share {r.true_share:.3f}: read {r.pattern_lost:+.3f} (healthy {b0.pattern_lost:+.3f}, change {r.pattern_lost - b0.pattern_lost:+.3f}), limit {r.detection_limit_pct:.1f} % -> {r.state}")
open(os.path.join(H, "cells_synthetic_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
