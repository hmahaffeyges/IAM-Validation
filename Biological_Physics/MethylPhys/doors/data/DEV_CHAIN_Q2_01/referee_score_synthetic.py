"""Synthetic test of referee_score.py (PHY-084), before any library is read. A stand-in grid (the committed ratio-1 seq runs in forward_out,
scaled by 1/r; a test of the scorer's logic, not a prediction) and planted libraries with the referee set's design (24 B, 12 CD4, 12 CD8,
10 monocytes; loose share 0.05-0.30 with slope 0.0284; 2 % library scatter): (a) planted on each cell type's prediction: must be MET with
pulls near 0; (b) planted 30 % high: must be NOT MET. Writes referee_score_synthetic.txt."""
import os, sys, glob, json, tempfile, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import referee_score as R
W = tempfile.mkdtemp(prefix="rs"); os.makedirs(f"{W}/grid")
for r in R.RATIOS:
    for f in glob.glob(os.path.join(H, "forward_out", "fc_seq_*.csv")):
        X = pd.read_csv(f); X["held_err_length"] /= r; X.to_csv(f"{W}/grid/pg_{r}_{X.param.iloc[0]}.csv", index=False)
G = R.grid(f"{W}/grid"); rng = np.random.default_rng(2); L = []
for tag, shift, want in (("on", 1.0, True), ("hi", 1.30, False)):
    d = f"{W}/{tag}"; os.makedirs(d); rows = []
    for cell, n, cp in (("b cells", 24, 1.45), ("cd4 t cells", 12, 1.50), ("cd8 t cells", 12, 1.55), ("monocytes", 10, 1.47)):
        mid = np.mean([R.predict(G, r, cp) for r in R.RATIOS])
        for k in range(n):
            g = f"{tag}{cell[:3]}{k}"; loose = rng.uniform(0.05, 0.3)
            json.dump(dict(gsm=g, held_err_length=mid * shift * (1 + rng.normal(0, 0.02)) + 0.0284 * loose, loose_share=loose, coupling=cp), open(f"{d}/{g}_count.json", "w")); rows.append((g, cell))
    pd.DataFrame(rows, columns=["gsm", "cell"]).to_csv(f"{W}/{tag}.csv", index=False)
    met = R.main(d, f"{W}/{tag}.csv", f"{W}/grid", f"{W}/out_{tag}")
    O = pd.read_csv(f"{W}/out_{tag}.csv")
    L.append(f"{tag} (planted x{shift}): {'MET' if met else 'NOT MET'}; expected {'MET' if want else 'NOT MET'}: {'ok' if met == want else 'WRONG'}; pulls {list(O.pull)}; combined u {list(O.u_combined)}")
open(os.path.join(H, "referee_score_synthetic.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
if any("WRONG" in x for x in L): sys.exit(1)
