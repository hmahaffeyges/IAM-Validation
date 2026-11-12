"""M or M + 1 (exploration on spent counts; nothing sealed). Is the healthy cell's restore/loss odds M or 1 + M?
The length-corrected lone-loss rate (held_err_length, run_sample.py --geo-pat --dev-count-only, chain Stage Q2) of the 153 Loyfer samples
(DEV_CHAIN_Q2_01/percall_out) and the 54 unseen ones (lone_loss_loyfer_out); neurons set aside (their marks outside CpGs are not counted);
zero-loose intercept by Theil-Sen, 95 % by resampling samples (2,000, seed 20261112). M from CANON/iam_canon.json.
Writes m_or_m_plus_1.csv, m_or_m_plus_1_shared_cells.csv, m_or_m_plus_1_summary.txt."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import theilslopes
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, "..", "..", "..", "..", "..")
M = float(json.load(open(os.path.join(ROOT, "CANON", "iam_canon.json")))["constants"]["M_cell"]["value"])
def load(d):
    R = []
    for f in sorted(glob.glob(os.path.join(H, d, "*.json"))):
        c = json.load(open(f))
        for x in (c if isinstance(c, list) else [c]):
            if isinstance(x, dict) and x.get("held_err_length") is not None: R.append(dict(gsm=x["sample"], cell=str(x["cell"]).lower(), obs=x["held_err_length"], loose=x["loose_share"]))
    X = pd.DataFrame(R).drop_duplicates("gsm"); X["neuron"] = X.cell.str.contains("neuron"); return X
A = load("../DEV_CHAIN_Q2_01/percall_out"); B = load("lone_loss_loyfer_out")
rng = np.random.default_rng(20261112)
def icpt(g):
    s, b, _, _ = theilslopes(g.obs, g.loose)
    bb = [theilslopes(g.obs.values[ix], g.loose.values[ix])[1] for ix in (rng.integers(0, len(g), len(g)) for _ in range(2000))]
    return b, s, np.percentile(bb, 2.5), np.percentile(bb, 97.5)
out = []
for lab, g in (("153 Loyfer, non-neurons", A[~A.neuron]), ("54 Loyfer unseen, non-neurons", B[~B.neuron]), ("both, non-neurons", pd.concat([A[~A.neuron], B[~B.neuron]]))):
    b, s, lo, hi = icpt(g)
    out.append(dict(set=lab, n=len(g), intercept=round(b, 5), lo95=round(lo, 5), hi95=round(hi, 5), slope=round(s, 4), M_in=bool(lo <= 1 / (1 + M) <= hi), M_plus_1_in=bool(lo <= 1 / (2 + M) <= hi)))
O = pd.DataFrame(out); O.to_csv(os.path.join(H, "m_or_m_plus_1.csv"), index=False)
for X in (A, B): X["ratio"] = X.obs / (1 / (1 + M) + 0.0284 * X.loose)
S = pd.DataFrame([dict(cell=c, n153=int((A.cell == c).sum()), r153=round(A[A.cell == c].ratio.median(), 3), n54=int((B.cell == c).sum()), r54=round(B[B.cell == c].ratio.median(), 3)) for c in sorted(set(A.cell) & set(B.cell))])
S.to_csv(os.path.join(H, "m_or_m_plus_1_shared_cells.csv"), index=False)
t = (f"M or M + 1 (exploration on spent counts; nothing sealed). M = {M}; odds M give 1/(1+M) = {1/(1+M):.5f}; odds 1 + M give 1/(2+M) = {1/(2+M):.5f}.\n"
     + O.to_string(index=False) + "\nShared cell types, median observed / M-based prediction (0.0284 x loose share added):\n" + S.to_string(index=False) +
     "\nReading: the two sets differ mainly by their cell mix (shared types agree, hepatocyte aside); pooled, odds of M are favored and 1 + M is not.\n")
open(os.path.join(H, "m_or_m_plus_1_summary.txt"), "w").write(t); print(t)
