"""Scores PHY-050's per-call firm rate (written before any sample was counted; sealed in doors/PHY_050_PERCALL_PREREG.md).
Prediction: the per-call loss rate of a fully held region (the intercept of held_e_call on the loose share, one row per healthy Loyfer
sample, ordinary least squares) equals 1/(1+M), M = dG_ATP / RT from the canon (54 kJ/mol, 310.15 K; 20.9405), that is 0.04558.
Bar 1: the intercept's 95 % interval contains 0.04558.  Bar 2: that interval is narrow enough to decide (half-width at most 0.003);
otherwise the verdict is INCONCLUSIVE.  Reported beside, not scored: the same fit on the counter's held error and on held_err_length; the
median per-call holding energy ln((1-e)/e) by cell type against ln M; the fit on cell-type medians.
Usage: python3 percall_score.py <folder of counts> <summary.txt>"""
import sys, os, json, glob, math
import numpy as np, pandas as pd, statsmodels.api as sm
M = 54000.0 / (8.314462618 * 310.15); PRED = 1 / (1 + M); HALF = 0.003
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "GSM*.json"))):
    o = json.load(open(f)); rows.append(dict(gsm=o["sample"], cell=o["cell"], loose=o["loose_share"], held_err=o["held_err"],
                                            held_err_length=o.get("held_err_length"), e_call=o.get("held_e_call")))
D = pd.DataFrame(rows).dropna(subset=["e_call", "loose"]); L = [f"PHY-050 per-call firm rate: {len(D)} healthy Loyfer samples, {D.cell.nunique()} cell types"]
def fit(col, d=None):
    d = D if d is None else d; r = sm.OLS(d[col], sm.add_constant(d.loose)).fit(); lo, hi = r.conf_int().loc["const"]
    return r.params["const"], lo, hi, r.params["loose"]
a, lo, hi, b = fit("e_call")
L.append(f"prediction 1/(1+M) = {PRED:.5f} (M {M:.4f}); per-call intercept {a:.5f}, 95 % interval {lo:.5f} to {hi:.5f}, slope {b:.4f}")
b1 = lo <= PRED <= hi; b2 = (hi - lo) / 2 <= HALF
L.append(f"bar 1 interval contains the prediction: {'met' if b1 else 'not met'}")
L.append(f"bar 2 half-width {(hi - lo) / 2:.5f} at most {HALF}: {'met' if b2 else 'not met'}")
L.append("verdict: " + ("MET" if (b1 and b2) else "INCONCLUSIVE" if not b2 else "NOT MET"))
for col in ("held_err", "held_err_length"):
    a2, l2, h2, s2 = fit(col); L.append(f"beside, not scored: {col} intercept {a2:.5f} ({l2:.5f} to {h2:.5f}), slope {s2:.4f}")
E = lambda e: math.log((1 - e) / e)
g = D.groupby("cell").agg(e=("e_call", "median"), L=("loose", "median"))
L.append(f"beside: median per-call holding energy over cell types {np.median([E(x) for x in g.e]):.3f} kT against ln M {math.log(M):.3f}")
r = sm.OLS(g.e, sm.add_constant(g.L)).fit(); ci = r.conf_int().loc["const"]
L.append(f"beside: fit on cell-type medians, intercept {r.params['const']:.5f} ({ci[0]:.5f} to {ci[1]:.5f})")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
