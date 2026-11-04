"""IAM-A on plasma with the held error's length bias removed (development; written and committed before any reading; nothing scored).
The instrument's job: a healthy plasma reads Normal on IAM-A. Synthetic test cfDNA 3 met: Stage Q2's held_err_length does not depend on how
long a sample's molecules are. Inputs: the one command's bundles of healthy pass 8 (the atlas's 140 inputs, pass 7's list and settings,
counted again so each carries held_err_length) and of plasma_seq_05 (the 30 tuning plasmas counted again from plasma_seq_02's saved
molecules, plasma_seq_02's settings). Two lines across the healthy cells, built as stage_q2_selfref.build_line builds the chain's line (per
cell, the median loose share and the opportunity-weighted held error; a Theil-Sen line): on the raw held error, and on held_err_length.
Each sample reads A = its held error over what its own loose share predicts on the line of the same kind. Expected if length is the
cause: the plasmas read near the healthy inputs on the corrected line, and as low as before (about 0.77) on the raw line.
Usage: python3 iama_length_line.py LAND SAMPLES_HEALTHY.csv OUT.csv OUT.txt"""
import os, sys, glob, json, pandas as pd
from scipy.stats import theilslopes
C = dict(pd.read_csv(sys.argv[2])[["gsm", "cell"]].astype(str).apply(lambda c: c.str.strip()).values); R = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "*_bundle.json"), recursive=True)):
    c = ((json.load(open(f)).get("iam_a") or {}).get("count") or {}); g = os.path.basename(f).replace("_bundle.json", "")
    if c.get("held_err") is None: continue
    R.append(dict(sample=g, kind="healthy input" if g in C else "plasma", cell=C.get(g, ""), loose_share=c.get("loose_share"), held_err=c.get("held_err"),
                  held_err_length=c.get("held_err_length"), held_opp=c.get("held_opp")))
D = pd.DataFrame(R); h = D[D.kind == "healthy input"].dropna(subset=["loose_share", "held_err", "held_err_length"])
def line(col):
    g = h.groupby("cell"); eps = pd.Series({k: (x[col] * x.held_opp).sum() / x.held_opp.sum() for k, x in g})
    s, i, _, _ = theilslopes(eps.values, g.loose_share.median().reindex(eps.index).values); return i, s, len(eps)
LN = {k: line(k) for k in ("held_err", "held_err_length")}
for k, (i, s, _) in LN.items(): D["A_" + k] = D[k] / (i + s * D.loose_share)
D.to_csv(sys.argv[3], index=False)
L = [f"line on {k}: firm {v[0]:.5f}, slope {v[1]:.5f}, {v[2]} cells" for k, v in LN.items()]
for k, G in D.groupby("kind"):
    L.append(f"{k} ({len(G)}): raw line median {G.A_held_err.median():.4f} (5 to 95 % {G.A_held_err.quantile(.05):.4f} to {G.A_held_err.quantile(.95):.4f}); "
             f"corrected line median {G.A_held_err_length.median():.4f} (5 to 95 % {G.A_held_err_length.quantile(.05):.4f} to {G.A_held_err_length.quantile(.95):.4f}); "
             f"held error {G.held_err.median():.5f}, corrected {G.held_err_length.median():.5f}, loose share {G.loose_share.median():.4f}")
open(sys.argv[4], "w").write("\n".join(L) + "\n"); print("\n".join(L))
