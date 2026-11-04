"""IAM-A on plasma, the molecule-level loose share (development; diagnostic, written and committed before it runs on the plasmas; nothing
scored). The instrument's job: a healthy plasma reads Normal on IAM-A. Synthetic test cfDNA 2 showed that mixing cells raises the tile loose
share but leaves the share of loose molecules (mol_low_share, recorded by Stage Q2) unchanged, so a line on the molecule-level share should
not be pulled down by plasma's mix.
Healthy reference: healthy_count_records.csv, the count fields of the IAM Sequencing Atlas's 153 inputs, copied from their one-command count
records (run_sample.py --dev-count-only). Two lines, built the way stage_q2_selfref.build_line builds the chain's line (per cell: the median
share and the opportunity-weighted held error; a Theil-Sen line across cells): on the tile loose share, and on the molecule-level share.
Each sample reads A = held error / what its own share predicts on the line. Printed: the healthy inputs on each line, and every plasma
bundle under BUNDLE_DIR (iam_a.count) on each line.
Usage: python3 iama_mol_lines.py BUNDLE_DIR OUT.csv OUT.txt"""
import os, sys, glob, json, pandas as pd
from scipy.stats import theilslopes
H = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(H, "healthy_count_records.csv"))
def line(col):
    g = T.dropna(subset=[col, "held_err"]).groupby("cell")
    C = pd.DataFrame({"L": g[col].median(), "eps": pd.Series({c: (x.held_err * x.held_opp).sum() / x.held_opp.sum() for c, x in g})})
    s, i, _, _ = theilslopes(C.eps.values, C.L.values); return i, s, len(C)
LN = {c: line(c) for c in ("loose_share", "mol_low_share")}
def A(rec, col): i, s, _ = LN[col]; return rec["held_err"] / (i + s * rec[col]) if rec.get(col) is not None and rec.get("held_err") is not None else None
R = [dict(sample=r["gsm"], kind="healthy input", cell=r["cell"], A_tile=A(r, "loose_share"), A_molecule=A(r, "mol_low_share"), loose_share=r["loose_share"], mol_low_share=r["mol_low_share"]) for r in T.to_dict("records")]
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "*_bundle.json"), recursive=True)):
    c = ((json.load(open(f)).get("iam_a") or {}).get("count") or {})
    if c.get("held_err") is None: continue
    R.append(dict(sample=os.path.basename(f).replace("_bundle.json", ""), kind="plasma", cell="", A_tile=A(c, "loose_share"), A_molecule=A(c, "mol_low_share"), loose_share=c.get("loose_share"), mol_low_share=c.get("mol_low_share")))
D = pd.DataFrame(R); D.to_csv(sys.argv[2], index=False)
L = [f"line on the {k}: firm {v[0]:.5f}, slope {v[1]:.5f}, {v[2]} cells" for k, v in LN.items()]
for k, G in D.groupby("kind"):
    L.append(f"{k} ({len(G)}): IAM-A on the tile line median {G.A_tile.median():.4f} (5 to 95 % {G.A_tile.quantile(.05):.4f} to {G.A_tile.quantile(.95):.4f}); "
             f"on the molecule line median {G.A_molecule.median():.4f} (5 to 95 % {G.A_molecule.quantile(.05):.4f} to {G.A_molecule.quantile(.95):.4f}); "
             f"tile share {G.loose_share.median():.4f}, molecule share {G.mol_low_share.median():.4f}")
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
