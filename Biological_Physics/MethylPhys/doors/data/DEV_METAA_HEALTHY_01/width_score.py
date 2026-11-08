"""Each cell's healthy width on arrays, from IAM-Atlas v2.1's own build donors (written and committed 2026-11-07 before any of its readings
exists; the fourth step to a working chain). The author: Normal is measured, not chosen; 1 is the middle. Each build donor's array read by the
one command against IAM-Atlas v2.1 (series run atlas_donors, --full-chain --cell-panel atlas). The cell a donor array is grouped under is the
cell the IAM-Deconvolver found most of in it (cells_found; never the label). For each such cell: the number of donor arrays, the median of
its own Met-A (does 1 sit in the middle), and the width: the standard deviation of its Met-A across those arrays, times sqrt(n / (n - 1)),
because each donor helped make the reference it is read against (exact for a reference that is the donors' mean; IAM-Atlas v2.1's levels are
a model's estimate, so the correction is stated, and the held-out sets check it when instruments are commissioned). A cell with fewer than 3
donor arrays gets no width ('not measured'). Also recorded: each array's main share, so step 3's share rule can be applied when it exists.
Usage: python3 width_score.py LAND OUT.csv OUT.txt"""
import sys, os, glob, json, numpy as np, pandas as pd
land, oc, ot = sys.argv[1:4]; rows = []
for f in sorted(glob.glob(os.path.join(land, "**", "GSM*.json"), recursive=True)):
    if f.endswith("_values.json"): continue
    B = json.load(open(f)); cf = (B.get("cells_found") or {}).get("cells") or []
    if not cf: rows.append(dict(gsm=os.path.basename(f)[:-5], cell=None, reason=str((B.get("cells_found") or {}).get("reason") or "no cells_found (older chain)"))); continue
    top = max(cf, key=lambda x: x.get("share") or 0)
    rows.append(dict(gsm=os.path.basename(f)[:-5], cell=top["cell"], share=top.get("share"), met_a=top.get("met_a"), note=top.get("met_a_note")))
X = pd.DataFrame(rows, columns=["gsm", "cell", "share", "met_a", "note", "reason"]).drop_duplicates("gsm"); X.to_csv(oc, index=False)
L = [f"IAM-Atlas v2.1's build donors read against the atlas: {len(X)} arrays; grouped by the cell the Deconvolver found most of"]
W = []
for c, g in X.dropna(subset=["met_a"]).groupby("cell"):
    n = len(g); med = g.met_a.median()
    w = float(g.met_a.std(ddof=1) * np.sqrt(n / (n - 1))) if n >= 3 else None
    W.append(dict(cell=c, n=n, median=round(med, 4), width=None if w is None else round(w, 4), main_share_median=round(g.share.median(), 3)))
    L.append(f"  {c[:34]:34} arrays {n:3} | median Met-A {med:.4f} | width {('%.4f' % w) if w is not None else 'not measured (fewer than 3)'} | main share median {g.share.median():.3f}")
L.append(f"arrays with no Met-A for their main cell: {int(X.met_a.isna().sum())}")
pd.DataFrame(W).to_csv(oc.replace(".csv", "_widths.csv"), index=False)
open(ot, "w").write("\n".join(L) + "\n"); print("\n".join(L[:6]))
