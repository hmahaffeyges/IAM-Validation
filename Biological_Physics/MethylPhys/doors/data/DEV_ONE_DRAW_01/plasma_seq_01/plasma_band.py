"""DEV-ONE-DRAW-01 plasma_seq_01: the healthy plasma band on sequencing (written and committed 2026-10-23 before 28 of the 30 plasmas are
read; the first two were read). It only summarizes what run_sample.py wrote (each plasma's bundle). For each reading over the healthy
plasmas: n, median, 25th and 75th percentiles, and the 5th to 95th percentile range; for IAM-A, the held error and the loose share apart, and
the rank correlation of IAM-A with the loose share (does the self-reference line, built on purified cells, carry plasma's loose share?); the
cell shares' medians, largest first. Nothing fitted.  Usage: python3 plasma_band.py FOLDER_OF_BUNDLES [OUT.txt]"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy import stats
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "plasma_band_summary.txt"); rows, sh = [], []
for f in sorted(glob.glob(os.path.join(src, "**", "SRR*_bundle.json"), recursive=True)):
    B = json.load(open(f)); q = B.get("iam_a") or {}; w = B.get("met_a_whole") or {}; sv = B.get("sequencing_values") or {}
    r = dict(run=os.path.basename(f).split("_")[0], **{"IAM-A": q.get("A"), "held error": q.get("held_err"), "loose share": q.get("loose_share"),
             "sites with a value": sv.get("sites_with_value"), "Met-A whole": w.get("A"), "ceiling distance": (w.get("ceiling") or {}).get("distance_to_ceiling"),
             "globe changed blocks": (B.get("globe") or {}).get("blocks_changed"), "sphere territories changed": (B.get("info_sphere") or {}).get("territories_changed")})
    for k, v in (w.get("by_place") or {}).items(): r["Met-A whole, " + k] = (v or {}).get("A")
    for k, v in ((B.get("territories") or {}).get("sets") or {}).items(): r["territory " + k] = v.get("A")
    rows.append(r); sh.append({x["cell"]: x["share"] for x in (B.get("cell_shares") or {}).get("cells") or []})
X = pd.DataFrame(rows).drop_duplicates("run").set_index("run"); S = pd.DataFrame(sh).fillna(0)
L = [f"plasma_seq_01 healthy band: {len(X)} plasmas read by the one command"]
for c in X.columns:
    v = pd.to_numeric(X[c], errors="coerce").dropna()
    if len(v): L.append(f"{c:42} n {len(v):2} | median {v.median():.4g} | 25-75 % {v.quantile(.25):.4g} to {v.quantile(.75):.4g} | 5-95 % {v.quantile(.05):.4g} to {v.quantile(.95):.4g}")
a, l = pd.to_numeric(X["IAM-A"], errors="coerce"), pd.to_numeric(X["loose share"], errors="coerce"); ok = a.notna() & l.notna()
if ok.sum() >= 5: rho, p = stats.spearmanr(a[ok], l[ok]); L.append(f"IAM-A against the loose share: Spearman rho {rho:+.2f} (p {p:.2g}, n {int(ok.sum())})")
L.append("cell shares, median over the plasmas: " + ", ".join(f"{c} {v:.3f}" for c, v in S.median().sort_values(ascending=False).head(10).items()))
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
