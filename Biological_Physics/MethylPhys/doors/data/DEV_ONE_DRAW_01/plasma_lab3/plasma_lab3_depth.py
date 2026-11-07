"""DEV-ONE-DRAW-01, plasma from a new laboratory (GSE186007), development after the second reading (not met; disclosed): does the healthy
plasmas' whole-sample Met-A scatter follow each plasma's depth? Reads only the bundles the one command wrote (LAND) and the committed readings.
Usage: python3 plasma_lab3_depth.py LAND READINGS_CSV OUT_TXT"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
land, rc, out = sys.argv[1:4]; rows = []
for f in glob.glob(os.path.join(land, "**", "*_bundle.json"), recursive=True):
    B = json.load(open(f)); v = B.get("sequencing_values") or {}; m = B.get("met_a_whole") or {}
    rows.append(dict(gsm=os.path.basename(f).split("_")[0], calls_median=v.get("calls_per_site_median"), sites=v.get("sites_with_value"), sites_used=m.get("sites"), n_eff=m.get("n_eff")))
    sh = {x["cell"]: x["share"] for x in ((B.get("cell_shares") or {}).get("cells") or [])}; rows[-1].update({"share|" + c: s for c, s in sh.items()})
    rows[-1]["raw_sum"] = (B.get("cell_shares") or {}).get("raw_sum")
D = pd.DataFrame(rows).merge(pd.read_csv(rc), on="gsm"); H = D[D.group == "HEA"].copy()
L = [f"healthy plasmas with a bundle: {len(H)}"]
for c in ("calls_median", "sites", "sites_used"):
    x = pd.to_numeric(H[c], errors="coerce"); k = x.notna() & H["Met-A whole sample, tared"].notna()
    if k.sum() >= 5:
        r = spearmanr(x[k], H.loc[k, "Met-A whole sample, tared"]); r2 = spearmanr(x[k], (H.loc[k, "Met-A whole sample, tared"] - 1).abs())
        L.append(f"{c}: median {x[k].median():.0f} (range {x[k].min():.0f} to {x[k].max():.0f}); Spearman with tared whole-sample Met-A {r.statistic:+.3f} (p {r.pvalue:.2g}); with its distance from 1 {r2.statistic:+.3f} (p {r2.pvalue:.2g})")
# 2026-11-06, version 2 (after depth was ruled out, disclosed): the Deconvolver's shares, each cell with a median share of at least 1 %
sc = [c for c in H.columns if c.startswith("share|") and pd.to_numeric(H[c], errors="coerce").fillna(0).median() >= 0.01] + ["raw_sum"]
R_ = []
for c in sc:
    x = pd.to_numeric(H[c], errors="coerce").fillna(0); r = spearmanr(x, H["Met-A whole sample, tared"])
    R_.append((abs(r.statistic), f"{c.replace('share|', 'share of ')}: median {x.median():.3f}; Spearman with tared whole-sample Met-A {r.statistic:+.3f} (p {r.pvalue:.2g})"))
L.append("shares (cells with a median share of at least 1 %, strongest first):"); L += ["  " + s for _, s in sorted(R_, reverse=True)]
L.append("rows (gsm, calls median, sites with a value, tared whole-sample Met-A):")
for t in H.sort_values("sites").itertuples(): L.append(f"  {t.gsm} {t.calls_median} {t.sites} {getattr(t, '_12', np.nan) if False else round(float(H.loc[t.Index, 'Met-A whole sample, tared']), 4)}")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L[:30]))
