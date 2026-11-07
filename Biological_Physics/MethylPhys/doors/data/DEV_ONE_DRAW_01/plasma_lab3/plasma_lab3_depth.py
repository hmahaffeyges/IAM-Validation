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
    # 2026-11-06, version 3 (after depth and the shares were ruled out, disclosed): the library and coverage fields the bundle records
    D_ = B.get("development") or {}; lf = D_.get("library_floor") or {}; ce = m.get("ceiling") or {}; gc = v.get("genome_coverage") or {}; fo = D_.get("foreign") or {}
    rows[-1].update({"floor unmarked rate": (lf.get("unmarked") or {}).get("rate"), "floor marked rate": (lf.get("marked") or {}).get("rate"), "floor rate": lf.get("rate"),
                     "methylated sites mean level": ce.get("methylated_sites_mean_beta"), "turnover share": ce.get("turnover_share"), "foreign f": fo.get("f"),
                     "coverage min ratio": gc.get("min_ratio"), "coverage median share": gc.get("median_share"), "effective sites": m.get("effective_sites"),
                     "se over sites": m.get("se_over_sites"), "the rest A": ((m.get("by_place") or {}).get("the rest") or {}).get("A"),
                     "chrX share": (gc.get("by_chrom") or {}).get("chrX")})
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
V3 = ("floor unmarked rate", "floor marked rate", "floor rate", "methylated sites mean level", "turnover share", "foreign f", "coverage min ratio",
      "coverage median share", "effective sites", "se over sites", "the rest A", "chrX share", "age")
R3 = []
for c in V3:
    if c not in H: continue
    x = pd.to_numeric(H[c], errors="coerce"); k = x.notna()
    if k.sum() >= 8 and x[k].nunique() > 2:
        r = spearmanr(x[k], H.loc[k, "Met-A whole sample, tared"]); R3.append((abs(r.statistic), f"{c}: median {x[k].median():.4g} (range {x[k].min():.4g} to {x[k].max():.4g}); Spearman {r.statistic:+.3f} (p {r.pvalue:.2g}, n {int(k.sum())})"))
L.append("library and coverage fields (version 3, strongest first):"); L += ["  " + s for _, s in sorted(R3, reverse=True)]
L.append("rows (gsm, calls median, sites with a value, tared whole-sample Met-A):")
for t in H.sort_values("sites").itertuples(): L.append(f"  {t.gsm} {t.calls_median} {t.sites} {getattr(t, '_12', np.nan) if False else round(float(H.loc[t.Index, 'Met-A whole sample, tared']), 4)}")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L[:30]))
