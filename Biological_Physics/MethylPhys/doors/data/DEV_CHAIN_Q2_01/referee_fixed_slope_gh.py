#!/usr/bin/env python3
"""PHY-084 after the result (exploration on spent counts; the sealed verdict NOT MET is unchanged; register A5/A6/A7/A8, 2026-11-15).
The sealed scorer read each cell type's intercept from its OWN Theil-Sen slope; the 11-13 rehearsal traced the B-cell low largely to that
own-slope step. Here the slope is the derived one (A7, 0.034625 per unit loose share, PHY-050) and each library gives its own intercept,
held_err_length - 0.034625 * loose_share; the type's reading is their median (2,000 library resamples for its spread). Both are set against
the level from M alone through the counter (no eraser share; mean of referee_out.csv's two ratio predictions divided back by each ratio) and
against the sealed middle. Runs on a GitHub machine: collects each library's *_count.json from its PHY-084 run artifact (erase-<GSM>,
2026-11-13) here, never on the owner's laptop. Writes xout/referee_fixed_slope_summary.txt and xout/referee_fixed_slope.csv (a few KB).
Usage (explore.yml): referee_fixed_slope_gh.py"""
import os, io, json, zipfile, requests, numpy as np, pandas as pd
from scipy.stats import theilslopes
H = {"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"}; R = "https://api.github.com/repos/" + os.environ["GITHUB_REPOSITORY"]
D = os.path.dirname(os.path.abspath(__file__)); SLOPE = 0.034625; SEED = 20261115; NB = 2000
S = pd.read_csv(os.path.join(D, "referee_samples.csv")); S["cell"] = S.cell.str.lower().str.strip()
REF = set(pd.read_csv(os.path.join(D, "referee_refused_intake.csv")).gsm)
P = pd.read_csv(os.path.join(D, "referee_out.csv")); P["cell"] = P.cell.str.lower().str.strip()
P["m_alone"] = (P.pred_ratio_0864 * 0.8642 + P.pred_ratio_0905 * 0.905) / 2
rows = []
for g in sorted(set(S.gsm) - REF):
    A = [a for a in requests.get(f"{R}/actions/artifacts", headers=H, params={"name": f"erase-{g}", "per_page": 30}, timeout=60).json().get("artifacts", [])
         if a["created_at"][:10] == "2026-11-13" and not a["expired"]]
    if not A: print(f"::warning::{g}: no PHY-084 artifact"); continue
    z = zipfile.ZipFile(io.BytesIO(requests.get(max(A, key=lambda a: a["size_in_bytes"])["archive_download_url"], headers=H, timeout=600).content))
    n = [m for m in z.namelist() if m.endswith("_count.json")]
    if not n: print(f"::warning::{g}: no count file in its artifact"); continue
    c = json.loads(z.read(n[0])); rows.append(dict(gsm=g, obs=c["held_err_length"], loose=c["loose_share"]))
X = pd.DataFrame(rows).merge(S[["gsm", "cell"]], on="gsm"); rng = np.random.default_rng(SEED); out = []
for cell, g in X.groupby("cell"):
    ic = (g.obs - SLOPE * g.loose).values; fixed = float(np.median(ic))
    bb = np.array([np.median(ic[rng.integers(0, len(ic), len(ic))]) for _ in range(NB)])
    s, b, _, _ = theilslopes(g.obs, g.loose); p = P[P.cell == cell].iloc[0]
    out.append(dict(cell=cell, n=len(g), own_slope=round(s, 4), own_intercept=round(b, 5), fixed_intercept=round(fixed, 5), fixed_se=round(float(bb.std(ddof=1)), 5),
                    m_alone=round(p.m_alone, 5), sealed_mid=round(p.pred_mid, 5), own_over_m=round(b / p.m_alone, 3), fixed_over_m=round(fixed / p.m_alone, 3)))
O = pd.DataFrame(out); os.makedirs("xout", exist_ok=True); O.to_csv("xout/referee_fixed_slope.csv", index=False)
t = (f"PHY-084 rescored with the derived slope {SLOPE} (exploration, after the fact; {len(X)} libraries):\n" + O.to_string(index=False) + "\n"
     f"median fixed/M-alone {O.fixed_over_m.median():.3f} (own-slope {O.own_over_m.median():.3f}); spread max/min fixed {O.fixed_over_m.max() / O.fixed_over_m.min():.3f} "
     f"(own-slope {O.own_over_m.max() / O.own_over_m.min():.3f})\n")
open("xout/referee_fixed_slope_summary.txt", "w").write(t); print(t)
for l in t.strip().splitlines(): print("::notice::" + l[:300])
