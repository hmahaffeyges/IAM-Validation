"""DEV-SELFTARE-03: the anchor range on real arrays, scored (exploration, not sealed). Each array's R (healthy and planted) tared against the median
healthy R of the other arrays on its slide (at least 3), R_rel; the loss read is (1 - R_rel) / f_neu, the per-array limit 2 x the healthy spread
of R_rel / f_neu. Usage: python3 anchor_range_real_01_score.py ROWS_DIR. Writes anchor_range_real_01_summary.txt here."""
import os, sys, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(sys.argv[1], "anchor_range_real_01_rows.csv")); T = T[T.planted_pct >= 0].copy(); T["slide"] = T.slide.astype(str)
h = T[T.planted_pct == 0]; rel = []
for _, x in T.iterrows():
    ref = h[(h.slide == x.slide) & (h.gsm != x.gsm)].R
    rel.append(x.R / ref.median() if len(ref) >= 3 else np.nan)
T["R_rel"] = rel; T["loss_read_pct"] = 100 * (1 - T.R_rel) / T.f_neu
hh = T[T.planted_pct == 0].dropna(subset=["R_rel"]); sd = hh.R_rel.std(ddof=1); fm = hh.f_neu.median()
L = [f"arrays read {T.gsm.nunique()}, tared on their slide {hh.gsm.nunique()}; healthy R_rel spread {sd:.4f}; median neutrophil share {fm:.3f}",
     f"healthy: loss read {hh.loss_read_pct.mean():+.2f} % (SD {hh.loss_read_pct.std():.2f}); per-array limit (2 x spread / share) {100 * 2 * sd / fm:.1f} %"]
base = hh.set_index("gsm")
for p in (2.0, 5.0):
    g = T[T.planted_pct == p].dropna(subset=["R_rel"]).set_index("gsm"); dl = (g.loss_read_pct - base.loss_read_pct).dropna(); dA = (g.A - base.A).dropna()
    se = dl.std(ddof=1) / np.sqrt(len(dl)); found = int((g.loss_read_pct > 100 * 2 * sd / fm).sum())
    L.append(f"planted {p:.0f} % at every site ({len(dl)} arrays): change in the loss read {dl.mean():+.2f} % (SE {se:.2f}); arrays above the per-array limit {found} of {len(g)}; "
             f"group of {len(dl)}: {'found' if g.loss_read_pct.mean() - base.loss_read_pct.reindex(g.index).mean() > 2 * hh.loss_read_pct.std() / np.sqrt(len(g)) else 'not found'}; Met-A change {dA.mean():+.4f}")
open(os.path.join(H, "anchor_range_real_01_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
