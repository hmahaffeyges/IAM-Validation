"""DEV-CEILING-02, step 4: the edges the breach line must sit between, from the step-3 gauge readings (author, 2026-10-13: not a
number all cancers share, a number established tumors do not go below - where past Normal does it warrant checking?).

# NO CHAIN: counting and percentiles over ceiling_gauge_04_rows.csv, which the chain's Stage 1 betas and its binary entropy produced.

Reported, no bar (this is not where the breach is SET - that needs lesions caught before cancer with known outcome, DEV-BREACH-01):
per cancer, the share of established tumors above 1.05, 1.07 and 1.10; the lowest tumors; the highest adjacent normals; the same for
the per-site share of sites at the full surface; and how many low-reading tumors already carry surface sites beyond every normal.
"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); T = pd.read_csv(os.path.join(HERE, "ceiling_gauge_04_rows.csv"))
L = ["breach edges from 219 tumor / adjacent-normal pairs on the cell gauge (development use of spent sets)", ""]
for c in ("liver", "breast", "colorectal", "lung"):
    n = T[(T.cancer == c) & (T.kind == "normal")]; t = T[(T.cancer == c) & (T.kind == "tumor")]
    L.append(f"{c}: tumors above 1.05 / 1.07 / 1.10 = {(t.A > 1.05).mean():.0%} / {(t.A > 1.07).mean():.0%} / {(t.A > 1.10).mean():.0%}; "
             f"tumor min {t.A.min():.3f}, 5th pct {t.A.quantile(.05):.3f}; adjacent normal 95th pct {n.A.quantile(.95):.3f}, max {n.A.max():.3f}; "
             f"tumors with a surface share above every adjacent normal: {(t.at_surface > n.at_surface.max()).mean():.0%}")
n = T[T.kind == "normal"]; t = T[T.kind == "tumor"]; low = t[t.A <= 1.10]
k = sum(r.at_surface > T[(T.cancer == r.cancer) & (T.kind == "normal")].at_surface.max() for r in low.itertuples())
L += ["", f"all four: adjacent normal 95th pct {n.A.quantile(.95):.3f}, max {n.A.max():.3f}; tumor 5th pct {t.A.quantile(.05):.3f}, min {t.A.min():.3f}",
      f"tumors at or below A 1.10: {len(low)} of {len(t)}; of these, already carrying a surface share above every normal in their cancer: {k}"]
open(os.path.join(HERE, "breach_edges_05_output.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
