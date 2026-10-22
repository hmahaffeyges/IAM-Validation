"""DEV-METAA-ATLAS-01, after design step 3 (2026-10-21; exploration on the committed readings, nothing new read; not sealed).
Step 3 (sorted_01/sorted_cells_01_summary.txt) read GSE184269's sorted cells against IAM-Atlas v2.1 through the chain: after the series tare
only 37 to 75 % read Normal. Each donor's five cell types sit on one slide (sorted_01/gse184269_sorted_slides.csv, from GEO's IDAT names),
so the question is how much of the spread is the slide (here also the donor). Per cell type: the readings as tared; then with each
reading's slide offset removed, taken from the same slide's OTHER cell types only (never the reading itself). Caution: here a slide is one
donor, so such a correction would also hide a loss shared by every cell of that donor (the self-tare trap, DEV-SELFTARE-03); it is a measure
of the slide's share of the spread, not a proposed reading. Writes slide_offset_02_summary.txt."""
import os, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(H, "sorted_01")
R = pd.read_csv(os.path.join(S, "sorted_cells_01_rows.csv")).merge(pd.read_csv(os.path.join(S, "gse184269_sorted_slides.csv")), on="gsm")
R["z"] = np.log(R.A_rel); R["zp"] = np.log(R.A_rel_planted)
W = R.pivot_table(index="slide", columns="atlas_cell", values="z"); C = W.corr(min_periods=8)
T = R.groupby("slide").z.agg(["sum", "count"]); R = R.join(T, on="slide"); R = R[R["count"] >= 3].copy(); R["off"] = (R["sum"] - R.z) / (R["count"] - 1)
L = ["correlation of tared readings between cell types on the same slide: " + ", ".join(f"{a}/{b} {C.loc[a, b]:.2f}" for i, a in enumerate(C) for b in list(C)[i + 1:]), ""]
for c, g in R.groupby("atlas_cell"):
    a = np.exp(g.z - g.z.median()); b = np.exp((g.z - g.off) - (g.z - g.off).median()); bp = np.exp((g.zp - g.off) - (g.z - g.off).median())
    L.append(f"{c}: n {len(g)} | as tared: spread {a.std():.4f}, Normal {((a >= .95) & (a <= 1.05)).mean():.2f} | the other cells' slide offset removed: spread {b.std():.4f}, "
             f"Normal {((b >= .95) & (b <= 1.05)).mean():.2f}, a planted 1 % loss {(bp - b).mean() / b.std():.1f} x the spread")
open(os.path.join(H, "slide_offset_02_summary.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
