"""The IAM Sequencing Atlas's sky sphere: how wide each cell's 95 % range is, by number of donors, against the sphere's color scale (0 to
0.15, IAM-Atlas v2.1's, kept so the two atlases read alike). From the committed columns (run_sample.py --build seq-sphere-cols).
Usage: python3 sphere_widths.py OUT.csv OUT.txt"""
import os, sys, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); SC = os.path.join(H, "..", "..", "..", "..", "chain", "Runtime Matrices", "Seq_Atlas", "sphere_cols")
K = pd.read_csv(os.path.join(H, "seq_atlas_inputs.csv")); K = K.groupby(K.cell.str.strip()).size()
rows = []
for k in (1, 2, 3):
    lo_t, hi_t = (pq.read_table(os.path.join(SC, f"{t}_part{k}.parquet")) for t in ("mu_lo", "mu_hi"))
    for c in lo_t.column_names:
        lo = lo_t.column(c).to_numpy().astype(float); hi = hi_t.column(c).to_numpy().astype(float); ok = (lo != 65535) & (hi != 65535); hw = (hi[ok] - lo[ok]) / 2e4
        rows.append(dict(cell=c, donors=int(K.get(c, 1)), half_width_median=round(float(np.median(hw)), 4), share_past_scale=round(float((hw > 0.15).mean()), 3)))
D = pd.DataFrame(rows).sort_values(["donors", "cell"]); D.to_csv(sys.argv[1], index=False); g = D.groupby("donors")
L = ["IAM Sequencing Atlas sky sphere: half-width of each cell's 95 % range (median over the cells in each group)"] + \
    [f"{k} donor(s): {len(x)} cells, half-width {x.half_width_median.median():.4f}, share past the 0.15 scale {x.share_past_scale.median():.3f}" for k, x in g]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
