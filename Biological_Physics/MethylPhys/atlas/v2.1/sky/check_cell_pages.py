"""Check of the 79 IAM-Atlas v2.1 cell pages (2026-10-18, author: "visit each one individually and verify the information is correct and
the map is correct"). For every cell: the half-width of the 95 % range recomputed from the compact atlas (mu_lo, mu_hi), the numbers the page
prints (website/art/out/cells/index.json), the drawn map's own pixels (sky_v21_halfwidth_nside256_nest.npz), and the samples behind the cell
(final_samples_v21.csv). Usage: python3 check_cell_pages.py MAPS_DIR OUT.csv"""
import sys, os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); A = os.path.normpath(os.path.join(H, "..")); R0 = os.path.normpath(os.path.join(H, "..", "..", "..", "..", ".."))
part = {n: k for k in (1, 2, 3) for n in pq.read_schema(f"{A}/mu_lo_part{k}.parquet").names}
I = {c["cell"]: c for c in json.load(open(os.path.join(R0, "website/art/out/cells/index.json")))["cells"]}
Z = np.load(os.path.join(sys.argv[1], "sky_v21_halfwidth_nside256_nest.npz")); cells = [str(c) for c in Z["cells"]]
S = pd.read_csv(os.path.join(R0, "Biological_Physics/MethylPhys/doors/data/DEV_ATLAS_V21_01/final_samples_v21.csv"))
rows = []
for c in cells:
    k = part[c]; lo = pq.read_table(f"{A}/mu_lo_part{k}.parquet", columns=[c]).column(c).to_numpy(); hi = pq.read_table(f"{A}/mu_hi_part{k}.parquet", columns=[c]).column(c).to_numpy()
    w = (hi.astype(np.float32) - lo.astype(np.float32)) / 2e4; w[(lo == 65535) | (hi == 65535)] = np.nan; m = Z[f"m{cells.index(c):02d}"].astype(float); s = S[S.cell.str.lower() == c.lower()]
    rows.append(dict(cell=c, samples=len(s), studies=s.source.nunique(), platforms="+".join(sorted(s.platform.astype(str).unique())), hw_median_atlas=round(float(np.nanmedian(w)), 4),
                     hw_p90_atlas=round(float(np.nanpercentile(w, 90)), 4), hw_median_page=I[c]["halfwidth_median"], hw_p90_page=I[c]["halfwidth_p90"], hw_median_map=round(float(np.nanmedian(m)), 4)))
R = pd.DataFrame(rows); R.to_csv(sys.argv[2], index=False)
d = max((R.hw_median_page - R.hw_median_atlas).abs().max(), (R.hw_p90_page - R.hw_p90_atlas).abs().max(), (R.hw_median_map - R.hw_median_atlas).abs().max())
print(f"cells {len(R)} | largest disagreement page/map vs atlas {d:.4f} | rank correlation samples vs median half-width {R[['samples','hw_median_atlas']].corr(method='spearman').iloc[0,1]:.2f}")
