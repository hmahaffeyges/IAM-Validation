"""Why own-cell Met-A reads low on outside laboratories' sequenced cells (diagnostic on the chain's own per-site tables; nothing scored).

The chain's per-site tables (NAME_metaa_sites.csv.gz, written by the one command with --dev-metaa-sites) show, at the identity sites of the
sample's own cell, the sample's level more extreme than the IAM Sequencing Atlas's level of that cell on both sides (higher at marked sites,
lower at unmarked ones). Two causes predict different things:
  (a) a share x of other cells' DNA in the atlas donors' sorts (or less of it in the outside sort): the reference is pulled toward the other
      cells, so the gap (sample level minus reference level) grows with the contrast (reference level minus the other cells' mean level at
      that site), with one x for both sides: gap = x / (1 - x) * contrast;
  (b) a laboratory difference tied to the level itself: within a narrow band of reference level the gap does not follow the contrast.
For each library and both sides, within bands of reference level, the slope of the gap on the contrast, and the x it implies. The other cells'
mean is taken over the atlas's other 56 cells at IAM-Atlas v2.1's sites (stage_seq_atlas.array_columns: mu, levels x 10,000).
Usage: python3 own_cell_gap.py SITES_DIR CELLS.csv OUT.csv OUT.txt
  CELLS.csv: gsm, atlas_cell (the sample's own cell), group; the per-site tables are SITES_DIR/<gsm>_metaa_sites.csv.gz
"""
import os, sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
H = os.path.dirname(os.path.abspath(__file__)); MP = os.path.join(H, *[".."] * 5)
AS = os.path.join(MP, "chain", "Runtime Matrices", "Seq_Atlas", "array_sites")
ids = pq.read_table(os.path.join(MP, "atlas", "v2.1", "sites.parquet")).column("cpg").to_pylist()
MU = pd.concat([pq.read_table(os.path.join(AS, f"mu_part{k}.parquet")).to_pandas() for k in (1, 2, 3)], axis=1).astype("float64")
MU = MU.where(MU < 65535) / 10000.0; MU.index = ids
C = pd.read_csv(sys.argv[2]); rows = []
BANDS = [(0, .1), (.7, .9), (.9, .95), (.95, 1.0001)]
for r in C.itertuples():
    f = os.path.join(sys.argv[1], f"{r.gsm}_metaa_sites.csv.gz")
    if not os.path.exists(f): continue
    T = pd.read_csv(f).set_index("site"); T = T[T.index.isin(MU.index)]
    other = MU.loc[T.index, [c for c in MU.columns if c != r.atlas_cell]].mean(axis=1)
    T["gap"] = T.sample_level - T.reference_level; T["contrast"] = T.reference_level - other
    for lo, hi in BANDS:
        x = T[(T.reference_level >= lo) & (T.reference_level < hi) & T.contrast.notna() & (T.reads >= 3)]
        if len(x) < 50: continue
        w = x.reads; b = np.polyfit(x.contrast, x.gap, 1, w=np.sqrt(w))
        rho = pd.Series(x.gap).corr(pd.Series(x.contrast), method="spearman")
        rows.append(dict(gsm=r.gsm, group=r.group, cell=r.atlas_cell, band=f"{lo}-{min(hi, 1)}", sites=len(x), mean_gap=round(float(np.average(x.gap, weights=w)), 4),
                         mean_contrast=round(float(x.contrast.mean()), 4), slope=round(float(b[0]), 4), x_implied=round(float(b[0] / (1 + b[0])), 4), rho=round(float(rho), 3)))
D = pd.DataFrame(rows); D.to_csv(sys.argv[3], index=False)
L = [f"own-cell gap at the identity sites: {D.gsm.nunique()} libraries (diagnostic, nothing scored)",
     "by band of reference level (medians over libraries): mean gap, slope of gap on contrast, implied other-cell share x, Spearman rho"]
for b_, g in D.groupby("band", sort=False):
    L.append(f"  {b_}: gap {g.mean_gap.median():+.4f}, contrast {g.mean_contrast.median():+.4f}, slope {g.slope.median():+.4f}, x {g.x_implied.median():+.4f}, rho {g.rho.median():+.3f} ({len(g)} libraries)")
L.append("(a) predicts a positive slope in every band with one x on both sides; (b) predicts slopes near 0 within bands")
open(sys.argv[4], "w").write("\n".join(L) + "\n"); print("\n".join(L))
