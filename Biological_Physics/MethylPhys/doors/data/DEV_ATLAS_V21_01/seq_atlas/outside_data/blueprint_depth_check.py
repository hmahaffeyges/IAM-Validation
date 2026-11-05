"""Is BLUEPRINT's low own-cell Met-A the atlas's depth trend? (diagnostic on committed outputs; nothing scored)
Inside the atlas's laboratory, sorted-cell Met-A falls with the library's depth against its reference donors (metaa_depth_diag.csv: slope per read
of depth difference). BLUEPRINT's libraries are deeper. This compares: BLUEPRINT's own-cell Met-A against its depth (calls per site), by cell,
beside what the atlas's trend would give at that depth.
Usage: python3 blueprint_depth_check.py OUT.txt   (run from this folder)"""
import sys, pandas as pd
from scipy.stats import spearmanr, linregress
DG = pd.read_csv("../../../DEV_METAA_ATLAS_CURVE_01/seq_atlas_01/metaa_depth_diag.csv")
T = pd.read_csv("blueprint_healthy_test.csv").merge(pd.read_csv("blueprint_islands_offset.csv")[["gsm", "calls_per_site_median"]], on="gsm")
T = T[T["Met-A own cell"].notna()]
lr = linregress(DG.d_depth, DG.A); g = T.groupby("cell")[["calls_per_site_median", "Met-A own cell"]].median()
L = [f"atlas trend (pass 4, {len(DG)} samples): Met-A = {lr.intercept:.4f} {lr.slope:+.5f} x (depth minus reference depth), r {lr.rvalue:.3f}; depth difference range {DG.d_depth.min():.0f} to {DG.d_depth.max():.0f}",
     f"BLUEPRINT ({len(T)} read): depth median {T.calls_per_site_median.median():.1f} calls per site; own-cell Met-A median {T['Met-A own cell'].median():.4f}; "
     f"Met-A against depth rho {spearmanr(T.calls_per_site_median, T['Met-A own cell']).correlation:+.3f} (the atlas trend is negative)",
     "by cell (median depth, median own-cell Met-A):"] + [f"  {c}: {r_.calls_per_site_median:.0f}, {r_['Met-A own cell']:.3f}" for c, r_ in g.iterrows()]
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
