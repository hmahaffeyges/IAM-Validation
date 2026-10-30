"""Met-A against the IAM Sequencing Atlas, site by site (diagnostic, written and committed before it runs; nothing scored). 20 of the 137
healthy readings (metaa_sites_samples.csv: every 7th by depth, spent) read again by the one command with the full chain and
--dev-metaa-sites, which writes each identity site's reads, sample level, reference level and measured and expected disorder. Reported: the
ratio of summed measured to summed expected disorder (Met-A's own form) by side of the site (reference level at most 0.5 or above), by the
site's reads (fifths) and by the sample's depth (thirds), so the extra disorder is placed.
Usage: python3 metaa_sites_score.py LANDED_DIR OUT.txt"""
import sys, os, glob, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); S = pd.read_csv(os.path.join(H, "metaa_sites_samples.csv")); parts = []
for r in S.itertuples():
    f = sorted(glob.glob(os.path.join(sys.argv[1], "**", r.gsm + "_metaa_sites.csv.gz"), recursive=True))
    if f: d = pd.read_csv(f[-1]); d["gsm"] = r.gsm; d["depth"] = r.depth; parts.append(d)
X = pd.concat(parts, ignore_index=True); X["side"] = (X.reference_level > 0.5).map({True: "marked", False: "unmarked"})
X["reads_fifth"] = pd.qcut(X.reads.rank(method="first"), 5, labels=False); X["depth_third"] = pd.qcut(X.depth.rank(method="dense"), 3, labels=False)
rat = lambda g: g.H_measured.sum() / g.H_expected.sum()
L = [f"Met-A site by site: {X.gsm.nunique()} of {len(S)} samples, {len(X):,} site readings; overall ratio {rat(X):.4f}", "by side:"]
for k, g in X.groupby("side"): L.append(f"  {k}: {len(g):,} sites, ratio {rat(g):.4f}, median reads {g.reads.median():.0f}, median sample level {g.sample_level.median():.4f}, median reference {g.reference_level.median():.4f}")
L.append("by side and the sample's depth (third 0 shallowest):")
for (k, t), g in X.groupby(["side", "depth_third"]): L.append(f"  {k}, depth third {t}: ratio {rat(g):.4f} over {g.gsm.nunique()} samples")
L.append("by the site's reads (fifths, 0 fewest):")
for k, g in X.groupby("reads_fifth"): L.append(f"  fifth {k} ({g.reads.min()} to {g.reads.max()} reads): ratio {rat(g):.4f}")
L.append("by reference level (unmarked side):")
u = X[X.side == "unmarked"]
for k, g in u.groupby(pd.cut(u.reference_level, [-0.001, 0.0, 0.02, 0.05, 0.1, 0.25, 0.5])): L.append(f"  {k}: {len(g):,} sites, ratio {rat(g):.4f}")
X.groupby("gsm").apply(lambda g: pd.Series(dict(depth=g.depth.iloc[0], ratio=rat(g), unmarked=rat(g[g.side == 'unmarked']), marked=rat(g[g.side == 'marked'])))).to_csv(os.path.join(sys.argv[1], "metaa_sites_by_sample.csv"))
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
