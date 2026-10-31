"""Healthy pass 3, site by site (written and committed before any of pass 3 is read; measured, not scored). From each healthy input's
per-site table of Met-A for the whole sample (NAME_whole_sites.csv.gz, written by the one command with --dev-whole-sites), on the
marked side (expected level at least 0.5) and the unmarked side: the reading's ratio there (weighted measured over weighted expected
disorder, as Met-A for the whole sample is formed), and the sample's level offset (its mean level minus the expected level's mean). Across
samples: the ratio on each side against the offset and against depth (Spearman). Usage: python3 pass3_sites.py LANDED_DIR OUT.txt"""
import sys, os, re, glob, pandas as pd, numpy as np
from scipy.stats import spearmanr
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "*_whole_sites.csv.gz"), recursive=True)):
    g = re.search(r"GSM\d+", os.path.basename(f)).group(0); D = pd.read_csv(f)
    for side, s in (("marked", D.expected_level >= 0.5), ("unmarked", D.expected_level < 0.5)):
        x = D[s]; rows.append(dict(gsm=g, side=side, sites=len(x), depth=float(x.reads.median()), ratio=float((x.weight * x.H_measured).sum() / (x.weight * x.H_expected).sum()),
                                   offset=float(x.sample_level.mean() - x.expected_level.mean())))
T = pd.DataFrame(rows); T.to_csv(os.path.join(os.path.dirname(sys.argv[2]) or ".", "pass3_sites.csv"), index=False)
L = [f"Healthy pass 3 site by site: {T.gsm.nunique()} samples with a per-site table", "side | samples | median ratio | median offset | ratio vs offset rho | ratio vs depth rho"]
for side, g in T.groupby("side"):
    L.append(f"{side} | {len(g)} | {g.ratio.median():.4f} | {g.offset.median():+.5f} | {spearmanr(g.ratio, g.offset).correlation:+.3f} | {spearmanr(g.ratio, g.depth).correlation:+.3f}")
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
