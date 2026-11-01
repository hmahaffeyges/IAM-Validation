"""Pass 4's outside neurons, the same measure as the depth diagnostic (metaa_depth_diag.py; added after scoring, disclosed; nothing scored).
From the chain's own per-site tables of sorted-cell Met-A (NAME_metaa_sites.csv.gz, written by the one command with --dev-metaa-sites) for
the outside samples: each sample's disorder per read at the identity sites at a fixed 10 reads (sites with at least 10 reads, thinned
without replacement), over the mean of the atlas's own neuron donors (metaa_depth_diag_donors.csv); against its Met-A
(selfree4_readings.csv) and its neuron share (selfree4_purity.csv). Usage: python3 selfree4_perread.py SITE_TABLE_DIR OUT.txt"""
import os, re, sys, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr, hypergeom
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(H, "..", "..", "..", "..", "chain"))); import stage_metaa_atlas as SA
j = np.arange(11); Hj = np.asarray(SA._H(pd.Series(j / 10)), "float64")
A = pd.read_csv(os.path.join(H, "selfree4_readings.csv")).set_index("gsm").A; own = pd.read_csv(os.path.join(H, "selfree4_purity.csv")).set_index("gsm").own
D = pd.read_csv(os.path.join(H, "metaa_depth_diag_donors.csv"), index_col=0); ref = D[D.cell == "cortical neurons"].h10; rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_metaa_sites.csv.gz"))):
    g = re.search(r"GSM\d+", os.path.basename(f)).group(0); t = pd.read_csv(f); t = t[t.reads >= 10]
    k = np.rint(t.sample_level * t.reads).astype(int).to_numpy(); n = t.reads.astype(int).to_numpy()
    rows.append(dict(gsm=g, h10=float((hypergeom.pmf(j[None, :], n[:, None], k[:, None], 10) @ Hj).mean()), depth=float(t.reads.median()), A=A.get(g), own=own.get(g)))
Q = pd.DataFrame(rows).dropna(subset=["A"]); Q["ratio"] = Q.h10 / ref.mean(); Q.to_csv(os.path.join(H, "selfree4_perread.csv"), index=False)
L = [f"Outside neurons, disorder per read at 10 reads: {len(Q)} samples, median {Q.h10.median():.4f}; the atlas's {len(ref)} neuron donors {ref.min():.4f} to {ref.max():.4f}, mean {ref.mean():.4f}",
     f"Met-A against the ratio (sample over the donors' mean): rho {spearmanr(Q.ratio, Q.A).correlation:+.3f}; medians {Q.A.median():.4f} and {Q.ratio.median():.4f}",
     f"the ratio against the neuron share: rho {spearmanr(Q.own, Q.ratio).correlation:+.3f}; samples inside the donors' range: {int(((Q.h10 >= ref.min()) & (Q.h10 <= ref.max())).sum())} of {len(Q)}"]
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
