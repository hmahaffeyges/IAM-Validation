"""Met-A against the IAM Sequencing Atlas, healthy test: the readings by how many donors the reference half holds (added after the test was
scored, disclosed; diagnostic, nothing scored). Each sample is read against the other sealed donor half of its own cell; a half of one donor
carries that donor's own differences, which counting does not predict. Donors are told apart by the atlas inputs' donor code (the title's
Z-number). Reads the committed readings and inputs only. Usage: python3 metaa_seqatlas_by_half.py"""
import os, re, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "..")
d = pd.read_csv(os.path.join(H, "metaa_seqatlas_readings.csv")); d = d[d.A.notna()].copy()
I = pd.read_csv(os.path.join(D, "DEV_ATLAS_V21_01", "seq_atlas", "seq_atlas_inputs.csv")); I["donor"] = I.title.str.extract(r"(Z0+\w+)$")[0].fillna(I.gsm)
n = I.groupby(["cell", "half"]).donor.nunique()
d["ref_half"] = d.half.map({"A": "B", "B": "A"}); d["ref_donors"] = [n.get((c, h), 0) for c, h in zip(d.cell, d.ref_half)]
d["one"] = d.ref_donors == 1; L = [f"Met-A against the sequencing atlas by the reference half's donors: {len(d)} readings"]
for k, g in d.groupby("ref_donors"): L.append(f"reference half of {k} donor(s): {len(g)} readings, median Met-A {g.A.median():.4f}, median depth {g.depth.median():,.0f}")
a, b = d[d.one].A, d[~d.one].A
L.append(f"one donor against more: median {a.median():.4f} against {b.median():.4f}, Mann-Whitney p {mannwhitneyu(a, b).pvalue:.2g}" if len(a) and len(b) else "one donor against more: not both present")
L.append(f"Met-A against depth, one-donor halves only: rho {spearmanr(d[d.one].A, d[d.one].depth).correlation:+.3f} over {int(d.one.sum())}")
L.append(f"Met-A against depth, halves of two or more: rho {spearmanr(d[~d.one].A, d[~d.one].depth).correlation:+.3f} over {int((~d.one).sum())}")
L.append(f"reference donors against depth: rho {spearmanr(d.ref_donors, d.depth).correlation:+.3f}")
d.to_csv(os.path.join(H, "metaa_seqatlas_by_half.csv"), index=False); open(os.path.join(H, "metaa_seqatlas_by_half.txt"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
