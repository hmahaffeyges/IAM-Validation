"""DEV-IAMA-COMMISSION-01 synthetic test 1 summary: from split_floor_01_output.csv, per depth share the largest |IAM-A ratio - 1| between halves
(three seeds) and molecules per half. Writes split_floor_01_summary.txt. Usage: python3 split_floor_01_summary.py"""
import os, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = pd.read_csv(os.path.join(HERE, "split_floor_01_output.csv")); R["d"] = (R.ratio - 1).abs()
out = [f"share {s}: molecules per half {int(g.molecules_per_half.iloc[0]):,}; largest |ratio - 1| {g.d.max():.4f}" for s, g in R.groupby("share")]
open(os.path.join(HERE, "split_floor_01_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
