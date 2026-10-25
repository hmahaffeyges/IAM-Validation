"""DEV-METAA-HEALTHY-01: version 4's per-site array-against-sequencing differences written as a table (step3_04/v4_offsets.csv), so version 5
reads the same numbers without the 44 readings (v4_score.offsets on them, unchanged).  Usage: python3 v4_offsets_write.py READINGS_DIR"""
import sys, os, glob, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); import v4_score as V
X = pd.concat([pd.read_csv(f, index_col=0) for f in glob.glob(os.path.join(sys.argv[1], "**", "*_v4.csv"), recursive=True)])
d, D, use, cells = V.offsets(X, V.side); pd.DataFrame({"side": V.side, "d": d}).to_csv(os.path.join(H, "step3_04", "v4_offsets.csv")); print(len(d), "sites;", ", ".join(cells))
