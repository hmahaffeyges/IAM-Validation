"""DEV-METAA-ATLAS-CURVE-01, B and T cells: each atlas column's median Met-A on the development arrays (bt_02/bt_curve_dev_summary.txt) beside
what IAM-Atlas v2.1 built the column from (DEV_ATLAS_V21_01/final_samples_v21.csv): samples, sources, sequencing share. Added after the
development read, disclosed; nothing scored. Usage: python3 bt_columns.py OUT.txt"""
import sys, os, re, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); A = pd.read_csv(os.path.join(H, "..", "DEV_ATLAS_V21_01", "final_samples_v21.csv"))
L = ["atlas column | development arrays | median Met-A | Normal | atlas samples | sources | sequencing samples | sequencing share"]
for l in open(os.path.join(H, "bt_02", "bt_curve_dev_summary.txt")):
    m = re.match(r"atlas column (.+?): (\d+) of \d+ read, median ([0-9.]+), Normal (\d+)", l)
    if not m: continue
    c = m.group(1); x = A[A.cell == c]; s = int((x.platform != "array").sum())
    L.append(f"{c} | {m.group(2)} | {m.group(3)} | {m.group(4)} | {len(x)} | {x.source.nunique()} | {s} | {s / max(len(x), 1):.3f}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
