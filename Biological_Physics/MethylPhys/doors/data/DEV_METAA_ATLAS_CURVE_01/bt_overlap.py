"""DEV-METAA-ATLAS-CURVE-01, B and T cells: which listed arrays are IAM-Atlas v2.1's own inputs (added after the development read, disclosed).
Usage: python3 bt_overlap.py OUT.txt"""
import sys, os, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); A = pd.read_csv(os.path.join(H, "..", "DEV_ATLAS_V21_01", "final_samples_v21.csv"))
ag = set(A["sample"].astype(str).str.split("_").str[0]); L = ["list | arrays | IAM-Atlas v2.1 inputs among them | series that are atlas sources"]
for f in ("bt_development_samples.csv", "bt_confirmation_samples.csv", "bt_samelab_samples.csv"):
    S = pd.read_csv(os.path.join(H, f)); g = S.gsm.astype(str)
    L.append(f"{f} | {len(S)} | {int(g.isin(ag).sum())} | {', '.join(sorted(set(S.gse) & set(A.source))) or 'none'}")
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
