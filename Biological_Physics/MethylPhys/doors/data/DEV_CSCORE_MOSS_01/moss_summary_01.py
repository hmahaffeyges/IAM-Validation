"""DEV-CSCORE-MOSS-01 summary quoted in the Met-A commissioning note: largest |A(mix) - A(unmixed leukocytes)| over the 9 real mixes.
Usage: python3 moss_summary_01.py   (reads moss_mixes_01_rows.csv; writes moss_summary_01_output.json)"""
import os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = pd.read_csv(os.path.join(HERE, "moss_mixes_01_rows.csv"))
base = float(R[R.mix == "unmixed leukocytes"].A.iloc[0]); M = R[R.mix != "unmixed leukocytes"]
d = (M.A - base).abs(); out = dict(A_unmixed=base, mixes=int(len(M)), largest_shift=round(float(d.max()), 4), largest_shift_mix=str(M.mix.iloc[int(d.values.argmax())]))
json.dump(out, open(os.path.join(HERE, "moss_summary_01_output.json"), "w"), indent=1); print(out)
