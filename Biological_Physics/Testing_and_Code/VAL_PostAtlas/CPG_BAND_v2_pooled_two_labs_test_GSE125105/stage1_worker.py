import sys, os, json, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1"))
from stage_1_idat_calibration import calibrate_idat_to_beta
jobs = json.load(open(sys.argv[1])); out = sys.argv[2]; B = {}; fails = []
for gsm, g, r in jobs:
    try: b, m = calibrate_idat_to_beta(g, r, verbose=False); B[gsm] = b.astype("float32")
    except Exception as e: fails.append((gsm, str(e)[:120]))
pd.DataFrame(B).to_pickle(out); json.dump(fails, open(out + ".fails.json", "w"))
