"""DEV-TISSUE-LUNG-01: GSE94785 IDATs through chain Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta), unchanged; one parquet per
GSM. Tumour arrays are refused by fetch_lung_01.lock until the seal. Usage: python3 calib_lung_01.py IDAT_DIR OUT_DIR"""
import os, sys, glob, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "chain"))
from stage_1_idat_calibration import calibrate_idat_to_beta
import fetch_lung_01 as F
idat, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
for g in sorted({os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(idat, "*_Grn.idat*"))}):
    F.lock(g); p = os.path.join(out, g + ".parquet")
    if os.path.exists(p): continue
    grn = glob.glob(os.path.join(idat, g + "_*_Grn.idat*"))[0]; red = grn.replace("_Grn.", "_Red.")
    o = calibrate_idat_to_beta(grn, red, verbose=False); b = o[0] if isinstance(o, tuple) else o
    pd.DataFrame({"beta": b}).to_parquet(p); print(g, int(pd.Series(b).notna().sum()), flush=True)
