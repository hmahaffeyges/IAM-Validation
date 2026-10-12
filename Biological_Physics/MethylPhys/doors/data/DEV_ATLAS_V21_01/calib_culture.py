"""DEV-ATLAS-V21-01 culture check: the GSE197512 baseline arrays through the chain's Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta),
unchanged. One parquet per array (GSM.parquet). Usage: python3 calib_culture.py IDAT_DIR OUT_DIR"""
import os, sys, glob, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "chain"))
from stage_1_idat_calibration import calibrate_idat_to_beta
IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
for g in sorted(glob.glob(os.path.join(IN, "*_Grn.idat"))):
    gsm = os.path.basename(g).split("_")[0]; out = os.path.join(OUT, f"{gsm}.parquet")
    if os.path.exists(out): continue
    o = calibrate_idat_to_beta(g, g.replace("_Grn.idat", "_Red.idat"), verbose=False); b = o[0] if isinstance(o, tuple) else o
    pd.DataFrame({"beta": b}).to_parquet(out); print(gsm, "probes", int(pd.Series(b).notna().sum()), flush=True)
