"""C-score test (a), second laboratory (betas are written as parquet WITH their probe names): GSE184269 granulocytes and whole bloods through the chain unchanged, scored against the
bar pre-registered in CSCORE_COMMISSIONING_PLAN.md before any array was fetched (pooled 34 arrays, at most 1 above 1.152).

Stage 1 from the IDATs (chain/stage_1_idat_calibration.py), then the chain's C-score with the same-run tare: references are the
other arrays of the same specimen type in this series. Refusals are reported and excluded from the proportion, with the count.
Usage: python3 score_band_test.py IDAT_DIR OUT_DIR
"""
import json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.abspath(os.path.join(HERE, "..", "..", "..", "chain"))
sys.path.insert(0, CH)
from stage_1_idat_calibration import calibrate_idat_to_beta
import conductor_v3 as C

idat, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
S = pd.read_csv(os.path.join(HERE, "band_test_GSE184269_samples.csv"))
files = os.listdir(idat)
rows = []
for r in S.itertuples():
    pair = {}
    for f in files:
        if f.startswith(r.gsm + "_"):
            if re.search(r"(?i)_grn", f): pair["Grn"] = os.path.join(idat, f)
            elif re.search(r"(?i)_red", f): pair["Red"] = os.path.join(idat, f)
    if len(pair) != 2: rows.append(dict(gsm=r.gsm, specimen=r.src, refusal="missing IDAT")); continue
    o = calibrate_idat_to_beta(pair["Grn"], pair["Red"], verbose=False)
    b = o[0] if isinstance(o, tuple) else o
    # the probe names ARE the data: the chain reads sites by name, and a bare value array is unusable
    # (first run of this script saved .npy and every array came back unread, 2026-10-13)
    pd.DataFrame({"beta": np.asarray(b, dtype=np.float32)}, index=pd.Index(b.index.astype(str), name="probe")).to_parquet(
        os.path.join(out, r.gsm + "_beta.parquet"))
    rows.append(dict(gsm=r.gsm, specimen=r.src, probes=int(pd.Series(b).notna().sum())))
    print(r.gsm, rows[-1].get("probes"), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(out, "calibrated.csv"), index=False)
print("calibrated", sum(1 for x in rows if "probes" in x), "of", len(S))
