"""DEV-FINGERPRINT-02 arrays: ENCODE HAIB 450K/EPIC IDATs (encode_arrays.csv: file, experiment, cell, channel, platform, md5, url; one array per
experiment) through the chain's Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta), unchanged. md5 checked. One parquet per experiment.
Usage: python3 calib_arrays.py WORKDIR normal|cancer"""
import os, sys, gzip, hashlib, requests, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(MP, "chain")); from stage_1_idat_calibration import calibrate_idat_to_beta
W, SET = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(W, "idat"), exist_ok=True); os.makedirs(os.path.join(W, "betas"), exist_ok=True)
NORMAL = ["epithelial cell of prostate", "hepatocyte", "epithelial cell of alveolus of lung", "bronchial epithelial cell", "mammary epithelial cell", "MCF 10A"]
A = pd.read_csv(os.path.join(HERE, "encode_arrays.csv")); A = A[A.cell.isin(NORMAL) == (SET == "normal")]
for exp, d in A.groupby("experiment"):
    out = os.path.join(W, "betas", exp + ".parquet")
    if os.path.exists(out): continue
    loc = {}
    for _, r in d.iterrows():
        ch = "Grn" if "green" in r.channel else "Red"; f = os.path.join(W, "idat", f"{r.file}_{ch}.idat")
        if not os.path.exists(f):
            b = requests.get(r.url, timeout=300).content
            assert hashlib.md5(b).hexdigest() == r.md5, f"md5 {r.file}"
            open(f, "wb").write(gzip.decompress(b) if b[:2] == b"\x1f\x8b" else b)
        loc[ch] = f
    o = calibrate_idat_to_beta(loc["Grn"], loc["Red"], verbose=False); b = o[0] if isinstance(o, tuple) else o
    pd.DataFrame({"beta": b}).to_parquet(out); print(exp, d.cell.iloc[0], d.platform.iloc[0][:28], "probes", int(pd.Series(b).notna().sum()), flush=True)
