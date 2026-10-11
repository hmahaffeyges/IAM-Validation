"""DEV-TISSUE-COLON-01: GSE131013 IDATs through the chain's Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta), unchanged; one parquet per
GSM. 'normal' = Mucosa and Normal; 'tumour' only after the seal is committed, sealed pairs only. Usage: python3 calib_colon.py WORKDIR normal|tumour"""
import os, sys, json, subprocess, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(MP, "chain")); from stage_1_idat_calibration import calibrate_idat_to_beta
W, which = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(W, "betas"), exist_ok=True); S = pd.read_csv(os.path.join(HERE, "colon_samples.csv"))
if which == "tumour":
    SEAL = os.path.join(HERE, "sealed_colon_tissue.json")
    assert os.path.exists(SEAL) and subprocess.run(["git", "-C", HERE, "ls-files", "--error-unmatch", os.path.basename(SEAL)], capture_output=True).returncode == 0, "tumour arrays are read only after the seal is committed"
    want = {p["tumour_gsm"] for p in json.load(open(SEAL))["pairs"]}
else: want = set(S[S.tissue.isin(["Mucosa", "Normal"])].gsm)
idat = os.listdir(os.path.join(W, "idat"))
for g in sorted(want):
    out = os.path.join(W, "betas", g + ".parquet")
    if os.path.exists(out): continue
    f = {c: next((os.path.join(W, "idat", x) for x in idat if x.startswith(g + "_") and f"_{c}.idat" in x), None) for c in ("Grn", "Red")}
    if None in f.values(): print(g, "missing IDAT", flush=True); continue
    o = calibrate_idat_to_beta(f["Grn"], f["Red"], verbose=False); b = o[0] if isinstance(o, tuple) else o
    pd.DataFrame({"beta": b}).to_parquet(out); print(g, "probes", int(pd.Series(b).notna().sum()), flush=True)
