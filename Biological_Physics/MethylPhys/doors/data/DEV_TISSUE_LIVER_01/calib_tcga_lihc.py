"""DEV-TISSUE-LIVER-01: TCGA-LIHC 450K arrays through the chain's Stage 1 (stage_1_idat_calibration.calibrate_idat_to_beta), unchanged.
Grn/Red files are paired by sample barcode (tcga_lihc_450k_files.csv; GDC names each file by its own UUID). One parquet per sample.
Normal tissue only, unless the seal is committed (fetch_tcga_lihc.py enforces the same for downloads).
Usage: python3 calib_tcga.py WORKDIR normal|tumour"""
import os, sys, subprocess, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(MP, "chain")); from stage_1_idat_calibration import calibrate_idat_to_beta
W, which = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(W, "betas"), exist_ok=True)
T = pd.read_csv(os.path.join(HERE, "tcga_lihc_450k_files.csv"))
if which == "tumour":
    seal = os.path.join(HERE, "sealed_liver_tissue.json"); rel = os.path.relpath(seal, subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip())
    assert os.path.exists(seal) and subprocess.run(["git", "-C", os.path.dirname(SEAL if "SEAL" in globals() else seal), "ls-files", "--error-unmatch", os.path.basename(rel)], capture_output=True).returncode == 0, "tumour arrays are read only after the seal is committed"
    import json; ids = {f["file_id"] for q in json.load(open(seal))["pairs"] for f in q["tumour_files"].values()}; T = T[T.file_id.isin(ids)]   # only the sealed pairs
else: T = T[T.sample_type == "Solid Tissue Normal"]
for smp, d in T.groupby("sample"):
    out = os.path.join(W, "betas", smp + ".parquet")
    if os.path.exists(out): continue
    loc = {("Grn" if "_Grn" in f else "Red"): os.path.join(W, "idat", f) for f in d.file_name}
    if len(loc) != 2 or not all(os.path.exists(v) for v in loc.values()): print(smp, "missing IDAT", flush=True); continue
    o = calibrate_idat_to_beta(loc["Grn"], loc["Red"], verbose=False); b = o[0] if isinstance(o, tuple) else o
    pd.DataFrame({"beta": b}).to_parquet(out); print(smp, "probes", int(pd.Series(b).notna().sum()), flush=True)
