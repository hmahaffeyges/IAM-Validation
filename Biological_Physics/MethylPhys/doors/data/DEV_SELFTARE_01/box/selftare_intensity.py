#!/usr/bin/env python3
"""DEV-SELFTARE-01 step 2 (box): noob methylated / unmethylated intensities at the subset sites, from the same methylprep call the chain's
Stage 1 makes (run_pipeline betas=True export=True save_control=True poobah=True; one-row sample sheet; probes with poobah p > 0.05 masked).
One output per array: inten/<gsm>.parquet (columns M, U, beta, poobah)."""
import os, sys, json, glob, tempfile, shutil, argparse
from concurrent.futures import ProcessPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("--chain"); ap.add_argument("--jobs"); ap.add_argument("--out"); ap.add_argument("--sites")
ap.add_argument("--workers", type=int, default=48); a = ap.parse_args()
sys.path.insert(0, a.chain)
import pandas as pd, numpy as np

def one(row):
    gsm = row["gsm"]; of = os.path.join(a.out, "inten", f"{gsm}.parquet")
    if os.path.exists(of): return gsm, "cached"
    try:
        import methylprep, stage_1_idat_calibration as S1
        sites = pd.read_csv(a.sites)["site"].astype(str)
        wd = tempfile.mkdtemp(prefix="selftare_"); bc = os.path.basename(row["grn"]).split("_")[1]; pos = "R01C01"
        S1._stage_idat(row["grn"], os.path.join(wd, f"{bc}_{pos}_Grn.idat")); S1._stage_idat(row["red"], os.path.join(wd, f"{bc}_{pos}_Red.idat"))
        open(os.path.join(wd, "samplesheet.csv"), "w").write(f"Sample_Name,Sentrix_ID,Sentrix_Position\n{bc},{bc},{pos}\n")
        methylprep.run_pipeline(wd, array_type="epic", betas=True, export=True, save_control=True, poobah=True,
                                sample_sheet_filepath=os.path.join(wd, "samplesheet.csv"))
        df = pd.read_csv(glob.glob(os.path.join(wd, "**", "*_processed.csv"), recursive=True)[0], index_col=0)
        df.index = df.index.astype(str)
        pc = [c for c in df.columns if "poobah" in c.lower()][0]
        mc = [c for c in df.columns if c.lower() in ("noob_meth", "meth")][0]; uc = [c for c in df.columns if c.lower() in ("noob_unmeth", "unmeth")][0]
        out = pd.DataFrame({"M": df[mc], "U": df[uc], "beta": df["beta_value"], "poobah": df[pc]}).reindex(sites).astype("float32")
        out.loc[out["poobah"] > 0.05, ["M", "U", "beta"]] = np.nan          # the chain's detection mask
        out.to_parquet(of); shutil.rmtree(wd, ignore_errors=True); return gsm, "ok"
    except Exception as e:
        return gsm, "error " + repr(e)[:300]

if __name__ == "__main__":
    os.makedirs(os.path.join(a.out, "inten"), exist_ok=True)
    jobs = pd.read_csv(a.jobs).to_dict("records")
    print(one(jobs[0]), flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for g, s in ex.map(one, jobs[1:]): print(g, s, flush=True)
    sites = pd.read_csv(a.sites)["site"].astype(str); cols = {}
    for r in jobs:
        f = os.path.join(a.out, "inten", f"{r['gsm']}.parquet")
        if os.path.exists(f):
            d = pd.read_parquet(f); cols[(r["gsm"], "M")] = d["M"]; cols[(r["gsm"], "U")] = d["U"]
    W = pd.DataFrame({f"{g}|{k}": v for (g, k), v in cols.items()}); W.index.name = "site"; W.to_parquet("intensity_subset.parquet")
    print(W.shape)
