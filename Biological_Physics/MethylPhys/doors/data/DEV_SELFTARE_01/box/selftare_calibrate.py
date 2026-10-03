#!/usr/bin/env python3
"""DEV-SELFTARE-01, step 1 (box): chain v3 Stage 1 + untared Met-A for every array, and the betas the self-tare needs.

Per array (one output file per array, spot-safe):
  betas/<gsm>.parquet  : full Stage-1 beta (chain calibrate_idat_to_beta, noob, poobah-masked), float32
  rec/<gsm>.json       : conductor_v3.run_neutrophil untared outputs (A, N, f_neu, ...), no references
Nothing here is new physics; it is the chain's own Stage 1 and Stage M/T with ref_A=None.
"""
import sys, os, json, glob, gzip, argparse, traceback
from concurrent.futures import ProcessPoolExecutor
ap = argparse.ArgumentParser(); ap.add_argument("--chain"); ap.add_argument("--jobs", default="jobs.csv"); ap.add_argument("--out")
ap.add_argument("--workers", type=int, default=48); a = ap.parse_args()
sys.path.insert(0, a.chain)
import pandas as pd, numpy as np

def one(row):
    gsm, grn, red, spec = row["gsm"], row["grn"], row["red"], row["specimen"]
    rf = os.path.join(a.out, "rec", f"{gsm}.json")
    if os.path.exists(rf): return gsm, "cached"
    try:
        import stage_1_idat_calibration as S1, conductor_v3 as C
        b, meta = S1.calibrate_idat_to_beta(grn, red, verbose=False, return_mask=True)
        b = b.iloc[:, 0] if hasattr(b, "columns") else b
        b = b.dropna(); b.index = b.index.astype(str)
        pd.DataFrame({"beta": b.astype("float32")}).to_parquet(os.path.join(a.out, "betas", f"{gsm}.parquet"))
        rec = {"gsm": gsm, "series": row["series"], "specimen": spec, "group": row.get("group"), "n_cpg": int(len(b))}
        if spec != "none":
            o = C.run_neutrophil(b, specimen=spec, ref_A=None, array_type="EPIC_v1", sample_id=gsm)
            m = o.get("met_a", {})
            rec.update(refusal=o.get("refusal"), A=m.get("A"), N=m.get("noise_index"), f_neu=m.get("fraction"),
                       n_sites=m.get("n_sites"), state=m.get("state"), reason=m.get("reason"))
        json.dump(rec, open(rf, "w"))
        return gsm, "ok"
    except Exception as e:
        json.dump({"gsm": gsm, "error": repr(e), "tb": traceback.format_exc()[-1500:]}, open(rf + ".err", "w"))
        return gsm, "error " + repr(e)[:200]

if __name__ == "__main__":
    os.makedirs(os.path.join(a.out, "betas"), exist_ok=True); os.makedirs(os.path.join(a.out, "rec"), exist_ok=True)
    jobs = pd.read_csv(a.jobs).to_dict("records")
    print(one(jobs[0]), flush=True)          # serial first: methylprep manifest fetch before the pool
    with ProcessPoolExecutor(a.workers) as ex:
        for g, s in ex.map(one, jobs[1:]): print(g, s, flush=True)
