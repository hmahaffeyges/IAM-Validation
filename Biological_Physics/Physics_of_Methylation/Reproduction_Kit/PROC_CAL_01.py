#!/usr/bin/env python3
"""PROC-CAL-01 — Stage 1: raw IDAT pair -> noob-calibrated beta, compared to the project's cached betas.

Environment: the 'methylprep' env (RUNBOOK.md §Environments): methylprep==1.7.1, numpy==1.26.4, pandas==1.5.3
             (methylprep calls DataFrame.append, removed in pandas 2). Set HOME to a writable dir; methylprep
             stores Illumina manifests in $HOME/.methylprep_manifest_files, fetched on first use from
             https://array-manifest-files.s3.amazonaws.com/ (450K: HumanMethylation450k_15017482_v3.csv.gz,
             EPIC: HumanMethylationEPIC_manifest_v2.csv.gz).
Input   : data/idats/<GSM>_Grn.idat.gz + _Red.idat.gz ; data/betas_cache.pkl (Stage-1 output shipped in 10_TEST_DATA.zip)
Expected: bit-identical (r = 1.000000, max|diff| = 0.000000). Observed 2026-09-19: 11/11.
Usage   : python PROC_CAL_01.py [GSM ...]   (default: every pair found under data/idats)
"""
import sys, os, glob, time, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpg_kit as K
sys.path.insert(0, K.ENGINE)
from stage_1_idat_calibration import calibrate_idat_to_beta

def run(gsms):
    cache = pickle.load(open(os.path.join(K.DATA, "betas_cache.pkl"), "rb"))
    rows = []; allpass = True
    for gsm in gsms:
        g = glob.glob(os.path.join(K.DATA, "idats", "**", f"{gsm}_Grn.idat.gz"), recursive=True)
        if not g: rows.append((gsm, "IDAT not found")); allpass = False; continue
        g = g[0]; r = g.replace("_Grn", "_Red")
        t = time.time(); beta, meta = calibrate_idat_to_beta(g, r, verbose=False)
        beta.to_csv(os.path.join(K.ROOT, "results", f"{gsm}_stage1_beta.csv"))
        line = f"{meta['array_type']} {len(beta):,} CpGs {time.time()-t:.0f}s"
        if gsm in cache:
            c = cache[gsm]; c = pd.Series(c) if not isinstance(c, pd.Series) else c
            com = beta.index.intersection(c.index); a = beta.loc[com].astype(float); b = c.loc[com].astype(float); ok = a.notna() & b.notna()
            d = (a[ok] - b[ok]).abs(); rr = np.corrcoef(a[ok], b[ok])[0, 1]
            ok_ = rr > 0.999999 and d.max() < 1e-6; allpass &= ok_
            line += f" | vs cache r={rr:.6f} max|diff|={d.max():.6f} -> {'PASS' if ok_ else 'FAIL'}"
        else: line += " | not in cache (no comparison)"
        rows.append((gsm, line))
    K.report("PROC-CAL-01", rows, "PASS" if allpass else "FAIL", path=os.path.join(K.ROOT, "results", "PROC_CAL_01.json"))

if __name__ == "__main__":
    os.makedirs(os.path.join(K.ROOT, "results"), exist_ok=True)
    gs = sys.argv[1:] or sorted({os.path.basename(p).split("_")[0] for p in glob.glob(os.path.join(K.DATA, "idats", "**", "*_Grn.idat.gz"), recursive=True)})
    run(gs)
