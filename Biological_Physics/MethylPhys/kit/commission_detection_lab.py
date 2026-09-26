#!/usr/bin/env python3
"""Commission a laboratory for Stage 2d foreign-cell detection: write its centre, weights and line into
detection_panel_v1.json from its OWN healthy whole-blood panel. This is the detection analogue of the laboratory zero.

Usage:
    python3 commission_detection_lab.py --lab GSE51032 --betas panel.parquet --pipeline GSE51032_450K [--min-n 36]

  panel.parquet   columns = arrays, index = CpG ids; healthy whole blood from this laboratory only
  --pipeline      the scale map the chain applies to this laboratory's betas (Stage 1s); the panel is mapped first
  --min-n         refuse below this (default 36: PROC-MF-02 B5 showed no gain past it on 450K)

What it does, in order - and each is a lesson from PROC-MF-01/02/03:
  1. maps the panel through the chain's own Stage 1s (the deconvolver and the gauge read mapped betas)
  2. runs the composition guard on every panel array and DROPS any it does not verify as blood-like, printing which -
     a control reading 0.2 on every solid-tissue column set the line in PROC-MF-03; it must never reach a line
  3. weights = 1/variance of the blood-only residual per marker, from THIS laboratory's panel (MF-03: weights from
     other laboratories were 6-25x less sharp on a fifth)
  4. centre = median raw amplitude per foreign cell on the panel
  5. line   = quantile min(1 - 1/n, 0.99) of the centred amplitude - a stated quantile, never the maximum
  6. sigma  = 1.4826 x MAD of the centred amplitudes on the panel
  7. records n, the arrays used, the arrays dropped, the date and the pipeline beside the entry
The chain's Stage 2d reads this file and nothing else; a laboratory not in it reports 'detection not commissioned'.
Produces no reading for any specimen.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(os.path.dirname(HERE), "chain")
sys.path.insert(0, CH)
import cpg_conductor as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", required=True); ap.add_argument("--betas", required=True); ap.add_argument("--pipeline", required=True)
    ap.add_argument("--min-n", type=int, default=36); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    P_path = str(C._find("detection_panel_v1.json")); P = json.load(open(P_path))
    M = P["markers"]; bc = P["_meta"]["blood_columns"]; fc = P["_meta"]["foreign_columns"]
    Ab = np.array([P["blood_ref"][c] for c in bc]).T; F = {c: np.array(P["foreign_ref"][c]) for c in fc}
    df = pd.read_parquet(a.betas) if a.betas.endswith(".parquet") else pd.read_csv(a.betas, index_col=0)
    df.index = df.index.map(str)
    print(f"panel: {df.shape[1]} arrays from {a.lab}; mapping through {a.pipeline}", flush=True)
    V, used, dropped = [], [], []
    for g in df.columns:
        mapped, _ = C.stage_1s_scale_map(df[g].dropna().to_dict(), a.pipeline)
        v = np.array([mapped.get(m, np.nan) for m in M], dtype=float)
        if np.isnan(v).mean() > 0.2:
            dropped.append((g, "fewer than 80 % of panel markers present")); continue
        # the composition guard: the chain's own verdict on whether this is blood
        try:
            aout = C.stage_a_cells(df[g].dropna().to_dict(), str(C._find("IAMAtlasREBUILD.csv")), {"pipeline": a.pipeline})
            bi = C.stage_b_identity(mapped, aout, 60, a.pipeline, lab_zero=None)
            ok = ((bi.get("immune") or {}).get("composition_verified") is not False)
        except Exception as e:
            ok = True; print(f"   guard could not run on {g}: {type(e).__name__} - array kept", flush=True)
        if not ok:
            dropped.append((g, "composition guard: not blood-like (foreign fraction %s)" % (bi["immune"].get("foreign_fraction")))); continue
        V.append(np.where(np.isnan(v), np.nanmedian(v), v)); used.append(g)
    n = len(used)
    print(f"usable arrays: {n} | dropped: {len(dropped)}", flush=True)
    for g, why in dropped: print(f"   dropped {g}: {why}")
    if n < a.min_n:
        print(f"REFUSED: {n} < {a.min_n} healthy arrays - a line from fewer is not a commissioning"); sys.exit(2)
    V = np.array(V)
    R = np.array([v - Ab @ nnls(Ab, v)[0] for v in V]); w = 1.0 / (R.var(axis=0, ddof=1) + 1e-8)
    def amp(v, c):
        fb, _ = nnls(Ab, v); b = Ab @ fb; r = v - b; t = F[c] - b / max(fb.sum(), 1e-9)
        return float(np.sum(w * t * r) / np.sum(w * t * t))
    q = min(1 - 1.0 / n, 0.99)
    entry = {"n_panel": n, "arrays": used, "dropped": dropped, "pipeline": a.pipeline,
             "commissioned": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
             "weights": np.round(w, 3).tolist(), "line_quantile": round(q, 4), "cells": {}}
    for c in fc:
        raw = np.array([amp(v, c) for v in V]); centre = float(np.median(raw)); cen = raw - centre
        entry["cells"][c] = {"centre": round(centre, 6), "line": round(float(np.quantile(cen, q)), 6), "line_quantile": round(q, 4),
                             "sigma": round(float(1.4826 * np.median(np.abs(cen))), 6),
                             "measured_detection_limit": P["_meta"].get("measured_limit_cells", {}).get(c, "not measured")}
    for c in ("Breast", "Colon_epithelial_cells", "Cortical_neurons", "Prostate"):
        if c in entry["cells"]: print(f"   {c:<24} centre {entry['cells'][c]['centre']:+.4f}  line {entry['cells'][c]['line']:.4f}  sigma {entry['cells'][c]['sigma']:.4f}")
    if a.dry_run:
        print("dry run - panel not written"); return
    P["laboratories"][a.lab] = entry
    json.dump(P, open(P_path, "w"), indent=1)
    print(f"wrote {a.lab} into {os.path.basename(P_path)} ({len(P['laboratories'])} laboratories commissioned)")


if __name__ == "__main__":
    main()
