#!/usr/bin/env python3
# INSTRUMENT-TEST: measures the ADOPTED Stage 2d on the full 732-array GSE87571 calibration (held-out false-positive
# rate per foreign cell). Calls the chain's own stage functions in the chain's own order; produces detection
# statistics only - no tier, no report. Run ALONE: never alongside another procedure.
"""PLAN item 6 / register row B-12. The 2026-09-26 attempt wrote only at the end and was killed after five hours with
nothing to show. This one writes ONE JSON SHARD PER ARRAY (results/heldout2d/<gsm>.json), skips shards that exist,
and combines once at the end - the rule from the calibration loss of 2026-09-25.

Per array: raw Stage 1 betas -> pipeline map -> deconvolve (class fractions) -> stage_b_identity (composition guard)
-> stage_2d_foreign_detection(lab=GSE87571). The panel's 12 arrays are not identified by accession in
detection_panel_v1.json, so the held-out rate is reported over all 732 and the in-sample 12 are noted as a
<= 1.6 % contamination of the denominator.
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd

W = os.getcwd(); CH = os.path.join(W, "iamrepo/Biological_Physics/MethylPhys/chain"); sys.path.insert(0, CH)
import cpg_conductor as C  # noqa: E402
OUT = os.path.join(W, "results/heldout2d"); os.makedirs(OUT, exist_ok=True)
ATLAS = os.path.join(W, "atlas_work/IAMAtlasREBUILD.csv"); LAB = "GSE87571"; PIPE = "stage1_noob_450K"


def main():
    pf = pd.read_parquet(os.path.join(W, "stage1_betas_GSE87571_FULL.parquet")); pf.index = pf.index.map(str)
    gsms = list(pf.columns); print(f"{len(gsms)} arrays", flush=True)
    dec_mod = C._load_module("legacy_iam_deconvolver", C._find("legacy_iam_deconvolver.py"))
    dec = dec_mod.legacyIAMDeconvolver(ATLAS, celltype_class_map=str(C._find("IAMAtlasREBUILD_celltype_to_class.json")), verbose=False)
    t0 = time.time(); done = 0
    for k, gsm in enumerate(gsms):
        shard = os.path.join(OUT, gsm + ".json")
        if os.path.exists(os.path.join(OUT, "STOP")): print("STOP file seen - halting cleanly at", k, flush=True); break
        if os.path.exists(shard): continue
        raw = pf[gsm].dropna().to_dict()
        mapped, scale_label = C.stage_1s_scale_map(raw, PIPE)
        r = dec.deconvolve(mapped); cf = dict(r.class_fractions); ctf = dict(r.celltype_fractions)
        bi = C.stage_b_identity(mapped, {"class_fractions": cf, "celltype_fractions": ctf}, 60, scale_label, lab_zero=None)
        fd = C.stage_2d_foreign_detection(mapped, {"class_fractions": cf, "celltype_fractions": ctf}, LAB, bi=bi)
        rec = {"gsm": gsm, "status": fd.get("status"), "detected": fd.get("detected"),
               "cells": {c: {kk: v.get(kk) for kk in ("f_hat", "sigma", "line", "detected")} for c, v in (fd.get("cells") or {}).items()},
               "composition_verified": (bi.get("immune") or {}).get("composition_verified"),
               "foreign_fraction": (bi.get("immune") or {}).get("foreign_fraction")}
        tmp = shard + ".tmp"; json.dump(rec, open(tmp, "w")); os.replace(tmp, shard); done += 1
        if done % 20 == 0: print(f"  {k+1}/{len(gsms)}  {(time.time()-t0)/done:.0f} s/array", flush=True)
    # ---- combine
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    n = len(recs); st = {}
    for r in recs: st[r["status"].split(":")[0] if r["status"] else "None"] = st.get(r["status"].split(":")[0] if r["status"] else "None", 0) + 1
    ok = [r for r in recs if r["status"] == "OK" or (r["status"] or "").startswith("OK_BUT")]
    cells = sorted({c for r in ok for c in (r["cells"] or {})})
    fp = {c: sum(1 for r in ok if (r["cells"].get(c) or {}).get("detected")) for c in cells}
    unspec = sum(1 for r in recs if (r["status"] or "").startswith("OK_BUT_UNSPECIFIC"))
    summary = {"n_arrays": n, "status_counts": st, "n_scored": len(ok), "unspecific": unspec,
               "false_positives_per_cell": {c: {"n": fp[c], "rate": fp[c] / max(len(ok), 1)} for c in cells},
               "line_rule": "per-laboratory 1 - 1/n quantile on a 12-array panel -> expected rate ~ 1/12 = 0.083 per cell if the null holds",
               "in_sample_note": "the 12 panel arrays are among the 732 (not identified by accession); denominator contamination <= 1.6 %"}
    json.dump({"summary": summary, "arrays": recs}, open(os.path.join(W, "handoff/detection_heldout.json"), "w"), indent=1)
    print(f"\n{n} arrays | status {st} | unspecific {unspec}")
    for c in cells: print(f"  {c:<24} FP {fp[c]:4d} / {len(ok)}  = {fp[c]/max(len(ok),1):.3f}")
    print("wrote handoff/detection_heldout.json")


if __name__ == "__main__":
    main()
