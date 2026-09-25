#!/usr/bin/env python3
"""PROC-FOREIGN-01 step 1: what foreign material does to the immune reading and its tier.

For every healthy host, for three non-blood classes, at six mixing fractions: score the mixture through the
chain's own path and record the reading, the tier word, and the one detector quantity the chain already
computes - the fraction of the specimen NOT assigned to the blood lineage.

The detector is chosen before any mixture is scored and is not invented here: `stage_4_6_patient_cmb` already
defines BLOOD_LINEAGE = (immune, progenitor, stem_adult) and the composition step already reports every class
fraction, so `foreign = 1 - (immune + progenitor + stem_adult)` is a quantity the chain has today.

Writes handoff/foreign01.json.
"""
import json
import lzma
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
REF = "iamrepo/Biological_Physics/MethylPhys/reference_data"
COHORTS = ["GSE87571", "GSE42861", "GSE111629", "GSE125105"]
FOREIGN = ["stromal", "secretory", "terminal"]
FRACS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35]
BLOOD = ("immune", "progenitor", "stem_adult")
sys.path.insert(0, CH)
import cpg_conductor as C
import cpg_tiers as T

ATLAS = "atlas_work/IAMAtlasREBUILD.csv"


def main():
    band = json.load(open(C._find("identity_band_v3.json")))
    coh = band["_meta"]["cohorts"]
    if isinstance(coh, str):
        import ast
        coh = ast.literal_eval(coh)
    zeros = {k.split("_")[0]: v.get("z_lab_full_cohort") for k, v in coh.items()}
    prior = {r["gsm"]: r for r in json.load(open("handoff/band01_arrays.json"))["arrays"]}

    # the foreign material: the atlas's own class means, read once
    head = pd.read_csv(ATLAS, nrows=0).columns.tolist()
    idcol = head[0]
    # the atlas stores a posterior mean, SD and interval per class: the mean column is "<class>_mean"
    want = [c + "_mean" for c in FOREIGN]
    missing = [c for c in want if c not in head]
    assert not missing, "atlas has no mean column for %s" % missing
    mu = pd.read_csv(ATLAS, usecols=[idcol] + want).set_index(idcol)
    mu.columns = FOREIGN
    print("foreign means: %s over %d loci" % (list(mu.columns), len(mu)), flush=True)

    dec_mod = C._load_module("walther_iam_deconvolver", C._find("walther_iam_deconvolver.py"))
    dec = dec_mod.WaltherIAMDeconvolver(ATLAS, celltype_class_map=str(
        C._find("IAMAtlasREBUILD_celltype_to_class.json")))

    def score(beta, age, gse):
        beta_rm, lab_scale = C.stage_1s_scale_map(beta, "stage1_noob_450K")
        fr = dict(dec.deconvolve(beta).class_fractions)
        bi = C.stage_b_identity(beta_rm, {"class_fractions": fr}, age, lab_scale, lab_zero=zeros[gse])
        imm = bi.get("immune", {})
        a = imm.get("A_abs")
        rep = imm.get("present", True)
        tier = None
        if a is not None:
            try:
                tier = T.tier_of(a, rep, imm.get("H_min"))
            except Exception:
                tier = None
        foreign = 1.0 - sum(fr.get(c, 0.0) for c in BLOOD)
        return {"A": a, "tier": (tier if isinstance(tier, str) else (tier or {}).get("tier")
                                 if isinstance(tier, dict) else tier),
                "foreign": round(float(foreign), 5),
                "fr": {k: round(float(v), 5) for k, v in fr.items() if v > 0.001}}

    rows = []
    if os.path.exists("handoff/foreign01.json"):
        rows = json.load(open("handoff/foreign01.json"))["rows"]
        print("resuming:", len({r["gsm"] for r in rows}), "hosts done", flush=True)
    done = {(r["gsm"], r["foreign_class"], r["f"]) for r in rows}

    for gse in COHORTS:
        with lzma.open(os.path.join(REF, "stage1_betas_%s.pkl.xz" % gse), "rb") as f:
            df = pickle.load(f)
        t0, n = time.time(), 0
        for gsm in df.columns:
            p = prior.get(gsm)
            if not p or p.get("age") is None:
                continue
            host = df[gsm].dropna()
            shared = host.index.intersection(mu.index)
            n += 1
            # The unspiked host is scored ONCE, then each foreign class at the spiked fractions only.
            # The first version put f = 0 inside the class loop with a break after it, so the break fired on
            # every class's first fraction and no spiked mixture was ever scored (2026-09-25).
            if (gsm, "none", 0.0) not in done:
                r = score(host.to_dict(), p["age"], gse)
                r.update({"gsm": gsm, "gse": gse, "foreign_class": "none", "f": 0.0, "n_mixed": 0})
                rows.append(r)
            for cls in FOREIGN:
                m = mu.loc[shared, cls]
                keep = m.notna()
                idx = shared[keep.values]
                mv = m[keep].to_numpy(dtype=float)
                for fr_ in FRACS[1:]:
                    if (gsm, cls, fr_) in done:
                        continue
                    b = host.copy()
                    b.loc[idx] = (1 - fr_) * b.loc[idx].to_numpy(dtype=float) + fr_ * mv
                    r = score(b.to_dict(), p["age"], gse)
                    r.update({"gsm": gsm, "gse": gse, "foreign_class": cls, "f": fr_,
                              "n_mixed": int(len(idx))})
                    rows.append(r)
            if n % 20 == 0:
                json.dump({"_meta": {"procedure": "PROC-FOREIGN-01", "fracs": FRACS, "foreign": FOREIGN,
                                     "detector": "1 - (immune + progenitor + stem_adult)"},
                           "rows": rows}, open("handoff/foreign01.json", "w"))
                print("  %s [%d] %s  %.1fs/host" % (gse, n, gsm, (time.time() - t0) / n), flush=True)
        json.dump({"_meta": {"procedure": "PROC-FOREIGN-01", "fracs": FRACS, "foreign": FOREIGN,
                             "detector": "1 - (immune + progenitor + stem_adult)"},
                   "rows": rows}, open("handoff/foreign01.json", "w"))
        print("%s DONE %.0fs, %d rows" % (gse, time.time() - t0, len(rows)), flush=True)
    print("TOTAL rows:", len(rows), flush=True)


if __name__ == "__main__":
    main()
