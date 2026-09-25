#!/usr/bin/env python3
"""PROC-LOD-01: what is the smallest secretory or cycling fraction this chain can see in whole blood?

An in-silico admixture series on real material. Each mixture is a real healthy blood array with a known
fraction of a reference class profile mixed in at the beta level, which is how methylation fractions combine:
beta_mix = (1 - f) * beta_blood + f * beta_class. The chain then runs on the mixture exactly as it would on a
specimen, and we ask what it recovers.

Two readouts, because they answer different questions:

  * the FRACTION the solver reports - does it recover 1 % as 1 %? Where does the estimate leave the noise?
    That is the limit of detection, and it is what a presence floor should be set from instead of a typed 0.02.
  * the class's A on its own identity loci - at 1 % admixture a class's loci still carry 99 % blood DNA, so
    its entropy is the background's. This curve says where, if anywhere, A starts tracking the class.

No disease cohort is opened: the matrix is healthy blood and the spike is an atlas reference profile.
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "stage1")
sys.path.insert(0, "Biological_Physics/MethylPhys/chain")

import numpy as np
import pandas as pd
from stage_1_idat_calibration import calibrate_idat_to_beta
import cpg_conductor as C

ATLAS = os.path.abspath("Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD.csv")
CH = "Biological_Physics/MethylPhys/chain"
IDAT = f"{CH}/TEST_DATA/idats"
# Read the class columns straight from the atlas (about 6 s). A pickle written by another environment's
# pandas/numpy cannot be read here - the methylprep environment carries numpy 1.x and the cache was written
# by 2.x, which fails on numpy._core. Cross-environment handoffs go through CSV or parquet, never pickle.
_CLS = ["stem_pluri", "stem_adult", "progenitor", "stromal", "cycling", "secretory", "terminal", "immune"]
PROF = pd.read_csv(os.path.abspath("Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD.csv"),
                   usecols=["cpg_id"] + [f"{c}_mean" for c in _CLS] + [f"{c}_sd" for c in _CLS],
                   low_memory=False).set_index("cpg_id")
IDENT = json.load(open([p for p in __import__("glob").glob(
    CH + "/**/iamatlas_gauge_identity_loci_v1_0.json", recursive=True) if "RETIRED" not in p][0]))
CMAP = json.load(open("Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD_celltype_to_class.json"))

HOSTS = {"GSM2333901": {"age": 72.0, "lab": "GSE87571", "zero": -0.0117},
         "GSM2333905": {"age": 74.0, "lab": "GSE87571", "zero": -0.0117},
         "GSM1051533": {"age": 55.0, "lab": "GSE42861", "zero": 0.0084}}
SPIKES = ["secretory", "cycling"]
FRACTIONS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def H(b):
    """Binary entropy of a beta value, the chain's own convention."""
    b = min(max(float(b), 1e-9), 1 - 1e-9)
    return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))


def class_A(beta_series, cls):
    """A on this class's identity loci, the same construction stage_b_identity uses."""
    loci = [c for c in IDENT[cls]["loci"] if c in beta_series.index]
    if not loci:
        return None, 0
    return H(float(beta_series.loc[loci].mean())) / IDENT[cls]["H_min"], len(loci)


def main():
    out = {}
    if os.path.exists("handoff/dilution.json"):
        out = json.load(open("handoff/dilution.json"))

    hosts = {}
    for g in HOSTS:
        beta, _ = calibrate_idat_to_beta(f"{IDAT}/{g}_Grn.idat.gz", f"{IDAT}/{g}_Red.idat.gz")
        s = (beta.iloc[:, 0] if hasattr(beta, "columns") else beta).dropna()
        hosts[g] = s
        print(f"host {g}: {len(s):,} loci calibrated", flush=True)

    # the pure reference profile's own A, the value a perfect reading of 100 % of that class would give
    for cls in SPIKES:
        ref = PROF[f"{cls}_mean"].dropna()
        a, n = class_A(ref, cls)
        print(f"reference {cls}: A on its own identity loci = {a:.4f} on {n:,} loci "
              f"(what a pure specimen of this class would read)", flush=True)
        out.setdefault("_reference_A", {})[cls] = {"A": a, "n_loci": n}

    for g, meta in HOSTS.items():
        host = hosts[g]
        for cls in SPIKES:
            ref = PROF[f"{cls}_mean"].dropna()
            shared = host.index.intersection(ref.index)
            for f in FRACTIONS:
                key = f"{g}|{cls}|{f}"
                if key in out:
                    continue
                t0 = time.time()
                mix = host.copy()
                if f > 0:
                    mix.loc[shared] = (1 - f) * host.loc[shared] + f * ref.loc[shared]
                a_spike, n_spike = class_A(mix, cls)
                try:
                    o = C.run_full(mix.to_dict(), ATLAS,
                                   cfg={"age": meta["age"], "pipeline": "stage1_noob_450K",
                                        "lab_zero": meta["zero"], "lab": meta["lab"],
                                        "substrate": "whole_blood"})
                except Exception as e:
                    out[key] = {"error": f"{type(e).__name__}: {e}"}
                    print(f"  {key}: {out[key]['error'][:100]}", flush=True)
                    json.dump(out, open("handoff/dilution.json", "w"), indent=1)
                    continue
                cl = (o.get("composition") or {}).get("class") or {}
                cells = (o.get("composition") or {}).get("celltype") or []
                rollup = {}
                for x in cells:
                    k = CMAP.get(x.get("cell"))
                    if k and x.get("pct"):
                        rollup[k] = round(rollup.get(k, 0.0) + float(x["pct"]), 4)
                so = (o.get("second_opinion") or {}).get("by_class") or {}
                imm = (o.get("classes") or {}).get("immune") or {}
                out[key] = {
                    "host": g, "spike_class": cls, "spike_fraction": f,
                    "n_shared_loci": int(len(shared)),
                    "recovered_class_level_pct": cl.get(cls, 0.0),
                    "recovered_cell_rollup_pct": rollup.get(cls, 0.0),
                    "recovered_needlet_frac": (so.get(cls) or {}).get("nilc"),
                    "all_class_level_pct": {k: float(v) for k, v in cl.items()},
                    "spike_class_A_on_identity_loci": None if a_spike is None else round(a_spike, 4),
                    "spike_class_n_loci": n_spike,
                    "immune_A_abs": imm.get("A_abs"), "immune_pct": cl.get("immune"),
                    "immune_placement": imm.get("placement"),
                    "agreement": (o.get("second_opinion") or {}).get("agreement"),
                    "secs": round(time.time() - t0, 1)}
                json.dump(out, open("handoff/dilution.json", "w"), indent=1)
                print(f"  {g} {cls} spiked {f*100:5.2f}% -> class-level {out[key]['recovered_class_level_pct']:>5}% "
                      f"| rollup {out[key]['recovered_cell_rollup_pct']:>6}% | needlet "
                      f"{out[key]['recovered_needlet_frac']} | A(spike class) "
                      f"{out[key]['spike_class_A_on_identity_loci']} | immune A'' {out[key]['immune_A_abs']} "
                      f"({out[key]['secs']}s)", flush=True)
    print("DONE", len([k for k in out if not k.startswith('_')]), "mixtures", flush=True)


if __name__ == "__main__":
    main()
