#!/usr/bin/env python3
"""PROC-SMALL-01 step 1: calibrate once, select markers once, then every configuration is cheap.

The dilution series re-read the 605 MB atlas on every mixture, which is why it took an hour. Here the
deconvolver is built once per marker configuration and reused, and every array is calibrated once and cached
restricted to the addresses any configuration uses. The configurations themselves then cost milliseconds.

Writes:
  handoff/small01_betas.parquet   - 43 arrays x the marker union (float32)
  handoff/small01_markers.json    - the two marker sets and the atlas means/SDs they carry
"""
import glob
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "stage1")
CH = "Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH)
sys.path.insert(0, f"{CH}/Walther_iam_deconvolver")

import numpy as np
import pandas as pd
from stage_1_idat_calibration import calibrate_idat_to_beta
import walther_iam_deconvolver as WD

ATLAS = os.path.abspath("Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD.csv")
CMAP = "Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD_celltype_to_class.json"
HOSTS = {"GSM2333901": 72.0, "GSM2333905": 74.0, "GSM1051533": 55.0}
IDAT_TEST = f"{CH}/TEST_DATA/idats"


def marker_sets():
    """Build the two marker configurations once. BASE is the commissioned selection; CONTRAST adds
    addresses chosen for the largest gap against immune, which is what a trace component in blood needs."""
    out = {}
    t0 = time.time()
    base = WD.WaltherIAMDeconvolver(ATLAS, CMAP, verbose=False)
    base._select_markers()
    out["BASE"] = base
    print(f"BASE markers: {len(base.class_ref):,} CpGs in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    con = WD.WaltherIAMDeconvolver(ATLAS, CMAP, verbose=False,
                                   contrast_pairs=[("secretory", "immune"), ("cycling", "immune")],
                                   n_contrast_markers_per_pair=2000)
    con._select_markers()
    out["CONTRAST"] = con
    print(f"CONTRAST markers: {len(con.class_ref):,} CpGs "
          f"(+{getattr(con, 'n_contrast_added', 0):,} contrast) in {time.time()-t0:.0f}s", flush=True)
    return out


def main():
    sets = marker_sets()
    union = sorted(set(sets["BASE"].class_ref) | set(sets["CONTRAST"].class_ref))
    print(f"marker union: {len(union):,} addresses", flush=True)

    # the atlas means each configuration carries, plus the posterior SDs for inverse-variance weighting
    CLS = list(WD.CLASSES)
    sd = pd.read_csv(ATLAS, usecols=["cpg_id"] + [f"{c}_sd" for c in CLS], low_memory=False).set_index("cpg_id")
    sd = sd.reindex(union)
    json.dump({"classes": CLS,
               "base": {c: sets["BASE"].class_ref[c] for c in list(sets["BASE"].class_ref)},
               "contrast": {c: sets["CONTRAST"].class_ref[c] for c in list(sets["CONTRAST"].class_ref)},
               "n_contrast_added": int(getattr(sets["CONTRAST"], "n_contrast_added", 0))},
              open("handoff/small01_markers.json", "w"))
    sd.to_parquet("handoff/small01_atlas_sd.parquet")
    print(f"atlas SDs cached for {sd.shape[0]:,} addresses", flush=True)

    # every array calibrated once, kept only on the union
    pat = re.compile(r"(GSM\d+)_(\d{9,12})_(R0\dC0\d)_(Grn|Red)\.idat\.gz$")
    pairs = {}
    for p in glob.glob("idats_gse87571/*.idat.gz") + glob.glob(f"{IDAT_TEST}/*.idat.gz"):
        m = pat.search(os.path.basename(p)) or re.search(r"(GSM\d+)_(Grn|Red)\.idat\.gz$", os.path.basename(p))
        if not m:
            continue
        g = m.group(1)
        ch = m.group(4) if m.lastindex and m.lastindex >= 4 else m.group(2)
        pairs.setdefault(g, {})[ch] = p

    nulls = sorted(json.load(open("handoff/floor_scan.json")).keys())
    want = [g for g in list(HOSTS) + nulls if g in pairs and len(pairs[g]) == 2]
    print(f"arrays to calibrate: {len(want)}", flush=True)

    cols, done = {}, {}
    if os.path.exists("handoff/small01_betas.parquet"):
        done = pd.read_parquet("handoff/small01_betas.parquet")
        cols = {c: done[c] for c in done.columns}
        print(f"resuming: {len(cols)} arrays already cached", flush=True)
    for n, g in enumerate(want, 1):
        if g in cols:
            continue
        t0 = time.time()
        try:
            beta, _ = calibrate_idat_to_beta(pairs[g]["Grn"], pairs[g]["Red"])
        except Exception as e:
            print(f"  {g}: {type(e).__name__}: {e}", flush=True)
            continue
        s = (beta.iloc[:, 0] if hasattr(beta, "columns") else beta).astype("float32")
        cols[g] = s.reindex(union)
        if n % 5 == 0 or n == len(want):
            pd.DataFrame(cols).to_parquet("handoff/small01_betas.parquet")
        print(f"  [{n}/{len(want)}] {g} in {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(cols)
    df.to_parquet("handoff/small01_betas.parquet")
    print(f"DONE betas {df.shape[0]:,} addresses x {df.shape[1]} arrays", flush=True)


if __name__ == "__main__":
    main()
