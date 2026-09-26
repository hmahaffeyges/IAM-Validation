#!/usr/bin/env python3
"""PROC-TISSUE-01 step 1: score every GSE131013 array through the chain's own path.

Order of operations is the pre-registration's, and it matters:

  1. groups are already frozen (handoff/tissue01_groups.json), written before any A was computed
  2. the SCORED CLASS is chosen from the HEALTHY MUCOSAE ALONE - the architecture class with the largest
     median deconvolved fraction there - so it cannot be picked after seeing which class separates
  3. only then is every array scored

A_abs = H(mean beta over that class's identity loci) / H_min, on betas passed through the chain's own
scale map. No laboratory zero and no age curve: both are per-laboratory or per-age constants and this is a
within-series comparison, so they cancel.
"""
import gzip
import json
import math
import sys

import numpy as np
import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH)
import cpg_conductor as C                                                    # noqa: E402

MATRIX = "tissue01/GSE131013_normalized_matrix.txt.gz"
OUT = "handoff/tissue01_scored.json"


def H(b):
    b = min(max(b, 1e-12), 1 - 1e-12)
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def main():
    meta = {d["gsm"]: d for d in json.load(open("handoff/tissue01_groups.json"))}
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
    ident = ident.get("classes", ident)
    loci = {k: set(str(x) for x in v["loci"]) for k, v in ident.items() if isinstance(v, dict) and v.get("loci")}
    hmin = {k: float(v["H_min"]) for k, v in ident.items() if isinstance(v, dict) and v.get("H_min")}

    dec_mod = C._load_module("walther_iam_deconvolver", C._find("walther_iam_deconvolver.py"))
    dec = dec_mod.WaltherIAMDeconvolver("atlas_work/IAMAtlasREBUILD.csv",
                                        celltype_class_map=str(C._find("IAMAtlasREBUILD_celltype_to_class.json")))
    keep = set().union(*loci.values())
    for attr in ("class_ref", "celltype_ref"):
        ref = getattr(dec, attr, None)
        if isinstance(ref, dict):
            keep |= set(map(str, ref.keys()))
        elif ref is not None and hasattr(ref, "index"):
            keep |= set(map(str, ref.index))
    print("loci the chain reads: %d" % len(keep), flush=True)

    with gzip.open(MATRIX, "rt", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
    print("matrix columns: %d | first five: %s" % (len(header), header[:5]), flush=True)
    cols = {}
    for i, h in enumerate(header):
        h = h.strip().strip('"')
        if h in meta:
            cols[h] = i
        else:
            for g, d in meta.items():
                t = (d.get("title") or "")
                if h and (h in t or t.endswith(h)):
                    cols[g] = i
                    break
    print("columns resolved to GSMs: %d of %d samples" % (len(cols), len(meta)), flush=True)
    missing = [g for g in meta if g not in cols]
    if missing:
        print("UNRESOLVED (no column in the matrix): %s"
              % [(g, meta[g]["group"], (meta[g].get("title") or "")[:46]) for g in missing], flush=True)
    assert len(cols) >= 200, "column mapping failed - inspect the header before proceeding"

    idx, data = [], []
    usecols = sorted(cols.values())
    back = {v: k for k, v in cols.items()}
    with gzip.open(MATRIX, "rt", errors="replace") as f:
        f.readline()
        for n, line in enumerate(f, 1):
            p = line.rstrip("\n").split("\t")
            cg = p[0].strip().strip('"')
            if cg not in keep:
                continue
            try:
                data.append([float(p[i]) if p[i] not in ("", "NA", "NaN") else np.nan for i in usecols])
            except (ValueError, IndexError):
                continue
            idx.append(cg)
            if n % 100000 == 0:
                print("  read %d rows, kept %d" % (n, len(idx)), flush=True)
    df = pd.DataFrame(np.array(data, dtype=np.float32), index=idx,
                      columns=[back[i] for i in usecols])
    print("kept matrix: %d loci x %d arrays" % df.shape, flush=True)
    df.to_parquet("tissue01/GSE131013_chain_loci.parquet")

    # ---- the scored class, from the HEALTHY MUCOSAE ALONE
    healthy = [g for g in df.columns if meta[g]["group"] == "mucosa"]
    fr_healthy = []
    for g in healthy:
        b = df[g].dropna().to_dict()
        fr_healthy.append(dict(dec.deconvolve(b).class_fractions))
    med = {k: float(np.median([f.get(k, 0.0) for f in fr_healthy]))
           for k in set().union(*[set(f) for f in fr_healthy])}
    SCORED = max(med, key=med.get)
    print("\nmedian class fractions in the %d HEALTHY mucosae: %s"
          % (len(healthy), {k: round(v, 4) for k, v in sorted(med.items(), key=lambda kv: -kv[1])}), flush=True)
    print("SCORED CLASS (largest median fraction in healthy only): %s\n" % SCORED, flush=True)
    assert SCORED in loci and SCORED in hmin, "the chosen class has no identity loci or floor"

    cls_loci = [x for x in df.index if x in loci[SCORED]]
    out = []
    for n, g in enumerate(df.columns, 1):
        b = df[g].dropna().to_dict()
        brm, lab = C.stage_1s_scale_map(b, "stage1_noob_450K")
        fr = dict(dec.deconvolve(b).class_fractions)
        vals = [brm[x] for x in cls_loci if x in brm]
        A = H(float(np.mean(vals))) / hmin[SCORED] if vals else None
        rec = dict(meta[g])
        rec.update({"A_abs": A, "n_loci": len(vals), "lab_flag": lab,
                    "fraction_scored_class": round(float(fr.get(SCORED, 0.0)), 4),
                    "epithelial_fraction": round(float(sum(fr.get(k, 0.0) for k in ("secretory", "terminal"))), 4),
                    "immune_fraction": round(float(fr.get("immune", 0.0)), 4),
                    "fractions": {k: round(v, 4) for k, v in fr.items()}})
        # every class, so a spread across classes can be told from a move in one
        rec["A_all_classes"] = {k: (H(float(np.mean([brm[x] for x in df.index if x in loci[k] and x in brm]))) / hmin[k])
                                for k in sorted(loci) if k in hmin}
        out.append(rec)
        if n % 40 == 0:
            print("  scored %d/%d" % (n, df.shape[1]), flush=True)

    json.dump({"scored_class": SCORED, "healthy_median_fractions": med, "rows": out}, open(OUT, "w"))
    ok = [r for r in out if r["A_abs"] is not None]
    print("\nscored %d arrays | median A_abs %.4f" % (len(ok), float(np.median([r["A_abs"] for r in ok]))), flush=True)


if __name__ == "__main__":
    main()
