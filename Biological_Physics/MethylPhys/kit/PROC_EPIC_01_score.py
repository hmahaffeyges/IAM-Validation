#!/usr/bin/env python3
"""PROC-EPIC-01 step 1: score every EPIC-Italy array through the commissioned chain.

One streaming pass over the 3 GB series matrix, keeping only the loci the chain reads (the identity-loci
union and the deconvolver's own markers), then every one of the 845 arrays scored through the chain's own
path - scale map, deconvolver, identity gauge, composition guard active.

All 845 are scored, not the 313 the pre-registration compares, because the author's rule is that data we
have already paid for gets used for everything it can answer: the overlapping 329 are needed to show what
the old discovery set looks like on the NEW surface, and the male controls are needed to measure the sex
term that justifies excluding them.

The reduced matrix is written to parquet so no later procedure re-parses 3 GB.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH)
import cpg_conductor as C                                                    # noqa: E402

E = "/Users/hmahaffeyges/early_gape_evidence/GSE130748_RAW_GSE51057:51032_VAL047_VAL048"
MATRIX = os.path.join(E, "GSE51032_replication/GSE51032_series_matrix.txt")
RED = "results/epic01/GSE51032_chain_loci.parquet"
os.makedirs("results/epic01", exist_ok=True)


def keep_set(dec):
    """Exactly the loci the chain reads: identity loci for every class, plus the deconvolver's markers."""
    ident = json.load(open(C._find("iamatlas_gauge_identity_loci_v1_0.json")))
    k = ident.get("classes", ident)
    keep = set()
    for _cls, rec in k.items():
        if isinstance(rec, dict) and rec.get("loci"):
            keep |= set(rec["loci"])
    n_ident = len(keep)
    # class_ref and celltype_ref are DICTS keyed by CpG, not frames - an index check silently added nothing
    # and would have left the deconvolver fitting on whatever markers happened to fall in the identity set
    # (2026-09-26).
    for attr in ("class_ref", "celltype_ref"):
        ref = getattr(dec, attr, None)
        if isinstance(ref, dict):
            keep |= set(map(str, ref.keys()))
        elif ref is not None and hasattr(ref, "index"):
            keep |= set(map(str, ref.index))
    print("keep set: %d identity loci, %d total with the deconvolver's markers" % (n_ident, len(keep)),
          flush=True)
    return keep


def stream_matrix(keep):
    """Rows are CpGs and columns are samples, so one pass keeping only the rows we need."""
    if os.path.exists(RED):
        df = pd.read_parquet(RED)
        print("reduced matrix already built: %d loci x %d arrays" % df.shape, flush=True)
        return df
    t0 = time.time()
    rows, index, cols = [], [], None
    with open(MATRIX, errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                cols = [c.strip('"') for c in next(f).rstrip("\n").split("\t")][1:]
                break
        n = 0
        for line in f:
            if line.startswith("!series_matrix_table_end"):
                break
            i = line.index("\t")
            cpg = line[:i].strip('"')
            n += 1
            if cpg not in keep:
                continue
            vals = line[i + 1:].rstrip("\n").split("\t")
            rows.append(np.array([np.nan if v in ("", "NA", "null") else float(v) for v in vals],
                                 dtype=np.float32))
            index.append(cpg)
            if len(index) % 25000 == 0:
                print("  kept %d of %d rows scanned (%.0fs)" % (len(index), n, time.time() - t0), flush=True)
    df = pd.DataFrame(np.vstack(rows), index=index, columns=cols)
    df.to_parquet(RED)
    print("reduced matrix: %d loci x %d arrays, %.0f MB, %.0fs"
          % (df.shape[0], df.shape[1], os.path.getsize(RED) / 1e6, time.time() - t0), flush=True)
    return df


def main():
    atlas = "atlas_work/IAMAtlasREBUILD.csv"
    assert os.path.exists(atlas), "decompress the atlas first"
    dec_mod = C._load_module("walther_iam_deconvolver", C._find("walther_iam_deconvolver.py"))
    dec = dec_mod.WaltherIAMDeconvolver(atlas, celltype_class_map=str(
        C._find("IAMAtlasREBUILD_celltype_to_class.json")))

    df = stream_matrix(keep_set(dec))
    meta = {r["gsm"]: r for r in json.load(open("handoff/epic_italy_meta.json"))}

    lz = C._load_module("lab_zero", C._find("lab_zero.py"))
    curve = {int(k): v for k, v in json.load(open(C._find("reference_age_curve_v1.json")))["curve"].items()}

    out, t0 = [], time.time()
    for n, gsm in enumerate(df.columns, 1):
        m = meta.get(gsm)
        if m is None:
            continue
        beta = df[gsm].dropna()
        if len(beta) < 10000:
            out.append({"gsm": gsm, "error": "only %d loci" % len(beta)})
            continue
        bd = beta.to_dict()
        brm, lab = C.stage_1s_scale_map(bd, "GSE51032_450K")
        fr = dict(dec.deconvolve(bd).class_fractions)
        # no commissioned laboratory zero for EPIC-Italy: the chain withholds placement and tier, which is
        # correct. A zero is a constant per laboratory and cancels in a case-versus-control comparison.
        bi = C.stage_b_identity(brm, {"class_fractions": fr}, m.get("age"), lab, lab_zero=None)
        im = bi.get("immune", {}) or {}
        c = lz.age_reference(m["age"], curve) if m.get("age") is not None else None
        A = im.get("A_mapped")
        out.append({"gsm": gsm, "held_out": m["held_out"], "icd": m["icd"], "ttd": m["ttd"],
                    "age": m["age"], "sex": m["sex"], "A_mapped": A, "age_c": c,
                    "A_prime": (None if (A is None or c is None) else round(A - c, 6)),
                    "n_loci": im.get("n_loci"), "fraction": im.get("fraction"),
                    "foreign_fraction": im.get("foreign_fraction"),
                    "composition_verified": im.get("composition_verified"),
                    "fractions": {k: round(v, 4) for k, v in fr.items()}})
        if n % 100 == 0:
            print("  [%d/%d] %.1fs/array" % (n, df.shape[1], (time.time() - t0) / n), flush=True)

    json.dump(out, open("handoff/epic01_scored.json", "w"))
    ok = [r for r in out if r.get("A_prime") is not None]
    print("\nscored %d arrays, %d with a reading | median A' %.4f"
          % (len(out), len(ok), float(np.median([r["A_prime"] for r in ok]))), flush=True)


if __name__ == "__main__":
    main()
