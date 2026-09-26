#!/usr/bin/env python3
# INSTRUMENT-TEST: extracts the deconvolver's marker loci from the raw EPIC-Italy series matrix for PROC-MF-03.
# Reads a data file and writes a reduced matrix; produces no reading of any kind.
"""One streaming pass over GSE51032_series_matrix.txt.gz (845 arrays, 485,577 loci) keeping ONLY the deconvolver's
marker loci - the 1,506 the detector uses plus the rest of celltype_ref so the full deconvolver can run on the same
matrix later. The 2026-09-26 lesson written into code: record how many of the detector's markers are present, so a
reduced matrix can never again be mistaken for one that carries the quantity a bar needs.
"""
import gzip
import json
import sys

import numpy as np
import pandas as pd

CH = "iamrepo/Biological_Physics/MethylPhys/chain"
sys.path.insert(0, CH)
import cpg_conductor as C  # noqa: E402

SRC = "/Users/hmahaffeyges/early_gape_evidence/GSE130748_RAW_GSE51057:51032_VAL047_VAL048/GSE51032_replication/GSE51032_series_matrix.txt.gz"
OUT = "results/mf03/GSE51032_deconvolver_markers.parquet"

dec_mod = C._load_module("walther_iam_deconvolver", C._find("walther_iam_deconvolver.py"))
dec = dec_mod.WaltherIAMDeconvolver("atlas_work/IAMAtlasREBUILD.csv",
                                    celltype_class_map=str(C._find("IAMAtlasREBUILD_celltype_to_class.json")), verbose=False)
want = set(map(str, dec.celltype_ref)) | set(map(str, dec.class_ref))
det = json.load(open("handoff/mf02_results.json"))
print(f"deconvolver marker loci wanted: {len(want):,}", flush=True)

rows = {}; hdr = None; n = 0
with gzip.open(SRC, "rt", errors="replace") as f:
    for l in f:
        if l.startswith("!series_matrix_table_begin"):
            break
    hdr = [x.strip().strip('"') for x in f.readline().rstrip("\n").split("\t")][1:]
    for l in f:
        if l.startswith("!series_matrix_table_end"):
            break
        n += 1
        cg, rest = l.split("\t", 1)
        cg = cg.strip('"')
        if cg in want:
            rows[cg] = rest
        if n % 100000 == 0:
            print(f"  {n:,} loci scanned, {len(rows):,} kept", flush=True)
df = pd.DataFrame({cg: np.array([np.nan if v in ("", "null", "NA") else float(v) for v in rest.rstrip("\n").split("\t")])
                   for cg, rest in rows.items()}, index=hdr).T
import os
os.makedirs("results/mf03", exist_ok=True)
df.to_parquet(OUT)
# how many of the DETECTOR's 1,506 markers (MF-02's common set) are here?
# MF-02's M is not saved by name; reconstruct: markers present on all 48 450K arrays = intersection recorded via n_markers only.
# So record coverage against the deconvolver's celltype_ref, which the detector's set is a subset of.
ct = [m for m in map(str, dec.celltype_ref) if m in df.index]
present_frac = df.loc[ct].notna().mean(axis=0)
rep = {"source": SRC, "arrays": int(df.shape[1]), "loci_kept": int(df.shape[0]), "celltype_ref_markers": len(dec.celltype_ref),
       "celltype_ref_present": len(ct), "median_present_per_array": float(present_frac.median()) * len(ct),
       "out": OUT}
json.dump(rep, open("handoff/mf03_extraction.json", "w"), indent=1)
print(f"\nwrote {OUT}: {df.shape[0]:,} loci x {df.shape[1]} arrays | celltype_ref markers present {len(ct):,} of {len(dec.celltype_ref):,} "
      f"| median per array {rep['median_present_per_array']:.0f}", flush=True)
