#!/usr/bin/env python3
# Toolkit: not yet wired into chain v3; enters the chain at commissioning with its own pre-registered check.  (SOP v3 section 2b, stage 3; chain/TOOLKIT.md)
"""Stage A - composition, atlas v2 solver (deconv_v2). Not called by chain v3 (conductor_v3.stage_a_composition uses blood_composition_EPIC_v1.json). Frozen settings (commissioned as PROC-DECONV-V2-01; specimen cell sets 2026-09-30).
In circulating blood the bone-marrow progenitors (~0.1 % of cells) are not modelled: left in, they absorbed a median 14.7 % of healthy
whole-blood arrays. The Moss vascular-endothelium profile is excluded (it reads as a blend of other endothelia, adipocytes and blood)."""
import pandas as pd
from deconv_v2 import DeconvV2
POOL1 = ["astrocytes", "microglia", "oligodendrocyte precursors", "vascular leptomeningeal cells"]
SOLVER = dict(markers="hybrid", margin=0.10, pair_margin=0.15, per_pair=60, k_near=8, top=600, sigma=0.02, drop=["vascular endothelium"])
BM = ["CMP (bone marrow)", "GMP (bone marrow)", "HSC (bone marrow)", "L-MPP (bone marrow)", "MEP (bone marrow)", "MPP (bone marrow)"]
SPECIMEN_DROP = {"whole blood": BM, "pbmc": BM, "buffy coat": BM, "bone marrow": [], "sorted": [], "tissue": []}
_CACHE = {}
def solver(atlas_parquet, specimen="whole blood"):
    k = (atlas_parquet, specimen.lower())
    if k not in _CACHE:
        A = pd.read_parquet(atlas_parquet)
        _CACHE[k] = DeconvV2(None, pooled_one_sample=POOL1, A=A, **dict(SOLVER, drop=SOLVER["drop"] + SPECIMEN_DROP.get(specimen.lower(), [])))
    return _CACHE[k]
