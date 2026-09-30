#!/usr/bin/env python3
"""MethylPhys v2 reader (PREVIEW, 2026-09-30). Atlas v2 only: composition by deconv_v2 (frozen settings), then each present cell's
A = H(mean beta over its v2 identity loci) / H_min(class). No v1 file, name, marker panel or scale map. Specimen on our Stage 1 scale."""
import json, numpy as np, pandas as pd
from deconv_v2 import DeconvV2
POOL1 = ["astrocytes", "microglia", "oligodendrocyte precursors", "vascular leptomeningeal cells"]
SOLVER = dict(markers="hybrid", margin=0.10, pair_margin=0.15, per_pair=60, k_near=8, top=600, sigma=0.02,
              drop=["vascular endothelium"])   # Moss profile reads as a blend of other endothelia, adipocytes and blood (prof_diag, 2026-09-30)
H = lambda b: -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
TIERS = [(0.95, 1.05, "NORMAL")]
BM = ["CMP (bone marrow)", "GMP (bone marrow)", "HSC (bone marrow)", "L-MPP (bone marrow)", "MEP (bone marrow)", "MPP (bone marrow)"]
# Specimen-type cell sets (2026-09-30). In circulating blood, bone-marrow progenitors are ~0.1 % of cells. Left in the whole-blood solve they
# took a median 14.7 % of 24 healthy GSE87571 arrays and left monocytes at 0.7 % and naive B at 0 %; without them the same arrays read
# monocytes 6.2 %, naive B 6.6 %, basophils 1.7 % (bm_where.log). A per-specimen scale term did not remove the sink (affine_test.log).
# In the 24 known mixtures they took ~0 either way. So: progenitors are modelled only where they physically are in quantity.

def tier_v15(A):
    """Tier word from the scheme in force (tier_breakpoints.json v1.5): SUPPRESSED < 0.95 <= NORMAL < 1.05 <= ELEVATED < 1.07 (Warburg line)
    <= SIGNIFICANTLY_ELEVATED < 1.10 <= BREACH. Fixed 2026-09-30: the preview reader had printed 'BELOW NORMAL' and had no tiers above ELEVATED."""
    if A is None or A != A: return None
    return "SUPPRESSED" if A < 0.95 else "NORMAL" if A < 1.05 else "ELEVATED" if A < 1.07 else "SIGNIFICANTLY_ELEVATED" if A < 1.10 else "BREACH"

SPECIMEN_DROP = {"whole blood": BM, "pbmc": BM, "buffy coat": BM, "bone marrow": [], "sorted": [], "tissue": []}
class ReaderV2:
    def __init__(self, atlas_parquet, identity_json):
        import pandas as _pd
        self.A = _pd.read_parquet(atlas_parquet); self.ID = json.load(open(identity_json))["cells"]; self.D = {}
    def solver(self, specimen):
        k = specimen.lower()
        if k not in self.D:
            self.D[k] = DeconvV2(None, pooled_one_sample=POOL1, A=self.A, **dict(SOLVER, drop=SOLVER["drop"] + SPECIMEN_DROP.get(k, [])))
        return self.D[k]
    def read(self, beta: pd.Series, specimen="whole blood", n_boot=100):
        beta = beta.copy(); beta.index = beta.index.astype(str)
        comp = self.solver(specimen).deconvolve(beta, n_boot=n_boot)
        cells = {}
        for c, f in comp["fractions"].items():
            if f < 0.01 or c not in self.ID: continue
            idl = self.ID[c]; v = beta.reindex(idl["loci"]).dropna()
            if len(v) < 100: cells[c] = dict(fraction=f, status="NOT READABLE", n_loci_read=int(len(v))); continue
            A = float(H(np.clip(v.mean(), 1e-9, 1 - 1e-9)) / idl["H_min"])
            cells[c] = dict(fraction=round(f, 4), ci=[round(x, 4) for x in comp["ci"][c]], present=comp["present"][c], A=round(A, 4),
                            tier=tier_v15(A),
                            carrying=f >= 0.30, n_loci_read=int(len(v)), klass=idl["klass"], flags=idl.get("flags", []))
        return dict(specimen=specimen, composition=comp, cells=cells, residual_mae=comp["residual_mae"])
