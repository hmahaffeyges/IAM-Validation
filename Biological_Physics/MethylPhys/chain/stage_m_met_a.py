#!/usr/bin/env python3
"""Stage M — Met-A (Metrology A). Development build v3. Not commissioned.

Met-A reads ONE cell type in ONE specimen against that cell type's own healthy floor on the SAME platform:

    Met-A = mean over the cell's identity sites of H(beta)  /  floor,     H(b) = -b log2 b - (1-b) log2(1-b)

SCOPE: neutrophils on EPIC v1 only (author ruling 2026-10-01). Identity sites and the floor are frozen in
Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json (EPIC neutrophils; 6 purified healthy arrays, each physical array counted once,
calibrated by this chain's own Stage 1). Floors of other cell types and of 450K are kept for development only in
metA_floors_v1_2_ALLCELLS_development.json and are not read here. Site rule: across-array SD <= 0.05 and mean beta 0.75-0.95
(methylated) or 0.05-0.25 (unmethylated), up to 3,000 per channel. The held-out precision of the floor (sites re-chosen on the other
arrays) is in metA_floors_v1_3_loo.csv and is printed with each reading.

One ruler: Normal = 0.95-1.05. This module reads a purified / sorted specimen of one cell type. In a mixture (whole blood) the reading is
made by conductor_v3.stage_m_blood against the composition-matched expectation, and each reading carries its own detection limit at its
fraction (no fixed fraction cut). Tier lines beyond Normal are withheld until measured on this scale."""
import json, os, math
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
FLOORS = os.path.join(HERE, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")
LOO = os.path.join(HERE, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3_loo.csv")
NORMAL = (0.95, 1.05)
SITE_COVERAGE_MIN = 0.9    # a reading needs >= 90 % of the identity sites measured (isolated and whole blood alike)
CELLS_IN_SCOPE = ("neutrophils",)   # author ruling 2026-10-01: neutrophils only until commissioned; add one cell type at a time
_F = None

def _floors():
    global _F
    if _F is None: _F = json.load(open(FLOORS))
    return _F

def _H(b):
    b = np.clip(np.asarray(b, dtype="float64"), 1e-6, 1 - 1e-6)
    return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))

_V2_NAME = r"_(?:TC|BC|TO|BO)\d{2}$"     # EPIC v2 probe names carry a design suffix (cg00000029_TC21); EPIC v1 and 450K names do not

def platform_of(beta):
    """Platform from the calibrated beta vector itself: 'EPIC_v2' if any probe name carries the EPIC v2 design suffix;
    otherwise 'EPIC' (v1) when the vector holds more than 700,000 probes, else '450K'."""
    idx = pd.Index(beta.index).astype(str)
    if idx.str.contains(_V2_NAME, regex=True).any(): return "EPIC_v2"
    return "EPIC" if len(idx) > 700_000 else "450K"

def available(platform=None):
    F = _floors()["platforms"]
    return {p: sorted(c for c in F[p] if c in CELLS_IN_SCOPE) for p in F if platform in (None, p)}

def precision(platform, cell):
    q = pd.read_csv(LOO); q = q[(q.platform == platform) & (q.cell == cell)]
    if q.empty: return None
    a = q.A_loo
    return dict(n_ref=int(len(a)), normal_fraction=round(float(((a >= NORMAL[0]) & (a <= NORMAL[1])).mean()), 3),
                sd=round(float(a.std()), 4), min=round(float(a.min()), 3), max=round(float(a.max()), 3))

def read(beta, cell, platform=None, specimen="purified"):
    """beta: pandas Series (probe -> beta) from Stage 1 of a purified / sorted specimen of one cell type. cell: floor name (see available()).
    Returns the Met-A record; A is None (with a reason) when the reading is not permitted."""
    platform = platform or platform_of(beta)
    F = _floors()["platforms"].get(platform, {})
    rec = dict(stage="M", reading="Met-A", cell=cell, platform=platform, specimen=specimen, fraction=None, A=None,
               band="Normal 0.95-1.05", build="development v3 (not commissioned)", floors_version=_floors()["version"])
    if cell not in CELLS_IN_SCOPE:
        rec["reason"] = f"'{cell}' is outside the commissioning scope (neutrophils only, 2026-10-01)"; return rec
    if cell not in F:
        rec["reason"] = f"no {platform} floor for '{cell}' (floors exist for: {', '.join(sorted(F))})"; return rec
    f = F[cell]; s = pd.Index(f["sites"]).intersection(beta.dropna().index)
    if len(s) < SITE_COVERAGE_MIN * f["n_sites"]:
        rec["reason"] = f"only {len(s)} of {f['n_sites']} identity sites measured"; return rec
    A = float(_H(beta.loc[s]).mean() / f["floor"])
    rec.update(A=round(A, 4), n_sites=int(len(s)), floor=round(f["floor"], 5),
               state="Normal" if NORMAL[0] <= A <= NORMAL[1] else ("above Normal" if A > NORMAL[1] else "below Normal"),
               floor_precision=precision(platform, cell))
    return rec

if __name__ == "__main__":
    import sys
    b = pd.read_csv(sys.argv[1], index_col=0).iloc[:, 0]; print(json.dumps(read(b, sys.argv[2]), indent=1))   # two-column CSV: cpg_id,beta
