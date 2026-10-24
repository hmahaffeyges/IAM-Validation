#!/usr/bin/env python3
"""Stage M - Met-A. COMMISSIONED 2026-10-09 on isolated neutrophils, EPIC v1 (doors/COMMISSIONING_NOTE_METAA_NEUTROPHILS.md).

Met-A reads ONE cell type in ONE specimen against that cell type's own healthy reference on the SAME platform:

    Met-A = mean over the cell's identity sites of H(beta)  /  H_ref,     H(b) = -b log2 b - (1-b) log2(1-b)

SCOPE: neutrophils on EPIC v1 (author ruling 2026-10-01). Identity sites and the healthy reference H_ref (key 'floor', a name older than the
gauge's marks; it is the healthy middle mark, not the floor) are frozen in
Runtime Matrices/Met_A_Floors/metA_floors_v1_3.json (EPIC neutrophils; 6 purified healthy arrays, each physical array counted once,
calibrated by this chain's own Stage 1). Healthy references of other cell types and of 450K are kept for development only in
metA_floors_v1_2_ALLCELLS_development.json and are not read here. Site rule: across-array SD <= 0.05 and mean beta 0.75-0.95
(methylated) or 0.05-0.25 (unmethylated), up to 3,000 per channel. The held-out precision of the healthy reference (sites re-chosen on the other
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
               band="Normal 0.95-1.05", build="chain v4 Stage M (Met-A commissioned on isolated neutrophils, EPIC v1, 2026-10-09)", floors_version=_floors()["version"])
    if cell not in CELLS_IN_SCOPE:
        rec["reason"] = f"'{cell}' is outside the commissioning scope (neutrophils only, 2026-10-01)"; return rec
    if cell not in F:
        rec["reason"] = f"no {platform} healthy reference for '{cell}' (references exist for: {', '.join(sorted(F))})"; return rec
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


# ---- the derived baseline (DEVELOPMENT, DEV-METAA-HEALTHY-01 step 3; 2026-10-24) ---------------------------------------------------------
# Met-A's healthy mark computed from the cell's own held errors instead of measured on reference arrays. Layer 1 found each healthy cell's
# level at the sites it holds to be 1 - eps_loss where the mark is held and eps_gain where it is held absent (sequencing scale). At each
# identity site that level is put on the array scale through IAM-Atlas v2.1's measured transfer (inverse of g, PCHIP through its knots), and
# the mark is the mean entropy over the sites. Nothing is fitted. Not included: the array's own measurement blur (it can only raise H).
TRANSFER = os.path.join(HERE, "..", "doors", "data", "DEV_ATLAS_V21_01", "transfer_v21_smooth.json")   # 2026-10-24: the curve the atlas was built with (run_v21_block.py), not the 42-knot table
def derived_baseline(held, eps_loss, eps_gain):
    """held: per identity site, True where the cell holds the mark (array level >= 0.5), False where it is held absent.
    Returns dict(H=mean bits, n_held, n_absent, level_held, level_absent) on the array scale."""
    import json as _j, numpy as _n
    from scipy.interpolate import PchipInterpolator as _P
    # sequencing scale -> array scale exactly as the atlas converted its sequencing readings (atlas_block_v21.py, mode v21inv): g the PCHIP
    # through the smooth curve's knots, inverted on a grid of 200,001 points with a 1e-12 tie-breaker, so g's flat ends invert to a definite
    # value (2026-10-24: the first version inverted the 42-knot table directly, which fails where g is flat; disclosed)
    T = _j.load(open(TRANSFER)); _g = _P(T["knots_x"], T["knots_y"]); _gx = _n.linspace(0, 1, 200001); _gv = _g(_gx) + 1e-12 * _gx
    ginv = lambda y: _n.interp(y, _gv, _gx)
    H = lambda b: float(-(b * _n.log2(b) + (1 - b) * _n.log2(1 - b)))
    xh, xa = float(ginv(1 - eps_loss)), float(ginv(eps_gain)); h = _n.asarray(held, bool)
    return dict(H=(h.sum() * H(xh) + (~h).sum() * H(xa)) / len(h), n_held=int(h.sum()), n_absent=int((~h).sum()), level_held=xh, level_absent=xa)
