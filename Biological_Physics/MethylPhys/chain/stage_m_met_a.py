#!/usr/bin/env python3
"""Stage M — Met-A (Metrology A). Development build v3, 2026-10-01. Not commissioned.

Met-A reads ONE cell type in ONE specimen against that cell type's own healthy floor on the SAME platform:

    Met-A = mean over the cell's identity sites of H(beta)  /  floor,     H(b) = -b log2 b - (1-b) log2(1-b)

SCOPE: neutrophils only (author ruling 2026-10-01) until the reading is commissioned with new VALs; other floors are kept in the file
for development but the chain does not report them. Identity sites and floors are frozen in Runtime Matrices/Met_A_Floors/metA_floors_v1_2.json, built from purified healthy arrays calibrated by this
chain's own Stage 1 (EPIC: Salas 2018/2022 purified blood cells; 450K: GSE63409 healthy bone-marrow progenitors). Site rule: across-array
SD <= 0.05 and mean beta 0.75-0.95 (methylated) or 0.05-0.25 (unmethylated), up to 3,000 per channel. The leave-one-out precision of every floor
is in metA_floors_v1_2_loo.csv and is printed with each reading.

One ruler: Normal = 0.95-1.05. A is reported for the DOMINANT cell of the specimen only (Stage A fraction >= MIN_FRACTION) or for a purified /
sorted specimen of a single type. Minor cells: fraction only (PROC-SCORE-01: a minor cell's A cannot be read out of a mixture).
Tier lines beyond Normal are withheld until measured on this scale."""
import json, os, math
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
FLOORS = os.path.join(HERE, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_2.json")
LOO = os.path.join(HERE, "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_2_loo.csv")
NORMAL = (0.95, 1.05)
MIN_FRACTION = 0.50
CELLS_IN_SCOPE = ("neutrophils",)   # author ruling 2026-10-01: neutrophils only until commissioned; add one cell type at a time
_F = None

def _floors():
    global _F
    if _F is None: _F = json.load(open(FLOORS))
    return _F

def _H(b):
    b = np.clip(np.asarray(b, dtype="float64"), 1e-6, 1 - 1e-6)
    return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))

def platform_of(beta):
    """EPIC (>700k probes) or 450K, from the calibrated beta vector."""
    return "EPIC" if len(beta) > 700_000 else "450K"

def available(platform=None):
    F = _floors()["platforms"]
    return {p: sorted(c for c in F[p] if c in CELLS_IN_SCOPE) for p in F if platform in (None, p)}

def precision(platform, cell):
    q = pd.read_csv(LOO); q = q[(q.platform == platform) & (q.cell == cell)]
    if q.empty: return None
    a = q.A_loo
    return dict(n_ref=int(len(a)), normal_fraction=round(float(((a >= NORMAL[0]) & (a <= NORMAL[1])).mean()), 3),
                sd=round(float(a.std()), 4), min=round(float(a.min()), 3), max=round(float(a.max()), 3))

def read(beta, cell, platform=None, fraction=None, specimen="purified"):
    """beta: pandas Series (probe -> beta) from Stage 1. cell: floor name (see available()). fraction: Stage A fraction of this cell, or None for
    a purified/sorted specimen. Returns the Met-A record; A is None (with a reason) when the reading is not permitted."""
    platform = platform or platform_of(beta)
    F = _floors()["platforms"].get(platform, {})
    rec = dict(stage="M", reading="Met-A", cell=cell, platform=platform, specimen=specimen, fraction=fraction, A=None,
               band="Normal 0.95-1.05", build="development v3 (not commissioned)", floors_version=_floors()["version"])
    if cell not in CELLS_IN_SCOPE:
        rec["reason"] = f"'{cell}' is outside the commissioning scope (neutrophils only, 2026-10-01)"; return rec
    if cell not in F:
        rec["reason"] = f"no {platform} floor for '{cell}' (floors exist for: {', '.join(sorted(F))})"; return rec
    if fraction is not None and fraction < MIN_FRACTION:
        rec["reason"] = f"minor cell (fraction {fraction:.3f} < {MIN_FRACTION}): fraction reported, A withheld"; return rec
    f = F[cell]; s = pd.Index(f["sites"]).intersection(beta.dropna().index)
    if len(s) < 0.9 * f["n_sites"]:
        rec["reason"] = f"only {len(s)} of {f['n_sites']} identity sites measured"; return rec
    A = float(_H(beta.loc[s]).mean() / f["floor"])
    rec.update(A=round(A, 4), n_sites=int(len(s)), floor=round(f["floor"], 5),
               state="Normal" if NORMAL[0] <= A <= NORMAL[1] else ("above Normal" if A > NORMAL[1] else "below Normal"),
               floor_precision=precision(platform, cell))
    return rec

def stage_m(beta, stage_a_out=None, specimen="whole blood", cell=None):
    """Chain entry. Purified/sorted specimen: pass cell=. Mixture: the dominant Stage A cell is read if it has a floor and fraction >= MIN_FRACTION."""
    if cell: return {"readings": [read(beta, cell, specimen=specimen)]}
    out = []
    cells = (stage_a_out or {}).get("cells", {})
    for ct, v in sorted(cells.items(), key=lambda kv: -(kv[1].get("fraction") or 0)):
        if not v.get("present"): continue
        out.append(read(beta, ct, fraction=v.get("fraction"), specimen=specimen) if ct in _floors()["platforms"].get(platform_of(beta), {})
                   else dict(stage="M", cell=ct, fraction=v.get("fraction"), A=None, reason="no floor for this cell on this platform"))
    return {"readings": out}

if __name__ == "__main__":
    import sys
    b = pd.read_parquet(sys.argv[1]).iloc[:, 0]; print(json.dumps(read(b, sys.argv[2]), indent=1))
