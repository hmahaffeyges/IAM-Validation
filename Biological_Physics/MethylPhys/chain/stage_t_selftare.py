#!/usr/bin/env python3
"""stage_t_selftare.py - Stage T step 1, the self-tare (self-tare II; adopted 2026-10-04 and run by conductor_v3.stage_t_selftare_ii on every
reading). Moved here unchanged from dev_stages.py on 2026-10-22 (Decoder finding F23): the step is commissioned and runs on every reading, so it
does not belong in the development module. Its matrix moved unchanged too, from Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json
to Runtime Matrices/Stage_T/selftare_typeII_EPIC_v1.json (same sha256), and is now a frozen input pinned by the release check (F1).

  anchors(beta)        this array's mean beta over each fixed-site set (type I low and high, type II low and high), with the count
  selftare_map(beta)   beta' = Lr + (beta - L)(Ur - Lr)/(U - L) per design, from this array's own anchors L, U onto the reference arrays' Lr, Ur
Nothing is fitted at run time and no other person enters the map: the anchors are the array's own; Lr and Ur are the frozen reference scale."""
import json, os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MATRIX = os.path.join(HERE, "Runtime Matrices", "Stage_T", "selftare_typeII_EPIC_v1.json")
_C = {}

def _json(name="selftare_typeII_EPIC_v1.json"):
    """The self-tare matrix, read once (any name for it, old or new, reads the same file)."""
    if "st" not in _C: _C["st"] = json.load(open(MATRIX)) if os.path.exists(MATRIX) else None
    return _C["st"]


def anchors(beta, st=None):
    """This array's anchors: mean beta over each fixed-site set (I_low, I_high, II_low, II_high) it measures, with the count."""
    st = st or _json()
    out = {}
    for k, sites in st["sets"].items():
        x = beta.reindex(sites).dropna(); out[k] = {"mean": float(x.mean()) if len(x) else None, "n": int(len(x))}
    return out


RULER = os.path.join(HERE, "Runtime Matrices", "Stage_T", "atlas_ruler_v1.json")

def ruler_map(beta):
    """The atlas ruler (development, 2026-10-23; builders/build_atlas_ruler.py): the self-tare's straight line for each probe design, with the
    array's own anchor means mapped onto IAM-Atlas v2.1's means at sites every cell type holds the same way (islands and shores never anchors).
    The anchor shift (the array's anchors before the map against the atlas's) is in the record; it is reported, never silently removed."""
    if "ruler" not in _C: _C["ruler"] = json.load(open(RULER)) if os.path.exists(RULER) else None
    if _C["ruler"] is None: return None, {"reason": "Runtime Matrices/Stage_T/atlas_ruler_v1.json not found"}
    b2, rec = selftare_map(beta, _C["ruler"]); rec = dict(rec or {}); rec["ruler"] = "atlas_ruler_v1 (IAM-Atlas v2.1 targets)"
    return b2, rec


def selftare_map(beta, st=None):
    """beta' = Lr + (beta - L)(Ur - Lr)/(U - L) per design at the sites whose design is recorded; other sites unchanged. Returns (beta', record)."""
    st = st or _json()
    if st is None: return None, {"reason": "Runtime Matrices/Stage_T/selftare_typeII_EPIC_v1.json not found"}
    an = anchors(beta, st); ra = st["ref_anchors"]; des = pd.Series(st["design"]); b2 = beta.copy(); maps = {}
    for d in ("I", "II"):
        L, U = an[f"{d}_low"]["mean"], an[f"{d}_high"]["mean"]; Lr, Ur = ra[f"{d}_low"], ra[f"{d}_high"]
        if L is None or U is None or U - L <= 0.1: maps[d] = None; continue
        s = des.index[des.values == d].intersection(b2.dropna().index)
        b2.loc[s] = (Lr + (b2.loc[s] - L) * (Ur - Lr) / (U - L)).clip(1e-6, 1 - 1e-6)
        maps[d] = {"L": round(L, 5), "U": round(U, 5), "L_ref": round(Lr, 5), "U_ref": round(Ur, 5), "slope": round((Ur - Lr) / (U - L), 5), "n_sites_mapped": int(len(s))}
    return b2, {"anchors": an, "maps": maps}


CURVE_BIN = 0.05; CURVE_MIN_SITES = 20; CURVE_AGREE = 0.10

def curve_cal(beta, cells, design):
    """Each array's response curve, per probe design, on its own calibrators (development, 2026-11-08, DEV-METAA-HEALTHY-01; the CMB route: each
    detector calibrated on its own data against known calibrators). The self-tare and the ruler map a straight line through two anchors (low,
    high); the GMP diagnostics showed each laboratory's arrays depart from IAM-Atlas v2.1 on each probe design differently, a bend a two-point
    line cannot take out. Calibrators: sites where every cell found in this sample (cells, from a first IAM-Deconvolver pass) holds the same
    level in the atlas (all within CURVE_AGREE of each other), outside cancer's territories (the islands and domains, never calibrators, so a
    change there is never calibrated away); a cell's identity sites are where cells differ, so they are never calibrators either. Per design,
    the calibrators are binned by their atlas level (CURVE_BIN); each bin with CURVE_MIN_SITES or more gives one point (the array's median, the
    atlas's median); the points, made monotone, map every site of that design (beyond the outer points, shifted by the outer offset).
    Nothing comes from other samples. Returns (mapped beta, record)."""
    import stage_metaa_atlas as SA, numpy as np
    b = pd.Series(beta, dtype="float64"); b.index = b.index.astype(str); des = pd.Series(design).reindex(b.index)
    MU = pd.concat([SA._col("mu", c).reindex(b.index).rename(c) for c in cells], axis=1)
    T = SA.territory_sets(); terr = b.index.isin(set(T["islands"]) | set(T["domains"]))
    ok = MU.notna().all(1).to_numpy() & ~terr & b.notna().to_numpy() & ((MU.max(1) - MU.min(1)) <= CURVE_AGREE).to_numpy()
    m = MU.mean(1); out = b.copy(); rec = {"cells": list(cells), "agree_within": CURVE_AGREE, "bin": CURVE_BIN, "min_sites": CURVE_MIN_SITES,
                                          "territories": "islands and domains are never calibrators", "designs": {}}
    for d in ("I", "II"):
        s = ok & (des == d).to_numpy(); xs, ts = b[s], m[s]
        bins = np.floor(ts.to_numpy() / CURVE_BIN).astype(int); P = []
        for k in np.unique(bins):
            w = bins == k
            if w.sum() >= CURVE_MIN_SITES: P.append((float(np.median(ts.to_numpy()[w])), float(np.median(xs.to_numpy()[w])), int(w.sum())))
        if len(P) < 3: rec["designs"][d] = {"mapped": False, "reason": f"{len(P)} calibrator levels with {CURVE_MIN_SITES} or more sites (3 needed)"}; continue
        P.sort(); tr = np.array([p[0] for p in P]); ob = np.maximum.accumulate(np.array([p[1] for p in P])); ob = ob + np.arange(len(ob)) * 1e-9
        sd = (des == d).to_numpy() & b.notna().to_numpy(); v = b[sd].to_numpy()
        y = np.interp(v, ob, tr); y = np.where(v < ob[0], v + (tr[0] - ob[0]), y); y = np.where(v > ob[-1], v + (tr[-1] - ob[-1]), y)
        out[sd] = np.clip(y, 1e-4, 1 - 1e-4)
        rec["designs"][d] = {"mapped": True, "calibrators": int(s.sum()), "levels": len(P), "points": [[round(p[0], 4), round(p[1], 4), p[2]] for p in P],
                             "largest_correction": round(float(np.max(np.abs(tr - ob))), 4), "sites_mapped": int(sd.sum())}
    return out, rec

