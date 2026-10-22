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
