#!/usr/bin/env python3
"""Stage Q - IAM-A (single-molecule sequencing), development build 2026-10-01, NOT commissioned. Scope: neutrophils.
IAM-A = H(eps) / (P_cell * H(eps0)), eps0 = 1/(1+exp(phi*M)) = 0.032 (canon), P_cell frozen in Runtime Matrices/IAM_A_Positions/iama_positions_v1.json.
eps = isolated copy errors / opportunities on qualifying molecules (>= 6 CpG calls, >= 80 % methylated), from the per-site table of the extractor
(columns pos, opp_A, err_A, opp_B, err_B). P is valid only for the read-level pipeline it was measured on; any other pipeline is refused."""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); POS = os.path.join(HERE, "Runtime Matrices", "IAM_A_Positions", "iama_positions_v1.json")
EPS0 = 0.0320
def _H(e): return float(-(e * np.log2(e) + (1 - e) * np.log2(1 - e)))
def read(site_table, cell="neutrophils", pipeline="loyfer_pat_v1", mask=None):
    P = json.load(open(POS)); rec = {"stage": "Q", "reading": "IAM-A", "cell": cell, "pipeline": pipeline, "build": "development v3", "A": None}
    c = P["cells"].get(cell)
    if c is None: rec["refusal"] = f"no frozen IAM-A position for {cell}"; return rec
    if c["pipeline"] != pipeline: rec["refusal"] = f"position for {cell} was measured on {c['pipeline']}, not {pipeline}: measure P on healthy {cell} with this pipeline first"; return rec
    D = site_table if mask is None else site_table[~site_table.pos.isin(mask)]
    o = {h: float(D[f"opp_{h}"].sum()) for h in "AB"}; e = {h: float(D[f"err_{h}"].sum()) for h in "AB"}
    if o["A"] + o["B"] < 1e5: rec["refusal"] = f"too few opportunities ({o['A'] + o['B']:.0f} < 100000)"; return rec
    eps = (e["A"] + e["B"]) / (o["A"] + o["B"]); A = _H(eps) / (c["P"] * _H(EPS0))
    halves = {h: _H(e[h] / o[h]) / (c["P"] * _H(EPS0)) for h in "AB" if o[h] > 5e4}
    rec.update(eps=round(eps, 6), A=round(A, 4), P=c["P"], E_kT=round(float(np.log((1 - eps) / eps)), 4), halves=halves,
               state="Normal" if 0.95 <= A <= 1.05 else ("above Normal" if A > 1.05 else "below Normal"), opportunities=int(o["A"] + o["B"]))
    return rec
