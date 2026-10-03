#!/usr/bin/env python3
# Toolkit: not yet wired into chain v3; enters the chain at commissioning with its own pre-registered check.  (SOP v3 section 2b, stage 3c; chain/TOOLKIT.md)
"""toolkit_foreign_detection.py - stage 3c, foreign-cell detection: one joint NNLS of the specimen's panel markers on
[blood reference columns | foreign-cell templates], read against each template's measured noise floor.

Extracted unchanged (apart from the panel path argument) on 2026-10-03 from the retired class-era conductor, where it was
Stage 2d (PROC-STAGE2D-03, doors/PROC_STAGE2D_03_OUTCOME.md); the conductor itself is archived privately. Panel:
Runtime Matrices/Celltype_Marker/detection_panel_v3.json (markers, blood_ref, foreign_ref, noise_floor, not_detectable).
The input beta was the class-era scale-mapped beta; at commissioning the check must state the beta scale it reads.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "Runtime Matrices", "Celltype_Marker", "detection_panel_v3.json")


def stage_2d_foreign_detection(beta_mapped, stage_a_out, lab, bi=None, panel_path=None):
    """Stage 2d - FOREIGN-CELL DETECTION as ONE JOINT FIT (PROC-STAGE2D-03, adopted 2026-09-27 by the author's ruling: B1's
    UNSPECIFIC bar failed as written (1.9 % vs 1 %) and the author ruled the 1.9 % a measurement, not a false alarm -
    doors/PROC_STAGE2D_03_OUTCOME.md).

    One NNLS of the specimen's panel markers (mapped) on [blood reference columns | all 21 foreign templates];
    f_hat_cell = coef_cell / sum(coef). The earlier design (blood fitted FIRST, template read in the residual) let the blood
    fit absorb foreign material: with honest spikes it responded at 2-13 % of a 5 % spike (PROC_STAGE2D_02_OUTCOME.md), and
    the 0.5 % limits of PROC-MF-03 came from injecting the panel's own template. The joint fit responds at 0.6-1.0 of f.
    Line: the instrument's NOISE FLOOR - 0.99 quantile over 732 arrays known to lack the cell (author's decision 2026-09-27),
    printed with N; a standing bias on blood is printed beside it. >= 3 templates above their floors = 'epithelial-like
    material, cell not resolved' (1.9 % of healthy arrays, the oldest). A template not resolved on this block prints NOT
    DETECTABLE. Gate: composition check verified blood-like. Never changes the blood composition.
    """
    import numpy as _np
    from scipy.optimize import nnls as _nnls
    out = {"status": None, "laboratory": lab, "cells": {}, "detected": [], "not_detectable": {}}
    try:
        P = json.load(open(panel_path or PANEL))
    except Exception as e:
        out["status"] = "NOT_RUN: detection_panel_v3.json not found (%s)" % type(e).__name__; return out
    imm = ((bi or {}).get("immune") or {})
    if imm.get("composition_verified") is False:
        out["status"] = "WITHHELD: the composition guard did not verify this specimen as blood-like (foreign fraction %s); a line is not applied to a specimen that is not blood" % imm.get("foreign_fraction")
        return out
    M = P["markers"]; v = _np.array([beta_mapped.get(m, _np.nan) for m in M], dtype=float); ok = ~_np.isnan(v)
    if ok.sum() < 0.8 * len(M):
        out["status"] = "NOT_RUN: only %d of %d panel markers present (need 80 %%)" % (int(ok.sum()), len(M)); return out
    cells = list(P["foreign_ref"])
    Ab = _np.array([P["blood_ref"][c] for c in P["blood_columns"]]).T[ok]
    T = _np.array([P["foreign_ref"][c] for c in cells]).T[ok]
    coef, _ = _nnls(_np.column_stack([Ab, T]), v[ok]); tot = max(float(coef.sum()), 1e-9)
    fb = coef[:Ab.shape[1]]; ff = coef[Ab.shape[1]:]
    out["foreign_mass_total"] = round(float(ff.sum() / tot), 5)
    NF = P["noise_floor"]["cells"]; ND = P["not_detectable"]
    for c, a in zip(cells, ff):
        f = float(a / tot)
        if c in ND:
            out["not_detectable"][c] = {"f_hat": round(f, 5), "reason": ND[c]["reason"]}; continue
        cp = NF[c]
        rec = {"f_hat": round(f, 5), "line": cp["line"], "standing_bias": cp["standing_bias"], "detected": bool(f > cp["line"]),
               "measured_detection_limit": cp["measured_detection_limit_90pct"]}
        out["cells"][c] = rec
        if rec["detected"]: out["detected"].append(c)
    out["status"] = "OK"; out["n_markers_used"] = int(ok.sum())
    out["noise_floor"] = {"measured_on": P["noise_floor"]["laboratory_measured_on"], "n_arrays": P["noise_floor"]["n_arrays"], "quantile": P["_meta"]["quantile"]}
    out["panel_n"] = P["noise_floor"]["n_arrays"]; out["line_rule"] = P["_meta"]["line_rule"]
    if len(out["detected"]) >= 3:
        out["status"] = ("OK_BUT_UNSPECIFIC: %d templates above their floors (foreign-like mass %.1f %%) - epithelial-like material, cell not resolved"
                         % (len(out["detected"]), out["foreign_mass_total"] * 100))
    return out
