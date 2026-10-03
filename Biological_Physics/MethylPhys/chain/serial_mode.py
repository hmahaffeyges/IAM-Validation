# Toolkit: not yet wired into chain v3; enters the chain at commissioning with its own pre-registered check.  (SOP v3 section 2b, stage 12b (difference of two draws: delta_sky, delta_cells); chain/TOOLKIT.md)
"""serial_mode.py - one person, two or more draws (PROC-SERIAL-01). Pure functions over bundles and beta vectors; the chain's
stages are untouched. Wired into run_sample.py (--prior) only after PROC-SERIAL-01's bars are scored."""
import json, hashlib
import numpy as np, pandas as pd

def check_same_person(now: dict, prior: dict):
    """Refuse unless patient hash, array type and pipeline match. Returns (ok, reason)."""
    cn, cp = now.get("context") or {}, prior.get("context") or {}
    for k, what in (("patient_hash", "patient"), ("array_type", "array type"), ("pipeline", "pipeline")):
        a, b = cn.get(k) or now.get(k), cp.get(k) or prior.get(k)
        if a is None or b is None: return False, f"serial mode refused: {what} not recorded on {'this' if a is None else 'the prior'} draw"
        if a != b: return False, f"serial mode refused: {what} differs between draws ({a!r} vs {b!r}) - two draws of one person only"
    return True, "same person, same array type, same pipeline"

def delta_cells(now: dict, prior: dict, floor: dict | None = None):
    """Per-cell dA for cells present in both draws; appeared / disappeared listed, not differenced."""
    cn, cp = now.get("cells_all") or {}, prior.get("cells_all") or {}
    rows, appeared, gone = [], [], []
    for c in sorted(set(cn) | set(cp)):
        a, b = cn.get(c) or {}, cp.get(c) or {}
        pn, pp = bool(a.get("present")) and a.get("A") is not None, bool(b.get("present")) and b.get("A") is not None
        if pn and pp:
            d = float(a["A"]) - float(b["A"]); fl = (floor or {}).get(c)
            rows.append({"cell": c, "class": a.get("class"), "A_prior": round(float(b["A"]), 4), "A_now": round(float(a["A"]), 4), "dA": round(d, 4),
                         "f_prior": b.get("fraction"), "f_now": a.get("fraction"), "floor": fl, "beyond_floor": (abs(d) > fl) if fl is not None else None,
                         "tier_prior": b.get("tier"), "tier_now": a.get("tier")})
        elif pn: appeared.append({"cell": c, "A_now": a.get("A"), "f_now": a.get("fraction")})
        elif pp: gone.append({"cell": c, "A_prior": b.get("A"), "f_prior": b.get("fraction")})
    return {"cells": rows, "appeared": appeared, "disappeared": gone}

def delta_sky(beta_now: pd.Series, beta_prior: pd.Series, floor_beta: float | None = None):
    """beta_now - beta_prior per address on the addresses both draws measured. No expectation, no sigma, no composition."""
    idx = beta_now.index.intersection(beta_prior.index); d = (beta_now.reindex(idx) - beta_prior.reindex(idx)).dropna()
    out = {"n_addresses": int(len(d)), "median_dbeta": float(d.median()), "mean_abs_dbeta": float(d.abs().mean()),
           "q99_abs_dbeta": float(d.abs().quantile(0.99)), "floor_beta": floor_beta,
           "frac_beyond_floor": (float((d.abs() > floor_beta).mean()) if floor_beta is not None else None),
           "n_only_now": int(len(beta_now.index.difference(beta_prior.index))), "n_only_prior": int(len(beta_prior.index.difference(beta_now.index)))}
    return d, out

def snp_noise_floor_A(pci_entry: dict, beta_now: pd.Series, beta_prior: pd.Series, snp_now: dict | None, snp_prior: dict | None, hmin: float):
    """F2: the SNP-probe noise of the two arrays propagated to A over the cell's identity loci - a LOWER BOUND on the change floor.
    var(mean beta) = sum(a + b*beta(1-beta)) / n^2 per array; dA ~ |dH/dbeta| * sqrt(var_now + var_prior) / H_min at the mean."""
    if not snp_now or not snp_prior: return None
    loci = [l for l in pci_entry["loci"] if l in beta_now.index and l in beta_prior.index]
    if len(loci) < 10: return None
    def var_mean(beta, snp):
        v = beta.reindex(loci).dropna(); a, b = float(snp.get("a") or 0), float(snp.get("b") or 0)
        return float(np.sum(a + b * v * (1 - v)) / len(v) ** 2), float(v.mean())
    vn, mn = var_mean(beta_now, snp_now); vp, mp = var_mean(beta_prior, snp_prior); m = min(max((mn + mp) / 2, 1e-6), 1 - 1e-6)
    dH = abs(np.log2((1 - m) / m))            # |dH/dbeta| at the mean
    return float(2.576 * dH * np.sqrt(vn + vp) / hmin)   # 99 % two-sided

def trajectory(bundles: list[dict]):
    """Bundles in draw order; per-cell A_1..A_N with dates. Numbers in order - no fit, no slope."""
    cells = sorted({c for b in bundles for c, v in (b.get("cells_all") or {}).items() if v.get("present") and v.get("A") is not None})
    dates = [((b.get("context") or {}).get("intake_date") or b.get("run_id")) for b in bundles]
    rows = []
    for c in cells:
        A = [((b.get("cells_all") or {}).get(c) or {}).get("A") if ((b.get("cells_all") or {}).get(c) or {}).get("present") else None for b in bundles]
        steps = [None if (A[i] is None or A[i - 1] is None) else float(A[i]) - float(A[i - 1]) for i in range(1, len(A))]
        rows.append({"cell": c, "A": A, "steps": steps, "last_sign": (None if not steps or steps[-1] is None else ("+" if steps[-1] > 0 else "-" if steps[-1] < 0 else "0"))})
    return {"draws": len(bundles), "dates": dates, "cells": rows}
