"""Shared helpers for development/sims (2026-10-09). Inputs only from this repository; seeds fixed by each script."""
import json, math, os
import numpy as np, pandas as pd
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XSP = os.path.join(REPO, "Biological_Physics/MethylPhys/doors/data/DEV_XSPECIES_TEMP_01")
EPS0 = 0.032        # IAM-A healthy reference copy error (CANON eps0_meth)
KM_DNMT1 = 4.4      # µM, SAM K_m used by DEV-SYNTH-LEVERS-01

def H2(e):
    return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))

def eps_at(A, lo=1e-4, hi=0.3):
    """Copy error giving IAM-A ratio A = H(eps)/H(EPS0)."""
    from scipy.optimize import brentq
    return brentq(lambda e: H2(e) / H2(EPS0) - A, lo, hi)

def xsp_table():
    """Per-run cross-species reads (xsp_results.jsonl, concatenated JSON) joined to the sample sheet; ok = >= 20,000 qualifying molecules."""
    raw = open(os.path.join(XSP, "xsp_results.jsonl")).read(); dec = json.JSONDecoder(); i = 0; recs = []
    while i < len(raw):
        while i < len(raw) and raw[i].isspace(): i += 1
        if i >= len(raw): break
        o, j = dec.raw_decode(raw, i); recs.append(o); i = j
    D = pd.read_csv(os.path.join(XSP, "temp_test_samples.csv")).merge(pd.DataFrame(recs), left_on="Run", right_on="run", how="left")
    D["ok"] = D.qualifying >= 20000
    return D

def within_species_sd():
    """Median over mammal species of the SD of ln(eps) across that species' liver runs."""
    D = xsp_table()
    w = D[D.ok & (D.cls == "Mammalia") & (D.tissue == "Liver")].groupby("species").eps.apply(lambda s: np.log(s).std()).dropna()
    return float(w.median())

def restore_ratio_A(S0, fold, renewed=1.0):
    """IAM-A ratio after SAM falls by `fold` from S0 µM: holding energy shifts by ln of the restore ratio (restore ∝ S/(K_m+S));
    only the renewed share of cells carries the new copy error (pooled molecules mix linearly in eps)."""
    g = (S0 / fold / (KM_DNMT1 + S0 / fold)) / (S0 / (KM_DNMT1 + S0))
    E0 = math.log((1 - EPS0) / EPS0); e1 = 1 / (1 + math.exp(E0 + math.log(g)))
    e = renewed * e1 + (1 - renewed) * EPS0
    return H2(e) / H2(EPS0), e
