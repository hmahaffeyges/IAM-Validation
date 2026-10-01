#!/usr/bin/env python3
"""Cellular Performance Gauge — conductor v3 (development build, 2026-10-01; NOT commissioned). Scope: neutrophils only.

Runs after Stage 0 (intake) and Stage 1 (IDAT calibration), which are unchanged:
  Stage A  composition    atlas v2 solver (deconv_v2, frozen settings in methylphys_v2.SOLVER; whole-blood cell set)
  Stage M  Met-A          neutrophil: isolated / sorted neutrophils -> own floor; whole blood -> composition-matched healthy expectation
  Stage MC Met-A C-score  clustering of the neutrophil residual map over the healthy baseline (development: band not set)
  Stage T  slide tare     A_rel = A / median A of reference arrays on the same slide (>= 3), when references are supplied
Frozen inputs (Runtime Matrices/Met_A_Floors): metA_floors_v1_2.json, metA_floors_v1_2_loo.csv, neutrophil_reference_v1.json.
IAM-A (sequencing) runs through stage_q_iam_a.py, not from arrays.
Formulas (canon: Met-A, C-score):
  H(b) = -b log2 b - (1-b) log2(1-b)
  isolated:    Met-A = mean_i H(beta_i) / floor
  whole blood: Met-A = mean_i H(beta_i) / mean_i H(e_i),  e_i = sum_c f_c mu_c,i  (f: this specimen's fractions; mu: purified healthy profiles)
  residual z_i = (H(beta_i) - H(ref_i)) / s_i  (ref_i = neutrophil mean H, or H(e_i) in blood; s_i = shrunk healthy SD of H among neutrophils)
  C = var(block means of 50 consecutive sites x sqrt 50) / var(z)  /  healthy median clustering
"""
import json, os
import numpy as np, pandas as pd
import stage_m_met_a as SM
HERE = os.path.dirname(os.path.abspath(__file__)); RM = os.path.join(HERE, "Runtime Matrices", "Met_A_Floors")
BUILD = "development v3 (not commissioned) - neutrophils only"
_REF = None
def ref():
    global _REF
    if _REF is None: _REF = json.load(open(os.path.join(RM, "neutrophil_reference_v1.json")))
    return _REF
H = SM._H
ISOLATED = ("isolated neutrophils", "sorted neutrophils", "purified neutrophils", "neutrophils")

_BC = None
def _bc():
    global _BC
    if _BC is None: _BC = json.load(open(os.path.join(RM, "blood_composition_EPIC_v1.json")))
    return _BC
def stage_a_composition(beta, specimen, atlas_parquet=None, identity_json=None):
    """EPIC blood composition (blood_composition_EPIC_v1): 8 groups from Salas purified EPIC cells; markers exclude the neutrophil sites;
    NNLS, sum 1. Same platform and same reference as the expectation profiles (the atlas v2 solver mixes platforms and under-read neutrophils)."""
    from scipy.optimize import nnls
    B = _bc(); y = beta.reindex(B["markers"]); M = pd.DataFrame(B["mu_markers"], index=B["markers"]); ok = y.notna()
    f, res = nnls(M[ok].values, y[ok].values); f = f / f.sum() if f.sum() > 0 else f
    fr = dict(zip(B["groups"], [float(v) for v in f]))
    return {"stage": "A", "method": "EPIC blood NNLS (blood_composition_EPIC_v1)", "fractions": fr, "n_markers_used": int(ok.sum()),
            "residual_mae": float(np.abs(y[ok].values - M[ok].values @ f).mean())}

def _clustering(z, w=50):
    o = z.dropna().values; nb = len(o) // w
    if nb < 10: return None
    b = o[:nb * w].reshape(nb, w).mean(1) * np.sqrt(w); return float(np.var(b) / np.var(o))

def stage_m_blood(beta, fractions):
    """Whole blood: Met-A = mean H(beta) / mean H(e) at the neutrophil sites, e = sum_g f_g mu_g (EPIC purified group profiles).
    The untared value carries the composition-estimation offset (about -0.05 on the known mixtures); the gauge state is printed only after
    the tare against healthy whole bloods run the same way (Stage T)."""
    B = _bc(); S = pd.Index(B["neutrophil_sites"]); P = {g: pd.Series(v, index=S, dtype="float64") for g, v in B["profiles_at_neutrophil_sites"].items()}
    fn = fractions.get("NEU", 0.0)
    rec = {"stage": "M", "reading": "Met-A", "cell": "neutrophils", "specimen": "whole blood", "fraction": round(fn, 4), "build": BUILD,
           "band": "Normal 0.95-1.05 (after tare)", "A": None}
    if fn < MIN_READ_FRACTION:
        rec["reason"] = f"neutrophil fraction {fn:.3f} < {MIN_READ_FRACTION}: fraction reported, A withheld"; return rec, None
    x = beta.reindex(S); e = sum(v * P[g] for g, v in fractions.items() if g in P); ok = x.notna() & e.notna()
    A = float(H(x[ok]).mean() / H(e[ok]).mean())
    mu = P["NEU"]; xd = x + fn * 0.01 * (0.5 - mu)            # a known 1 % loss of the neutrophils' pattern, at this specimen's own fraction
    rec["shift_per_1pct_loss"] = round(float(H(xd[ok].clip(1e-6, 1 - 1e-6)).mean() / H(e[ok]).mean()) - A, 5)
    rec.update(_ceiling(x[ok], mu[ok]))
    rec.update(A=round(A, 4), n_sites=int(ok.sum()), expectation="composition-matched healthy (EPIC purified group profiles x this specimen's fractions)",
               state="untared: read A_rel (Stage T)")
    R = ref(); Sr = pd.Index(R["sites_ordered"])
    z = (H(beta.reindex(Sr)) - H(e.reindex(Sr))) / pd.Series(R["neutrophil_H_sd_shrunk"], index=Sr)
    return rec, z

MIN_READ_FRACTION = 0.20   # below this the 1 % shift is under 0.01 and too few sites carry the cell (DEV-LOWFRAC-01: 5 healthy arrays under 0.40)

def _ceiling(x, mu):
    """Entropy ceiling: per-site H peaks at beta = 0.5. Met-A is monotone in pattern loss only while the cell's methylated sites stay above 0.5
    (DNMT-01: A_meth saturates at 1/H(floor) once the methylated sites reach beta ~0.5)."""
    hi = mu > 0.5
    if not hi.any(): return {}
    m = float(x[hi].mean())
    return {"methylated_sites_mean_beta": round(m, 4), "past_entropy_ceiling": bool(m < 0.5)}

def stage_m_isolated(beta):
    rec = SM.read(beta, "neutrophils", specimen="isolated neutrophils")
    R = ref(); S = pd.Index(R["sites_ordered"])
    B = _bc(); Sb = pd.Index(B["neutrophil_sites"]); mu = pd.Series(B["profiles_at_neutrophil_sites"]["NEU"], index=Sb, dtype="float64")
    x = beta.reindex(Sb); ok = x.notna() & mu.notna(); rec.update(_ceiling(x[ok], mu[ok]))
    if rec.get("A") is not None:
        xd = (x + 0.01 * (0.5 - mu))[ok].clip(1e-6, 1 - 1e-6); rec["shift_per_1pct_loss"] = round(float(H(xd).mean() / H(x[ok].clip(1e-6, 1 - 1e-6)).mean() * rec["A"]) - rec["A"], 5)
    z = (H(beta.reindex(S)) - pd.Series(R["neutrophil_H_mean"], index=S)) / pd.Series(R["neutrophil_H_sd_shrunk"], index=S)
    return rec, z

def stage_mc_cscore(z):
    R = ref(); c = _clustering(z) if z is not None else None
    if c is None: return {"stage": "MC", "reading": "Met-A C-score", "C": None, "reason": "no residual map"}
    return {"stage": "MC", "reading": "Met-A C-score", "C": round(c / R["healthy_clustering_median"], 4), "clustering": round(c, 4),
            "healthy_baseline": R["healthy_clustering_median"], "n_healthy_baseline": len(R["healthy_clustering_LOO"]),
            "healthy_range": [min(R["healthy_clustering_LOO"]) / R["healthy_clustering_median"], max(R["healthy_clustering_LOO"]) / R["healthy_clustering_median"]],
            "status": "development: healthy band not yet set", "frac_abs_z_gt3": round(float((z.abs() > 3).mean()), 4)}

def stage_t_tare(A, slide_ref_A, shift_1pct=None):
    refs = [a for a in (slide_ref_A or []) if a is not None]
    if A is None: return {"stage": "T", "A_rel": None, "reason": "no A"}
    if len(refs) < 3: return {"stage": "T", "A_rel": None, "reason": f"untared: {len(refs)} same-slide reference arrays (>= 3 required)"}
    m = float(np.median(refs)); Ar = A / m
    sd = float(np.std(np.array(refs) / m, ddof=1)); dl = (2 * sd / shift_1pct) if shift_1pct and shift_1pct > 0 else None
    return {"stage": "T", "A_rel": round(Ar, 4), "slide_reference_median": round(m, 4), "n_refs": len(refs), "reference_spread_sd": round(sd, 4),
            "detection_limit_pct_loss": (round(dl, 2) if dl else None),
            "detection_note": "smallest loss of the cell's pattern (percent) this specimen could show: 2 x reference spread / shift per 1 % loss",
            "state": "Normal" if SM.NORMAL[0] <= Ar <= SM.NORMAL[1] else ("above Normal" if Ar > SM.NORMAL[1] else "below Normal")}

def run_neutrophil(beta, specimen="whole blood", atlas_parquet=None, identity_json=None, slide_ref_A=None, stage1_meta=None):
    """beta: pd.Series from Stage 1 (EPIC). Returns the v3 bundle."""
    beta = beta.copy(); beta.index = beta.index.astype(str)
    out = {"build": BUILD, "specimen": specimen, "platform": SM.platform_of(beta), "scope": "neutrophils only",
           "floors_version": SM._floors()["version"], "reference_version": ref()["version"]}
    if out["platform"] != "EPIC":
        out["refusal"] = "no frozen neutrophil floor for this platform yet (EPIC only; 450K pending)"; return out
    if specimen.lower() in ISOLATED:
        m, z = stage_m_isolated(beta); out["composition"] = {"stage": "A", "note": "isolated neutrophils: composition not solved"}
    else:
        a = stage_a_composition(beta, specimen); out["composition"] = a
        m, z = stage_m_blood(beta, a["fractions"])
    out["met_a"] = m; out["met_a_cscore"] = stage_mc_cscore(z); out["tare"] = stage_t_tare(m.get("A"), slide_ref_A, m.get("shift_per_1pct_loss"))
    out["withheld"] = ["tier lines beyond Normal (not yet measured on this scale)", "other cell types (outside commissioning scope)"]
    return out
