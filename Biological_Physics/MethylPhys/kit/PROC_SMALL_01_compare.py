#!/usr/bin/env python3
"""PROC-SMALL-01 step 2: score four configurations against the pre-registered bars.

Configurations, all on the same cached betas:
  BASE      the commissioned selection, unweighted NNLS (what the chain does today)
  GLS       same markers, inverse-variance weights from the atlas posterior SDs
  CONTRAST  markers chosen for the largest gap against immune, unweighted
  BOTH      contrast markers and inverse-variance weights

For every configuration and every array we compute, per class, both
  * the NNLS point estimate of the fraction, and
  * a detection statistic: refit the class design matrix WITHOUT that class and take the drop in residual
    sum of squares over the full model's mean square. The point estimate is pinned at the non-negativity
    boundary for a trace component; the residual improvement is not, which is the whole hypothesis.

The statistic's nominal degrees of freedom are meaningless under a non-negativity constraint, so no
distributional assumption is made: the threshold is the 95th percentile measured on healthy donors who play
no part in choosing the configuration (the 40 Uppsala nulls minus the two development hosts).
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import nnls

CLS_ORDER = None
HOSTS = {"GSM2333901": "A", "GSM2333905": "B", "GSM1051533": "C"}
DEV = ["GSM2333901", "GSM2333905"]
HELD_OUT = "GSM1051533"
SPIKES = ["secretory", "cycling"]
FRACTIONS = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

mk = json.load(open("handoff/small01_markers.json"))
CLASSES = mk["classes"]
betas = pd.read_parquet("handoff/small01_betas.parquet")
sd = pd.read_parquet("handoff/small01_atlas_sd.parquet")


def design(ref_dict, addresses):
    """R (addresses x classes) of atlas class means, and the address order it was built on."""
    addr = [a for a in addresses if a in ref_dict]
    R = np.full((len(addr), len(CLASSES)), np.nan)
    for i, a in enumerate(addr):
        means = ref_dict[a]
        row = [means[c] for c in CLASSES if c in means]
        fill = float(np.mean(row)) if row else 0.5
        for j, c in enumerate(CLASSES):
            R[i, j] = means.get(c, fill)
    return R, addr


CONFIG = {}
for name, key, weighted in (("BASE", "base", False), ("GLS", "base", True),
                            ("CONTRAST", "contrast", False), ("BOTH", "contrast", True)):
    R, addr = design(mk[key], list(betas.index))
    if weighted:
        v = sd.reindex(addr)[[f"{c}_sd" for c in CLASSES]].to_numpy(dtype=float)
        var = np.nanmean(np.square(v), axis=1)
        var = np.where(np.isfinite(var) & (var > 1e-8), var, np.nanmedian(var[np.isfinite(var)]))
        w = 1.0 / var
        w = w / np.nanmean(w)
    else:
        w = np.ones(R.shape[0])
    CONFIG[name] = {"R": R, "addr": addr, "w": w}
    print(f"{name}: {R.shape[0]:,} addresses x {R.shape[1]} classes | weights "
          f"{'inverse-variance' if weighted else 'uniform'} "
          f"(range {w.min():.3g}-{w.max():.3g})", flush=True)


def fit(cfg, y, drop=None):
    """Weighted NNLS fit; returns (fractions over CLASSES or None for the dropped column, weighted RSS)."""
    R, w = cfg["R"], cfg["w"]
    keep = [j for j, c in enumerate(CLASSES) if c != drop]
    ok = np.isfinite(y) & np.all(np.isfinite(R[:, keep]), axis=1)
    sw = np.sqrt(w[ok])
    A = R[np.ix_(ok, keep)] * sw[:, None]
    b = y[ok] * sw
    f, _ = nnls(A, b)
    rss = float(np.sum((A @ f - b) ** 2))
    s = f.sum()
    frac = {}
    if s > 0:
        for idx, j in enumerate(keep):
            frac[CLASSES[j]] = float(f[idx] / s)
    return frac, rss, int(ok.sum())


def score(cfg, y):
    """Fractions from the constrained fit, plus a detection statistic that is NOT boundary-degenerate.

    The pre-registered F form - refit without the class, take the drop in residual sum of squares - was run
    first and found degenerate on 2026-09-23: when a class's coefficient sits at the non-negativity boundary,
    removing its column changes the fit by exactly nothing, so F was identically 0 on 36 of 38 healthy donors
    and could not respond to a small spike either. Inheriting the boundary was the very failure the statistic
    was meant to escape.

    Replaced with the standard added-variable (score) test, which is continuous by construction and is the
    same idea done properly: fit the model WITHOUT the candidate class, take the residual, project the
    candidate's profile off the columns already in the model, and regress one on the other UNCONSTRAINED. The
    coefficient may be negative, which is what lets the statistic vary below the boundary. Bars unchanged;
    the threshold is still the 95th percentile on donors not used to choose a configuration.
    """
    R, w = cfg["R"], cfg["w"]
    full, rss_full, n = fit(cfg, y)
    out = {"fractions": full, "rss": rss_full, "n_addresses": n, "F": {}, "t": {}}
    k = len(CLASSES)
    for c in SPIKES:
        j = CLASSES.index(c)
        keep = [q for q in range(k) if q != j]
        ok = np.isfinite(y) & np.all(np.isfinite(R), axis=1)
        sw = np.sqrt(w[ok])
        A = R[np.ix_(ok, keep)] * sw[:, None]
        b = y[ok] * sw
        f0, _ = nnls(A, b)
        resid = b - A @ f0
        rss_null = float(np.sum(resid ** 2))
        ms = rss_full / max(n - k, 1)
        out["F"][c] = float((rss_null - rss_full) / ms) if ms > 0 else 0.0
        # added-variable test: the candidate profile, with the fitted columns projected out
        pc = R[ok, j] * sw
        active = [i for i, v in enumerate(f0) if v > 1e-12]
        if active:
            Aa = A[:, active]
            coef, *_ = np.linalg.lstsq(Aa, pc, rcond=None)
            pt = pc - Aa @ coef
        else:
            pt = pc - pt.mean() if False else pc - pc.mean()
        denom = float(pt @ pt)
        if denom <= 1e-12:
            out["t"][c] = 0.0
            continue
        beta = float(pt @ resid) / denom
        dof = max(len(pt) - len(active) - 1, 1)
        s2 = max(float(resid @ resid) / dof, 1e-18)
        se = (s2 / denom) ** 0.5
        out["t"][c] = beta / se if se > 0 else 0.0
    return out


def mixture(y, cls, f, ref_addr, ref_dict):
    if f <= 0:
        return y
    prof = np.array([ref_dict[a].get(cls, np.nan) for a in ref_addr])
    m = y.copy()
    ok = np.isfinite(prof)
    m[ok] = (1 - f) * y[ok] + f * prof[ok]
    return m


results = {"_config": {k: {"n_addresses": int(v["R"].shape[0])} for k, v in CONFIG.items()},
           "nulls": {}, "mixtures": {}}

null_ids = [g for g in betas.columns if g not in HOSTS]
print(f"\nnull donors (threshold set, none used to choose a configuration): {len(null_ids)}", flush=True)
for name, cfg in CONFIG.items():
    for g in null_ids:
        y = betas[g].reindex(cfg["addr"]).to_numpy(dtype=float)
        s = score(cfg, y)
        results["nulls"].setdefault(name, {})[g] = {"F": s["F"], "t": s["t"],
                                                    "frac": {c: round(s["fractions"].get(c, 0.0), 5)
                                                             for c in SPIKES + ["immune"]}}
    Fs = {c: sorted(results["nulls"][name][g]["t"][c] for g in null_ids) for c in SPIKES}
    thr = {c: float(np.percentile(Fs[c], 95)) for c in SPIKES}
    results.setdefault("_threshold", {})[name] = thr
    print(f"  {name}: null score-test t, 95th percentile " +
          ", ".join(f"{c} {thr[c]:.2f} (median {np.median(Fs[c]):.2f}, max {Fs[c][-1]:.2f})" for c in SPIKES),
          flush=True)

print("\nmixtures", flush=True)
for name, cfg in CONFIG.items():
    ref = mk["base" if name in ("BASE", "GLS") else "contrast"]
    for g in HOSTS:
        if g not in betas.columns:
            print(f"  {name} {g}: NOT CALIBRATED - excluded", flush=True)
            continue
        y0 = betas[g].reindex(cfg["addr"]).to_numpy(dtype=float)
        for cls in SPIKES:
            for f in FRACTIONS:
                y = mixture(y0, cls, f, cfg["addr"], ref)
                s = score(cfg, y)
                results["mixtures"].setdefault(name, {})[f"{g}|{cls}|{f}"] = {
                    "F": s["F"][cls], "t": s["t"][cls], "frac_pct": round(100 * s["fractions"].get(cls, 0.0), 3),
                    "immune_frac": round(s["fractions"].get("immune", 0.0), 5)}
json.dump(results, open("handoff/small01_results.json", "w"), indent=1)
print("wrote handoff/small01_results.json", flush=True)
