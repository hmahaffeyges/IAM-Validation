"""Atlas v2.1 commissioning scorers (readiness L15; committed 2026-10-14 before the atlas exists). The bars are in sealed_commission_v21.json;
these functions apply them. planted_commission_v21.py tests every scorer on synthetic data before any real held-out array is read."""
import numpy as np, pandas as pd
FLOOR = 0.01 ** 2; NOISE = 20; RNG_SEED = 20261014

def predictive_interval(draws, sd_draws, kind="array", g=None, cov=None, scatter=None, level=0.90, seed=RNG_SEED):
    """draws, sd_draws: (K, sites) posterior draws of mu and of the donor spread for ONE cell. Returns (lo, hi) per site of the posterior
    predictive for one new sample from a new laboratory (offset 0). Array: N(mu, sd^2 + FLOOR). Sequencing: N(g(mu), slope^2 sd^2 + g(1-g)/cov
    + s(mu)^2 + FLOOR), the fit's own likelihood. NOISE noise samples per draw."""
    rng = np.random.default_rng(seed); K, S = draws.shape
    mu = np.repeat(draws, NOISE, 0).astype(float); sd = np.repeat(sd_draws, NOISE, 0).astype(float)
    if kind == "array": loc, var = mu, sd ** 2 + FLOOR
    else:
        h = g(mu); slope = (g(mu + 0.01) - g(mu - 0.01)) / 0.02
        var = slope ** 2 * sd ** 2 + np.clip(h * (1 - h), 0, 0.25) / np.maximum(cov, 1) + scatter(mu) ** 2 + FLOOR; loc = h
    y = loc + rng.standard_normal(loc.shape) * np.sqrt(var); a = (1 - level) / 2
    return np.nanquantile(y, a, 0), np.nanquantile(y, 1 - a, 0)

def u1_coverage(lo, hi, y):
    """Share of measured sites (finite y and interval) where one held-out sample's reading lies inside the interval."""
    k = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi); return float(((y[k] >= lo[k]) & (y[k] <= hi[k])).mean()) if k.any() else float("nan")

def u1_verdict(cov_by_set, lo_bar=0.80, hi_bar=0.97):
    """cov_by_set: {set: mean coverage over its samples} for one cell and platform. COMMISSIONED if every independent set is inside the bar."""
    if not cov_by_set: return "NOT ASSESSED"
    return "COMMISSIONED" if all(lo_bar <= v <= hi_bar for v in cov_by_set.values()) else "FAILED"

def u2_share(set_mean, anchor, tol):
    """Share of blind sites where a held-out set's mean for one panel cell lies within tol of the atlas epithelial anchor value."""
    k = np.isfinite(set_mean) & np.isfinite(anchor); return float((np.abs(set_mean[k] - anchor[k]) <= tol).mean()) if k.any() else float("nan")

def u2_verdict(shares, bar=0.90, epithelium=None):
    """shares: {panel cell: share} for one tissue. COMMISSIONED if the epithelium is assessed and every assessed panel cell meets the bar."""
    if not shares or epithelium not in shares: return "NOT ASSESSED"
    return "COMMISSIONED" if all(v >= bar for v in shares.values()) else "FAILED"

def u3_neutrophil(est, truth):
    """Bar 1 (DEV-ATLAS-COMMISSION-01, unchanged): mean absolute error <= 0.02 and every sample within 0.05."""
    e = np.abs(np.asarray(est) - np.asarray(truth)); return dict(mae=float(e.mean()), max=float(e.max()), met=bool(e.mean() <= 0.02 and e.max() <= 0.05))
