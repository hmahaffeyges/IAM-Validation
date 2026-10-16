#!/usr/bin/env python3
"""Stage A21 - atlas v2.1 commissioning scorers, as AMENDED 2026-10-16 before the atlas exists (author: "Seal the amendment").
The instrument's job: a healthy cell from any laboratory reads inside the atlas's range for that cell, and a mixed blood sample splits into its
cells correctly. Amendment to sealed_commission_v21.json (doors/data/DEV_ATLAS_V21_01/commission_v21.py is the original, kept unchanged):
 1. the U1 range carries the measured laboratory-to-laboratory spread: the SD of the fit's own array laboratory offsets d
    (transfer_v21_final.json, read at run time; 18 laboratories, 0.0065 when sealed), added to the donor spread for arrays
 2. a sample or set the chain withholds (Stage 1 QC, the identity and sex check, the composition guard) is listed, not scored
 3. U1: a cell passes when >= 95 % of its healthy held-out samples, all laboratories pooled, have coverage in 0.80-0.97 (was: every set)
 4. U2: a tissue passes when the epithelium is assessed and >= 95 % of its assessed panel cells reach share >= 0.90 (was: every cell)
 5. U3: neutrophil mean absolute error <= 0.02 (unchanged) and >= 95 % of samples within 0.05 (was: every sample)
Every set, cell and sample is still reported individually; the bars 0.80-0.97, 0.90, 0.02 and 0.05 are as sealed."""
import os, json, numpy as np
STAGE = "A21 atlas commissioning (amended 2026-10-16)"
FLOOR = 0.01 ** 2; NOISE = 20; RNG_SEED = 20261014; SHARE = 0.95
HERE = os.path.dirname(os.path.abspath(__file__))
TERMS = os.path.join(HERE, "..", "doors", "data", "DEV_ATLAS_V21_01", "transfer_v21_final.json")
def lab_sd(terms=TERMS):
    """SD of the fit's own array-laboratory offsets d (measured, never typed)."""
    A = json.load(open(terms))["array"]; d = np.array([v["d"] for v in A.values() if isinstance(v, dict) and "d" in v], float)
    return float(d.std(ddof=1))
def predictive_interval(draws, sd_draws, kind="array", g=None, cov=None, scatter=None, level=0.90, seed=RNG_SEED, lab=None):
    """Posterior predictive for one new sample from a new laboratory. Array: N(mu, sd^2 + lab^2 + FLOOR) (amendment 1).
    Sequencing: N(g(mu), slope^2 sd^2 + g(1-g)/cov + s(mu)^2 + FLOOR), unchanged (its laboratory scatter s(mu) was already in)."""
    rng = np.random.default_rng(seed); lab = lab_sd() if lab is None else lab
    mu = np.repeat(draws, NOISE, 0).astype(float); sd = np.repeat(sd_draws, NOISE, 0).astype(float)
    if kind == "array": loc, var = mu, sd ** 2 + lab ** 2 + FLOOR
    else:
        h = g(mu); slope = (g(mu + 0.01) - g(mu - 0.01)) / 0.02
        var = slope ** 2 * sd ** 2 + np.clip(h * (1 - h), 0, 0.25) / np.maximum(cov, 1) + scatter(mu) ** 2 + FLOOR; loc = h
    y = loc + rng.standard_normal(loc.shape) * np.sqrt(var); a = (1 - level) / 2
    return np.nanquantile(y, a, 0), np.nanquantile(y, 1 - a, 0)
def u1_coverage(lo, hi, y):
    k = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi); return float(((y[k] >= lo[k]) & (y[k] <= hi[k])).mean()) if k.any() else float("nan")
def u1_verdict(cov_by_sample, withheld=(), lo_bar=0.80, hi_bar=0.97):
    """cov_by_sample: {sample: coverage} for one cell and platform, all laboratories pooled. withheld: samples the chain withheld (listed, not scored)."""
    S = {k: v for k, v in cov_by_sample.items() if k not in set(withheld) and np.isfinite(v)}
    if not S: return dict(verdict="NOT ASSESSED", n=0, share=None, withheld=sorted(withheld))
    sh = float(np.mean([lo_bar <= v <= hi_bar for v in S.values()]))
    return dict(verdict="COMMISSIONED" if sh >= SHARE else "FAILED", n=len(S), share=sh, withheld=sorted(set(withheld) & set(cov_by_sample)))
def u2_share(set_mean, anchor, tol):
    k = np.isfinite(set_mean) & np.isfinite(anchor); return float((np.abs(set_mean[k] - anchor[k]) <= tol).mean()) if k.any() else float("nan")
def u2_verdict(shares, bar=0.90, epithelium=None):
    if not shares or epithelium not in shares: return dict(verdict="NOT ASSESSED", share=None)
    sh = float(np.mean([v >= bar for v in shares.values()]))
    return dict(verdict="COMMISSIONED" if sh >= SHARE else "FAILED", share=sh, failing=sorted(k for k, v in shares.items() if v < bar))
def u3_neutrophil(est, truth, withheld_mask=None):
    e = np.abs(np.asarray(est, float) - np.asarray(truth, float))
    if withheld_mask is not None: e = e[~np.asarray(withheld_mask, bool)]
    w = float((e <= 0.05).mean()); return dict(mae=float(e.mean()), within_005=w, max=float(e.max()), met=bool(e.mean() <= 0.02 and w >= SHARE))
