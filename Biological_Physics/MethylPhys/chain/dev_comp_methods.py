#!/usr/bin/env python3
"""dev_comp_methods.py - DEVELOPMENT - not commissioned. Copied unchanged in method on 2026-10-04 from doors/data/DEV_ATLAS_EPIC_01/code/comp_methods.py
(DEV-ATLAS-EPIC-01, 2026-10-03) so the development flags --dev-nilc and --dev-atlas-e (chain/dev_stages.py) read it from the chain. Composition methods for EPIC whole blood.

  NNLS8      chain v3 Stage A as shipped: NNLS on the 963 markers of blood_composition_EPIC_v1.json (8 Salas EPIC groups), sum 1.
  ATLAS_a    atlas v2 solver (deconv_v2.DeconvV2, frozen SOLVER settings, whole-blood cell set: bone-marrow progenitors and the Moss
             vascular endothelium out), atlas means as stored (= array scale; sequencing sources mapped through source_terms_v1.json
             inside the atlas fit), weights 1/(sum_c f_c^2 v_ci + sigma^2), v = posterior SD^2 + between-person SD^2 (atlas column `donor_sd`).
  ATLAS_b    as ATLAS_a, but only atlas cells measured on arrays (roster 'platforms' contains 'array'): WGBS-only and pooled cells out.
  ATLAS_c    as ATLAS_a, with cells merged by the atlas's own twin / cross-source rule (twin_family_thresholds_v1.json, records/10):
             a cell with no array measurement whose marker-profile correlation with an array-measured cell is >= cross_source_r is
             that cell on another platform -> merged into it (its column removed; 'twins become one cell'); two array cells merge only
             if r >= twin_r AND the sample-level test separates them at < min_separating_loci CpGs.
  NILC_*     constrained internal linear combination (CMB component separation): for each target cell g the minimum-variance weights
             w_g with w_g . a_g = 1 and w_g . a_k = 0 (k != g) [and, D1, w_g . 1 = 0: a constant beta offset is deprojected].
             W = C^-1 A (A^T C^-1 A)^-1. Linear, unbiased if the templates are right, no positivity and no sum-1 (the sum is a check).
             Noise per estimate sqrt(diag (A^T C^-1 A)^-1) is printed (the Cramer-Rao bound of the template set).
All composition site sets exclude the 6,000 neutrophil identity sites (chain rule: the composition must not read the cell it feeds).
Nothing here is fitted to the test specimens.
"""
import json, os, sys, glob
import numpy as np, pandas as pd
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deconv_v2 import DeconvV2

H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
POOL1 = ["astrocytes", "microglia", "oligodendrocyte precursors", "vascular leptomeningeal cells"]
SOLVER = dict(markers="hybrid", margin=0.10, pair_margin=0.15, per_pair=60, k_near=8, top=600, sigma=0.02)   # frozen (stage_a_composition_v2.py)
BM = ["CMP (bone marrow)", "GMP (bone marrow)", "HSC (bone marrow)", "L-MPP (bone marrow)", "MEP (bone marrow)", "MPP (bone marrow)"]
DROP_WB = ["vascular endothelium"] + BM
GROUP = {"neutrophils": "NEU", "eosinophils": "EOS", "basophils": "BASO", "monocytes": "MONO",
         "b cells": "B", "naive b cells": "B", "memory b cells": "B", "nk cells": "NK",
         "cd4 t cells": "CD4T", "naive cd4 t cells": "CD4T", "memory cd4 t cells": "CD4T", "regulatory t cells": "CD4T",
         "t central memory cd4": "CD4T", "t effector memory cd4": "CD4T",
         "cd8 t cells": "CD8T", "naive cd8 t cells": "CD8T", "effector memory cd8 t cells": "CD8T", "t effector cell cd8": "CD8T",
         "colon macrophages": "OTHER_IMMUNE", "lung alveolar macrophages": "OTHER_IMMUNE", "lung interstitial macrophages": "OTHER_IMMUNE",
         "microglia": "OTHER_IMMUNE"}
SUBGROUP = {"naive cd4 t cells": "CD4nv", "memory cd4 t cells": "CD4mem", "t central memory cd4": "CD4mem", "t effector memory cd4": "CD4mem",
            "regulatory t cells": "Treg", "naive cd8 t cells": "CD8nv", "effector memory cd8 t cells": "CD8mem", "t effector cell cd8": "CD8mem",
            "naive b cells": "Bnv", "memory b cells": "Bmem"}
BLOOD8 = ["NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T"]
grp = lambda c: GROUP.get(c, c if c in BLOOD8 else "NONBLOOD")


def load_atlas(path):
    A = pd.read_parquet(path)
    if "cpg_id" in A.columns: A = A.set_index("cpg_id")
    A.index = A.index.astype(str); return A


def atlas_cells(A):
    return sorted({c[:-5] for c in A.columns if c.endswith("_mean")})


def subset_atlas(A, keep):
    cols = [f"{c}_{s}" for c in keep for s in ("mean", "sd", "ci_lo", "ci_hi", "donor_sd", "n_obs")]
    return A[cols]


def build_deconv(A, keep, exclude_sites):
    """DeconvV2 on the cells in `keep` (frozen SOLVER), with the neutrophil identity sites removed from the locus pool."""
    S = subset_atlas(A, keep); S = S.loc[S.index.difference(exclude_sites)]
    return DeconvV2(None, pooled_one_sample=POOL1, A=S, **dict(SOLVER, drop=[]))


def twin_merge(A, D, roster, twin_tab, thr, candidates):
    """The atlas's own rule (twin_family_thresholds_v1.json; records/10_twin_test_sample_level.csv) applied on the markers of solver D.
    Returns (merge map {cell: into}, pair table)."""
    plat = roster.set_index("cell")["platforms"].astype(str).to_dict()
    has_array = {c: ("array" in plat.get(c, "")) and ("WGBS-pooled" != plat.get(c, "")) for c in candidates}
    idx = {c: k for k, c in enumerate(D.cells)}
    mu = D.mu; rows = []
    sep = {}
    for r in twin_tab.itertuples():
        if pd.notna(r.nonoverlap_sep): sep[frozenset((r.a, r.b))] = float(r.nonoverlap_sep)
    for i, p in enumerate(candidates):
        for q in candidates[i + 1:]:
            x, y = mu[:, idx[p]], mu[:, idx[q]]; r = float(np.corrcoef(x, y)[0, 1])
            rows.append(dict(a=p, b=q, r_markers=r, a_array=has_array[p], b_array=has_array[q], mean_abs_diff=float(np.abs(x - y).mean()),
                             sample_sep=sep.get(frozenset((p, q)), np.nan)))
    P = pd.DataFrame(rows); merge = {}
    for c in candidates:
        if has_array[c]: continue
        Q = P[((P.a == c) & P.b_array) | ((P.b == c) & P.a_array)].copy()
        if Q.empty: continue
        Q["other"] = np.where(Q.a == c, Q.b, Q.a); best = Q.sort_values("r_markers", ascending=False).iloc[0]
        if best.r_markers >= thr["cross_source_r"]: merge[c] = best.other
    for r in P.itertuples():   # same-platform twins
        if r.a_array and r.b_array and r.r_markers >= thr["twin_r"] and pd.notna(r.sample_sep) and r.sample_sep < thr["min_separating_loci"]:
            merge[r.b] = r.a
    P["rule_outcome"] = [("merge %s -> %s" % (r.a, merge[r.a]) if merge.get(r.a) == r.b else
                          "merge %s -> %s" % (r.b, merge[r.b]) if merge.get(r.b) == r.a else "keep") for r in P.itertuples()]
    return merge, P


def mixture_twin(D, candidates, r_thr):
    """Identifiability in a mixture (atlas only): a cell whose template, at solver D's markers, is reproduced by a non-negative,
    sum-1 combination of the other candidate cells with correlation >= r_thr (the atlas twin_r) cannot be told apart from that mixture
    of cells in a specimen: it is a twin of a mixture (the D6 rule: mixtures out). Removed one at a time, highest r first, re-tested.
    Returns (removed {cell: {components}}, log rows)."""
    idx = {c: k for k, c in enumerate(D.cells)}; keep = list(candidates); removed = {}; rows = []; it = 0
    while len(keep) > 2:
        it += 1; best = None
        for c in keep:
            others = [o for o in keep if o != c]; Mo = D.mu[:, [idx[o] for o in others]]; y = D.mu[:, idx[c]]; lam = 1e3
            f, _ = nnls(np.vstack([Mo, lam * np.ones((1, len(others)))]), np.concatenate([y, [lam]]))
            fit = Mo @ f; r = float(np.corrcoef(y, fit)[0, 1])
            comp = {o: round(float(w), 3) for o, w in zip(others, f) if w >= 0.02}
            rows.append(dict(iteration=it, cell=c, r_mixture=r, rms=float(np.sqrt(((y - fit) ** 2).mean())), mean_abs=float(np.abs(y - fit).mean()), components=json.dumps(comp)))
            if best is None or r > best[1]: best = (c, r, comp)
        if best[1] < r_thr: break
        removed[best[0]] = best[2]; keep.remove(best[0])
    return removed, rows


def crlb(D, f=None, sigma=0.02):
    """Fraction SD bound for solver D's design at the markers (one array, noise v + sigma^2, equality sum f = 1 by elimination)."""
    M = D.mu; K = M.shape[1]; f = np.full(K, 1.0 / K) if f is None else f
    w = 1.0 / ((D.v * f[None, :] ** 2).sum(1) + sigma ** 2)
    F = M.T @ (w[:, None] * M)
    # constrained: f = f0 + Z t with Z an orthonormal basis of {sum = 0}
    Z = np.linalg.svd(np.ones((1, K)))[2][1:].T
    C = Z @ np.linalg.pinv(Z.T @ F @ Z) @ Z.T
    sd = np.sqrt(np.clip(np.diag(C), 0, None)); R = C / np.outer(sd + 1e-12, sd + 1e-12)
    return sd, R


class NNLS8:
    def __init__(self, bc):
        self.groups = bc["groups"]; self.markers = pd.Index(bc["markers"]); self.M = pd.DataFrame(bc["mu_markers"], index=bc["markers"])[self.groups]
    def deconvolve(self, beta):
        y = beta.reindex(self.markers); ok = y.notna().values
        f, _ = nnls(self.M.values[ok], y.values[ok]); f = f / f.sum() if f.sum() > 0 else f
        return dict(fractions=dict(zip(self.groups, map(float, f))), residual_mae=float(np.abs(y.values[ok] - self.M.values[ok] @ f).mean()), n_markers_used=int(ok.sum()))


def ledoit_wolf(X):
    """Analytic Ledoit-Wolf shrinkage of the sample covariance of rows of X (n x p, centred) toward mu*I. Parameter-free."""
    n, p = X.shape; S = X.T @ X / n; mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    b2 = min(d2, sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / n ** 2 / p)
    a = b2 / d2 if d2 > 0 else 1.0
    return a * mu * np.eye(p) + (1 - a) * S, float(a)


class NILC:
    """Constrained ILC. mu, v: sites x K arrays (templates, template variance). cov: 'atlas' (diag mean_k v + sigma^2),
    'atlas+tech' (diag v + technical variance), 'atlas+techLW' (diag v + Ledoit-Wolf technical covariance). offset: deproject a constant."""
    def __init__(self, sites, cells, mu, v, cov="atlas", sigma=0.02, tech_var=None, tech_cov=None, offset=False):
        self.sites = pd.Index(sites); self.cells = list(cells); self.mu = np.asarray(mu, float); self.v = np.asarray(v, float)
        self.cov = cov; self.sigma = sigma; self.tech_var = None if tech_var is None else np.asarray(tech_var, float)
        self.tech_cov = tech_cov; self.offset = offset
    def _solve(self, m, f):
        A = self.mu[m]; tv = (self.v[m] * f[None, :] ** 2).sum(1)
        if self.offset: A = np.hstack([A, np.ones((A.shape[0], 1))])
        if self.cov == "atlas":
            c = tv + self.sigma ** 2; CiA = A / c[:, None]
        elif self.cov == "atlas+tech":
            c = tv + self.tech_var[m]; CiA = A / c[:, None]
        else:
            C = self.tech_cov[np.ix_(m, m)] + np.diag(tv); CiA = np.linalg.solve(C, A)
        G = np.linalg.inv(A.T @ CiA); W = CiA @ G
        return W, np.sqrt(np.clip(np.diag(G), 0, None))
    def deconvolve(self, beta, passes=2):
        y = beta.reindex(self.sites).values.astype(float); m = np.isfinite(y); K = len(self.cells)
        f = np.full(K, 1.0 / K)
        for _ in range(passes):
            W, sd = self._solve(m, f); est = W.T @ y[m]; f = np.clip(est[:K], 0, 1)
        fr = est[:K]
        return dict(fractions=dict(zip(self.cells, map(float, fr))), noise_sd=dict(zip(self.cells, map(float, sd[:K]))),
                    offset=(float(est[K]) if self.offset else None), sum=float(fr.sum()), n_sites_used=int(m.sum()),
                    residual_mae=float(np.abs(y[m] - self.mu[m] @ fr - (est[K] if self.offset else 0)).mean()))


def to_groups(fr):
    g = {}
    for c, v in fr.items(): g[grp(c)] = g.get(grp(c), 0.0) + v
    for c, v in fr.items():
        if c in SUBGROUP: g[SUBGROUP[c]] = g.get(SUBGROUP[c], 0.0) + v
    g["GRAN"] = g.get("NEU", 0) + g.get("EOS", 0) + g.get("BASO", 0); g["BNK"] = g.get("B", 0) + g.get("NK", 0)
    return g


def met_a(beta, fractions, profiles, ns):
    """Whole-blood Met-A = mean H(beta) / mean H(e) at the neutrophil sites, e = sum f_g mu_g; profiles: dict cell -> Series over ns.
    Cells with f < 1e-4 are skipped; sites unmeasured in any used profile are dropped."""
    use = {c: f for c, f in fractions.items() if f >= 1e-4 and c in profiles}
    if not use: return None, 0, 0.0
    tot = sum(use.values()); e = sum((f / tot) * profiles[c] for c, f in use.items())
    x = beta.reindex(ns); ok = x.notna() & e.notna()
    return float(H(x[ok].values).mean() / H(e[ok].values).mean()), int(ok.sum()), float(tot)


def read_beta(path):
    p = glob.glob(path); assert len(p) == 1, (path, p)
    b = pd.read_parquet(p[0]).iloc[:, 0].astype("float64"); b.index = b.index.astype(str); return b
