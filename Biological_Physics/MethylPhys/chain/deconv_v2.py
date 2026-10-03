#!/usr/bin/env python3
# Toolkit: not yet wired into chain v3; enters the chain at commissioning with its own pre-registered check.  (SOP v3 section 2b, stage 3; chain/TOOLKIT.md)
"""IAM-Atlas v2 composition solver (doors/PROC_DECONV_V2_01_PREREG.md). Built on atlas v2 alone: no v1 file, panel, name or code.
Reference mu and variance v = mu_sd^2 + donor_sd^2 per cell and locus; loci measured by every modelled cell; per-cell markers
chosen against the NEAREST other cell (margin >= 0.20, top 200 by margin / pooled SD); one joint solve, f >= 0, sum f = 1,
weights 1 / (sum_c f_c^2 v_ci + sigma^2) iterated 5 times; presence by 200 bootstrap resamples of the marker loci."""
import json, numpy as np, pandas as pd
from scipy.optimize import nnls

class DeconvV2:
    def __init__(self, atlas_parquet, pooled_one_sample=(), margin=0.20, top=200, min_markers=20, sigma=0.02, drop=(), markers="nearest", k_near=6, per_pair=40, pair_margin=0.10, A=None):
        cols = pd.read_parquet(atlas_parquet, columns=None).columns if False else None
        A = pd.read_parquet(atlas_parquet) if A is None else A
        self.cells = sorted({c.rsplit("_", 1)[0] for c in A.columns if c.endswith("_mean")} - set(drop))
        mu = A[[f"{c}_mean" for c in self.cells]].to_numpy(np.float64)
        v = (A[[f"{c}_sd" for c in self.cells]].to_numpy(np.float64) ** 2 + A[[f"{c}_donor_sd" for c in self.cells]].to_numpy(np.float64) ** 2)
        nobs = A[[f"{c}_n_obs" for c in self.cells]].to_numpy()
        need = np.array([1 if c in set(pooled_one_sample) else 2 for c in self.cells])
        ok = np.all(nobs >= need[None, :], axis=1) & np.all(np.isfinite(mu), axis=1) & np.all(np.isfinite(v), axis=1)
        self.loci_all = A.index[ok].astype(str).values; mu = mu[ok]; v = v[ok]
        K = len(self.cells); self.sep = {}; pick = set()
        if markers in ("pairwise", "hybrid"):
            # every cell against each of its k nearest cells (mean |difference| over all measured loci): the loci that tell THAT pair apart
            D = np.array([[np.abs(mu[:, a] - mu[:, b]).mean() for b in range(K)] for a in range(K)]); np.fill_diagonal(D, np.inf)
            for k in range(K):
                near = np.argsort(D[k])[:k_near]; n_k = 0
                for j in near:
                    d = np.abs(mu[:, k] - mu[:, j]); pool = np.sqrt((v[:, k] + v[:, j]) / 2.0)
                    cand = np.where(d >= pair_margin)[0]; sel = cand[np.argsort(-(d[cand] / pool[cand]))[:per_pair]]
                    pick.update(sel.tolist()); n_k += len(sel)
                self.sep[self.cells[k]] = dict(n_markers=int(n_k), nearest_cell_overall=self.cells[near[0]], separable=bool(len(np.where(np.abs(mu[:, k] - mu[:, near[0]]) >= pair_margin)[0]) >= min_markers))
        if markers in ("nearest", "hybrid"):
            for k in range(K):
                d = mu[:, [j for j in range(K) if j != k]] - mu[:, [k]]
                ad = np.abs(d); jn = ad.argmin(axis=1); near = ad[np.arange(len(ad)), jn]
                others = [j for j in range(K) if j != k]
                pool = np.sqrt((v[:, k] + v[np.arange(len(v)), np.array(others)[jn]]) / 2.0)
                cand = np.where(near >= margin)[0]
                score = near[cand] / pool[cand]; sel = cand[np.argsort(-score)[:top]]
                if markers == "nearest":
                    self.sep[self.cells[k]] = dict(n_markers=int(len(sel)), nearest_cell_overall=self.cells[others[np.bincount(jn).argmax()]], separable=bool(len(sel) >= min_markers))
                pick.update(sel.tolist())
        idx = np.array(sorted(pick)); self.loci = self.loci_all[idx]; self.mu = mu[idx]; self.v = v[idx]; self.sigma = sigma
        self.meta = dict(n_cells=K, n_loci_all_measured=int(ok.sum()), n_markers=int(len(idx)), margin=margin, top=top, sigma=sigma, drop=list(drop), markers=markers, k_near=k_near, per_pair=per_pair, pair_margin=pair_margin)

    def _solve(self, y, M, V):
        K = M.shape[1]; f = np.full(K, 1.0 / K); lam = 1e3
        for _ in range(5):
            w = 1.0 / ((V * f[None, :] ** 2).sum(axis=1) + self.sigma ** 2); sw = np.sqrt(w)
            Aa = np.vstack([M * sw[:, None], lam * np.ones((1, K))]); bb = np.concatenate([y * sw, [lam]])
            f, _ = nnls(Aa, bb, maxiter=5000); s = f.sum(); f = f / s if s > 0 else np.full(K, 1.0 / K)
        return f

    def deconvolve(self, beta: pd.Series, n_boot=200, seed=0, present_floor=0.005, affine=False):
        y = beta.reindex(self.loci).to_numpy(np.float64); m = np.isfinite(y)
        M, V, yy = self.mu[m], self.v[m], y[m]
        a, b = 0.0, 1.0
        if affine:   # the specimen's own scale: beta_obs = a + b * (M f), fitted from this specimen alone, alternating with f
            f = self._solve(yy, M, V)
            for _ in range(8):
                p = M @ f; b, a = np.polyfit(p, yy, 1)
                f = self._solve(np.clip((yy - a) / b, 1e-3, 1 - 1e-3), M, V)
            yy = np.clip((yy - a) / b, 1e-3, 1 - 1e-3)
        else:
            f = self._solve(yy, M, V)
        rng = np.random.default_rng(seed); B = np.zeros((max(n_boot,1), len(f))) + f
        for b in range(n_boot):
            i = rng.integers(0, len(yy), len(yy)); B[b] = self._solve(yy[i], M[i], V[i])
        lo, hi = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
        return dict(n_markers_used=int(m.sum()), residual_mae=float(np.abs(yy - M @ f).mean()), scale=dict(a=float(a), b=float(b)),
                    fractions={c: float(f[k]) for k, c in enumerate(self.cells)},
                    ci={c: [float(lo[k]), float(hi[k])] for k, c in enumerate(self.cells)},
                    present={c: bool(lo[k] > present_floor) for k, c in enumerate(self.cells)})
