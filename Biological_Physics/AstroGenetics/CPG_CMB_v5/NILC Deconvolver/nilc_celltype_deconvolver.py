#!/usr/bin/env python3
"""
nilc_celltype_deconvolver.py — NILC at the CELL-TYPE level.

The class-level NILC (nilc_deconvolver-2.py) builds its reference from the 8 class
means and stops there. This module takes the SAME inverse-variance, departure-from-
consensus GLS down to the 115 cell types, weighting each cell by its OWN posterior SD
from the rebuilt IAMAtlas. It exists so the two deconvolvers can agree CELL-BY-CELL
(Walther's constrained NNLS ∩ NILC's variance-weighted GLS), not just class-by-class.

Why this is meaningful now and was not before: the rebuilt IAMAtlas is separable at the
cell level (see IAMAtlas_FLATNESS_LESSON.md) and carries per-cell mean + posterior SD
columns (`<Cell>_mean`, `<Cell>_sd`) — the covariances the MCMC produced. The old flat
atlas could not have supported this; the rebuild can.

PER-SAMPLE ONLY. No cohort, no population. Each cell is read against its own atlas
posterior. Returns per-cell fractions; presence is decided by agreement with Walther.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path


class NILCCelltypeDeconvolver:
    def __init__(self, atlas_path, marker_path, ridge_lambda=1e-3,
                 min_markers=200, top_markers_per_cell=100, verbose=False):
        self.atlas_path = str(atlas_path)
        self.marker_path = str(marker_path)
        self.ridge_lambda = ridge_lambda
        self.min_markers = min_markers
        self.top_markers_per_cell = top_markers_per_cell
        self.verbose = verbose
        self._load()

    def _load(self):
        md = json.load(open(self.marker_path))
        mbc = md.get("markers_by_celltype") or md.get("markers") or {}
        self.celltypes = sorted(mbc.keys())
        # Drop bulk/aggregate pseudo-cells at the CELL level (whole_blood, PBMC,
        # granulocytes, plasma, ...): they absorb fraction mass and starve real
        # cells. The class tier resolves these via class columns, not cell columns,
        # so this only affects cell resolution. Matches the chain's _AGG filter.
        _BULK = ("pbmc", "whole_blood", "buffy", "leukocyte", "_blood", "blood_",
                 "plasma", "granulocytes", "mononuclear", "wbc", "bulk")
        self.celltypes = [c for c in self.celltypes
                          if not any(b in c.lower() for b in _BULK)]
        self.celltype_to_class = md.get("celltype_to_class", {})
        # marker pool: union of each kept cell's top-N markers
        pool = set()
        for ct in self.celltypes:
            pool.update(mbc.get(ct, [])[:self.top_markers_per_cell])
        markers = sorted(pool)

        mean_cols = [f"{c}_mean" for c in self.celltypes]
        sd_cols = [f"{c}_sd" for c in self.celltypes]
        comp = "xz" if self.atlas_path.endswith(".xz") else None
        atlas = pd.read_csv(self.atlas_path, usecols=["cpg_id"] + mean_cols + sd_cols,
                            compression=comp)
        atlas = atlas[atlas["cpg_id"].isin(markers)].copy()

        # Impute missing posterior means with the column median; mark via inflated SD so
        # the GLS down-weights them automatically (same policy as the class-level NILC).
        for c in self.celltypes:
            mc, sc = f"{c}_mean", f"{c}_sd"
            nan = atlas[mc].isna()
            if nan.sum() > 0:
                atlas.loc[nan, mc] = atlas[mc].median()
                atlas.loc[nan, sc] = atlas[sc].max()
            med_sd = atlas[sc].median()
            atlas[sc] = atlas[sc].fillna(med_sd if med_sd > 0 else 0.05).clip(lower=1e-3)

        self.cpg_ids = atlas["cpg_id"].values
        self.X = atlas[mean_cols].values            # (m × 115) per-cell means
        self.SIGMA = atlas[sd_cols].values          # (m × 115) per-cell posterior SDs
        if self.verbose:
            print(f"[NILC-cell] {len(self.celltypes)} cells, {len(self.cpg_ids)} marker CpGs")

    @staticmethod
    def _project_simplex(v):
        u = np.sort(v)[::-1]
        css = np.cumsum(u)
        K = len(v)
        idx = np.nonzero(u * np.arange(1, K + 1) > (css - 1))[0]
        if len(idx) == 0:
            return np.maximum(v, 0)
        rho = idx[-1]
        theta = (css[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0.0)

    def deconvolve(self, beta_dict, presence_bootstrap=True, n_boot=200,
                   presence_seed=0, presence_ci=(2.5, 97.5), presence_zero=1e-6):
        bv = {k: v for k, v in beta_dict.items()
              if v is not None and not (isinstance(v, float) and np.isnan(v))}
        mask = np.array([c in bv for c in self.cpg_ids])
        K = len(self.celltypes)
        if mask.sum() < self.min_markers:
            return {"status": "INSUFFICIENT_MARKERS", "fractions": {},
                    "n_markers_used": int(mask.sum())}
        X = self.X[mask]
        SIGMA = self.SIGMA[mask]
        beta = np.array([bv[c] for c in self.cpg_ids[mask]])

        # departure-from-consensus: subtract the per-CpG mean across cells from both the
        # reference and the patient, orthogonalizing the columns (each row sums to ~0).
        consensus = X.mean(axis=1)
        Xd = X - consensus[:, None]
        bd = beta - consensus
        # inverse-variance weights from each cell's own posterior SD (vector, never m×m)
        sigma2 = np.maximum((SIGMA ** 2).mean(axis=1), 1e-6)
        w = 1.0 / sigma2

        XtWX = Xd.T @ (w[:, None] * Xd)
        ridge = self.ridge_lambda * np.trace(XtWX) / K * np.eye(K)
        try:
            f_dep = np.linalg.solve(XtWX + ridge, Xd.T @ (w * bd))
        except np.linalg.LinAlgError:
            return {"status": "ILL_CONDITIONED", "fractions": {},
                    "n_markers_used": int(mask.sum())}
        f_raw = np.full(K, 1.0 / K) + f_dep
        f_proj = self._project_simplex(f_raw)

        beta_recon = X @ f_proj
        residual_mae = float(np.mean(np.abs(beta - beta_recon)))

        # --- cell-level presence gate (bootstrap marker resample) -------------
        # Resample the matched marker rows with replacement, re-solve the GLS and
        # re-project to the simplex, and call a cell PRESENT iff the lower bound of
        # its bootstrap fraction CI > presence_zero. Mirrors the Walther-cell gate
        # so the two can be intersected cell-by-cell (the agreement gate).
        present = {}
        fraction_ci = {}
        m = Xd.shape[0]
        if presence_bootstrap and n_boot and m >= 1:
            rng = np.random.default_rng(presence_seed)
            boot = np.zeros((int(n_boot), K))
            for b in range(int(n_boot)):
                idx = rng.integers(0, m, m)
                Xb, bb, wb = Xd[idx], bd[idx], w[idx]
                XtWXb = Xb.T @ (wb[:, None] * Xb)
                ridgeb = self.ridge_lambda * np.trace(XtWXb) / K * np.eye(K)
                try:
                    fdb = np.linalg.solve(XtWXb + ridgeb, Xb.T @ (wb * bb))
                except np.linalg.LinAlgError:
                    boot[b] = f_proj
                    continue
                boot[b] = self._project_simplex(np.full(K, 1.0 / K) + fdb)
            lo_p, hi_p = presence_ci
            los = np.percentile(boot, lo_p, axis=0)
            his = np.percentile(boot, hi_p, axis=0)
            for i, c in enumerate(self.celltypes):
                fraction_ci[c] = [float(los[i]), float(his[i])]
                present[c] = bool(los[i] > presence_zero)
            presence_method = f"bootstrap_marker_resample_n{int(n_boot)}_ci{lo_p}-{hi_p}"
        else:
            for i, c in enumerate(self.celltypes):
                fraction_ci[c] = [float(f_proj[i]), float(f_proj[i])]
                present[c] = bool(f_proj[i] > presence_zero)
            presence_method = "point_fraction_no_bootstrap"

        return {
            "status": "OK",
            "fractions": {c: float(f_proj[i]) for i, c in enumerate(self.celltypes)},
            "raw_fractions": {c: float(f_raw[i]) for i, c in enumerate(self.celltypes)},
            "present": present,
            "fraction_ci": fraction_ci,
            "presence_method": presence_method,
            "n_present": int(sum(present.values())),
            "residual_mae": residual_mae,
            "n_markers_used": int(mask.sum()),
            "n_markers_total": len(self.cpg_ids),
        }
