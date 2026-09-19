#!/usr/bin/env python3
"""
age_axis_foreground.py — L4 age-axis foreground module (Phase B3)

Formalizes the CPG-VAL-007 seed (age-axis subtraction) as a reusable foreground module
conforming to the v1 foreground_registry.py interface:

    age_module.fit(beta_matrix, ages, hc_mask)    # train on HC samples
    age_module.subtract_from(beta_matrix, ages)   # remove age component → cleaned β

This is L4 of the chain of custody: foreground subtraction. Age is one of the foreground
axes the CMB community calls "secondaries" — sources of structure in the observed signal
that are not the cosmological (disease-relevant, in our case) signal we care about.

Per-CpG age model
-----------------
For each CpG i, fit:

    β_i(a) = α_i + γ_i * a + ε_i

where a is donor age, α_i is the per-CpG intercept (β at age 0, extrapolated), γ_i is
the per-CpG age slope, and ε_i is the residual (the signal AFTER age subtraction).

Training is on HC samples only — case samples are excluded so disease signal doesn't
contaminate the age regression. Once trained, the same (α_i, γ_i) is applied to ALL
samples (HC and case) to subtract the age component.

Output of subtract_from(β, age):

    β_cleaned[i, sample] = β[i, sample] - γ_i * age[sample]

Note: intercept α_i is NOT subtracted; the goal is to remove the AGE-DEPENDENT structure
while preserving the per-CpG baseline. After subtraction, β_cleaned at age=0 equals
β + 0 = β (intercept preserved); β_cleaned at age=50 equals β - 50*γ (age component removed).

This is the standard "regress out a covariate" move from epidemiology, applied per-CpG.

Per-CpG age layer artifact
--------------------------
After fitting, the module can emit `IAMAtlas_age_layer.csv` per the v1 Roadmap spec:

    cpg_id, intercept_alpha, slope_gamma, r_squared, n_samples

This file lives alongside the IAMAtlas REBUILD as a reusable per-CpG annotation, usable
by any downstream analysis that needs age-adjusted β values.

Robustness against the CPG-VAL-007 seed
---------------------------------------
The CPG-VAL-007 seed computed an 8-class age axis at the A-score level and projected it
out before computing the Mahalanobis distance. This module operates one level deeper —
at the per-CpG β level — which is what L4 component-separation requires. The two are
compatible: this module can be composed with the A-score-level age axis as a two-stage
treatment (β-level age subtraction first, then A-score-level residual age axis).

Usage
-----
    from age_axis_foreground import AgeAxisForeground
    age_fg = AgeAxisForeground()
    age_fg.fit(beta_matrix, ages_array, hc_mask)  # train on HC
    cleaned = age_fg.subtract_from(beta_matrix, ages_array)  # apply to all
    age_fg.save_layer('/path/to/IAMAtlas_age_layer.csv')

CLI: see __main__ at end.
"""

import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd


@dataclass
class AgeFitDiagnostics:
    """Diagnostics from age-axis fitting."""
    n_cpgs_fitted: int
    n_cpgs_converged: int          # fits with successful slope estimation
    n_samples_used: int            # HC samples used for fitting
    age_range: tuple
    slope_distribution: dict       # min, 25%, median, 75%, max of slopes
    r_squared_distribution: dict
    top_age_correlated_cpgs: list  # top 10 by |slope| × √n (informative ones)
    convergence_threshold_pct: float

    def to_dict(self):
        return asdict(self)


class AgeAxisForeground:
    """
    L4 foreground module for age-axis subtraction.

    Conforms to the v1 foreground_registry.py interface:
        - fit(beta, ages, hc_mask)
        - subtract_from(beta, ages) → cleaned_beta
        - is_fitted: bool
        - save_layer(path) → write per-CpG slope/intercept artifact

    Per-CpG linear regression on HC samples only. Excluded patients (NaN ages, ill-
    conditioned CpGs) are tracked in the diagnostics. The module fits and applies
    independently per CpG — no cross-CpG smoothing — so a low-R² CpG simply has a near-
    zero slope and contributes nothing to the cleaning, rather than being smoothed by
    neighboring high-R² CpGs.
    """

    def __init__(self, min_samples: int = 30, min_age_range: float = 10.0):
        self.min_samples = min_samples
        self.min_age_range = min_age_range
        self.is_fitted = False
        self.cpg_ids = None
        self.slopes = None       # γ_i per CpG (length M)
        self.intercepts = None   # α_i per CpG (length M)
        self.r_squared = None    # R² per CpG
        self.n_samples = None    # number of HC samples contributing per CpG
        self.diagnostics: Optional[AgeFitDiagnostics] = None

    def fit(self, beta_matrix: np.ndarray, ages: np.ndarray,
            hc_mask: np.ndarray, cpg_ids=None):
        """
        Fit per-CpG age regression on HC samples.

        Args:
            beta_matrix : (N × M) — N samples × M CpGs
            ages : (N,) — donor ages, NaN-tolerant
            hc_mask : (N,) bool — True for HC samples to use in training
            cpg_ids : optional (M,) list of CpG identifiers
        """
        if beta_matrix.shape[0] != len(ages):
            raise ValueError(f"beta rows ({beta_matrix.shape[0]}) != ages length ({len(ages)})")
        if beta_matrix.shape[0] != len(hc_mask):
            raise ValueError(f"beta rows ({beta_matrix.shape[0]}) != hc_mask length ({len(hc_mask)})")

        # Restrict to HC samples with non-NaN ages
        train_mask = hc_mask & ~np.isnan(ages)
        ages_train = ages[train_mask]
        beta_train = beta_matrix[train_mask]
        N_train = train_mask.sum()
        if N_train < self.min_samples:
            raise ValueError(f"Only {N_train} HC samples have valid ages; need >= {self.min_samples}")
        age_range = float(ages_train.max() - ages_train.min())
        if age_range < self.min_age_range:
            raise ValueError(f"Age range {age_range:.1f} too narrow; need >= {self.min_age_range}")

        M = beta_matrix.shape[1]
        self.cpg_ids = cpg_ids if cpg_ids is not None else np.arange(M)
        self.slopes = np.zeros(M)
        self.intercepts = np.zeros(M)
        self.r_squared = np.zeros(M)
        self.n_samples = np.zeros(M, dtype=int)

        # Per-CpG linear regression via numpy lstsq (vectorized where possible)
        # For NaN-tolerance per CpG, we loop — fast enough for ~7K CpGs × 600 samples
        a = ages_train
        a_mean = a.mean()
        a_centered = a - a_mean
        a_var = (a_centered ** 2).sum()

        n_converged = 0
        for j in range(M):
            b = beta_train[:, j]
            valid = ~np.isnan(b)
            n = valid.sum()
            self.n_samples[j] = n
            if n < self.min_samples or a_var == 0:
                self.slopes[j] = 0.0
                self.intercepts[j] = np.nanmean(b) if n > 0 else 0.5
                self.r_squared[j] = 0.0
                continue

            b_v = b[valid]
            a_v = a[valid]
            a_v_mean = a_v.mean()
            b_mean = b_v.mean()
            num = ((a_v - a_v_mean) * (b_v - b_mean)).sum()
            den = ((a_v - a_v_mean) ** 2).sum()
            if den == 0:
                self.slopes[j] = 0.0
                self.intercepts[j] = b_mean
                self.r_squared[j] = 0.0
                continue

            slope = num / den
            intercept = b_mean - slope * a_v_mean
            b_pred = intercept + slope * a_v
            ss_res = ((b_v - b_pred) ** 2).sum()
            ss_tot = ((b_v - b_mean) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            self.slopes[j] = slope
            self.intercepts[j] = intercept
            self.r_squared[j] = r2
            n_converged += 1

        # Build diagnostics
        slopes_abs = np.abs(self.slopes)
        informative_score = slopes_abs * np.sqrt(self.n_samples)  # weight by sample size
        top_idx = np.argsort(-informative_score)[:10]
        top_cpgs = [(str(self.cpg_ids[i]), float(self.slopes[i]),
                     float(self.r_squared[i]), int(self.n_samples[i])) for i in top_idx]

        self.diagnostics = AgeFitDiagnostics(
            n_cpgs_fitted=M,
            n_cpgs_converged=n_converged,
            n_samples_used=int(N_train),
            age_range=(float(ages_train.min()), float(ages_train.max())),
            slope_distribution={
                'min': float(self.slopes.min()),
                'p25': float(np.percentile(self.slopes, 25)),
                'median': float(np.median(self.slopes)),
                'p75': float(np.percentile(self.slopes, 75)),
                'max': float(self.slopes.max()),
                'mean_abs': float(slopes_abs.mean()),
            },
            r_squared_distribution={
                'min': float(self.r_squared.min()),
                'p25': float(np.percentile(self.r_squared, 25)),
                'median': float(np.median(self.r_squared)),
                'p75': float(np.percentile(self.r_squared, 75)),
                'max': float(self.r_squared.max()),
                'pct_above_0.1': float((self.r_squared > 0.1).mean() * 100),
                'pct_above_0.3': float((self.r_squared > 0.3).mean() * 100),
            },
            top_age_correlated_cpgs=top_cpgs,
            convergence_threshold_pct=100.0 * n_converged / M,
        )
        self.is_fitted = True
        return self

    def subtract_from(self, beta_matrix: np.ndarray, ages: np.ndarray) -> np.ndarray:
        """
        Apply the trained age axis to subtract age component from β.

        Returns:
            cleaned : (N × M) β with age component removed at each CpG.
        """
        if not self.is_fitted:
            raise RuntimeError("Module not fitted yet. Call .fit(...) first.")
        if beta_matrix.shape[1] != len(self.slopes):
            raise ValueError(f"beta cols ({beta_matrix.shape[1]}) != fitted slopes ({len(self.slopes)})")
        N = beta_matrix.shape[0]
        if len(ages) != N:
            raise ValueError(f"ages length ({len(ages)}) != beta rows ({N})")

        # cleaned[i, j] = β[i, j] - γ_j * (age[i] - age_mean_train)
        # Subtracting just γ * age is also valid — it shifts the intercept but preserves
        # the age component's contribution. We use this formulation for stability.
        age_arr = np.where(np.isnan(ages), 0.0, ages)
        age_component = np.outer(age_arr, self.slopes)
        cleaned = beta_matrix - age_component

        # Where age was NaN, fall back to original β (no correction applied)
        nan_age_mask = np.isnan(ages)
        if nan_age_mask.any():
            cleaned[nan_age_mask] = beta_matrix[nan_age_mask]
        return cleaned

    def save_layer(self, path: str):
        """Save per-CpG age layer artifact (the IAMAtlas_age_layer.csv per the spec)."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted layer")
        df = pd.DataFrame({
            'cpg_id': self.cpg_ids,
            'intercept_alpha': self.intercepts,
            'slope_gamma': self.slopes,
            'r_squared': self.r_squared,
            'n_samples': self.n_samples,
        })
        df.to_csv(path, index=False)
        return path

    def save_diagnostics(self, path: str):
        if self.diagnostics is None:
            raise RuntimeError("No diagnostics; module not fitted.")
        with open(path, 'w') as f:
            json.dump(self.diagnostics.to_dict(), f, indent=2)
        return path


# =========================================================================
# CLI
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="Fit and apply per-CpG age-axis foreground")
    ap.add_argument("--beta-matrix", required=True,
                    help="CSV: rows=CpGs (first col cpg_id), cols=samples (one column 'gsm')")
    ap.add_argument("--metadata", required=True,
                    help="CSV with columns: gsm, age, arm")
    ap.add_argument("--hc-label", default="hc")
    ap.add_argument("--age-col", default="age")
    ap.add_argument("--arm-col", default="arm")
    ap.add_argument("--layer-out", required=True, help="Output age-layer CSV path")
    ap.add_argument("--diagnostics-out", help="Optional diagnostics JSON path")
    args = ap.parse_args()

    bdf = pd.read_csv(args.beta_matrix)
    cpg_col = bdf.columns[0]
    cpgs = bdf[cpg_col].values
    sample_cols = [c for c in bdf.columns if c != cpg_col]
    beta_matrix = bdf[sample_cols].values.T  # (N × M)

    meta = pd.read_csv(args.metadata)
    meta = meta.set_index('gsm')
    ages = np.array([meta.loc[s, args.age_col] if s in meta.index else np.nan
                     for s in sample_cols])
    arms = np.array([meta.loc[s, args.arm_col] if s in meta.index else 'unknown'
                     for s in sample_cols])
    hc_mask = np.array([str(a).lower() == args.hc_label for a in arms])

    age_fg = AgeAxisForeground()
    age_fg.fit(beta_matrix, ages, hc_mask, cpg_ids=cpgs)
    age_fg.save_layer(args.layer_out)
    if args.diagnostics_out:
        age_fg.save_diagnostics(args.diagnostics_out)

    print(f"Age layer saved: {args.layer_out}")
    print(f"CpGs fitted: {age_fg.diagnostics.n_cpgs_fitted}")
    print(f"CpGs converged: {age_fg.diagnostics.n_cpgs_converged} "
          f"({age_fg.diagnostics.convergence_threshold_pct:.1f}%)")
    print(f"Top age-correlated CpGs:")
    for cpg, slope, r2, n in age_fg.diagnostics.top_age_correlated_cpgs[:5]:
        print(f"  {cpg}  slope={slope:+.5f}/yr  R²={r2:.3f}  n={n}")


if __name__ == "__main__":
    main()
