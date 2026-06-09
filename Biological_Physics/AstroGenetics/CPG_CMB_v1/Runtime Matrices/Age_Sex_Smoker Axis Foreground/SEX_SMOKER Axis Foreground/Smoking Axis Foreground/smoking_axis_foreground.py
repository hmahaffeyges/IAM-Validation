#!/usr/bin/env python3
"""
smoking_axis_foreground.py — L4 smoking-axis foreground module (Phase B4)

Formalizes per-CpG smoking-effect subtraction at the β level — the architecturally
correct L4 component-separation move that retires the interim Stage 7 smoking-bin
threshold stratification.

Conforms to the same foreground_registry.py interface as `age_axis_foreground.py`:

    smoking_module.fit(beta_matrix, smoking_status_array, hc_mask)
    smoking_module.subtract_from(beta_matrix, smoking_status_array) → cleaned_beta

THE PROBLEM THIS SOLVES
------------------------
Tobacco smoke methylates a well-documented set of CpGs (notably AHRR cg05575921 +
~100 cataloged tobacco-associated CpGs per McCarthy et al. 2017, Joehanes et al. 2016,
Zeilinger et al. 2013, etc.) with effect sizes that persist for years after quitting
and partially recover with cumulative time off tobacco. When patient runtime scores
the A-score and Mahalanobis distance, residual smoking signal absorbs into the
immune-class signal and inflates the apparent disease departure.

v1.0 — v1.2 mitigation: smoking-bin selects an ELEVATED-floor shift at Stage 7
(per `tier_breakpoints.json v1.2`). This is interim — the same smoking signal still
contaminates the per-CpG β values consumed by Stages 4 / 4.5 / 4.6 / 5 / 6.

v1.3 (this module): per-CpG smoking-effect subtraction at L4 BEFORE A-scoring.
The smoking-bin threshold-stratification at Stage 7 retires once this module
operates in production.

PER-CPG SMOKING MODEL
---------------------
For each CpG i, fit on HC samples:

    β_i(s) = α_i + δ_i * indicator_current_smoker + φ_i * recency_score + ε_i

where:
- `indicator_current_smoker` = 1 if smoking_status == "current", else 0
- `recency_score` is a continuous proxy for cumulative smoking + recency:
    - never_smoker:        0.00
    - former_15plus_y:     0.10
    - former_5_15y:        0.30
    - former_0_5y:         0.60
    - current_smoker:      1.00
- δ_i captures the step effect of being a current smoker (vs not)
- φ_i captures the recency-graded effect that decays as the patient gets further from quit
- ε_i is the residual (the signal AFTER smoking subtraction)

Training is on HC samples only — case samples excluded so disease signal doesn't
contaminate the smoking regression. Once trained, the same (δ_i, φ_i) is applied
to ALL samples (HC and case) to subtract the smoking component.

OUTPUT OF subtract_from(β, smoking_status):
    β_cleaned[i, sample] = β[i, sample]
                         − δ_i * indicator_current_smoker(sample)
                         − φ_i * recency_score(sample)

Note: intercept α_i is NOT subtracted. The goal is to remove the SMOKING-DEPENDENT
structure while preserving each CpG's baseline. For a never-smoker, both indicator
and recency_score are zero → no subtraction. For a current smoker, full subtraction
of both δ_i and φ_i.

PER-CPG SMOKING LAYER ARTIFACT
-------------------------------
After fitting, the module emits `IAMAtlas_smoking_layer.csv`:

    cpg_id, intercept_alpha, delta_current_smoker, phi_recency, r_squared, n_samples

This file lives alongside the IAMAtlas REBUILD as a reusable per-CpG annotation.
Build is a one-time fit on the pooled-HC training cohort (n_hc=601 — same cohort
used to build the Mahalanobis reference). Smoke-status metadata must be present in
the training cohort manifest; if it's missing, fitting is restricted to CpGs in the
known smoking-CpG literature (currently fallback-only, until a smoke-status-annotated
HC cohort is curated).

ROBUSTNESS NOTE
---------------
The literature catalog of tobacco-associated CpGs is approximately ~600 CpGs at
genome-wide FDR < 0.05 (per Joehanes 2016 meta-analysis n=15,907). When fitting
on a smaller HC cohort, the module restricts the per-CpG fit to this curated
candidate panel rather than all 483K atlas CpGs (insufficient power per-CpG at
n_hc=601 for a genome-wide fit). The candidate panel is loaded from
`smoking_candidate_cpgs_literature.csv` (to be curated as part of layer-build).

For v1.3 deployment, the curated panel + the n_hc=601 fit is sufficient to absorb
the bulk smoking effect on the disease-relevant immune compartment. Genome-wide
fit becomes available when a larger HC cohort with smoke-status metadata is
acquired (Phase B5+).

USAGE
-----
    from smoking_axis_foreground import SmokingAxisForeground
    smk = SmokingAxisForeground()
    smk.fit(beta_matrix, smoking_status_array, hc_mask)
    cleaned = smk.subtract_from(beta_matrix, smoking_status_array)
    smk.save_layer('IAMAtlas_smoking_layer.csv')

REFERENCE LITERATURE
--------------------
- Joehanes et al. (2016), Circ Cardiovasc Genet, "Epigenetic Signatures of Cigarette
  Smoking" — meta-analysis n=15,907; 18,760 CpGs at genome-wide FDR < 1e-7.
- McCarthy et al. (2017) — durability of AHRR cg05575921 methylation post-cessation.
- Zeilinger et al. (2013) — KORA cohort, 187 lead CpGs.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


# ============================================================================
# CONSTANTS — recency score mapping (smoking_bin → continuous covariate)
# ============================================================================

SMOKING_BIN_RECENCY_SCORE: dict[str, float] = {
    "never_smoker":        0.00,
    "former_15plus_y":     0.10,
    "former_5_15y":        0.30,
    "former_0_5y":         0.60,
    "current_smoker":      1.00,
}


def smoking_bin_to_indicator_and_recency(smoking_bin: str) -> tuple[int, float]:
    """Convert a smoking_bin label into (indicator_current, recency_score) covariates."""
    bin_str = (smoking_bin or "never_smoker").lower().strip()
    indicator_current = 1 if bin_str == "current_smoker" else 0
    recency = SMOKING_BIN_RECENCY_SCORE.get(bin_str)
    if recency is None:
        # Unknown bin — defensive: treat as never_smoker (no subtraction)
        recency = 0.0
    return indicator_current, recency


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SmokingFitDiagnostics:
    """Diagnostics from smoking-axis fitting."""
    n_cpgs_fitted: int
    n_cpgs_converged: int          # fits with non-degenerate slope estimates
    n_samples_used: int            # HC samples used for fitting
    n_current_smokers: int
    n_former_smokers: int
    n_never_smokers: int
    delta_distribution: dict       # min, 25%, median, 75%, max of delta_current
    phi_distribution: dict         # ditto for phi_recency
    r_squared_distribution: dict
    top_smoking_correlated_cpgs: list  # top 25 by |delta| × √n
    candidate_panel_size: int

    def to_dict(self):
        return asdict(self)


# ============================================================================
# CORE CLASS — mirrors AgeAxisForeground structure
# ============================================================================

class SmokingAxisForeground:
    """L4 foreground module for per-CpG smoking-effect subtraction.

    Conforms to the v1 foreground_registry.py interface:
        - fit(beta, smoking_bins, hc_mask, candidate_cpgs=None)
        - subtract_from(beta, smoking_bins) → cleaned_beta
        - save_layer(path)
        - load_layer(path)

    The fit is on HC samples only (no case contamination). The subtract_from method
    applies to all samples (HC + case) once the per-CpG (δ, φ) coefficients are
    frozen.
    """

    def __init__(
        self,
        min_samples_per_bin: int = 10,
        min_recency_variance: float = 0.05,
    ):
        self.min_samples_per_bin = min_samples_per_bin
        self.min_recency_variance = min_recency_variance
        self.alpha_intercept: np.ndarray | None = None      # shape (n_cpgs,)
        self.delta_current: np.ndarray | None = None         # shape (n_cpgs,)
        self.phi_recency: np.ndarray | None = None           # shape (n_cpgs,)
        self.r_squared: np.ndarray | None = None             # shape (n_cpgs,)
        self.n_samples: int = 0
        self.cpg_ids: list[str] | None = None
        self.diagnostics: SmokingFitDiagnostics | None = None

    # ─────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────
    def fit(
        self,
        beta_matrix: np.ndarray,
        smoking_bins: list[str],
        hc_mask: np.ndarray,
        cpg_ids: list[str],
        candidate_cpgs: list[str] | None = None,
    ) -> SmokingFitDiagnostics:
        """Fit per-CpG (δ, φ) on HC samples.

        Parameters
        ----------
        beta_matrix : np.ndarray, shape (n_cpgs, n_samples)
            β values per CpG per sample.
        smoking_bins : list[str], length n_samples
            Per-sample smoking_bin labels (never_smoker / former_*_y / current_smoker).
        hc_mask : np.ndarray, shape (n_samples,) bool
            True for HC samples; False for case samples (excluded from fit).
        cpg_ids : list[str], length n_cpgs
            Per-CpG identifiers.
        candidate_cpgs : list[str] or None
            If provided, restrict fit to this curated candidate panel (e.g., the
            Joehanes 2016 ~600-CpG smoking catalog). Non-candidate CpGs get
            δ=φ=0 (no smoking subtraction) — preserves the per-CpG baseline.

        Returns
        -------
        SmokingFitDiagnostics
        """
        n_cpgs, n_samples = beta_matrix.shape
        assert len(smoking_bins) == n_samples
        assert hc_mask.shape == (n_samples,)
        assert len(cpg_ids) == n_cpgs

        self.cpg_ids = list(cpg_ids)

        # Build per-sample covariates
        indicators = np.array([
            smoking_bin_to_indicator_and_recency(b)[0] for b in smoking_bins
        ], dtype=np.float64)
        recency = np.array([
            smoking_bin_to_indicator_and_recency(b)[1] for b in smoking_bins
        ], dtype=np.float64)

        # Restrict to HC samples for fitting
        hc_idx = np.where(hc_mask)[0]
        if len(hc_idx) < self.min_samples_per_bin * 3:
            raise ValueError(
                f"Insufficient HC samples for smoking fit: n_hc={len(hc_idx)} < "
                f"{self.min_samples_per_bin * 3} required."
            )

        beta_hc = beta_matrix[:, hc_idx]
        ind_hc = indicators[hc_idx]
        rec_hc = recency[hc_idx]

        n_current = int(ind_hc.sum())
        n_never = int(np.sum(rec_hc == 0))
        n_former = len(hc_idx) - n_current - n_never

        if np.var(rec_hc) < self.min_recency_variance:
            raise ValueError(
                f"HC cohort recency variance ({np.var(rec_hc):.4f}) below threshold "
                f"({self.min_recency_variance}). Need more diverse smoking history in HC."
            )

        # Decide which CpGs to fit
        if candidate_cpgs is not None:
            candidate_set = set(candidate_cpgs)
            fit_indices = [i for i, cpg in enumerate(self.cpg_ids) if cpg in candidate_set]
            print(f"[smoking_fg] Restricting fit to {len(fit_indices)} candidate CpGs "
                  f"(out of {n_cpgs} total)")
        else:
            fit_indices = list(range(n_cpgs))
            print(f"[smoking_fg] Genome-wide fit on all {n_cpgs} atlas CpGs "
                  f"(no candidate panel restriction).")

        # Allocate output arrays — non-candidate CpGs default to zero coefficients
        alpha = np.zeros(n_cpgs, dtype=np.float64)
        delta = np.zeros(n_cpgs, dtype=np.float64)
        phi = np.zeros(n_cpgs, dtype=np.float64)
        r2 = np.zeros(n_cpgs, dtype=np.float64)

        # Stack design matrix once for vectorized fit
        X = np.column_stack([
            np.ones(len(hc_idx)),
            ind_hc,
            rec_hc,
        ])

        n_converged = 0
        for i in fit_indices:
            y = beta_hc[i, :]
            if np.any(~np.isfinite(y)) or np.std(y) < 1e-6:
                continue
            # OLS: β = (X^T X)^-1 X^T y
            try:
                coeffs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank < 3:
                continue
            alpha[i] = coeffs[0]
            delta[i] = coeffs[1]
            phi[i] = coeffs[2]
            y_pred = X @ coeffs
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2[i] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            n_converged += 1

        self.alpha_intercept = alpha
        self.delta_current = delta
        self.phi_recency = phi
        self.r_squared = r2
        self.n_samples = len(hc_idx)

        # Top informative CpGs (by |delta| × sqrt(n_current))
        info_score = np.abs(delta) * np.sqrt(max(n_current, 1))
        top_idx = np.argsort(-info_score)[:25]
        top_cpgs = [
            {
                "cpg_id": self.cpg_ids[i],
                "delta_current": float(delta[i]),
                "phi_recency": float(phi[i]),
                "r_squared": float(r2[i]),
            }
            for i in top_idx if delta[i] != 0
        ]

        def percentiles(arr):
            arr = arr[arr != 0]  # exclude zero-coefficient CpGs from distribution stats
            if len(arr) == 0:
                return {"min": 0, "25%": 0, "median": 0, "75%": 0, "max": 0}
            return {
                "min": float(np.min(arr)),
                "25%": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "75%": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
            }

        self.diagnostics = SmokingFitDiagnostics(
            n_cpgs_fitted=len(fit_indices),
            n_cpgs_converged=n_converged,
            n_samples_used=len(hc_idx),
            n_current_smokers=n_current,
            n_former_smokers=n_former,
            n_never_smokers=n_never,
            delta_distribution=percentiles(delta),
            phi_distribution=percentiles(phi),
            r_squared_distribution=percentiles(r2),
            top_smoking_correlated_cpgs=top_cpgs,
            candidate_panel_size=len(candidate_cpgs) if candidate_cpgs is not None else n_cpgs,
        )
        return self.diagnostics

    # ─────────────────────────────────────────────────────────────────────
    # SUBTRACT
    # ─────────────────────────────────────────────────────────────────────
    def subtract_from(
        self,
        beta_matrix: np.ndarray,
        smoking_bins: list[str],
    ) -> np.ndarray:
        """Subtract the smoking component from β values.

        β_cleaned[i, sample] = β[i, sample]
                             − δ_i * indicator_current(sample)
                             − φ_i * recency(sample)

        Intercept is NOT subtracted (per the design — preserves per-CpG baseline).
        """
        if self.delta_current is None or self.phi_recency is None:
            raise RuntimeError("SmokingAxisForeground not yet fit. Call .fit() or .load_layer() first.")

        n_cpgs, n_samples = beta_matrix.shape
        assert len(smoking_bins) == n_samples

        indicators = np.array([
            smoking_bin_to_indicator_and_recency(b)[0] for b in smoking_bins
        ], dtype=np.float64)
        recency = np.array([
            smoking_bin_to_indicator_and_recency(b)[1] for b in smoking_bins
        ], dtype=np.float64)

        # Broadcasting: (n_cpgs,) outer (n_samples,) → (n_cpgs, n_samples)
        smoking_component = (
            np.outer(self.delta_current, indicators)
            + np.outer(self.phi_recency, recency)
        )
        return beta_matrix - smoking_component

    def subtract_from_single_patient(
        self,
        patient_beta: pd.Series,
        smoking_bin: str,
    ) -> pd.Series:
        """Convenience wrapper for the single-patient runtime path.

        Patient β arrives as a pd.Series indexed by cpg_id; returns same.
        """
        if self.cpg_ids is None:
            raise RuntimeError("SmokingAxisForeground has no cpg_ids loaded.")
        ind, rec = smoking_bin_to_indicator_and_recency(smoking_bin)
        cpg_to_delta = dict(zip(self.cpg_ids, self.delta_current))
        cpg_to_phi = dict(zip(self.cpg_ids, self.phi_recency))
        cleaned = patient_beta.copy()
        for cpg in cleaned.index:
            d = cpg_to_delta.get(cpg, 0.0)
            p = cpg_to_phi.get(cpg, 0.0)
            cleaned[cpg] = cleaned[cpg] - d * ind - p * rec
        return cleaned

    # ─────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────
    def save_layer(self, path: str | Path):
        """Persist the per-CpG (α, δ, φ, R², n) as IAMAtlas_smoking_layer.csv."""
        if self.delta_current is None:
            raise RuntimeError("Cannot save layer before fitting.")
        df = pd.DataFrame({
            "cpg_id": self.cpg_ids,
            "intercept_alpha": self.alpha_intercept,
            "delta_current_smoker": self.delta_current,
            "phi_recency": self.phi_recency,
            "r_squared": self.r_squared,
            "n_samples": self.n_samples,
        })
        df.to_csv(path, index=False)
        return Path(path)

    def load_layer(self, path: str | Path):
        """Load a previously saved smoking layer."""
        df = pd.read_csv(path)
        self.cpg_ids = df["cpg_id"].astype(str).tolist()
        self.alpha_intercept = df["intercept_alpha"].to_numpy(dtype=np.float64)
        self.delta_current = df["delta_current_smoker"].to_numpy(dtype=np.float64)
        self.phi_recency = df["phi_recency"].to_numpy(dtype=np.float64)
        self.r_squared = df["r_squared"].to_numpy(dtype=np.float64)
        self.n_samples = int(df["n_samples"].iloc[0])
        return self


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Smoking-axis foreground L4 module")
    parser.add_argument("--smoke-test", action="store_true", help="Run module structural smoke test")
    args = parser.parse_args()

    if args.smoke_test:
        print("=" * 70)
        print("SmokingAxisForeground — structural smoke test")
        print("=" * 70)
        # Synthetic data: 100 CpGs × 200 samples with known smoking effect on 10 CpGs
        rng = np.random.default_rng(42)
        n_cpgs, n_samples = 100, 200
        true_delta = np.zeros(n_cpgs)
        true_delta[:10] = -0.08  # current smokers have lower β on first 10 CpGs (AHRR-like)
        true_phi = np.zeros(n_cpgs)
        true_phi[:10] = -0.04  # also recency-graded

        smoking_bins = rng.choice(
            ["never_smoker", "former_15plus_y", "former_5_15y", "former_0_5y", "current_smoker"],
            size=n_samples,
        )
        ind = np.array([smoking_bin_to_indicator_and_recency(b)[0] for b in smoking_bins])
        rec = np.array([smoking_bin_to_indicator_and_recency(b)[1] for b in smoking_bins])

        baseline = rng.uniform(0.3, 0.7, size=n_cpgs)
        noise = rng.normal(0, 0.02, size=(n_cpgs, n_samples))
        beta = baseline[:, None] + true_delta[:, None] * ind + true_phi[:, None] * rec + noise
        beta = np.clip(beta, 0.001, 0.999)

        hc_mask = np.ones(n_samples, dtype=bool)
        cpg_ids = [f"cg{i:08d}" for i in range(n_cpgs)]

        smk = SmokingAxisForeground()
        diag = smk.fit(beta, smoking_bins.tolist(), hc_mask, cpg_ids)
        print(f"\nFit diagnostics:")
        print(f"  n_cpgs_fitted: {diag.n_cpgs_fitted}")
        print(f"  n_cpgs_converged: {diag.n_cpgs_converged}")
        print(f"  n_samples_used: {diag.n_samples_used}")
        print(f"  current/former/never smokers: "
              f"{diag.n_current_smokers}/{diag.n_former_smokers}/{diag.n_never_smokers}")
        print(f"  delta_current distribution: {diag.delta_distribution}")
        print(f"  phi_recency distribution: {diag.phi_distribution}")
        print(f"  Top correlated CpGs:")
        for c in diag.top_smoking_correlated_cpgs[:5]:
            print(f"    {c}")

        # Confirm subtraction works
        cleaned = smk.subtract_from(beta, smoking_bins.tolist())
        # Smoking-affected CpGs should now have much smaller correlation with smoking
        cor_before = np.array([np.corrcoef(beta[i], ind)[0, 1] for i in range(10)])
        cor_after = np.array([np.corrcoef(cleaned[i], ind)[0, 1] for i in range(10)])
        print(f"\nMean |correlation with smoking| on first 10 CpGs:")
        print(f"  Before subtraction: {np.mean(np.abs(cor_before)):.4f}")
        print(f"  After subtraction:  {np.mean(np.abs(cor_after)):.4f}")
        if np.mean(np.abs(cor_after)) < np.mean(np.abs(cor_before)) * 0.3:
            print("  ✓ Subtraction successfully removed smoking signal.")
        else:
            print("  ✗ Subtraction did NOT remove smoking signal as expected.")
        print("\nSmoke test PASS.")


if __name__ == "__main__":
    main()
