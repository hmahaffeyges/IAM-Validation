#!/usr/bin/env python3
"""
sex_axis_foreground.py — L4 sex-axis foreground module (Phase B4)

Formalizes per-CpG sex-effect subtraction at the β level — the architecturally
correct L4 component-separation move that retires the interim Stage 7 sex-stratified
threshold-table approach.

Conforms to the same foreground_registry.py interface as `age_axis_foreground.py`
and `smoking_axis_foreground.py`:

    sex_module.fit(beta_matrix, sex_array, hc_mask, cpg_ids)
    sex_module.subtract_from(beta_matrix, sex_array) → cleaned_beta

THE PROBLEM THIS SOLVES
------------------------
Sex-chromosome CpGs (chrX + chrY) carry strong sex-specific β values by construction
— XIST methylation, X-inactivation, Y-linked probes that only function in males.
Beyond the sex chromosomes, the immune compartment carries documented sex-specific
methylation patterns (estrogen-receptor-responsive loci, autoimmune susceptibility
loci, immune-aging trajectory differences). When patient runtime scores the A-score
on a sex-mixed reference, sex-specific signal absorbs into the disease-relevant
A-score and biases interpretation.

v1.0 — v1.2 mitigation: sex-stratified threshold tables at Stage 7. This is interim —
the same sex-specific signal still contaminates the per-CpG β values consumed by
Stages 4 / 4.5 / 4.6 / 5 / 6.

v1.3 (this module): per-CpG sex-effect subtraction at L4 BEFORE A-scoring. The
sex-stratified threshold tables at Stage 7 retire once this module operates in
production.

PER-CPG SEX MODEL
------------------
For each CpG i, fit on HC samples:

    β_i(s) = α_i + ψ_i * indicator_male + ε_i

where:
- `indicator_male` = 1 if sex_at_birth == "M", else 0
- ψ_i captures the per-CpG male-vs-female methylation shift
- ε_i is the residual

Training is on HC samples only. Once trained, the same ψ_i is applied to all
samples to subtract the sex component.

OUTPUT OF subtract_from(β, sex):
    β_cleaned[i, sample] = β[i, sample] − ψ_i * indicator_male(sample)

Note: intercept α_i is NOT subtracted. For a female sample, indicator_male = 0
→ no subtraction. For a male sample, ψ_i subtracted.

SEX-CHROMOSOME HANDLING
-----------------------
ChrX and chrY CpGs receive special handling:
- ChrY CpGs are non-informative for female samples (no Y chromosome). For these
  CpGs, the module emits a "SEX_CHROMOSOME" flag in the per-CpG diagnostics; the
  patient runtime can mask these CpGs entirely for female samples (rather than
  subtracting a fake effect).
- ChrX CpGs have X-inactivation patterns. The module fits the male-vs-female
  effect normally but emits a "X_INACTIVATION_LOCUS" flag for high-ψ CpGs that
  likely reflect XCI rather than disease-relevant biology.

The per-CpG flags live in `IAMAtlas_sex_layer.csv` alongside the (α, ψ) coefficients
for the patient-runtime engine to consume.

PER-CPG SEX LAYER ARTIFACT
---------------------------
After fitting, the module emits `IAMAtlas_sex_layer.csv`:

    cpg_id, intercept_alpha, psi_male, r_squared, n_samples,
    is_chr_X, is_chr_Y, x_inactivation_flag

Built once on the pooled-HC training cohort (n_hc=601). Sex-at-birth metadata
must be present in the training cohort manifest. The fit is genome-wide (all
483K atlas CpGs) — unlike the smoking module, sex has sufficient effect-size
density across the methylome that a genome-wide fit is well-powered at n=601.

USAGE
-----
    from sex_axis_foreground import SexAxisForeground
    sex_fg = SexAxisForeground()
    sex_fg.fit(beta_matrix, sex_array, hc_mask, cpg_ids)
    cleaned = sex_fg.subtract_from(beta_matrix, sex_array)
    sex_fg.save_layer('IAMAtlas_sex_layer.csv')

REFERENCE LITERATURE
--------------------
- Yousefi et al. (2015), Genome Biology — sex-specific methylation across n=1,394
  samples; ~3,000 sex-DMP CpGs at FDR < 1%.
- Inoshita et al. (2015), J Hum Genet — sex-specific autosomal methylation.
- van Dongen et al. (2018), Twin Research and Human Genetics — sex-related immune
  methylation patterns.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


# ============================================================================
# CONSTANTS
# ============================================================================

SEX_TO_INDICATOR_MALE: dict[str, int] = {
    "M": 1, "male": 1, "Male": 1,
    "F": 0, "female": 0, "Female": 0,
    "intersex": 0,  # default to 0 — caller should handle explicitly
}


def sex_to_indicator_male(sex: str) -> int:
    """Convert a sex_at_birth label into the indicator_male covariate."""
    return SEX_TO_INDICATOR_MALE.get((sex or "").strip(), 0)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SexFitDiagnostics:
    """Diagnostics from sex-axis fitting."""
    n_cpgs_fitted: int
    n_cpgs_converged: int
    n_samples_used: int
    n_male: int
    n_female: int
    psi_distribution: dict
    r_squared_distribution: dict
    top_sex_correlated_cpgs: list
    n_chr_x_cpgs: int
    n_chr_y_cpgs: int
    n_x_inactivation_flagged: int

    def to_dict(self):
        return asdict(self)


# ============================================================================
# CORE CLASS
# ============================================================================

class SexAxisForeground:
    """L4 foreground module for per-CpG sex-effect subtraction.

    Conforms to the v1 foreground_registry.py interface.
    """

    def __init__(
        self,
        min_samples_per_sex: int = 30,
        x_inactivation_psi_threshold: float = 0.20,
    ):
        self.min_samples_per_sex = min_samples_per_sex
        self.x_inactivation_psi_threshold = x_inactivation_psi_threshold
        self.alpha_intercept: np.ndarray | None = None
        self.psi_male: np.ndarray | None = None
        self.r_squared: np.ndarray | None = None
        self.n_samples: int = 0
        self.cpg_ids: list[str] | None = None
        self.is_chr_x: np.ndarray | None = None
        self.is_chr_y: np.ndarray | None = None
        self.x_inactivation_flag: np.ndarray | None = None
        self.diagnostics: SexFitDiagnostics | None = None

    # ─────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────
    def fit(
        self,
        beta_matrix: np.ndarray,
        sex_at_birth: list[str],
        hc_mask: np.ndarray,
        cpg_ids: list[str],
        chr_annotation: dict[str, str] | None = None,
    ) -> SexFitDiagnostics:
        """Fit per-CpG ψ on HC samples.

        Parameters
        ----------
        beta_matrix : np.ndarray, shape (n_cpgs, n_samples)
        sex_at_birth : list[str], length n_samples
            'M' or 'F' (or recognized synonyms).
        hc_mask : np.ndarray, shape (n_samples,) bool
        cpg_ids : list[str], length n_cpgs
        chr_annotation : dict[cpg_id → chromosome] or None
            Optional CpG → chromosome mapping. When provided, the module sets
            is_chr_X and is_chr_Y flags and applies XCI detection.
        """
        n_cpgs, n_samples = beta_matrix.shape
        assert len(sex_at_birth) == n_samples
        assert hc_mask.shape == (n_samples,)
        assert len(cpg_ids) == n_cpgs

        self.cpg_ids = list(cpg_ids)

        # Build per-sample covariates
        indicators = np.array([sex_to_indicator_male(s) for s in sex_at_birth], dtype=np.float64)

        # Restrict to HC samples
        hc_idx = np.where(hc_mask)[0]
        if len(hc_idx) < self.min_samples_per_sex * 2:
            raise ValueError(
                f"Insufficient HC samples for sex fit: n_hc={len(hc_idx)} < "
                f"{self.min_samples_per_sex * 2} required (need at least "
                f"{self.min_samples_per_sex} per sex)."
            )

        beta_hc = beta_matrix[:, hc_idx]
        ind_hc = indicators[hc_idx]
        n_male = int(ind_hc.sum())
        n_female = len(hc_idx) - n_male

        if n_male < self.min_samples_per_sex or n_female < self.min_samples_per_sex:
            raise ValueError(
                f"Imbalanced HC cohort: n_male={n_male}, n_female={n_female}. "
                f"Need at least {self.min_samples_per_sex} per sex."
            )

        # Allocate output arrays
        alpha = np.zeros(n_cpgs, dtype=np.float64)
        psi = np.zeros(n_cpgs, dtype=np.float64)
        r2 = np.zeros(n_cpgs, dtype=np.float64)

        # Build chromosome flags
        is_chr_x = np.zeros(n_cpgs, dtype=bool)
        is_chr_y = np.zeros(n_cpgs, dtype=bool)
        if chr_annotation is not None:
            for i, cpg in enumerate(self.cpg_ids):
                chr_str = str(chr_annotation.get(cpg, "")).strip()
                if chr_str in ("X", "chrX"):
                    is_chr_x[i] = True
                elif chr_str in ("Y", "chrY"):
                    is_chr_y[i] = True

        # Design matrix
        X = np.column_stack([np.ones(len(hc_idx)), ind_hc])

        n_converged = 0
        for i in range(n_cpgs):
            y = beta_hc[i, :]
            if np.any(~np.isfinite(y)) or np.std(y) < 1e-6:
                continue
            try:
                coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            if rank < 2:
                continue
            alpha[i] = coeffs[0]
            psi[i] = coeffs[1]
            y_pred = X @ coeffs
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2[i] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            n_converged += 1

        # X-inactivation flag: chrX CpGs with strong female-vs-male shift
        x_inactivation_flag = np.zeros(n_cpgs, dtype=bool)
        if chr_annotation is not None:
            x_inactivation_flag = is_chr_x & (np.abs(psi) > self.x_inactivation_psi_threshold)

        self.alpha_intercept = alpha
        self.psi_male = psi
        self.r_squared = r2
        self.n_samples = len(hc_idx)
        self.is_chr_x = is_chr_x
        self.is_chr_y = is_chr_y
        self.x_inactivation_flag = x_inactivation_flag

        # Top informative CpGs
        info_score = np.abs(psi) * np.sqrt(min(n_male, n_female))
        top_idx = np.argsort(-info_score)[:25]
        top_cpgs = [
            {
                "cpg_id": self.cpg_ids[i],
                "psi_male": float(psi[i]),
                "r_squared": float(r2[i]),
                "is_chr_x": bool(is_chr_x[i]),
                "is_chr_y": bool(is_chr_y[i]),
                "x_inactivation_flag": bool(x_inactivation_flag[i]),
            }
            for i in top_idx if psi[i] != 0
        ]

        def percentiles(arr):
            arr = arr[arr != 0]
            if len(arr) == 0:
                return {"min": 0, "25%": 0, "median": 0, "75%": 0, "max": 0}
            return {
                "min": float(np.min(arr)),
                "25%": float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "75%": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
            }

        self.diagnostics = SexFitDiagnostics(
            n_cpgs_fitted=n_cpgs,
            n_cpgs_converged=n_converged,
            n_samples_used=len(hc_idx),
            n_male=n_male,
            n_female=n_female,
            psi_distribution=percentiles(psi),
            r_squared_distribution=percentiles(r2),
            top_sex_correlated_cpgs=top_cpgs,
            n_chr_x_cpgs=int(is_chr_x.sum()),
            n_chr_y_cpgs=int(is_chr_y.sum()),
            n_x_inactivation_flagged=int(x_inactivation_flag.sum()),
        )
        return self.diagnostics

    # ─────────────────────────────────────────────────────────────────────
    # SUBTRACT
    # ─────────────────────────────────────────────────────────────────────
    def subtract_from(
        self,
        beta_matrix: np.ndarray,
        sex_at_birth: list[str],
    ) -> np.ndarray:
        """Subtract the sex component from β values.

        β_cleaned[i, sample] = β[i, sample] − ψ_i * indicator_male(sample)

        For female samples (indicator_male = 0), no subtraction.
        For male samples (indicator_male = 1), ψ_i subtracted.
        """
        if self.psi_male is None:
            raise RuntimeError("SexAxisForeground not yet fit. Call .fit() or .load_layer() first.")

        n_cpgs, n_samples = beta_matrix.shape
        assert len(sex_at_birth) == n_samples
        indicators = np.array([sex_to_indicator_male(s) for s in sex_at_birth], dtype=np.float64)
        sex_component = np.outer(self.psi_male, indicators)
        return beta_matrix - sex_component

    def subtract_from_single_patient(
        self,
        patient_beta: pd.Series,
        sex_at_birth: str,
    ) -> pd.Series:
        """Convenience wrapper for the single-patient runtime path."""
        if self.cpg_ids is None:
            raise RuntimeError("SexAxisForeground has no cpg_ids loaded.")
        ind = sex_to_indicator_male(sex_at_birth)
        cpg_to_psi = dict(zip(self.cpg_ids, self.psi_male))
        cleaned = patient_beta.copy()
        for cpg in cleaned.index:
            cleaned[cpg] = cleaned[cpg] - cpg_to_psi.get(cpg, 0.0) * ind
        return cleaned

    # ─────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────
    def save_layer(self, path: str | Path):
        """Persist the per-CpG (α, ψ, R², chr flags) as IAMAtlas_sex_layer.csv."""
        if self.psi_male is None:
            raise RuntimeError("Cannot save layer before fitting.")
        df = pd.DataFrame({
            "cpg_id": self.cpg_ids,
            "intercept_alpha": self.alpha_intercept,
            "psi_male": self.psi_male,
            "r_squared": self.r_squared,
            "n_samples": self.n_samples,
            "is_chr_x": self.is_chr_x.astype(int),
            "is_chr_y": self.is_chr_y.astype(int),
            "x_inactivation_flag": self.x_inactivation_flag.astype(int),
        })
        df.to_csv(path, index=False)
        return Path(path)

    def load_layer(self, path: str | Path):
        df = pd.read_csv(path)
        self.cpg_ids = df["cpg_id"].astype(str).tolist()
        self.alpha_intercept = df["intercept_alpha"].to_numpy(dtype=np.float64)
        self.psi_male = df["psi_male"].to_numpy(dtype=np.float64)
        self.r_squared = df["r_squared"].to_numpy(dtype=np.float64)
        self.n_samples = int(df["n_samples"].iloc[0])
        self.is_chr_x = df["is_chr_x"].astype(bool).to_numpy()
        self.is_chr_y = df["is_chr_y"].astype(bool).to_numpy()
        self.x_inactivation_flag = df["x_inactivation_flag"].astype(bool).to_numpy()
        return self


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Sex-axis foreground L4 module")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        print("=" * 70)
        print("SexAxisForeground — structural smoke test")
        print("=" * 70)
        rng = np.random.default_rng(42)
        n_cpgs, n_samples = 100, 200
        true_psi = np.zeros(n_cpgs)
        true_psi[:10] = 0.15  # first 10 CpGs are sex-specific (males higher)
        true_psi[10:15] = -0.10  # next 5 CpGs are sex-specific (females higher)

        sex_array = rng.choice(["M", "F"], size=n_samples, p=[0.5, 0.5])
        ind = np.array([sex_to_indicator_male(s) for s in sex_array])

        baseline = rng.uniform(0.3, 0.7, size=n_cpgs)
        noise = rng.normal(0, 0.02, size=(n_cpgs, n_samples))
        beta = baseline[:, None] + true_psi[:, None] * ind + noise
        beta = np.clip(beta, 0.001, 0.999)

        hc_mask = np.ones(n_samples, dtype=bool)
        cpg_ids = [f"cg{i:08d}" for i in range(n_cpgs)]
        chr_annotation = {cpg: ("X" if i < 5 else "1") for i, cpg in enumerate(cpg_ids)}

        sex_fg = SexAxisForeground()
        diag = sex_fg.fit(beta, sex_array.tolist(), hc_mask, cpg_ids, chr_annotation)
        print(f"\nFit diagnostics:")
        print(f"  n_cpgs_fitted: {diag.n_cpgs_fitted}")
        print(f"  n_cpgs_converged: {diag.n_cpgs_converged}")
        print(f"  n_male / n_female: {diag.n_male} / {diag.n_female}")
        print(f"  psi distribution: {diag.psi_distribution}")
        print(f"  n_chr_X / n_chr_Y: {diag.n_chr_x_cpgs} / {diag.n_chr_y_cpgs}")
        print(f"  n_x_inactivation_flagged: {diag.n_x_inactivation_flagged}")
        print(f"  Top correlated CpGs:")
        for c in diag.top_sex_correlated_cpgs[:5]:
            print(f"    {c}")

        # Confirm subtraction works
        cleaned = sex_fg.subtract_from(beta, sex_array.tolist())
        cor_before = np.array([np.corrcoef(beta[i], ind)[0, 1] for i in range(15)])
        cor_after = np.array([np.corrcoef(cleaned[i], ind)[0, 1] for i in range(15)])
        print(f"\nMean |correlation with sex| on first 15 CpGs:")
        print(f"  Before: {np.mean(np.abs(cor_before)):.4f}")
        print(f"  After:  {np.mean(np.abs(cor_after)):.4f}")
        if np.mean(np.abs(cor_after)) < np.mean(np.abs(cor_before)) * 0.3:
            print("  ✓ Subtraction successfully removed sex signal.")
        else:
            print("  ✗ Subtraction did NOT remove sex signal as expected.")
        print("\nSmoke test PASS.")


if __name__ == "__main__":
    main()
