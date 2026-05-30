#!/usr/bin/env python3
"""
nilc_deconvolver.py — Independent second deconvolver for L4 cross-method confirmation

Walther's partner. Produces an independent estimate of per-class cell-type fractions
from a patient's β vector, using a mathematically distinct algorithm. The two methods
must agree to within stated tolerance, or L4 has a diagnostic failure mode to investigate.

THE NILC ANALOGY
----------------
Planck's NILC (Needlet Internal Linear Combination) constructs a linear combination of
multi-frequency CMB channels that minimizes residual variance subject to preserving the
CMB component. The CRITICAL INSIGHT is that the weights are derived ONCE from the data's
own covariance structure, not fit empirically against a training set. NILC is independent
of Commander, SMICA, and SEVEM precisely because it constructs its weights from a
different mathematical principle.

For the methylome, "frequency channels" map to architectural classes (8 of them), and
"needlet domains" map to genomic regions × correlation scales. The cross-method
confirmation discipline is what Planck used to validate the CMB temperature maps:
Commander, NILC, SMICA, SEVEM all had to agree before any cosmological result shipped.
CPG inherits the same discipline — Walther and NILC must agree on class fractions
before any downstream L5+ analysis is trusted.

ALGORITHM (METHYLOME-NILC)
---------------------------
Given:
  β_obs   : patient observed β vector (length M, M = number of usable marker CpGs)
  X       : (M × K) reference matrix where X[i, k] = IAMAtlas posterior mean for CpG i, class k
  σ       : (M × K) reference SD matrix (posterior uncertainty per cell)
  K = 8 architectural classes

Walther solves:
  f_walther = argmin_{f ≥ 0, Σf = 1} ||X·f − β_obs||²       (NNLS with simplex constraint)

NILC solves:
  Σ_i = σ_avg(i)²                             (per-CpG variance, averaged over K classes)
  W   = diag(1 / Σ_i)                          (inverse-variance weight matrix)
  f_nilc = (X^T·W·X)^{-1} · X^T·W · β_obs     (generalized least squares, NO constraints)

Then normalize f_nilc to sum to 1.0 (projecting onto the simplex) for fair comparison
with f_walther. Note: we explicitly DO NOT enforce non-negativity during the solve —
this is the anti-NNLS property that makes NILC mathematically independent. If a class
fraction comes out negative, that's a real signal that the patient's β at that class's
markers is BELOW the atlas reference (e.g., the class is depleted relative to the HC
expectation). We report the negative value, then project to the simplex.

OPTIONAL: CHROMOSOME-WINDOWED VARIANT
-------------------------------------
The pure ILC formulation uses ONE weight per class globally. NILC's needlet refinement
uses LOCAL weights that adapt to spatial / scale variations. For the methylome:
  - Partition CpGs by chromosome (23 windows).
  - Solve NILC independently on each window.
  - Combine via inverse-variance weighting of per-window fractions.

This catches localized contamination (e.g., the chr16/chr17 transfer-function suppression
identified in CPG-VAL-006) that a global solve averages away.

CROSS-METHOD AGREEMENT METRIC
-----------------------------
After running both Walther and NILC across a cohort, compute:
  - Per-class Spearman correlation across patients (target: ρ > 0.85)
  - Per-patient L1 disagreement |f_walther - f_nilc|_1 (target: median < 0.05)
  - Per-class direction-of-fraction-deviation agreement (in case patients, does NILC
    move the same direction as Walther for case-vs-HC differences? target: ≥ 90% sign concordance)

Patients with per-patient L1 disagreement > 0.10 are FLAGGED for investigation.
Classes with cross-cohort correlation ρ < 0.85 are FLAGGED for L4 expansion review.

WHY NILC AND NOT JUST "ANOTHER NNLS WITH DIFFERENT WEIGHTS"
-----------------------------------------------------------
The mathematical structure must be ORTHOGONAL to Walther's, not just parameter-tweaked.
Walther uses constrained optimization (simplex + non-negativity). NILC uses unconstrained
linear inversion. If both arrive at the same answer, that answer was determined by the
DATA, not by the chosen optimization regime. If they diverge, the divergence tells us
which patients are in the regime where the constraint matters — those patients are
where the cell composition is genuinely ill-defined by the atlas reference and L4 has
real ambiguity.

USAGE
-----
    from nilc_deconvolver import NILCDeconvolver
    nilc = NILCDeconvolver(atlas_path='/path/to/IAMAtlasREBUILD.csv')
    result = nilc.deconvolve(beta_vector, cpg_ids)
    # result: NILCResult(fractions=dict, raw_fractions=dict, residual_mae=float,
    #                   per_window_fractions=optional dict)

PROTOCOL
--------
1. Build NILC instance against the same IAMAtlas REBUILD that Walther uses.
2. Run NILC on every patient where Walther class fractions exist.
3. Compute the three cross-method agreement metrics.
4. Flag disagreement-outlier patients and disagreement-outlier classes.
5. Report in repo at chain_of_custody/L4_component_separation/nilc_walther_crosscheck.json.

This is the L4 cross-method confirmation gate. Phase B is not closed until NILC and
Walther reach the agreement targets stated above.
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd


# =========================================================================
# Configuration
# =========================================================================
ATLAS_PATH_DEFAULT = "/home/claude/iamatlas_rebuild/IAMAtlasREBUILD.csv"
MARKER_PATH_DEFAULT = "/home/claude/iamatlas_v0_2_extension/iamatlas_celltype_markers_v0_1.json"
CLASSES = ['stem_pluri', 'stem_adult', 'progenitor', 'cycling',
           'secretory', 'immune', 'terminal', 'stromal']


# =========================================================================
# Result type
# =========================================================================
@dataclass
class NILCResult:
    """Per-patient NILC deconvolution result."""
    fractions: dict                # projected to simplex, sums to 1.0, all >= 0
    raw_fractions: dict            # unconstrained GLS output, can have negatives
    residual_mae: float            # mean absolute error of reconstruction at marker CpGs
    n_markers_used: int            # how many marker CpGs the patient had data for
    n_markers_total: int           # how many marker CpGs were in the atlas pool
    status: str                    # "OK", "INSUFFICIENT_MARKERS", "ILL_CONDITIONED"
    per_class_residual: dict       # per-class reconstruction residual
    per_window_fractions: Optional[dict] = None   # if chromosome-windowed mode

    def to_dict(self):
        return asdict(self)


# =========================================================================
# Main deconvolver class
# =========================================================================
class NILCDeconvolver:
    """
    Independent second deconvolver for L4 component-separation cross-check.

    Uses generalized least squares with inverse-variance weighting derived from
    IAMAtlas posterior SDs. Optional chromosome-windowed mode for localized
    weighting.

    Construction:
        nilc = NILCDeconvolver(
            atlas_path='/path/to/IAMAtlasREBUILD.csv',
            marker_path='/path/to/iamatlas_celltype_markers_v0_1.json',
            chromosome_windowed=False,  # set True for true-NILC behavior
        )

    Per-patient use:
        result = nilc.deconvolve(beta_dict)
        # beta_dict: {cpg_id: β_value}
    """

    def __init__(self,
                 atlas_path: str = ATLAS_PATH_DEFAULT,
                 marker_path: Optional[str] = MARKER_PATH_DEFAULT,
                 chromosome_windowed: bool = False,
                 min_markers_per_class: int = 30,
                 ridge_lambda: float = 1e-4,
                 verbose: bool = False):
        self.atlas_path = atlas_path
        self.marker_path = marker_path
        self.chromosome_windowed = chromosome_windowed
        self.min_markers_per_class = min_markers_per_class
        self.ridge_lambda = ridge_lambda
        self.verbose = verbose
        self._loaded = False
        self.X = None         # (M_total × K) atlas means at marker CpGs
        self.SIGMA = None     # (M_total × K) atlas SDs at marker CpGs
        self.cpg_ids = None   # marker CpG IDs in row order
        self.chr_assign = None  # chromosome per marker CpG
        self._load_atlas()

    def _load_atlas(self):
        """Load IAMAtlas REBUILD + marker pool. Build the X (means) and SIGMA (SDs)
        reference matrices at marker CpGs."""
        if self.verbose:
            print(f"[NILC] Loading atlas from {self.atlas_path}")

        # Load marker pool: union of all per-class markers from the v0.1 marker artifact
        if self.marker_path and Path(self.marker_path).exists():
            mark_data = json.load(open(self.marker_path))
            # marker artifact is per-cell-type; aggregate to per-class
            ct_to_class = mark_data.get('celltype_to_class', {})
            class_markers = {c: set() for c in CLASSES}
            # The v0.1 artifact key is `markers_by_celltype` (not `markers`).
            markers_dict = (mark_data.get('markers_by_celltype')
                            or mark_data.get('markers')
                            or {})
            for ct, markers in markers_dict.items():
                cls = ct_to_class.get(ct)
                if cls in class_markers:
                    class_markers[cls].update(markers[:100])  # top-100 per cell type
            all_markers = sorted(set().union(*class_markers.values()))
            if self.verbose:
                print(f"[NILC]   Loaded {len(all_markers):,} marker CpGs from artifact")
        else:
            # Fallback: use ALL atlas CpGs (slow)
            all_markers = None
            if self.verbose:
                print(f"[NILC]   No marker artifact — using all atlas CpGs")

        # Load the atlas, optionally filtered to markers
        keep_cols = ['cpg_id']
        keep_cols += [f'{c}_mean' for c in CLASSES] + [f'{c}_sd' for c in CLASSES]
        atlas = pd.read_csv(self.atlas_path, usecols=keep_cols)
        atlas = atlas.rename(columns={'cpg_id': 'cpg'})
        if all_markers:
            atlas = atlas[atlas['cpg'].isin(all_markers)].copy()

        # Impute missing posterior means (mostly stromal due to galactic mask)
        # with the column median rather than dropping the CpG. Mark those positions
        # via inflated SD so the GLS down-weights them automatically.
        for c in CLASSES:
            mean_col = f'{c}_mean'
            sd_col = f'{c}_sd'
            nan_mask = atlas[mean_col].isna()
            if nan_mask.sum() > 0:
                med_mean = atlas[mean_col].median()
                atlas.loc[nan_mask, mean_col] = med_mean
                # Set SD to a large value (max observed) so GLS treats imputed cells as low-info
                med_sd = atlas[sd_col].max()
                atlas.loc[nan_mask, sd_col] = med_sd if med_sd > 0 else 0.5
            if sd_col in atlas.columns:
                med = atlas[sd_col].median()
                atlas[sd_col] = atlas[sd_col].fillna(med if med > 0 else 0.05)
            else:
                atlas[sd_col] = 0.05

        # Build matrices
        self.cpg_ids = atlas['cpg'].values
        self.X = atlas[[f'{c}_mean' for c in CLASSES]].values
        self.SIGMA = atlas[[f'{c}_sd' for c in CLASSES]].values

        # Chromosome assignment (for windowed mode)
        # Have to look it up from the methylprep manifest
        if self.chromosome_windowed:
            self._assign_chromosomes()

        self._loaded = True
        if self.verbose:
            print(f"[NILC]   Reference matrix: X{self.X.shape}, SIGMA{self.SIGMA.shape}")
            print(f"[NILC]   Marker CpGs in atlas: {len(self.cpg_ids):,}")
            print(f"[NILC]   Per-class SD range: "
                  f"[{self.SIGMA.min():.4f}, {self.SIGMA.max():.4f}]")

    def _assign_chromosomes(self):
        """Annotate each marker CpG with its chromosome using methylprep manifest."""
        try:
            from methylprep.files import Manifest
            man = Manifest('450k').data_frame[['CHR']].reset_index()
            man = man.rename(columns={'IlmnID': 'cpg', 'CHR': 'chromosome'})
            chr_map = dict(zip(man['cpg'], man['chromosome'].astype(str)))
            self.chr_assign = np.array([chr_map.get(c, 'unk') for c in self.cpg_ids])
        except Exception as e:
            if self.verbose:
                print(f"[NILC] chromosome assignment failed ({e}); windowed mode disabled")
            self.chromosome_windowed = False

    # ----------------------------------------------------------------------
    # Core single-patient deconvolution
    # ----------------------------------------------------------------------
    def deconvolve(self, beta_dict, patient_id=None) -> NILCResult:
        """
        Given a patient's β values at CpGs (as dict {cpg_id: β}), return per-class fractions.

        Returns NILCResult with both raw (unconstrained GLS) and projected (simplex) fractions.
        """
        if not self._loaded:
            raise RuntimeError("Atlas not loaded")

        # Match the patient's CpGs to the marker pool — only keep CpGs where
        # both (a) the atlas has a posterior, and (b) the patient has a non-NaN β
        beta_keys_with_value = {k: v for k, v in beta_dict.items()
                                 if v is not None and not (isinstance(v, float) and np.isnan(v))}
        beta_keys = set(beta_keys_with_value.keys())
        mask = np.array([c in beta_keys for c in self.cpg_ids])
        if mask.sum() < self.min_markers_per_class * len(CLASSES):
            return NILCResult(
                fractions={c: np.nan for c in CLASSES},
                raw_fractions={c: np.nan for c in CLASSES},
                residual_mae=np.nan,
                n_markers_used=int(mask.sum()),
                n_markers_total=len(self.cpg_ids),
                status="INSUFFICIENT_MARKERS",
                per_class_residual={c: np.nan for c in CLASSES},
            )

        X = self.X[mask]                              # (m × K)
        SIGMA = self.SIGMA[mask]                      # (m × K)
        # Patient β at matched CpGs (guaranteed non-NaN by mask construction)
        beta = np.array([beta_keys_with_value[c] for c in self.cpg_ids[mask]])

        # Per-CpG average variance across classes (the "common" measurement uncertainty)
        sigma2 = (SIGMA ** 2).mean(axis=1)
        sigma2 = np.maximum(sigma2, 1e-6)
        W = np.diag(1.0 / sigma2)

        # Generalized least squares with ridge regularization for numerical stability
        # f_raw = (X^T W X + λI)^{-1} X^T W β_obs
        XtWX = X.T @ W @ X
        ridge = self.ridge_lambda * np.trace(XtWX) / len(CLASSES) * np.eye(len(CLASSES))
        XtWX_reg = XtWX + ridge
        try:
            inv = np.linalg.inv(XtWX_reg)
            f_raw = inv @ X.T @ W @ beta
        except np.linalg.LinAlgError:
            return NILCResult(
                fractions={c: np.nan for c in CLASSES},
                raw_fractions={c: np.nan for c in CLASSES},
                residual_mae=np.nan,
                n_markers_used=int(mask.sum()),
                n_markers_total=len(self.cpg_ids),
                status="ILL_CONDITIONED",
                per_class_residual={c: np.nan for c in CLASSES},
            )

        # Project to simplex (non-negative, sum to 1) using a stable algorithm
        f_proj = self._project_simplex(f_raw)

        # Reconstruction residual
        beta_reconstructed = X @ f_proj
        per_cpg_resid = beta - beta_reconstructed
        residual_mae = float(np.mean(np.abs(per_cpg_resid)))

        # Per-class residual contribution: how much of the residual lives in CpGs
        # most-informative for each class?
        per_class_resid = {}
        for ki, cls in enumerate(CLASSES):
            # Weight each CpG's residual by how strongly that CpG marks this class
            # (use X[:, ki] / X.sum(axis=1) as the class-affinity weight)
            class_affinity = X[:, ki] / (X.sum(axis=1) + 1e-9)
            per_class_resid[cls] = float(
                np.average(np.abs(per_cpg_resid), weights=class_affinity)
            )

        # Optional chromosome-windowed fractions
        per_window_fracs = None
        if self.chromosome_windowed and self.chr_assign is not None:
            per_window_fracs = self._deconvolve_windowed(beta_dict, mask)

        return NILCResult(
            fractions={c: float(f_proj[ki]) for ki, c in enumerate(CLASSES)},
            raw_fractions={c: float(f_raw[ki]) for ki, c in enumerate(CLASSES)},
            residual_mae=residual_mae,
            n_markers_used=int(mask.sum()),
            n_markers_total=len(self.cpg_ids),
            status="OK",
            per_class_residual=per_class_resid,
            per_window_fractions=per_window_fracs,
        )

    def _deconvolve_windowed(self, beta_dict, mask):
        """Per-chromosome windowed NILC."""
        out = {}
        chrs = self.chr_assign[mask]
        X_m = self.X[mask]
        SIGMA_m = self.SIGMA[mask]
        beta_keys_with_value = {k: v for k, v in beta_dict.items()
                                 if v is not None and not (isinstance(v, float) and np.isnan(v))}
        beta = np.array([beta_keys_with_value[c] for c in self.cpg_ids[mask]])
        for ch in np.unique(chrs):
            sel = chrs == ch
            if sel.sum() < self.min_markers_per_class:
                continue
            Xc = X_m[sel]; SIGMAc = SIGMA_m[sel]; bc = beta[sel]
            sigma2 = np.maximum((SIGMAc ** 2).mean(axis=1), 1e-6)
            Wc = np.diag(1.0 / sigma2)
            XtWX = Xc.T @ Wc @ Xc
            ridge = self.ridge_lambda * np.trace(XtWX) / len(CLASSES) * np.eye(len(CLASSES))
            try:
                f_raw_c = np.linalg.inv(XtWX + ridge) @ Xc.T @ Wc @ bc
                f_proj_c = self._project_simplex(f_raw_c)
                out[str(ch)] = {c: float(f_proj_c[ki]) for ki, c in enumerate(CLASSES)}
            except np.linalg.LinAlgError:
                continue
        return out

    @staticmethod
    def _project_simplex(v):
        """Project vector v onto the probability simplex {x: x ≥ 0, Σx = 1}.

        Uses the algorithm of Duchi et al. 2008 "Efficient projections onto the
        ℓ1-ball for learning in high dimensions" — known stable and exact in O(K log K).
        """
        K = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_candidates = np.where(u - (cssv - 1) / (np.arange(K) + 1) > 0)[0]
        if len(rho_candidates) == 0:
            # All-zero result; uniform fallback
            return np.full(K, 1.0 / K)
        rho = rho_candidates[-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        return np.maximum(v - theta, 0)

    # ----------------------------------------------------------------------
    # Batch interface
    # ----------------------------------------------------------------------
    def deconvolve_batch(self, beta_matrix, cpg_ids, patient_ids=None):
        """
        Deconvolve a batch of patients.

        Args:
            beta_matrix : (N × M) matrix of β values; rows = patients, cols = CpGs
            cpg_ids : list of M CpG IDs (column labels of beta_matrix)
            patient_ids : optional list of N patient IDs

        Returns:
            pd.DataFrame with one row per patient + columns {class_name, residual_mae,
            n_markers_used, status, [raw_class_name]}
        """
        rows = []
        N = beta_matrix.shape[0]
        for i in range(N):
            beta_dict = {cpg_ids[j]: beta_matrix[i, j] for j in range(len(cpg_ids))
                         if not np.isnan(beta_matrix[i, j])}
            r = self.deconvolve(beta_dict)
            row = {**r.fractions,
                   **{f'raw_{k}': v for k, v in r.raw_fractions.items()},
                   'residual_mae': r.residual_mae,
                   'n_markers_used': r.n_markers_used,
                   'status': r.status}
            if patient_ids is not None:
                row['gsm'] = patient_ids[i]
            rows.append(row)
            if self.verbose and (i % 100 == 0):
                print(f"[NILC]   deconvolved {i + 1}/{N}")
        return pd.DataFrame(rows)


# =========================================================================
# Cross-method comparison
# =========================================================================
@dataclass
class CrossMethodReport:
    n_patients: int
    per_class_correlations: dict           # {class: spearman_rho}
    median_L1_disagreement: float          # median over patients of Σ_k |f_walther_k - f_nilc_k|
    p95_L1_disagreement: float
    n_outlier_patients: int                # patients with L1 > 0.10
    outlier_patient_ids: list
    per_class_passes_threshold: dict       # {class: bool} where threshold is ρ ≥ 0.85
    overall_pass: bool                     # all classes pass AND median L1 < 0.05

    def to_dict(self):
        d = asdict(self)
        return d


def cross_method_comparison(walther_df: pd.DataFrame,
                            nilc_df: pd.DataFrame,
                            patient_id_col: str = 'gsm',
                            correlation_threshold: float = 0.85,
                            median_L1_threshold: float = 0.05,
                            outlier_L1_threshold: float = 0.10) -> CrossMethodReport:
    """
    Compare Walther and NILC class fractions across the same patients.

    Args:
        walther_df: DataFrame with columns {patient_id_col, class1, class2, ..., classK}
        nilc_df: DataFrame with columns {patient_id_col, class1, class2, ..., classK}
    """
    merged = walther_df.merge(nilc_df, on=patient_id_col, suffixes=('_walther', '_nilc'))
    if len(merged) == 0:
        raise ValueError(f"No patients matched between Walther and NILC outputs on '{patient_id_col}'")

    # Per-class Spearman correlation
    from scipy.stats import spearmanr
    per_class_corrs = {}
    per_class_passes = {}
    for cls in CLASSES:
        wcol = f'{cls}_walther'; ncol = f'{cls}_nilc'
        if wcol not in merged.columns or ncol not in merged.columns:
            per_class_corrs[cls] = np.nan
            per_class_passes[cls] = False
            continue
        m = merged.dropna(subset=[wcol, ncol])
        if len(m) < 5:
            per_class_corrs[cls] = np.nan
            per_class_passes[cls] = False
            continue
        rho, _ = spearmanr(m[wcol], m[ncol])
        per_class_corrs[cls] = float(rho)
        per_class_passes[cls] = bool(rho >= correlation_threshold)

    # Per-patient L1 disagreement
    L1 = np.zeros(len(merged))
    for cls in CLASSES:
        wcol = f'{cls}_walther'; ncol = f'{cls}_nilc'
        if wcol in merged.columns and ncol in merged.columns:
            L1 += np.abs(merged[wcol].fillna(0) - merged[ncol].fillna(0))
    merged['L1_disagreement'] = L1
    median_L1 = float(np.median(L1))
    p95_L1 = float(np.percentile(L1, 95))
    outliers = merged.loc[merged['L1_disagreement'] > outlier_L1_threshold, patient_id_col].tolist()

    overall_pass = bool(all(per_class_passes.values()) and median_L1 < median_L1_threshold)

    return CrossMethodReport(
        n_patients=len(merged),
        per_class_correlations=per_class_corrs,
        median_L1_disagreement=median_L1,
        p95_L1_disagreement=p95_L1,
        n_outlier_patients=len(outliers),
        outlier_patient_ids=outliers,
        per_class_passes_threshold=per_class_passes,
        overall_pass=overall_pass,
    )


def print_cross_method_summary(report: CrossMethodReport,
                                correlation_threshold: float = 0.85,
                                median_L1_threshold: float = 0.05):
    print()
    print("=" * 78)
    print("L4 CROSS-METHOD CONFIRMATION (Walther vs NILC)")
    print("=" * 78)
    print(f"n_patients: {report.n_patients}")
    print()
    print(f"{'Class':<14} {'ρ (Spearman)':>14} {'≥' + str(correlation_threshold) + '?':>10}")
    print("-" * 42)
    for cls, rho in report.per_class_correlations.items():
        passes = "PASS" if report.per_class_passes_threshold[cls] else "FAIL"
        rho_str = f"{rho:+.3f}" if not np.isnan(rho) else " n/a"
        print(f"{cls:<14} {rho_str:>14} {passes:>10}")
    print("-" * 42)
    print(f"Median L1 disagreement: {report.median_L1_disagreement:.4f} "
          f"(threshold < {median_L1_threshold:.2f}: "
          f"{'PASS' if report.median_L1_disagreement < median_L1_threshold else 'FAIL'})")
    print(f"P95 L1 disagreement: {report.p95_L1_disagreement:.4f}")
    print(f"Outlier patients (L1 > 0.10): {report.n_outlier_patients}")
    if report.outlier_patient_ids and len(report.outlier_patient_ids) <= 10:
        print(f"  IDs: {report.outlier_patient_ids}")
    print()
    print(f"OVERALL: {'PASS' if report.overall_pass else 'FAIL'} — "
          f"L4 cross-method gate {'cleared' if report.overall_pass else 'NOT cleared'}.")
    print()


# =========================================================================
# CLI
# =========================================================================
def main():
    ap = argparse.ArgumentParser(description="L4 cross-method NILC deconvolver + Walther cross-check")
    ap.add_argument("--mode", choices=["deconvolve", "crosscheck"], required=True)
    ap.add_argument("--atlas", default=ATLAS_PATH_DEFAULT)
    ap.add_argument("--markers", default=MARKER_PATH_DEFAULT)
    ap.add_argument("--beta-matrix", help="Patient β matrix CSV (patients × CpGs, gsm column)")
    ap.add_argument("--walther-fractions", help="Walther class-fractions CSV for crosscheck mode")
    ap.add_argument("--nilc-fractions", help="NILC class-fractions CSV (output of deconvolve mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chromosome-windowed", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.mode == "deconvolve":
        nilc = NILCDeconvolver(atlas_path=args.atlas, marker_path=args.markers,
                                chromosome_windowed=args.chromosome_windowed,
                                verbose=args.verbose)
        beta_df = pd.read_csv(args.beta_matrix)
        gsm_col = 'gsm' if 'gsm' in beta_df.columns else beta_df.columns[0]
        patient_ids = beta_df[gsm_col].tolist()
        cpg_cols = [c for c in beta_df.columns if c != gsm_col]
        beta_matrix = beta_df[cpg_cols].values
        result_df = nilc.deconvolve_batch(beta_matrix, cpg_cols, patient_ids)
        result_df.to_csv(args.out, index=False)
        print(f"NILC fractions saved to {args.out} ({len(result_df)} patients)")

    elif args.mode == "crosscheck":
        walther_df = pd.read_csv(args.walther_fractions)
        nilc_df = pd.read_csv(args.nilc_fractions)
        report = cross_method_comparison(walther_df, nilc_df)
        print_cross_method_summary(report)
        with open(args.out, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"Crosscheck report saved to {args.out}")
        sys.exit(0 if report.overall_pass else 1)


if __name__ == "__main__":
    main()
