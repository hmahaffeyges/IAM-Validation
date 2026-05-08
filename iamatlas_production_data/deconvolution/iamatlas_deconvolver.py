#!/usr/bin/env python3
"""
IAMAtlas Deconvolver
=====================

Takes a customer's β vector (per-CpG methylation values from an IDAT) and
returns fractional composition across the cell types the IAMAtlas covers.

Replaces external dependencies on EpiDISH, CIBERSORT, NNLS-against-Loyfer,
and similar tools. Same math; runs against the IAMAtlas posterior matrix
instead of any single source atlas.

ARCHITECTURE
============
1. Load IAMAtlas per-cell-type matrix (cpg_id, ct_1_mean, ct_2_mean, ...)
2. Load customer's β vector (cpg_id → β)
3. Find informative CpGs: those where IAMAtlas posterior SD is small
   (i.e., the matrix is confident about that CpG's cell-type-specific value)
   AND where between-cell-type variance is high (i.e., the CpG actually
   discriminates between cell types).
4. Build the reference matrix R: shape (n_cpg, n_celltype), each column
   is the IAMAtlas posterior mean β for that cell type at the CpGs.
5. Build the customer vector y: shape (n_cpg,), the customer's β at those
   same CpGs.
6. Solve the constrained linear system y = R × f for fractional composition f:
     - f ≥ 0 (non-negativity)
     - Σ f_i = 1 (proportions sum to 1)
     via scipy.optimize.nnls then renormalization.
7. Compute residuals: y_hat = R × f vs y. Residual SD per CpG.
8. Return: (fractions, residuals, diagnostic info)

WHY THIS IS BETTER THAN NNLS-AGAINST-LOYFER
============================================
- Reference matrix has tighter posteriors (multi-atlas pooled, not single-atlas)
- Reference includes posterior SD, so we can weight CpGs by reliability
- Outputs include per-architecture-class aggregation (sum cell-type fractions
  within class) for multi-resolution scoring
- No dependency on R/Bioconductor packages

LIMITATIONS
===========
- Does not yet propagate full Bayesian uncertainty in fractions (gives point
  estimates). For posterior intervals on f, run multi-start NNLS with
  resampled reference draws. v1 ships point estimates.
- Customer's β must be on the HM450 array universe (or EPIC v1 with
  HM450 overlap CpGs). Any CpG not in IAMAtlas is dropped.

USAGE
=====
    from deconvolve import IAMAtlasDeconvolver

    deconv = IAMAtlasDeconvolver("IAMAtlas.csv")
    customer_betas = {"cg00012345": 0.45, "cg00012346": 0.78, ...}
    result = deconv.deconvolve(customer_betas)
    print(result.fractions)              # cell type → fraction
    print(result.class_fractions)        # architecture class → fraction (sum)
    print(result.residual_sd)            # how well the fit explains the data

Source: IAMPerformance Inter-Domain Research Institute, Entiat WA
Date: 2026-05-04
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


CLASSES = ["stem_pluri", "stem_adult", "progenitor", "stromal",
           "cycling", "secretory", "immune", "terminal"]


@dataclass
class DeconvolutionResult:
    """Result returned by IAMAtlasDeconvolver.deconvolve()"""
    fractions: dict          # cell_type → fraction (0–1, sum to 1)
    class_fractions: dict    # architecture class → fraction (sum within class)
    n_cpgs_used: int         # how many informative CpGs went into the fit
    n_cpgs_customer_total: int  # total CpGs customer provided
    n_cpgs_in_matrix: int    # how many of customer's CpGs were in IAMAtlas
    residual_mae: float      # mean absolute residual between y and R×f
    residual_sd: float       # SD of residuals (uncertainty signal)
    used_celltypes: list     # cell types kept after filtering
    diagnostic: dict = field(default_factory=dict)


class IAMAtlasDeconvolver:
    """
    Deconvolution engine using IAMAtlas posterior brightness as reference.
    
    Loads the matrix once at instantiation; each .deconvolve(customer_β)
    call is fast (~1 sec per customer).
    """

    def __init__(self, matrix_path: str | Path,
                 sd_threshold: float = 0.10,
                 betweencell_var_threshold: float = 0.02,
                 verbose: bool = True):
        """
        Parameters
        ----------
        matrix_path : path to IAMAtlas CSV (with per-cell-type columns)
        sd_threshold : keep CpGs only if max posterior SD across cell types < this.
                       Default 0.10 means we trust CpGs where the atlas is
                       confident within ±0.10 β units.
        betweencell_var_threshold : keep CpGs only if between-cell-type variance
                       > this. Removes CpGs that don't discriminate (e.g.,
                       fully methylated everywhere).
        verbose : print loading progress
        """
        self.matrix_path = Path(matrix_path)
        self.sd_threshold = sd_threshold
        self.betweencell_var_threshold = betweencell_var_threshold
        self.verbose = verbose

        # Per-CpG reference: cpg_id → {celltype: (mean, sd, class)}
        self.ref = {}
        # Cell type vocabulary
        self.celltypes = []
        self.celltype_to_class = {}

        self._load_matrix()
        self._filter_informative_cpgs()

    def _load_matrix(self):
        """Parse the IAMAtlas CSV. Detects per-cell-type columns automatically."""
        if self.verbose:
            print(f"Loading IAMAtlas matrix: {self.matrix_path}")
        with open(self.matrix_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            
            # Detect cell-type columns: anything ending in _mean that isn't
            # one of the 8 architecture-class columns
            ct_columns = []
            for c in fieldnames:
                if not c.endswith("_mean"):
                    continue
                stem = c[:-len("_mean")]
                if stem in CLASSES:
                    continue  # skip class-level columns
                ct_columns.append(stem)
            
            self.celltypes = ct_columns
            if self.verbose:
                print(f"  Detected {len(self.celltypes)} cell-type columns")
            
            if not self.celltypes:
                # Fall back: matrix only has class-level columns. Use those
                # as "cell types" — coarser deconvolution but still valid.
                self.celltypes = list(CLASSES)
                if self.verbose:
                    print(f"  No cell-type columns found; falling back to class-level deconvolution")
            
            # Map cell type → class. For class-level fallback, identity map.
            for ct in self.celltypes:
                if ct in CLASSES:
                    self.celltype_to_class[ct] = ct
                else:
                    # We need the class assignment from the matrix metadata.
                    # Without per-row class column for cell types, we infer
                    # by checking which class the cell type came from in
                    # the per-class output files (loaded externally).
                    # For now, default to "unknown" — caller can supply mapping.
                    self.celltype_to_class[ct] = "unknown"
            
            # Read each row
            for row in reader:
                cpg = row["cpg_id"]
                cell_data = {}
                for ct in self.celltypes:
                    mean_str = row.get(f"{ct}_mean")
                    sd_str = row.get(f"{ct}_sd")
                    if mean_str in (None, "", "NA"):
                        continue
                    try:
                        mean = float(mean_str)
                        sd = float(sd_str) if sd_str not in (None, "", "NA") else 0.0
                    except ValueError:
                        continue
                    if not (0 <= mean <= 1):
                        continue
                    cell_data[ct] = (mean, sd)
                if cell_data:
                    self.ref[cpg] = cell_data

        if self.verbose:
            print(f"  Loaded {len(self.ref)} CpGs with per-cell-type data")

    def _filter_informative_cpgs(self):
        """
        Build self.informative_cpgs — only CpGs where the matrix is both
        precise (low SD) and discriminative (high between-cell variance).
        """
        informative = {}
        n_filtered_sd = 0
        n_filtered_var = 0
        n_filtered_coverage = 0
        for cpg, cell_data in self.ref.items():
            # All cell types must be present (or close to it) for a clean fit
            if len(cell_data) < max(2, len(self.celltypes) // 2):
                n_filtered_coverage += 1
                continue
            sds = [sd for _, sd in cell_data.values()]
            means = [m for m, _ in cell_data.values()]
            max_sd = max(sds)
            if max_sd > self.sd_threshold:
                n_filtered_sd += 1
                continue
            mean_avg = sum(means) / len(means)
            var = sum((m - mean_avg) ** 2 for m in means) / len(means)
            if var < self.betweencell_var_threshold:
                n_filtered_var += 1
                continue
            informative[cpg] = cell_data
        
        self.informative_cpgs = informative
        if self.verbose:
            print(f"  Informative CpGs after filtering: {len(self.informative_cpgs)}")
            print(f"    Filtered for high SD: {n_filtered_sd}")
            print(f"    Filtered for low between-cell variance: {n_filtered_var}")
            print(f"    Filtered for low coverage: {n_filtered_coverage}")

    def deconvolve(self, customer_betas: dict) -> DeconvolutionResult:
        """
        Deconvolve a customer's β vector into fractional cell-type composition.
        
        Parameters
        ----------
        customer_betas : dict {cpg_id: β_value}
        
        Returns
        -------
        DeconvolutionResult
        """
        try:
            import numpy as np
            from scipy.optimize import nnls
        except ImportError:
            raise ImportError(
                "Deconvolver requires numpy and scipy. "
                "Install with: pip install numpy scipy"
            )

        # Find CpGs that are both in informative set AND in customer's data
        usable_cpgs = []
        for cpg in self.informative_cpgs:
            if cpg in customer_betas:
                b = customer_betas[cpg]
                if isinstance(b, (int, float)) and 0 <= b <= 1:
                    usable_cpgs.append(cpg)

        n_cpgs_in_matrix = sum(1 for c in customer_betas if c in self.ref)
        n_cpgs_used = len(usable_cpgs)

        if n_cpgs_used < 100:
            return DeconvolutionResult(
                fractions={},
                class_fractions={},
                n_cpgs_used=n_cpgs_used,
                n_cpgs_customer_total=len(customer_betas),
                n_cpgs_in_matrix=n_cpgs_in_matrix,
                residual_mae=float("nan"),
                residual_sd=float("nan"),
                used_celltypes=[],
                diagnostic={"status": "INSUFFICIENT_INFORMATIVE_CPGS",
                            "needed": 100, "got": n_cpgs_used},
            )

        # Determine which cell types have coverage at every usable CpG.
        # Cell types missing β at too many CpGs get dropped.
        ct_coverage = {ct: 0 for ct in self.celltypes}
        for cpg in usable_cpgs:
            for ct in self.informative_cpgs[cpg]:
                ct_coverage[ct] += 1
        coverage_threshold = int(0.9 * n_cpgs_used)
        used_celltypes = [ct for ct, c in ct_coverage.items() if c >= coverage_threshold]
        if len(used_celltypes) < 2:
            return DeconvolutionResult(
                fractions={},
                class_fractions={},
                n_cpgs_used=n_cpgs_used,
                n_cpgs_customer_total=len(customer_betas),
                n_cpgs_in_matrix=n_cpgs_in_matrix,
                residual_mae=float("nan"),
                residual_sd=float("nan"),
                used_celltypes=[],
                diagnostic={"status": "INSUFFICIENT_CELLTYPE_COVERAGE",
                            "celltype_coverage": ct_coverage},
            )

        # Build R (reference) and y (customer)
        # R: shape (n_cpg, n_celltype). Use posterior mean β for each cell type.
        # For CpGs missing a particular cell type, fall back to global mean β
        # at that CpG (less informative but keeps the fit numerically stable).
        n_cpg = n_cpgs_used
        n_ct = len(used_celltypes)
        R = np.zeros((n_cpg, n_ct))
        y = np.zeros(n_cpg)
        weights = np.ones(n_cpg)
        for i, cpg in enumerate(usable_cpgs):
            cell_data = self.informative_cpgs[cpg]
            global_mean = sum(m for m, _ in cell_data.values()) / len(cell_data)
            for j, ct in enumerate(used_celltypes):
                if ct in cell_data:
                    R[i, j] = cell_data[ct][0]
                else:
                    R[i, j] = global_mean
            y[i] = customer_betas[cpg]
            # Weight by inverse of max SD at this CpG (lower SD → higher weight)
            max_sd = max(sd for _, sd in cell_data.values())
            weights[i] = 1.0 / max(max_sd, 1e-3)

        # Apply weights via row-scaling
        sqrt_w = np.sqrt(weights)
        R_w = R * sqrt_w[:, None]
        y_w = y * sqrt_w

        # Solve y_w = R_w × f with f ≥ 0
        f, _ = nnls(R_w, y_w)
        # Renormalize to sum to 1
        if f.sum() > 0:
            f = f / f.sum()

        # Compute residuals on un-weighted scale
        y_hat = R @ f
        residual = y - y_hat
        residual_mae = float(np.mean(np.abs(residual)))
        residual_sd = float(np.std(residual))

        # Pack results
        fractions = {ct: float(f[i]) for i, ct in enumerate(used_celltypes)}
        class_fractions = {cls: 0.0 for cls in CLASSES}
        for ct, frac in fractions.items():
            cls = self.celltype_to_class.get(ct, "unknown")
            if cls in class_fractions:
                class_fractions[cls] += frac

        return DeconvolutionResult(
            fractions=fractions,
            class_fractions=class_fractions,
            n_cpgs_used=n_cpgs_used,
            n_cpgs_customer_total=len(customer_betas),
            n_cpgs_in_matrix=n_cpgs_in_matrix,
            residual_mae=residual_mae,
            residual_sd=residual_sd,
            used_celltypes=used_celltypes,
            diagnostic={"status": "OK",
                        "weight_strategy": "inverse_max_posterior_sd"},
        )


def set_celltype_class_map(deconvolver: IAMAtlasDeconvolver, mapping: dict):
    """
    Register the cell-type → architecture-class mapping for class aggregation.
    
    Mapping comes from the canonical cell_to_class file built in Step 1.
    Without this, class_fractions will all show as "unknown".
    """
    for ct, cls in mapping.items():
        if ct in deconvolver.celltypes and cls in CLASSES:
            deconvolver.celltype_to_class[ct] = cls


# ============================================================
# CLI for testing
# ============================================================

def _cli_main():
    parser = argparse.ArgumentParser(description="IAMAtlas deconvolver CLI")
    parser.add_argument("--matrix", required=True, help="Path to IAMAtlas CSV")
    parser.add_argument("--customer_betas", required=True,
                        help="Path to customer JSON file: {cpg_id: β, ...}")
    parser.add_argument("--celltype_class_map", default=None,
                        help="Path to JSON: {celltype: class, ...}")
    parser.add_argument("--output", default="deconvolution_result.json")
    args = parser.parse_args()

    deconv = IAMAtlasDeconvolver(args.matrix)
    
    if args.celltype_class_map:
        with open(args.celltype_class_map) as f:
            mapping = json.load(f)
        set_celltype_class_map(deconv, mapping)
    
    with open(args.customer_betas) as f:
        customer = json.load(f)
    
    result = deconv.deconvolve(customer)
    
    print(f"\nDeconvolution result:")
    print(f"  CpGs used: {result.n_cpgs_used}")
    print(f"  Residual MAE: {result.residual_mae:.4f}")
    print(f"\n  Cell-type fractions:")
    for ct, f in sorted(result.fractions.items(), key=lambda x: -x[1])[:10]:
        print(f"    {ct:<30}  {f:.4f}")
    print(f"\n  Architecture-class fractions:")
    for cls, f in sorted(result.class_fractions.items(), key=lambda x: -x[1]):
        print(f"    {cls:<15}  {f:.4f}")

    out = {
        "fractions": result.fractions,
        "class_fractions": result.class_fractions,
        "n_cpgs_used": result.n_cpgs_used,
        "n_cpgs_customer_total": result.n_cpgs_customer_total,
        "n_cpgs_in_matrix": result.n_cpgs_in_matrix,
        "residual_mae": result.residual_mae,
        "residual_sd": result.residual_sd,
        "used_celltypes": result.used_celltypes,
        "diagnostic": result.diagnostic,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull result: {args.output}")


if __name__ == "__main__":
    _cli_main()
