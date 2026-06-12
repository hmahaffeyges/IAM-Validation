#!/usr/bin/env python3
"""
Walther IAM Deconvolver
=======================

Cell-fraction estimator built specifically for IAMAtlas. Takes a customer's
methylation beta vector and returns:
  - per-CLASS fractions   (the 8 IAM architecture classes) -- PRIMARY, reliable
  - per-cell-type fractions (within-class)                 -- SECONDARY, indicative

WHY THIS EXISTS / DESIGN NOTES
------------------------------
This is a ground-up replacement for the earlier prototype deconvolver, which
assumed reference characteristics that IAMAtlas does not have. The atlas was
measured to behave like this (IAMAtlas v0.1):
  * between-cell-type variance is COMPRESSED (median ~0.0003, max ~0.0067):
    per-cell posterior means sit close together, so individual cell types are
    only weakly separable. Absolute variance thresholds reject everything.
  * between-CLASS variance is much larger and reliable -- the 8 architecture
    classes ARE well separated. This is where the trustworthy signal lives.
  * posterior SD is small (median ~0.011): the atlas is confident.
  * many (cpg, celltype) pairs are EMPTY (cell never measured at that CpG):
    these must never be treated as real values.
  * the matrix is large (~1.2 GB uncompressed): must stream, not load whole.

So the Walther deconvolver:
  1. selects markers by RANK within the atlas (no absolute thresholds), and
     selects them primarily for CLASS discrimination.
  2. solves the CLASS mixture first (reliable), then optionally refines to
     cell types WITHIN each present class (indicative, clearly labelled).
  3. streams the matrix and keeps only marker rows in memory.
  4. is empty-cell aware throughout.
  5. reports honest per-class confidence and fit diagnostics.

USAGE
-----
    from walther_iam_deconvolver import WaltherIAMDeconvolver

    d = WaltherIAMDeconvolver("IAMAtlas.csv",
                              celltype_class_map="IAMAtlas_celltype_to_class.json")
    result = d.deconvolve(customer_betas)        # {cpg_id: beta}
    print(result.class_fractions)                # PRIMARY  {'immune':0.74,...}
    print(result.celltype_fractions)             # SECONDARY (indicative)
    print(result.diagnostics)                    # markers matched, fit, confidence

Requires numpy + scipy.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


# The 8 IAM architecture classes (class-level brightness columns in the matrix)
CLASSES = ["stem_pluri", "stem_adult", "progenitor", "stromal",
           "cycling", "secretory", "immune", "terminal"]


@dataclass
class DeconvolutionResult:
    # PRIMARY output -- trust this
    class_fractions: dict = field(default_factory=dict)
    # SECONDARY output -- indicative only (within-class cell types are
    # weakly separable in v0.1)
    celltype_fractions: dict = field(default_factory=dict)
    # diagnostics
    diagnostics: dict = field(default_factory=dict)
    status: str = "OK"


class WaltherIAMDeconvolver:

    def __init__(self, matrix_path, celltype_class_map=None,
                 n_class_markers_per_class=600,
                 max_celltype_markers=4000,
                 verbose=True):
        """
        Parameters
        ----------
        matrix_path : path to IAMAtlas.csv (decompressed).
        celltype_class_map : path to IAMAtlas_celltype_to_class.json, OR a dict,
                             OR None (then cell-type refinement is disabled and
                             only class-level deconvolution runs).
        n_class_markers_per_class : how many top class-discriminating CpGs to
                             keep per class for the class-level solve.
        max_celltype_markers : cap on CpGs kept for the (secondary) cell-type
                             refinement.
        verbose : print progress.
        """
        self.matrix_path = Path(matrix_path)
        self.verbose = verbose
        self.n_class_markers_per_class = n_class_markers_per_class
        self.max_celltype_markers = max_celltype_markers

        # cell type -> class
        self.celltype_to_class = {}
        if isinstance(celltype_class_map, dict):
            self.celltype_to_class = dict(celltype_class_map)
        elif celltype_class_map is not None:
            with open(celltype_class_map) as f:
                self.celltype_to_class = json.load(f)

        # populated by _scan_matrix
        self.class_cols = {}        # class -> column index of <class>_mean
        self.celltype_cols = {}     # celltype -> (mean_col, sd_col)
        self.celltypes = []

        # marker reference tables (only marker rows kept in memory)
        # class markers: cpg -> {class: mean}
        self.class_ref = {}
        # celltype markers: cpg -> {celltype: (mean, sd)}
        self.celltype_ref = {}

        self._scan_header()
        self._select_markers()

    # ------------------------------------------------------------------
    def _scan_header(self):
        if self.verbose:
            print(f"Scanning header: {self.matrix_path}")
        with open(self.matrix_path) as f:
            header = next(csv.reader(f))
        idx = {name: i for i, name in enumerate(header)}
        # class-level mean columns
        for cls in CLASSES:
            col = f"{cls}_mean"
            if col in idx:
                self.class_cols[cls] = idx[col]
        # per-cell-type mean/sd columns (anything _mean not in CLASSES)
        for name, i in idx.items():
            if name.endswith("_mean"):
                stem = name[:-len("_mean")]
                if stem in CLASSES:
                    continue
                sd_col = idx.get(f"{stem}_sd")
                self.celltype_cols[stem] = (i, sd_col)
        self.celltypes = list(self.celltype_cols.keys())
        if self.verbose:
            print(f"  {len(self.class_cols)} class columns, "
                  f"{len(self.celltypes)} cell-type columns")
        # fill any missing class assignments for detected cell types
        for ct in self.celltypes:
            self.celltype_to_class.setdefault(ct, "unknown")

    # ------------------------------------------------------------------
    @staticmethod
    def _between_var(values):
        """Population variance of a list of floats."""
        n = len(values)
        if n < 2:
            return 0.0
        m = sum(values) / n
        return sum((v - m) ** 2 for v in values) / n

    def _select_markers(self):
        """
        ONE streaming pass over the matrix. For every CpG, compute:
          - between-CLASS variance (using the 8 class-level means present)
          - between-cell-type variance (using per-cell-type means present)
        Keep, by RANK, using BOUNDED MIN-HEAPS so memory stays flat regardless
        of atlas size (we never hold all 483K rows in memory at once):
          - top class-discriminating CpGs -> class_ref
          - top one-vs-rest CpGs per class -> ensures every class is represented
          - top cell-type-discriminating CpGs -> celltype_ref
        No absolute thresholds: selection adapts to the atlas's own scale.
        Empty cells are skipped (never treated as values).
        """
        import heapq
        if self.verbose:
            print("Selecting markers (one streaming pass, bounded memory)...")

        global_quota = self.n_class_markers_per_class * len(CLASSES)
        # min-heaps of (key, tiebreak, payload); smallest key is heap[0], so we
        # pop the smallest when over capacity -> heap retains the top-N largest.
        class_heap = []           # key=class_var, payload=(cpg, cls_means)
        ct_heap = []              # key=ct_var,    payload=(cpg, ct_data)
        per_class_heap = {c: [] for c in CLASSES}  # key=separation, payload=(cpg, cls_means)
        tie = 0

        def push_bounded(heap, key, payload, cap):
            nonlocal tie
            tie += 1
            if len(heap) < cap:
                heapq.heappush(heap, (key, tie, payload))
            elif key > heap[0][0]:
                heapq.heapreplace(heap, (key, tie, payload))

        with open(self.matrix_path) as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                cpg = row[0]

                # ---- class-level ----
                cls_means = {}
                for cls, col in self.class_cols.items():
                    v = row[col]
                    if v not in ("", "NA"):
                        try:
                            fv = float(v)
                            if 0.0 <= fv <= 1.0:
                                cls_means[cls] = fv
                        except ValueError:
                            pass
                if len(cls_means) >= 2:
                    vals = list(cls_means.values())
                    cvar = self._between_var(vals)
                    if cvar > 0:
                        push_bounded(class_heap, cvar, (cpg, cls_means), global_quota)
                        mean_all = sum(vals) / len(vals)
                        for cls, mv in cls_means.items():
                            sep = abs(mv - mean_all)
                            if sep > 0:
                                push_bounded(per_class_heap[cls], sep,
                                             (cpg, cls_means),
                                             self.n_class_markers_per_class)

                # ---- cell-type level ----
                ct_data = {}
                for ct, (mcol, scol) in self.celltype_cols.items():
                    v = row[mcol]
                    if v in ("", "NA"):
                        continue
                    try:
                        fv = float(v)
                    except ValueError:
                        continue
                    if not (0.0 <= fv <= 1.0):
                        continue
                    sd = 0.0
                    if scol is not None:
                        sv = row[scol]
                        if sv not in ("", "NA"):
                            try:
                                sd = float(sv)
                            except ValueError:
                                sd = 0.0
                    ct_data[ct] = (fv, sd)
                if len(ct_data) >= 3:
                    cvar = self._between_var([m for m, _ in ct_data.values()])
                    if cvar > 0:
                        push_bounded(ct_heap, cvar, (cpg, ct_data),
                                     self.max_celltype_markers)

        # ---- assemble class marker reference ----
        chosen = {}
        for _, _, (cpg, means) in class_heap:
            chosen[cpg] = means
        for cls in CLASSES:
            for _, _, (cpg, means) in per_class_heap[cls]:
                chosen.setdefault(cpg, means)
        self.class_ref = chosen

        # ---- assemble cell-type marker reference ----
        self.celltype_ref = {cpg: data for _, _, (cpg, data) in ct_heap}

        if self.verbose:
            print(f"  class markers: {len(self.class_ref)} CpGs")
            print(f"  cell-type markers: {len(self.celltype_ref)} CpGs")

    # ------------------------------------------------------------------
    def _solve_nnls(self, R, y, weights=None):
        import numpy as np
        from scipy.optimize import nnls
        if weights is not None:
            sw = np.sqrt(weights)
            R = R * sw[:, None]
            y = y * sw
        f, _ = nnls(R, y)
        s = f.sum()
        if s > 0:
            f = f / s
        return f

    # ------------------------------------------------------------------
    def deconvolve(self, customer_betas, refine_celltypes=True):
        """
        customer_betas : dict {cpg_id: beta in [0,1]}
        Returns DeconvolutionResult.
        """
        import numpy as np

        # ===== TIER 1: CLASS-LEVEL (primary, reliable) =====
        usable = [c for c in self.class_ref
                  if c in customer_betas
                  and isinstance(customer_betas[c], (int, float))
                  and 0.0 <= customer_betas[c] <= 1.0]
        diag = {"n_customer_cpgs": len(customer_betas),
                "n_class_markers_matched": len(usable)}

        if len(usable) < 50:
            return DeconvolutionResult(
                status="INSUFFICIENT_CLASS_MARKERS",
                diagnostics={**diag, "needed": 50})

        # Build class reference matrix over the present classes only.
        present_classes = [c for c in CLASSES
                           if any(c in self.class_ref[cpg] for cpg in usable)]
        # require each class present at >= 60% of usable markers to be solvable
        cov = {c: sum(1 for cpg in usable if c in self.class_ref[cpg])
               for c in present_classes}
        thr = int(0.6 * len(usable))
        solve_classes = [c for c in present_classes if cov[c] >= thr]
        if len(solve_classes) < 2:
            return DeconvolutionResult(
                status="INSUFFICIENT_CLASS_COVERAGE",
                diagnostics={**diag, "class_coverage": cov})

        n = len(usable)
        k = len(solve_classes)
        R = np.zeros((n, k))
        y = np.zeros(n)
        for i, cpg in enumerate(usable):
            means = self.class_ref[cpg]
            row_vals = [means[c] for c in solve_classes if c in means]
            fill = sum(row_vals) / len(row_vals) if row_vals else 0.5
            for j, c in enumerate(solve_classes):
                R[i, j] = means.get(c, fill)
            y[i] = customer_betas[cpg]

        f = self._solve_nnls(R, y)
        class_fractions = {c: float(f[j]) for j, c in enumerate(solve_classes)}
        # classes not solved get 0
        for c in CLASSES:
            class_fractions.setdefault(c, 0.0)

        # class-level fit residual
        pred = R @ f
        resid = float(np.mean(np.abs(pred - y)))
        diag["class_residual_mae"] = resid
        diag["classes_solved"] = solve_classes
        diag["class_marker_coverage"] = cov

        # per-class confidence: fraction of markers supporting that class,
        # scaled by fit quality. Honest, simple, bounded [0,1].
        fit_quality = max(0.0, 1.0 - resid / 0.2)  # resid 0 ->1, 0.2 ->0
        class_confidence = {c: round(min(1.0, (cov.get(c, 0) / len(usable)) * fit_quality), 3)
                            for c in solve_classes}
        diag["class_confidence"] = class_confidence

        result = DeconvolutionResult(
            class_fractions={c: round(v, 4) for c, v in class_fractions.items()},
            diagnostics=diag,
            status="OK")

        # ===== TIER 2: CELL-TYPE REFINEMENT (secondary, indicative) =====
        if refine_celltypes and self.celltype_ref:
            ct_usable = [c for c in self.celltype_ref
                         if c in customer_betas
                         and isinstance(customer_betas[c], (int, float))
                         and 0.0 <= customer_betas[c] <= 1.0]
            diag["n_celltype_markers_matched"] = len(ct_usable)
            if len(ct_usable) >= 50:
                # cell types with >=80% coverage across ct_usable markers
                ctcov = {}
                for cpg in ct_usable:
                    for ct in self.celltype_ref[cpg]:
                        ctcov[ct] = ctcov.get(ct, 0) + 1
                ct_thr = int(0.8 * len(ct_usable))
                use_ct = [ct for ct, c in ctcov.items() if c >= ct_thr]
                if len(use_ct) >= 2:
                    nn = len(ct_usable)
                    kk = len(use_ct)
                    Rc = np.zeros((nn, kk))
                    yc = np.zeros(nn)
                    wc = np.ones(nn)
                    for i, cpg in enumerate(ct_usable):
                        d = self.celltype_ref[cpg]
                        gm = sum(m for m, _ in d.values()) / len(d)
                        for j, ct in enumerate(use_ct):
                            Rc[i, j] = d[ct][0] if ct in d else gm
                        yc[i] = customer_betas[cpg]
                        msd = max((sd for _, sd in d.values()), default=1e-3)
                        wc[i] = 1.0 / max(msd, 1e-3)
                    fc = self._solve_nnls(Rc, yc, wc)
                    ct_fr = {ct: round(float(fc[j]), 4)
                             for j, ct in enumerate(use_ct) if fc[j] > 1e-4}
                    result.celltype_fractions = dict(
                        sorted(ct_fr.items(), key=lambda x: -x[1]))
                    diag["celltype_note"] = (
                        "INDICATIVE ONLY: within-class cell types are weakly "
                        "separable in IAMAtlas v0.1; trust class_fractions for "
                        "decisions. Per-cell-type weight may shift between "
                        "similar cells in the same class.")
        return result


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Walther IAM Deconvolver")
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--map", default=None, help="celltype_to_class.json")
    ap.add_argument("--betas", required=True, help="JSON {cpg_id: beta}")
    args = ap.parse_args()
    d = WaltherIAMDeconvolver(args.matrix, celltype_class_map=args.map)
    with open(args.betas) as f:
        betas = json.load(f)
    r = d.deconvolve(betas)
    print("\nstatus:", r.status)
    print("CLASS fractions (PRIMARY):")
    for c, v in sorted(r.class_fractions.items(), key=lambda x: -x[1]):
        print(f"  {c:<14} {v:.4f}")
    if r.celltype_fractions:
        print("cell-type fractions (INDICATIVE):")
        for c, v in list(r.celltype_fractions.items())[:10]:
            print(f"  {c:<22} {v:.4f}")
    print("diagnostics:", json.dumps(r.diagnostics, indent=2, default=str))
