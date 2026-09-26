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


# Below this bootstrap-fraction lower-CI bound a class is treated as numerical
# dust (NNLS returns ~1e-15 noise), not a real detection. This is a numerical
# floor to separate true zero from float noise -- NOT a biological threshold;
# the biological detection emerges from the bootstrap CI itself.
PRESENCE_NUMERICAL_ZERO = 1e-6


@dataclass
class DeconvolutionResult:
    # PRIMARY output -- trust this
    class_fractions: dict = field(default_factory=dict)
    # CELL-TYPE output -- per-cell fractions. The rebuilt IAMAtlas is separable
    # at the cell level (IAMAtlas_FLATNESS_LESSON.md), so these carry their own
    # bootstrap presence gate (celltype_present) rather than riding the class call.
    celltype_fractions: dict = field(default_factory=dict)
    celltype_unresolvable: list = field(default_factory=list)
    celltype_twins_dropped: dict = field(default_factory=dict)
    celltype_families: dict = field(default_factory=dict)
    celltype_shared: dict = field(default_factory=dict)
    celltype_exclusive_n: dict = field(default_factory=dict)
    # diagnostics
    diagnostics: dict = field(default_factory=dict)
    status: str = "OK"
    # presence gate (bootstrap detection test): per-class fraction CI and a
    # present/absent verdict. A class is PRESENT iff the lower CI bound > 0.
    class_fraction_ci: dict = field(default_factory=dict)
    class_present: dict = field(default_factory=dict)
    presence_method: str = "none"
    # CELL-LEVEL presence gate (same bootstrap detection test, applied per cell
    # type). A cell is PRESENT iff the lower bound of its bootstrap fraction CI
    # > PRESENCE_NUMERICAL_ZERO. Consumed by the Walther-cell ∩ NILC-cell
    # agreement gate downstream.
    celltype_fraction_ci: dict = field(default_factory=dict)
    celltype_present: dict = field(default_factory=dict)
    celltype_presence_method: str = "none"


class WaltherIAMDeconvolver:

    def __init__(self, matrix_path, celltype_class_map=None,
                 n_class_markers_per_class=600,
                 max_celltype_markers=4000,
                 n_celltype_markers_per_celltype=60,
                 min_celltype_coverage=0.01,
                 verbose=True,
                 contrast_pairs=None,
                 n_contrast_markers_per_pair=300):
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
        # 2026-09-26: EVERY CELL GETS MARKERS. The cell-type reference was one global top-N by between-cell
        # variance, which five stomach entries dominated (2,400 of 4,000) while 26 cells - Breast, Prostate,
        # Kidney, Lung, Bladder, Uterus and 14 immune labels - had NONE, so the solver could not return them for
        # any specimen: pure breast tissue read as neurons and stomach. The class level already keeps a per-class
        # heap 'so every class is represented'; this is the same rule one level down (author's greenlit repair).
        self.n_celltype_markers_per_celltype = n_celltype_markers_per_celltype
        # 2026-09-26 COVERAGE: the atlas is twelve source families defined on 252 to 482,421 loci. A cell defined
        # at 252 loci was being FILLED with the grand mean at every other marker, so it absorbed mass from every
        # specimen (the 'erythroblast hub'); a cell defined at 6,105 loci (Breast, Colon, Prostate, Kidney, Lung,
        # Liver ...) could never win a marker ranked over 483k loci and so was unfindable in its own tissue.
        # Now: only cells defined on >= min_celltype_coverage of the atlas enter the cell-level solve, and cell
        # markers are drawn only from loci where EVERY candidate cell is defined - no filling. Sub-floor cells stay
        # in the atlas as reference and are reported as 'not resolvable on this platform', not as fraction 0.
        self.min_celltype_coverage = min_celltype_coverage
        self.celltype_coverage = {}
        self.celltype_unresolvable = []
        # PROC-SEP-01 (2026-09-19): optional pairwise-contrast markers. The default per-class
        # criterion sep = |class mean - field mean| never selects a CpG where two classes differ
        # from EACH OTHER but both sit near the field mean - which is exactly the HSC / progenitor
        # case (1,290 Atlas CpGs > 0.2 apart; ~129 chosen). For each (a, b) in contrast_pairs the
        # top n_contrast_markers_per_pair CpGs by |mean_a - mean_b| are forced into class_ref.
        # OFF by default (None): sealed results are unchanged unless a caller opts in.
        self.contrast_pairs = [tuple(p) for p in contrast_pairs] if contrast_pairs else []
        self.n_contrast_markers_per_pair = n_contrast_markers_per_pair

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
        self._scan_coverage()
        self._select_markers()
        self._resolve_twins()
        self._add_exclusive_markers()

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

    def _scan_coverage(self):
        """Find the SOLVE BLOCK: the cells and loci that are mutually defined, so the cell solve never fills.
        Pass 1: coverage per cell; candidates = cells defined on >= min_celltype_coverage of the atlas.
        Pass 2: a locus enters the block if >= 80% of candidates are defined there; a candidate stays if it is
        defined at >= 90% of block loci. The intersection-of-everyone rule left 12 loci (2026-09-26): entries
        from small sources at 1-2% coverage do not overlap the 6,105-locus solid-tissue family, and the block
        must be the largest mutually-covered set, not everyone's intersection."""
        counts = {ct: 0 for ct in self.celltype_cols}
        n = 0
        with open(self.matrix_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                n += 1
                for ct, (mcol, _) in self.celltype_cols.items():
                    if row[mcol] not in ("", "NA"):
                        counts[ct] += 1
        self.n_atlas_loci = n
        self.celltype_coverage = {ct: c / n for ct, c in counts.items()} if n else {}
        _BULK_NAMES = ("pbmc", "whole_blood", "buffy", "leukocyte", "_blood", "blood_", "plasma", "granulocytes",
                       "mononuclear", "wbc", "bulk")
        cand = [ct for ct, c in self.celltype_coverage.items() if c >= self.min_celltype_coverage
                and not any(b in ct.lower() for b in _BULK_NAMES)]   # aggregates are not cells; out before families form
        # pass 2: per-locus count of defined candidates -> block mask; then candidates' coverage within the block
        # choose the block by MAXIMISING THE NUMBER OF CELLS resolvable together (>= 2,000 shared loci), not a fixed 80%:
        # a fixed fraction collapsed from 5,343 loci to 518 when three aggregates left the candidate list (2026-09-26)
        per_locus = []   # (cpg, frozenset of defined candidates)
        with open(self.matrix_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                defined = tuple(ct for ct in cand if row[self.celltype_cols[ct][0]] not in ("", "NA"))
                if len(defined) >= 3:
                    per_locus.append((row[0], defined))
        best = ((0, 0), None, None)
        for need in range(3, len(cand) + 1):
            loci = [(c, d) for c, d in per_locus if len(d) >= need]
            if len(loci) < 200:
                break
            cnt = {ct: 0 for ct in cand}
            for _, d in loci:
                for ct in d:
                    cnt[ct] += 1
            kept = [ct for ct in cand if cnt[ct] / len(loci) >= 0.9]
            # objective: the MOST CELLS the array can resolve together, given at least 2,000 mutually-defined loci;
            # cells x loci picked 384k loci x 21 blood/brain/stomach cells and lost every solid tissue (2026-09-26)
            score = (len(kept) if len(loci) >= 2000 else 0, len(loci))
            if score > best[0]:
                best = (score, need, kept)
        _, need, kept = best
        self._block = {c for c, d in per_locus if len(d) >= need}
        incount = {ct: 0 for ct in cand}
        for c, d in per_locus:
            if len(d) >= need:
                for ct in d:
                    incount[ct] += 1
        nb = max(1, len(self._block))
        self.celltype_candidates = sorted(kept)
        self.celltype_unresolvable = sorted(ct for ct in self.celltype_cols if ct not in self.celltype_candidates)
        self.celltype_block_coverage = {ct: incount.get(ct, 0) / nb for ct in cand}
        if self.verbose:
            print(f"  coverage: solve block = {len(self._block):,} loci x {len(self.celltype_candidates)} cells "
                  f"(mutually defined); {len(self.celltype_unresolvable)} cells are reference-only on this platform")

    def _resolve_twins(self, twin_r=0.985):
        """Cells the array cannot tell apart are not solved apart (2026-09-26). Single-linkage on marker-profile
        correlation r > twin_r within one architecture class. Within a cluster: a member from a LOWER-coverage source is
        the same cell measured on 1% of the array and is dropped; members of EQUAL coverage (same source) become a
        RESOLUTION FAMILY solved as one column, whose fraction is reported to every member flagged as shared."""
        import numpy as np
        cand = list(self.celltype_candidates)
        if not self.celltype_ref or len(cand) < 2:
            self.celltype_families = {}; self.celltype_twins_dropped = {}; return
        cpgs = list(self.celltype_ref)
        M = np.array([[self.celltype_ref[c][ct][0] for ct in cand] for c in cpgs], dtype=float)
        # CORRELATION, not |delta beta|: a cross-platform copy of the same cell is offset by 0.02-0.04 in beta but
        # keeps the pattern (EPIC copies r 0.988-0.998 to their 450K originals; granulocytes/neutrophils 0.999),
        # while genuinely different cells sit at r <= 0.978 (CD4 vs CD8; neutrophil vs monocyte). Same class required.
        Mc = M - np.nanmean(M, axis=0, keepdims=True)
        Mc = np.nan_to_num(Mc)
        nrm = np.linalg.norm(Mc, axis=0); nrm[nrm == 0] = 1.0
        Rm = (Mc / nrm).T @ (Mc / nrm)
        cls = {ct: self.celltype_to_class.get(ct) for ct in cand}
        # single linkage
        parent = list(range(len(cand)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                # cross-SOURCE pair (one member on < 5% of the array, the other on > 50%) is the same cell measured on
                # another platform: merge at r > 0.98 (NK-cells_EPIC sat at 0.984 and took 72% of the 'non-blood' mass
                # in healthy blood, 2026-09-26). Equal-coverage pairs keep the stricter bar: CD4 vs CD8 is 0.978.
                ci, cj = self.celltype_coverage.get(cand[i], 0), self.celltype_coverage.get(cand[j], 0)
                cross = (min(ci, cj) < 0.05 and max(ci, cj) > 0.5)
                thr = 0.98 if cross else twin_r
                if Rm[i, j] > thr and cls[cand[i]] == cls[cand[j]]:
                    parent[find(i)] = find(j)
        clusters = {}
        for i, ct in enumerate(cand):
            clusters.setdefault(find(i), []).append(ct)
        self.celltype_families = {}; self.celltype_twins_dropped = {}
        keep = set(cand)
        for members in clusters.values():
            if len(members) < 2:
                continue
            cov = {m: round(self.celltype_coverage.get(m, 0.0), 3) for m in members}
            top = max(cov.values())
            high = [m for m in members if cov[m] >= top - 0.05]
            low = [m for m in members if m not in high]
            for m in low:
                keep.discard(m); self.celltype_twins_dropped[m] = high[0]
            if len(high) > 1:
                fam = "family:" + "+".join(sorted(high))
                self.celltype_families[fam] = sorted(high)
                for m in high:
                    keep.discard(m)
                keep.add(fam)
                for c in cpgs:
                    d = self.celltype_ref[c]
                    ms = [d[m][0] for m in high if m in d]; sds = [d[m][1] for m in high if m in d]
                    if ms:
                        d[fam] = (sum(ms) / len(ms), max(sds) if sds else 0.0)
        self.celltype_solve_columns = sorted(keep)
        if self.verbose:
            print(f"  twins: {len(self.celltype_twins_dropped)} lower-coverage twins dropped "
                  f"{sorted(self.celltype_twins_dropped)}; {len(self.celltype_families)} resolution families: "
                  + "; ".join(f"{k}={v}" for k, v in self.celltype_families.items()))

    def _add_exclusive_markers(self, margin=0.15):
        """UNIQUENESS markers (author, 2026-09-26: 'filter them based on their uniqueness rather than their
        similarity'). For every solve column, block loci where the column beats its NEAREST RIVAL by > margin are
        added to the cell-type reference. Measured: on constructed mixtures these cut the maximum composition error
        from 0.06 to 0.04 with no spurious cell; alone they leave the blood background unpinned (9.7% non-blood in
        healthy blood), so they are a UNION with the variance-ranked set (2.4%), never a replacement."""
        import numpy as np
        cols = list(getattr(self, "celltype_solve_columns", []))
        cand = list(self.celltype_candidates)
        if len(cols) < 3:
            self.n_exclusive_added = 0; return
        fam = self.celltype_families
        added = 0
        self.exclusive_markers = {c: [] for c in cols}
        with open(self.matrix_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                cpg = row[0]
                if cpg not in self._block:
                    continue
                d = {}
                ok = True
                for ct in cand:
                    mcol, scol = self.celltype_cols[ct]
                    v = row[mcol]
                    if v in ("", "NA"):
                        ok = False; break
                    sd = 0.0
                    if scol is not None and row[scol] not in ("", "NA"):
                        try: sd = float(row[scol])
                        except ValueError: sd = 0.0
                    d[ct] = (float(v), sd)
                if not ok:
                    continue
                prof = {}
                for c in cols:
                    if c in fam:
                        ms = [d[m][0] for m in fam[c] if m in d]
                        prof[c] = (sum(ms) / len(ms), max(d[m][1] for m in fam[c] if m in d)) if ms else None
                    else:
                        prof[c] = d.get(c)
                if any(v is None for v in prof.values()):
                    continue
                vals = np.array([prof[c][0] for c in cols])
                for j, c in enumerate(cols):
                    others = np.delete(vals, j)
                    if np.min(np.abs(others - vals[j])) > margin:
                        self.exclusive_markers[c].append(cpg)
                        if cpg not in self.celltype_ref:
                            entry = dict(d)
                            for fc in fam:
                                if prof.get(fc) is not None:
                                    entry[fc] = prof[fc]
                            self.celltype_ref[cpg] = entry
                            added += 1
                        break
        self.n_exclusive_added = added
        if self.verbose:
            thin = sorted((c, len(v)) for c, v in self.exclusive_markers.items() if len(v) < 20)
            print(f"  exclusive markers: {added} added (union with the variance set -> {len(self.celltype_ref)}); "
                  f"columns with < 20 exclusive loci: {thin}")

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
        self._cand_set = set(getattr(self, 'celltype_candidates', list(self.celltype_cols)))
        # min-heaps of (key, tiebreak, payload); smallest key is heap[0], so we
        # pop the smallest when over capacity -> heap retains the top-N largest.
        class_heap = []           # key=class_var, payload=(cpg, cls_means)
        ct_heap = []              # key=ct_var,    payload=(cpg, ct_data)
        per_ct_heap = {}          # key=one-vs-rest separation, payload=(cpg, ct_data); one heap per cell type
        per_class_heap = {c: [] for c in CLASSES}  # key=separation, payload=(cpg, cls_means)
        pair_heap = {pr: [] for pr in self.contrast_pairs}  # key=|mean_a - mean_b|
        tie = 0

        def push_bounded(heap, key, payload, cap):
            nonlocal tie
            tie += 1
            if cap <= 0:
                return
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
                        for pr in self.contrast_pairs:
                            a, b = pr
                            if a in cls_means and b in cls_means:
                                d = abs(cls_means[a] - cls_means[b])
                                if d > 0:
                                    push_bounded(pair_heap[pr], d, (cpg, cls_means),
                                                 self.n_contrast_markers_per_pair)

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
                # coverage rule: candidates only, and every candidate must be defined here (no filling)
                ct_data = {ct: v for ct, v in ct_data.items() if ct in self._cand_set}
                if cpg in self._block and len(ct_data) == len(self._cand_set) and len(ct_data) >= 3:
                    cvar = self._between_var([m for m, _ in ct_data.values()])
                    if cvar > 0:
                        push_bounded(ct_heap, cvar, (cpg, ct_data),
                                     self.max_celltype_markers)
                        # per-cell quota: this cell's separation from the mean of ALL OTHER cells at this locus
                        if self.n_celltype_markers_per_celltype > 0:
                            _tot = sum(m for m, _ in ct_data.values()); _n = len(ct_data)
                            for _ct, (_m, _) in ct_data.items():
                                _rest = (_tot - _m) / (_n - 1)
                                _sep = abs(_m - _rest)
                                if _sep > 0:
                                    push_bounded(per_ct_heap.setdefault(_ct, []), _sep, (cpg, ct_data),
                                                 self.n_celltype_markers_per_celltype)

        # ---- assemble class marker reference ----
        chosen = {}
        for _, _, (cpg, means) in class_heap:
            chosen[cpg] = means
        for cls in CLASSES:
            for _, _, (cpg, means) in per_class_heap[cls]:
                chosen.setdefault(cpg, means)
        self.n_contrast_added = 0
        for pr in self.contrast_pairs:
            for _, _, (cpg, means) in pair_heap[pr]:
                if cpg not in chosen:
                    chosen[cpg] = means; self.n_contrast_added += 1
        self.class_ref = chosen

        # ---- assemble cell-type marker reference ----
        self.celltype_ref = {cpg: data for _, _, (cpg, data) in ct_heap}
        self.n_celltype_quota_added = 0
        for _ct, _h in per_ct_heap.items():
            for _, _, (cpg, data) in _h:
                if cpg not in self.celltype_ref:
                    self.celltype_ref[cpg] = data; self.n_celltype_quota_added += 1

        if self.verbose:
            print(f"  class markers: {len(self.class_ref)} CpGs" + (f" (incl. {self.n_contrast_added} contrast markers for {self.contrast_pairs})" if self.contrast_pairs else ""))
            print(f"  cell-type markers: {len(self.celltype_ref)} CpGs (incl. {self.n_celltype_quota_added} from the per-cell quota of {self.n_celltype_markers_per_celltype})")

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
    def deconvolve(self, customer_betas, refine_celltypes=True,
                   presence_bootstrap=True, n_boot=200, presence_seed=0,
                   presence_ci=(2.5, 97.5)):
        """
        customer_betas : dict {cpg_id: beta in [0,1]}
        presence_bootstrap : if True, bootstrap the class fractions to produce
            per-class CIs and a present/absent verdict (the substrate presence
            gate). n_boot resamples of the matched markers, re-solved by NNLS.
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

        # --- Presence gate (bootstrap detection test) ------------------------
        # Resample the matched markers with replacement, re-solve NNLS, and call
        # a class PRESENT iff the lower bound of its bootstrap fraction CI > 0.
        # NNLS non-negativity makes this self-calibrating: a truly-absent class
        # is pinned to 0 in many resamples (lower CI -> 0 -> NOT present); a
        # present class carries weight in nearly all resamples (lower CI > 0).
        # Also supplies the per-class fraction CIs the BUILD_SPEC asks for.
        class_fraction_ci, class_present = {}, {}
        if presence_bootstrap and n_boot and n >= 1:
            rng = np.random.default_rng(presence_seed)
            boot = np.empty((int(n_boot), k))
            for b in range(int(n_boot)):
                idx = rng.integers(0, n, size=n)
                boot[b] = self._solve_nnls(R[idx], y[idx])
            lo_p, hi_p = presence_ci
            los = np.percentile(boot, lo_p, axis=0)
            his = np.percentile(boot, hi_p, axis=0)
            for j, c in enumerate(solve_classes):
                class_fraction_ci[c] = [round(float(los[j]), 4), round(float(his[j]), 4)]
                class_present[c] = bool(los[j] > PRESENCE_NUMERICAL_ZERO)
            presence_method = f"bootstrap_marker_resample_n{int(n_boot)}_ci{lo_p}-{hi_p}"
            diag["presence_n_boot"] = int(n_boot)
        else:
            for j, c in enumerate(solve_classes):
                class_fraction_ci[c] = [round(float(f[j]), 4), round(float(f[j]), 4)]
                class_present[c] = bool(f[j] > PRESENCE_NUMERICAL_ZERO)
            presence_method = "point_fraction_no_bootstrap"
            diag["presence_n_boot"] = 0
        # classes that never entered the solve are not assessable in this substrate
        for c in CLASSES:
            class_fraction_ci.setdefault(c, [0.0, 0.0])
            class_present.setdefault(c, False)

        result = DeconvolutionResult(
            class_fractions={c: round(v, 4) for c, v in class_fractions.items()},
            diagnostics=diag,
            status="OK",
            class_fraction_ci=class_fraction_ci,
            class_present=class_present,
            presence_method=presence_method)

        # ===== TIER 2: CELL-TYPE RESOLUTION (own bootstrap presence gate) =====
        # The rebuilt IAMAtlas is separable at the cell level, so cell types get
        # the SAME bootstrap detection test the class level uses above: resample
        # the matched markers, re-solve NNLS, and call a cell PRESENT iff the
        # lower bound of its bootstrap fraction CI > PRESENCE_NUMERICAL_ZERO.
        # The Walther-cell ∩ NILC-cell agreement gate consumes celltype_present.
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
                # Drop bulk/aggregate pseudo-cells at the CELL level (they absorb
                # mass and starve real cells); matches NILC-cell and the chain _AGG.
                _BULK = ("pbmc", "whole_blood", "buffy", "leukocyte", "_blood",
                         "blood_", "plasma", "granulocytes", "mononuclear", "wbc", "bulk")
                use_ct = [ct for ct, c in ctcov.items() if c >= ct_thr
                          and not any(b in ct.lower() for b in _BULK)]
                # coverage/twin rule (2026-09-26): only the solve columns - dropped twins and family members are
                # not solved as separate columns; a family column is solved once and its fraction shared out below
                _solve_cols = set(getattr(self, 'celltype_solve_columns', use_ct))
                use_ct = [ct for ct in use_ct if ct in _solve_cols]
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
                    result.celltype_unresolvable = list(self.celltype_unresolvable)
                    result.celltype_twins_dropped = dict(self.celltype_twins_dropped)
                    result.celltype_families = dict(self.celltype_families)
                    result.celltype_exclusive_n = {c: len(v) for c, v in getattr(self, 'exclusive_markers', {}).items()}
                    # every family member receives the family fraction, flagged shared - never five measurements
                    _exp = {}
                    for k, v in result.celltype_fractions.items():
                        if k in self.celltype_families:
                            for m in self.celltype_families[k]:
                                _exp[m] = v
                        else:
                            _exp[k] = v
                    result.celltype_fractions = dict(sorted(_exp.items(), key=lambda x: -x[1]))
                    result.celltype_shared = {m: k for k, ms in self.celltype_families.items() for m in ms}

                    # --- cell-level presence gate (bootstrap marker resample) ---
                    ct_ci = {}
                    ct_present = {}
                    if presence_bootstrap and n_boot and nn >= 1:
                        rng = np.random.default_rng(presence_seed + 1)
                        bootc = np.zeros((int(n_boot), kk))
                        for b in range(int(n_boot)):
                            idx = rng.integers(0, nn, nn)
                            bootc[b] = self._solve_nnls(Rc[idx], yc[idx], wc[idx])
                        lo_p, hi_p = presence_ci
                        los = np.percentile(bootc, lo_p, axis=0)
                        his = np.percentile(bootc, hi_p, axis=0)
                        for j, ct in enumerate(use_ct):
                            ct_ci[ct] = [round(float(los[j]), 4), round(float(his[j]), 4)]
                            ct_present[ct] = bool(los[j] > PRESENCE_NUMERICAL_ZERO)
                        ct_method = f"bootstrap_marker_resample_n{int(n_boot)}_ci{lo_p}-{hi_p}"
                    else:
                        for j, ct in enumerate(use_ct):
                            ct_ci[ct] = [round(float(fc[j]), 4), round(float(fc[j]), 4)]
                            ct_present[ct] = bool(fc[j] > PRESENCE_NUMERICAL_ZERO)
                        ct_method = "point_fraction_no_bootstrap"
                    result.celltype_fraction_ci = ct_ci
                    result.celltype_present = ct_present
                    result.celltype_presence_method = ct_method
                    diag["n_celltype_present"] = int(sum(ct_present.values()))
                    diag["celltype_note"] = (
                        "Per-cell fractions carry their own bootstrap presence "
                        "gate (celltype_present). The rebuilt IAMAtlas is "
                        "separable at the cell level; cell calls are admitted by "
                        "the Walther-cell n NILC-cell agreement gate, not by the "
                        "class call.")
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
