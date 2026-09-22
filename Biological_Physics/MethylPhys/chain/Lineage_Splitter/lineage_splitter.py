"""lineage_splitter.py — Tool B of the two-tool composition design (PROC-SEP-03, 2026-09-19).

Tool A (WaltherIAMDeconvolver) answers "what is in the tube" on field-ranked class markers.
Tool B takes ONE compartment Tool A found (default: haematopoietic = progenitor + stem_adult) and asks
only "how does that mass divide along the lineage", using ONLY the CpGs where the member classes differ
(contrast CpGs, weighted by |mean_a - mean_b|), solving a small non-negative problem on the compartment
mass, and carrying its own condition-number check. If the sub-problem is ill-conditioned for THIS
sample the tool reports the compartment and says why. It cannot manufacture information the Atlas
lacks; it can say honestly whether it is there.

Usage
-----
    from lineage_splitter import LineageSplitter
    L = LineageSplitter(atlas_csv, members=("progenitor","stem_adult"), min_delta=0.2, kappa_max=10.0)
    r = L.split(beta_dict, compartment_fraction=0.12)
    r.status  -> "SPLIT" | "COMPARTMENT_ONLY"
    r.fractions -> {"progenitor": ..., "stem_adult": ...}   (sum to compartment_fraction when SPLIT)
    r.kappa, r.n_cpgs, r.reason
"""
import csv, json
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import nnls

@dataclass
class SplitResult:
    status: str
    fractions: dict
    compartment_fraction: float
    kappa: float
    n_cpgs: int
    residual_mae: float
    reason: str = ""
    weights_used: bool = True

class LineageSplitter:
    def __init__(self, atlas_csv, members=("progenitor", "stem_adult"), min_delta=0.20,
                 max_cpgs=2000, kappa_max=10.0, weight_power=1.0, verbose=False):
        self.members = tuple(members); self.min_delta = min_delta; self.max_cpgs = max_cpgs
        self.kappa_max = kappa_max; self.weight_power = weight_power
        self.ref = {}   # cpg -> np.array of member means
        with open(atlas_csv, newline="") as f:
            rd = csv.reader(f); hdr = next(rd)
            cols = [hdr.index(f"{m}_mean") for m in self.members]
            cand = []
            for row in rd:
                try: v = [float(row[c]) for c in cols]
                except ValueError: continue
                if any(not (0.0 <= x <= 1.0) for x in v): continue
                d = max(v) - min(v)
                if d >= min_delta: cand.append((d, row[0], v))
            cand.sort(reverse=True)
            for d, cpg, v in cand[:max_cpgs]: self.ref[cpg] = np.array(v)
        if verbose: print(f"[LineageSplitter] {len(self.ref)} contrast CpGs for {self.members} (|delta| >= {min_delta})")

    def split(self, beta, compartment_fraction, background=None):
        """beta: dict cpg->beta. compartment_fraction: Tool A's mass for the compartment.
        background: optional dict cpg->beta of the NON-compartment mixture (Tool A's other classes
        times their reference) to subtract; if None, the compartment is fit directly to the observed beta
        scaled by its fraction (adequate when the contrast CpGs are near-invariant across other classes)."""
        cpgs = [c for c in self.ref if c in beta]
        if len(cpgs) < 50:
            return SplitResult("COMPARTMENT_ONLY", {m: None for m in self.members}, compartment_fraction,
                               float("nan"), len(cpgs), float("nan"), "fewer than 50 contrast CpGs present")
        X = np.array([self.ref[c] for c in cpgs]); y = np.array([beta[c] for c in cpgs])
        if background is not None:
            y = y - np.array([background.get(c, 0.0) for c in cpgs])
        else:
            y = y * compartment_fraction
        w = (X.max(1) - X.min(1)) ** self.weight_power
        sw = np.sqrt(w)
        Xw = X * sw[:, None] * compartment_fraction; yw = y * sw
        kappa = float(np.linalg.cond(Xw))
        f, _ = nnls(Xw, yw)
        tot = f.sum()
        fr = {m: float(f[i]) * compartment_fraction / tot if tot > 0 else 0.0 for i, m in enumerate(self.members)}
        resid = float(np.mean(np.abs(Xw @ f - yw)))
        if kappa > self.kappa_max:
            return SplitResult("COMPARTMENT_ONLY", {m: None for m in self.members}, compartment_fraction,
                               kappa, len(cpgs), resid, f"sub-problem condition number {kappa:.1f} > {self.kappa_max}; the Atlas does not determine this split for this sample")
        return SplitResult("SPLIT", fr, compartment_fraction, kappa, len(cpgs), resid, "")
