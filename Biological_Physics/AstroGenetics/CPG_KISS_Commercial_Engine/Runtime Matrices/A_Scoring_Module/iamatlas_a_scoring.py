"""IAMAtlas A-score scoring module.

Two scoring surfaces, both computing A = mean(H(beta) / H_min(class)) at marker CpGs.

  score_per_class(customer_betas, ...)     -> 8 class A-scores
  score_per_celltype(customer_betas, ...)  -> 115 cell-type A-scores

Both return per-result diagnostics: A-score, coverage, confidence, status.

This module is SEPARATE from the Walther IAM Deconvolver. The deconvolver
computes cell-type FRACTIONS via NNLS; this module computes cell-type A-SCORES
via entropy-at-markers. Different math, different CpG sets, different failure
modes - kept separate so each component is independently testable.

Companion artifacts (must be loaded once at session start):
  - iamatlas_celltype_markers_v0_1.json  (the per-class + per-celltype marker lists)
  - H_min_by_class table                  (the 8 architectural floors)

Author: WaltherMayer + Heath W. Mahaffey
Build session: EDEAR_Physics_Roadmap TODO 1.1 (2026-05-29)
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Status codes returned in every result dict
STATUS_OK = "OK"
STATUS_INSUFFICIENT_MARKERS = "INSUFFICIENT_MARKERS"
STATUS_NO_MARKER_OVERLAP = "NO_MARKER_OVERLAP"

# Tunable gate: minimum number of matched markers to attempt scoring
MIN_MARKERS_FOR_SCORING = 20

# Confidence saturation residual: scores with mean-pairwise-deviation above this floor map to fit_quality=0
CONFIDENCE_RESIDUAL_FLOOR = 0.20


def _shannon_bits(beta_array: np.ndarray) -> np.ndarray:
    """Per-element Shannon entropy in bits, NaN-safe.
    H(b) = -b log2 b - (1-b) log2(1-b) for b in (0,1), 0 elsewhere.
    """
    b = np.asarray(beta_array, dtype=float)
    out = np.zeros_like(b)
    valid = (b > 0) & (b < 1) & np.isfinite(b)
    bv = b[valid]
    out[valid] = -bv * np.log2(bv) - (1 - bv) * np.log2(1 - bv)
    return out


def _scoring_confidence(coverage_frac: float, dispersion: float) -> float:
    """Confidence for an A-score = coverage_frac * fit_quality, bounded [0,1].

    coverage_frac = (n_markers_matched / n_markers_expected)
    fit_quality = max(0, 1 - dispersion / CONFIDENCE_RESIDUAL_FLOOR)
    dispersion = stdev of (H(beta)/H_min) across the matched markers; lower = cleaner.
    """
    fit_quality = max(0.0, 1.0 - dispersion / CONFIDENCE_RESIDUAL_FLOOR)
    return float(min(1.0, coverage_frac * fit_quality))


def _score_one(beta_series: pd.Series,
               marker_cpgs: List[str],
               h_min: float) -> Dict:
    """Score a single (cell-type or class) A-score on one patient.

    beta_series: pandas Series indexed by CpG ID, values are beta floats in [0,1]
    marker_cpgs: list of CpG IDs for this target
    h_min:       architectural floor for the target's class
    """
    n_expected = len(marker_cpgs)
    matched = [c for c in marker_cpgs if c in beta_series.index]
    n_matched = len(matched)

    if n_matched == 0:
        return {
            "A": float("nan"),
            "n_markers_expected": n_expected,
            "n_markers_matched": 0,
            "coverage": 0.0,
            "confidence": 0.0,
            "status": STATUS_NO_MARKER_OVERLAP,
        }

    vals = beta_series.loc[matched].dropna().astype(float).values
    n_usable = len(vals)
    if n_usable < MIN_MARKERS_FOR_SCORING:
        return {
            "A": float("nan"),
            "n_markers_expected": n_expected,
            "n_markers_matched": n_usable,
            "coverage": n_usable / n_expected,
            "confidence": 0.0,
            "status": STATUS_INSUFFICIENT_MARKERS,
        }

    per_cpg_a = _shannon_bits(vals) / h_min
    a_score = float(np.mean(per_cpg_a))
    dispersion = float(np.std(per_cpg_a, ddof=1)) if n_usable >= 2 else 0.0
    coverage = n_usable / n_expected
    confidence = _scoring_confidence(coverage, dispersion)
    return {
        "A": a_score,
        "n_markers_expected": n_expected,
        "n_markers_matched": n_usable,
        "coverage": coverage,
        "confidence": confidence,
        "status": STATUS_OK,
    }


def score_per_class(customer_betas: Dict[str, float],
                    class_markers: Dict[str, List[str]],
                    h_min_by_class: Dict[str, float]) -> Dict[str, Dict]:
    """Score all 8 architecture-class A-scores for one patient.

    customer_betas:   {cpg_id: beta} dict
    class_markers:    {class_name: [cpg_id, ...]} dict, e.g. from the marker artifact
                       (this is the per-class one-vs-rest markers, not the deconvolver's class_ref)
    h_min_by_class:   {class_name: H_min float} dict

    Returns: {class_name: {A, n_markers_expected, n_markers_matched, coverage, confidence, status}}
    """
    beta_series = pd.Series(customer_betas)
    out = {}
    for cls, markers in class_markers.items():
        if cls not in h_min_by_class:
            continue
        out[cls] = _score_one(beta_series, markers, h_min_by_class[cls])
    return out


def score_per_celltype(customer_betas: Dict[str, float],
                       celltype_markers: Dict[str, List[str]],
                       celltype_to_class: Dict[str, str],
                       h_min_by_class: Dict[str, float]) -> Dict[str, Dict]:
    """Score all 115 cell-type A-scores for one patient.

    Each cell type's H_min is looked up via its class membership.
    """
    beta_series = pd.Series(customer_betas)
    out = {}
    for ct, markers in celltype_markers.items():
        cls = celltype_to_class.get(ct)
        if cls is None or cls not in h_min_by_class:
            continue
        result = _score_one(beta_series, markers, h_min_by_class[cls])
        result["class"] = cls
        out[ct] = result
    return out


def load_artifact(artifact_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load the iamatlas_celltype_markers artifact and return its component dicts.

    Returns: (artifact_metadata, celltype_markers, celltype_to_class, h_min_by_class)
    """
    with open(artifact_path) as f:
        artifact = json.load(f)
    return (
        {k: v for k, v in artifact.items() if k != "markers_by_celltype"},
        artifact["markers_by_celltype"],
        artifact["celltype_to_class"],
        artifact["H_min_by_class"],
    )


def regression_guard_bimodal():
    """LESSON-ASCORE-02 guard (SOP v1.4.0 §105).

    The A-score MUST be mean_i(H(beta_i)/H_min) -- the mean of per-CpG
    entropies -- NEVER H(beta_mean)/H_min. This guard scores a known bimodal
    panel and asserts the mean-of-per-CpG value. Any build that reverts to the
    entropy-of-the-mean form (the 2026-06-11 regression) fails here.
    """
    import pandas as _pd
    import math as _math

    def _H(b):
        if b <= 0.0 or b >= 1.0:
            return 0.0
        return -b * _math.log2(b) - (1 - b) * _math.log2(1 - b)

    h_min = 0.84  # representative class floor, illustrative
    # Bimodal locked panel: 50 loci locked low, 50 locked high (a normal marker panel,
    # well above MIN_MARKERS_FOR_SCORING). beta_mean lands at 0.5 -> the H(beta_mean)
    # trap reads max entropy, while mean-of-per-CpG correctly reads the locked floor.
    cpgs = [f"cg_lo_{i}" for i in range(50)] + [f"cg_hi_{i}" for i in range(50)]
    betas = _pd.Series([0.1] * 50 + [0.9] * 50, index=cpgs)

    result = _score_one(betas, cpgs, h_min)
    got = result["A"]

    expected_mean_of_H = sum(_H(b) / h_min for b in betas.values) / len(betas)  # ~0.4690/0.84
    broken_H_of_mean = _H(float(betas.mean())) / h_min                          # H(0.5)/0.84 = 1/0.84

    assert abs(got - expected_mean_of_H) < 1e-9, (
        f"A-SCORE REGRESSION: got A={got:.6f}, expected mean-of-per-CpG-entropies "
        f"{expected_mean_of_H:.6f}. The formula must be mean_i(H(beta_i)/H_min), "
        f"not H(beta_mean)/H_min. See SOP v1.4.0 LESSON-ASCORE-02 (§105)."
    )
    assert abs(got - broken_H_of_mean) > 0.4, (
        f"A-SCORE REGRESSION: got A={got:.6f}, which matches the BROKEN "
        f"H(beta_mean)/H_min={broken_H_of_mean:.6f}. See SOP v1.4.0 §105."
    )
    return got, expected_mean_of_H, broken_H_of_mean


if __name__ == "__main__":
    # Regression guard runs FIRST, always, with no artifact required.
    _g, _exp, _broken = regression_guard_bimodal()
    print(f"[regression_guard] bimodal panel A={_g:.4f} == mean-of-per-CpG {_exp:.4f} "
          f"(broken H(beta_mean) would be {_broken:.4f}) -- PASS")

    # Self-test: load the artifact and score a synthetic uniform-beta patient
    import sys
    if len(sys.argv) < 2:
        print("Usage: python iamatlas_a_scoring.py <path_to_iamatlas_celltype_markers_v0_1.json>")
        sys.exit(0)
    meta, ct_markers, cmap, hmin = load_artifact(sys.argv[1])
    print(f"Artifact: {meta['artifact_id']}")
    print(f"  source_atlas:    {meta['source_atlas']}")
    print(f"  n_celltypes:     {meta['n_celltypes']}")
    print(f"  n_top_per_ct:    {meta['n_top_per_celltype']}")

    # Score a synthetic uniform-beta=0.5 patient at all marker CpGs
    all_cpgs = set()
    for cts in ct_markers.values():
        all_cpgs.update(cts)
    synth_betas = {c: 0.5 for c in all_cpgs}
    print(f"\nSynthetic patient: beta=0.5 at all {len(synth_betas)} marker CpGs")

    ct_scores = score_per_celltype(synth_betas, ct_markers, cmap, hmin)
    print(f"\nSelf-test: scored {len(ct_scores)} cell types")
    classes_seen = sorted(set(r["class"] for r in ct_scores.values() if "class" in r))
    print(f"  Classes covered: {classes_seen}")
    # At beta=0.5 H(b)=1; A = 1/H_min so each class A = 1/H_min
    sample = list(ct_scores.items())[0]
    print(f"  Example: {sample[0]} (class={sample[1]['class']}) A={sample[1]['A']:.4f} (expected {1/hmin[sample[1]['class']]:.4f} = 1/H_min)")
