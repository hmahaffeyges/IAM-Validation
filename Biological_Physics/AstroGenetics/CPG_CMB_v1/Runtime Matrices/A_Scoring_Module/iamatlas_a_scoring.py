"""IAMAtlas A-score scoring module.

Two scoring surfaces, both computing A = H(mean beta) / H_min(class) at marker CpGs
(entropy of the mean beta across the panel, NOT the mean of per-CpG entropies).

  score_per_class(customer_betas, ...)     -> 8 class A-scores
  score_per_celltype(customer_betas, ...)  -> 115 cell-type A-scores

Both return per-result diagnostics: A-score, coverage, confidence, status.

This module is SEPARATE from the Walther IAM Deconvolver. The deconvolver
computes cell-type FRACTIONS via NNLS; this module computes cell-type A-SCORES
via entropy-at-markers. Different math, different CpG sets, different failure
modes - kept separate so each component is independently testable.

Companion artifacts (must be loaded once at session start):
  - iamatlas_celltype_markers_v0_2.json  (loaded here ONLY for the H_min table and
                                          the cell->class map. Its marker lists are
                                          DISCRIMINATIVE one-vs-rest deconvolution
                                          markers and MUST NOT be scored by the
                                          A-score — see below.)
  - H_min_by_class table                  (the 8 architectural floors)

  A-SCORE LOCI (the CpGs actually scored): per-class MOST-METHYLATED loci from the
  IAMAtlas (iamatlas_a_score_loci_v1_0.json), supplied by the caller. NEVER score
  the discriminative markers: they are mixed-direction, so their panel mean -> 0.5
  and H(0.5)=1.0 pins every A at the ceiling 1/H_min (the all-BREACH bug, root-caused
  2026-06-11; Recipe Part 4 / line 884; SOP §103). Two CpG sets, two jobs:
  discriminative -> Stage 1 deconvolution; most-methylated -> Stage 4 A-score.

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

    # A-score = H(beta_mean) / H_min(class): take the MEAN beta across the panel
    # FIRST, then the entropy of that single mean, then divide by the class floor.
    # (Prior code computed mean(H(beta_i)) -- the average of per-CpG entropies --
    #  which is a different quantity and collapses bimodal marker panels toward
    #  the floor, so healthy samples read ~0.4-0.5 instead of ~1.0.)
    beta_mean = float(np.mean(vals))
    a_score = float(_shannon_bits(np.array([beta_mean]))[0] / h_min)
    per_cpg_a = _shannon_bits(vals) / h_min  # retained only as a dispersion diagnostic
    dispersion = float(np.std(per_cpg_a, ddof=1)) if n_usable >= 2 else 0.0
    coverage = n_usable / n_expected
    confidence = _scoring_confidence(coverage, dispersion)
    return {
        "A": a_score,
        "v_s": beta_mean,
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


# ---------------------------------------------------------------------------
# Substrate-aware envelope (Recipe Part 3.1-3.4)
# ---------------------------------------------------------------------------
# Production scores methylation only. This envelope makes the A-score
# substrate-aware so the four cfDNA fragmentomics substrates (nucleosome
# occupancy, fuzziness, windowed protection score, fragment size) plug in
# later WITHOUT re-architecting the chain. Fragmentomics are OFF by default:
# they are simply absent from the substrate dict for an array-only patient,
# in which case A_combined == A_active == A_methyl by construction.
#
# The methyl scoring path above (_score_one / score_per_class /
# score_per_celltype) is unchanged; this section is purely additive.

# Published single-substrate AUC weights (Recipe Part 2.2 / 3.2).
SUBSTRATE_AUC_WEIGHTS = {
    "methyl": 0.866,
    "nucl":   0.852,
    "fuzz":   0.779,
    "wps":    0.761,
    "frag":   0.940,
}

# Recipe 3.3: a substrate is "saturated" when its A sits within this distance
# of its own ceiling (1/H_min); saturated substrates are excluded from A_active.
SATURATION_EPS = 0.005

# Recipe 3.4: ceiling invariant tolerance, A <= 1/H_min + CEILING_EPS.
CEILING_EPS = 1e-9


def substrate_ceiling(h_min: float) -> float:
    """Recipe 3.4: A_ceiling(c, s) = 1 / H_min(c, s)."""
    return 1.0 / h_min


def substrate_a_score(value: float, h_min: float) -> float:
    """Recipe 3.1: per-substrate A_{c,s} = H(v_s) / H_min(c, s).

    value: the substrate's representative value in (0,1) for this class
           (for methylation this is the panel mean beta; for the cfDNA
           substrates it is the analogous [0,1] measure).
    Asserts the ceiling invariant (3.4); H(v) <= 1 so A <= 1/H_min always
    holds for correct inputs -- a violation flags an upstream bug.
    """
    a = float(_shannon_bits(np.array([value]))[0] / h_min)
    ceiling = substrate_ceiling(h_min)
    assert a <= ceiling + CEILING_EPS, (
        f"ceiling invariant violated: A={a:.6f} > 1/H_min={ceiling:.6f}")
    return a


def a_combined(substrate_a: Dict[str, float]) -> Optional[float]:
    """Recipe 3.2: AUC-weighted mean across ALL available substrates.

    substrate_a: {substrate_name: A_value}. Only keys present contribute,
    so a methyl-only dict returns A_methyl unchanged.
    Returns None if no usable substrate is present.
    """
    num = den = 0.0
    for s, a in substrate_a.items():
        if a is None or not np.isfinite(a):
            continue
        w = SUBSTRATE_AUC_WEIGHTS.get(s)
        if w is None:
            continue
        num += w * a
        den += w
    return (num / den) if den > 0 else None


def a_active(substrate_a: Dict[str, float],
             h_min_by_substrate: Dict[str, float]) -> Optional[float]:
    """Recipe 3.3: AUC-weighted mean over NON-saturated substrates only.

    A substrate is dropped when |A_s - ceiling_s| <= SATURATION_EPS.
    For serial monitoring of advanced disease / chemotherapy response, where
    A_combined can sit deceptively pinned while one non-saturated substrate
    still carries the honest progression signal. With no saturation (e.g. an
    array-only or early-timepoint patient) A_active == A_combined.
    """
    active = {}
    for s, a in substrate_a.items():
        if a is None or not np.isfinite(a):
            continue
        hm = h_min_by_substrate.get(s)
        if hm is None:
            continue
        if abs(a - substrate_ceiling(hm)) > SATURATION_EPS:
            active[s] = a
    return a_combined(active) if active else None


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


if __name__ == "__main__":
    # Self-test: load the artifact and score a synthetic uniform-beta patient
    import sys

    # --- Substrate-envelope self-test (Recipe 3.1-3.4; needs no artifact) ---
    print("Substrate-envelope self-test (Recipe 3.1-3.4):")
    hm_term = 0.7728          # terminal methyl H_min (ceiling 1/H_min = 1.2939)
    hm_frag = 0.687936        # fragment-size H_min (G-003b non-methyl posterior)
    # ceiling invariant (3.4): beta=0.5 -> H=1 -> A = 1/H_min exactly
    a_ceil = substrate_a_score(0.5, hm_term)
    assert abs(a_ceil - 1.0 / hm_term) < 1e-9, a_ceil
    print(f"  ceiling:      A(beta=0.5, terminal) = {a_ceil:.4f} = 1/H_min ({1/hm_term:.4f})  [OK]")
    # methyl-only collapses: A_combined == A_active == A_methyl
    a_m = substrate_a_score(0.78, hm_term)
    assert abs(a_combined({"methyl": a_m}) - a_m) < 1e-12
    assert abs(a_active({"methyl": a_m}, {"methyl": hm_term}) - a_m) < 1e-12
    print(f"  methyl-only:  A_combined == A_active == A_methyl = {a_m:.4f}  [OK]")
    # A_combined (3.2): AUC-weighted mean across substrates
    sa = {"methyl": 1.05, "frag": 1.20}
    expect = (0.866 * 1.05 + 0.940 * 1.20) / (0.866 + 0.940)
    assert abs(a_combined(sa) - expect) < 1e-12
    print(f"  A_combined:   (methyl=1.05, frag=1.20) = {a_combined(sa):.4f} = AUC-weighted mean  [OK]")
    # A_active (3.3): drop a saturated substrate, keep the one still carrying signal
    sat = {"methyl": 1.0 / hm_term, "frag": 1.20}        # methyl pinned at ceiling, frag not
    hmsub = {"methyl": hm_term, "frag": hm_frag}
    aa, ac = a_active(sat, hmsub), a_combined(sat)
    assert abs(aa - 1.20) < 1e-12                          # only frag survives
    assert ac > aa + 1e-3                                  # combined pinned higher by saturated methyl
    print(f"  saturation:   A_combined={ac:.4f} (pinned) vs A_active={aa:.4f} (honest progression)  [OK]")
    print("  envelope self-test PASSED\n")

    if len(sys.argv) < 2:
        print("Usage: python iamatlas_a_scoring.py <path_to_iamatlas_celltype_markers_v0_2.json>")
        sys.exit(1)
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
