#!/usr/bin/env python3
"""
bidirectional_decomposition.py — Stage 4.5 of the Walther Clinical Engine

Implements the four-step bidirectional discipline at patient runtime, mirroring
the sealed VAL-051 directional A-score methodology exactly.

THE PROBLEM THIS SOLVES
------------------------
At Stage 4, the pooled-entropy A-score (A = H(β_mean) / H_min) is direction-
agnostic because Shannon entropy is symmetric around β=0.5. When a disease
produces a bidirectional pattern — some CpGs going UP, others going DOWN — the
pooled β_mean barely moves and the pooled-entropy A-score reads NULL. VAL-050
hit this exactly: on the 18-CpG IMM panel applied to AIBL AD vs HC, pooled
entropy A-score returned d=+0.077 (effectively null).

VAL-051 recovered the same signal at d=+0.624 by using a directional weighted
composite z-score on a 7-CpG sub-panel. The directional decomposition is what
made the AD-instance immune pattern visible.

This module runs at PATIENT runtime to surface the same bidirectional patterns
that the VAL discipline catches at validation time. Per VAL discipline: every
VAL has a PREREG specifying direction. Patient runtime has no PREREG per
patient — the engine must autonomously decompose and decide.

THE SEALED FORMULA (mirrors val051_analyze.py:112-121 exactly)
---------------------------------------------------------------
```python
def a_dir_score(patient_beta, panel_cpgs_with_stats):
    contribs = []
    for cpg, r in panel_cpgs_with_stats.items():
        b = patient_beta.get(cpg)
        if b is None or not (0 < b < 1):
            continue
        z = (b - r['mean_hc_train']) / r['sd_hc_train'] if r['sd_hc_train'] > 0 else 0
        contribs.append(r['direction'] * z)
    if len(contribs) < max(3, int(0.7 * len(panel_cpgs_with_stats))):
        return None
    return sum(contribs) / len(contribs)
```

Positive composite → patient methylation matches the AD direction.
Negative composite → patient methylation matches the HC direction (anti-AD).
Near-zero composite → no directional signal.

PARTIAL-COVERAGE FLAG
---------------------
The four-step discipline: if fewer than 70% of panel CpGs are present in the
patient's β data, the directional composite is not reported (returns None).
The orchestrator interprets None as INSUFFICIENT_COVERAGE for the directional
read; pooled-entropy A from Stage 4 still works.

BIDIRECTIONAL FLAG
------------------
FLAG_BIDIRECTIONAL is set when:
- Pooled-entropy A-score is at-or-near baseline (within 0.05 of 1.0)
  AND
- |directional composite z-score| > 0.40 (matches VAL-051 d>0.40 effect-size
  threshold)

In English: pooled is mute but the directional composite is loud — the
classical hallmark of a bidirectional pattern being cancelled by pooled
averaging. When flagged, customer-facing reporting at Stage 7 uses the
directional composite (with sign + magnitude) rather than the pooled A-score.

REFERENCE
---------
- Sealed VAL-051 panel (7 CpGs, Rule A): immune class, AD-direction-anchored.
  Files: val051_panel_ruleA.json (cpg list + per-CpG mean_hc_train, sd_hc_train,
  direction, FDR q-value).
- Sealed VAL-050: 18-CpG IMM_CPGS_EPIC parent panel where pooled A returned null.
- Future expansion: CPG-VAL-019 (cancer-positive vs AD-negative direction
  discrimination), additional per-class directional panels as the CPG-VAL series
  produces sealed evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

ARCHITECTURAL_CLASSES: list[str] = [
    "stem_pluri", "stem_adult", "stromal", "progenitor",
    "cycling", "secretory", "terminal", "immune"
]

H_MIN_BY_CLASS: dict[str, float] = {
    "terminal":    0.7728,
    "immune":      0.838889,
    "secretory":   0.8433,
    "progenitor":  0.8522,
    "cycling":     0.8561,
    "stromal":     0.8630,
    "stem_adult":  0.8737,
    "stem_pluri":  0.9822,
}

# Minimum panel-CpG coverage to compute the directional composite.
# Mirrors val051_analyze.py:120 — max(3, 70% of panel size).
MIN_PANEL_CPG_ABSOLUTE: int = 3
MIN_PANEL_CPG_FRACTION: float = 0.70

# Bidirectional flag thresholds (defaults chosen to match VAL-051 effect-size
# baseline; tunable per-card via the panel JSON if needed).
DEFAULT_POOLED_NEAR_BASELINE_TOLERANCE: float = 0.05  # |A_pooled - 1.0| < 0.05
DEFAULT_DIRECTIONAL_COMPOSITE_THRESHOLD: float = 0.40  # |z_dir| > 0.40


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CpGDirectionalSpec:
    """Per-CpG directional spec frozen from a VAL panel."""
    cpg_id: str
    direction: int            # +1 (up in disease) or -1 (down in disease)
    mean_hc_train: float      # training-set HC mean β (from VAL training split)
    sd_hc_train: float        # training-set HC SD β
    q_fdr: float | None = None
    delta_beta: float | None = None  # training-set Δβ (AD - HC) for audit


@dataclass
class DirectionalPanel:
    """Per-class directional panel — a list of CpGs with signed directions + HC reference stats."""
    cls: str
    panel_id: str
    panel_source_val: str     # e.g. "VAL-051" — which VAL produced this panel
    cpgs: list[CpGDirectionalSpec]
    h_min: float              # H_min(class) — used for pooled-entropy comparator
    pooled_panel_cpgs: list[str] | None = None  # broader panel for pooled-entropy comparator (e.g., 18-CpG IMM panel parent of 7-CpG Rule A)
    description: str = ""

    @property
    def n_cpgs(self) -> int:
        return len(self.cpgs)

    @property
    def n_positive(self) -> int:
        return sum(1 for s in self.cpgs if s.direction > 0)

    @property
    def n_negative(self) -> int:
        return sum(1 for s in self.cpgs if s.direction < 0)


@dataclass
class BidirectionalResult:
    """Stage 4.5 per-class output."""
    cls: str
    panel_id: str
    panel_source_val: str
    n_panel_cpgs: int
    n_covered: int
    coverage_fraction: float
    a_pooled_entropy: float | None       # Stage 4 pooled-entropy A-score (the null comparator)
    a_directional_composite: float | None  # the sealed VAL-051 directional composite
    flag_bidirectional: bool
    flag_insufficient_coverage: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.cls,
            "panel_id": self.panel_id,
            "panel_source_val": self.panel_source_val,
            "n_panel_cpgs": self.n_panel_cpgs,
            "n_covered": self.n_covered,
            "coverage_fraction": round(self.coverage_fraction, 4),
            "a_pooled_entropy": round(self.a_pooled_entropy, 4) if self.a_pooled_entropy is not None else None,
            "a_directional_composite": round(self.a_directional_composite, 4) if self.a_directional_composite is not None else None,
            "flag_bidirectional": self.flag_bidirectional,
            "flag_insufficient_coverage": self.flag_insufficient_coverage,
            "interpretation": self.interpretation,
        }


@dataclass
class BidirectionalReport:
    """Aggregate Stage 4.5 output across all 8 classes."""
    patient_id: str
    per_class_results: dict[str, BidirectionalResult]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "stage": "4.5",
            "per_class": {cls: r.to_dict() for cls, r in self.per_class_results.items()},
            "notes": self.notes,
        }

    @property
    def any_bidirectional_flagged(self) -> bool:
        return any(r.flag_bidirectional for r in self.per_class_results.values())

    @property
    def flagged_classes(self) -> list[str]:
        return [cls for cls, r in self.per_class_results.items() if r.flag_bidirectional]


# ============================================================================
# PANEL LOADING
# ============================================================================

def load_directional_panels(panel_json_path: Path | str) -> dict[str, DirectionalPanel]:
    """Load per-class directional panels from a JSON file.

    Expected JSON schema (directional_panels_v1_0.json):

        {
            "version": "v1.0",
            "panels": {
                "immune": {
                    "panel_id": "VAL-051 Rule A 7-CpG AD-direction-anchored",
                    "panel_source_val": "VAL-051",
                    "h_min": 0.838889,
                    "description": "...",
                    "pooled_panel_cpgs": [...18 CpGs from IMM_CPGS_EPIC...],
                    "cpgs": [
                        {"cpg_id": "cg16867657", "direction": 1, "mean_hc_train": 0.7309, "sd_hc_train": 0.0474, ...},
                        ...
                    ]
                },
                ...
            }
        }

    Classes without sealed directional panels are returned with `cpgs=[]` (no
    directional decomposition available; pooled-entropy A from Stage 4 still
    valid).
    """
    panel_json_path = Path(panel_json_path)
    with open(panel_json_path) as f:
        raw = json.load(f)

    panels: dict[str, DirectionalPanel] = {}
    raw_panels = raw.get("panels", {})

    for cls in ARCHITECTURAL_CLASSES:
        spec = raw_panels.get(cls)
        if not spec:
            # No sealed panel for this class — return placeholder
            panels[cls] = DirectionalPanel(
                cls=cls,
                panel_id=f"NO_SEALED_PANEL_for_{cls}",
                panel_source_val="N/A",
                cpgs=[],
                h_min=H_MIN_BY_CLASS[cls],
                pooled_panel_cpgs=None,
                description=(
                    f"No sealed directional panel for class '{cls}' as of "
                    "directional_panels_v1_0.json. Pooled-entropy A from Stage 4 "
                    "is the only A-score for this class until a panel is sealed."
                ),
            )
            continue

        cpgs = [
            CpGDirectionalSpec(
                cpg_id=c["cpg_id"],
                direction=int(c["direction"]),
                mean_hc_train=float(c["mean_hc_train"]),
                sd_hc_train=float(c["sd_hc_train"]),
                q_fdr=c.get("q_fdr"),
                delta_beta=c.get("delta_beta"),
            )
            for c in spec["cpgs"]
        ]
        panels[cls] = DirectionalPanel(
            cls=cls,
            panel_id=spec["panel_id"],
            panel_source_val=spec["panel_source_val"],
            cpgs=cpgs,
            h_min=spec.get("h_min", H_MIN_BY_CLASS[cls]),
            pooled_panel_cpgs=spec.get("pooled_panel_cpgs"),
            description=spec.get("description", ""),
        )

    return panels


# ============================================================================
# SCORING FUNCTIONS — MIRROR THE SEALED VAL-051 FORMULA
# ============================================================================

def score_directional_composite(
    patient_beta: pd.Series,
    panel: DirectionalPanel,
) -> tuple[float | None, int, int]:
    """Mirror VAL-051 `a_dir_score` exactly.

    For each CpG in the panel:
      z = (β_patient - mean_hc_train) / sd_hc_train
      contrib = direction * z   (multiplies in the frozen disease direction)

    Returns
    -------
    (composite, n_covered, n_total) : (float | None, int, int)
        composite = sum(contribs) / len(contribs), or None if coverage too low.

    See val051_analyze.py:112-121 for the sealed reference implementation.
    """
    if not panel.cpgs:
        return None, 0, 0

    contribs = []
    for spec in panel.cpgs:
        b = patient_beta.get(spec.cpg_id)
        if b is None or not (0.0 < b < 1.0):
            continue
        if spec.sd_hc_train > 0:
            z = (b - spec.mean_hc_train) / spec.sd_hc_train
        else:
            z = 0.0
        contribs.append(spec.direction * z)

    n_covered = len(contribs)
    n_total = len(panel.cpgs)
    min_required = max(MIN_PANEL_CPG_ABSOLUTE, int(MIN_PANEL_CPG_FRACTION * n_total))

    if n_covered < min_required:
        return None, n_covered, n_total

    return sum(contribs) / n_covered, n_covered, n_total


def score_pooled_entropy(
    patient_beta: pd.Series,
    cpg_list: list[str],
    h_min: float,
) -> float | None:
    """Mirror VAL-050 / val051_analyze.py `a_entropy_pooled` exactly.

    Computes the pooled-entropy A-score on the parent panel (typically the 18-CpG
    IMM_CPGS_EPIC for immune class).

    Returns A = H(β_mean) / H_min, or None if coverage too low.
    """
    import math

    vals = [
        patient_beta[c]
        for c in cpg_list
        if (c in patient_beta.index) and pd.notna(patient_beta.get(c))
        and 0.0 < patient_beta.get(c, 0.0) < 1.0
    ]
    if len(vals) < 12:  # matches val051_analyze.py:126
        return None

    mean_b = sum(vals) / len(vals)
    if mean_b <= 0.0 or mean_b >= 1.0:
        return None

    H = -mean_b * math.log2(mean_b) - (1.0 - mean_b) * math.log2(1.0 - mean_b)
    return H / h_min


def bidirectional_flag(
    a_pooled: float | None,
    a_directional: float | None,
    *,
    pooled_near_baseline_tolerance: float = DEFAULT_POOLED_NEAR_BASELINE_TOLERANCE,
    directional_threshold: float = DEFAULT_DIRECTIONAL_COMPOSITE_THRESHOLD,
) -> bool:
    """Bidirectional pattern detection.

    Sets FLAG_BIDIRECTIONAL when:
    - Pooled A-score is near baseline (within `pooled_near_baseline_tolerance` of 1.0)
      AND
    - |directional composite| > `directional_threshold`

    In English: pooled is mute but directional is loud — the bidirectional
    cancellation signature.

    Returns False when either input is None (insufficient coverage).
    """
    if a_pooled is None or a_directional is None:
        return False
    pooled_near_baseline = abs(a_pooled - 1.0) < pooled_near_baseline_tolerance
    directional_loud = abs(a_directional) > directional_threshold
    return pooled_near_baseline and directional_loud


def interpret_directional_result(
    a_pooled: float | None,
    a_directional: float | None,
    flag: bool,
    insufficient_coverage: bool,
    panel: DirectionalPanel,
) -> str:
    """Generate a human-readable interpretation string for the report builder."""
    if insufficient_coverage:
        return (
            f"INSUFFICIENT_COVERAGE — fewer than 70% of {panel.n_cpgs} panel CpGs "
            f"found in patient β; directional decomposition not reported."
        )
    if not panel.cpgs:
        return "NO_PANEL — no sealed directional panel for this class yet."
    if a_directional is None:
        return "UNCOMPUTED — directional composite returned None despite coverage check."

    pooled_str = f"{a_pooled:.3f}" if a_pooled is not None else "N/A (no pooled comparator coverage)"

    if flag:
        direction = "disease-direction" if a_directional > 0 else "anti-disease-direction"
        return (
            f"BIDIRECTIONAL_PATTERN_DETECTED — pooled A={pooled_str} (near baseline) "
            f"but directional composite={a_directional:+.3f} ({direction}). "
            f"Customer-facing tier uses the directional composite."
        )
    if abs(a_directional) > DEFAULT_DIRECTIONAL_COMPOSITE_THRESHOLD:
        direction = "disease-direction" if a_directional > 0 else "anti-disease-direction"
        return (
            f"DIRECTIONAL_SIGNAL_PRESENT — pooled A={pooled_str} also moves; "
            f"directional composite={a_directional:+.3f} ({direction}). "
            f"Pooled and directional agree."
        )
    return (
        f"WITHIN_BASELINE — pooled A={pooled_str}, directional composite="
        f"{a_directional:+.3f}. No bidirectional pattern detected."
    )


# ============================================================================
# TOP-LEVEL ENTRY: COMPUTE STAGE 4.5 FOR ALL CLASSES
# ============================================================================

def compute_per_class_bidirectional_decomposition(
    patient_beta: pd.Series,
    panels: dict[str, DirectionalPanel],
    *,
    patient_id: str = "patient",
    pooled_near_baseline_tolerance: float = DEFAULT_POOLED_NEAR_BASELINE_TOLERANCE,
    directional_threshold: float = DEFAULT_DIRECTIONAL_COMPOSITE_THRESHOLD,
) -> BidirectionalReport:
    """Stage 4.5 top-level entry point.

    For each of the 8 architectural classes that has a sealed directional panel,
    compute the directional composite + pooled-entropy comparator + bidirectional
    flag. Classes without a sealed panel get a placeholder result (NO_PANEL).
    """
    per_class_results: dict[str, BidirectionalResult] = {}
    notes: list[str] = []

    for cls in ARCHITECTURAL_CLASSES:
        if cls not in panels:
            notes.append(f"WARNING: no panel entry for class '{cls}' — skipped.")
            continue

        panel = panels[cls]
        a_dir, n_covered, n_total = score_directional_composite(patient_beta, panel)

        # Pooled-entropy comparator (uses the broader parent panel if available)
        pooled_cpgs = panel.pooled_panel_cpgs or [s.cpg_id for s in panel.cpgs]
        a_pooled = score_pooled_entropy(patient_beta, pooled_cpgs, panel.h_min) if pooled_cpgs else None

        insufficient_coverage = (a_dir is None) and bool(panel.cpgs)

        flag = bidirectional_flag(
            a_pooled, a_dir,
            pooled_near_baseline_tolerance=pooled_near_baseline_tolerance,
            directional_threshold=directional_threshold,
        )

        result = BidirectionalResult(
            cls=cls,
            panel_id=panel.panel_id,
            panel_source_val=panel.panel_source_val,
            n_panel_cpgs=n_total,
            n_covered=n_covered,
            coverage_fraction=(n_covered / n_total) if n_total > 0 else 0.0,
            a_pooled_entropy=a_pooled,
            a_directional_composite=a_dir,
            flag_bidirectional=flag,
            flag_insufficient_coverage=insufficient_coverage,
            interpretation=interpret_directional_result(
                a_pooled, a_dir, flag, insufficient_coverage, panel
            ),
        )
        per_class_results[cls] = result

        logger.info(
            "Class %s: panel=%s, covered=%d/%d, A_pooled=%s, A_dir=%s, flag=%s",
            cls, panel.panel_id, n_covered, n_total,
            f"{a_pooled:.3f}" if a_pooled is not None else "None",
            f"{a_dir:+.3f}" if a_dir is not None else "None",
            flag,
        )

    return BidirectionalReport(
        patient_id=patient_id,
        per_class_results=per_class_results,
        notes=notes,
    )


def save_bidirectional_report(
    report: BidirectionalReport,
    out_dir: Path | str,
) -> Path:
    """Persist the bidirectional report JSON to the audit-trail folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report.patient_id}_stage_4_5_bidirectional_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    return out_path


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def _cli_main():
    """Smoke test — loads panels + runs against a tiny synthetic patient.

    Production invocation comes from walther_clinical.py at Stage 4.5:

        from bidirectional_decomposition import (
            load_directional_panels,
            compute_per_class_bidirectional_decomposition,
            save_bidirectional_report,
        )

        panels = load_directional_panels("Bidirectional_Decomposition/directional_panels_v1_0.json")

        report = compute_per_class_bidirectional_decomposition(
            patient_beta=patient_beta_cleaned,
            panels=panels,
            patient_id=patient_metadata["patient_id"],
        )

        if report.any_bidirectional_flagged:
            # Customer-facing tier uses directional composite for flagged classes
            ...

        save_bidirectional_report(report, out_dir=f"reports/{patient_id}/stage_4_5/")
    """
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4.5 bidirectional decomposition")
    parser.add_argument(
        "--panel-json",
        default="directional_panels_v1_0.json",
        help="Path to directional panels JSON",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Load panels and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.smoke_test:
        print(f"Loading directional panels from {args.panel_json}")
        panels = load_directional_panels(args.panel_json)
        print(f"Loaded panels for {len(panels)} classes:")
        for cls, panel in panels.items():
            if panel.cpgs:
                print(f"  {cls:12s}  panel={panel.panel_id}")
                print(f"               source={panel.panel_source_val}, n_cpgs={panel.n_cpgs} "
                      f"({panel.n_positive} positive, {panel.n_negative} negative)")
            else:
                print(f"  {cls:12s}  NO_SEALED_PANEL")

        # Tiny synthetic test
        print("\nRunning tiny synthetic test (uniform β = 0.5)...")
        synthetic_patient = pd.Series(
            {spec.cpg_id: 0.5 for cls in panels for spec in panels[cls].cpgs}
        )
        report = compute_per_class_bidirectional_decomposition(
            synthetic_patient, panels, patient_id="synthetic_test"
        )
        print(f"Bidirectional flagged classes: {report.flagged_classes}")
        for cls, r in report.per_class_results.items():
            if r.a_directional_composite is not None or r.flag_insufficient_coverage:
                print(f"  {cls}: {r.interpretation}")
        print("\nSmoke test PASS.")
        return

    parser.print_help()


if __name__ == "__main__":
    _cli_main()
