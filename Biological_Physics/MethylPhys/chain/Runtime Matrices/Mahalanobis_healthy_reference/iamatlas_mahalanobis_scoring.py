"""IAMAtlas Mahalanobis scoring -- DERIVED, AGE-MATCHED class gauge (Option A, v2).

One number per patient: the derived departure of the patient's CLASS gauge A-scores
(the 8 identity-loci readings) from the AGE-MATCHED healthy band, summed over the
classes ASSESSABLE in the patient's substrate.

DERIVED-IAMAtlas-ONLY + AGE-MATCHED. The reference is NOT a pooled cohort. Per class,
per patient age, the healthy floor is age_reference_matrix (single source, read via
cpg_gauge_engine.age_band):
  mu_c(age)    = A_mean(class, age)              age-matched healthy expectation
  sigma_c(age) = (A_mean - A_p10)/1.2816          age-matched half-band (p10..p90 -> sd)
  distance     = sqrt( sum_c ((A_c - mu_c)/sigma_c)^2 )  over assessable classes.

Supersedes the fixed mu=1.0 / sigma=0.02 derived floor (v1_0_derived): mu=1.0 was
confirmed on CLASS-level readings (~1.0) but the feature space was the 115 per-cell
SEPARATION A-scores (~0.56) -> z ~ -22 for healthy. Option A scores the CLASS GAUGE
(identity loci, age-matched ~0.9-1.0) against its OWN age band, so a healthy patient
sums to ~0. Cohort comparables are still used NOWHERE.

Author: WaltherMayerAI + Heath W. Mahaffey
Age-matched class-gauge rebuild: 2026-06-30 (Option A; supersedes v1_0_derived).
"""
import json
import os
from math import sqrt
from typing import Dict, Optional, Iterable
import numpy as np

_Z90 = 1.2816  # p10/p90 = mean -/+ 1.2816*sd


def _load_gauge():
    """Import cpg_gauge_engine -- the single source of the age_reference_matrix band."""
    import importlib.util
    p = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "cpg_gauge_engine.py"))
    spec = importlib.util.spec_from_file_location("cpg_gauge_engine", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class MahalanobisHealthyHull:
    """Derived, age-matched class-gauge reference. Load once, call .score() per patient."""

    def __init__(self, reference_path: str, gauge=None):
        with open(reference_path) as f:
            ref = json.load(f)
        if ref.get("method") != "age_matched_class_gauge":
            raise ValueError(
                "Refusing a non age-matched-class-gauge Mahalanobis reference: "
                f"{ref.get('artifact_id', reference_path)}. Option A uses the age-matched "
                "class-gauge reference only "
                "(mahalanobis_healthy_reference_v2_0_age_matched_derived.json).")
        self.artifact_id = ref["artifact_id"]
        self.feature_names = ref["feature_names"]     # the 8 architecture classes
        self.n_features = len(self.feature_names)
        self.method = ref["method"]
        self._g = gauge or _load_gauge()

    def _thresholds(self, n: int):
        if n <= 0:
            return 0.0, 0.0
        try:
            from scipy.stats import chi2
            return float(sqrt(chi2.ppf(0.95, n))), float(sqrt(chi2.ppf(0.99, n)))
        except Exception:  # chi2 mean+k*sd fallback (mean=n, var=2n)
            return float(sqrt(n + 2.0 * sqrt(2.0 * n))), float(sqrt(n + 3.0 * sqrt(2.0 * n)))

    def score(self, class_ascores: Dict[str, float], age: Optional[int] = None,
              assessable: Optional[Iterable[str]] = None) -> Dict:
        """Age-matched derived departure over the class gauge.

        class_ascores : {class: A-score} (identity-loci gauge). None/NaN class EXCLUDED.
        age           : patient age (sets the age-matched band). None -> gauge default.
        assessable    : optional explicit set of assessable classes.
        """
        aset = set(assessable) if assessable is not None else None
        contribs = []
        for cls in self.feature_names:
            if aset is not None and cls not in aset:
                continue
            a = class_ascores.get(cls)
            if isinstance(a, dict):          # accept a stage-4 rec {A, ...}
                a = a.get("A")
            if a is None:
                continue
            try:
                a = float(a)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(a):
                continue
            p10, mean, p90 = self._g.age_band(cls, age)
            sigma = max((mean - p10) / _Z90, 1e-6)
            contribs.append((cls, a, mean, (a - mean) / sigma))

        n = len(contribs)
        distance = float(sqrt(sum(z * z for _, _, _, z in contribs)))
        thr95, thr99 = self._thresholds(n)
        order = sorted(contribs, key=lambda x: abs(x[3]), reverse=True)[:10]
        top = [{"rank": i + 1, "class": c, "z_shift": z, "patient_A": a,
                "age_matched_mean": mu} for i, (c, a, mu, z) in enumerate(order)]
        return {
            "mahalanobis_distance": distance,
            "n_features_assessable": n,
            "alarm_threshold_p95": thr95,
            "alarm_threshold_p99": thr99,
            "mahalanobis_beyond_band": bool(n > 0 and distance > thr95),
            "status": "OK" if n > 0 else "NO_ASSESSABLE_FEATURES",
            "top_axis_contributions": top,
            "method": self.method,
        }


if __name__ == "__main__":
    import sys
    ref = sys.argv[1] if len(sys.argv) > 1 else \
        "mahalanobis_healthy_reference_v2_0_age_matched_derived.json"
    hull = MahalanobisHealthyHull(ref)
    classes = hull.feature_names
    # healthy AT the age-matched mean -> distance ~0
    at_band = {c: hull._g.age_band(c, 45)[1] for c in classes}
    print("at age-band mean (age 45):",
          round(hull.score(at_band, age=45)["mahalanobis_distance"], 6), "(expect ~0)")
    # one class pushed to +2 sigma (its p90) -> that axis contributes ~2
    one = dict(at_band); one["immune"] = hull._g.age_band("immune", 45)[2]
    r = hull.score(one, age=45)
    print("one class at p90 (age 45):", round(r["mahalanobis_distance"], 4),
          "beyond_band=", r["mahalanobis_beyond_band"])
    # assessable subset (whole blood: immune only)
    g = hull.score(at_band, age=45, assessable=["immune"])
    print("immune-only assessable:", round(g["mahalanobis_distance"], 6),
          "n=", g["n_features_assessable"])
