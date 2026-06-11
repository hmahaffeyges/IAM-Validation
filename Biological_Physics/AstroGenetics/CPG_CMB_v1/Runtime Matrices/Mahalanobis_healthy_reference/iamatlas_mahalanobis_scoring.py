"""IAMAtlas Mahalanobis scoring -- DERIVED reference only.

One number per patient: the derived departure of the patient's per-cell-type
A-score vector from the healthy floor, summed over the cell types ASSESSABLE in
the patient's substrate.

DERIVED-IAMAtlas-ONLY. The reference is NOT a pooled cohort. It is:
  mu    = 1.0         the architectural floor (healthy = at floor by construction)
  sigma = 0.02        the derived tier-band spread (NORMAL [0.95,1.04), ELEVATED
                      onset 1.04 = +2 sigma, the 95% healthy reference interval)
  Sigma = sigma^2 I   diagonal; off-diagonal cross-cell-type correlations are
                      population-empirical and deliberately excluded.

  distance = sqrt( sum_c ((A_c - 1)/sigma)^2 )  over substrate-assessable cells.

This module REFUSES a pooled-cohort reference. Cohort comparables are not used
anywhere in CPG/IAM.

Author: WaltherMayerAI + Heath W. Mahaffey
Derived-reference rebuild: 2026-06-11 (supersedes the pooled v0_5 hull).
"""
import json
from typing import Dict, Optional, Iterable
import numpy as np


class MahalanobisHealthyHull:
    """Derived healthy-floor reference. Load once, call .score() per patient."""

    def __init__(self, reference_path: str):
        with open(reference_path) as f:
            ref = json.load(f)
        is_derived = (ref.get("method") == "derived_floor_tierband"
                      or ref.get("covariance_method") == "derived_diagonal")
        if not is_derived:
            raise ValueError(
                "Refusing a non-derived (pooled-cohort) Mahalanobis reference: "
                f"{ref.get('artifact_id', reference_path)}. CPG/IAM uses the DERIVED "
                "floor+tier-band reference only "
                "(mahalanobis_healthy_reference_v1_0_derived.json).")
        self.artifact_id = ref["artifact_id"]
        self.feature_names = ref["feature_names"]
        self.n_features = ref.get("n_features", len(self.feature_names))
        self.mu = float(ref["mu_floor"])
        self.sigma = float(ref["sigma"])
        self.method = "derived_floor_tierband"

    def score(self, celltype_ascores: Dict[str, float],
              assessable: Optional[Iterable[str]] = None) -> Dict:
        """Derived departure distance for one patient.

        celltype_ascores : {celltype: A-score}. A non-assessable cell (gated out
                           upstream) carries A=None/NaN and is EXCLUDED here.
        assessable       : optional explicit set of assessable cell types; if None,
                           every cell with a finite A-score contributes.
        """
        aset = set(assessable) if assessable is not None else None
        contribs = []
        for ct in self.feature_names:
            if aset is not None and ct not in aset:
                continue
            a = celltype_ascores.get(ct)
            if a is None:
                continue
            try:
                a = float(a)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(a):
                continue
            z = (a - self.mu) / self.sigma
            contribs.append((ct, a, z))

        n = len(contribs)
        d_sq = float(sum(z * z for _, _, z in contribs))
        distance = float(np.sqrt(d_sq))
        order = sorted(contribs, key=lambda x: abs(x[2]), reverse=True)[:10]
        top10 = [{"rank": i + 1, "celltype": ct, "z_shift": z,
                  "patient_value": a, "floor": self.mu}
                 for i, (ct, a, z) in enumerate(order)]
        return {
            "mahalanobis_distance": distance,
            "n_features_assessable": n,
            "status": "OK" if n > 0 else "NO_ASSESSABLE_FEATURES",
            "top10_axis_contributions": top10,
            "method": self.method,
            "sigma": self.sigma,
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python iamatlas_mahalanobis_scoring.py <derived_reference.json>")
        sys.exit(1)
    hull = MahalanobisHealthyHull(sys.argv[1])
    print(f"Loaded {hull.artifact_id} (mu={hull.mu}, sigma={hull.sigma}, n={hull.n_features})")
    at_floor = {ct: 1.0 for ct in hull.feature_names}
    print("at-floor distance:", round(hull.score(at_floor)["mahalanobis_distance"], 6), "(expect 0)")
    one = dict(at_floor); one[hull.feature_names[0]] = 1.04   # +2 sigma
    print("one cell at ELEVATED onset 1.04:", round(hull.score(one)["mahalanobis_distance"], 4), "(expect 2.0)")
    g = hull.score(at_floor, assessable=hull.feature_names[:3])
    print("gated to 3 assessable cells -> n =", g["n_features_assessable"], "distance =", g["mahalanobis_distance"])
