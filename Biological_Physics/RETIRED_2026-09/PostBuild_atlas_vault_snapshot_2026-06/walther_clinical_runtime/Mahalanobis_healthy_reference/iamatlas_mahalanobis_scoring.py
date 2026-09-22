"""IAMAtlas Mahalanobis hyper-volume scoring module.

Companion to iamatlas_a_scoring.py. Produces a single-number patient summary:
the Mahalanobis distance of the patient's 115-cell-type A-score vector from the
pooled-healthy-cohort centroid in the inverse-covariance-weighted hyper-volume.

This is the multi-dimensional analog of the CMB community's joint posterior
ellipsoid (banana degeneracy / hyper-volume). It gives every EDEAR report ONE
number that summarizes "how far is this patient from the healthy reference
hyper-volume, on a statistically interpretable scale."

Validation anchor (built into mahalanobis_healthy_reference_v0_1.json):
  GSE51057 >10yr breast pre-dx: Cohen's d = +1.871 (95% CI [+1.014, +2.856])
  GSE51032 >10yr breast pre-dx: Cohen's d = +2.088 (95% CI [+1.502, +2.735])
  Beats Xu-538 disease-trained panel by +0.752 on GSE51032.
  Not breast-trained — universal departure-from-healthy summary across all cards.

Author: WaltherMayerAI + Heath W. Mahaffey
Build session: EDEAR_Physics_Roadmap TODO 1.2 (2026-05-29)
"""

import json
from typing import Dict, Optional, List
import numpy as np


class MahalanobisHealthyHull:
    """Healthy-cohort hyper-volume reference.

    Load the reference artifact once at session startup, then call
    .score(per_celltype_ascore_dict) per patient.
    """

    def __init__(self, reference_path: str):
        with open(reference_path) as f:
            ref = json.load(f)
        self.artifact_id = ref["artifact_id"]
        self.feature_names = ref["feature_names_valid"]
        self.n_features = ref["n_features"]
        self.centroid = np.array(ref["centroid"], dtype=float)
        self.cov = np.array(ref["covariance_matrix"], dtype=float)
        self.inv_cov = np.linalg.inv(self.cov)
        self.shrinkage = ref.get("shrinkage")
        self.anchor = ref.get("validation_anchor", {})

    def score(self, celltype_ascores: Dict[str, float]) -> Dict:
        """Compute the Mahalanobis distance for one patient.

        celltype_ascores: {celltype_name: A-score float} dict from score_per_celltype()

        Returns: dict with distance, status, percentile_vs_hc (if reference distribution
                 of HC distances is available), per_axis_decomposition (top contributors).
        """
        # Build feature vector in the canonical order
        vec = np.array([celltype_ascores.get(ct, np.nan) for ct in self.feature_names])
        missing_mask = np.isnan(vec)
        n_missing = int(missing_mask.sum())

        if n_missing > 0:
            # Mean-impute missing features (the deconvolver should have provided most)
            vec_imputed = vec.copy()
            vec_imputed[missing_mask] = self.centroid[missing_mask]
            status = "PARTIAL_DATA" if n_missing > 5 else "OK"
        else:
            vec_imputed = vec
            status = "OK"

        diff = vec_imputed - self.centroid
        d_sq = float(diff @ self.inv_cov @ diff)
        distance = float(np.sqrt(d_sq))

        # Per-axis contribution (univariate standardized shift, easy to interpret)
        # diagonal of cov gives feature variances
        hc_sd = np.sqrt(np.diag(self.cov))
        per_axis_z = diff / hc_sd
        # Top 10 contributors by absolute z
        order = np.argsort(np.abs(per_axis_z))[::-1][:10]
        top10 = [
            {
                "rank": int(i + 1),
                "celltype": self.feature_names[idx],
                "z_shift": float(per_axis_z[idx]),
                "patient_value": float(vec_imputed[idx]),
                "hc_centroid": float(self.centroid[idx]),
            }
            for i, idx in enumerate(order)
        ]

        return {
            "mahalanobis_distance": distance,
            "n_features_used": self.n_features - n_missing,
            "n_features_imputed": n_missing,
            "status": status,
            "top10_axis_contributions": top10,
            "reference_anchor": self.anchor,
        }


if __name__ == "__main__":
    # Self-test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python iamatlas_mahalanobis_scoring.py <path_to_mahalanobis_healthy_reference_v0_1.json>")
        sys.exit(1)
    hull = MahalanobisHealthyHull(sys.argv[1])
    print(f"Loaded {hull.artifact_id}")
    print(f"  Features: {hull.n_features}")
    print(f"  Shrinkage: {hull.shrinkage}")
    print(f"  Validation anchor: {json.dumps(hull.anchor, indent=2)[:300]}")
    # Synthetic patient at centroid (Mahalanobis distance should be ~0)
    test_betas = {ct: hull.centroid[i] for i, ct in enumerate(hull.feature_names)}
    result = hull.score(test_betas)
    print(f"\nSelf-test (patient at centroid): distance = {result['mahalanobis_distance']:.6f} (expected ~0)")
