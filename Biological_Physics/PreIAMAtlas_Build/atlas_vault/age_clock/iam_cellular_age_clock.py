#!/usr/bin/env python3
"""
iam_cellular_age_clock.py — Phase B3-bonus / TODO 2.4 (methods-paper prototype)

THE NOVEL CLAIM
---------------
Existing methylation-age clocks (Horvath 2013, Hannum 2013, PhenoAge 2018, GrimAge 2019,
DunedinPACE 2022) train elastic-net regression directly on β values at hundreds to
thousands of CpGs. The training set picks which CpGs matter, and the resulting clock is
an empirical aggregate with no architectural interpretation.

The IAM cellular age clock uses the 8 architecture-class A-scores as features instead.
A-score = H(β) / H_min(class). The clock predicts chronological age from how each
patient's cellular architecture departs from the IAM physics floor.

WHY THIS COULD BE A METHODS PAPER OF ITS OWN
--------------------------------------------
- 8 features instead of hundreds → trivially interpretable
- Each coefficient has biological meaning ("this much per-year drift in immune-class
  departure-from-floor")
- Calibration-free at the substrate level (H_min comes from physics, not training)
- Should generalize across populations and platforms because the architectural floor
  is universal
- If MAE < 5 years on independent test (matching Horvath chronological-age accuracy),
  it's a standalone publication

EXIT GATE (per v1 Roadmap spec)
-------------------------------
- MAE < 5 years on independent test cohort
- Per-class A-score coefficients biologically interpretable
- Clock generalizes within ±20% MAE across cohorts

USAGE
-----
    from iam_cellular_age_clock import IAMCellularAgeClock
    clock = IAMCellularAgeClock()
    clock.fit(a_scores_train, ages_train, hc_mask_train)
    predicted = clock.predict(a_scores_test)
    departure = ages_test - predicted   # IAM cellular age departure
    clock.diagnostics()                  # MAE, R², per-class coefficients
"""

import json
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

CLASSES = ['stem_pluri','stem_adult','progenitor','cycling','secretory','immune','terminal','stromal']
H_MIN_BY_CLASS = {
    'terminal': 0.7728, 'immune': 0.838889, 'secretory': 0.843264,
    'cycling': 0.856055, 'progenitor': 0.852216, 'stromal': 0.86295,
    'stem_adult': 0.873718, 'stem_pluri': 0.982166,
}


@dataclass
class ClockDiagnostics:
    n_train: int
    n_test: int
    train_age_range: tuple
    test_age_range: tuple
    mae_train: float
    mae_test: float
    rmse_train: float
    rmse_test: float
    r2_train: float
    r2_test: float
    spearman_test: float
    bias_test: float            # mean(predicted - actual) on test set
    per_class_coefficients: dict
    intercept: float
    mae_gate_pass: bool         # MAE_test < 5 years
    cross_cohort_drift_pct: float  # how much MAE drifts when held-out is a different cohort

    def to_dict(self): return asdict(self)


class IAMCellularAgeClock:
    """
    Predicts chronological age from 8-class A-scores using elastic-net regression.

    The 8 features are the per-class A-scores in the canonical order:
        [stem_pluri, stem_adult, progenitor, cycling, secretory, immune, terminal, stromal]

    Fit on HC samples only. Test on held-out HC samples (and optionally on cases to
    measure cellular age acceleration in disease — that's the downstream clinical use).
    """

    def __init__(self, alpha: float = 0.5, l1_ratio: float = 0.5,
                 use_elastic_net: bool = True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.use_elastic_net = use_elastic_net
        self.coefficients_ = None     # shape (8,)
        self.intercept_ = None
        self.diagnostics_: Optional[ClockDiagnostics] = None
        self.is_fitted_ = False

    def fit(self, a_scores_train: np.ndarray, ages_train: np.ndarray,
            a_scores_test: Optional[np.ndarray] = None,
            ages_test: Optional[np.ndarray] = None,
            cohorts_test: Optional[np.ndarray] = None,
            cohorts_train: Optional[np.ndarray] = None):
        """
        Fit clock. Train on (a_scores_train, ages_train); evaluate on held-out test set.

        Args:
            a_scores_train : (N × 8) A-scores in CLASSES order
            ages_train : (N,) chronological ages
            a_scores_test, ages_test : (M × 8), (M,) held-out test set
            cohorts_train, cohorts_test : cohort labels for cross-cohort drift analysis
        """
        valid_train = ~np.isnan(ages_train) & ~np.isnan(a_scores_train).any(axis=1)
        Xtr = a_scores_train[valid_train]
        ytr = ages_train[valid_train]
        N = len(ytr)

        if self.use_elastic_net:
            try:
                from sklearn.linear_model import ElasticNet
                model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=10000)
                model.fit(Xtr, ytr)
                self.coefficients_ = model.coef_
                self.intercept_ = float(model.intercept_)
            except ImportError:
                # Fall back to ordinary least squares with numpy
                X_with_intercept = np.column_stack([np.ones(N), Xtr])
                beta, *_ = np.linalg.lstsq(X_with_intercept, ytr, rcond=None)
                self.intercept_ = float(beta[0])
                self.coefficients_ = beta[1:]
        else:
            X_with_intercept = np.column_stack([np.ones(N), Xtr])
            beta, *_ = np.linalg.lstsq(X_with_intercept, ytr, rcond=None)
            self.intercept_ = float(beta[0])
            self.coefficients_ = beta[1:]

        # Train predictions
        ytr_pred = self._predict_raw(Xtr)
        train_resid = ytr_pred - ytr
        mae_tr = float(np.mean(np.abs(train_resid)))
        rmse_tr = float(np.sqrt(np.mean(train_resid ** 2)))
        ss_res_tr = np.sum(train_resid ** 2)
        ss_tot_tr = np.sum((ytr - ytr.mean()) ** 2)
        r2_tr = 1.0 - ss_res_tr / ss_tot_tr if ss_tot_tr > 0 else 0.0

        # Test predictions (if held-out provided)
        if a_scores_test is not None and ages_test is not None:
            valid_te = ~np.isnan(ages_test) & ~np.isnan(a_scores_test).any(axis=1)
            Xte = a_scores_test[valid_te]
            yte = ages_test[valid_te]
            yte_pred = self._predict_raw(Xte)
            test_resid = yte_pred - yte
            mae_te = float(np.mean(np.abs(test_resid)))
            rmse_te = float(np.sqrt(np.mean(test_resid ** 2)))
            ss_res_te = np.sum(test_resid ** 2)
            ss_tot_te = np.sum((yte - yte.mean()) ** 2)
            r2_te = 1.0 - ss_res_te / ss_tot_te if ss_tot_te > 0 else 0.0
            from scipy.stats import spearmanr
            spearman_te = float(spearmanr(yte, yte_pred)[0]) if len(yte) > 5 else np.nan
            bias_te = float(test_resid.mean())
            test_age_range = (float(yte.min()), float(yte.max()))
            n_test = int(valid_te.sum())

            # Cross-cohort drift if labels available
            cross_drift = np.nan
            if cohorts_test is not None:
                cohorts_te_valid = cohorts_test[valid_te]
                per_cohort_mae = {}
                for ch in np.unique(cohorts_te_valid):
                    cm = cohorts_te_valid == ch
                    if cm.sum() < 5: continue
                    per_cohort_mae[ch] = float(np.mean(np.abs(yte_pred[cm] - yte[cm])))
                if len(per_cohort_mae) >= 2:
                    maes = list(per_cohort_mae.values())
                    cross_drift = 100.0 * (max(maes) - min(maes)) / np.mean(maes)
        else:
            mae_te = rmse_te = r2_te = spearman_te = bias_te = np.nan
            test_age_range = (np.nan, np.nan)
            n_test = 0
            cross_drift = np.nan

        # Diagnostics
        per_class_coef = {c: float(self.coefficients_[i]) for i, c in enumerate(CLASSES)}
        self.diagnostics_ = ClockDiagnostics(
            n_train=N, n_test=n_test,
            train_age_range=(float(ytr.min()), float(ytr.max())),
            test_age_range=test_age_range,
            mae_train=mae_tr, mae_test=mae_te,
            rmse_train=rmse_tr, rmse_test=rmse_te,
            r2_train=r2_tr, r2_test=r2_te,
            spearman_test=spearman_te,
            bias_test=bias_te,
            per_class_coefficients=per_class_coef,
            intercept=self.intercept_,
            mae_gate_pass=bool(mae_te < 5.0) if not np.isnan(mae_te) else False,
            cross_cohort_drift_pct=cross_drift,
        )
        self.is_fitted_ = True
        return self

    def _predict_raw(self, X):
        return self.intercept_ + X @ self.coefficients_

    def predict(self, a_scores):
        if not self.is_fitted_: raise RuntimeError("Clock not fitted")
        return self._predict_raw(a_scores)

    def cellular_age_departure(self, a_scores, chronological_ages):
        """
        Cellular age departure = predicted - chronological.

        Positive departure → cellular architecture appears older than chronological age
        (acceleration). Negative → appears younger. This is the analog of Horvath's
        "epigenetic age acceleration."
        """
        return self.predict(a_scores) - chronological_ages

    def save_diagnostics(self, path: str):
        if self.diagnostics_ is None: raise RuntimeError("No diagnostics")
        with open(path, 'w') as f:
            json.dump(self.diagnostics_.to_dict(), f, indent=2, default=str)


# =========================================================================
# Helper: compute 8-class A-scores from β matrix
# =========================================================================
def compute_8class_ascores(beta_dict_per_patient, ascoring_module_path=None):
    """
    Compute 8 architectural-class A-scores per patient using the IAMAtlas v0.1
    A-scoring module (iamatlas_a_scoring.py).
    """
    import sys
    sys.path.insert(0, '/home/claude/iamatlas_v0_2_extension')
    from iamatlas_a_scoring import score_per_class, load_artifact
    artifact_path = '/home/claude/iamatlas_v0_2_extension/iamatlas_celltype_markers_v0_1.json'
    markers, hmin, ct_to_class, atlas_means = load_artifact(artifact_path)
    rows = []
    for gsm, beta_dict in beta_dict_per_patient.items():
        scores = score_per_class(beta_dict, markers, hmin, ct_to_class, atlas_means)
        row = {'gsm': gsm}
        for cls in CLASSES:
            row[f'A_{cls}'] = scores.get(cls, {}).get('A', np.nan)
        rows.append(row)
    return pd.DataFrame(rows)
