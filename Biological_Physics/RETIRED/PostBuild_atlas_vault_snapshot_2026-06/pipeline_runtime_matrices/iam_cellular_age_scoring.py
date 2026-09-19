#!/usr/bin/env python3
"""
iam_cellular_age_scoring.py — Per-class cellular age via canonical IAM inversion (Stage 3).

CANONICAL PER RECIPE §6.3 (v3 of v1_CPG_Recipe.md, lines 1884-1911):

    "For each class with a non-saturated A-score, EDEAR computes a per-class
    cellular age by inverting the age-baseline curve from Part 2.3 — the age
    at which the class population mean β equals the customer's measured β.
    Linear interpolation in (β_mean, decade) for class_name to find the age
    at which population β_mean = beta_class.
    Note: β_mean DECREASES with age in healthy reference."

CANONICAL PER RECIPE §2.3 (line 358):

    "A is computed via A = H(β_mean) / H_min(class, methyl)."

WHAT THIS REPLACES
------------------
v1 — was a Horvath-style elastic-net regression on A-scores. Wrong (was a
     training-set-based clock, not a physics inversion).
v2 — tried to invert against the A_mean column of the baseline using the
     wrong A-score formula. Wrong (used mean(H(β_i))/H_min instead of
     H(β_mean)/H_min — different by Jensen's inequality).

v3 — does what the Recipe actually says. β_mean inversion, per class, no
     A-scores in the loop. The reference matrix supplies (decade → β_mean)
     per class as a calibrated lookup curve. The patient's β_mean per class
     is computed once per class. The cellular age is the age at which the
     baseline curve crosses the patient's β_mean for that class.

USAGE
-----
    from iam_cellular_age_v3 import IAMCellularAgeV3
    clock = IAMCellularAge()
    res = clock.score_patient(patient_beta_dict, chronological_age=54)
    # res.cellular_age_per_class -> {class: age_years}     ← 8 numbers
    # res.beta_mean_per_class -> {class: β_mean}            ← raw inputs
    # res.summary_cellular_age -> float (n-weighted mean)   ← single number
    # res.compartments_accelerated -> [(class, +years)]
    # res.compartments_decelerated -> [(class, -years)]
    # res.saturated_classes -> list                          ← out-of-range
"""

import json
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

CLASSES = ['stem_pluri','stem_adult','progenitor','cycling','secretory',
           'immune','terminal','stromal']

H_MIN_BY_CLASS = {
    'terminal': 0.7728, 'immune': 0.838889, 'secretory': 0.843264,
    'cycling': 0.856055, 'progenitor': 0.852216, 'stromal': 0.86295,
    'stem_adult': 0.873718, 'stem_pluri': 0.982166,
}

DEFAULT_REF_PATH = Path(__file__).parent / 'age_reference_matrix.json'
DEFAULT_MARKERS_PATH = Path('/home/claude/iamatlas_v0_2_extension/iamatlas_celltype_markers_v0_1.json')


@dataclass
class CellularAgeResult:
    """Per-patient cellular age — 8 per-class numbers, optional summary."""
    patient_id: Optional[str]
    chronological_age: Optional[float]

    # Raw inputs to the inversion (the canonical β_mean per class)
    beta_mean_per_class: dict           # {class: β_mean} — mean of β over class markers

    # Eight per-class cellular ages (the inversion readback)
    cellular_age_per_class: dict        # {class: age_years}

    # Per-class A-scores using the CANONICAL formula H(β_mean)/H_min
    a_score_per_class: dict             # {class: A}

    # Per-class status: 'OK' / 'SATURATED_HIGH' (β above youngest bin's β_mean) /
    # 'SATURATED_LOW' (β below oldest bin's β_mean) / 'INSUFFICIENT_CPGS'
    status_per_class: dict

    # Per-class CpG counts used in β_mean
    n_cpgs_per_class: dict

    # Summary cellular age — n_samples-weighted mean across non-saturated classes
    summary_cellular_age: Optional[float]

    # Disagreement structure (the readout single-number clocks can't produce)
    age_spread: Optional[float]         # max - min across class cellular ages
    age_median: Optional[float]
    age_iqr: Optional[float]

    # Compartment classification vs chronological age (if supplied)
    compartments_accelerated: list      # [(class, +years above +concordance_window)]
    compartments_decelerated: list
    compartments_concordant: list
    saturated_classes: list             # classes where β was outside baseline range

    # Status / notes
    status: str
    notes: list

    def to_dict(self):
        return asdict(self)


class IAMCellularAge:
    """
    Canonical IAM cellular age clock per Recipe §6.3.

    For each architectural class:
      - Compute β_mean = mean of patient's β over that class's marker CpGs
      - Invert the baseline's (decade → β_mean) curve to find the age at which
        baseline β_mean equals patient β_mean
      - That's the per-class cellular age

    No training. No regression. The baseline IS the instrument; the patient's β_mean
    IS the measurement; the cellular age IS the readback.
    """

    def __init__(self, ref_matrix_path: Optional[str] = None,
                 markers_artifact_path: Optional[str] = None,
                 markers_per_class: Optional[dict] = None,
                 min_cpgs_per_class: int = 30):
        self.ref_matrix_path = Path(ref_matrix_path) if ref_matrix_path else DEFAULT_REF_PATH
        self.min_cpgs_per_class = min_cpgs_per_class
        self.classes = CLASSES
        self._load_ref()
        if markers_per_class is not None:
            self.class_markers = {c: set(markers_per_class.get(c, [])) for c in CLASSES}
        else:
            mp = Path(markers_artifact_path) if markers_artifact_path else DEFAULT_MARKERS_PATH
            self._load_markers(mp)

    def _load_ref(self):
        """Load 80-cell baseline from age_reference_matrix.json."""
        data = json.load(open(self.ref_matrix_path))
        self.ref = {k: v for k, v in data.items() if k != '_meta'}
        for c in self.classes:
            if c not in self.ref:
                raise ValueError(f"Class {c} missing from age reference matrix")
            self.ref[c] = sorted(self.ref[c], key=lambda r: r['age_midpoint'])

    def _load_markers(self, mp: Path):
        """Load per-class marker pool from celltype-marker artifact."""
        ma = json.load(open(mp))
        ct_to_class = ma['celltype_to_class']
        markers_by_ct = ma['markers_by_celltype']
        cm = {c: set() for c in CLASSES}
        for ct, mks in markers_by_ct.items():
            cls = ct_to_class.get(ct)
            if cls in cm:
                cm[cls].update(mks[:100])
        self.class_markers = cm

    # ----------------------------------------------------------------
    # Core β_mean inversion (the canonical operation per §6.3)
    # ----------------------------------------------------------------
    def invert_beta_to_age(self, arch_class: str, patient_beta_mean: float) -> tuple:
        """
        Find the age at which baseline β_mean(class, age) = patient_beta_mean.

        β_mean DECREASES with age in healthy reference (per §6.3 line 1900).
        So patient with β_mean above the youngest decade's β_mean is "younger than
        youngest reference" — saturate high (assign youngest age, flag SATURATED_HIGH).
        Patient with β_mean below the oldest decade's β_mean is "older than oldest
        reference" — saturate low (assign oldest age, flag SATURATED_LOW).

        Returns (cellular_age, status).
        """
        bins = self.ref[arch_class]
        ages = np.array([b['age_midpoint'] for b in bins])
        b_means = np.array([b['beta_mean'] for b in bins])

        # β_mean decreases with age → sorted ascending in age means sorted descending in β
        # patient β >= max(b_means) → younger than youngest bin
        if patient_beta_mean >= b_means.max():
            return (float(ages[b_means.argmax()]), 'SATURATED_HIGH')
        # patient β <= min(b_means) → older than oldest bin
        if patient_beta_mean <= b_means.min():
            return (float(ages[b_means.argmin()]), 'SATURATED_LOW')

        # Linear interpolation between bracketing bins
        for i in range(len(bins) - 1):
            b_lo, b_hi = b_means[i], b_means[i + 1]
            age_lo, age_hi = ages[i], ages[i + 1]
            # Since β_mean decreases with age: b_lo > b_hi
            if (b_hi <= patient_beta_mean <= b_lo) or (b_lo <= patient_beta_mean <= b_hi):
                if b_lo == b_hi:
                    return (float((age_lo + age_hi) / 2), 'OK_FLAT')
                # fraction of the way from b_lo down to b_hi
                frac = (b_lo - patient_beta_mean) / (b_lo - b_hi)
                age = float(age_lo + frac * (age_hi - age_lo))
                return (age, 'OK')

        # Fallback (shouldn't reach with monotonic data)
        return (float(np.interp(patient_beta_mean, b_means[::-1], ages[::-1])), 'OK_FALLBACK')

    # ----------------------------------------------------------------
    # Per-patient scoring
    # ----------------------------------------------------------------
    @staticmethod
    def _shannon_bits(beta: float) -> float:
        """Shannon binary entropy in bits."""
        if beta <= 0.0 or beta >= 1.0:
            return 0.0
        return -beta * np.log2(beta) - (1 - beta) * np.log2(1 - beta)

    def score_patient(self, beta_dict: dict, chronological_age: Optional[float] = None,
                      patient_id: Optional[str] = None,
                      concordance_window_years: float = 5.0) -> CellularAgeResult:
        """
        Score a single patient.

        beta_dict: {cpg_id: β_value} — patient's per-CpG methylation
        chronological_age: optional, for compartment classification
        """
        # Drop NaN β values
        clean_beta = {c: v for c, v in beta_dict.items()
                      if v is not None and not (isinstance(v, float) and np.isnan(v))}

        beta_mean_per_class = {}
        n_cpgs_per_class = {}
        cellular_age_per_class = {}
        a_score_per_class = {}
        status_per_class = {}
        notes = []

        for cls in self.classes:
            # β values at this class's marker CpGs (intersect with patient's CpGs)
            markers = self.class_markers[cls]
            betas = [clean_beta[c] for c in markers if c in clean_beta]
            n_cpgs_per_class[cls] = len(betas)

            if len(betas) < self.min_cpgs_per_class:
                beta_mean_per_class[cls] = np.nan
                cellular_age_per_class[cls] = np.nan
                a_score_per_class[cls] = np.nan
                status_per_class[cls] = 'INSUFFICIENT_CPGS'
                notes.append(f"{cls}: only {len(betas)} CpGs available, need {self.min_cpgs_per_class}")
                continue

            # CANONICAL β_mean (Recipe §6.3)
            beta_mean = float(np.mean(betas))
            beta_mean_per_class[cls] = beta_mean

            # Inversion (Recipe §6.3)
            age, status = self.invert_beta_to_age(cls, beta_mean)
            cellular_age_per_class[cls] = age
            status_per_class[cls] = status

            # CANONICAL A-score (Recipe §2.3 line 358): A = H(β_mean) / H_min
            a_score_per_class[cls] = float(self._shannon_bits(beta_mean) / H_MIN_BY_CLASS[cls])

        # Summary cellular age — n_samples-weighted mean across non-saturated classes
        non_sat = [c for c in self.classes if status_per_class[c] == 'OK']
        if non_sat:
            # Weight each class by the total n_samples backing its baseline curve
            class_weights = {c: sum(b['n_samples'] for b in self.ref[c]) for c in non_sat}
            wt_total = sum(class_weights.values())
            summary = sum(cellular_age_per_class[c] * class_weights[c] for c in non_sat) / wt_total
        else:
            summary = None

        # Disagreement structure
        valid_ages = np.array([cellular_age_per_class[c] for c in self.classes
                                if status_per_class[c] in {'OK','SATURATED_HIGH','SATURATED_LOW'}
                                and not np.isnan(cellular_age_per_class[c])])
        if len(valid_ages) >= 2:
            age_spread = float(valid_ages.max() - valid_ages.min())
            age_median = float(np.median(valid_ages))
            age_iqr = float(np.percentile(valid_ages, 75) - np.percentile(valid_ages, 25))
        else:
            age_spread = age_median = age_iqr = None

        # Compartment classification
        accelerated, decelerated, concordant = [], [], []
        if chronological_age is not None:
            for cls in self.classes:
                age = cellular_age_per_class[cls]
                if np.isnan(age) or status_per_class[cls] in {'INSUFFICIENT_CPGS'}: continue
                delta = age - chronological_age
                tag = (cls, round(delta, 2))
                if delta > concordance_window_years:
                    accelerated.append(tag)
                elif delta < -concordance_window_years:
                    decelerated.append(tag)
                else:
                    concordant.append(tag)

        saturated = [c for c in self.classes
                     if status_per_class[c] in {'SATURATED_HIGH', 'SATURATED_LOW'}]

        n_ok = sum(1 for s in status_per_class.values() if s == 'OK')
        if n_ok >= 6:
            overall = 'OK'
        elif n_ok >= 4:
            overall = 'OK_PARTIAL'
        elif n_ok >= 1:
            overall = 'OK_LIMITED'
        else:
            overall = 'ALL_SATURATED_OR_INSUFFICIENT'

        return CellularAgeResult(
            patient_id=patient_id,
            chronological_age=chronological_age,
            beta_mean_per_class={c: (float(v) if not np.isnan(v) else None)
                                 for c, v in beta_mean_per_class.items()},
            cellular_age_per_class={c: (float(v) if not np.isnan(v) else None)
                                    for c, v in cellular_age_per_class.items()},
            a_score_per_class={c: (float(v) if not np.isnan(v) else None)
                               for c, v in a_score_per_class.items()},
            status_per_class=status_per_class,
            n_cpgs_per_class=n_cpgs_per_class,
            summary_cellular_age=summary,
            age_spread=age_spread,
            age_median=age_median,
            age_iqr=age_iqr,
            compartments_accelerated=accelerated,
            compartments_decelerated=decelerated,
            compartments_concordant=concordant,
            saturated_classes=saturated,
            status=overall,
            notes=notes,
        )

    def score_batch(self, beta_matrix_df: pd.DataFrame, metadata_df: Optional[pd.DataFrame] = None,
                    cpg_col: str = 'CpGs', age_col: str = 'age',
                    id_col: str = 'gsm') -> pd.DataFrame:
        """
        Score a cohort.

        beta_matrix_df: rows=CpGs, columns = patient IDs (one column = cpg_col)
        metadata_df: optional, indexed by id_col, supplies age
        """
        cpgs = beta_matrix_df[cpg_col].values
        patient_cols = [c for c in beta_matrix_df.columns if c != cpg_col]

        if metadata_df is not None:
            md = metadata_df.set_index(id_col) if id_col in metadata_df.columns else metadata_df
        else:
            md = None

        rows = []
        for pid in patient_cols:
            beta_dict = dict(zip(cpgs, beta_matrix_df[pid].values))
            chrono = None
            if md is not None and pid in md.index:
                chrono = md.loc[pid, age_col] if not pd.isna(md.loc[pid, age_col]) else None
                if chrono is not None: chrono = float(chrono)
            res = self.score_patient(beta_dict, chrono, pid)
            row = {'gsm': pid, 'chronological_age': chrono}
            for c in self.classes:
                row[f'beta_mean_{c}'] = res.beta_mean_per_class[c]
                row[f'cellular_age_{c}'] = res.cellular_age_per_class[c]
                row[f'A_{c}'] = res.a_score_per_class[c]
                row[f'status_{c}'] = res.status_per_class[c]
                row[f'n_cpgs_{c}'] = res.n_cpgs_per_class[c]
            row['summary_cellular_age'] = res.summary_cellular_age
            row['age_spread'] = res.age_spread
            row['age_median'] = res.age_median
            row['age_iqr'] = res.age_iqr
            row['n_accelerated'] = len(res.compartments_accelerated)
            row['n_decelerated'] = len(res.compartments_decelerated)
            row['n_concordant'] = len(res.compartments_concordant)
            row['n_saturated'] = len(res.saturated_classes)
            row['accelerated_classes'] = ','.join(c for c, _ in res.compartments_accelerated)
            row['decelerated_classes'] = ','.join(c for c, _ in res.compartments_decelerated)
            row['saturated_classes'] = ','.join(res.saturated_classes)
            row['status'] = res.status
            rows.append(row)
        return pd.DataFrame(rows)
