#!/usr/bin/env python3
"""
walther_clinical.py — CPG clinical orchestrator.

CURRENT SCOPE: Stages 3, 4, 4.5, 4.6, 5, and 6, built to BUILD_SPEC v1.3 (Option C
age architecture). Stages 0-2, 7-10 are declared as explicit NotImplemented
placeholders so the contract is honest about what is and is not built yet.

Option C (hybrid, V1 simplification)
------------------------------------
Stage 3 FORKS the per-patient beta vector into two products:

  * cleaned_beta : age + smoking + sex foreground subtracted. Age is a NUISANCE
                   here. Consumed by Stage 4 (A-score), 4.5 (bidirectional),
                   4.6 (brightness), 5 (Mahalanobis), 8 (disease matching).
  * beta_raw     : the calibrated beta with NO foreground subtraction, passed
                   through untouched. Age is the SIGNAL here. Consumed ONLY by
                   Stage 6 (cellular age inversion).

This fork is what resolves the Stage 3 <-> Stage 6 collision: the cellular-age
inversion reads age OUT of beta by matching per-class beta_mean to the age-indexed
baseline. If it were handed the age-subtracted beta, that signal would be gone and
every patient would read near the training-cohort mean age.

Cellular age in V1 is CLASS-LEVEL (8 per-class absolute ages + n-weighted summary).
The per-cell (115) confidence-weighted total-departure method (decision D6) is V2
work and is intentionally NOT implemented here.

Beta convention
---------------
Per-patient beta is carried as a pandas Series indexed by cpg_id (matches the
foreground modules' subtract_from_single_patient signature). Stage 6's
IAMCellularAge.score_patient consumes a plain {cpg_id: beta} dict, so beta_raw is
converted with .to_dict() at the Stage 6 boundary.

Module loading
--------------
The real runtime modules live in folders whose names contain spaces, so they are
loaded by absolute file path via importlib rather than by package import. Paths are
declared in DEFAULT_CONFIG (repo-relative) and can be overridden per deployment.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — repo-relative paths to the real runtime dependencies.
# Resolved relative to CPG_CMB_v1/ (two levels up from this file's folder).
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
CPG_ROOT = _THIS_DIR.parent  # .../AstroGenetics/CPG_CMB_v1

DEFAULT_CONFIG = {
    # Foreground modules (Stage 3)
    "age_module_path":   CPG_ROOT / "Runtime Matrices/Age_Sex_Smoker Axis Foreground/Age Axis Foreground/age_axis_foreground.py",
    "smoking_module_path": CPG_ROOT / "Runtime Matrices/Age_Sex_Smoker Axis Foreground/SEX_SMOKER Axis Foreground/Smoking Axis Foreground/smoking_axis_foreground.py",
    "sex_module_path":   CPG_ROOT / "Runtime Matrices/Age_Sex_Smoker Axis Foreground/SEX_SMOKER Axis Foreground/Sex Axis Foreground/sex_axis_foreground.py",
    # Frozen per-CpG layers (Stage 3)
    "age_layer_csv":     CPG_ROOT / "Runtime Matrices/Age_Sex_Smoker Axis Foreground/Age Axis Foreground/IAMAtlas_age_layer.csv",
    "smoking_layer_csv": CPG_ROOT / "Runtime Matrices/Age_Sex_Smoker Axis Foreground/SEX_SMOKER Axis Foreground/Smoking Axis Foreground/IAMAtlas_smoking_layer.csv",
    "sex_layer_csv":     CPG_ROOT / "Runtime Matrices/Age_Sex_Smoker Axis Foreground/SEX_SMOKER Axis Foreground/Sex Axis Foreground/IAMAtlas_sex_layer.csv",
    # Cellular age (Stage 6)
    "cellular_age_module_path": CPG_ROOT / "Runtime Matrices/Age_Sex_Smoker Axis Foreground/Age Axis Foreground/iam_cellular_age_scoring.py",
    "age_reference_matrix_json": CPG_ROOT / "Runtime Matrices/Age_Reference_Matrix 80_cells/age_reference_matrix.json",
    "celltype_markers_json": CPG_ROOT / "Runtime Matrices/Celltype_Marker/iamatlas_celltype_markers_v0_2.json",
    # Stage 4 — A-score
    "a_scoring_module_path": CPG_ROOT / "Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py",
    # Stage 4.5 — bidirectional decomposition
    "bidirectional_module_path": CPG_ROOT / "Runtime Matrices/Bidirectional Decomposition py/bidirectional_decomposition.py",
    "directional_panels_json": CPG_ROOT / "Runtime Matrices/Bidirectional Decomposition py/directional_panels_v1_0.json",
    # Stage 4.6 — brightness comparison / Mollweide
    "brightness_module_path": CPG_ROOT / "Runtime Matrices/Mollweide & Brightness Comparison/patient_brightness_comparison.py",
    "brightness_archives_dir": CPG_ROOT / "IAM_Atlas/iamatlas_class_archives",
    # Stage 5 — Mahalanobis
    "mahalanobis_module_path": CPG_ROOT / "Runtime Matrices/Mahalanobis_healthy_reference/iamatlas_mahalanobis_scoring.py",
    "mahalanobis_reference_json": CPG_ROOT / "Runtime Matrices/Mahalanobis_healthy_reference/mahalanobis_healthy_reference_v0_5.json",
    # Stage 9 — brilliance maps (HEALPix Mollweide)
    "cpg_healpix_mapping_npy": CPG_ROOT / "Runtime Matrices/cpg healpix mapping/iamatlas_cpg_to_healpix_nside128.npy",
    # Behaviour flags
    "apply_smoking_foreground": True,   # applied iff the smoking layer CSV exists
    "apply_sex_foreground": True,       # applied iff the sex layer CSV exists
}


def _load_module(path, name):
    """Load a module from an absolute file path (handles spaces in folder names)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Stage 3 — Foreground subtraction (Option C fork)
# ---------------------------------------------------------------------------
@dataclass
class Stage3Output:
    beta_raw: pd.Series        # untouched calibrated beta  -> Stage 6 ONLY
    cleaned_beta: pd.Series    # age (+ smoking + sex) removed -> Stage 4/4.5/4.6/5/8
    foregrounds_applied: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def stage_3_foreground_fork(beta_calibrated: pd.Series,
                            patient_age: Optional[float],
                            patient_sex: Optional[str] = None,
                            patient_smoking_bin: Optional[str] = None,
                            config: Optional[dict] = None) -> Stage3Output:
    """Stage 3 per BUILD_SPEC v1.3.

    Args:
        beta_calibrated : pd.Series of calibrated beta indexed by cpg_id (Stage 1 output).
        patient_age     : chronological age in years (None -> no age subtraction).
        patient_sex     : 'M'/'F'/'male'/'female' etc (None or no layer -> skip sex).
        patient_smoking_bin : never / former_15plus_y / former_5_15y / former_0_5y /
                              current (None or no layer -> skip smoking).
    Returns:
        Stage3Output with beta_raw (untouched) and cleaned_beta (foregrounds removed).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if not isinstance(beta_calibrated, pd.Series):
        raise TypeError("beta_calibrated must be a pd.Series indexed by cpg_id")

    # beta_raw is the untouched calibrated beta, reserved for Stage 6.
    beta_raw = beta_calibrated.copy()

    applied, notes = [], []

    # --- Age (always, per v1.2 default; the architecturally required L4 axis) ---
    age_mod = _load_module(cfg["age_module_path"], "age_axis_foreground")
    afg = age_mod.AgeAxisForeground()
    afg.load_layer(str(cfg["age_layer_csv"]))
    cleaned_beta = afg.subtract_from_single_patient(beta_raw, patient_age)
    applied.append("age")
    if patient_age is None:
        notes.append("patient_age is None -> age component not subtracted (cleaned_beta == beta_raw for age axis)")

    # --- Smoking (if layer present and enabled) ---
    smk_csv = Path(cfg["smoking_layer_csv"])
    if cfg.get("apply_smoking_foreground", True) and smk_csv.exists() and patient_smoking_bin is not None:
        smk_mod = _load_module(cfg["smoking_module_path"], "smoking_axis_foreground")
        smk = smk_mod.SmokingAxisForeground()
        smk.load_layer(str(smk_csv))
        cleaned_beta = smk.subtract_from_single_patient(cleaned_beta, patient_smoking_bin)
        applied.append("smoking")
    else:
        notes.append("smoking foreground not applied (layer missing, disabled, or smoking_bin None) "
                      "-- interim Stage 7 smoking-bin threshold stratification absorbs residual")

    # --- Sex (if layer present and enabled) ---
    sex_csv = Path(cfg["sex_layer_csv"])
    if cfg.get("apply_sex_foreground", True) and sex_csv.exists() and patient_sex is not None:
        sex_mod = _load_module(cfg["sex_module_path"], "sex_axis_foreground")
        sex_fg = sex_mod.SexAxisForeground()
        sex_fg.load_layer(str(sex_csv))
        cleaned_beta = sex_fg.subtract_from_single_patient(cleaned_beta, patient_sex)
        applied.append("sex")
    else:
        notes.append("sex foreground not applied (layer missing, disabled, or sex None) "
                      "-- interim Stage 7 sex-stratified threshold tables absorb residual")

    # batch / ancestry: documented gap (modules not built) per BUILD_SPEC v1.3 Stage 3.4
    notes.append("batch/ancestry foregrounds NOT subtracted at CpG level (documented gap)")

    return Stage3Output(beta_raw=beta_raw, cleaned_beta=cleaned_beta,
                        foregrounds_applied=applied, notes=notes)


# ---------------------------------------------------------------------------
# Stage 6 — Cellular age inversion (consumes beta_raw, NOT cleaned_beta)
# ---------------------------------------------------------------------------
def stage_6_cellular_age(beta_raw: pd.Series,
                         chronological_age: Optional[float],
                         patient_id: Optional[str] = None,
                         config: Optional[dict] = None):
    """Stage 6 per BUILD_SPEC v1.3.

    Inverts the age-indexed baseline on the PRE-foreground beta_raw to produce
    class-level absolute cellular ages. Returns the module's CellularAgeResult
    (8 per-class ages, n-weighted summary_cellular_age, accelerated/decelerated/
    concordant compartments vs chronological age, saturation flags, overall status).

    CRITICAL: pass beta_raw here, never cleaned_beta -- see module docstring.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if not isinstance(beta_raw, pd.Series):
        raise TypeError("beta_raw must be a pd.Series indexed by cpg_id")

    ca_mod = _load_module(cfg["cellular_age_module_path"], "iam_cellular_age_scoring")
    ca = ca_mod.IAMCellularAge(
        ref_matrix_path=str(cfg["age_reference_matrix_json"]),
        markers_artifact_path=str(cfg["celltype_markers_json"]),
    )
    # score_patient consumes a {cpg_id: beta} dict, not a Series.
    age_result = ca.score_patient(
        beta_dict=beta_raw.to_dict(),
        chronological_age=chronological_age,
        patient_id=patient_id,
    )
    return age_result


# ---------------------------------------------------------------------------
# Stage 9 helper — Personal Brilliance Maps (HEALPix Mollweide). Consumes the
# Stage 4.6 PatientBrightnessReport. Requires healpy at runtime.
# ---------------------------------------------------------------------------
def render_brilliance_maps(brightness_report, output_dir, patient_id="patient",
                           config=None):
    """Render the patient's Personal Brilliance Maps (Appendix A3):
      - 8 individual per-class panels  -> personal_brilliance_map_{class}.png
      - 1 whole-atlas 8-panel figure   -> personal_brilliance_map_whole_atlas.png

    These are the patient's per-class z-score departures projected onto the same
    HEALPix NSIDE=128 grid as the Cosmic Methylome Background reference (Plate 1),
    so the customer's maps sit on the identical ruler as the atlas reference.

    brightness_report : the PatientBrightnessReport from stage_4_6_brightness.
    Requires healpy (pip install healpy) at runtime.
    """
    import numpy as np
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    br_mod = _load_module(cfg["brightness_module_path"], "patient_brightness_comparison")
    cpg_to_pixel = np.load(str(cfg["cpg_healpix_mapping_npy"]))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}

    # 8 individual per-class panels
    for cls, departure in brightness_report.per_class_results.items():
        pixel_values = br_mod.project_to_healpix_pixels(departure, cpg_to_pixel)
        fig = br_mod.render_patient_mollweide_panel(
            pixel_values, cls,
            title=f"{cls.upper()} \u00b7 personal departure vs Cosmic Methylome Background")
        panel_path = out / f"{patient_id}_personal_brilliance_map_{cls}.png"
        fig.savefig(panel_path, dpi=120, bbox_inches="tight", facecolor="black")
        import matplotlib.pyplot as plt
        plt.close(fig)
        written[cls] = panel_path

    # 1 whole-atlas (all 8 classes in one figure)
    whole = out / f"{patient_id}_personal_brilliance_map_whole_atlas.png"
    br_mod.render_patient_cosmic_methylome(brightness_report, cpg_to_pixel, whole)
    written["whole_atlas"] = whole
    return written



def _not_built(stage):
    raise NotImplementedError(
        f"{stage} is not implemented in this build. Current scope: Stage 3 + Stage 6 "
        f"per BUILD_SPEC v1.3. See the build spec for the remaining stage contracts."
    )

def stage_0_intake(*a, **k):            _not_built("Stage 0 (intake)")
def stage_1_calibration_beta(*a, **k):  _not_built("Stage 1 (calibration & beta)")
def stage_2_deconvolution(*a, **k):     _not_built("Stage 2 (deconvolution)")
# ---------------------------------------------------------------------------
# Stage 4 — A-score (per-class + per-cell-type). Consumes cleaned_beta.
# ---------------------------------------------------------------------------
def _aggregate_markers_to_class(markers_by_celltype, celltype_to_class):
    """Union the per-cell-type marker CpGs up to their parent class (dedup, order-preserving)."""
    from collections import OrderedDict
    class_markers = {}
    for ct, markers in markers_by_celltype.items():
        cls = celltype_to_class.get(ct)
        if cls is None:
            continue
        d = class_markers.setdefault(cls, OrderedDict())
        for m in markers:
            d[m] = None
    return {cls: list(d.keys()) for cls, d in class_markers.items()}


def stage_4_a_score(cleaned_beta: pd.Series, config: Optional[dict] = None) -> dict:
    """Stage 4 per BUILD_SPEC v1.3 — consumes cleaned_beta (foreground-removed).

    A = H(beta_mean) / H_min(class) over panel CpGs, per class and per cell type.
    Returns {'class_ascores', 'celltype_ascores', 'h_min_by_class', 'celltype_to_class'}.
    (95% CI propagation from atlas posteriors is added at the report layer; the
    point A-scores are produced here.)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    a_mod = _load_module(cfg["a_scoring_module_path"], "iamatlas_a_scoring")
    meta, ct_markers, ct_to_class, h_min = a_mod.load_artifact(str(cfg["celltype_markers_json"]))
    beta_dict = cleaned_beta.to_dict()
    class_markers = _aggregate_markers_to_class(ct_markers, ct_to_class)
    class_ascores = a_mod.score_per_class(beta_dict, class_markers, h_min)
    celltype_ascores = a_mod.score_per_celltype(beta_dict, ct_markers, ct_to_class, h_min)
    return {
        "class_ascores": class_ascores,           # {class: {A, coverage, status, ...}}  (8)
        "celltype_ascores": celltype_ascores,     # {celltype: {A, class, status, ...}}  (115)
        "h_min_by_class": h_min,
        "celltype_to_class": ct_to_class,
    }


# ---------------------------------------------------------------------------
# Stage 4.5 — Bidirectional decomposition. Consumes cleaned_beta.
# ---------------------------------------------------------------------------
def stage_4_5_bidirectional(cleaned_beta: pd.Series, patient_id: str = "patient",
                            config: Optional[dict] = None):
    """Stage 4.5 per BUILD_SPEC v1.3 — directional composite + pooled-entropy comparator
    + FLAG_BIDIRECTIONAL per class, on cleaned_beta. Recovers signal that pooled averaging
    hides (the VAL-050 null vs VAL-051 directional lesson). Returns a BidirectionalReport.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    bd_mod = _load_module(cfg["bidirectional_module_path"], "bidirectional_decomposition")
    panels = bd_mod.load_directional_panels(str(cfg["directional_panels_json"]))
    return bd_mod.compute_per_class_bidirectional_decomposition(
        cleaned_beta, panels, patient_id=patient_id)


# ---------------------------------------------------------------------------
# Stage 4.6 — Per-class brightness departure / personal methylome. Consumes cleaned_beta.
# ---------------------------------------------------------------------------
def stage_4_6_brightness(cleaned_beta: pd.Series, patient_id: str = "patient",
                         config: Optional[dict] = None):
    """Stage 4.6 per BUILD_SPEC v1.3 — per-CpG z-score departure of the patient from each
    class's healthy reference (the customer's personal methylome map; the Mollweide panel
    is rendered downstream at Stage 9). Returns a PatientBrightnessReport.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    br_mod = _load_module(cfg["brightness_module_path"], "patient_brightness_comparison")
    references = br_mod.load_all_8_class_references(str(cfg["brightness_archives_dir"]))
    return br_mod.compute_all_8_class_departures(cleaned_beta, references, patient_id=patient_id)


# ---------------------------------------------------------------------------
# Stage 5 — Mahalanobis distance of the 115 per-cell-type A-scores vs the HC hull.
# Consumes the Stage 4 per-cell-type A-scores (NOT beta directly).
# ---------------------------------------------------------------------------
def stage_5_mahalanobis(stage4_output: dict, config: Optional[dict] = None) -> dict:
    """Stage 5 per BUILD_SPEC v1.3 — multivariate distance of the patient's 115 per-cell-type
    A-scores from the healthy-cohort hull centroid. stage4_output is the dict from
    stage_4_a_score; its 'celltype_ascores' supply the feature vector. Returns the module's
    score dict (mahalanobis_distance, status, top10_axis_contributions, ...).
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    m_mod = _load_module(cfg["mahalanobis_module_path"], "iamatlas_mahalanobis_scoring")
    hull = m_mod.MahalanobisHealthyHull(str(cfg["mahalanobis_reference_json"]))
    ct_results = stage4_output.get("celltype_ascores", stage4_output)
    celltype_a = {ct: (v["A"] if isinstance(v, dict) else v) for ct, v in ct_results.items()}
    return hull.score(celltype_a)
def stage_7_tiers(*a, **k):             _not_built("Stage 7 (tier breakpoints)")
def stage_8_dual_matching(*a, **k):     _not_built("Stage 8 (dual matching)")
def stage_9_report(*a, **k):            _not_built("Stage 9 (report assembly)")
def stage_10_delivery(*a, **k):         _not_built("Stage 10 (delivery)")


if __name__ == "__main__":
    print("walther_clinical.py — Stages 3, 4, 4.5, 4.6, 5 & 6 implemented (BUILD_SPEC v1.3).")
    print("Stage 3 forks beta_raw (-> Stage 6) and cleaned_beta (-> Stages 4/4.5/4.6/5).")
    print("Stage 4 = A-scores (8 class + 115 cell type); 4.5 = bidirectional; 4.6 = brightness;")
    print("Stage 5 = Mahalanobis on the 115 cell-type A-scores; Stage 6 = cellular age on beta_raw.")
    print("Stages 0-2, 7-10 raise NotImplementedError by design.")
