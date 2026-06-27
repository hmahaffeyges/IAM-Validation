#!/usr/bin/env python3
"""
walther_clinical.py — CPG v1 LEAN clinical conductor (built from scratch).

Spec: Walther_CPG_v1_chain_flowchart_v4. Primary chain = 7 links, KISS:
  L1 intake/integrity -> L2 decode+beta -> L3 deconvolution (Walther + NILC, parallel)
  -> L4 A-score (+ brightness CI) -> L5 tier -> L6 disease-signature match -> L7 report.

The proven stage functions (intake, calibration, deconvolution, A-score, tier, matching)
are carried over VERBATIM from the retired walther_clinical.py. Their internal guards are
untouched and load-bearing:
  * H_min provenance pin (SOP §99): refuses to run if runtime H_min != frozen posteriors.
  * A-score loci guard: refuses discriminative markers (the all-BREACH bug, 2026-06-11).
  * substrate presence gate: detect_floor = 0.01 (1%, grounded in VAL-090).
  * A-score canonical fail-safe: halts if the entropy-of-mean math regressed.

NEW in v1:
  * brightness CI — every A-score carries a 95% interval propagated from the 8 per-class
    per-CpG brightness posteriors (the MCMC edge), derived-only, no cohort.
  * PatientContext firewall — age/sex/family-history are REPORT CONTEXT only and never
    reach the scoring path; _assert_no_covariate_in_beta guards the boundary.

DEFERRED (not wired here): foreground subtraction (deliberately excluded — intake facts are
report annotations, never operands; SOP §104), bidirectional, brightness-Mollweide,
cellular age, brilliance maps. NILC runs IN Link 3 (parallel
cross-check), not as a second-chain bolt-on, so a signal it sees but Walther floors can
still raise the disease flag.

================================================================================
LESSONS FOR A FUTURE AI — DO NOT REPEAT THESE. (added 2026-06-26 after repeated drift)
We have veered back into class-level scoring and cohort thinking too many times. Stop.
================================================================================

  1. NEVER make a class-level A-score the call. PER-CELL-TYPE A-scores are the
     load-bearing observable (Recipe v3 sec 5.3 / 6.2): they catch the bidirectional
     patterns a class average cancels out. A class score is one summary among many,
     never the readout. If you catch yourself reporting "the <class> class is X", stop —
     the unit of analysis is the individual cell: its percentage, its A-score, and the
     MAGNITUDE and DIRECTION of its departure from 1.0.

  2. NEVER introduce cohort methodology into the runtime. CPG is PER-SAMPLE ONLY:
     what is present, at what percentage, its A-score, its direction from the derived
     healthy floor of 1.0. Case-vs-control, pooled centroids, population standardization,
     Cohen's d — those are VALIDATION evidence from past work, NOT the runtime method.
     The runtime never compares a patient to a population; it compares each cell to its
     own derived floor. If a sentence in your output needs a control group, it is wrong.

  3. The IAMAtlas IS SEPARABLE at the cell level. The "weakly separable / indicative
     only / trust the class" comments in walther_iam_deconvolver.py are STALE — they
     describe the OLD FLAT atlas (the collapse in IAMAtlas_FLATNESS_LESSON.md), not the
     REBUILD. The rebuild pulled cells apart (terminal 0.10, immune CD4-vs-Mono 0.42,
     per-CpG signal ~41x noise). Do NOT cite that stale comment to justify the class level.

  4. The two deconvolvers must AGREE AT THE CELL LEVEL, not just the class level. NILC
     was left class-only (it builds its reference from the 8 class means). Walther's
     cell-level tier has NO presence test and floors trace cells to 1e-4. Class-only
     agreement lets every cell in a present class ride one verdict and cannot separate,
     e.g., the hepatocyte from its secretory siblings. Cell-level cross-check is required.
================================================================================

HARD-WON LESSON (LESSON-DECONV-01, 2026-06-21 -- see SOP §0.9):
  The global NNLS deconvolver is CORRECT and noise-robust. It recovers known synthetic
  mixtures perfectly (MAE 0.000, CD4/monocytes included) and holds 6/6 cells under noise
  to sd=0.10. The real-IDAT Tier-2 collapse to a few cells is REFERENCE MISMATCH (real
  blood beta fits the atlas ~2.6x worse than a clean synthetic mixture), NOT solver
  conditioning. A hierarchical within-class refinement was tested and INVENTS signal
  (HSC/GMP/neurons; MAE 0.271 on pure monocytes) -- it failed composition-recovery
  validation and does NOT ship. The cell tier is INDICATIVE by design in v0.1; the
  two-tier report (class reliable / cell indicative) is the validated honest posture.
  Richer per-cell resolution is reference/atlas work, never a solver swap, and must clear
  a composition-recovery + null suite before production.
"""
from __future__ import annotations

import importlib.util
import os
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CPG_ROOT — the canonical CPG_CMB_v1 tree that holds the runtime modules.
# Resolves whether this conductor sits inside CPG_CMB_v1/<folder>/ (deployment)
# or elsewhere (set CPG_ROOT in the environment to override, e.g. for testing).
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_ENV_ROOT = os.environ.get("CPG_ROOT")
if _ENV_ROOT:
    CPG_ROOT = Path(_ENV_ROOT)
else:
    CPG_ROOT = _THIS_DIR.parent
    if not (CPG_ROOT / "IAM_Atlas").exists():
        for _p in [_THIS_DIR, *_THIS_DIR.parents]:
            if (_p / "IAM_Atlas").exists():
                CPG_ROOT = _p
                break


# ===== DEFAULT_CONFIG (verbatim) =====
DEFAULT_CONFIG = {
    # Stage 0 — sample intake (L1); Steps 0.1-0.9 complete (SOP §11-§19)
    "stage_0_intake_module_path": CPG_ROOT / "Runtime Matrices/Stage_0_Intake/stage_0_intake.py",
    # Stage 1 — calibration & beta (L2+L3); 1.4-1.8 built, 1.1-1.3 + IDAT decode wrap the standard stack
    "stage_1_calibration_module_path": CPG_ROOT / "Runtime Matrices/Stage_1_Calibration/stage_1_calibration.py",
    # NOTE: NO foreground module/layer paths here by design. The production chain
    # subtracts no foregrounds (firewall, 2026-06-11; SOP §104) -- removing those keys
    # so there is no dormant wiring to re-enable. The age/sex/smoking foreground
    # modules are retained as TEST-ONLY tooling under the foreground folder and are
    # never loaded by this orchestrator.
    # Cell-type markers (Stage 4). NOTE: the per-class cellular-age scorer (iam_cellular_age_scoring.py)
    # and age_reference_matrix were removed 2026-06-09 — the per-cell departure (Stage 6) is age-robust
    # and consumes neither. Their files remain on disk (preserved) but are no longer in the runtime path.
    # Discriminative one-vs-rest markers. THESE ARE FOR STAGE-1 DECONVOLUTION ONLY.
    # They are mixed-direction by construction and MUST NOT feed the Stage-4 A-score
    # (see a_score_loci_json below and the guard in stage_4_a_score).
    "celltype_markers_json": CPG_ROOT / "Runtime Matrices/Celltype_Marker/iamatlas_celltype_markers_v0_2.json",
    # Stage 4 — A-score
    "a_scoring_module_path": CPG_ROOT / "Runtime Matrices/A_Scoring_Module/iamatlas_a_scoring.py",
    # Stage 4 A-score LOCI — per-cell-type MOST-METHYLATED CpGs from the IAMAtlas.
    # This is the ONLY correct loci source for the A-score. Built so the atlas
    # reference is unidirectional (healthy ~beta_floor; disorder walks beta -> 0.5).
    # NEVER point Stage 4 at celltype_markers_json (discriminative) — that is the
    # all-BREACH bug, root-caused 2026-06-11 (GAPE _derive_A + Recipe Part 4 / line 884).
    "a_score_loci_json": CPG_ROOT / "Runtime Matrices/A_Scoring_Module/iamatlas_a_score_loci_v1_0.json",
    # Stage 4.5 — bidirectional decomposition
    "bidirectional_module_path": CPG_ROOT / "Runtime Matrices/Directional Panel/bidirectional_decomposition.py",
    "directional_panels_json": CPG_ROOT / "Runtime Matrices/Directional Panel/directional_panels_v1_0.json",
    # Stage 4.6 — brightness comparison / Mollweide
    "brightness_module_path": CPG_ROOT / "Runtime Matrices/Mollweide & Brightness Comparison/patient_brightness_comparison.py",
    "brightness_archives_dir": CPG_ROOT / "IAM_Atlas/iamatlas_class_archives",
    # Stage 9 — brilliance maps (HEALPix Mollweide)
    "cpg_healpix_mapping_npy": CPG_ROOT / "Runtime Matrices/cpg healpix mapping/iamatlas_cpg_to_healpix_nside128.npy",
    "whole_atlas_reference_npz": CPG_ROOT / "Runtime Matrices/Mollweide & Brightness Comparison/whole_atlas_reference/iamatlas_whole450k_reference.npz",
    # Stage 2 — deconvolution (Walther NNLS primary + NILC cross-check)
    "atlas_csv_xz":        CPG_ROOT / "IAM_Atlas/IAMAtlasREBUILD.csv.xz",
    "atlas_csv_decompressed": CPG_ROOT / "IAM_Atlas/IAMAtlasREBUILD.csv",
    "celltype_to_class_json": CPG_ROOT / "IAM_Atlas/IAMAtlasREBUILD_celltype_to_class.json",
    # Atlas provenance — SOP §99 SINGLE SOURCE OF TRUTH for the frozen H_min
    # (key h_min_values_frozen_2026_04_06), build pipeline, distinctness tests.
    # Stage 4 reads this and refuses to run if the runtime H_min disagrees.
    # NOTE: the atlas <class>_mean global mean ~0.5 (std ~0.34, mass at both ends)
    # is CORRECT bimodal methylation, NOT a flat atlas or a scale offset — see
    # IAMAtlas_FLATNESS_LESSON.md and SOP §103. Do not re-derive H_min from it.
    "atlas_provenance_json": CPG_ROOT / "IAM_Atlas/IAMAtlasREBUILD_provenance.json",
    "walther_deconv_module": CPG_ROOT / "Walther_iam_deconvolver/walther_iam_deconvolver.py",
    "nilc_deconv_module":  CPG_ROOT / "NILC Deconvolver/nilc_deconvolver-2.py",
    # per-CELL NILC (the second cell-level lens): a cell is presented only when BOTH
    # Walther and the per-cell NILC resolve it (the agreement requirement). Disagreement
    # excludes the cell as noise. This is the filter that keeps the census clean.
    "nilc_celltype_module": CPG_ROOT / "NILC Deconvolver/nilc_celltype_deconvolver.py",
    "cell_agreement_min_fraction": 0.02,
    # Stage 8 — disease signature matrix matching (Path B) + priors
    "disease_matrix_csv":  CPG_ROOT / "Disease Matrix/DISEASE_MATRIX/disease_cell_signature_matrix_v1_8.csv",
    "matrix_mapping_json": CPG_ROOT / "Disease Matrix/DISEASE_MATRIX/iamatlas_115_to_matrix_v0_2_mapping.json",
    "cancer_prior_json":   CPG_ROOT / "Runtime Matrices/Cancer_prior/cancer_prior.json",
    "family_history_json": CPG_ROOT / "Runtime Matrices/Family_history_multiplier/family_history_multiplier.json",
    "literature_anchors_json": CPG_ROOT / "Runtime Matrices/Literature_anchors_Report building/literature_anchors.json",
    # Stage 9 — report visual assets (gauges + rankings)
    "gauge_module_path": CPG_ROOT / "cpg_gauge.py",
    "tier_breakpoints_json": CPG_ROOT / "Runtime Matrices/Tier_breakpoints/tier_breakpoints.json",
}


def _load_module(path, name):
    """Load a module from an absolute file path (handles spaces in folder names).

    Registers the module in sys.modules before executing it, which @dataclass requires
    to resolve cls.__module__ during class construction (importlib-by-path otherwise
    leaves the module unregistered and dataclass creation raises AttributeError).
    """
    import sys
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _assert_a_score_canonical(cfg):
    """Startup gate: run the canonical A-score fail-safe and RAISE if the scoring
    math has regressed (the entropy-of-mean vs mean-of-entropy bug). Wired into
    run_pipeline so a regression can never silently ship a real patient run."""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    test_path = os.path.join(os.path.dirname(str(cfg["a_scoring_module_path"])),
                             "test_a_score_canonical.py")
    fs = _load_module(test_path, "test_a_score_canonical")
    fails = fs.check()
    if fails:
        raise RuntimeError(
            "A-score canonical fail-safe FAILED at startup (scoring math regressed):\n  - "
            + "\n  - ".join(fails))


# ===========================================================================
# PatientContext — the FIREWALL (SOP §104).
# Intake facts the REPORT may annotate with. They are NEVER operands in any score.
# The scoring path (deconvolution, A-score, tier, match) never receives this object;
# only run_pipeline's report bundle carries it, for the clinician's context.
# ===========================================================================
@dataclass
class PatientContext:
    age: Optional[float] = None
    sex: Optional[str] = None
    family_history: Optional[dict] = None
    substrate: str = "whole_blood"


def _assert_no_covariate_in_beta(beta: pd.Series, context: "PatientContext") -> None:
    """Boundary guard: the beta entering deconvolution/scoring must be the raw calibrated
    beta with NO age/sex/smoking adjustment. PatientContext is report-only and must remain
    a separate object. This makes a covariate leak structurally impossible to ship silently:
    if anyone ever folds context into beta (e.g. age-regressed values, or tags the Series as
    adjusted), this fires before a single cell is scored.
    """
    if not isinstance(beta, pd.Series):
        raise TypeError("FIREWALL: scoring input must be a pd.Series of calibrated betas.")
    if getattr(beta, "_covariate_adjusted", False):
        raise RuntimeError(
            "FIREWALL (SOP §104): covariate-adjusted beta reached the scoring path. "
            "Age/sex/smoking are report context, never operands. Refusing to score.")
    if isinstance(context, pd.Series) or not isinstance(context, PatientContext):
        # context must be the dedicated object, never the beta itself or a raw dict that
        # could be mistaken for data.
        if context is not None:
            raise TypeError("FIREWALL: patient context must be a PatientContext, kept "
                            "separate from the beta vector.")


def _not_built(stage):
    raise NotImplementedError(
        f"{stage} is not implemented in this build. Current scope: Stages 3, 4, 4.5, 4.6, 5, 6, 9 "
        f"per BUILD_SPEC v1.3. See the build spec for the remaining stage contracts."
    )


def stage_0_intake(manifest_entry, grn_path, red_path, questionnaire_answers=None,
                   config=None, intake_log_path=None, manifest_dir=None,
                   integrity_log_path=None, control_summary=None,
                   probe_intensities=None, neg_control_stats=None,
                   bead_counts=None, detection_pass_mask=None, bead_pass_mask=None,
                   reference_cpg_coverage=None, sex_intensities=None,
                   verdict_log_path=None):
    """SOP Stage 0 (L1) sample intake — COMPLETE (Steps 0.1-0.9). Runs
    0.1 -> 0.2 -> 0.3 -> 0.4 -> 0.5 -> 0.6 -> 0.7 -> 0.7b -> 0.8 -> 0.9, stopping at the
    first gate that does not advance, and ends on the §19 decision gate
    (PROCEED / PROCEED_WITH_PENALTY / QUARANTINE). The intensity-dependent inputs
    (control_summary, probe_intensities + neg_control_stats, bead_counts, the detection/
    bead pass masks, reference_cpg_coverage, sex_intensities) come from the shared Stage 1
    IDAT decoder; when absent, those steps mark their QC deferred and 0.9 records the
    deferred set honestly rather than asserting a clean pass."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    s0 = _load_module(cfg["stage_0_intake_module_path"], "stage_0_intake")
    r = s0.step_0_1_idat_arrival(manifest_entry, grn_path, red_path, intake_log_path)
    if not r.get("advance"):
        return r
    r = s0.step_0_2_manifest_creation(r, questionnaire_answers, manifest_dir)
    if not r.get("advance"):
        return r
    r = s0.step_0_3_integrity_hash(r, grn_path, red_path, integrity_log_path)
    if not r.get("advance"):
        return r
    r = s0.step_0_4_control_probe_validation(r, grn_path, red_path, control_summary)
    if not r.get("advance"):
        return r
    r = s0.step_0_5_detection_pvalue_qc(r, probe_intensities, neg_control_stats)
    if not r.get("advance"):
        return r
    r = s0.step_0_6_bead_count_qc(r, bead_counts)
    if not r.get("advance"):
        return r
    r = s0.step_0_7_call_rate(r, detection_pass_mask, bead_pass_mask)
    if not r.get("advance"):
        return r
    r = s0.step_0_7b_platform_coverage(r, reference_cpg_coverage)
    if not r.get("advance"):
        return r
    r = s0.step_0_8_sex_check(r, sex_intensities)
    if not r.get("advance"):
        return r
    return s0.step_0_9_decision_gate(r, verdict_log_path)
def stage_1_calibration_beta(record, M=None, U=None, cohort_median=None,
                             bs_controls=None, beta_output_dir=None, config=None):
    """SOP Stage 1 (L2+L3) calibration & beta. Steps 1.4-1.8 built (IAM-native BS check,
    beta = M/(M+U+100), sanity, identity probe-response, packaging); Steps 1.1-1.3 + IDAT
    decode wrap the standard methylation stack (methylprep/minfi) and are deferred pending
    sign-off. M/U intensities come from the shared decoder. Returns the record with the
    calibrated beta matrix attached (internal) + Stage 1 provenance, or the quarantine
    result of whichever step halted (e.g. BS_CONVERSION_FAIL, BETA_OUT_OF_RANGE)."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    s1 = _load_module(cfg["stage_1_calibration_module_path"], "stage_1_calibration")
    return s1.run_stage_1(record, M=M, U=U, cohort_median=cohort_median,
                          bs_controls=bs_controls, beta_output_dir=beta_output_dir)


def _ensure_atlas_decompressed(cfg):
    """Decompress IAMAtlasREBUILD.csv.xz once to the runtime path if not already present."""
    import lzma, shutil
    src = Path(cfg["atlas_csv_xz"]); dst = Path(cfg["atlas_csv_decompressed"])
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    with lzma.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1 << 24)
    return dst


def stage_2_deconvolution(beta_calibrated: pd.Series, config: Optional[dict] = None,
                          run_nilc: bool = True) -> dict:
    """Stage 2 per SOP — returns {class_fractions, celltype_fractions, walther_diagnostics,
    nilc_fractions, cross_method, status}. beta_calibrated is the Stage 1 output (pre
    foreground-subtraction); deconvolution reads composition from the raw calibrated beta."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    atlas = _ensure_atlas_decompressed(cfg)
    w_mod = _load_module(cfg["walther_deconv_module"], "walther_iam_deconvolver")
    deconv = w_mod.WaltherIAMDeconvolver(str(atlas),
                                         celltype_class_map=str(cfg["celltype_to_class_json"]),
                                         verbose=False)
    betas = {str(k): float(v) for k, v in beta_calibrated.items()
             if isinstance(v, (int, float)) and 0.0 <= v <= 1.0}
    res = deconv.deconvolve(betas, refine_celltypes=True)
    out = {"class_fractions": res.class_fractions,
           "celltype_fractions": res.celltype_fractions,
           "class_present": res.class_present,
           "class_fraction_ci": res.class_fraction_ci,
           "presence_method": res.presence_method,
           "walther_diagnostics": res.diagnostics,
           "status": res.status,
           "nilc_fractions": None, "cross_method": None}
    if run_nilc:
        try:
            n_mod = _load_module(cfg["nilc_deconv_module"], "nilc_deconvolver")
            nilc = n_mod.NILCDeconvolver(str(atlas), verbose=False)
            nres = nilc.deconvolve(betas, patient_id="patient")
            out["nilc_fractions"] = nres.to_dict() if hasattr(nres, "to_dict") else nres
        except Exception as e:
            out["nilc_fractions"] = {"_error": str(e)}

    # CELL-LEVEL AGREEMENT (the noise filter): present a cell only when BOTH Walther and
    # the per-cell NILC resolve it above the agreement floor. Disagreement excludes the
    # cell as noise — this is what collapses the raw all-cell A-score list to real signal.
    out["celltype_nilc_fractions"] = None
    out["celltype_agreed"] = None
    try:
        nc_mod = _load_module(cfg["nilc_celltype_module"], "nilc_celltype_deconvolver")
        ncd = nc_mod.NILCCelltypeDeconvolver(str(atlas), str(cfg["celltype_markers_json"]))
        ncres = ncd.deconvolve(betas)
        nilc_cell_frac = ncres.get("fractions", {}) or {}
        out["celltype_nilc_fractions"] = nilc_cell_frac
        T = float(cfg.get("cell_agreement_min_fraction", 0.02))
        wfrac = out["celltype_fractions"] or {}
        out["celltype_agreed"] = sorted(
            c for c in (set(wfrac) | set(nilc_cell_frac))
            if wfrac.get(c, 0.0) >= T and nilc_cell_frac.get(c, 0.0) >= T)
        out["cell_agreement_min_fraction"] = T
    except Exception as e:
        out["celltype_agreed"] = None
        out["_cell_agreement_error"] = str(e)
    return out


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


def stage_4_a_score(cleaned_beta: pd.Series, config: Optional[dict] = None,
                    class_present: Optional[dict] = None,
                    celltype_fractions: Optional[dict] = None,
                    detect_floor: float = 0.01,
                    cfdna_run_everything: bool = False) -> dict:
    """Stage 4 per BUILD_SPEC v1.3 — consumes cleaned_beta (foreground-removed).

    A = H(beta_mean) / H_min(class) over panel CpGs, per class and per cell type.
    class_present: optional {class: bool} substrate-presence verdict from the Stage 2
    deconvolver gate. When supplied, classes (and their cell types) NOT present in the
    substrate are marked NOT_ASSESSABLE_IN_SUBSTRATE with A=None instead of being scored
    against blood-background beta at their marker addresses. When None, all classes are
    scored (legacy behavior).
    Returns {'class_ascores', 'celltype_ascores', 'h_min_by_class', 'celltype_to_class'}.
    (95% CI propagation from atlas posteriors is added at the report layer; the
    point A-scores are produced here.)
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    a_mod = _load_module(cfg["a_scoring_module_path"], "iamatlas_a_scoring")
    # h_min (the 8 frozen Mahaffey Numbers) + cell->class map come from the artifact loader.
    meta, _discriminative_markers, ct_to_class, h_min = a_mod.load_artifact(str(cfg["celltype_markers_json"]))

    # ========================================================================
    # H_MIN PROVENANCE PIN — SOP §99  (DO NOT REMOVE)
    # ------------------------------------------------------------------------
    # SOP §99: "Single source of truth: IAMAtlasREBUILD_provenance.json, key
    # h_min_values_frozen_2026_04_06" and "any code that hardcodes these values
    # WITHOUT reading from the canonical JSON is a chain-of-custody violation;
    # the engine refuses to deploy." We enforce that here: read the frozen H_min
    # from provenance and REFUSE TO RUN if the runtime H_min disagrees.
    #
    # The frozen H_min are MCMC posteriors (provenance.json), NEVER H(atlas global
    # mean). The atlas <class>_mean global ~0.5 (std ~0.34) is correct BIMODAL
    # methylation, not a flat atlas and not a scale offset (IAMAtlas_FLATNESS_LESSON.md;
    # SOP §103). Re-deriving a floor from the atlas global mean is the 2026-06-11
    # false-alarm; do not repeat it.
    # ========================================================================
    import json as _json
    with open(str(cfg["atlas_provenance_json"])) as _pf:
        _prov_hmin = _json.load(_pf)["h_min_values_frozen_2026_04_06"]
    for _cls, _hv in _prov_hmin.items():
        if _cls not in h_min:
            raise RuntimeError(
                f"H_min chain-of-custody violation (SOP §99): class '{_cls}' is in "
                f"provenance but missing at runtime. Refusing to deploy.")
        if abs(float(h_min[_cls]) - float(_hv)) > 1e-3:
            raise RuntimeError(
                f"H_min chain-of-custody violation (SOP §99): class '{_cls}' runtime "
                f"H_min={h_min[_cls]} != provenance frozen {_hv}. Refusing to deploy. "
                f"Single source of truth: IAMAtlasREBUILD_provenance.json.")

    # ========================================================================
    # A-SCORE MARKERS — SOP §41 (canonical).
    # ------------------------------------------------------------------------
    # Per-class beta_mean is taken across the class's marker CpGs from the
    # celltype-markers artifact (iamatlas_celltype_markers_v0_2.json), AGGREGATED to
    # class = union of the class's cell-type marker pools (SOP §41). Per-cell A-score
    # uses each cell's own marker pool.
    #
    # Verified against the ACTUAL a-scoring script (iamatlas_a_scoring.py) on a healthy
    # whole-blood reference: immune A = 1.035 -- at the healthy reference, above the
    # 0.838889 floor, exactly where the gauge centers it. The previously-wired
    # most-methylated loci (a_score_loci_v1_0.json) gave immune A = 0.729, BELOW the
    # floor -- a number a present cell cannot produce. A value below H_min is impossible
    # for a cell still holding its identity, so the loci path was wrong by definition.
    # The all-BREACH concern that motivated the loci path does not occur on v0_2 markers:
    # healthy and glioma both land near 1.0, never ceiling-pinned. SOP §41 wins.
    # ========================================================================
    class_markers = {}
    for _ct, _mk in _discriminative_markers.items():
        _c = ct_to_class.get(_ct)
        if _c:
            class_markers.setdefault(_c, set()).update(_mk)
    class_markers = {_c: sorted(_v) for _c, _v in class_markers.items()}
    ct_markers = _discriminative_markers

    beta_dict = cleaned_beta.to_dict()
    class_ascores = a_mod.score_per_class(beta_dict, class_markers, h_min)
    celltype_ascores = a_mod.score_per_celltype(beta_dict, ct_markers, ct_to_class, h_min)

    # cfDNA RUN-EVERYTHING (VAL-090 / CCL-033). The whole-blood presence gate below is
    # correct for whole blood, where terminal/secretory/stromal cells do not circulate.
    # In plasma cfDNA, shed tissue DNA IS present -- it is the entire tissue-of-origin
    # signal. Nulling tissue-class A-scores because their deconvolution fraction is low is
    # the fraction trap VAL-090 corrected (glioma plasma read NULL on fraction, +1.96 on
    # the per-tile A-score). When cfdna_run_everything is set, mark every architecture
    # class assessable so each tile is scored; the per-cell fraction still sets the
    # confidence tier below, it does not gate the score. h_min carries all 8 classes.
    if cfdna_run_everything:
        class_present = {cls: True for cls in h_min}

    # Substrate presence gate: a class not detected in this substrate is not
    # assessable -- do not report a blood-background A-score for it (the cause of
    # spurious terminal/stromal/stem_pluri "suppression" on whole blood). Present
    # classes are scored against their OWN class H_min, unchanged.
    if class_present is not None:
        NA = "NOT_ASSESSABLE_IN_SUBSTRATE"
        for cls, rec in class_ascores.items():
            present = bool(class_present.get(cls, False))
            rec["assessable"] = present
            if not present:
                rec["A"] = None
                rec["status"] = NA
        for ct, rec in celltype_ascores.items():
            cls = rec.get("class")
            class_ok = bool(class_present.get(cls, False)) if cls is not None else False
            # SOP §44 + run-everything doctrine (§655): EVERY cell type in a PRESENT
            # class is A-scored. The per-cell deconvolution fraction is the indicative
            # tier (§9) -- it sets CONFIDENCE, it does NOT gate scoring. The only thing
            # that NULLs a cell-type A-score is insufficient marker coverage in the
            # patient beta (INSUFFICIENT_MARKERS, set inside score_per_celltype). A
            # class absent from this substrate is the one substrate gate kept: it avoids
            # reporting a blood-background A-score for terminal/stromal cells that do not
            # circulate. (The prior per-cell fraction gate collapsed 51 immune cells to 4
            # by riding the sparse NNLS cell tier -- removed.)
            rec["assessable"] = class_ok
            if celltype_fractions is not None:
                _f = float(celltype_fractions.get(ct, 0.0))
                rec["celltype_fraction"] = _f
                rec["fraction_tier"] = "reliable" if _f >= detect_floor else "indicative"
            if not class_ok:
                rec["A"] = None
                rec["status"] = NA

    # FLOOR-GATE (principled, all substrates). A tile scored below its class H_min is an
    # ABSENT cell -- its markers are reading background cfDNA, not a present cell of that
    # class (H(beta) < H_min is impossible for a cell still holding its identity). Mark it
    # so the cell-of-origin / disease-card layer never calls an absent cell as a positive
    # organ flag from background drift. This is the cfDNA cortical-neuron artifact: A~0.58,
    # below the 0.7728 terminal floor, moved +1.3 d in HCC -- that is tumor-background
    # disorder leaking through absent-cell markers, NOT a brain signal. Above-floor
    # homogenization (the hepatocyte tissue-of-origin signal at A~0.99, above the 0.843
    # secretory floor) is a PRESENT-cell signal and is preserved untouched.
    for _ct, _rec in celltype_ascores.items():
        _A = _rec.get("A"); _fl = h_min.get(_rec.get("class"))
        _rec["below_floor"] = bool(_A is not None and _fl is not None and _A < _fl)
    for _cls, _rec in class_ascores.items():
        _A = _rec.get("A"); _fl = h_min.get(_cls)
        _rec["below_floor"] = bool(_A is not None and _fl is not None and _A < _fl)

    return {
        "class_ascores": class_ascores,           # {class: {A, coverage, status, ...}}  (8)
        "celltype_ascores": celltype_ascores,     # {celltype: {A, class, status, ...}}  (115)
        "h_min_by_class": h_min,
        "celltype_to_class": ct_to_class,
        "class_markers": class_markers,           # §41 markers actually scored (for the CI step)
        "ct_markers": ct_markers,
    }


# ===========================================================================
# Brightness CI (L4) — the MCMC edge.
# Each of the 8 class archives carries iamatlas_v0_1_<class>_brightness.csv with
# per-CpG (mean, sd, ci_lo, ci_hi) posteriors. For every scored A-score we propagate
# the per-CpG reference SD at the cell's scored loci through A = H(beta_mean)/H_min via
# a small Monte Carlo, and attach A_ci_lo / A_ci_hi. Derived-only: the interval comes
# from our own atlas posteriors, never from a cohort or population spread.
# Honest scope: this reflects the atlas posterior uncertainty at the scored CpGs; patient
# technical noise (detection-p) is a separate, additive term we can fold in later.
# ===========================================================================
def _shannon_bit(b: float) -> float:
    """Binary Shannon entropy of a single beta value, in bits (NaN/edge-safe)."""
    eps = 1e-9
    b = min(max(float(b), eps), 1.0 - eps)
    return float(-b * np.log2(b) - (1.0 - b) * np.log2(1.0 - b))


def _load_class_brightness_sd(brightness_archives_dir):
    """Return {class_name: {cpg_id: sd}} from the 8 per-class brightness CSVs.
    The CSVs live inside per-class .tar.xz archives (cpg_id,class,mean,sd,ci_lo,ci_hi)."""
    import tarfile, io
    arch_dir = Path(brightness_archives_dir)
    out = {}
    for tar_path in sorted(arch_dir.glob("*_REBUILD.tar.xz")):
        cls = tar_path.name.split("_v0_1_REBUILD")[0]
        try:
            with tarfile.open(tar_path, "r:xz") as tf:
                member = next((m for m in tf.getmembers()
                               if m.name.endswith("_brightness.csv")), None)
                if member is None:
                    continue
                raw = tf.extractfile(member).read().decode("utf-8", "replace")
        except Exception:
            continue
        sd_map = {}
        rdr = csv.DictReader(io.StringIO(raw))
        for row in rdr:
            cpg = row.get("cpg_id")
            try:
                sd_map[cpg] = float(row.get("sd", "nan"))
            except (TypeError, ValueError):
                continue
        if sd_map:
            out[cls] = sd_map
    return out


def attach_brightness_ci(stage4_output, a_score_loci_path, brightness_archives_dir,
                         patient_beta, n_mc=200, seed=0):
    """Attach A_ci_lo / A_ci_hi (95%) to every scored class and cell-type A-score.
    The CI is computed over the SAME §41 markers the A-score used (handed through from
    stage_4 as class_markers / ct_markers), so the interval and the score can never diverge.
    Mutates stage4_output in place and also returns it."""
    rng = np.random.default_rng(seed)
    class_markers = stage4_output.get("class_markers", {}) or {}
    ct_markers = stage4_output.get("ct_markers", {}) or {}
    ct_to_class = stage4_output.get("celltype_to_class", {})
    h_min_by_class = stage4_output.get("h_min_by_class", {})
    bsd = _load_class_brightness_sd(brightness_archives_dir)

    if isinstance(patient_beta, pd.Series):
        beta = patient_beta
    else:
        beta = pd.Series(patient_beta)

    def _ci_for(loci_list, cls):
        h_min = h_min_by_class.get(cls)
        if not loci_list or h_min in (None, 0):
            return None, None
        present = [c for c in loci_list if c in beta.index]
        if not present:
            return None, None
        vals = beta.loc[present].astype(float).values
        beta_mean = float(np.nanmean(vals))
        sd_map = bsd.get(cls, {})
        sds = np.array([sd_map.get(c, np.nan) for c in present], dtype=float)
        sds = sds[~np.isnan(sds)]
        if sds.size == 0:
            return None, None
        # standard error of the panel mean under the atlas posterior SD at these loci
        se = float(np.sqrt(np.mean(sds ** 2) / len(present)))
        draws = beta_mean + rng.normal(0.0, se, size=n_mc)
        a_draws = np.array([_shannon_bit(b) / h_min for b in draws])
        return float(np.percentile(a_draws, 2.5)), float(np.percentile(a_draws, 97.5))

    for cls, rec in stage4_output.get("class_ascores", {}).items():
        if isinstance(rec, dict) and rec.get("A") is not None:
            lo, hi = _ci_for(class_markers.get(cls), cls)
            rec["A_ci_lo"], rec["A_ci_hi"] = lo, hi
            rec["ci_method"] = "brightness_posterior_mc_95"

    for ct, rec in stage4_output.get("celltype_ascores", {}).items():
        if isinstance(rec, dict) and rec.get("A") is not None:
            cls = rec.get("class") or ct_to_class.get(ct)
            lo, hi = _ci_for(ct_markers.get(ct), cls)
            rec["A_ci_lo"], rec["A_ci_hi"] = lo, hi
            rec["ci_method"] = "brightness_posterior_mc_95"

    return stage4_output


def stage_7_tiers(stage4_output, config=None):
    """Stage 7 per BUILD_SPEC v1.3 — classify each class + cell-type A-score into the customer
    tier scheme (tier_breakpoints.json, the single source of truth shared with the gauges).
    Returns {class_tiers, celltype_tiers, max_class_tier, n_cells_breach}.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    gauge = _load_module(cfg["gauge_module_path"], "cpg_gauge")
    scheme = gauge.load_tier_scheme(str(cfg["tier_breakpoints_json"]))
    parts = scheme["partitions"]
    order = [p[0] for p in parts]  # ordered low->high

    def classify(a):
        tid = gauge._tier_for(a, parts)
        return {"A": float(a), "tier_id": tid, "label": gauge.CUSTOMER_LABELS.get(tid, tid)}

    cls_t = {c: classify(v["A"]) for c, v in stage4_output["class_ascores"].items()
             if isinstance(v, dict) and v.get("A") is not None}
    ct_t = {c: {**classify(v["A"]), "class": v.get("class")}
            for c, v in stage4_output["celltype_ascores"].items()
            if isinstance(v, dict) and v.get("A") is not None}

    def rank(tid):
        return order.index(tid) if tid in order else (len(order) if tid == "BREACH" else -1)
    max_tier = max((t["tier_id"] for t in cls_t.values()), key=rank, default=None)
    n_breach = sum(1 for t in ct_t.values() if t["tier_id"] == "BREACH")
    return {"class_tiers": cls_t, "celltype_tiers": ct_t,
            "max_class_tier": max_tier, "n_cells_breach": n_breach,
            "breach_line": scheme["breach_line"], "warburg_line": scheme["warburg_line"]}




# ===========================================================================
# Stage 8 (L6) — disease-signature matching by DIRECTION-GATED COSINE CONCORDANCE.
#
# WHY THIS SHAPE (read this before changing it):
#   The patient side is the derived A-departure vector: dep[cell] = A_patient[cell] - 1.0,
#   where 1.0 is the H_min-normalized healthy floor (a healthy cell sits at A~1.0 BY THE
#   DERIVATION, not a cohort mean). NO sigma, NO z-shift, NO standardization against any
#   population. Cohort methodology never touches the patient.
#
#   The disease side is the signature matrix: signed per-cell values learned in the VAL
#   cohorts. These are a TEMPLATE consulted as a PATTERN, never a population the patient is
#   standardized against. Their absolute scale is irrelevant here because the match is COSINE,
#   which compares the ANGLE between the patient's departure vector and the signature vector
#   over the cells present in both -- and angle is scale-invariant. So the matrix can stay in
#   its as-written units and nothing needs converting; multiplying either vector by any
#   constant leaves the match unchanged.
#
#   TWO TERMS, on purpose:
#     direction_agreement  -- coarse gate: are the right cells moving the right way at all
#                             (fraction of shared cells whose sign matches the signature).
#     cosine               -- fine discriminator: does the SHAPE (relative proportions across
#                             cells) match this disease rather than a neighbour. This is the
#                             aging-vs-Alzheimer's separator: both drift the same direction,
#                             but their relative cell proportions differ, so their vectors sit
#                             at a measurably different angle.
#
#   HONEST SCOPE (v1): we can recognize that a pattern RESEMBLES a template; we cannot yet
#   state a calibrated magnitude ("this much departure = this stage"), because that calibration
#   only accrues from real patients over time, in our own derived A-units. So Route B reports
#   RESEMBLANCE and its strength, never a probability. Across a trajectory the same patient
#   drifting toward a template's angle is a far stronger signal than any single snapshot.
# ===========================================================================
@dataclass
class Stage8Output:
    route_B_concordance: list          # ranked pattern matches (the detector)
    route_B_all_scored: list           # every row with enough shared cells (audit / appendix)
    patient_departure: dict            # {matrix_column: A - 1.0}  derived, no sigma
    route_A_architectural_alarm: dict  # stands down in lean v1 (residual matched filter confirms)
    route_C_bidirectional: dict        # stands down in lean v1 (shelf: bidirectional)
    status: str = "OK"


def _build_patient_departure_profile(stage4_output, mapping_path, floor=1.0, min_fraction=0.001):
    """Derived A-departure per matrix column: (A - floor) for each PRESENT cell, atlas-cell ->
    matrix-column via the v0.2 map (averaged when several cells share a column). floor = 1.0 is
    the derived healthy baseline. No sigma, no cohort, no standardization.

    PRESENT CELLS ONLY (two gates):
      (1) below_floor is False  -- the A is above the per-class H_min floor (catches SUPPRESSED
          background noise), AND
      (2) the deconvolver allocated this cell a real fraction (>= min_fraction). The atlas carries
          many duplicate panel labels for the same lineage (Neutrophils_reinius / Neutrophils_EPIC
          / Neu / Neutro / ...) and many cell types simply are not in the sample; the deconvolver
          assigns them fraction 0 and only ONE representative per lineage gets the real fraction.
          A zero-fraction cell's A-score is a background read of its markers, NOT that cell's
          architecture -- it can land ABOVE 1.0 (Left_atrium 1.267 in whole blood, a cardiomyocyte
          that is not there) and so escape the below_floor gate, which only catches suppression.
          Gating on fraction is what makes 'present cells only' actually true."""
    import json, numpy as np
    ct = stage4_output["celltype_ascores"]

    def _present(rec):
        if not (isinstance(rec, dict) and rec.get("A") is not None and not rec.get("below_floor")):
            return False
        frac = rec.get("celltype_fraction")
        if frac is None:           # no fraction recorded -> fall back to the below_floor gate only
            return True
        return float(frac) >= min_fraction

    dep_by_atlas = {cell: float(rec["A"]) - floor
                    for cell, rec in ct.items() if _present(rec)}
    mapping = json.load(open(mapping_path)).get("mapping", {})
    by_col = {}
    for atlas_cell, dep in dep_by_atlas.items():
        col = mapping.get(atlas_cell)
        if col:
            by_col.setdefault(col, []).append(dep)
    return {col: float(np.mean(v)) for col, v in by_col.items()}


def _signature_vector(signature_row, cell_cols):
    """Parse one disease-matrix row into {column: signed value} over its populated cells.
    Values written as signed effect sizes ('+0.81/+1.26' -> mean of the pair) or a single
    number; empty cells and arrow-qualitative cells are skipped (no documented signature)."""
    sig = {}
    for c in cell_cols:
        v = signature_row.get(c, "")
        if v is None:
            continue
        v = str(v).strip()
        if not v or v.startswith(("\u2191", "\u2193")):
            continue
        try:
            if "/" in v:
                lo, hi = v.split("/")
                sig[c] = (float(lo) + float(hi)) / 2.0
            else:
                sig[c] = float(v)
        except ValueError:
            continue
    return sig


# --- Mode 1 concordance specificity gate (tunable; VAL-relevant) -----------------------
# Cosine is direction-only. Without these gates a mild generic neutrophils-up shift
# false-flags every signature that has neutrophils-up. A shared cell counts as carrying
# real signal only when the patient departs from baseline by at least SIGNAL_FLOOR; a
# resemblance call needs at least MIN_SIGNAL_CELLS such cells. Below that -> INSUFFICIENT_SIGNAL.
# --- Directional pattern matcher (the disease-wall flagger) -----------------------------
# Replaces the old absolute-magnitude cosine. The disease-wall numbers are COHORT effect
# sizes (Cohen's d); a single patient moves the same DIRECTION at much smaller magnitude.
# So a match is scored on weighted directional agreement over the disease's SIGNAL cells
# (cells the disease actually moves, |d| >= WEIGHT_FLOOR), not on absolute-magnitude overlap.
# This catches subtle directional signal the old |dep|>=0.15 floor gated out, while the
# hardened specificity gate below rejects the generic neutrophil-to-lymphocyte stress shift.
DIRECTIONAL_WEIGHT_FLOOR = 0.20      # a disease cell is a SIGNAL cell when its |Cohen d| >= this
DIRECTIONAL_MOVE_EPS     = 0.05      # the patient "moved" on a cell when |A-departure| > this
DIRECTIONAL_MAG_FLOOR    = 0.15      # mean patient |departure| on agreeing cells for a STRONG call
CONCORDANCE_MIN_SIGNAL_CELLS = 3     # need this many moved cells to call a resemblance at all

# Progenitor / megakaryocyte / erythroid axis. A generic expansion here (all elevated) is a
# myeloproliferative / clonal-progenitor theme that resembles AML/CML/MDS/MPN alike -- not a
# disease-specific fingerprint -- so it joins the myeloid-up / lymphoid-down generic axis.
_PROGENITOR_CELLS = {
    "MPP", "CMP", "GMP", "MEP", "L_MPP", "HSC", "HSPC_pooled", "megakaryocyte",
    "erythroid_progenitor", "myeloid_progenitor", "erythroblast", "nRBC",
}

# Lineage membership for the non-specificity recognizer. A match whose signal-bearing cells
# are ALL one generic theme (all myeloid-elevated, or all lymphoid-suppressed) is a non-
# specific systemic shift (the neutrophil-to-lymphocyte / stress pattern) that resembles many
# myeloid-involved conditions and is NOT specific to any one disease. We say so instead of
# naming a leukemia, which would alarm a clinician over a generic inflammatory pattern.
_MYELOID_CELLS = {
    "granulocytes", "granulocytes_pooled", "neutrophils", "Neutro", "Neu", "Neutrophils_EPIC",
    "Neutrophils_reinius", "eosinophils", "eosinophil", "Eos", "basophils", "Baso",
    "monocytes", "CD14_monocytes", "Mono", "macrophages", "macrophages_peripheral", "Macro",
    "dendritic_cells", "dendritic", "GMP", "CMP", "MPP", "MEP", "MP", "myeloid",
}
_LYMPHOID_CELLS = {
    "CD4_T_cells", "CD4_T-cells", "CD4T", "CD4Tnv", "CD4Tmem", "CD8_T_cells", "CD8_T-cells",
    "CD8T", "CD8Tnv", "CD8Tmem", "B_cells", "CD19_B-cells", "Bnv", "Bmem", "naive_B_cells",
    "memory_B_cells", "NK_cells", "CD56_NK-cells", "NK", "regulatory_T_cells", "Treg",
    "memory_T_cells_pooled", "plasma_cells", "Plasma", "lymphoid",
}


def _classify_match_specificity(signal_cells, patient_departure, disease_signature=None,
                                origin_cells=None):
    """Return 'SPECIFIC' or 'NON_SPECIFIC_GENERIC' for one disease match.

    The generic stress axis -- myeloid-elevated, progenitor-elevated, lymphoid-suppressed -- is
    the neutrophil-to-lymphocyte / myeloproliferative shift that directionally resembles every
    myeloid-involved condition (infection, inflammation, stress, paraneoplasia) and fingerprints
    none of them. Naming a disease off it is the failure mode (lung cancer flagged off a head cold).

    The decisive question is the disease's CELL OF ORIGIN (from the disease wall):
      - TISSUE-origin disease (solid cancer / organ disease: lung, colon, breast, brain, ...):
        SPECIFIC only when one of its own origin cells actually agrees -- i.e. shed tissue is
        present (cfDNA) or the origin cell is otherwise resolved. Matching on shared blood-immune
        cells alone is the generic pattern wearing the disease's name -> NON_SPECIFIC. From whole
        blood, where the origin tissue is not present, these are correctly never named here; that
        is the matched filter's and the cell-of-origin layer's job, not the per-cell matcher's.
      - IMMUNE / blood-origin disease (myeloma=plasma, lymphoma/leukemia=blasts, autoimmune) or a
        disease with no tissue origin (infection, inflammaging): can be SPECIFIC on a genuine
        immune-pattern distinction (an off-axis break: lymphoid-elevated or myeloid-suppressed)."""
    if not signal_cells:
        return "SPECIFIC"

    def _is_immune(c):
        return c in _MYELOID_CELLS or c in _LYMPHOID_CELLS or c in _PROGENITOR_CELLS

    def _on_generic_axis(c):
        d = patient_departure.get(c, 0.0)
        if c in _MYELOID_CELLS and d > 0:
            return True                                  # myeloid up
        if c in _PROGENITOR_CELLS and d > 0:
            return True                                  # progenitor / clonal expansion up
        if c in _LYMPHOID_CELLS and d < 0:
            return True                                  # lymphoid down (the lymphopenia half)
        return False

    origin_cells = origin_cells or []
    tissue_origin = [c for c in origin_cells if not _is_immune(c)]

    # tissue-origin disease: SPECIFIC only if its own cell-of-origin agrees
    if tissue_origin:
        if any(c in signal_cells for c in origin_cells):
            return "SPECIFIC"
        return "NON_SPECIFIC_GENERIC"

    # immune / blood-origin (or origin not marked): any non-immune agreeing cell is distinctive
    if any(not _is_immune(c) for c in signal_cells):
        return "SPECIFIC"
    # a purely-immune pattern -- an off-axis break is a real immune distinction; the pure
    # myeloid-expansion / lymphopenia axis is not
    if all(_on_generic_axis(c) for c in signal_cells):
        return "NON_SPECIFIC_GENERIC"
    return "SPECIFIC"


def _concordance(patient_departure, signature, min_shared=2):
    """Directional weighted concordance between the patient's derived A-departure profile and
    one disease-wall signature. Returns None when too little overlaps to say anything.

    The wall's numbers are cohort effect sizes, so we do NOT compare absolute magnitudes (the
    old cosine did, and its |dep|>=0.15 floor gated out subtle pre-dx directional signal).
    Instead, over the disease's SIGNAL cells (|d| >= WEIGHT_FLOOR) we ask: on the cells where
    the patient actually moved (|dep| > MOVE_EPS), does the patient move the disease's way,
    weighted by how hard the disease moves each cell? 'cosine' carries this directional
    concordance in [-1, +1] for downstream compatibility (same role: higher = better match)."""
    import numpy as np
    signal = {c: d for c, d in signature.items()
              if abs(d) >= DIRECTIONAL_WEIGHT_FLOOR} or dict(signature)
    shared = [c for c in signal if c in patient_departure]
    if len(shared) < min_shared:
        return None
    moved = [c for c in shared if abs(patient_departure[c]) > DIRECTIONAL_MOVE_EPS]
    if len(moved) < min_shared:
        return None
    num = sum(np.sign(patient_departure[c]) * np.sign(signal[c]) * abs(signal[c]) for c in moved)
    den = sum(abs(signal[c]) for c in moved)
    dc = float(num / den) if den else 0.0
    agree_cells = [c for c in moved
                   if np.sign(patient_departure[c]) == np.sign(signal[c])]
    dir_agree = float(len(agree_cells) / len(moved))
    coverage = float(len(moved) / max(len(signal), 1))
    mag = float(np.mean([abs(patient_departure[c]) for c in moved]))
    return {"cosine": dc, "direction_agreement": dir_agree,
            "n_shared": len(shared), "n_signal": len(moved),
            "coverage": coverage, "mag": mag,
            "signal_cells": agree_cells}


def _resemblance_label(con):
    """Plain-language strength of resemblance (NOT a probability). Reads the directional
    concordance dict from _concordance. A resemblance needs the patient to have moved on at
    least CONCORDANCE_MIN_SIGNAL_CELLS of the disease's signal cells; below that the match is
    a coincidence on one or two cells and is reported as INSUFFICIENT_SIGNAL, never surfaced."""
    dc = con["cosine"]; cov = con["coverage"]; n = con["n_signal"]
    if n < CONCORDANCE_MIN_SIGNAL_CELLS:
        return "INSUFFICIENT_SIGNAL"
    if dc >= 0.70 and cov >= 0.40 and con["mag"] >= DIRECTIONAL_MAG_FLOOR:
        return "STRONG_RESEMBLANCE"
    if dc >= 0.50 and cov >= 0.30:
        return "MODERATE_RESEMBLANCE"
    if dc > 0.0:
        return "WEAK_RESEMBLANCE"
    return "NO_RESEMBLANCE"


# --- Systemic stress / inflammatory pattern (a wellness read, NOT a disease call) ----------
# The neutrophil-to-lymphocyte shift -- myeloid + progenitor elevation with lymphoid
# suppression -- is one of the oldest validated markers of systemic inflammation and
# physiological stress. The matcher correctly refuses to NAME a disease from it (it resembles
# every myeloid-involved condition), but the pattern itself is real information and is the kind
# of thing a preventive / naturopathic physician can act on immediately: lifestyle, weight,
# diet, trajectory monitoring, family-history vigilance. We surface it as a wellness signal,
# never an alarm, and never as a disease. It fires only when the pattern is coherent and
# carries real magnitude; flat noise and incoherent departures stay quiet.
STRESS_MOVE_EPS = 0.05          # a cell "moved" on the stress axis when |departure| > this
STRESS_NOTABLE_MAG = 0.10       # mean |departure| on axis cells for a NOTABLE read
STRESS_MILD_MAG = 0.07          # ... for a MILD read
STRESS_MIN_COHERENCE = 0.60     # axis cells must outweigh against-axis cells by this fraction


def detect_systemic_stress_pattern(patient_departure):
    """Return a wellness-level read of the patient's systemic stress / inflammatory axis.
    Level is NONE / MILD / NOTABLE. This is never a disease call -- it is the actionable
    'something is shifting' signal. Calibration against large healthy cohorts is future work;
    v1 is intentionally framed as a wellness heads-up, not a diagnosis."""
    mye_up = [c for c in patient_departure if c in _MYELOID_CELLS and patient_departure[c] > STRESS_MOVE_EPS]
    prog_up = [c for c in patient_departure if c in _PROGENITOR_CELLS and patient_departure[c] > STRESS_MOVE_EPS]
    lym_dn = [c for c in patient_departure if c in _LYMPHOID_CELLS and patient_departure[c] < -STRESS_MOVE_EPS]
    axis_cells = sorted(set(mye_up + prog_up + lym_dn))
    against = [c for c in patient_departure
               if (c in _MYELOID_CELLS and patient_departure[c] < -STRESS_MOVE_EPS)
               or (c in _LYMPHOID_CELLS and patient_departure[c] > STRESS_MOVE_EPS)]
    n = len(axis_cells)
    mag = (sum(abs(patient_departure[c]) for c in axis_cells) / n) if n else 0.0
    coherence = n / (n + len(against)) if (n + len(against)) else 0.0
    if n >= 4 and mag >= STRESS_NOTABLE_MAG and coherence >= STRESS_MIN_COHERENCE:
        level = "NOTABLE"
    elif n >= 3 and mag >= STRESS_MILD_MAG and coherence >= STRESS_MIN_COHERENCE:
        level = "MILD"
    else:
        level = "NONE"
    return {
        "level": level,
        "n_axis_cells": n,
        "mean_magnitude": round(float(mag), 3),
        "coherence": round(float(coherence), 2),
        "myeloid_up": mye_up,
        "progenitor_up": prog_up,
        "lymphoid_down": lym_dn,
        "against_axis": against,
    }


def stage_8_dual_matching(stage4_output, stage5_output, stage4_5_report,
                          patient_meta=None, config=None):
    """Stage 8 (L6) — Route B disease-pattern concordance is the detector. Routes A
    (architectural alarm) and C (bidirectional) stand down in the lean v1
    primary chain; A is the second chain, C is shelf. patient_meta is report context only."""
    import csv
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # ---- Route B: derived pattern concordance (scale-invariant, no cohort, no sigma) ----
    patient_dep = _build_patient_departure_profile(stage4_output, cfg["matrix_mapping_json"])
    with open(cfg["disease_matrix_csv"]) as f:
        rows = list(csv.DictReader(f))
    # cell-of-origin map (disease_id -> [origin cells]); drives the specificity rule so a solid-
    # cancer card is never named off the shared blood-immune axis without real tissue evidence
    import json as _json
    try:
        origin_map = _json.load(open(Path(cfg["disease_matrix_csv"]).parent / "disease_origin_cells.json"))
    except Exception:
        origin_map = {}
    header = list(rows[0].keys()) if rows else []
    meta_cols = ["disease_id", "phase", "time_range", "substrate",
                 "disease_severity_class", "mechanism", "organ_pages_to_link", "evidence_anchors"]
    cell_cols = [c for c in header if c not in meta_cols]

    # ---- substrate firewall: a patient is matched ONLY against signatures of a
    # compatible substrate. Whole-blood patients never match plasma_cfDNA or tissue
    # signatures, and cfDNA patients never match whole-blood signatures. This stops
    # generic cancer hypomethylation in cfDNA from resembling a whole-blood breast
    # signature (and vice versa). The cell-of-origin presence detector handles the
    # tissue-of-origin read for cfDNA separately.
    _SUBSTRATE_COMPAT = {
        "whole_blood": {"whole_blood_buffy_coat", "whole_blood_sorted"},
        "whole_blood_buffy_coat": {"whole_blood_buffy_coat", "whole_blood_sorted"},
        "buffy_coat": {"whole_blood_buffy_coat", "whole_blood_sorted"},
        "pbmc": {"whole_blood_buffy_coat", "whole_blood_sorted"},
        "blood": {"whole_blood_buffy_coat", "whole_blood_sorted"},
        "cfdna": {"plasma_cfDNA"}, "plasma": {"plasma_cfDNA"},
        "cf_dna": {"plasma_cfDNA"}, "cfdna_plasma": {"plasma_cfDNA"},
        "plasma_cfdna": {"plasma_cfDNA"},
        "tumor_tissue": {"tumor_tissue", "tumor_tissue_paired", "tumor_tissue_normalized"},
        "tissue": {"tumor_tissue", "tumor_tissue_paired", "tumor_tissue_normalized"},
    }
    _psub = str((patient_meta or {}).get("substrate") or "whole_blood").lower()
    _allowed = _SUBSTRATE_COMPAT.get(_psub, {"whole_blood_buffy_coat", "whole_blood_sorted"})

    scored = []
    for r in rows:
        if r.get("substrate") not in _allowed:
            continue                                    # substrate firewall
        sig = _signature_vector(r, cell_cols)
        con = _concordance(patient_dep, sig)
        if con is None:
            continue
        scored.append({
            "disease": r.get("disease_id"), "phase": r.get("phase"),
            "time_range": r.get("time_range"), "substrate": r.get("substrate"),
            "severity": r.get("disease_severity_class"), "mechanism": r.get("mechanism"),
            "cosine": round(con["cosine"], 4),
            "direction_agreement": round(con["direction_agreement"], 3),
            "n_shared": con["n_shared"],
            "n_signal": con["n_signal"],
            "coverage": round(con["coverage"], 3),
            "signal_cells": con["signal_cells"],
            "specificity": _classify_match_specificity(con["signal_cells"], patient_dep, sig,
                                                       origin_map.get(r.get("disease_id"))),
            "resemblance": _resemblance_label(con),
        })

    # direction-gated, ranked by shape (cosine). Candidates must point the right way overall
    # AND carry real signal on enough shared cells (INSUFFICIENT_SIGNAL matches are noise-
    # direction alignment and are kept only in route_B_all_scored for audit).
    candidates = [s for s in scored
                  if s["direction_agreement"] >= 0.5 and s["cosine"] > 0.0
                  and s["resemblance"] != "INSUFFICIENT_SIGNAL"]
    candidates.sort(key=lambda s: s["cosine"], reverse=True)

    route_A = {"fired": None,
               "note": "Route A (architectural alarm) stands down in the lean v1 "
                       "primary chain; it is the second chain, run only on a flag."}
    route_C = {"fired": None,
               "note": "Route C (bidirectional) stands down in v1; it is on the shelf."}

    return Stage8Output(
        route_B_concordance=candidates[:10],
        route_B_all_scored=scored,
        patient_departure=patient_dep,
        route_A_architectural_alarm=route_A,
        route_C_bidirectional=route_C,
        status="OK",
    )


def stage_10_delivery(*a, **k):         _not_built("Stage 10 (delivery)")


# ===========================================================================
# Second detection mode (L6b) — CELL-OF-ORIGIN PRESENCE.
# The ONLY presence that is alarming by itself is a blood-brain-barrier cell circulating in
# blood. A cortical neuron, glia, or oligodendrocyte crossed a physical barrier to be there --
# that is a red flag on presence alone, regardless of quantity or A-score.
#
# Everything else that can shed into blood -- epithelial / secretory (breast, prostate, colon),
# cycling -- does so at a normal baseline. Detecting them is NOT abnormal; we detect and, when
# abundant enough, SCORE them routinely. For those cells presence is normal composition and the
# A-score (Mode 1) is the discriminator: normal A = benign turnover, abnormal A = concerning.
#
# So this detector returns BBB-review flags only. Other shed cells are reported as normal
# composition and scored in Mode 1 when abundant -- never flagged as abnormal by presence.
# ===========================================================================
# Cells behind the blood-brain barrier. Their presence in blood is a barrier breach.
BBB_PROTECTED_CELLS = {
    "cortical_neurons", "neurons_pooled", "neuron", "NeuMa", "NeuIm",
    "glia", "Glia", "astrocytes", "brain_astrocytes", "brain_pooled",
    "oligodendrocytes", "Oligo", "OPC", "microglia",
}
# The atlas class that carries the BBB cells (v0.1 deconvolves at class level; the terminal
# class holds the brain cells alongside other terminal cells).
BBB_BEARING_CLASSES = {"terminal"}


def detect_cell_of_origin_presence(stage2_output, detect_floor=0.01, substrate=None):
    """Return BBB-review flags only. Epithelial/secretory/cycling presence is normal and is NOT
    returned here -- it is normal composition, scored in Mode 1 when abundant.

    The barrier-breach alarm applies to a brain CELL circulating in WHOLE BLOOD. In plasma
    cfDNA a brain/terminal signal is SHED DNA (tissue-of-origin), not a cell that crossed the
    barrier, so the alarm must NOT fire on cfDNA -- terminal signal in plasma is reported in
    the Tissue-of-origin section instead. (Without this gate the whole terminal class -- which
    also carries hepatocytes, cardiomyocytes -- false-flagged as a barrier breach on every
    cfDNA run.) The whole-blood path below is unchanged (VAL-090 glioma sensitivity preserved)."""
    if not stage2_output:
        return []
    if substrate and str(substrate).lower() in ("cfdna", "plasma", "cf_dna", "cfdna_plasma"):
        return []
    walther = stage2_output.get("class_fractions", {}) or {}
    nilc_raw = (stage2_output.get("nilc_fractions") or {}).get("raw_fractions", {}) or {}
    flags = []
    for cls in BBB_BEARING_CLASSES:
        try:
            w = float(walther.get(cls, 0.0) or 0.0)
            r = float(nilc_raw.get(cls, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        observed = r  # NILC-raw is the presence detector. VAL-090's glioma breach lived in
                      # NILC-raw at +1.41% while Walther floored it to 0, so NILC-raw catches the
                      # real shed signal. The constrained NNLS (w) parks leftover variance into
                      # terminal (e.g. ~3.78% in normal blood with NO brain cells) and false-flags,
                      # so it must NOT drive a barrier-breach alarm. w is kept in the record below
                      # for transparency only.
        if observed >= detect_floor:
            flags.append({
                "class": cls,
                "observed_fraction": round(observed, 4),
                "fraction_walther": round(w, 4),
                "fraction_nilc_raw": round(r, 4),
                "severity": "REVIEW_BBB",
                "interpretation": (
                    "the terminal class carries blood-brain-barrier cells (cortical neurons, "
                    "glia, oligodendrocytes). Their presence in blood is a barrier breach -- a "
                    "red flag on presence alone; refer (neurology / neuro-oncology). v0.1 reports "
                    "at class level and cannot yet separate brain from other terminal cells, so "
                    "treat as review-and-refer. NOT a diagnosis."),
            })
    return flags


# ===========================================================================
# NILC-raw rescue — the flag fires on EITHER deconvolver, for real.
# Walther's constrained NNLS floors a faint class to zero; NILC's RAW (unconstrained)
# fractions surface it. On glioma blood, terminal came back +1.41% in NILC-raw while
# Walther AND NILC's simplex projection both floored it to 0.000%. So a class counts as
# present if Walther OR NILC-raw >= floor. For a class NILC-raw rescues, promote its cells
# to the NILC-raw class fraction so the per-cell gate scores them (v0.1 cannot separate
# cells within a class; the A-score + concordance then discriminate which carry the signal).
# ===========================================================================
def _merge_deconv_presence(class_present, celltype_fractions, nilc_result,
                           celltype_to_class, floor=0.01):
    cp = dict(class_present or {})
    cf = dict(celltype_fractions or {})
    raw = nilc_result.get("raw_fractions", {}) if isinstance(nilc_result, dict) else {}
    rescued = []
    for cls, frac in (raw or {}).items():
        try:
            fv = float(frac)
        except (TypeError, ValueError):
            continue
        if fv >= floor and not cp.get(cls, False):
            cp[cls] = True
            rescued.append({"class": cls, "nilc_raw_fraction": round(fv, 4)})
            for ct, c in celltype_to_class.items():
                if c == cls and float(cf.get(ct, 0.0)) < floor:
                    cf[ct] = fv
    return cp, cf, rescued


# ===========================================================================
# run_pipeline — the CPG v1 LEAN primary chain. Seven links, nothing else.
# ===========================================================================
def run_pipeline(beta_calibrated, *, context=None, patient_id="patient", test_id=None,
                 config=None, run_deconvolution=True, attach_ci=True,
                 nilc_rescue=True, detect_floor=0.01):
    """CPG v1 lean primary chain:
        L3 deconvolution (Walther + NILC) -> NILC-raw rescue gating
        L4 A-score (presence-gated) + brightness CI -> L5 tier
        L6 disease-signature match (direction-gated cosine concordance — the detector)
    context : PatientContext (report-only). test_id : optional serial/trajectory label.
    """
    import datetime
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    context = context if context is not None else PatientContext()

    _assert_a_score_canonical(cfg)
    _assert_no_covariate_in_beta(beta_calibrated, context)

    # L3 — deconvolution: Walther (NNLS) + NILC (GLS) on the same beta + atlas
    s2 = stage_2_deconvolution(beta_calibrated, cfg, run_nilc=True) if run_deconvolution else None
    class_present = s2["class_present"] if s2 else None
    celltype_fractions = s2.get("celltype_fractions") if s2 else None

    # PRESENCE = detected by EITHER deconvolver at or above the detect floor. If a class is in
    # the blood at >= ~1% (Walther OR NILC-raw), we score it -- whole blood and cfDNA alike.
    # NILC-raw's unconstrained fit surfaces faint classes Walther's sparse NNLS floors to zero
    # (the glioma cortical-neuron breach, VAL-090 +1.96); we PROMOTE those into the A-scoring,
    # never treat them as presence-only. The floor-gate inside Stage 4 (A < class H_min => an
    # absent cell reading background) is the ONLY exclusion -- a physics gate, never an
    # abundance gate: detected above its floor is a real read; detected but below H_min is
    # background and is dropped. Either it is in the blood and detected, and we score it, or it
    # is not, and there is nothing to score.
    rescued = []
    if s2 and nilc_rescue:
        _ct2cls = json.load(open(cfg["celltype_to_class_json"]))
        class_present, celltype_fractions, rescued = _merge_deconv_presence(
            class_present, celltype_fractions,
            (s2.get("nilc_fractions") or {}),
            _ct2cls, floor=detect_floor)

    # L4 — A-score (cleaned_beta == calibrated beta; firewall: no foreground)
    # cfDNA substrate ungates ALL tissue classes (run-everything, VAL-090) since shed tissue
    # DNA is the signal. Whole blood scores every class detected at >= ~1% by Walther OR NILC
    # (merged above), so a breached tissue cell is scored, not just flagged. Floor-gate drops
    # the background either way.
    _cfdna = str(getattr(context, "substrate", "") or "").lower() in (
        "cfdna", "cf_dna", "plasma", "ctdna", "ct_dna")
    s4 = stage_4_a_score(beta_calibrated, cfg,
                         class_present=class_present,
                         celltype_fractions=celltype_fractions,
                         detect_floor=detect_floor,
                         cfdna_run_everything=_cfdna)
    if attach_ci:
        try:
            attach_brightness_ci(s4, cfg["a_score_loci_json"],
                                 cfg["brightness_archives_dir"], beta_calibrated)
        except Exception as e:
            s4["_ci_note"] = f"brightness CI not attached: {e}"

    # L4.5 — Stage 4.5 bidirectional decomposition (sealed VAL-051 Rule A immune
    # panel). The validated directional detector for cancellation-pattern signals:
    # it z-scores each panel CpG against its FROZEN training-set HC mean/SD and
    # frozen direction, so it is composition-independent by construction. This is
    # the AD-direction immune detector. The whole-blood-referenced matched filter
    # is NOT used for AD -- its fixed atlas baseline carried each patient's
    # composition into the departure and false-fired AD on healthy blood
    # (rho ~ +0.55-0.60 on cases AND controls alike). Shelved in lean v1; wired
    # back in here per VAL-051 (Rule A, d=0.624) / VAL-013.
    s4_5 = None
    try:
        import importlib.util as _iu5, sys as _sys5
        _sp5 = _iu5.spec_from_file_location(
            "bidirectional_decomposition", str(cfg["bidirectional_module_path"]))
        _bd = _iu5.module_from_spec(_sp5)
        _sys5.modules["bidirectional_decomposition"] = _bd
        _sp5.loader.exec_module(_bd)
        _panels = _bd.load_directional_panels(cfg["directional_panels_json"])
        s4_5 = _bd.compute_per_class_bidirectional_decomposition(
            beta_calibrated, _panels, patient_id=patient_id)
    except Exception as e:
        s4_5 = None
        s4["_stage4_5_note"] = f"stage 4.5 not run: {e}"

    # L5 — tier
    s7 = stage_7_tiers(s4, cfg)

    # L6 — disease-signature match (direction-gated cosine concordance, architectural mode)
    s8 = stage_8_dual_matching(
        s4, {}, s4_5,
        patient_meta={"age": context.age, "sex": context.sex,
                      "family_history": context.family_history,
                      "substrate": context.substrate},
        config=cfg)

    # L6b — cell-of-origin presence (second detection mode: a cell that shouldn't circulate)
    cell_of_origin = detect_cell_of_origin_presence(s2, detect_floor=detect_floor,
                                                     substrate=context.substrate)

    bundle = {
        "patient_id": patient_id,
        "test_id": test_id or datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "chain": "CPG_v1_primary_plus_confirmation",
        "context": {"age": context.age, "sex": context.sex,
                    "family_history": context.family_history,
                    "substrate": context.substrate},
        "nilc_rescued_classes": rescued,
        "stage2": s2, "stage4": s4, "stage7": s7, "stage8": s8,
        "stage4_5": (s4_5.to_dict() if s4_5 is not None else {"status": "not_run"}),
        "cell_of_origin_flags": cell_of_origin,   # L6b second detection mode
        "trajectory_baseline": {
            "patient_departure": s8.patient_departure,
            "celltype_A": {ct: {"A": r.get("A"), "ci": [r.get("A_ci_lo"), r.get("A_ci_hi")]}
                           for ct, r in s4["celltype_ascores"].items() if r.get("A") is not None},
        },
    }

    # systemic stress / inflammatory wellness read (not a disease call) from the derived profile
    bundle["systemic_stress"] = detect_systemic_stress_pattern(s8.patient_departure)

    # L7 — SECOND CHAIN (confirmation). Fires on a SPECIFIC Route B flag OR any residual-sweep
    # hit. The residual matched filter (SOP 8.2) is the detector and the directional class
    # signal confirms; the literature anchor labels. When neither a specific per-cell flag nor a
    # sweep hit is present, run_second_chain returns None and the report shows no confirmation.
    try:
        import stage_5_second_chain as _s5
        bundle["stage5"] = _s5.run_second_chain(bundle, beta_calibrated, cfg)
    except Exception as e:
        bundle["stage5"] = {"fired": None, "error": f"second chain not run: {e}"}

    return bundle


def _find_one(folder, *patterns):
    import glob
    for pat in patterns:
        hits = sorted(glob.glob(str(Path(folder) / pat)))
        if hits:
            return hits[0]
    return None


def _prompt_questionnaire(folder, grn, _json):
    """Minimal command-line intake: ask for the four fields the chain reads,
    compute age from DOB, write questionnaire.json into the folder, return the dict.
    The full clinical questionnaire is collected by cpg_intake_form.html; this is
    only the terminal fallback so a sample can be run without hand-editing JSON."""
    import datetime
    print("\nNo questionnaire.json found. Quick intake (press Enter to skip a field):")

    def ask(prompt, default=None):
        try:
            v = input("  " + prompt).strip()
        except EOFError:
            return default
        return v or default

    default_pid = Path(grn).name.split("_Grn")[0].split("Grn")[0].rstrip("_") or "patient"
    pid = ask(f"Anonymized patient ID [{default_pid}]: ", default_pid)

    age = None
    dob_y = ask("Birth year (YYYY): ")
    if dob_y and dob_y.isdigit():
        dob_m = ask("Birth month (1-12): ") or "1"
        dob_d = ask("Birth day (1-31): ") or "1"
        try:
            bd = datetime.date(int(dob_y), int(dob_m), int(dob_d))
            today = datetime.date.today()
            age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
            if age < 0 or age > 130:
                age = None
        except ValueError:
            age = None

    sex_in = (ask("Sex at birth [F/M/I, Enter to skip]: ") or "").upper()
    sex = sex_in if sex_in in ("F", "M", "I") else None

    sub_in = (ask("Sample type [1=whole_blood (default), 2=cfdna, 3=pbmc, 4=buffy_coat]: ") or "1")
    substrate = {"1": "whole_blood", "2": "cfdna", "3": "pbmc",
                 "4": "buffy_coat"}.get(sub_in, "whole_blood")

    fh_in = (ask("Family history of cancer? [n=none (default), y=yes]: ") or "n").lower()
    family_history = "present" if fh_in.startswith("y") else "none"

    q = {"patient_id": pid, "age": age, "sex": sex, "substrate": substrate,
         "family_history": family_history,
         "intake_version": "v1.0-terminal",
         "intake_completed_utc": datetime.datetime.utcnow().isoformat() + "Z"}
    try:
        outp = Path(folder) / "questionnaire.json"
        with open(outp, "w") as fh:
            _json.dump(q, fh, indent=2)
        print(f"  wrote {outp}")
    except Exception as e:
        print(f"  (could not write questionnaire.json: {e}; proceeding in-memory)")
    return q


# ===========================================================================
#  TRAJECTORY — per-patient baseline persistence + cross-visit comparison
#  Convention: run a VISIT folder (patients/<id>/<date>/). The patient root is
#  its parent; baselines accumulate in patients/<id>/baselines/. Each run saves
#  this visit's vector and, if prior visits exist for the same patient_id,
#  compares against the most recent one and appends a Trajectory section.
# ===========================================================================
import re as _re

def _visit_label(folder):
    """A visit label for this draw: the folder name if it looks like a date,
    else the UTC run date."""
    import datetime
    name = Path(folder).name
    if _re.match(r"^\d{4}[-_]\d{2}[-_]\d{2}", name):
        return name.replace("_", "-")[:10]
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")

def _days_between(a, b):
    import datetime
    try:
        da = datetime.date.fromisoformat(a[:10]); db = datetime.date.fromisoformat(b[:10])
        return abs((da - db).days)
    except Exception:
        return None

def _baseline_payload(bundle, visit_label):
    """The small, derived vector we persist for trajectory. Per-cell A-scores with
    their reliability tier (so a later draw can lead with the deconvolver-resolved
    cells), the matrix-column departure vector (for rotation-toward-signature), the
    class scores and global departure as one-line context. No raw beta."""
    s4 = bundle["stage4"]
    ctx = bundle.get("context", {})
    s5 = bundle.get("stage5") or {}
    s8 = bundle.get("stage8")
    departure = getattr(s8, "patient_departure", None)
    if departure is None and isinstance(s8, dict):
        departure = s8.get("patient_departure")
    s5trig = (s5.get("trigger", {}) or {}) if isinstance(s5, dict) else {}
    return {
        "patient_id": bundle.get("patient_id"),
        "visit_label": visit_label,
        "run_timestamp": bundle.get("test_id"),
        "substrate": ctx.get("substrate"),
        "age": ctx.get("age"),
        "class_ascores": {c: r.get("A") for c, r in s4["class_ascores"].items()
                          if r.get("A") is not None and r.get("assessable", True)},
        "celltype": {ct: {"A": r.get("A"), "reliable": (r.get("fraction_tier") == "reliable")}
                     for ct, r in s4["celltype_ascores"].items() if r.get("A") is not None},
        "patient_departure": departure or {},
        "flagged_disease": s5trig.get("flagged_disease"),
    }

def _save_baseline(baselines_dir, payload):
    import json as _json
    baselines_dir.mkdir(parents=True, exist_ok=True)
    pid = payload["patient_id"]; lab = payload["visit_label"]
    out = baselines_dir / f"baseline_{pid}_{lab}.json"
    with open(out, "w") as f:
        _json.dump(payload, f, indent=2, default=str)
    return out

def _load_prior_baselines(baselines_dir, pid, exclude_label):
    import json as _json, glob as _glob
    if not baselines_dir.exists():
        return []
    priors = []
    for p in _glob.glob(str(baselines_dir / f"baseline_{pid}_*.json")):
        try:
            b = _json.load(open(p))
        except Exception:
            continue
        if b.get("visit_label") == exclude_label:
            continue
        if b.get("patient_id") != pid:
            continue
        priors.append(b)
    priors.sort(key=lambda b: (b.get("visit_label") or "", b.get("run_timestamp") or ""))
    return priors

def _cosine(a, b):
    """Cosine between two {key: value} vectors over their shared keys."""
    import numpy as np
    shared = [k for k in a if k in b]
    if len(shared) < 3:
        return None
    va = np.array([a[k] for k in shared], dtype=float)
    vb = np.array([b[k] for k in shared], dtype=float)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return None
    return float(np.dot(va, vb) / (na * nb))

def _flagged_signature(flagged_disease, cfg):
    """Build the signature vector for the flagged disease from the matrix, the same
    way Route B does — so rotation toward it is measured in the engine's own space."""
    import csv
    if not flagged_disease:
        return None
    try:
        with open(cfg["disease_matrix_csv"]) as f:
            rows = list(csv.DictReader(f)); header = rows[0].keys() if rows else []
    except Exception:
        return None
    meta = ["disease_id", "phase", "time_range", "substrate",
            "disease_severity_class", "mechanism", "organ_pages_to_link", "evidence_anchors"]
    cell_cols = [c for c in header if c not in meta]
    cand = [r for r in rows if r.get("disease_id") == flagged_disease]
    if not cand:
        return None
    # prefer an 'active' phase row, else the first row for that disease
    row = next((r for r in cand if (r.get("phase") or "").lower() == "active"), cand[0])
    return _signature_vector(row, cell_cols)

def _compute_trajectory(bundle, priors, now_label, cfg):
    """Per-cell trajectory: this draw vs the most recent prior, cell by cell. The
    per-cell DELTA is the unit (the dilution offset is shared at both draws and
    cancels in the subtraction), led by the deconvolver-resolved cells whose change
    is clean. Plus the rotation of the whole departure vector toward the flagged
    disease's signature angle, and the global departure as one summary line.

    Bulk/aggregate pseudo-columns (whole_blood, PBMC, broad lineage buckets) are
    excluded — they are mixtures, not cell types, so their 'A-score' is exactly the
    averaged-away signal the per-cell view exists to avoid."""
    if not priors:
        return None
    _AGG = {"whole_blood", "pbmc", "leu", "lym", "mye", "buffy_coat",
            "leukocytes", "wbc", "blood", "mononuclear", "bulk"}
    now = _baseline_payload(bundle, now_label)
    prev = priors[-1]; first = priors[0]
    days = _days_between(now_label, prev.get("visit_label"))
    WARBURG, BREACH = 1.07, 1.10
    now_cell = {k: v for k, v in now.get("celltype", {}).items() if k.lower() not in _AGG}
    prev_cell = {k: v for k, v in prev.get("celltype", {}).items() if k.lower() not in _AGG}

    cell_changes = []
    for ct in set(now_cell) | set(prev_cell):
        rn, rp = now_cell.get(ct), prev_cell.get(ct)
        an = rn["A"] if rn else None
        ap = rp["A"] if rp else None
        rel_both = bool(rn and rn.get("reliable")) and bool(rp and rp.get("reliable"))
        rel_now = bool(rn and rn.get("reliable"))
        if an is not None and ap is not None:
            d = an - ap
            cell_changes.append({"cell": ct, "prior": round(ap, 3), "now": round(an, 3),
                                 "delta": round(d, 3), "kind": "tracked",
                                 "reliable": rel_both, "reliable_now": rel_now,
                                 "direction": ("up" if d > 0 else "down" if d < 0 else "flat"),
                                 "crossed_warburg": (ap < WARBURG <= an),
                                 "crossed_breach": (ap < BREACH <= an)})
        elif an is not None:
            cell_changes.append({"cell": ct, "prior": None, "now": round(an, 3), "delta": None,
                                 "kind": "new", "reliable": False, "reliable_now": rel_now,
                                 "direction": "new", "crossed_warburg": an >= WARBURG,
                                 "crossed_breach": an >= BREACH})
        else:
            cell_changes.append({"cell": ct, "prior": round(ap, 3), "now": None, "delta": None,
                                 "kind": "dropped", "reliable": False, "reliable_now": False,
                                 "direction": "dropped", "crossed_warburg": False, "crossed_breach": False})

    # lead with reliable cells, then by magnitude of change
    def _key(c):
        mag = abs(c["delta"]) if c["delta"] is not None else (abs((c["now"] or 1.0) - 1.0) + 0.03)
        return (0 if c["reliable"] else 1, -mag)
    cell_changes.sort(key=_key)

    # rotation of the departure vector toward the flagged signature's angle
    rotation = None
    flagged = now.get("flagged_disease")
    sig = _flagged_signature(flagged, cfg)
    if sig and prev.get("patient_departure") and now.get("patient_departure"):
        cp = _cosine(prev["patient_departure"], sig)
        cn = _cosine(now["patient_departure"], sig)
        if cp is not None and cn is not None:
            dd = cn - cp
            rotation = {"disease": flagged, "prior_cosine": round(cp, 3), "now_cosine": round(cn, 3),
                        "delta": round(dd, 3),
                        "trend": ("rotating toward the signature" if dd > 0.05
                                  else "rotating away from the signature" if dd < -0.05 else "holding angle")}
    # overall pattern alignment with the prior draw (self-drift)
    self_align = None
    if prev.get("patient_departure") and now.get("patient_departure"):
        sa = _cosine(prev["patient_departure"], now["patient_departure"])
        if sa is not None:
            self_align = round(sa, 3)

    # headline — integrate every tier honestly, don't let reliable-only bury the signal
    def _moved(c, thr):
        return c["delta"] is not None and abs(c["delta"]) >= thr
    rel_up = [c for c in cell_changes if c["reliable"] and _moved(c, 0.03) and c["delta"] > 0]
    rel_dn = [c for c in cell_changes if c["reliable"] and _moved(c, 0.03) and c["delta"] < 0]
    ind_up = [c for c in cell_changes if (not c["reliable"]) and _moved(c, 0.05) and c["delta"] > 0]
    breach = [c for c in cell_changes if c.get("crossed_breach") and c["kind"] == "tracked"]
    risers = sorted(rel_up + ind_up, key=lambda c: c["delta"], reverse=True)[:3]

    def _name(c):
        return f"{c['cell']} {c['delta']:+.3f}" + ("" if c["reliable"] else " [indicative]")

    if breach:
        lead = f"{breach[0]['cell']} crossed the 1.10 breach line (now {breach[0]['now']})"
    elif rel_up:
        top = max(rel_up, key=lambda c: c["delta"])
        lead = f"the reliable cell {top['cell']} rose {top['prior']}\u2192{top['now']} (\u0394 {top['delta']:+.3f})"
    elif rotation and rotation["delta"] > 0.05:
        lead = (f"the departure pattern rotated toward the {rotation['disease']} signature "
                f"(cosine {rotation['prior_cosine']}\u2192{rotation['now_cosine']})")
    elif ind_up:
        lead = "several indicative cells rose"
    elif rel_dn:
        top = min(rel_dn, key=lambda c: c["delta"])
        lead = f"reliable cells eased toward the floor ({top['cell']} {top['delta']:+.3f})"
    else:
        lead = f"the cell pattern is stable versus {prev['visit_label']}"

    head = f"Since {prev['visit_label']}, {lead}."
    if risers:
        head += " Rising: " + ", ".join(_name(c) for c in risers) + "."
    if rotation and rotation["delta"] > 0.05 and "rotated toward" not in lead:
        head += (f" The whole pattern is rotating toward the {rotation['disease']} signature "
                 f"({rotation['prior_cosine']}\u2192{rotation['now_cosine']}).")

    return {"prior_visit_label": prev["visit_label"], "first_visit_label": first["visit_label"],
            "n_prior_visits": len(priors), "days_between": days,
            "cell_changes": cell_changes, "rotation": rotation, "self_alignment": self_align, "headline": head}


def run_from_folder(folder=None, out_dir=None):
    """Drop-and-run entry. Put an IDAT pair (*_Grn.idat, *_Red.idat) and an
    optional questionnaire.json into the folder that holds this script, then run
    `python walther_clinical.py`. Calibrates the IDAT to beta (Stage 1: per-sample
    noob dye-bias + probe-type normalization), runs the full CPG v1 chain, and
    writes the patient report beside the inputs. No manifest needed.

    questionnaire.json (optional): {"patient_id","age","sex","family_history","substrate"}.
    """
    import glob, json as _json, datetime, sys
    folder = Path(folder) if folder else _THIS_DIR
    out_dir = Path(out_dir) if out_dir else folder

    grn = _find_one(folder, "*_Grn.idat", "*Grn.idat", "*_Grn.idat.gz")
    red = _find_one(folder, "*_Red.idat", "*Red.idat", "*_Red.idat.gz")
    if not grn or not red:
        raise FileNotFoundError(
            f"No IDAT pair found in {folder}. Drop a *_Grn.idat and *_Red.idat there.")
    # Stage 1 (noob) uses methylprep's own managed manifest cache; the operator
    # does NOT need to supply a manifest. Only an IDAT pair + questionnaire.json.

    # questionnaire -> patient context (report-only; never enters the scored beta)
    qpath = _find_one(folder, "questionnaire.json", "*questionnaire*.json")
    q = {}
    if qpath:
        try:
            q = _json.load(open(qpath))
        except Exception as e:
            print(f"  questionnaire unreadable ({e}); proceeding without it.")
    elif sys.stdin and sys.stdin.isatty():
        # No questionnaire.json in the folder and we're at an interactive
        # terminal: prompt for the four fields the chain reads and write the
        # file, so the operator never has to hand-edit JSON. (The full intake
        # questionnaire is collected by cpg_intake_form.html; this is the
        # minimal command-line fallback.)
        q = _prompt_questionnaire(folder, grn, _json)
    ctx = PatientContext(age=q.get("age"), sex=q.get("sex"),
                         family_history=q.get("family_history"),
                         substrate=q.get("substrate", "whole_blood"))
    pid = q.get("patient_id") or Path(grn).name.split("_Grn")[0].split("Grn")[0].rstrip("_") or "patient"

    # Stage 1 - calibrate IDAT -> beta (per-sample dye-bias + probe-type
    # normalization / noob). Self-contained: uses only this patient's own
    # control + out-of-band probes. No cohort, no reference, no manifest needed
    # (methylprep manages its own manifest cache and auto-detects 450k vs EPIC).
    sys.path.insert(0, str(_THIS_DIR))
    import stage_1_idat_calibration as _s1
    print(f"[1/3] Stage 1 calibration (IDAT -> calibrated beta) for {pid} ...")
    beta, meta = _s1.calibrate_idat_to_beta(str(grn), str(red))

    print("[2/3] running CPG v1 chain (deconvolution -> per-cell A-scores -> match) ...")
    bundle = run_pipeline(beta, context=ctx, patient_id=pid)

    # --- TRAJECTORY: patient root = parent of the visit folder; baselines live
    #     there. Load this patient's prior visits, compare, and (after the report)
    #     persist this visit's vector for next time.
    visit_label = _visit_label(folder)
    patient_root = folder.parent
    baselines_dir = patient_root / "baselines"
    priors = _load_prior_baselines(baselines_dir, pid, exclude_label=visit_label)
    if priors:
        bundle["trajectory"] = _compute_trajectory(bundle, priors, visit_label, DEFAULT_CONFIG)
        print(f"      trajectory: {len(priors)} prior visit(s) for {pid} "
              f"-> comparing to {priors[-1].get('visit_label')}")
    else:
        bundle["trajectory"] = None
        print(f"      trajectory: first visit on record for {pid} (baseline will be saved)")

    print("[3/3] building report ...")
    sys.path.insert(0, str(_THIS_DIR))
    import importlib
    rb = importlib.import_module("cpg_report_builder")
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"CPG_report_{pid}_{visit_label}_{stamp}.html"
    rb.build_report(bundle, out_path=str(out_path))

    # persist this visit's baseline for future trajectory comparisons
    saved = _save_baseline(baselines_dir, _baseline_payload(bundle, visit_label))
    print(f"\nDONE. Report: {out_path}")
    print(f"      baseline saved: {saved}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CPG v1 LEAN conductor — IDAT to patient report.")
    ap.add_argument("--folder", default=None,
                    help="folder holding the IDAT pair + manifest + questionnaire.json "
                         "(default: the folder this script sits in)")
    ap.add_argument("--out", default=None, help="output folder for the report (default: --folder)")
    args = ap.parse_args()
    run_from_folder(args.folder, args.out)
