#!/usr/bin/env python3
"""disease_matching.py - Stage 8, disease-pattern concordance. FUTURE WORK, not a commissioned stage.

Extracted verbatim 2026-09-25 from walther_clinical.py (the superseded v1 conductor, commit b0517e9), which was
retired in the same change. This is the ONLY part of that 1,769-line file the live chain ever called:
cpg_conductor.py loaded the whole module to reach stage_8_dual_matching and nothing else.

WHAT THIS DOES. Given a patient's per-cell A-departure profile and a disease signature matrix, it measures a
directional weighted concordance between them and returns a plain-language strength of resemblance - never a
probability, never a diagnosis. It also reports whether a match is specific to the signature's own cells or
generic.

WHY IT IS MARKED FUTURE. Disease evidence for the commissioned chain is Issue 004 work, after sealed runs on
the current chain (author's instruction). The disease matrix this reads is pre-atlas, built on the
marker-union surface, and the RECON D2 finding says that surface and the identity gauge move in OPPOSITE
directions with age - so its signatures must not be quoted beside a reading from the commissioned gauge until
they have been re-derived. The code is kept, wired and runnable so that work starts from something tested
rather than from a rewrite; it is not part of any reading the chain reports today.

Extracted rather than rewritten on purpose: the functions and their constants are byte-faithful to the
version that was in use, so a future comparison is against the real thing.
"""
from __future__ import annotations

import os
import csv
import json
import numpy
import re as _re

from pathlib import Path

import numpy as np
import pandas as pd


# ---- constants, as they were in walther_clinical.py ----
# CPG_ROOT - the tree that holds the runtime modules, resolved as the retired v1 conductor resolved it
# (set CPG_ROOT in the environment to override). Carried over verbatim so DEFAULT_CONFIG's paths mean
# what they meant; nothing in the live chain reads them today.
_THIS_DIR = Path(__file__).resolve().parent
_ENV_ROOT = os.environ.get("CPG_ROOT")
if _ENV_ROOT:
    CPG_ROOT = Path(_ENV_ROOT)
else:
    CPG_ROOT = _THIS_DIR.parent
    if not (CPG_ROOT / "atlas").exists() and not (CPG_ROOT / "atlas").exists():
        for _p in [_THIS_DIR, *_THIS_DIR.parents]:
            if (_p / "MethylPhys/atlas").exists():
                CPG_ROOT = _p
                break

STRESS_MILD_MAG = 0.07
STRESS_MIN_COHERENCE = 0.60
STRESS_MOVE_EPS = 0.05
STRESS_NOTABLE_MAG = 0.10

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
    # Stage 4 A-score LOCI — per-class IDENTITY CpGs (beta near the class H_min_beta,
    # i.e. where a healthy cell sits at its characteristic methylation). This is the
    # gauge instrument: the mean beta over these loci, read as A = H(beta_mean)/H_min,
    # lands healthy on the age-matched band (immune ~0.90 young -> 1.00 age 95). It is
    # NOT the most-methylated extremes (old a_score_loci_v1_0 read immune 0.729) and
    # NOT the discriminative one-vs-rest markers (celltype_markers_json, the all-BREACH
    # bug root-caused 2026-06-11). Two loci sets, two jobs (Issue 002 §103): identity
    # loci -> the gauge here; discriminative markers -> Stage-1 deconvolution + the
    # separation/Cohen's-d disease-matching surface, never this A-score.
    "a_score_loci_json": CPG_ROOT / "Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json",
    # Stage 4 GAUGE engine — A = H(one mean beta)/H_min, age-matched placement +
    # Issue-002 severity ladder (cpg_gauge_engine.read). Loads age_reference_matrix.
    "gauge_engine_path": CPG_ROOT / "cpg_gauge_engine.py",
    # L6c derived age-matched departure (Option A): the class GAUGE scored against the
    # age_reference_matrix band, + overall cellular age. DERIVED-only, no cohort.
    "mahalanobis_module_path": CPG_ROOT / "Runtime Matrices/Mahalanobis_healthy_reference/iamatlas_mahalanobis_scoring.py",
    "mahalanobis_reference_json": CPG_ROOT / "Runtime Matrices/Mahalanobis_healthy_reference/mahalanobis_healthy_reference_v2_0_age_matched_derived.json",
    # Stage 4.5 — bidirectional decomposition
    "bidirectional_module_path": CPG_ROOT / "Runtime Matrices/Directional Panel/bidirectional_decomposition.py",
    "directional_panels_json": CPG_ROOT / "Runtime Matrices/Directional Panel/directional_panels_v1_0.json",
    # Stage 4.6 — brightness comparison / Mollweide
    "brightness_module_path": CPG_ROOT / "Runtime Matrices/Mollweide & Brightness Comparison/patient_brightness_comparison.py",
    "brightness_archives_dir": CPG_ROOT / "atlas/iamatlas_class_archives",
    # Stage 9 — brilliance maps (HEALPix Mollweide)
    "cpg_healpix_mapping_npy": CPG_ROOT / "Runtime Matrices/cpg healpix mapping/iamatlas_cpg_to_healpix_nside128.npy",
    "whole_atlas_reference_npz": CPG_ROOT / "Runtime Matrices/Mollweide & Brightness Comparison/whole_atlas_reference/iamatlas_whole450k_reference.npz",
    # Stage 2 — deconvolution (Walther NNLS primary + NILC cross-check)
    "atlas_csv_xz":        CPG_ROOT / "atlas/IAMAtlasREBUILD.csv.xz",
    "atlas_csv_decompressed": CPG_ROOT / "atlas/IAMAtlasREBUILD.csv",
    "celltype_to_class_json": CPG_ROOT / "atlas/IAMAtlasREBUILD_celltype_to_class.json",
    # Atlas provenance — SOP §99 SINGLE SOURCE OF TRUTH for the frozen H_min
    # (key h_min_values_frozen_2026_04_06), build pipeline, distinctness tests.
    # Stage 4 reads this and refuses to run if the runtime H_min disagrees.
    # NOTE: the atlas <class>_mean global mean ~0.5 (std ~0.34, mass at both ends)
    # is CORRECT bimodal methylation, NOT a flat atlas or a scale offset — see
    # IAMAtlas_FLATNESS_LESSON.md and SOP §103. Do not re-derive H_min from it.
    "atlas_provenance_json": CPG_ROOT / "atlas/IAMAtlasREBUILD_provenance.json",
    "walther_deconv_module": CPG_ROOT / "Walther_iam_deconvolver/walther_iam_deconvolver.py",
    "nilc_deconv_module":  CPG_ROOT / "NILC Deconvolver/nilc_deconvolver-2.py",
    # per-CELL NILC (the second cell-level lens): a cell is presented only when BOTH
    # Walther and the per-cell NILC resolve it (the agreement requirement). Disagreement
    # excludes the cell as noise. This is the filter that keeps the census clean.
    "nilc_celltype_module": CPG_ROOT / "NILC Deconvolver/nilc_celltype_deconvolver.py",
    "cell_agreement_min_fraction": 0.02,
    # Stage 8 — disease signature matrix matching (Path B) + priors
    "disease_matrix_csv":  CPG_ROOT / "Disease Matrix/DISEASE_MATRIX/disease_cell_signature_matrix_v1_13.csv",
    "matrix_mapping_json": CPG_ROOT / "Disease Matrix/DISEASE_MATRIX/iamatlas_115_to_matrix_v0_2_mapping.json",
    "cancer_prior_json":   CPG_ROOT / "Runtime Matrices/Cancer_prior/cancer_prior.json",
    "family_history_json": CPG_ROOT / "Runtime Matrices/Family_history_multiplier/family_history_multiplier.json",
    "literature_anchors_json": CPG_ROOT / "Runtime Matrices/Literature_anchors_Report building/literature_anchors.json",
    # Stage 9 — report visual assets (gauges + rankings)
    "gauge_module_path": CPG_ROOT / "cpg_gauge.py",
    "tier_breakpoints_json": CPG_ROOT / "Runtime Matrices/Tier_breakpoints/tier_breakpoints.json",
}

CONCORDANCE_MIN_SIGNAL_CELLS = 3
DIRECTIONAL_MAG_FLOOR    = 0.15
DIRECTIONAL_MOVE_EPS     = 0.05
DIRECTIONAL_WEIGHT_FLOOR = 0.20
_LYMPHOID_CELLS = {
    "CD4_T_cells", "CD4_T-cells", "CD4T", "CD4Tnv", "CD4Tmem", "CD8_T_cells", "CD8_T-cells",
    "CD8T", "CD8Tnv", "CD8Tmem", "B_cells", "CD19_B-cells", "Bnv", "Bmem", "naive_B_cells",
    "memory_B_cells", "NK_cells", "CD56_NK-cells", "NK", "regulatory_T_cells", "Treg",
    "memory_T_cells_pooled", "plasma_cells", "Plasma", "lymphoid",
}
_MYELOID_CELLS = {
    "granulocytes", "granulocytes_pooled", "neutrophils", "Neutro", "Neu", "Neutrophils_EPIC",
    "Neutrophils_reinius", "eosinophils", "eosinophil", "Eos", "basophils", "Baso",
    "monocytes", "CD14_monocytes", "Mono", "macrophages", "macrophages_peripheral", "Macro",
    "dendritic_cells", "dendritic", "GMP", "CMP", "MPP", "MEP", "MP", "myeloid",
}
_PROGENITOR_CELLS = {
    "MPP", "CMP", "GMP", "MEP", "L_MPP", "HSC", "HSPC_pooled", "megakaryocyte",
    "erythroid_progenitor", "myeloid_progenitor", "erythroblast", "nRBC",
}


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
    # PROC-MATCH-01 M1 (2026-09-21): FAIL CLOSED. A missing or unparsable origin map used to become {} and the
    # specificity rule degraded silently; now Stage 8 refuses to match at all.
    _op = Path(cfg["disease_matrix_csv"]).parent / "disease_origin_cells.json"
    try:
        origin_map = _json.load(open(_op))
        if not isinstance(origin_map, dict) or not origin_map: raise ValueError("origin map empty")
    except Exception as _e:
        return Stage8Output(route_B_concordance=[], route_B_all_scored=[], patient_departure={}, route_A_architectural_alarm={},
                            route_C_bidirectional={}, status=f"NOT AVAILABLE - cell-of-origin map unreadable ({_op.name}: {_e}); Stage 8 refuses to match (fail-closed, PROC-MATCH-01)")
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
