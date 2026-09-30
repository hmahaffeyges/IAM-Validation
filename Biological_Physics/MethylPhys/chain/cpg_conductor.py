#!/usr/bin/env python3
"""CPG Conductor — clean from-scratch orchestrator (2026-07).

Replaces the v1 conductor, which was retired on 2026-09-25 after its one live function (stage_8_dual_matching) was extracted to disease_matching.py. Wires the KISS files in the order
Heath specified, each stage a small pure function, per-cell A always paired with
its deconvolved fraction so presence and score are read together.

Inputs it needs (all in the working dir or repo):
  - IAMAtlasREBUILD.csv                 (decompressed atlas, the deconvolver reference)
  - IAMAtlasREBUILD_celltype_to_class.json
  - iamatlas_celltype_markers_v0_2.json (per-cell discriminative markers + H_min_by_class)
  - iamatlas_gauge_identity_loci_v1_0.json  (Stage B, the class gauge - internal gate; the age curve, band and lab zero were retired 2026-09-27; age_reference_matrix.json is diagnostic only)
  - iamatlas_mahalanobis_scoring.py + mahalanobis_healthy_reference_v2_0_*.json (Stage C)
  - directional_panels_v1_0.json, bidirectional_decomposition.py       (Stage C)
  - tier_breakpoints.json, disease_cell_signature_matrix_v1_13.csv,
    iamatlas_115_to_matrix_v0_2_mapping.json                           (Stage C)

STAGE A (this file, wired + tested):
  deconvolve(beta) -> class_fractions + celltype_fractions   (the RATIOS)
  score_per_celltype(beta) -> 115 per-cell A-scores          (via v0_2 markers)
  -> paired: {cell: {A, fraction, class, present}}   present = fraction >= detect_floor


BETA SCALE (LESSON-SCALE-01, 2026-09-20): H_min and the Atlas live on the Roadmap/GenomicStudio beta scale. Stage-1 noob beta is +0.066 higher on the identity loci; GEO author-processed EPIC +0.037. ANY absolute A reading must first map patient beta onto the Roadmap scale via Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json (PROVISIONAL). Within-pipeline comparisons do not need it. Do NOT re-derive H_min per pipeline. Record: Record/VAL_PostAtlas/CPG_PHASE1_identity_band_GSE87571/OUTCOME.md

LAB ZERO (LAB-ZERO-02, 2026-09-20): absolute A requires a per-lab zero from a 40-array healthy panel read against reference_age_curve_v1.json (lab_zero.py, PROC-PANEL-03) in addition to the pipeline map; readings without it are lab_zero=UNSET and not reportable as absolute.
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── Resolve chain files whether laid out flat (the CPG_TRIAL_CODE working folder) or in the
#    repository tree (Biological_Physics/MethylPhys/chain/{Runtime Matrices/*, legacy_iam_deconvolver/} and
#    Biological_Physics/MethylPhys/atlas/). Added 2026-09-19; first run of the conductor from the repo.
# One deconvolver per (atlas, class-map) pair, per process. See stage_a_cells for why this is safe.
_DEC_CACHE = {}

_SEARCH = [HERE, HERE / "legacy_iam_deconvolver", HERE / "Runtime Matrices" / "A_Scoring_Module",
           HERE / "Runtime Matrices" / "Celltype_Marker", HERE / "Runtime Matrices" / "Directional Panel", HERE / "Runtime Matrices" / "Tier_breakpoints",
           # the atlas: HERE is MethylPhys/chain, so its sibling; the second form covers a flat working folder
           HERE.parent / "atlas", HERE.parent.parent / "MethylPhys" / "atlas",
           HERE / 'Runtime Matrices' / 'Percell_Reference',
           HERE / 'Runtime Matrices' / 'Patient_CMB',   # 2026-09-27: presence_floors_v1.json and the residual scales live here; off the path until today   # 2026-09-26: percell_reference_v0_3.json has lived here since 09-22 and _find could not see it,
           HERE / 'Runtime Matrices' / 'Intake',   # 2026-09-27: intake_thresholds_v1.json
           # so the per-cell healthy reference (A' = H/H_ref, 1.0 healthy by construction, with
           # per-lab p10/p90 bands and a held-out check) was unreachable from the chain
           ]
def _find(name, required=True):
    for d in _SEARCH:
        p = d / name
        if p.exists(): return p
    if not required: return None
    raise FileNotFoundError(f"{name}: not found in any of {[str(d.relative_to(HERE.parent)) for d in _SEARCH]}")
DETECT_FLOOR = 0.01  # 1% — a cell below this is treated as absent (fraction sets presence)


def _load_module(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec so @dataclass resolves __module__
    spec.loader.exec_module(mod)
    return mod


# _trace_detect() removed 2026-09-30 with Stage 2c (chain/RETIRED_2026-09/trace_detection_2026-09-30/WHY.md).


def stage_a_cells(beta_dict, atlas_csv, cfg=None):
    """Stage A — find the cell types in the sample, their ratios, and their A-scores.

    beta_dict : {cpg_id: beta} calibrated patient betas
    atlas_csv : path to decompressed IAMAtlasREBUILD.csv (deconvolver reference)
    Returns: {'class_fractions', 'celltype_fractions', 'cells'} where cells is
             {cell: {A, coverage, confidence, status, class, fraction, present}}.
    """
    cfg = cfg or {}
    dec_mod = _load_module("legacy_iam_deconvolver", _find("legacy_iam_deconvolver.py"))
    asc = _load_module("iamatlas_a_scoring", _find("iamatlas_a_scoring.py"))
    c2c_path = _find("IAMAtlasREBUILD_celltype_to_class.json")
    markers_path = _find("iamatlas_celltype_markers_v0_2.json")

    # 1. Deconvolve -> the ratio of each class and cell type present
    # The deconvolver re-reads the 605 MB atlas on construction, and this line ran on EVERY specimen -
    # about two minutes per array, which makes panel-scale work (318 to 732 healthy arrays) impossible
    # through the chain entry point, and the entry point is now the only permitted route (propagate rule
    # 10). It holds NO per-specimen state - deconvolve(beta_dict) takes the specimen as an argument - so
    # caching it by reference path cannot change a reading. Proven by invariance check, not asserted:
    # kit/PROC_CACHE_01.py recomputes bundles produced BEFORE this change and requires every reported
    # value to be identical. 2026-09-26.
    _ck = (str(atlas_csv), str(c2c_path))
    dec = _DEC_CACHE.get(_ck)
    if dec is None:
        dec = dec_mod.legacyIAMDeconvolver(str(atlas_csv), celltype_class_map=str(c2c_path))
        _DEC_CACHE[_ck] = dec
    # 2026-09-26: the deconvolver reads MAPPED betas, as the class gauge always did. Raw stage-1 betas sit ~0.07
    # above the atlas; with solid-tissue columns now solvable that offset landed on tissue (4.1% -> 2.4% -> 0.0%
    # median non-blood in healthy blood once mapped and twins were merged). Mapping therefore happens FIRST.
    _pipeline = (cfg or {}).get('pipeline', 'stage1_noob_450K') if isinstance(cfg, dict) else 'stage1_noob_450K'
    try:
        _beta_mapped, _ = stage_1s_scale_map(beta_dict, _pipeline)
    except Exception as _e:   # a pipeline with no map falls back to raw, and SAYS so in the record
        _beta_mapped = beta_dict
        print('  per-cell scoring on RAW betas - no scale map for pipeline %r (%s)' % (_pipeline, _e))
    result = dec.deconvolve(_beta_mapped)
    class_fr = dict(result.class_fractions)
    ct_fr = dict(result.celltype_fractions)

    # 2. Per-cell A-scores via the v0_2 discriminative markers (mean-of-per-CpG H/H_min)
    _meta, ct_markers, c2c, h_min = asc.load_artifact(str(markers_path))
    # PER-CELL IDENTITY LOCI (2026-09-26). The per-cell A was computed on discriminative marker
    # panels, which the reference audit disqualified: a cell's own atlas mean read far below 1.0.
    # This artifact gives 102 of 115 cells a panel of loci sitting AT their class floor, built by
    # the same criterion as the eight class panels; each cell's own reference reads 0.9362-1.0204.
    _pci_path = _find("iamatlas_percell_identity_loci_v1_1.json", required=False)
    _pci = asc.load_percell_identity(str(_pci_path)) if _pci_path else None
    try:
        _pref = None   # 2026-09-27: the per-cell atlas 'reference' was removed by the author's ruling - the only reference is A = 1.00
    except Exception:
        _pref = None
    # 2026-09-27: the per-cell laboratory offset (a per-lab zero from healthy arrays, percell_reference_identity_v1_0.json)
    # was removed from the record by the author's ruling - no zero from any population is attached to a cell's A.
    # SCALE MAP BEFORE PER-CELL SCORING (2026-09-26). The class gauge maps stage-1 betas onto the atlas
    # scale before it reads them; this path never did, and read RAW betas that sit ~0.07 above the atlas
    # at the identity loci (the offset PROC-COV-01 measured). Raw, every present cell in healthy blood
    # read SUPPRESSED (Neutrophils 0.87, NK 0.80); mapped, the same arrays read 0.968-1.053 = NORMAL.
    scores = asc.score_per_celltype(_beta_mapped, ct_markers, c2c, h_min, celltype_identity_loci=_pci, percell_reference=_pref)

    # 3. Pair A with fraction — presence comes from the ratio, not the A-score
    # 2026-09-26: the interval on a reading is computed by the scorer ON THE SAME SURFACE as the reading (identity loci,
    # resampled) and travels with it, together with the cell's MCMC reference. The marker-CpG bootstrap that lived here
    # produced an interval on the retired surface that did not contain the A it stood beside (HSC 0.917 vs 0.750-0.881).
    cells = {}
    for ct, r in scores.items():
        frac = float(ct_fr.get(ct, 0.0))
        cells[ct] = {
            "A": r.get("A"),
            "reading_ci": r.get("reading_ci"),
            "reference": r.get("reference"),
            "coverage": r.get("coverage"),
            "confidence": r.get("confidence"),
            "status": r.get("status"),
            # which surface and which formula produced this A - so the failsafe can SEE a reversion to marker
            # panels or to the wrong form, instead of a reader having to remember (2026-09-26)
            "A_raw": r.get("A"),
            "surface": r.get("surface"),
            "formula": r.get("formula"),
            "jensen_gap": r.get("jensen_gap"),
            "class": c2c.get(ct),
            "fraction": frac,
            "present": frac >= DETECT_FLOOR,
        }
    # 2026-09-26 RESOLVABILITY, carried so the report never shows a family as several measurements and never shows
    # a sub-floor cell as 'fraction 0': shared -> the family a cell's fraction belongs to; unresolvable -> cells the
    # array cannot resolve (defined on < 1% of loci); twins_dropped -> lower-coverage copies of a solved cell.
    _shared = dict(getattr(result, "celltype_shared", {}) or {})
    _unres = list(getattr(result, "celltype_unresolvable", []) or [])
    _twins = dict(getattr(result, "celltype_twins_dropped", {}) or {})
    _excl = dict(getattr(result, "celltype_exclusive_n", {}) or {})
    for _ct, _rec in cells.items():
        if not isinstance(_rec, dict):
            continue
        if _ct in _shared:
            _rec["shared_with"] = _shared[_ct]
        if _ct in _unres:
            _rec["resolvable"] = False; _rec["fraction"] = None
            _rec["status"] = (_rec.get("status") or "") + "|NOT_RESOLVABLE_ON_PLATFORM"
        elif _ct in _twins:
            _rec["resolvable"] = False; _rec["fraction"] = None
            _rec["status"] = (_rec.get("status") or "") + "|TWIN_OF:" + _twins[_ct]
        else:
            _rec["resolvable"] = True
        _rec["exclusive_markers"] = _excl.get(_shared.get(_ct, _ct))
    return {"class_fractions": class_fr, "celltype_fractions": ct_fr, "cells": cells,
            "celltype_shared": _shared, "celltype_unresolvable": _unres, "celltype_twins_dropped": _twins,
            "celltype_families": dict(getattr(result, "celltype_families", {}) or {}), "celltype_exclusive_n": _excl}



def stage_b_classes(beta_dict, stage_a_out, cfg=None):
    """Stage B - per-class GAUGE. **AS WIRED (2026-07 -> today): A = H(beta_mean)/H_min over the
    class MARKER UNION (iamatlas_celltype_markers_v0_2.json, ~3k bimodal CpGs per class), read
    against age_reference_matrix, which was compiled on the same marker-union statistic
    (MPHYS_WEB_v13 _AGE_REFERENCE).** This is NOT the identity-loci gauge that SOP §41/§106,
    Issue 002 and the sentence below describe; PROC-N7-01 (2026-09-19) showed the marker-union
    statistic reads a synthetic healthy mixture as BREACH (1.13) and misses real adenoma (0.93),
    while identity-loci H(beta_mean) reads 0.99 and 1.10 respectively. The switch to identity
    loci is gated on an identity-loci band (Phase 1, GSE87571); until then every reading carries
    gauge_surface = "marker_union". Original docstring follows.
    A = H(beta_mean)/H_min over the class
    IDENTITY loci (iamatlas_gauge_identity_loci_v1_0.json), read against the age-matched
    band via cpg_gauge_engine. Paired with the Stage A fraction: a class below the
    substrate presence floor reads blood-background, flagged BACKGROUND_LOW_FRACTION,
    NOT a finding. This is the doctor's fuel gauge (never the separation surface)."""
    import numpy as np
    cfg = cfg or {}
    age = cfg.get("age", 60)
    ge = _load_module("cpg_gauge_engine", _find("cpg_gauge_engine.py"))
    asc = _load_module("iamatlas_a_scoring", _find("iamatlas_a_scoring.py"))
    # PRODUCTION A-score (MPHYS_WEB_v13 + Reproduction Paper v3): beta_mean = customer's mean
    # beta over the class MARKER/informative CpGs (NOT identity loci). ge.read() then does
    # A = H(beta_mean)/H_min + age-matched placement + tier.
    _meta, ctm, c2c, _hmin = asc.load_artifact(str(_find("iamatlas_celltype_markers_v0_2.json")))
    cls_markers = {}
    for ct, mk in ctm.items():
        cls_markers.setdefault(c2c.get(ct), []).extend(mk)
    class_fr = stage_a_out.get("class_fractions", {})
    classes = {}
    for cls in ["immune", "secretory", "cycling", "terminal",
                "stromal", "progenitor", "stem_adult", "stem_pluri"]:
        cpgs = list(set(cls_markers.get(cls, [])))
        vals = [beta_dict[c] for c in cpgs if c in beta_dict]
        if not vals:
            classes[cls] = {"A": None, "status": "NO_MARKERS"}
            continue
        mb = float(np.mean(vals))
        r = ge.read(mb, cls, age=age)
        A = r.get("A") if isinstance(r, dict) else r
        frac = float(class_fr.get(cls, 0.0))
        present = frac >= DETECT_FLOOR
        classes[cls] = {
            "A": round(A, 4), "beta_mean": round(mb, 4),
            "placement": (r.get("placement") if isinstance(r, dict) else ge.placement(A, cls, age)),
            "tier": (r.get("tier") if isinstance(r, dict) else ge.tier(A, cls, age)),
            "fraction": round(frac, 4), "present": present,
            "status": "OK" if present else "BACKGROUND_LOW_FRACTION",
            "gauge_surface": "marker_union",   # PROC-N7-01: not identity loci; see docstring
            "n_cpgs": len(vals),
        }
    return {"class_gauge": classes, "age": age}




def stage_1s_scale_map(beta_dict, pipeline):
    """Stage 1s (LESSON-SCALE-01, SOP s109): put patient beta on the Roadmap scale that H_min and the Atlas
    were calibrated on. beta_roadmap = (beta - intercept) / slope from beta_scale_maps_v1.json, keyed by the
    pipeline tag Stage 1 stamps on its output. Returns (mapped_dict, scale_label). If the pipeline has no
    fitted map the beta is returned UNCHANGED with label 'UNMAPPED' - downstream must not report a tier from it.
    Applied to the GAUGE path only: Stage 2 composition was verified on unmapped Stage-1 beta and is scale-tolerant
    (simplex-constrained NNLS); the wired marker-union gauge keeps unmapped beta because ITS band was compiled that way."""
    if pipeline == "atlas_scale":   # beta already on the Roadmap/atlas scale (synthetic patients, N7); identity map
        return dict(beta_dict), "MAPPED(atlas_scale identity; slope 1, intercept 0)"
    maps = json.load(open(_find("beta_scale_maps_v1.json")))["maps"]
    m = maps.get(pipeline or "", {})
    if m.get("slope") is None:
        return beta_dict, f"UNMAPPED({pipeline or 'unknown pipeline'})"
    a, b = float(m["slope"]), float(m["intercept"])
    return {k: min(1.0, max(0.0, (v - b) / a)) for k, v in beta_dict.items()}, f"MAPPED({pipeline}->roadmap; slope {a}, intercept {b}; {maps and json.load(open(_find('beta_scale_maps_v1.json')))['_meta']['version']})"


def stage_b_identity(beta_mapped, stage_a_out, age=None, scale_label="", lab_zero=None):
    """THE CLASS GAUGE - INTERNAL GATE ONLY (PROC-SWITCH-01, 2026-09-21; RULING A3; author's ruling 2026-09-27).
    A = H(beta_mean) / H_min over the class IDENTITY loci on mapped beta. Healthy is A = 1.00 by the physics; the tier
    scale is the tolerance. Nothing is subtracted from A and no population places it.

    2026-09-27: the three-layer reference (A_abs = A - c(decade) - z_lab, placed in identity_band_v3) was REMOVED. The
    age curve, the laboratory zero and the band were cohort statistics - where healthy donors sat on the gauge - and the
    author ruled that where people sit is never a correction to a cell. reference_age_curve_v1.json, identity_band_v3.json
    and lab_zero.py moved to RETIRED_2026-09. `age` and `lab_zero` are accepted and ignored so old callers do not break;
    age is carried as context in the bundle by run_full, never used here.

    What this stage still does on the live path: the class reading (printed nowhere; the report reads cells), and the
    PROC-FOREIGN-01 composition check that decides whether the specimen is whole blood - the tier on a cell is withheld
    when it is not. s108: on whole blood only immune and the joint haematopoietic-progenitor component are read."""
    import math
    ident = json.load(open(_find("iamatlas_gauge_identity_loci_v1_0.json")))
    ident = {k: v for k, v in ident.items() if isinstance(v, dict) and "loci" in v}
    def H(b): b = min(max(b, 1e-12), 1 - 1e-12); return -b * math.log2(b) - (1 - b) * math.log2(1 - b)
    fr = stage_a_out["class_fractions"]
    out = {}
    groups = {"immune": ["immune"], "haematopoietic_progenitor": ["progenitor", "stem_adult"]}
    mapped = str(scale_label).startswith("MAPPED")
    for name, members in groups.items():
        loci = [c for cls in members for c in ident[cls]["loci"]]
        vals = [beta_mapped[c] for c in loci if c in beta_mapped]
        frac = sum(fr.get(c, 0) for c in members)
        if not vals or frac < 0.01:
            out[name] = {"present": False, "fraction": round(frac, 4)}; continue
        hm = ident["progenitor" if name != "immune" else "immune"]["H_min"]        # joint uses progenitor's floor (PREREG s3)
        A = H(sum(vals) / len(vals)) / hm
        rec = {"present": True, "fraction": round(frac, 4), "A": round(A, 4), "A_mapped": round(A, 4), "n_loci": len(vals),
               "gauge_surface": "identity_loci", "scale": scale_label, "H_min": hm,
               "reportable": bool(mapped), "internal_gate": True,
               "reason": None if mapped else "beta not on the calibration scale (UNMAPPED)"}
        # PROC-FOREIGN-01: the composition check, on every path
        try:
            _g = json.load(open(_find('composition_guard_v1.json')))
            _fgn = 1.0 - sum(float(fr.get(_c, 0.0)) for _c in _g['blood_lineage'])
            rec['foreign_fraction'] = round(_fgn, 4)
            rec['composition_verified'] = bool(_fgn <= _g['foreign_fraction_max'])
            if not rec['composition_verified']:
                rec['reason'] = ('composition unverified: %.1f%% of this specimen is assigned outside the blood lineage, '
                                 'above the %.1f%% the gauge was commissioned for. Cell tiers are withheld (PROC-FOREIGN-01).'
                                 % (100 * _fgn, 100 * _g['foreign_fraction_max']))
        except FileNotFoundError:
            rec['composition_verified'] = None      # no guard file: withhold nothing, say nothing
        out[name] = rec
    return out

def stage_4_5_bidirectional(beta_dict, cfg=None):
    """Stage 4.5 (SOP §46.5) - bidirectional decomposition. Signed directional
    composite that catches bidirectional CpG patterns the pooled entropy cancels.
    v1.0 panels: immune class only (VAL-051 AD). Other classes -> NO_PANEL honestly."""
    import pandas as pd
    bd = _load_module("bidirectional_decomposition", _find("bidirectional_decomposition.py"))
    panels = bd.load_directional_panels(_find("directional_panels_v1_0.json"))
    beta_series = pd.Series(beta_dict)
    report = bd.compute_per_class_bidirectional_decomposition(beta_series, panels, patient_id="patient")
    out = {}
    for cls, r in report.per_class_results.items():
        out[cls] = {"a_directional": getattr(r, "a_directional_composite", None),
                    "a_pooled": getattr(r, "a_pooled_entropy", None),
                    "flag_bidirectional": getattr(r, "flag_bidirectional", None),
                    "n_covered": getattr(r, "n_covered", None),
                    "interpretation": getattr(r, "interpretation", None)}
    return {"bidirectional": out, "any_flagged": getattr(report, "any_bidirectional_flagged", False)}


# stage_5_mahalanobis: REMOVED 2026-09-27 (author's ruling - a cohort statistic; the code is in RETIRED_2026-09/cohort_gauge_layers_2026-09-27/README.md's list)

# stage_5_hull_marker_union: REMOVED 2026-09-27 - not called by run_full; a hull distance from a population (RETIRED_2026-09/cohort_gauge_layers_2026-09-27/README.md)



# stage_6_cellular_age: REMOVED 2026-09-27 (author's ruling - a cohort statistic; the code is in RETIRED_2026-09/cohort_gauge_layers_2026-09-27/README.md's list)

# stage_6_cellular_age_marker_union: REMOVED 2026-09-27 (author's ruling - a cohort statistic; the code is in RETIRED_2026-09/cohort_gauge_layers_2026-09-27/README.md's list)


def stage_4_6_patient_sky(beta_rm, stage_a_out, cfg=None, atlas_csv=None):
    """Stage 4.6 - the patient's sky. z_i = (beta_i - sum_c f_c mu_ci) / sigma_i on the mapped beta, where
    sigma_i^2 = sum_c f_c^2 sd_ci^2 (the atlas posterior, propagated through this specimen's composition)
               + sigma_arr^2(beta_i)  (this array's own noise from its 65 SNP probes: a + b*beta(1-beta); cfg['snp_noise']).
    No laboratory zero, no laboratory spread, no panel (2026-09-27: the 40-array panel scales were retired as population layers).
    Without the array's SNP probes (a betas-only input) sigma_arr is unknown and the sky is drawn on the atlas term alone with
    the caption saying so - a picture, not a reading. Class panels gated by the presence floors as before.
    Development note (PROC-SKY-01, 2026-09-27): robust SD of z 0.92-1.16 on three laboratories; 0.60 on GSE125105, whose arrays
    are low-signal input (FINDING_GSE125105_LOW_SIGNAL.md) - handled at intake, not here."""
    import importlib.util, os as _os, json as _json, pandas as _pd, numpy as _np
    cfg = cfg or {}
    spec = importlib.util.spec_from_file_location("stage_4_6_patient_cmb", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "stage_4_6_patient_cmb.py"))
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    rt = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "Runtime Matrices", "Patient_CMB")
    floors = _json.load(open(_os.path.join(rt, "presence_floors_v1.json")))["floors"]
    mu, sd = S.load_atlas_mean_sd(atlas_csv or _find("IAMAtlasREBUILD.csv"))
    ident = _json.load(open(_find("iamatlas_gauge_identity_loci_v1_0.json"))); loci = {c: v["loci"] for c, v in ident.items() if isinstance(v, dict) and "loci" in v}
    beta = _pd.Series(beta_rm, dtype=float); beta.index = beta.index.map(str)
    snp = cfg.get("snp_noise") or {}
    slope = float((_json.load(open(_find("beta_scale_maps_v1.json")))["maps"].get(cfg.get("pipeline") or "stage1_noob_450K") or {}).get("slope", 1.0))
    sky = S.patient_sky_sigma(beta, stage_a_out["class_fractions"], mu, sd, snp.get("a"), snp.get("b"), slope, loci, S.load_mapping(), presence_floors_by_class=floors)
    out = {"available": True, "sigma": "atlas posterior + this array's SNP-probe noise" if snp.get("a") is not None else "atlas posterior only (no SNP probes in this input)",
           "snp_noise": snp or None, "presence_floors": floors,
           "all": sky["all"], "classes": {c: {k: v for k, v in d.items() if k != "pixels"} for c, d in sky["classes"].items()},
           "_sky": sky}   # full arrays for render_plate; stripped by the report builder
    return out

# stage_8_matching: REMOVED 2026-09-27 - not called by run_full; signature matching, which the chain does not do (RETIRED_2026-09/cohort_gauge_layers_2026-09-27/README.md)


def stage_2b_second_opinion(beta_dict, stage_a_out, atlas_csv, cfg=None):
    """Row 2b - the second opinion. NILC (needlet internal linear combination, the Planck component-separation
    method) run beside legacy's constrained NNLS and compared AT THE CLASS LEVEL, which is where PROC-SEP-03
    showed the atlas is separable. Ships as a second column plus an agreement flag, never as the composition the
    report stands on: legacy is conservative and commissioned, NILC is variance-weighted and deliberately
    sensitive, and the disagreement is information (PROC-NILC-01 - the disagreement WAS the finding). Cell-level
    disagreement inside one lineage is EXPECTED, not a defect, because the atlas cannot split the blood classes.
    Returns {} if the module or the marker file is unavailable, so the chain never depends on it.
    """
    import importlib.util, os, numpy as np
    try:
        mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nilc_celltype_deconvolver.py")
        if not os.path.exists(mp): return {"available": False, "reason": "nilc_celltype_deconvolver.py not present"}
        spec = importlib.util.spec_from_file_location("nilc_mod", mp); nl = importlib.util.module_from_spec(spec); spec.loader.exec_module(nl)
        d = nl.NILCCelltypeDeconvolver(str(atlas_csv), _find("iamatlas_celltype_markers_v0_2.json"))
        out = d.deconvolve(beta_dict)
        nf = out.get("fractions", out) if isinstance(out, dict) else out
        nf = {k: float(v) for k, v in dict(nf).items()}
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}"[:200]}
    c2c = json.load(open(_find("IAMAtlasREBUILD_celltype_to_class.json")))
    wf = stage_a_out.get("celltype_fractions", {})
    def by_class(frac):
        out = {}
        for cell, v in frac.items():
            cl = c2c.get(cell)
            if cl: out[cl] = out.get(cl, 0.0) + float(v)
        return out
    W, N = by_class(wf), by_class(nf)
    classes = sorted(set(W) | set(N))
    rows = {c: {"legacy": round(W.get(c, 0.0), 4), "nilc": round(N.get(c, 0.0), 4),
                "abs_diff": round(abs(W.get(c, 0.0) - N.get(c, 0.0)), 4)} for c in classes}
    L1c = sum(r["abs_diff"] for r in rows.values())
    L1cell = sum(abs(wf.get(k, 0.0) - nf.get(k, 0.0)) for k in set(wf) | set(nf))
    # agreement bar: class-level L1 <= 0.10 is agreement; above it the two methods are telling different stories
    agree = L1c <= 0.10
    return {"available": True, "by_class": rows, "L1_class": round(L1c, 4), "L1_cell": round(L1cell, 4),
            "agreement": "AGREE" if agree else "DISAGREE", "bar": "class-level L1 <= 0.10",
            "note": ("the two solvers agree on the architecture-class composition the report stands on; cell-level "
                     "differences inside a lineage are expected and are not scored") if agree else
                    ("the two solvers disagree at the class level - per PROC-NILC-01 that is information, not a "
                     "defect in either: read the class table and treat the composition as uncertain"),
            "reported_composition": "legacy (constrained NNLS) - NILC is a second opinion only"}

def stage_2d_foreign_detection(beta_mapped, stage_a_out, lab, bi=None):
    """Stage 2d - FOREIGN-CELL DETECTION as ONE JOINT FIT (PROC-STAGE2D-03, adopted 2026-09-27 by the author's ruling: B1's
    UNSPECIFIC bar failed as written (1.9 % vs 1 %) and the author ruled the 1.9 % a measurement, not a false alarm -
    doors/PROC_STAGE2D_03_OUTCOME.md).

    One NNLS of the specimen's panel markers (mapped) on [blood reference columns | all 21 foreign templates];
    f_hat_cell = coef_cell / sum(coef). The earlier design (blood fitted FIRST, template read in the residual) let the blood
    fit absorb foreign material: with honest spikes it responded at 2-13 % of a 5 % spike (PROC_STAGE2D_02_OUTCOME.md), and
    the 0.5 % limits of PROC-MF-03 came from injecting the panel's own template. The joint fit responds at 0.6-1.0 of f.
    Line: the instrument's NOISE FLOOR - 0.99 quantile over 732 arrays known to lack the cell (author's decision 2026-09-27),
    printed with N; a standing bias on blood is printed beside it. >= 3 templates above their floors = 'epithelial-like
    material, cell not resolved' (1.9 % of healthy arrays, the oldest). A template not resolved on this block prints NOT
    DETECTABLE. Gate: composition check verified blood-like. Never changes the blood composition.
    """
    import numpy as _np
    from scipy.optimize import nnls as _nnls
    out = {"status": None, "laboratory": lab, "cells": {}, "detected": [], "not_detectable": {}}
    try:
        P = json.load(open(_find("detection_panel_v3.json")))
    except Exception as e:
        out["status"] = "NOT_RUN: detection_panel_v3.json not found (%s)" % type(e).__name__; return out
    imm = ((bi or {}).get("immune") or {})
    if imm.get("composition_verified") is False:
        out["status"] = "WITHHELD: the composition guard did not verify this specimen as blood-like (foreign fraction %s); a line is not applied to a specimen that is not blood" % imm.get("foreign_fraction")
        return out
    M = P["markers"]; v = _np.array([beta_mapped.get(m, _np.nan) for m in M], dtype=float); ok = ~_np.isnan(v)
    if ok.sum() < 0.8 * len(M):
        out["status"] = "NOT_RUN: only %d of %d panel markers present (need 80 %%)" % (int(ok.sum()), len(M)); return out
    cells = list(P["foreign_ref"])
    Ab = _np.array([P["blood_ref"][c] for c in P["blood_columns"]]).T[ok]
    T = _np.array([P["foreign_ref"][c] for c in cells]).T[ok]
    coef, _ = _nnls(_np.column_stack([Ab, T]), v[ok]); tot = max(float(coef.sum()), 1e-9)
    fb = coef[:Ab.shape[1]]; ff = coef[Ab.shape[1]:]
    out["foreign_mass_total"] = round(float(ff.sum() / tot), 5)
    NF = P["noise_floor"]["cells"]; ND = P["not_detectable"]
    for c, a in zip(cells, ff):
        f = float(a / tot)
        if c in ND:
            out["not_detectable"][c] = {"f_hat": round(f, 5), "reason": ND[c]["reason"]}; continue
        cp = NF[c]
        rec = {"f_hat": round(f, 5), "line": cp["line"], "standing_bias": cp["standing_bias"], "detected": bool(f > cp["line"]),
               "measured_detection_limit": cp["measured_detection_limit_90pct"]}
        out["cells"][c] = rec
        if rec["detected"]: out["detected"].append(c)
    out["status"] = "OK"; out["n_markers_used"] = int(ok.sum())
    out["noise_floor"] = {"measured_on": P["noise_floor"]["laboratory_measured_on"], "n_arrays": P["noise_floor"]["n_arrays"], "quantile": P["_meta"]["quantile"]}
    out["panel_n"] = P["noise_floor"]["n_arrays"]; out["line_rule"] = P["_meta"]["line_rule"]
    if len(out["detected"]) >= 3:
        out["status"] = ("OK_BUT_UNSPECIFIC: %d templates above their floors (foreign-like mass %.1f %%) - epithelial-like material, cell not resolved"
                         % (len(out["detected"]), out["foreign_mass_total"] * 100))
    return out


def run_full(beta_dict, atlas_csv, cfg=None):
    """Full conductor: Stage A -> B (wired) -> 1s scale map -> B-identity -> 4.5 -> 5 -> 6, one bundle
    in the shape build_dashboard_v1.py consumes. Present-gated throughout.
    cfg["pipeline"] MUST be the tag Stage 1 stamped (meta["pipeline"], e.g. "stage1_noob_450K"); without it the
    identity-loci gauge is emitted with scale UNMAPPED and reportable=False (LESSON-SCALE-01, SOP s109)."""
    cfg = cfg or {}
    age = cfg.get("age", 60)
    a = stage_a_cells(beta_dict, atlas_csv, cfg)
    so = stage_2b_second_opinion(beta_dict, a, atlas_csv, cfg) if (cfg or {}).get("second_opinion", True) else {"available": False, "reason": "not requested"}
    b = stage_b_classes(beta_dict, a, cfg={"age": age})                         # marker-union statistic: DIAGNOSTIC ONLY since PROC-SWITCH-01 (feeds Stages 5/6 pending their recalibration)
    beta_rm, scale_label = stage_1s_scale_map(beta_dict, cfg.get("pipeline"))   # Stage 1s: calibration scale for the gauge
    bi = stage_b_identity(beta_rm, a, age, scale_label, lab_zero=cfg.get("lab_zero"))   # THE REPORTED GAUGE (row B, commissioned PROC-SWITCH-01)
    fd = stage_2d_foreign_detection(beta_rm, a, cfg.get("lab"), bi)                  # Stage 2d: foreign-cell detection (author-adopted 2026-09-26, PROC-MF-01/02/03)
    # 2026-09-27 (PROC-STAGE2D-02 B7; author caught a Glia BREACH on a healthy blood reference array): in a whole-blood
    # specimen a cell from a NON-BLOOD class is scored only when Stage 2d DETECTS it. The solver's fraction crossing the
    # 1 % cell floor is not presence for a foreign cell - at 1 % the cell's identity loci carry the other 99 % of the
    # specimen and read a false tier (FRACTION_AND_A). The detector is the presence test; its noise floor was measured on
    # arrays known to lack the cell. A template the detector cannot read (not_detectable) is likewise not scored.
    _blood_specimen = str(cfg.get("substrate", "whole blood")).lower().startswith("whole blood")
    if _blood_specimen and isinstance(fd, dict) and fd.get("status", "").startswith("OK"):
        _det = set(fd.get("detected") or [])
        _c2c_path = _find("IAMAtlasREBUILD_celltype_to_class.json", required=False)
        _c2c = json.load(open(_c2c_path)) if _c2c_path else {}
        _c2c = _c2c.get("celltype_to_class", _c2c)
        for _ct, _v in a["cells"].items():
            if not _v.get("present"): continue
            _cls = _v.get("class") or _c2c.get(_ct)
            if _cls in ("immune", "progenitor", "stem_adult"): continue           # the blood architecture
            if _ct in _det: continue
            _v["present"] = False; _v["A_solver_only"] = _v.get("A"); _v["A"] = None
            _v["status"] = ("NOT_DETECTED: solver fraction %.4f in a whole-blood specimen, Stage 2d %s - not scored"
                            % (_v.get("fraction") or 0, "cannot read this template (not detectable on this block)" if _ct in (fd.get("not_detectable") or {}) else "below its noise floor"))
    # every present cell under 5 % carries its fraction beside the tier: a minority cell's reading is the fraction confound
    for _ct, _v in a["cells"].items():
        if _v.get("present") and (_v.get("fraction") or 0) < 0.05: _v["minority_cell"] = True
    present_cls = [c for c, v in b["class_gauge"].items() if v.get("present")]
    bd = stage_4_5_bidirectional(beta_dict, cfg)
    sky = stage_4_6_patient_sky(beta_rm, a, cfg=cfg, atlas_csv=atlas_csv)                              # Stage 4.6: the patient's sky (row 4.6 built, commissioning WITHHELD - PROC-CMB-04 C2')
    # 2026-09-26: stage_5_mahalanobis, stage_5_hull_marker_union, stage_6_cellular_age and
    # stage_6_cellular_age_marker_union are NO LONGER CALLED on the live path. They computed cohort statistics
    # (a Mahalanobis distance against a healthy panel, an age clock) that the author ruled irrelevant to an A-score
    # and that no tab printed. The functions remain defined for the sealed procedures that reference them.
    # per-cell separation, present cells only, with class
    cells = [{"cell": ct, "class": v["class"], "A": round(v["A"], 3) if v["A"] is not None else None,
              "fraction": round(v["fraction"], 4)}
             for ct, v in a["cells"].items() if v["present"] and v["A"] is not None]
    cells.sort(key=lambda c: -c["fraction"])
    return {
        "context": {"age": age, "substrate": cfg.get("substrate", "whole blood")},
        "composition": {"class": {c: round(f * 100, 1) for c, f in a["class_fractions"].items() if f > 0.001},
                        "celltype": [{"cell": c["cell"], "pct": c["fraction"] * 100, "flag": False} for c in cells],
                    "resolvability": {"shared": a.get("celltype_shared", {}), "families": a.get("celltype_families", {}),
                                      "unresolvable": a.get("celltype_unresolvable", []), "twins_dropped": a.get("celltype_twins_dropped", {}),
                                      "exclusive_markers": a.get("celltype_exclusive_n", {})}},
        "cells": cells,
        "cells_all": a["cells"],
        "second_opinion": so,
        # Stage 2c (PROC-SMALL-01, 2026-09-23): trace-class DETECTION, a side channel that changes no
        # existing number. The composition solve pins a trace component at the non-negativity boundary, so a
        # score test with inverse-variance weights answers presence where the point estimate cannot. Its
        # verdict may name a class only at about 5 %; below that it reports epithelial-like material.
        # Stage 2c RETIRED 2026-09-30: its per-class thresholds were the 95th percentile of 38 healthy donors (a population
        # setting a number) and it read two CLASSES, not cells. Foreign material is found per cell by the joint fit (Stage 2d);
        # the no-population replacement for sub-resolution material is the out-of-span map (PROC-OUTSPAN-01).
        "trace_detection": {"_meta": {"available": False, "retired": True, "reason": "Stage 2c retired 2026-09-30: its thresholds were a population statistic; replaced by the per-cell joint fit (Stage 2d) and the out-of-span map (PROC-OUTSPAN-01)"}},
        # 2026-09-25: the flags are data, not presentation - so a batch run, a ledger and a program can all
        # ask "did anything go wrong" without parsing HTML.   # RAW betas: the panel was calibrated on the same
                                                      # input the composition solver receives, and the
                                                      # mapped scale shifts the statistic by ~24 units
                        # row 2b: NILC beside legacy, class-level agreement flag                     # 2026-09-22 (row 9): every one of the 115 atlas cells scored, placed or not - the report shows all of them
        "patient_sky": sky,                              # Stage 4.6 (row 4.6): NOT AVAILABLE without the lab's residual scale
        "foreign_detection": fd,                          # Stage 2d: per foreign cell f_hat, sigma, line, detected - only for commissioned laboratories
        "classes": bi,                                   # THE REPORTED GAUGE: identity loci, mapped, age-referenced, lab-zeroed (Issue 003 s3.5)
        "scale": scale_label, "lab_zero": ("UNSET" if cfg.get("lab_zero") is None else cfg.get("lab_zero")),
        "bidirectional": bd["bidirectional"],
    }


if __name__ == "__main__":
    import pickle
    import sys
    import numpy as np
    beta = pickle.load(open(sys.argv[1] if len(sys.argv) > 1
                            else "/tmp/calibrated_beta_GSM1051533.pkl", "rb"))
    beta_dict = {k: float(v) for k, v in beta.items()}
    ATLAS = str(Path(__file__).resolve().parent.parent / "atlas" / "IAMAtlasREBUILD.csv")  # Biological_Physics/MethylPhys/atlas/ — decompress IAMAtlasREBUILD.csv.xz there first
    out = stage_a_cells(beta_dict, ATLAS)
    present = {k: v for k, v in out["cells"].items() if v["present"]}
    print("STAGE A — %d cell types, %d present (fraction >= 1%%)" % (len(out["cells"]), len(present)))
    print("top class fractions:", sorted(out["class_fractions"].items(), key=lambda x: -x[1])[:4])
    print("present cells (A paired with fraction):")
    for ct, v in sorted(present.items(), key=lambda kv: -kv[1]["fraction"])[:8]:
        A = v["A"]
        print("  %-20s frac=%5.1f%%  A=%s" % (ct, v["fraction"] * 100,
              ("%.3f" % A if A is not None else "-")))
    print()
    b = stage_b_classes(beta_dict, out, cfg={"age": 60})
    print("STAGE B - per-class GAUGE (fuel gauge, age-matched):")
    for cls, v in b["class_gauge"].items():
        if v.get("A") is None:
            continue
        mark = "" if v["present"] else "  <- background (absent, not a finding)"
        print("  %-11s A=%.3f  %-10s %-9s frac=%4.1f%%%s" % (
            cls, v["A"], v["placement"], v["tier"], v["fraction"] * 100, mark))
    print()
    bd = stage_4_5_bidirectional(beta_dict, cfg={"age": 60})
    print("STAGE 4.5 - bidirectional (immune panel only in v1.0):")
    for cls, r in bd["bidirectional"].items():
        if r["a_directional"] is not None or (r["interpretation"] and "NO_PANEL" not in str(r["interpretation"])):
            print("  %-11s a_dir=%s pooled=%s flag=%s" % (cls,
                  ("%.3f" % r["a_directional"] if r["a_directional"] is not None else "-"),
                  ("%.3f" % r["a_pooled"] if r["a_pooled"] is not None else "-"), r["flag_bidirectional"]))
    print("  any bidirectional flagged:", bd["any_flagged"])
    print()
    # (2026-09-27: the self-test's Stage 5 / Stage 6 blocks were removed with the stages - cohort statistics no tab prints)