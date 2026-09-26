#!/usr/bin/env python3
"""CPG Conductor — clean from-scratch orchestrator (2026-07).

Replaces the v1 conductor, which was retired on 2026-09-25 after its one live function (stage_8_dual_matching) was extracted to disease_matching.py. Wires the KISS files in the order
Heath specified, each stage a small pure function, per-cell A always paired with
its deconvolved fraction so presence and score are read together.

Inputs it needs (all in the working dir or repo):
  - IAMAtlasREBUILD.csv                 (decompressed atlas, the deconvolver reference)
  - IAMAtlasREBUILD_celltype_to_class.json
  - iamatlas_celltype_markers_v0_2.json (per-cell discriminative markers + H_min_by_class)
  - iamatlas_gauge_identity_loci_v1_0.json, reference_age_curve_v1.json, identity_band_v3.json, lab_zero.py  (Stage B, the reported gauge; age_reference_matrix.json is diagnostic only)
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
#    repository tree (Biological_Physics/MethylPhys/chain/{Runtime Matrices/*, Walther_iam_deconvolver/} and
#    Biological_Physics/MethylPhys/atlas/). Added 2026-09-19; first run of the conductor from the repo.
# One deconvolver per (atlas, class-map) pair, per process. See stage_a_cells for why this is safe.
_DEC_CACHE = {}

_SEARCH = [HERE, HERE / "Walther_iam_deconvolver", HERE / "Runtime Matrices" / "A_Scoring_Module",
           HERE / "Runtime Matrices" / "Celltype_Marker", HERE / "Runtime Matrices" / "Directional Panel",
           HERE / "Runtime Matrices" / "Mahalanobis_healthy_reference", HERE / "Runtime Matrices" / "Tier_breakpoints",
           HERE / "Runtime Matrices" / "Cellular_Age",
           # the atlas: HERE is MethylPhys/chain, so its sibling; the second form covers a flat working folder
           HERE.parent / "atlas", HERE.parent.parent / "MethylPhys" / "atlas",
           HERE / 'Runtime Matrices' / 'Percell_Reference',   # 2026-09-26: percell_reference_v0_3.json has lived here since 09-22 and _find could not see it,
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


def _trace_detect(betas, substrate=None):
    """Stage 2c. Never raises into a reading: a missing panel reports unavailable and the chain proceeds."""
    try:
        mod = _load_module("stage_2c_trace_detection", _find("stage_2c_trace_detection.py"))
        return mod.detect(betas if isinstance(betas, dict) else dict(betas), substrate=substrate)
    except Exception as e:
        return {"_meta": {"available": False, "reason": f"{type(e).__name__}: {e}"}}


def stage_a_cells(beta_dict, atlas_csv, cfg=None):
    """Stage A — find the cell types in the sample, their ratios, and their A-scores.

    beta_dict : {cpg_id: beta} calibrated patient betas
    atlas_csv : path to decompressed IAMAtlasREBUILD.csv (deconvolver reference)
    Returns: {'class_fractions', 'celltype_fractions', 'cells'} where cells is
             {cell: {A, coverage, confidence, status, class, fraction, present}}.
    """
    cfg = cfg or {}
    dec_mod = _load_module("walther_iam_deconvolver", _find("walther_iam_deconvolver.py"))
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
        dec = dec_mod.WaltherIAMDeconvolver(str(atlas_csv), celltype_class_map=str(c2c_path))
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
    _pci_path = _find("iamatlas_percell_identity_loci_v1_0.json", required=False)
    _pci = asc.load_percell_identity(str(_pci_path)) if _pci_path else None
    # PER-CELL LABORATORY OFFSET (2026-09-26): one number per laboratory from the calibration record,
    # applied to every cell's A as stage_b_identity applies z_lab. Without it four laboratories
    # disagree on every cell by ~0.055; with it, by 0.007. Unknown laboratory -> UNSET, fail-closed.
    _pcr_path = _find("percell_reference_identity_v1_0.json", required=False)
    _pcr_meta = json.load(open(_pcr_path)).get("_meta", {}) if _pcr_path else {}
    _lab = (cfg or {}).get("lab") if isinstance(cfg, dict) else None
    _lab_off = (_pcr_meta.get("laboratory_offset") or {}).get("values", {}).get(_lab)
    # SCALE MAP BEFORE PER-CELL SCORING (2026-09-26). The class gauge maps stage-1 betas onto the atlas
    # scale before it reads them; this path never did, and read RAW betas that sit ~0.07 above the atlas
    # at the identity loci (the offset PROC-COV-01 measured). Raw, every present cell in healthy blood
    # read SUPPRESSED (Neutrophils 0.87, NK 0.80); mapped, the same arrays read 0.968-1.053 = NORMAL.
    scores = asc.score_per_celltype(_beta_mapped, ct_markers, c2c, h_min, celltype_identity_loci=_pci)

    # 3. Pair A with fraction — presence comes from the ratio, not the A-score
    # Uncertainty ON THE READING (2026-09-22): resample this cell's own marker CpGs, 500 draws, 95 % interval.
    # This is the error bar on A for THIS sample - a different quantity from the healthy range for that cell, and
    # from the atlas posterior SD of a class mean. The June reports conflated the third with the second: they printed
    # "95 % CI" that was mean +/- 1.96 x the posterior SD OF THE MEAN, an uncertainty-of-an-average used as a
    # population spread, which is why healthy cells appeared to sit 10 sigma out. Never that quantity again.
    import numpy as _np
    _rng = _np.random.default_rng(20260922)
    def _H1(b):
        b = _np.clip(_np.asarray(b, float), 1e-12, 1 - 1e-12)
        return -b * _np.log2(b) - (1 - b) * _np.log2(1 - b)
    _boot = {}
    for _ct, _mk in ct_markers.items():
        _cl = c2c.get(_ct); _hm = h_min.get(_cl) if _cl else None
        _v = [beta_dict[c] for c in _mk if c in beta_dict] if isinstance(_mk, (list, tuple)) else []
        if not _hm or len(_v) < 10: continue
        _h = _H1(_v) / _hm; _n = len(_h)
        _d = _h[_rng.integers(0, _n, size=(500, _n))].mean(axis=1)
        _boot[_ct] = {"ci_lo": float(_np.percentile(_d, 2.5)), "ci_hi": float(_np.percentile(_d, 97.5)),
                      "n_markers_found": int(_n), "n_markers_panel": int(len(_mk))}

    cells = {}
    for ct, r in scores.items():
        frac = float(ct_fr.get(ct, 0.0))
        cells[ct] = {
            "A": r.get("A"),
            "reading_ci": _boot.get(ct),
            "coverage": r.get("coverage"),
            "confidence": r.get("confidence"),
            "status": r.get("status"),
            # which surface and which formula produced this A - so the failsafe can SEE a reversion to marker
            # panels or to the wrong form, instead of a reader having to remember (2026-09-26)
            "A_raw": r.get("A"),
            "lab_offset": (round(_lab_off, 4) if _lab_off is not None else "UNSET"),
            "A_zeroed": (round(r["A"] - _lab_off, 4) if (_lab_off is not None and r.get("A") is not None and r.get("A") == r.get("A")) else None),
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
    (GAPE_WEB_v13 _AGE_REFERENCE).** This is NOT the identity-loci gauge that SOP §41/§106,
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
    # PRODUCTION A-score (GAPE_WEB_v13 + Reproduction Paper v3): beta_mean = customer's mean
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


def stage_b_identity(beta_mapped, stage_a_out, age, scale_label, lab_zero=None):
    """THE REPORTED GAUGE (PROC-SWITCH-01, 2026-09-21; SOP s41/s106; RULING A3): A = H(beta_mean)/H_min over the
    class IDENTITY loci on mapped beta, then the three-layer reference (Issue 003 s3.5):
        A_abs = A_mapped - c(decade) - z_lab
    c from reference_age_curve_v1.json (PROC-PANEL-03), z_lab from the laboratory's 40-array healthy panel
    (lab_zero.py). Placement in identity_band_v3.json (pooled p10-p90 of A_abs over four zeroed healthy cohorts).
    lab_zero None -> 'UNSET': A_mapped is still returned, but A_abs/placement are None and reportable=False.
    s108 reporting rule: on whole blood only immune and the joint haematopoietic-progenitor component are read."""
    import math
    ident = json.load(open(_find("iamatlas_gauge_identity_loci_v1_0.json")))
    ident = {k: v for k, v in ident.items() if isinstance(v, dict) and "loci" in v}
    band = json.load(open(_find("identity_band_v3.json")))
    lz = _load_module("lab_zero", _find("lab_zero.py"))
    curve = lz.load_curve(_find("reference_age_curve_v1.json"))
    def H(b): b = min(max(b, 1e-12), 1 - 1e-12); return -b * math.log2(b) - (1 - b) * math.log2(1 - b)
    fr = stage_a_out["class_fractions"]
    out = {}
    groups = {"immune": ["immune"], "haematopoietic_progenitor": ["progenitor", "stem_adult"]}
    mapped = scale_label.startswith("MAPPED")
    for name, members in groups.items():
        loci = [c for cls in members for c in ident[cls]["loci"]]
        vals = [beta_mapped[c] for c in loci if c in beta_mapped]
        frac = sum(fr.get(c, 0) for c in members)
        if not vals or frac < 0.01:
            out[name] = {"present": False, "fraction": round(frac, 4)}; continue
        hm = ident["progenitor" if name != "immune" else "immune"]["H_min"]        # joint uses progenitor's floor (PREREG s3)
        A = H(sum(vals) / len(vals)) / hm
        rec = {"present": True, "fraction": round(frac, 4), "A_mapped": round(A, 4), "n_loci": len(vals),
               "gauge_surface": "identity_loci", "scale": scale_label, "H_min": hm}
        # PROC-FOREIGN-01: attach the composition check on EVERY path, so a reader sees the foreign
        # fraction even when the tier is withheld for an earlier reason - the adenoma is withheld for a
        # missing laboratory zero, and the guard's own field was simply absent from its record.
        try:
            _g = json.load(open(_find('composition_guard_v1.json')))
            _fgn = 1.0 - sum(float(fr.get(_c, 0.0)) for _c in _g['blood_lineage'])
            rec['foreign_fraction'] = round(_fgn, 4)
            rec['composition_verified'] = bool(_fgn <= _g['foreign_fraction_max'])
        except FileNotFoundError:
            pass                       # no guard file: withhold nothing, claim nothing
        if name == "immune":
            c = lz.age_reference(age, curve) if age is not None else None
            rec["age_reference_c"] = None if c is None else round(c, 4)
            if lab_zero is None or c is None or not mapped:
                rec.update({"lab_zero": "UNSET" if lab_zero is None else round(lab_zero, 4), "A_abs": None, "placement": None,
                            "band": band["pooled"], "reportable": False, "tier": None,
                            "reason": ("no laboratory zero (Issue 003 s3.5; lab_zero.py)" if lab_zero is None else "beta not on the calibration scale" if not mapped else "no age")})
            else:
                A_abs = A - c - lab_zero
                rec.update({"lab_zero": round(lab_zero, 4), "A_abs": round(A_abs, 4), "band": band["pooled"],
                            "placement": "BELOW_BAND" if A_abs < band["pooled"]["p10"] else "ABOVE_BAND" if A_abs > band["pooled"]["p90"] else "IN_BAND",
                            "reportable": True, "band_status": "identity_band_v3 (four zeroed labs, n=1,379; LOO 0.75-0.84, PROC-PANEL-03)"})
                _t,_n=_load_module("cpg_tiers", HERE/"cpg_tiers.py").tier_of(A_abs, True, rec.get("H_min")); rec.update({"tier": _t, "tier_note": _n})   # Stage 7 (PROC-TIER-01): one JSON-driven tier
                # PROC-FOREIGN-01 (2026-09-25): the gauge is commissioned on WHOLE BLOOD. A specimen carrying
                # material outside the blood lineage still gets an honest reading, but no tier word - measured
                # on 318 healthy arrays mixed with atlas stromal, secretory and terminal material: at every
                # level where a tenth of hosts changed tier, this threshold caught at least 95.9 % of them.
                try:
                    _g = json.load(open(_find('composition_guard_v1.json')))
                    _foreign = 1.0 - sum(float(fr.get(_c, 0.0)) for _c in _g['blood_lineage'])
                    rec['foreign_fraction'] = round(_foreign, 4)
                    rec['composition_verified'] = bool(_foreign <= _g['foreign_fraction_max'])
                    if not rec['composition_verified']:
                        rec.update({'tier': None, 'tier_note': None, 'reportable': False,
                                    'reason': ('composition unverified: %.1f%% of this specimen is assigned '
                                               'outside the blood lineage, above the %.1f%% the gauge was '
                                               'commissioned for. The reading stands; no tier is printed '
                                               '(PROC-FOREIGN-01).')
                                              % (100 * _foreign, 100 * _g['foreign_fraction_max'])})
                except FileNotFoundError:
                    rec['composition_verified'] = None      # no guard file: withhold nothing, say nothing

        else:
            rec.update({"A_abs": None, "placement": None, "reportable": False, "tier": None, "reason": "no band for this component yet (s108)"})
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


def stage_5_mahalanobis(identity_out, cfg=None):
    """Stage 5 - THE REPORTED DEPARTURE (PROC-MAHA-01, 2026-09-21; SOP s47-51 re-based on row B).
    Input is the REPORTED gauge (stage_b_identity): for each component with a commissioned band,
        z = (A_abs - 1.000) / sigma,   sigma = (p90 - p10) / (2 * 1.2816)  from identity_band_v3
    distance = sqrt(sum z^2) over the n assessable components; thresholds sqrt(chi2(0.95|0.99, n)).
    Components without a band (haematopoietic-progenitor joint; all non-blood classes) are not assessable.
    On whole blood today n = 1: the departure is how many band-widths from the healthy line the immune
    reading sits, and the report says so. Not reportable when the gauge is not (UNSET / UNMAPPED).
    Keys: the long names the report interface (build_methylphys.py) reads and the short aliases run_full exposes."""
    import math
    try:
        from scipy.stats import chi2
        thr = lambda n: (math.sqrt(chi2.ppf(0.95, n)), math.sqrt(chi2.ppf(0.99, n)))
    except Exception:
        thr = lambda n: (math.sqrt(n + 2.0 * math.sqrt(2.0 * n)), math.sqrt(n + 3.0 * math.sqrt(2.0 * n)))
    contribs = []; unreportable = []
    for name, v in identity_out.items():
        if not v.get("present") or not v.get("band"): continue
        if not v.get("reportable"): unreportable.append(name); continue
        b = v["band"]; sigma = (b["p90"] - b["p10"]) / (2 * 1.2816); z = (v["A_abs"] - 1.0) / sigma
        contribs.append({"class": name, "patient_A": v["A_abs"], "age_matched_mean": 1.0, "sigma": round(sigma, 5), "z": round(z, 3),
                         "band_widths_from_line": round((v["A_abs"] - 1.0) / (b["p90"] - b["p10"]), 3)})
    n = len(contribs)
    if n == 0:
        status = ("gauge not reportable (" + ", ".join(unreportable) + ")") if unreportable else "no assessable component with a commissioned band"
        out = {"mahalanobis_distance": None, "n_features_assessable": 0, "n_assessable": 0, "reportable": False, "status": status,
               "alarm_threshold_p95": None, "alarm_threshold_p99": None, "mahalanobis_beyond_band": None, "top_axis_contributions": []}
    else:
        d = math.sqrt(sum(c["z"] ** 2 for c in contribs)); t95, t99 = thr(n)
        contribs.sort(key=lambda c: -abs(c["z"]))
        out = {"mahalanobis_distance": round(d, 4), "n_features_assessable": n, "n_assessable": n, "reportable": True,
               "alarm_threshold_p95": round(t95, 4), "alarm_threshold_p99": round(t99, 4),
               "mahalanobis_beyond_band": bool(d > t95), "beyond_p99": bool(d > t99), "top_axis_contributions": contribs,
               "status": "one banded axis (immune) - the distance is |z_immune|" if n == 1 else f"{n} banded axes",
               "reference": "identity_band_v3 (four zeroed labs, n=1,379); mu = 1.000; sigma from p10-p90"}
    # PROC-MAHA-02: the laboratory's empirical false-alarm rate travels with the number (row 5b = the chip term behind it)
    band_meta = json.load(open(_find("identity_band_v3.json")))["_meta"]
    lab_key = (cfg or {}).get("lab"); _coh = band_meta.get("cohorts", {})
    if isinstance(_coh, str):
        import ast as _ast; _coh = _ast.literal_eval(_coh)
    # 2026-09-22 (row 9 build): band cohorts are keyed "GSE87571_Uppsala"; cfg["lab"] may be the bare accession - match on prefix
    rec = (_coh.get(lab_key) or next((v for k, v in _coh.items() if lab_key and (k.startswith(lab_key) or lab_key.startswith(k))), None)) if lab_key else None
    if rec and "tail_p95" in rec:
        out.update({"lab_false_alarm_p95": rec["tail_p95"], "lab_false_alarm_p99": rec["tail_p99"], "lab_false_alarm_source": f"measured on {rec['n']} healthy arrays at {lab_key}",
                    "lab_false_alarm_sentence": f"At this laboratory {round(100*rec['tail_p95'])} of 100 healthy donors read beyond p95 on this axis ({round(100*rec['tail_p99'])} of 100 beyond p99); the excess over 5 is the chip term (row 5b)."})
    else:
        lo, hi = band_meta.get("four_lab_tail_range_p95", [0.044, 0.098])
        out.update({"lab_false_alarm_p95": None, "lab_false_alarm_p99": None, "lab_false_alarm_source": "not measured for this laboratory",
                    "lab_false_alarm_sentence": f"This laboratory's healthy false-alarm rate is not measured; across four commissioned laboratories {round(100*lo)}-{round(100*hi)} of 100 healthy donors read beyond p95 on this axis (chip term, row 5b)."})
    out.update({"distance": out["mahalanobis_distance"], "beyond": out["mahalanobis_beyond_band"],
                "driver": contribs[0]["class"] if contribs else None})
    return {"departure": out, "class_ascores_scored": {c["class"]: c["patient_A"] for c in contribs}}

def stage_5_hull_marker_union(stage_b_out, cfg=None):
    """DIAGNOSTIC ONLY since PROC-MAHA-01 (2026-09-21): the pre-switch eight-class derived hull on the marker-union readings. Never the reported departure.
    Stage 5 (SOP §47-51, Option A) - derived age-matched departure of the patient's
    CLASS GAUGE A-scores from the age band. Scores PRESENT classes only (absent-class
    background is excluded). One number + top-axis decomposition. No cohort."""
    cfg = cfg or {}
    age = cfg.get("age", stage_b_out.get("age", 60))
    mh = _load_module("iamatlas_mahalanobis_scoring", _find("iamatlas_mahalanobis_scoring.py"))
    ge = _load_module("cpg_gauge_engine", _find("cpg_gauge_engine.py"))
    ref = _find("mahalanobis_healthy_reference_v2_0_age_matched_derived.json")
    hull = mh.MahalanobisHealthyHull(str(ref), gauge=ge)
    # Mahalanobis presence gate (manifest adjudicator fix): a class contributes ONLY if
    # abundance >= 3% AND it is outside the age-matched NORMAL band (placement != IN_BAND).
    # This removes trace non-substrate classes reading blood-background (the false-positive
    # inflator) and in-band healthy classes (which are not a departure).
    maha_floor = cfg.get("maha_floor", 0.03)
    require_out = cfg.get("require_outside_band", True)
    ca = {}
    for cls, v in stage_b_out.get("class_gauge", {}).items():
        frac = v.get("fraction", 0.0)
        # AGE-MATCHED gate (offset-free gauge): count only genuine departures from the
        # age band - (a) >=3% AND ABOVE_BAND elevation, OR (b) >=15% AND INVERSION.
        # A mild below-band dip reads NORMAL and never counts. The manifest's absolute
        # [0.95,1.04) gate is offset-era and false-positives here (healthy age-matched ~0.90).
        # ELEVATION-ONLY: disease drives A up (ABOVE_BAND). The inversion arm is
        # disabled - the age band is offset-era, so healthy blood reads below it and a
        # naive inversion arm false-positives (immune 0.807 INVERSION on a healthy 58M).
        # Re-enable inversion only after age_reference_matrix is rebuilt offset-free.
        plc = v.get("placement")
        ca[cls] = v.get("A") if (frac >= maha_floor and plc == "ABOVE_BAND") else None
    result = hull.score(ca, age=age)
    return {"departure": result, "class_ascores_scored": {k: v for k, v in ca.items() if v is not None}}



def stage_6_cellular_age(identity_out, cfg=None):
    """Stage 6 - CELLULAR AGE IS NOT REPORTABLE AT SINGLE-ARRAY RESOLUTION (PROC-AGE-01, 2026-09-21).
    The healthy immune identity-gauge curve rises 0.47 mA per year against a within-laboratory SD of 0.0235:
    inverting it resolves age to ~50 years per array (1,379 healthy donors, four labs: 15.9% within +/-10 yr,
    Spearman rho 0.27). The population aging trajectory (CPG-VAL-015, mammalian paper) is REPRODUCED by this curve; what is below resolution is one person's
    position on it. This stage reports the resolution, not an age; the inversion
    is available as diagnostic_cellular_age for lineage only."""
    r = json.load(open(_find("age01_results.json"))) if _find("age01_results.json", required=False) else {"resolution_yr": 50, "A1_within10": 0.159, "A3_rho": 0.271}
    im = identity_out.get("immune", {})
    return {"reportable": False, "cellular_age": None, "chronological_age": (cfg or {}).get("age"),
            "resolution_yr": round(r["resolution_yr"]), "healthy_within_10yr": r["A1_within10"], "spearman_rho": r["A3_rho"],
            "sentence": (f"Cellular age is not reported: on the immune identity gauge the healthy age curve moves 0.47 mA/yr against a "
                         f"within-laboratory spread of 0.0235, so one array resolves age to about {round(r['resolution_yr'])} years "
                         f"({round(100*r['A1_within10'])} of 100 healthy donors within +/-10 yr; PROC-AGE-01). The healthy aging TRAJECTORY itself is real and "
                         f"reproduced here (0.47 mA/yr, monotone by decade, four laboratories; CPG-VAL-015 found the same slope on Hannum) - it is a population "
                         f"measurement, and one array cannot resolve a person's position on it."),
            "gauge_A_abs": im.get("A_abs")}

def stage_6_cellular_age_marker_union(beta_dict, cfg=None):
    """DIAGNOSTIC ONLY since PROC-AGE-01 (2026-09-21): inverts the superseded marker-union age matrix. Never reported.
    Stage 6 (SOP §Stage 6) - per-class cellular age by IAM inversion of the
    age_reference_matrix: the age at which population beta_mean equals the patient's
    beta_mean, per class. iam_cellular_age_scoring.py."""
    cfg = cfg or {}
    ca = _load_module("iam_cellular_age_scoring", _find("iam_cellular_age_scoring.py"))
    cls_name = [n for n in dir(ca) if n.lower().startswith("iamcellularage")][0]
    # beta_mean SOURCE = IDENTITY loci (the axis the age curve is built on), NOT discriminative markers
    loci = json.load(open(_find("iamatlas_gauge_identity_loci_v1_0.json")))
    markers_per_class = {c: v["loci"] for c, v in loci.items()
                         if isinstance(v, dict) and "loci" in v}
    clock = getattr(ca, cls_name)(
        ref_matrix_path=str(_find("age_reference_matrix.json")),
        markers_per_class=markers_per_class)
    res = clock.score_patient(beta_dict, chronological_age=cfg.get("age"))
    per_class = dict(getattr(res, "cellular_age_per_class", {}))
    # present-gate: absent classes invert background noise -> summarize PRESENT only
    present = cfg.get("present_classes")
    if present:
        vals = [per_class[c] for c in present if c in per_class and isinstance(per_class[c], (int, float))]
        summary = round(sum(vals) / len(vals), 1) if vals else None
    else:
        summary = getattr(res, "summary_cellular_age", None)
    return {"cellular_age_per_class": per_class, "summary_cellular_age_present": summary,
            "summary_cellular_age_all": getattr(res, "summary_cellular_age", None),
            "present_classes": present, "chronological_age": getattr(res, "chronological_age", cfg.get("age"))}


def stage_4_6_patient_sky(beta_rm, stage_a_out, cfg=None, atlas_csv=None):
    """Stage 4.6 - the patient's sky (PROC-CMB-04, 2026-09-21). z_i = (beta_i - sum_c f_c mu_ci - m_lab,i) / s_lab,i on the mapped
    beta; class panels gated by the measured presence floors. Needs the laboratory's residual scale (Runtime Matrices/Patient_CMB/
    residual_scale_<lab>.npz, built from the same 40-array panel as the lab zero); without it the sky is NOT AVAILABLE, never approximated.
    Calibration on record: healthy held-out arrays show 2.6-3.2% of CpGs beyond |z|=2 (scale ~1.1x conservative; CMB-04 C2' failed as sealed)."""
    import importlib.util, os as _os, json as _json, pandas as _pd
    cfg = cfg or {}
    spec = importlib.util.spec_from_file_location("stage_4_6_patient_cmb", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "stage_4_6_patient_cmb.py"))
    S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
    lab = cfg.get("lab"); rt = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "Runtime Matrices", "Patient_CMB")
    sp = _os.path.join(rt, f"residual_scale_{lab}.npz") if lab else None
    if not sp or not _os.path.exists(sp):
        return {"available": False, "status": f"NOT AVAILABLE - no residual scale for laboratory {lab!r} (build it from the lab's 40-array healthy panel: PROC-CMB-04)", "classes": {}}
    scale = S.load_scale(sp); floors = _json.load(open(_os.path.join(rt, "presence_floors_v1.json")))["floors"]
    means = S.load_atlas_means(atlas_csv or _find("IAMAtlasREBUILD.csv"))
    ident = _json.load(open(_find("iamatlas_gauge_identity_loci_v1_0.json"))); loci = {c: v["loci"] for c, v in ident.items() if isinstance(v, dict) and "loci" in v}
    sky = S.patient_sky(_pd.Series(beta_rm, dtype=float), stage_a_out["class_fractions"], means, scale, loci, S.load_mapping(), presence_floors_by_class=floors)
    out = {"available": True, "lab": lab, "scale_panel_n": scale["n_panel"], "presence_floors": floors,
           "all": sky["all"], "classes": {c: {k: v for k, v in d.items() if k != "pixels"} for c, d in sky["classes"].items()},
           "calibration_note": "healthy held-out arrays read 2.6-3.2% of CpGs beyond |z|=2 (PROC-CMB-04); a healthy sky is quiet at that level, not at 5%",
           "_sky": sky}   # full arrays for render_plate; stripped by the report builder
    return out

def stage_8_matching(stage_a_out, cfg=None):
    """NOT A CHAIN STAGE (author's ruling 2026-09-21). Disease-pattern concordance against disease_cell_signature_matrix_v1_13 - a matrix
    compiled from pre-build and early post-build VALs, i.e. the preliminary record. The chain reports what it measured (cells detected,
    fractions, A per cell and class, placement, flags) and names no disease. This function is kept callable for RECORD-SIDE study only
    (logging cohort behaviour as trusted-chain cohorts accumulate); run_full does not call it and the report never shows its output.
    Earlier text (superseded): ROW 8 OPEN (PROC-MATCH-01, 2026-09-21):
    the departure profile is (A_cell - 1.0) over PRESENT cells, but healthy per-cell A on this surface sits at ~0.44-0.52 with class
    H_min 0.77-0.98, so on healthy whole blood only ~3 cells enter the profile. The reference level must be re-derived on this surface
    (healthy per-cell level from the four-lab panels) before any match is reported. Until then the output is DIAGNOSTIC and not reportable.
    Origin gate fails CLOSED (missing/unreadable disease_origin_cells.json -> status NOT AVAILABLE, zero candidates)."""
    cfg = cfg or {}
    W = _load_module("disease_matching", _find("disease_matching.py"))
    md = HERE / "Disease Matrix" / "DISEASE_MATRIX"
    c2c = json.load(open(_find("IAMAtlasREBUILD_celltype_to_class.json"))); HM = json.load(open(_find("iamatlas_celltype_markers_v0_2.json"))).get("H_min_by_class", {})
    s4 = {"celltype_ascores": {cell: {"A": r.get("A"), "below_floor": bool(r.get("A") is not None and r["A"] < HM.get(c2c.get(cell), 0)),
                                       "celltype_fraction": r.get("fraction")} for cell, r in stage_a_out["cells"].items()}}
    out = W.stage_8_dual_matching(s4, None, None, patient_meta={"substrate": cfg.get("substrate", "whole_blood")},
                                  config={"disease_matrix_csv": str(md / "disease_cell_signature_matrix_v1_13.csv"), "matrix_mapping_json": str(md / "iamatlas_115_to_matrix_v0_2_mapping.json")})
    return {"available": out.status == "OK", "status": out.status, "reportable": False, "row_status": "OPEN - departure reference not commissioned on the separation surface (PROC-MATCH-01)",
            "n_present_cells": len(out.patient_departure), "patient_departure": out.patient_departure,
            "route_B_top": out.route_B_concordance[:5], "n_scored": len(out.route_B_all_scored)}


def stage_2b_second_opinion(beta_dict, stage_a_out, atlas_csv, cfg=None):
    """Row 2b - the second opinion. NILC (needlet internal linear combination, the Planck component-separation
    method) run beside Walther's constrained NNLS and compared AT THE CLASS LEVEL, which is where PROC-SEP-03
    showed the atlas is separable. Ships as a second column plus an agreement flag, never as the composition the
    report stands on: Walther is conservative and commissioned, NILC is variance-weighted and deliberately
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
    rows = {c: {"walther": round(W.get(c, 0.0), 4), "nilc": round(N.get(c, 0.0), 4),
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
            "reported_composition": "Walther (constrained NNLS) - NILC is a second opinion only"}

def stage_2d_foreign_detection(beta_mapped, stage_a_out, lab, bi=None):
    """Stage 2d - FOREIGN-CELL DETECTION. Adopted by the author's decision 2026-09-26, scoped to laboratories with a
    commissioned panel in detection_panel_v1.json (PROC-MF-01/02/03: 0.5-1 % of Breast / colon / neurons / prostate
    in blood on four 450K laboratories, honest sigma, no bias; the fifth-laboratory bar failed each time, so a
    laboratory not in the panel gets NO line and says so).

    Inverse-variance weighted template amplitude against the specimen's own blood background:
        b = blood-only NNLS reconstruction on the panel's markers; r = v - b; t = mu_cell - b / sum(f_blood)
        f_hat = sum w t r / sum w t^2 - centre_lab ;  detected iff f_hat > line_lab(cell)
    Gate: runs only when the composition guard verified the specimen as blood-like (a control reading 0.2 on every
    foreign cell at once is substrate mismatch, PROC-MF-03 - it must never reach a line). Reports per foreign cell
    (f_hat, sigma, line, detected, measured_detection_limit). Never changes the blood composition (MF-02/03 B6).
    """
    import numpy as _np
    from scipy.optimize import nnls as _nnls
    out = {"status": None, "laboratory": lab, "cells": {}, "detected": []}
    try:
        P = json.load(open(_find("detection_panel_v1.json")))
    except Exception as e:
        out["status"] = "NOT_RUN: detection_panel_v1.json not found (%s)" % type(e).__name__; return out
    if lab not in P["laboratories"]:
        out["status"] = "NOT_COMMISSIONED: detection is not commissioned for laboratory %r - no line is borrowed" % lab
        out["commissioned_laboratories"] = sorted(P["laboratories"]); return out
    imm = ((bi or {}).get("immune") or {})
    if imm.get("composition_verified") is False:
        out["status"] = "WITHHELD: the composition guard did not verify this specimen as blood-like (foreign fraction %s); a line is not applied to a specimen that is not blood" % imm.get("foreign_fraction")
        return out
    M = P["markers"]; v = _np.array([beta_mapped.get(m, _np.nan) for m in M], dtype=float); ok = ~_np.isnan(v)
    if ok.sum() < 0.8 * len(M):
        out["status"] = "NOT_RUN: only %d of %d panel markers present (need 80 %%)" % (int(ok.sum()), len(M)); return out
    Ab = _np.array([P["blood_ref"][c] for c in P["_meta"]["blood_columns"]]).T[ok]
    Lp = P["laboratories"][lab]; w = _np.array(Lp["weights"])[ok]
    fb, _ = _nnls(Ab, v[ok]); b = Ab @ fb; r = v[ok] - b; bg = b / max(fb.sum(), 1e-9)
    for c, ref in P["foreign_ref"].items():
        t = _np.array(ref)[ok] - bg
        a = float(_np.sum(w * t * r) / _np.sum(w * t * t)); cp = Lp["cells"][c]
        f = a - cp["centre"]
        rec = {"f_hat": round(f, 5), "sigma": cp["sigma"], "z": round(f / cp["sigma"], 2) if cp["sigma"] else None, "line": cp["line"],
               "detected": bool(f > cp["line"]), "measured_detection_limit": cp["measured_detection_limit"]}
        out["cells"][c] = rec
        if rec["detected"]: out["detected"].append(c)
    out["status"] = "OK"; out["n_markers_used"] = int(ok.sum()); out["panel_n"] = Lp["n_panel"]
    out["line_rule"] = P["_meta"]["line_rule"]
    # specificity note (PROC-MF-03): every foreign cell rising together is substrate mismatch, not detection
    if len(out["detected"]) >= max(3, len(P["foreign_ref"]) // 2):
        out["status"] = "OK_BUT_UNSPECIFIC: %d of %d foreign cells detected together - substrate-mismatch signature, not a detection of any one cell" % (len(out["detected"]), len(P["foreign_ref"]))
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
    present_cls = [c for c, v in b["class_gauge"].items() if v.get("present")]
    bd = stage_4_5_bidirectional(beta_dict, cfg)
    sky = stage_4_6_patient_sky(beta_rm, a, cfg=cfg, atlas_csv=atlas_csv)                              # Stage 4.6: the patient's sky (row 4.6 built, commissioning WITHHELD - PROC-CMB-04 C2')
    m = stage_5_mahalanobis(bi, cfg={"age": age, "lab": cfg.get("lab")})                            # THE REPORTED DEPARTURE on the identity gauge (row 5, PROC-MAHA-01)
    m_diag = stage_5_hull_marker_union(b, cfg={"age": age})                    # diagnostic only
    reliable_cls = [c for c, v in b["class_gauge"].items() if v.get("fraction", 0) >= 0.15]
    ag = stage_6_cellular_age(bi, cfg={"age": age})                                     # NOT REPORTABLE at single-array resolution (PROC-AGE-01); prints the resolution
    ag_diag = stage_6_cellular_age_marker_union(beta_dict, cfg={"age": age, "present_classes": reliable_cls})   # diagnostic only
    # per-cell separation, present cells only, with class
    cells = [{"cell": ct, "class": v["class"], "A": round(v["A"], 3) if v["A"] is not None else None,
              "fraction": round(v["fraction"], 4)}
             for ct, v in a["cells"].items() if v["present"] and v["A"] is not None]
    cells.sort(key=lambda c: -c["fraction"])
    dep = m["departure"]
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
        "trace_detection": _trace_detect(beta_dict, (cfg or {}).get("substrate")),
        # 2026-09-25: the flags are data, not presentation - so a batch run, a ledger and a program can all
        # ask "did anything go wrong" without parsing HTML.   # RAW betas: the panel was calibrated on the same
                                                      # input the composition solver receives, and the
                                                      # mapped scale shifts the statistic by ~24 units
                        # row 2b: NILC beside Walther, class-level agreement flag                     # 2026-09-22 (row 9): every one of the 115 atlas cells scored, placed or not - the report shows all of them
        "patient_sky": sky,                              # Stage 4.6 (row 4.6): NOT AVAILABLE without the lab's residual scale
        "foreign_detection": fd,                          # Stage 2d: per foreign cell f_hat, sigma, line, detected - only for commissioned laboratories
        "classes": bi,                                   # THE REPORTED GAUGE: identity loci, mapped, age-referenced, lab-zeroed (Issue 003 s3.5)
        "diagnostic_marker_union": {c: {"A": v["A"], "tier": v["tier"], "placement": v["placement"],
                        "fraction": round(v["fraction"], 4), "present": v["present"], "band": v.get("band"), "gauge_surface": "marker_union",
                        "status": "DIAGNOSTIC ONLY - not the reported A (PROC-N7-01, PROC-SWITCH-01)"}
                    for c, v in b["class_gauge"].items() if v.get("A") is not None},
        "scale": scale_label, "lab_zero": ("UNSET" if cfg.get("lab_zero") is None else cfg.get("lab_zero")),
        "pending_recalibration": {"stage_5_mahalanobis": False, "stage_6_cellular_age": False,
                                  "note": "Stage 5 re-based on the identity gauge (PROC-MAHA-01/02); Stage 6 closed as NOT REPORTABLE at single-array resolution (PROC-AGE-01) - no reported path reads the marker-union statistics"},
        "departure": dep,                                # identity-gauge departure; long keys + short aliases (PROC-MAHA-01)
        "mahalanobis": dep,                              # the key cpg_report_builder._departure_section reads
        "diagnostic_hull_marker_union": m_diag["departure"],
        "bidirectional": bd["bidirectional"],
        "cellular_age": ag,                              # reportable False; resolution + sentence (PROC-AGE-01)
        "diagnostic_cellular_age": {"summary": ag_diag["summary_cellular_age_present"], "chrono": age, "per_class": ag_diag["cellular_age_per_class"],
                                    "status": "DIAGNOSTIC ONLY - marker-union age matrix inversion; never reported (PROC-AGE-01)"},
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
    m = stage_5_mahalanobis(b, cfg={"age": 60})
    dep = m["departure"]
    dist = dep.get("mahalanobis_distance") if isinstance(dep, dict) else dep
    beyond = dep.get("mahalanobis_beyond_band") if isinstance(dep, dict) else None
    print("STAGE 5 - Mahalanobis Option A (age-matched class-gauge departure):")
    print("  classes scored:", list(m["class_ascores_scored"].keys()))
    print("  departure=%s  beyond age band=%s" % (
          ("%.2f" % dist if dist is not None else "-"), beyond))
    print()
    ag = stage_6_cellular_age(beta_dict, cfg={"age": 60})
    print("STAGE 6 - cellular age (IAM inversion, chrono=60):")
    print("  summary cellular age:", ag["summary_cellular_age"])
    for cls, yr in list(ag["cellular_age_per_class"].items())[:5]:
        print("    %-11s %s yr" % (cls, round(yr,1) if isinstance(yr,(int,float)) else yr))
