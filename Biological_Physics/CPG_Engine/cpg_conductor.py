#!/usr/bin/env python3
"""CPG Conductor — clean from-scratch orchestrator (2026-07).

Replaces the confusing walther_clinical.py. Wires the KISS files in the order
Heath specified, each stage a small pure function, per-cell A always paired with
its deconvolved fraction so presence and score are read together.

Inputs it needs (all in the working dir or repo):
  - IAMAtlasREBUILD.csv                 (decompressed atlas, the deconvolver reference)
  - IAMAtlasREBUILD_celltype_to_class.json
  - iamatlas_celltype_markers_v0_2.json (per-cell discriminative markers + H_min_by_class)
  - iamatlas_gauge_identity_loci_v1_0.json, age_reference_matrix.json  (Stage B)
  - iamatlas_mahalanobis_scoring.py + mahalanobis_healthy_reference_v2_0_*.json (Stage C)
  - directional_panels_v1_0.json, bidirectional_decomposition.py       (Stage C)
  - tier_breakpoints.json, disease_cell_signature_matrix_v1_13.csv,
    iamatlas_115_to_matrix_v0_2_mapping.json                           (Stage C)

STAGE A (this file, wired + tested):
  deconvolve(beta) -> class_fractions + celltype_fractions   (the RATIOS)
  score_per_celltype(beta) -> 115 per-cell A-scores          (via v0_2 markers)
  -> paired: {cell: {A, fraction, class, present}}   present = fraction >= detect_floor
"""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── Resolve chain files whether laid out flat (the CPG_TRIAL_CODE working folder) or in the
#    repository tree (Biological_Physics/CPG_Engine/{Runtime Matrices/*, Walther_iam_deconvolver/} and
#    Biological_Physics/IAM_Atlas/). Added 2026-09-19; first run of the conductor from the repo.
_SEARCH = [HERE, HERE / "Walther_iam_deconvolver", HERE / "Runtime Matrices" / "A_Scoring_Module",
           HERE / "Runtime Matrices" / "Celltype_Marker", HERE / "Runtime Matrices" / "Directional Panel",
           HERE / "Runtime Matrices" / "Mahalanobis_healthy_reference", HERE / "Runtime Matrices" / "Tier_breakpoints",
           HERE / "Runtime Matrices" / "Cellular_Age", HERE.parent / "IAM_Atlas"]
def _find(name):
    for d in _SEARCH:
        p = d / name
        if p.exists(): return p
    raise FileNotFoundError(f"{name}: not found in any of {[str(d.relative_to(HERE.parent)) for d in _SEARCH]}")
DETECT_FLOOR = 0.01  # 1% — a cell below this is treated as absent (fraction sets presence)


def _load_module(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec so @dataclass resolves __module__
    spec.loader.exec_module(mod)
    return mod


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
    dec = dec_mod.WaltherIAMDeconvolver(str(atlas_csv), celltype_class_map=str(c2c_path))
    result = dec.deconvolve(beta_dict)
    class_fr = dict(result.class_fractions)
    ct_fr = dict(result.celltype_fractions)

    # 2. Per-cell A-scores via the v0_2 discriminative markers (mean-of-per-CpG H/H_min)
    _meta, ct_markers, c2c, h_min = asc.load_artifact(str(markers_path))
    scores = asc.score_per_celltype(beta_dict, ct_markers, c2c, h_min)

    # 3. Pair A with fraction — presence comes from the ratio, not the A-score
    cells = {}
    for ct, r in scores.items():
        frac = float(ct_fr.get(ct, 0.0))
        cells[ct] = {
            "A": r.get("A"),
            "coverage": r.get("coverage"),
            "confidence": r.get("confidence"),
            "status": r.get("status"),
            "class": c2c.get(ct),
            "fraction": frac,
            "present": frac >= DETECT_FLOOR,
        }
    return {"class_fractions": class_fr, "celltype_fractions": ct_fr, "cells": cells}



def stage_b_classes(beta_dict, stage_a_out, cfg=None):
    """Stage B - per-class GAUGE (SOP §41-43): A = H(beta_mean)/H_min over the class
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
        }
    return {"class_gauge": classes, "age": age}



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


def stage_5_mahalanobis(stage_b_out, cfg=None):
    """Stage 5 (SOP §47-51, Option A) - derived age-matched departure of the patient's
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



def stage_6_cellular_age(beta_dict, cfg=None):
    """Stage 6 (SOP §Stage 6) - per-class cellular age by IAM inversion of the
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


def run_full(beta_dict, atlas_csv, cfg=None):
    """Full conductor: Stage A -> B -> 4.5 -> 5 -> 6, assembled into one bundle
    in the shape build_dashboard_v1.py consumes. Present-gated throughout."""
    cfg = cfg or {}
    age = cfg.get("age", 60)
    a = stage_a_cells(beta_dict, atlas_csv, cfg)
    b = stage_b_classes(beta_dict, a, cfg={"age": age})
    present_cls = [c for c, v in b["class_gauge"].items() if v.get("present")]
    bd = stage_4_5_bidirectional(beta_dict, cfg)
    m = stage_5_mahalanobis(b, cfg={"age": age})
    reliable_cls = [c for c, v in b["class_gauge"].items() if v.get("fraction", 0) >= 0.15]
    ag = stage_6_cellular_age(beta_dict, cfg={"age": age, "present_classes": reliable_cls})
    # per-cell separation, present cells only, with class
    cells = [{"cell": ct, "class": v["class"], "A": round(v["A"], 3) if v["A"] is not None else None,
              "fraction": round(v["fraction"], 4)}
             for ct, v in a["cells"].items() if v["present"] and v["A"] is not None]
    cells.sort(key=lambda c: -c["fraction"])
    dep = m["departure"]
    return {
        "context": {"age": age, "substrate": cfg.get("substrate", "whole blood")},
        "composition": {"class": {c: round(f * 100, 1) for c, f in a["class_fractions"].items() if f > 0.001},
                        "celltype": [{"cell": c["cell"], "pct": c["fraction"] * 100, "flag": False} for c in cells]},
        "cells": cells,
        "classes": {c: {"A": v["A"], "tier": v["tier"], "placement": v["placement"],
                        "fraction": round(v["fraction"], 4), "present": v["present"], "band": v.get("band")}
                    for c, v in b["class_gauge"].items() if v.get("A") is not None},
        "departure": {"distance": dep.get("mahalanobis_distance"), "beyond": dep.get("mahalanobis_beyond_band"),
                      "driver": (dep.get("top", [{}])[0].get("class") if dep.get("top") else None)},
        "bidirectional": bd["bidirectional"],
        "cellular_age": {"summary": ag["summary_cellular_age_present"], "chrono": age,
                         "per_class": ag["cellular_age_per_class"]},
    }


if __name__ == "__main__":
    import pickle
    import sys
    import numpy as np
    beta = pickle.load(open(sys.argv[1] if len(sys.argv) > 1
                            else "/tmp/calibrated_beta_GSM1051533.pkl", "rb"))
    beta_dict = {k: float(v) for k, v in beta.items()}
    ATLAS = str(Path(__file__).resolve().parent.parent / "IAM_Atlas" / "IAMAtlasREBUILD.csv")  # Biological_Physics/IAM_Atlas/ — decompress IAMAtlasREBUILD.csv.xz there first
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
