"""Stage 5 — Second chain (confirmation). Fires ONLY when Stage 8 Route B flags.

Three components, in priority order:
  A. Derived Mahalanobis global departure  — the adjudicator. Answers "is this a
     real, significant departure of the patient's cell architecture from the
     healthy floor, or a bulk-composition resemblance?" Derived reference only
     (mu=1.0, sigma=0.02 from the tier band, diagonal Sigma). A chi-square test
     on the assessable cells turns the distance into a verdict.
  B. Literature-anchor evidence  — labels the patient's most-departed class A-score
     against PUBLISHED anchors (healthy / disease / cancer) for that class. Context
     for the clinician, never the statistical reference.
  C. Residual-map concordance  — if a residual map exists for the flagged disease
     (breast, AD, or the immune cross-disease universal alarm), measures whether
     the patient's per-CpG departure FROM THE DERIVED ATLAS BASELINE fires in the
     disease's direction. Graceful fallback when no map exists yet.

Derived-only throughout. The Mahalanobis reference is derived; the residual-map
baseline is the derived atlas per-CpG class mean. No cohort comparables anywhere.

The flag gate fires on ANY Route B match above threshold (including the
non-specific ones) — "when in doubt, run it out." The Mahalanobis is exactly what
adjudicates a non-specific resemblance.
"""
import os, json, math, tarfile, io, csv
from pathlib import Path


# ---- disease -> architectural class (for the literature-anchor lookup) ----
DISEASE_TO_CLASS = {
    "breast_cancer": "secretory", "pancreatic_cancer": "secretory",
    "prostate_cancer": "secretory", "gastric_cancer": "secretory",
    "colorectal_cancer": "cycling", "lung_cancer": "cycling",
    "esophageal_cancer_eac": "cycling", "esophageal_cancer_escc": "cycling",
    "bladder_cancer": "cycling", "cervical_cancer": "cycling",
    "kidney_cancer": "cycling", "hcc": "cycling",
    "leukemia_aml": "immune", "leukemia_b_all": "immune", "leukemia_cll": "immune",
    "leukemia_cml": "immune", "leukemia_t_all": "immune", "lymphoma_dlbcl": "immune",
    "mds": "immune", "mpn": "immune", "multiple_myeloma": "immune", "thymoma": "immune",
    "glioma_gbm": "terminal", "glioma_lgg": "terminal",
    "alzheimers_disease": "terminal", "frontotemporal_dementia": "terminal",
    "parkinsons_disease": "terminal", "als": "terminal",
    "psp_cbd_tauopathies": "terminal", "schizophrenia": "terminal",
    "aortic_dissection_BAV": "stromal", "pah": "stromal",
}

# ---- chi-square 0.95 critical values (DOF 1..60), so we need no scipy ----
_CHI2_95 = {1:3.841,2:5.991,3:7.815,4:9.488,5:11.070,6:12.592,7:14.067,8:15.507,
            9:16.919,10:18.307,11:19.675,12:21.026,13:22.362,14:23.685,15:24.996,
            16:26.296,17:27.587,18:28.869,19:30.144,20:31.410,21:32.671,22:33.924,
            23:35.172,24:36.415,25:37.652,26:38.885,27:40.113,28:41.337,29:42.557,
            30:43.773,31:44.985,32:46.194,33:47.400,34:48.602,35:49.802,36:50.998,
            37:52.192,38:53.384,39:54.572,40:55.758,41:56.942,42:58.124,43:59.304,
            44:60.481,45:61.656,46:62.830,47:64.001,48:65.171,49:66.339,50:67.505,
            51:68.669,52:69.832,53:70.993,54:72.153,55:73.311,56:74.468,57:75.624,
            58:76.778,59:77.931,60:79.082}

def _chi2_crit(dof):
    if dof in _CHI2_95: return _CHI2_95[dof]
    # Wilson-Hilferty for dof > 60
    return dof * (1 - 2.0/(9*dof) + 1.6449 * math.sqrt(2.0/(9*dof)))**3


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
#  A. Mahalanobis global departure (the adjudicator)
# ===========================================================================
def _mahalanobis(stage4, cfg, class_fractions=None):
    """Derived global-departure adjudicator on the CLASS-level A-scores. A class
    counts only if it is GENUINELY PRESENT in the substrate (not a trace background
    fraction the deconvolver assigned — at trace abundance its A-score is reading
    background, not architecture) and ACTUALLY OUTSIDE the derived NORMAL tier band
    [0.95, 1.04). A class inside the band is normal and contributes nothing; a trace
    class is absence, not departure. This stops suppressed non-substrate classes
    (e.g. stem_pluri, terminal in blood) from inflating the distance. Reads the
    derived sigma from the derived reference. Diagonal, derived, no cohort."""
    ref = json.load(open(cfg["mahalanobis_reference_json"]))
    sigma = float(ref["sigma"])              # 0.02, derived from the NORMAL tier band
    NORMAL_LO, NORMAL_HI = 0.95, 1.04        # derived NORMAL tier band
    MIN_FRACTION = 0.03                       # genuinely present (not trace background)
    class_fractions = class_fractions or {}
    classes = stage4["class_ascores"]
    contribs, excluded = [], []
    for cls, rec in classes.items():
        a = rec.get("A")
        if a is None or not rec.get("assessable", True):
            continue
        frac = class_fractions.get(cls)
        if frac is not None and frac < MIN_FRACTION:
            excluded.append({"class": cls, "A": round(a, 3),
                             "reason": f"trace abundance ({frac*100:.1f}%) — reading background, not architecture"})
            continue
        if a > NORMAL_HI:
            z = (a - NORMAL_HI) / sigma       # band-relative: only the part outside normal counts
        elif a < NORMAL_LO:
            z = (a - NORMAL_LO) / sigma
        else:
            z = 0.0
        contribs.append((cls, a, z))
    n = len(contribs)
    d_sq = float(sum(z * z for _, _, z in contribs))
    d = math.sqrt(d_sq)
    crit = _chi2_crit(n) if n > 0 else None
    beyond = (n > 0 and d_sq > crit)
    order = sorted(contribs, key=lambda x: abs(x[2]), reverse=True)
    return {
        "feature_space": "class_level_A_scores (present classes, band-relative to [0.95,1.04))",
        "distance": round(d, 3),
        "n_present_classes": n,
        "d_squared": round(d_sq, 2),
        "chi2_0.95_threshold": round(crit, 2) if crit else None,
        "beyond_healthy_band": bool(beyond),
        "verdict": ("departure beyond the derived healthy band"
                    if beyond else "within the derived healthy band"),
        "driving_classes": [
            {"class": c, "A": round(a, 3), "z_from_floor": round(z, 2)}
            for c, a, z in order],
        "excluded_classes": excluded,
        "reference": "derived (normal band [0.95,1.04), sigma=0.02, diagonal) — not a cohort",
    }


# ===========================================================================
#  B. Literature-anchor evidence (labels, not the reference)
# ===========================================================================
def _literature_anchor(stage4, flagged_disease, cfg):
    cls = DISEASE_TO_CLASS.get(flagged_disease)
    if cls is None:
        return {"status": "no_class_mapping",
                "note": f"no class mapping for {flagged_disease}; anchor step skipped"}
    try:
        anchors_all = json.load(open(cfg["literature_anchors_json"]))["data"]
    except Exception as e:
        return {"status": "anchors_unavailable", "note": str(e)}
    anchors = anchors_all.get(cls)
    if not anchors:
        return {"status": "no_anchor_for_class", "class": cls,
                "note": f"no published anchors loaded for class {cls}"}
    rec = stage4["class_ascores"].get(cls)
    pA = rec.get("A") if rec else None
    if pA is None:
        return {"status": "class_not_scored", "class": cls,
                "note": f"patient {cls} class not assessable in this substrate"}
    nearest = min(anchors, key=lambda a: abs(a["A"] - pA))
    return {
        "status": "OK",
        "class": cls,
        "patient_class_A": round(pA, 3),
        "nearest_published_anchor": {
            "label": nearest["label"], "A": nearest["A"],
            "context": nearest["context"], "source": nearest["source"]},
        "anchor_ladder": [
            {"label": a["label"], "A": a["A"], "context": a["context"],
             "source": a["source"]} for a in sorted(anchors, key=lambda x: x["A"])],
        "interpretation": (
            f"patient {cls} A={round(pA,3)} is consistent with the published "
            f"\"{nearest['label']}\" anchor (A={nearest['A']}, {nearest['source']})"),
    }


# ===========================================================================
#  C. Residual-map concordance (derived atlas baseline)
# ===========================================================================
# flagged disease -> (residual map csv, direction-sign column, baseline cell mean)
def _residual_map_spec(flagged_disease, cfg):
    root = Path(cfg["mahalanobis_reference_json"]).parents[2]   # CPG_ROOT
    cards = root / "Disease Cards : Residual Maps"
    breast = (cards / "Breast_EPIC/breast_epic_residual_maps/"
              "breast_epic_residual_map_chr_annotated.csv")
    ad = None
    for c in (cards / "AD_EPIC").rglob("*residual_map*chr*annotated*.csv"):
        ad = c; break
    immune = (cards / "Immune_Atlas/Immune_Atlas_residual_maps/immune_atlas_residual_maps/"
              "immune_atlas_cross_disease_universal_alarm_residual_map_v0_1.csv")
    cls = DISEASE_TO_CLASS.get(flagged_disease)
    if flagged_disease == "breast_cancer" and breast.exists():
        return {"map": breast, "sign_col": "breast_sign", "strong_col": "breast_concordant_strong",
                "baseline_cell": "whole_blood", "label": "breast residual map"}
    if flagged_disease == "alzheimers_disease" and ad and ad.exists():
        return {"map": ad, "sign_col": "ad_sign", "strong_col": "ad_concordant_strong",
                "baseline_cell": "whole_blood", "label": "AD residual map"}
    if cls == "immune" and immune.exists():
        # cross-disease universal alarm: fires when the patient's immune CpGs move
        # in the same direction the cross-disease signature does
        return {"map": immune, "sign_col": "breast_sign", "strong_col": None,
                "baseline_cell": "whole_blood", "label": "immune cross-disease universal alarm"}
    return None


def _atlas_baseline_for_cpgs(cpg_set, baseline_cell, cfg):
    """Derived per-CpG healthy baseline = atlas class-mean for `baseline_cell`,
    read from the immune class brightness archive we already ship."""
    root = Path(cfg["mahalanobis_reference_json"]).parents[2]
    arch = root / "IAM_Atlas/iamatlas_class_archives/immune_v0_1_REBUILD.tar.xz"
    if not arch.exists():
        return {}
    col = baseline_cell + "_mean"
    base = {}
    with tarfile.open(arch) as t:
        member = [m for m in t.getmembers() if m.name.endswith(".csv")][0]
        f = io.TextIOWrapper(t.extractfile(member), encoding="utf-8")
        rdr = csv.DictReader(f)
        if col not in rdr.fieldnames:
            return {}
        for row in rdr:
            cg = row["cpg_id"]
            if cg in cpg_set:
                v = row.get(col, "")
                if v not in ("", None):
                    try: base[cg] = float(v)
                    except ValueError: pass
    return base


def _residual_concordance(beta, flagged_disease, cfg):
    spec = _residual_map_spec(flagged_disease, cfg)
    if spec is None:
        return {"status": "no_map",
                "note": f"residual map not yet built for {flagged_disease}; "
                        "Mahalanobis + anchor still stand"}
    # load the map's CpGs + disease direction
    rows = []
    with open(spec["map"]) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    sign_col = spec["sign_col"]
    strong_col = spec["strong_col"]
    use = []
    for r in rows:
        if strong_col and str(r.get(strong_col)).strip().lower() not in ("true", "1"):
            continue
        try:
            sgn = float(r.get(sign_col, "nan"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(sgn) or sgn == 0:
            continue
        use.append((r["cpg"], 1.0 if sgn > 0 else -1.0))
    if not use:
        # immune map has no per-disease strong flag; fall back to top |breast_mean_abs_d|
        scored = []
        for r in rows:
            try: m = float(r.get("breast_mean_abs_d", "nan"))
            except (TypeError, ValueError): continue
            if math.isfinite(m):
                try: sgn = float(r.get("breast_sign", "0"))
                except (TypeError, ValueError): sgn = 0
                scored.append((r["cpg"], m, 1.0 if sgn > 0 else -1.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        use = [(cg, s) for cg, _m, s in scored[:1000]]
    cpg_set = set(cg for cg, _ in use)
    base = _atlas_baseline_for_cpgs(cpg_set, spec["baseline_cell"], cfg)
    agree = total = 0
    for cg, disease_dir in use:
        if cg not in base or cg not in beta.index:
            continue
        dep = float(beta.loc[cg]) - base[cg]      # patient departure from derived baseline
        if abs(dep) < 0.02:                        # ignore near-floor noise
            continue
        total += 1
        if (dep > 0) == (disease_dir > 0):
            agree += 1
    if total == 0:
        return {"status": "insufficient_overlap", "map": spec["label"],
                "note": "no overlapping above-noise CpGs between patient and map"}
    frac = agree / total
    return {
        "status": "OK",
        "map": spec["label"],
        "cpgs_compared": total,
        "directional_concordance": round(frac, 3),
        "interpretation": (
            f"{round(frac*100)}% of {total} disease-direction CpGs in the "
            f"{spec['label']} move the patient's way "
            f"({'consistent with' if frac >= 0.6 else 'not strongly consistent with'} "
            f"the {flagged_disease} residual pattern)"),
    }


# ===========================================================================
#  Orchestrator — fires only on a Route B flag
# ===========================================================================
def run_second_chain(bundle, beta, cfg):
    """Return the confirmation dict, or None when no flag fired (gate stays closed)."""
    s8 = bundle.get("stage8")
    matches = getattr(s8, "route_B_concordance", None)
    if matches is None and isinstance(s8, dict):
        matches = s8.get("route_B_concordance")
    if not matches:
        return None                          # gate closed — no second chain, no report section

    flagged = matches[0]["disease"]
    stage4 = bundle["stage4"]
    s2 = bundle.get("stage2") or {}
    class_fractions = s2.get("class_fractions") if isinstance(s2, dict) else None

    maha = _mahalanobis(stage4, cfg, class_fractions)
    anchor = _literature_anchor(stage4, flagged, cfg)
    try:
        resid = _residual_concordance(beta, flagged, cfg)
    except Exception as e:
        resid = {"status": "error", "note": f"residual step error: {e}"}

    # ---- integrated verdict: Mahalanobis adjudicates, anchor + residual qualify ----
    flagged_class = DISEASE_TO_CLASS.get(flagged)
    driving = {d["class"]: d for d in maha["driving_classes"]}
    flagged_class_elevated = (
        flagged_class in driving and driving[flagged_class]["A"] >= 1.04)
    anchor_near_disease = (
        anchor.get("status") == "OK"
        and anchor.get("nearest_published_anchor", {}).get("context") in ("disease", "cancer"))
    resid_supports = (
        resid.get("status") == "OK" and resid.get("directional_concordance", 0) >= 0.6)

    if not maha["beyond_healthy_band"]:
        overall = (
            f"Within the derived healthy band (Mahalanobis d={maha['distance']}, "
            f"threshold not crossed). The Route B resemblance to {flagged} is most "
            f"consistent with bulk-composition shape, not a real global departure.")
    elif flagged_class_elevated and (anchor_near_disease or resid_supports):
        overall = (
            f"Consistent with a real departure in the pattern of {flagged} "
            f"(Mahalanobis d={maha['distance']} beyond the band; the {flagged_class} "
            f"class is elevated and the anchor/residual support the {flagged} direction).")
    else:
        drv = ", ".join(f"{d['class']} {d['A']}" for d in maha["driving_classes"]
                        if abs(d["A"] - 1.0) >= 0.04)[:120] or "low-level shifts"
        overall = (
            f"A global departure is present (Mahalanobis d={maha['distance']} beyond "
            f"the band, driven by {drv}), but it is NOT in the pattern of {flagged}: "
            f"the {flagged_class or 'flagged'} class sits near its healthy anchor and the "
            f"residual concordance is low. The {flagged} flag is not supported by the "
            f"confirmation layer.")

    return {
        "fired": True,
        "trigger": {
            "flagged_disease": flagged,
            "route_B_resemblance": matches[0].get("resemblance"),
            "route_B_specificity": matches[0].get("specificity"),
            "n_matches_above_threshold": len(matches),
            "gate_policy": "fires on any Route B match above threshold (when in doubt, run it out)",
        },
        "mahalanobis": maha,
        "literature_anchor": anchor,
        "residual_map": resid,
        "overall_verdict": overall,
    }
