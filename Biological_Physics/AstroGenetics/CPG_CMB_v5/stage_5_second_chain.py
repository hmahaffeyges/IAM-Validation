"""Stage 5 — Second chain (confirmation). Fires ONLY when Stage 8 Route B flags.

Three components, in priority order:
  A. Residual-map matched filter (SOP 8.2) — if a residual map exists for the
     flagged disease (breast, AD, or the immune cross-disease universal alarm),
     measures the Pearson correlation between the patient's per-CpG departure
     FROM THE DERIVED ATLAS BASELINE and the disease's signed residual, with a
     Fisher confidence interval. Fires when the CI is clear of zero in EITHER
     direction (elevation OR suppression). This is the detection instrument.
  B. Directional class signal — whether the flagged disease's architectural class
     A-score departs from the healthy floor (|A - 1.0| >= 0.04), in either
     direction. Read straight off Stage 4; no global adjudicator.
  C. Literature-anchor evidence — labels the patient's flagged-class A-score
     against PUBLISHED anchors (healthy / disease / cancer) for that class.
     Context for the clinician, never the statistical reference.

Derived-only throughout. The residual-map baseline is the derived atlas per-CpG
class mean. No cohort comparables anywhere, and no Mahalanobis adjudicator: the
disease matrix (Route B) is the primary detector and these three channels confirm.

The flag gate fires on ANY Route B match above threshold (including the
non-specific ones) — "when in doubt, run it out." The matched filter and the
directional class signal then confirm or fail to confirm the resemblance.
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
    "immune_universal_alarm": "immune",   # sentinel: the cross-disease universal alarm sweep
}

def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    root = Path(cfg["brightness_archives_dir"]).parents[1]   # CPG_ROOT
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
        return {"map": breast, "sign_col": "d_GSE51057", "strong_col": "concordant_strong",
                "baseline_cell": "whole_blood", "label": "breast residual map"}
    if flagged_disease == "alzheimers_disease" and ad and ad.exists():
        return {"map": ad, "sign_col": "d_AIBL", "strong_col": "concordant_strong",
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
    root = Path(cfg["brightness_archives_dir"]).parents[1]
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
                        "literature anchor + directional class signal still stand"}
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
        use.append((r["cpg"], sgn))            # keep continuous signed d for the matched filter
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
    deps = []; dds = []
    for cg, disease_d in use:
        if cg not in base or cg not in beta.index:
            continue
        deps.append(float(beta.loc[cg]) - base[cg])   # patient departure from derived baseline
        dds.append(disease_d)                          # sealed disease direction (signed d)
    n = len(deps)
    if n < 30:
        return {"status": "insufficient_overlap", "map": spec["label"],
                "note": f"only {n} overlapping CpGs between patient and map"}
    # SOP 8.2 matched filter: Pearson rho between patient per-CpG departure and the
    # disease's signed residual. Pearson centres both vectors, so a global cohort-vs-atlas
    # baseline offset cancels -- the readout is the disease pattern, not the offset.
    mx = sum(deps) / n; my = sum(dds) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(deps, dds))
    sxx = sum((a - mx) ** 2 for a in deps); syy = sum((b - my) ** 2 for b in dds)
    rho = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
    z = math.atanh(max(min(rho, 0.999), -0.999)); se = 1.0 / math.sqrt(n - 3)
    lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
    fires = lo > 0 or hi < 0          # CI clear of zero in either direction = real departure
    if fires and rho > 0:   sense = f"consistent with the {flagged_disease} residual pattern"
    elif fires:             sense = f"opposite to the {flagged_disease} residual pattern"
    else:                   sense = f"not distinguishable from null for {flagged_disease}"
    return {
        "status": "OK",
        "map": spec["label"],
        "cpgs_compared": n,
        "rho": round(rho, 3),
        "ci": [round(lo, 3), round(hi, 3)],
        "fires": fires,
        "interpretation": (
            f"matched-filter rho={round(rho, 3)} (95% CI [{round(lo, 3)}, {round(hi, 3)}]) "
            f"over {n} CpGs; {sense}"),
    }


# ===========================================================================
#  RUN-everything residual sweep — the safety net.
#  The per-cell matcher (Stage 8 Route B) leaves distributional diseases quiet: breast
#  pre-dx lives in secretory homogenization (a per-CpG variance collapse), not a per-cell
#  mean shift, so the matcher correctly does not flag it. If the matched filter only ran on
#  the per-cell top match, breast's residual map would never be tested and the case would be
#  missed. So we ALWAYS sweep every available residual map (breast, AD, immune cross-disease
#  universal alarm) for a whole-blood-compatible patient, independent of the per-cell ranking.
#  The maps are whole-blood-derived, so the sweep self-skips for cfDNA / tissue substrates
#  (the cell-of-origin detector handles those).
# ===========================================================================
_RESIDUAL_SWEEP_DISEASES = ["breast_cancer", "alzheimers_disease", "immune_universal_alarm"]
_WHOLE_BLOOD_SUBSTRATES = {
    "whole_blood", "whole_blood_buffy_coat", "buffy_coat", "pbmc", "blood", "",
}


def _residual_sweep(beta, cfg, substrate):
    """Always run the SOP 8.2 matched filter against each available residual map, regardless
    of what the per-cell matcher ranked. Returns the per-map results and the list that fired."""
    sub = str(substrate or "").lower()
    if sub not in _WHOLE_BLOOD_SUBSTRATES:
        return {"status": "skipped_substrate",
                "note": f"residual maps are whole-blood-derived; not applicable to substrate '{sub}'",
                "results": {}, "fired": []}
    results = {}
    for d in _RESIDUAL_SWEEP_DISEASES:
        try:
            results[d] = _residual_concordance(beta, d, cfg)
        except Exception as e:
            results[d] = {"status": "error", "note": f"sweep error for {d}: {e}"}
    # A DETECTION is a positive-rho fire: the patient moves the SAME way as the disease's
    # signed residual pattern. A negative-rho fire is anti-correlation (opposite to the
    # pattern) -- not a detection. Null check across EPIC-Italy healthy controls confirmed
    # healthy blood anti-correlates with the AD/immune maps (rho ~ -0.1 to -0.17) and sits at
    # zero on the breast map, while breast cases fire breast POSITIVE (rho +0.06 to +0.09).
    fired = [d for d, r in results.items()
             if r.get("status") == "OK" and r.get("fires") and r.get("rho", 0.0) > 0]
    return {"status": "OK", "results": results, "fired": fired}


# ===========================================================================
#  Orchestrator — per-cell Route B flag OR the residual sweep
# ===========================================================================
def run_second_chain(bundle, beta, cfg):
    """Return the confirmation dict, or None when no flag fired (gate stays closed)."""
    s8 = bundle.get("stage8")
    matches = getattr(s8, "route_B_concordance", None)
    if matches is None and isinstance(s8, dict):
        matches = s8.get("route_B_concordance")
    matches = matches or []

    stage4 = bundle["stage4"]
    substrate = (bundle.get("context") or {}).get("substrate")

    # RUN-everything safety net: always sweep the available residual maps, independent of the
    # per-cell ranking. This is what catches breast pre-dx, whose distributional signal the
    # per-cell matcher correctly leaves quiet.
    try:
        sweep = _residual_sweep(beta, cfg, substrate)
    except Exception as e:
        sweep = {"status": "error", "note": f"residual sweep error: {e}", "results": {}, "fired": []}
    sweep_fired = sweep.get("fired") or []

    # gate: open if a SPECIFIC concern-worthy per-cell match exists OR the residual sweep fired.
    # The per-cell flag we confirm is the top SPECIFIC, at-least-moderate match. Non-specific
    # generic-axis matches (the neutrophil-to-lymphocyte / myeloproliferative shift) are handled
    # by the report's Mode 1 line, never escalated here as a named disease.
    flagged_matches = [m for m in matches
                       if m.get("specificity") == "SPECIFIC"
                       and m.get("resemblance") in ("STRONG_RESEMBLANCE", "MODERATE_RESEMBLANCE")]
    flagged = flagged_matches[0]["disease"] if flagged_matches else None

    if not flagged and not sweep_fired:
        return None                          # nothing specific from either instrument

    if flagged:
        anchor = _literature_anchor(stage4, flagged, cfg)
        try:
            resid = _residual_concordance(beta, flagged, cfg)
        except Exception as e:
            resid = {"status": "error", "note": f"residual step error: {e}"}
        flagged_class = DISEASE_TO_CLASS.get(flagged)
        fc_rec = stage4["class_ascores"].get(flagged_class) if flagged_class else None
        flagged_class_departed = (
            fc_rec is not None and fc_rec.get("A") is not None
            and fc_rec.get("assessable", True) and abs(fc_rec["A"] - 1.0) >= 0.04)
        anchor_near_disease = (
            anchor.get("status") == "OK"
            and anchor.get("nearest_published_anchor", {}).get("context") in ("disease", "cancer"))
        resid_supports = (resid.get("status") == "OK" and resid.get("fires", False))
        supports = [s for s, ok in (
            ("residual matched filter", resid_supports),
            ("directional class signal", flagged_class_departed),
            ("literature anchor", anchor_near_disease)) if ok]
    else:
        anchor = {"status": "no_flag",
                  "note": "no per-cell Route B flag; the residual sweep is the trigger"}
        resid = {"status": "no_flag",
                 "note": "no per-cell Route B flag; see residual_sweep for the matched-filter result"}
        supports = []

    # ---- integrated verdict: confirmed findings lead; unconfirmed resemblances never headline ----
    if sweep_fired:
        named = [("breast pattern" if d == "breast_cancer"
                  else "AD pattern" if d == "alzheimers_disease"
                  else "immune cross-disease universal alarm") for d in sweep_fired]
        sweep_sentence = (f"Residual matched-filter sweep fired for: {', '.join(named)}. "
                          f"The matched filter is the detector here.")
    else:
        sweep_sentence = ""

    if flagged and supports:
        # a per-cell flag with its own confirmation is the strongest statement
        overall = (f"Consistent with the pattern of {flagged}: Route B matrix resemblance "
                   f"confirmed by {', '.join(supports)}.")
        if sweep_sentence:
            overall += " " + sweep_sentence
    elif sweep_fired:
        # the confirmed sweep leads; an unconfirmed per-cell resemblance is demoted, never named,
        # because on a sparse present-cell profile the generic stress pattern resembles many
        # myeloid-involved conditions at once and is not a fingerprint of any one of them
        overall = sweep_sentence
        if flagged:
            overall += (" The per-cell matcher also noted non-specific composition resemblances "
                        "that its confirmatory channels did not support; these are consistent with "
                        "the systemic pattern, not a specific second disease.")
    elif flagged:
        # no sweep, only an unconfirmed per-cell resemblance -> explicitly non-specific
        overall = (f"Route B noted a composition resemblance to {flagged}, but its confirmatory "
                   f"channels (residual matched filter, directional class signal, literature "
                   f"anchor) did not support it; treated as non-specific, not a disease call.")
    else:
        overall = ("No per-cell Route B flag and no sweep hit; nothing specific from either "
                   "instrument.")

    return {
        "fired": True,
        "trigger": {
            "flagged_disease": flagged,
            "flagged_confirmed": bool(flagged and supports),
            "route_B_resemblance": flagged_matches[0].get("resemblance") if flagged_matches else None,
            "route_B_specificity": flagged_matches[0].get("specificity") if flagged_matches else None,
            "n_specific_matches": len(flagged_matches),
            "residual_sweep_fired": sweep_fired,
            "gate_policy": "fires on a SPECIFIC concern-worthy Route B match OR any residual-sweep "
                           "hit (when in doubt, run it out)",
        },
        "literature_anchor": anchor,
        "residual_map": resid,
        "residual_sweep": sweep,
        "overall_verdict": overall,
    }
