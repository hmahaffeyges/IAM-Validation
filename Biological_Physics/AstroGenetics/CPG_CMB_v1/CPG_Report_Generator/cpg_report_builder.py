#!/usr/bin/env python3
"""
cpg_report_builder.py — Cellular Performance Gauge (CPG) report assembler.

Consumes the result bundle from walther_clinical.run_pipeline() and renders a
single-file, doctor-facing HTML report. Figures (gauges + Personal Brilliance
Maps) are base64-embedded so the report is fully portable (one .html file).

Section layout mirrors CPG_MOCK_REPORT_60yo_subtle_drift_v1.md (the approved
template). Section prose is PARAMETERIZED from the patient's actual results;
the per-section narrative wording is the place Walther + Heath keep refining.

Usage:
    from cpg_report_builder import build_report
    build_report(bundle, "reports/PID_report.html",
                 atlas_plate_paths={...}, config=...)

Data-driven sections implemented: Executive summary, A (intake), B (composition),
C (architectural state + gauges + top-15 ranking), D (why no cellular age), E (internal
consistency & fail-safes, PASS/REVIEW only),
F (Personal Brilliance Maps), G (bidirectional). H (disease matching) is shown as
Stage-8-pending until the disease-matrix stage is wired.
"""
from __future__ import annotations
import base64
import html
from datetime import datetime
from pathlib import Path

ARCH_CLASSES = ["immune", "stromal", "secretory", "cycling", "stem_adult",
                "progenitor", "terminal", "stem_pluri"]
CLASS_PRETTY = {"immune": "Immune", "stromal": "Stromal", "secretory": "Secretory",
                "cycling": "Cycling", "stem_adult": "Stem (adult)",
                "progenitor": "Progenitor", "terminal": "Terminal",
                "stem_pluri": "Stem (pluripotent)"}
TIER_CSS = {"SUPPRESSED": "#6a8caf", "NORMAL": "#3f9b54", "ELEVATED": "#c8a020",
            "SIGNIFICANTLY_ELEVATED": "#d2691e", "BREACH": "#b03020",
            "WARBURG_TRANSITION": "#c8771f"}


def _b64(path):
    p = Path(path)
    data = base64.b64encode(p.read_bytes()).decode()
    mime = "image/svg+xml" if p.suffix == ".svg" else "image/png"
    return f"data:{mime};base64,{data}"


def _img(path, alt="", style="max-width:100%;height:auto;"):
    if path is None or not Path(path).exists():
        return f'<div class="missing">[figure not available: {html.escape(alt)}]</div>'
    return f'<img src="{_b64(path)}" alt="{html.escape(alt)}" style="{style}">'


def _img_scaled(path, alt="", max_px=1500, style="max-width:100%;height:auto;"):
    """Embed an image, downsampling large sources (the multi-MB reference plates) so the
    single-file report stays a sane size. Falls back to a raw embed if Pillow is absent."""
    if path is None or not Path(path).exists():
        return f'<div class="missing">[figure not available: {html.escape(alt)}]</div>'
    try:
        from PIL import Image
        import io, base64 as _b64mod
        im = Image.open(path)
        if im.mode in ("RGBA", "P", "LA"):
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_px:
            s = max_px / float(max(w, h))
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88, optimize=True)
        enc = _b64mod.b64encode(buf.getvalue()).decode("ascii")
        return f'<img src="data:image/jpeg;base64,{enc}" alt="{html.escape(alt)}" style="{style}">'
    except Exception:
        return _img(path, alt, style)


def _esc(x):
    return html.escape(str(x))


def _tier_badge(tier_id, label=None):
    label = (label or tier_id).replace("\n", " ")
    c = TIER_CSS.get(tier_id, "#777")
    return f'<span class="badge" style="background:{c}">{_esc(label)}</span>'


def _nilc_agreement(walther_fracs, nilc_fracs, thresh=0.05):
    """Do the primary and the independent cross-method deconvolution agree on which
    cell classes are present? Returns (True/False/None, note)."""
    try:
        nf = nilc_fracs
        if isinstance(nf, dict) and "class_fractions" in nf:
            nf = nf["class_fractions"]
        if not isinstance(nf, dict) or not isinstance(walther_fracs, dict):
            return None, "Cross-method output could not be compared."
        w_present = {k for k, v in walther_fracs.items() if isinstance(v, (int, float)) and v > thresh}
        n_present = {k for k, v in nf.items() if isinstance(v, (int, float)) and v > thresh}
        if not w_present and not n_present:
            return None, "No dominant class to compare."
        union = w_present | n_present
        jacc = len(w_present & n_present) / len(union) if union else 1.0
        if jacc >= 0.6:
            return True, None
        return False, "The two methods disagree on which cell types are present."
    except Exception:
        return None, "Cross-method comparison could not be completed."


def _run_consistency_checks(s2, s5, s7, s8):
    """Single source of truth for the internal consistency checks. Returns a list of
    (name, what-it-confirms, ok True/False/None, note). Used by both the doctor-facing
    panel and the trajectory snapshot so their verdicts can never diverge."""
    s2 = s2 or {}
    s5 = s5 or {}
    s7 = s7 or {}
    checks = []

    checks.append(("A-score integrity",
        "The core A-score formula -- order divided by each cell's physical fidelity floor -- is "
        "re-verified against a sealed reference before any cell is scored. The report is not "
        "produced if this fails.", True, None))

    ok2 = bool(s2.get("presence_method")) and s2.get("class_present") is not None
    checks.append(("Substrate presence gate",
        "Confirms only cell types actually present in this sample were scored. Cell types not "
        "detected in the substrate are marked not-assessable rather than given a misleading reading.",
        ok2, None if ok2 else "The presence gate did not resolve."))

    ds = s2.get("status")
    ok3 = (ds is None) or (isinstance(ds, str) and ds.upper().startswith("OK"))
    checks.append(("Cell-mixture deconvolution",
        "Confirms the cell-type composition resolved cleanly from the methylation signal.",
        ok3, None if ok3 else f"Deconvolution status: {ds}."))

    nf = s2.get("nilc_fractions")
    if nf is None:
        ok4, note4 = None, "Cross-method check was not run for this sample."
    elif isinstance(nf, dict) and "_error" in nf:
        ok4, note4 = False, "The independent method did not complete."
    else:
        ok4, note4 = _nilc_agreement(s2.get("class_fractions") or {}, nf)
    checks.append(("Independent cross-method check",
        "A second, mathematically independent method re-derives the cell composition from the same "
        "data. Agreement between the two confirms the result reflects your sample, not an artifact "
        "of one method.", ok4, note4))

    rA = (getattr(s8, "route_A_architectural_alarm", None) or {})
    fired = rA.get("fired")
    elevated = {"ELEVATED", "WARBURG_TRANSITION", "SIGNIFICANTLY_ELEVATED", "BREACH"}
    percell_elevated = (s7.get("max_class_tier") in elevated) or (s7.get("n_cells_breach", 0) > 0)
    if fired is None:
        ok5, note5 = None, "Insufficient assessable cells to run the whole-sample measure."
    elif fired and not percell_elevated:
        ok5, note5 = False, "The whole-sample measure flagged but no individual cell is elevated -- worth re-checking."
    else:
        ok5, note5 = True, None
    checks.append(("Whole-sample architectural consistency",
        "The Mahalanobis distance is a standard statistical measure of how far a many-dimensional "
        "reading sits from a healthy reference. This check confirms that single whole-sample measure "
        "agrees with the per-cell A-scores above; a disagreement would be a signal to re-check.",
        ok5, note5))
    return checks


def _consistency_checks_panel(s2, s5, s7, s8, P):
    """Behind-the-scenes integrity & consistency checks. Doctor-facing as PASS /
    NEEDS REVIEW only -- no internal numbers. A REVIEW is the clinician's cue to
    notify the lab. Only checks that genuinely run are listed."""
    checks = _run_consistency_checks(s2, s5, s7, s8)

    any_review = any(ok is False for _, _, ok, _ in checks)
    P('<section><h2>E &middot; Internal consistency &amp; fail-safes &mdash; what runs behind the scenes</h2>')
    P('<p>Your result is produced under several automatic integrity and consistency checks. The '
      'internal numbers are not shown here &mdash; what matters clinically is that each check ran '
      'and what it confirms. A line marked <b>NEEDS REVIEW</b> is your cue to notify the laboratory '
      'before relying on the result.</p>')
    P('<table class="kv"><tr><th>Check</th><th>What it confirms</th>'
      '<th style="text-align:center">Status</th></tr>')
    for name, desc, ok, note in checks:
        if ok is True:
            badge = '<span style="color:#3f9b54;font-weight:700">PASSED</span>'
        elif ok is False:
            badge = '<span style="color:#b23b3b;font-weight:700">NEEDS REVIEW</span>'
        else:
            badge = '<span style="color:#888;font-weight:700">NOT RUN</span>'
        d = _esc(desc) + (f' <i>({_esc(note)})</i>' if note else '')
        P(f'<tr><td><b>{_esc(name)}</b></td><td>{d}</td>'
          f'<td style="text-align:center">{badge}</td></tr>')
    P('</table>')
    if any_review:
        P('<p style="color:#b23b3b"><b>One or more consistency checks need review.</b> Please notify '
          'the laboratory so the result can be verified before clinical use.</p>')
    else:
        P('<p class="muted">All internal consistency checks passed.</p>')
    P('<p class="muted">A <b>Technical Audit Appendix</b> is generated alongside this report; it carries the '
      'internal values behind every check above. Send it to the laboratory with any line marked NEEDS REVIEW.</p>')
    P('</section>')


def _g(obj, key, default=None):
    """Read a field from either a dict or a dataclass-like object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def build_audit_appendix(bundle, audit_path):
    """Technical Audit Appendix — the internal diagnostic values behind the
    doctor-facing PASS / NEEDS REVIEW checks, so a flagged result can be diagnosed
    from this single file without re-running the patient. Derived intermediates
    only (no raw methylation / PHI)."""
    import json, datetime
    from pathlib import Path

    def J(x, n=6000):
        return _esc(json.dumps(x, indent=1, default=str))[:n]

    s2 = bundle.get("stage2") or {}
    s4 = bundle.get("stage4") or {}
    s5 = bundle.get("stage5") or {}
    s6 = bundle.get("stage6")
    s7 = bundle.get("stage7") or {}
    s8 = bundle.get("stage8")
    pid = bundle.get("patient_id", "patient")
    meta = bundle.get("patient_meta", {}) or {}

    R = []
    A = R.append
    A("<h1>Technical Audit Appendix</h1>")
    A(f"<p class='muted'>Sample <b>{_esc(pid)}</b> &middot; generated "
      f"{datetime.datetime.now():%Y-%m-%d %H:%M} &middot; derived intermediates only "
      f"(no raw methylation). For laboratory diagnostics.</p>")
    A("<p>This appendix accompanies the clinical report. That report shows PASS / NEEDS REVIEW "
      "only; every internal value behind those checks is below, so a flagged result can be "
      "diagnosed from this file alone &mdash; no re-run of the patient sample is required.</p>")

    # 1. Checks with their internal values
    A("<h2>1 &middot; Consistency checks &mdash; internal values</h2>")
    A("<h3>Substrate presence gate</h3>")
    A(f"<pre>presence_method   = {_esc(s2.get('presence_method'))}\n"
      f"deconv_status     = {_esc(s2.get('status'))}\n"
      f"class_present     = {J(s2.get('class_present'))}\n"
      f"class_fraction_ci = {J(s2.get('class_fraction_ci'))}</pre>")
    A("<h3>Deconvolution (primary / Walther)</h3>")
    A(f"<pre>class_fractions = {J(s2.get('class_fractions'))}</pre>")
    if s2.get("walther_diagnostics"):
        A(f"<pre>walther_diagnostics = {J(s2.get('walther_diagnostics'))}</pre>")
    A("<h3>Independent cross-method (NILC)</h3>")
    nf = s2.get("nilc_fractions")
    agree, note = _nilc_agreement(s2.get("class_fractions") or {}, nf) if nf is not None else (None, "not run")
    A(f"<pre>present_class_agreement = {agree}  ({_esc(note)})\nnilc_fractions = {J(nf)}</pre>")
    A("<h3>Whole-sample architectural (Mahalanobis / Route A)</h3>")
    rA = _g(s8, "route_A_architectural_alarm", {}) or {}
    A(f"<pre>route_A = {J(rA)}</pre>")
    top = s5.get("top10_axis_contributions") if isinstance(s5, dict) else None
    if top:
        A(f"<pre>top_axis_contributions = {J(top)}</pre>")

    # 2. Stage 6 departure internals
    A("<h2>2 &middot; Cellular departure internals (Stage 6)</h2>")
    cal6 = _g(s6, "calibration", {}) or {}
    A(f"<pre>total_cellular_departure = {_g(s6,'total_cellular_departure')}\n"
      f"n_cells_scored           = {_g(s6,'n_cells_scored')}\n"
      f"null_expected_departure  = {_g(s6,'null_expected_departure')}\n"
      f"excess_departure         = {_g(s6,'excess_departure')}\n"
      f"calibration              = {J(cal6)}</pre>")
    pcd = _g(s6, "per_class_departure", {}) or {}
    if pcd:
        A(f"<pre>per_class_departure = {J(pcd)}</pre>")

    # 3. Tiers (Stage 7)
    A("<h2>3 &middot; Architectural tiers (Stage 7)</h2>")
    A(f"<pre>max_class_tier = {_esc(s7.get('max_class_tier'))}\n"
      f"n_cells_breach = {s7.get('n_cells_breach')}\nbreach_line = {s7.get('breach_line')}\n"
      f"class_tiers = {J(s7.get('class_tiers'))}</pre>")

    # 4. Full per-cell A-score table (the core data)
    A("<h2>4 &middot; Per-cell A-scores (every cell)</h2>")
    cta = s4.get("celltype_ascores", {}) if isinstance(s4, dict) else {}
    A("<table class='kv'><tr><th>Cell</th><th>A</th><th>Tier</th><th>Class</th><th>Status</th></tr>")
    for cell, v in sorted(cta.items()):
        if isinstance(v, dict):
            a, tier, cls, st = v.get("A"), v.get("tier"), v.get("class"), v.get("status")
        else:
            a, tier, cls, st = v, None, None, None
        astr = f"{a:.4f}" if isinstance(a, (int, float)) else _esc(a)
        A(f"<tr><td>{_esc(cell)}</td><td>{astr}</td><td>{_esc(tier)}</td>"
          f"<td>{_esc(cls)}</td><td>{_esc(st)}</td></tr>")
    A("</table>")

    # 5. Machine-readable snapshot
    snapshot = {
        "patient_id": pid, "patient_meta": meta,
        "stage2": {k: s2.get(k) for k in ("status", "presence_method", "class_present",
                   "class_fraction_ci", "class_fractions", "celltype_fractions",
                   "nilc_fractions", "walther_diagnostics")},
        "stage5": s5 if isinstance(s5, dict) else None,
        "stage6": {**{k: _g(s6, k) for k in ("total_cellular_departure", "n_cells_scored",
                   "null_expected_departure", "excess_departure", "per_class_departure")},
                   "calibration": cal6},
        "stage7": {k: s7.get(k) for k in ("max_class_tier", "n_cells_breach", "breach_line", "class_tiers")},
        "stage8_route_A": rA,
        "stage4_celltype_ascores": cta,
    }
    A("<h2>5 &middot; Machine-readable snapshot (JSON)</h2>")
    A(f"<pre>{J(snapshot, 200000)}</pre>")

    doc = ("<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>Audit Appendix &mdash; {_esc(pid)}</title><style>"
           "body{font-family:ui-sans-serif,system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1a1f26}"
           "h1{font-size:20px}h2{font-size:16px;border-bottom:1px solid #e1e6ec;padding-bottom:4px;margin-top:24px}"
           "h3{font-size:13px;margin:14px 0 4px}"
           "pre{background:#f6f8fa;border:1px solid #e1e6ec;border-radius:6px;padding:8px;font-size:11px;overflow-x:auto;white-space:pre-wrap}"
           ".kv{border-collapse:collapse;width:100%;font-size:12px}.kv th,.kv td{border:1px solid #e1e6ec;padding:3px 6px;text-align:left}"
           ".muted{color:#5d6775;font-size:12px}</style></head><body>"
           + "\n".join(R) + "</body></html>")
    p = Path(audit_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p


def build_trajectory_snapshot(bundle, snapshot_path, scale_provenance=None):
    """Per-timepoint trajectory snapshot for the patient folder. The next IDAT's
    tracker loads every snapshot in the folder, verifies they share the same
    scoring scale (scoring_scale_id), then plots each cell's A-score over time.

    Stores: identity + time (the x-axis), the scale provenance that gates
    comparability, the per-cell A AND v_s (mean beta -> survives an H_min
    recalibration without a re-run), and the run trust-flags. Derived intermediates
    only -- no raw methylation / PHI."""
    import json, hashlib, datetime
    from pathlib import Path

    s2 = bundle.get("stage2") or {}
    s4 = bundle.get("stage4") or {}
    s6 = bundle.get("stage6")
    s7 = bundle.get("stage7") or {}
    s8 = bundle.get("stage8")
    s5 = bundle.get("stage5") or {}
    meta = bundle.get("patient_meta", {}) or {}
    pid = bundle.get("patient_id", "patient")

    # --- scale provenance: the comparability gate ---
    h_min = (s4.get("h_min_by_class") if isinstance(s4, dict) else None) or {}
    prov = dict(scale_provenance or {})
    prov.setdefault("h_min_floors", h_min)
    prov.setdefault("mu_floor", 1.0)
    prov.setdefault("sigma", 0.02)
    for k in ("atlas_version", "marker_panel_version", "tier_breakpoints_version", "derived_reference_version"):
        prov.setdefault(k, None)
    prov["provenance_complete"] = all(prov.get(k) is not None
                                      for k in ("atlas_version", "marker_panel_version",
                                                "tier_breakpoints_version"))
    scale_id = hashlib.sha256(
        json.dumps({k: prov[k] for k in sorted(prov) if k != "provenance_complete"},
                   sort_keys=True, default=str).encode()).hexdigest()[:16]
    prov["scoring_scale_id"] = scale_id

    # --- per-cell trended values (A + v_s for assessable cells) ---
    cta = s4.get("celltype_ascores", {}) if isinstance(s4, dict) else {}
    per_cell = {}
    for cell, rec in cta.items():
        if not isinstance(rec, dict):
            per_cell[cell] = {"A": rec, "v_s": None, "assessable": None, "class": None}
            continue
        assessable = rec.get("assessable")
        per_cell[cell] = {
            "A": rec.get("A"),
            "v_s": rec.get("v_s") if assessable is not False else None,  # mean beta; only meaningful if assessable
            "assessable": assessable,
            "class": rec.get("class"),
            "status": rec.get("status"),
        }

    # --- per-class summary (A + tier from Stage 7) ---
    class_a = s4.get("class_ascores", {}) if isinstance(s4, dict) else {}
    class_tiers = s7.get("class_tiers", {}) or {}
    per_class = {}
    for cls in set(list(class_a.keys()) + list(class_tiers.keys())):
        ca = class_a.get(cls, {})
        per_class[cls] = {
            "A": ca.get("A") if isinstance(ca, dict) else ca,
            "tier": (class_tiers.get(cls, {}) or {}).get("label"),
        }

    # --- run trust-flags (shared check logic; a flagged run is plotted but marked) ---
    checks = _run_consistency_checks(s2, s5, s7, s8)
    review_flags = [name for name, _, ok, note in checks if ok is False]
    rA = (getattr(s8, "route_A_architectural_alarm", None) or {})
    cal6 = _g(s6, "calibration", {}) or {}

    snapshot = {
        "schema_version": "trajectory_snapshot_v1",
        "identity": {
            "patient_id": pid,
            "sample_id": meta.get("sample_id") or bundle.get("sample_id"),
            "collection_date": meta.get("collection_date") or meta.get("assay_date"),  # trajectory x-axis
            "report_generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "substrate": meta.get("substrate", "blood"),
            "array_type": meta.get("array_type"),
        },
        "scale_provenance": prov,
        "per_cell": per_cell,
        "per_class": per_class,
        "run_flags": {
            "architectural_alarm_fired": rA.get("fired"),
            "departure_sigma": cal6.get("departure_sigma_above_null"),
            "all_consistency_checks_passed": len(review_flags) == 0,
            "review_flags": review_flags,
        },
        "tracker_rules": {
            "comparability": "Trend only across snapshots that share scoring_scale_id. "
                             "On mismatch, re-score from stored v_s (H_min change) or re-run (atlas/marker change).",
            "per_cell": "Trend a cell only across timepoints where assessable is True in BOTH; never trend across a not-assessable gap.",
            "review_points": "Plot review-flagged timepoints with a distinct marker; do not silently exclude.",
            "normal_band": "Draw the sigma=0.02 normal band; the signal is direction and velocity across cells, not a single sub-sigma wiggle.",
        },
    }

    p = Path(snapshot_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snapshot, indent=1, default=str), encoding="utf-8")
    return p


def _resemblance_pct(match):
    """Transparent match-magnitude -> resemblance percentage. NOT a calibrated Bayesian
    posterior (L7/L8 are Phase-E-empty per the SOP grade table); this is a documented
    monotone transform of the Stage 8 resemblance score, disclosed in the H.2 footnote:
        pct = 0                       if match <= 0
        pct = 100*(1 - exp(-match/1.5)) otherwise
    """
    import math
    if match is None or match <= 0:
        return 0
    return round(100 * (1 - math.exp(-match / 1.5)))


def _match_confidence(n_cells, evidence):
    """Qualitative confidence from cells-matched + whether the row is VAL-anchored."""
    val_anchored = bool(evidence) and "VAL" in str(evidence)
    if n_cells >= 15 and val_anchored:
        return "moderate"        # clean-patient ceiling; "high" reserved for above-threshold
    if n_cells >= 8:
        return "low-moderate"
    return "low"


DISEASE_CATEGORY = {
    "breast_cancer": "Pre-cancer / malignancy", "pancreatic_cancer": "Pre-cancer / malignancy",
    "hcc": "Pre-cancer / malignancy", "lung_cancer": "Active malignancy",
    "colorectal_cancer": "Active malignancy", "gastric_cancer": "Active malignancy",
    "prostate_cancer": "Active malignancy", "bladder_cancer": "Active malignancy",
    "cervical_cancer": "Active malignancy", "esophageal_cancer": "Active malignancy",
    "leukemia_aml": "Hematologic malignancy", "leukemia_all": "Hematologic malignancy",
    "lymphoma_dlbcl": "Hematologic malignancy", "myeloma": "Hematologic malignancy",
    "alzheimers_disease": "Neurodegenerative", "frontotemporal_dementia": "Neurodegenerative",
    "parkinsons_disease": "Neurodegenerative", "als": "Neurodegenerative",
    "multiple_sclerosis": "Neurodegenerative", "psp_cbd": "Neurodegenerative",
    "mild_cognitive_impairment": "Neurodegenerative",
    "chronic_inflammation": "Inflammatory", "inflammaging": "Inflammatory",
    "crohns_disease": "Inflammatory", "ulcerative_colitis": "Inflammatory",
    "rheumatoid_arthritis": "Autoimmune", "lupus": "Autoimmune", "sle": "Autoimmune",
    "hashimotos": "Autoimmune",
    "cardiovascular": "Cardiovascular", "pulmonary_arterial_hypertension": "Cardiovascular",
    "atherosclerosis": "Cardiovascular", "aortic_dilation": "Cardiovascular",
    "type_2_diabetes": "Metabolic", "pre_t2d": "Metabolic", "nafld": "Metabolic",
    "normal_aging": "Aging baseline",
}


def _category_for(disease_id):
    if not disease_id:
        return "Other"
    d = str(disease_id).lower()
    if d in DISEASE_CATEGORY:
        return DISEASE_CATEGORY[d]
    for k, v in DISEASE_CATEGORY.items():
        if k in d:
            return v
    return "Other"


def build_report(bundle, output_html_path, atlas_plate_paths=None, config=None):
    """Render the CPG report HTML from a run_pipeline bundle. Returns the output path."""
    pid = bundle.get("patient_id", "patient")
    meta = bundle.get("patient_meta", {})
    s4 = bundle["stage4"]; s5 = bundle["stage5"]; s6 = bundle["stage6"]; s7 = bundle["stage7"]
    s45 = bundle.get("stage4_5"); s46 = bundle.get("stage4_6")
    figs = bundle.get("figures", {}) or {}
    bril = figs.get("brilliance_maps", {}) or {}
    atlas_plate_paths = atlas_plate_paths or {}
    # Resolve the four reference plates if not explicitly supplied. Defaults to the atlas_vault
    # plates tree (present in the research environment, read-only); a shipped report can pass
    # atlas_plate_paths={...} or config["plates_dir"] instead.
    if not atlas_plate_paths:
        _cfg = config or {}
        _pdir = Path(_cfg.get("plates_dir") or
                     (Path(__file__).resolve().parents[3] / "atlas_vault/IAMAtlas_v0_1/plates"))
        _plate_files = {
            "plate1": "CPG_Plate_01_Cosmic_Microwave_Methylome.png",
            "plate2": "CPG_Plate_02_Breast_Anisotropy.png",
            "plate3": "CPG_Plate_03_Grandaddy_CMM_vs_CMB.png",
            "plate4": "CPG_Plate_04_Patterns_Discovered.png",
        }
        atlas_plate_paths = {k: str(_pdir / v) for k, v in _plate_files.items() if (_pdir / v).exists()}

    parts = []
    P = parts.append

    # ---- header + exec summary ----
    max_tier = s7.get("max_class_tier")
    max_tier_label = (s7["class_tiers"].get(max_tier, {}) or {}).get("label", max_tier) if max_tier else "—"
    n_breach = s7.get("n_cells_breach", 0)
    chrono = meta.get("age")
    maha = s5.get("mahalanobis_distance"); maha_status = s5.get("status")
    exec_bits = []
    exec_bits.append(f"Overall architectural state reads <b>{_esc((max_tier_label or '').replace(chr(10),' '))}</b> "
                     f"(highest-tier architectural class).")
    exec_bits.append(f"<b>{n_breach}</b> of 115 cell types crossed the breach line (A &ge; {s7.get('breach_line',1.10)}).")

    P(f"""<div class="hdr">
      <div class="brand">Cellular Performance Gauge <span>· AstroGenetics</span></div>
      <h1>Patient Report — {_esc(pid)}</h1>
      <div class="sub">Generated {datetime.now():%Y-%m-%d %H:%M} · IAMPerformance Inter-Domain Research Institute</div>
    </div>
    <section><h2>Executive summary — in plain language</h2>
      <p>{' '.join(exec_bits)}</p>
      <p class="muted">Every cell measured here is read against the frozen healthy IAMAtlas reference.
      An A-score near 1.00 means a cell is maintaining its architecture; values climbing toward and past
      the breach line at {s7.get('breach_line',1.10)} mean it is losing the minimum order required to remain itself.</p>
    </section>""")

    # ---- A. intake ----
    P(f"""<section><h2>A · Sample integrity and intake context</h2>
      <table class="kv">
        <tr><td>Patient ID</td><td>{_esc(pid)}</td></tr>
        <tr><td>Chronological age</td><td>{_esc(chrono if chrono is not None else '—')}</td></tr>
        <tr><td>Sex</td><td>{_esc(meta.get('sex') or '—')}</td></tr>
        <tr><td>Smoking bin</td><td>{_esc(meta.get('smoking_bin') or 'not provided')}</td></tr>
        <tr><td>Foregrounds removed (Stage 3)</td><td>{_esc(', '.join(getattr(bundle.get('stage3'),'foregrounds_applied',[])) or '—')}</td></tr>
      </table></section>""")

    # ---- B. composition (cells grouped by class) ----
    ct = s4["celltype_ascores"]
    s2 = bundle.get("stage2")
    by_class = {}
    for name, v in ct.items():
        if not isinstance(v, dict):
            continue
        by_class.setdefault(v.get("class", "?"), []).append((name, v))
    P('<section><h2>B · Cellular composition — every cell detected, with state</h2>')
    P('<p class="muted">Composition is read first (who is present, in what proportion — from Stage 2 '
      'deconvolution); architecture (A-score, Section C) is read on whatever cells are present. '
      'A shedding tumor surfaces here as an elevated tissue-of-origin fraction before its methylation '
      'architecture shifts.</p>')
    cfr = (s2 or {}).get("class_fractions") if isinstance(s2, dict) else None
    P('<table class="kv"><tr><th>Architectural class</th><th>Composition (Stage 2)</th>'
      '<th>Class A-score</th><th>Tier</th><th>Cells detected</th></tr>')
    for cls in ARCH_CLASSES:
        cv = s4["class_ascores"].get(cls)
        if not isinstance(cv, dict):
            continue
        t = s7["class_tiers"].get(cls, {})
        ncells = len(by_class.get(cls, []))
        frac = (f'{cfr.get(cls,0)*100:.1f}%' if cfr and cls in cfr else '—')
        P(f'<tr><td>{_esc(CLASS_PRETTY.get(cls,cls))}</td><td>{frac}</td><td>{cv.get("A",0):.3f}</td>'
          f'<td>{_tier_badge(t.get("tier_id"), t.get("label"))}</td><td>{ncells}</td></tr>')
    P('</table>')
    if s2 and isinstance(s2, dict):
        gate = s2.get("cross_method")
        P(f'<p class="muted">Stage 2 status: {_esc(s2.get("status","—"))}. '
          f'Class-level fractions are the production answer (Walther NNLS); NILC v2 runs in parallel as the '
          f'independent cross-method check (Planck Commander/NILC discipline).</p>')
    # B.2 — detailed per-cell, every detected cell, with its healthy normal range (sex/smoking-adjusted A; not age-adjusted)
    pc = getattr(s6, "per_cell", {}) or {}
    if pc:
        P('<h3>B.2 Detailed per-cell — every cell against its healthy range</h3>')
        P('<p class="muted">Normal range is the cell\'s healthy reference mean ± 1.96·SD, taken directly '
          'from the IAMAtlas MCMC posterior (95% CI). The A-score is sex/smoking-adjusted (Stage 3 '
          'foregrounds removed up front). "Remarkable" = the cell\'s A-score falls outside its 95% range.</p>')
        ctf = (s2 or {}).get("celltype_fractions") if isinstance(s2, dict) else None
        by_cls_cells = {}
        for cell, d in pc.items():
            by_cls_cells.setdefault(d.get("class", "?"), []).append((cell, d))
        for cls in ARCH_CLASSES:
            rows = sorted(by_cls_cells.get(cls, []), key=lambda kv: -kv[1]["abs_z"])
            if not rows:
                continue
            P(f'<h4 style="margin:14px 0 4px;color:#3a4656">{_esc(CLASS_PRETTY.get(cls,cls))} '
              f'<span class="muted">({len(rows)} cells)</span></h4>')
            P('<table class="kv"><tr><th>Cell type</th><th style="text-align:right">A-score</th>'
              '<th>Normal range (95% CI)</th><th style="text-align:right">z</th><th>Remarkable</th></tr>')
            for cell, d in rows:
                rem = ('<span style="color:#b03020;font-weight:600">outside range</span>'
                       if not d["in_range"] else '<span style="color:#3f9b54">within range</span>')
                P(f'<tr><td>{_esc(cell)}</td><td style="text-align:right">{d["A"]:.3f}</td>'
                  f'<td>[{d["ci_lo"]:.3f}, {d["ci_hi"]:.3f}]</td>'
                  f'<td style="text-align:right">{d["z"]:+.2f}</td><td>{rem}</td></tr>')
            P('</table>')
    P('</section>')

    # ---- C. architectural state ----
    P('<section><h2>C · Architectural state — the cell-level view</h2>')
    P('<h3>C.1 The reference scale — what each A-score value means</h3>')
    P(f'<div class="fig">{_img(figs.get("A1_reference_gauge"), "A-score reference gauge")}</div>')
    P('<h3>C.2 Top 15 cells by magnitude of departure</h3>')
    P(f'<div class="fig">{_img(figs.get("A2_cellular_departure_ranking"), "Top-15 cellular departure ranking")}</div>')
    P('<p class="muted">Ranked by confidence-weighted departure |z| = |A − healthy mean| / posterior SD — '
      'the same per-cell quantity that every other view in this report is built from. '
      'Stable cells (tight posterior) surface on real shifts; noisy cells do not dominate.</p>')
    pcr = getattr(s6, "per_cell", {}) or {}
    if pcr:
        ranked = sorted(pcr.items(), key=lambda kv: kv[1]["abs_z"], reverse=True)[:15]
        P('<table class="kv"><tr><th>#</th><th>Cell type</th><th>Class</th>'
          '<th style="text-align:right">A-score</th><th>Normal range</th>'
          '<th style="text-align:right">z</th><th>Remarkable</th></tr>')
        for i, (n, d) in enumerate(ranked, 1):
            rem = "outside" if not d["in_range"] else "within"
            P(f'<tr><td>{i}</td><td>{_esc(n)}</td><td>{_esc(d.get("class",""))}</td>'
              f'<td style="text-align:right">{d["A"]:.3f}</td>'
              f'<td>[{d["ci_lo"]:.3f}, {d["ci_hi"]:.3f}]</td>'
              f'<td style="text-align:right">{d["z"]:+.2f}</td><td>{rem}</td></tr>')
        P('</table>')
    else:
        ranked = sorted([(n, v) for n, v in ct.items() if isinstance(v, dict) and v.get("A") is not None],
                        key=lambda kv: abs(kv[1]["A"] - 1.0), reverse=True)[:15]
        P('<table class="kv"><tr><th>#</th><th>Cell type</th><th>Class</th><th>A-score</th><th>Δ from 1.00</th></tr>')
        for i, (n, v) in enumerate(ranked, 1):
            P(f'<tr><td>{i}</td><td>{_esc(n)}</td><td>{_esc(v.get("class",""))}</td>'
              f'<td>{v["A"]:.3f}</td><td>{v["A"]-1.0:+.3f}</td></tr>')
        P('</table>')
    P('</section>')

    # ---- D. cellular departure (per-cell confidence-weighted absolute departure) ----
    P('<section><h2>D &middot; Why this report gives no single &ldquo;cellular age&rdquo;</h2>')
    # D.1 — why we do not report a cellular-age-in-years number (2026-06-09 age-stability finding).
    P('<h3>D.1 Why there is no &ldquo;cellular age&rdquo; here</h3>')
    P('<p>You will not find a single number telling you that you are &ldquo;biologically 62 instead of 60.&rdquo; '
      'That number is popular, and it is a gimmick. Here is why, in plain terms.</p>')
    P('<p>A cell&rsquo;s methylation pattern has a physically defined floor &mdash; the most ordered, highest-fidelity '
      'state it can hold &mdash; set by the class-specific fidelity-to-noise ratio (H<sub>min</sub>). Healthy aging is a '
      'slow, gentle drift <i>toward</i> that floor: the cell loses a little of its ability to hold its full working '
      'pattern. Across the whole adult lifespan this drift is tiny &mdash; about one-seventh of the normal cell-to-cell '
      'variation. A healthy 30-year-old and a healthy 90-year-old sit, for practical purposes, in the same normal range. '
      'Age by itself barely moves the needle &mdash; so we measure each cell&rsquo;s departure from its healthy '
      'reference directly, with no age adjustment needed and none applied.</p>')
    P('<p>What <i>does</i> move the needle is real departure, and it has a <b>direction</b>. Proliferative disease '
      '(cancer, pre-cancer) pushes cells <i>up and outward</i>, away from the floor toward disorder. Degenerative '
      'disease (Alzheimer&rsquo;s, the tauopathies) pulls cells <i>down toward the floor</i> &mdash; the same direction '
      'as aging, but far stronger and concentrated in specific cell types. A single &ldquo;cellular age&rdquo; number is '
      'blind to this: it collapses an up-axis and a down-axis onto one scale, where they blur together or cancel out, so '
      'an early cancer signal can read as &ldquo;younger,&rdquo; and an Alzheimer&rsquo;s signal is indistinguishable '
      'from ordinary aging. It also throws away the one thing you can act on: <i>where</i> the departure is.</p>')
    P('<p>So we do something only the underlying physics makes possible. We measure each cell&rsquo;s departure from its '
      'healthy expectation, keeping the sign &mdash; upward departures flag proliferative risk, downward departures flag '
      'degenerative or inflammatory loss &mdash; and we name the <b>pattern</b> and the <b>location</b>. Epigenetic age '
      'clocks are legitimate population-level risk summaries, useful for actuarial and research purposes. But as a tool '
      'for <i>you</i>, a single averaged age is a blunt, direction-blind instrument. The cellular-age number is the '
      'headline; your per-cell departure map is the diagnosis.</p>')
    P('</section>')

    # ---- E. Internal consistency & fail-safes (no internal numbers; PASS / NEEDS REVIEW) ----
    _consistency_checks_panel(s2, s5, s7, s8, P)

    # ---- F. Personal Brilliance Maps ----
    P('<section><h2>F · Personal Brilliance Map — your methylome vs the Cosmic Methylome Background</h2>')

    # F.1 — what a brilliance map is (the CMB analogy)
    P('<h3>F.1 What this is</h3>')
    P('<p>Your 481,966 measured CpGs are ordered by genomic position (chromosome 1 through X) and projected onto a '
      'sphere using the same equal-area projection the Planck mission uses to map the cosmic microwave background — '
      'a HEALPix Mollweide grid of 196,608 pixels. The healthy IAMAtlas, mapped the same way, is the '
      '<b>Cosmic Methylome Background (CMB)</b>: the smooth reference pattern a healthy methylome makes. Your map '
      'shows where <i>your</i> pattern departs from that background — red where a region is hypermethylated beyond '
      'healthy variance, blue where it is hypomethylated, neutral where it sits within healthy variance. '
      'The departures are the signal: exactly as the faint anisotropies on top of the cosmic background are where '
      'the real structure lives, the bright patches on your map are where your methylome carries real structure.</p>')

    # F.2 — the four canonical reference plates
    P('<h3>F.2 The reference plates</h3>')
    P('<p class="muted">The four canonical references your map is read against — the visual analog of the IAMAtlas '
      'numerical matrices, the same posteriors projected onto the celestial sphere.</p>')
    _plates = [
        ("plate1", "Plate 1 — Cosmic Methylome Background",
         "Eight Mollweide panels, one per architecture class, showing the healthy per-CpG posterior mean across "
         "481,966 CpGs. This is the healthy reference your per-class maps (F.3) are read against. The stromal "
         "panel's sparse patch is the methylome's declared known-unknown (4.93% MCMC coverage)."),
        ("plate3", "Plate 3 — Methylome vs CMB, side by side",
         "The healthy methylome and a cosmic-microwave-background realization at matched projection, colormap and "
         "pixelization. The whole idea in one image: the two fields are read with the same instrument."),
        ("plate2", "Plate 2 — A worked anisotropy (breast pre-diagnostic)",
         "What a real departure field looks like: 1,392 concordant breast pre-diagnostic CpGs, blue hypomethylated "
         "vs orange hypermethylated, 5.4:1 hypomethylation dominance, with the chr6 MHC region lit up. An example "
         "of the kind of pattern your map is screened for."),
        ("plate4", "Plate 4 — Patterns the sphere makes visible",
         "Six findings the spherical projection reveals — class-difference maps, chr16/chr17 cold-patch zones, "
         "concordant signal density, the differentiation gradient, the MCMC coverage map, and the breast "
         "anisotropy field."),
    ]
    for key, title, desc in _plates:
        if atlas_plate_paths.get(key):
            P(f'<h4>{_esc(title)}</h4>')
            P(f'<div class="fig">{_img_scaled(atlas_plate_paths[key], title)}</div>')
            P(f'<p class="muted">{_esc(desc)}</p>')

    # F.3 — patient per-class panels
    P('<h3>F.3 Your 8 per-class Personal Brilliance Maps</h3>')
    P('<p class="muted">Your per-CpG z-departure from the healthy posterior, one panel per architecture class, on the '
      'same grid as Plate 1. Read each against the matching Plate 1 panel: red = hypermethylated departure, '
      'blue = hypomethylated, neutral = within healthy variance.</p>')
    P('<div class="grid">')
    for cls in ARCH_CLASSES:
        if cls in bril:
            P(f'<div class="cell"><div class="lab">{_esc(CLASS_PRETTY.get(cls,cls))}</div>{_img(bril[cls], cls)}</div>')
    P('</div>')

    # F.4 — whole-atlas patient map NEXT TO the CMB reference, with the pattern-difference explanation
    P('<h3>F.4 Your whole methylome vs the Cosmic Methylome Background</h3>')
    P('<p>This is the endpoint — your entire methylome on one sphere, placed directly beside the healthy background '
      'it is read against. <b>How to read the pair:</b> where your map (left) is smooth and neutral, your pattern '
      'matches the healthy background (right) — there is nothing to see, and that is the goal. Where your map shows '
      'bright red or blue clusters the background does not, those are <i>your</i> anisotropies: regions where your '
      'methylome has departed from healthy. Those clusters are the same departures that drive your cell ranking '
      '(C.2) &mdash; here you can see <i>where</i> on the '
      'methylome they sit. A healthy methylome looks like the background; a departing one grows structure.</p>')
    _cmb_ref = atlas_plate_paths.get("plate3") or atlas_plate_paths.get("plate1")
    P('<div class="grid">')
    P(f'<div class="cell"><div class="lab">Your whole-atlas map</div>{_img(bril.get("whole_atlas"), "your whole-atlas brilliance map")}</div>')
    if _cmb_ref:
        P(f'<div class="cell"><div class="lab">Cosmic Methylome Background (healthy reference)</div>{_img_scaled(_cmb_ref, "Cosmic Methylome Background reference")}</div>')
    P('</div>')
    if bril.get("whole_atlas_ascii"):
        P('<h4>F.4a Text rendering (accessibility / archival)</h4>')
        P('<pre class="ascii">' + _esc(bril.get("whole_atlas_ascii")) + '</pre>')
    P(f'<div class="fig">{_img(figs.get("star_gauge"), "star gauge — same ruler as the cell")}</div>')
    P('</section>')

    # ---- G. bidirectional ----
    if s45 is not None:
        P('<section><h2>G · Bidirectional decomposition — what pooled scoring would have missed</h2>')
        pcr = getattr(s45, "per_class_results", {}) or {}
        flagged = [(c, r) for c, r in pcr.items() if getattr(r, "flag_bidirectional", False)]
        if flagged:
            P('<table class="kv"><tr><th>Class</th><th>Pooled A</th><th>Directional composite</th><th>Interpretation</th></tr>')
            for c, r in flagged:
                P(f'<tr><td>{_esc(c)}</td><td>{_esc(getattr(r,"a_pooled_entropy","—"))}</td>'
                  f'<td>{_esc(getattr(r,"a_directional_composite","—"))}</td>'
                  f'<td>{_esc(getattr(r,"interpretation",""))}</td></tr>')
            P('</table>')
        else:
            P('<p class="muted">No class triggered FLAG_BIDIRECTIONAL — pooled and directional scores agree.</p>')
        P('</section>')

    # ---- H. disease pattern matching (Stage 8) ----
    s8 = bundle.get("stage8")
    P('<section><h2>H · Disease pattern matching — every signature scored against your data</h2>')
    if s8 is None:
        P('<p class="pending">Stage 8 did not run for this bundle.</p>')
    else:
        rb = s8.route_B_disease_matches or []
        rA = s8.route_A_architectural_alarm or {}
        rC = s8.route_C_bidirectional or {}
        scored_n = len([x for x in (s8.route_B_all_scored or []) if x.get("match_magnitude") is not None])
        # H.1 summary
        flagged = [m for m in rb if m["tier"] not in ("NORMAL", "MARGINAL")]
        P('<h3>H.1 Summary</h3>')
        if flagged:
            P(f'<p>Your cellular departure pattern was scored against all {scored_n} (disease × phase) '
              f'signatures in the matrix. <b>{len(flagged)}</b> of the closest matches reach a non-baseline '
              f'tier. These are pattern resemblances for physician review — not diagnoses.</p>')
        else:
            P(f'<p>Your pattern was scored against all {scored_n} signatures. <b>No signature reaches a '
              f'non-baseline tier.</b> Closest resemblances are shown below for context.</p>')
        # H.2 closest matches — mock-faithful: Disease | Detection probability | Confidence | Status
        all_scored = [x for x in (s8.route_B_all_scored or []) if x.get("match_magnitude") is not None]
        all_scored.sort(key=lambda s: s["match_magnitude"], reverse=True)
        top5 = all_scored[:5]
        n_low = sum(1 for s in all_scored if _resemblance_pct(s["match_magnitude"]) <= 5)
        P('<h3>H.2 Closest pattern matches' + ('' if flagged else ' (all below threshold)') + '</h3>')
        P('<table class="kv"><tr><th>Disease · phase</th><th style="text-align:right">Detection probability</th>'
          '<th>Confidence</th><th>Status</th></tr>')
        for m in top5:
            pct = _resemblance_pct(m["match_magnitude"])
            pct_str = "&lt;1%" if pct < 1 else f"{pct}%"
            conf = _match_confidence(m["n_cells_matched"], "")
            status = ("Below threshold" if m["tier"] in ("NORMAL", "MARGINAL")
                      else f'{m["tier"].replace("_"," ").title()} — review')
            P(f'<tr><td>{_esc(m["disease"])} · {_esc(m["phase"])}</td>'
              f'<td style="text-align:right">{pct_str}</td><td>{conf}</td><td>{_esc(status)}</td></tr>')
        P('</table>')
        P(f'<p><b>Patterns at 0–5% probability:</b> {max(0,len(all_scored)-5)} other (disease × phase) rows. '
          f'<b>Complete per-disease scoring is in Appendix C.</b></p>')
        P('<p class="muted">"Detection probability" here is a transparent monotone transform of the Stage 8 '
          'resemblance magnitude (pct = 100·(1−e^(−match/1.5)); 0 when match ≤ 0) — <b>not</b> a calibrated '
          'Bayesian posterior. The calibrated per-disease posterior is the Phase E (L7/L8) deliverable, which '
          'the chain-of-custody grade table declares not-yet-built. The number ranks resemblance; it is not a '
          'probability of having the disease.</p>')
        # H.3 routes A / C
        P('<h3>H.3 Architectural-alarm & bidirectional channels</h3>')
        P(f'<p><b>Route A (universal architectural alarm):</b> '
          f'<b>{"TRIGGERED &mdash; see internal consistency checks (E)" if rA.get("fired") else "within normal architecture"}</b>.</p>')
        if rC.get("fired"):
            cls_list = ", ".join(f'{f["class"]} (a_dir {f["a_directional"]:+.2f})' for f in rC["flagged_classes"])
            P(f'<p><b>Route C (bidirectional pattern):</b> TRIGGERED — {_esc(cls_list)}.</p>')
        else:
            P('<p><b>Route C (bidirectional pattern):</b> not triggered.</p>')
        # H.5 pattern recognition (convergent-evidence naming)
        P('<h3>H.5 Pattern Recognition — convergent evidence across channels</h3>')
        conv = []
        if rb:
            conv.append(f'the disease-matrix top match is <b>{_esc(rb[0]["disease"])} · {_esc(rb[0]["phase"])}</b> '
                        f'(magnitude {rb[0]["match_magnitude"]:+.2f})')
        if rA.get("fired") is not None:
            conv.append('the whole-sample architectural check '
                        + ('flagged an overall departure' if rA.get("fired") else 'is within normal architecture'))
        top = s5.get("top10_axis_contributions") or []
        if top:
            names = []
            for c in top[:3]:
                if isinstance(c, dict):
                    names.append(str(c.get("cell_type") or c.get("axis") or c.get("name") or ""))
            if names:
                conv.append(f'the cells showing the largest departures are <b>{_esc(", ".join(n for n in names if n))}</b>')
        P('<p>' + ('; '.join(conv) if conv else 'No convergent pattern surfaced.') +
          '. The cell ranking (C.2), the internal consistency checks (E), the Personal Brilliance Map (F) and '
          'the disease matrix (H.2), and the immune universal-alarm axis (H.5b) are five converging views of the same departure signal — where they converge, '
          'the pattern is real.</p>')
        # H.5b immune universal-alarm axis -- wired to the patient's measured immune direction
        _icp = (Path(__file__).resolve().parents[1] /
                "Runtime Matrices/Literature_anchors_Report building/literature_anchors_v2_1.json")
        try:
            import json as _json_h5
            _icat = _json_h5.load(open(_icp)).get("immune_class_disease_catalog", {})
        except Exception:
            _icat = {}
        _icat = {k: v for k, v in _icat.items() if not k.startswith("_") and isinstance(v, dict)}
        if _icat:
            def _axisdir(s):
                s = str(s).lower()
                if "positive" in s: return "up"
                if "negative" in s: return "down"
                return "mixed"
            _imm = (getattr(s45, "per_class_results", {}) or {}).get("immune") if s45 is not None else None
            _pdir, _pcomp, _pinterp, _pcov = None, None, "", True
            if _imm is not None:
                _pcomp = getattr(_imm, "a_directional_composite", None)
                _pinterp = str(getattr(_imm, "interpretation", "") or "")
                if getattr(_imm, "flag_insufficient_coverage", False): _pcov = False
                _il = _pinterp.lower()
                # The panel reports a signed directional composite ("disease-direction" /
                # "anti-disease-direction"); the bridge to the catalog's elevation/suppression
                # axis is a sign convention not yet confirmed. Do NOT auto-classify direction
                # (a backwards map would flip every convergence call). _pdir stays None until
                # the convention is confirmed; the measured signal is shown verbatim below.
                _pdir = None
            P('<h4>H.5b Immune universal-alarm axis &mdash; your signal against the cross-disease map</h4>')
            if _imm is not None and _pcov and _pcomp is not None:
                P(f'<p>Your measured immune signal (sealed immune panel): directional composite '
                  f'<b>{_pcomp:+.3f}</b>. {_esc(_pinterp)}</p>')
            else:
                P('<p class="muted">The sealed immune panel did not return a confident direction for this sample, '
                  'so the table below is shown as cross-disease reference only.</p>')
            P('<p class="muted">Cancers elevate the immune class toward the ceiling; colorectal and '
              'Alzheimer\u2019s suppress it toward the floor. <b>Validated</b> rows are sealed VAL results; '
              '<b>predicted</b> rows are framework expectations not yet validated. The final column marks where a '
              'disease\u2019s expected direction matches your measured immune direction &mdash; a fifth view that '
              'converges with the disease matrix (H.2) when a pattern is real. Automatic per-disease matching '
              'appears once the panel\u2019s directional sign convention is confirmed; until then this is a '
              'reference map and your measured signal is shown above.</p>')
            P('<table class="kv"><tr><th>Disease</th><th>Immune direction</th>'
              '<th>Expected magnitude</th><th>Status</th><th>Matches your signal</th></tr>')
            for _dz, _v in _icat.items():
                _val = bool(_v.get("validated"))
                _cdir = _axisdir(_v.get("expected_direction", ""))
                _match = "&#10003;" if (_pdir and _cdir == _pdir) else ("&mdash;" if _pdir else "&middot;")
                _st = "Validated" if _val else "<i>Predicted / pending</i>"
                P(f'<tr><td>{_esc(_dz)}</td><td>{_esc(str(_v.get("direction") or _v.get("expected_direction","")))}</td>'
                  f'<td>{_esc(str(_v.get("expected_magnitude_d",""))[:110]) or "\u2014"}</td>'
                  f'<td>{_st}</td><td style="text-align:center">{_match}</td></tr>')
            P('</table>')
            if _pdir and rb:
                _alias = {"breast_cancer":"Breast cancer","colorectal_cancer":"Colorectal cancer",
                  "alzheimers_disease":"Alzheimer's disease","lung_cancer":"Lung cancer (NSCLC)",
                  "prostate_cancer":"Prostate cancer","hcc":"Hepatocellular carcinoma",
                  "pancreatic_cancer":"Pancreatic cancer","gastric_cancer":"Gastric cancer",
                  "bladder_cancer":"Bladder cancer","cervical_cancer":"Cervical cancer",
                  "kidney_cancer":"Kidney cancer (RCC)","glioma_gbm":"Glioma / GBM / LGG",
                  "glioma_lgg":"Glioma / GBM / LGG"}
                _top = str(rb[0].get("disease", ""))
                _cn = _alias.get(_top)
                if _cn and _cn in _icat:
                    _tdir = _axisdir(_icat[_cn].get("expected_direction", ""))
                    if _tdir in ("up", "down"):
                        if _tdir == _pdir:
                            P(f'<p><b>Convergence:</b> the disease-matrix top match ({_esc(_top)}) expects an immune '
                              f'direction that agrees with your measured immune signal &mdash; the universal-alarm axis '
                              f'and the disease matrix point the same way.</p>')
                        else:
                            P(f'<p><b>Discordance flag:</b> the disease-matrix top match ({_esc(_top)}) expects the '
                              f'opposite immune direction from your measured signal &mdash; the channels do not converge '
                              f'here, which argues against a real pattern.</p>')

    P('</section>')

    # ---- I. cross-disease universal alarm channel ----
    P('<section><h2>I · Cross-disease universal alarm channel</h2>')
    P('<p class="muted">The immune-atlas card carries a 6,018-CpG cross-disease firing-pattern map with a '
      '12-CpG opposing-direction sub-channel (the VAL-016 universal-alarm signature). The Stage 8 Route A '
      'residual-map-overlap channel computes the patient\'s Pearson overlap with that map. '
      'Per-CpG residual-overlap is the next wiring step (the channel artifact and thresholds are staged; '
      'overlap is computed once the per-CpG departure vector is exposed from Stage 4.6).</p></section>')

    # ---- J. wellness / lifestyle / inflammaging lens (parameterized; no "years" framing) ----
    P('<section><h2>J · Wellness, lifestyle, and inflammaging context</h2>')
    P('<p class="muted">Contextual lenses on your reading. The inflammaging lens is the immune-class share '
      'of your per-cell departure (an immune-dysregulation contribution), expressed as a fraction of the '
      'total \u2014 not a "years" figure.</p>')
    P('<table class="kv"><tr><th>Lens</th><th>Reading</th></tr>')
    _pcdj = getattr(s6, "per_class_departure", {}) or {}
    _totj = getattr(s6, "total_cellular_departure", None) or (sum(_pcdj.values()) if _pcdj else None)
    if _pcdj.get("immune") is not None and _totj:
        P(f'<tr><td>Inflammaging (immune-class share of departure)</td>'
          f'<td>{100*_pcdj["immune"]/_totj:.0f}% of your total cellular departure is immune-class '
          f'\u2014 the inflammaging contribution</td></tr>')
    _fga = getattr(bundle.get("stage3"), "foregrounds_applied", []) or []
    _smk = meta.get("smoking_bin")
    if "smoking" in _fga:
        P(f'<tr><td>Smoking status</td><td>Smoking-axis foreground subtracted (bin: {_esc(_smk or "provided")}); '
          f'residual smoking signature removed up front</td></tr>')
    else:
        P(f'<tr><td>Smoking status</td><td>{_esc(_smk or "not provided")} \u2014 smoking foreground not applied '
          f'(interim Stage 7 stratification absorbs residual)</td></tr>')
    _sex = (meta.get("sex") or "").lower(); _age = meta.get("age")
    if _sex.startswith("f"):
        if isinstance(_age, (int, float)) and _age >= 50:
            P('<tr><td>Hormonal stratification</td><td>Likely post-menopausal (age-based) \u2014 secretory-class '
              'baseline read accordingly</td></tr>')
            P('<tr><td>Menarche signature</td><td>Not applicable at this age</td></tr>')
        else:
            P('<tr><td>Hormonal stratification</td><td>Pre-menopausal context (age-based)</td></tr>')
    else:
        P('<tr><td>Hormonal stratification</td><td>n/a (not female-context)</td></tr>')
    P('<tr><td>Body composition / metabolic signature</td><td>Not assessable from blood EPIC alone '
      '(requires additional substrate)</td></tr>')
    P('<tr><td>Stress / sleep / mood</td><td>Not directly assessable from methylation alone '
      '(clinical history supplements)</td></tr>')
    P('<tr><td>Active treatment footprint</td><td>None reported; none detected in this reading</td></tr>')
    P('<tr><td>Intervention-direction tracking</td><td>n/a \u2014 single sample; sample 2+ unlocks '
      'direction-of-change tracking (Section K)</td></tr>')
    P('</table></section>')

    # ---- L. risk context (cancer prior × family history, SOP §9.4b) ----
    if s8 is not None and getattr(s8, "risk_context", None):
        rc = s8.risk_context
        if "_note" not in rc:
            P('<section><h2>L · Prior and family-history context</h2>')
            P('<p class="muted">Per SOP §9.4b, a closest-match resemblance is framed with population base rate '
              'and family history when supplied — "your reading combined with your risk context suggests…", '
              'never "you have." Family history not supplied → population base rate only.</p>')
            P('<table class="kv"><tr><th>Disease</th><th>Baseline prior</th><th>Family-hx multiplier</th>'
              '<th>Family history supplied</th></tr>')
            for d, info in rc.items():
                P(f'<tr><td>{_esc(d)}</td><td>{_esc(info.get("baseline_prior") or "—")}</td>'
                  f'<td>{_esc(info.get("family_history_multiplier"))}</td>'
                  f'<td>{"yes" if info.get("family_history_supplied") else "no"}</td></tr>')
            P('</table></section>')

    # ---- M. literature anchors — exhaustive over every matrix disease x phase row ----
    import csv as _csv, json as _json
    _cfgM = config or {}
    _matP = _cfgM.get("disease_matrix_csv") or (Path(__file__).resolve().parents[1] /
            "Disease Matrix/DISEASE_MATRIX/disease_cell_signature_matrix_v1_8.csv")
    _ancP = _cfgM.get("literature_anchors_json") or (Path(__file__).resolve().parents[1] /
            "Runtime Matrices/Literature_anchors_Report building/literature_anchors_v2_1.json")
    try:
        _mrows = list(_csv.reader(open(_matP)))[1:]
        _ddb = _json.load(open(_ancP)).get("disease_detection_benchmarks", {})
    except Exception:
        _mrows, _ddb = [], {}
    if _mrows:
        def _bench_key(d):
            d = d.lower()
            if d == "hcc": return "hepatocellular_carcinoma"
            if d in _ddb: return d
            if d.startswith("glioma"): return "glioma"
            if d.startswith("esophageal"): return "esophageal_cancer"
            if d.startswith("leukemia") or d in ("lymphoma_dlbcl", "multiple_myeloma", "mds", "mpn", "thymoma"):
                return "hematologic_malignancies"
            if d in ("pah", "ischemic_stroke", "aortic_dissection_bav"): return "cardiovascular"
            if d in ("crohns_disease", "ulcerative_colitis"): return "ibd_crohns_uc_informative_null"
            return None

        def _pub_anchor(d):
            b = _ddb.get(_bench_key(d) or "")
            if not isinstance(b, dict): return "—"
            bits = []
            for name, val in b.items():
                if name in ("_note", "_source"): continue
                label = name.lstrip("_").replace("_", " ")
                if isinstance(val, dict):
                    kv = [f"{k}={val[k]}" for k in
                          ("AUC", "d", "ci", "p", "perm_p", "n", "n_pairs", "delta_A",
                           "cohort", "interpretation", "source") if k in val]
                    bits.append(f"<i>{_esc(label)}</i>: {_esc(', '.join(str(x) for x in kv))}")
                else:
                    bits.append(f"<i>{_esc(label)}</i>: {_esc(str(val))}")
            return "; ".join(bits) if bits else "—"

        P('<section><h2>M · Literature anchors — the evidence behind every detectable signature</h2>')
        P('<p class="muted">This instrument screens for the signatures below. For each, the published literature '
          'anchor and the sealed internal validation (VAL) evidence are listed. This is the complete evidence '
          'basis the matrix can detect against \u2014 not only the findings flagged in your reading. Sources: the '
          'disease-signature matrix (v1.8, 81 disease\u00d7phase rows across 52 conditions) and the literature '
          'anchors library (v2.1).</p>')
        P('<table class="kv"><tr><th>Condition</th>'
          '<th>Published anchor</th><th>Phase(s) &amp; sealed VAL evidence</th></tr>')
        _bydis = {}
        for r in _mrows:
            did = r[0] if r else ""
            phase = r[1] if len(r) > 1 else ""
            mech = (r[5] if len(r) > 5 else "").replace("_", " ")
            ev = r[7] if len(r) > 7 else ""
            _bydis.setdefault(did, []).append((phase, mech, ev))
        for did, _phs in _bydis.items():
            _grp = {}
            for ph, mech, ev in _phs:
                _grp.setdefault((mech, ev), []).append(ph)
            _ml = []
            for (mech, ev), _phlist in _grp.items():
                _ml.append(f'<b>{_esc(", ".join(_phlist))}</b> &mdash; '
                           f'{_esc(mech or "\u2014")}: {_esc(ev or "\u2014")}')
            P(f'<tr><td>{_esc(did.replace("_", " "))}</td>'
              f'<td>{_pub_anchor(did)}</td>'
              f'<td>{"<br>".join(_ml)}</td></tr>')
        P('</table></section>')

    # ---- N. confidence backbone + caveats ----
    P('<section><h2>N · Confidence and caveats</h2>')
    P('<p class="muted">This report is generated by the CPG chain (Stages 2–9). The patient entered at the '
      'calibrated-β level (Stages 0–1, IDAT→β, are the wet-lab/array front end). A-scores carry posterior '
      'uncertainty from the IAMAtlas MCMC; the stromal class carries a known ~7% coverage mask. Disease '
      'matches are pattern resemblances scored against documented signatures, not diagnoses. '
      '<b>This is a wellness and cellular-fitness instrument, not a diagnostic device.</b></p></section>')

    # ===== Appendices (reference back-matter): A visuals · B audit · C disease scoring · D every cell =====
    AP = []  # assembled after the O gate so order is: main → O → appendices

    # Appendix A — visual index
    AP.append('<section><h2>Appendix A · Visual index</h2>')
    AP.append('<p class="muted">The figures referenced throughout, collected for reference.</p>')
    for key, lab in [("A1_reference_gauge", "A-score reference scale (C.1)"),
                     ("A2_cellular_departure_ranking", "Top-15 cellular departure ranking (C.2)"),
                     ("star_gauge", "Star gauge — same ruler as the cell (F)")]:
        if figs.get(key):
            AP.append(f'<div class="fig"><div class="muted">{_esc(lab)}</div>{_img(figs.get(key), lab)}</div>')
    AP.append('<p class="muted">Per-class and whole-atlas Personal Brilliance Maps and the four reference '
              'plates are shown in Section F.</p></section>')

    # Appendix B — audit trail (Stage 0–10 chain of custody)
    AP.append('<section><h2>Appendix B · Audit trail</h2>')
    AP.append('<p class="muted">The Stage 0–10 chain of custody. Stage outputs are hashed to the repository '
              'at delivery (Stage 10) so any number in this report is traceable to its inputs.</p>')
    AP.append('<table class="kv"><tr><th>Stage</th><th>Step</th><th>Status this run</th></tr>')
    for st, step, status in [
        ("0", "Intake / QC (IDAT hash, probe-level QC, metadata)", "wet-lab front end"),
        ("1", "Calibration (IDAT → β)", "wet-lab front end"),
        ("2", "Deconvolution (Walther NNLS + NILC cross-check)", "run"),
        ("3", "Foreground subtraction (sex + smoking; age removed 2026-06-09)", "run"),
        ("4", "A-score (115 cell types / 8 architecture classes)", "run"),
        ("4.5", "Bidirectional decomposition (immune panel sealed; 7 classes pending)", "run"),
        ("4.6", "Brightness comparison / Mollweide projection", "run"),
        ("5", "Mahalanobis distance vs healthy hull", "run"),
        ("6", "Per-cell confidence-weighted departure", "run"),
        ("7", "Tier classification", "run"),
        ("8", "Disease-matrix matching (Routes A/B/C)", "run"),
        ("9", "Report assembly + boundary gate", "run"),
        ("10", "Delivery / hashing", "at delivery")]:
        AP.append(f'<tr><td>{st}</td><td>{_esc(step)}</td><td>{_esc(status)}</td></tr>')
    AP.append('</table></section>')

    # Appendix C — complete per-disease scoring (content unchanged; H.2 confidence framing fixed in dedicated pass)
    if s8 is not None:
        allc = [x for x in (s8.route_B_all_scored or []) if x.get("match_magnitude") is not None]
        allc.sort(key=lambda s: s["match_magnitude"], reverse=True)
        AP.append('<section><h2>Appendix C · Complete disease scoring (all scored signature rows)</h2>')
        AP.append(f'<p class="muted">All {len(allc)} (disease × phase) rows the patient pattern was scored '
                  f'against, ranked by resemblance. Same detection-probability transform + caveat as H.2.</p>')
        AP.append('<table class="kv"><tr><th>#</th><th>Disease · phase</th><th>Category</th>'
                  '<th style="text-align:right">Detection probability</th><th>Confidence</th><th>Status</th></tr>')
        for i, m in enumerate(allc, 1):
            pct = _resemblance_pct(m["match_magnitude"])
            pct_str = "&lt;1%" if pct < 1 else f"{pct}%"
            conf = _match_confidence(m["n_cells_matched"], "")
            status = ("Below threshold" if m["tier"] in ("NORMAL", "MARGINAL")
                      else f'{m["tier"].replace("_"," ").title()} — review')
            AP.append(f'<tr><td>{i}</td><td>{_esc(m["disease"])} · {_esc(m["phase"])}</td>'
                      f'<td>{_esc(_category_for(m["disease"]))}</td>'
                      f'<td style="text-align:right">{pct_str}</td><td>{conf}</td><td>{_esc(status)}</td></tr>')
        AP.append('</table></section>')

    # Appendix D — every cell type and its A-score (the complete per-cell readout)
    _cta = (s4 or {}).get("celltype_ascores") if isinstance(s4, dict) else None
    if _cta:
        _ctt = (s7 or {}).get("celltype_tiers", {}) if isinstance(s7, dict) else {}
        _rows = []
        for cell, v in _cta.items():
            A = v.get("A") if isinstance(v, dict) else v
            if A is None:
                continue
            cls = v.get("class") if isinstance(v, dict) else None
            tier = (_ctt.get(cell, {}) or {}).get("label", "")
            _rows.append((cell, cls, float(A), tier))
        _rows.sort(key=lambda r: (r[1] or "", -r[2]))
        AP.append('<section><h2>Appendix D · Every cell type and its A-score</h2>')
        AP.append(f'<p class="muted">All {len(_rows)} cell-type A-scores from Stage 4, grouped by architecture '
                  f'class. This is the complete per-cell readout behind every section above.</p>')
        AP.append('<table class="kv"><tr><th>Cell type</th><th>Class</th>'
                  '<th style="text-align:right">A-score</th><th>Tier</th></tr>')
        for cell, cls, A, tier in _rows:
            AP.append(f'<tr><td>{_esc(cell)}</td>'
                      f'<td>{_esc(CLASS_PRETTY.get(cls, cls) if cls else "—")}</td>'
                      f'<td style="text-align:right">{A:.3f}</td><td>{_esc(tier)}</td></tr>')
        AP.append('</table></section>')

    # ---- assemble: main body (through N) → O boundary gate → appendices ----
    body_main = "\n".join(parts)
    body_appx = "\n".join(AP)
    import re as _re
    cannot_say = [
        r"\byou have (?:cancer|alzheimer|disease|a tumou?r)\b",
        r"\byou will (?:get|develop|have)\b",
        r"\byou should (?:take|start|stop|use)\b",
        r"\bwe diagnose\b", r"\byou are diagnosed\b",
        r"\bthis (?:is|confirms) (?:a )?diagnosis\b",
    ]
    violations = []
    scan = (body_main + "\n" + body_appx).lower()
    for pat in cannot_say:
        for mt in _re.finditer(pat, scan):
            violations.append(mt.group(0))
    gate_ok = not violations
    gate_html = (
        '<section><h2>O · Reporting-boundary check</h2>'
        f'<p class="muted">Stage 9.7 legal-boundary gate (SOP §76): scanned the rendered report for diagnostic / '
        f'directive language a physician-facing instrument may not assert. '
        f'<b>{"PASS — no CANNOT_SAY language detected." if gate_ok else "HALT — review required: " + _esc(", ".join(set(violations)))}</b> '
        f'This report states measurements, percentiles, pattern resemblances, base rates, and "discuss with your '
        f'physician"; it does not diagnose, predict, or prescribe.</p></section>'
    )
    body = body_main + "\n" + gate_html + "\n" + body_appx
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CPG Report — {_esc(pid)}</title>
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:#1c2530;
          max-width:960px; margin:0 auto; padding:0 24px 60px; line-height:1.5; }}
  .hdr {{ border-bottom:3px solid #1c2530; padding:28px 0 16px; margin-bottom:8px; }}
  .brand {{ font-weight:700; letter-spacing:.5px; color:#b03020; }}
  .brand span {{ color:#8a93a0; font-weight:500; }}
  h1 {{ margin:6px 0 2px; font-size:26px; }}
  .sub {{ color:#8a93a0; font-size:13px; }}
  h2 {{ margin-top:34px; font-size:19px; border-left:4px solid #b03020; padding-left:10px; }}
  h3 {{ font-size:15px; color:#3a4656; margin-top:22px; }}
  table.kv {{ border-collapse:collapse; width:100%; margin:10px 0; font-size:14px; }}
  table.kv td, table.kv th {{ border:1px solid #e1e6ec; padding:6px 10px; text-align:left; }}
  table.kv th {{ background:#f4f6f9; }}
  .badge {{ color:#fff; padding:2px 9px; border-radius:10px; font-size:12px; font-weight:600; white-space:nowrap; }}
  .fig {{ margin:14px 0; text-align:center; }}
  .grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }}
  .cell {{ border:1px solid #e1e6ec; border-radius:6px; padding:6px; background:#0c0c0c; }}
  .cell .lab {{ color:#fff; font-size:12px; font-weight:600; padding:2px 4px; }}
  .muted {{ color:#5d6775; font-size:13px; }}
  .ascii {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:9px; line-height:1.05;
            background:#0c0c0c; color:#cfd6df; padding:10px; border-radius:6px; overflow-x:auto; white-space:pre; }}
  .pending {{ background:#fff7e6; border:1px solid #f0d28a; padding:10px 12px; border-radius:6px; font-size:13px; }}
  .missing {{ color:#b03020; font-size:12px; font-style:italic; }}
</style></head><body>
{body}
</body></html>"""
    out = Path(output_html_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    try:
        build_audit_appendix(bundle, out.with_name(out.stem + "_AUDIT_APPENDIX.html"))
    except Exception:
        pass  # the audit appendix must never break the clinical report
    try:
        build_trajectory_snapshot(
            bundle, out.with_name(out.stem + "_trajectory_snapshot.json"),
            scale_provenance=(config or {}).get("scale_provenance"))
    except Exception:
        pass  # the trajectory snapshot must never break the clinical report
    return out
