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
C (architectural state + gauges + top-15 ranking), D (cellular age), E (Mahalanobis),
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


def _esc(x):
    return html.escape(str(x))


def _tier_badge(tier_id, label=None):
    label = (label or tier_id).replace("\n", " ")
    c = TIER_CSS.get(tier_id, "#777")
    return f'<span class="badge" style="background:{c}">{_esc(label)}</span>'


def build_report(bundle, output_html_path, atlas_plate_paths=None, config=None):
    """Render the CPG report HTML from a run_pipeline bundle. Returns the output path."""
    pid = bundle.get("patient_id", "patient")
    meta = bundle.get("patient_meta", {})
    s4 = bundle["stage4"]; s5 = bundle["stage5"]; s6 = bundle["stage6"]; s7 = bundle["stage7"]
    s45 = bundle.get("stage4_5"); s46 = bundle.get("stage4_6")
    figs = bundle.get("figures", {}) or {}
    bril = figs.get("brilliance_maps", {}) or {}
    atlas_plate_paths = atlas_plate_paths or {}

    parts = []
    P = parts.append

    # ---- header + exec summary ----
    max_tier = s7.get("max_class_tier")
    max_tier_label = (s7["class_tiers"].get(max_tier, {}) or {}).get("label", max_tier) if max_tier else "—"
    n_breach = s7.get("n_cells_breach", 0)
    chrono = meta.get("age")
    age_med = getattr(s6, "age_median", None)
    maha = s5.get("mahalanobis_distance"); maha_status = s5.get("status")
    exec_bits = []
    exec_bits.append(f"Overall architectural state reads <b>{_esc((max_tier_label or '').replace(chr(10),' '))}</b> "
                     f"(highest-tier architectural class).")
    exec_bits.append(f"<b>{n_breach}</b> of 115 cell types crossed the breach line (A &ge; {s7.get('breach_line',1.10)}).")
    if age_med is not None and chrono is not None:
        delta = age_med - chrono
        dirn = "older than" if delta > 1 else ("younger than" if delta < -1 else "consistent with")
        exec_bits.append(f"Median cellular age <b>{age_med:.0f}</b> vs chronological <b>{chrono:.0f}</b> "
                         f"({dirn} chronological age).")
    if maha is not None:
        exec_bits.append(f"Universal architectural departure (Mahalanobis) = <b>{maha:.1f}</b> ({_esc(maha_status)}).")

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
    P('</section>')

    # ---- C. architectural state ----
    P('<section><h2>C · Architectural state — the cell-level view</h2>')
    P('<h3>C.1 The reference scale — what each A-score value means</h3>')
    P(f'<div class="fig">{_img(figs.get("A1_reference_gauge"), "A-score reference gauge")}</div>')
    P('<h3>C.2 Top 15 cells by magnitude of departure</h3>')
    P(f'<div class="fig">{_img(figs.get("A2_cellular_departure_ranking"), "Top-15 cellular departure ranking")}</div>')
    # the same data as a table
    ranked = sorted([(n, v) for n, v in ct.items() if isinstance(v, dict) and v.get("A") is not None],
                    key=lambda kv: abs(kv[1]["A"] - 1.0), reverse=True)[:15]
    P('<table class="kv"><tr><th>#</th><th>Cell type</th><th>Class</th><th>A-score</th><th>Δ from 1.00</th></tr>')
    for i, (n, v) in enumerate(ranked, 1):
        P(f'<tr><td>{i}</td><td>{_esc(n)}</td><td>{_esc(v.get("class",""))}</td>'
          f'<td>{v["A"]:.3f}</td><td>{v["A"]-1.0:+.3f}</td></tr>')
    P('</table></section>')

    # ---- D. cellular age ----
    P('<section><h2>D · Cellular aging — departure from age-adjusted normal</h2>')
    cap = getattr(s6, "cellular_age_per_class", {}) or {}
    P(f'<p>Median cellular age <b>{_esc(getattr(s6,"age_median","—"))}</b> '
      f'(chronological {_esc(getattr(s6,"chronological_age","—"))}); '
      f'spread {_esc(getattr(s6,"age_spread","—"))}.</p>')
    accel = getattr(s6, "compartments_accelerated", []) or []
    decel = getattr(s6, "compartments_decelerated", []) or []
    if accel:
        P(f'<p><b>Accelerated compartments:</b> {_esc(", ".join(map(str, accel)))}</p>')
    if decel:
        P(f'<p><b>Decelerated compartments:</b> {_esc(", ".join(map(str, decel)))}</p>')
    if cap:
        P('<table class="kv"><tr><th>Class</th><th>Cellular age</th></tr>')
        for cls, age in cap.items():
            P(f'<tr><td>{_esc(CLASS_PRETTY.get(cls,cls))}</td><td>{age:.1f}</td></tr>')
        P('</table>')
    P('</section>')

    # ---- E. Mahalanobis ----
    P('<section><h2>E · Universal architectural departure — Mahalanobis distance</h2>')
    P(f'<p>Distance from the healthy-cohort centroid across all 115 cell-type axes: '
      f'<b>{maha:.2f}</b> — status <b>{_esc(maha_status)}</b> '
      f'({s5.get("n_features_used","?")} features used, {s5.get("n_features_imputed",0)} imputed).</p>')
    top = s5.get("top10_axis_contributions") or []
    if top:
        P('<table class="kv"><tr><th>#</th><th>Axis (cell type)</th><th>Contribution</th></tr>')
        for i, c in enumerate(top, 1):
            if isinstance(c, dict):
                nm = c.get("cell_type") or c.get("axis") or c.get("name") or list(c.values())[0]
                val = c.get("contribution") or c.get("value") or c.get("z")
                P(f'<tr><td>{i}</td><td>{_esc(nm)}</td><td>{_esc(round(val,3) if isinstance(val,(int,float)) else val)}</td></tr>')
        P('</table>')
    P('</section>')

    # ---- F. Personal Brilliance Maps ----
    P('<section><h2>F · Personal Brilliance Map — your methylation pattern vs the Cosmic Methylome Background</h2>')
    P('<p class="muted">Your departures projected onto the same HEALPix sphere as the reference atlas. '
      'Eight per-class panels (one per architecture class) plus one whole-atlas map of your entire methylome '
      'against the whole-450K reference — the same mathematics the Planck mission uses on the microwave sky.</p>')
    if atlas_plate_paths.get("plate1"):
        P('<h3>F.2 Reference atlas (Plate 1 — Cosmic Methylome Background)</h3>')
        P(f'<div class="fig">{_img(atlas_plate_paths["plate1"], "Plate 1 reference")}</div>')
    P('<h3>F.3 Your 8 per-class Personal Brilliance Maps</h3>')
    P('<div class="grid">')
    for cls in ARCH_CLASSES:
        if cls in bril:
            P(f'<div class="cell"><div class="lab">{_esc(CLASS_PRETTY.get(cls,cls))}</div>{_img(bril[cls], cls)}</div>')
    P('</div>')
    P('<h3>F.4 The whole-atlas Personal Brilliance Map — your entire methylome vs whole-450K CMB</h3>')
    P(f'<div class="fig">{_img(bril.get("whole_atlas"), "whole-atlas brilliance map")}</div>')
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
        # H.2 closest matches
        P('<h3>H.2 Closest pattern matches</h3>')
        P('<table class="kv"><tr><th>Signature (disease · phase)</th><th>Match magnitude</th>'
          '<th>Cells matched</th><th>Tier</th></tr>')
        for m in rb:
            P(f'<tr><td>{_esc(m["disease"])} · {_esc(m["phase"])}</td>'
              f'<td>{m["match_magnitude"]:+.3f}</td><td>{m["n_cells_matched"]}</td>'
              f'<td>{_tier_badge(m["tier"], m["tier"])}</td></tr>')
        P('</table>')
        P('<p class="muted">Match magnitude is a sign-aligned, √n-weighted concordance between your per-cell '
          'departures and each signature\'s documented per-cell Cohen\'s d (engine schema v1.2). '
          'It is a resemblance score, not a probability of disease.</p>')
        # H.3 routes A / C
        P('<h3>H.3 Architectural-alarm & bidirectional channels</h3>')
        P(f'<p><b>Route A (universal architectural alarm):</b> Mahalanobis {rA.get("mahalanobis_d",0):.1f} '
          f'vs hull p95 {rA.get("p95","—")} — <b>{"TRIGGERED" if rA.get("fired") else "within hull"}</b>.</p>')
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
        if maha is not None:
            conv.append(f'the Mahalanobis distance is <b>{maha:.1f}</b> ({_esc(maha_status)})')
        top = s5.get("top10_axis_contributions") or []
        if top:
            names = []
            for c in top[:3]:
                if isinstance(c, dict):
                    names.append(str(c.get("cell_type") or c.get("axis") or c.get("name") or ""))
            if names:
                conv.append(f'the cells driving the distance are <b>{_esc(", ".join(n for n in names if n))}</b>')
        P('<p>' + ('; '.join(conv) if conv else 'No convergent pattern surfaced.') +
          '. The cell ranking (C.2), the Mahalanobis decomposition (E), the Personal Brilliance Map (F) and '
          'the disease matrix (H.2) are four views of the same per-cell departure data — where they converge, '
          'the pattern is real.</p>')

    P('</section>')

    # ---- I. cross-disease universal alarm channel ----
    P('<section><h2>I · Cross-disease universal alarm channel</h2>')
    P('<p class="muted">The immune-atlas card carries a 6,018-CpG cross-disease firing-pattern map with a '
      '12-CpG opposing-direction sub-channel (the VAL-016 universal-alarm signature). The Stage 8 Route A '
      'residual-map-overlap channel computes the patient\'s Pearson overlap with that map. '
      'Per-CpG residual-overlap is the next wiring step (the channel artifact and thresholds are staged; '
      'overlap is computed once the per-CpG departure vector is exposed from Stage 4.6).</p></section>')

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

    # ---- N / O. confidence backbone + caveats ----
    P('<section><h2>N · Confidence and caveats</h2>')
    P('<p class="muted">This report is generated by the CPG chain (Stages 2–9). The patient entered at the '
      'calibrated-β level (Stages 0–1, IDAT→β, are the wet-lab/array front end). A-scores carry posterior '
      'uncertainty from the IAMAtlas MCMC; the stromal class carries a known ~7% coverage mask. Disease '
      'matches are pattern resemblances scored against documented signatures, not diagnoses. '
      '<b>This is a wellness and cellular-fitness instrument, not a diagnostic device.</b></p></section>')

    body = "\n".join(parts)

    # ---- §76 legal-boundary gate — scan the rendered body for CANNOT_SAY language ----
    import re as _re
    cannot_say = [
        r"\byou have (?:cancer|alzheimer|disease|a tumou?r)\b",
        r"\byou will (?:get|develop|have)\b",
        r"\byou should (?:take|start|stop|use)\b",
        r"\bwe diagnose\b", r"\byou are diagnosed\b",
        r"\bthis (?:is|confirms) (?:a )?diagnosis\b",
    ]
    violations = []
    low = body.lower()
    for pat in cannot_say:
        for mt in _re.finditer(pat, low):
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
    body = body + "\n" + gate_html
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
  .pending {{ background:#fff7e6; border:1px solid #f0d28a; padding:10px 12px; border-radius:6px; font-size:13px; }}
  .missing {{ color:#b03020; font-size:12px; font-style:italic; }}
</style></head><body>
{body}
</body></html>"""
    out = Path(output_html_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out
