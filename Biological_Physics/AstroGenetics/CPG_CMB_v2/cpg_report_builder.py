#!/usr/bin/env python3
"""
cpg_report_builder.py — CPG v1 LEAN report builder.

Consumes the bundle from walther_clinical.run_pipeline and renders one HTML report that
answers the only question v1 is built to answer: do any of the 17 core diseases show, in
which of the two detection modes, and how strongly. Every value carries its CI. The whole
methodology is written out in plain language for the clinician AND for a future AI, so no
context is lost between sessions.

DESIGN (matches Walther_CPG_v1_chain_flowchart_v4):
  Mode 1 - ARCHITECTURAL concordance: scale-invariant cosine of the patient's derived
           A-departure vector against each disease signature. The workhorse for field-effect
           and systemic disease, where the signal is written into the ABUNDANT immune cells we
           can score cleanly. Reports RESEMBLANCE, never a probability.
  Mode 2 - CELL-OF-ORIGIN presence: a cell circulating in whole blood that should not be there.
           BBB-protected CNS cells (cortical neurons, glia, oligodendrocytes) are a hard red
           flag on presence ALONE -- they crossed a physical barrier to be in blood. Other shed
           cells (epithelial) are interpreted as presence PLUS A-score: normal A = benign
           turnover, abnormal A = concerning. A flag for specialist referral, never a diagnosis.
NO cohort methodology anywhere: the patient is never standardized against a population.
"""
from __future__ import annotations
import html as _html
import json
from datetime import datetime

# Cells that live behind the blood-brain barrier. Their presence in blood is a barrier breach
# -- a red flag regardless of quantity or A-score.
BBB_PROTECTED = {
    "cortical_neurons", "neurons_pooled", "neuron", "NeuMa", "NeuIm",
    "glia", "Glia", "astrocytes", "brain_astrocytes", "brain_pooled",
    "oligodendrocytes", "Oligo", "OPC", "microglia",
}

# A-score gauge bounds — the meaningful range. A value outside this is not a real
# architectural reading; it is the signature of un-normalized input (see the guard).
GAUGE_LO, GAUGE_HI = 0.90, 1.35

# Bulk/aggregate reference profiles that live in the atlas but are NOT individual cells.
# They must never appear in a per-cell list, and they distort the picture by absorbing
# deconvolution mass that belongs to real cells (whole_blood alone took ~29% on real blood).
NON_CELL_AGGREGATES = {
    "whole_blood", "PBMC", "pbmc", "buffy_coat", "leukocytes", "leukocytes_total",
    "blood", "blood_total", "wbc", "WBC",
}


def _is_cell(name):
    return name not in NON_CELL_AGGREGATES


def a_score_gauge_svg(scored=None, width=860):
    """The A-score reference gauge (0.90-1.20), with optional patient markers."""
    lo, hi = 0.90, 1.20
    ml, mr, barY, barH = 54, 24, 70, 64
    W = width; innerW = W - ml - mr
    def x(a):
        a = max(lo, min(hi, a)); return ml + (a - lo) / (hi - lo) * innerW
    regions = [(0.90,0.95,"#AEC6E4","SUPPRESSED",""),(0.95,1.04,"#A8D5A8","NORMAL","healthy range"),
               (1.04,1.07,"#F6DDA0","ELEVATED",""),(1.07,1.10,"#F0C674","SIG. ELEVATED",""),
               (1.10,1.20,"#D98A82","BREACH","")]
    s = [f'<svg viewBox="0 0 {W} 210" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;font-family:system-ui,sans-serif">']
    s.append(f'<text x="{W/2}" y="26" text-anchor="middle" font-size="15" font-weight="700" fill="#16202a">A-score reference gauge &#8212; the calibration scale</text>')
    for a0,a1,fill,lab,sub in regions:
        x0,x1 = x(a0),x(a1); s.append(f'<rect x="{x0:.1f}" y="{barY}" width="{(x1-x0):.1f}" height="{barH}" fill="{fill}"/>'); cx=(x0+x1)/2
        if lab in ("NORMAL","BREACH"):
            s.append(f'<text x="{cx:.1f}" y="{barY+barH/2-2:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#16202a">{lab}</text>')
            if sub: s.append(f'<text x="{cx:.1f}" y="{barY+barH/2+13:.1f}" text-anchor="middle" font-size="10.5" fill="#16202a">{sub}</text>')
    s.append(f'<line x1="{x(1.0):.1f}" y1="{barY-8}" x2="{x(1.0):.1f}" y2="{barY+barH+8}" stroke="#16202a" stroke-width="2.5"/>')
    s.append(f'<text x="{x(1.0):.1f}" y="{barY-14}" text-anchor="middle" font-size="11" font-weight="700" fill="#16202a">A = 1.0  healthy reference</text>')
    s.append(f'<line x1="{x(1.07):.1f}" y1="{barY-8}" x2="{x(1.07):.1f}" y2="{barY+barH+8}" stroke="#C77514" stroke-width="2" stroke-dasharray="6 4"/>')
    s.append(f'<text x="{x(1.07):.1f}" y="{barY-14}" text-anchor="middle" font-size="9.5" font-weight="700" fill="#C77514">1.07 Warburg</text>')
    for a,lab in [(1.055,"ELEVATED"),(1.085,"SIG. ELEVATED"),(0.925,"SUPPRESSED")]:
        s.append(f'<text x="{x(a):.1f}" y="{barY+barH+20:.1f}" text-anchor="middle" font-size="9" fill="#4A555C">{lab}</text>')
    s.append(f'<text x="{x(1.135):.1f}" y="{barY-14:.1f}" text-anchor="middle" font-size="9.5" font-weight="700" fill="#B2182B">1.10 breach &#8594; malignancy</text>')
    for t in [0.90,0.95,1.00,1.05,1.10,1.15,1.20]:
        s.append(f'<line x1="{x(t):.1f}" y1="{barY+barH+26}" x2="{x(t):.1f}" y2="{barY+barH+31}" stroke="#9aa3ab" stroke-width="1"/>')
        s.append(f'<text x="{x(t):.1f}" y="{barY+barH+43:.1f}" text-anchor="middle" font-size="10" fill="#4A555C" font-family="ui-monospace,Menlo,monospace">{t:.2f}</text>')
    s.append(f'<text x="{W/2}" y="{barY+barH+60:.1f}" text-anchor="middle" font-size="10.5" fill="#4A555C">Architectural A-score &#160;(mean of H(&#946;) / H_min(class) over panel CpGs)</text>')
    if scored:
        for lab,a in scored.items():
            if a is None: continue
            on = lo <= a <= hi; mx = x(a); my = barY-2; col = "#16202a" if on else "#B2182B"
            s.append(f'<polygon points="{mx:.1f},{my:.1f} {mx-5:.1f},{my-9:.1f} {mx+5:.1f},{my-9:.1f}" fill="{col}"/>')
    s.append('</svg>'); return "".join(s)


def departure_ranking_svg(bundle, width=900, max_rows=15, reliable_only=True):
    """Diverging departure ranking. Per SOP §30, the cell tier is indicative and weakly
    separable in v0.1: cells the deconvolver does not resolve read the bulk mixture at their
    one-vs-rest markers and pin spuriously. So the CHART shows the reliable (deconvolver-
    resolved) cells -- the real per-cell fingerprint -- ranked by distance from the 1.0
    baseline. The full per-cell table (all present-class cells, with the confidence column)
    is shown below the chart so nothing is hidden. Direction is the diagnostic part: a bar
    points RIGHT when elevated, LEFT when suppressed."""
    s4 = bundle["stage4"]["celltype_ascores"]
    hmin = bundle["stage4"].get("h_min_by_class", {}) or {}
    def _above_floor(ct, r):
        floor = hmin.get(r.get("class"))
        return floor is None or r["A"] >= float(floor)
    cells = [(ct, r) for ct, r in s4.items()
             if r.get("A") is not None and _is_cell(ct) and _above_floor(ct, r)]
    if reliable_only:
        rel = [(ct, r) for ct, r in cells if r.get("fraction_tier") == "reliable"]
        if len(rel) >= 1:
            cells = rel
    if not cells:
        return "<p class='meta'>No cells scored above the presence floor on this sample.</p>"
    cells.sort(key=lambda kv: -abs(kv[1]["A"] - 1.0))
    total = len(cells)
    if max_rows:
        cells = cells[:max_rows]
    n = len(cells)
    _title = (f"Cellular departure ranking &#8212; top {n} of {total} resolved cells"
              if total > n else f"Cellular departure ranking &#8212; {n} resolved cell(s)")
    allv = [1.0]
    for _, r in cells:
        allv += [r["A"], r.get("A_ci_lo") or r["A"], r.get("A_ci_hi") or r["A"]]
    dmax = max(0.10, max(abs(v - 1.0) for v in allv))
    xlo, xhi = 1.0 - dmax * 1.12, 1.0 + dmax * 1.12
    mL, mR, top, rh, bot = 300, 70, 80, 30, 78
    H = top + n * rh + bot
    innerW = width - mL - mR

    def X(a):
        a = max(xlo, min(xhi, a))
        return mL + (a - xlo) / (xhi - xlo) * innerW

    ABBR = {"immune": "IMM", "secretory": "SEC", "stromal": "STR", "progenitor": "PRO",
            "cycling": "CYC", "terminal": "TER", "stem_adult": "SAD", "stem_pluri": "SPL"}

    def col(a):
        if a >= 1.10: return "#B2182B"
        if a >= 1.07: return "#D98A45"
        if a > 1.04:  return "#E8B04B"
        if a >= 0.95: return "#86C28B"
        if a >= 0.90: return "#6B9BC4"
        return "#3E6E99"

    o = [f'<svg viewBox="0 0 {width} {H}" xmlns="http://www.w3.org/2000/svg" '
         'style="max-width:100%;height:auto;font-family:system-ui,sans-serif">']
    o.append(f'<text x="{width/2}" y="26" text-anchor="middle" font-size="15" font-weight="700" '
             f'fill="#16202a">{_title}</text>')
    o.append(f'<text x="{width/2}" y="45" text-anchor="middle" font-size="10.5" fill="#6b7780">'
             'ranked by distance from baseline &#183; right = elevated &#183; left = suppressed '
             '&#183; bold border = deconvolver-resolved (reliable) &#183; whiskers = 95% CI</text>')
    pT, pB = top - 10, top + n * rh
    for a0, a1, fill in [(xlo, 0.95, "#EAF2F8"), (0.95, 1.04, "#EAF6EA"), (1.04, 1.07, "#FBF3DF"),
                         (1.07, 1.10, "#F7E6CC"), (1.10, xhi, "#F6DEDC")]:
        if a1 <= xlo or a0 >= xhi: continue
        x0, x1 = X(max(a0, xlo)), X(min(a1, xhi))
        o.append(f'<rect x="{x0:.1f}" y="{pT}" width="{(x1-x0):.1f}" height="{(pB-pT):.1f}" fill="{fill}"/>')
    o.append(f'<line x1="{X(1.0):.1f}" y1="{pT}" x2="{X(1.0):.1f}" y2="{pB+8}" stroke="#16202a" stroke-width="2"/>')
    o.append(f'<text x="{X(1.0):.1f}" y="{pB+32}" text-anchor="middle" font-size="10" font-weight="700" fill="#16202a">1.00 healthy</text>')
    if xhi >= 1.07:
        o.append(f'<line x1="{X(1.07):.1f}" y1="{pT}" x2="{X(1.07):.1f}" y2="{pB}" stroke="#C77514" stroke-width="1.4" stroke-dasharray="6 4"/>')
        o.append(f'<text x="{X(1.07):.1f}" y="{pT-16}" text-anchor="middle" font-size="9" fill="#C77514" font-weight="700">1.07 Warburg</text>')
    if xhi >= 1.10:
        o.append(f'<line x1="{X(1.10):.1f}" y1="{pT}" x2="{X(1.10):.1f}" y2="{pB}" stroke="#B2182B" stroke-width="1.4" stroke-dasharray="2 3"/>')
        o.append(f'<text x="{X(1.10):.1f}" y="{pT-4}" text-anchor="middle" font-size="9" fill="#B2182B" font-weight="700">1.10 breach</text>')
    import math as _math
    _step = 0.10 if (xhi - xlo) > 0.45 else 0.05
    _t = _math.ceil(xlo / _step) * _step
    while _t <= xhi + 1e-9:
        tk = round(_t, 2)
        o.append(f'<line x1="{X(tk):.1f}" y1="{pB+2}" x2="{X(tk):.1f}" y2="{pB+7}" stroke="#9aa3ab" stroke-width="1"/>')
        o.append(f'<text x="{X(tk):.1f}" y="{pB+50}" text-anchor="middle" font-size="9" fill="#6b7780" font-family="ui-monospace,Menlo,monospace">{tk:.2f}</text>')
        _t += _step
    for i, (ct, r) in enumerate(cells):
        a = r["A"]; lo = r.get("A_ci_lo"); hi = r.get("A_ci_hi")
        cy = top + i * rh + rh / 2
        ab = ABBR.get(r.get("class"), (r.get("class") or "")[:3].upper())
        x1, xa = X(1.0), X(a)
        bx0, bx1 = min(x1, xa), max(x1, xa)
        _reliable = (r.get("fraction_tier") == "reliable")
        _stroke = ("#16202a", "1.6") if _reliable else ("#9aa3ab", "0.5")
        o.append(f'<rect x="{bx0:.1f}" y="{cy-9:.1f}" width="{max(1,bx1-bx0):.1f}" height="18" '
                 f'fill="{col(a)}" stroke="{_stroke[0]}" stroke-width="{_stroke[1]}"/>')
        if lo is not None and hi is not None:
            o.append(f'<line x1="{X(lo):.1f}" y1="{cy:.1f}" x2="{X(hi):.1f}" y2="{cy:.1f}" stroke="#3a3a3a" stroke-width="1"/>')
            for xx in (X(lo), X(hi)):
                o.append(f'<line x1="{xx:.1f}" y1="{cy-4:.1f}" x2="{xx:.1f}" y2="{cy+4:.1f}" stroke="#3a3a3a" stroke-width="1"/>')
        bbb = " &#9888;" if ct in BBB_PROTECTED else ""
        o.append(f'<text x="{mL-14:.1f}" y="{cy+4:.1f}" text-anchor="end" font-size="11.5" fill="#16202a">'
                 f'{i+1}. {_esc(ct)}{bbb} <tspan fill="#7a8690">({ab})</tspan></text>')
        vx = (bx1 + 6) if a >= 1.0 else (bx0 - 6)
        anc = "start" if a >= 1.0 else "end"
        o.append(f'<text x="{vx:.1f}" y="{cy+4:.1f}" text-anchor="{anc}" font-size="11" font-weight="700" fill="#16202a">{a:.3f}</text>')
    o.append('</svg>')
    return "".join(o)


def _input_scale_ok(s4):
    """The real floor is per-class H_min, not a fixed gauge number. A class scored BELOW
    its own H_min floor is impossible for a cell still holding its identity, so that is the
    signature of a data/scale problem. BUT a single phantom class (e.g. stem_pluri flagged
    present in whole blood at ~0% then scored below floor) must NOT condemn the whole sample:
    genuine un-normalized beta drags the MAJORITY of classes off-scale, not one. So the guard
    fails only when MORE THAN ONE assessable class is below floor (systemic), not for a lone
    phantom. Returns (ok, n_below_floor, n_assessable)."""
    hmin = s4.get("h_min_by_class", {}) or {}
    below = []
    n = 0
    for cls, r in (s4.get("class_ascores", {}) or {}).items():
        a = r.get("A")
        if a is None:
            continue
        n += 1
        floor = hmin.get(cls)
        if floor is not None and a < float(floor):
            below.append(cls)
    return (len(below) <= 1), len(below), n


def _esc(x):
    return _html.escape(str(x)) if x is not None else ""


def _fmt_ci(lo, hi):
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def _exec_summary(bundle):
    flags = bundle.get("cell_of_origin_flags", [])
    ok, off, total = _input_scale_ok(bundle["stage4"])
    lines = []
    if not ok:
        lines.append(f"Architectural mode (Mode 1): <b>not assessable on this sample</b> — "
                     f"{off} of {total} classes scored below their own H_min floor, which is "
                     f"impossible for a cell that is present, so the β feeding the score is "
                     f"suspect. Per-class A-scores are withheld rather than reported as artifacts.")
    else:
        conc = bundle["stage8"].route_B_concordance
        strong = [m for m in conc
                  if m["resemblance"] in ("STRONG_RESEMBLANCE", "MODERATE_RESEMBLANCE")]
        if strong:
            top = strong[0]
            spec = top.get("specificity", "SPECIFIC")
            if spec == "NON_SPECIFIC_MYELOID":
                lines.append("Architectural mode (Mode 1): <b>non-specific systemic myeloid pattern</b> "
                             "&#8212; the signal is a generic myeloid/neutrophil shift (the kind seen in "
                             "infection, inflammation, stress and many conditions), not a fingerprint specific "
                             "to any one disease. Not flagged as a named malignancy.")
            elif spec == "NON_SPECIFIC_LYMPHOID":
                lines.append("Architectural mode (Mode 1): <b>non-specific systemic lymphoid-suppression "
                             "pattern</b> &#8212; a generic lymphopenia seen across many conditions, not "
                             "specific to any one disease.")
            else:
                lines.append(f"Architectural mode (Mode 1): closest pattern is <b>{_esc(top['disease'])}</b> "
                             f"({_esc(top['resemblance']).replace('_',' ').lower()}, cosine {top['cosine']:+.2f}, "
                             f"{top.get('n_signal',0)} signal-bearing cells). Resemblance, not a probability.")
        else:
            lines.append("Architectural mode (Mode 1): no disease pattern carries enough signal-bearing "
                         "shared cells to register a meaningful resemblance.")
    if flags:
        cls = ", ".join(_esc(f["class"]) for f in flags)
        lines.append(f"Cell-of-origin mode (Mode 2): <b class='red'>blood-brain-barrier cells circulating</b> "
                     f"(terminal class, {cls}) — barrier breach, refer for specialist evaluation.")
    else:
        lines.append("Cell-of-origin mode (Mode 2): no blood-brain-barrier cells circulating. "
                     "(Epithelial/cycling cells in blood are normal and are not flagged.)")
    return lines


def _mode1_rows(bundle):
    rows = ""
    for m in bundle["stage8"].route_B_concordance[:10]:
        resem = _esc(m['resemblance']).replace('_', ' ').lower()
        spec = m.get("specificity", "SPECIFIC")
        disease = _esc(m['disease'])
        if spec in ("NON_SPECIFIC_MYELOID", "NON_SPECIFIC_LYMPHOID"):
            disease = "<span class='muted'>non-specific systemic pattern</span> (" + disease + ")"
            resem += " <span class='muted'>(generic shift &#8212; not disease-specific)</span>"
        rows += (f"<tr><td>{disease}</td><td>{_esc(m['phase'])}</td>"
                 f"<td class='num'>{m['cosine']:+.3f}</td>"
                 f"<td class='num'>{m['direction_agreement']:.2f}</td>"
                 f"<td class='num'>{m.get('n_signal', 0)}</td>"
                 f"<td>{resem}</td></tr>")
    return rows or "<tr><td colspan='6' class='muted'>No pattern with enough signal-bearing shared cells.</td></tr>"


def _mode2_rows(bundle):
    flags = bundle.get("cell_of_origin_flags", [])
    rows = ""
    for f in flags:
        rows += (f"<tr><td>{_esc(f['class'])}</td>"
                 f"<td class='num'>{f['observed_fraction']*100:.2f}%</td>"
                 f"<td class='num'>{f['fraction_walther']*100:.2f}% / {f['fraction_nilc_raw']*100:.2f}%</td>"
                 f"<td><b class='red'>REVIEW — BBB</b></td>"
                 f"<td>{_esc(f['interpretation'])}</td></tr>")
    return rows or ("<tr><td colspan='5' class='muted'>No blood-brain-barrier cells circulating. "
                    "Epithelial, cycling and other shed cells in blood are normal and are not flagged here — "
                    "see composition.</td></tr>")


def _composition_rows(bundle):
    s2 = bundle.get("stage2") or {}
    walther = s2.get("class_fractions", {}) or {}
    nilc = (s2.get("nilc_fractions") or {})
    nilc_s = nilc.get("fractions", {}) if isinstance(nilc, dict) else {}
    nilc_r = nilc.get("raw_fractions", {}) if isinstance(nilc, dict) else {}
    rows = ""
    for cls in sorted(walther, key=lambda c: -walther.get(c, 0)):
        rows += (f"<tr><td>{_esc(cls)}</td>"
                 f"<td class='num'>{walther.get(cls,0)*100:.2f}%</td>"
                 f"<td class='num'>{nilc_s.get(cls,0)*100:.2f}%</td>"
                 f"<td class='num'>{nilc_r.get(cls,0)*100:+.2f}%</td></tr>")
    return rows


def _tier(a):
    if a is None:
        return ""
    if a >= 1.10:
        return "BREACH"
    if a >= 1.07:
        return "sig. elevated"
    if a > 1.04:
        return "elevated"
    if a >= 0.95:
        return "normal"
    return "suppressed"


def _ascore_rows(bundle):
    s4 = bundle["stage4"]
    cta = s4["celltype_ascores"]
    fr = (bundle.get("stage2") or {}).get("celltype_fractions", {}) or {}
    hmin = s4.get("h_min_by_class", {}) or {}
    rows = ""
    for ct, r in sorted(cta.items(), key=lambda kv: -(kv[1].get("A") or -9)):
        a = r.get("A")
        if a is None or not _is_cell(ct):
            continue
        cls = r.get("class")
        floor = hmin.get(cls)
        # A cell scored below its class H_min floor cannot exist as a present cell of that
        # class (H(beta) < H_min is physically impossible) -- it is an absent cell's markers
        # reading background, NOT a suppressed real cell. Exclude it; do not present it.
        if floor is not None and a < float(floor):
            continue
        below = False
        tier = "below floor" if below else _tier(a)
        frac = r.get("celltype_fraction")
        frac_txt = f"{frac*100:.1f}%" if frac is not None else "&#8212;"
        conf = r.get("fraction_tier", "indicative")
        bbb = " <span class='tag-bbb'>BBB</span>" if ct in BBB_PROTECTED else ""
        rows += (f"<tr><td>{_esc(ct)}{bbb}</td><td>{_esc(cls)}</td>"
                 f"<td class='num'>{frac_txt}</td>"
                 f"<td class='num'>{a:.3f}</td>"
                 f"<td class='num'>{_fmt_ci(r.get('A_ci_lo'), r.get('A_ci_hi'))}</td>"
                 f"<td>{_esc(tier)}</td><td>{_esc(conf)}</td></tr>")
    return rows


def _confirmation_section(bundle):
    """Render the second-chain Confirmation section. Returns '' when no flag fired
    (so the report ends after the primary chain, untouched)."""
    s5 = bundle.get("stage5")
    if not s5 or not s5.get("fired"):
        return ""
    m = s5["mahalanobis"]; an = s5["literature_anchor"]; rd = s5["residual_map"]
    tr = s5["trigger"]; ctx = bundle.get("context", {})

    band = "beyond" if m["beyond_healthy_band"] else "within"
    band_col = "var(--red)" if m["beyond_healthy_band"] else "var(--accent)"

    drv = "".join(
        f"<tr><td>{_esc(d['class'])}</td><td class='num'>{d['A']:.3f}</td>"
        f"<td class='num'>{d['z_from_floor']:+.1f}</td></tr>"
        for d in m["driving_classes"])

    if an.get("status") == "OK":
        ladder = "".join(
            f"<tr><td>{_esc(a['label'])}</td><td class='num'>{a['A']}</td>"
            f"<td>{_esc(a['context'])}</td><td class='meta'>{_esc(a['source'])}</td></tr>"
            for a in an["anchor_ladder"])
        anchor_html = (
            f"<p>Patient <b>{_esc(an['class'])}</b> class A = "
            f"<b>{an['patient_class_A']:.3f}</b>, nearest published anchor: "
            f"<b>{_esc(an['nearest_published_anchor']['label'])}</b> "
            f"(A={an['nearest_published_anchor']['A']}, "
            f"{_esc(an['nearest_published_anchor']['context'])}).</p>"
            f"<table><thead><tr><th>Published anchor</th><th>A</th><th>Context</th>"
            f"<th>Source</th></tr></thead><tbody>{ladder}</tbody></table>")
    else:
        anchor_html = f"<p class='muted'>Literature anchor: {_esc(an.get('note', an.get('status')))}</p>"

    if rd.get("status") == "OK":
        resid_html = f"<p>{_esc(rd['interpretation'])} <span class='meta'>({rd['cpgs_compared']} CpGs, {_esc(rd['map'])})</span></p>"
    else:
        resid_html = f"<p class='muted'>Residual map: {_esc(rd.get('note', rd.get('status')))}</p>"

    return f"""
<h2>Confirmation — second chain <span class="meta">(ran because Stage 8 flagged {_esc(tr['flagged_disease'])})</span></h2>
<div class="method">
<p>The second chain is independent and runs only on a Route B flag ({_esc(tr['gate_policy'])}).
It does not change anything above; it adjudicates whether the flag reflects a real departure.
Read alongside context: age {_esc(ctx.get('age'))} · sex {_esc(ctx.get('sex'))} · substrate {_esc(ctx.get('substrate'))}.</p>
<div style="background:#fff;border:1px solid var(--line);border-left:4px solid {band_col};border-radius:0 8px 8px 0;padding:10px 16px;margin:10px 0">
<b style="color:{band_col}">Verdict:</b> {_esc(s5['overall_verdict'])}</div>

<h3>A · Global departure (derived Mahalanobis — the adjudicator)</h3>
<p class="explain"><b>What this is and why it matters.</b> A <b>Mahalanobis distance</b> measures, in a single number,
how far the patient's whole cell architecture sits from healthy — but unlike a plain distance it weights each cell
class by how much it normally varies, so a small shift on a steady class counts for more than a large shift on a noisy
one. That gives the number a known statistical meaning (how far from the healthy floor in combined standard deviations),
which we test against a &chi;&sup2; threshold. We use it because the primary chain answers <em>"what does this pattern
resemble?"</em> but not <em>"is the architecture actually departed, and by how much?"</em> This is the step that answers
the second question and adjudicates the flag. Because the reference is the <b>derived healthy floor</b> — every class at
1.0 by construction, not a group of sick or healthy people — the number compares the patient to the architecture's own
baseline, never to a cohort.</p>
<p>d&sup2; = <b>{m['d_squared']}</b> vs &chi;&sup2;<sub>0.95</sub> threshold <b>{m['chi2_0.95_threshold']}</b>
over {m['n_present_classes']} present classes &rarr; <b style="color:{band_col}">{_esc(band)} the derived healthy band</b>.
Reference: {_esc(m['reference'])}.</p>
<table><thead><tr><th>Driving class</th><th>A-score</th><th>z from floor</th></tr></thead><tbody>{drv}</tbody></table>

<h3>B · Literature-anchor evidence <span class="meta">(published context, not the reference)</span></h3>
{anchor_html}

<h3>C · Residual-map concordance <span class="meta">(patient departure from the derived atlas baseline)</span></h3>
{resid_html}
</div>"""


def _trajectory_section(bundle):
    """Render the cross-visit Trajectory section (per-cell). Returns '' on a first
    visit, so the section appears only from the second draw on."""
    tj = bundle.get("trajectory")
    if not tj:
        return ""
    cells = tj.get("cell_changes", [])
    # split reliable vs indicative; lead with reliable
    reliable = [c for c in cells if c.get("reliable")]
    indicative = [c for c in cells if not c.get("reliable")]

    def _row(c, soft=False):
        if c.get("crossed_breach") or (c["kind"] == "new" and (c["now"] or 0) >= 1.10):
            col = "var(--red)"
        elif (c["delta"] is not None and c["delta"] > 0) or (c["kind"] == "new"):
            col = "#b7791f"
        elif c["delta"] is not None and c["delta"] < 0:
            col = "var(--accent)"
        else:
            col = "var(--soft)"
        note = (" · crossed 1.10 breach" if c.get("crossed_breach") and c["kind"] == "tracked"
                else " · newly detected" if c["kind"] == "new"
                else " · no longer detected" if c["kind"] == "dropped" else "")
        prior_s = f"{c['prior']:.3f}" if c["prior"] is not None else "&mdash;"
        now_s = f"{c['now']:.3f}" if c["now"] is not None else "&mdash;"
        delta_s = f"{c['delta']:+.3f}" if c["delta"] is not None else "&mdash;"
        name = _esc(c["cell"]) + ("" if not soft else "")
        wt = "600" if not soft else "400"
        return (f"<tr><td style='font-weight:{wt}'>{name}</td><td class='num'>{prior_s}</td>"
                f"<td class='num'>{now_s}</td>"
                f"<td class='num' style='color:{col}'>{delta_s}</td>"
                f"<td style='color:{col};font-size:12px'>{_esc(c['direction'])}{note}</td></tr>")

    rel_rows = "".join(_row(c) for c in reliable[:14])
    ind_rows = "".join(_row(c, soft=True) for c in indicative[:8])
    ind_block = (f"<tr><td colspan='5' class='meta' style='padding-top:8px'>indicative cells "
                 f"(softer read — abundance can confound a single cell's change)</td></tr>{ind_rows}"
                 if ind_rows else "")

    # rotation toward the flagged signature
    rot = tj.get("rotation"); rot_html = ""
    if rot:
        rcol = ("var(--red)" if rot["trend"] == "rotating toward the signature"
                else "var(--accent)" if rot["trend"] == "rotating away from the signature" else "var(--soft)")
        rot_html = (f"<p><b>Pattern rotation toward the {_esc(rot['disease'])} signature:</b> cosine "
                    f"<b>{rot['prior_cosine']}</b> &rarr; <b>{rot['now_cosine']}</b> "
                    f"(<span style='color:{rcol}'>{_esc(rot['trend'])}</span>). This is the whole departure "
                    f"vector's angle, not any one cell — the strongest single indicator of where the pattern is headed.</p>")
    sa = tj.get("self_alignment")
    sa_html = (f"<p>Overall pattern alignment with the prior draw: <b>{sa}</b> "
               f"(1.00 = identical direction; lower means the cell pattern has rotated).</p>" if sa is not None else "")
    gd = tj.get("global_departure"); gd_html = ""
    if gd:
        gcol = ("var(--red)" if gd["trend"] == "moving away from healthy"
                else "var(--accent)" if gd["trend"] == "returning toward healthy" else "var(--soft)")
        gd_html = (f"<p>Global departure (one summary number): <b>{gd['prior']}</b> &rarr; <b>{gd['now']}</b> "
                   f"(<span style='color:{gcol}'>{_esc(gd['trend'])}</span>).</p>")
    days = tj.get("days_between")
    span = f"{days} days earlier" if days is not None else "the prior draw"
    return f"""
<h2>Trajectory <span class="meta">(this draw vs {_esc(tj['prior_visit_label'])}, {span})</span></h2>
<div class="method">
<p class="explain"><b>Why the cells, not the class.</b> A single test is a snapshot; the architecture's
<em>direction of travel</em> is what separates a stable system from one walking toward disease — and that lives in the
cells, not the class average, which can sit flat while the cells beneath it move in opposite directions. This compares
each cell to the same patient's previous draw: the per-cell <em>change</em> is the unit, because each cell's dilution
offset is shared at both draws and cancels in the subtraction. The deconvolver-resolved cells lead (their change is
clean); the indicative cells follow as a softer read. The strongest signal of all is the whole pattern's
<em>angle</em> swinging toward a disease signature across visits.</p>
<p><b>{_esc(tj['headline'])}</b> {tj['n_prior_visits']} prior visit(s) on record; earliest {_esc(tj['first_visit_label'])}.</p>
{rot_html}
<p style="margin-bottom:4px;font-weight:600">Per-cell change (reliable cells first)</p>
<table><thead><tr><th>Cell</th><th>Prior A</th><th>Now A</th><th>&Delta;</th><th>Direction</th></tr></thead>
<tbody>{rel_rows}{ind_block}</tbody></table>
{sa_html}
{gd_html}
</div>"""


def build_report(bundle, out_path=None):
    ctx = bundle.get("context", {})
    rescued = bundle.get("nilc_rescued_classes", [])
    rescued_txt = ", ".join(f"{_esc(r.get('class'))} ({r.get('nilc_raw_fraction')})"
                            for r in rescued if "class" in r) or "none"
    exec_lines = "".join(f"<li>{l}</li>" for l in _exec_summary(bundle))
    snapshot = {
        "patient_id": bundle.get("patient_id"), "test_id": bundle.get("test_id"),
        "chain": bundle.get("chain"),
        "mode1_top": bundle["stage8"].route_B_concordance[:5],
        "mode2_flags": bundle.get("cell_of_origin_flags", []),
        "nilc_rescued": rescued,
        "trajectory_baseline_departure": bundle.get("trajectory_baseline", {}).get("patient_departure", {}),
    }

    # Input-scale guard + reference gauge. If the scored (abundant) cells fall off the gauge,
    # the patient beta is not on the atlas scale -> suppress per-cell A rather than print artifacts.
    _scale_ok, _off, _total = _input_scale_ok(bundle["stage4"])
    hmin = bundle["stage4"].get("h_min_by_class", {}) or {}

    # ---- TIER 1: class-level, RELIABLE (SOP §30 Tier 1, the production call) ----
    class_gauge = {cls: r["A"] for cls, r in bundle["stage4"].get("class_ascores", {}).items()
                   if r.get("A") is not None and r.get("status") == "OK"}
    gauge_html = a_score_gauge_svg(class_gauge if _scale_ok else None)
    if _scale_ok:
        _crows = ""
        for cls, r in sorted(bundle["stage4"].get("class_ascores", {}).items(),
                             key=lambda kv: -(kv[1].get("A") or -9)):
            a = r.get("A")
            if a is None or r.get("status") != "OK":
                continue
            floor = hmin.get(cls)
            below = (floor is not None and a < float(floor))
            tier = "below floor" if below else _tier(a)
            _crows += (f"<tr><td>{_esc(cls)}</td><td class='num'>{a:.3f}</td>"
                       f"<td class='num'>{_fmt_ci(r.get('A_ci_lo'), r.get('A_ci_hi'))}</td>"
                       f"<td class='num'>{floor:.3f}</td><td>{_esc(tier)}</td></tr>")
        class_reliable_html = (
            "<p class='meta'>SOP &sect;30 Tier 1 / &sect;44: the 8 architectural-class A-scores are the "
            "<b>production-grade, reliable</b> readout &#8212; this is the call. A = H(&#946;_mean)/H_min(class): "
            "1.0 is the healthy reference, the floor is the class H_min, the ceiling is 1/H_min.</p>"
            "<table><thead><tr><th>Architecture class</th><th>A-score</th><th>95% CI (MCMC)</th>"
            f"<th>floor (H_min)</th><th>tier</th></tr></thead><tbody>{_crows}</tbody></table>")
    else:
        class_reliable_html = (
            f"<div class='caveat'><b>Class A-scores withheld.</b> {_off} of {_total} classes scored below "
            "their own H_min floor &#8212; impossible for a present cell, so the &#946; is not on the atlas "
            "scale (a data problem, not a reading). Mode 2 (cell-of-origin presence) does not depend on this "
            "and is reported normally below.</div>")

    # ---- TIER 2: cell-level, INDICATIVE (SOP §30 Tier 2 — weakly separable in v0.1) ----
    departure_html = departure_ranking_svg(bundle, max_rows=24, reliable_only=False) if _scale_ok else ""
    if _scale_ok:
        n_cells = sum(1 for r in bundle["stage4"].get("celltype_ascores", {}).values()
                      if r.get("A") is not None)
        cell_indicative_html = (
            f"<div class='caveat' style='border-color:#c9a227;background:#fffdf3'>"
            "<b>Indicative tier &#8212; read alongside, not instead of, the class call above.</b> "
            "Per SOP &sect;30, within-class cell types are the Tier-2 <em>finer-grained decomposition</em> and "
            "are <b>weakly separable in IAMAtlas v0.1</b> &#8212; trust the class-level scores for the call. "
            "The <b>chart shows the deconvolver-resolved cells</b> (real per-cell signal); the <b>table below "
            "lists every present-class cell</b> with a confidence column, so nothing is hidden. Every cell type "
            "in a present class is A-scored (SOP &sect;44, independent of deconvolution fraction); the fraction "
            "is a confidence cue, never a gate. An <em>indicative</em> cell reads the bulk mixture at its "
            "one-vs-rest markers and can pin near the ceiling or below its floor &#8212; that is a low-confidence "
            "background read, not a finding.</div>"
            f"<p class='meta'>{n_cells} cell type(s) in present classes, ranked by departure from baseline.</p>"
            "<table><thead><tr><th>Cell type</th><th>Class</th><th>Fraction</th><th>A-score</th>"
            f"<th>95% CI</th><th>tier</th><th>confidence</th></tr></thead><tbody>{_ascore_rows(bundle)}</tbody></table>")
    else:
        cell_indicative_html = ""

    confirmation_html = _confirmation_section(bundle)
    trajectory_html = _trajectory_section(bundle)

    H = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CPG Report — {_esc(bundle.get('patient_id'))}</title>
<style>
:root{{--ink:#16202a;--soft:#516170;--line:#dfe3e8;--paper:#fbfcfd;--accent:#0e6b53;--red:#b2182b;--amber:#b45309;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",Helvetica,sans-serif;line-height:1.5}}
.wrap{{max-width:920px;margin:0 auto;padding:34px 26px 70px}}
h1{{font-size:26px;margin:0 0 2px}} h2{{font-size:18px;margin:30px 0 8px;border-bottom:2px solid var(--ink);padding-bottom:4px}}
h3{{font-size:14px;margin:16px 0 4px;color:var(--accent)}}
.meta{{font-size:13px;color:var(--soft);font-family:ui-monospace,Menlo,monospace}}
.ctx{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:12px 0;font-size:13px}}
.ctx b{{color:var(--soft);font-weight:600}}
ul.exec{{background:#fff;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:0 8px 8px 0;padding:12px 16px 12px 32px;margin:10px 0}}
ul.exec li{{margin:5px 0}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0 4px}}
th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--soft)}}
td.num{{font-family:ui-monospace,Menlo,monospace;text-align:right;white-space:nowrap}}
.muted{{color:var(--soft);font-style:italic}} .red{{color:var(--red)}}
.tag-bbb{{font-size:9px;background:#fbe9eb;color:var(--red);padding:1px 5px;border-radius:4px;font-weight:700;letter-spacing:.04em}}
.method{{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 18px 14px;margin:8px 0;font-size:13px}}
.method p{{margin:9px 0}} .method b{{color:var(--ink)}}
.caveat{{background:#fff7ed;border:1px solid #f3d9b5;border-radius:8px;padding:8px 16px;font-size:13px}}
.explain{{background:#f3f7f6;border-left:3px solid var(--accent);border-radius:0 6px 6px 0;padding:9px 14px;margin:8px 0;font-size:12.5px;color:#33414d}}
.explain b{{color:var(--ink)}}
.foot{{margin-top:30px;font-size:11px;color:var(--soft);font-family:ui-monospace,Menlo,monospace}}
details{{margin-top:10px}} summary{{cursor:pointer;font-size:12px;color:var(--soft)}}
pre{{background:#0f1720;color:#e7edf3;padding:12px;border-radius:8px;overflow:auto;font-size:11px}}
</style></head><body><div class="wrap">

<h1>Cellular Performance Gauge — Report</h1>
<div class="meta">Patient {_esc(bundle.get('patient_id'))} &nbsp;·&nbsp; Test {_esc(bundle.get('test_id'))}
 &nbsp;·&nbsp; {datetime.utcnow():%Y-%m-%d} &nbsp;·&nbsp; chain {_esc(bundle.get('chain'))}</div>

<div class="ctx"><b>Context (read alongside, never used in scoring):</b>
age {_esc(ctx.get('age'))} · sex {_esc(ctx.get('sex'))} · substrate {_esc(ctx.get('substrate'))}
· family history {_esc(ctx.get('family_history') or 'none provided')}</div>

<h2>Executive summary</h2>
<ul class="exec">{exec_lines}</ul>

<h2>How to read this report <span class="meta">(for the clinician and for a future AI)</span></h2>
<div class="method">
<p>CPG runs <b>two independent detection modes</b>. A finding in either is worth attention; they answer different questions.</p>
<p><b>Two deconvolvers, by design.</b> A blood sample is a mixture of dozens of cell types, so before anything else CPG
has to estimate how much of each is present from the bulk methylation signal — a step called <b>deconvolution</b>. CPG
runs this twice, with two methods built on different mathematics. The first (<b>Walther</b>, a constrained fit) is
deliberately conservative: it will not report a cell unless the evidence forces it, so faint traces are floored to zero.
That gives the <b>reliable composition</b> the rest of the report stands on. The second (<b>NILC</b>, adapted from the
variance-weighted technique used to clean cosmic-microwave-background maps) is deliberately sensitive: it surfaces the
faint shed signal the conservative fit suppresses. Where the two agree, confidence is high. Where they disagree, the
disagreement is itself information — it is how a cell quietly shedding into blood gets caught early. Two lenses, each
catching what the other can miss; neither alone would be as trustworthy as the pair.</p>
<p><b>Mode 1 — Architectural concordance.</b> Each cell gets an <b>A-score</b>, a derived measure of how far its
methylation architecture has walked from its healthy floor (A ≈ 1.0 by derivation; no cohort, no population). The
pattern of those departures across cells is compared to each disease's signature by <b>scale-invariant cosine</b>
— the angle between the patient's departure vector and the signature, so absolute magnitudes never need converting and
nothing standardizes the patient against a population. This is the workhorse for field-effect and systemic disease,
whose signal sits in the <b>abundant immune cells we can score cleanly</b>. It reports <b>resemblance, not a
probability</b>: v1 can recognize that a pattern looks like a template learned in validation cohorts; the calibrated
magnitude that would turn resemblance into "this stage of this disease" only accrues from real patients over years, in
our own derived A-units. Direction agreement is the coarse gate; cosine shape is the fine discriminator (it is what
separates aging from Alzheimer's — same direction, different proportions).</p>
<p><b>Mode 2 — Cell-of-origin presence.</b> A different test: a cell <b>circulating in blood that should not be
there</b>. The one presence that is alarming by itself is a <b>blood-brain-barrier cell</b> (cortical neuron, glia,
oligodendrocyte) — it crossed a physical barrier to be in blood, so quantity and A-score are irrelevant to the call;
refer. Every other cell that can shed into blood — epithelial/secretory (breast, prostate, colon), cycling — does so at
a normal baseline. <b>Detecting them is not abnormal</b>; we detect and, when abundant enough, score them routinely. So
their presence is normal composition (see the composition table), and the A-score is the discriminator only when the
cell is abundant enough to score. A cell at ~1% is ~99% blood, so its architecture cannot be scored from the bulk signal
— we detect such a cell's presence but do not assign it an A-score. The sensitive front end is the unconstrained NILC
deconvolver, which surfaces faint shed signal the constrained fit floors to zero.</p>
<p><b>Confidence intervals</b> on every A-score are derived from the atlas's own MCMC posteriors (the 8 per-class
per-CpG brightness tables), not from a cohort. <b>Trajectory:</b> this report stores the patient's derived departure
vector as a baseline; a second or third test compares drift toward a signature's angle, which is a far stronger signal
than any single snapshot.</p>
</div>

<h2>Mode 1 — Architectural concordance</h2>
<table><thead><tr><th>Disease</th><th>Phase</th><th>Cosine (shape)</th><th>Direction</th><th>Shared cells</th><th>Resemblance</th></tr></thead>
<tbody>{_mode1_rows(bundle)}</tbody></table>

<h2>Mode 2 — Cell-of-origin presence</h2>
<table><thead><tr><th>Class</th><th>Observed</th><th>Walther / NILC-raw</th><th>Severity</th><th>Interpretation</th></tr></thead>
<tbody>{_mode2_rows(bundle)}</tbody></table>
<p class="meta">NILC-raw rescued (surfaced where Walther floored): {rescued_txt}</p>

<h2>Cell-level architectural departure <span class="meta">(the A-score readout — every cell found in blood)</span></h2>
{trajectory_html}
{departure_html}
{cell_indicative_html}
{confirmation_html}

<h2>Honest limits</h2>
<div class="caveat">
A cell at ~1% of blood is ~99% diluted, so its architecture cannot be scored from the bulk signal — we detect such a
cell's presence (Mode 2) but do not assign it an A-score; only cells abundant enough to read their own loci are scored.
Healthy cells land at A&#8776;1.0 only when the patient beta is on the atlas scale. Mode 2 flags blood-brain-barrier cells only; epithelial, cycling and other
shed cells in blood are normal and are not flagged. None of the resemblance magnitude bands are calibrated yet — that is
what real patient data builds, in our own derived units. This report flags and refers; it does not diagnose.
</div>

<details><summary>Machine-readable snapshot (for a future AI)</summary>
<pre>{_esc(json.dumps(snapshot, indent=2, default=str))}</pre></details>

<div class="foot">CPG v1 lean · two-mode detection · scale-invariant cosine, no cohort · MCMC-derived CI · flag-and-refer, not diagnostic</div>
</div></body></html>"""

    if out_path:
        with open(out_path, "w") as f:
            f.write(H)
    return H


if __name__ == "__main__":
    print("cpg_report_builder.py — build_report(bundle, out_path) -> HTML.")
