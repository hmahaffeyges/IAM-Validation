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
from pathlib import Path
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


_REF_GAUGE_CACHE = {}
def _reference_gauge_html(include_star=True):
    """The FIXED A-score reference gauge -- the calibration scale the doctor and patient read
    every cell against (SUPPRESSED / NORMAL 0.95-1.04 / ELEVATED / 1.07 Warburg / 1.10 BREACH).
    Carries NO patient data; rendered from tier_breakpoints.json via cpg_gauge so it can never
    drift from the canonical tier scheme. Optionally stacks the star gauge (same ruler)."""
    import os, base64, tempfile
    key = "ref_star" if include_star else "ref"
    if key in _REF_GAUGE_CACHE:
        return _REF_GAUGE_CACHE[key]
    try:
        import cpg_gauge
        root = Path(os.environ.get("CPG_ROOT") or os.environ.get("CPG_ENGINE_ROOT") or Path(__file__).resolve().parent)
        tb = str(root / "Runtime Matrices" / "Tier_breakpoints" / "tier_breakpoints.json")
        def _png_b64(render):
            tmp = tempfile.mktemp(suffix=".png"); render(tmp)
            b = base64.b64encode(open(tmp, "rb").read()).decode(); os.unlink(tmp); return b
        # Fixed reference gauge: embed the canonical approved PNG as-is (it carries no patient
        # data and must not drift). Fall back to a fresh render only if the asset is missing.
        _asset = root / "assets" / "A1_reference_gauge.png"
        if _asset.exists():
            ref = base64.b64encode(_asset.read_bytes()).decode()
        else:
            ref = _png_b64(lambda p: cpg_gauge.render_reference_gauge(tb, p))
        html = ('<p class="meta">Fixed reference scale &#8212; read each cell\'s A-score below against '
                'this ruler. It carries no patient data.</p>'
                '<img class="gauge" alt="A-score reference gauge - the calibration scale" '
                f'src="data:image/png;base64,{ref}"/>')
        if include_star:
            try:
                star = _png_b64(lambda p: cpg_gauge.render_star_gauge(cpg_gauge.FIGC_STARS, tb, p))
                html += ('<details><summary>Same ruler, a star (AstroGenetics companion)</summary>'
                         '<img class="gauge" alt="star gauge - same ruler as the cell" '
                         f'src="data:image/png;base64,{star}"/></details>')
            except Exception:
                pass
        _REF_GAUGE_CACHE[key] = html
        return html
    except Exception as e:
        return f'<div class="caveat">Reference gauge unavailable ({_esc(str(e))}).</div>'


def departure_ranking_svg(bundle, width=900, max_rows=15, reliable_only=True):
    """Diverging departure ranking. Per SOP §30, the cell tier is indicative and weakly
    separable in v0.1: cells the deconvolver does not resolve read the bulk mixture at their
    one-vs-rest markers and pin spuriously. So the CHART shows the reliable (deconvolver-
    resolved) cells -- the real per-cell fingerprint -- ranked by distance from the 1.0
    baseline. The full per-cell table (all present-class cells, with the confidence column)
    is shown below the chart so nothing is hidden. Direction is the diagnostic part: a bar
    points RIGHT when elevated, LEFT when suppressed."""
    s4 = bundle["stage4"]["celltype_ascores"]
    # vKISS cell set: every cell scored above its class floor (Walther alone, no AND-gate).
    cells = [(ct, r) for ct, r in s4.items()
             if r.get("A") is not None and _is_cell(ct)
             and not r.get("below_floor")]
    if not cells:
        return "<p class='meta'>No cells scored above the presence floor on this sample.</p>"
    cells.sort(key=lambda kv: -abs(kv[1]["A"] - 1.0))
    total = len(cells)
    if max_rows:
        cells = cells[:max_rows]
    n = len(cells)
    _title = (f"Cellular departure ranking &#8212; top {n} of {total} present cells"
              if total > n else f"Cellular departure ranking &#8212; {n} present cell(s)")
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
             '&#183; whiskers = 95% CI</text>')
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


# --- Disease surfacing gate -------------------------------------------------
# A disease is SURFACED and NAMED (Mode 1 table, exec summary, second-chain header,
# trajectory rotation) only when it is disease-SPECIFIC and its shape resemblance
# (cosine) reaches the concern threshold. Below the threshold, or non-specific, the
# match is RETAINED (collapsed section + machine-readable snapshot) but never surfaced,
# so a clean report does not headline a scary disease name on a generic/weak pattern.
# The number shown is shape resemblance (cosine x100) -- NOT a probability of disease.
CONCERN_COSINE = 0.60

def _match_pct(m):
    try:
        return max(0, round(float(m.get("cosine", 0.0)) * 100))
    except Exception:
        return 0

def _is_concern(m):
    return (m is not None
            and m.get("specificity", "SPECIFIC") == "SPECIFIC"
            and m.get("resemblance") != "INSUFFICIENT_SIGNAL"
            and float(m.get("cosine", 0.0)) >= CONCERN_COSINE)

def _flagged_match(bundle, disease_id):
    try:
        for m in (bundle["stage8"].route_B_concordance or []):
            if m.get("disease") == disease_id:
                return m
    except Exception:
        pass
    return None


def _input_scale_ok(s4, substrate=None):
    """The real floor is per-class H_min, not a fixed gauge number. A class scored BELOW
    its own H_min floor is impossible for a cell still holding its identity, so for WHOLE BLOOD
    that is the signature of a data/scale problem. BUT a single phantom class must NOT condemn
    the whole sample: genuine un-normalized beta drags the MAJORITY of classes off-scale, not
    one. So the whole-blood guard fails only when MORE THAN ONE assessable class is below floor.

    cfDNA is different: run-everything intentionally scores absent tissue classes, and most of
    them are legitimately below floor (that tissue is not shed into plasma) — that is EXPECTED,
    not off-scale. So for cfDNA the scale anchor is the immune class, which is always present in
    blood/plasma: if immune is on-scale, the beta is on the atlas scale. Returns (ok, n_below, n)."""
    hmin = s4.get("h_min_by_class", {}) or {}
    cls_a = s4.get("class_ascores", {}) or {}
    below = [c for c, r in cls_a.items()
             if r.get("A") is not None and hmin.get(c) is not None and r["A"] < float(hmin[c])]
    n = sum(1 for r in cls_a.values() if r.get("A") is not None)
    cfdna = str(substrate or "").lower() in ("cfdna", "cf_dna", "plasma", "ctdna", "ct_dna")
    if cfdna:
        imm = cls_a.get("immune", {}) or {}
        a, floor = imm.get("A"), hmin.get("immune")
        ok = (a is None) or (floor is None) or (a >= float(floor))
        return ok, len(below), n
    return (len(below) <= 1), len(below), n


def _esc(x):
    return _html.escape(str(x)) if x is not None else ""


def _fmt_ci(lo, hi):
    if lo is None or hi is None:
        return "—"
    return f"[{lo:.3f}, {hi:.3f}]"


def _exec_summary(bundle):
    flags = bundle.get("cell_of_origin_flags", [])
    ok, off, total = _input_scale_ok(bundle["stage4"], (bundle.get("context") or {}).get("substrate"))
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
            if spec == "NON_SPECIFIC_GENERIC":
                lines.append("Architectural mode (Mode 1): <b>non-specific systemic pattern</b> "
                             "&#8212; the signal sits on the generic stress axis (myeloid/progenitor "
                             "elevation with lymphoid suppression &#8212; the neutrophil-to-lymphocyte "
                             "shift seen in infection, inflammation, stress and many conditions), not a "
                             "fingerprint specific to any one disease. Not flagged as a named malignancy.")
            elif _is_concern(top):
                lines.append(f"Architectural mode (Mode 1): your per-cell <b>A-scores</b> are the measurement &#8212; read them "
                             f"against the reference gauge below (H_min floor &#183; ~1.00 mid healthy band &#183; 1.10 breach). "
                             f"The <i>shape</i> of your departures across {top.get('n_signal',0)} cells most resembles the "
                             f"<b>{_esc(top['disease'])}</b> pattern template ({_match_pct(top)}% shape match). This is the shape of a "
                             f"pattern, <b>not a diagnosis, not a probability, and not a stage</b> &#8212; many conditions and benign states "
                             f"share pattern shapes. The confirmation chain below is what tests whether it is a real departure.")
            else:
                lines.append(f"Architectural mode (Mode 1): no disease-specific pattern reached the "
                             f"{int(CONCERN_COSINE*100)}% concern threshold (closest specific {_match_pct(top)}%). "
                             f"Below-threshold resemblances are retained for the clinician, not surfaced.")
        else:
            lines.append("Architectural mode (Mode 1): no disease pattern carries enough signal-bearing "
                         "shared cells to register a meaningful resemblance.")
    if flags:
        cls = ", ".join(_esc(f["class"]) for f in flags)
        lines.append(f"Cell-of-origin mode (Mode 2): <b class='red'>blood-brain-barrier cells circulating</b> "
                     f"(terminal class, {cls}) — barrier breach, refer for specialist evaluation.")
    else:
        lines.append("Cell-of-origin mode (Mode 2): no barrier-restricted cells circulating, and no tissue "
                     "cell present above its expected level. (Tissue cells at expected levels are normal; an amount "
                     "well above expected is watched as a possible shedding signal and scored when abundant enough.)")

    # Mode 3 — systemic stress / inflammatory wellness read (never a disease call, never alarm)
    ss = bundle.get("systemic_stress") or {}
    lvl = ss.get("level", "NONE")
    if lvl in ("NOTABLE", "MILD"):
        adj = "a clear" if lvl == "NOTABLE" else "a mild"
        lines.append(
            f"Wellness signal (Mode 3): <b style='color:#b7791f'>{adj} systemic stress / inflammatory pattern</b> "
            f"&#8212; myeloid and progenitor activity running high with lymphoid suppression (the methylation analog "
            f"of the neutrophil-to-lymphocyte shift), across {ss.get('n_axis_cells', 0)} cells. This is "
            f"<b>not a disease finding</b> and names no disease &#8212; it is a non-specific, <i>actionable</i> "
            f"wellness signal. It is the kind of thing to act on now: lifestyle, weight, diet, and trajectory "
            f"monitoring, weighed more heavily with a family history of cancer. The value of seeing it early is "
            f"the chance to change course &#8212; re-test to follow the trajectory.")
    else:
        lines.append("Wellness signal (Mode 3): no coherent systemic stress / inflammatory pattern "
                     "in the immune compartment.")
    return lines


def _mode1_rows(bundle):
    # Surface a disease ONLY when it is disease-SPECIFIC and its shape resemblance (cosine)
    # reaches the concern threshold. Weak / below-threshold / non-specific matches are NOT
    # surfaced (they would only alarm) but are RETAINED in a collapsed section below and in
    # the machine-readable snapshot. The number is shape resemblance (cosine x100), not a
    # probability of disease.
    conc = bundle["stage8"].route_B_concordance or []
    surfaced = [m for m in conc if _is_concern(m)]
    retained = [m for m in conc
                if m.get("resemblance") != "INSUFFICIENT_SIGNAL" and not _is_concern(m)]

    def _row(m, status):
        spec = m.get("specificity", "SPECIFIC")
        tag = "" if spec == "SPECIFIC" else f" <span class='meta'>({spec.replace('_',' ').lower()})</span>"
        return (f"<tr><td>{_esc(m['disease'])}{tag}</td><td>{_esc(m['phase'])}</td>"
                f"<td class='num'>{_match_pct(m)}%</td>"
                f"<td class='num'>{m['direction_agreement']:.2f}</td>"
                f"<td class='num'>{m.get('n_signal', 0)}</td>"
                f"<td>{status}</td></tr>")

    rows = "".join(_row(m, "shape match \u2014 not a diagnosis") for m in surfaced)
    if not rows:
        rows = (f"<tr><td colspan='6' class='muted'>No disease-specific pattern reached the "
                f"{int(CONCERN_COSINE*100)}% concern threshold. Below-threshold and non-specific "
                f"resemblances are retained below (and in the machine-readable snapshot), not "
                f"surfaced, to avoid false alarm.</td></tr>")
    if retained:
        retained.sort(key=lambda m: -float(m.get("cosine", 0.0)))
        rrows = "".join(_row(m, "below concern") for m in retained)
        rows += (f"<tr><td colspan='6' style='padding:2px 0'>"
                 f"<details><summary class='meta'>Screened, below the {int(CONCERN_COSINE*100)}% "
                 f"concern threshold or non-specific ({len(retained)}) &#8212; retained, not surfaced"
                 f"</summary><table><tbody>{rrows}</tbody></table></details></td></tr>")
    return rows


def _mode2_rows(bundle):
    flags = bundle.get("cell_of_origin_flags", [])
    rows = ""
    for f in flags:
        rows += (f"<tr><td>{_esc(f['class'])}</td>"
                 f"<td class='num'>{f['observed_fraction']*100:.2f}%</td>"
                 f"<td class='num'>{f['fraction_walther']*100:.2f}%</td>"
                 f"<td><b class='red'>REVIEW — BBB</b></td>"
                 f"<td>{_esc(f['interpretation'])}</td></tr>")
    return rows or ("<tr><td colspan='5' class='muted'>No barrier-restricted cells circulating, and no tissue "
                    "cell present above its expected level. Tissue cells at expected levels are normal; CPG watches for "
                    "any cell elevated well above expectation as a possible shedding signal, and scores its architecture "
                    "when it is abundant enough to read — see composition.</td></tr>")


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
        # vKISS noise fix: a wide A-CI = poorly-constrained (low-representation) reference
        # (Microglia/Kupffer/thin aliases ~0.07 posterior sd vs ~0.004 for well-covered cells).
        # Flag it so a thin-reference read is never mistaken for a confident finding.
        _lo, _hi = r.get("A_ci_lo"), r.get("A_ci_hi")
        if _lo is not None and _hi is not None and (_hi - _lo) > 0.05:
            conf = "indicative \u00b7 thin reference"
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
    an = s5["literature_anchor"]; rd = s5["residual_map"]
    tr = s5["trigger"]; ctx = bundle.get("context", {})

    _fm = _flagged_match(bundle, tr.get("flagged_disease"))
    _flag_label = (f"flagged {_esc(tr['flagged_disease'])}" if _is_concern(_fm)
                   else "screened a pattern for confirmation")
    verdict_col = "var(--red)" if _is_concern(_fm) else "var(--accent)"

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

    # RUN-everything residual sweep: every available map, independent of the per-cell rank
    sw = s5.get("residual_sweep") or {}
    sw_rows = sw.get("results") or {}
    if sw_rows:
        _label = {"breast_cancer": "breast", "alzheimers_disease": "Alzheimer's",
                  "immune_universal_alarm": "immune cross-disease alarm"}
        body = []
        for d, r in sw_rows.items():
            nm = _label.get(d, d)
            if r.get("status") == "OK":
                if r.get("fires") and r.get("rho", 0) > 0:
                    det = "<b style='color:var(--red)'>detected (consistent direction)</b>"
                elif r.get("fires"):
                    det = "opposite to pattern (not a detection)"
                else:
                    det = "not distinguishable from null"
                body.append(f"<tr><td>{_esc(nm)}</td><td class='num'>{r['rho']:+.3f}</td>"
                            f"<td class='num'>[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]</td>"
                            f"<td class='num'>{r['cpgs_compared']}</td><td>{det}</td></tr>")
            else:
                body.append(f"<tr><td>{_esc(nm)}</td><td colspan='3' class='meta'>"
                            f"{_esc(r.get('note', r.get('status')))}</td><td>&mdash;</td></tr>")
        sweep_html = (f"<table><thead><tr><th>Map</th><th>rho</th><th>95% CI</th>"
                      f"<th>CpGs</th><th>Result</th></tr></thead><tbody>{''.join(body)}</tbody></table>")
    elif sw.get("status") == "skipped_substrate":
        sweep_html = f"<p class='muted'>Residual sweep skipped: {_esc(sw.get('note'))}</p>"
    else:
        sweep_html = "<p class='muted'>Residual sweep: no maps available for this substrate.</p>"

    # Stage 4.5 AD directional read (composition-independent; AD's validated detector,
    # surfaced here because AD is deliberately NOT in the matched-filter sweep above).
    adx = s5.get("ad_directional")
    if adx and adx.get("composite") is not None:
        _comp = float(adx["composite"]); _adflag = bool(adx.get("flags_ad_direction"))
        _adcol = "var(--red)" if _adflag else "var(--accent)"
        _adstate = ("flags an AD-direction immune pattern" if _adflag
                    else "no AD-direction flag &mdash; graded read only")
        ad_html = (
            "<h3>AD directional read <span class=\"meta\">(Stage 4.5 &mdash; composition-independent, sealed VAL-051 Rule A)</span></h3>"
            "<p class=\"explain\"><b>What this is.</b> Architecturally, AD is <b>suppression toward the H_min floor</b> &mdash; "
            "advanced aging of informational fidelity, the cell&#39;s methylation drifting down toward its entropy floor (the AIBL "
            "per-cell-type fan-out is uniformly negative: 20 significant suppressed cells, zero elevated). The reason we read it with a "
            "sealed 7-CpG <b>directional panel</b> rather than the pooled A-score is purely statistical: at the individual-CpG level a "
            "minority of sites move the other way, so they partly cancel in the <i>pooled</i> number. The panel z-scores 7 sealed CpGs "
            "against a frozen per-CpG reference and multiplies by each CpG&#39;s frozen direction, so cell composition cannot leak in and "
            "the suppression signal is not washed out. The composite below is a directional-panel score, <b>not an A-score</b>.</p>"
            f"<div style=\"background:#fff;border:1px solid var(--line);border-left:4px solid {_adcol};"
            f"border-radius:0 8px 8px 0;padding:8px 14px;margin:8px 0\">Directional-panel score (not an A-score) = "
            f"<b style=\"color:{_adcol}\">{_comp:+.3f}</b> &middot; {_esc(adx.get('lean',''))} &middot; {_adstate}."
            f"<br><span class='meta'>Positive = toward the AD-suppression direction; negative = away from it (anti-AD / healthy lean). "
            f"Flags only past |0.40|.</span></div>"
            "<p class=\"meta\">Reference is AIBL-trained: it discriminates within AIBL but does not transfer to other-platform "
            "cohorts. Flags only on a clear move (|composite| &gt; 0.40), so a non-transferring cohort is a miss, never a false "
            "alarm. The single-patient AD signal is diffuse (AUC ~0.67) &mdash; read this as a lean, not a call.</p>")
    else:
        ad_html = ""

    return f"""
<h2>Confirmation — second chain <span class="meta">(ran because Stage 8 {_flag_label})</span></h2>
<div class="method">
<p>The second chain is independent and runs on a specific Route B flag OR any residual-sweep hit
({_esc(tr['gate_policy'])}).
It does not change anything above; it confirms whether the flag reflects a real departure in the
disease's own direction. Read alongside context: age {_esc(ctx.get('age'))} &middot; sex {_esc(ctx.get('sex'))} &middot; substrate {_esc(ctx.get('substrate'))}.</p>
<div style="background:#fff;border:1px solid var(--line);border-left:4px solid {verdict_col};border-radius:0 8px 8px 0;padding:10px 16px;margin:10px 0">
<b style="color:{verdict_col}">Verdict:</b> {_esc(s5['overall_verdict'])}</div>
{ad_html}
<h3>A &middot; Residual-map matched filter <span class="meta">(the detection instrument — SOP 8.2)</span></h3>
<p class="explain"><b>What this is.</b> For the flagged disease we hold a sealed <b>residual map</b> — the per-CpG
direction the disease moves methylation, fixed in advance from validated case/control cohorts. We measure the
correlation between the patient's per-CpG departure (from the derived healthy atlas baseline) and that sealed
direction, with a 95% confidence interval. When the interval is clear of zero, the patient's methylation is moving
the disease's way — elevation or suppression both count. This is the same matched-filter statistic used to pull
faint signals out of noise in gravitational-wave and cosmic-microwave-background searches.</p>
{resid_html}

<h3>B &middot; Literature-anchor evidence <span class="meta">(published context, not the reference)</span></h3>
{anchor_html}

<h3>C &middot; RUN-everything residual sweep <span class="meta">(every available map, independent of the per-cell rank)</span></h3>
<p class="explain"><b>Why this runs on every sample.</b> Some diseases announce themselves in a per-CpG
distributional pattern rather than a per-cell shift &mdash; breast pre-diagnosis is one (its signal is a quiet
homogenization of the secretory compartment). The per-cell matcher leaves those quiet by design, so the matched
filter sweeps every sealed map on its own rather than waiting for a per-cell flag. A detection is a positive
correlation with the disease's signed residual; an anti-correlation is not a detection. Healthy controls sit at
or below zero on these maps.</p>
{sweep_html}
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

    # rotation toward the flagged signature -- name the disease only when it is of concern
    # (SPECIFIC and >= the concern threshold). Otherwise describe the rotation neutrally so a
    # clean report does not headline a scary disease name on a generic/below-threshold pattern.
    rot = tj.get("rotation"); rot_html = ""
    if rot:
        rcol = ("var(--red)" if rot["trend"] == "rotating toward the signature"
                else "var(--accent)" if rot["trend"] == "rotating away from the signature" else "var(--soft)")
        _rfm = _flagged_match(bundle, rot.get("disease"))
        _sig = (f"the {_esc(rot['disease'])} signature" if _is_concern(_rfm)
                else "the screened signature (held below the concern threshold)")
        rot_html = (f"<p><b>Pattern rotation toward {_sig}:</b> cosine "
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


_CFDNA_SUBSTRATES = {"cfdna", "cf_dna", "plasma", "ctdna", "ct_dna", "serum"}


def _is_cfdna(bundle):
    sub = str((bundle.get("context", {}) or {}).get("substrate", "") or "").lower()
    return sub in _CFDNA_SUBSTRATES


def _tissue_of_origin_section(bundle):
    """cfDNA ONLY. Shed tissue DNA circulates in plasma, so the non-immune tissue tiles carry
    a tissue-of-origin signal that whole-blood Mode 2 (BBB-only) never surfaces. Rank the
    tissue tiles by departure from the derived reference (A=1.0) in EITHER direction:
    homogenization (A<1.0, the HCC hepatocyte signature) and elevation (A>1.0) both carry
    information. Floor-gated: a tile below its class H_min is an absent cell reading background,
    not a tissue read, and is excluded (the cortical-neuron artifact). Returns '' on whole blood."""
    if not _is_cfdna(bundle):
        return ""
    s4 = bundle["stage4"]
    cta = s4.get("celltype_ascores", {}) or {}
    hmin = s4.get("h_min_by_class", {}) or {}
    ranked = []
    for ct, r in cta.items():
        a = r.get("A")
        if a is None or not _is_cell(ct):
            continue
        cls = r.get("class")
        if cls == "immune":          # immune is the ~70% systemic plasma background, not tissue-of-origin
            continue
        if r.get("below_floor"):     # absent cell reading background -- not a tissue read
            continue
        floor = hmin.get(cls)
        if floor is not None and a < float(floor):
            continue
        ranked.append((abs(a - 1.0), ct, cls, a))
    ranked.sort(reverse=True)
    body = ""
    for dep, ct, cls, a in ranked[:12]:
        direction = "homogenization (below reference)" if a < 1.0 else "elevation (above reference)"
        bbb = " <span class='tag-bbb'>BBB</span>" if ct in BBB_PROTECTED else ""
        body += (f"<tr><td>{_esc(ct)}{bbb}</td><td>{_esc(cls)}</td>"
                 f"<td class='num'>{a:.3f}</td><td class='num'>{dep:.3f}</td>"
                 f"<td>{_esc(direction)}</td></tr>")
    if not body:
        body = "<tr><td colspan='5' class='muted'>No tissue tile departs from the derived reference.</td></tr>"
    return ("<h2>Tissue-of-origin <span class=\"meta\">(plasma cfDNA &#8212; shed tissue DNA)</span></h2>"
            "<p class='meta'>In plasma, shed tissue DNA carries the tissue-of-origin signal &#8212; the opposite of "
            "whole blood, where these same cells would be absent. Non-immune tiles are ranked by <b>departure from "
            "the derived reference</b> (A=1.0) in either direction. Homogenization (A&lt;1.0) and elevation "
            "(A&gt;1.0) both carry information; a single timepoint is a <b>flag</b>, the <b>trajectory</b> across "
            "draws is the call. Floor-gated: absent-cell background reads are excluded.</p>"
            "<table><thead><tr><th>Tissue tile</th><th>Class</th><th>A-score</th><th>|departure|</th>"
            f"<th>direction</th></tr></thead><tbody>{body}</tbody></table>")


_CARD_DIR = Path(__file__).resolve().parent / "Disease Cards : Residual Maps"
_DISEASE_CARD_FILES = {   # substring of the flagged route-B disease_id -> card json (relative to _CARD_DIR)
    "breast": "Breast_EPIC/breast_epic_card_json/breast-epic_card_v3_1.json",
    "alzheim": "AD_EPIC/AD_immune_card_json/ad-immune_card_v3_1.json",
    "_ad_": "AD_EPIC/AD_immune_card_json/ad-immune_card_v3_1.json",
}
_UNIVERSAL_CARD = "Immune_Atlas/immune-atlas_card_v2_0.json"


def _load_card(relpath):
    try:
        with open(_CARD_DIR / relpath) as f:
            return json.load(f)
    except Exception:
        return None


def _match_cards(bundle):
    """The immune-atlas card is the universal gateway and ALWAYS renders. Disease-specific
    cards (breast, AD) are loaded and rendered when stage-8 route B flags their disease.
    Returns [(card_id, card_dict, match_or_None)] — match is the route-B row that fired it."""
    out = []
    uni = _load_card(_UNIVERSAL_CARD)
    if uni:
        out.append((uni.get("card_id", "immune-atlas"), uni, None))
    try:
        flagged = bundle["stage8"].route_B_concordance or []
    except Exception:
        flagged = []
    seen = set()
    for m in flagged:
        did = str(m.get("disease", "")).lower()
        for key, rel in _DISEASE_CARD_FILES.items():
            k = key.strip("_")
            if k and k in did and rel not in seen:
                card = _load_card(rel)
                if card:
                    out.append((card.get("card_id", rel), card, m))
                    seen.add(rel)
    return out


def _disease_card_section(bundle):
    """Clinical interpretation layer: render each matched card's claim + honest limits.
    Cards are the interpretation over the architectural A-scores; they do not diagnose."""
    cards = _match_cards(bundle)
    if not cards:
        return ""
    blocks = ""
    for cid, card, match in cards:
        claim = ((card.get("clinical_claim") or {}).get("summary") or "")[:650]
        disease = card.get("disease") or card.get("card_type") or ""
        ver = card.get("card_version", "")
        limits = card.get("honest_limitations")
        if isinstance(limits, dict):
            limits = "; ".join(str(v) for v in list(limits.values())[:3])
        elif isinstance(limits, list):
            limits = "; ".join(str(v) for v in limits[:3])
        limits = (str(limits or ""))[:500]
        basis = ("universal baseline &#8212; always reported" if match is None else
                 f"flagged via route B: {_esc(str(match.get('disease')))} &#8212; "
                 f"{_esc(str(match.get('resemblance')))} (cosine {_esc(str(match.get('cosine')))})")
        ver_disp = str(ver) if str(ver).lower().startswith("v") else "v" + str(ver)
        blocks += (f"<div class='dcard'><h3>{_esc(str(cid))} "
                   f"<span class='meta'>{_esc(ver_disp)} &#183; {_esc(str(disease))}</span></h3>"
                   f"<p class='meta'>Reference basis: {basis}</p>"
                   f"<p>{_esc(claim)}</p>"
                   + (f"<p class='meta'><b>Card limits:</b> {_esc(limits)}</p>" if limits else "")
                   + "</div>")
    return ("<h2>Disease cards <span class=\"meta\">(clinical interpretation layer)</span></h2>"
            "<p class='meta'>The immune-atlas card is the universal baseline and always reports. "
            "Disease-specific cards render their clinical claim and limits when stage 8 flags their "
            "disease. Cards interpret the architectural A-scores; they flag and refer, they do not "
            "diagnose.</p>"
            "<style>.dcard{border:1px solid #e3e3e3;border-left:4px solid #0e6b53;border-radius:6px;"
            "padding:10px 14px;margin:8px 0;background:#fbfdfc}.dcard h3{margin:0 0 4px;font-size:14px}</style>"
            + blocks)


def _cell_census_table(bundle):
    """Every cell, scored individually — NO class average (a class score cancels the
    bidirectional departures that carry the signal). Present cells (above their class floor)
    are shown by departure magnitude in BOTH directions; suppression is signal as much as
    elevation. Present cells within the normal band beyond the first ~10 collapse so a clean
    report stays short. Below-floor reads (absent cells reading background) sit in their own
    collapsed, labelled section and are never fed to the disease matrix."""
    s4 = bundle["stage4"]
    cta = s4.get("celltype_ascores", {}) or {}
    LO, HI, MIN_SHOW = 0.95, 1.04, 10
    present, noise, disagreed = [], [], []   # vKISS: no AND-gate; disagreed stays empty
    for ct, r in cta.items():
        a = r.get("A")
        if a is None or not _is_cell(ct):
            continue
        row = (ct, r.get("class"), r.get("celltype_fraction"), a, r.get("A_ci_lo"), r.get("A_ci_hi"))
        if r.get("below_floor"):
            noise.append(row)
        else:
            present.append(row)   # vKISS: every above-floor cell is a present cell (Walther alone)
    present.sort(key=lambda x: -abs(x[3] - 1.0))
    departed = [p for p in present if p[3] < LO or p[3] > HI]
    normal = [p for p in present if LO <= p[3] <= HI]
    visible = departed[:]
    if len(visible) < MIN_SHOW:
        visible += normal[:MIN_SHOW - len(visible)]
    vis_set = {id(p) for p in visible}
    collapsed = [p for p in present if id(p) not in vis_set]

    def _row(p):
        ct, cls, frac, a, lo, hi = p
        d = a - 1.0
        direction = "elevated" if a > HI else "suppressed" if a < LO else "normal"
        ftxt = f"{frac*100:.1f}%" if frac is not None else "&#8212;"
        bbb = " <span class='tag-bbb'>BBB</span>" if ct in BBB_PROTECTED else ""
        return (f"<tr><td>{_esc(ct)}{bbb}</td><td>{_esc(cls)}</td><td class='num'>{ftxt}</td>"
                f"<td class='num'>{a:.3f}</td><td class='num'>{d:+.3f}</td>"
                f"<td>{_esc(_tier(a))}</td><td>{_esc(direction)}</td>"
                f"<td class='num'>{_fmt_ci(lo, hi)}</td></tr>")

    head = ("<tr><th>Cell type</th><th>Class</th><th>%</th><th>A-score</th>"
            "<th>departure</th><th>where on gauge</th><th>direction</th><th>95% CI</th></tr>")
    vis_rows = "".join(_row(p) for p in visible) or \
        "<tr><td colspan='8' class='muted'>No present cells scored.</td></tr>"
    intro_agree = ""   # vKISS: no AND-gate, every above-floor cell is presented
    out = (f"<p class='meta'>{intro_agree}Every present cell, scored on its own &#8212; no class average. "
           f"{len(departed)} of {len(present)} present cells depart from the normal band "
           f"(A&nbsp;0.95&#8211;1.04), in either direction; a suppressed cell carries signal as much as an "
           f"elevated one. Listed by departure magnitude; the percentage is that cell's share of the blood.</p>"
           f"<table><thead>{head}</thead><tbody>{vis_rows}</tbody></table>")
    if collapsed:
        out += (f"<details><summary>Show all {len(present)} present cells "
                f"(+{len(collapsed)} more, within the normal band)</summary>"
                f"<table><thead>{head}</thead><tbody>{''.join(_row(p) for p in collapsed)}</tbody></table></details>")
    # (deconvolver-disagreement exclusion removed in vKISS — no NILC, no AND-gate)
    if noise:
        noise.sort(key=lambda x: -abs(x[3] - 1.0))
        out += (f"<details><summary>Below-floor reads ({len(noise)}) &#8212; absent cells reading background, "
                f"NOT present-cell signal, excluded from the disease match</summary>"
                f"<p class='meta'>A tile below its class floor is an absent cell whose markers read background, "
                f"not that cell's own architecture. Shown for completeness; never fed to the matrix.</p>"
                f"<table><thead>{head}</thead><tbody>{''.join(_row(p) for p in noise)}</tbody></table></details>")
    return out


def _exec_render(renderer_path, inject, strip_substrings):
    """Run a sealed standalone renderer (builders/render_*.py) WITHOUT modifying it:
    strip its hardcoded file-I/O lines, inject the data dict it expects as `D`, and
    capture its `HTML` output variable. The renderer file on disk is never touched."""
    src = Path(renderer_path).read_text()
    for s in strip_substrings:
        src = src.replace(s, "")
    ns = dict(inject)
    exec(compile(src, str(renderer_path), "exec"), ns)
    return ns.get("HTML", "")


def _strawman_section(bundle):
    """Pattern-recognition straw man: this patient's per-cell architecture laid on the
    eight-class grid next to the disease rows it flagged, plus the full reference
    disease-signature wall (the crown jewel) collapsed beneath it.

    The patient is measured by physics (A = H(beta)/H_min, derived floor, no cohort);
    the wall supplies only the DIRECTION each disease moves each cell, learned from
    validation cohorts -- the demarcation line. Built from the EXISTING bundle (no chain
    re-run). The present-cell gate (above floor AND real deconvolver fraction) is what
    excludes the zero-fraction breach reads that inflate the class-level ranking, so the
    straw man is the honest per-cell view. Both walls are rendered by the sealed
    renderers (files untouched) and embedded in iframes so their styling stays isolated."""
    try:
        import os
        roots = [Path(__file__).resolve().parent,
                 Path(os.environ.get("CPG_ENGINE_ROOT", "/home/claude/work/FILES FOR AI/CPG_CMB_v5"))]
        def _find(*rel):
            for r in roots:
                for rp in rel:
                    p = r / rp
                    if p.exists():
                        return p
            return None
        cj_path  = _find("builders/strawman_data_v2.json", "Crown Jewel and Patient Strawman/strawman_data_v2.json")
        map_path = _find("Disease Matrix/DISEASE_MATRIX/iamatlas_115_to_matrix_v0_2_mapping.json")
        rp_wall  = _find("builders/render_patient_wall.py")
        rp_crown = _find("builders/render_strawman_v2.py")
        if not (cj_path and map_path and rp_wall):
            return ""  # straw man assets not found; omit silently
        CJ = json.loads(cj_path.read_text())
        mapping = json.loads(map_path.read_text())["mapping"]

        # ---- 1. patient cells from the bundle: PRESENT cells only, mapped, tiered ----
        # (same gate as build_patient_wall.py: above floor AND deconvolver fraction >= MIN)
        s4 = bundle.get("stage4") or {}
        ct = s4.get("celltype_ascores") or {}
        MIN_FRACTION = 0.001
        def _present(rec):
            if not (isinstance(rec, dict) and rec.get("A") is not None and not rec.get("below_floor")):
                return False
            frac = rec.get("celltype_fraction")
            return True if frac is None else float(frac) >= MIN_FRACTION
        bycol = {}
        for cell, rec in ct.items():
            if not _present(rec):
                continue
            col = mapping.get(cell)
            if not col:
                continue
            bycol.setdefault(col, []).append((float(rec["A"]), rec.get("A_ci_lo"), rec.get("A_ci_hi"), cell))
        patient_cells = {}
        for col, vals in bycol.items():
            A = sum(v[0] for v in vals) / len(vals); dep = A - 1.0
            cis = [(v[1], v[2]) for v in vals if v[1] is not None and v[2] is not None]
            ci = ([sum(c[0] for c in cis)/len(cis), sum(c[1] for c in cis)/len(cis)] if cis else None)
            if   A < 0.95: tier = "SUPPRESSED"
            elif A < 1.04: tier = "NORMAL"
            elif A < 1.07: tier = "ELEVATED"
            elif A < 1.10: tier = "SIGNIFICANTLY_ELEVATED"
            else:          tier = "BREACH"
            confident = bool(ci and (ci[0] >= 1.04 or ci[1] <= 0.95))
            patient_cells[col] = {"A": round(A, 3), "dep": round(dep, 3),
                                  "ci": [round(c, 3) for c in ci] if ci else None,
                                  "tier": tier, "confident": confident,
                                  "atlas": [v[3] for v in vals]}

        # ---- 2. flagged diseases from the EXISTING second chain (sweep + confirmed + AD direction) ----
        s5 = bundle.get("stage5") or {}
        trig = s5.get("trigger", {}) or {}
        _alias = {"breast_cancer": "breast_cancer", "alzheimers_disease": "alzheimers_disease",
                  "immune_universal_alarm": "immune"}
        flagged = []
        for d in (trig.get("residual_sweep_fired") or []):
            flagged.append({"disease": _alias.get(d, d), "via": "matched filter (residual sweep)"})
        if trig.get("flagged_confirmed") and trig.get("flagged_disease"):
            flagged.append({"disease": trig["flagged_disease"], "via": "per-cell matcher (confirmed)"})
        adx = s5.get("ad_directional")
        if adx and adx.get("flags_ad_direction"):
            flagged.append({"disease": "alzheimers_disease",
                            "via": "Stage 4.5 directional composite (VAL-051 Rule A)"})
        flagged_ids = {f["disease"] for f in flagged}
        cj_rows = [r for r in CJ["disease_rows"] if r["disease"] in flagged_ids]

        # ---- 3. assemble the data dict render_patient_wall.py expects as D ----
        ctx = bundle.get("context", {}) or {}
        stress = bundle.get("systemic_stress") or {"level": "NONE", "n_axis_cells": 0, "mean_magnitude": 0}
        D = {"patient_id": bundle.get("patient_id") or "", "substrate": ctx.get("substrate") or "",
             "age": ctx.get("age") or "", "sex": ctx.get("sex") or "",
             "patient_cells": patient_cells, "stress": stress,
             "flagged": flagged, "flagged_rows": cj_rows,
             "verdict": s5.get("overall_verdict", "") or "",
             "sec_cols": CJ["sec_cols"], "sections_used": CJ["sections_used"]}

        # ---- 4. render BOTH walls with the sealed renderers (their files are not modified) ----
        patient_wall = _exec_render(rp_wall, {"D": D}, [
            'D=json.load(open("/home/claude/patient_wall_data.json"))',
            'open("/home/claude/IAM_Patient_StrawMan.html","w").write(HTML)'])
        crown_wall = ""
        crown_v3 = _find("Crown Jewel and Patient Strawman/IAM_Disease_Wall_CROWN_JEWEL_v3.html",
                         "outputs/IAM_Disease_Wall_CROWN_JEWEL_v3.html",
                         "IAM_Disease_Wall_CROWN_JEWEL_v3.html")
        if crown_v3:
            crown_wall = crown_v3.read_text(encoding="utf-8")   # the sealed v1_13 wall, embedded as-is
        elif rp_crown.exists():
            crown_wall = _exec_render(rp_crown, {"D": CJ}, [
                'D=json.load(open("/home/claude/strawman_data_v2.json"))',
                'open("/home/claude/IAM_Disease_Wall_strawman_v2.html","w").write(HTML)'])

        # ---- 5. embed both, collapsed; iframe keeps the dark wall styling isolated ----
        def _frame(h, height):
            return (f'<iframe srcdoc="{_html.escape(h, quote=True)}" '
                    f'style="width:100%;height:{height}px;border:1px solid var(--line);'
                    f'border-radius:8px;background:#100c08"></iframe>')
        n_flag = len(flagged)
        out = ['<h2>Pattern recognition — straw man <span class="meta">(per-cell architecture vs the disease signature wall)</span></h2>']
        out.append('<div class="explain"><b>How this is read.</b> The patient\'s own per-cell A-score architecture is '
                   'laid on the same eight-class grid as the reference disease-signature wall, so the two can be set '
                   'side by side. The patient is measured by physics (A = H(&#946;)/H_min, a derived floor, no cohort); '
                   'the wall supplies only the <b>direction</b> each disease moves each cell, learned from validation '
                   'cohorts. Where the patient\'s red or blue lines up with a disease row\'s red or blue, the patient is '
                   'moving that pattern\'s way. A single cell is never the call &#8212; the shape across cells is. '
                   'Present cells only (above floor, real deconvolver fraction): the zero-fraction reads that inflate '
                   'the class-level ranking are excluded here.</div>')
        out.append(f'<details><summary>Your straw man &#8212; this patient\'s per-cell wall vs {n_flag} flagged pattern(s)</summary>{_frame(patient_wall, 560)}</details>')
        if crown_wall:
            out.append('<details><summary>Reference disease-signature wall (crown jewel) &#8212; the full pattern '
                       'catalog the patient is matched against</summary>' + _frame(crown_wall, 640) + '</details>')
        return "\n".join(out)
    except Exception as e:
        return f'<div class="caveat">Straw man pattern view unavailable: {_html.escape(str(e))}</div>'



def _plate_thumb_b64(path, width=520):
    """Downscale a sky-map plate to a small base64 PNG for the folded Stage 4.6 showcase."""
    try:
        from PIL import Image
        import io, base64
        im = Image.open(path).convert("RGB")
        if im.width > width:
            im = im.resize((width, int(im.height * width / im.width)))
        buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _cosmic_methylome_section(bundle, plate_path=None):
    """Stage 4.6 — the patient's Cosmic Methylome Background, rendered here.

    Projects this patient's per-class departure z = (beta - mu_class)/sd_class onto a HEALPix
    Mollweide sky. sd_class is the atlas per-CpG biological spread (it scales with locus lability),
    so z reads departure relative to how much a healthy person varies, not reference precision.
    Classes the blood does not contain read as a uniform offset and are labeled reference-only.
    """
    import os, base64, tempfile
    from pathlib import Path as _P
    patient_img = None; present = None
    beta = bundle.get("patient_beta")
    if beta is not None:
        try:
            import cpg_patient_cmb as _cmb
            root = _P(os.environ.get("CPG_ENGINE_ROOT") or os.environ.get("CPG_ROOT") or ".")
            csv = root / "IAM_Atlas" / "IAMAtlasREBUILD.csv"
            xz = root / "IAM_Atlas" / "IAMAtlasREBUILD.csv.xz"
            if not csv.exists() and xz.exists():
                import lzma, shutil
                with lzma.open(xz) as fi, open(csv, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            if csv.exists():
                atlas = _cmb.load_atlas_mean_sd(str(csv))
                cmb, _pix = _cmb.compute_patient_cmb(beta, atlas)
                tmp = _P(tempfile.gettempdir()) / f"cmb_{bundle.get('patient_id','patient')}.png"
                _cmb.render_patient_cmb(cmb, str(tmp), str(bundle.get("patient_id", "patient")))
                present = [c for c in _cmb.CLASSES if cmb[c]["assessable"]]
                patient_img = base64.b64encode(tmp.read_bytes()).decode()
        except Exception:
            patient_img = None
    ref_thumb = _plate_thumb_b64(plate_path) if plate_path else None
    blocks = []
    if patient_img:
        blocks.append("<img alt='Personal Cosmic Methylome' style='max-width:100%;border:1px solid "
                      f"var(--line);border-radius:8px;margin:8px 0' src='data:image/png;base64,{patient_img}'>")
        blocks.append("<p class='meta'>Assessable from this whole-blood draw: <b>"
                      f"{_esc(', '.join(present))}</b>. Panels for tissue not present in blood are greyed "
                      "and labeled reference-only &#8212; a uniform offset there is the absence of that tissue, "
                      "not a finding.</p>")
    elif ref_thumb:
        blocks.append("<img alt='Cosmic Methylome Background' style='max-width:100%;border:1px solid "
                      f"var(--line);border-radius:8px;margin:8px 0' src='data:image/png;base64,{ref_thumb}'>")
    return (
        "<details><summary><b>Cosmic Methylome Background</b> &#8212; this methylome on the celestial sphere "
        "(AstroGenetics companion)</summary>"
        "<p>CPG is the first to map and score the methylome with the visualization tools of cosmology. Each "
        "architecture class is projected onto a HEALPix NSIDE=128 Mollweide sky &#8212; the equal-area, full-sky "
        "convention used for the cosmic microwave background. The map below is this patient's personal sky: each "
        "panel shows the per-CpG departure z = (&beta;<sub>patient</sub> &#8722; &mu;<sub>class</sub>) / "
        "&sigma;<sub>class</sub>, where &sigma;<sub>class</sub> is the atlas per-CpG spread &#8212; how much a "
        "healthy person varies at that locus &#8212; so z reads departure relative to healthy biological "
        "variation, not reference precision.</p>"
        + "".join(blocks) +
        "<p class='meta'>The assessable panels are the detector; the matched-filter and hull adjudication above "
        "act on these same departures. Same MCMC-built IAMAtlas that sets every A-score's 95% CI.</p></details>")


def _how_cpg_works_section():
    """The AstroGenetics method explainer, folded. Reads the prose from the sealed HTML file so it
    can be edited independently of the builder."""
    import os, re as _re
    from pathlib import Path as _P
    for r in [_P(__file__).resolve().parent,
              _P(os.environ.get("CPG_ENGINE_ROOT", "") or "."),
              _P(os.environ.get("CPG_ROOT", "") or ".")]:
        p = r / "CPG_AstroGenetics_explainer_section.html"
        if p.exists():
            html = _re.sub(r"<!--.*?-->", "", p.read_text(encoding="utf-8"), flags=_re.S)
            html = _re.sub(r"<h2>.*?</h2>", "", html, count=1, flags=_re.S)  # summary supplies the title
            return ("<details class=\"howworks\"><summary><b>How CPG works</b> "
                    "&#8212; the AstroGenetics method</summary>" + html + "</details>")
    return ""


def build_report(bundle, out_path=None):
    ctx = bundle.get("context", {})
    rescued = bundle.get("nilc_rescued_classes", [])
    def _pct_frac(v):
        try:
            return f"{float(v)*100:.1f}%"
        except Exception:
            return str(v)
    rescued_txt = ", ".join(f"{_esc(r.get('class'))} ({_pct_frac(r.get('nilc_raw_fraction'))})"
                            for r in rescued if "class" in r) or "none"
    exec_lines = "".join(f"<li>{l}</li>" for l in _exec_summary(bundle))
    _s4 = bundle.get("stage4", {}) or {}
    _cells = []
    for _ct, _r in (_s4.get("celltype_ascores") or {}).items():
        _a = _r.get("A")
        if _a is None:
            continue
        _cells.append({"cell": _ct, "class": _r.get("class"),
                       "pct": round((_r.get("celltype_fraction") or 0) * 100, 2),
                       "A": round(_a, 4), "departure": round(_a - 1.0, 4),
                       "below_floor": bool(_r.get("below_floor"))})
    _cells.sort(key=lambda c: -abs(c["departure"]))
    snapshot = {
        "patient_id": bundle.get("patient_id"), "test_id": bundle.get("test_id"),
        "chain": bundle.get("chain"),
        "substrate": (bundle.get("context", {}) or {}).get("substrate"),
        "cells": _cells,
        "cards_matched": [cid for cid, _c, _m in _match_cards(bundle)],
        "mode1_top": bundle["stage8"].route_B_concordance[:5],
        "mode2_flags": bundle.get("cell_of_origin_flags", []),
        "nilc_rescued": rescued,
        "trajectory_baseline_departure": bundle.get("trajectory_baseline", {}).get("patient_departure", {}),
        "stage4_5_directional": bundle.get("stage4_5"),
        "ad_directional": ((bundle.get("stage5") or {}).get("ad_directional")
                           if isinstance(bundle.get("stage5"), dict) else None),
        "errors": bundle.get("errors") or bundle.get("warnings") or [],
    }

    # Input-scale guard + reference gauge. If the scored (abundant) cells fall off the gauge,
    # the patient beta is not on the atlas scale -> suppress per-cell A rather than print artifacts.
    _scale_ok, _off, _total = _input_scale_ok(bundle["stage4"], (bundle.get("context") or {}).get("substrate"))
    hmin = bundle["stage4"].get("h_min_by_class", {}) or {}

    # ---- TIER 1: class-level, RELIABLE (SOP §30 Tier 1, the production call) ----
    class_gauge = {cls: r["A"] for cls, r in bundle["stage4"].get("class_ascores", {}).items()
                   if r.get("A") is not None and r.get("status") == "OK"}
    # The gauge is the FIXED reference scale (no patient markers); each cell's position is read
    # against it in the per-cell census below. (Was: class-level markers, which we no longer show.)
    gauge_html = _reference_gauge_html(include_star=True)
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

    # ---- Per-cell census: every cell scored individually, no class average ----
    departure_html = departure_ranking_svg(bundle, max_rows=24) if _scale_ok else ""
    cell_census_html = _cell_census_table(bundle) if _scale_ok else ""

    confirmation_html = _confirmation_section(bundle)
    trajectory_html = _trajectory_section(bundle)
    tissue_of_origin_html = _tissue_of_origin_section(bundle)
    disease_card_html = _disease_card_section(bundle)
    strawman_html = _strawman_section(bundle)
    # Stage 4.6 — Cosmic Methylome Background (folded). Resolve the reference plate if present.
    from pathlib import Path as _P
    _plate = None
    for _c in [_P(__file__).parent / "AstroGenetics_and_NullSuite_assets/Mollweide & Brightness Comparison/Plates/SkyMaps of the Methylome/CPG_Plate_01_Cosmic_Methylome_Background.png",
               _P("Biological_Physics/atlas_vault/IAMAtlas_v0_1/plates/CPG_Plate_01_Cosmic_Methylome_Background.png")]:
        if _c.exists(): _plate=str(_c); break
    cosmic_methylome_html = _cosmic_methylome_section(bundle, _plate)
    how_cpg_works_html = _how_cpg_works_section()
    _shed_note = ("In plasma cfDNA, shed epithelial, cycling and other tissue cells ARE the tissue-of-origin "
                  "signal and are surfaced in the Tissue-of-origin section above; Mode 2 BBB flagging still applies."
                  if _is_cfdna(bundle) else
                  "Mode 2 flags blood-brain-barrier cells only; epithelial, cycling and other "
                  "shed cells in blood are normal and are not flagged.")

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
.gauge{{width:100%;max-width:900px;display:block;margin:6px auto 14px;border:1px solid var(--line);border-radius:8px;background:#fff}}
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

{how_cpg_works_html}

<h2>How to read this report <span class="meta">(for the clinician and for a future AI)</span></h2>
<div class="method">
<p>CPG asks <b>two detection questions</b> — the two modes below — from a single <b>whole-blood</b> draw, and adds a confirmation chain (derived global-departure adjudication, and trajectory across visits) whenever a pattern flags or a prior draw exists. A finding in either mode is worth attention; they answer different questions.</p>
<p><b>The deconvolver answers one question only: what cells are in the mix.</b> A blood sample is a mixture of dozens of cell types, so CPG first estimates how much of each is present from the bulk methylation signal (a step called <b>deconvolution</b>, the Walther constrained fit). This resolves the composition so the per-cell reads below are real cells, not bulk-mixture background. <b>It never fires the detection call</b> — the call is the A-score read straight from the methylation, and the derived healthy hull. Composition informs; it does not decide.</p>
<p><b>Mode 1 — Architectural concordance.</b> Each cell gets an <b>A-score</b>, a derived measure of how far its
methylation architecture has walked from its healthy floor (A ≈ 1.0 by derivation; no cohort, no population). The
pattern of those departures across cells is compared to each disease's signature by <b>scale-invariant cosine</b>
— the angle between the patient's departure vector and the signature, so absolute magnitudes never need converting and
nothing standardizes the patient against a population. This is the workhorse for field-effect and systemic disease,
whose signal sits in the <b>abundant immune cells we can score cleanly</b>. A-scores are produced from whole blood — the immune compartment and any shed cells abundant enough to score. It reports <b>resemblance, not a
probability</b>: v1 can recognize that a pattern looks like a template learned in validation cohorts; the calibrated
magnitude that would turn resemblance into "this stage of this disease" only accrues from real patients over years, in
our own derived A-units. Direction agreement is the coarse gate; cosine shape is the fine discriminator (it is what
separates aging from Alzheimer's — same direction, different proportions).</p>
<p><b>What separates this from a plain cell-of-origin readout:</b> listing which cells are present in plasma is table stakes. CPG measures how far each present cell's methylation architecture has departed from its own derived healthy floor — a physics-derived scale with no cohort and no population — and reads the pattern of those departures against disease signatures. The presence is the easy part; the architectural departure is the measurement.</p>
<p><b>Mode 2 — Cell-of-origin presence.</b> A different test: a cell <b>circulating in blood that should not be
there</b>. The one presence that is alarming by itself is a <b>blood-brain-barrier cell</b> (cortical neuron, glia,
oligodendrocyte) — it crossed a physical barrier to be in blood, so quantity and A-score are irrelevant to the call;
refer. Every other cell that can shed into blood — epithelial/secretory (breast, prostate, colon), cycling — does so at
a normal baseline. <b>Detecting them is not abnormal</b>; we detect and, when abundant enough, score them routinely. So
their presence is normal composition (see the composition table), and the A-score is the discriminator only when the
cell is abundant enough to score. A cell at ~1% is ~99% blood, so its architecture cannot be scored from the bulk signal
— we flag such a cell's presence as a barrier-breach. The sensitive front end is the cell's own <b>A-score</b>: VAL-090 showed the A-score catches a shed cortical neuron (1.292%) that the deconvolver fraction nulls — the A-score, not a fraction, is the shed-cell detector.</p>
<p><b>Confidence intervals</b> on every A-score are derived from the atlas's own MCMC posteriors (the 8 per-class
per-CpG brightness tables), not from a cohort. <b>Trajectory:</b> this report stores the patient's derived departure
vector as a baseline; a second or third test compares drift toward a signature's angle, which is a far stronger signal
than any single snapshot.</p>
</div>

<h2>Mode 1 — Architectural concordance</h2>
<table><thead><tr><th>Disease</th><th>Phase</th><th>Shape match</th><th>Direction</th><th>Shared cells</th><th>Status</th></tr></thead>
<tbody>{_mode1_rows(bundle)}</tbody></table>

<h2>Mode 2 — Cell-of-origin presence</h2>
<table><thead><tr><th>Class</th><th>Observed</th><th>Presence</th><th>Severity</th><th>Interpretation</th></tr></thead>
<tbody>{_mode2_rows(bundle)}</tbody></table>
{tissue_of_origin_html}
{disease_card_html}

<h2>Cell-level architectural departure <span class="meta">(the A-score readout — every cell found in blood)</span></h2>
{gauge_html}
{trajectory_html}
{departure_html}
{cell_census_html}
{cosmic_methylome_html}
{confirmation_html}
{strawman_html}

<h2>Reading the result</h2>
<div class="caveat">
A cell has to make up enough of the sample for its own methylation signal to rise above the background mixture before it can be scored; those cells get an A-score against their derived healthy floor. A cell diluted below that (~1% of blood, ~99% background) is still detected and reported (Mode 2), just not assigned an A-score.
Healthy cells land at A&#8776;1.0 when the patient beta is on the atlas scale. {_shed_note} The resemblance bands are not yet magnitude-calibrated — that accrues from real patients in our own derived units. This report flags and refers; it does not diagnose.
</div>

<details><summary>Machine-readable snapshot (for a future AI)</summary>
<pre>{_esc(json.dumps(snapshot, indent=2, default=str))}</pre></details>

<div class="foot">CPG vKISS · two detection modes · one deconvolver (Walther — composition/presence, gates no call) · whole blood · derived A-score + healthy hull, no cohort · MCMC-derived CI · matched-filter + trajectory confirmation · flags and refers, not diagnostic</div>
</div></body></html>"""

    if out_path:
        with open(out_path, "w") as f:
            f.write(H)
    return H


if __name__ == "__main__":
    print("cpg_report_builder.py — build_report(bundle, out_path) -> HTML.")
