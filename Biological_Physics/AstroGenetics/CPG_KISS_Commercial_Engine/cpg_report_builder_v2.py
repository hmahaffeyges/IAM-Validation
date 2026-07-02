"""CPG report builder v2 — built on the CURRENT chain (2026-06-30).

Written fresh against the corrected chain so it carries none of the old traps:

  * The gauge is A = H(beta_mean)/H_min over IDENTITY loci, read against the
    AGE-MATCHED band (age_reference_matrix), NOT a fixed 0.95-1.04 line and NOT
    "A=1.0 is healthy". Healthy is the age band; the report shows placement
    (BELOW / IN / ABOVE the band) + the Issue-002 severity tier straight from
    the gauge engine (class_ascores already carry placement/tier/band).
  * class_ascores = the GAUGE (per-class, the fuel gauge). celltype_ascores =
    the SEPARATION surface (per-cell, discriminative) — a different instrument,
    shown as disease-matching, never as the gauge.
  * Cellular age is a headline delta BESIDE the per-class breakdown, never a lone
    number (a single number collapses up/down departures — see REAL_IDAT note).
  * The age-matched departure (Mahalanobis Option A) is the one-number distance
    from the age band, with the classes that drive it.
  * Bidirectional signals (AD) are shown from the directional panel (Stage 4.5),
    because they cancel in the pooled A-score.

Reads the run_pipeline bundle. Emits a single self-contained HTML string; writes
it if out_path is given. No cohort comparisons anywhere (DERIVED-only).

Old cpg_report_builder.py is kept as a structural/style reference only.
"""
from datetime import datetime
import html as _html


# ─── small helpers ───────────────────────────────────────────────────────────
def _esc(x):
    return _html.escape(str(x)) if x is not None else ""


def _pct(v):
    try:
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return _esc(v)


def _fmt_ci(lo, hi):
    if lo is None or hi is None:
        return "&mdash;"
    return f"[{lo:.3f}, {hi:.3f}]"


def _get(obj, name, default=None):
    """Read attr-or-key defensively (stage8 is an object; others are dicts)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# tier -> colour + plain-language
_TIER_STYLE = {
    "NORMAL":     ("var(--accent)", "within normal range"),
    "MARGINAL":   ("var(--amber)",  "marginally elevated"),
    "DETECTABLE": ("var(--amber)",  "elevated — detectable"),
    "URGENT":     ("var(--red)",    "elevated — urgent"),
    "BREACH":     ("var(--red)",    "floor breach"),
    "INVERSION":  ("#6d28d9",       "inversion (identity loss)"),
    "SUPPRESSED": ("var(--soft)",   "suppressed"),
    "N/A":        ("var(--soft)",   "not assessable"),
}
_PLACEMENT_TXT = {
    "IN_BAND":     "in age band",
    "ABOVE_BAND":  "above age band",
    "BELOW_BAND":  "below age band",
}


# ─── CSS (house style, reused verbatim from the v1 report) ───────────────────
_CSS = """
:root{--ink:#16202a;--soft:#516170;--line:#dfe3e8;--paper:#fbfcfd;--accent:#0e6b53;--red:#b2182b;--amber:#b45309;}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",Helvetica,sans-serif;line-height:1.5}
.wrap{max-width:920px;margin:0 auto;padding:34px 26px 70px}
h1{font-size:26px;margin:0 0 2px} h2{font-size:18px;margin:30px 0 8px;border-bottom:2px solid var(--ink);padding-bottom:4px}
h3{font-size:14px;margin:16px 0 4px;color:var(--accent)}
.meta{font-size:13px;color:var(--soft);font-family:ui-monospace,Menlo,monospace}
.ctx{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:12px 0;font-size:13px}
.ctx b{color:var(--soft);font-weight:600}
ul.exec{background:#fff;border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:0 8px 8px 0;padding:12px 16px 12px 32px;margin:10px 0}
ul.exec li{margin:5px 0}
table{width:100%;border-collapse:collapse;font-size:13px;margin:6px 0 4px}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--soft)}
td.num{font-family:ui-monospace,Menlo,monospace;text-align:right;white-space:nowrap}
.muted{color:var(--soft);font-style:italic} .red{color:var(--red)}
.pill{font-size:10px;padding:1px 7px;border-radius:10px;font-weight:700;letter-spacing:.03em;color:#fff;white-space:nowrap}
.method{background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 18px 14px;margin:8px 0;font-size:13px}
.method p{margin:9px 0} .method b{color:var(--ink)}
.caveat{background:#fff7ed;border:1px solid #f3d9b5;border-radius:8px;padding:8px 16px;font-size:13px;margin:8px 0}
.explain{background:#f3f7f6;border-left:3px solid var(--accent);border-radius:0 6px 6px 0;padding:9px 14px;margin:8px 0;font-size:12.5px;color:#33414d}
.explain b{color:var(--ink)}
.headline{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:10px 0;display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
.headline .big{font-size:30px;font-weight:700}
.bandbar{position:relative;height:26px;border-radius:5px;background:linear-gradient(90deg,#eef2f1,#eef2f1);border:1px solid var(--line);margin:3px 0}
.foot{margin-top:30px;font-size:11px;color:var(--soft);font-family:ui-monospace,Menlo,monospace}
"""


# ─── age-matched class gauge (the spine) ─────────────────────────────────────
def _class_gauge_section(bundle):
    s4 = bundle.get("stage4", {}) or {}
    ca = s4.get("class_ascores", {}) or {}
    rows = ""
    n_assess = 0
    for cls, r in sorted(ca.items(), key=lambda kv: -((kv[1] or {}).get("departure") or -9)):
        A = r.get("A")
        assessable = r.get("assessable", True)
        if A is None or (isinstance(A, float) and A != A) or not assessable:
            rows += (f"<tr><td>{_esc(cls)}</td><td class='num muted' colspan='5'>"
                     f"not assessable in this substrate</td></tr>")
            continue
        n_assess += 1
        band = r.get("band", {}) or {}
        placement = r.get("placement") or ""
        tier = r.get("tier") or "N/A"
        col, _txt = _TIER_STYLE.get(tier, ("var(--soft)", tier.lower()))
        band_txt = (f"{band.get('p10', float('nan')):.3f}&ndash;{band.get('p90', float('nan')):.3f}"
                    if band else "&mdash;")
        dep = r.get("departure")
        dep_txt = (f"{dep:+.3f}" if isinstance(dep, (int, float)) else "&mdash;")
        rows += (
            f"<tr><td>{_esc(cls)}</td>"
            f"<td class='num'>{A:.3f}</td>"
            f"<td class='num'>{_fmt_ci(r.get('A_ci_lo'), r.get('A_ci_hi'))}</td>"
            f"<td class='num'>{band_txt}</td>"
            f"<td class='num'>{dep_txt}</td>"
            f"<td><span class='pill' style='background:{col}'>{_esc(tier)}</span> "
            f"<span class='muted'>{_PLACEMENT_TXT.get(placement, '')}</span></td></tr>")
    if not rows:
        rows = "<tr><td colspan='6' class='muted'>No class A-scores in bundle.</td></tr>"
    note = (
        "<p class='meta'>The gauge, per class: <b>A = H(&#946;_mean)/H_min</b> over the class "
        "identity loci, read against the <b>age-matched band</b> (age_reference_matrix, this "
        "patient's age). Healthy is the band, not a fixed line &mdash; A = 1.0 is the architectural "
        "commitment line (the age-95 value), not where healthy sits. <b>Placement</b> is vs the "
        "patient's age peers (below / in / above the p10&ndash;p90 band); <b>tier</b> is the absolute "
        "severity ladder (NORMAL &lt; 1.01, MARGINAL &ge; 1.01, DETECTABLE &ge; 1.05, URGENT &ge; 1.07, "
        "BREACH &ge; 1.10). INVERSION = a genuine far-below identity-loss reading, a finding, not an error. "
        "No cohort comparison anywhere.</p>")
    empty = ("<div class='caveat'>No class was assessable-above-floor in this substrate. On whole "
             "blood, tissue classes are absent (read as background); a below-floor read on a present "
             "class is an input-scale problem (Stage-1 normalization) &mdash; the class A-scores are "
             "withheld rather than printed as artifacts.</div>") if n_assess == 0 else ""
    return (
        "<h2>Cellular Performance Gauge <span class='meta'>(age-matched, per architecture class)</span></h2>"
        + note + empty +
        "<table><thead><tr><th>Architecture class</th><th>A-score</th><th>95% CI</th>"
        "<th>age band (p10&ndash;p90)</th><th>departure</th><th>reading</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>")


# ─── cellular age (headline beside per-class, never a lone number) ───────────
def _cellular_age_section(bundle):
    cage = bundle.get("cellular_age") or {}
    overall = cage.get("cellular_age")
    chrono = cage.get("chronological_age")
    per = cage.get("per_class") or {}
    if overall is None or not per:
        return (
            "<h2>Cellular age</h2>"
            "<div class='caveat'>Cellular age not computed for this sample &mdash; it requires at "
            "least one architecture class assessable and holding identity (above its floor). On a "
            "whole-blood draw that is normally the immune class; a below-floor immune read points to "
            "Stage-1 normalization, not to a real age. Reported once the input is on the atlas scale.</div>")
    delta = cage.get("delta_years")
    delta_txt = ""
    if delta is not None:
        col = "var(--accent)" if abs(delta) <= 5 else ("var(--amber)" if abs(delta) <= 12 else "var(--red)")
        verdict = ("about where it should be for the patient's age" if abs(delta) <= 5
                   else ("aging faster than chronological age" if delta > 0
                         else "reading younger than chronological age"))
        delta_txt = (f"<span class='big' style='color:{col}'>{delta:+.0f} yr</span>"
                     f"<span class='muted'>vs chronological age &mdash; {verdict}</span>")
    prows = "".join(
        f"<tr><td>{_esc(c)}</td><td class='num'>{a:.0f} yr</td>"
        f"<td class='num'>{(a - chrono):+.0f} yr</td></tr>"
        for c, a in sorted(per.items(), key=lambda kv: -abs(kv[1] - (chrono or kv[1])))
    ) if chrono is not None else "".join(
        f"<tr><td>{_esc(c)}</td><td class='num'>{a:.0f} yr</td><td class='num muted'>&mdash;</td></tr>"
        for c, a in per.items())
    return (
        "<h2>Cellular age <span class='meta'>(headline + which cells drive it)</span></h2>"
        f"<div class='headline'><div><span class='meta'>overall cellular age</span><br>"
        f"<span class='big'>{overall:.0f} yr</span></div><div>{delta_txt}</div></div>"
        "<p class='explain'>The overall number is the friendly headline: it says whether cells are aging "
        "about on pace. It is <b>not</b> the finding on its own &mdash; a single age collapses up- and "
        "down-departures onto one scale and can hide a real signal. The per-class breakdown below is where "
        "the actionable information is: it says <b>which</b> compartment is off-pace. A single class running "
        "old is a lead; all classes running old is genuine accelerated aging.</p>"
        "<table><thead><tr><th>Architecture class</th><th>cellular age</th><th>vs chronological</th>"
        "</tr></thead><tbody>" + prows + "</tbody></table>")


# ─── age-matched departure (Mahalanobis Option A) ────────────────────────────
def _departure_section(bundle):
    m = bundle.get("mahalanobis") or {}
    d = m.get("mahalanobis_distance")
    if d is None:
        return ("<h2>Age-matched departure</h2><p class='muted'>Not computed "
                f"({_esc(m.get('status', 'no assessable classes'))}).</p>")
    thr = m.get("alarm_threshold_p95")
    beyond = m.get("mahalanobis_beyond_band")
    n = m.get("n_features_assessable", 0)
    col = "var(--red)" if beyond else "var(--accent)"
    verdict = ("beyond the age-matched healthy band" if beyond
               else "within the age-matched healthy band")
    top = m.get("top_axis_contributions", []) or []
    trows = "".join(
        f"<tr><td>{_esc(t.get('class'))}</td><td class='num'>{t.get('patient_A', float('nan')):.3f}</td>"
        f"<td class='num'>{t.get('age_matched_mean', float('nan')):.3f}</td>"
        f"<td class='num'>{t.get('z_shift', float('nan')):+.2f}</td></tr>"
        for t in top[:6])
    return (
        "<h2>Age-matched departure <span class='meta'>(one number, derived — Option A)</span></h2>"
        f"<div class='headline'><div><span class='meta'>departure distance</span><br>"
        f"<span class='big' style='color:{col}'>{d:.2f}</span></div>"
        f"<div><span class='muted'>alarm threshold (p95) {thr:.2f} &middot; {n} class(es) assessable "
        f"&middot; <b style='color:{col}'>{verdict}</b></span></div></div>"
        "<p class='explain'>Each assessable class gauge is z-scored against its own age-matched band "
        "(&mu; = A_mean(class, age), &sigma; from the p10&ndash;p90 spread) and the distances summed. "
        "The reference is <b>derived</b> from the age matrix, not a pooled cohort. The threshold adapts to "
        "how many classes were assessable (&chi;&sup2;). The classes below are the ones driving the distance.</p>"
        + (f"<table><thead><tr><th>class</th><th>patient A</th><th>age-matched mean</th><th>z</th>"
           f"</tr></thead><tbody>{trows}</tbody></table>" if trows else ""))


# ─── directional read for bidirectional signals (Stage 4.5 / AD) ─────────────
def _directional_section(bundle):
    s45 = bundle.get("stage4_5")
    if not s45 or (isinstance(s45, dict) and s45.get("status") == "not_run"):
        return ""
    # s45 may be a dataclass-dict; pull the immune AD directional read if present
    panels = _get(s45, "per_class") or _get(s45, "panels") or {}
    body = ""
    if isinstance(panels, dict) and panels:
        prows = ""
        for cls, p in panels.items():
            score = _get(p, "a_dir_score", _get(p, "directional_score"))
            flag = _get(p, "flag_bidirectional", _get(p, "flag"))
            if score is None:
                continue
            col = "var(--red)" if flag else "var(--soft)"
            prows += (f"<tr><td>{_esc(cls)}</td><td class='num'>{score:+.3f}</td>"
                      f"<td><span class='muted' style='color:{col}'>"
                      f"{'FLAG' if flag else 'no flag'}</span></td></tr>")
        if prows:
            body = ("<table><thead><tr><th>class</th><th>directional score</th><th>bidirectional</th>"
                    f"</tr></thead><tbody>{prows}</tbody></table>")
    if not body:
        return ""
    return (
        "<h2>Directional read <span class='meta'>(Stage 4.5 — composition-independent, sealed VAL-051)</span></h2>"
        "<p class='explain'>Some signals (Alzheimer's-direction immune drift is the validated case) are "
        "<b>bidirectional</b> at the CpG level &mdash; some loci rise, some fall &mdash; so they cancel in the "
        "pooled A-score and read near-normal on the gauge. This read scores a sealed, sign-anchored panel per "
        "CpG, so the departure is recovered instead of washed out. It is a <b>directional-panel score, not an "
        "A-score</b>, and it confirms rather than replaces the gauge above.</p>" + body)


# ─── composition (Stage 2) ───────────────────────────────────────────────────
def _composition_section(bundle):
    s2 = bundle.get("stage2") or {}
    fr = _get(s2, "class_fractions") or _get(s2, "fractions") or {}
    if not fr:
        return ""
    rows = "".join(f"<tr><td>{_esc(c)}</td><td class='num'>{_pct(v)}</td></tr>"
                   for c, v in sorted(fr.items(), key=lambda kv: -(kv[1] or 0)) if (v or 0) > 0.001)
    return ("<h2>Composition <span class='meta'>(Stage 2 — informs, never decides)</span></h2>"
            "<p class='explain'>What cells are in the mix, from the bulk methylation (the Walther "
            "constrained deconvolution). This resolves the composition so the gauge reads real cells, "
            "not bulk background. It does <b>not</b> fire any detection call.</p>"
            f"<table><thead><tr><th>class</th><th>fraction</th></tr></thead><tbody>{rows}</tbody></table>")


# ─── disease matching (Stage 8) + cell-of-origin (Mode 2) ────────────────────
def _matching_section(bundle):
    s8 = bundle.get("stage8")
    conc = _get(s8, "route_B_concordance", []) or []
    origin = bundle.get("cell_of_origin_flags", []) or []
    parts = []
    if conc:
        rows = ""
        for m in conc[:6]:
            name = _get(m, "disease", _get(m, "disease_id", "?"))
            cos = _get(m, "cosine", _get(m, "concordance"))
            rows += (f"<tr><td>{_esc(name)}</td>"
                     f"<td class='num'>{cos:+.2f}</td></tr>" if cos is not None
                     else f"<tr><td>{_esc(name)}</td><td class='num muted'>&mdash;</td></tr>")
        parts.append(
            "<h3>Mode 1 — architectural concordance</h3>"
            "<p class='explain'>The <i>shape</i> of the patient's departures across cells, compared to each "
            "disease signature by scale-invariant cosine (angle, not magnitude &mdash; no standardizing "
            "against any population). A high cosine means the departure pattern resembles that signature; the "
            "confirmation chain is what tests whether it is a real departure.</p>"
            f"<table><thead><tr><th>disease signature</th><th>cosine</th></tr></thead><tbody>{rows}</tbody></table>")
    if origin:
        orows = "".join(f"<tr><td>{_esc(_get(o, 'cell', o))}</td>"
                        f"<td>{_esc(_get(o, 'note', ''))}</td></tr>" for o in origin[:8])
        parts.append(
            "<h3>Mode 2 — cell-of-origin presence</h3>"
            "<p class='explain'>A second, independent detection mode: a cell that should not circulate in "
            "whole blood appearing anyway (blood-brain-barrier and shed tissue cells). Presence itself is the "
            "signal, independent of the A-score.</p>"
            f"<table><thead><tr><th>cell</th><th>note</th></tr></thead><tbody>{orows}</tbody></table>")
    if not parts:
        return ("<h2>Disease matching</h2><p class='muted'>No concordance match or cell-of-origin flag "
                "raised on this sample.</p>")
    return "<h2>Disease matching</h2>" + "".join(parts)


# ─── executive summary (auto from the bundle) ────────────────────────────────
def _exec_summary(bundle):
    lines = []
    s4 = bundle.get("stage4", {}) or {}
    ca = s4.get("class_ascores", {}) or {}
    flagged = [(c, r) for c, r in ca.items()
               if r.get("assessable", True) and r.get("A") is not None
               and r.get("tier") not in (None, "NORMAL", "N/A") and not r.get("below_floor")]
    if flagged:
        for c, r in sorted(flagged, key=lambda kv: -(kv[1].get("departure") or 0))[:3]:
            lines.append(f"<b>{_esc(c)}</b> reads {_esc(r.get('tier'))} "
                         f"({r.get('A'):.3f}, {_PLACEMENT_TXT.get(r.get('placement'), '')}).")
    else:
        lines.append("All assessable class gauges read within their age-matched normal range.")
    m = bundle.get("mahalanobis") or {}
    if m.get("mahalanobis_distance") is not None:
        lines.append(("Age-matched departure is <b class='red'>beyond</b> the healthy band."
                      if m.get("mahalanobis_beyond_band")
                      else "Age-matched departure is within the healthy band.")
                     + f" (distance {m['mahalanobis_distance']:.2f})")
    cage = bundle.get("cellular_age") or {}
    if cage.get("delta_years") is not None:
        lines.append(f"Overall cellular age {cage['cellular_age']:.0f} yr "
                     f"({cage['delta_years']:+.0f} vs chronological).")
    lines.append("<span class='muted'>All readings are consistent-with statements from a derived "
                 "reference; none is a population comparison or a diagnosis.</span>")
    return lines


# ─── main ─────────────────────────────────────────────────────────────────────
def build_report(bundle, out_path=None):
    ctx = bundle.get("context", {}) or {}
    exec_lines = "".join(f"<li>{l}</li>" for l in _exec_summary(bundle))
    body = (
        _composition_section(bundle)
        + _class_gauge_section(bundle)
        + _cellular_age_section(bundle)
        + _departure_section(bundle)
        + _directional_section(bundle)
        + _matching_section(bundle)
    )
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CPG Report — {_esc(bundle.get('patient_id'))}</title>
<style>{_CSS}</style></head><body><div class="wrap">
<h1>Cellular Performance Gauge — Report</h1>
<div class="meta">Patient {_esc(bundle.get('patient_id'))} &nbsp;·&nbsp; Test {_esc(bundle.get('test_id'))}
 &nbsp;·&nbsp; {datetime.utcnow():%Y-%m-%d} &nbsp;·&nbsp; chain {_esc(bundle.get('chain'))}</div>
<div class="ctx"><b>Context (read alongside, never used in scoring):</b>
age {_esc(ctx.get('age'))} · sex {_esc(ctx.get('sex'))} · substrate {_esc(ctx.get('substrate'))}
· family history {_esc(ctx.get('family_history') or 'none provided')}</div>
<h2>Executive summary</h2>
<ul class="exec">{exec_lines}</ul>
<div class="method">
<p>CPG reads a single <b>whole-blood</b> draw. The <b>gauge</b> places each architecture class against
its <b>age-matched</b> healthy band (derived, no cohort); the <b>age-matched departure</b> sums those into
one number; <b>cellular age</b> is the friendly headline with the per-class breakdown beside it; and the
<b>directional read</b> recovers bidirectional signals that cancel in a pooled score. Composition informs;
it never fires a call. Every line is a consistent-with statement, not a diagnosis.</p>
</div>
{body}
<div class="foot">CPG report builder v2 · derived-reference only · A = H(&#946;_mean)/H_min, age-matched ·
generated {datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}</div>
</div></body></html>"""
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html


if __name__ == "__main__":
    import sys
    print("cpg_report_builder_v2: import and call build_report(bundle). "
          "Sections: composition, age-matched gauge, cellular age, departure (Option A), "
          "directional (Stage 4.5), disease matching.", file=sys.stderr)
