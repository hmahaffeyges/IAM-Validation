#!/usr/bin/env python3
"""Report for chain v4 (2026-10-17). One self-contained HTML page per sample.

Front: the reading at a glance (every commissioned instrument, its reading, its state and its commissioning status), then one section per
instrument: the cell shares (IAM-Deconvolver against IAM-Atlas v2.1), Met-A on its gauge, the C-Score with its departures drawn on the
HEALPix sphere, and IAM-A against the sample's own reference on its gauge. Back: the technical record, the complete chain v3 report
(intake, calibration, tare, noise gate, red flags, safeguards, integrity, inventory, run it yourself), every section id unchanged so the
release check reads it as before. Research use; not a diagnostic test."""
import html, os, re, datetime
import report_v3 as R3
from report_v3 import CLAIM_SCAN, REPORT_SECTIONS   # noqa: F401  (the same prose rule and section ids)
E = html.escape
CSS = """<style>
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:960px;margin:24px auto;color:#1d2330;line-height:1.45;padding:0 16px}
h1{font-size:26px;margin:4px 0}h2{font-size:19px;border-bottom:2px solid #1d2330;padding-bottom:4px;margin-top:34px}
.meta{color:#556;font-size:13px}.note{background:#f4f6fa;border-left:4px solid #3a5a9b;padding:8px 12px;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:7px 9px;border-bottom:1px solid #dde2ea;text-align:left;vertical-align:top}
th{background:#f4f6fa;font-weight:600}.chip{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600}
.com{background:#dff3e4;color:#17602b}.dev{background:#fff1d6;color:#7a5200}.ref{background:#fde2e2;color:#8a1f1f}.na{background:#eceff3;color:#556}
.bar{background:#3a5a9b;height:12px;border-radius:2px}.bt{background:#eceff3;width:240px;border-radius:2px}.big{font-size:22px;font-weight:700}
.tech{margin-top:46px;border-top:3px double #1d2330;padding-top:8px}.tech h2{font-size:16px}img{max-width:100%}
</style>"""

def chip(status):
    s = str(status or "").lower()
    k = "com" if s.startswith("commissioned") else "ref" if s.startswith(("refused", "withheld", "not read")) else "na" if s.startswith("not run") else "dev"
    return f"<span class='chip {k}'>{E(str(status))}</span>"

def _glance(o):
    sh, m, c, t, q = o.get("cell_shares") or {}, o.get("met_a") or {}, o.get("met_a_cscore") or {}, o.get("tare") or {}, o.get("iam_a") or {}
    rows = []
    if sh.get("cells"):
        top = sorted(sh["cells"], key=lambda r: -r["share"])[:3]
        rows.append(("IAM-Deconvolver", "Which cells, in what shares", ", ".join(f"{r['cell']} {100*r['share']:.1f} %" for r in top), "",
                     "commissioned (neutrophils, monocytes, CD4 T, CD8 T)"))
    elif sh: rows.append(("IAM-Deconvolver", "Which cells, in what shares", "", "", "withheld: " + str(sh.get("withheld", ""))))
    else: rows.append(("IAM-Deconvolver", "Which cells, in what shares", "", "", "not run (no array input)"))
    if m.get("A") is not None:
        val = f"A_rel {t['A_rel']}" if t.get("A_rel") is not None else f"A {m['A']} (untared)"
        st = "commissioned" if m.get("commissioning") else "development (whole blood)"
        rows.append(("Met-A", "How far the neutrophil's tagging level has moved", val, str(t.get("state") or m.get("state", "")).split(". ")[0][:140], st))
    elif o.get("met_a") is not None or o.get("refusal"):
        rows.append(("Met-A", "How far the neutrophil's tagging level has moved", "", "", "refused: " + str(o.get("refusal") or m.get("reason", ""))))
    else: rows.append(("Met-A", "How far the neutrophil's tagging level has moved", "", "", "not run (no array input)"))
    if c.get("C") is not None:
        val = f"C_rel {c['C_rel']}" if c.get("C_rel") is not None else f"C {c['C']} (untared)"
        st = ("commissioned" if c.get("commissioning") else "not tared: needs 3 same-run healthy references" if c.get("C_rel") is None and o.get("specimen") == "whole blood"
              else "development")
        rows.append(("C-Score", "Are the departures clustered or scattered", val, str(c.get("C_call") or ""), st))
    else: rows.append(("C-Score", "Are the departures clustered or scattered", "", "", "not run" if not o.get("met_a") else "not read: " + str(c.get("reason", ""))))
    if q.get("A") is not None:
        st = "commissioned" if q.get("commissioning") else "development (" + str(q.get("cell")) + ")"
        rows.append(("IAM-A", "Is each mark still held, against the sample's own reference", f"A {q['A']}", str(q.get("state", "")), st))
    elif q: rows.append(("IAM-A", "Is each mark still held, against the sample's own reference", "", "", "not read: " + str(q.get("refusal", ""))))
    else: rows.append(("IAM-A", "Is each mark still held, against the sample's own reference", "", "", "not run (no sequencing input)"))
    return "<table><tr><th>Instrument</th><th>Question it answers</th><th>Reading</th><th>State</th><th>Status</th></tr>" + "".join(
        f"<tr><td><b>{E(a)}</b></td><td>{E(b)}</td><td class='big'>{E(v)}</td><td>{E(s)}</td><td>{chip(st)}</td></tr>" for a, b, v, s, st in rows) + "</table>"

def _shares(o):
    sh = o.get("cell_shares")
    if not sh: return ""
    h = "<h2 id='v4-shares'>Cell shares: IAM-Deconvolver against IAM-Atlas v2.1</h2>"
    if not sh.get("cells"): return h + f"<p>{chip('withheld')} {E(str(sh.get('withheld')))}</p>"
    rows = "".join(f"<tr><td>{E(r['cell'])}</td><td><div class='bt'><div class='bar' style='width:{max(1, int(240*r['share']))}px'></div></div></td>"
                   f"<td><b>{100*r['share']:.1f} %</b></td><td>{100*r['interval_95'][0]:.1f} to {100*r['interval_95'][1]:.1f} %</td><td>{chip(r['status'])}</td></tr>"
                   for r in sorted(sh["cells"], key=lambda r: -r["share"]))
    return (h + f"<table><tr><th>Cell</th><th></th><th>Share</th><th>Range (information)</th><th>Status</th></tr>{rows}</table>"
            f"<p class='note'>Read on {sh.get('sites_used')} of {sh.get('markers')} marker sites. Each cell's healthy profile, with its uncertainty and the spread "
            "between people, comes from the 79-cell atlas fitted as one model across 21 public sources. Commissioned on mixtures from a laboratory the "
            "atlas never saw: median share error 0.016. The ranges are printed as information; they are not calibrated.</p>")

def _meta(o):
    m, t = o.get("met_a") or {}, o.get("tare") or {}
    if m.get("A") is None: return ""
    gA = t.get("A_rel") if t.get("A_rel") is not None else (m["A"] if m.get("specimen") != "whole blood" else None)
    if str(m.get("state", "")).startswith("withheld"): gA = None   # the noise gate withholds the state: no gauge is drawn (as the v3 record)
    return ("<h2 id='v4-met-a'>Met-A: the neutrophil's tagging level on the gauge</h2>" + (R3._gauge(gA, label=f"Met-A {gA}") if gA is not None else "")
            + f"<p>{chip('commissioned' if m.get('commissioning') else 'development (whole blood)')} {E(str(t.get('state') or m.get('state', '')))}</p>")

def _cscore(o):
    c, k = o.get("met_a_cscore") or {}, o.get("cscore_sky") or {}
    if c.get("C") is None: return ""
    img = ""
    if k.get("drawn"):
        import stage_v4 as V4
        img = (f"<img alt='C-Score departures on the sphere' src='{V4.png_data_uri(k['png'])}'>"
               f"<p class='note'>The same departures the C-Score scores, its {k['sites']:,} sites in genomic order laid on a HEALPix sphere (NSIDE {k['nside']}, "
               "NESTED order, where neighboring sites fill compact patches), the way a microwave sky is drawn. Clustered departures show as patches; "
               "scattered ones as speckle. A picture of the reading, not a new reading.</p>")
    val = c.get("C_rel") if c.get("C_rel") is not None else c.get("C")
    return (f"<h2 id='v4-cscore'>C-Score: clustered or scattered</h2><p class='big'>{'C_rel' if c.get('C_rel') is not None else 'C'} {E(str(val))}</p>"
            f"<p>{chip('commissioned' if c.get('commissioning') else 'development')} {E(str(c.get('C_call', c.get('C_rel_reason', ''))))}. "
            "Healthy reads near 1; above 1.152 is a clustered departure.</p>" + img)

def _iama(o):
    q = o.get("iam_a") or {}
    if not q or q.get("stage") != "Q2": return ""
    h = "<h2 id='v4-iam-a'>IAM-A: the sample against its own reference</h2>"
    if q.get("A") is None: return h + f"<p>{chip('not read')} {E(str(q.get('refusal', '')))}</p>"
    st = "commissioned" if q.get("commissioning") else "development"
    return (h + R3._gauge(q["A"], q["band"]["lo"], q["band"]["hi"], label=f"IAM-A {q['A']}", instrument="IAM-A")
            + f"<p>{chip(st)} <b>{E(str(q.get('state')))}</b> (Normal {q['band']['lo']:.3f} to {q['band']['hi']:.3f}). Held error {q.get('held_err')} against "
            f"{q.get('predicted')} predicted by this sample's own loose share ({q.get('loose_share')}). No outside reference is used: the sample is read against itself.</p>")

def build(o, out, sid):
    tech = os.path.splitext(out)[0] + "_v3record.tmp.html"
    r = R3.build(o, tech, sid); t = open(tech, encoding="utf-8").read(); os.remove(tech)
    body = re.search(r"<body[^>]*>(.*)</body>", t, re.S).group(1)
    when = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    front = [f"<p class='meta'>IAM chain v4 · run {E(str(o.get('run_id')))} · {when}</p>", f"<h1>Sample report: {E(sid)}</h1>",
             f"<p class='meta'>Specimen {E(str(o.get('specimen', '')))} · platform {E(str(o.get('platform') or o.get('array_type') or 'sequencing'))}</p>",
             "<p class='note'><b>Research use only. Not a diagnostic test.</b> Each reading is printed with its commissioning status: "
             "<i>commissioned</i> means it passed a test sealed before its data was seen; <i>development</i> means it has not.</p>",
             "<h2 id='v4-glance'>At a glance</h2>", _glance(o), _shares(o), _meta(o), _cscore(o), _iama(o)]
    prose = re.sub(r"<[^>]+>", " ", "\n".join(front))
    if re.search(CLAIM_SCAN, prose, re.I): raise ValueError("report_v4: front prose carries a word the report may not print: " + re.search(CLAIM_SCAN, prose, re.I).group(0))
    H = (f"<html><head><meta charset='utf-8'><title>IAM chain v4 - {E(sid)}</title>{CSS}</head><body>" + "\n".join(front)
         + "<div class='tech'><h2 id='v4-technical'>Technical record</h2>" + body + "</div></body></html>")
    open(out, "w", encoding="utf-8").write(H)
    return {**r, "out": out, "bytes": len(H)}
