#!/usr/bin/env python3
"""Report for chain v4 (2026-10-17; two views 2026-10-18). One self-contained HTML page per sample, and the same report as two PDFs.

Two views of the same readings (author 2026-10-16/18: researchers get the table, everyone else the gauge): the RESEARCHER view opens first
(results table; AstroGenetics, the physics behind each reading; the sample's sky beside its cell's healthy sky from IAM-Atlas v2.1; one section
per instrument), and the PLAIN view (the floor-to-ceiling gauges, one sentence each, and the sky). The technical record follows both.

Front: the reading at a glance (every commissioned instrument, its reading, its state and its commissioning status), then one section per
instrument: the cell shares (IAM-Deconvolver against IAM-Atlas v2.1), Met-A on its gauge, the C-Score with its departures drawn on the
HEALPix sphere, and IAM-A against the sample's own reference on its gauge. Back: the technical record, the complete chain v3 report
(intake, calibration, tare, noise gate, red flags, safeguards, integrity, inventory, run it yourself), every section id unchanged so the
release check reads it as before. Research use; not a diagnostic test."""
import html, os, re, datetime, json, math, base64
import report_v3 as R3
import report_figs_v4 as RF
HERE = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(HERE, "..")
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
.sub2{font-size:15px;color:#3a5a9b;margin:2px 0 10px;font-weight:600;letter-spacing:.2px}
.views{margin:18px 0 6px}.views input{display:none}.views label{display:inline-block;padding:6px 16px;border:1px solid #9aa3b2;cursor:pointer;font-size:14px;margin-right:-1px}
.views label:first-of-type{border-radius:6px 0 0 6px}.views label:last-of-type{border-radius:0 6px 6px 0}
#vr:checked~.vbar label[for=vr],#vp:checked~.vbar label[for=vp]{background:#1d2330;color:#fff}
#vr:checked~.plain{display:none}#vp:checked~.research{display:none}
.phys td:first-child{white-space:nowrap;font-weight:600}.phys .eq{font-family:Georgia,serif}
.pair{display:flex;gap:10px;flex-wrap:wrap}.pair figure{flex:1 1 440px;margin:0}.pair figcaption{font-size:12.5px;color:#556;margin-top:2px}
.plain p.say{font-size:15.5px;margin:4px 0 18px}
@media print{.vbar{display:none}.plain,.research{display:block!important}.plain{page-break-before:always}}
.explain{background:#f5f7fa;border-left:3px solid #8a96a8;padding:6px 12px;margin:10px 0 18px}.explain p{margin:5px 0;font-size:14px}</style>"""
def _uri(path): return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode() if path and os.path.exists(path) else None
def _H(p): return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
def _sci(v):
    m_, e_ = f"{v:.1e}".split("e"); return f"{m_}&times;10<sup>{int(e_)}</sup>"
def _marks(): return json.load(open(os.path.join(CH, "Runtime Matrices", "Gauge", "cell_gauge_marks_v1.json"), encoding="utf-8"))

_NAMES = {"cd4 t cells": "CD4 T cells", "cd8 t cells": "CD8 T cells", "b cells": "B cells", "nk cells": "NK cells", "neutrophils": "Neutrophils", "monocytes": "Monocytes"}
def _href():   # Met-A's healthy reference in bits per site, read from its frozen file (key 'floor'; it is the middle mark), never typed
    import json as _j; return float(_j.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Runtime Matrices", "Met_A_Floors", "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]["floor"])
def cellname(c):   # 2026-10-18: cell types printed as a reader writes them (the data keeps the atlas's own names)
    c = str(c); return _NAMES.get(c.lower(), c[:1].upper() + c[1:])
def untested(what, inst, shown=True, noun="score"):   # author 2026-10-18: one plain sentence for a reading outside the commissioned scope
    w = cellname(what) if what.lower() in _NAMES else what   # (the run record keeps status "development"; checks and scoring unchanged)
    if not shown: return f"{inst} is not yet officially commissioned on {w}."
    lead = "This share is" if noun == "shares" else "This score is"
    return f"{lead} preliminary until {inst} is officially commissioned on {w}."
def chip(status):
    s = str(status or "").lower()
    k = "com" if s.startswith("commissioned") else "ref" if s.startswith(("refused", "withheld", "not read")) else "na" if s.startswith("not run") else "dev"
    return f"<span class='chip {k}'>{E(str(status))}</span>"

def _glance(o):
    return "<table><tr><th>Instrument</th><th>Question it answers</th><th>Reading</th><th>State</th><th>Status</th></tr>" + "".join(
        f"<tr><td><b>{E(a)}</b></td><td>{E(b)}</td><td class='big'>{E(v)}</td><td>{E(s)}</td><td>{chip(st)}</td></tr>" for a, b, v, s, st in glance_rows(o)) + "</table>"

def glance_rows(o):
    sh, m, c, t, q = o.get("cell_shares") or {}, o.get("met_a") or {}, o.get("met_a_cscore") or {}, o.get("tare") or {}, o.get("iam_a") or {}
    rows = []
    if sh.get("cells"):
        top = sorted(sh["cells"], key=lambda r: -r["share"])[:3]
        rows.append(("IAM-Deconvolver", "Which cells, in what shares", ", ".join(f"{cellname(r['cell'])} {100*r['share']:.1f} %" for r in top), "",
                     "commissioned (neutrophils, monocytes, CD4 T, CD8 T)" if any(x.get("status") == "commissioned" for x in sh["cells"])
                     else untested(f"a panel of {len(sh['cells'])} cell types", "the IAM-Deconvolver", noun="shares")))
    elif sh: rows.append(("IAM-Deconvolver", "Which cells, in what shares", "", "", "withheld: " + str(sh.get("withheld", ""))))
    else: rows.append(("IAM-Deconvolver", "Which cells, in what shares", "", "", "not run (no array input)"))
    if m.get("A") is not None:
        wh = str(m.get("state", "")).startswith("withheld")
        val = "no reading" if wh else (f"A_rel {t['A_rel']}" if t.get("A_rel") is not None else f"A {m['A']} (untared)")
        st = "commissioned" if m.get("commissioning") else untested(m.get("specimen") or "whole blood", "Met-A", shown=not str(m.get("state", "")).startswith("withheld"))
        rows.append(("Met-A", "How far the neutrophil's tagging level has moved", val,
                     "withheld by the array's noise check" if wh else str(t.get("state") or m.get("state", "")).split(". ")[0][:140], st))
    elif o.get("met_a") is not None or o.get("refusal"):
        rows.append(("Met-A", "How far the neutrophil's tagging level has moved", "", "", "refused: " + str(o.get("refusal") or m.get("reason", ""))))
    else: rows.append(("Met-A", "How far the neutrophil's tagging level has moved", "", "", "not run (no array input)"))
    ec = o.get("met_a_each_cell") or {}
    if ec.get("cells"):   # 2026-10-22: Met-A for each cell of a whole blood, against IAM-Atlas v2.1 in the person's own shares (development, flagged)
        rd = [x for x in ec["cells"] if x.get("A") is not None]
        rows.append(("Met-A, each cell", "How far each blood cell's tagging level has moved (against IAM-Atlas v2.1 in this person's own cell mix)",
                     ", ".join(f"{cellname(x['cell'])} {x['A']:.3f}" for x in rd) + (" (untared)" if rd else ""), "",
                     untested("each blood cell in whole blood", "Met-A")))
    mw = o.get("met_a_whole") or {}
    if mw.get("A") is not None:   # 2026-10-23: Met-A for the whole sample over every atlas site (development, flagged)
        bp = mw.get("by_place") or {}
        rows.append(("Met-A, whole sample", "Has the sample as a whole departed from its own expected healthy self, over every atlas site",
                     f"A {mw['A']} ({mw['sites']:,} sites; " + "; ".join(f"{k} {v['A']}" for k, v in bp.items() if v) + ")", "",
                     "development, not commissioned; " + str(mw.get("state") or "untared")))
    elif mw.get("reason"): rows.append(("Met-A, whole sample", "Has the sample as a whole departed", "", "", "not read: " + str(mw.get("reason"))[:120]))
    te = (o.get("territories") or {}).get("sets") or {}
    if any(v.get("A") is not None for v in te.values()):   # 2026-10-23: cancer's own territories (development, flagged)
        rows.append(("Territories", "Gains at islands every healthy cell keeps unmarked; losses in loosely held domains",
                     "; ".join(f"{k} A {v.get('A')}, level change {v.get('level_change'):+}" for k, v in te.items() if v.get("A") is not None), "",
                     "development, not commissioned; untared"))
    sp = o.get("info_sphere") or {}
    if sp.get("territories_read"):   # 2026-10-23: the information sphere, on by default (development, flagged)
        rows.append(("Information sphere", "Which cell programs moved, and how far the sample sits from its own expected healthy self",
                     f"{sp.get('territories_changed')} of {sp.get('territories_read')} territories changed; distance {sp.get('fisher_rao_rms_rad')} rad", "",
                     "development, not commissioned (page " + str(sp.get("html") or "") + ")"))
    elif sp.get("reason"): rows.append(("Information sphere", "Which cell programs moved", "", "", "not drawn: " + str(sp.get("reason"))[:120]))
    if c.get("C") is not None:
        val = f"C_rel {c['C_rel']}" if c.get("C_rel") is not None else f"C {c['C']} (untared)"
        st = ("commissioned" if c.get("commissioning") else "not tared: needs 3 same-run healthy references" if c.get("C_rel") is None and o.get("specimen") == "whole blood"
              else untested(o.get("specimen") or "this specimen", "the C-Score"))
        rows.append(("C-Score, commissioned", "Are the neutrophil's departures clustered or scattered", val, str(c.get("C_call") or ""), st))
    else: rows.append(("C-Score, commissioned", "Are the neutrophil's departures clustered or scattered", "", "", "not run" if not o.get("met_a") else "not read: " + str(c.get("reason", ""))))
    ce = o.get("cscore_each_cell") or {}; wg = ce.get("whole_genome") or {}   # 2026-10-23: the other three C-Scores, Group-A and the globe, always listed
    rows.append(("C-Score, whole genome", "Are the departures clustered, over every atlas site", f"C {wg['C']}" if wg.get("C") is not None else "", "", "development, not commissioned" if wg.get("C") is not None else "not read: " + str(ce.get("reason") or "no cell shares")))
    each = [x for x in ce.get("cells", []) if x.get("C") is not None]
    rows.append(("C-Score, each cell", "Are each cell's departures clustered", ", ".join(f"{cellname(x['cell'])} {x['C']:.3f}" for x in sorted(each, key=lambda x: -(x.get("share") or 0))[:6]), "", "development, not commissioned" if each else "not read"))
    qc = (q or {}).get("cscore") or {}
    rows.append(("C-Score, IAM-A", "Are the held-error departures clustered, region by region", f"C {qc['C']}" if qc.get("C") is not None else "", "", "development" if qc.get("C") is not None else "not run (needs sequencing input)"))
    g = o.get("globe") or {}
    rows.append(("Globe", "Where on the genome the sample changed", f"{g.get('blocks_changed')} of {g.get('blocks'):,} blocks changed" if g.get("drawn") else "", "", "development, not commissioned" if g.get("drawn") else "not drawn: " + str(g.get("reason") or "")[:120]))
    ga = o.get("group_a") or {}
    rows.append(("Group-A", "Does this person's group differ from its healthy group", str(ga.get("reading") or ""), "", str(ga.get("status") or "a group reading: shown in the series summary when this sample is read as one member of a series")))
    if q.get("A") is not None:
        st = "commissioned" if q.get("commissioning") else untested(str(q.get("cell")), "IAM-A")
        rows.append(("IAM-A", "Is each mark still held, against the sample's own reference", f"A {q['A']}", str(q.get("state", "")), st))
    elif q: rows.append(("IAM-A", "Is each mark still held, against the sample's own reference", "", "", "not read: " + str(q.get("refusal", ""))))
    else: rows.append(("IAM-A", "Is each mark still held, against the sample's own reference", "", "", "not run (no sequencing input)"))
    return rows

def _shares(o):
    sh = o.get("cell_shares")
    if not sh: return ""
    h = "<h2 id='v4-shares'>Cell shares: IAM-Deconvolver against IAM-Atlas v2.1</h2>"
    if not sh.get("cells"): return h + f"<p>{chip('withheld')} {E(str(sh.get('withheld')))}</p>"
    rows = "".join(f"<tr><td>{E(cellname(r['cell']))}</td><td><div class='bt'><div class='bar' style='width:{max(1, int(240*r['share']))}px'></div></div></td>"
                   f"<td><b>{100*r['share']:.1f} %</b></td><td>{100*r['interval_95'][0]:.1f} to {100*r['interval_95'][1]:.1f} %</td><td>{chip(r['status'] if r['status'] != 'development' else untested(r['cell'] if 'cell' in r else 'this cell type', 'the IAM-Deconvolver', noun='shares'))}</td></tr>"
                   for r in sorted(sh["cells"], key=lambda r: -r["share"]))
    return (h + f"<table><tr><th>Cell</th><th></th><th>Share</th><th>Range (information)</th><th>Status</th></tr>{rows}</table>"
            f"<p class='note'>Read on {sh.get('sites_used')} of {sh.get('markers')} marker sites. Each cell's healthy profile, with its uncertainty and the spread "
            "between people, comes from the 79-cell atlas fitted as one model across 21 public sources. Commissioned on mixtures from a laboratory the "
            "atlas never saw: median share error 0.016. The ranges are printed as information; they are not calibrated.</p>")

def _meta(o):
    m, t = o.get("met_a") or {}, o.get("tare") or {}
    if m.get("A") is None: return ""
    gA = t.get("A_rel") if t.get("A_rel") is not None else (m["A"] if m.get("specimen") not in ("whole blood", "buffy coat") else None)
    if str(m.get("state", "")).startswith("withheld"): gA = None   # the noise gate withholds the state: no gauge is drawn (as the v3 record)
    _f = (o.get("_figs") or {}).get("met_a")
    return ("<h2 id='v4-met-a'>Met-A: the neutrophil's tagging level on the gauge</h2>" + (f"<img alt='Met-A gauge' src='{_uri(_f)}'>" if _f else "")
            + f"<p>State: <b>{'withheld by the array&#39;s noise check' if str(m.get('state', '')).startswith('withheld') else E(str(t.get('state') or m.get('state', '')))}</b></p>"
            + f"<p>{chip('commissioned' if m.get('commissioning') else untested(m.get('specimen') or 'whole blood', 'Met-A', shown=not str(m.get('state', '')).startswith('withheld')))}</p>")

def _cscore(o):
    c, k = o.get("met_a_cscore") or {}, o.get("cscore_sky") or {}
    if c.get("C") is None: return ""
    _f = (o.get("_figs") or {}).get("cscore")
    img = (f"<img alt='C-Score bar' src='{_uri(_f)}'>" if _f else "") + _skypair(o)
    val = c.get("C_rel") if c.get("C_rel") is not None else c.get("C")
    return (f"<h2 id='v4-cscore'>C-Score: clustered or scattered</h2><p class='big'>{'C_rel' if c.get('C_rel') is not None else 'C'} {E(str(val))}</p>"
            f"<p>{chip('commissioned' if c.get('commissioning') else ('not tared: needs 3 same-run healthy references (whole blood is its commissioned scope)' if o.get('specimen') == 'whole blood' and c.get('C_rel') is None else untested(o.get('specimen') or 'this specimen', 'the C-Score')))} {E(str(c.get('C_call', c.get('C_rel_reason', ''))))}. "
            "Healthy reads near 1; above 1.152 is a clustered departure.</p>" + img)

def _iama(o):
    q = o.get("iam_a") or {}
    if not q or q.get("stage") != "Q2": return ""
    h = "<h2 id='v4-iam-a'>IAM-A: the sample against its own reference</h2>"
    if q.get("A") is None: return h + f"<p>{chip('not read')} {E(str(q.get('refusal', '')))}</p>"
    st = "commissioned" if q.get("commissioning") else untested(str(q.get("cell")), "IAM-A")
    _f = (o.get("_figs") or {}).get("iam_a")
    return (h + (f"<img alt='IAM-A gauge' src='{_uri(_f)}'>" if _f else "")
            + f"<p>{chip(st)} <b>{E(str(q.get('state')))}</b> (Normal {q['band']['lo']:.3f} to {q['band']['hi']:.3f}). Held error {q.get('held_err')} against "
            f"{q.get('predicted')} predicted by this sample's own loose share ({q.get('loose_share')}). No outside reference is used: the sample is read against itself.</p>"
            + (f"<p>Counting: {(q.get('counting') or {}).get('opportunities', 0):,} held-error chances counted; from counting alone A lies within "
               f"{q['counting']['A_95'][0]:.3f} to {q['counting']['A_95'][1]:.3f} (95 %), {100 * q['counting']['share_of_band_width']:.0f} % of the Normal band's width.</p>"
               if (q.get("counting") or {}).get("A_95") else ""))


def _figs(o, base):
    """The gauge pictures for this sample (PNG beside the report); a withheld or unread instrument gets none."""
    F, G = {}, _marks(); m, t, c, q = o.get("met_a") or {}, o.get("tare") or {}, o.get("met_a_cscore") or {}, o.get("iam_a") or {}
    gA = t.get("A_rel") if t.get("A_rel") is not None else (m.get("A") if m.get("specimen") not in ("whole blood", "buffy coat") else None)
    if m.get("A") is not None and gA is not None and not str(m.get("state", "")).startswith("withheld"):
        st = str(t.get("state") or m.get("state", "")).split(":")[0].split("(")[0].strip()[:40]
        F["met_a"] = RF.gauge_png(base + "_gauge_met_a.png", "Met-A", float(gA), float(G["floor"]["A"]), float(G["full_surface"]["Met-A"]), (0.95, 1.05),
                                  f"neutrophils, {o.get('specimen', '')}", st)
    if c.get("C") is not None:
        cv = c.get("C_rel") if c.get("C_rel") is not None else None
        F["cscore"] = RF.cscore_png(base + "_gauge_cscore.png", cv, 1.152, 0.877, f"{o.get('specimen', '')}",
                                    str(c.get("C_call") or ("not tared: no reading on the band" if cv is None else "")))
    if q.get("A") is not None and q.get("predicted"):
        M = G["M_cell"]["value"]; ep = float(q["predicted"])
        F["iam_a"] = RF.gauge_png(base + "_gauge_iam_a.png", "IAM-A", float(q["A"]), (1 / (1 + math.exp(M))) / ep, 0.5 / ep, (q["band"]["lo"], q["band"]["hi"]),
                                  f"{cellname(q.get('cell', ''))}, against its own reference", str(q.get("state", "")))
    return F

PHYS_INTRO = ("One law runs under every reading here: a mark held against thermal noise costs energy, and the loss of that hold shows as entropy, "
              "in bits. Each reading is a ratio to its healthy reference, so healthy reads 1. Its gauge runs from 0, the floor (below it a cell can no "
              "longer hold its identity), to saturation, the most the scale can register (every identity site fully scrambled). Both ends are derived "
              "from physics, not fitted to data.")
def _physics(o):
    rows = physics_rows(o)
    if not rows: return ""
    return ("<h2 id='v4-physics'>AstroGenetics: the physics behind each reading</h2><p>" + PHYS_INTRO + "</p>"
            "<table class='phys'><tr><th>Instrument</th><th>What was measured</th><th>The physics</th></tr>"
            + "".join(f"<tr><td>{E(a)}</td><td>{b}</td><td>{c_}</td></tr>" for a, b, c_ in rows) + "</table>")

def physics_rows(o):
    """AstroGenetics: each reading in the physics it rests on. Every number from this run's record or a committed reference file."""
    G = _marks(); M = G["M_cell"]["value"]; rows = []; m, t, c, q = o.get("met_a") or {}, o.get("tare") or {}, o.get("met_a_cscore") or {}, o.get("iam_a") or {}
    if q.get("A") is not None and q.get("predicted") and q.get("held_err") is not None:   # the own-reference reading only (the retired position form has no prediction)
        e, ep, L = float(q["held_err"]), float(q["predicted"]), float(q.get("loose_share") or 0); emin = 1 / (1 + math.exp(M))
        rows.append(("IAM-A", f"Held error &epsilon; = {e:.4f}: of the qualifying molecules, the share of sites left unwritten between written neighbors. "
                     f"This sample's loose share L = {L:.3f} predicts &epsilon;<sub>pred</sub>(L) = {ep:.4f} from the self-reference line; "
                     f"<b>A = &epsilon;/&epsilon;<sub>pred</sub> = {float(q['A']):.3f}</b>.",
                     f"In bits: H(&epsilon;) = {_H(e):.3f} bits per site. Holding energy E/kT = ln((1&minus;&epsilon;)/&epsilon;) = {math.log((1 - e) / e):.2f}. "
                     f"<b>Gauge: 0 &middot; healthy 1 &middot; saturation {0.5 / ep:.2f}.</b> Healthy 1 is this sample's own predicted held error, {ep:.4f}. "
                     f"Saturation: half the sites unwritten, nothing held (H = 1 bit per site). The floor sits at A = {_sci(emin / ep)}, effectively 0: "
                     f"the least error one ATP per site (M = {M}) can buy against thermal noise."))
    if m.get("A") is not None and str(m.get("state", "")).startswith("withheld"):
        rows.append(("Met-A", "No reading: the noise gate withheld the state on this array (" + E(str(m["state"]).split(":", 1)[-1].split("(")[0].strip()) + ").",
                     "A reading is given only when enough of the array's noise sites are measured to show the array is clean enough to read."))
    elif m.get("A") is not None:
        a = t.get("A_rel") if t.get("A_rel") is not None else m["A"]
        ref_txt = (f"Healthy 1 is the median of this run's {t.get('n_refs')} healthy references (their own Met-A {float(t['reference_median']):.3f}). "
                   if t.get("A_rel") is not None and t.get("reference_median") is not None else "Healthy 1 is the committed healthy reference of the cell type. ")
        bits = (f"On isolated neutrophils the healthy reference is {_href():.3f} bits per site, so this reading is {float(a) * _href():.3f} bits per site. "
                if m.get("specimen") not in ("whole blood", "buffy coat") else "In whole blood the expectation is the healthy profile of each cell type, weighted by this sample's cell shares. ")
        rows.append(("Met-A", f"&lang;H(&beta;)&rang; over {m.get('n_sites', '')} identity sites of the neutrophil, divided by its healthy expectation; "
                     f"<b>{'A_rel' if t.get('A_rel') is not None else 'A'} = {float(a):.3f}</b>.",
                     f"<b>Gauge: 0 &middot; healthy 1 &middot; saturation {float(G['full_surface']['Met-A']):.2f}.</b> " + ref_txt + bits
                     + "The floor (0 on the gauge) is the least entropy a cell can carry and still hold its identity; saturation is every identity site fully scrambled."))
    if c.get("C") is not None:
        rows.append(("C-Score", f"How the departures of Met-A's identity sites cluster along the genome: the variance of the means of blocks of "
                     f"{c.get('block_sites', 10)} neighboring sites over the variance of single sites, divided by the healthy median; "
                     f"<b>{'C_rel' if c.get('C_rel') is not None else 'C'} = {float(c.get('C_rel') if c.get('C_rel') is not None else c['C']):.3f}</b>.",
                     "A ratio of clustering, so it has no derived floor or ceiling. Healthy reads near 1; the Normal band 0.877 to 1.152 was set on "
                     "617 healthy arrays from 19 studies and held in two laboratories that had no part in setting it. The sky below shows the same departures."))
    return rows

SKY_CAP = ("Gray is where this sample matches healthy neutrophils from IAM-Atlas v2.1 (the shade is the healthy cell's own tagging). Color marks only "
           "blocks of 10 neighboring identity sites, in genome order, whose mean departure is beyond the line pure chance crosses less than once in 20 "
           "samples (5 % family-wise over all blocks): red where the pattern is more disordered than healthy (marks lost), blue where it is more ordered. "
           "This is the globe's own change rule, so the flat picture and the globe count the same blocks. A picture of the reading; nothing is scored from it.")
def _skypair(o):   # the overlay replaces the side-by-side pair (author 2026-10-18: "quieten the matches and showcase the changes")
    c, a = o.get("changes_sky") or {}, o.get("atlas_sky") or {}
    if not c.get("drawn"): return ""
    hw = a.get("range_halfwidth") or {}
    sure = (f" How sure the healthy reference is at these sites: its 95 % range is half {hw['median']:.3f} wide at the median site (0 would be exact; "
            f"nine sites in ten within {hw['p90']:.3f})." if hw else "")
    return (f"<h3 id='v4-skypair'>What changed: the sample laid over its healthy reference</h3><figure><img alt='This sample laid over healthy neutrophils' "
            f"src='{_uri(c['png'])}'><figcaption>{SKY_CAP}{sure}</figcaption></figure>")

def _plain(o, F):
    sh, m, t, c, q = o.get("cell_shares") or {}, o.get("met_a") or {}, o.get("tare") or {}, o.get("met_a_cscore") or {}, o.get("iam_a") or {}
    P = ["<h2 id='v4-plain'>The readings, in plain words</h2><p>Each gauge runs from 0, the floor (below it a cell can no longer hold its identity), "
         "through 1, healthy, to saturation, the most the scale can register. Healthy cells sit in the green band around 1. Moving right means the marks are being lost.</p>"]
    if sh.get("cells"):
        top = sorted(sh["cells"], key=lambda r: -r["share"])[:4]
        P.append("<h3>Which cells are in the sample</h3><p class='say'>" + E(", ".join(f"{cellname(r['cell'])} {100 * r['share']:.0f} %" for r in top))
                 + ", read against the 79-cell IAM-Atlas v2.1.</p>")
    for key, name, say in (("met_a", "Met-A", "How far the neutrophils' tagging level sits from healthy."),
                           ("iam_a", "IAM-A", "Whether each mark is still held, read against this sample's own reference: no other person is needed."),
                           ("cscore", "C-Score", "Whether the changes are scattered, as in healthy cells, or clustered along the genome.")):
        if key in F: P.append(f"<h3>{name}</h3><img alt='{name} gauge' src='{_uri(F[key])}'><p class='say'>{E(say)}</p>")
        elif key == "met_a" and str(m.get("state", "")).startswith("withheld"):
            P.append("<h3>Met-A</h3><p class='say'>No reading on this sample: the array's own noise check could not confirm it was clean enough to read, so Met-A held back rather than guess.</p>")
        elif key == "iam_a" and q: P.append(f"<h3>IAM-A</h3><p class='say'>Not read: {E(str(q.get('refusal', '')).split('.')[0])}.</p>")
    k = o.get("cscore_sky") or {}
    if k.get("drawn"): P.append(f"<h3>The sample on the sphere</h3><img alt='sky' src='{_uri(k['png'])}'><p class='say'>Each dot of color is one place in the genome. "
                                "Speckle is what healthy looks like; patches mean the changes are bunched together.</p>")
    return "".join(P)


# ---- 2026-10-23 (author: "That report needs updating with ALL those tests"; "both Met-A, all 4 C-Scores, Territory Score, Group-A, IAM-A,
# Globe and Information Sphere all correctly represented"): ONE source, detail_tables(o), drawn by the HTML report AND both PDFs. Every
# instrument always has its table: a reading, or the reason it did not fire.
BLOOD_FAMILIES = ("T cells", "B and NK cells", "Myeloid cells")
def _fam_of():
    try:
        import stage_info_sphere as _IS
        return {c: n for n, _, cs in _IS.FAMILIES for c in cs}
    except Exception: return {}
def _f(v, d=4): return "" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))
DEV = "development, not commissioned"

def detail_tables(o):
    """[(id, title, intro, header, rows, note)] - plain text cells; one per instrument, in the order the chain reads them."""
    T = []; sp_ = str(o.get("specimen") or "").lower(); F = _fam_of()
    sh = (o.get("cell_shares") or {}).get("cells") or []
    ru = o.get("atlas_ruler") or {}; mp = ru.get("maps") or {}
    T.append(("v4-atlas", "IAM-Atlas v2.1 and the ruler",
              "Every atlas reading compares this sample with IAM-Atlas v2.1: 79 healthy cell types, 813,951 sites, built once from healthy donors. The ruler puts the array on the atlas's scale first, one straight line per probe design through sites every healthy cell type holds unmarked and marked (islands and their shores are never anchors). The shift is shown, never hidden.",
              ["Probe design", "Array low anchor", "Atlas low", "Array high anchor", "Atlas high", "Slope"],
              [[d, _f(m.get("L")), _f(m.get("L_ref")), _f(m.get("U")), _f(m.get("U_ref")), _f(m.get("slope"))] for d, m in mp.items() if m],
              (DEV + "." if mp else "Not applied: " + str(ru.get("reason") or "no ruler record") + ".")))
    odd = [r for r in sh if (r.get("share") or 0) >= 0.02 and F.get(r["cell"]) and F.get(r["cell"]) not in BLOOD_FAMILIES] if sp_ in ("whole blood", "buffy coat", "pbmc") else []
    if odd:
        T.append(("v4-odd", "Cells not usually found in circulating blood",
                  "The IAM-Deconvolver reports what fits. A share like this can be the platform's own difference from the atlas, or a change the healthy atlas has no name for. It is shown, not hidden, and read with care.",
                  ["Cell", "Share", "Family"], [[cellname(r["cell"]), f"{100*r['share']:.1f} %", F.get(r["cell"])] for r in odd], ""))
    w = o.get("met_a_whole") or {}
    T.append(("v4-whole", "Met-A, whole sample: every atlas site",
              "Disorder at every atlas site against the atlas mixed in this sample's own cell shares, each site counted by how firmly a healthy sample of this mix holds it. The sensitive reading: it moves when anything in the sample departs. Healthy reads 1.",
              ["Sites", "Met-A", "Standard error", "Sites read"],
              ([["whole sample", _f(w.get("A")), _f(w.get("se_over_sites"), 5), f"{w.get('sites', 0):,}"]] + [[k, _f(v["A"]), _f(v.get("se_over_sites"), 5), f"{v.get('sites', 0):,}"] for k, v in (w.get("by_place") or {}).items() if v]) if w.get("A") is not None else [],
              (f"A 1 % loss of the held pattern moves this reading by about {_f(w.get('shift_per_1pct_loss'))}. State: {w.get('state', 'untared')}. {DEV}." if w.get("A") is not None else "Not read: " + str(w.get("reason") or "no cell shares") + ".")))
    ec = o.get("met_a_each_cell") or {}; cells = sorted(ec.get("cells", []), key=lambda x: -(x.get("share") or 0))
    T.append(("v4-each", "Met-A for each cell: which cell moved",
              "Each cell found, read at its own identity sites against its healthy profile in the atlas, in this sample's mix. When the whole-sample reading moves, these say which cell.",
              ["Cell", "Share", "Met-A", "Note"],
              [[cellname(x["cell"]), f"{100*(x.get('share') or 0):.1f} %", _f(x.get("A")) or "-", x.get("coverage_flag") or x.get("reason") or ""] for x in cells],
              DEV + ". Untared." if cells else "Not read: " + str(ec.get("reason") or "no cell shares") + "."))
    m = o.get("met_a") or {}; t = o.get("tare") or {}
    T.append(("v4-meta-c", "Met-A, commissioned: the neutrophil on its gauge",
              "The commissioned reading: the neutrophil's disorder at its 6,000 identity sites, against its frozen healthy reference, tared against same-run healthy arrays. Commissioned on isolated neutrophils, EPIC v1.",
              ["Reading", "Value", "State"],
              [["A_rel" if t.get("A_rel") is not None else "A (untared)", _f(t.get("A_rel") if t.get("A_rel") is not None else m.get("A")), str(t.get("state") or m.get("state") or "")[:160]]] if m.get("A") is not None else [],
              ("" if m.get("A") is not None else "Not read: " + str(m.get("reason") or (o.get("refusal") or "")) [:200] + ".")))
    te = (o.get("territories") or {}).get("sets") or {}
    T.append(("v4-territories", "Territories: islands and loosely held domains",
              "Islands: sites every healthy cell type keeps unmarked, where a mark gained is a departure no healthy cell makes. Domains: marked sites some healthy cell types hold loosely, where a held mark is most easily lost. Each read against the atlas in this sample's mix; healthy reads 1.",
              ["Territory", "Disorder (A)", "Level change", "Standard error", "Sites measured"],
              [[k, _f(v.get("A")), (f"{v.get('level_change'):+.4f}" if v.get("level_change") is not None else ""), _f(v.get("se_over_sites"), 5), f"{v.get('measured', 0):,} of {v.get('set_size', 0):,}"] for k, v in te.items()],
              (DEV + ". State: " + str((o.get("territories") or {}).get("state") or "untared") + ".") if te else "Not read."))
    cl = []   # 2026-10-23: distance to the entropy ceiling (CAN-005), for every atlas reading that carries it
    def _cl(name, A, c):
        if c and c.get("A_sat"): cl.append([name, _f(A), _f(c.get("A_sat"), 3), _f(c.get("distance_to_ceiling"), 4), _f(c.get("turnover_share"), 4), "yes" if c.get("past_entropy_ceiling") else "no"])
    _cl("Met-A, whole sample", w.get("A"), w.get("ceiling"))
    for k, v in (w.get("by_place") or {}).items():
        if v: _cl("  " + k, v.get("A"), v.get("ceiling"))
    for x in cells:
        if x.get("A") is not None: _cl("Met-A, " + cellname(x["cell"]), x.get("A"), x.get("ceiling"))
    for k, v in te.items():
        if v.get("A") is not None: _cl("Territory, " + k, v.get("A"), v.get("ceiling"))
    T.append(("v4-ceiling", "Distance to the ceiling",
              "How close each reading sits to the ceiling the physics sets for it: 1 is the healthy expectation, 0 the ceiling.",
              ["Reading", "Value", "Saturation", "Distance to the ceiling", "Turnover share", "Methylated sites past 0.5"], cl,
              (DEV + ".") if cl else "Not read: no atlas reading carries it on this sample."))
    c = o.get("met_a_cscore") or {}; ce = o.get("cscore_each_cell") or {}; wg = ce.get("whole_genome") or {}; q = o.get("iam_a") or {}
    cr = [["1. C-Score, commissioned (neutrophil identity sites)", _f(c.get("C_rel") if c.get("C_rel") is not None else c.get("C")), str(c.get("C_call") or c.get("reason") or ("not read" if c.get("C") is None else ""))[:120], "commissioned" if c.get("commissioning") else ("commissioned on whole blood; untared here (needs 3 same-run healthy references for the band)" if c.get("C") is not None and sp_ == "whole blood" else ("outside what it was commissioned on" if c.get("C") is not None else ""))],
          ["2. C-Score, whole genome (every atlas site)", _f(wg.get("C")), "" if wg.get("C") is not None else str(ce.get("reason") or "not read"), DEV]]
    each = [x for x in ce.get("cells", []) if x.get("C") is not None]
    cr.append(["3. C-Score, each cell", ("; ".join(f"{cellname(x['cell'])} {x['C']:.3f}" for x in sorted(each, key=lambda x: -(x.get("share") or 0))[:8])) or "-", f"{len(each)} cells read" if each else "not read", DEV])
    qc = q.get("cscore") or {}
    cr.append(["4. IAM-A C-Score, region by region (sequencing)", _f(qc.get("C")), "" if qc.get("C") is not None else ("needs single molecules: sequencing input" if not q else "not built yet"), DEV])
    T.append(("v4-cscores", "The four C-Scores: are the departures clustered along the genome?",
              "Each asks the same question of a different map: scattered departures read near 1; departures that run along the genome read higher.", ["C-Score", "Value", "Note", "Status"], cr, ""))
    T.append(("v4-iama", "IAM-A: is each mark still held, against the sample's own reference",
              "Single molecules only: the share of marks left unwritten between written neighbors, against what this sample's own loose share predicts. No other person is needed.",
              ["Reading", "Value", "State"], [["A", _f(q.get("A")), str(q.get("state") or "")]] if q.get("A") is not None else [],
              ("commissioned on healthy CD8 T cells" if q.get("commissioning") else DEV) + "." if q.get("A") is not None else ("Not read: " + str(q.get("refusal")) + "." if q else "Not run: this sample has no sequencing input (arrays carry no single molecules).")))
    ga = o.get("group_a") or {}
    T.append(("v4-groupa", "Group-A: this person's group against its healthy group",
              "Group-A reads a group, not one person: it compares a group's readings with a healthy group's, window by window, after every member is read by this same command. Commissioned on EPIC v1 whole blood.",
              ["Group", "Reading", "Status"], [[str(ga.get("group")), str(ga.get("reading")), str(ga.get("status") or "")]] if ga.get("reading") is not None else [],
              "" if ga.get("reading") is not None else "Not part of a single-sample report: when this sample is read as one member of a series (run_series.py), Group-A's result is in that series' summary."))
    g = o.get("globe") or {}
    T.append(("v4-globe-sum", "The globe: where on the genome it changed",
              "Every site keeps one place on the sphere, set by its position in the genome; the sample is laid over its healthy reference and only changed blocks of 10 neighboring sites light up.",
              ["Layer", "Changed blocks", "Blocks read"],
              [["whole genome", f"{g.get('blocks_changed', 0):,}", f"{g.get('blocks', 0):,}"]] + ([["Met-A's identity sites", f"{g.get('metaa_blocks_changed')}", f"{g.get('metaa_blocks')}"]] if g.get("metaa_blocks") else []) if g.get("drawn") else [],
              ("Reference: " + str(g.get("reference") or "") + ". " + DEV + ". The turning globe: " + os.path.basename(str(g.get("html") or "the globe page beside this report")) + ".") if g.get("drawn") else "Not drawn: " + str(g.get("reason") or "") + "."))
    sp = o.get("info_sphere") or {}
    T.append(("v4-sphere", "The information sphere: which cell programs moved",
              "Every site placed by how it behaves across the atlas's 79 cell types, so sites that work together sit together. The sample is read territory by territory against its own expected healthy self, with its distance from that self in the sphere's own measure (the Fisher-Rao angle).",
              ["Territories read", "Territories changed", "Distance (rad)"],
              [[str(sp.get("territories_read")), str(sp.get("territories_changed")), _f(sp.get("fisher_rao_rms_rad"))]] if sp.get("territories_read") else [],
              (DEV + ". Page: " + os.path.basename(str(sp.get("html") or "")) + ".") if sp.get("territories_read") else "Not drawn: " + str(sp.get("reason") or "") + "."))
    return T

REPORT_TEXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Runtime Matrices", "Report", "report_text.json")
EXPLAIN_LABELS = (("measures", "What it measures"), ("source", "Where it comes from"), ("compares", "What it is compared to"),
                  ("physics", "The physics"), ("significance", "Why it matters"), ("sky", "From the sky"), ("status", "Status"))
_RT = []
def report_text():
    """The report's explanations, generated from CANON/catalogue.json by CANON/report_text.py (one source; never written in this file)."""
    if not _RT: _RT.append(json.load(open(REPORT_TEXT, encoding="utf-8")))
    return _RT[0]
def explain(sec):
    """[(label, text)] for one report section, in the fixed order of EXPLAIN_LABELS; empty when the section has no entry."""
    x = report_text()["sections"].get(sec) or {}
    return [(lab, x[k]) for k, lab in EXPLAIN_LABELS if x.get(k)]
def _explain_html(sec):
    e = explain(sec)
    return ("<div class='explain'>" + "".join(f"<p><b>{E(l)}.</b> {E(t)}</p>" for l, t in e) + "</div>") if e else ""
def _intro_html():
    i = report_text()["intro"]
    return f"<h2 id='v4-intro'>{E(i['title'])}</h2>" + "".join(f"<p>{E(t)}</p>" for t in i["paragraphs"])

def _sections_v5(o):
    h = []
    for i, title, intro, head, rows, note in detail_tables(o):
        h.append(f"<h2 id='{i}'>{E(title)}</h2><p>{E(intro)}</p>")
        if rows: h.append("<table><tr>" + "".join(f"<th>{E(x)}</th>" for x in head) + "</tr>" + "".join("<tr>" + "".join(f"<td>{E(str(x))}</td>" for x in r) + "</tr>" for r in rows) + "</table>")
        if note: h.append(f"<p class='note'>{E(note)}</p>")
        h.append(_explain_html(i))
    return "".join(h)

def build(o, out, sid):
    G, GP = o.pop("_globe_html", "") or "", o.pop("_globe_prose", "") or ""   # taken out first: the technical record must not print the globe's data
    tech = os.path.splitext(out)[0] + "_v3record.tmp.html"
    r = R3.build(o, tech, sid); t = open(tech, encoding="utf-8").read(); os.remove(tech)
    body = re.search(r"<body[^>]*>(.*)</body>", t, re.S).group(1)
    when = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    o["_figs"] = F = _figs(o, os.path.splitext(os.path.abspath(out))[0])
    front = [f"<p class='meta'>IAM chain v4 · run {E(str(o.get('run_id')))} · {when}</p>", f"<h1>Sample report: {E(sid)}</h1>",
             "<p class='sub2'>AstroGenetics: the physics of methylation, read on one sample</p>",
             f"<p class='meta'>Specimen {E(str(o.get('specimen', '')))} · platform {E(str(o.get('platform') or o.get('array_type') or 'sequencing'))}</p>",
             (f"<p class='note' style='border-left-color:#b33;background:#fdf0f0'><b>{E(o['banner'])}</b></p>" if o.get("banner") else ""),
             "<p class='note'><b>Research use only. Not a diagnostic test.</b> Each reading is printed with its commissioning status: "
             "<i>commissioned</i> means it passed a test sealed before its data was seen; a reading outside what an instrument was tested on is still shown, and says so.</p>",
             "<div class='views'><input type='radio' name='v' id='vr' checked><input type='radio' name='v' id='vp'>"
             "<div class='vbar'><label for='vr'>Researcher</label><label for='vp'>Plain</label></div>",
             "<div class='research'>", "<h2 id='v4-glance'>At a glance</h2>", _glance(o), _intro_html(), _physics(o), _shares(o), _explain_html("v4-shares"), _sections_v5(o), _meta(o), _cscore(o), _iama(o), "</div>",
             "<div class='plain'>", _plain(o, F), "</div></div>"]
    prose = re.sub(r"<[^>]+>", " ", re.sub(r"src='data:[^']+'", "", re.sub(r"<!--GAUGE-->.*?<!--/GAUGE-->", " ", "\n".join(front), flags=re.S)))   # as report_v3: the gauge's fixed legend is a scale, not prose
    if re.search(CLAIM_SCAN, prose, re.I): raise ValueError("report_v4: front prose carries a word the report may not print: " + re.search(CLAIM_SCAN, prose, re.I).group(0))
    # the turning globe, inside the researcher view; its own prose (outside and inside its script's data) is scanned
    gp = re.sub(r"<[^>]+>", " ", re.sub(r"<script>.*?</script>", " ", G, flags=re.S)) + " " + GP
    if re.search(CLAIM_SCAN, gp, re.I): raise ValueError("report_v4: the globe's prose carries a word the report may not print: " + re.search(CLAIM_SCAN, gp, re.I).group(0))
    if G: front.insert(front.index("</div>"), G)
    H = (f"<html><head><meta charset='utf-8'><title>IAM chain v4 - {E(sid)}</title>{CSS}</head><body>" + "\n".join(front)
         + "<div class='tech'><h2 id='v4-technical'>Technical record</h2>" + body + "</div></body></html>")
    open(out, "w", encoding="utf-8").write(H); o.pop("_figs", None)
    pdfs = {}
    try:
        import report_pdf_v4 as RP
        pdfs = RP.build_pdfs(o, F, out, sid, when)
    except Exception as e: pdfs = {"error": f"{type(e).__name__}: {e}"[:200]}
    return {**r, "out": out, "bytes": len(H), "pdfs": pdfs}
