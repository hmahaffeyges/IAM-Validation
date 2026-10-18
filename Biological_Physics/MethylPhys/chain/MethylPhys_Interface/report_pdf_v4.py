"""The chain v4 report as two PDFs (2026-10-18): <report>_researcher.pdf and <report>_plain.pdf, built from the same rows and pictures as the
HTML page (report_v4.glance_rows, physics_rows, the gauge and sky PNGs), so the formats cannot disagree. Unicode text in DejaVu Sans (shipped
with matplotlib). Research use only; not a diagnostic test."""
import os, html, re
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
_FD = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for nm, f in (("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf"), ("DVI", "DejaVuSans-Oblique.ttf")):
    if nm not in pdfmetrics.getRegisteredFontNames(): pdfmetrics.registerFont(TTFont(nm, os.path.join(_FD, f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")
INK = colors.HexColor("#1d2330"); SUB = colors.HexColor("#556070")
S = {k: ParagraphStyle(k, fontName="DV", **v) for k, v in dict(
    h1=dict(fontSize=18, leading=22, textColor=INK, spaceAfter=2), sub=dict(fontSize=10.5, leading=14, textColor=colors.HexColor("#3a5a9b")),
    meta=dict(fontSize=8.5, leading=11, textColor=SUB), h2=dict(fontSize=12.5, leading=16, textColor=INK, spaceBefore=12, spaceAfter=5),
    body=dict(fontSize=9, leading=12.2, textColor=INK), small=dict(fontSize=7.8, leading=10.2, textColor=SUB),
    note=dict(fontSize=8.5, leading=11.5, textColor=INK, backColor=colors.HexColor("#f4f6fa"), borderPadding=5, leftIndent=4, rightIndent=4)).items()}

def P(t, st="body"):
    t = html.unescape(str(t)).replace("<sup>", "<super>").replace("</sup>", "</super>")
    t = re.sub(r"<(?!/?(b|i|sub|super|br/?)\b)[^>]+>", "", t)   # only the tags reportlab understands
    return Paragraph(t, S[st])

def _img(path, w=7.0):
    if not path or not os.path.exists(path): return None
    from PIL import Image as PI
    W, H = PI.open(path).size; return Image(path, width=w * inch, height=w * inch * H / W)

def _head(o, sid, when, view):
    return [P(f"IAM chain v4 · run {o.get('run_id')} · {when} · {view} view", "meta"), P(f"Sample report: {sid}", "h1"),
            P("AstroGenetics: the physics of methylation, read on one sample", "sub"),
            P(f"Specimen {o.get('specimen', '')} · platform {o.get('platform') or o.get('array_type') or 'sequencing'}", "meta"), Spacer(1, 6),
            P("<b>Research use only. Not a diagnostic test.</b> Each reading is printed with its commissioning status: <i>commissioned</i> means it "
              "passed a test sealed before its data was seen; a reading outside what an instrument was tested on is still shown, and says so.", "note"), Spacer(1, 4)]

def _tbl(rows, widths, head=True):
    t = Table(rows, colWidths=[w * inch for w in widths], repeatRows=1 if head else 0)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#dde2ea")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f4f6fa") if head else colors.white),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)])); return t

def build_pdfs(o, F, out, sid, when):
    import report_v4 as R
    base = os.path.splitext(os.path.abspath(out))[0]; res = {}
    def doc(path, story):
        SimpleDocTemplate(path, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                          title=f"IAM sample report {sid}", author="IAM chain v4").build(story); return path
    # researcher
    st = _head(o, sid, when, "Researcher") + [P("At a glance", "h2")]
    st.append(_tbl([[P("<b>Instrument</b>"), P("<b>Question it answers</b>"), P("<b>Reading</b>"), P("<b>State</b>"), P("<b>Status</b>")]]
                   + [[P(f"<b>{a}</b>"), P(b), P(f"<b>{v}</b>"), P(s_, "small"), P(x, "small")] for a, b, v, s_, x in R.glance_rows(o)], [1.2, 1.5, 1.25, 1.65, 1.7]))
    ph = R.physics_rows(o)
    if ph:
        st += [P("AstroGenetics: the physics behind each reading", "h2"), P(R.PHYS_INTRO), Spacer(1, 4),
               _tbl([[P("<b>Instrument</b>"), P("<b>What was measured</b>"), P("<b>The physics</b>")]] + [[P(f"<b>{a}</b>"), P(b), P(c)] for a, b, c in ph], [0.8, 3.1, 3.4])]
    sh = o.get("cell_shares") or {}
    if sh.get("cells"):
        st += [P("Cell shares: IAM-Deconvolver against IAM-Atlas v2.1", "h2"),
               _tbl([[P("<b>Cell</b>"), P("<b>Share</b>"), P("<b>Range (information)</b>"), P("<b>Status</b>")]]
                    + [[P(R.cellname(r["cell"])), P(f"<b>{100 * r['share']:.1f} %</b>"), P(f"{100 * r['interval_95'][0]:.1f} to {100 * r['interval_95'][1]:.1f} %"),
                        P(r["status"] if r["status"] != "development" else R.untested(r["cell"], "the IAM-Deconvolver"), "small")]
                       for r in sorted(sh["cells"], key=lambda r: -r["share"])], [1.9, 1.0, 1.7, 2.7])]
    for k, t_ in (("met_a", "Met-A on its gauge"), ("iam_a", "IAM-A on its gauge"), ("cscore", "C-Score on its bar")):
        i = _img(F.get(k))
        if i: st += [KeepTogether([P(t_, "h2"), i])]
    k, a = o.get("cscore_sky") or {}, o.get("atlas_sky") or {}
    if k.get("drawn"):
        pair = [[_img(k["png"], 3.45), _img(a.get("png"), 3.45) if a.get("drawn") else P("")],
                [P("This sample: where each identity site departs from healthy neutrophils (z).", "small"),
                 P(f"Healthy neutrophils from IAM-Atlas v2.1 at the same {a.get('sites', 0):,} sites, on the same pixels." if a.get("drawn") else "", "small")]]
        st += [KeepTogether([P("The sample's sky beside the healthy sky", "h2"), _tbl(pair, [3.55, 3.55], head=False),
                             P("Each site keeps one place on a HEALPix sphere, set by its position in the genome, so the two pictures line up site for site. "
                               "Healthy departures look like speckle; clustered departures show as patches. Pictures of the readings; nothing is scored from them.", "small")])]
    st += [Spacer(1, 8), P("The full technical record (intake, calibration, tare, noise gate, safeguards, integrity, and how to run it yourself) is in the HTML report.", "small")]
    res["researcher"] = doc(base + "_researcher.pdf", st)
    # plain
    st = _head(o, sid, when, "Plain") + [P("The readings, in plain words", "h2"),
          P("Each gauge runs from the floor, below which a cell can no longer hold its identity, to the ceiling, where every mark is a coin flip. "
            "Healthy cells sit in the green band. Moving right means more writing than healthy.")]
    if sh.get("cells"):
        top = sorted(sh["cells"], key=lambda r: -r["share"])[:4]
        st += [P("Which cells are in the sample", "h2"), P(", ".join(f"{R.cellname(r['cell'])} {100 * r['share']:.0f} %" for r in top) + ", read against the 79-cell IAM-Atlas v2.1.")]
    for key, name, say in (("met_a", "Met-A", "How far the neutrophils' tagging level sits from healthy."),
                           ("iam_a", "IAM-A", "Whether each mark is still held, read against this sample's own reference: no other person is needed."),
                           ("cscore", "C-Score", "Whether the changes are scattered, as in healthy cells, or clustered along the genome.")):
        i = _img(F.get(key))
        if i: st += [KeepTogether([P(name, "h2"), i, P(say)])]
        elif key == "met_a" and str((o.get("met_a") or {}).get("state", "")).startswith("withheld"):
            st += [P("Met-A", "h2"), P("No reading on this sample: the array's own noise check could not confirm it was clean enough to read, so Met-A held back rather than guess.")]
    if k.get("drawn"): st += [KeepTogether([P("The sample on the sphere", "h2"), _img(k["png"], 5.4),
                              P("Each dot of color is one place in the genome. Speckle is what healthy looks like; patches mean the changes are bunched together.")])]
    res["plain"] = doc(base + "_plain.pdf", st)
    return res
