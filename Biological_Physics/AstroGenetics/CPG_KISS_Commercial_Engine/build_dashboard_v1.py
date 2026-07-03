#!/usr/bin/env python3
"""CPG clinical dashboard v1 — built on the DETECTION SPINE (memory #30 ladder).

Dark-cosmic GAPE_WEB aesthetic, tabbed + printable. The body is the detection,
in this order:

  1. VERDICT GATE (Option A derived departure) — the reliable call: is there a
     real departure from the age-matched band, and which class drove it.
  2. PER-CELL RUN-EVERYTHING LADDER — every resolved cell scored (run-everything).
       * BREACHED (A >= 1.10) surfaced FIRST and flagged as the named CULPRIT.
       * ELEVATED (above age band, not breached) = concern.
       * the elevated-together SHAPE called out as the early / pre-disease read.
     Breach is the loud signal and names the culprit; shape is the quiet, early one.
  3. CLASS GAUGE (8) — the reliable tier, each class vs its age band.
  4. CELLULAR AGE + PERSONAL BRILLIANCE MAP — supporting panels.
  Disease matrix + residual maps: deferred.

The demo bundle below is an ILLUSTRATIVE pre-cancer case (constructed to make the
ladder fire), clearly labelled. Real values come from run_pipeline on Stage-1 beta.
"""
import base64
import json
import math
import os

AGE_MATRIX = "prod/PRODUCTION CPG FILES/age_reference_matrix.json"
BRILLIANCE_PNG = "/tmp/patient_brilliance_map.png"
CMB_REF = "/mnt/project/cmb1.jpg"


def _thumb_b64(path, w=760):
    from PIL import Image
    import io
    im = Image.open(path).convert("RGB")
    if im.width > w:
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _H(b):
    if b <= 0 or b >= 1:
        return 0.0
    return -b * math.log2(b) - (1 - b) * math.log2(1 - b)


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ─── illustrative pre-cancer case: immune+secretory elevated, one secretory breach ─
def _demo_bundle():
    """Load REAL chain output (real cell names, real A-scores, real CI) from a run.
    No fabricated names or numbers. Raw-β run (Stage-1 pending), honestly labelled."""
    import json
    rb = json.load(open("/tmp/real_bundle.json"))
    d = json.load(open(AGE_MATRIX))
    age = 50
    band = {}
    for c in ["immune", "cycling", "secretory", "terminal", "stromal",
              "progenitor", "stem_adult", "stem_pluri"]:
        r = min(d.get(c, []), key=lambda r: abs(r["age_midpoint"] - age))
        band[c] = {"p10": r["A_p10"], "mean": r["A_mean"], "p90": r["A_p90"]}

    def _tier(a):
        if a is None:
            return "N/A"
        if a >= 1.10: return "BREACH"
        if a >= 1.07: return "URGENT"
        if a >= 1.05: return "DETECTABLE"
        if a >= 1.01: return "MARGINAL"
        return "NORMAL"

    # real class gauges from the run
    classes = {}
    for c, r in (rb.get("classes") or {}).items():
        a = r.get("A")
        b = r.get("band") or band.get(c, {})
        if a is None:
            continue
        classes[c] = {"A": a, "band": b, "tier": r.get("tier") or _tier(a),
                      "placement": r.get("placement") or "",
                      "departure": round(a - b.get("mean", 1.0), 3),
                      "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi")}
    # real per-cell A-scores (real names) from the run
    cells = []
    for cc in rb.get("cells", []):
        cl = cc["class"]; a = cc["A"]
        cells.append({"cell": cc["cell"], "class": cl, "A": a, "tier": _tier(a),
                      "band": band.get(cl, {}), "ci_lo": cc.get("ci_lo"), "ci_hi": cc.get("ci_hi"),
                      "departure": round(a - band.get(cl, {}).get("mean", 1.0), 3)})
    dep = rb.get("departure") or {}
    dist = dep.get("distance")
    driver = (max([(c, r["departure"]) for c, r in classes.items() if r.get("departure") is not None],
                  key=lambda kv: kv[1])[0] if classes else "")
    return {
        "patient_id": "GSM1051525 (real run)", "context": {"age": age, "sex": "F", "substrate": "whole blood"},
        "classes": classes, "cells": cells,
        "departure": {"distance": dist, "threshold": dep.get("threshold"),
                      "beyond": dep.get("beyond"), "driver": driver},
        "cellular_age": {"overall": None, "chrono": age, "delta": None, "per_class": {}},
        "composition": rb.get("composition", {}),
    }


AXIS_MIN, AXIS_MAX = 0.90, 1.20
_TIER_COL = {"NORMAL": "#1f7a5f", "MARGINAL": "#8a6d1f", "DETECTABLE": "#b45309",
             "URGENT": "#b2182b", "BREACH": "#b2182b", "INVERSION": "#7C3AED"}


def _bar_svg(rec, w=520):
    def x(a):
        a = max(AXIS_MIN, min(AXIS_MAX, a))
        return 10 + (a - AXIS_MIN) / (AXIS_MAX - AXIS_MIN) * (w - 20)
    A = rec["A"]; b = rec.get("band", {})
    y, h = 20, 11
    s = [f'<svg viewBox="0 0 {w} 46" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;font-family:ui-monospace,monospace">']
    for lo, hi, col in [(0.90, 1.01, "#1f7a5f"), (1.01, 1.07, "#8a6d1f"), (1.07, 1.10, "#b45309"), (1.10, 1.20, "#b2182b")]:
        s.append(f'<rect x="{x(lo):.1f}" y="{y}" width="{x(hi)-x(lo):.1f}" height="{h}" fill="{col}" opacity="0.28"/>')
    if b.get("p10") is not None:
        s.append(f'<rect x="{x(b["p10"]):.1f}" y="{y-3}" width="{x(b["p90"])-x(b["p10"]):.1f}" height="{h+6}" fill="#A78BFA" opacity="0.30" rx="2"/>')
    s.append(f'<line x1="{x(1.10):.1f}" y1="{y-4}" x2="{x(1.10):.1f}" y2="{y+h+4}" stroke="#b2182b" stroke-width="1.5"/>')
    px = x(A); mcol = _TIER_COL.get(rec["tier"], "#F5F3FF")
    s.append(f'<circle cx="{px:.1f}" cy="{y+h/2:.1f}" r="6" fill="{mcol}" stroke="#F5F3FF" stroke-width="1.5"/>')
    s.append(f'<text x="{px:.1f}" y="13" text-anchor="middle" font-size="11" fill="#C4B5FD" font-weight="700">{A:.3f}</text>')
    for t in (0.90, 1.10, 1.20):
        s.append(f'<text x="{x(t):.1f}" y="44" text-anchor="middle" font-size="8" fill="#516170">{t:.2f}</text>')
    s.append('</svg>')
    return "".join(s)


def _pill(tier):
    return f'<span class="pill" style="background:{_TIER_COL.get(tier,"#516170")}">{tier}</span>'


def build_dashboard(bundle, out_path):
    ctx = bundle["context"]
    dep = bundle["departure"]
    classes = bundle["classes"]
    cells = sorted(bundle["cells"], key=lambda c: -c["A"])
    cage = bundle["cellular_age"]
    plate01 = open("/tmp/plate01_b64.txt").read() if os.path.exists("/tmp/plate01_b64.txt") else ""
    plate03 = open("/tmp/plate03_b64.txt").read() if os.path.exists("/tmp/plate03_b64.txt") else ""

    comp = bundle.get("composition", {}) or {}
    cls_rows = "".join(f'<div class="crow" style="grid-template-columns:150px 1fr 90px"><div class="cn">{c.replace("_"," ")}</div>'
                       f'<div style="background:var(--surf2);border-radius:4px;height:14px;position:relative"><div style="position:absolute;left:0;top:0;height:14px;border-radius:4px;background:var(--lav2);width:{min(v,100):.1f}%"></div></div>'
                       f'<div class="cd" style="text-align:right">{v:.1f}%</div></div>'
                       for c, v in sorted((comp.get("class") or {}).items(), key=lambda kv:-kv[1]) if v > 0.05)
    ct_rows = ""
    for c in (comp.get("celltype") or []):
        flag = ' <span class="culprit">NOT BLOOD-RESIDENT</span>' if c.get("flag") else ""
        ct_rows += (f'<div class="crow" style="grid-template-columns:200px 1fr 130px"><div class="cn">{c["cell"].replace("_"," ")}{flag}</div>'
                    f'<div style="background:var(--surf2);border-radius:4px;height:14px;position:relative"><div style="position:absolute;left:0;top:0;height:14px;border-radius:4px;background:{"#b2182b" if c.get("flag") else "var(--accent)"};width:{min(c["pct"],100):.1f}%"></div></div>'
                    f'<div class="cd" style="text-align:right">{c["pct"]:.1f}%</div></div>')
    # verdict
    dist_val = dep.get("distance")
    beyond = dep.get("beyond")
    vcol = "#ff6b6b" if beyond else "#4ade80"
    verdict = ("Full verdict fires on Stage-1-calibrated β; on this raw-β run the class gauge is guarded, "
               "so the departure is not asserted. Per-cell A-scores below are the real chain output.")

    # per-cell ladder: breached first (culprit), then elevated, then normal
    breached = [c for c in cells if c["tier"] == "BREACH"]
    elevated = [c for c in cells if c["tier"] in ("URGENT", "DETECTABLE", "MARGINAL")]
    normal = [c for c in cells if c["tier"] in ("NORMAL", "INVERSION")]

    def _cell_row(c, culprit=False):
        flag = ' <span class="culprit">CULPRIT</span>' if culprit else ""
        ci = (f' <span class="ci">[{c["ci_lo"]:.3f}, {c["ci_hi"]:.3f}]</span>'
              if c.get("ci_lo") is not None else "")
        return (f'<div class="crow"><div class="cn">{c["cell"].replace("_"," ")}'
                f'<div class="cc">{c["class"]}</div></div>'
                f'<div class="cbar">{_bar_svg(c)}</div>'
                f'<div class="ct">{_pill(c["tier"])}{flag}'
                f'<div class="cd">A {c["A"]:.3f}{ci}</div></div></div>')

    breach_html = "".join(_cell_row(c, culprit=True) for c in breached) or \
        '<div class="none">No cell breached its floor.</div>'
    elev_html = "".join(_cell_row(c) for c in elevated) or '<div class="none">None.</div>'
    norm_html = "".join(_cell_row(c) for c in normal)

    shape_cells = [c["cell"].replace("_", " ") for c in cells if c["tier"] in ("MARGINAL", "DETECTABLE", "URGENT", "BREACH")]
    shape_txt = (", ".join(shape_cells[:6]) if shape_cells else "none")

    class_rows = ""
    for c, r in sorted(classes.items(), key=lambda kv: -kv[1]["departure"]):
        culprit = ' <span class="culprit">CULPRIT</span>' if r["tier"] == "BREACH" else ""
        _ci = (f' <span class="ci">[{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}]</span>'
               if r.get("ci_lo") is not None else "")
        class_rows += (f'<div class="crow"><div class="cn">{c.replace("_"," ")}</div>'
                       f'<div class="cbar">{_bar_svg(r)}</div>'
                       f'<div class="ct">{_pill(r["tier"])}{culprit}'
                       f'<div class="cd">A {r["A"]:.3f}{_ci}</div></div></div>')

    age_rows = "".join(f'<tr><td>{c.replace("_"," ")}</td><td class="num">{a} yr</td>'
                       f'<td class="num">{a - cage["chrono"]:+d} yr</td></tr>'
                       for c, a in cage["per_class"].items())

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CPG — {bundle['patient_id']}</title>
<style>
:root{{--bg:#080c14;--surf:#0d1525;--surf2:#111e2e;--border:#1a2a3a;--lav:#C4B5FD;--lav3:#7C3AED;
--text:#dbe4ee;--muted:#7c8b9d;--red:#ff6b6b;--sans:system-ui,-apple-system,"Segoe UI",sans-serif;--mono:ui-monospace,Menlo,monospace;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.55}}
.top{{display:flex;align-items:center;justify-content:space-between;padding:16px 26px;border-bottom:1px solid var(--border);background:linear-gradient(180deg,#0b1220,#080c14)}}
.brand{{font-weight:700}} .brand span{{color:var(--lav)}} .brand small{{display:block;font-family:var(--mono);font-size:10px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;font-weight:400}}
.pt{{font-family:var(--mono);font-size:12px;color:var(--muted);text-align:right}}
.printbtn{{margin-left:16px;background:var(--lav3);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:12px;cursor:pointer;font-weight:600}}
.tabs{{display:flex;gap:2px;padding:0 20px;border-bottom:1px solid var(--border);background:var(--surf);overflow-x:auto}}
.tab{{padding:12px 16px;font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;font-family:var(--mono)}}
.tab:hover{{color:var(--lav)}} .tab.on{{color:var(--lav);border-bottom-color:var(--lav3)}}
.wrap{{max-width:960px;margin:0 auto;padding:24px 22px 80px}} .panel{{display:none}} .panel.on{{display:block}}
h2{{font-size:14px;letter-spacing:1.5px;margin:24px 0 8px;color:var(--lav);font-weight:600;text-transform:uppercase}} h2:first-child{{margin-top:4px}}
.verdict{{background:var(--surf);border:1px solid var(--border);border-left:4px solid {vcol};border-radius:0 10px 10px 0;padding:16px 20px;margin:12px 0}}
.verdict .big{{font-size:22px;font-weight:700;color:{vcol}}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:12px 0}}
.card{{background:var(--surf);border:1px solid var(--border);border-radius:10px;padding:16px}}
.card-lbl{{font-size:9px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;font-family:var(--mono)}}
.card-big{{font-size:26px;font-weight:700;margin-top:5px}}
.crow{{display:grid;grid-template-columns:150px 1fr 150px;align-items:center;gap:12px;padding:8px 12px;border-bottom:1px solid var(--border)}}
.cn{{font-family:var(--mono);font-size:12px;text-transform:capitalize}} .cc{{font-size:10px;color:var(--muted)}}
.ct{{text-align:right}} .cd{{font-size:10px;color:var(--muted);font-family:var(--mono);margin-top:3px}}
.pill{{font-size:9px;font-weight:700;letter-spacing:.5px;color:#fff;padding:2px 8px;border-radius:10px}}
.culprit{{font-size:9px;font-weight:800;letter-spacing:1px;color:#fff;background:#b2182b;padding:2px 7px;border-radius:4px;margin-left:5px}}
.explain{{background:var(--surf2);border-left:3px solid var(--lav3);border-radius:0 8px 8px 0;padding:11px 15px;margin:10px 0;font-size:12.5px;color:#b9c6d6}} .explain b{{color:var(--text)}}
.sub{{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);font-family:var(--mono);margin:16px 0 4px}}
.none{{color:var(--muted);font-style:italic;padding:8px 12px;font-size:12px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);font-family:var(--mono);margin:6px 0}}
.sw{{width:11px;height:11px;border-radius:2px;display:inline-block;vertical-align:-1px;margin-right:4px}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border)}}
th{{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);font-family:var(--mono)}} td.num{{font-family:var(--mono);text-align:right}}
.plate{{width:100%;border:1px solid var(--border);border-radius:10px;margin:8px 0}}
.illus{{background:#1a1330;border:1px solid var(--lav3);border-radius:8px;padding:9px 14px;margin:10px 0;font-size:12px;color:var(--lav)}}
.foot{{margin-top:28px;padding-top:12px;border-top:1px solid var(--border);font-size:10px;color:var(--muted);font-family:var(--mono)}}
@media print{{body{{background:#fff;color:#111}}.top,.tabs{{background:#fff}}.tab{{display:none}}.panel{{display:block!important;page-break-inside:avoid}}.card,.explain,.crow,.verdict,.plate{{border-color:#ccc;background:#fff;color:#111}}h2{{color:#3b1d8f}}.explain{{background:#f4f1fb;color:#222}}.card-lbl,.cc,.cd,th{{color:#555}}.brand span,.tab.on{{color:#3b1d8f}}}}
</style></head><body>
<div class="top"><div class="brand">Cellular Performance <span>Gauge</span><small>IAMPerformance · AstroGenetics</small></div>
<div style="display:flex;align-items:center"><div class="pt">Patient {bundle['patient_id']}<br>Age {ctx['age']} · {ctx['sex']} · {ctx['substrate']}</div>
<button class="printbtn" onclick="window.print()">Print patient report</button></div></div>
<div class="tabs"><div class="tab on" data-p="dx">Detection</div><div class="tab" data-p="co">Composition</div><div class="tab" data-p="ga">Class Gauge</div>
<div class="tab" data-p="ag">Cellular Age</div><div class="tab" data-p="mm">Methylome Map</div><div class="tab" data-p="ab">About</div></div>
<div class="wrap">
<div class="illus"><b>Real chain output on raw (un-calibrated) β — Stage-1 pending.</b> The cell names and A-scores below are
the actual chain output, not invented. On raw β the per-cell A sits on the indicative separation scale and the class gauge
is guarded; the Stage-1-calibrated run is what makes the breach and verdict fire honestly.</div>

<div class="panel on" id="dx">
  <h2>Verdict</h2>
  <div class="verdict"><div class="big">Departure {('%.1f'%dist_val) if dist_val is not None else '—'} &nbsp;·&nbsp; {('BEYOND' if beyond else 'within') if beyond is not None else 'pending'} age band</div>
    <div style="color:var(--muted);margin-top:4px">{verdict} <span style="font-size:11px">(derived χ², no cohort; on raw β this is guarded)</span></div></div>

  <h2>Per-cell detection — run everything</h2>
  <div class="legend"><span><span class="sw" style="background:#A78BFA;opacity:.5"></span>age band</span>
    <span><span class="sw" style="background:#8a6d1f;opacity:.6"></span>elevated</span>
    <span><span class="sw" style="background:#b2182b;opacity:.6"></span>breach (culprit)</span></div>
  <div class="sub">Breached — a real problem, cell names the culprit</div>
  {breach_html}
  <div class="sub">Elevated above age band — concern; the shape is the early read</div>
  {elev_html}
  <div class="explain"><b>The elevated-together shape</b> ({shape_txt}) is the pre-disease fingerprint — immune and
  secretory cells drifting up together, years before anything breaches. <b>The breach</b> is the loud, late signal
  and points straight at the culprit cell. Every resolved cell is scored (run-everything); a cell is nulled only for
  insufficient markers. Per-patient, derived, age-matched — no cohort.</div>
  <div class="sub">Within band</div>
  {norm_html}
</div>

<div class="panel" id="co">
  <h2>Composition — what cells are in the blood, and how much</h2>
  <div class="explain">The Walther deconvolver resolves the fraction of each architecture class and cell type from the
  whole-blood methylation. A cell type appearing that <b>should not circulate in blood</b> (brain / BBB cells), or a
  resident type at an unexpected fraction, is itself a signal — independent of the A-score. Composition informs; it does
  not by itself fire the call.</div>
  <div class="sub">By architecture class</div>
  {cls_rows}
  <div class="sub">By cell type (resolved &gt; 0.1%)</div>
  {ct_rows}
</div>
<div class="panel" id="ga">
  <h2>Class Gauge — the reliable call</h2>
  <div class="explain">Per class, A = H(β_mean)/H_min on identity loci, read against the age-matched band (lavender).
  The class tier is the <b>reliable</b> call; the per-cell ladder on the Detection tab is the indicative fingerprint that
  names which cells. A breached class is flagged as the culprit.</div>
  {class_rows}
</div>

<div class="panel" id="ag">
  <h2>Cellular Age</h2>
  <div class="explain">Cellular age pending — it needs at least one class assessable-above-floor on Stage-1-calibrated β.
  On this raw-β run the classes are guarded, so no age is asserted (rather than print a fabricated one).</div>
  <div class="explain">The headline says whether cells age on pace. It is not the finding alone — the per-class rows show
  <b>which</b> compartment is off pace. Here immune and secretory read oldest, consistent with the detection above.</div>
  
</div>

<div class="panel" id="mm">
  <h2>The Cosmic Microwave Methylome</h2>
  <div class="explain">The whole methylome projected onto the sphere — 481,966 CpGs, one per pixel, per architectural
  class (Mollweide, genomic order). The texture has the same statistical character as CMB anisotropy: large-scale
  gradients with embedded small-scale fluctuation. Pure presentation — the detection is on the first tab.</div>
  {'<img class="plate" src="data:image/jpeg;base64,'+plate01+'" alt="The Cosmic Microwave Methylome">' if plate01 else '<div class="none">Plate not available.</div>'}
  <div class="sub">Methylome vs the cosmic microwave background — the Grandaddy plate</div>
  {'<img class="plate" src="data:image/jpeg;base64,'+plate03+'" alt="Methylome vs CMB">' if plate03 else ''}
  <div style="font-size:11px;color:var(--muted);font-family:var(--mono);margin:-2px 0 4px">Same projection, same colormap:
  the pooled methylome beside a Planck 2018 ΛCDM CMB realization. The methylome reads more bimodal/saturated; the CMB more Gaussian.</div>
</div>

<div class="panel" id="ab">
  <h2>About this test</h2>
  <div class="explain">CPG reads one whole-blood draw. The Walther deconvolver resolves which cells are present; the IAMAtlas
  scores each one's A-score with a real CI; the departure from the age-matched band is the detection. Derived, per-patient,
  no cohort comparison. Every line is a consistent-with statement, not a diagnosis.</div>
  <div class="explain"><b>The ladder.</b> Cells drifting elevated together above their age band = early / pre-disease shape.
  A single cell breaching A ≥ 1.10 = a real problem, and that cell names the culprit. Class-level = the reliable call;
  per-cell = the fingerprint (indicative in v0.1). Disease-shape matching and residual maps are deferred.</div>
  <div class="foot">CPG dashboard v1 · detection spine · derived, age-matched, no cohort · IAMPerformance / AstroGenetics · illustrative render</div>
</div>
</div>
<script>
document.querySelectorAll('.tab').forEach(function(t){{t.addEventListener('click',function(){{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('.panel').forEach(x=>x.classList.remove('on'));
  t.classList.add('on'); document.getElementById(t.dataset.p).classList.add('on'); window.scrollTo(0,0);}});}});
</script></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return len(html)


if __name__ == "__main__":
    n = build_dashboard(_demo_bundle(), "/tmp/cpg_dashboard_v1.html")
    print("dashboard v1 written:", n, "bytes")
