import json, html
D=json.load(open("/home/claude/patient_wall_data.json"))
sec_cols=D["sec_cols"]; sections=D["sections_used"]; pc=D["patient_cells"]; stress=D["stress"]
SECLABEL={"lymphoid":"Lymphoid","myeloid":"Myeloid","progenitor":"Progenitor","cycling":"Cycling",
 "secretory":"Secretory","terminal":"Terminal","stromal":"Stromal","stem":"Stem"}
SECACCENT={"lymphoid":"#8b83e6","myeloid":"#e0a23a","progenitor":"#5fb0a8","cycling":"#c77fd6",
 "secretory":"#5a9bd4","terminal":"#d98a6a","stromal":"#8aa86a","stem":"#b0926a"}
BBB={"cortical_neurons","neurons_pooled","glia","astrocytes","brain_pooled","oligodendrocytes","OPC","microglia","NeuMa","NeuIm"}
def short(c):
    s=c.replace("_cells","").replace("_pooled","\u00b7p").replace("regulatory_T","Treg").replace("naive_","n").replace("memory_","m").replace("_T","T").replace("_B","B")
    return s if len(s)<=14 else s[:13]+"\u2026"
ordered=[c for s in sections for c in sec_cols[s]]

TIER_BG={"SUPPRESSED":"#4a72a8","NORMAL":"#2f7a4f","ELEVATED":"#b8923a",
         "SIGNIFICANTLY_ELEVATED":"#cc6f2e","BREACH":"#c4493d"}
TIER_NAME={"SUPPRESSED":"suppressed (A<0.95)","NORMAL":"normal / healthy (0.95-1.04)",
           "ELEVATED":"elevated (1.04-1.07)","SIGNIFICANTLY_ELEVATED":"past Warburg line (1.07-1.10)",
           "BREACH":"breach (A>=1.10)"}
def patient_cell(col):
    info=pc.get(col)
    if not info: return '<td class="blank"></td>'
    tier=info["tier"]; A=info["A"]; dep=info["dep"]; conf=info["confident"]; ci=info["ci"]
    bg=TIER_BG[tier]
    cls="pcell" + ("" if (conf or tier=="NORMAL") else " unc")
    if conf and tier!="NORMAL": cls+=" conf"
    citxt=(f" CI[{ci[0]},{ci[1]}]" if ci else "")
    tip=html.escape(f"A={A} ({TIER_NAME[tier]}){citxt}")
    return f'<td class="{cls}" style="background:{bg}" title="{tip}">{A:.2f}</td>'

# disease (crown jewel) cell — cohort d
def disease_cell(row,col):
    v=row["cells"].get(col)
    if v is None: return '<td class="blank"></td>'
    d=v.get("d"); arr=v.get("arr")
    if d is None and arr:
        c="#d6584e" if arr=="up" else "#4f86d6"; sym="\u25b2" if arr=="up" else "\u25bc"
        return f'<td class="q" style="color:{c}">{sym}</td>'
    a=min(abs(d)/2.0,1.0)*0.8+0.18
    bg=f"rgba(214,88,78,{a:.2f})" if d>=0 else f"rgba(79,134,214,{a:.2f})"
    return f'<td style="background:{bg}">{d:+.2f}</td>'

sec_band="".join(f'<th class="secband" colspan="{len(sec_cols[s])}" style="color:{SECACCENT[s]};border-bottom:2px solid {SECACCENT[s]}">{SECLABEL[s]}</th>' for s in sections if sec_cols[s])
cell_head="".join(f'<th class="cellh{" bbb" if c in BBB else ""}" title="{html.escape(c)}">{html.escape(short(c))}</th>' for s in sections for c in sec_cols[s])

# patient row
patient_row=f'<tr class="prow"><td class="rowlabel pl"><div class="dz">YOUR BLOODWORK</div><div class="ph">{html.escape(D["patient_id"])} &middot; {html.escape(D["substrate"])} &middot; {D["age"]}{html.escape(D["sex"])}</div></td>{"".join(patient_cell(c) for c in ordered)}</tr>'

# flagged disease rows (order by phase: long_pre_dx first)
PHASE_ORD=["long_pre_dx","mid_pre_dx","mid_late_pre_dx","near_dx","at_dx","long_pre_dx_post_build_v3_0","tumor_tissue"]
frows=sorted(D["flagged_rows"], key=lambda r: PHASE_ORD.index(r["phase"]) if r["phase"] in PHASE_ORD else 99)
disease_rows_html=[]
for r in frows:
    dname=r["disease"].replace("_"," ").replace(" cancer","").title()
    ph=r["phase"].replace("_"," "); tr=r.get("time_range") or ""
    disease_rows_html.append(f'<tr class="drow"><td class="rowlabel"><div class="dz2">{html.escape(dname)}</div><div class="ph">{html.escape(ph)}{" \u00b7 "+html.escape(tr) if tr else ""}</div></td>{"".join(disease_cell(r,c) for c in ordered)}</tr>')

# stress banner
lvl=stress["level"]; lvlcol={"NOTABLE":"#b7791f","MILD":"#caa46a","NONE":"#5a8f6a"}[lvl]
stress_txt=("a clear" if lvl=="NOTABLE" else "a mild") if lvl!="NONE" else "no"
stress_banner=(f'<div class="stress" style="border-left:4px solid {lvlcol}"><b style="color:{lvlcol}">Wellness signal &mdash; systemic stress / inflammatory read:</b> '
  + (f'{stress_txt} systemic stress / inflammatory pattern ({stress["n_axis_cells"]} cells, mean departure {stress["mean_magnitude"]}). '
     'This names <b>no disease</b>. It is an actionable wellness signal &mdash; lifestyle, weight, diet, and trajectory monitoring, weighed more heavily with a family history of cancer. The point of seeing it early is the chance to change course.'
     if lvl!="NONE" else 'no coherent systemic stress pattern in the immune compartment.') + '</div>')

flagged_names=", ".join(html.escape(f["disease"].replace("_"," ")) + f' <span class="via">via {html.escape(f["via"])}</span>' for f in D["flagged"]) or "none"

HTML=f"""<!doctype html><meta charset="utf-8">
<style>
:root{{--bg:#100c08;--panel:#181109;--blank:#1e160d;--line:#2a1f12;--ink:#efe6d8;--muted:#b6a890;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:12px/1.3 ui-sans-serif,system-ui,sans-serif}}
.wrap{{padding:18px 20px 40px}} h1{{font-size:18px;margin:0 0 2px}}
.sub1{{color:var(--muted);font-size:12px;margin:0 0 12px;max-width:1100px}}
.stress{{background:var(--panel);border:1px solid var(--line);border-radius:0 8px 8px 0;padding:9px 14px;margin:0 0 10px;font-size:12px;max-width:1150px}}
.flag{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 14px;margin:0 0 12px;font-size:12px}}
.via{{color:#8a7c64;font-size:10.5px}}
.legend{{display:flex;flex-wrap:wrap;gap:13px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 14px;margin-bottom:12px;font-size:11px}}
.sw{{display:inline-block;width:13px;height:13px;border-radius:3px;vertical-align:-2px;margin-right:5px}}
.tablewrap{{overflow:auto;border:1px solid var(--line);border-radius:12px}}
table{{border-collapse:separate;border-spacing:0;font-variant-numeric:tabular-nums}}
th,td{{padding:0;text-align:center}}
td{{width:40px;min-width:40px;height:32px;border-right:1px solid #00000040;border-bottom:1px solid #00000040;color:#fff;font-size:10px;font-weight:600}}
td.blank{{background:var(--blank)}} td.q{{background:var(--blank);font-size:11px}}
td.green{{background:#27583f;color:#bfe6cf}}
td.pcell.unc{{opacity:.5;background-image:repeating-linear-gradient(45deg,#0000,#0000 3px,#ffffff22 3px,#ffffff22 5px)}}
td.pcell.conf{{box-shadow:inset 0 0 0 2px #ffffffcc}}
.secband{{position:sticky;top:0;z-index:5;background:var(--panel);font-size:11.5px;font-weight:700;padding:5px 4px}}
.cellh{{position:sticky;top:27px;z-index:4;background:var(--panel);color:var(--muted);font-size:9px;font-weight:600;height:62px;width:40px;min-width:40px;writing-mode:vertical-rl;transform:rotate(180deg);padding:4px 0;border-right:1px solid var(--line)}}
.cellh.bbb{{color:#e6b85c}}
.rowlabel{{position:sticky;left:0;z-index:3;background:var(--panel);text-align:left;min-width:172px;width:172px;padding:5px 9px;border-right:2px solid var(--line)}}
.rowlabel.pl{{background:#23341f;border-right:2px solid #3a7a4a}}
.prow td{{height:40px;font-size:11px}}
.dz{{font-weight:800;font-size:12px;color:#cdeccf;letter-spacing:.3px}} .dz2{{font-weight:700;font-size:12px;color:var(--ink)}}
.ph{{color:var(--muted);font-size:10px}}
.cornerblank{{min-width:172px;width:172px;border-right:2px solid var(--line);position:sticky;left:0;top:0;z-index:6;background:var(--panel)}}
.sephdr td{{background:#0b0805;height:26px;border:0}}
.seplabel{{position:sticky;left:0;background:#0b0805;text-align:left;padding:5px 10px;color:#caa46a;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;min-width:172px;border-right:2px solid var(--line)}}
</style>
<div class="wrap">
<h1>Patient straw man <span style="color:#7d7059;font-size:12px;font-weight:400">&mdash; {html.escape(D["patient_id"])}</span></h1>
<p class="sub1">The patient's own per-cell architecture, laid out on the same eight-class grid as the disease wall so the two can be set side by side. The top row is this bloodwork; beneath it are the validated patterns the patient flagged, pulled straight from the wall, for direct comparison.</p>
{stress_banner}
<div class="flag"><b>Flagged for comparison:</b> {flagged_names}<br><span class="via">{html.escape(D["verdict"])}</span></div>
<div class="legend">
<span style="color:var(--muted);font-weight:700;margin-right:2px">Your cells (A-score gauge tiers):</span>
<span><span class="sw" style="background:#4a72a8"></span>Suppressed &lt;0.95</span>
<span><span class="sw" style="background:#2f7a4f"></span><b>Normal / healthy</b> 0.95&ndash;1.04</span>
<span><span class="sw" style="background:#b8923a"></span>Elevated 1.04&ndash;1.07</span>
<span><span class="sw" style="background:#cc6f2e"></span>Past Warburg 1.07&ndash;1.10</span>
<span><span class="sw" style="background:#c4493d"></span>Breach &ge;1.10</span>
<span><span class="sw" style="background:#c4493d;box-shadow:inset 0 0 0 2px #fff"></span>outline = CI clears normal (confident)</span>
<span><span class="sw" style="background:#cc6f2e;background-image:repeating-linear-gradient(45deg,#0000,#0000 2px,#fff4 2px,#fff4 4px)"></span>hatched = CI uncertain</span>
<span><span class="sw" style="background:var(--blank)"></span>not scored</span>
<span style="color:var(--muted)">disease rows below show Cohen's d</span>
</div>
<div class="tablewrap"><table>
<thead>
<tr><th class="cornerblank"></th>{sec_band}</tr>
<tr><th class="rowlabel" style="top:27px;position:sticky">cell &rarr;</th>{cell_head}</tr>
</thead>
<tbody>
{patient_row}
<tr class="sephdr"><td class="seplabel">&darr; validated patterns you flagged</td><td colspan="{len(ordered)}"></td></tr>
{''.join(disease_rows_html)}
</tbody>
</table></div>
<p class="sub1" style="margin-top:12px">Read it like a clinician: where the patient's red or blue lines up with a disease row's red or blue, the patient is moving that pattern's way; where the patient is green, that cell is in the healthy band. A single cell is never the call &mdash; the shape across cells is. This patient's confirmed signal is the breast residual matched filter (see the report); the per-cell view here shows the systemic stress pattern that accompanies it.</p>
</div>"""
open("/home/claude/IAM_Patient_StrawMan.html","w").write(HTML)
print("rendered patient straw man:", len(HTML), "bytes |", len(ordered), "columns | 1 patient row +", len(frows), "disease rows")
