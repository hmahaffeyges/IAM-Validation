import json, html
D=json.load(open("/home/claude/strawman_data_v2.json"))
rows=D["disease_rows"]; sec_cols=D["sec_cols"]; sections=D["sections_used"]; comp=D["completeness"]
SECLABEL={"lymphoid":"Lymphoid","myeloid":"Myeloid","progenitor":"Progenitor","cycling":"Cycling",
 "secretory":"Secretory","terminal":"Terminal","stromal":"Stromal","stem":"Stem"}
SECACCENT={"lymphoid":"#8b83e6","myeloid":"#e0a23a","progenitor":"#5fb0a8","cycling":"#c77fd6",
 "secretory":"#5a9bd4","terminal":"#d98a6a","stromal":"#8aa86a","stem":"#b0926a"}
CLS_LABEL={"secretory":"Secretory","cycling":"Cycling","terminal":"Terminal","immune":"Immune",
 "progenitor":"Progenitor","stem_pluri":"Stem (pluripotent)","stromal":"Stromal","stem_adult":"Stem (adult)"}
BBB={"cortical_neurons","neurons_pooled","glia","astrocytes","brain_pooled","brain_astrocytes",
 "oligodendrocytes","OPC","microglia","NeuMa","NeuIm"}
def short(c):
    s=c.replace("_cells","").replace("_pooled","\u00b7p").replace("regulatory_T","Treg")\
       .replace("naive_","n").replace("memory_","m").replace("_T","T").replace("_B","B")
    return s if len(s)<=14 else s[:13]+"\u2026"
CAT=[("Solid tumours",["breast_cancer","colorectal_cancer","lung_cancer","gastric_cancer","pancreatic_cancer",
   "prostate_cancer","bladder_cancer","kidney_cancer","hcc","esophageal_cancer_eac","esophageal_cancer_escc",
   "cervical_cancer","glioma_gbm","glioma_lgg","thymoma"]),
 ("Haematologic",["leukemia_aml","leukemia_b_all","leukemia_t_all","leukemia_cll","leukemia_cml",
   "lymphoma_dlbcl","multiple_myeloma","mds","mpn"]),
 ("Neurodegenerative",["alzheimers_disease","parkinsons_disease","frontotemporal_dementia","als",
   "psp_cbd_tauopathies","multiple_sclerosis"]),
 ("Cardiovascular",["aortic_dissection_BAV","pah","ischemic_stroke"]),
 ("Autoimmune / inflammatory",["rheumatoid_arthritis","lupus_sle","psoriasis","crohns_disease",
   "ulcerative_colitis","active_allergies","inflammaging"]),
 ("Infection / immune-activation",["chronic_cmv","chronic_ebv","chronic_hiv","chronic_hepatitis_bc",
   "recent_infection_viral","recent_infection_bacterial","recent_vaccination"]),
 ("Neuropsychiatric",["major_depression","schizophrenia"]),
 ("Physiologic / other",["pregnancy","normal_aging","active_chemotherapy"])]
order={}; n=0
for cat,ds in CAT:
    for d in ds: order[d]=(n,cat); n+=1
rows_sorted=sorted(rows,key=lambda r:(order.get(r["disease"],(999,"Other"))[0], r.get("time_range") or ""))

def cell_html(c,info,is_origin):
    cls_o=" origin" if is_origin else ""
    if info is None: return f'<td class="blank{cls_o}"></td>'
    d=info["d"]; arr=info["arr"]
    if d is None and arr:
        col="#d6584e" if arr=="up" else "#4f86d6"; sym="\u25b2" if arr=="up" else "\u25bc"
        return f'<td class="q{cls_o}" style="color:{col}">{sym}</td>'
    a=min(abs(d)/2.0,1.0)*0.8+0.18
    bg=f"rgba(214,88,78,{a:.2f})" if d>=0 else f"rgba(79,134,214,{a:.2f})"
    return f'<td class="{cls_o.strip()}" style="background:{bg}">{d:+.2f}</td>'

sec_band="".join(f'<th class="secband" colspan="{len(sec_cols[s])}" style="color:{SECACCENT[s]};border-bottom:2px solid {SECACCENT[s]}">{SECLABEL[s]}</th>'
  for s in sections if sec_cols[s])
cell_head="".join(f'<th class="cellh{" bbb" if c in BBB else ""}" title="{html.escape(c)}">{html.escape(short(c))}{" \u26a0" if c in BBB else ""}</th>'
  for s in sections for c in sec_cols[s])
ordered_cells=[c for s in sections for c in sec_cols[s]]

body=[]; last_cat=None
for r in rows_sorted:
    cat=order.get(r["disease"],(999,"Other"))[1]
    if cat!=last_cat:
        body.append(f'<tr class="cathdr"><td class="catlabel">{html.escape(cat)}</td><td colspan="{len(ordered_cells)}"></td></tr>'); last_cat=cat
    dname=r["disease"].replace("_"," ").replace(" cancer","").title()
    ph=r["phase"].replace("_"," "); tr=r["time_range"]
    sub=r["substrate"].replace("_"," ").replace("whole blood buffy coat","WB").replace("plasma cfDNA","cfDNA").replace("tumor tissue","tissue")
    persp=r.get("perspective",""); mech=r.get("mechanism",""); anc=r.get("anchors","")
    tip=html.escape(f"{persp}\n\nMechanism: {mech}\n\nValidation anchors: {anc}")
    origins=set(r.get("origin_cells",[]))
    label=(f'<div class="dz">{html.escape(dname)}</div>'
           f'<div class="ph">{html.escape(ph)}{" \u00b7 "+html.escape(tr) if tr else ""}</div>'
           f'<div class="sub">{html.escape(sub)}{"  \u24d8" if persp else ""}</div>')
    cells="".join(cell_html(c, r["cells"].get(c), c in origins) for c in ordered_cells)
    body.append(f'<tr title="{tip}"><td class="rowlabel">{label}</td>{cells}</tr>')

# completeness panel
def comp_block():
    parts=[]
    for cls in ["immune","secretory","cycling","terminal","progenitor","stromal","stem_pluri"]:
        items=comp.get(cls,[])
        gen=[c for c,k in items if k=="origin_unfilled"]
        if gen:
            parts.append(f'<div class="cgrp"><span class="ctitle">{CLS_LABEL.get(cls,cls)}</span> '
                         +" ".join(f'<span class="cpill">{html.escape(c)}</span>' for c in gen)+'</div>')
    return "".join(parts)

HTML=f"""<!doctype html><meta charset="utf-8">
<style>
:root{{--bg:#100c08;--panel:#181109;--blank:#1e160d;--line:#2a1f12;--ink:#efe6d8;--muted:#b6a890;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:12px/1.3 ui-sans-serif,system-ui,sans-serif}}
.wrap{{padding:18px 20px 40px}} h1{{font-size:18px;margin:0 0 2px}}
.sub1{{color:var(--muted);font-size:12px;margin:0 0 14px;max-width:1100px}}
.legend{{display:flex;flex-wrap:wrap;gap:14px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:9px 14px;margin-bottom:14px;font-size:11.5px}}
.sw{{display:inline-block;width:13px;height:13px;border-radius:3px;vertical-align:-2px;margin-right:5px}}
.tablewrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;max-height:78vh}}
table{{border-collapse:separate;border-spacing:0;font-variant-numeric:tabular-nums}}
th,td{{padding:0;text-align:center}}
td{{width:38px;min-width:38px;height:34px;border-right:1px solid #00000040;border-bottom:1px solid #00000040;color:#fff;font-size:10px;font-weight:600}}
td.blank{{background:var(--blank)}} td.q{{background:var(--blank);font-size:11px}}
td.origin{{box-shadow:inset 0 0 0 2px #d4a24a}}
.secband{{position:sticky;top:0;z-index:5;background:var(--panel);font-size:11.5px;font-weight:700;padding:5px 4px}}
.cellh{{position:sticky;top:27px;z-index:4;background:var(--panel);color:var(--muted);font-size:9px;font-weight:600;height:64px;width:38px;min-width:38px;writing-mode:vertical-rl;transform:rotate(180deg);padding:4px 0;border-right:1px solid var(--line)}}
.cellh.bbb{{color:#e6b85c}}
tr[title]:hover .rowlabel{{background:#241a0f}}
.rowlabel{{position:sticky;left:0;z-index:3;background:var(--panel);text-align:left;min-width:188px;width:188px;padding:5px 9px;border-right:2px solid var(--line);cursor:help}}
.dz{{font-weight:700;font-size:12px;color:var(--ink)}} .ph{{color:var(--muted);font-size:10px}}
.sub{{color:#8a7c64;font-size:9px;text-transform:uppercase;letter-spacing:.4px}}
.cornerblank{{min-width:188px;width:188px;border-right:2px solid var(--line);position:sticky;left:0;top:0;z-index:6;background:var(--panel)}}
.cathdr td{{background:#0b0805;height:24px;border:0}}
.catlabel{{position:sticky;left:0;background:#0b0805;text-align:left;padding:4px 10px;color:#caa46a;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;min-width:188px;border-right:2px solid var(--line)}}
.comp{{margin-top:18px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px}}
.comp h2{{font-size:13px;margin:0 0 4px;color:#caa46a}} .comp p{{color:var(--muted);font-size:11px;margin:0 0 10px;max-width:1000px}}
.cgrp{{margin:4px 0;font-size:11px}} .ctitle{{display:inline-block;min-width:130px;color:var(--ink);font-weight:700}}
.cpill{{display:inline-block;background:#221812;border:1px solid var(--line);border-radius:5px;padding:1px 6px;margin:2px;color:#9c8e76;font-size:10px}}
.prov{{margin-top:12px;color:#7d7059;font-size:10.5px;max-width:1100px}}
</style>
<div class="wrap">
<h1>IAM Disease Signature Wall &mdash; official straw man <span style="color:#7d7059;font-size:12px;font-weight:400">v2 (validated)</span></h1>
<p class="sub1">Disease matrix v1.8 &middot; {len(rows_sorted)} disease&times;time-range patterns &middot; {len(ordered_cells)} cells across 8 architecture classes. Every immune-cell direction cross-checked against the immune-atlas card lens (81/81) and each row's VAL anchors. Hover any disease row for its clinical perspective, mechanism, and validation anchors.</p>
<div class="legend">
<span><span class="sw" style="background:rgba(214,88,78,.85)"></span><b>Red = elevation</b></span>
<span><span class="sw" style="background:rgba(79,134,214,.85)"></span><b>Blue = suppression</b></span>
<span><span class="sw" style="background:var(--blank)"></span>Dark = no signature</span>
<span style="color:#e6b85c">&#9650;/&#9660; = direction only</span>
<span><span class="sw" style="box-shadow:inset 0 0 0 2px #d4a24a;background:var(--blank)"></span>gold ring = cell of origin</span>
<span style="color:#e6b85c">&#9888; = barrier-breach cell</span>
<span><b>number = Cohen's d</b> (validation effect size, not the A-scale)</span>
</div>
<div class="tablewrap"><table>
<thead>
<tr><th class="cornerblank"></th>{sec_band}</tr>
<tr><th class="rowlabel" style="top:27px;position:sticky">disease &middot; stage &middot; substrate</th>{cell_head}</tr>
</thead>
<tbody>{''.join(body)}</tbody>
</table></div>
<div class="comp">
<h2>Cells defined in the atlas with no validated disease signature yet</h2>
<p>These cell-of-origin types exist in the 115-cell atlas but no disease row in matrix v1.8 carries a validated departure for them. They are shown here for completeness &mdash; not fabricated as blank tiles in the wall. As cohorts accrue, these fill in.</p>
{comp_block()}
</div>
<p class="prov">Provenance: per-cell directions seeded from disease_cell_signature_matrix_v1.8 (sha 1ed44ccc&hellip;), immune directions verified against immune-atlas card v2.0 disease_immune_lens, breast/AD architectural directions verified against breast-epic v3.1 (Mahalanobis d=1.88/2.10) and ad-immune v3.1 cards. Residual maps (breast/AD/immune) feed the per-CpG SOP&nbsp;8.2 matched filter &mdash; the confirmation layer that runs alongside this per-cell wall. Numbers are cohort effect sizes; a single patient moves the same direction at smaller magnitude.</p>
</div>"""
open("/home/claude/IAM_Disease_Wall_strawman_v2.html","w").write(HTML)
print("rendered v2:", len(HTML), "bytes |", len(rows_sorted),"rows x",len(ordered_cells),"cells")
