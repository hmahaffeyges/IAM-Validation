#!/usr/bin/env python3
"""MethylPhys CPG - the researcher interface / report builder (row 9, in build 2026-09-22; UNSEALED until it runs end to end).

Renders ONE self-contained HTML from cpg_conductor.run_full's bundle plus the runtime files the chain actually read.
Tabs: Reading | Every cell | Departure | Sky | Healthy reference | Integrity | Chain | Physics | Story | Record | Run.
Two audiences (Clinician / Researcher) toggle on one run. Print reflows Reading + Every cell + Departure + Sky.
Vocabulary guard: the measurement tabs may not carry a disease name, a diagnosis, a verdict, or an age in years.
Author's report specification (Issue 003 p4): cells detected and %, A per cell and per class with placement and tier
(tiers on the CLASS gauge, where they are commissioned), the Stage 5 departure with the laboratory's false-alarm rate,
the patient's sky, every flag; BREACH is a gauge reading (a temperature), not a comparison to any disease record.
"""
import os, sys, json, html, math, base64, hashlib, csv, re, time, glob, ast, subprocess
HERE=os.path.dirname(os.path.abspath(__file__)); ENGINE=os.path.dirname(HERE); BIO=os.path.dirname(ENGINE); REPO=os.path.dirname(BIO)
RT=os.path.join(ENGINE,"Runtime Matrices")
def _find(name, root=BIO):
    for dp,_,fs in os.walk(root):
        if "RETIRED" in dp or ".git" in dp: continue
        if name in fs: return os.path.join(dp,name)
    raise FileNotFoundError(name)
def _sha(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()
def _j(p): return json.load(open(p))
def _e(x): return html.escape(str(x))
def _git_head():
    try: return subprocess.run(["git","-C",REPO,"rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip() or "HEAD"
    except Exception: return "HEAD"
GH="https://github.com/hmahaffeyges/IAM-Validation"
def _gh(path, sha): return f"{GH}/blob/{sha}/{path.replace(os.sep,'/')}"
def _rel(p): return os.path.relpath(p, REPO)

# ---------------- runtime constants (the report reads the same files the chain did) ----------------
def load_runtime():
    R={}
    R["band"]=_j(_find("identity_band_v3.json")); m=R["band"]["_meta"]; coh=m["cohorts"]; R["cohorts"]=ast.literal_eval(coh) if isinstance(coh,str) else coh
    R["maps"]=_j(_find("beta_scale_maps_v1.json")); R["age"]=_j(_find("reference_age_curve_v1.json")); R["tiers"]=_j(_find("tier_breakpoints.json"))
    R["floors"]=_j(_find("presence_floors_v1.json")); R["ident"]=_j(_find("iamatlas_gauge_identity_loci_v1_0.json"))
    R["c2c"]=_j(_find("IAMAtlasREBUILD_celltype_to_class.json")); R["panels"]=_j(_find("directional_panels_v1_0.json"))
    R["files"]={n:_find(n) for n in ["identity_band_v3.json","beta_scale_maps_v1.json","reference_age_curve_v1.json","tier_breakpoints.json","presence_floors_v1.json",
               "iamatlas_gauge_identity_loci_v1_0.json","iamatlas_celltype_markers_v0_2.json","IAMAtlasREBUILD_celltype_to_class.json","directional_panels_v1_0.json",
               "iamatlas_cpg_to_healpix_nside128.npz","cpg_conductor.py","cpg_gauge_engine.py","lab_zero.py","cpg_tiers.py","stage_4_6_patient_cmb.py","walther_iam_deconvolver.py",
               "bidirectional_decomposition.py","iamatlas_a_scoring.py","stage_1_idat_calibration.py","IAMAtlasREBUILD_provenance.json"] if _exists(n)}
    R["sha"]=_git_head()
    sys.path.insert(0,ENGINE); import cpg_tiers as T; sch=T.scheme(); R["tier_bands"]=sch["bands"]; R["tier_version"]=sch.get("version")   # ONE tier definition (PROC-TIER-01)
    return R
def _exists(n):
    try: _find(n); return True
    except FileNotFoundError: return False
TIER_COL={"SUPPRESSED":"#8fb3e6","NORMAL":"#86c28b","ELEVATED":"#f2d27a","SIGNIFICANTLY_ELEVATED":"#f0b04a","BREACH":"#d9736a","AT_CEILING":"#b2182b"}
CLASSES=["stem_pluri","stem_adult","progenitor","cycling","secretory","immune","terminal","stromal"]
CLASS_LABEL={"stem_pluri":"Stem (pluripotent)","stem_adult":"Stem (adult)","progenitor":"Progenitor","cycling":"Cycling epithelial","secretory":"Secretory / glandular","immune":"Immune & haematopoietic","terminal":"Terminal / post-mitotic","stromal":"Stromal & connective"}

# ---------------- gauges as inline SVG (crisp on screen and paper, no image library) ----------------
def ruler_svg(value, bands, band_p10=None, band_p90=None, lo=0.88, hi=1.20, w=760, h=64, label=None, muted=False, ceiling=None):
    def X(a): return 40+ (min(max(a,lo),hi)-lo)/(hi-lo)*(w-80)
    parts=[f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A-score gauge">']
    for name,a,b in bands:
        if b<lo or a>hi: continue
        col=TIER_COL.get(name,"#ccc"); parts.append(f'<rect x="{X(a):.1f}" y="18" width="{max(X(min(b,hi))-X(max(a,lo)),0):.1f}" height="18" fill="{col}" opacity="{0.35 if muted else 0.9}"/>')
    for t in (0.90,0.95,1.00,1.04,1.07,1.10,1.15,1.20):
        if lo<=t<=hi: parts.append(f'<line x1="{X(t):.1f}" y1="36" x2="{X(t):.1f}" y2="41" stroke="#999"/><text x="{X(t):.1f}" y="52" font-size="9" fill="#bbb" text-anchor="middle">{t:.2f}</text>')
    if band_p10 is not None and band_p90 is not None:
        parts.append(f'<rect x="{X(band_p10):.1f}" y="12" width="{X(band_p90)-X(band_p10):.1f}" height="30" fill="none" stroke="#e8e8ff" stroke-width="1.5" stroke-dasharray="3,2"/><text x="{X(band_p90)+3:.1f}" y="15" font-size="8.5" fill="#cfd8ff">healthy p10-p90</text>')
    parts.append(f'<line x1="{X(1.0):.1f}" y1="10" x2="{X(1.0):.1f}" y2="42" stroke="#fff" stroke-width="1.5"/>')
    if ceiling and lo<=ceiling<=hi: parts.append(f'<line x1="{X(ceiling):.1f}" y1="10" x2="{X(ceiling):.1f}" y2="42" stroke="#b2182b" stroke-width="2" stroke-dasharray="4,2"/><text x="{X(ceiling):.1f}" y="9" font-size="8.5" fill="#f4a" text-anchor="middle">ceiling 1/H_min</text>')
    if value is not None and not muted:
        x=X(value); parts.append(f'<polygon points="{x:.1f},18 {x-6:.1f},6 {x+6:.1f},6" fill="#fff"/><line x1="{x:.1f}" y1="18" x2="{x:.1f}" y2="36" stroke="#fff" stroke-width="2"/><text x="{x:.1f}" y="62" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">{value:.3f}</text>')
    if label: parts.append(f'<text x="40" y="9" font-size="10" fill="#ddd">{_e(label)}</text>')
    parts.append('</svg>'); return "".join(parts)
def bar(pct, col="#7fa8cc", w=160):
    return f'<svg viewBox="0 0 {w} 10" width="{w}" height="10"><rect width="{w}" height="10" fill="#222"/><rect width="{max(0,min(1,pct))*w:.1f}" height="10" fill="{col}"/></svg>'

# ---------------- vocabulary guard (measurement tabs only) ----------------
FORBIDDEN=re.compile(r"\b(cancer|carcinoma|tumou?r|malignan\w*|alzheimer\w*|dementia|leukemia|lymphoma|diagnos\w*|verdict|culprit|cellular age|years? old|prognos\w*|disease)\b",re.I)
def guard(section_html, tab):
    txt=re.sub(r"<[^>]+>"," ",section_html); bad=sorted(set(m.group(0).lower() for m in FORBIDDEN.finditer(txt)))
    if bad: raise ValueError(f"vocabulary guard [{tab}]: {bad}")
    return section_html

# ======================= measurement tabs =======================
GAUGE_EXPLAINER="""<div class="explain"><h3>What the class gauge is, in plain words</h3>
<p><b>Loci.</b> <i>Locus</i> is Latin for "place"; <i>loci</i> is the plural. In genetics a locus is an address on the DNA - chromosome 6, position 31,400,000, say.
A CpG locus is an address where a C is followed by a G and the cell can attach a methyl tag or not. Your genome has about 28 million of them; the array reads about 480,000.</p>
<p><b>Identity loci.</b> Think of a school with eight grades, every student with a locker. Most lockers tell you nothing about the grade. Some are characteristic: every ninth-grader's
holds the same geometry book. Those are the identity lockers for that grade. For each of the eight cell classes we found the CpG addresses where healthy cells of that class all sit at
about the same methylation level - the addresses that say "I am an immune cell." For immune that is about 42,000 addresses.</p>
<p><b>The gauge.</b> Go to those addresses in this sample and average how methylated they are. A healthy immune cell holds that average at a characteristic level (about 0.73 on a
0-to-1 scale). Turn the average into an entropy - how disordered the pattern is: 0 if every address is fully on or off, 1 bit if every address is a coin flip - and divide by the entropy
a healthy immune cell holds (H_min, measured once from 37 reference cells and frozen). Healthy divides to 1.0. If the class is losing its grip on its identity pattern the addresses drift
toward coin-flip, entropy rises and A climbs toward the ceiling. <b>A is how well this class of cells is still holding its own identity pattern, compared with a healthy one.</b></p>
<p><b>Three corrections before the number is placed.</b> The laboratory's pipeline shifts every beta a little (a measured slope and intercept, the <i>pipeline map</i>); each laboratory
sits at its own constant offset (the <i>laboratory zero</i>, measured from 40 of its healthy arrays); and healthy A rises slowly with age (the <i>age curve</i>, measured on 1,379 healthy
donors). The corrected value A'' is placed in the healthy band (the middle 80% of healthy donors, four laboratories) and given its tier. <b>BREACH</b> is a reading on this ruler - like a
temperature of 104 F - not a comparison to any prior cohort.</p></div>"""

def tab_reading(o, R, sid):
    H=[]; comp=o["composition"]["class"]; tot=sum(comp.values()) or 1.0; sc=(0.01 if tot>1.5 else 1.0)   # class composition is stored in percent (open conductor item)
    ctx=o.get("context",{}); cls=o["classes"]
    H.append(f"<h2>Reading - {_e(sid)}</h2><table class='kv'><tr><td>Specimen</td><td>{_e(ctx.get('substrate','whole blood'))}</td><td>Substrate</td><td>DNA methylation (450K/EPIC beta)</td></tr>"
             f"<tr><td>Declared age</td><td>{_e(ctx.get('age','-'))}</td><td>Pipeline</td><td>{_e(o.get('scale','-'))}</td></tr>"
             f"<tr><td>Laboratory</td><td>{_e(o['patient_sky'].get('lab') or (o.get('cfg') or {}).get('lab','-'))}</td><td>Laboratory zero</td><td>{o.get('lab_zero')!s}</td></tr></table>")
    # composition
    H.append("<h3>1. What is in the sample (Stage 2 - Walther deconvolver against the 115-cell atlas)</h3><div class='cols'><div><h4>By architecture class</h4><table class='t'>")
    for c in CLASSES:
        f=comp.get(c,0)*sc; H.append(f"<tr><td>{CLASS_LABEL[c]}</td><td>{bar(f)}</td><td class='n'>{100*f:.1f} %</td></tr>")
    H.append("</table></div><div><h4>By cell type (every cell the deconvolver placed)</h4><table class='t'>")
    for r in sorted(o["composition"]["celltype"], key=lambda r:-r["pct"]):
        H.append(f"<tr><td>{_e(r['cell'])}{' <span class=flag>not expected in this specimen</span>' if r.get('flag') else ''}</td><td>{bar(r['pct']/100 if r['pct']>1.5 else r['pct'],'#a0c8a0')}</td><td class='n'>{(r['pct'] if r['pct']>1.5 else r['pct']*100):.1f} %</td></tr>")
    H.append("</table><p class='m'>Only cells the constrained fit was forced to place appear here; every one of the 115 atlas cells is scored on the <b>Every cell</b> tab.</p></div></div>")
    # class gauges
    H.append("<h3>2. The class gauge - A'' on the identity loci, against the healthy band</h3>")
    # the conductor reads stem_adult + progenitor as ONE joint component (haematopoietic_progenitor) on shared identity loci - one gauge row, not two
    order=[c for c in CLASSES if c not in ("stem_adult","progenitor")]; order.insert(1,"haematopoietic_progenitor")
    for c in order:
        if c=="haematopoietic_progenitor":
            rec=cls.get("haematopoietic_progenitor",{}); label="Haematopoietic progenitor (stem_adult + progenitor, joint identity loci)"; frac=(comp.get("stem_adult",0)+comp.get("progenitor",0))*sc; hm=R["ident"].get("progenitor",{}).get("H_min")
        else:
            rec=cls.get(c,{}); label=CLASS_LABEL[c]; frac=comp.get(c,0)*sc; hm=R["ident"].get(c,{}).get("H_min")
        ceiling=(1.0/hm) if hm else None
        if rec and rec.get("reportable"):
            b=rec.get("band") or {}; H.append(f"<div class='gauge'><div class='gl'><b>{label}</b> · fraction {100*rec.get('fraction',0):.1f} % · {rec.get('n_loci','?'):,} identity loci · H_min {rec.get('H_min')}"
                     f"<span class='tier' style='background:{TIER_COL.get(rec.get('tier'),'#555')}'>{_e(rec.get('tier'))}</span> <span class='pl'>{_e(rec.get('placement'))}</span></div>"
                     +ruler_svg(rec["A_abs"],R["tier_bands"],b.get("p10"),b.get("p90"),ceiling=ceiling)+
                     f"<div class='m'>A_mapped {rec.get('A_mapped')} - age term {rec.get('age_reference_c')} - laboratory zero {rec.get('lab_zero')} = <b>A'' {rec.get('A_abs')}</b>; band {b.get('p10')}-{b.get('p90')} ({_e(rec.get('band_status'))}); tiers {_e(rec.get('tier_note'))}</div></div>")
        else:
            why=(rec or {}).get("reason") or ("present in the sample; no commissioned healthy band for this class on this specimen yet" if frac>=0.02 else "not present above the presence floor in this specimen")
            A=(rec or {}).get("A_mapped"); H.append(f"<div class='gauge muted'><div class='gl'><b>{label}</b> · fraction {100*frac:.1f} %"+(f" · A_mapped {A}" if A else "")+f" <span class='tier' style='background:#444'>NOT REPORTABLE</span></div>"+ruler_svg(None,R["tier_bands"],muted=True,ceiling=ceiling)+f"<div class='m'>{_e(why)}. A gauge without a commissioned band prints no placement and no tier.</div></div>")
    H.append(GAUGE_EXPLAINER)
    return guard("".join(H),"Reading")

def tab_cells(o, R, percell_ref=None):
    cells=o.get("cells_all") or {}; comp_cells={r["cell"]:r for r in o["composition"]["celltype"]}
    H=["<h2>Every cell - all 115 atlas cell types, scored</h2>",
       "<p>Per-cell A = H(mean beta over that cell's ~100 discriminative marker CpGs) / H_min(class). This is the <b>separation surface</b> - the surface that produced the sealed breast anchors (r = 1.00000) - and it is where <i>which cell moved, and in which direction</i> is read. "
       "Healthy cells do not sit at 1.0 on this ratio (each cell's markers have their own natural entropy, and in bulk blood a rare cell's markers mostly carry other cells' DNA), so a cell is read against <b>its own healthy range</b>, measured on its own markers from the four laboratories' healthy arrays. "
       "Tier words (BREACH etc.) print on a cell only once its own reference is commissioned; until then the column shows the range status.</p>"]
    by={}; 
    for cell,r in cells.items(): by.setdefault(r.get("class","?"),[]).append((cell,r))
    for c in CLASSES:
        rows=sorted(by.get(c,[]), key=lambda kv:-(kv[1].get("A") or 0)); 
        if not rows: continue
        H.append(f"<h3>{CLASS_LABEL[c]} <span class='m'>({len(rows)} cells; H_min {R['ident'].get(c,{}).get('H_min','-')})</span></h3><table class='t cells'><tr><th>cell type</th><th>placed</th><th>fraction</th><th>A (marker surface)</th><th>coverage</th><th>healthy range (own markers)</th><th>direction</th></tr>")
        for cell,r in rows:
            A=r.get("A"); fr=r.get("fraction") or 0; ref=(percell_ref or {}).get(cell)
            if ref: rng=f"{ref['p10']:.3f}-{ref['p90']:.3f} (n={ref['n']})"; dirn=("above" if A>ref['p90'] else "below" if A<ref['p10'] else "within")
            else: rng="<span class='pend'>pending - reference build in progress</span>"; dirn="-"
            H.append(f"<tr class='{'placed' if fr>0 else ''}'><td>{_e(cell)}</td><td>{'yes' if fr>0 else '-'}</td><td class='n'>{100*fr:.1f} %</td><td class='n'>{'' if A is None else f'{A:.3f}'}</td><td class='n'>{r.get('coverage',0):.2f}</td><td>{rng}</td><td>{dirn}</td></tr>")
        H.append("</table>")
    bd=o.get("bidirectional",{}); H.append("<h3>Direction - Stage 4.5 bidirectional composite</h3><p>Pooled entropy folds hypo- and hyper-methylation together; the signed composite keeps the sign, per sealed panel. Panels exist only where one was sealed (immune, VAL-051 / CPG-VAL-019); the other classes say so.</p><table class='t'><tr><th>class</th><th>signed composite</th><th>pooled A on the panel</th><th>panel</th><th>reading</th></tr>")
    for c in CLASSES:
        b=bd.get(c,{}); ad=b.get('a_directional'); ap=b.get('a_pooled'); ads=('' if ad is None else '%+.3f'%float(ad)); aps=('' if ap is None else '%.3f'%float(ap))
        if ad is None: rd="no sealed directional panel for this class"
        else:
            rd=("pooled and signed composite both move; the signed composite is "+("in" if float(ad)>0 else "opposite to")+" the sealed panel's direction" if abs(float(ad))>=0.5 or bool(b.get('flag_bidirectional')) else "within baseline on the sealed panel")
            rd+=f" (panel: {_e(R['panels'].get('immune',{}).get('source','VAL-051 Rule A, 7 CpGs'))})"
        H.append(f"<tr><td>{CLASS_LABEL[c]}</td><td class='n'>{ads}</td><td class='n'>{aps}</td><td>{b.get('n_covered',0)} CpGs</td><td class='m'>{rd}</td></tr>")
    H.append("</table>"); return guard("".join(H),"Every cell")

def tab_departure(o, R):
    d=o["departure"]; H=[f"<h2>Departure - how far this sample sits from the healthy centre</h2>"]
    if d.get("reportable"):
        D=d["mahalanobis_distance"]; t95=d["alarm_threshold_p95"]; t99=d["alarm_threshold_p99"]; w=760
        X=lambda v: 40+min(v,4.0)/4.0*(w-80)
        svg=(f'<svg viewBox="0 0 {w} 70" width="100%" height="70"><rect x="40" y="20" width="{X(t95)-40:.0f}" height="18" fill="#86c28b" opacity=".8"/><rect x="{X(t95):.0f}" y="20" width="{X(t99)-X(t95):.0f}" height="18" fill="#f2d27a" opacity=".8"/><rect x="{X(t99):.0f}" y="20" width="{w-40-X(t99):.0f}" height="18" fill="#d9736a" opacity=".8"/>'
             f'<polygon points="{X(D):.0f},20 {X(D)-6:.0f},8 {X(D)+6:.0f},8" fill="#fff"/><text x="{X(D):.0f}" y="64" font-size="11" fill="#fff" text-anchor="middle" font-weight="bold">{D:.2f}</text>'
             f'<text x="{X(t95):.0f}" y="50" font-size="9" fill="#ddd" text-anchor="middle">p95 {t95:.2f}</text><text x="{X(t99):.0f}" y="50" font-size="9" fill="#ddd" text-anchor="middle">p99 {t99:.2f}</text><text x="40" y="50" font-size="9" fill="#bbb">0</text><text x="{w-40}" y="50" font-size="9" fill="#bbb" text-anchor="end">4+</text></svg>')
        words=("within the healthy scatter" if not d["mahalanobis_beyond_band"] else "beyond the p95 line" if not d["beyond_p99"] else "beyond the p99 line")
        H.append(svg+f"<p><b>In words:</b> the distance is {D:.2f} healthy spreads over {d['n_assessable']} banded class axis{'es' if d['n_assessable']!=1 else ''} ({_e(d.get('driver','immune'))}). The alarm line at {t95:.2f} is where 5 of 100 healthy people sit by chance; this sample is <b>{words}</b>. "
                 f"{_e(d.get('lab_false_alarm_sentence',''))}</p><p class='m'>Reference: {_e(d.get('reference'))}. Status: {_e(d.get('status'))}. With one commissioned class band, the departure is simply |z| of that class; as further class bands are commissioned the distance becomes a true multi-axis Mahalanobis distance over all assessable classes.</p>")
        H.append("<table class='t'><tr><th>axis</th><th>patient A''</th><th>healthy centre</th><th>spread (sigma)</th><th>z</th></tr>"+"".join(f"<tr><td>{_e(c['class'])}</td><td class='n'>{c['patient_A']:.4f}</td><td class='n'>{c['age_matched_mean']:.3f}</td><td class='n'>{c['sigma']:.4f}</td><td class='n'>{c['z']:+.2f}</td></tr>" for c in d.get("top_axis_contributions",[]))+"</table>")
    else: H.append(f"<p class='pend'>NOT REPORTABLE - {_e(d.get('status'))}</p>")
    return guard("".join(H),"Departure")

def tab_sky(o, R, sid, workdir):
    s=o["patient_sky"]; H=["<h2>The patient's sky - Stage 4.6</h2>"]
    H.append("<p>Every CpG the chain reads is placed on a sphere in genomic order (HEALPix, NSIDE 128 - the projection Planck used for the microwave background). At each address the chain computes what this sample's <i>own composition</i> predicts (the Stage 2 fractions mixed over the atlas class means), subtracts the laboratory's per-address zero, and divides by the laboratory's healthy spread at that address - both measured from the same 40 healthy arrays that set the laboratory zero. "
             "The plate shows that residual z. A healthy sky is <b>quiet</b>: 2.6-3.2 % of addresses beyond |z| = 2 on the four commissioned laboratories (the Gaussian expectation is 5 %; the scale is ~1.1x conservative and that constant is printed on every plate). A class panel renders only when Stage 2 places the class above its measured presence floor; masked panels say so.</p>")
    if s.get("available") and s.get("_sky") is not None:
        try:
            sys.path.insert(0,ENGINE); import stage_4_6_patient_cmb as S
            png=os.path.join(workdir,f"sky_{sid}.png"); S.render_plate(s["_sky"],png,f"{sid} - residual z on the laboratory's own zero and scale ({s.get('lab')}, panel n={s.get('scale_panel_n')})")
            H.append(f"<img class='plate' src='data:image/png;base64,{base64.b64encode(open(png,'rb').read()).decode()}' alt='patient sky plate'/>")
        except Exception as e: H.append(f"<p class='pend'>plate not rendered: {_e(e)}</p>")
        a=s["all"]; H.append(f"<p><b>Whole sky:</b> {a['n']:,} addresses; {100*a['frac_abs_z_gt2']:.1f} % beyond |z| = 2 (healthy 2.6-3.2 %); median z {a['median_z']:+.3f}; mean |z| {a['mean_abs_z']:.3f}.</p><table class='t'><tr><th>class panel</th><th>Stage 2 fraction</th><th>presence floor</th><th>status</th><th>addresses</th><th>% beyond |z|=2</th><th>median z</th></tr>")
        for c in CLASSES:
            r=s["classes"].get(c,{}); H.append(f"<tr><td>{CLASS_LABEL[c]}</td><td class='n'>{100*r.get('fraction',0):.1f} %</td><td class='n'>{100*r.get('presence_floor',0):.0f} %</td><td>{_e(r.get('status'))}</td><td class='n'>{r.get('n','') if r.get('assessable') else ''}</td><td class='n'>{('%.1f'%(100*r['frac_abs_z_gt2'])) if r.get('assessable') else ''}</td><td class='n'>{('%+.3f'%r['median_z']) if r.get('assessable') else ''}</td></tr>")
        H.append(f"</table><p class='m'>{_e(s.get('calibration_note',''))}</p>")
    else: H.append(f"<p class='pend'>SKY NOT AVAILABLE - {_e(s.get('reason') or 'this laboratory has no commissioned residual scale (40 healthy arrays through Stage 1 are required; see Healthy reference)')}</p>")
    H.append("<h3>How to read a plate</h3><p>Blue = less methylated than this sample's own composition predicts at that address; red = more. A healthy plate is salt-and-pepper with no structure. Structure - a band, a patch, one class panel lit while the others are quiet - is what the sky is for: it shows <i>where in the genome</i> a departure lives, which no single number can. The plate is a residual map, the same object a cosmologist looks at after subtracting the model from the data.</p>")
    return guard("".join(H),"Sky")

# ======================= reference / integrity / chain / physics / story / record / run =======================
def _link(R, path_or_name, label=None):
    p=path_or_name if os.path.isabs(path_or_name) else R["files"].get(path_or_name)
    if not p or not os.path.exists(p): return _e(label or path_or_name)
    return f"<a href='{_gh(_rel(p),R['sha'])}' target='_blank'>{_e(label or os.path.basename(p))}</a> <span class='sha'>{_sha(p)[:12]}</span>"

def tab_reference(R, percell_status):
    m=R["band"]["_meta"]; coh=R["cohorts"]; mp=R["maps"]["maps"]["stage1_noob_450K"]; age=R["age"]
    H=["<h2>The healthy reference - who, how many, processed how, and what was measured from them</h2>",
       "<p>Every constant a reading is corrected by was measured on healthy people, and a reader is entitled to see exactly which. Nothing on this page comes from a disease sample. The class floors (H_min) are a separate, earlier layer - 37 published reference cell methylomes calibrated by MCMC in April 2026 (G-002) - and the two layers are never mixed.</p>",
       "<h3>1. Who</h3><table class='t'><tr><th>laboratory</th><th>GEO accession</th><th>country</th><th>healthy-arm rule</th><th>n healthy arrays</th><th>laboratory zero z_L</th><th>healthy tail beyond p95</th></tr>"]
    LAB={"GSE87571_Uppsala":("Uppsala","Sweden","all arrays: disease state = normal"),"GSE42861_Karolinska":("Karolinska","Sweden","disease state = Normal (controls of an RA study)"),"GSE111629_UCLA":("UCLA","USA","disease state = PD-free control"),"GSE125105_Munich":("Munich","Germany","diagnosis = control")}
    for k,v in coh.items():
        lab,cty,rule=LAB.get(k,(k,"","")); acc=k.split("_")[0]
        H.append(f"<tr><td>{lab}</td><td><a href='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}' target='_blank'>{acc}</a></td><td>{cty}</td><td>{rule}</td><td class='n'>{v['n']}</td><td class='n'>{float(v['z_lab_full_cohort']):+.4f}</td><td class='n'>{100*float(v['tail_p95']):.1f} %</td></tr>")
    H.append(f"</table><p class='m'>Total {m.get('n')} healthy whole-blood arrays, four laboratories, four countries. Band built {_e(m.get('built'))} by {_e(m.get('procedure'))}.</p>")
    H.append("<h3>2. How processed</h3><p>Raw IDAT pairs (Red + Grn) from GEO through <b>Stage 1</b> - methylprep <i>noob</i> (dye-bias and probe-type normalisation) - to beta. The authors' own processed matrices on GEO are <b>not</b> used: each laboratory's normalisation places beta on a different scale, and the offset between pipelines is larger than the healthy band (LESSON-SCALE-01, measured). "
             f"Stage 1 code: {_link(R,'stage_1_idat_calibration.py')}.</p>")
    H.append(f"<h3>3. What was measured from them, in order</h3><table class='t'><tr><th>layer</th><th>constant</th><th>fitted on</th><th>procedure</th><th>file</th></tr>"
             f"<tr><td>Pipeline map</td><td>beta_roadmap = (beta - {mp['intercept']}) / {mp['slope']}</td><td>{_e(mp['fit_cohort'])}, {mp['n_loci']} identity loci; transfer: {_e(mp['transfer_test'])}</td><td>PHASE 1 / 1c, LESSON-SCALE-01</td><td>{_link(R,'beta_scale_maps_v1.json')}</td></tr>"
             f"<tr><td>Laboratory zero</td><td>z_L = median[A_i - c(decade_i)] - 1 over 40 healthy arrays</td><td>each laboratory's own panel; four values above</td><td>PROC-PANEL-01..03, LAB-ZERO-01/02</td><td>{_link(R,'lab_zero.py')}</td></tr>"
             f"<tr><td>Age curve</td><td>c(decade): {_e(json.dumps(age.get('curve',{})))}</td><td>{m.get('n')} healthy donors, leave-one-lab-out</td><td>PROC-PANEL-02/03, PROC-AGE-01</td><td>{_link(R,'reference_age_curve_v1.json')}</td></tr>"
             f"<tr><td>Healthy band (immune)</td><td>p10-p90 = {R['band'].get('immune',{}).get('p10','?')}-{R['band'].get('immune',{}).get('p90','?')}</td><td>{m.get('n')} zeroed, age-referenced donors</td><td>PROC-SWITCH-02</td><td>{_link(R,'identity_band_v3.json')}</td></tr>"
             f"<tr><td>Sky zero and scale</td><td>per-address mean and spread of the composition residual</td><td>the same 40-array panel per laboratory</td><td>PROC-CMB-01..05</td><td>{_link(R,'stage_4_6_patient_cmb.py')}</td></tr>"
             f"<tr><td>Presence floors</td><td>{_e({k:v for k,v in R['floors'].items() if not k.startswith('_')})}</td><td>160 healthy panel arrays</td><td>PROC-CMB-03/04</td><td>{_link(R,'presence_floors_v1.json')}</td></tr>"
             f"<tr><td>Tier onset</td><td>NORMAL = healthy central 95 % = [0.95, 1.04); 1.07 and 1.10 are the physics lines</td><td>{m.get('n')} donors</td><td>PROC-TIER-01/02</td><td>{_link(R,'tier_breakpoints.json')}</td></tr>"
             f"<tr><td>Per-cell healthy range</td><td>{_e(percell_status)}</td><td>80 arrays per laboratory (seed 2029: 40 build, 40 held out)</td><td>working note, unsealed</td><td>-</td></tr></table>")
    H.append("<h3>4. The data itself</h3><p>The calibrated beta matrices the constants were fitted on are published beside them (per laboratory, filtered to the 153,444 CpGs the chain reads, with the GSM list, Stage 1 version and SHA-256 in a manifest). "
             "<span class='pend'>Rebuild in progress 2026-09-22: the 1,379-array Stage 1 output behind PROC-PANEL-03 was not preserved; 80 arrays per laboratory are being re-run through Stage 1 and will be linked here the moment they exist. A reference layer is not considered commissioned until its data is published beside it (RUNBOOK).</span></p>")
    H.append("<h3>5. What is NOT in the reference</h3><ul><li>No disease sample, no case arm of any cohort.</li><li>No author-processed beta; no typed literature values (the April 80-cell age table was retired for that reason - PROC-RECORD-03).</li><li>The class floors H_min are not fitted here; they come from the 37-cell G-002 calibration and are frozen.</li></ul>")
    return "".join(H)   # not a measurement tab: quotes GEO characteristic fields verbatim ("disease state: normal")

CMB_TWINS=[("MCMC atlas calibration","Cosmological parameter estimation (Planck likelihood chains)","Posterior mean and SD per CpG per class; class floors H_min with R-hat < 1.001 on 37 reference cells","G-002 / G-003b, IAMAtlasREBUILD"),
 ("Pipeline map","Instrument calibration transfer (cross-calibrating detectors)","affine beta map, fit on 32,688 identity loci; transfers to a second laboratory at median A 1.0097","beta_scale_maps_v1.json"),
 ("Laboratory zero","Monopole / dipole removal","one constant per laboratory from 40 healthy arrays; four labs on one scale","lab_zero.py, identity_band_v3.json"),
 ("Age curve","Foreground subtraction","healthy A rises 0.47 mA/yr by decade, same slope on four labs; subtracted before placement","reference_age_curve_v1.json"),
 ("Two deconvolvers (Walther, NILC)","Component separation (NILC is the Planck method)","conservative constrained fit for the composition the report stands on; sensitive variance-weighted method for faint components; disagreement is information","walther_iam_deconvolver.py"),
 ("Residual sky","The anisotropy map / residual map after model subtraction","z per address on the laboratory's own zero and spread; healthy 2.6-3.2 % beyond |z|=2","stage_4_6_patient_cmb.py"),
 ("Presence floors","Galactic mask","a class panel renders only above the floor measured on 160 healthy arrays","presence_floors_v1.json"),
 ("Pre-registration and the falsification register","Blind analysis","every bar written and hashed before the run; failures kept on the record as sealed","Testing_and_Code/PROC_data")]

def tab_integrity(o, R, refusals):
    H=["<h2>Integrity - the fail-safes that kept this reading honest, and what each one measured</h2>",
       "<p>The chain was built with the toolkit cosmology developed for reading a faint signal against a calibrated reference. Each safeguard below has a named twin in that toolkit, a measured constant, and a file with a hash. Where a safeguard could not be applied to this sample, the chain refused to print a number rather than approximate one.</p>",
       "<h3>1. What this run refused, and why</h3>"+("<ul>"+"".join(f"<li>{_e(r)}</li>" for r in refusals)+"</ul>" if refusals else "<p>Nothing was refused on this sample.</p>"),
       "<h3>2. The safeguards, with their cosmology twins</h3><table class='t'><tr><th>safeguard</th><th>cosmology twin</th><th>what it measured / does here</th><th>where</th></tr>"]
    for a,b,c,d in CMB_TWINS: H.append(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td class='m'>{_e(d)}</td></tr>")
    H.append("</table><h3>3. Constants applied to this sample</h3><table class='kv'>")
    imm=o["classes"].get("immune",{}); H.append(f"<tr><td>pipeline</td><td>{_e(o.get('scale'))}</td></tr><tr><td>laboratory zero</td><td>{o.get('lab_zero')}</td></tr><tr><td>age term c(decade)</td><td>{imm.get('age_reference_c')}</td></tr><tr><td>band</td><td>{_e(imm.get('band_status'))}</td></tr><tr><td>tiers</td><td>{_e(imm.get('tier_note'))}</td></tr><tr><td>false-alarm rate, this laboratory</td><td>{_e(o['departure'].get('lab_false_alarm_source'))}</td></tr><tr><td>sky scale</td><td>{'panel n=%s'%o['patient_sky'].get('scale_panel_n') if o['patient_sky'].get('available') else 'not available'}</td></tr></table>")
    H.append("<h3>4. Files read by this run (repository at commit "+_e(R["sha"])+")</h3><table class='t'><tr><th>file</th><th>sha256</th></tr>"+"".join(f"<tr><td><a href='{_gh(_rel(p),R['sha'])}' target='_blank'>{_e(n)}</a></td><td class='sha'>{_sha(p)}</td></tr>" for n,p in sorted(R["files"].items()))+"</table>")
    H.append("<h3>5. Two rules this report obeys</h3><ul><li><b>Detection rule.</b> No definitive statement about what the chain can or cannot detect appears until the chain has been run on that question under seal - 'not yet tested', never 'cannot'. A vocabulary guard refuses to write the measurement tabs if they name a condition, a verdict, or an age in years.</li><li><b>Sealing rule.</b> A pre-registration is written only for a built tool tested against a bar; building is exploration recorded in a dated working note. This report generator is in build and unsealed.</li></ul>")
    return "".join(H)

CHAIN=[("0","Intake","declared age, specimen, substrate; IDAT integrity hash","stage_0_intake.py"),("1","Calibration","IDAT (Red+Grn) -> beta, methylprep noob","stage_1_idat_calibration.py"),
 ("1s","Pipeline map","beta -> reference scale (slope, intercept)","beta_scale_maps_v1.json"),("2","Composition","Walther constrained NNLS against the 115-cell atlas -> cell and class fractions","walther_iam_deconvolver.py"),
 ("A","Per-cell A","H(mean beta over each cell's markers)/H_min(class), all 115 cells","iamatlas_a_scoring.py"),("B","Class gauge","H(mean beta over identity loci)/H_min - c(decade) - z_L, placed in the band, tiered","cpg_gauge_engine.py"),
 ("4.5","Direction","signed directional composite on sealed panels","bidirectional_decomposition.py"),("4.6","Sky","composition-residual z on the laboratory's zero and scale, HEALPix NSIDE 128","stage_4_6_patient_cmb.py"),
 ("5","Departure","distance over banded class axes, chi-square lines, laboratory false-alarm rate","cpg_conductor.py"),("7","Tiers","one JSON-driven tier function; AT_CEILING at 1/H_min","cpg_tiers.py"),
 ("9","Report","this document, from the bundle","build_methylphys.py")]
def tab_chain(R):
    H=["<h2>The chain - every stage this report came from, with the live file</h2><p>Orchestrated by <code>cpg_conductor.run_full</code>. Stage 6 (cellular age in years) and Stage 8 (disease matrix) are not chain stages: single-array age resolution is ~50 years (PROC-AGE-01), and the matrix was compiled from the preliminary record and names conditions, which is not the instrument's job (author's ruling 2026-09-21). The marker-union class statistic retired by PROC-N7-01 runs as a diagnostic only and is not shown.</p><table class='t'><tr><th>stage</th><th>name</th><th>what it does</th><th>file (commit "+_e(R["sha"])+")</th></tr>"]
    for st,nm,what,f in CHAIN: H.append(f"<tr><td>{st}</td><td><b>{nm}</b></td><td>{what}</td><td>{_link(R,f) if f in R['files'] else _e(f)}</td></tr>")
    H.append("</table><p class='m'>The June manifest (CPG_KISS_Chain_Files.md) described the pre-switch chain and must not be used: it names the identity-loci gauge a 'false road' (reversed by PROC-N7-01 and PROC-SWITCH-02) and calls the eight H_min values 'Mahaffey numbers' (they are measured class entropy references; the Mahaffey number is E_drive/k_BT).</p>")
    return "".join(H)

def tab_physics():
    kB=1.380649e-23; T=310.15; R_=8.314462618; EL=kB*T*math.log(2); M=54000/(R_*T)
    return f"""<h2>The physics - Landauer metrology</h2>
<p><b>Premise (peer-reviewed).</b> Writing or erasing one bit costs at least k_B T ln 2 = {EL:.3e} J at body temperature (Landauer 1961). The methylome is a written pattern and obeys that bound - shown for cytosine methylation by Sanchez &amp; Mackenzie (2016, PLoS ONE) using information thermodynamics. This work stands on that premise; it was reached independently (cosmology -> quantum hardware -> semiconductors -> cells) and first read on 2026-09-20. Cited, not built upon.</p>
<h3>Three quantities, never to be confused</h3><table class='t'><tr><th>symbol</th><th>name</th><th>what it is</th><th>units</th><th>varies by</th><th>fixed by</th></tr>
<tr><td>M</td><td>Mahaffey number</td><td>E_drive / k_B T - how many thermal quanta the writing process spends per irreversible operation</td><td>none (ratio)</td><td>substrate</td><td>biochemistry / device physics</td></tr>
<tr><td>H_min(c)</td><td>class entropy reference</td><td>the binary entropy a healthy cell of class c holds at its identity loci</td><td>bits</td><td>class (8)</td><td>healthy reference cells, calibrated once (MCMC, 37 cells), frozen</td></tr>
<tr><td>A</td><td>the gauge</td><td>H(beta_mean at identity loci) / H_min(c)</td><td>none (ratio)</td><td>class x sample</td><td>the sample, over the frozen reference</td></tr></table>
<h3>The Mahaffey number across substrates</h3><table class='t'><tr><th>substrate</th><th>E_drive</th><th>k_B T at</th><th>M</th></tr>
<tr><td>Apple M1 transistor</td><td>switching energy</td><td>348 K</td><td>~117</td></tr><tr><td>cell nucleus</td><td>Delta G_ATP per hydrolysis (54 kJ/mol)</td><td>310 K</td><td>{M:.2f}</td></tr><tr><td>Al transmon qubit</td><td>Delta_Al ln 2</td><td>T_gap = Delta_Al/k_B</td><td>1.000</td></tr></table>
<p>Thermal noise is not subtracted; it is the <b>ruler</b>. The denominator k_B T is the energy of one thermal fluctuation at the operating temperature, so M says how many quanta each substrate pays to write and hold a state against noise it cannot escape. The qubit is cooled until the gap is the only scale left and M lands on exactly 1; the cell runs at 310 K and pays 21. Same equation, opposite corner of the temperature axis.</p>
<h3>Measured, not derived</h3><p>An energy bound per operation constrains the <i>cost</i> of writing a pattern; it says nothing about the <i>entropy</i> of the pattern a healthy class holds. H_min is therefore measured - MCMC on 37 published reference methylomes, posterior R-hat &lt; 1.001, bootstrap cross-check with every frozen value inside its interval (PROC-HMIN-BOOT-01) - and frozen. That is the stronger claim, because it is checkable: the calibration script and its 37-cell database with every DOI are in the Record tab.</p>
<h3>Why identity loci</h3><p>H is concave, so H(mean beta) over a set of addresses is largest when the addresses average to a coin flip. Marker loci are chosen to be extreme and opposite between cells; averaged over a mixture they read as disorder that is not there (the defect PROC-N7-01 caught). Identity loci are the addresses where a healthy class sits at one level, so the mean carries meaning and A = 1 is healthy by construction. The gauge is two-to-one in beta (H is symmetric about 0.5) and has a structural ceiling at 1/H_min.</p>
<h3>Filter and ruler</h3><p>Sanchez &amp; Mackenzie use k_B T ln 2 to model the thermal <i>background</i> and remove it, so that regulatory signal stands out against a control centroid. Here the same constant sets the <i>unit</i>, and the healthy floor is a statement of how far above it a class holds its pattern. One filters, one calibrates. Both are legitimate; the second is what allows a single sample to be read with no control group in the room.</p>"""

def tab_story():
    return """<h2>The story - what the physics of methylation is, and how it was found</h2>
<p>This instrument did not begin in a biology laboratory. It began with a question about why bound systems - stars, galaxies, the expanding universe - hold the states they hold at finite thermodynamic cost, and with the measurement toolkit cosmology built to answer questions like it: calibrate a reference by Markov-chain Monte Carlo, freeze it, and read every new observation as a departure from it, with the analysis pre-registered so the answer cannot be tuned after the fact. The same fidelity ratio was then read on quantum processors (where the thermal quantum is the only scale left) and on semiconductor logic (where it is a Landauer efficiency), and finally on the cell, whose methylome is a written surface held against thermal noise at 310 K.</p>
<p>The premise that the methylome obeys Landauer's bound was already in the peer-reviewed literature (Sanchez &amp; Mackenzie 2016) - reached from biology, and read here for the first time in September 2026, after the chain existed. Two independent arrivals at the same constant are the best thing that can happen to a premise. What is added here is a fixed zero (the class floors), a single-sample absolute reading with no control group, composition read first, and the cosmology toolkit applied as a set of fail-safes: pipeline map, laboratory zero, age curve, residual sky, presence floors, blind analysis.</p>
<p>The work was done between March and September 2026 by an independent researcher with no formal training in physics or genetics, in the open, in a public repository, with every validation - 175 index rows, pre-atlas and post-atlas - pre-registered and sealed, and with every failure kept on the record. Those validations were the design record: run to learn how to build the chain this report comes from. They are cited for what they found and for the history; the tests that will be declared with full confidence are the ones the commissioned chain runs under seal, and only those will be submitted for peer review.</p>
<p>The positioning is deliberately narrow. Not the largest atlas, not all diseases, not early detection, not a claim that one atlas is the right structure. Only this: cells compute and write to a two-dimensional surface, the physics of that writing is known, and the healthy range for it can be calculated and read against a fixed zero. <i>We don't lose because we can't detect everything; we only lose if we can't detect anything.</i></p>
<p>Field name: <b>Physics of Methylation: Landauer Metrology</b>. Instrument: <b>MethylPhys CPG</b> - the Cellular Performance Gauge. The full engineering manual is GAPE Issue 003; the short methods paper is <i>Landauer Metrology of the Methylome</i>. Both are linked from the Record tab.</p>"""

def tab_record(R):
    H=["<h2>Record - every validation and every sealed procedure, linked</h2>"]
    idx=os.path.join(BIO,"Testing_and_Code","VAL_INDEX.csv")
    if os.path.exists(idx):
        rows=list(csv.DictReader(open(idx,encoding="utf-8")))
        H.append(f"<p>{len(rows)} index rows (series G, T, VAL pre-atlas, CPG-VAL post-atlas), rebuilt from the sealed inventory (PROC-HISTORY-01). All of these are the <b>design record</b>: preliminary tests run to build the chain. <a href='{_gh('Biological_Physics/Testing_and_Code/VAL_INDEX.csv',R['sha'])}' target='_blank'>VAL_INDEX.csv</a> · <a href='https://doi.org/10.5281/zenodo.19633499' target='_blank'>Zenodo deposit (pre-atlas code)</a></p><details><summary>Show the index</summary><table class='t small'><tr>"+"".join(f"<th>{_e(k)}</th>" for k in list(rows[0].keys())[:8])+"</tr>")
        for r in rows: H.append("<tr>"+"".join(f"<td>{_e(str(r.get(k,''))[:90])}</td>" for k in list(rows[0].keys())[:8])+"</tr>")
        H.append("</table></details>")
    pd_=os.path.join(BIO,"Testing_and_Code","PROC_data")
    if os.path.isdir(pd_):
        H.append("<h3>Sealed procedures on the commissioned chain (PROC-*)</h3><table class='t'><tr><th>procedure</th><th>files</th><th>link</th></tr>")
        for d in sorted(os.listdir(pd_)):
            p=os.path.join(pd_,d)
            if os.path.isdir(p): H.append(f"<tr><td>{_e(d)}</td><td>{', '.join(sorted(os.listdir(p)))[:120]}</td><td><a href='{GH}/tree/{R['sha']}/Biological_Physics/Testing_and_Code/PROC_data/{d}' target='_blank'>open</a></td></tr>")
        H.append("</table>")
    H.append(f"<h3>Documents</h3><ul><li><a href='{GH}/tree/{R['sha']}/Biological_Physics/Physics_of_Methylation/Issue003' target='_blank'>GAPE Issue 003</a> - the engineering manual (RC1)</li><li><a href='{GH}/tree/{R['sha']}/Biological_Physics/Physics_of_Methylation/Papers' target='_blank'>Landauer Metrology of the Methylome</a> - the methods paper (draft)</li><li><a href='{GH}/blob/{R['sha']}/Biological_Physics/Physics_of_Methylation/Reproduction_Kit/RUNBOOK.md' target='_blank'>RUNBOOK</a> · <a href='{GH}/blob/{R['sha']}/Biological_Physics/Physics_of_Methylation/CHAIN_COMMISSIONING.md' target='_blank'>CHAIN_COMMISSIONING</a> · <a href='{GH}/blob/{R['sha']}/Biological_Physics/HANDOFF.md' target='_blank'>HANDOFF</a></li></ul>")
    return "".join(H)

def tab_run(R):
    return f"""<h2>Run it yourself</h2><p>The chain is open. Clone the repository, verify it on the eleven commissioning arrays, then run your own IDATs. A local server (in build) will drive this same page live, stage by stage.</p>
<pre>git clone {GH}.git
cd IAM-Validation/Biological_Physics
# environment: python 3.11, numpy, pandas, scipy, methylprep (Stage 1)
python3 -c "import lzma,shutil; shutil.copyfileobj(lzma.open('IAM_Atlas/IAMAtlasREBUILD.csv.xz','rb'), open('IAM_Atlas/IAMAtlasREBUILD.csv','wb'))"   # 605 MB atlas
cd Physics_of_Methylation/Reproduction_Kit && python3 test_gauge_switch.py && python3 test_tiers.py && python3 test_patient_sky.py   # conformance on the commissioning arrays
# your sample: Red + Grn IDAT pair -> Stage 1 -> run_full -> this report
python3 CPG_Engine/MethylPhys_Interface/build_methylphys.py --idat-grn SAMPLE_Grn.idat.gz --idat-red SAMPLE_Red.idat.gz --age 58 --lab NEW --out report.html</pre>
<p><b>Inputs the chain needs:</b> the IDAT pair (or a calibrated beta vector), declared age, specimen (whole blood today; plasma cfDNA, tissue, CSF, urine, stool reserved - each needs its own pipeline map, laboratory zero and healthy band before it reads), substrate (methylation today; nucleosome occupancy, fuzziness, WPS and fragment size reserved with their own floors), and the laboratory. A laboratory the chain has not seen prints NOT REPORTABLE until 40 of its healthy arrays have been run to set its zero and sky scale - the instructions for that are in the RUNBOOK.</p>
<p><b>Cohort mode</b> (for validation runs): every sample is read absolutely as above, then the report adds the distribution of readings by arm and the sealed pre-registration bars. This is how tests on the commissioned chain will be reported.</p>
<p>Repository commit for this page: <code>{_e(R['sha'])}</code>.</p>"""

# ======================= shell =======================
CSS="""
:root{--bg:#0b0d12;--pn:#12161f;--tx:#e6e8ee;--mu:#9aa3b2;--ac:#7fa8cc;--ln:#2a3140}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
header{padding:18px 28px 10px;border-bottom:1px solid var(--ln);display:flex;justify-content:space-between;align-items:flex-end;gap:20px;flex-wrap:wrap}
header h1{margin:0;font-size:22px;letter-spacing:.3px}header h1 small{display:block;font-size:12px;color:var(--mu);font-weight:normal;margin-top:3px}
.aud{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--mu)}.aud button{background:#1b2130;color:var(--tx);border:1px solid var(--ln);padding:5px 12px;border-radius:14px;cursor:pointer}.aud button.on{background:var(--ac);color:#000;border-color:var(--ac)}
nav{display:flex;flex-wrap:wrap;gap:2px;padding:8px 28px;border-bottom:1px solid var(--ln);position:sticky;top:0;background:var(--bg);z-index:5}
nav button{background:none;border:none;color:var(--mu);padding:8px 12px;cursor:pointer;border-bottom:2px solid transparent;font-size:13px}nav button.on{color:#fff;border-bottom-color:var(--ac)}
main{padding:18px 28px 60px;max-width:1240px}section.tab{display:none}section.tab.on{display:block}
h2{font-size:19px;margin:6px 0 10px}h3{font-size:15px;margin:22px 0 8px;color:#cfd8ff}h4{margin:8px 0 4px;font-size:13px;color:var(--mu)}
table.t{border-collapse:collapse;width:100%;margin:6px 0 10px}table.t th{text-align:left;color:var(--mu);font-weight:normal;font-size:12px;border-bottom:1px solid var(--ln);padding:5px 8px}table.t td{padding:4px 8px;border-bottom:1px solid #1a1f2b;vertical-align:top}
table.t.small td,table.t.small th{font-size:11px;padding:2px 5px}table.kv td{padding:3px 12px 3px 0;color:var(--mu)}table.kv td:nth-child(even){color:var(--tx)}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.m{color:var(--mu);font-size:12px}.pend{color:#f2d27a}.flag{color:#f0b04a;font-size:11px;margin-left:6px}.sha{color:#5f6b80;font:11px monospace}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:26px}.gauge{background:var(--pn);border:1px solid var(--ln);border-radius:8px;padding:10px 14px;margin:8px 0}.gauge.muted{opacity:.7}.gl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:4px}
.tier{font-size:11px;font-weight:bold;color:#000;padding:2px 8px;border-radius:10px}.pl{color:var(--mu);font-size:12px}.explain{background:#0f1420;border-left:3px solid var(--ac);padding:10px 16px;margin:16px 0;font-size:13px}.explain p{margin:6px 0}
tr.placed td{color:#fff;font-weight:600}img.plate{width:100%;border-radius:6px;margin:8px 0}pre{background:#0f1420;border:1px solid var(--ln);padding:12px;overflow:auto;font-size:12px}details summary{cursor:pointer;color:var(--ac)}
.res{display:none}body.researcher .res{display:initial}body.researcher tr.res{display:table-row}body.researcher div.res{display:block}
footer{padding:14px 28px;border-top:1px solid var(--ln);color:var(--mu);font-size:12px}
@media print{body{background:#fff;color:#000}header,nav,footer,.aud{display:none}section.tab{display:none}section.tab.print{display:block;page-break-after:always}.gauge{border:1px solid #999;background:#fff}h3{color:#000}table.t th{color:#333}.m{color:#444}svg text{fill:#000}}
"""
JS="""
function tab(id){document.querySelectorAll('section.tab').forEach(s=>s.classList.toggle('on',s.id===id));document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));location.hash=id}
function aud(a){document.body.classList.toggle('researcher',a==='researcher');document.querySelectorAll('.aud button').forEach(b=>b.classList.toggle('on',b.dataset.a===a));localStorage.setItem('mp_aud',a)}
window.addEventListener('DOMContentLoaded',()=>{aud(localStorage.getItem('mp_aud')||'researcher');tab((location.hash||'#reading').slice(1))});
"""
TABS=[("reading","Reading",True),("cells","Every cell",True),("departure","Departure",True),("sky","Sky",True),("reference","Healthy reference",False),("integrity","Integrity",False),("chain","Chain",False),("physics","Physics",False),("story","Story",False),("record","Record",False),("run","Run",False)]

def refusals_from(o):
    r=[]
    for c,rec in o["classes"].items():
        if not rec.get("reportable"): r.append(f"class gauge '{c}': {rec.get('reason') or 'no commissioned band'} -> no placement, no tier")
    if not o["patient_sky"].get("available"): r.append("sky: no commissioned residual scale for this laboratory -> not rendered")
    if not o["departure"].get("reportable"): r.append("departure: "+str(o["departure"].get("status")))
    if o.get("lab_zero") is None: r.append("laboratory zero UNSET -> no absolute reading on any class")
    ca=o.get("cellular_age",{}); r.append(f"cellular age in years: not reported (single-array resolution ~{ca.get('resolution_yr','50')} yr, PROC-AGE-01) - the age-matched healthy reference is what the chain uses instead")
    return r

def build(o, out_html, sample_id="sample", percell_ref=None, percell_status="in build - 80 healthy arrays per laboratory through Stage 1 (started 2026-09-22)"):
    R=load_runtime(); wd=os.path.dirname(os.path.abspath(out_html)) or "."; os.makedirs(wd,exist_ok=True)
    sec={"reading":tab_reading(o,R,sample_id),"cells":tab_cells(o,R,percell_ref),"departure":tab_departure(o,R),"sky":tab_sky(o,R,sample_id,wd),
         "reference":tab_reference(R,percell_status),"integrity":tab_integrity(o,R,refusals_from(o)),"chain":tab_chain(R),"physics":tab_physics(),"story":tab_story(),"record":tab_record(R),"run":tab_run(R)}
    imm=o["classes"].get("immune",{}); head=(f"immune A'' {imm.get('A_abs')} · {imm.get('placement')} · {imm.get('tier')}" if imm.get("reportable") else "class gauge not reportable on this sample")
    nav="".join(f"<button data-t='{i}' onclick=\"tab('{i}')\">{n}</button>" for i,n,_ in TABS)
    body="".join(f"<section class='tab{' print' if p else ''}' id='{i}'>{sec[i]}</section>" for i,n,p in TABS)
    page=f"""<!doctype html><html><head><meta charset='utf-8'><title>MethylPhys CPG - {_e(sample_id)}</title><style>{CSS}</style><script>{JS}</script></head><body>
<header><h1>MethylPhys <span style='color:var(--ac)'>CPG</span> <small>Physics of Methylation: Landauer Metrology · Cellular Performance Gauge · the physics of methylation, read against a fixed zero</small></h1>
<div><div class='aud'>view: <button data-a='clinician' onclick="aud('clinician')">Clinician</button><button data-a='researcher' onclick="aud('researcher')">Researcher</button> <button onclick='window.print()'>Print report</button></div>
<div class='m' style='text-align:right;margin-top:6px'>{_e(sample_id)} · {head} · generated {time.strftime('%Y-%m-%d %H:%M')} · repo {_e(R['sha'])}</div></div></header>
<nav>{nav}</nav><main>{body}</main>
<footer>MethylPhys CPG · report generator in build (row 9, unsealed) · every number on the measurement tabs comes from cpg_conductor.run_full and the runtime files listed under Integrity · this is a measurement record, not a clinical interpretation · IAMPerformance · <a href='{GH}' style='color:var(--ac)'>repository</a></footer></body></html>"""
    open(out_html,"w",encoding="utf-8").write(page); return {"out":out_html,"bytes":len(page),"refusals":refusals_from(o)}

if __name__=="__main__":
    import argparse, pickle; ap=argparse.ArgumentParser(); ap.add_argument("--bundle",help="pickle or json of run_full output"); ap.add_argument("--out",default="methylphys_report.html"); ap.add_argument("--id",default="sample"); a=ap.parse_args()
    o=pickle.load(open(a.bundle,"rb")) if a.bundle.endswith(".pkl") else json.load(open(a.bundle)); print(build(o,a.out,a.id))
