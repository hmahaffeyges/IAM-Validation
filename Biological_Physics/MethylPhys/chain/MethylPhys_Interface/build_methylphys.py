#!/usr/bin/env python3
"""MethylPhys CPG - the researcher interface / report builder (row 9, in build 2026-09-22; UNSEALED until it runs end to end).

Renders ONE self-contained HTML from cpg_conductor.run_full's bundle plus the runtime files the chain actually read.
Tabs: Reading | Every cell | Departure | Sky | Healthy reference | Integrity | Chain | Physics | Story | Record | Run.
Two audiences (Clinician / Researcher) toggle on one run. Print reflows Reading + Every cell + Departure + Sky.
Vocabulary guard: the measurement tabs may not carry a disease name, a diagnosis, a verdict, or an age in years.
Author's report specification (Operations Manual, report chapter): cells detected and %, A per cell and per class with placement and tier
(tiers on the CLASS gauge, where they are commissioned), the Stage 5 departure with the laboratory's false-alarm rate,
the patient's sky, every flag; BREACH is a gauge reading (a temperature), not a comparison to any disease record.
"""
import os, sys, json, html, math, base64, hashlib, csv, re, time, glob, ast, subprocess

def _bio_root(start=None):
    """The directory that CONTAINS MethylPhys/ - i.e. Biological_Physics.

    Derived by ascending from this file until a directory holding 'MethylPhys' is found, rather than by counting
    dirname() calls. The 2026-09-22 move (CPG_Engine -> MethylPhys/chain, Testing_and_Code -> Record) changed the
    depth of every script by one; a counted chain resolved to MethylPhys and silently stopped finding Record/.
    """
    import os as _os
    d=_os.path.dirname(_os.path.abspath(start or __file__))
    for _ in range(8):
        if _os.path.isdir(_os.path.join(d,"MethylPhys")): return d
        if _os.path.basename(d)=="MethylPhys": return _os.path.dirname(d)
        nd=_os.path.dirname(d)
        if nd==d: break
        d=nd
    return _os.path.dirname(_os.path.dirname(_os.path.abspath(start or __file__)))

HERE=os.path.dirname(os.path.abspath(__file__)); ENGINE=os.path.dirname(HERE); BIO=_bio_root(); REPO=os.path.dirname(BIO)
RT=os.path.join(ENGINE,"Runtime Matrices")
def _find(name, root=BIO, allow_retired=False):
    for dp,_,fs in os.walk(root):
        if (not allow_retired and "RETIRED" in dp) or ".git" in dp: continue
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
    R["maps"]=_j(_find("beta_scale_maps_v1.json")); R["tiers"]=_j(_find("tier_breakpoints.json"))
    R["floors"]=_j(_find("presence_floors_v1.json")); R["ident"]=_j(_find("iamatlas_gauge_identity_loci_v1_0.json"))
    R["c2c"]=_j(_find("IAMAtlasREBUILD_celltype_to_class.json")); R["panels"]=_j(_find("directional_panels_v1_0.json"))
    R["files"]={n:_find(n, allow_retired=True) for n in ["beta_scale_maps_v1.json","tier_breakpoints.json","presence_floors_v1.json",
               "iamatlas_gauge_identity_loci_v1_0.json","iamatlas_celltype_markers_v0_2.json","IAMAtlasREBUILD_celltype_to_class.json","directional_panels_v1_0.json",
               "iamatlas_cpg_to_healpix_nside128.npz","cpg_conductor.py","cpg_gauge_engine.py","cpg_tiers.py","stage_4_6_patient_cmb.py","legacy_iam_deconvolver.py",
               "bidirectional_decomposition.py","iamatlas_a_scoring.py","stage_1_idat_calibration.py","IAMAtlasREBUILD_provenance.json",
               "IAMAtlasREBUILD.csv.xz","IAMAtlasREBUILD_celltype_to_class.json","iamatlas_cpg_to_healpix_nside128.npy","iamatlas_cpg_to_healpix_nside128.provenance.json",
               "RUNBOOK.md","CHAIN_COMMISSIONING.md","HANDOFF.md","nilc_celltype_deconvolver.py","run_sample.py","release_check.py",
               "test_gauge_switch.py","test_tiers.py","test_patient_sky.py","PROC_ANCHOR_01.py","PROC_FORMULA_01.py",
               "PROC_DECON_01.py","PROC_SEP_03.py","PROC_BIDIR_01.py","finding_check.py","lineage_splitter.py","README_CPG_Plates.md","README_HEALPix_Mapping.md","CPG_Gauge_Cell.png","CPG_Gauge_Cosmic.png","healthy_sky_vs_cmb.png"] if _exists(n)}
    sys.path.insert(0,ENGINE); import cpg_gauge_engine as _G
    R["hmin_table"]=_G.H_MIN_TABLE; R["sub_order"]=_G.SUB_ORDER; R["auc"]=_G.AUC_W
    _ts=R["tiers"]["tier_system_v1_2"]; R["warburg"]=next(x for x in _ts["tiers"] if x["tier_id"]=="WARBURG_TRANSITION")
    R["breach_line"]=_ts["breach_line_value"]; R["warburg_line"]=_ts["warburg_line_value"]
    import glob as _glob
    R["findings"]=[]
    for _fp in sorted(_glob.glob(os.path.join(BIO,"Record","VAL_FINDINGS","*_finding.json"))):
        try: R["findings"].append(_j(_fp))
        except Exception: pass
    try: R["inv"]=_j(_find("chain_inventory_v1.json"))
    except FileNotFoundError: R["inv"]=None
    try:
        _cg=_j(_find("iamatlas_collinearity_groups_v0_1.json")); R["cgroup"]=_cg["cell_to_group"]
        # each group value is a dict carrying members / classes / singleton - take the member list, not the keys
        R["cgmembers"]={g:(m.get("members") if isinstance(m,dict) else m) for g,m in _cg["groups"].items()}
        R["cgmeta"]=_cg.get("_metadata",{})
    except FileNotFoundError: R["cgroup"]={}; R["cgmembers"]={}
    try: R["excl"]=_j(_find("percell_exclusivity_v0.json"))["entries"]
    except FileNotFoundError: R["excl"]={}
    try: R["release"]=_j(_find("release_check.json"))
    except FileNotFoundError: R["release"]=None
    except FileNotFoundError: R["percell"]=None
    R["sha"]=_git_head()
    sys.path.insert(0,ENGINE); import cpg_tiers as T; sch=T.scheme(); R["tier_bands"]=sch["bands"]; R["tier_version"]=sch.get("version")   # ONE tier definition (PROC-TIER-01)
    return R
def _exists(n, allow_retired=True):
    try: _find(n, allow_retired=allow_retired); return True
    except FileNotFoundError: return False
TIER_COL={"SUPPRESSED":"#8fb3e6","NORMAL":"#86c28b","ELEVATED":"#f2d27a","SIGNIFICANTLY_ELEVATED":"#f0b04a","BREACH":"#d9736a"}
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
# MEASURE, DON'T COMPARE (author, 2026-09-27): the report describes this specimen's reading, the frozen constants and the
# calibration files by name. Nothing about any population. These words fail the render on the measurement tabs.
COHORT=re.compile(r"\b(cohorts?|donors?|healthy (band|range|reference|people|person|population|null|threshold|panel)|reference range|population (band|range)|central 95|percentiles?|p10|p90|middle 80|"
                  r"age (term|curve)|age-referenced|laboratory zero|lab zero|z_lab|placement|placed in|in_band|above_band|below_band|mahalanobis|a little under 1|compared with a healthy|against a healthy|ceiling|at_ceiling)\b",re.I)
# CHANGELOG GUARD (2026-09-22, at the author's instruction: "We are handing them a finished product not a log of
# my mistakes or changes"). The condition-name guard above protects against clinical overclaim; this one protects
# against the opposite failure - narrating the instrument's construction to someone who asked for the instrument.
# A finished report states what is true now. Build history lives in ROW9_WORKING_NOTE.md and the registers.
# Applied to EVERY tab, including the two exempt from the vocabulary guard, because those exemptions are for
# naming conditions and files, not for telling the reader about earlier drafts.
CHANGELOG=re.compile(r"\[updated\]|the author corrected|(?<![a-z])he corrected|corrected himself|my (first|earlier|own) "
                     r"(version|reading|wording|attempt|diagnosis|filter|headline|error|defect|output|mistake)|"
                     r"\bI had\b|\bI wrote\b|quietly rewritten|no longer true|stale-data failure|found independently|"
                     r"was caught (by|on)|supersede the (May|April|June) 20\d\d text|earlier draft", re.I)
def no_changelog(section_html, tab):
    txt=re.sub(r"<[^>]+>"," ",section_html)
    bad=sorted(set(m.group(0).lower() for m in CHANGELOG.finditer(txt)))
    if bad: raise ValueError(f"changelog guard [{tab}]: this is a finished report, not a build log -> {bad}")
    return section_html

def guard(section_html, tab):
    txt=re.sub(r"<[^>]+>"," ",section_html); bad=sorted(set(m.group(0).lower() for m in FORBIDDEN.finditer(txt)))
    if bad: raise ValueError(f"vocabulary guard [{tab}]: {bad}")
    coh=sorted(set(m.group(0).lower() for m in COHORT.finditer(txt)))
    if coh: raise ValueError(f"MEASURE, DON'T COMPARE guard [{tab}]: population vocabulary on a measurement tab -> {coh}")
    return no_changelog(section_html, tab)

def deepdive(R, topic=""):
    """The manual and the paper, linked wherever a reader may want to go further (author: they can deep dive all they want from the link)."""
    pdf="Biological_Physics/MethylPhys/manual/MethylPhys_CPG_Operations_Manual.pdf"
    # the paper: link the compiled PDF when one sits beside the source (the author compiles in Overleaf and commits it);
    # a reader who follows the link should get a document, not LaTeX source (author, 2026-09-27)
    import os as _os
    _pdf_name="Physics_of_Methylation__Landauer_Metrology.pdf"   # the author's Overleaf build, committed beside the .tex (2026-09-27)
    _pdf_local=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),"..","..","papers",_pdf_name)
    tex=("Biological_Physics/MethylPhys/papers/"+_pdf_name if _os.path.exists(_pdf_local)
         else "Biological_Physics/MethylPhys/papers/Landauer_Metrology_of_the_Methylome.tex")
    paper_label="Physics of Methylation: Landauer Metrology (PDF)" if _os.path.exists(_pdf_local) else "Physics of Methylation: Landauer Metrology (LaTeX source)"
    t=f" on {topic}" if topic else ""
    return (f"<div class='dd'><b>Go deeper{t}.</b> Everything on this tab is treated at full length in the Operations Manual - <a href='{GH}/blob/{R['sha']}/{pdf}' target='_blank'>MethylPhys CPG Operations Manual</a> "
            f"(the physics section, every sealed procedure with its outcome as found, the complete validation history, the reconciliation tables, the falsification register and the future-goals list) - and in the short methods paper, "
            f"<a href='{GH}/blob/{R['sha']}/{tex}' target='_blank'>{paper_label}</a>. Both live in the same repository as the code that produced this page, at the same commit.</div>")

def posbar(A, lo, hi, w=150):
    """Where this reading sits against that cell's OWN healthy range. Green inside, amber within half a
    band-width outside, red beyond. Carries no tier: tiers exist only on the class gauge."""
    if A is None or lo is None or hi is None: return ""
    span=max(hi-lo,1e-6); mid=(lo+hi)/2.0; d=(A-mid)/span
    col="#3fa45b" if abs(d)<=0.5 else ("#d68910" if abs(d)<=1.0 else "#c0392b")
    lo_v,hi_v=mid-2.0*span, mid+2.0*span
    X=lambda v:(min(max(v,lo_v),hi_v)-lo_v)/(hi_v-lo_v)*w
    gl,gr=X(lo),X(hi)
    return (f"<svg viewBox='0 0 {w} 14' width='{w}' height='14' style='vertical-align:middle'>"
            f"<rect width='{w}' height='14' fill='#1b1f2a'/><rect x='{gl:.1f}' width='{max(gr-gl,1):.1f}' height='14' fill='#24422e'/>"
            f"<line x1='{X(mid):.1f}' y1='0' x2='{X(mid):.1f}' y2='14' stroke='#3fa45b' stroke-width='1'/>"
            f"<circle cx='{X(A):.1f}' cy='7' r='4.2' fill='{col}'/></svg>")

# (COLS_LEGEND, the June marker-surface column legend, removed 2026-09-27: unused, and it described a healthy range)

# ======================= measurement tabs =======================
# (GAUGE_EXPLAINER, the class-gauge ruler explainer with the three corrections, removed 2026-09-27: unused, and it described the retired layers)

def _trace_one_liner(o):
    """One line on the Reading tab. A reader who opens one tab should see what Stage 2c found."""
    td = o.get("trace_detection") or {}
    m = td.get("_meta") or {}
    if not m.get("available") or m.get("retired"):
        return ""
    if not m.get("calibrated_for_this_substrate", True):
        return ("<p class='m'><b>Trace-class detection:</b> not calibrated for this substrate "
                "(the thresholds are a whole-blood measurement), so no call is made here. The statistics "
                "are on the Every cell tab.</p>")
    hits = [c for c in ("secretory", "cycling") if (td.get(c) or {}).get("detected")]
    if hits:
        return ("<p class='warn'><b>Trace-class detection: evidence of epithelial-like material</b> "
                "(statistic above the healthy threshold for %s). At this limit the class cannot be named - "
                "attribution needs about 5 %%. No fraction and no A are reported for it. Full numbers on the "
                "Every cell tab.</p>" % ", ".join(hits))
    return ("<p class='m'><b>Trace-class detection:</b> no evidence of secretory or cycling material above "
            "the healthy null (%s). This is a separate test from the composition above, which cannot see a "
            "component this small. Full numbers on the Every cell tab.</p>"
            % ", ".join("%s t=%s vs %s" % (c, (td.get(c) or {}).get("t"), (td.get(c) or {}).get("threshold"))
                        for c in ("secretory", "cycling") if c in td))


try:
    import cpg_tiers as T
except Exception:
    T=None

def tab_reading(o, R, sid):
    """THE PHYSICS FRONT PAGE (author, 2026-09-26). What is in the sample and how much of each cell; then each detected
    cell's A against its architecture class's H_min. Healthy is A = 1.00 by the physics and the tier scale's NORMAL
    (0.95-1.05) is the tolerance; departure from healthy is each cell's distance from 1.00. No pooled class A is printed
    as a reading: the class gauge runs as an internal gate (is this specimen blood-like?) and is named as such in one
    sentence. Nothing on this page places the patient among other people."""
    H=[]; ctx=o.get("context",{}); cls=o.get("classes") or {}; cells=o.get("cells_all") or {}
    fd=o.get("foreign_detection") or {}; lab=o["patient_sky"].get("lab") or (o.get("cfg") or {}).get("lab","-")
    det_state=("commissioned" if str(fd.get("status","")).startswith("OK") else ("not commissioned" if str(fd.get("status","")).startswith("NOT_COMMISSIONED") else _e(fd.get("status") or "not run")))
    H.append(f"<h2>Reading - {_e(sid)}</h2><table class='kv'><tr><td>Specimen</td><td>{_e(ctx.get('substrate','whole blood'))}</td><td>Substrate</td><td>DNA methylation (450K/EPIC beta)</td></tr>"
             f"<tr><td>Declared age</td><td>{_e(ctx.get('age','-'))}</td><td>Pipeline map</td><td>{_e(o.get('scale','-'))}</td></tr>"
             f"<tr><td>Laboratory</td><td>{_e(lab)}</td><td>Foreign-cell detection</td><td>{det_state} for this laboratory</td></tr></table>")
    H.append("<p><b>Healthy is A = 1.00.</b> Every cell type has one physical floor, the H_min of its architecture class, and a healthy cell of any type reads "
             "A = H / H_min = 1.00 on it. NORMAL is 0.95-1.05; below 0.95 is SUPPRESSED; 1.05-1.07 ELEVATED; the 1.07 Warburg line and the 1.10 breach line are "
             "the physics' two inflection points. The reading below is each cell in this sample against its own floor. Nothing about any population is part of any number on this page.</p>")
    # ---- 1. cells found, and each one's A
    fam_seen=set(); rows=[]; sub=[]; unres=[]
    for name,v in cells.items():
        if not isinstance(v,dict): continue
        if v.get("resolvable") is False: unres.append(name); continue
        fr=v.get("fraction") or 0.0
        key=v.get("shared_with") or name
        if key in fam_seen: continue
        if not v.get("present"):
            if fr>0: sub.append((name,fr))
            continue
        fam_seen.add(key); rows.append((key,name,v,fr))
        # every other member of the family: same fraction, its OWN A (2026-09-26 - the first pass hid the neutrophil A behind the eosinophil row)
        if v.get("shared_with"):
            for other,ov in cells.items():
                if other!=name and isinstance(ov,dict) and ov.get("shared_with")==key and ov.get("present"): rows.append((key+" / "+other,other,ov,fr))
    rows.sort(key=lambda r:-r[3])
    H.append("<h3>1. What is in the sample, and how each cell reads</h3>")
    if not rows:
        H.append("<p class='pend'>No cell cleared its presence floor - nothing is scored.</p>")
    else:
        H.append("<table class='t'><tr><th>cell</th><th>class</th><th>fraction</th><th>A</th><th>95 % interval on the reading</th><th>tier</th></tr>")
        outside=[]
        for key,name,v,fr in rows:
            A=v.get("A"); cl=v.get("class"); hm=(R.get("ident") or {}).get(cl,{}).get("H_min")
            try: tier,_=T.tier_of(A, True, hm) if A is not None else (None,None)
            except Exception: tier=None
            d=(A-1.0) if A is not None else None
            ci=v.get("reading_ci") or {}; ci_s=(f"{ci['ci_lo']:.4f} - {ci['ci_hi']:.4f}" if ci.get("ci_lo") is not None else "-")
            label=(_e(name)+f" <span class='m'>(in family {_e(str(v.get('shared_with')).replace('family:',''))} - one shared fraction, own A)</span>") if v.get("shared_with") else _e(name)
            H.append(f"<tr><td>{label}</td><td class='m'>{_e(CLASS_LABEL.get(cl,cl))}</td><td class='n'>{100*fr:.1f} %</td>"
                     f"<td class='n'><b>{A:.4f}</b></td>" if A is not None else f"<tr><td>{label}</td><td class='m'>{_e(CLASS_LABEL.get(cl,cl))}</td><td class='n'>{100*fr:.1f} %</td><td class='n'>-</td>")
            H.append(f"<td class='n m'>{ci_s}</td>"
                     +(f"<td><span class='tier' style='background:{TIER_COL.get(tier,'#555')}'>{_e(tier)}</span></td>" if tier else "<td class='m'>not scored</td>")+"</tr>")
            if tier and tier!="NORMAL": outside.append((key,A,tier))
        H.append("</table>")
        n_normal=sum(1 for k,n,v,fr in rows if v.get("A") is not None)-len(outside)
        H.append(f"<p><b>{len(rows)} cell{'s' if len(rows)!=1 else ''} found above the presence floor.</b> {n_normal} read NORMAL. "
                 +(("Outside NORMAL: "+"; ".join(f"<b>{_e(k)}</b> at A = {a:.4f} ({_e(t)}, {'below' if a<1 else 'above'} 1.00 by {abs(a-1):.4f})" for k,a,t in outside)+".") if outside else "None outside NORMAL.")+"</p>")
    H.append(f"<p class='m'>{len(sub)} atlas cell{'s' if len(sub)!=1 else ''} placed below the presence floor - detected in trace amounts, not scored (a cell must be present to be measured; fraction is a detection gate, not a correction to A). "
             f"{len(unres)} entr{'ies' if len(unres)!=1 else 'y'} not resolvable on this platform. Every cell, scored or not, is on the <b>Every cell</b> tab. "
             "A is H(mean beta over the cell's identity loci) / H_min of its class, on scale-mapped betas. Nothing is added to or subtracted from it. "
             "The tier is read on A against 1.00 on the tier scale (NORMAL 0.95-1.05). Each cell's 95 % interval and identifiability are on the <b>Every cell</b> tab.</p>")
    # ---- 2. foreign cells (Stage 2d)
    H.append("<h3>2. Foreign cells - is there anything in this blood that is not blood?</h3>")
    st=str(fd.get("status") or "")
    if st=="OK":
        det=fd.get("detected") or []
        H.append(("<p><b>Detected above this laboratory's own line: "+_e(", ".join(det))+".</b> A presence statement, not a clinical statement; the cell's A is read only where it clears its presence floor.</p>") if det
                 else "<p><b>No foreign template above its noise floor.</b></p>")
    elif st.startswith("OK_BUT_UNSPECIFIC"):
        H.append(f"<p class='pend'><b>Unspecific.</b> {_e(st.split(': ',1)[-1])} Every foreign column rising together is what a specimen that is not blood-like looks like; no single detection is read.</p>")
    elif st.startswith("NOT_COMMISSIONED"):
        H.append("<p class='pend'>Detection is <b>not commissioned</b> for this laboratory: no line is borrowed from another. The composition table above is the only statement about foreign cells. "
                 "Commission it on &ge; 36 of this laboratory's own healthy whole-blood arrays (kit/commission_detection_lab.py).</p>")
    else:
        H.append(f"<p class='pend'>{_e(st or 'Stage 2d not run on this bundle.')}</p>")
    H.append(_trace_one_liner(o))
    # ---- 3. the instrument
    imm=cls.get("immune") or {}
    H.append("<h3>3. The instrument</h3>")
    H.append(f"<p><b>Laboratory {_e(lab)}</b> &middot; pipeline map <code>{_e(o.get('scale','-'))}</code> (this laboratory's processing route onto the atlas scale, fitted on paired reads) &middot; "
             f"foreign-cell detection {det_state}"+(f" (panel n = {fd.get('panel_n')})" if fd.get("panel_n") else "")+". "
             "The commissioning panel calibrates the instrument - what this scanner and this processing do to a known input - and never the definition of healthy.</p>")
    H.append(f"<p class='m'>Internal gate: the pooled class gauge runs only to ask whether this specimen is blood-like (composition verified: <b>{_e(imm.get('composition_verified'))}</b>"
             +(f", foreign fraction {imm.get('foreign_fraction')}" if imm.get("foreign_fraction") is not None else "")+"). "
             "A pooled class A is not a reading and is not printed; the cells are.</p>")
    H.append(deepdive(R,"the gauge"))
    return guard("".join(H),"Reading")

def _trace_block(o):
    """Stage 2c - trace-class detection. Presence only: no fraction, no A, no tier below the attribution
    limit, because a trace class cannot be scored in this substrate at any fraction a blood draw presents."""
    td = o.get("trace_detection") or {}
    m = td.get("_meta") or {}
    if m.get("retired"):
        return []
    if not m.get("available"):
        return ["<h3>Trace-class detection</h3>",
                "<p class='m'>Not run for this specimen: %s</p>" % (m.get("reason") or "no panel")]
    H = ["<h3>Trace-class detection - is there any epithelial-like material here at all?</h3>",
         "<p>The composition solve above cannot answer this. It is a non-negative fit, and a non-negativity "
         "constraint pins a component that small at exactly zero (PROC-TRACE-01: a spike below 5 % returned 0.00 "
         "on two of three constructed mixtures). This panel asks the question the "
         "other way round: fit the specimen <i>without</i> the class, then test whether the class's own "
         "profile explains what is left over, weighting each address by how well the atlas pinned it down. "
         "The threshold per class is a detector constant from the trace panel file, set before the method was chosen (PROC-TRACE-01).</p>"]
    if not m.get("calibrated_for_this_substrate", True):
        H.append("<p class='warn'><b>Uncalibrated on this substrate.</b> The thresholds are a whole-blood "
                 "measurement (declared here: %s). The statistic is printed for reference only and no "
                 "call is made here.</p>" % (m.get("substrate_declared") or "not declared"))
    H.append("<table><tr><th>class</th><th>t</th><th>threshold</th><th>panel median</th>"
             "<th>what this says</th></tr>")
    for c in ("secretory", "cycling"):
        r = td.get(c) or {}
        if not r:
            continue
        H.append("<tr><td class='m'>%s</td><td class='m'>%s</td><td class='m'>%s</td><td class='m'>%s</td>"
                 "<td>%s</td></tr>" % (c, r.get("t"), r.get("threshold"), r.get("null_median_t"),
                                       r.get("verdict", "-")))
    H.append("</table>")
    H.append("<p class='m'>Limits, measured: material of this kind is detectable from about <b>%s %%</b>, and "
             "naming <i>which</i> class requires about <b>%s %%</b> - at the lower limit a cycling component "
             "lifts the secretory statistic marginally over its own threshold, so the two are not separable "
             "there. No fraction and no A are reported for a detected trace class: at 20 %% admixture a "
             "class's own identity loci still read 0.85 against 0.99 for a pure specimen, so the entropy "
             "measured there is the background's, not the class's.</p>"
             % (100 * m.get("detection_limit", 0.02), 100 * m.get("attribution_requires", 0.05)))
    return H


def _detection_block(o):
    """Stage 2d, foreign-cell detection - read from the bundle, never recomputed here. Adopted by the author's decision
    2026-09-26 and scoped to laboratories with a commissioned panel; the status string says exactly why nothing is
    printed when nothing is."""
    fd=o.get("foreign_detection")
    H=["<h3>Foreign-cell detection (Stage 2d)</h3>", "<p>One joint fit of this specimen's panel markers on the blood reference and all 21 foreign templates; each template's floor is the instrument's noise floor, the 0.99 quantile over "+str((fd.get('noise_floor') or {}).get('n_arrays') or '')+" arrays known to lack the cell (PROC-STAGE2D-03). A cell is named when one template clears its floor; three or more together read as epithelial-like material with the cell not resolved (measured on 1.9 % of healthy arrays, the oldest). A detected foreign cell prints its fraction; its A is not read below a scoring floor still to be tested.</p>"]
    if not fd:
        H.append("<p class='pend'>NOT RUN - this bundle predates Stage 2d.</p>"); return "".join(H)
    st=fd.get("status") or ""
    H.append("<p class='m'>A foreign cell is one that does not belong to whole blood. For each, the chain fits the specimen's own blood background, "
             "takes the residual, and asks how much of that cell's atlas profile is in it, weighting every locus by how quiet it is in healthy blood "
             "at commissioned laboratories. The reading is f&#770; (an estimated fraction), with the laboratory's own line above which the chain says "
             "<b>detected</b>. The line is drawn from that laboratory's commissioning panel and from nothing else; a laboratory without a panel gets no line.</p>")
    if not st.startswith("OK"):
        H.append(f"<p class='pend'>{_e(st)}</p>")
        if fd.get("commissioned_laboratories"): H.append(f"<p class='m'>Laboratories with a commissioned detection panel: {_e(', '.join(fd['commissioned_laboratories']))}.</p>")
        return "".join(H)
    if st.startswith("OK_BUT_UNSPECIFIC"):
        H.append(f"<p class='pend'><b>Unspecific.</b> {_e(st.split(': ',1)[-1])} No single cell is reported as detected from this pattern.</p>")
    det=fd.get("detected") or []
    H.append(f"<p>Laboratory <b>{_e(fd.get('laboratory'))}</b> &middot; panel n = {fd.get('panel_n')} &middot; {fd.get('n_markers_used')} markers &middot; line rule: {_e(fd.get('line_rule') or '')}. "
             + (f"<b>Detected: {_e(', '.join(det))}</b>." if det and not st.startswith("OK_BUT_UNSPECIFIC") else ("<b>Unspecific - every column rose together; no single detection is read.</b>" if st.startswith("OK_BUT_UNSPECIFIC") else "<b>No foreign cell above its line.</b>")) + "</p>")
    H.append("<table class='t'><tr><th>foreign template</th><th>f&#770;</th><th>noise floor</th><th>standing bias on blood</th><th>detected</th><th>measured detection limit (90 %)</th></tr>")
    cells=fd.get("cells") or {}
    order=sorted(cells, key=lambda c: -(cells[c].get("f_hat") or 0))
    def _n(x, d=4): return "" if x is None else (f"{x:.{d}f}" if isinstance(x,(int,float)) else _e(str(x)))
    for c in order:
        r=cells[c]; lim=r.get("measured_detection_limit")
        H.append(f"<tr><td>{_e(c)}</td><td class='n'>{_n(r.get('f_hat'),5)}</td><td class='n'>{_n(r.get('line'),5)}</td><td class='n'>{_n(r.get('standing_bias'),5)}</td>"
                 f"<td>{'<b>yes</b>' if r.get('detected') else 'no'}</td><td class='n'>{('not resolved at 10 %' if lim is None else f'{float(lim)*100:.0f} %')}</td></tr>")
    for c,r in (fd.get("not_detectable") or {}).items():
        H.append(f"<tr><td>{_e(c)}</td><td class='n'>{_n(r.get('f_hat'),5)}</td><td colspan='4'>NOT DETECTABLE on this block - {_e(r.get('reason') or '')}</td></tr>")
    H.append("</table>")
    H.append("<p class='m'>The measured detection limit is the smallest spiked fraction the detector found in at least 90 % of trials at this false-positive rate "
             "(PROC-MF-02/03, four 450K laboratories). A cell without one has a line but no measured sensitivity; a detection on it is a reading to follow up, not a result. "
             "A detection of a foreign cell is a statement about <i>presence</i>; the cell's A-score is a separate reading and is printed only where the cell clears its presence floor.</p>")
    return "".join(H)


def tab_cells(o, R, percell_ref=None):
    # 2026-09-27: what each present cell IS (the author's biology, guarded) - one line under the table, from the runtime file
    try: _CD=_j(_find("cell_descriptions_v1.json"))["cells"]
    except Exception as _ex: _CD={}; print("cells tab: cell_descriptions_v1.json not loaded:", type(_ex).__name__, str(_ex)[:80])
    sys.path.insert(0,ENGINE); import cpg_tiers as T
    lab=(o.get('patient_sky') or {}).get('lab') or (o.get('cfg') or {}).get('lab')
    cells=o.get("cells_all") or {}; comp_cells={r["cell"]:r for r in o["composition"]["celltype"]}
    H=["<h2>Every cell - all 115 atlas cell types</h2>"] + _trace_block(o) + [
       "<p><b>One surface, one formula, one gauge.</b> A = H(mean beta over the cell's IDENTITY loci) / H_min of its architecture class "
       "(RULING A3; LESSON-SURFACE-01). H_min is the minimum entropy a cell of that class must hold to keep its identity; A is where the cell sits on that bar - "
       "its informational fidelity against its own minimum requirement. The gauge reads 1.00 when the cell holds exactly what its identity costs; the tier scale "
       "(NORMAL 0.95-1.05) is the tolerance about that point. Nothing about any person or group is part of this number. Where people of a given age "
       "sit on the gauge is a separate observation about people, reported as such and never applied to a cell.</p>",
       "<details open><summary><b>What each column is, and why an instrument prints it</b></summary><table class='t'><tr><th>column</th><th>what it is</th><th>why it is on the page</th></tr>"
       "<tr><td>cell type</td><td>one of the 115 atlas cell types</td><td>the object being measured - a gauge reads one object at a time</td></tr>"
       "<tr><td>present</td><td>the cell's fraction is above its presence floor</td><td>the detection gate. Below it the cell's identity loci carry OTHER cells' bytes and read by artefact (PROC-CEIL-01). A scale with nothing on it prints no weight</td></tr>"
       "<tr><td>fraction</td><td>the deconvolved share of the sample's DNA that is this cell</td><td>how much of the object is on the scale - so the reading is known to be OF this cell. Never a correction to A</td></tr>"
       "<tr><td>A</td><td>H(mean beta over the cell's identity loci) / H_min of its class</td><td>the measurement. Landauer floor in the denominator, measured entropy in the numerator; absolute and dimensionless</td></tr>"
       "<tr><td>95 % interval on the reading</td><td>resample THIS array's identity loci for the cell</td><td>repeatability of this reading - how far A moves if the cell's loci had been sampled differently. From the specimen, not from anyone else</td></tr>"
       ""
       "<tr><td>tier</td><td>A against 1.00 on the tier scale (tier_breakpoints.json)</td><td>the tolerance about the fixed point, one file, one function</td></tr>"
       "<tr><td>identity loci on this platform</td><td>how many of the cell's identity addresses exist on this array type (450K or EPIC), out of the atlas's list for the cell</td><td>coverage of the instrument, NOT a finding: every address reads something whether or not the cell is in the sample. Presence is the separate question answered by the fraction</td></tr>"
       "<tr><td>identifiability</td><td>exclusive loci against the nearest rival cell, and whether the cell is solved alone, as a family, or not at all on this platform</td><td>what the detector can and cannot distinguish. Says whether 'this is a CD4 T-cell' is a defensible statement or a shared reading</td></tr>"
       "</table></details>",
       "<p><b>A cell below its presence floor gets no A and no tier.</b> PROC-CEIL-01 measured why: an absent cell's identity addresses carry the specimen's other cells, "
       "which average near a coin flip, so an absent cell reads high or low by artefact. Its row says <i>not present - not scored</i>. Fraction is a detection gate, never "
       "a correction to A. A resolution family shares ONE fraction (the array cannot split it) but every member is scored on its own identity loci and prints its own A.</p>"]
    # 2026-09-26 RESOLVABILITY: a resolution family (cells the array cannot tell apart) is ONE measurement, never several;
    # a cell defined on < 1% of the platform's loci is 'not resolvable', never 'fraction 0'; a lower-coverage copy of a
    # solved cell is named as that cell's twin.
    _res=(o.get('composition') or {}).get('resolvability') or {}
    _unres=set(_res.get('unresolvable') or []); _twins=_res.get('twins_dropped') or {}; _fams=_res.get('families') or {}; _shared=_res.get('shared') or {}
    H.append("<h3>What limits a per-cell claim</h3>"
      "<p><b>Resolvability is a property of the atlas and the platform, not of your sample.</b> Two cells whose atlas profiles cannot be told apart on this array's loci are solved as one <i>resolution family</i> with one shared fraction; each member still prints its own A on its own identity loci. A cell whose atlas entry is defined on too few of this platform's loci is <i>not resolvable</i> and is listed, never printed as fraction 0. The <b>exclusive loci</b> column counts identity addresses at which this cell differs from its nearest rival by more than 0.2 beta - the more there are, the more the reading is about this cell and not a shared block.</p>")
    H.append("<p class='m'><b>Resolvability.</b> "+(f"{len(_unres)} atlas entries are defined on under 1 % of this platform's loci and are not solved for (listed at the end); " if _unres else '')
             +(f"{len(_twins)} lower-coverage copies of solved cells were folded into their originals ({', '.join(f'{k} = {v}' for k,v in sorted(_twins.items()))}); " if _twins else '')
             +(('<b>resolution families</b>, solved as one column with one fraction shared by every member: '+'; '.join(' + '.join(v) for v in _fams.values())) if _fams else 'no resolution families')+'.</p>')
    H.append(_detection_block(o))   # Stage 2d, 2026-09-26

    by={}; 
    for cell,r in cells.items(): by.setdefault(r.get("class","?"),[]).append((cell,r))
    for c in CLASSES:
        rows=sorted(by.get(c,[]), key=lambda kv:-(kv[1].get("A") or 0))
        hm=R["ident"].get(c,{}).get("H_min")
        if not rows: continue
        H.append(f"<h3>{CLASS_LABEL[c]} <span class='m'>({len(rows)} cells; H_min {R['ident'].get(c,{}).get('H_min','-')})</span></h3><table class='t cells'><tr><th>cell type</th><th>present</th><th>fraction</th><th>A</th>""<th>95 % interval on the reading</th><th>tier</th><th>identity loci on this platform</th><th>identifiability</th></tr>")
        for cell,r in rows:
            if cell in _unres or cell in _twins: continue   # listed separately below, never as fraction 0
            A=r.get("A"); fr=r.get("fraction") or 0
            ci=r.get("reading_ci") or {}
            cis=("%.3f - %.3f"%(ci["ci_lo"],ci["ci_hi"])) if ci.get("ci_lo") is not None and str(ci.get("surface","")).startswith("identity") else "<span class='m'>-</span>"
            nf=("%d of %d"%(ci["n_loci_found"],ci["n_loci_panel"])) if ci.get("n_loci_found") else ("%d of %d"%(r.get("n_markers_matched",0),r.get("n_markers_expected",0)) if r.get("n_markers_expected") else "-")
            present=bool(r.get("present"))
            if present and A is not None:
                gt,_=T.tier_of(A, True, hm); tier=f"<span class='tier' style='background:{TIER_COL.get(gt,'#555')}'>{_e(gt)}</span>"
            else: tier="<span class='m'>not present - not scored</span>"
            ex=r.get("exclusive_markers"); in_block=(ex is not None) or (cell in _shared) or (fr>0)
            exs=(f"{ex} exclusive loci" if ex is not None else "")
            gid=(R.get("cgroup") or {}).get(cell); mem=(R.get("cgmembers") or {}).get(gid) or []
            res=(f"family: {_e(str(_shared[cell]).replace('family:',''))}" if cell in _shared else (("not separable from "+_e(", ".join(m for m in mem if m!=cell)[:60])) if len(mem)>1 else "solved alone"))
            ident=(f"{exs} &middot; {res}" if exs else (res if in_block else "not in this platform's solve block (defined on too few of its loci) - fraction is not solved for"))
            Ashow=(f"{A:.4f}" if present and A is not None else "<span class='m'>-</span>")
            H.append(f"<tr class='{'placed' if present else ''}'><td>{_e(cell)}</td><td>{'yes' if present else '-'}</td><td class='n'>{100*fr:.1f} %</td><td class='n'>{Ashow}</td>"f"<td class='n'>{cis if present else '-'}</td><td>{tier}</td><td class='n'>{nf}</td><td class='m'>{ident}</td></tr>")
        H.append("</table>")
    if _unres:
        H.append("<details><summary class='m'>"+f"{len(_unres)} atlas entries not resolvable on this platform (defined on under 1 % of its loci) - kept in the atlas as reference, not solved for"+"</summary><p class='m'>"+', '.join(sorted(_unres))+"</p></details>")
    bd=o.get("bidirectional",{}); H.append("<details><summary class='m'>Direction - Stage 4.5 bidirectional composite (a panel-derived quantity kept for the record, not a reading: a per-cell A below or above 1.00 already carries direction)</summary><p>Pooled entropy folds hypo- and hyper-methylation together; the signed composite keeps the sign, per sealed panel. Panels exist only where one was sealed (immune, VAL-051 / CPG-VAL-019); the other classes say so.</p><table class='t'><tr><th>class</th><th>signed composite</th><th>pooled A on the panel</th><th>panel</th><th>reading</th></tr>")
    for c in CLASSES:
        b=bd.get(c,{}); ad=b.get('a_directional'); ap=b.get('a_pooled'); ads=('' if ad is None else '%+.3f'%float(ad)); aps=('' if ap is None else '%.3f'%float(ap))
        if ad is None: rd="no sealed directional panel for this class"
        else:
            rd=("pooled and signed composite both move; the signed composite is "+("in" if float(ad)>0 else "opposite to")+" the sealed panel's direction" if abs(float(ad))>=0.5 or bool(b.get('flag_bidirectional')) else "within baseline on the sealed panel")
            rd+=f" (panel: {_e(R['panels'].get('immune',{}).get('source','VAL-051 Rule A, 7 CpGs'))})"
        H.append(f"<tr><td>{CLASS_LABEL[c]}</td><td class='n'>{ads}</td><td class='n'>{aps}</td><td>{b.get('n_covered',0)} CpGs</td><td class='m'>{rd}</td></tr>")
    H.append("</details>")
    if _CD:
        _pres=[(c,v) for c,v in (o.get('cells_all') or {}).items() if isinstance(v,dict) and v.get('present') and _CD.get(c,{}).get('what')]
        if _pres:
            H.append("<h3>What the cells found in this specimen are</h3><p class='m'>The author's biology, one paragraph per cell present, from <code>cell_descriptions_v1.json</code>; the full entries for all cells are the Operations Manual's cells chapter. Nothing here is a reading.</p>")
            for c,v in sorted(_pres, key=lambda kv:-(kv[1].get('fraction') or 0)):
                e=_CD[c]; H.append(f"<p><b>{_e(e.get('title') or c)}</b>" + (f" <span class='m'>({_e(e['other_names'])})</span>" if e.get('other_names') else '') + f" - {_e(e['what'])}" + (f" <i>{_e(e['where'])}</i>" if e.get('where') else '') + "</p>")
    H.append("</table>"); return guard("".join(H),"Every cell")

# tab_departure: removed 2026-09-26 by the author's decision - the Mahalanobis departure is a cohort distance on the pooled class, not a reading

SKY_WHY = ("<h2>The sky - what it is, why it is a cosmologist's object, and what it buys a geneticist</h2>"
 "<p>This page is not an analogy. Every step below is a measurement problem cosmologists had to solve before the microwave "
 "background could be read at all, and each has a counterpart this chain already runs. The point of putting a methylome on a "
 "sphere is not that the picture resembles theirs - it is that <b>thirty years of machinery for reading a noisy field on a "
 "sphere becomes available</b>, with the names changed.</p>"

 "<h3>A sky map is not a photograph</h3>"
 "<p>This is the first thing cosmology had to unlearn. What Planck delivers is not a picture of the sky; it is <i>the residual "
 "left after a model is subtracted from a measurement</i>, at a declared resolution, with a mask over the parts that cannot be "
 "measured, and a noise level quoted per pixel. The famous image is the last step of a long pipeline, and every step of that "
 "pipeline had to be invented. Two of those steps decide everything: the monopole (2.725 K) and the dipole (the Earth's own "
 "motion through the radiation) are real, enormous, and <i>not the signal</i>. Remove them wrongly and the anisotropy - one part "
 "in 100,000 - is buried a thousand times over. That is not a subtlety; it is the measurement.</p>"

 "<h3>The correspondence, step by step</h3>"
 "<table class='t'><tr><th>what cosmology had to learn</th><th>why it mattered</th><th>what it is in this chain</th></tr>"
 "<tr><td>Subtract the monopole and dipole before anything else</td><td>they are real and they are not the signal</td>"
 "<td>nothing is subtracted from the sky or from a cell's A. The residual is this specimen against its own composition expectation; A = 1.00 is the physical fixed point. Whether the array's own probes can tare the instrument is PROC-TARE-01's question (sealed, not commissioned)</td></tr>"
 "<tr><td>Mask the galaxy</td><td>part of the sky cannot be measured; do not guess what is behind it</td>"
 "<td>the <b>presence floors</b>. <b>Below its floor a class is not there</b> - the specimen holds no detectable amount of it - so the chain "
 "masks it black and reports nothing about it. <i>The analogy is close but not exact, and the difference is worth stating:</i> the galaxy hides a "
 "sky that really is behind it, whereas an absent class has nothing behind the mask. Both are refusals to report where the instrument cannot see, "
 "for different reasons. PROC-CEIL-01 measured what happens without the refusal: on whole blood the absent classes read high by artefact, "
 "because their identity addresses carry other cells' DNA and average near a coin flip - so an unmasked healthy sample would report two "
 "classes past breach</td></tr>"
 "<tr><td>Component separation - dust, synchrotron, free-free</td><td>you cannot read the sky until you separate what lies in "
 "front of it</td><td><b>Stage 2, the composition step.</b> Not a loose parallel: the second solver in this chain <i>is</i> NILC, "
 "the needlet internal linear combination Planck used, pointed at cell types instead of foregrounds</td></tr>"
 "<tr><td>Deconvolve the instrument beam</td><td>resolution is finite and must be declared, not assumed</td>"
 "<td>about 2.2 CpG addresses per pixel at NSIDE 128 is this instrument's beam. The smoothed panel in the figure below is the "
 "beam-smoothed map, which is the only fair comparison to a published CMB image</td></tr>"
 "<tr><td>A noise covariance per pixel, not one number for the map</td><td>pixels are not equally trustworthy</td>"
 "<td>each address's <b>own sigma</b>: the atlas posterior at that address combined with this array's own SNP-probe noise - the denominator of every z on this page. No other person's array enters it</td></tr>"
 "<tr><td>Score anomalies against simulations, never an analytic null</td><td>a real sky carries built-in correlations, so a "
 "Gaussian null is simply the wrong null</td><td>the <b>spatially-shuffled null</b> - and this chain has now measured exactly why "
 "it is required (the warning further down)</td></tr>"
 "<tr><td>Then, and only then, the angular power spectrum</td><td>the <i>scale</i> of the structure is where the physics lives</td>"
 "<td><b>not built - this is the frontier.</b> What it would be worth is the next paragraph</td></tr></table>"

 "<h3>What this buys a geneticist that a list of differentially methylated regions does not</h3>"
 "<p>Epigenomics today answers <i>which CpGs moved</i>: per-site tests with a false-discovery correction, or differentially "
 "methylated regions found in windows whose size is chosen in advance. Both force you to pick a scale before you look. A spectrum "
 "asks a different question - <b>at what genomic scale does this departure live?</b> - and answers it at every scale at once with "
 "no window chosen. The multiple-comparison problem comes with it: cosmology calls it the <i>look-elsewhere effect</i> and has "
 "spent decades on it, and its answer is simulation rather than dividing an alpha by a large number.</p>"
 "<p>The spatial correlation measured here is the first point of that spectrum. A healthy methylome has a characteristic "
 "correlation scale. <b>If a condition changes that scale rather than the value at any single address, a spectrum sees it and no "
 "per-site test can.</b> That is a new and falsifiable observable, and it is the reason this is a method rather than a picture.</p>"

 "<h3>One thing a geneticist has that a cosmologist would trade almost anything for</h3>"
 "<p>There is one microwave sky. It cannot be re-observed, it will not change, and every cosmological anomaly argument is limited "
 "by that single realisation. <b>A methylome sky can be measured again on the same person.</b> Two draws six months apart give a "
 "difference map - and a difference map is where this whole toolkit is strongest, because the static structure (the genomic "
 "correlation, the laboratory's own character, that individual's baseline) cancels and only what changed survives. That is not a "
 "metaphor; it is an experimental design a clinic can execute, and it is the strongest argument for treating a methylome this way.</p>"


 "<h3>What else the MCMC gives us, and what we are not yet using</h3>"
 "<p>The atlas does not store one number per cell type per address. It stores the <b>posterior mean, the posterior SD and the credible-interval "
 "bounds</b> - 122 standard-deviation columns with interval columns beside them - because MCMC produces a distribution, and keeping only its "
 "centre throws away most of what was computed. Two uses are live: the floors were fitted this way and frozen; and a second, posterior-weighted solver cross-checks the composition.</p>"
 "<p><b>What the atlas does not carry, measured.</b> A cross-class covariance at an address would let the composition step say <i>these two are individually uncertain but their sum is well determined</i>. PROC-COV-01 (2026-09-26) checked the MCMC archives: only marginal summaries were written, each locus's posterior is independent by model construction, and the classes were run as separate jobs - so no cross-class covariance was ever estimated and none can be recovered. The residual covariance on real specimens was measured instead and found to be a reproducible reference misfit, mostly removable; that is recorded in PROC_COV_01_OUTCOME.md.</p>"

 "<h3>Brilliance - the first tool taken from cosmology, and where it went</h3>"
 "<p>The first CMB-derived instrument in this work was not the sky; it was <b>brightness</b>. Surface brightness is how astronomy states an "
 "intensity that does not depend on the distance to the source or the size of the telescope, and the brightness layer built alongside the atlas "
 "applied that idea to an architecture class: how strongly does this class shine at its own addresses, on a scale that does not depend on how much "
 "of it happens to be in the tube. That was the right instinct, and it is why a per-class expectation exists at all. The sky "
 "builds its expectation from <i>this sample's own composition</i>, which a precomputed per-class file cannot "
 "do. The lineage is worth stating, because the first import from cosmology is still load-bearing one layer down.</p>"

 "<h3>Two maps from one patient - the difference map</h3>"
 "<p>In the author's words, and it is the strongest argument on this page: <i>a methylome sky can be re-measured on the same patient. Two draws "
 "six months apart gives a difference map - and difference maps are where cosmology's entire toolkit is most powerful, because the static "
 "foregrounds cancel. That is not a metaphor, it is an experimental design a clinic can execute, and it is the strongest reason the analogy is a "
 "method rather than a decoration.</i></p>"
 "<p><b>Is it sensitive enough to see a change in one island? Measured, not asserted.</b> On the Uppsala panel the between-person spread per "
 "address is 0.029 in beta units (IQR 0.018-0.046). The quietest 5 % of addresses - spread 0.0089 - bound the purely technical part from above, "
 "since nothing can be quieter than the noise. A paired difference of two draws from one person removes that person's baseline "
 "and the genomic correlation, leaving technical noise times root two:</p>"
 "<table class='t'><tr><th>addresses averaged</th><th>what that is</th><th>detectable change in methylation (2 sigma, paired)</th></tr>"
 "<tr><td class='n'>1</td><td>one CpG</td><td class='n'>2.5 percentage points</td></tr>"
 "<tr><td class='n'>5</td><td>a few sites in one promoter</td><td class='n'>1.1 points</td></tr>"
 "<tr><td class='n'>20</td><td><b>a CpG island</b></td><td class='n'><b>0.6 points</b></td></tr>"
 "<tr><td class='n'>100</td><td>a small domain</td><td class='n'>0.25 points</td></tr>"
 "<tr><td class='n'>1,000</td><td>a large domain, or a class panel</td><td class='n'>0.08 points</td></tr></table>"
 "<p>Reported effects at a CpG island typically run from several to twenty points. <b>So yes: at island scale and above, a paired difference map "
 "should resolve changes an order of magnitude smaller than effects already in the literature</b>, and the limit is set by array noise rather than "
 "by anything in this chain. Two caveats travel with that. The technical term is an <i>upper bound inferred from cross-sectional data</i>, because "
 "<b>no repeat draws of the same person exist in anything held here</b> - the EPIC-Italy foundation arrays carry no subject identifier, so no second draw can be linked - and a real serial test must re-measure it from actual replicates. And a difference map cancels the laboratory only if both draws "
 "went through the same laboratory and pipeline; otherwise the pipeline map must be applied to each before differencing. "
 "A serial mode - the same patient read twice, with the difference printed - is a named next step of the chain.</p>"

 "<h3>Is the sphere necessary? A straight answer</h3>"
 "<p><b>For the measurement, no. For the toolkit, yes.</b> The residual is a one-dimensional sequence along the genome; the sphere is a "
 "space-filling reindexing of it. What makes that legitimate rather than ornamental is that the reindexing <b>preserves locality</b> - measurable, "
 "not assumed: every one of the 196,608 pixels holds CpGs that are genomically contiguous and on a single chromosome, with a median span of "
 "<b>511 base pairs</b> (nine in ten under 20 kb); a quarter of all neighbouring-pixel pairs sit within 10 CpGs of each other in genomic order and "
 "fewer than 1 per cent are more than 10,000 apart, against about 161,000 for a random assignment. Angular distance therefore tracks genomic "
 "distance at the scales that matter, which is what makes needlets, a harmonic decomposition and a power spectrum mean anything here - and it is "
 "why the correlation noted below is genomic rather than an artefact of the projection.</p>"
 "<p><b>For a clinician reading one patient, a sphere is probably the wrong picture,</b> and this page would rather say so than defend it. A "
 "geneticist thinks in chromosome coordinates and an ellipse has no chromosomes on it. Better formats for the clinical read: a <b>per-chromosome "
 "linear track</b> - immediately interpretable, the format every genome browser already uses - or a <b>Hilbert-curve layout</b>, a space-filling "
 "curve in the plane already used in genomics, which keeps locality about as well as the sphere while staying rectangular and printable. The sphere "
 "earns its place where the <i>statistics</i> are spherical. The intended end state is both: the sphere for the spectral analysis, a linear or "
 "Hilbert track for what a clinician sees, and the same residual underneath so the two cannot disagree.</p>"

 "<h3>Acknowledgement - whose tools these are</h3>"
 "<p>Everything on this page except the biology was built by the cosmology community over some thirty years, out of necessity, because they had one "
 "noisy sky and no way to obtain another. HEALPix; the internal-linear-combination and needlet methods that make component separation work; beam "
 "deconvolution; per-pixel noise covariances; simulation-based nulls, and the discipline of scoring an anomaly against a null rather than against "
 "intuition - none of it was invented here. The contribution is the recognition that a methylome is the same kind of object and can be measured "
 "with the same instruments. <b>We are the messenger, not the inventor</b>, and the right response to a cosmologist reading this page is thanks.</p>"
 "<p>What is handed over is also more than a ruler for cellular fidelity. It is a <b>toolkit, and a different way of seeing the methylome</b> - as "
 "a field on a manifold with a model, a mask, a beam and a noise budget, rather than as a list of sites with p-values attached. What a geneticist "
 "builds with that is not ours to predict, which is rather the point of handing it over.</p>"
 "<p class='m'><b>On precedence, stated the way a referee will read it.</b> We are not aware of prior work that projects a methylome "
 "onto a sphere and applies component-separation and sky-statistics machinery to it. The search behind that sentence, run 2026-09-22, with its "
 "actual counts rather than a summary: in <b>Europe PMC</b>, <i>spherical harmonic</i> + <i>methylome</i> returned <b>0</b>, "
 "<i>angular power spectrum</i> + <i>(methylation OR epigenome)</i> returned <b>0</b>, and <i>needlet</i> + <i>(genome OR methylation)</i> returned "
 "<b>0</b>; <i>HEALPix</i> + <i>methylation</i> returned <b>2</b> records and <i>HEALPix</i> + <i>genome</i> returned <b>4</b>, all of them "
 "structural-biology papers (cryo-EM structures of a vesicular stomatitis virus polymerase complex, human IAPP fibrils, and others) in which HEALPix "
 "appears as the orientation-sampling scheme and methylation only incidentally - none projects a methylome. A query pairing <i>cosmic microwave "
 "background</i> with <i>methylation</i> returns over 1,500 records and is pure abbreviation collision (CMB is also conditioned medium, chronic mountain "
 "sickness), which is why it is not evidence either way. In <b>arXiv</b>, five query forms - HEALPix + methylation, power spectrum + methylome, cosmic "
 "microwave background + epigenome, needlet + biological, spherical harmonics + DNA methylation - each returned <b>0</b>. Sanchez &amp; Mackenzie brought "
 "the Landauer bound to methylation but no sky; CMB pipelines have not been pointed at a genome. <b>This is a bounded search, not a proof of absence</b>, "
 "and a reader who knows of prior work should say so. One detail worth keeping, because those two HEALPix hits make the point better than a zero would "
 "have: <b>HEALPix is already used in biology</b>, as the angular-sampling scheme for particle orientations in cryo-electron microscopy. The "
 "pixelisation is in the toolbox already; this is a different use of it.</p>")

def tab_sky(o, R, sid, workdir):
    s=o["patient_sky"]; H=["<div class='callout'><b>What this sky is compared to.</b> Not to a healthy picture - there is no reference image anywhere in this comparison. Every pixel is <span class='m'>z = (&beta; &minus; &Sigma;<sub>c</sub> f<sub>c</sub>&mu;<sub>c</sub> &minus; m<sub>lab</sub>) / s<sub>lab</sub></span>: this specimen's own methylation at that address, minus what <b>its own composition</b> predicts for it (the atlas mean of each class, weighted by the fractions Stage 2 measured in <i>this</i> specimen), minus this laboratory's measured zero, divided by this laboratory's measured per-address spread across its panel arrays. So z is the departure from this specimen's own composition expectation, in units of the laboratory's per-address spread. The number to read is the fraction of addresses beyond |z| = 2; the four commissioned laboratories' panels read 2.6-3.2 %, the instrument's quiet level. The per-address zero m_lab is measured on the laboratory's panel; a constructed specimen composed from the atlas itself reads far from quiet against it, so replacing that zero and spread with the atlas's own per-address posterior is a queued chain change (ENHANCEMENTS 0g).  Red is above the composition expectation, blue below, black not assessable. Two plates are comparable only within one laboratory, because m<sub>lab</sub> and s<sub>lab</sub> are that laboratory's own.</div>", SKY_WHY, "<h3>This sample's sky - Stage 4.6</h3>"]
    # 2026-09-25: say it at the top, not in the refusal list at the back.
    if not (s.get("available") and s.get("_sky") is not None):
        H.insert(0, "<p class='warn'><b>No sky was rendered for this specimen.</b> " +
                 html.escape(str(s.get("reason") or "this laboratory has no commissioned residual scale, so "
                                  "there is no zero to take residuals against")) +
                 ". Every figure on this page is a reference illustration, identical in every report - none "
                 "of them is this specimen.</p>")
    H.append("<p>Every CpG the chain reads is placed on a sphere in genomic order (HEALPix, NSIDE 128 - the projection Planck used for the microwave background). At each address the chain computes what this sample's <i>own composition</i> predicts (the Stage 2 fractions mixed over the atlas class means), subtracts the laboratory's per-address zero, and divides by the laboratory's healthy spread at that address - both measured on the laboratory's commissioning panel. "
             "The plate shows that residual z. A healthy sky is <b>quiet</b>: 2.6-3.2 % of addresses beyond |z| = 2 on the four commissioned laboratories (the Gaussian expectation is 5 %; the scale is ~1.1x conservative and that constant is printed on every plate). A class panel renders only when Stage 2 places the class above its measured presence floor; masked panels say so.</p>")
    if s.get("available") and s.get("_sky") is not None:
        try:
            sys.path.insert(0,ENGINE); import stage_4_6_patient_cmb as S
            png=os.path.join(workdir,f"sky_{sid}.png"); S.render_plate(s["_sky"],png,f"{sid} - residual z: this specimen against its own composition expectation; sigma from the atlas posterior and this array's SNP probes")
            s["_plate_drawn"] = True
            H.append("<figure><img class='plate' src='data:image/png;base64," +
                     base64.b64encode(open(png, 'rb').read()).decode() +
                     "' alt='this specimen&#39;s sky plate'/><figcaption><b>THIS SPECIMEN: " +
                     html.escape(str(sid)) + ".</b> Rendered from this specimen's own residuals "
                     "at this run. It is the only image on this page that is about this "
                     "specimen; every other figure below is a reference illustration."
                     "</figcaption></figure>")
        except Exception as e:
            s["_plate_drawn"] = False; s["_plate_error"] = _e(e)
            # 2026-09-25: this used to be a small grey note under four reference pictures, so a
            # reader saw a sky that was not theirs and had no way to know. It is a warning now.
            H.insert(0, "<p class='warn'><b>This specimen's own sky plate was NOT drawn.</b> "
                     "Reason: " + _e(e) + ". Every figure on this page is therefore a reference "
                     "illustration, identical in every report - none of them is this specimen. "
                     "The per-class statistics below ARE this specimen's.</p>")
        a=s["all"]; H.append(f"<p><b>Whole sky:</b> {a['n']:,} addresses; {100*a['frac_abs_z_gt2']:.1f} % beyond |z| = 2 (healthy 2.6-3.2 %); median z {a['median_z']:+.3f}; mean |z| {a['mean_abs_z']:.3f}.</p><table class='t'><tr><th>class panel</th><th>Stage 2 fraction</th><th>presence floor</th><th>status</th><th>addresses</th><th>% beyond |z|=2</th><th>median z</th></tr>")
        for c in CLASSES:
            r=s["classes"].get(c,{}); H.append(f"<tr><td>{CLASS_LABEL[c]}</td><td class='n'>{100*r.get('fraction',0):.1f} %</td><td class='n'>{100*r.get('presence_floor',0):.0f} %</td><td>{_e(r.get('status'))}</td><td class='n'>{r.get('n','') if r.get('assessable') else ''}</td><td class='n'>{('%.1f'%(100*r['frac_abs_z_gt2'])) if r.get('assessable') else ''}</td><td class='n'>{('%+.3f'%r['median_z']) if r.get('assessable') else ''}</td></tr>")
        H.append(f"</table><p class='m'>{_e(s.get('calibration_note',''))}</p>")
    else: H.append(f"<p class='pend'>{_e(s.get('status') or 'SKY WITHHELD')}. The picture returns when a per-address spread that belongs to the instrument fits every commissioned laboratory (PROC-SKY-01 outcome, doors).</p>")
    # Archival reference plates removed at the author's direction 2026-09-22: this tab carries only the sky THIS run
    # generated, plus the one standing comparison figure.
    cmp_png=R["files"].get("healthy_sky_vs_cmb.png")
    if cmp_png:
        H.append("<h3>The comparison figure</h3><figure><img src='data:image/png;base64,"+base64.b64encode(open(cmp_png,'rb').read()).decode()+"' style='width:100%'/>"
          "<figcaption>Top: the author's photograph of the Planck CMB temperature residual. Below it, a healthy methylome sky from this chain at full "
          "resolution and beam-smoothed, same projection, same colour convention. The comparison is a difference, not a resemblance, and the difference is "
          "what makes the cellular map readable. <b>Reference figure</b> - the same in every report, not this specimen.</figcaption></figure>")
    H.append("<div class='warn'><b>If you are going to look for a patch, a band or a region in one of these maps, read this first.</b> "
      "A healthy sky is <b>not</b> spatially featureless. Beam-smoothing a healthy sky on the sphere (32 nearest pixels) leaves a spread of 0.171, against "
      "0.131 &plusmn; 0.001 for the same values spatially shuffled - <b>1.31&times;, 57&sigma;</b>. That mottling is real: methylation is correlated along the "
      "genome, and this projection places pixels in genomic order, so neighbouring addresses carry correlated residuals. It is a property of healthy biology, "
      "not a departure.<br><br><b>Consequence: any test that looks for structure - a patch, a band, a region - must be scored against a SPATIALLY-SHUFFLED "
      "null, not a Gaussian one.</b> A Gaussian null will call this healthy baseline a finding every time. The per-address test this report prints (the "
      "fraction of addresses beyond |z| = 2) is per-pixel and therefore unaffected. Measured 2026-09-22 while building the figure above; recorded as an open "
      "item in the working note.</div>")
    H.append("<h3>How to read a plate</h3><p>Blue = less methylated than this sample's own composition predicts at that address; red = more. A healthy plate is salt-and-pepper with no structure. Structure - a band, a patch, one class panel lit while the others are quiet - is what the sky is for: it shows <i>where in the genome</i> a departure lives, which no single number can. The plate is a residual map, the same object a cosmologist looks at after subtracting the model from the data.</p>")
    return guard("".join(H),"Sky")

# ======================= reference / integrity / chain / physics / story / record / run =======================
def _link(R, path_or_name, label=None):
    p=path_or_name if os.path.isabs(path_or_name) else R["files"].get(path_or_name)
    if not p or not os.path.exists(p): return _e(label or path_or_name)
    return f"<a href='{_gh(_rel(p),R['sha'])}' target='_blank'>{_e(label or os.path.basename(p))}</a> <span class='sha'>{_sha(p)[:12]}</span>"

def tab_reference(R, percell_status):
    """THE INSTRUMENT (author, 2026-09-26: measuring, not comparing). Three things a measuring instrument publishes:
    its physical constants, its calibration, and the arrays the calibration was done on. No band, no zero applied to
    any A - nothing on this page is a comparison layer."""
    mp=R["maps"]["maps"]["stage1_noob_450K"]
    H=[GH, "<h2>The instrument - its constants and its calibration</h2>",
       "<p>A reading on this report is A = H(mean beta over a cell's identity loci) / H_min of its architecture class. Healthy is A = 1.00 by the physics. "
       "This page lists what that measurement depends on, in two layers that are never mixed: physical constants (frozen, from reference cell methylomes) and instrument calibration "
       "(how a laboratory's array is brought onto the atlas scale, where a cell counts as present, how quiet the foreign-cell detector is). Nothing on this page adds to or subtracts from any cell's A.</p>"]
    # ---- 1. constants
    SUBL={"methyl":"methylation","nucl":"nucleosome occupancy","fuzz":"nucleosome fuzziness","wps":"WPS","frag":"DELFI fragment size"}
    subs=list(R.get("sub_order") or ["methyl"]); hm_tab=R.get("hmin_table") or {}
    H.append("<h3>1. Physical constants - the floors</h3>"
             "<p>H_min is the minimum entropy a cell of that class must hold to keep its identity; A = H / H_min. One floor per class per substrate. Only the methylation column is read on this report; the other four substrates have frozen floors and no pipeline map, so they are RESERVED (see Coverage).</p>"
             "<table class='t'><tr><th>architecture class</th><th>H_min (methylation)</th>"+"".join(f"<th class='m'>H_min {SUBL.get(x,x)} <span class='pend'>RESERVED</span></th>" for x in subs if x!="methyl")+"</tr>")
    for cl,ent in sorted((R.get("ident") or {}).items()):
        if not isinstance(ent,dict) or ent.get("H_min") is None: continue
        hm=float(ent["H_min"]); row=hm_tab.get(cl) or []
        extra="".join(f"<td class='n m'>{float(v):.4f}</td>" for x,v in zip(subs,row) if x!="methyl") if row else ""
        H.append(f"<tr><td>{_e(cl)}</td><td class='n'>{hm:.4f}</td>{extra}</tr>")
    H.append("</table><p class='m'>Methylation floors: eight class floors fitted by MCMC on 37 published reference cell methylomes (G-002, April 2026) and frozen. Samplers, the 37-cell table and a re-run script: <a href='https://doi.org/10.5281/zenodo.22905819'>10.5281/zenodo.22905819</a>.</p>")
    # ---- 2. calibration
    fl={k:v for k,v in (R.get("floors") or {}).items() if not str(k).startswith("_")}
    H.append("<h3>2. Instrument calibration - what a laboratory's array needs before a cell can be read</h3>"
             "<p>Each file below is a constant of the instrument. What it was fitted on is in the file's own <code>_meta</code>, not on this page.</p>"
             "<table class='t'><tr><th>calibration</th><th>what it does</th><th>file</th></tr>"
             f"<tr><td>Pipeline map</td><td>beta_atlas = (beta - {mp['intercept']}) / {mp['slope']}: puts a laboratory's Stage 1 betas on the atlas scale - a scale transfer between two measuring pipelines, applied to every address alike.</td><td>{_link(R,'beta_scale_maps_v1.json')}</td></tr>"
             f"<tr><td>Presence floors</td><td>a class or cell below its floor is not scored - an absent cell's loci carry the specimen's other cells and read by artefact (PROC-CEIL-01). Floors: {_e(', '.join(f'{k} {float(v):.3f}' for k,v in sorted(fl.items()) if isinstance(v,(int,float))))}</td><td>{_link(R,'presence_floors_v1.json')}</td></tr>"
             f"<tr><td>Foreign-cell detector panel</td><td>per-laboratory noise floor and detection line for each foreign cell (Stage 2d). A laboratory without a commissioned panel reports 'detection not commissioned', never a borrowed line.</td><td>{_link(R,'detection_panel_v1.json')}</td></tr>"
             f"<tr><td>Sky per-address scale</td><td>the spread of the composition residual at each address, so the Sky tab can draw z; the sky is a picture, not a reading.</td><td>{_link(R,'stage_4_6_patient_cmb.py')}</td></tr>"
             f"<tr><td>Tier scale</td><td>NORMAL [0.95, 1.05) about the fixed point A = 1.00; ELEVATED to 1.07 (Warburg line); BREACH at 1.10. One file, one function.</td><td>{_link(R,'tier_breakpoints.json')}</td></tr>"
             "</table>")
    H.append(deepdive(R,"the instrument: its constants and its calibration"))
    return "".join(H)

CMB_TWINS=[("MCMC atlas calibration","Cosmological parameter estimation (Planck likelihood chains)","Posterior mean and SD per CpG per class; class floors H_min with R-hat < 1.001 on 37 reference cells","G-002 / G-003b, IAMAtlasREBUILD"),
 ("Pipeline map","Instrument calibration transfer (cross-calibrating detectors)","affine beta map, fit on 32,688 identity loci; transfers to a second laboratory at median A 1.0097","beta_scale_maps_v1.json"),
 ("Composition solver (legacy)","Constrained component fitting","conservative NNLS against the atlas: a cell is placed only when the evidence forces it, so the composition the report stands on is not inflated","legacy_iam_deconvolver.py"),
 ("Composition cross-check (NILC) - runs as the cross-check column on Safeguards","Needlet internal linear combination: the Planck component-separation method","variance-weighted and deliberately sensitive. Switched off in July 2026 because it disagreed with legacy on every blood sample; PROC-NILC-01 found the disagreement WAS the finding - NILC was reporting that the atlas cannot split the blood classes, which PROC-SEP-03 then measured (7/7 blood SPLIT, kappa < 10). It is commissioning row 2b and is not in this run; the implementation is linked below","nilc_celltype_deconvolver.py"),
 ("Residual sky","The anisotropy map / residual map after model subtraction","z per address: beta minus this specimen's own composition expectation, over sigma from the atlas posterior and this array's SNP-probe noise; no panel, no laboratory file (2026-09-27)","stage_4_6_patient_cmb.py"),
 ("Presence floors","Galactic mask","a class panel renders only above its presence floor","presence_floors_v1.json"),
 ("Pre-registration and the falsification register","Blind analysis","every bar written and hashed before the run; failures kept on the record as sealed","Record/PROC_data")]

def tab_integrity(o, R, refusals):
    H=["<h2>Integrity - the fail-safes that kept this reading honest, and what each one measured</h2>",
       "<p>The chain was built with the toolkit cosmology developed for reading a faint signal against a calibrated reference. Each safeguard below has a named twin in that toolkit, a measured constant, and a file with a hash. Where a safeguard could not be applied to this sample, the chain refused to print a number rather than approximate one.</p>",
       "<h3>1. What this run refused, and why</h3>"+("<ul>"+"".join(f"<li>{_e(r)}</li>" for r in refusals)+"</ul>" if refusals else "<p>Nothing was refused on this sample.</p>"),
       "<h3>2. The safeguards, with their cosmology twins</h3><table class='t'><tr><th>safeguard</th><th>cosmology twin</th><th>what it measured / does here</th><th>where</th></tr>"]
    for a,b,c,d in CMB_TWINS: H.append(f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td class='m'>{_e(d)}</td></tr>")
    H.append("</table><h3>3. Instrument constants read by this run (the pipeline map and the tier file; no constant is applied to any cell's A)</h3><table class='kv'>")
    imm=o["classes"].get("immune",{}); H.append(f"<tr><td>pipeline</td><td>{_e(o.get('scale'))}</td></tr><tr><td>tiers</td><td>tier_breakpoints.json {_e(R['tiers']['_meta'].get('version'))}</td></tr></table>")
    H.append("<h3>4. Files read by this run (repository at commit "+_e(R["sha"])+")</h3><table class='t'><tr><th>file</th><th>sha256</th></tr>"+"".join(f"<tr><td><a href='{_gh(_rel(p),R['sha'])}' target='_blank'>{_e(n)}</a></td><td class='sha'>{_sha(p)}</td></tr>" for n,p in sorted(R["files"].items()))+"</table>")
    H.append("<h3>5. Two rules this report obeys</h3><ul><li><b>Detection rule.</b> No definitive statement about what the chain can or cannot detect appears until the chain has been run on that question under seal - 'not yet tested', never 'cannot'. A vocabulary guard refuses to write the measurement tabs if they name a condition, a verdict, or an age in years.</li><li><b>Sealing rule.</b> A pre-registration is written only for a built tool tested against a bar; building is exploration recorded in a dated working note. This report generator is in build and unsealed.</li></ul>")
    return "".join(H)

CHAIN=[
 ("0","Intake","declared age, specimen, substrate; file integrity hash","stage_0_intake.py",
  {"in":"the IDAT pair (or a calibrated beta vector), declared age, specimen, substrate, laboratory","out":"a checked context record; a SHA-256 of each input file so the run can be tied to exactly these bytes",
   "why":"every later constant is specimen-specific and laboratory-specific. If the specimen is not declared, the presence floors do not apply, and the chain must not guess.",
   "refuses":"an undeclared specimen, or an array type the pipeline map was not fitted on","commissioned":"row 1 of the commissioning table"}),
 ("1","Calibration","IDAT (Red + Grn) -> beta, methylprep noob","stage_1_idat_calibration.py",
  {"impl":"calibrate_idat_to_beta","in":"the raw two-channel intensity files the scanner writes","out":"one beta value per CpG (413,058 on 450K), beta = methylated / (methylated + unmethylated)",
   "why":"raw intensities carry dye bias (the two colour channels are not equally efficient) and probe-type bias (the array has two chemistries). noob normalisation removes both using the array's own out-of-band control probes. Author-processed matrices published on GEO are deliberately NOT used: each laboratory normalises differently, and the offset between two pipelines is larger than the whole NORMAL tolerance (LESSON-SCALE-01, measured).",
   "refuses":"nothing - it either produces betas or errors","commissioned":"PROC-CAL-01: raw IDAT through this stage reproduced the cached betas the conformance tests were built on, exactly"}),
 ("1s","Pipeline map","beta -> the reference scale (one slope, one intercept)","beta_scale_maps_v1.json",
  {"impl":"stage_1s_scale_map","in":"this pipeline's betas","out":"betas on the scale the class floors were calibrated on",
   "why":"the floors were measured on published reference methylomes processed a particular way. A sample processed differently sits at a different zero. The map is an affine fit on the identity loci, measured once per pipeline - the same operation as cross-calibrating two detectors before comparing their readings.",
   "refuses":"a pipeline with no fitted map: the reading is marked NOT REPORTABLE rather than placed on someone else's scale","commissioned":"PHASE 1 / 1c; transfer verified on a second laboratory"}),
 ("2","Composition","constrained fit against the atlas -> which cell types, and in what proportion","legacy_iam_deconvolver.py",
  {"impl":"stage_a_cells","in":"the mapped betas and the atlas","out":"a fraction for each of the 115 atlas cell types and each of the 8 architecture classes",
   "why":"a blood tube is a mixture. Before anything can be said about how well a class holds its pattern, you have to know how much of that class is in the tube. The solver is a constrained non-negative fit and is deliberately conservative: it places a cell only when the evidence forces it, so the composition the rest of the report stands on is not inflated. That is why whole blood typically shows a handful of placed cells rather than all 115 - it is the specification, not a defect. Every one of the 115 is still scored (see Every cell).",
   "refuses":"a class below its measured presence floor is not carried forward into the gauge or the sky","commissioned":"PROC-DECON-01: reproduced the answer key shipped with the test package to the manifest's precision"}),
 ("2","The atlas","IAMAtlasREBUILD: per-CpG, per-cell-type posterior mean and SD - 483,092 CpGs x 115 cell types","IAMAtlasREBUILD.csv.xz",
  {"impl":"stage_a_cells","in":"open published reference methylation measurements, reconciled to one set of cell-type definitions","out":"a full posterior at every position for every cell type, with a credible interval attached",
   "why":"published reference panels are built by different laboratories, on different arrays, with different definitions of the same cell type, and they carry no statement of their own uncertainty and no concept of a healthy floor. Run a sample through them separately and you get fractions that do not live on a common scale. The atlas is a hierarchical Bayesian reconciliation sampled by MCMC - the same class of inference cosmology runs against the Planck likelihood - which is what produces an uncertainty at every pixel. That uncertainty is the point: it is the difference between 'inside or outside a line' and 'this far from the floor, give or take this much'.",
   "refuses":"nothing - it is data; but only 4.9 % of stromal CpGs have a converged posterior, and the sky masks what has not converged","commissioned":"the reference the whole chain reads; provenance file linked below"}),
 ("2","115 cells -> 8 classes","the map from every atlas cell type to its architecture class","IAMAtlasREBUILD_celltype_to_class.json",
  {"impl":"stage_a_cells","in":"a cell-type name","out":"one of the eight architecture classes, which selects that cell's H_min",
   "why":"the floors are per class, not per cell type, because the floor follows from how much information that architecture must protect (the eight sandcastles). This file is the only place the assignment lives.",
   "refuses":"an unmapped cell type is not scored","commissioned":"shipped with the atlas"}),
 ("2b","Composition cross-check (NILC)","variance-weighted component separation - the Planck method. runs as the cross-check column on Safeguards","nilc_celltype_deconvolver.py",
  {"impl":"stage_2b_second_opinion","in":"the same mapped betas","out":"an independent set of fractions, computed with opposite biases: sensitive where the constrained solver is conservative",
   "why":"in cosmology you never separate components one way only. NILC (needlet internal linear combination) is what Planck uses. Here it was switched off in July 2026 because it disagreed with the constrained solver on every blood sample - which looked like a defect in NILC. PROC-NILC-01 found the disagreement WAS the finding: NILC was reporting that the atlas cannot split the blood classes, which PROC-SEP-03 then measured directly (7 of 7 blood classes inseparable, and the separability statistic quantified). The tool was right and was cut for being right. It is commissioning row 2b and is not in this run.",
   "refuses":"n/a - not currently in the chain","commissioned":"NOT commissioned. Row 2b is open: the decision is whether its output ships as a second column or as a disagreement flag"}),
 ("A","Per-cell A","H(mean beta over each cell's IDENTITY loci) / H_min of its architecture class - the commissioned gauge's form on the identity surface (RULING A3, LESSON-SURFACE-01); a healthy cell of any type reads 1.00","iamatlas_a_scoring.py",
  {"impl":"stage_a_cells","in":"the mapped betas, each cell type's identity loci, its class's frozen H_min, and the cell's Stage 2 fraction as the presence gate","out":"one A per PRESENT cell with its 95 % interval on the identity loci;  ",
   "why":"this is the reading: one A per present cell, on that cell's own identity addresses, against its class floor. Deconvolution comes first so a cell is read only where it is present (PROC-CEIL-01: an absent cell's addresses read other cells' bytes). The form is H(mean beta)/H_min - RULING A3, the same arithmetic as the class gauge; the marker-panel form (mean of per-CpG H) governs the retired discriminative surface and is never crossed with it (LESSON-SURFACE-01). test_percell_physics.py asserts surface, form and scale on every push",
   "refuses":"a cell with fewer than the minimum matched markers is not scored","commissioned":"PROC-ANCHOR-01: the sealed 648-array foundation per-cell scores reproduced from raw public data at r = 1.00000, max difference 0.00004"}),
 ("B","Class gauge - internal gate","H(mean beta over identity loci) / H_min per class; the composition check (PROC-FOREIGN-01)","cpg_conductor.py",
  {"impl":"stage_b_classes","in":"the mapped betas, the identity loci for each class, the frozen H_min, the Stage 2 class fractions","out":"the class reading (printed nowhere - the report reads cells) and the composition check: the fraction of the specimen outside the blood lineage, against the PROC-FOREIGN-01 bound","why":"the gauge is commissioned on whole blood; a specimen that is not whole blood gets its cell readings and no tier word","refuses":"a tier on any cell when the composition is not verified as blood","commissioned":"PROC-SWITCH-01 (the identity surface); PROC-FOREIGN-01 (the composition check)"}),
 ("4.5","Direction","signed directional composite on sealed panels","bidirectional_decomposition.py",
  {"impl":"stage_4_5_bidirectional","in":"the mapped betas and a sealed directional panel (per-CpG reference mean and expected direction)","out":"a signed composite per class: which way the departure points",
   "why":"pooled entropy folds hyper- and hypo-methylation together - two opposite movements can average to 'normal'. Keeping the sign is how a class that is drifting in a structured way is distinguished from one that is genuinely quiet. Panels exist only where one has been sealed (immune); the other classes say so rather than guessing.",
   "refuses":"a class with no sealed panel returns no composite","commissioned":"PROC-BIDIR-01, all five bars, including re-extraction of the 726-array set from the raw 5.1 GB GEO file with zero difference"}),
 ("4.6","Sky","composition-residual z at every address - sigma from the atlas posterior and this array's own SNP probes - projected on a sphere","stage_4_6_patient_cmb.py",
  {"impl":"stage_4_6_patient_sky","in":"the mapped betas, the Stage 2 fractions, the laboratory's per-address zero and spread, the presence floors, the CpG-to-pixel mapping","out":"a residual map - one z per address - and per-class panels with their beyond-|z|=2 fractions",
   "why":"a single number per class cannot say WHERE in the genome a departure lives. The sky can. The expectation at each address is what this sample's own composition predicts, so the map is a residual in the cosmologist's sense: data minus model. A whole-blood sky is quiet at 2.6-3.2 % beyond |z| = 2 on the four commissioned laboratories.",
   "refuses":"a laboratory with no measured residual scale -> the sky is not rendered at all; a class below its presence floor -> that panel is masked, and says so","commissioned":"PROC-CMB-01 through 05. The retired brightness formula it replaced divided by the atlas posterior spread of a class mean and compared whole blood against a pure-class mean, reading most of a healthy genome as anomalous - that is closed"}),
 ("4.6","CpG -> sky mapping","every atlas CpG to one of 196,608 pixels in genomic order","iamatlas_cpg_to_healpix_nside128.npy",
  {"impl":"stage_4_6_patient_sky","in":"chromosome and position for each CpG","out":"a HEALPix pixel index, NSIDE 128",
   "why":"HEALPix is the projection Planck used: equal-area pixels, so no part of the map is visually over-weighted. Genomic order means neighbouring addresses are neighbouring pixels, which is what makes a structured departure look structured.",
   "refuses":"an unannotated CpG goes to a sentinel pixel and is excluded","commissioned":"deterministic across builds, and verified to assign all 483,092 CpGs to exactly the same pixels as the mapping built with the atlas for the reference plates"}),
 ("7","Tiers","one tier function, read from one file","cpg_tiers.py",
  {"impl":"stage_b_identity","in":"a present cell's A and its class H_min","out":"one tier word and a note, or nothing",
   "why":"before this stage the engine carried three disagreeing definitions of where NORMAL ends. Now every tier word in the chain - including the colours on this page - comes from one function reading one file. NORMAL is [0.95, 1.05) about the fixed point; 1.07 and 1.10 are the physics lines.",
   "refuses":"a non-reportable reading gets no tier at all - a tier without a commissioned band would be a fabrication","commissioned":"PROC-TIER-01, with a kit test that exercises every boundary from the file, both sides, and fails if a literal breakpoint reappears in engine code"}),
 ("2d","Foreign-cell detection","inverse-variance matched-template amplitude per foreign cell, against the laboratory's own line","detection_panel_v1.json",
  {"impl":"stage_2d_foreign_detection",
   "in":"the mapped betas at the panel's 1,506 markers; the laboratory name; the composition guard's verdict",
   "out":"per foreign cell: f-hat (estimated fraction), sigma, z, the laboratory's line, detected yes/no, and the measured detection limit where a procedure measured one",
   "why":"the deconvolver's constrained fit is pinned at exactly zero for a trace component and cannot respond to half a percent of a foreign cell (PROC-MF-01: NNLS limits 2-5 %). Borrowed from CMB point-source detection: fit the cell's profile, minus the specimen's own blood background, to the residual with each locus weighted by how quiet it is in commissioned healthy blood (inverse-variance weighting, the map-maker's weight). Limits 0.5-1 % for Breast, colon, neurons and prostate on four 450K laboratories (PROC-MF-02/03). The full-covariance matched filter was tried first and tied NNLS - 1,506 markers against 36 arrays is under-determined forty-fold (PROC-MF-01).",
   "refuses":"any laboratory without a commissioned detection panel - a line set on four laboratories fired on 43-46 % of a fifth laboratory's whole-blood arrays at full marker resolution (PROC-MF-03 B8), so no line is ever borrowed; a specimen the composition guard did not verify as blood-like; and it reports every-cell-rising-together as substrate mismatch rather than as a detection",
   "commissioned":"ADOPTED BY THE AUTHOR'S DECISION 2026-09-26 (register row B-12), scoped to the four laboratories in detection_panel_v1.json; PROC-MF-01 (full-covariance matched filter) FAILED B1-B5 and was not adopted; PROC-MF-02 and PROC-MF-03 (this inverse-variance detector) met B1-B6 on four 450K laboratories and failed the fifth-laboratory bar each time. A new laboratory is commissioned by kit/commission_detection_lab.py on >= 36 of its own whole-blood arrays: centre, per-locus weights and a detection line from its own panel."}),
 ("B","The class gauge - INTERNAL GATE","identity loci -> A per class; foreign fraction against the PROC-FOREIGN-01 bound","iamatlas_gauge_identity_loci_v1_0.json",
  {"impl":"stage_b_identity",
   "in":"the mapped betas, the class's identity loci and its floor, the Stage 2 class fractions",
   "out":"A for the pooled class - used only to decide whether the specimen is whole blood (the foreign fraction against the PROC-FOREIGN-01 bound); not printed as a reading. The report leads with the per-cell A (stage A)",
   "why":"this is the surface the gauge reports on. Identity loci sit near beta = 0.73 for the class, so a reading is a reading of the class's own pattern rather than of a marker panel chosen to separate two groups",
   "refuses":"a tier on any cell when the composition is not verified as whole blood; a reading on unmapped beta",
   "commissioned":"PROC-SWITCH-01 (the identity surface); PROC-FOREIGN-01 (the composition check)"}),
 ("9","Report","this page, rendered from the bundle","build_methylphys.py",
  {"in":"the conductor's output bundle and the runtime files listed on Integrity","out":"this document",
   "why":"the report is part of the instrument, not decoration: it decides what is shown and what is withheld. A vocabulary guard refuses to write the measurement tabs if they name a condition, a verdict, or an age in years - it has caught the author's own wording more than once.",
   "refuses":"writing the file at all if the guard trips","commissioned":"row 9 - IN BUILD, UNSEALED. It seals when a report has been read line by line against the commissioning table"}),
]
DOCS=[("RUNBOOK.md","how to run the chain, and how to commission a new laboratory (a new pipeline -> its pipeline map; a new laboratory -> its foreign-cell detection line)"),
 ("CHAIN_COMMISSIONING.md","the commissioning table: which stage is commissioned, by which sealed procedure, and what is still open"),
 ("HANDOFF.md","the state of the work, for the next reader"),("README_CPG_Plates.md","the reference plates and their conventions"),
 ("README_HEALPix_Mapping.md","how every CpG was placed on the sphere")]


def _chain_sequence():
    """The step order as derived from the code by chain/build_chain_sequence.py.

    Added 2026-09-23. The CHAIN table below is written by hand, which is why it carries what each stage refuses
    and which procedure commissioned it - but a hand-written table can drift from what the code calls, and by
    that date three drifts had accumulated. This loads the derivation so the page can say which entries are in
    the live path and which are not, and so the build fails if a step the code runs is missing from the table.
    """
    import json, os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chain_sequence.json")
    try:
        d = json.load(open(p))
    except Exception:
        return None
    live, stages = set(), []
    for st in d["live_path"]:
        live.add(st["step"]); live.add(st["where"])
        # a stage is a measured step: the conductor's stage functions and the calibration module, not the report
        if st["step"].startswith("stage_"):
            stages.append(st["step"])
    notwired = {}
    for st in d["not_in_live_path"]:
        notwired[st["step"]] = st["status"]; notwired[st["where"]] = st["status"]
    fns = [st["step"] for st in d["live_path"] if st["step"].startswith("stage_")]
    fns.append("calibrate_idat_to_beta")   # the calibration step, named by the function run_sample.py calls
    return {"live": live, "stages": stages, "fns": fns, "not_wired": notwired,
            "gaps": [g for g in d.get("role_gaps", []) if "NO path calls it" in g["status"]],
            "n_live": len(d["live_path"])}

def tab_chain(R):
    H=["<h2>The chain - every stage this report came from</h2>",
       "<p>Orchestrated by <code>cpg_conductor.run_full</code>. Open any stage for what goes in, what comes out, why it exists, what it refuses to do, and which sealed procedure commissioned it. Stages 5, 6 and 8 of the June specification (a distance from a population, an age in years, matching a pattern to a condition) are <b>not</b> in the chain: none is a measurement of a cell. The class statistics are internal gates and are not shown.</p>"]
    SEQ=_chain_sequence()
    if SEQ:
        H.append(f"<p class='m'>The order below is the order the code calls: {SEQ['n_live']} steps, derived from "
                 f"<code>run_sample.py</code> and <code>cpg_conductor.run_full</code> by "
                 f"<code>chain/build_chain_sequence.py</code> and cross-checked against this table at build time. "
                 f"An entry marked NOT IN THE LIVE PATH is implemented in the tree but not called by a run - it "
                 f"has to be invoked deliberately.</p>")
    for st,nm,what,f,d in CHAIN:
        link=_link(R,f) if f in R["files"] else f"<span class='m'>{_e(f)}</span>"
        flag=""
        if SEQ and f in SEQ["not_wired"]:
            flag=(f" <span style='color:#c0392b;font-weight:600'>NOT IN THE LIVE PATH</span> "
                  f"<span class='m'>({_e(SEQ['not_wired'][f])})</span>")
        H.append(f"<details class='stage'><summary><span class='stg'>{_e(st)}</span> <b>{_e(nm)}</b> - {what} &nbsp; {link}{flag}</summary>"
                 f"<table class='kv'><tr><td>goes in</td><td>{d['in']}</td></tr><tr><td>comes out</td><td>{d['out']}</td></tr>"
                 f"<tr><td>why it exists</td><td>{d['why']}</td></tr><tr><td>what it refuses</td><td>{d['refuses']}</td></tr>"
                 f"<tr><td>commissioned by</td><td>{d['commissioned']}</td></tr></table></details>")
    H.append("<h3>The documents the chain is governed by</h3><table class='t'><tr><th>document</th><th>what it is</th></tr>"+"".join(f"<tr><td>{_link(R,n)}</td><td>{d}</td></tr>" for n,d in DOCS if n in R["files"])+"</table>")
    if SEQ:
        # the table describes the measurement stages, not the renderer that draws this page
        declared = {d.get("impl") for _, _, _, _, d in CHAIN if d.get("impl")}
        absent = [st for st in SEQ["fns"] if st not in declared]
        assert not absent, "a stage the code runs has no row in the CHAIN table: %s" % absent
        stale = [d for d in declared if d not in SEQ["fns"] and d not in SEQ["not_wired"]]
        assert not stale, "the CHAIN table claims a step the code does not run: %s" % stale
    if SEQ and SEQ.get("gaps"):
        H.append("<h3>Named as chain, called by nothing</h3><p>The chain inventory gives these files "
                 "<code>role=chain</code>, and no path in the tree calls them. They are listed because a report "
                 "that presents a step the instrument does not perform is worse than one that omits it.</p>"
                 "<table class='t'><tr><th>file</th><th>status</th></tr>"
                 + "".join(f"<tr><td class='m'>{_e(g['file'])}</td><td>{_e(g['status'])}</td></tr>"
                           for g in SEQ["gaps"]) + "</table>")
    H.append(deepdive(R,"any stage above"))
    return "".join(H)

def tab_physics(R):
    hm=R["ident"]["immune"]["H_min"]
    kB=1.380649e-23; T=310.15; Rg=8.314462618; EL=kB*T*math.log(2); M=54000/(Rg*T)
    nloci={c:len(v.get("loci",[])) for c,v in R["ident"].items() if isinstance(v,dict) and "loci" in v}
    return f"""<h2>The physics, in plain language</h2>
<p class='m'><b>Who this is for.</b> The oncologist, the molecular biologist, the lab director, the informed reader. You do not need to follow a single line of physics to use what follows, and nothing in the first seven sections needs a formula. Where a constant appears it is a textbook constant, shown so that you can check it. The formulas are at the end, marked optional.</p>

<h3>1. The one idea</h3>
<p>Every living cell is doing the same thing every moment: <b>spending energy to hold itself in order against the natural pull toward disorder</b>. Order does not maintain itself. It has to be paid for, continuously, or it decays. This instrument measures how much margin a cell has between the order it is maintaining and the minimum cost of maintaining any order at all. That margin is the reading.</p>

<h3>2. Where the number comes from - and why you will recognise every piece</h3>
<p>This is the part worth seeing, because it shows the method rests on quantities you already know from biochemistry, not on anything exotic. For a human cell:</p>
<p style='text-align:center;font-size:16px'><b>n = &Delta;G<sub>ATP</sub> &divide; (R &middot; T) = 54,000 &divide; (8.314 &times; 310.15) = {M:.2f}</b></p>
<table class='t'><tr><th>term</th><th>value</th><th>what it is</th></tr>
<tr><td>&Delta;G<sub>ATP</sub></td><td>~54,000 J/mol</td><td>the free energy released by hydrolysing one mole of ATP under cellular conditions - the standard energy packet a cell spends to do one increment of ordering work. The same number in every cell-biology text.</td></tr>
<tr><td>R</td><td>8.314 J/mol&middot;K</td><td>the gas constant - the bookkeeping constant that puts energy and temperature into the same units.</td></tr>
<tr><td>T</td><td>310.15 K</td><td>body temperature. Exactly 37 &deg;C, in Kelvin.</td></tr></table>
<p>So the denominator R&middot;T is the <i>thermal energy scale at body temperature</i> - the size of a single random thermal kick at 37 &deg;C, the noise packet trying to randomise the cell's order. And the ratio is simply <b>about 21 packets of ordering for every 1 packet of noise</b>. That is the whole foundation: two numbers any biochemist already carries, divided by each other. Nothing is hidden in it. What is new is not the ingredients - it is the recognition that their ratio defines a fixed floor of cellular existence, and that the same kind of floor appears in physical systems with no biology in them at all.</p>

<h3>3. Why this is NOT metabolism - the distinction that matters most</h3>
<p>The instant you say "ATP" and "body temperature", a clinician thinks <i>metabolism</i>. It is important to be precise: this is not a metabolic rate and not a BMR measurement.</p>
<ul><li><b>Metabolism is a flow - a rate.</b> Calories <i>per day</i>. How fast the body burns fuel, rising with exercise, fever, illness. It has <i>time</i> in it: energy divided by time.</li>
<li><b>This margin is a ratio</b> - a proportion with no time in it. The size of one ordering packet compared with the size of one noise packet. Energy divided by energy. You could freeze a cell at a single instant and the margin would still be {M:.2f}.</li></ul>
<p><i>Metabolism is the river's flow rate. This is the river's depth relative to the rocks. Same water - but one is how fast it moves, the other is how far it sits above the bottom.</i></p>
<p>The tell is the units: metabolism has <i>per day</i> in it; the margin has no time in it at all. A patient with a perfectly normal metabolism can still have a cell population whose margin has narrowed - and that narrowing is what this reads and a metabolic panel cannot.</p>

<h3>4. Three things, not two</h3>
<p>It is tempting to say "ATP is the information and temperature is the noise", but the clean picture has three parts, and the distinction matters:</p>
<table class='t'><tr><th>#</th><th>part</th><th>role</th></tr>
<tr><td>1</td><td><b>The noise</b> (R&middot;T)</td><td>thermal motion - the eraser, constantly trying to randomise the cell's order. The floor's denominator.</td></tr>
<tr><td>2</td><td><b>The power to resist it</b> (ATP)</td><td>the energy budget the cell spends holding its pattern against the eraser. The floor's numerator. Not the information itself - the force that maintains it.</td></tr>
<tr><td>3</td><td><b>The information being protected</b></td><td>the cell's actual methylation pattern - the chemical marks along the DNA that make a neuron a neuron and a liver cell a liver cell. <b>This is what the instrument reads</b>, and it is what differs from one cell type to the next.</td></tr></table>
<p>Parts 1 and 2 are <b>universal</b>: every human cell runs on the same ATP at the same 37 &deg;C, so the floor's energy is the same everywhere in the body. But the <i>information</i> each cell type must protect is different - a neuron carries a tightly specified pattern, a stem cell deliberately carries a looser one. One universal energy floor, applied to cell types carrying different amounts of information, yields <b>a different minimum-order threshold for each cell class</b>.</p>

<h3>5. Eight sandcastles on one beach</h3>
<p>The tide (thermal noise) is the same for all of them. The strength of each builder's repair-bucket (the ATP margin) is the same for all of them. But the castles are different shapes - some intricate and detailed, some simple mounds - so <b>the minimum shape each can erode to and still be recognisably itself is different</b>. One law, one tide, one bucket; eight castles, eight thresholds. That is why this reports a separate reading for each cell architecture instead of one averaged number: the law is universal, but the information each cell defends is not.</p>
<p>Concretely, for each of the eight classes we found the DNA addresses where every healthy cell of that class sits at about the same methylation level - the addresses that say "I am an immune cell" rather than the ones that distinguish one immune cell from another. We call them <b>identity loci</b>: {nloci.get('immune',0):,} for immune, {nloci.get('cycling',0):,} for cycling epithelial, {nloci.get('secretory',0):,} for secretory. Then we measured, once, how tidily a healthy cell of each class holds those addresses, and <b>froze</b> the eight numbers - the class floors H_min. They have not been touched since. Each of the 115 cell types also has its own identity addresses within its class (the addresses where that cell, specifically, sits at one level), and that is where the cell is read.</p>

<h3>6. Tidiness has a number, and the reading is a ratio</h3>
<p>Look at the tags at a set of addresses and ask: how predictable are they? If every address is definitely on or definitely off, the pattern is perfectly tidy - you could guess any one of them. If every address is a coin flip, it is maximally scrambled. Information theory gives this a number called <b>entropy</b>, from 0 (perfectly tidy) to 1 (pure coin flip); it is the same quantity a physicist uses for disorder in a gas and an engineer for noise on a line, and it is computed from the array data alone.</p>
<p><b>In what order, and for what.</b> Take a sample. <b>First</b> calibrate the raw array to the atlas's scale (the laboratory's pipeline map - what this scanner and this processing do to a known input). <b>Second</b>, find out what is in it: the deconvolver fits the sample as a mixture of the atlas's cell types and returns a fraction for each. A cell below its presence floor is not there to be read, and is not read - fraction is a detection gate, never a correction. <b>Third</b>, for each cell that is present: go to that cell's identity addresses, compute the entropy of the mean methylation there, and divide by the frozen floor of its architecture class ({hm} for every immune cell). The answer is <b>A</b>, one per cell present. A = 1.00 means that cell holds exactly the entropy its identity costs; NORMAL is 0.95 to 1.05 on the tier scale. A = 1.10 means the pattern is 10 % more scrambled than its identity requires - the cell is losing its grip on who it is; 1.07 is the Warburg line and 1.10 the breach line, the physics' two inflection points. Below 0.95 is SUPPRESSED. Because the floor is a frozen constant and the fixed point is A = 1.00, <b>one sample can be read on its own</b>. Deconvolution comes before scoring, always: a pooled reading over a whole class is not a reading of any cell, and this report prints none; the class gauge runs only as an internal gate that asks whether a blood sample is blood-like.</p>

<h3>7. The same floor appears far outside biology - which is why we trust it</h3>
<p>The reason to have confidence that this floor is real, and not a biological coincidence, is that the identical ratio governs systems with no biology in them at all. The same form - ordering energy divided by the minimum cost set by temperature - describes:</p>
<table class='t'><tr><th>system</th><th>what it spends to maintain order</th><th>the floor it pays against</th><th>ratio</th></tr>
<tr><td><b>A living cell</b></td><td>ATP free energy, at 37 &deg;C</td><td>thermal energy at body temperature</td><td>{M:.2f}</td></tr>
<tr><td><b>A computer chip</b></td><td>switching energy per bit-flip</td><td>the minimum cost to write one bit (Landauer's limit)</td><td>~81 (Apple M1)</td></tr>
<tr><td><b>A quantum computer</b></td><td>the energy of a quantum operation</td><td>the same bit-writing floor - which is why these machines must be chilled to near absolute zero, to lower the floor itself</td><td>1.000 (Al transmon)</td></tr></table>
<p>Landauer's principle - that writing or erasing one bit has a minimum, unavoidable energy cost set by temperature - is an established result in physics, published in 1961 and used routinely in computing. A microchip today runs far above that floor, but as chips are pushed to be more efficient they approach it, and the floor is what eventually limits them. A quantum computer fights the same floor from the other direction: it cannot easily lower its operating energy, so it lowers the <i>temperature</i> instead, dropping the floor toward zero so its fragile information can survive. The cell cannot cool. It runs at 310 K and pays its 21 quanta.</p>
<p>The point for a clinician is simply this: <b>the floor used here for a cell is the same kind of floor an engineer uses for a chip.</b> A recognised principle of physics, applied to biology. That is the difference between a measurement and a metaphor - the same arithmetic reads a cell, a transistor and a refrigerated qubit, and it was not invented for any one of them.</p>

<h3>8. What is measured, and what is not derived</h3>
<p>One point of honesty that matters to a reviewer, and that has been corrected in this work since earlier drafts: the energy bound above constrains the <b>cost of writing</b> a pattern. It does not, by itself, tell you the <b>entropy of the pattern a healthy class holds</b>. So the eight class levels are <b>measured, not derived</b> - fitted once, in April 2026, from 37 published reference cell methylomes using Markov-chain Monte Carlo (the same class of inference cosmology runs against the Planck likelihood), with convergence checked and a bootstrap cross-check in which every frozen value falls inside its interval - and then frozen before any sample in this work was scored against them. That is the stronger claim, because it is checkable: the calibration script and its 37-cell database with every DOI are linked from the Record tab. Nothing about the reference is withheld.</p>

<h3>9. The atlas, and what MCMC and a posterior actually mean</h3>
<p>Two numbers on this report came out of a fitting procedure rather than a direct measurement, and a reader is entitled to know what kind of object they are. Neither idea is difficult; both are usually explained in a way that assumes you already know them.</p>
<p><b>The atlas.</b> Every reading here is a comparison: this sample against what each cell type looks like when it is healthy. The atlas is that reference - 483,092 CpG addresses by 115 cell types, each measured entry a methylation level with an uncertainty attached. It is sparse: the source atlases behind it were measured on different platforms, so no address carries every cell type and most cells are known at a subset of addresses; the deconvolver solves only where its candidate cells are all defined, and a cell known at too few addresses for this platform is reported as not resolvable rather than as absent. Without it there is nothing to compare to and no way to ask which cell type a departure belongs to; the composition step, the per-cell readings and the sky all read out of it. It is the single most consequential file in the chain, which is why its provenance record and checksum are linked on the Chain tab rather than described.</p>
<p><b>Why an atlas entry needs an uncertainty and not just a value.</b> A reference built from six published samples of a rare cell type is not as trustworthy as one built from sixty, and a plain average hides which is which. Carrying an uncertainty per entry is what lets the chain refuse to place a cell on weak evidence instead of guessing; a second, posterior-weighted solver runs as a cross-check on the composition and is never a reading.</p>
<p><b>MCMC, in plain words.</b> Markov chain Monte Carlo. Suppose you want the methylation level of one cell type at one address and you have a handful of noisy published measurements. You could average them - but then you have one number and no idea how much to trust it. Instead you ask: of all the values this address <i>could</i> have, which are consistent with the data I have? MCMC answers that by taking a long random walk through the candidate values, stepping more often toward values that fit the data better, and keeping a record of everywhere it went. Run it long enough and the record is the answer: values it visited often are plausible, values it rarely visited are not. It samples an answer rather than solving for one, and it works on problems where solving is impossible.</p>
<p><b>The posterior is that record.</b> Not a single number - a distribution: for this address in this cell type, the range of levels consistent with the evidence and how strongly each is supported. From it come the two numbers the atlas stores: the <b>posterior mean</b> (the centre, which is the atlas value) and the <b>posterior SD</b> (how wide the range is, i.e. how well the data pinned it down). Wide means the evidence was thin.</p>
<p><b>Why posteriors matter here, concretely - two places.</b> <b>One:</b> the class floors H_min were fitted this way from 37 published reference cell methylomes, with the chains run to convergence (the standard convergence statistic, R-hat, below 1.001 - it compares independent walks and asks whether they ended up describing the same distribution; if they disagree the answer is not yet trustworthy). Those eight values were then frozen and have not been re-fitted since - they are constants in this instrument, not parameters it tunes - and a separate bootstrap cross-check (PROC-HMIN-BOOT-01, 2026-09-20) placed all eight inside their intervals. <b>Two:</b> a second, posterior-weighted solver runs as a cross-check on the composition.</p>
<div class='warn'><b>And the one place a posterior must never be used.</b> The posterior SD describes how well the atlas pinned down an <i>average</i>. It is <b>not</b> how much healthy people differ from one another. Those are different quantities and the second is typically many times larger. Using <i>healthy mean &plusmn; 1.96 &times; the posterior SD of the mean</i> as a normal range is a category error with a large practical cost: it makes ordinary healthy samples appear to sit many standard deviations outside normal, because the interval it produces is the precision of an average rather than the spread of a population. This report therefore prints two separately labelled intervals and never a population range: the <b>95 % interval on this sample's own reading</b> (from resampling the cell's identity addresses on this array) and the <b>95 % interval on the cell's reference</b> (from the atlas posterior - how well the fixed point itself is known). Neither is a healthy range, and no healthy range is printed: healthy is A = 1.00 by the physics, with NORMAL 0.95 to 1.05 as the tolerance.</div>

<details><summary><b>Optional - the formulas and the three quantities</b></summary>
<table class='t'><tr><th>symbol</th><th>name</th><th>what it is</th><th>units</th><th>varies by</th><th>fixed by</th></tr>
<tr><td>M</td><td>Mahaffey number (the cellular margin)</td><td>E_drive / k_B T - how many thermal quanta the writing process spends per irreversible operation</td><td>none (ratio)</td><td>substrate</td><td>biochemistry / device physics</td></tr>
<tr><td>H_min(c)</td><td>class entropy reference</td><td>the binary entropy a healthy cell of class c holds at its identity loci</td><td>bits</td><td>class (8)</td><td>reference cell methylomes, MCMC-calibrated once, frozen</td></tr>
<tr><td>A</td><td>the gauge</td><td>H(beta_mean at identity loci) / H_min(c)</td><td>none (ratio)</td><td>class x sample</td><td>the sample, over the frozen reference</td></tr></table>
<p>Binary entropy: H(&beta;) = -&beta; log&#8322;&beta; - (1-&beta;) log&#8322;(1-&beta;). Landauer bound at body temperature: E &ge; k_B T ln 2 = {EL:.3e} J.</p>
<p><b>Why identity loci and not marker loci.</b> H is concave, so H(mean &beta;) over a set of addresses is largest when the addresses average to a coin flip. Marker loci are deliberately chosen to be extreme and opposite between cell types; averaged over a mixture they read as disorder that is not there - a real defect this chain had and that a synthetic-patient test caught (PROC-N7-01). Identity loci are the addresses where a healthy class sits at one level, so the mean carries meaning and A = 1 is healthy by construction. The gauge is two-to-one in &beta; (H is symmetric about 0.5).</p>
<p><b>Filter and ruler.</b> Sanchez &amp; Mackenzie (2016) used k_B T ln 2 to model the thermal <i>background</i> and remove it, so regulatory signal stands out against a control centroid - the premise that the methylome obeys Landauer's bound is theirs and is peer-reviewed. Here the same constant sets the <i>unit</i>, and the healthy level is a statement of how far above it a class holds its pattern. One filters, one calibrates. This work reached the constant independently (cosmology &rarr; quantum hardware &rarr; semiconductors &rarr; cells) and read their papers on 2026-09-20; they are cited, not built upon.</p></details>"""+deepdive(R,"the physics")

HOWTO_LEVELS=[("SUPPRESSED - below 0.95","A below 0.95","the cell is holding its pattern more tightly than its identity requires. Real, and not automatically good: strongly hypomethylated states read here (post-chemotherapy and immunosuppressed samples sit near 0.90 in the design record)."),
 ("NORMAL - 0.95 to 1.05","A within 5 % of 1.00","the cell is holding its identity pattern within five per cent of the entropy its identity costs. Symmetric about the fixed point; no population defines the edges."),
 ("ELEVATED - 1.05 to 1.07","A above 1.05","the pattern is measurably looser than its identity requires. The cell is drifting. This is the regime where a reading is worth a second sample or a closer look, and where nothing has failed yet."),
 ("at 1.07 - the Warburg line","WARBURG_TRANSITION (a line, not a band)","a line in the tier file, not a band - see below."),
 ("at 1.10","BREACH","the no-return line. The cell is no longer holding its identity pattern at the level that defines it.")]

def tab_howto(R):
    H=[f"""<h2>How to read the gauge</h2>
<p>A thermometer is only useful if its reading means something definite. This tab sets out what A means at each level, where the lines are, and - just as important - what the reading does not claim. <b>The instrument reports a cellular state; the clinician decides what to do about it.</b> The roles are distinct and kept distinct on purpose.</p>
<h3>What A is measuring</h3>
<p>A does not measure how many cells there are, or how large a mass is, or whether anything is present anywhere. It measures one thing: <b>how much informational fidelity a cell holds against the minimum its identity requires - A = H / H_min</b> - in plain terms, how well a cell is still maintaining its own identity. A well-differentiated cell doing its job sits at 1.00; a cell that has lost its architectural fidelity reads high. This is closer to a measure of <i>grade</i> (degree of dedifferentiation) than of size or burden, and that distinction is what makes the reading behave the way a clinician would want.</p>
<h3>The test for any number on this report</h3>\n<p><b>Could this number exist if nobody else's sample had ever been measured?</b> Every number on a reading passes it: H, the floor H_min, A, the array's own calibration. A band, a percentile, an age curve or a range of people would fail it - which is why none is printed. A cell's class only names the floor it is divided by; no class has an A.</p>\n<h3>Where the locker analogy helps</h3>
<p>Think of a school with eight grades, every student with a locker. Most lockers tell you nothing about the grade. Some are characteristic: every ninth-grader's holds the same geometry book. Those are the identity lockers. The gauge walks the identity lockers for one cell and asks how consistently they still hold what that class's lockers hold. It is not counting students.</p>
<h3>The levels</h3><p>Every reading on this report is <b>one cell type's A</b>: the entropy of the mean methylation over that cell's identity addresses, divided by the frozen floor H_min of its architecture class. Healthy is A = 1.00 by the physics; the tier scale below is the tolerance around it, read from one file (<code>tier_breakpoints.json</code>) by one function.</p><table class='t'><tr><th>level</th><th>on the gauge</th><th>what it means</th></tr>"""]
    for a,c,d in HOWTO_LEVELS:
        H.append(f"<tr><td><b>{a}</b></td><td>{c}</td><td>{d}</td></tr>")
    H.append(f"""</table><p class='m'>NORMAL (0.95-1.05) is the tier file's tolerance about the physical fixed point A = 1.00.<i>instrument</i> (a laboratory's pipeline map, a detector's noise floor) and never defines healthy. The tier onsets come from one file (<code>tier_breakpoints.json</code> {_sha(R['files']['tier_breakpoints.json'])[:12] if R['files'].get('tier_breakpoints.json') else ''}) read by one function.</p>""")
    H.append("<h3>Age</h3><p>The declared age is carried as context on the record and is not an operand of any reading.</p>")
    w=R["warburg"]; BR=R["breach_line"]
    H.append("<h3>Two different things, and neither is A = 1.00 sitting on a floor</h3>"
      "<p>This is worth getting exactly right, because one is a physical constant and one is the fixed point, and they are easy to conflate.</p>"
      "<table class='t'><tr><th></th><th>what it is</th><th>where it sits</th><th>what crossing it means</th></tr>"
      "<tr><td><b>A = 1.00</b><br><span class='m'>the fixed point</span></td><td>a cell holding exactly the entropy its identity costs. It is the <i>middle of the green band</i>, not an edge of anything.</td><td>mid-NORMAL. NORMAL (0.95-1.05) is the tolerance about it.</td><td>nothing - it is the point everything else is measured from.</td></tr>"
      "<tr><td><b>H_min(class)</b><br><span class='m'>the class entropy reference</span></td><td>a <b>constant in the denominator</b>, in bits: the entropy level below which that architecture cannot hold the pattern that makes it that kind of cell. It is what makes A dimensionless.</td><td>it is <i>not a mark on the A axis at all</i>. It is the unit the axis is drawn in.</td><td>a reading that falls well below 1.00 is the suppressed / inverted direction - the pattern is over-ordered or erased rather than loosened (post-chemotherapy and immunosuppressed samples sit near 0.90).</td></tr>"
      "</table>")
    H.append("<h3>The saturation wall chart - 8 classes x 5 substrates</h3>"
      f"<p>Each cell shows the class's frozen floor H_min on that substrate; the flags are the author's own from the Issue 002 saturation wall chart: <b>SAT</b> = saturates below the breach line at {BR}, so that substrate cannot register a floor breach for that class; <b>TGT</b> = tight headroom (saturates below 1.15), where a severe departure approaches saturation; unflagged = full headroom past breach. The floors come from the frozen 40-value table (methylation from the G-002 calibration, the four cfDNA/chromatin substrates from G-003b). Only the methylation column is lit today - see Coverage.</p>"
      "<table class='t'><tr><th>class</th>" + "".join(f"<th>{sub}{' (lit)' if sub=='methyl' else ''}</th>" for sub in R["sub_order"]) + "</tr>"
      + "".join("<tr><td>"+CLASS_LABEL.get(c,c)+"</td>"+"".join(
            (f"<td class='n' style='background:#2a1d1d'>{v:.4f} <b>SAT</b></td>" if 1.0/v < BR
             else f"<td class='n' style='background:#2a2a1d'>{v:.4f} <b>TGT</b></td>" if 1.0/v < 1.15
             else f"<td class='n'>{v:.4f}</td>")
            for v in R["hmin_table"][c])+"</tr>" for c in CLASSES if c in R["hmin_table"])
      + "</table>"
      f"<p class='m'><b>This chart is not new here.</b> It is the author's saturation wall chart (Issue 002, April 2026, p12), rebuilt from the engine's live floor table: 40 rows, <b>15 SAT and 2 TGT</b> - the same cells, the same flags. Issue 002 frames it as the direct analogue of the Dennard scaling walls in semiconductor physics: the frequency, power and cost walls beyond which a technology stops improving. <b>15 of the 40 combinations saturate below breach.</b> Nucleosome occupancy caps 7 of the 8 classes (its floors are all near 0.98-0.99, so there is almost no range above healthy before saturation). Fuzziness caps the three stem and progenitor classes (pluripotent stem, adult stem, progenitor). WPS caps a different three - <b>terminal</b>, adult stem and progenitor - and notably does <i>not</i> cap pluripotent stem, whose WPS limit is {1.0/R['hmin_table']['stem_pluri'][3]:.3f}, just above the line. On methylation - the one lit column - <b>pluripotent stem is capped at {1.0/R['hmin_table']['stem_pluri'][0]:.3f}</b>, which is below breach and barely above NORMAL, so a pluripotent-stem breach cannot be read on methylation at all.</p>"
      )


    H.append(f"<h3>The Warburg line at {R['warburg_line']}</h3><p>{_e(w['physics_meaning'])}</p>"
      f"<p class='m'><b>How it is placed.</b> {R['warburg_line']} is a boundary <i>line</i> in the breakpoints file, not a tier band - the file carries it with a null range and an explicit anchor flag, so no reading is ever labelled 'Warburg'; the line is reported as crossed or not. Its value is inherited from the Issue 002 tier system, where it was set from the published methylation range over which the glycolytic program becomes structurally committed, and it has not been re-derived on the commissioned chain. That re-derivation - measuring where the metabolic and structural regimes actually part on this gauge - is a named open item, and until it is done the line should be read as the corpus's stated boundary rather than a result of this chain.</p>"
      f"<p class='m'>The clinician-facing wording the file itself carries: <i>{_e(w['customer_paragraph'])}</i></p>")
    H.append(f"""<h3>What a high reading does not mean</h3>
<p>A high reading is a statement about the state of a cell population, not a clinical conclusion. Being precise about this matters, because over-reading it is exactly the failure mode the instrument must avoid.</p>
<ul>
<li><b>It does not locate anything.</b> The reading says that cells of a given class with loosened fidelity are detectable in this specimen - not where they are, nor whether they have organised into anything. Localising is the clinical workup, not the reading.</li>
<li><b>It does not name a cause.</b> The same threshold is crossed by senescent cells. The level marks a regime, not a mechanism; which mechanism applies is a separate question A alone does not answer, and this report never guesses at it.</li>
<li><b>It does not mean inevitability.</b> It means a population has crossed into the regime where structural fidelity is lost. What follows depends on biology, location, burden and clinical response - none of which this instrument measures.</li>
<li><b>It is not a grade of any lesion.</b> A tracks epigenomic fidelity, which correlates with histological grade but is its own axis. It does not replace histology. What it does is tell you <i>which compartments have drifted and by how much</i>, so that attention goes where the cells have actually changed.</li></ul>
<p>The honest reading of a high value is therefore: <i>a population of cells in this compartment has lost structural fidelity, and this compartment warrants a closer look.</i> A prompt, not a conclusion.</p>
<h3>Would a well-differentiated benign growth read high?</h3>
<p>In general it should not, and that is the question that reveals what the instrument measures. A lipoma, a fibroid or a simple nevus is made of cells that are still recognisably their own type - they have lost growth control, not identity. Because A reads identity fidelity rather than cell number, a well-differentiated benign lesion is expected to read at or near 1.00. <b>The instrument is not tripped by the presence of extra cells; it responds to the loss of what makes a cell that kind of cell.</b> The corollary is that a reading is independent of a lesion's shape - a flat lesion produces the same reading as a raised one, because the instrument reads methylation entropy rather than looking for a shape.</p>
<p class='m'>An ordered series - normal, benign neoplasm, dysplasia, the established condition - read on the commissioned chain is a named next test; until it has been run this report claims nothing about it.</p>
<h3>Two things the gauge is not</h3>
<ul><li><b>Not a clock.</b> Age is not an operand of any reading and this report never prints an age.</li>
<li><b>Not a fuel measurement.</b> The energy the cell spends holding its pattern is real, but A measures the pattern, not the fuel. A cell can be starving with a tidy pattern or well fed with a scrambled one.</li></ul>""")
    H.append(deepdive(R,"reading the gauge and the tiers"))
    return "".join(H)

def tab_story(R=None):
    """The story, in the author's own words.

    This is a finished section of a finished instrument. Where the source text predates a measurement made while
    commissioning this chain, the text here is simply the current one - no change log, no dated corrections, no
    [updated] markers. The build history belongs in ROW9_WORKING_NOTE.md and the registers, not in front of a
    reader who is being handed a product (author, 2026-09-22)."""
    H=["<h2>What is astro-genetics?</h2>",
       "<p class='m'>In the author's words.</p>",

       "<h3>Who this is for</h3>",
       "<p>The oncologist, the molecular biologist, the lab director, the informed patient. You do not need to follow a single line of cosmology to "
       "use what follows. The point is not to teach you astrophysics. The point is to explain why a cellular-health measurement is built on the same "
       "mathematics that describes stars and galaxies, and why that is a practical advantage rather than a poetic flourish. Where a number from "
       "cosmology appears, it is there so that a colleague who does know cosmology can check it independently and tell you whether it is right. The "
       "cosmology has already been checked by the field that owns it. What is new is the use of those same checked tools on cells.</p>",

       "<h3>The one-sentence version</h3>",
       "<p><b>Biology has a measurement problem that cosmology solved the tooling for more than twenty years ago, and astro-genetics is what happens "
       "when you hand biology that tooling.</b> Cosmologists were forced, by the data, to build a precise toolkit for reading the state of a system "
       "against a calibrated reference and saying how far it sits from a transition. They built it to measure the universe; it turns out to be "
       "exactly what reading a cell requires. Astro-genetics points that toolkit at the cell. The tools transfer directly because the underlying "
       "accounting is the same. That is the whole idea.</p>",

       "<h3>Stargazers and trailblazers</h3>",
       "<p>The work applies the discipline of cosmic cartography to the human epigenome. Cartography is the right word, and it is chosen carefully: "
       "<b>we map what is there.</b> We do not invent the territory; we find it, chart it, and put the chart in the hands of people who can use it. "
       "The CpG sites of the genome are already there. The healthy floor of each cell class is already there, as a physical quantity waiting to be "
       "measured. The work is to produce the chart that lets a clinician find their way around it.</p>",
       "<p>What makes the transfer more than borrowed vocabulary is that two specific correspondences hold, observable to observable, not as "
       "metaphors but as the same kind of inference run on the same kind of object.</p>",
       "<p><b>The first correspondence</b> is between the microwave background and architectural drift. The microwave background is a snapshot of "
       "the universe at one moment, with tiny temperature variations encoding the structure of everything that came after; the cosmological community "
       "spent thirty years learning to extract that structure. A cell's methylation pattern is the same kind of snapshot: a frozen record at the "
       "moment of measurement, with tiny entropy variations encoding the architectural state that came before and the trajectory that is coming. The "
       "observable differs - microkelvin temperature against Shannon entropy of methylation - but the structure of the inference is the same. Both "
       "are readings of redundantly encoded classical information, the kind of encoding Zurek's quantum Darwinism describes: a system's state "
       "written redundantly across an environment, recoverable by an observer reading any sufficient fragment. The microwave background and the "
       "methylation landscape are both that environmental record, at two scales.</p>",
       "<p><b>The second correspondence is the mechanism, and it is the load-bearing one.</b> Inhomogeneous decoherence at the cosmic horizon "
       "corresponds to inhomogeneous floor crossings at the epigenome. The horizon writes information at the Landauer cost of k_B T ln 2 per bit; "
       "the epigenome writes information at the chemical cost of maintaining a methylation mark, which sits at a fixed multiple of the same Landauer "
       "floor. The writing is irreversible at both scales. Where the horizon shows uneven, structured encoding as matter clusters unevenly, the "
       "epigenome shows uneven, structured floor crossings as some cell compartments saturate before others. The same irreversible "
       "writing-to-a-surface process governs both. <b>That is why the same instrument reads both.</b></p>",
       "<p><i>Stargazers see what is there; trailblazers go where no one has been.</i> The framework is built by doing both at once: reading the "
       "chart the universe already wrote, and being first to read it at the cellular scale. Astro-genetics is the name for that second reading.</p>",

       "<h3>Why your cells follow the same rule as the stars</h3>",
       "<p>There is an old idea, older than physics, that the universe runs on balance. When something gains, something gives. The Greeks called it "
       "the principle of opposites; Aristotle wrote about it twenty-five centuries ago. Modern physics rediscovered it in the 1800s and gave it a "
       "name that hides how simple it is: the <b>virial theorem</b>. In any stable, bound system the energy of motion and the energy of position "
       "settle into a fixed ratio. Half and half.</p>",
       "<p>Until recently this was thought to apply only to things held together by gravity. The position taken here is that the same balance applies "
       "everywhere - not because something enforces it, but because anything that violates the balance cannot be sustained at finite cost and "
       "therefore does not persist. What we see when we look around, from galaxies to cells, is what the balance allows to exist. Everything that did "
       "not balance has already come and gone.</p>",
       "<p><b>Here is the part that connects the cosmos to the clinic.</b> Every time a physical system does something irreversible it pays a cost in "
       "information. This is not a metaphor; it is Landauer's principle, an established result: writing or erasing one bit at temperature T costs at "
       "least k_B T ln 2. When a star burns, that cost is paid in two halves - one shows up as motion, which we see as heat and light; the other as "
       "the way the star bends space around itself, which we feel as gravity. A cell does the same thing. Every time it reads its genes, builds "
       "proteins or divides, it pays the same kind of cost, split into the same two halves. One half is metabolism, the moment-to-moment work. The "
       "other is written into the DNA - not into the genes themselves, but into small chemical tags placed at specific sites along the strand. "
       "<b>The methylation pattern is the cell's running ledger of what it has been doing and what it is meant to do next.</b></p>",
       "<p>In other words, the cell uses its DNA as a notebook and writes the receipt for every action. The pattern of receipts in a healthy cell "
       "looks one way. The pattern in a cell drifting toward trouble looks different. Reading those patterns is what this instrument does.</p>",
       "<p>The balance that keeps a star from collapsing is the same balance that keeps a cell healthy. When a cell's pattern drifts too far from its "
       "balance point, the cell is in trouble for the same structural reason a star past a certain mass is in trouble: both have run out of capacity "
       "to keep paying their costs in the normal way. <b>We did not invent this rule; we learned to read it.</b></p>",

       "<h3>Where the gauge sits, and what is open to inspection</h3>",
       "<p>A = 1.00 is the <b>fixed point</b> - a cell holding exactly the entropy its identity costs, in the middle of the normal band. H_min is the "
       "constant in the denominator, the unit the axis is drawn in. The two are distinct and the <b>How to "
       "read</b> tab sets them out side by side.</p>",
       "<p>The eight class floors, the calibration code that fitted them and the bootstrap cross-check (PROC-HMIN-BOOT-01) that agrees with them are <b>public</b>, "
       "under an open licence. The floors and their provenance are on the <b>Instrument</b> tab; the code is linked from <b>Files</b>. Nothing "
       "in the metrology is withheld - a fixed zero that a reader cannot inspect is not a fixed zero.</p>",

       "<h3>Two names, and which is which</h3>",
       "<p><b>Astro-genetics</b> is the programme: cosmology's measurement tools pointed at the epigenome. <b>Physics of methylation: Landauer "
       "metrology</b> is the narrower field name for the metrology itself - the frozen floors, the single-sample absolute reading - and it is the title the Operations Manual and the methods paper carry, because a methods paper should claim only what it "
       "measures. Both names are the author's; they describe different scopes of the same work. The programme document is "
       f"<a href='{GH}/tree/{R['sha']}/Biological_Physics/MethylPhys/papers/IAM_for_physicists' target='_blank'>The Informational Actualization Model - A Technical Reference for Physicists</a> "
       "(source and figures; the compiled PDF sits beside them once the author has built it); the methods paper is linked under the gauge above.</p>"]
    plates=[("CPG_Gauge_Cosmic.png","The same gauge read on a star. The cellular and cosmic readings are one instrument at two scales - the author's figure.")
]   # the CMB comparison figure lives on the Sky tab; embedding it twice doubled the file
    for fn,cap in plates:
        pth=R["files"].get(fn) if R else None
        if pth and os.path.exists(pth):
            H.append(f"<figure><img src='data:image/png;base64,{base64.b64encode(open(pth,'rb').read()).decode()}' style='width:100%'/>"
                     f"<figcaption>{_e(cap)} <b>Reference figure</b> - the same in every report, not this specimen.</figcaption></figure>")
    if R: H.append(deepdive(R,"the framework and its lineage"))
    return guard("".join(H),"Story")


def tab_record(R):
    H=["<h2>Record - every validation and every sealed procedure, linked</h2>"]
    idx=os.path.join(BIO,"Record","VAL_INDEX.csv")
    if os.path.exists(idx):
        rows=list(csv.DictReader(open(idx,encoding="utf-8")))
        H.append(f"<p>{len(rows)} index rows (series G, T, VAL pre-atlas, CPG-VAL post-atlas), rebuilt from the sealed inventory (PROC-HISTORY-01). All of these are the <b>design record</b>: preliminary tests run to build the chain. <a href='{_gh('Biological_Physics/Record/VAL_INDEX.csv',R['sha'])}' target='_blank'>VAL_INDEX.csv</a> · <a href='https://doi.org/10.5281/zenodo.19633499' target='_blank'>Zenodo deposit (pre-atlas code)</a></p><details><summary>Show the index</summary><table class='t small'><tr>"+"".join(f"<th>{_e(k)}</th>" for k in list(rows[0].keys())[:8])+"</tr>")
        for r in rows: H.append("<tr>"+"".join(f"<td>{_e(str(r.get(k,''))[:90])}</td>" for k in list(rows[0].keys())[:8])+"</tr>")
        H.append("</table></details>")
    pd_=os.path.join(BIO,"Record","PROC_data")
    if os.path.isdir(pd_):
        H.append("<p class='m'>The G-002 index row carries the status text it was sealed with ('values proprietary'); the eight floors have since been published with their samplers at <a href='https://doi.org/10.5281/zenodo.22905819'>10.5281/zenodo.22905819</a>. The row is the record and is not edited; this note is the correction.</p>")
        H.append("<h3>Sealed procedures on the commissioned chain (PROC-*)</h3><table class='t'><tr><th>procedure</th><th>files</th><th>link</th></tr>")
        for d in sorted(os.listdir(pd_)):
            p=os.path.join(pd_,d)
            if os.path.isdir(p): H.append(f"<tr><td>{_e(d)}</td><td>{', '.join(sorted(os.listdir(p)))[:120]}</td><td><a href='{GH}/tree/{R['sha']}/Biological_Physics/Record/PROC_data/{d}' target='_blank'>open</a></td></tr>")
        H.append("</table>")
    H.append(f"<h3>Documents</h3><ul><li><a href='{GH}/tree/{R['sha']}/Biological_Physics/MethylPhys/manual' target='_blank'>MethylPhys CPG Operations Manual</a></li><li><a href='{GH}/tree/{R['sha']}/Biological_Physics/MethylPhys/papers' target='_blank'>Landauer Metrology of the Methylome</a> - the methods paper (draft)</li><li><a href='{GH}/blob/{R['sha']}/Biological_Physics/MethylPhys/doors/RUNBOOK.md' target='_blank'>RUNBOOK</a> · <a href='{GH}/blob/{R['sha']}/Biological_Physics/MethylPhys/doors/CHAIN_COMMISSIONING.md' target='_blank'>CHAIN_COMMISSIONING</a> · <a href='{GH}/blob/{R['sha']}/Biological_Physics/HANDOFF.md' target='_blank'>HANDOFF</a></li></ul>")
    return "".join(H)

SPECIMENS=[("whole blood","450K / EPIC array","immune-dominant by construction; the only specimen with a commissioned pipeline map and detection panel today","lit"),
 ("plasma cell-free DNA","array or WGS","the specimen the multi-substrate argument needs: fragment size, WPS and nucleosome occupancy only exist in cfDNA. Read once in this work as a second substrate; the predicted two-channel discriminator failed and is recorded as failed","reserved"),
 ("solid tissue / biopsy","450K / EPIC array","runs end to end today and reads plausible composition (cycling, secretory and terminal present where whole blood has none) - but no tissue laboratory has a commissioned zero or band, so every class prints NOT REPORTABLE","reserved"),
 ("cerebrospinal fluid","array","named in the corpus; never run","reserved"),
 ("urine","array","named in the corpus; never run","reserved"),
 ("stool","array","named in the corpus; never run","reserved"),
 ("saliva / buccal","array","named in the corpus; never run","reserved")]
SUBSTRATE_DESC={"methyl":("DNA methylation beta","the fraction of DNA molecules carrying a methyl tag at one address. What an array measures."),
 "nucl":("nucleosome occupancy","how often a stretch of DNA is wrapped on a histone. Read from cfDNA coverage depth."),
 "fuzz":("nucleosome fuzziness","how precisely positioned those nucleosomes are - the spread, not the mean."),
 "wps":("windowed protection score","the footprint a bound protein leaves on cfDNA fragment ends."),
 "frag":("fragment size","the length distribution of cfDNA fragments; the DELFI substrate.")}

def tab_coverage(R):
    H=["<h2>Coverage - what is lit, what is reserved, and what each one needs</h2>",
       "<p>The instrument is one law read on a surface, and it generalises in two independent directions: <b>which specimen</b> the DNA came from, and <b>which physical substrate</b> is measured on it. Each combination needs its own instrument calibration before a number can be reported - a pipeline map onto the atlas scale, presence floors for its cells, and (for foreign-cell detection) a laboratory noise panel - and each has its own floor and saturation limit. Healthy itself needs no layer: it is A = 1.00. Today <b>one cell of this grid is lit: DNA methylation on whole blood.</b> Everything else prints its status rather than a number. This page exists so that no reader assumes otherwise, and so that a collaborator can see exactly what contributing one cell would take.</p>",
       "<h3>The five substrates</h3><table class='t'><tr><th>substrate</th><th>what it measures</th><th>published single-substrate discrimination (AUC)</th><th>specimen it requires</th><th>status here</th></tr>"]
    for k in R["sub_order"]:
        nm,d=SUBSTRATE_DESC[k]; req="any DNA" if k=="methyl" else ("plasma cfDNA" if k in ("wps","frag") else "plasma cfDNA (or chromatin assay)")
        st="<b>LIT</b> - commissioned on whole blood" if k=="methyl" else "<span class='pend'>RESERVED</span> - floor frozen, no pipeline map"
        H.append(f"<tr><td><b>{k}</b></td><td>{nm} - {d}</td><td class='n'>{R['auc'].get(k,'-')}</td><td>{req}</td><td>{st}</td></tr>")
    H.append("</table><p class='m'>Each substrate has its own frozen floor for each of the eight classes (the 40-value table on the Instrument tab), so a reading on one substrate is never compared against another's floor. The AUC column is the published single-substrate discrimination from the source literature, carried in the engine as a weight for combining substrates once more than one is lit; it is not a result of this chain.</p>")
    H.append("<h3>The specimens</h3><table class='t'><tr><th>specimen</th><th>measured on</th><th>what is known</th><th>status</th></tr>")
    for nm,on,note,st in SPECIMENS:
        badge="<b>LIT</b>" if st=="lit" else "<span class='pend'>RESERVED</span>"
        H.append(f"<tr><td><b>{nm}</b></td><td>{on}</td><td>{note}</td><td>{badge}</td></tr>")
    H.append("</table>")
    H.append("<h3>The grid</h3><table class='t'><tr><th>specimen \\ substrate</th>"+"".join(f"<th>{k}</th>" for k in R["sub_order"])+"</tr>")
    for nm,on,note,st in SPECIMENS:
        row=[f"<tr><td>{nm}</td>"]
        for k in R["sub_order"]:
            if nm=="whole blood" and k=="methyl": row.append("<td style='background:#1d3a24'><b>LIT</b></td>")
            elif k!="methyl" and "blood" in nm: row.append("<td class='m'>needs cfDNA</td>")
            elif k!="methyl": row.append("<td class='m'>-</td>")
            else: row.append("<td class='pend'>needs 3 layers</td>")
        H.append("".join(row)+"</tr>")
    H.append("</table>")
    H.append("""<h3>What lighting one cell requires</h3>"
      "<p><b>A pipeline map.</b> One affine fit taking that specimen-and-substrate's values onto the scale the floors were calibrated on. Measured once per processing pipeline, on identity loci.</p>"
      "<p><b>Presence floors and a detection panel.</b> Healthy specimens of that kind through the same Stage 1, so the deconvolver's presence floors and the foreign-cell detector's line are that specimen's and that laboratory's own (36 or more arrays for a line). This calibrates what the instrument does to a known input; it does not define healthy.</p>"
      "<p>A cell's A is read against 1.00 on every specimen and substrate; the tolerance is the tier scale."
      "<p class='m'>The frozen floors, by contrast, do transfer - they are a property of the cell class, not of the specimen or the laboratory.</p>
<p class='m'>The honest position, stated the way the record requires: for every reserved cell above, the chain has <b>not yet been tested</b> on that combination. That is not a statement that it cannot read it.</p>""")
    H.append(deepdive(R,"the substrates, their floors and the saturation argument"))
    return "".join(H)


def _intake_block(o):
    """Stage 0's record for this specimen, or NOT RUN. A gate that did not run is never shown as a pass."""
    rec = o.get("intake")
    if not rec:
        why = ("intake was skipped with --no-intake" if o.get("intake_skipped") else
               "this bundle was built from a beta table, so the file-level gates have nothing to read"
               if o.get("from_betas") else "no Stage 0 record was supplied with this bundle")
        return ("<h3>Stage 0 - chain of custody</h3>"
                "<p class='pend'>NOT RUN - " + why + ". The reading below is unaffected in value, but this "
                "specimen carries no arrival gate, no integrity hash and no QC decision.</p>" +
                "<p class='m'>What each refusal means and what to do about it is on the Troubleshooting tab of this report.</p>")
    rows = [("Stage 0 decision", rec.get("stage0_verdict") or "-"),
            ("hard failures", ", ".join(rec.get("stage0_hard_fail") or []) or "none"),
            ("borderline", ", ".join(rec.get("stage0_borderline") or []) or "none"),
            ("deferred", ", ".join(rec.get("stage0_deferred_qc") or []) or "none"),
            ("array type", f"{rec.get('array_type')} declared, {rec.get('array_type_detected')} read from the header"),
            ("control probes (0.4)", rec.get("ctrl_qc") or "-"),
            ("detection p (0.5)", rec.get("detection_qc") or "-"),
            ("bead count (0.6)", rec.get("bead_qc") or "-"),
            ("call rate (0.7)", f"{rec.get('call_rate')} - {rec.get('call_rate_status')}"),
            ("reference coverage (0.7b)", f"{rec.get('hm450_reference_coverage')} - {rec.get('hm450_coverage_gate')}"),
            ("sex check (0.8)", f"{rec.get('sex_check')} (predicted {rec.get('predicted_sex')}, "
                                f"declared {rec.get('declared_sex')})"),
            ("integrity", rec.get("integrity_status") or "-"),
            ("Grn sha256", (rec.get("grn_sha256") or "")[:24] + "..."),
            ("Red sha256", (rec.get("red_sha256") or "")[:24] + "..."),
            ("sample run id", rec.get("sample_run_id") or "-")]
    H = ["<h3>Stage 0 - chain of custody, this specimen</h3>",
         "<p class='m'>The gates the specimen passed before anything was calibrated or scored, as recorded by "
         "<code>stage_0_intake.py</code> (SOP sections 11-19). A QUARANTINE decision stops the chain: no report "
         "is written at all, so a report in your hands means these gates were cleared or explicitly deferred.</p>",
         "<table class='t'>"]
    for k, v in rows:
        H.append(f"<tr><td class='m'>{k}</td><td>{v}</td></tr>")
    H.append("</table>")
    H.append("<p class='m'>What each refusal means and what to do about it is on the Troubleshooting "
             "tab of this report, under the step that refused it in the SOP (sections 11 to 19), and "
             "in the Operations Manual section 3b with the healthy distribution behind every threshold.</p>")
    if rec.get("stage0_deferred_qc"):
        H.append("<p class='m'><b>Deferred means not measured.</b> A deferred gate is neither a pass nor a "
                 "failure - it is a check this run could not make, named so that nobody reads its silence as "
                 "consent.</p>")
    return "".join(H)


def tab_troubleshooting(o, R):
    """What to do when the chain refuses - the same content as the SOP's step sections and the Operations Manual s3b.

    A reader holding one reading should not have to open the procedure to find out what a refusal means, so
    the refusals live here too, keyed by the exact string the chain prints. Added 2026-09-23 at the author's
    instruction that this belongs in the report, the SOP and the manual rather than in a document beside them.
    """
    SOP = ("https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/MethylPhys/sop/"
           "MethylPhys_CPG_SOP.md")
    MAN = ("https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/MethylPhys/manual/"
           "MethylPhys_CPG_Operations_Manual.pdf")
    OUT = ("https://github.com/hmahaffeyges/IAM-Validation/blob/main/Biological_Physics/MethylPhys/doors/"
           "PROC_STAGE0_02_OUTCOME.md")
    H = ["<h2>Troubleshooting - what each refusal means, and what to do</h2>",
         "<p>The chain has three ways of not giving you an answer and they mean different things. "
         "<b>QUARANTINE</b>: Stage 0 refused the specimen - nothing is scored, no report is written, the run "
         "exits with code 2. <b>NOT REPORTABLE</b> (also UNSET, NOT ASSESSABLE): the measurement was made and "
         "the chain will not place a number on it, because a reference it needs does not exist - the value is "
         "unplaced, not wrong. <b>DEFERRED</b>: a check could not be made, and a deferred check is never a "
         "pass. A fourth, <b>PROVISIONAL</b>, means a threshold exists in the procedure but has never been "
         "measured against healthy specimens, so the value is printed and not refused on.</p>",
         "<p class='m'>Full detail under the step that refused: <a href='" + SOP + "'>the SOP, sections 11 to "
         "19</a>. The same material with the healthy distributions: <a href='" + MAN + "'>the Operations Manual, section "
         "3b</a>. The 732-array run these numbers come from: <a href='" + OUT + "'>PROC-STAGE0-02</a>.</p>",
         "<h3>Stage 0 refused the specimen</h3>",
         "<table><tr><th>what was printed</th><th>what it found</th><th>what to do</th></tr>"]
    for a, b, c in (
        ("QUARANTINE_INCOMPLETE_MANIFEST", "a required manifest field is missing or empty",
         "the seven fields are exact: sentrix_id, array_type, patient_id, intake_date, substrate, "
         "declared_sex, declared_chronological_age. Pass --sex and --age; an array with no recorded age "
         "cannot clear this gate"),
        ("QUARANTINE_MANIFEST_INVALID", "a field is present but not acceptable; the flag names it",
         "array_type must be HM450K, EPIC_v1 or EPIC_v2 - '450k' is rejected. patient_id must be a hashed "
         "token of at least 16 alphanumeric characters; run_sample.py hashes it for you"),
        ("QUARANTINE_MISSING_CHANNEL", "one of the two IDAT files is absent",
         "both --grn and --red are required; a missing Red channel cannot be recovered from the Grn"),
        ("QUARANTINE_TRUNCATED_UPLOAD", "an IDAT is smaller than 1 MB",
         "the transfer did not finish - re-fetch. The flag prints the size it found"),
        ("QUARANTINE_ARRAY_TYPE_MISMATCH", "the declared type and the file's own header disagree",
         "believe the header: omit --array-type and the chain reads it from the file. A 450K array reports "
         "622,399 addresses, an EPIC v1 about 1,051,943"),
        ("QUARANTINE_CORRUPT_IDAT", "the decoder reached the file and failed on it",
         "re-fetch. A file can pass the 1 MB floor and still be truncated inside its compressed stream, so a "
         "size check is not an integrity check"),
        ("RE_TRANSMISSION_DETECTED", "these exact bytes were already taken in against this custody log",
         "for a legitimate re-run use a different intake log, or none. If a duplicate was not expected, find "
         "out who submitted the first one"),
        ("sex MISMATCH", "chrX and chrY intensities disagree with the declared sex",
         "check the paperwork first - this call agrees with published labels on 729 of 731 healthy arrays. A "
         "array of unknown sex cannot clear this gate"),
        ("FAIL_LOW_DETECTION / CALL_RATE_FAIL", "too many probes are indistinguishable from background",
         "a specimen or hybridisation problem, not a configuration one: healthy arrays clear these with two "
         "decimal places to spare (table below)"),
        ("coverage FAIL", "under 80 per cent of the reference CpGs survived calibration",
         "usually the wrong array type - check the platform before the specimen"),
        ("WARN_LOW_BEAD_COUNT", "a borderline flag, not a refusal: the sample is scored with a penalty",
         "nothing for one array; a plate where many warn together is a scanning pattern worth raising with "
         "the core facility")):
        H.append("<tr><td class='m'>" + a + "</td><td>" + b + "</td><td>" + c + "</td></tr>")
    H.append("</table>")
    H.append("<h3>What whole-blood arrays measure at intake, so you can tell a bad array from a bad configuration</h3>")
    H.append("<p class='m'>731 whole-blood arrays, four Sentrix-chip years. A result far from these "
             "is the array; a result at zero or one is the configuration.</p>")
    H.append("<table><tr><th>check</th><th>threshold</th><th>panel median</th><th>worst</th>"
             "<th>outside the threshold</th></tr>")
    for a, b, c, d, e in (("detection p", "&ge; 0.99", "0.9994", "0.9951", "0 of 731"),
                          ("call rate", "&ge; 0.98", "0.9983", "0.9889", "0 of 731"),
                          ("bead count", "&ge; 0.995", "0.9988", "0.9900", "9 of 731 (warn)"),
                          ("bisulfite conversion", "&ge; 0.95", "0.7979", "0.6354", "731 of 731"),
                          ("sex call", "agreement", "729 of 731 agree", "-", "2, both recorded as NA")):
        H.append("<tr><td>" + a + "</td><td>" + b + "</td><td>" + c + "</td><td>" + d + "</td><td>" + e +
                 "</td></tr>")
    H.append("</table>")
    H.append("<p class='m'><b>The bisulfite row is why one gate is PROVISIONAL.</b> A threshold that refuses "
             "731 of 731 healthy specimens is not measuring specimen quality, so the chain prints the value "
             "and does not refuse on it, and the decision records it as deferred rather than passed. No "
             "specimen is passed that a calibrated gate would fail, and none is refused on a number nobody "
             "has measured.</p>")
    H.append("<h3>It ran, but no number was placed</h3>")
    H.append("<table><tr><td>detection not commissioned for this laboratory</td><td>the foreign-cell detector has no noise panel for this laboratory; the cells' A are still read - only the foreign-cell line is missing</td><td>commission the laboratory once: 36 or more healthy whole-blood arrays through the same Stage 1, then kit/commission_detection_lab.py</td></tr>")
    H.append("</table>")
    H.append("<h3>It will not start</h3>")
    H.append("<table><tr><th>symptom</th><th>cause</th><th>fix</th></tr>")
    for a, b, c in (
        ("calibration hangs with no output, or fails on a manifest download",
         "the decoder wants to download the array manifest into a home directory it cannot write",
         "point HOME at a writable cache for the run; the first run fetches the manifest once, after which "
         "calibration is about 26 s per array"),
        ("atlas not found: IAMAtlasREBUILD.csv.xz", "the atlas is stored compressed",
         "nothing to do - the runner decompresses it once (605 MB) and says so"),
        ("Missing optional dependency 'pyarrow'", "the synthetic generator writes parquet",
         "pip install pyarrow"),
        ("a batch script dies with a process-pool error", "some environments forbid process pools",
         "use threads, as the published batch scripts do"),
        ("ModuleNotFoundError on a stage module", "the chain directory is not on the path",
         "run run_sample.py from its own directory; if files were moved, build_chain_sequence.py names what "
         "is no longer reachable")):
        H.append("<tr><td>" + a + "</td><td>" + b + "</td><td>" + c + "</td></tr>")
    H.append("</table>")
    H.append("<h3>The reading itself looks wrong</h3>")
    H.append("<p>Two layers make a cell's reading absolute and they are separate on purpose: the floor (H_min per class, calibrated by MCMC, never re-derived per pipeline) and the pipeline map (the same healthy blood reads 0.737 on the reference scale and 0.815 through Stage 1 noob from raw IDATs - a within-pipeline comparison cancels that offset and never sees it, an absolute reading does not). A reading that skips the map is not slightly wrong. The check that catches it in one line: run a handful of your own healthy specimens. The A of every present cell should land inside NORMAL (0.95-1.05) with no zero applied - that is how the map was verified in the first place. If they do not, the map is missing or wrong for this pipeline, and the chain will have printed which.</p>")
    H.append("<h3>Foreign-cell detection (Stage 2d) printed something you did not expect</h3>")
    H.append("<table><tr><th>what was printed</th><th>what it means</th><th>what to do</th></tr>")
    for a, b, c in (
        ("DETECTION_NOT_COMMISSIONED", "this laboratory has no detection panel in detection_panel_v1.json",
         "commission it on >= 36 of the laboratory's own healthy whole-blood arrays; the chain never borrows another laboratory's line, "
         "because a line set on four laboratories fired on 43-46 % of a fifth laboratory's whole-blood arrays at full marker resolution (PROC-MF-03 B8)"),
        ("FOREIGN_UNSPECIFIC - N of 21 foreign cells detected together", "the specimen is not blood-like to the detector: every solid-tissue column rises at once",
         "check the substrate, the pipeline flag (--pipeline) and whether this laboratory's panel was commissioned on data processed the same way; "
         "do not read any one detection"),
        ("FOREIGN_CELL_DETECTED with 'not measured' in the limit column", "the cell has a line from the null but its sensitivity was never measured by a procedure",
         "treat it as a reading to follow up; only Breast, Colon_epithelial_cells, Cortical_neurons and Prostate carry a measured limit today"),
        ("a detected cell whose A-score is not printed", "detection is presence; the A needs the cell above its presence floor in the deconvolution",
         "this is correct behaviour - a fraction of 0.5 % can be detected and still be too small to score"),
        ("a family: row in the detection table", "cells the array cannot tell apart (stomach diff / undiff, the progenitors) are detected as one column",
         "the fraction belongs to the family; no member is singled out"),
    ):
        H.append(f"<tr><td><code>{_e(a)}</code></td><td>{_e(b)}</td><td>{_e(c)}</td></tr>")
    H.append("</table>")
    H.append("<h3>How these were found, which is how to look for yours</h3>")
    H.append("<p class='m'>A gate that cannot read its input never fires - the header reader opened IDAT "
             "files raw while public downloads are gzipped, so the array-type check silently never ran on "
             "public data; it did not error, it returned 'unreadable' and everything continued. A value that "
             "fails to propagate looks like a value that is wrong - an identifier dropped between two steps "
             "made the next step refuse every array in a 732-array set, and the message blamed the data. A "
             "refusal that does not stop the run is reported as something else downstream. And a failure "
             "logged as 'deferred' is worse than no check at all. If a check has never reported a failure, "
             "test it with input you know is bad.</p>")
    return "".join(H)


def _cmb_tools(o):
    """The CMB tool registry, evaluated on this bundle."""
    try:
        sys.path.insert(0, ENGINE)
        import cmb_tools as CT
        return CT.evaluate(o)
    except Exception as e:
        return [{"id": "REGISTRY", "tool": "the CMB tool registry itself", "borrowed_from": "-",
                 "what_it_does_here": "-", "where": "cmb_tools.py", "status": "FAIL",
                 "evidence": "the registry could not be evaluated: %s" % _e(e)}]


def _cmb_tool_table(o):
    """Every tool borrowed from cosmology, with its state on this run. A FAIL is also a red flag.

    Added 2026-09-25 on the author's instruction: the borrowings will multiply, so they are a register with a
    status per run rather than a paragraph. NOT_BUILT entries are listed on purpose - the shelf is part of
    the record.
    """
    T = _cmb_tools(o)
    if not T:
        return ""
    order = {"FAIL": 0, "PASS": 1, "NOT_RUN": 2, "NOT_APPLICABLE": 3, "NOT_BUILT": 4}
    T = sorted(T, key=lambda t: (order.get(t["status"], 9), t["id"]))
    counts = {}
    for t in T:
        counts[t["status"]] = counts.get(t["status"], 0) + 1
    H = ["<h3>The cosmology toolkit, and what it did on this specimen</h3>",
         "<p>Every method this chain borrowed from CMB analysis, with its state on this run. "
         "<b>NOT_BUILT</b> means borrowed in principle and not implemented yet - listed on purpose. A "
         "<b>FAIL</b> is carried to the Red flags tab as well.</p>",
         "<p>" + " &nbsp;&middot;&nbsp; ".join("<b>%d %s</b>" % (n, k) for k, n in
                 sorted(counts.items(), key=lambda kv: order.get(kv[0], 9))) + "</p>",
         "<table><tr><th>state</th><th>tool</th><th>borrowed from</th><th>what it does here</th>"
         "<th>implemented in</th><th>evidence on this run</th></tr>"]
    for t in T:
        H.append("<tr><td class='m'><b>%s</b></td><td>%s</td><td class='m'>%s</td><td>%s</td>"
                 "<td class='m'>%s</td><td>%s</td></tr>"
                 % (t["status"], html.escape(t["tool"]), html.escape(t["borrowed_from"]),
                    html.escape(t["what_it_does_here"]), html.escape(t["where"]),
                    html.escape(str(t["evidence"]))))
    H.append("</table>")
    return "".join(H)


def tab_safeguards(o, R):
    """Every guard the chain has, with its result. Written by release_check.py (one command, commissioning row N)
    and read here - a guard that has not been run prints NOT RUN, never a pass."""
    rel=R.get("release"); H=["<h2>Safeguards - every guard, and whether it passed</h2>", _intake_block(o),
      "<p>The chain's guarantee is not that it is clever; it is that the things that could make it wrong are each checked by something that fails loudly. "
      "This page is written by one command - <code>release_check.py</code> - and read here. A guard that could not run prints what it needs. "
      "<b>A guard that has not been run prints NOT RUN and is never shown as a pass.</b></p>"]
    if not rel:
        H.append("<p class='pend'>NOT RUN - no release_check.json found. Run <code>python3 release_check.py</code> in the reproduction kit.</p>")
    else:
        BADGE={"PASS":"<b style='color:#3fa45b'>PASS</b>","FAIL":"<b style='color:#c0392b'>FAIL</b>",
               "SKIPPED":"<b style='color:#d68910'>SKIPPED</b>","INCONCLUSIVE":"<b style='color:#d68910'>INCONCLUSIVE</b>"}
        H.append(f"<p class='m'>Run {_e(rel['run_at'])} at commit <code>{_e(rel['commit'])}</code> &middot; "
                 f"<b>{rel['n_pass']} pass, {rel['n_fail']} fail, {rel['n_skipped']} skipped, {rel.get('n_inconclusive',0)} inconclusive</b>. "
                 f"This report was generated at commit <code>{_e(R['sha'])}</code>"
                 + ("" if rel['commit']==R['sha'] else " - <b>which is not the commit the guards were run at; re-run the release check</b>") + ".</p>")
        H.append("<table class='t'><tr><th>guard</th><th>result</th><th>what it guards against</th><th>what it printed</th></tr>")
        for g in rel["guards"]:
            det=g.get("detail") or ""
            if g["status"]=="SKIPPED" and g.get("needs"): det=f"needs {g['needs']}"
            H.append(f"<tr><td><b>{_e(g['name'])}</b><br><span class='m'>{_e(g['argv'])}</span></td><td>{BADGE.get(g['status'],g['status'])}</td>"
                     f"<td class='m'>{_e(g['guards'])}</td><td class='m'>{_e(det)}</td></tr>")
        H.append("</table>")
        H.append("<p class='m'><b>How to read a SKIPPED.</b> It means the guard exists and runs, but the data it checks against is not on this machine - "
                 "a multi-gigabyte public data set, or the test package. It is not a pass and it is not a failure; it is a statement that this particular "
                 "run could not exercise it. A reader who downloads the named data gets the result.</p>")
    # the second opinion, per sample
    so=o.get("second_opinion") or {}
    H.append("<h3>Composition cross-check - NILC beside the constrained solver</h3>")
    if not so.get("available"):
        H.append(f"<p class='pend'>NOT RUN for this sample - {_e(so.get('reason','not requested'))}</p>")
    else:
        ok=so["agreement"]=="AGREE"
        H.append(f"<p>Result: <b style='color:{'#3fa45b' if ok else '#d68910'}'>{so['agreement']}</b> against the bar <i>{_e(so['bar'])}</i>. "
                 f"Class-level L1 {so['L1_class']}, cell-level L1 {so['L1_cell']}. {_e(so['note'])}</p>"
                 "<table class='t'><tr><th>class</th><th>legacy (reported)</th><th>NILC (cross-check)</th><th>difference</th></tr>"
                 +"".join(f"<tr><td>{_e(CLASS_LABEL.get(c,c))}</td><td class='n'>{100*v['legacy']:.1f} %</td><td class='n'>{100*v['nilc']:.1f} %</td>"
                          f"<td class='n'>{100*v['abs_diff']:.1f} pp</td></tr>" for c,v in so["by_class"].items())
                 +"</table>"
                 "<p class='m'>Why two solvers. legacy's constrained fit is conservative: a cell is placed only when the evidence forces it, which is why the "
                 "composition the report stands on is not inflated. NILC - the needlet internal linear combination, the component-separation method Planck uses - "
                 "is variance-weighted and deliberately sensitive to faint components. In July 2026 NILC was switched off for disagreeing with legacy on every "
                 "blood sample; PROC-NILC-01 later found the disagreement was the finding, not the fault: NILC was reporting that the atlas cannot separate the "
                 "blood classes, which PROC-SEP-03 then measured directly. It is back, as a second column and an agreement flag - never as the reported "
                 "composition. Cell-level disagreement <i>inside</i> one lineage is expected and is not scored; class-level disagreement is.</p>")
    H.append(deepdive(R,"the guards and the null suite"))
    H.append(_cmb_tool_table(o))   # the register, 2026-09-25
    return guard("".join(H),"Safeguards")


ROADMAP=[
 ("now","Find a serial cohort - two draws, same person","the difference map is the strongest design on the Sky tab and nothing held here has a repeat draw; the technical noise term is currently an upper bound inferred from cross-sectional data","EPIC-Italy and the Uppsala follow-up arms are the candidates; needs repeat-draw metadata, which the public extracts do not carry"),
 ("now","Merge the atlas's duplicate labels to one lineage per entry","the same lineage appears under several atlas entries whose per-cell readings differ by more than anything biological, purely by which reference panel defined the markers; until they are merged nobody can say 'which cell moved'","a merge rule plus a re-measured per-entry reference; it is the blocker on per-cell reporting"),
 ("gate 0","Rebuild the eight class bands from one healthy cohort through one pipeline","RECON B1 - the bands still carry the history of how they were assembled","the four-laboratory panel exists; this is the re-fit"),
 ("gate 0","Wire the Stage 0 to Stage 1 intensity hand-off","four QC checks are deferred rather than running, because Stage 0 never sees the raw intensities","PROC-STAGE0-01; map rows 10-11"),
 ("gate 0","Reproduce the breast pre-symptomatic-window anchor from raw IDATs","the sealed anchor reproduces from processed betas; from raw IDATs it has not been run end to end","sprint F1"),
 ("after gate 0","The full MCMC covariance in the composition step","the atlas carries the covariance between cell types at each address and the chain ignores it; a generalised-least-squares separation would use it, and it is exactly what the blood-separability finding needs","the largest piece of unspent evidence in the chain (see Sky)"),
 ("after gate 0","Cellular variance as cosmic variance","one methylome is one realisation; how much of the spread between healthy people is irreducible sampling rather than biology","map row 3"),
 ("after gate 0","Transfer function of the chain","what the pipeline does to a signal of known size and shape, measured rather than assumed","map row 14; CCL-039"),
 ("after gate 0","Nuisance marginalisation in the gauge","age, sex, chip and laboratory are currently subtracted as point estimates; marginalising over them propagates their uncertainty into the reading","map row 21"),
 ("after the anchor","C(d) - the two-point correlation of residuals against genomic distance, per class","the first real step toward a power spectrum, and the quantity behind the spatial null; a healthy correlation scale is a measurable baseline","map rows 17, 23, 25; sprint C1"),
 ("after the anchor","The angular power spectrum itself","the scale-resolved observable the sphere was built for: at what genomic scale does a departure live, with the look-elsewhere effect handled by simulation","not started; the frontier named on the Sky tab"),
 ("after the anchor","Banana degeneracy - the 2D posterior shape for A-score pairs","two classes can be individually uncertain and jointly well determined; the shape of that degeneracy is the honest error bar and it is not an ellipse","map row 38; sprint C3 - the author's outstanding request, 'I never got my banana degeneracy'"),
 ("after the anchor","Per-card likelihood, marginalised, with MCMC posteriors","a proper likelihood per class rather than a point reading against a band","map rows 28, 37; sprint E2/E3"),
 ("after the anchor","Formal blinding for confirmation runs","the analyst should not know the arm; the pre-registration protocol allows it, the tooling does not enforce it","map row 78"),
 ("clinical format","A per-chromosome linear track and a Hilbert-curve layout","a sphere is the right object for spherical statistics and the wrong picture for a clinician; the same residual should be renderable in chromosome coordinates","see 'Is the sphere necessary' on the Sky tab"),
 ("substrates","Urine, CSF, and the within-patient tissue / plasma / urine trio","each needs its own pipeline map and presence floors before it can read; the trio would test whether one person's classes agree across specimens","the Operations Manual section 7; see Coverage"),
 ("not now","Bispectrum and trispectrum; Minkowski functionals; isotropy and alignment tests","higher-order sky statistics; they need the power spectrum first and a reason to look","map rows 45, 53-60, 69"),
 ("not now","5mC / 5hmC as an E/B-mode separation; multi-omics cross-correlation","a genuinely deep parallel - two components of one field - but it needs oxidative-bisulphite data the chain has never seen","map rows 4, 49, 70"),
 ("does not translate","Rees-Sciama; Rayleigh scattering","recorded so nobody spends a week on them: the analogy breaks, and saying so is part of the map","map rows 63, 68"),
]

# tab_roadmap: removed 2026-09-26 by the author's decision - the enhancement list is internal planning, not for a researcher; it stays in doors/ENHANCEMENTS.md



def tab_inventory(R):
    """Every live file of the chain, enumerated by build_chain_inventory.py rather than hand-listed, so a file
    cannot be silently omitted. Added 2026-09-22 after the author asked whether the Chain tab lists everything:
    an audit found 20 load-bearing files linked nowhere, including the atlas itself."""
    inv=R.get("inv")
    if not inv: return guard("<h2>Chain inventory</h2><p class='pend'>chain_inventory_v1.json not found - run build_chain_inventory.py</p>","Files")
    m=inv["_meta"]; rows=inv["files"]
    ROLE=[("chain","In the chain","Executed on every run, in stage order. The conductor resolves each of these by name and fails loudly if one is missing."),
          ("reference","Reference and calibration data","The files the chain reads: the atlas and its per-cell references, the class floors, the identity loci, the pipeline maps, the presence floors, the detector panel, the sky scales. Every one is listed on the Run tab of each report with the hash the run saw."),
          ("interface","Interface","This report and the commands that drive it."),
          ("guard","Guards, procedures and doors","The conformance tests, the sealed procedures, the protocol gate, and the documents a reader should start from."),
          ("record","Present but NOT in the chain","Callable and kept for the record - most of it from the preliminary era. run_full() does not read any of it. The disease matrix is here because the author removed disease matching from the chain on 2026-09-21."),
          ("superseded","Superseded","A later file does the job; kept so the lineage is visible.")]
    H=["<h2>Every file the chain uses</h2>",
       f"<p>Enumerated from the live tree by <code>build_chain_inventory.py</code> at commit <code>{_e(m.get('commit',''))}</code> - "
       f"<b>{m['n_files']} files</b>, each with its role, its purpose, its size and its SHA-256. The table is generated, not maintained by hand: "
       f"any file without a description is emitted as <b>UNDESCRIBED</b> and counted here, so a gap is visible rather than silent. "
       f"Undescribed at this build: <b>{m['n_undescribed']}</b>.</p>",
       "<p class='m'>Roles are measured, not asserted: <i>in the chain</i> means resolved by <code>cpg_conductor._find()</code> or loaded by a module "
       "that is; <i>not in the chain</i> means present and callable but never reached from <code>run_full()</code>.</p>"]
    for role,label,blurb in ROLE:
        rs=[r for r in rows if r["role"]==role]
        if not rs: continue
        H.append(f"<h3>{_e(label)} <span class='m'>({len(rs)})</span></h3><p class='m'>{_e(blurb)}</p>")
        H.append("<table class='t'><tr><th>file</th><th>stage</th><th>what it is and why it is there</th><th>size</th><th>sha256</th></tr>")
        for r in rs:
            nm=_link(R,r["file"]) if r["file"] in R["files"] else _e(r["file"])
            sz=f"{r['bytes']/1e6:.1f} MB" if r["bytes"]>1e6 else f"{r['bytes']//1000} KB" if r["bytes"]>1500 else f"{r['bytes']} B"
            H.append(f"<tr><td>{nm}<br><span class='m'>{_e(r['path'])}</span></td><td class='m'>{_e(r['stage'])}</td>"
                     f"<td>{_e(r['description'])}</td><td class='n'>{sz}</td><td class='m'><code>{_e(r['sha256_12'])}</code></td></tr>")
        H.append("</table>")
    if m.get("undescribed"):
        H.append("<div class='warn'><b>Undescribed at this build:</b> "+", ".join(f"<code>{_e(x)}</code>" for x in m["undescribed"])+
                 " - present in the tree with no description in the generator. Listed here rather than omitted.</div>")
    H.append(deepdive(R,"the chain and its files"))
    # EXEMPT from the vocabulary guard, for the same reason the Healthy-reference tab is: an inventory that cannot
    # name disease_cell_signature_matrix_v1_13.csv is a false inventory. The guard still applies to every
    # measurement tab. Nothing here is a statement about this sample - it is a list of files on disk.
    return no_changelog("".join(H),"Files")



def tab_findings(R):
    """Every validation run on the commissioned chain, from its structured finding record (val_finding.py,
    schema val_finding_v1). Written 2026-09-22, before the first run, so that every run is comparable to every
    other and nothing has to be reconstructed afterwards."""
    fs=R.get("findings") or []
    H=["<h2>Validation findings</h2>",
       "<p>Each run on this chain writes a structured record - <code>val_finding.py</code>, schema "
       "<code>val_finding_v1</code> - and this tab reads those records. The schema was written <i>before</i> the first "
       "run, on the principle that a record designed afterwards is shaped by whatever was convenient to save.</p>",
       "<h3>What a finding record holds, and why each part is there</h3>",
       "<table class='t'><tr><th>block</th><th>what it holds</th><th>why a researcher needs it</th></tr>",
       "<tr><td><b>instrument</b></td><td>the SHA-256 of all 14 reference layers - floors, identity loci, markers, "
       "exclusivity, collinearity groups, pipeline maps, tiers, presence floors, sky mapping, "
       "directional panels, atlas provenance - plus the repository commit</td>"
       "<td>a reading is only meaningful against a stated instrument. When a layer is re-sealed these hashes change, "
       "so two findings can be compared only if their fingerprints match - and the record makes that checkable "
       "rather than assumed</td></tr>",
       "<tr><td><b>samples</b></td><td>one row per array: arm, age, every class reading, the departure, the sky "
       "summary, and the refusals that applied</td><td>the unit of this instrument is a per-sample absolute "
       "reading. Storing the samples means every summary above them can be recomputed, and an arm difference can "
       "never quietly become the result</td></tr>",
       "<tr><td><b>cells</b> and <b>groups</b></td><td>per atlas entry and per lineage group: where it was placed, "
       "the median reading by arm, the <b>direction</b> (above / below / within the NORMAL tolerance about A = 1.00), the "
       "<b>magnitude</b> in units of that entry's healthy spread, the <b>prevalence</b> (what fraction of the arm "
       "departed), its panel exclusivity, and the claim level this permits</td>"
       "<td>this is the answer to <i>which cells are doing what, in which direction, by how much</i>. Magnitude is "
       "in healthy spreads rather than raw A so that a loose entry and a tight one are comparable; prevalence "
       "separates 'most of the arm moved a little' from 'a few moved a lot'</td></tr>",
       "<tr><td><b>bars</b></td><td>every pre-registered bar with its threshold, its measured value, and PASS or "
       "FAIL AS SEALED</td><td>the outcome is scored against what was written before the run, not after it</td></tr>",
       "<tr><td><b>not_assessable</b></td><td>what could not be read, and why</td><td>the honest half of any "
       "result - a class below its presence floor is absent, not normal</td></tr>",
       "<tr><td><b>matrix_evidence</b></td><td>for this condition and specimen: the per-group direction, magnitude "
       "and prevalence, tagged with the instrument fingerprint</td>"
       "<td>the record-side precursor of a signature matrix (not consulted by the chain). <b>Accumulating evidence, not a matching rule:</b> a matrix becomes "
       "possible only once several conditions have been measured on the same instrument, and the chain does not "
       "read this block back to classify anything. A finding that names no condition carries none of it</td></tr>",
       "</table>",
       "<p class='m'>Two rules are built into the writer rather than left to the operator. A finding cannot be "
       "written without its instrument fingerprint. And the per-cell claim level is taken from the reference, not "
       "chosen per run: <code>individual</code> where the entry is separable and its panel exclusive, "
       "<code>group_only</code> where the atlas cannot separate it from its group, <code>withheld_panel_shared</code> "
       "where the marker panel is mostly shared with other entries.</p>"]
    if not fs:
        H.append("<div class='warn'><b>No findings recorded yet.</b> The schema and its writer are in place and "
                 "exercised end to end on the eleven commissioning arrays; the first real run will appear here. "
                 "Records live in <code>Record/VAL_FINDINGS/</code> and are linked from the Record tab.</div>")
    else:
        H.append("<h3>Recorded runs</h3><table class='t'><tr><th>VAL</th><th>title</th><th>condition</th>"
                 "<th>specimen</th><th>arms</th><th>bars</th><th>entries with a departure</th><th>instrument commit</th></tr>")
        for f in fs:
            arms=f.get("cohort",{}).get("arms") or {}
            dep=sum(1 for c in (f.get("cells") or {}).values()
                    if any((v or {}).get("direction") in ("above","below") for v in (c.get("by_arm") or {}).values()))
            bars=f.get("bars") or []
            bp=sum(1 for b in bars if b.get("passed") is True)
            H.append(f"<tr><td><b>{_e(f.get('val_id',''))}</b></td><td>{_e(f.get('title',''))}</td>"
                     f"<td>{_e(f.get('condition') or '-')}</td><td>{_e(f.get('specimen') or '-')}</td>"
                     f"<td class='m'>{_e(', '.join(f'{k} n={v}' for k,v in arms.items()))}</td>"
                     f"<td class='n'>{bp} / {len(bars)} passed</td><td class='n'>{dep}</td>"
                     f"<td class='m'><code>{_e((f.get('instrument') or {}).get('_commit','')[:8])}</code></td></tr>")
        H.append("</table>")
        for f in fs:
            cells=f.get("cells") or {}
            rows=[(k,v) for k,v in cells.items()
                  if any((x or {}).get("direction") in ("above","below") for x in (v.get("by_arm") or {}).values())]
            rows.sort(key=lambda kv: -abs(max((abs((x or {}).get("magnitude_in_healthy_spreads") or 0)
                                                for x in (kv[1].get("by_arm") or {}).values()), default=0)))
            H.append(f"<details><summary><b>{_e(f.get('val_id',''))}</b> - {_e(f.get('title',''))} "
                     f"<span class='m'>({len(rows)} entries departed)</span></summary>")
            if f.get("bars"):
                H.append("<table class='t'><tr><th>bar</th><th>pre-registered statement</th><th>threshold</th>"
                         "<th>measured</th><th>as sealed</th></tr>")
                for b in f["bars"]:
                    v="PASS" if b.get("passed") is True else "FAIL AS SEALED" if b.get("passed") is False else "not scored"
                    H.append(f"<tr><td>{_e(b.get('bar',''))}</td><td>{_e(b.get('statement',''))}</td>"
                             f"<td class='n'>{_e(b.get('threshold'))}</td><td class='n'>{_e(b.get('measured'))}</td>"
                             f"<td><b>{v}</b></td></tr>")
                H.append("</table>")
            if rows:
                H.append("<table class='t'><tr><th>entry</th><th>claim level</th><th>arm</th><th>direction</th>"
                         "<th>magnitude (healthy spreads)</th><th>prevalence</th><th>n placed</th></tr>")
                for k,v in rows[:60]:
                    for a,x in (v.get("by_arm") or {}).items():
                        if (x or {}).get("direction") not in ("above","below"): continue
                        pv=x.get("prevalence_above") if x["direction"]=="above" else x.get("prevalence_below")
                        H.append(f"<tr><td>{_e(k)}</td><td class='m'>{_e(v.get('claim_level'))}</td><td>{_e(a)}</td>"
                                 f"<td><b>{_e(x['direction'])}</b></td><td class='n'>{_e(x.get('magnitude_in_healthy_spreads'))}</td>"
                                 f"<td class='n'>{_e(pv)}</td><td class='n'>{_e(x.get('n_placed'))}</td></tr>")
                H.append("</table>")
            for na in (f.get("not_assessable") or []):
                H.append(f"<p class='m'>not assessable: {_e(na.get('what'))} - {_e(na.get('reason'))}</p>")
            H.append("</details>")
    H.append(deepdive(R,"the validation record"))
    return no_changelog("".join(H),"Findings")   # exempt from the vocabulary guard only: a finding names the condition it measured


def _provenance_block(o):
    """What produced THIS reading: the chain commit, the decoder version, and a hash of every input read.

    Added 2026-09-23. The run already recorded this in its bundle, but a reader holding the report could not
    see which atlas or which band produced the number in front of them - and two readings from different
    months are only comparable if that is on the page. Covariate KEYS are listed without their values: the
    phenotype belongs in the custody record, not in a reading's prose.
    """
    v = o.get("versions") or {}
    if not v:
        return ["<h3>What produced this reading</h3>",
                "<p class='m'>This report was built from a bundle with no version block - it predates the "
                "2026-09-23 capture, so the input hashes were not recorded at run time.</p>"]
    H = ["<h3>What produced this reading</h3>",
         "<p>Two readings are comparable only if they came from the same inputs. Everything this run read is "
         "hashed below, so a reviewer can tell at a glance whether this reading and another used the same "
         "atlas, the same floors, the same identity loci and the same pipeline map.</p>",
         "<table><tr><th>run timestamp (UTC)</th><td class='m'>%s</td></tr>"
         "<tr><th>chain commit</th><td class='m'>%s%s</td></tr>"
         "<tr><th>decoder</th><td class='m'>methylprep %s, Python %s</td></tr></table>"
         % (v.get("run_timestamp_utc", "-"), v.get("chain_commit", "-"),
            " <b>(working tree had uncommitted changes)</b>" if v.get("chain_dirty") else "",
            v.get("methylprep", "-"), v.get("python", "-"))]
    ins = v.get("inputs") or {}
    if ins:
        H.append("<table><tr><th>input the chain read</th><th>SHA-256 (first 12)</th><th>size</th></tr>")
        for k in sorted(ins):
            d = ins[k] or {}
            H.append("<tr><td class='m'>%s</td><td class='m'>%s</td><td class='m'>%.1f MB</td></tr>"
                     % (k, d.get("sha256_12", "-"), (d.get("bytes") or 0) / 1e6))
        H.append("</table>")
    cov = ((o.get("intake") or {}).get("covariates") or {})
    if cov:
        # Not even the KEYS are printed: the vocabulary guard refused "diagnosis" on 2026-09-23, and it was
        # right to - a key name carries the phenotype as surely as its value does. The count tells a reader
        # the capture happened; the custody record tells them what it captured.
        H.append("<p><b>%d covariate field%s recorded with this run</b>, held in the custody record and the "
                 "bundle rather than here. A reading states what was measured; what the specimen was "
                 "declared to be is not a measurement, and naming it on this page would invite the number "
                 "to be read as a finding about it.</p>" % (len(cov), "" if len(cov) == 1 else "s"))
    else:
        H.append("<p class='m'>No covariates were recorded with this run. A run intended for a later "
                 "cross-sample analysis should pass them (<code>--covariate study=NAME</code>), because "
                 "nothing downstream can recover a phenotype the run did not capture.</p>")
    return H


SEVERITY = {"STOP": 0, "WITHHELD": 1, "CAUTION": 2, "NOTE": 3}


def red_flags(o, R=None):
    """Every red flag in one list, so none of them can hide in the middle of a tab.

    Each entry: code (stable, for a program), severity, where (which tab it came from), what happened, and
    what to do about it. Added 2026-09-25 after a missing dependency silently removed the specimen's own sky
    plate from every report and said so only in a grey note.
    """
    F = []

    def add(code, sev, where, what, todo):
        F.append({"code": code, "severity": sev, "where": where, "what": what, "what_to_do": todo})

    # 1. intake
    intake = o.get("intake") or {}
    v = intake.get("stage0_verdict")
    if v and str(v).upper().startswith("QUARANTINE"):
        add("STAGE0_QUARANTINE", "STOP", "Safeguards",
            "Stage 0 refused this specimen: %s" % v,
            "Nothing was scored. The cause is in the intake record; re-submit the specimen once it is fixed.")
    for k in (intake.get("stage0_deferred_qc") or []):
        add("STAGE0_DEFERRED", "CAUTION", "Safeguards",
            "An intake check could not be measured and was DEFERRED: %s" % k,
            "A deferred check is not a pass. Supply what it needs (usually the array's own control probes) "
            "or read the run knowing that gate did not fire.")
    if intake and not intake.get("integrity"):
        add("NO_INTEGRITY_HASH", "CAUTION", "Integrity",
            "No integrity hash was recorded for the raw files.",
            "Run with --intake-log so the custody record carries both file hashes.")
    if o.get("intake_skipped"):
        add("INTAKE_SKIPPED", "CAUTION", "Safeguards",
            "Stage 0 was skipped for this run (--no-intake).",
            "Every file-level gate is unmeasured. Use --no-intake only for a beta-file path where there are "
            "no IDATs to check.")

    # 2. the gauge: what was withheld and why
    for c, rec in (o.get("classes") or {}).items():
        if not rec.get("reportable"):
            add("GAUGE_WITHHELD", "WITHHELD", "Reading",
                "Class '%s': %s" % (c, rec.get("reason") or "no commissioned band"),
                "No placement and no tier are printed for this class. The number, where present, is the "
                "measurement; the missing piece is the healthy reference to judge it against.")
    # 2b. foreign-cell detection (Stage 2d, 2026-09-26)
    fd = o.get("foreign_detection") or {}
    _st = fd.get("status") or ""
    if _st.startswith("NOT_COMMISSIONED"):
        add("DETECTION_NOT_COMMISSIONED", "WITHHELD", "Cells",
            "Foreign-cell detection has no commissioned panel for this laboratory; no line is borrowed from another.",
            "Commission the laboratory's detection panel on >= 36 of its own healthy whole-blood arrays (kit/commission_detection_lab.py); "
            "until then the composition is the only statement about foreign cells.")
    elif _st.startswith("OK_BUT_UNSPECIFIC"):
        add("FOREIGN_UNSPECIFIC", "CHECK", "Cells",
            _st.split(": ", 1)[-1],
            "Every foreign cell rising together is what a specimen that is not blood-like looks like to the detector - substrate, "
            "processing, or a platform the panel was not commissioned on. Do not read any single detection from this run.")
    elif _st == "OK" and fd.get("detected"):
        add("FOREIGN_CELL_DETECTED", "CHECK", "Cells",
            "Above the laboratory's own line: %s" % ", ".join(fd["detected"]),
            "A presence statement, not a diagnosis. Confirm on a second draw; read the cell's A only if it clears its presence floor. "
            "Cells with no measured detection limit are readings to follow up, not results.")
    elif _st.startswith("WITHHELD"):
        add("DETECTION_WITHHELD", "WITHHELD", "Cells", _st.split(": ", 1)[-1],
            "The composition guard did not verify the specimen as blood-like, so no detection line is applied.")
        add("NO_LAB_ZERO", "WITHHELD", "Reading",
            "This laboratory has no commissioned zero, so no absolute reading is possible on any class.",
            "Commission the laboratory (PROC-MAHA-01) on its own healthy arrays, or read this run only for "
            "composition.")


    # 4. the sky
    sky = o.get("patient_sky") or {}
    if not sky.get("available"):
        add("NO_SKY", "WITHHELD", "Sky",
            "No sky was rendered: %s" % (sky.get("reason") or "no commissioned residual scale for this "
                                         "laboratory"),
            "Every figure on the Sky tab is a reference illustration, not this specimen.")
    elif not sky.get("_plate_drawn", True):
        add("NO_PLATE", "CAUTION", "Sky",
            "The specimen's own plate could not be drawn: %s" % sky.get("_plate_error", "unknown"),
            "Install matplotlib (chain/requirements.txt) and re-run. Until then the Sky tab shows reference "
            "figures only, and the per-class |z| statistics are readable only in the bundle.")

    # 5. the second opinion
    so = o.get("second_opinion") or {}
    if so.get("available") and so.get("agreement") == "DISAGREE":
        add("SOLVERS_DISAGREE", "CAUTION", "Every cell",
            "The two deconvolvers disagree at class level (L1 = %s)." % so.get("L1_class"),
            "The composition is less certain than a single solver suggests. Look at which class carries the "
            "disagreement before quoting any fraction.")

    # 6. trace detection
    td = o.get("trace_detection") or {}
    tm = td.get("_meta") or {}
    if tm.get("retired"):
        pass
    elif not tm.get("available"):
        add("NO_TRACE_PANEL", "NOTE", "Every cell",
            "Trace-class detection did not run: %s" % (tm.get("reason") or "panel unavailable"),
            "Presence of a trace class was not tested on this specimen either way.")
    elif not tm.get("calibrated_for_this_substrate", True):
        add("TRACE_UNCALIBRATED", "CAUTION", "Every cell",
            "Trace-class detection is calibrated for whole blood; this specimen is declared '%s'."
            % (tm.get("substrate_declared") or "not declared"),
            "The statistic is printed but no call is made. Do not read it as a negative.")
    else:
        for c in ("secretory", "cycling"):
            if (td.get(c) or {}).get("detected"):
                add("TRACE_DETECTED", "NOTE", "Every cell",
                    "Evidence of epithelial-like material (%s statistic above the healthy threshold)." % c,
                    "At this limit the class cannot be named - attribution needs about 5 %. No fraction and "
                    "no A are reportable for it.")

    # 7. the cosmology toolkit: a borrowed method that ran and failed its own condition
    try:
        sys.path.insert(0, ENGINE)
        import cmb_tools as _CT
        for _t in _CT.failures(o):
            add("CMB_TOOL_FAIL", "CAUTION", "Safeguards",
                "%s (%s) failed its own check: %s" % (_t["tool"], _t["id"], _t["evidence"]),
                "A borrowed method that ran and did not hold. See the cosmology-toolkit table "
                "on the Safeguards tab.")
    except Exception:
        pass

    # 8. age
    # (age-clock flag removed 2026-09-26 with the stage)

    F.sort(key=lambda x: (SEVERITY.get(x["severity"], 9), x["code"]))
    F=[f for f in F if not (isinstance(f,dict) and f.get('code') in _FLAGS_NOT_SHOWN)]
    return F


def _lab_bound(fa):
    """The |z| this laboratory's own healthy arrays reach 5 % of the time.

    The commissioned band is pooled over four laboratories. A laboratory whose healthy arrays cross it at a
    rate fa (rather than the nominal 0.05) is wider than the pool by the factor that maps its own tail onto
    the nominal one; under a normal approximation that is 1.96 / z(1 - fa/2), and the laboratory's own 95 %
    bound is 1.96 times it. Approximation, and labelled as one wherever it is printed.
    """
    if fa is None:
        return None
    try:
        fa = float(fa)
    except (TypeError, ValueError):
        return None
    if not (0 < fa < 0.5):
        return None
    try:
        from statistics import NormalDist
        z_at_fa = NormalDist().inv_cdf(1 - fa / 2.0)
    except Exception:
        return None
    if z_at_fa <= 0:
        return None
    return 1.959964 * (1.959964 / z_at_fa)


_FLAGS_NOT_SHOWN={"NO_CELLULAR_AGE","GAUGE_WITHHELD"}   # about quantities this report does not print (2026-09-26)
def tab_redflags(o, R=None):
    """Everything that went wrong, was withheld, or could not be measured - in one place."""
    F = o.get("red_flags") if isinstance(o.get("red_flags"), list) else red_flags(o, R)
    H = ["<h2>Red flags - everything this run refused, withheld or could not measure</h2>",
         "<p>One place, so nothing has to be noticed in the middle of a long tab. Ordered by severity. "
         "<b>STOP</b> means nothing was scored; <b>WITHHELD</b> means a number exists but the reference to "
         "judge it against does not; <b>CAUTION</b> means read the result differently because of something "
         "about this run; <b>NOTE</b> is a statement of fact worth carrying.</p>"]
    if not F:
        H.append("<p class='ok'><b>No red flags.</b> Every gate fired, every component the chain reports "
                 "had its reference, and nothing was withheld.</p>")
    else:
        counts = {}
        for f in F:
            counts[f["severity"]] = counts.get(f["severity"], 0) + 1
        H.append("<p>" + " &nbsp;·&nbsp; ".join("<b>%d %s</b>" % (n, k) for k, n in
                 sorted(counts.items(), key=lambda kv: SEVERITY.get(kv[0], 9))) + "</p>")
        H.append("<table><tr><th>severity</th><th>code</th><th>tab</th><th>what happened</th>"
                 "<th>what to do</th></tr>")
        for f in F:
            H.append("<tr><td class='m'><b>%s</b></td><td class='m'>%s</td><td class='m'>%s</td><td>%s</td>"
                     "<td>%s</td></tr>" % (f["severity"], f["code"], f["where"],
                                           html.escape(str(f["what"])), html.escape(str(f["what_to_do"]))))
        H.append("</table>")
    H.append("<details><summary>The same list as JSON, for a reader that is a program</summary>"
             "<pre><code>" + html.escape(json.dumps({"red_flags": F}, indent=1)) + "</code></pre></details>")
    return "".join(H)


def _propagate_state():
    """What propagate.py last reported, so a report says whether the documents were current when it was built.

    The author's point, 2026-09-25: the record of how a result was produced matters as much as the result.
    propagate.py writes chain/propagate_status.json; if it is absent or stale this says so rather than
    implying the documents were checked.
    """
    p = os.path.join(ENGINE, "propagate_status.json")
    try:
        with open(p, encoding="utf-8") as f:
            st = json.load(f)
    except Exception:
        return ("<h3>Repository state when this report was built</h3><p class='warn'>No propagation status "
                "was found. Nobody has run <span class='m'>chain/propagate.py</span> in this working tree, "
                "so it is not known whether the SOP, the manual, the register and the reviewer manifest were "
                "current for this chain.</p>")
    ok = st.get("pass") is True
    rows = "".join("<tr><td class='m'>%s</td><td>%s</td><td>%s</td></tr>"
                   % ("PASS" if r.get("ok") else "FAIL", html.escape(str(r.get("rule"))),
                      html.escape(str(r.get("detail"))))
                   for r in st.get("rules", []))
    return ("<h3>Repository state when this report was built</h3>"
            "<p>%s <span class='m'>chain/propagate.py</span> last ran <b>%s</b> at commit "
            "<span class='m'>%s</span>: it regenerated every derived document and checked %d rules that "
            "cannot be generated because a human wrote them. %s</p>"
            "<table><tr><th>state</th><th>rule</th><th>detail</th></tr>%s</table>"
            "<p class='m'>A rule that fails means a document has drifted from the tree - the SOP, the Operations Manual, the commissioning register or the reviewer manifest no longer describes the code "
            "that produced this reading.</p>"
            % ("<b class='ok'>Every document was current.</b>" if ok else
               "<b class='warn'>At least one document had drifted.</b>",
               html.escape(str(st.get("when") or "?")), html.escape(str(st.get("commit") or "?")),
               len(st.get("rules", [])),
               "" if ok else "The failures are listed below.", rows))


def _repository_section(o, out_path=None):
    """The loop's return path: what THIS run produced, where each file belongs in the repository, and whether it is
    already there. Read from kit/file_run.py's own plan(), so the page and the command cannot disagree (2026-09-26)."""
    H=["<h3>7. Repository - what this run should add to the tree, and where</h3>",
       "<p class='m'>The repository writes the documents; a run writes back to the repository. Every file below has one destination. "
       "<code>python3 kit/file_run.py --report &lt;this report&gt; [--procedure PROC-XXX-NN] [--commit]</code> files them, rebuilds the run index, "
       "runs the documentation gate, and pushes only if the gate passes. A run that is evidence for a sealed procedure names the procedure and lands in kit/results as well.</p>"]
    try:
        import importlib.util, os as _os
        kp=_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),"kit","file_run.py")
        spec=importlib.util.spec_from_file_location("file_run",kp); fr=importlib.util.module_from_spec(spec); spec.loader.exec_module(fr)
        rep=out_path or (o.get("context") or {}).get("report_path")
        if not rep or not _os.path.exists(rep):
            H.append("<p class='pend'>The report path was not known at build time; run file_run.py on the saved report to see the plan.</p>"); return "".join(H)
        rid,sid,rows=fr.plan(rep, o.get("run_id"), sid=(o.get("context") or {}).get("sample_id"))
        H.append(f"<p>Run <b>{_e(rid)}</b> &middot; specimen <b>{_e(sid)}</b></p><table class='t'><tr><th>file</th><th>belongs at</th><th>why</th><th>state</th></tr>")
        for r in rows:
            col={"committed":"#3fa45b","on disk, not committed":"#d68910","not filed":"#c0392b"}.get(r["state"],"#888")
            H.append(f"<tr><td class='m'>{_e(_os.path.basename(r['source']))}</td><td><code>{_e(r['destination'])}</code></td><td class='m'>{_e(r['why'])}</td><td><b style='color:{col}'>{_e(r['state'])}</b></td></tr>")
        H.append("</table>")
        H.append("<p class='m'>Also updated by the same command: <code>chain/example_runs/RUN_INDEX.csv</code> (rebuilt from the run folders) and, through the gate, every generated document that lists runs.</p>")
    except Exception as e:
        H.append(f"<p class='pend'>Plan not available: {_e(type(e).__name__)} - {_e(str(e)[:120])}</p>")
    return "".join(H)



def _kit_checks(R):
    """The kit's tests and guards, read from the tree at render time with each file's own first docstring line -
    a typed list went stale twice (0e, author 2026-09-26)."""
    import glob as _g
    kit = next((k for k in (os.path.join(ENGINE, "kit"), os.path.join(os.path.dirname(ENGINE), "kit")) if os.path.isdir(k)), os.path.join(ENGINE, "kit")); out = []
    paths = sorted(_g.glob(os.path.join(kit, "test_*.py"))) + sorted(_g.glob(os.path.join(kit, "release_check.py"))) + sorted(_g.glob(os.path.join(kit, "finding_check.py")))
    for path in paths:
        name = os.path.basename(path)
        try:
            src = open(path, encoding="utf-8", errors="replace").read()
            m = re.search(r'"{3}(.*?)"{3}', src, re.S) or re.search(r"'{3}(.*?)'{3}", src, re.S)
            d = (m.group(1).strip().split("\n")[0] if m else "").strip()[:160] or "(no docstring)"
        except OSError:
            d = "(unreadable)"
        if name not in R["files"]:
            R["files"][name] = path
        out.append((name, d))
    return out


def _run_files_check(o, R):
    """0e (author, 2026-09-26): 'linked to the repo every test that runs so it can detect and confirm or flag that the list
    of files is or is no longer accurate.' Every runtime file THIS run read is listed from the bundle's version block and
    checked against the tree at render time: present, tracked by git, and hashing to what the run recorded. Anything else
    prints as a flag, never silently."""
    v = (o.get("versions") or {}); ins = v.get("inputs") or {}
    if not ins:
        return "<h3>Files this run read</h3><p class='m'>The bundle carries no version block, so this check cannot run.</p>"
    try:
        tracked = set(subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True, text=True).stdout.split("\n"))
    except Exception:
        tracked = set()
    rows = []; bad = 0
    for rel in sorted(ins):
        rec = ins[rel] or {}; want = rec.get("sha256_12", "")
        cands = [os.path.join(ENGINE, rel), os.path.join(BIO, rel), os.path.join(REPO, rel)]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            st = "<b>MISSING from the tree</b>"; bad += 1
        else:
            have = _sha(path)[:12]; relrepo = os.path.relpath(path, REPO)
            if want and have != want:
                st = "<b>CHANGED since this run</b> (tree " + have + ")"; bad += 1
            elif relrepo not in tracked and (relrepo + ".xz") in tracked:
                st = "decompressed locally from the tracked .xz"          # the atlas ships compressed; the CSV is derived, not a drift
            elif relrepo not in tracked:
                st = "<b>present but NOT TRACKED</b>"; bad += 1
            else:
                st = "matches the tree"
        rows.append(f"<tr><td class='m'>{_e(rel)}</td><td class='m'>{_e(want)}</td><td>{st}</td></tr>")
    verdict = ("all " + str(len(ins)) + " match") if not bad else (str(bad) + " FLAGGED")
    head = (f"<h3>Files this run read - checked against the repository at render time</h3>"
            f"<p class='m'>{len(ins)} runtime files were read by this run and hashed at run time. Each is checked here against the "
            f"working tree and git: <b>{verdict}</b>. A flag means the report and the tree disagree - the file changed, moved, or was "
            f"never committed - and the reading should not be quoted until that is resolved.</p>")
    return head + "<table class='t'><tr><th>file</th><th>SHA-256 at run</th><th>tree now</th></tr>" + "".join(rows) + "</table>"


def tab_run(o, R):
    """Run it yourself. Every file named here is linked at this commit, and every command is one that
    actually works - the earlier version advertised an IDAT entry point that did not exist (fixed 2026-09-22
    by writing run_sample.py)."""
    def L(name, label=None):
        return _link(R, name, label)
    H=["<h2>Run it yourself</h2>"] + _provenance_block(o) + [
       "<p>The chain is open. Clone the repository, verify it against the eleven commissioning arrays, then run your own sample. Every file below "
       "is linked at the exact commit this report was built from, with its SHA-256, so you can check you are running what this page describes.</p>",
       "<h3>1. Clone and prepare</h3>",
       "<pre>git clone https://github.com/hmahaffeyges/IAM-Validation.git\n"
       "cd IAM-Validation/Biological_Physics\n"
       "# python 3.11 with numpy, pandas, scipy; add methylprep only if you will calibrate raw IDATs\n"
       "# the atlas ships compressed - decompress it once (605 MB). There is no system xz dependency:\n"
       "python3 -c \"import lzma,shutil; shutil.copyfileobj(lzma.open('MethylPhys/atlas/IAMAtlasREBUILD.csv.xz','rb'), open('MethylPhys/atlas/IAMAtlasREBUILD.csv','wb'))\"</pre>",
       "<h3>2. Verify the chain before trusting it on your data</h3>",
       "<pre>cd MethylPhys/kit\npython3 release_check.py            # every guard, one command, writes results/release_check.json</pre>",
       "<p>That is the same command whose output the <b>Safeguards</b> tab prints. It exits non-zero if any guard fails; a guard that cannot run for "
       "want of data is reported as skipped with what it needs, never as a pass. The individual guards, if you want them one at a time:</p>",
       "<table class='t'><tr><th>file</th><th>what it checks</th></tr>"]
    for n,d in _kit_checks(R): H.append(f"<tr><td>{L(n)}</td><td>{_e(d)}</td></tr>")
    H.append("</table>")
    H.append("<h3>3. Run your own sample</h3>"
       "<pre># an Illumina IDAT pair (Stage 1 needs methylprep):\npython3 MethylPhys/chain/MethylPhys_Interface/run_sample.py \\\n"
       "    --grn SAMPLE_Grn.idat.gz --red SAMPLE_Red.idat.gz \\\n    --age 58 --lab MYLAB --specimen \"whole blood\" --out report.html\n\n"
       "# or a beta table you calibrated yourself - a CSV of cpg_id,beta:\npython3 MethylPhys/chain/MethylPhys_Interface/run_sample.py \\\n"
       "    --betas mysample.csv --age 58 --lab MYLAB --out report.html</pre>")
    if "run_sample.py" in R["files"]: H.append(f"<p>The runner: {L('run_sample.py')} - it calls Stage 1, then the conductor, then this report builder. "
       f"The builder can also be driven directly from a saved bundle: {L('build_methylphys.py')} <code>--bundle out.pkl</code>.</p>")
    H.append("<h3>4. What the chain needs from you, and what it will refuse</h3>"
       "<table class='t'><tr><th>input</th><th>why</th><th>if you omit it</th></tr>"
       "<tr><td>the IDAT pair, or a calibrated beta vector</td><td>the measurement</td><td>nothing runs</td></tr>"
       "<tr><td>declared age</td><td>carried as context on the record; not an operand of any reading</td><td>-</td></tr>"
       "<tr><td>specimen</td><td>presence floors are specimen-specific</td><td>the run is refused - the chain will not guess</td></tr>"
       "<tr><td>laboratory identity</td><td>the pipeline map, the sky's per-address scale and the foreign-cell detection line are the laboratory's own</td><td>every present cell is still read; the sky and foreign-cell detection print 'not commissioned for this laboratory' with the reason</td></tr>"
       "<tr><td>pipeline name</td><td>a beta from a different normalisation sits on a different scale (LESSON-SCALE-01)</td>"
       "<td>the conductor refuses an unmapped reading rather than placing it</td></tr></table>")
    H.append("<h3>5. Commissioning your own laboratory</h3>"
       "<p>Every present cell's A is read on any laboratory's arrays with no panel at all: healthy is A = 1.00 and the floors are frozen. What a laboratory "
       "commissions once is the <b>instrument</b> around that reading - the pipeline map onto the atlas scale, the sky's per-address scale, and the "
       "foreign-cell detector's line (36 or more healthy whole-blood arrays through Stage 1, then <code>kit/commission_detection_lab.py</code>). Until then "
       "those two sections print 'not commissioned for this laboratory' and say why. See " + L("RUNBOOK.md") + " and " + L("CHAIN_COMMISSIONING.md") + ".</p>")
    H.append(_run_files_check(o, R))   # 0e (author, 2026-09-26): the file list is read from the run and checked against the tree, never typed
    H.append("<h3>6. Validation mode</h3><p>For a validation run, every sample is read absolutely as above and the report then adds the distribution of "
       "readings by arm beside the sealed pre-registration bars. The protocol - seal before you run, register the finding, close it in code, teach "
       "every door, rebuild, read, push with copies - is in the RUNBOOK, and the gate that enforces it is <code>finding_check.py</code>.</p>")
    H.append(f"<p class='m'>Repository commit for this page: <code>{_e(R['sha'])}</code>. Every link above resolves at that commit, so a file that has "
       f"changed since will not silently substitute itself.</p>")
    H.append(deepdive(R,"running the chain"))
    H.append(_repository_section(o, (o.get("context") or {}).get("report_path")))   # the loop's return path, 2026-09-26
    H.append(_propagate_state())   # how we got here, 2026-09-25
    return guard("".join(H),"Run")


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
details.legend{background:var(--pn);border:1px solid var(--ln);border-radius:6px;padding:8px 12px;margin:4px 0 14px}details.legend summary{cursor:pointer;color:#cfd8ff}
details.stage{background:var(--pn);border:1px solid var(--ln);border-radius:6px;padding:8px 12px;margin:5px 0}details.stage summary{cursor:pointer}details.stage[open]{border-color:var(--ac)}span.stg{display:inline-block;min-width:34px;font:bold 11px monospace;color:#000;background:var(--ac);border-radius:4px;padding:1px 5px;text-align:center}
div.warn{background:#241a14;border-left:3px solid #d68910;padding:12px 16px;margin:16px 0;font-size:13px}
div.dd{background:#0f1420;border-left:3px solid #6a8;padding:10px 16px;margin:20px 0 4px;font-size:13px}
footer{padding:14px 28px;border-top:1px solid var(--ln);color:var(--mu);font-size:12px}
/* the audience toggle (2026-09-22): it previously set a class nothing listened to. Researcher-only
   material - provenance hashes, the optional formula folds, and the record/engineering tabs - is hidden
   in clinician view; nothing is hidden from the researcher. */
body:not(.researcher) .resr{display:none !important}
body:not(.researcher) nav button.resr{display:none !important}
body.researcher .clin-only-note{display:none}
@media print{body{background:#fff;color:#000}header,nav,footer,.aud{display:none}section.tab{display:none}section.tab.print{display:block;page-break-after:always}
  body.researcher section.tab.printr{display:block;page-break-after:always}
  body.researcher .resr{display:block !important}
  details{display:block !important}details>*{display:revert}details:not([open])>*:not(summary){display:block}
  table,figure,div.gauge,details.stage{page-break-inside:avoid}
  h2,h3{page-break-after:avoid}
  a[href^='http']:after{content:' [' attr(href) ']';font-size:8pt;color:#555;word-break:break-all}.gauge{border:1px solid #999;background:#fff}h3{color:#000}table.t th{color:#333}.m{color:#444}svg text{fill:#000}}
"""
JS="""
function tab(id){document.querySelectorAll('section.tab').forEach(s=>s.classList.toggle('on',s.id===id));document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));location.hash=id}
function aud(a){document.body.classList.toggle('researcher',a==='researcher');document.querySelectorAll('.aud button').forEach(b=>b.classList.toggle('on',b.dataset.a===a));localStorage.setItem('mp_aud',a)}
window.addEventListener('DOMContentLoaded',()=>{aud(localStorage.getItem('mp_aud')||'researcher');tab((location.hash||'#reading').slice(1))});
"""
# Measured 2026-09-25 by diffing every tab between a healthy blood donor and an adenoma tissue specimen:
# these nine were byte-identical, i.e. they are reference material and carry nothing about the specimen in
# front of you. Seven tabs do carry it: reading, cells, departure, sky, integrity, safeguards, run.
REFERENCE_TABS = ("howto", "story", "physics", "findings", "trouble", "reference", "coverage",
                  "record")
REFERENCE_BANNER = ("<p class='m' style='border-left:3px solid #bbb;padding-left:8px'>Reference material - "
                    "this tab is the same in every report and carries no measurement of this specimen.</p>")

TABS=[  # id, label, in the CLINICIAN print set, audience ("c" = both, "r" = researcher only)
 ("reading","Reading",True,"c"),("howto","How to read",True,"c"),("cells","Every cell",True,"c"),("sky","Sky",True,"c"),("physics","Physics",False,"c"),("story","Story",False,"c"),
 ("reference","Instrument",False,"r"),("coverage","Coverage",False,"r"),("flags","Red flags",True,"c"),("safeguards","Safeguards",False,"r"),("trouble","Troubleshooting",False,"r"),
 ("integrity","Integrity",False,"r"),("chain","Chain",False,"r"),("files","Files",False,"r"),("findings","Findings",False,"r"),
 ("record","Record",False,"r"),("run","Run",False,"r")]

def refusals_from(o):
    r=[]
    # the pooled class gauge is an internal gate and is not printed, so its refusals are not listed here (2026-09-26)
    if not o["patient_sky"].get("available"): r.append("sky: no commissioned residual scale for this laboratory -> not rendered")
    fd=o.get("foreign_detection") or {}
    if fd.get("status") and fd.get("status")!="OK": r.append(f"foreign-cell detection: {fd.get('status')} - {fd.get('reason') or ''}".rstrip(" -"))
    return r

ISHA=None
def build(o, out_html, sample_id="sample", percell_ref=None, percell_status="in build - 80 healthy arrays per laboratory through Stage 1 (started 2026-09-22)"):
    global ISHA; ISHA=_sha(os.path.abspath(__file__))[:12]
    R=load_runtime(); wd=os.path.dirname(os.path.abspath(out_html)) or "."; os.makedirs(wd,exist_ok=True)
    sec={"reading":tab_reading(o,R,sample_id),"cells":tab_cells(o,R,percell_ref if percell_ref is not None else R.get("percell")),"sky":tab_sky(o,R,sample_id,wd),
         "reference":tab_reference(R,percell_status),"integrity":tab_integrity(o,R,refusals_from(o)),"chain":tab_chain(R),"files":tab_inventory(R),"findings":tab_findings(R),"physics":tab_physics(R),"howto":tab_howto(R),"coverage":tab_coverage(R),"safeguards":tab_safeguards(o,R),"flags":tab_redflags(o,R),"trouble":tab_troubleshooting(o,R),"story":tab_story(R),"record":tab_record(R),"run":tab_run(o,R)}
    for _rt in REFERENCE_TABS:
        if _rt in sec:
            sec[_rt] = REFERENCE_BANNER + sec[_rt]
    imm=o["classes"].get("immune",{}); head=("composition verified: whole blood" if imm.get("composition_verified") else ("composition not verified as whole blood - cell tiers withheld" if imm.get("composition_verified") is False else "composition check not run"))
    nav="".join(f"<button class='{'resr' if a=='r' else ''}' data-t='{i}' onclick=\"tab('{i}')\">{n}</button>" for i,n,_,a in TABS)
    _rprint={"reading","howto","cells","sky","reference","safeguards","trouble","integrity","chain","files","coverage"}
    body="".join(f"<section class='tab{' print' if p else ''}{' printr' if i in _rprint else ''}"
                 f"{' resr' if a=='r' else ''}' id='{i}'>{sec[i]}</section>" for i,n,p,a in TABS)
    page=f"""<!doctype html><html><head><meta charset='utf-8'><title>MethylPhys CPG - {_e(sample_id)}</title><style>{CSS}</style><script>{JS}</script></head><body data-interface-sha='{ISHA}'>
<header><h1>MethylPhys <span style='color:var(--ac)'>CPG</span> <small>Physics of Methylation: Landauer Metrology · Cellular Performance Gauge · the physics of methylation, read against a fixed zero</small></h1>
<div><div class='aud'>view: <button data-a='clinician' onclick="aud('clinician')">Clinician</button><button data-a='researcher' onclick="aud('researcher')">Researcher</button> <button onclick='window.print()'>Print report</button></div>
<div class='m' style='text-align:right;margin-top:6px'>{_e(sample_id)} · {head} · generated {time.strftime('%Y-%m-%d %H:%M')} · repo {_e(R['sha'])}</div></div></header>
<nav>{nav}</nav><main>{body}</main>
<footer>MethylPhys CPG · report generator in build (row 9, unsealed) · every number on the measurement tabs comes from cpg_conductor.run_full and the runtime files listed under Integrity · this is a measurement record, not a clinical interpretation · IAMPerformance · <a href='{GH}' style='color:var(--ac)'>repository</a></footer></body></html>"""
    open(out_html,"w",encoding="utf-8").write(page); return {"out":out_html,"bytes":len(page),"refusals":refusals_from(o)}

if __name__=="__main__":
    import argparse, pickle; ap=argparse.ArgumentParser(); ap.add_argument("--bundle",help="pickle or json of run_full output"); ap.add_argument("--out",default="methylphys_report.html"); ap.add_argument("--id",default="sample"); a=ap.parse_args()
    o=pickle.load(open(a.bundle,"rb")) if a.bundle.endswith(".pkl") else json.load(open(a.bundle)); print(build(o,a.out,a.id))
