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
    R["band"]=_j(_find("identity_band_v3.json")); m=R["band"]["_meta"]; coh=m["cohorts"]; R["cohorts"]=ast.literal_eval(coh) if isinstance(coh,str) else coh
    R["maps"]=_j(_find("beta_scale_maps_v1.json")); R["age"]=_j(_find("reference_age_curve_v1.json")); R["tiers"]=_j(_find("tier_breakpoints.json"))
    R["floors"]=_j(_find("presence_floors_v1.json")); R["ident"]=_j(_find("iamatlas_gauge_identity_loci_v1_0.json"))
    R["c2c"]=_j(_find("IAMAtlasREBUILD_celltype_to_class.json")); R["panels"]=_j(_find("directional_panels_v1_0.json"))
    R["files"]={n:_find(n, allow_retired=True) for n in ["identity_band_v3.json","beta_scale_maps_v1.json","reference_age_curve_v1.json","tier_breakpoints.json","presence_floors_v1.json",
               "iamatlas_gauge_identity_loci_v1_0.json","iamatlas_celltype_markers_v0_2.json","IAMAtlasREBUILD_celltype_to_class.json","directional_panels_v1_0.json",
               "iamatlas_cpg_to_healpix_nside128.npz","cpg_conductor.py","cpg_gauge_engine.py","lab_zero.py","cpg_tiers.py","stage_4_6_patient_cmb.py","walther_iam_deconvolver.py",
               "bidirectional_decomposition.py","iamatlas_a_scoring.py","stage_1_idat_calibration.py","IAMAtlasREBUILD_provenance.json",
               "IAMAtlasREBUILD.csv.xz","IAMAtlasREBUILD_celltype_to_class.json","iamatlas_cpg_to_healpix_nside128.npy","iamatlas_cpg_to_healpix_nside128.provenance.json",
               "RUNBOOK.md","CHAIN_COMMISSIONING.md","HANDOFF.md","nilc_celltype_deconvolver.py","run_sample.py","release_check.py",
               "test_gauge_switch.py","test_tiers.py","test_patient_sky.py","test_lab_zero.py","PROC_ANCHOR_01.py","PROC_FORMULA_01.py",
               "PROC_DECON_01.py","PROC_SEP_03.py","PROC_BIDIR_01.py","finding_check.py","percell_reference_v0.json","reference_age_curve_v1.json","lineage_splitter.py","README_CPG_Plates.md","README_HEALPix_Mapping.md","CPG_Gauge_Cell.png","CPG_Gauge_Cosmic.png","healthy_sky_vs_cmb.png"] if _exists(n)}
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
    try: R["percell"]=_j(_find("percell_reference_v0.json"))
    except FileNotFoundError: R["percell"]=None
    R["sha"]=_git_head()
    sys.path.insert(0,ENGINE); import cpg_tiers as T; sch=T.scheme(); R["tier_bands"]=sch["bands"]; R["tier_version"]=sch.get("version")   # ONE tier definition (PROC-TIER-01)
    return R
def _exists(n, allow_retired=True):
    try: _find(n, allow_retired=allow_retired); return True
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
    return no_changelog(section_html, tab)

def deepdive(R, topic=""):
    """The manual and the paper, linked wherever a reader may want to go further (author: they can deep dive all they want from the link)."""
    pdf="Biological_Physics/MethylPhys/manual/MethylPhys_CPG_Operations_Manual.pdf"
    tex="Biological_Physics/MethylPhys/papers/Landauer_Metrology_of_the_Methylome.tex"
    t=f" on {topic}" if topic else ""
    return (f"<div class='dd'><b>Go deeper{t}.</b> Everything on this tab is treated at full length in the engineering manual - <a href='{GH}/blob/{R['sha']}/{pdf}' target='_blank'>GAPE Issue 003</a> "
            f"(~300 pages: the physics section, every sealed procedure with its outcome as found, the complete validation history, the reconciliation tables, the falsification register and the future-goals list) - and in the short methods paper, "
            f"<a href='{GH}/blob/{R['sha']}/{tex}' target='_blank'>Landauer Metrology of the Methylome</a>. Both live in the same repository as the code that produced this page, at the same commit.</div>")

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

COLS_LEGEND=("<details class='legend' open><summary><b>What each column means</b> - three of them are easy to confuse, so they are defined here</summary><table class='t'>"
 "<tr><th>column</th><th>what it is</th><th>what it is NOT</th></tr>"
 "<tr><td><b>placed</b></td><td>the composition solver found enough evidence to give this sample some fraction of this cell type</td>"
 "<td>a blank is not 'absent' - a lineage's whole fraction often lands on one representative entry, so its aliases read 0 %</td></tr>"
 "<tr><td><b>fraction</b></td><td>how much of the sample's DNA the solver attributes to this cell type</td><td>not a cell count</td></tr>"
 "<tr><td><b>A (marker surface)</b></td><td>this cell's reading: the mean of the per-CpG entropies over its own marker addresses, divided by its class's frozen floor H_min</td>"
 "<td>not the class gauge - that is on the Reading tab, computed on identity loci, and it is the only surface carrying tiers</td></tr>"
 "<tr><td><b>95 % interval on the reading</b></td><td><b>the error bar on THIS number</b>, from resampling this cell's own marker CpGs 500 times. Narrow means the markers agree with each other</td>"
 "<td><b>not</b> the healthy range and <b>not</b> the atlas posterior spread. The June reports printed the atlas posterior SD of a class <i>mean</i> as a '95 % CI' - the uncertainty of an average used as a population spread - which is why healthy cells there appeared to sit 10 sigma out. Three different quantities; this report labels each one</td></tr>"
 "<tr><td><b>healthy range (own markers)</b></td><td>the 10th-90th percentile of this same reading across 40 healthy arrays from <b>this patient's own laboratory</b> (or the four pooled if that laboratory has no panel)</td>"
 "<td>not a decision threshold - no tier is assigned on this surface</td></tr>"
 "<tr><td><b>markers found</b></td><td>how many of that cell's panel CpGs were present and finite on this array, out of the panel total</td>"
 "<td><b>not</b> a confidence interval - it is panel coverage. A low count is what widens the interval to its left</td></tr>"
 "<tr><td><b>position vs healthy</b></td><td>the picture of the two middle columns: green band = that cell's healthy range, dot = this sample, drawn over +/-2 band-widths. Green inside, amber within half a band-width outside, red beyond</td>"
 "<td>not a tier and not a severity</td></tr></table></details>")

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
donors). The corrected value A'' is placed in the healthy band (the middle 80% of healthy donors, four laboratories) and given its tier. <b>BREACH at 1.10</b> means the class has lost its floor - the level that defines it. It is a reading on this ruler, like a temperature of 104 F, and not a comparison to any prior cohort. It is not the ceiling: the ceiling is saturation, 1/H_min, and it is a different limit (see How to read).</p></div>"""

def _trace_one_liner(o):
    """One line on the Reading tab. A reader who opens one tab should see what Stage 2c found."""
    td = o.get("trace_detection") or {}
    m = td.get("_meta") or {}
    if not m.get("available"):
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
    H.append(_trace_one_liner(o))
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
    H.append(GAUGE_EXPLAINER); H.append(deepdive(R,"the gauge"))
    H.append("<h3>Is this composition plausible? Measured across 40 healthy donors of this laboratory</h3>"
      "<p>A haematologist's first instinct on seeing one number - <i>62.6 per cent neutrophils?</i> - is to ask whether the solver just says that "
      "every time. It does not, and the way to show it is the distribution across donors rather than one array. Every one of the 40 healthy "
      "Uppsala panel arrays was put through the composition step; this is what came out, beside the textbook differential white-cell count a "
      "clinical laboratory would report on the same tube:</p>"
      "<table class='t'><tr><th>atlas entry</th><th>donors placing it</th><th>median % when placed</th><th>range across donors</th>"
      "<th>textbook differential (% of white cells)</th></tr>"
      "<tr><td>Neutrophils</td><td class='n'>40 / 40</td><td class='n'>47.0</td><td class='n'>31.9 - 68.7</td><td class='n'>40 - 70</td></tr>"
      "<tr><td>CD4 T cells</td><td class='n'>38 / 40</td><td class='n'>18.7</td><td class='n'>1.2 - 34.3</td><td class='n'rowspan='3'>lymphocytes 20 - 45 in total</td></tr>"
      "<tr><td>CD8 T cells</td><td class='n'>32 / 40</td><td class='n'>6.3</td><td class='n'>0.6 - 30.2</td></tr>"
      "<tr><td>CD19 B cells</td><td class='n'>34 / 40</td><td class='n'>2.3</td><td class='n'>0.4 - 7.2</td></tr>"
      "<tr><td>CD56 NK cells</td><td class='n'>40 / 40</td><td class='n'>13.4</td><td class='n'>4.3 - 32.5</td><td class='n'>2 - 10</td></tr>"
      "<tr><td>CD14 monocytes</td><td class='n'>40 / 40</td><td class='n'>8.4</td><td class='n'>3.5 - 14.3</td><td class='n'>2 - 10</td></tr>"
      "<tr><td>GMP (granulocyte-monocyte progenitor)</td><td class='n'>25 / 40</td><td class='n'>4.6</td><td class='n'>0.1 - 16.2</td><td class='n'>not counted clinically</td></tr></table>"
      "<p><b>What that shows.</b> Neutrophils dominate every healthy donor, which is correct - they are the most abundant white cell in blood - and "
      "the solver's median of 47 per cent sits inside the textbook range, with donor-to-donor variation of 32 to 69 per cent. <b>The other immune "
      "cells are all there:</b> T cells in 38 of 40 donors, NK and monocytes in all 40, B cells in 34. A single array showing 62.6 per cent "
      "neutrophils is a high-normal donor, not a solver that only knows one answer. Summing this report's own placed cells reproduces the immune "
      "class fraction the gauge is read on, which is the internal consistency check that matters.</p>"
      "<p><b>Two honest departures from the clinical count, both worth a reader's attention.</b> First, <b>NK cells read high</b> - a median of 13 "
      "per cent against a textbook 2 to 10 - and the most likely reason is the one PROC-SEP-03 measured directly: the atlas cannot fully separate "
      "the lymphoid entries, so an NK panel absorbs signal that belongs to T cells. That is a known limit of the reference, not a finding about the "
      "donor, and it is why per-cell readings inside one lineage are not scored against each other. Second, a minority of donors place a trace of "
      "something implausible - gastric or glial entries at under 2 per cent in 1 to 15 of 40 donors. Those are the conservative solver's "
      "false placements at the edge of its evidence threshold; they are reported rather than hidden, and their size is the reason they do not "
      "change a class reading. Eosinophils and basophils, which a clinical count reports at a few per cent, have no atlas entry at all - so they "
      "are not missing from this sample, they are missing from the reference, which the Coverage tab states.</p>"
      "<p class='m'>Measured 2026-09-22 on the 40 build-panel arrays of GSE87571 (raw IDAT through Stage 1, this laboratory only); median 7 atlas "
      "entries placed per donor, range 5 to 9 of 115. The textbook differential ranges are the standard clinical reference intervals for a white-cell "
      "differential and are shown for orientation, not as a validation target - a methylation-based composition and a microscope count are "
      "different measurements of the same tube.</p>")

    return guard("".join(H),"Reading")

def _trace_block(o):
    """Stage 2c - trace-class detection. Presence only: no fraction, no A, no tier below the attribution
    limit, because a trace class cannot be scored in this substrate at any fraction a blood draw presents."""
    td = o.get("trace_detection") or {}
    m = td.get("_meta") or {}
    if not m.get("available"):
        return ["<h3>Trace-class detection</h3>",
                "<p class='m'>Not run for this specimen: %s</p>" % (m.get("reason") or "no panel")]
    H = ["<h3>Trace-class detection - is there any epithelial-like material here at all?</h3>",
         "<p>The composition solve above cannot answer this. It is a non-negative fit, and a non-negativity "
         "constraint pins a component that small at exactly zero - measured on 48 mixtures of real healthy "
         "blood, a spike below 5 % returned 0.00 on two of three donors. This panel asks the question the "
         "other way round: fit the specimen <i>without</i> the class, then test whether the class's own "
         "profile explains what is left over, weighting each address by how well the atlas pinned it down. "
         "A healthy donor sits at t &asymp; -16; the threshold is the 95th percentile of 38 healthy donors "
         "who played no part in choosing the method.</p>"]
    if not m.get("calibrated_for_this_substrate", True):
        H.append("<p class='warn'><b>Uncalibrated on this substrate.</b> The thresholds are a whole-blood "
                 "measurement (declared here: %s). The statistic is printed for reference only and no "
                 "call is made here.</p>" % (m.get("substrate_declared") or "not declared"))
    H.append("<table><tr><th>class</th><th>t</th><th>threshold</th><th>healthy median</th>"
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


def tab_cells(o, R, percell_ref=None):
    sys.path.insert(0,ENGINE); import cpg_tiers as T
    lab=(o.get('patient_sky') or {}).get('lab') or (o.get('cfg') or {}).get('lab')
    cells=o.get("cells_all") or {}; comp_cells={r["cell"]:r for r in o["composition"]["celltype"]}
    H=[COLS_LEGEND,"<h2>Every cell - all 115 atlas cell types, on the gauge</h2>"] + _trace_block(o) + [
       "<p><b>Per-cell A</b> = the mean over that cell's ~100 discriminative marker CpGs of H(beta at that CpG), divided by H_min for its class - "
       "the mean of the per-CpG entropies, <i>not</i> the entropy of the mean beta (the scoring module refuses the second form by assertion: marker CpGs "
       "are chosen to be extreme and opposite, so their mean beta lands near a coin flip and would read maximal disorder on a healthy sample). "
       "This is the same ratio the gauge is drawn in - the cellular gauge figure's own axis reads <i>mean of H(beta)/H_min(class) over panel CpGs</i>.</p>",
       "<p><b>Why there are two columns and not one.</b> Raw per-cell A has no common zero: measured across the four laboratories' healthy panels, the healthy "
       "median of this ratio runs from about 0.55 to 1.15 depending on the atlas entry, because each entry's marker panel has its own natural entropy and in bulk "
       "blood a rare cell's markers mostly carry other cells' DNA. So the landmarks cannot be read off the raw number. They can be read off it once each entry is "
       "put on its own measured zero: <b>cell-zeroed A = A - (this entry's healthy median in this laboratory - 1.000)</b>, which makes a healthy reading of that "
       "entry sit at 1.000 and then places it on exactly the gauge the class uses - NORMAL, MARGINAL, the Warburg line at 1.07, DETECTABLE, BREACH at 1.10, and "
       "the ceiling at 1/H_min. Same physics, same landmarks, one zero per entry instead of one per class.</p>",
       "<p><b>What this column is and is not.</b> The zero is <i>measured</i>, from 40 healthy arrays of that laboratory - not assumed. But the placement of tier "
       "words on this surface is <b>exploratory and not commissioned</b>: the tier breakpoints were commissioned against the class gauge, and the held-out check on "
       "the per-cell reference currently sits at 75 % against a nominal 80 %. What would commission it: the alias merge (one lineage, one entry), the held-out "
       "coverage at bar, and a sealed test on a cohort where the per-cell placement is checked against a known answer. Until then read the tier word on a cell as "
       "<i>where the physics puts it</i>, and the class gauge on the Reading tab as <i>what the chain reports</i>.</p>",
       "<p><b>And no tier at all on a cell below its presence floor</b> - PROC-CEIL-01 measured why: on healthy whole blood the classes that are absent read at or "
       "past their ceiling, purely because their identity addresses carry blood's values, which average near a coin flip. A high reading on an absent cell is an "
       "artefact of absence, never a severity.</p>",
       "<p><b>Measured, before trusting the column.</b> The gauge's NORMAL band is 0.090 wide. Across all 460 entry-by-laboratory "
       "combinations the healthy 10th-90th spread of a cell's own reading has median 0.059 (IQR 0.047-0.080) - tighter than the band, which is what makes "
       "placement meaningful. But <b>20 % of entries are wider than the band</b>, and they are a recognisable set: the T-cell and NK entries run 0.21-0.23 "
       "(CD56_NK-cells 0.225, CD8Tmem 0.215, NK 0.213, CD4T 0.210) against the tightest at 0.038-0.041 (Gran, nRBC, adipocyte, Eosinophils_reinius). For an "
       "entry whose healthy spread exceeds the NORMAL band, a perfectly healthy person can read SUPPRESSED or ELEVATED on the cell-zeroed gauge - so this "
       "report prints the number and <b>withholds the tier word</b> for those entries, naming the spread instead. Fixing it properly means a per-entry band, "
       "which is the alias merge plus a wider panel, and is an open item rather than a wording choice.</p>",
       "<p class='m'>The class gauge on the Reading tab is <b>not</b> these numbers pooled. It is a separate measurement on that class's identity loci - the addresses "
       "where every healthy cell of the class sits at one level - and it is the surface the floor was calibrated against and the reference layers were fitted on. "
       "The two answer different questions: the class gauge asks whether this architecture is holding its pattern; the per-cell reading asks which cell type, and "
       "therefore which organ, is where the departure sits.</p>"]
    H.append("<h3>Two different limits on a per-cell claim, and they are not the same limit</h3>"
      "<p><b>1. The atlas cannot separate some entries from each other.</b> That was measured in June and is in the chain's own files: "
      "<code>iamatlas_collinearity_groups_v0_1.json</code> clusters the 115 entries at centred-cosine 0.95 in departure-from-consensus space and gets "
      "<b>94 groups, 10 of them multi-member</b>. Its own note is the right statement of the limit: cells within a group are methylation-collinear and "
      "<i>not individually identifiable by deconvolution</i>. The groups are the ones a haematologist would predict - the six gastric entries together; "
      "CD4 with CD8 T cells (in three different naming conventions); HSC with L-MPP and MPP; CMP with MEP; dendritic with macrophage; eosinophil with "
      "monocyte and neutrophil. Where a row belongs to a multi-member group, <b>the honest unit of the claim is the group</b>, and the column says so.</p>"
      "<p><b>2. Some marker panels are not exclusive to their entry.</b> A separate and independent problem, measured 2026-09-22. "
      "Cortical_neurons and stem_pluri share 91 of ~100 markers yet sit in <i>different</i> collinearity groups - the atlas can tell them apart; their "
      "panels cannot. So a row can be separable in the atlas and still carry a number that reads a shared block.</p>"
      "<p class='m'>A per-cell claim therefore needs both: a group that is a single member (or a claim made at group level), and a panel exclusive "
      "enough to be about that entry. Neither is a property of your sample; both are properties of the reference, and both are printed.</p>")
    H.append("<div class='warn'><b>Read the exclusivity column before believing any single row.</b> The marker panels were selected one-vs-rest against the <i>mean</i> of the other cell types - a criterion that scores a globally extreme CpG highly for every cell type in which it is extreme. Measured 2026-09-22: <b>33.8 % of the 6,738 marker CpGs belong to more than one entry's panel</b> (one serves 11 of them), and the median entry's panel is only <b>37 % exclusive</b> to it. At the extreme, <b>macrophage's panel is 0 % exclusive</b> - every marker it has also belongs to another entry - and Cortical_neurons, dendritic, erythroblast, small_intestine and tcell are all near 1 %. Twenty-six of the 115 entries sit in pairs sharing at least half their markers, and 28 of those pairs span <i>different architecture classes</i>: Cortical_neurons and stem_pluri share 91 markers, small_intestine and tcell share 82. Where a panel is mostly shared, the number below reads a shared block rather than that cell type, so <b>the individual direction claim is withheld for the 36 entries under 25 % exclusivity</b> and the number is printed with its exclusivity beside it. This is a property of the reference, not of any sample. The runtime marker file is deliberately unchanged - the sealed foundation-cohort anchors reproduce on it, so repairing the selection criterion requires a re-seal, and that is on the Roadmap.</div>")
    by={}; 
    for cell,r in cells.items(): by.setdefault(r.get("class","?"),[]).append((cell,r))
    for c in CLASSES:
        rows=sorted(by.get(c,[]), key=lambda kv:-(kv[1].get("A") or 0))
        hm=R["ident"].get(c,{}).get("H_min")
        if not rows: continue
        H.append(f"<h3>{CLASS_LABEL[c]} <span class='m'>({len(rows)} cells; H_min {R['ident'].get(c,{}).get('H_min','-')})</span></h3><table class='t cells'><tr><th>cell type</th><th>placed</th><th>fraction</th><th>A (marker surface)</th><th>95 % interval on the reading</th><th>healthy range (own markers)</th><th>markers found</th><th>panel exclusive to this entry</th><th>lineage group</th><th>on the gauge (cell-zeroed)</th><th>position vs healthy</th></tr>")
        for cell,r in rows:
            A=r.get("A"); fr=r.get("fraction") or 0; e=((percell_ref or {}).get("entries") or {}).get(cell)
            ref=None; src=""
            if e:
                lr=(e.get("labs") or {}).get(lab)
                if lr: ref={"p10":lr["A_p10"],"p50":lr.get("A_p50"),"p90":lr["A_p90"],"n":lr["n_panel"]}; src="this lab"
                elif e.get("pooled"): ref={"p10":e["pooled"].get("A_p10"),"p50":e["pooled"].get("A_p50"),"p90":e["pooled"].get("A_p90"),"n":e["pooled"]["n"]}; src="4 labs pooled"
            if ref and ref.get("p10") is not None and A is not None:
                rng=f"{ref['p10']:.3f}-{ref['p90']:.3f} <span class='m'>(n={ref['n']}, {src})</span>"
                dirn=("<b>above</b>" if A>ref["p90"] else "<b>below</b>" if A<ref["p10"] else "within")
            else: rng="<span class='pend'>no reference for this entry</span>"; dirn="-"
            ci=r.get("reading_ci") or {}
            cis=("%.3f - %.3f"%(ci["ci_lo"],ci["ci_hi"])) if ci.get("ci_lo") is not None else "<span class='m'>too few markers</span>"
            mf=("%d of %d"%(ci["n_markers_found"],ci["n_markers_panel"])) if ci.get("n_markers_found") else ("%.2f"%(r.get('coverage') or 0))
            # PANEL EXCLUSIVITY GUARD (2026-09-22). The marker panels were selected one-vs-rest against the MEAN of
            # the other cell types, a criterion that scores a globally extreme CpG highly for every cell type in
            # which it is extreme. Measured: 33.8 % of the 6,738 marker CpGs serve more than one entry, and the
            # median entry's panel is only 37 % exclusive. Where a panel is mostly shared, the number is a reading
            # of a shared block rather than of that cell type: print it, withhold the individual direction claim,
            # and show the exclusivity so the reader can see why.
            # LINEAGE GROUP (iamatlas_collinearity_groups_v0_1, built 2026-06-26): entries inside one group are
            # methylation-collinear - the atlas cannot separate them, so the honest unit of a per-cell claim is
            # the GROUP, not the member. Independent of panel exclusivity: two entries can be separable in the
            # atlas and still share most of their marker panel.
            gid=(R.get("cgroup") or {}).get(cell)
            mem=(R.get("cgmembers") or {}).get(gid) or []
            if len(mem)>1:
                others=[m for m in mem if m!=cell]
                gtxt=(f"<span style='color:#d68910'>{_e(gid)}</span> <span class='m'>with {_e(', '.join(others[:3]))}"
                      f"{' +%d'%(len(others)-3) if len(others)>3 else ''} - not separable; read at group level</span>")
            elif gid: gtxt=f"<span class='m'>{_e(gid)} (alone - separable)</span>"
            else: gtxt="<span class='m'>not in the grouping</span>"
            ex=(R.get("excl") or {}).get(cell) or {}
            exf=ex.get("exclusivity"); ex_ok=bool(ex.get("individual_claim_ok", True))
            exs="" if exf is None else (f"{100*exf:.0f} %" if ex_ok else f"<span style='color:#d68910'>{100*exf:.0f} %</span>")
            bar=(posbar(A,(ref or {}).get("p10"),(ref or {}).get("p90")) if ex_ok
                 else "<span class='m'>claim withheld - panel mostly shared</span>")
            if not ex_ok: dirn=""
            # ON THE GAUGE, per cell. A is shifted by this entry's own measured healthy zero so that a healthy
            # reading of THIS entry sits at 1.000; the gauge landmarks (0.95 / 1.05 / 1.07 Warburg / 1.10 breach
            # / 1/H_min ceiling) are properties of the ratio and then apply to the cell exactly as to the class.
            # Without the shift they cannot: measured healthy medians on the marker surface run 0.55-1.15 across
            # atlas entries because the panels differ, so raw per-cell A has no common zero.
            gA=None; gt=""; gnote=""
            z0=(ref or {}).get("p50")
            if A is not None and z0:
                gA=A-(z0-1.0)
                nb=T.scheme(); _n=[b for b in nb["bands"] if b[0]=="NORMAL"]
                nw=(_n[0][2]-_n[0][1]) if _n else 0.09
                wide=(ref.get("p90") is not None and ref.get("p10") is not None and (ref["p90"]-ref["p10"])>nw)
                if fr>0 and not wide:
                    gt,gn=T.tier_of(gA, True, hm)
                    gt=f"<b>{_e(gt)}</b>"
                elif fr>0 and wide:
                    gt=("<span class='m' title='this entry&#39;s healthy spread is wider than the gauge&#39;s NORMAL band'>"
                        f"no tier - healthy spread {ref['p90']-ref['p10']:.3f} &gt; NORMAL band {nw:.3f}</span>")
                else:
                    gt="<span class='m'>no tier - below presence floor</span>"
                    gnote=""
            elif A is not None:
                gt="<span class='m'>no measured zero for this entry</span>"
            gcell=(f"<span class='n'>{gA:.3f}</span> {gt}" if gA is not None else gt)
            H.append(f"<tr class='{'placed' if fr>0 else ''}'><td>{_e(cell)}</td><td>{'yes' if fr>0 else '-'}</td><td class='n'>{100*fr:.1f} %</td><td class='n'>{'' if A is None else f'{A:.3f}'}</td><td class='n'>{cis}</td><td>{rng}</td><td class='n'>{mf}</td><td class='n'>{exs}</td><td>{gtxt}</td><td>{gcell}</td><td>{bar} {dirn}</td></tr>")
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
    H.append("<h3>What this number is, and where it comes from</h3>"
      "<p>The number on the dial is a <b>Mahalanobis distance</b>. It deserves its name on the page, because a reader who recognises it knows "
      "immediately what kind of object it is, and a reader who does not is entitled to a plain explanation.</p>"
      "<p><b>The idea.</b> Suppose you want to say how unusual a person is on two measurements at once - height and weight, say. Raw distance is "
      "useless: 10 kg and 10 cm are not comparable quantities, and in any case tall people weigh more, so the two measurements are not "
      "independent. Mahalanobis' answer, in 1936, was to measure distance in units of <i>how much healthy people vary</i>, and to account for "
      "the fact that the measurements move together. A distance of 1 means one typical spread away from the centre; 2 means twice that. It "
      "converts incommensurable units into one number that means <i>how surprising is this</i>.</p>"
      "<p><b>Why that is the right tool here.</b> A patient has one reading per architecture class. Asking whether they are unusual is exactly the "
      "two-measurement problem multiplied: the classes have different spreads, and they move together (a sample rich in progenitors is poorer in "
      "something else, by construction - the fractions sum to one). A per-class table cannot answer <i>is this person unusual overall</i> without "
      "some rule for combining, and combining z-scores by eye is how a reader talks themselves into a pattern. One number, with a stated alarm "
      "line, is the honest form.</p>"
      "<p><b>Where it comes from - and no, not from cosmology.</b> Prasanta Chandra Mahalanobis introduced it at the Indian Statistical Institute "
      "in 1936, measuring human skulls: he needed to say how far one population sat from another on several correlated measurements at once. It is "
      "a statistician's invention, not a physicist's. <b>But cosmology is one of its heaviest users</b>, and under a different name: every "
      "cosmological parameter fit computes a chi-squared of the form (data minus model) transposed, times the inverse covariance matrix, times "
      "(data minus model) - and that quadratic form <i>is</i> a squared Mahalanobis distance. When a Planck likelihood reports how well a model "
      "fits the sky, that is the number it is computing. So the honest lineage is: the statistic is Mahalanobis' from 1936, the discipline of "
      "building it on a properly measured covariance matrix - and of never trusting it until the covariance itself is measured - is what "
      "cosmology contributed. The same pattern as HEALPix, which also came from outside cosmology before cosmology made it standard.</p>"
      "<p><b>What is honest about this number today.</b> With one commissioned class band it is not yet doing the work it is built for: with a "
      "single axis the distance is simply the absolute value of that class's z-score, and the covariance has nothing to act on. It becomes a true "
      "multi-axis distance when further class bands are commissioned - and the covariance it will then need is the same one the atlas already "
      "carries and the chain does not yet use (see the Sky tab). The number below is correct and it is thin; the report says which.</p>")

    return guard("".join(H),"Departure")

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
 "<td>the <b>laboratory zero</b> and the <b>age term</b>, each measured - 40 healthy arrays of that laboratory, and the "
 "four-laboratory age curve - and subtracted before any reading is placed</td></tr>"
 "<tr><td>Mask the galaxy</td><td>part of the sky cannot be measured; do not guess what is behind it</td>"
 "<td>the <b>presence floors</b>. <b>Below its floor a class is not there</b> - the specimen holds no detectable amount of it - so the chain "
 "masks it black and reports nothing about it. <i>The analogy is close but not exact, and the difference is worth stating:</i> the galaxy hides a "
 "sky that really is behind it, whereas an absent class has nothing behind the mask. Both are refusals to report where the instrument cannot see, "
 "for different reasons. PROC-CEIL-01 measured what happens without the refusal: on healthy whole blood the absent classes read at or past their "
 "ceiling, because their identity addresses carry other cells' DNA and average near a coin flip - so an unmasked healthy sample would report two "
 "classes past breach</td></tr>"
 "<tr><td>Component separation - dust, synchrotron, free-free</td><td>you cannot read the sky until you separate what lies in "
 "front of it</td><td><b>Stage 2, the composition step.</b> Not a loose parallel: the second solver in this chain <i>is</i> NILC, "
 "the needlet internal linear combination Planck used, pointed at cell types instead of foregrounds</td></tr>"
 "<tr><td>Deconvolve the instrument beam</td><td>resolution is finite and must be declared, not assumed</td>"
 "<td>about 2.2 CpG addresses per pixel at NSIDE 128 is this instrument's beam. The smoothed panel in the figure below is the "
 "beam-smoothed map, which is the only fair comparison to a published CMB image</td></tr>"
 "<tr><td>A noise covariance per pixel, not one number for the map</td><td>pixels are not equally trustworthy</td>"
 "<td>that laboratory's <b>measured per-address spread</b> from its own healthy panel - the denominator of every z on this page</td></tr>"
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
 "bounds</b> - 123 standard-deviation columns with interval columns beside them - because MCMC produces a distribution, and keeping only its "
 "centre throws away most of what was computed. Three uses are live: the floors were fitted this way and frozen; the sky weights each class panel "
 "by how well the atlas pinned that class down; and the second solver uses each entry's posterior SD as its inverse-variance weight, which is what "
 "makes it sensitive to faint components.</p>"
 "<p><b>What is not yet used - the largest piece of unspent evidence in the chain.</b> MCMC also gives the <i>covariance between cell types at the "
 "same address</i>: if two cell types are hard to tell apart there, the chains wander together, and that correlation is exactly what a proper "
 "generalised-least-squares separation needs. The chain currently treats each entry's uncertainty as independent - the conservative choice, and the "
 "wasteful one. A full covariance would let the composition step say <i>these two are individually uncertain but their sum is well determined</i>, "
 "which is precisely the situation PROC-SEP-03 found in blood. The credible intervals are also asymmetric near the ends of the scale and are "
 "currently summarised by a symmetric SD. Both are on the list below.</p>"

 "<h3>Brilliance - the first tool taken from cosmology, and where it went</h3>"
 "<p>The first CMB-derived instrument in this work was not the sky; it was <b>brightness</b>. Surface brightness is how astronomy states an "
 "intensity that does not depend on the distance to the source or the size of the telescope, and the brightness layer built alongside the atlas "
 "applied that idea to an architecture class: how strongly does this class shine at its own addresses, on a scale that does not depend on how much "
 "of it happens to be in the tube. That was the right instinct, and it is why a per-class expectation exists at all. It has been <b>superseded "
 "rather than retired</b>: the sky now builds its expectation from <i>this sample's own composition</i>, which a precomputed per-class file cannot "
 "do. The lineage is worth stating, because the first import from cosmology is still load-bearing one layer down.</p>"

 "<h3>Two maps from one patient - the difference map</h3>"
 "<p>In the author's words, and it is the strongest argument on this page: <i>a methylome sky can be re-measured on the same patient. Two draws "
 "six months apart gives a difference map - and difference maps are where cosmology's entire toolkit is most powerful, because the static "
 "foregrounds cancel. That is not a metaphor, it is an experimental design a clinic can execute, and it is the strongest reason the analogy is a "
 "method rather than a decoration.</i></p>"
 "<p><b>Is it sensitive enough to see a change in one island? Measured, not asserted.</b> On the Uppsala panel the between-person spread per "
 "address is 0.029 in beta units (IQR 0.018-0.046). The quietest addresses - 5th percentile, 0.0089 - bound the purely technical part from above, "
 "since nothing can be quieter than the noise. A paired difference of two draws from one person removes that person's baseline, the laboratory zero "
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
 "<b>no repeat draws of the same person exist in anything held here</b> - the EPIC-Italy foundation cohort is 460 distinct participants with no "
 "second sample - so a real serial test must re-measure it from actual replicates. And a difference map cancels the laboratory only if both draws "
 "went through the same laboratory and pipeline; otherwise the pipeline map and laboratory zero must be applied to each before differencing. "
 "Finding a serial cohort is the first item on the list below.</p>"

 "<h3>Is the sphere necessary? A straight answer</h3>"
 "<p><b>For the measurement, no. For the toolkit, yes.</b> The residual is a one-dimensional sequence along the genome; the sphere is a "
 "space-filling reindexing of it. What makes that legitimate rather than ornamental is that the reindexing <b>preserves locality</b> - measurable, "
 "not assumed: every one of the 196,608 pixels holds CpGs that are genomically contiguous and on a single chromosome, with a median span of "
 "<b>511 base pairs</b> (90th percentile 20 kb); a quarter of all neighbouring-pixel pairs sit within 10 CpGs of each other in genomic order and "
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
    s=o["patient_sky"]; H=["<div class='callout'><b>What this sky is compared to.</b> Not to a healthy picture - there is no reference image anywhere in this comparison. Every pixel is <span class='m'>z = (&beta; &minus; &Sigma;<sub>c</sub> f<sub>c</sub>&mu;<sub>c</sub> &minus; m<sub>lab</sub>) / s<sub>lab</sub></span>: this specimen's own methylation at that address, minus what <b>its own composition</b> predicts for it (the atlas mean of each class, weighted by the fractions Stage 2 measured in <i>this</i> specimen), minus this laboratory's measured zero, divided by this laboratory's measured per-address spread across its own healthy donors. So the comparison is to a healthy cohort's <b>statistics</b>, and the number to read is the fraction of addresses beyond |z| = 2 against the healthy range of 2.6-3.2 per cent. Red is above the composition expectation, blue below, black not assessable. Two plates are comparable only within one laboratory, because m<sub>lab</sub> and s<sub>lab</sub> are that laboratory's own.</div>", SKY_WHY, "<h3>This sample's sky - Stage 4.6</h3>"]
    # 2026-09-25: say it at the top, not in the refusal list at the back.
    if not (s.get("available") and s.get("_sky") is not None):
        H.insert(0, "<p class='warn'><b>No sky was rendered for this specimen.</b> " +
                 html.escape(str(s.get("reason") or "this laboratory has no commissioned residual scale, so "
                                  "there is no zero to take residuals against")) +
                 ". Every figure on this page is a reference illustration, identical in every report - none "
                 "of them is this specimen.</p>")
    H.append("<p>Every CpG the chain reads is placed on a sphere in genomic order (HEALPix, NSIDE 128 - the projection Planck used for the microwave background). At each address the chain computes what this sample's <i>own composition</i> predicts (the Stage 2 fractions mixed over the atlas class means), subtracts the laboratory's per-address zero, and divides by the laboratory's healthy spread at that address - both measured from the same 40 healthy arrays that set the laboratory zero. "
             "The plate shows that residual z. A healthy sky is <b>quiet</b>: 2.6-3.2 % of addresses beyond |z| = 2 on the four commissioned laboratories (the Gaussian expectation is 5 %; the scale is ~1.1x conservative and that constant is printed on every plate). A class panel renders only when Stage 2 places the class above its measured presence floor; masked panels say so.</p>")
    if s.get("available") and s.get("_sky") is not None:
        try:
            sys.path.insert(0,ENGINE); import stage_4_6_patient_cmb as S
            png=os.path.join(workdir,f"sky_{sid}.png"); S.render_plate(s["_sky"],png,f"{sid} - residual z on the laboratory's own zero and scale ({s.get('lab')}, panel n={s.get('scale_panel_n')})")
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
    else: H.append(f"<p class='pend'>SKY NOT AVAILABLE - {_e(s.get('reason') or 'this laboratory has no commissioned residual scale (40 healthy arrays through Stage 1 are required; see Healthy reference)')}</p>")
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
    H.append("<h3>The floors, and how to check them yourself</h3>"
      "<p>The eight class floors this reading divides by were fitted by MCMC on 37 published reference cells - "
      "32 walkers, 500 burn-in and 5,000 production steps, five independent chains. The samplers, the 37 cells as "
      "a table, and a script that re-runs the calibration in about fifteen seconds are deposited at "
      "<a href='https://doi.org/10.5281/zenodo.22905819'>10.5281/zenodo.22905819</a>. Re-running it on 2026-09-22 returned every floor inside its own posterior "
      "standard deviation - largest difference 0.000245 against SDs of 0.0069 to 0.0088, R-hat below 1.001 on all "
      "eight parameters. The posterior samples themselves were never written to disk by any run, so the deposit "
      "carries the re-run rather than an archive of chains.</p>")
    H.append("<h3>5. What is NOT in the reference</h3><ul><li>No disease sample, no case arm of any cohort.</li><li>No author-processed beta; no typed literature values (the April 80-cell age table was retired for that reason - PROC-RECORD-03).</li><li>The class floors H_min are not fitted here; they come from the 37-cell G-002 calibration and are frozen.</li></ul>")
    H.append(deepdive(R,"the healthy reference and how each constant was measured"))
    return "".join(H)   # not a measurement tab: quotes GEO characteristic fields verbatim ("disease state: normal")

CMB_TWINS=[("MCMC atlas calibration","Cosmological parameter estimation (Planck likelihood chains)","Posterior mean and SD per CpG per class; class floors H_min with R-hat < 1.001 on 37 reference cells","G-002 / G-003b, IAMAtlasREBUILD"),
 ("Pipeline map","Instrument calibration transfer (cross-calibrating detectors)","affine beta map, fit on 32,688 identity loci; transfers to a second laboratory at median A 1.0097","beta_scale_maps_v1.json"),
 ("Laboratory zero","Monopole / dipole removal","one constant per laboratory from 40 healthy arrays; four labs on one scale","lab_zero.py, identity_band_v3.json"),
 ("Age curve","Foreground subtraction","healthy A rises 0.47 mA/yr by decade, same slope on four labs; subtracted before placement","reference_age_curve_v1.json"),
 ("Composition solver (Walther)","Constrained component fitting","conservative NNLS against the atlas: a cell is placed only when the evidence forces it, so the composition the report stands on is not inflated","walther_iam_deconvolver.py"),
 ("Second opinion (NILC) - VINDICATED, NOT YET RE-WIRED","Needlet internal linear combination: the Planck component-separation method","variance-weighted and deliberately sensitive. Switched off in July 2026 because it disagreed with Walther on every blood sample; PROC-NILC-01 found the disagreement WAS the finding - NILC was reporting that the atlas cannot split the blood classes, which PROC-SEP-03 then measured (7/7 blood SPLIT, kappa < 10). It is commissioning row 2b and is not in this run; the implementation is linked below","nilc_celltype_deconvolver.py"),
 ("Residual sky","The anisotropy map / residual map after model subtraction","z per address on the laboratory's own zero and spread; healthy 2.6-3.2 % beyond |z|=2","stage_4_6_patient_cmb.py"),
 ("Presence floors","Galactic mask","a class panel renders only above the floor measured on 160 healthy arrays","presence_floors_v1.json"),
 ("Pre-registration and the falsification register","Blind analysis","every bar written and hashed before the run; failures kept on the record as sealed","Record/PROC_data")]

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

CHAIN=[
 ("0","Intake","declared age, specimen, substrate; file integrity hash","stage_0_intake.py",
  {"in":"the IDAT pair (or a calibrated beta vector), declared age, specimen, substrate, laboratory","out":"a checked context record; a SHA-256 of each input file so the run can be tied to exactly these bytes",
   "why":"every later constant is specimen-specific and laboratory-specific. If the specimen is not declared, the presence floors and the healthy band do not apply, and the chain must not guess.",
   "refuses":"an undeclared specimen, or an array type the pipeline map was not fitted on","commissioned":"row 1 of the commissioning table"}),
 ("1","Calibration","IDAT (Red + Grn) -> beta, methylprep noob","stage_1_idat_calibration.py",
  {"impl":"calibrate_idat_to_beta","in":"the raw two-channel intensity files the scanner writes","out":"one beta value per CpG (413,058 on 450K), beta = methylated / (methylated + unmethylated)",
   "why":"raw intensities carry dye bias (the two colour channels are not equally efficient) and probe-type bias (the array has two chemistries). noob normalisation removes both using the array's own out-of-band control probes. Author-processed matrices published on GEO are deliberately NOT used: each laboratory normalises differently, and the offset between two pipelines is larger than the whole healthy band (LESSON-SCALE-01, measured).",
   "refuses":"nothing - it either produces betas or errors","commissioned":"PROC-CAL-01: raw IDAT through this stage reproduced the cached betas the conformance tests were built on, exactly"}),
 ("1s","Pipeline map","beta -> the reference scale (one slope, one intercept)","beta_scale_maps_v1.json",
  {"impl":"stage_1s_scale_map","in":"this pipeline's betas","out":"betas on the scale the class floors were calibrated on",
   "why":"the floors were measured on published reference methylomes processed a particular way. A sample processed differently sits at a different zero. The map is an affine fit on the identity loci, measured once per pipeline - the same operation as cross-calibrating two detectors before comparing their readings.",
   "refuses":"a pipeline with no fitted map: the reading is marked NOT REPORTABLE rather than placed on someone else's scale","commissioned":"PHASE 1 / 1c; transfer verified on a second laboratory"}),
 ("2","Composition","constrained fit against the atlas -> which cell types, and in what proportion","walther_iam_deconvolver.py",
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
 ("2b","Second opinion (NILC)","variance-weighted component separation - the Planck method. VINDICATED, not yet re-wired","nilc_celltype_deconvolver.py",
  {"impl":"stage_2b_second_opinion","in":"the same mapped betas","out":"an independent set of fractions, computed with opposite biases: sensitive where the constrained solver is conservative",
   "why":"in cosmology you never separate components one way only. NILC (needlet internal linear combination) is what Planck uses. Here it was switched off in July 2026 because it disagreed with the constrained solver on every blood sample - which looked like a defect in NILC. PROC-NILC-01 found the disagreement WAS the finding: NILC was reporting that the atlas cannot split the blood classes, which PROC-SEP-03 then measured directly (7 of 7 blood classes inseparable, and the separability statistic quantified). The tool was right and was cut for being right. It is commissioning row 2b and is not in this run.",
   "refuses":"n/a - not currently in the chain","commissioned":"NOT commissioned. Row 2b is open: the decision is whether its output ships as a second column or as a disagreement flag"}),
 ("A","Per-cell A","mean of per-CpG H over each cell's discriminative markers, divided by H_min(class) - all 115 cells","iamatlas_a_scoring.py",
  {"impl":"stage_a_cells","in":"the mapped betas and each cell type's ~100 discriminative marker CpGs","out":"one A per atlas cell type, with marker coverage",
   "why":"this is the surface on which 'which cell moved, and in which direction' can be read, because the markers are chosen to differ between cell types. It is also the surface the sealed cohort anchors were computed on. The formula must be the mean of the per-CpG entropies: marker CpGs are extreme and opposite, so the entropy of their mean beta would read maximal disorder on a healthy sample. The module asserts against that mistake on every import.",
   "refuses":"a cell with fewer than the minimum matched markers is not scored","commissioned":"PROC-ANCHOR-01: the sealed 648-sample foundation-cohort per-cell scores reproduced from raw public data at r = 1.00000, max difference 0.00004"}),
 ("B","Class gauge","H(mean beta over identity loci) / H_min, minus the age term, minus the laboratory zero; placed in the band","cpg_gauge_engine.py",
  {"impl":"stage_b_classes","in":"the mapped betas, the identity loci for each class, the frozen H_min, the age curve, the laboratory zero","out":"A'' per class with its band placement and tier - the reported reading",
   "why":"identity loci are the addresses where a healthy class all sits at one level, so their mean carries meaning and the entropy of that mean is the right statistic here (the opposite of the per-cell surface, deliberately). This stage replaced an earlier version that computed the class gauge over the marker union - which a synthetic-patient test caught reading every healthy patient far above band (PROC-N7-01). The retired statistic still runs, labelled diagnostic, and is not shown on this report.",
   "refuses":"no laboratory zero, or no commissioned band for that class on that specimen -> no placement, no tier, NOT REPORTABLE with the reason printed","commissioned":"PROC-SWITCH-01 -> PROC-SWITCH-02; held-out synthetic healthy read 1.001 with 100 % in band, against 1.125 and 0 % on the retired statistic"}),
 ("4.5","Direction","signed directional composite on sealed panels","bidirectional_decomposition.py",
  {"impl":"stage_4_5_bidirectional","in":"the mapped betas and a sealed directional panel (per-CpG healthy mean and expected direction)","out":"a signed composite per class: which way the departure points",
   "why":"pooled entropy folds hyper- and hypo-methylation together - two opposite movements can average to 'normal'. Keeping the sign is how a class that is drifting in a structured way is distinguished from one that is genuinely quiet. Panels exist only where one has been sealed (immune); the other classes say so rather than guessing.",
   "refuses":"a class with no sealed panel returns no composite","commissioned":"PROC-BIDIR-01, all five bars, including re-extraction of the 726-sample cohort from the raw 5.1 GB GEO file with zero difference"}),
 ("4.6","Sky","composition-residual z at every address, on the laboratory's own zero and spread, projected on a sphere","stage_4_6_patient_cmb.py",
  {"impl":"stage_4_6_patient_sky","in":"the mapped betas, the Stage 2 fractions, the laboratory's per-address zero and spread, the presence floors, the CpG-to-pixel mapping","out":"a residual map - one z per address - and per-class panels with their beyond-|z|=2 fractions",
   "why":"a single number per class cannot say WHERE in the genome a departure lives. The sky can. The expectation at each address is what this sample's own composition predicts, so the map is a residual in the cosmologist's sense: data minus model. A healthy sky is quiet at 2.6-3.2 % beyond |z| = 2 on the four commissioned laboratories.",
   "refuses":"a laboratory with no measured residual scale -> the sky is not rendered at all; a class below its presence floor -> that panel is masked, and says so","commissioned":"PROC-CMB-01 through 05. The retired brightness formula it replaced divided by the atlas posterior spread of a class mean and compared whole blood against a pure-class mean, reading most of a healthy genome as anomalous - that is closed"}),
 ("4.6","CpG -> sky mapping","every atlas CpG to one of 196,608 pixels in genomic order","iamatlas_cpg_to_healpix_nside128.npy",
  {"impl":"stage_4_6_patient_sky","in":"chromosome and position for each CpG","out":"a HEALPix pixel index, NSIDE 128",
   "why":"HEALPix is the projection Planck used: equal-area pixels, so no part of the map is visually over-weighted. Genomic order means neighbouring addresses are neighbouring pixels, which is what makes a structured departure look structured.",
   "refuses":"an unannotated CpG goes to a sentinel pixel and is excluded","commissioned":"deterministic across builds, and verified to assign all 483,092 CpGs to exactly the same pixels as the mapping built with the atlas for the reference plates"}),
 ("5","Departure","distance over the banded class axes, with chi-square lines and the laboratory's false-alarm rate","cpg_conductor.py",
  {"impl":"stage_5_mahalanobis","in":"each reportable class's A'' and the healthy band","out":"one distance, the p95 and p99 lines, and this laboratory's measured healthy false-alarm rate",
   "why":"a clinician needs one number for 'how unusual is this sample overall', and it has to come with how often healthy people trip it. With one commissioned class band the distance is just |z| of that class; as further class bands are commissioned it becomes a true multi-axis distance. The laboratory's own false-alarm rate travels with the number because it differs measurably between laboratories - the residual cause is the Sentrix chip term, which is its own open row (5b).",
   "refuses":"no banded axis -> no distance","commissioned":"PROC-MAHA-01 -> PROC-MAHA-02, with one laboratory's healthy tail exceeding the sealed bar recorded as a failure and the false-alarm rate printed on every report as the remedy"}),
 ("7","Tiers","one tier function, read from one file; AT_CEILING at 1/H_min","cpg_tiers.py",
  {"impl":"stage_b_identity","in":"A'', whether the reading is reportable, and the class H_min","out":"one tier word and a note, or nothing",
   "why":"before this stage the engine carried three disagreeing definitions of where NORMAL ends. Now every tier word in the chain - including the colours on this page - comes from one function reading one file. NORMAL is the healthy central 95 %; 1.07 and 1.10 are the physics lines; above 1/H_min the arithmetic cannot go, so the report prints AT_CEILING with the ceiling value rather than a number above it.",
   "refuses":"a non-reportable reading gets no tier at all - a tier without a commissioned band would be a fabrication","commissioned":"PROC-TIER-01, with a kit test that exercises every boundary from the file, both sides, and fails if a literal breakpoint reappears in engine code"}),
 ("B","The reported gauge","identity loci -> A, then the decade curve and the laboratory zero -> A''","iamatlas_gauge_identity_loci_v1_0.json",
  {"impl":"stage_b_identity",
   "in":"the mapped betas, the class's identity loci and its floor, the donor's declared age, the laboratory's zero",
   "out":"A'' - the number this report leads with: entropy over the floor, corrected to the healthy line for that decade and to this laboratory's own zero",
   "why":"this is the surface the gauge reports on. Identity loci sit near beta = 0.73 in every healthy donor of the class, so a departure is a departure of the class's own pattern rather than of a marker panel chosen to separate two groups",
   "refuses":"without a laboratory zero the placement, the tier and the departure are all withheld and only A_mapped prints; without a declared age the absolute reading is withheld",
   "commissioned":"PROC-MAHA-01 and PROC-MAHA-02 (the zero and the band); the chip term was measured and NOT commissioned (PROC-MAHA-03)"}),
 ("5d","Marker-union hull (diagnostic)","the same departure computed on the marker-union surface","iamatlas_mahalanobis_scoring.py",
  {"impl":"stage_5_hull_marker_union",
   "in":"the mapped betas and the age-matched healthy reference",
   "out":"a Mahalanobis distance on the marker-union surface, carried in the bundle as a diagnostic",
   "why":"the marker-union surface is what the pre-atlas work measured. Keeping it beside the reported gauge lets the two be compared deliberately",
   "refuses":"it is never reported beside the identity gauge: the two surfaces move in opposite directions with age (RECON D2), so quoting them together would invite a false comparison",
   "commissioned":"diagnostic only - no procedure commissions it for reporting"}),
 ("6","Age reference","where the healthy line for that decade sits","reference_age_curve_v1.json",
  {"impl":"stage_6_cellular_age",
   "in":"the donor's declared age and the reference curve measured on 1,379 healthy donors from four laboratories",
   "out":"the decade correction subtracted from A before the reading is placed",
   "why":"fidelity falls with age on this surface at 0.47 mA/yr, so a reading must be judged against the healthy line for that decade rather than against a single population mean",
   "refuses":"a cellular age in years for one patient is NOT reportable: a lifetime of drift is 0.047 against a within-laboratory healthy spread of 0.0235 (PROC-AGE-01). The curve corrects a reading; it does not date a person",
   "commissioned":"PROC-AGE-01 - the trajectory is reproduced on four laboratories, the per-patient inversion is closed"}),
 ("6d","Age on the marker union (diagnostic)","the same age arithmetic on the marker-union surface","iam_cellular_age_scoring.py",
  {"impl":"stage_6_cellular_age_marker_union",
   "in":"the mapped betas and the marker-union age reference",
   "out":"a diagnostic age statistic on the pre-atlas surface",
   "why":"it is the quantity the earlier work reported, kept for comparison",
   "refuses":"not reported, and never quoted beside the identity-gauge age arithmetic - the sign differs by surface",
   "commissioned":"diagnostic only"}),
 ("9","Report","this page, rendered from the bundle","build_methylphys.py",
  {"in":"the conductor's output bundle and the runtime files listed on Integrity","out":"this document",
   "why":"the report is part of the instrument, not decoration: it decides what is shown and what is withheld. A vocabulary guard refuses to write the measurement tabs if they name a condition, a verdict, or an age in years - it has caught the author's own wording more than once.",
   "refuses":"writing the file at all if the guard trips","commissioned":"row 9 - IN BUILD, UNSEALED. It seals when a report has been read line by line against the commissioning table"}),
]
DOCS=[("RUNBOOK.md","how to run the chain, and how to commission a new laboratory (40 healthy arrays -> its zero and sky scale)"),
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
       "<p>Orchestrated by <code>cpg_conductor.run_full</code>. Open any stage for what goes in, what comes out, why it exists, what it refuses to do, and which sealed procedure commissioned it. Stage 6 (an age in years) and Stage 8 (matching a pattern to a condition) are <b>not</b> chain stages: single-array age resolution is about 50 years, and naming a condition is not this instrument's job. The retired marker-union class statistic runs as a diagnostic only and is not shown.</p>"]
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
    H.append("<p class='m'>The June manifest (CPG_KISS_Chain_Files.md) described the pre-switch chain and must not be used: it names the identity-loci gauge a 'false road' (reversed by PROC-N7-01 and PROC-SWITCH-02) and calls the eight H_min values 'Mahaffey numbers' (they are measured class entropy references; the Mahaffey number is E_drive/k_BT).</p>")
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
    kB=1.380649e-23; T=310.15; Rg=8.314462618; EL=kB*T*math.log(2); M=54000/(Rg*T)
    nloci={c:len(v.get("loci",[])) for c,v in R["ident"].items() if isinstance(v,dict) and "loci" in v}
    b=R["band"].get("immune",{}); hm=R["ident"]["immune"]["H_min"]
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
<p>Concretely, for each of the eight classes we found the DNA addresses where every healthy cell of that class sits at about the same methylation level - the addresses that say "I am an immune cell" rather than the ones that distinguish one immune cell from another. We call them <b>identity loci</b>: {nloci.get('immune',0):,} for immune, {nloci.get('cycling',0):,} for cycling epithelial, {nloci.get('secretory',0):,} for secretory. Then we measured, once, how tidily a healthy cell of each class holds those addresses, and <b>froze</b> the eight numbers. They have not been touched since.</p>

<h3>6. Tidiness has a number, and the reading is a ratio</h3>
<p>Look at the tags at a set of addresses and ask: how predictable are they? If every address is definitely on or definitely off, the pattern is perfectly tidy - you could guess any one of them. If every address is a coin flip, it is maximally scrambled. Information theory gives this a number called <b>entropy</b>, from 0 (perfectly tidy) to 1 (pure coin flip); it is the same quantity a physicist uses for disorder in a gas and an engineer for noise on a line, and it is computed from the array data alone.</p>
<p>Take a sample. Go to the immune identity addresses. Compute the entropy. Divide by the frozen healthy level for immune cells ({hm}). The answer is <b>A</b>. A = 1.00 means this sample's immune cells hold their identity pattern exactly as tidily as healthy ones do. A = 1.10 means the pattern is 10 % more scrambled than healthy - the cells are losing their grip on who they are. The middle 80 % of healthy donors land between {b.get('p10','?')} and {b.get('p90','?')}, and that band is drawn on every gauge. Because the reference is frozen, <b>one sample can be read on its own</b> - no control group in the room, no cohort required.</p>

<h3>7. The same floor appears far outside biology - which is why we trust it</h3>
<p>The reason to have confidence that this floor is real, and not a biological coincidence, is that the identical ratio governs systems with no biology in them at all. The same form - ordering energy divided by the minimum cost set by temperature - describes:</p>
<table class='t'><tr><th>system</th><th>what it spends to maintain order</th><th>the floor it pays against</th><th>ratio</th></tr>
<tr><td><b>A living cell</b></td><td>ATP free energy, at 37 &deg;C</td><td>thermal energy at body temperature</td><td>{M:.2f}</td></tr>
<tr><td><b>A computer chip</b></td><td>switching energy per bit-flip</td><td>the minimum cost to write one bit (Landauer's limit)</td><td>~117 (Apple M1)</td></tr>
<tr><td><b>A quantum computer</b></td><td>the energy of a quantum operation</td><td>the same bit-writing floor - which is why these machines must be chilled to near absolute zero, to lower the floor itself</td><td>1.000 (Al transmon)</td></tr></table>
<p>Landauer's principle - that writing or erasing one bit has a minimum, unavoidable energy cost set by temperature - is an established result in physics, published in 1961 and used routinely in computing. A microchip today runs far above that floor, but as chips are pushed to be more efficient they approach it, and the floor is what eventually limits them. A quantum computer fights the same floor from the other direction: it cannot easily lower its operating energy, so it lowers the <i>temperature</i> instead, dropping the floor toward zero so its fragile information can survive. The cell cannot cool. It runs at 310 K and pays its 21 quanta.</p>
<p>The point for a clinician is simply this: <b>the floor used here for a cell is the same kind of floor an engineer uses for a chip.</b> A recognised principle of physics, applied to biology. That is the difference between a measurement and a metaphor - the same arithmetic reads a cell, a transistor and a refrigerated qubit, and it was not invented for any one of them.</p>

<h3>8. What is measured, and what is not derived</h3>
<p>One point of honesty that matters to a reviewer, and that has been corrected in this work since earlier drafts: the energy bound above constrains the <b>cost of writing</b> a pattern. It does not, by itself, tell you the <b>entropy of the pattern a healthy class holds</b>. So the eight class levels are <b>measured, not derived</b> - fitted once, in April 2026, from 37 published reference cell methylomes using Markov-chain Monte Carlo (the same class of inference cosmology runs against the Planck likelihood), with convergence checked and a bootstrap cross-check in which every frozen value falls inside its interval - and then frozen before any sample in this work was scored against them. That is the stronger claim, because it is checkable: the calibration script and its 37-cell database with every DOI are linked from the Record tab. Nothing about the reference is withheld.</p>

<h3>9. The atlas, and what MCMC and a posterior actually mean</h3>
<p>Two numbers on this report came out of a fitting procedure rather than a direct measurement, and a reader is entitled to know what kind of object they are. Neither idea is difficult; both are usually explained in a way that assumes you already know them.</p>
<p><b>The atlas.</b> Every reading here is a comparison: this sample against what each cell type looks like when it is healthy. The atlas is that reference - 483,092 CpG addresses by 115 cell types, each entry a methylation level with an uncertainty attached. Without it there is nothing to compare to and no way to ask which cell type a departure belongs to; the composition step, the per-cell readings and the sky all read out of it. It is the single most consequential file in the chain, which is why its provenance record and checksum are linked on the Chain tab rather than described.</p>
<p><b>Why an atlas entry needs an uncertainty and not just a value.</b> A reference built from six published samples of a rare cell type is not as trustworthy as one built from sixty, and a plain average hides which is which. Carrying an uncertainty per entry is what lets the chain refuse to place a cell on weak evidence instead of guessing, and it is what the second solver weights by.</p>
<p><b>MCMC, in plain words.</b> Markov chain Monte Carlo. Suppose you want the methylation level of one cell type at one address and you have a handful of noisy published measurements. You could average them - but then you have one number and no idea how much to trust it. Instead you ask: of all the values this address <i>could</i> have, which are consistent with the data I have? MCMC answers that by taking a long random walk through the candidate values, stepping more often toward values that fit the data better, and keeping a record of everywhere it went. Run it long enough and the record is the answer: values it visited often are plausible, values it rarely visited are not. It samples an answer rather than solving for one, and it works on problems where solving is impossible.</p>
<p><b>The posterior is that record.</b> Not a single number - a distribution: for this address in this cell type, the range of levels consistent with the evidence and how strongly each is supported. From it come the two numbers the atlas stores: the <b>posterior mean</b> (the centre, which is the atlas value) and the <b>posterior SD</b> (how wide the range is, i.e. how well the data pinned it down). Wide means the evidence was thin.</p>
<p><b>Why posteriors matter here, concretely - three places.</b> <b>One:</b> the class floors H_min were fitted this way from 37 published reference cell methylomes, with the chains run to convergence (the standard diagnostic, R-hat, below 1.001 - it compares independent walks and asks whether they ended up describing the same distribution; if they disagree the answer is not yet trustworthy). Those eight values were then frozen and have not been re-fitted since - they are constants in this instrument, not parameters it tunes - and a separate leave-one-out bootstrap agreed with every one of them. <b>Two:</b> the sky weights each class panel by how well the atlas pinned that class down. <b>Three:</b> the second solver uses each entry's posterior SD as its inverse-variance weight, which is what makes it sensitive, and what made its disagreement with the conservative solver informative rather than noise.</p>
<div class='warn'><b>And the one place a posterior must never be used.</b> The posterior SD describes how well the atlas pinned down an <i>average</i>. It is <b>not</b> how much healthy people differ from one another. Those are different quantities and the second is typically many times larger. Using <i>healthy mean &plusmn; 1.96 &times; the posterior SD of the mean</i> as a normal range is a category error with a large practical cost: it makes ordinary healthy samples appear to sit many standard deviations outside normal, because the interval it produces is the precision of an average rather than the spread of a population. This report therefore prints three separately labelled quantities and never mixes them: the <b>uncertainty on this sample own reading</b> (from resampling its own CpGs), the <b>healthy range</b> (measured across healthy people), and the <b>atlas posterior</b> (how well the reference itself is known - used for weighting, never as a range).</div>

<details><summary><b>Optional - the formulas and the three quantities</b></summary>
<table class='t'><tr><th>symbol</th><th>name</th><th>what it is</th><th>units</th><th>varies by</th><th>fixed by</th></tr>
<tr><td>M</td><td>Mahaffey number (the cellular margin)</td><td>E_drive / k_B T - how many thermal quanta the writing process spends per irreversible operation</td><td>none (ratio)</td><td>substrate</td><td>biochemistry / device physics</td></tr>
<tr><td>H_min(c)</td><td>class entropy reference</td><td>the binary entropy a healthy cell of class c holds at its identity loci</td><td>bits</td><td>class (8)</td><td>healthy reference cells, MCMC-calibrated once, frozen</td></tr>
<tr><td>A</td><td>the gauge</td><td>H(beta_mean at identity loci) / H_min(c)</td><td>none (ratio)</td><td>class x sample</td><td>the sample, over the frozen reference</td></tr></table>
<p>Binary entropy: H(&beta;) = -&beta; log&#8322;&beta; - (1-&beta;) log&#8322;(1-&beta;). Landauer bound at body temperature: E &ge; k_B T ln 2 = {EL:.3e} J.</p>
<p><b>Why identity loci and not marker loci.</b> H is concave, so H(mean &beta;) over a set of addresses is largest when the addresses average to a coin flip. Marker loci are deliberately chosen to be extreme and opposite between cell types; averaged over a mixture they read as disorder that is not there - a real defect this chain had and that a synthetic-patient test caught (PROC-N7-01). Identity loci are the addresses where a healthy class sits at one level, so the mean carries meaning and A = 1 is healthy by construction. The gauge is two-to-one in &beta; (H is symmetric about 0.5) and has a structural ceiling at 1/H_min.</p>
<p><b>Filter and ruler.</b> Sanchez &amp; Mackenzie (2016) used k_B T ln 2 to model the thermal <i>background</i> and remove it, so regulatory signal stands out against a control centroid - the premise that the methylome obeys Landauer's bound is theirs and is peer-reviewed. Here the same constant sets the <i>unit</i>, and the healthy level is a statement of how far above it a class holds its pattern. One filters, one calibrates. This work reached the constant independently (cosmology &rarr; quantum hardware &rarr; semiconductors &rarr; cells) and read their papers on 2026-09-20; they are cited, not built upon.</p></details>"""+deepdive(R,"the physics")

HOWTO_LEVELS=[("below the healthy range","A'' below the band","the class is holding its pattern more tightly than the healthy reference. Real, and not automatically good: strongly hypomethylated states read here, and so does a class whose sampled population is unusually uniform."),
 ("within the healthy range","A'' inside the band, NORMAL","the class is holding its identity pattern the way healthy cells of that class do, for this age and this laboratory."),
 ("elevated","above the band","the pattern is measurably looser than healthy. The class is drifting. This is the regime where a reading is worth a second sample or a closer look, and where nothing has failed yet."),
 ("at 1.07 - the Warburg line","WARBURG_TRANSITION (a line, not a band)","the class has flipped toward aerobic glycolysis. This is a boundary in the intervention strategy, not in the arithmetic - see below."),
 ("at 1.10","BREACH","the no-return line. The class is no longer holding its identity pattern at the level that defines it. It is not the ceiling: on several class-and-substrate combinations the ceiling sits BELOW this line, so breach cannot be reached there at all."),
 ("at the ceiling (1/H_min for that class and substrate)","AT_CEILING","saturation: the entropy of the identity loci has reached its structural maximum, so the ratio cannot go higher. The report prints the ceiling value, never a number above it. The ceiling is NOT where identity is lost - that is the floor at 1.10. Saturation is a separate limit, and it differs by class and by substrate.")]

def tab_howto(R):
    b=R["band"]["pooled"]   # identity_band_v3 is keyed pooled / per_decade, class named in _meta - as cpg_conductor reads it
    H=[f"""<h2>How to read the gauge</h2>
<p>A thermometer is only useful if its reading means something definite. This tab sets out what A means at each level, where the lines are, and - just as important - what the reading does not claim. <b>The instrument reports a cellular state; the clinician decides what to do about it.</b> The roles are distinct and kept distinct on purpose.</p>
<h3>What A is measuring</h3>
<p>A does not measure how many cells there are, or how large a mass is, or whether anything is present anywhere. It measures one thing: <b>how far a cell population's methylation pattern has drifted from the healthy reference for its architecture class</b> - in plain terms, how well a cell is still maintaining its own identity. A well-differentiated cell doing its job sits near the reference; a cell that has lost its architectural fidelity reads high. This is closer to a measure of <i>grade</i> (degree of dedifferentiation) than of size or burden, and that distinction is what makes the reading behave the way a clinician would want.</p>
<h3>Where the locker analogy helps</h3>
<p>Think of a school with eight grades, every student with a locker. Most lockers tell you nothing about the grade. Some are characteristic: every ninth-grader's holds the same geometry book. Those are the identity lockers. The gauge walks the identity lockers for one class and asks how consistently they still hold what that class's lockers hold. It is not counting students.</p>
<h3>The levels</h3><table class='t'><tr><th>level</th><th>on the gauge</th><th>what it means</th></tr>"""]
    for a,c,d in HOWTO_LEVELS: H.append(f"<tr><td><b>{a}</b></td><td>{c}</td><td>{d}</td></tr>")
    H.append(f"""</table><p class='m'>The healthy band on this report is the middle 80 % of {R['band']['_meta'].get('n')} healthy donors from four laboratories in four countries, after each donor's age term and their laboratory's constant are removed: {b.get('p10','?')} to {b.get('p90','?')}. The tier onsets come from one file ({_link(R,'tier_breakpoints.json')}) read by one function - the engine no longer carries a second opinion about where a line is.</p>""")
    H.append("<h3>The band is age-referenced - here it is, decade by decade</h3><table class='t'><tr><th>decade</th><th>healthy donors</th><th>p10</th><th>median</th><th>p90</th><th>age term removed before placement</th></tr>"
      + "".join(f"<tr><td>{d}s</td><td class='n'>{v.get('n','')}</td><td class='n'>{v['p10']}</td><td class='n'>{v['p50']}</td><td class='n'>{v['p90']}</td><td class='n'>{R['age']['curve'].get(d,0):+.4f}</td></tr>"
                 for d,v in sorted(R["band"]["per_decade"].items(), key=lambda kv: int(kv[0])))
      + "</table><p class='m'>Healthy A rises about 0.47 milli-A per year across these four laboratories - a whole lifetime of that drift is about 0.047, roughly the width of the healthy band itself. The decade term is subtracted before a sample is placed, so the band a patient is read against is the one for their own decade. The curve is built leave-one-laboratory-out, so no laboratory sets its own age reference.</p>")
    for fn,cap in (("CPG_Gauge_Cell.png","<b>The cellular gauge.</b> A = 1.00 is the healthy reference, in the middle of the NORMAL band - not an edge. Below it is the suppressed / inverted direction (post-chemotherapy and immunosuppressed samples sit near 0.90); above it the loosening runs through MARGINAL, the Warburg line at 1.07, DETECTABLE, and BREACH at 1.10. Author's figure."),
                   ("CPG_Gauge_Cosmic.png","<b>The same gauge on a star</b>, which is where the ceiling becomes obvious. A main-sequence star sits healthy; an isolated white dwarf reads above healthy but is <i>ceiling-capped below breach</i> - it has no mechanism to gain mass, so it cannot reach the no-return event however spent it looks. Only a collapse-capable core reaches A = 1 in the gravitational sense, where the Chandrasekhar and TOV limits and the Schwarzschild condition all land. Author's figure.")):
        pth=R["files"].get(fn)
        if pth: H.append(f"<figure><img src='data:image/png;base64,{base64.b64encode(open(pth,'rb').read()).decode()}' style='width:100%;max-width:860px;border:1px solid var(--ln);border-radius:6px'><figcaption class='m'>{cap} <b>Reference figure</b> - the same in every report, not this specimen.</figcaption></figure>")
    w=R["warburg"]; BR=R["breach_line"]
    H.append("<h3>Three different things, and none of them is A = 1.00 sitting on a floor</h3>"
      "<p>This is worth getting exactly right, because two of these are physical limits and one is a calibration point, and they are easy to conflate.</p>"
      "<table class='t'><tr><th></th><th>what it is</th><th>where it sits</th><th>what crossing it means</th></tr>"
      "<tr><td><b>A = 1.00</b><br><span class='m'>the healthy reference</span></td><td>the calibration point: the reading a healthy cell of that class gives. It is the <i>middle of the green band</i>, not an edge of anything.</td><td>mid-NORMAL. The commissioned healthy band on this chain is the central 95 % of 1,379 healthy donors, and it is nearly symmetric about 1.00.</td><td>nothing - it is the point everything else is measured from.</td></tr>"
      "<tr><td><b>H_min(class)</b><br><span class='m'>the class entropy reference</span></td><td>a <b>constant in the denominator</b>, in bits: the entropy level below which that architecture cannot hold the pattern that makes it that kind of cell. It is what makes A dimensionless.</td><td>it is <i>not a mark on the A axis at all</i>. It is the unit the axis is drawn in.</td><td>a reading that falls well below 1.00 is the suppressed / inverted direction - the pattern is over-ordered or erased rather than loosened (post-chemotherapy and immunosuppressed samples sit near 0.90).</td></tr>"
      "<tr><td><b>1/H_min</b><br><span class='m'>the ceiling - saturation</span></td><td>the arithmetic limit: the entropy of those addresses has reached its structural maximum, so the ratio cannot go higher. It differs by class <b>and</b> by substrate.</td><td>anywhere. For some class-and-substrate combinations it sits <b>above</b> breach; for others <b>below</b> it (see the chart).</td><td>the instrument has run out of scale - not the cell out of identity. Reported as AT_CEILING with the value, never as a number above it.</td></tr></table>"
      f"<p><b>And the consequence that matters.</b> Where the ceiling sits below the breach line at {BR}, that class <i>cannot reach breach on that substrate at all</i> - it saturates first. A system can be far above healthy, visibly spent, and still be structurally incapable of the no-return event on the surface you happen to be measuring. The author's cosmic gauge makes the point on a star: an isolated white dwarf reads above healthy and is ceiling-capped below breach, because it has no mechanism to gain mass; only a collapse-capable core reaches A = 1 in the gravitational sense, where the Chandrasekhar and TOV limits and the Schwarzschild condition all land. <b>Ceiling-capped is not the same as safe, and it is not the same as healthy - it means this substrate cannot tell you.</b> That is the argument for five substrates stated as a limit rather than a preference.</p>")
    H.append("<h3>The saturation wall chart - 8 classes x 5 substrates</h3>"
      f"<p>Each cell shows the class's frozen floor H_min on that substrate and, in brackets, the ceiling 1/H_min it implies; the flags are the author's own from the Issue 002 saturation wall chart: <b>SAT</b> = saturates below the breach line at {BR}, so that substrate cannot register a floor breach for that class; <b>TGT</b> = tight ceiling (below 1.15), where severe disease approaches saturation; unflagged = full headroom past breach. The floors come from the frozen 40-value table (methylation from the G-002 calibration, the four cfDNA/chromatin substrates from G-003b). Only the methylation column is lit today - see Coverage.</p>"
      "<table class='t'><tr><th>class</th>" + "".join(f"<th>{sub}{' (lit)' if sub=='methyl' else ''}</th>" for sub in R["sub_order"]) + "</tr>"
      + "".join("<tr><td>"+CLASS_LABEL.get(c,c)+"</td>"+"".join(
            (f"<td class='n' style='background:#2a1d1d'>{v:.4f} <span class='m'>[{1.0/v:.3f}]</span> <b>SAT</b></td>" if 1.0/v < BR
             else f"<td class='n' style='background:#2a2a1d'>{v:.4f} <span class='m'>[{1.0/v:.3f}]</span> <b>TGT</b></td>" if 1.0/v < 1.15
             else f"<td class='n'>{v:.4f} <span class='m'>[{1.0/v:.3f}]</span></td>")
            for v in R["hmin_table"][c])+"</tr>" for c in CLASSES if c in R["hmin_table"])
      + "</table>"
      f"<p class='m'><b>This chart is not new here.</b> It is the author's saturation wall chart (GAPE Issue 002, April 2026, p12), rebuilt from the engine's live floor table and reproducing it exactly: 40 rows, every ceiling equal to 1/H_min to three decimals, <b>15 SAT and 2 TGT</b> - the same cells, the same flags. Issue 002 frames it as the direct analogue of the Dennard scaling walls in semiconductor physics: the frequency, power and cost walls beyond which a technology stops improving. <b>15 of the 40 combinations saturate below breach.</b> Nucleosome occupancy caps 7 of the 8 classes (its floors are all near 0.98-0.99, so there is almost no range above healthy before saturation). Fuzziness caps the three stem and progenitor classes (pluripotent stem, adult stem, progenitor). WPS caps a different three - <b>terminal</b>, adult stem and progenitor - and notably does <i>not</i> cap pluripotent stem, whose WPS ceiling is {1.0/R['hmin_table']['stem_pluri'][3]:.3f}, just above the line. On methylation - the one lit column - <b>pluripotent stem is capped at {1.0/R['hmin_table']['stem_pluri'][0]:.3f}</b>, which is below breach and barely above the healthy band, so a pluripotent-stem breach cannot be read on methylation at all.</p>"
      f"<p class='m'><b>The reference clusters past breach, and where they come from.</b> The breakpoints file carries senescent cells at {R['tiers']['tier_system_v1_2']['reference_clusters_past_breach']['senescent_cells']['a_low']}-{R['tiers']['tier_system_v1_2']['reference_clusters_past_breach']['senescent_cells']['a_high']} and malignant cells at {R['tiers']['tier_system_v1_2']['reference_clusters_past_breach']['malignant_cells']['a_low']}-{R['tiers']['tier_system_v1_2']['reference_clusters_past_breach']['malignant_cells']['a_high']}. Issue 002 identifies the malignant end as <b>terminal-class</b> measurements: lower-grade glioma at A = 1.2846 and glioblastoma at A = 1.256, both from Ceccarelli 2016 (TCGA, n = 516 and n = 149), with terminal-class cancers showing the largest departures in the 28-cancer panel. That is consistent with the ceilings - the terminal class has the highest methylation ceiling of any class at {1.0/R['hmin_table']['terminal'][0]:.3f} - and it makes the arithmetic striking rather than loose: <b>the largest cancer signal in the panel sits within 0.01 of its own class ceiling</b>, which is exactly the regime where a second substrate stops being a refinement and becomes the only way to keep measuring. The cluster's upper bound of {R['tiers']['tier_system_v1_2']['reference_clusters_past_breach']['malignant_cells']['a_high']} is above even that ceiling, so it cannot be a methylation reading for any class; it is carried here as a corpus reference range, not re-measured on this chain, and the two open items it raises are listed on the Record tab.</p>"
      f"<p class='m'><b>And one refinement from Issue 002 worth stating precisely.</b> In its own words, the A = 1.00 line 'represents the architectural commitment point, not a mathematical floor' - and under the unfloored formula its healthy reference cells sit slightly <i>below</i> it, around A = 0.97, because the healthy beta produces an entropy slightly under the MCMC central estimate of H_min. On this chain healthy reads 1.00 rather than 0.97, and not because the formula changed: the laboratory zero and the age term are measured from healthy donors of that laboratory and that decade, which places the healthy population on 1.00 by construction. The two are the same instrument reading the same floor - one quotes the raw ratio, the other quotes it after the two measured offsets - and any number quoted from Issue 002 has to say which of the two it is.</p>")


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
<p>The honest reading of a high value is therefore: <i>a population of cells in this compartment has lost structural fidelity, and this compartment warrants a closer look.</i> A prompt, not a verdict.</p>
<h3>Would a well-differentiated benign growth read high?</h3>
<p>In general it should not, and that is the question that reveals what the instrument measures. A lipoma, a fibroid or a simple nevus is made of cells that are still recognisably their own type - they have lost growth control, not identity. Because A reads identity fidelity rather than cell number, a well-differentiated benign lesion is expected to read at or near the healthy reference. <b>The instrument is not tripped by the presence of extra cells; it responds to the loss of what makes a cell that kind of cell.</b> The corollary is that a reading is independent of a lesion's shape - a flat lesion produces the same reading as a raised one, because the instrument reads methylation entropy rather than looking for a shape.</p>
<p class='m'>The design record contains an ordered series consistent with this on colorectal and breast material - progressively higher readings from normal through benign neoplasm and dysplasia to established disease, crossing the line only at the end. Those values were produced on the pre-atlas surface and the earlier statistic, before the corrections on this report existed, so they are <b>history, not a result of this chain</b>. Re-measuring that series on the commissioned chain is a named next test, and until it has been run this report claims nothing about it.</p>
<h3>Two things the gauge is not</h3>
<ul><li><b>Not a clock.</b> Healthy A drifts upward with age and the drift is corrected for, but a whole lifetime of it is about the size of the scatter between two healthy people. One array cannot place a person on that curve, and this report never prints an age.</li>
<li><b>Not a fuel measurement.</b> The energy the cell spends holding its pattern is real, but A measures the pattern, not the fuel. A cell can be starving with a tidy pattern or well fed with a scrambled one.</li></ul>""")
    H.append(deepdive(R,"reading the gauge, the tiers and the healthy reference"))
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

       "<h3>The ceiling is informational, not gravitational - and that is why it travels</h3>",
       "<p>It is worth being precise about what this re-description does and does not claim. Consider how a black hole forms. The conventional "
       "account, the one every astronomer uses, is gravitational: mass concentrates until gravity is intense enough that nothing escapes, and a "
       "horizon forms. The framework here tells the same story in a different language: a region of space can only record so much information on its "
       "boundary before that boundary is full, and when the local accounting saturates that limit a new surface must form for the writing to "
       "continue. <b>These are not competing claims.</b> They predict the identical threshold at the identical radius; one is the other stated in the "
       "language of information rather than force. The gravity an astronomer measures is, in this reading, the stored record of the accounting rather "
       "than a separate cause.</p>",
       "<p><b>The reason it matters is that it travels where the gravitational language cannot follow.</b> A single cell is far too small for its mass "
       "to bend space measurably, so the gravitational account has nothing to say about it. The saturation account does. A cell maintains its identity "
       "by writing its ledger onto the surface available to it - not spacetime, but its own chromatin. A healthy cell sits comfortably below the limit "
       "of what that surface can hold. A cell drifting toward failure is one whose accounting is climbing toward saturation, writing more and more "
       "onto a surface with only so much room, until it can no longer maintain its identity on its existing terms. The ceiling is informational, not "
       "gravitational, which is exactly why the same law reads the star - where we used to call the ceiling gravity - and the cell, where there is no "
       "gravity to speak of but the ceiling is just as real.</p>",

       "<h3>Where the gauge sits, and what is open to inspection</h3>",
       "<p>A = 1.00 is the <b>healthy reference</b> - the reading a healthy cell of that class gives, in the middle of the normal band. H_min is the "
       "constant in the denominator, the unit the axis is drawn in; the ceiling at 1/H_min is saturation. The three are distinct and the <b>How to "
       "read</b> tab sets them out side by side.</p>",
       "<p>The eight class floors, the calibration code that fitted them and the leave-one-out bootstrap that cross-checks them are <b>public</b>, "
       "under an open licence. The floors and their provenance are on the <b>Healthy reference</b> tab; the code is linked from <b>Files</b>. Nothing "
       "in the metrology is withheld - a fixed zero that a reader cannot inspect is not a fixed zero.</p>",

       "<h3>Two names, and which is which</h3>",
       "<p><b>Astro-genetics</b> is the programme: cosmology's measurement tools pointed at the epigenome. <b>Physics of methylation: Landauer "
       "metrology</b> is the narrower field name for the metrology itself - the fixed zero, the single-sample absolute reading, the three-layer "
       "reference - and it is the title the engineering manual and the methods paper carry, because a methods paper should claim only what it "
       "measures. Both names are the author's; they describe different scopes of the same work.</p>"]
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
        H.append("<h3>Sealed procedures on the commissioned chain (PROC-*)</h3><table class='t'><tr><th>procedure</th><th>files</th><th>link</th></tr>")
        for d in sorted(os.listdir(pd_)):
            p=os.path.join(pd_,d)
            if os.path.isdir(p): H.append(f"<tr><td>{_e(d)}</td><td>{', '.join(sorted(os.listdir(p)))[:120]}</td><td><a href='{GH}/tree/{R['sha']}/Biological_Physics/Record/PROC_data/{d}' target='_blank'>open</a></td></tr>")
        H.append("</table>")
    H.append(f"<h3>Documents</h3><ul><li><a href='{GH}/tree/{R['sha']}/Biological_Physics/MethylPhys/manual' target='_blank'>GAPE Issue 003</a> - the engineering manual (RC1)</li><li><a href='{GH}/tree/{R['sha']}/Biological_Physics/MethylPhys/papers' target='_blank'>Landauer Metrology of the Methylome</a> - the methods paper (draft)</li><li><a href='{GH}/blob/{R['sha']}/Biological_Physics/MethylPhys/doors/RUNBOOK.md' target='_blank'>RUNBOOK</a> · <a href='{GH}/blob/{R['sha']}/Biological_Physics/MethylPhys/doors/CHAIN_COMMISSIONING.md' target='_blank'>CHAIN_COMMISSIONING</a> · <a href='{GH}/blob/{R['sha']}/Biological_Physics/HANDOFF.md' target='_blank'>HANDOFF</a></li></ul>")
    return "".join(H)

SPECIMENS=[("whole blood","450K / EPIC array","immune-dominant by construction; the only specimen with a commissioned band today","lit"),
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
       "<p>The instrument is one law read on a surface, and it generalises in two independent directions: <b>which specimen</b> the DNA came from, and <b>which physical substrate</b> is measured on it. Each combination needs its own three reference layers before a number can be reported - a pipeline map, a laboratory zero, and a healthy band - and each has its own floor and saturation limit. Today <b>one cell of this grid is lit: DNA methylation on whole blood.</b> Everything else prints its status rather than a number. This page exists so that no reader assumes otherwise, and so that a collaborator can see exactly what contributing one cell would take.</p>",
       "<h3>The five substrates</h3><table class='t'><tr><th>substrate</th><th>what it measures</th><th>published single-substrate discrimination (AUC)</th><th>specimen it requires</th><th>status here</th></tr>"]
    for k in R["sub_order"]:
        nm,d=SUBSTRATE_DESC[k]; req="any DNA" if k=="methyl" else ("plasma cfDNA" if k in ("wps","frag") else "plasma cfDNA (or chromatin assay)")
        st="<b>LIT</b> - commissioned on whole blood" if k=="methyl" else "<span class='pend'>RESERVED</span> - floor frozen, no pipeline map / laboratory zero / band"
        H.append(f"<tr><td><b>{k}</b></td><td>{nm} - {d}</td><td class='n'>{R['auc'].get(k,'-')}</td><td>{req}</td><td>{st}</td></tr>")
    H.append("</table><p class='m'>Each substrate has its own frozen floor for each of the eight classes (the 40-value table on the How-to-read tab), so a reading on one substrate is never compared against another's floor. The AUC column is the published single-substrate discrimination from the source literature, carried in the engine as a weight for combining substrates once more than one is lit; it is not a result of this chain.</p>")
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
    H.append("""<h3>What lighting one cell requires</h3><ol>
<li><b>A pipeline map.</b> One affine fit taking that specimen-and-substrate's values onto the scale the floors were calibrated on. Measured once per processing pipeline, on identity loci.</li>
<li><b>A laboratory zero.</b> 40 healthy arrays from the laboratory that will run the samples, of that specimen, read against the age curve. One constant. The RUNBOOK has the procedure.</li>
<li><b>A healthy band.</b> The middle 80 % of healthy readings for that specimen, ideally across more than one laboratory so that transfer can be tested leave-one-laboratory-out.</li></ol>
<p>None of the three can be borrowed from whole blood: the offset between two pipelines on the <i>same</i> specimen is already larger than the healthy band, which is the measured lesson that forced this design. The frozen floors, by contrast, do transfer - they are a property of the cell class, not of the specimen or the laboratory.</p>
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
             "in Issue 003 section 3b with the healthy distribution behind every threshold.</p>")
    if rec.get("stage0_deferred_qc"):
        H.append("<p class='m'><b>Deferred means not measured.</b> A deferred gate is neither a pass nor a "
                 "failure - it is a check this run could not make, named so that nobody reads its silence as "
                 "consent.</p>")
    return "".join(H)


def tab_troubleshooting(o, R):
    """What to do when the chain refuses - the same content as the SOP's step sections and Issue 003 s3b.

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
         "19</a>. The same material with the healthy distributions: <a href='" + MAN + "'>Issue 003, section "
         "3b</a>. The 732-array run these numbers come from: <a href='" + OUT + "'>PROC-STAGE0-02</a>.</p>",
         "<h3>Stage 0 refused the specimen</h3>",
         "<table><tr><th>what was printed</th><th>what it found</th><th>what to do</th></tr>"]
    for a, b, c in (
        ("QUARANTINE_INCOMPLETE_MANIFEST", "a required manifest field is missing or empty",
         "the seven fields are exact: sentrix_id, array_type, patient_id, intake_date, substrate, "
         "declared_sex, declared_chronological_age. Pass --sex and --age; a donor with no recorded age "
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
         "donor of unknown sex cannot clear this gate"),
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
    H.append("<h3>What healthy specimens measure, so you can tell a bad array from a bad configuration</h3>")
    H.append("<p class='m'>731 healthy whole-blood arrays, four Sentrix-chip years. A result far from these "
             "is the array; a result at zero or one is the configuration.</p>")
    H.append("<table><tr><th>check</th><th>threshold</th><th>healthy median</th><th>worst</th>"
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
    H.append("<table><tr><th>the reason printed</th><th>why</th><th>what to do</th></tr>")
    for a, b, c in (
        ("no laboratory zero &rarr; no placement, no tier",
         "this laboratory has never been measured, and between-laboratory offsets reach 0.046 in A - larger "
         "than most effects anyone wants to see",
         "commission the laboratory once: 40 healthy arrays of any age mix through the same Stage 1, then "
         "lab_zero.py. Panels under 40 are refused by design"),
        ("sky: no commissioned residual scale", "same cause, same panel", "same fix"),
        ("UNMAPPED", "no pipeline map was applied, so the values are not on the scale the floors were "
         "calibrated on",
         "pass a pipeline that exists in beta_scale_maps_v1.json; stage1_noob_450K is the right one for raw "
         "IDATs through this chain"),
        ("no band for this component yet", "that class has no measured healthy band",
         "nothing to fix - the fraction and A are still printed"),
        ("cellular age in years: not reported", "one array resolves age to about 50 years",
         "nothing to fix; the age-matched healthy reference is what the chain uses instead"),
        ("NOT ASSESSABLE, f below the presence floor",
         "that class is below its measured presence floor in this specimen",
         "nothing to fix - below its floor a class is not there")):
        H.append("<tr><td class='m'>" + a + "</td><td>" + b + "</td><td>" + c + "</td></tr>")
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
    H.append("<p>Three layers make a reading absolute and they are separate on purpose: the <b>floor</b> "
             "(H_min per class, calibrated by MCMC, never re-derived per pipeline), the <b>pipeline map</b> "
             "(the same healthy blood reads 0.737 on the reference scale and 0.815 through Stage 1 noob from "
             "raw IDATs - a within-pipeline comparison cancels that offset and never sees it, an absolute "
             "reading does not), and the <b>laboratory zero</b> (measured on 40 healthy arrays; four cohorts "
             "on one scale sit at 0, +0.024, -0.021 and -0.046 in A). A reading that skips either of the last "
             "two is not slightly wrong.</p>")
    H.append("<p><b>The check that catches it in one line:</b> run a handful of your own healthy specimens. "
             "Their median A&Prime; should land near 1.00 - that is how the map and the zero were verified in "
             "the first place. If they do not, one of the two is missing, and the chain will have printed "
             "which.</p>")
    H.append("<h3>How these were found, which is how to look for yours</h3>")
    H.append("<p class='m'>A gate that cannot read its input never fires - the header reader opened IDAT "
             "files raw while public downloads are gzipped, so the array-type check silently never ran on "
             "public data; it did not error, it returned 'unreadable' and everything continued. A value that "
             "fails to propagate looks like a value that is wrong - an identifier dropped between two steps "
             "made the next step refuse every array in a 732-array cohort, and the message blamed the data. A "
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
                 "a multi-gigabyte public cohort, or the test package. It is not a pass and it is not a failure; it is a statement that this particular "
                 "run could not exercise it. A reader who downloads the named data gets the result.</p>")
    # the second opinion, per sample
    so=o.get("second_opinion") or {}
    H.append("<h3>Second opinion on the composition - NILC beside the constrained solver</h3>")
    if not so.get("available"):
        H.append(f"<p class='pend'>NOT RUN for this sample - {_e(so.get('reason','not requested'))}</p>")
    else:
        ok=so["agreement"]=="AGREE"
        H.append(f"<p>Result: <b style='color:{'#3fa45b' if ok else '#d68910'}'>{so['agreement']}</b> against the bar <i>{_e(so['bar'])}</i>. "
                 f"Class-level L1 {so['L1_class']}, cell-level L1 {so['L1_cell']}. {_e(so['note'])}</p>"
                 "<table class='t'><tr><th>class</th><th>Walther (reported)</th><th>NILC (second opinion)</th><th>difference</th></tr>"
                 +"".join(f"<tr><td>{_e(CLASS_LABEL.get(c,c))}</td><td class='n'>{100*v['walther']:.1f} %</td><td class='n'>{100*v['nilc']:.1f} %</td>"
                          f"<td class='n'>{100*v['abs_diff']:.1f} pp</td></tr>" for c,v in so["by_class"].items())
                 +"</table>"
                 "<p class='m'>Why two solvers. Walther's constrained fit is conservative: a cell is placed only when the evidence forces it, which is why the "
                 "composition the report stands on is not inflated. NILC - the needlet internal linear combination, the component-separation method Planck uses - "
                 "is variance-weighted and deliberately sensitive to faint components. In July 2026 NILC was switched off for disagreeing with Walther on every "
                 "blood sample; PROC-NILC-01 later found the disagreement was the finding, not the fault: NILC was reporting that the atlas cannot separate the "
                 "blood classes, which PROC-SEP-03 then measured directly. It is back, as a second column and an agreement flag - never as the reported "
                 "composition. Cell-level disagreement <i>inside</i> one lineage is expected and is not scored; class-level disagreement is.</p>")
    H.append(deepdive(R,"the guards and the null suite"))
    H.append(_cmb_tool_table(o))   # the register, 2026-09-25
    return guard("".join(H),"Safeguards")


ROADMAP=[
 ("now","Find a serial cohort - two draws, same person","the difference map is the strongest design on the Sky tab and nothing held here has a repeat draw; the technical noise term is currently an upper bound inferred from cross-sectional data","EPIC-Italy and the Uppsala follow-up arms are the candidates; needs repeat-draw metadata, which the public extracts do not carry"),
 ("now","Merge the atlas's duplicate labels to one lineage per entry","the same lineage appears under several atlas entries whose per-cell readings differ by more than anything biological, purely by which reference panel defined the markers; until they are merged nobody can say 'which cell moved'","a merge rule plus a re-measured per-entry reference; it is the blocker on per-cell reporting"),
 ("now","Per-entry healthy bands wide enough to carry a tier word","20 per cent of atlas entries have a healthy spread wider than the gauge's NORMAL band, so their tier word is withheld and only the number is printed","the T-cell and NK entries are the widest; needs a per-entry band rather than the class band"),
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
 ("substrates","Urine, CSF, and the within-patient tissue / plasma / urine trio","each needs its own pipeline map, laboratory zero and healthy band before it can read; the trio would test whether one person's classes agree across specimens","Issue 003 section 7; see Coverage"),
 ("not now","Bispectrum and trispectrum; Minkowski functionals; isotropy and alignment tests","higher-order sky statistics; they need the power spectrum first and a reason to look","map rows 45, 53-60, 69"),
 ("not now","5mC / 5hmC as an E/B-mode separation; multi-omics cross-correlation","a genuinely deep parallel - two components of one field - but it needs oxidative-bisulphite data the chain has never seen","map rows 4, 49, 70"),
 ("does not translate","Rees-Sciama; Rayleigh scattering","recorded so nobody spends a week on them: the analogy breaks, and saying so is part of the map","map rows 63, 68"),
]

def tab_roadmap(R):
    H=["<h2>What is being considered next</h2>",
       "<p>Everything on this list is either named in the translation map between the microwave background and the methylome, or came out of a "
       "measurement made while building this chain. It is ordered by what has to happen first, not by how interesting it is. <b>Nothing here is a "
       "claim</b> - an item on this list has not been done, and several are listed precisely so that nobody spends a week rediscovering why they "
       "do not work.</p>",
       "<table class='t'><tr><th>when</th><th>item</th><th>why it matters</th><th>what it needs / where it comes from</th></tr>"]
    for when,item,why,needs in ROADMAP:
        H.append(f"<tr><td class='m'>{_e(when)}</td><td><b>{_e(item)}</b></td><td>{_e(why)}</td><td class='m'>{_e(needs)}</td></tr>")
    H.append("</table>")
    H.append("<p class='m'>The ordering rule is the author's, from the day an earlier version of this list was attempted all at once: "
             "<i>too ambitious too quick - these should have been worked on long after the bones were trusted.</i> The bones are the commissioning "
             "table; the Safeguards tab says which of them currently hold.</p>")
    H.append(deepdive(R,"any item on this list"))
    return guard("".join(H),"Roadmap")



def tab_inventory(R):
    """Every live file of the chain, enumerated by build_chain_inventory.py rather than hand-listed, so a file
    cannot be silently omitted. Added 2026-09-22 after the author asked whether the Chain tab lists everything:
    an audit found 20 load-bearing files linked nowhere, including the atlas itself."""
    inv=R.get("inv")
    if not inv: return guard("<h2>Chain inventory</h2><p class='pend'>chain_inventory_v1.json not found - run build_chain_inventory.py</p>","Files")
    m=inv["_meta"]; rows=inv["files"]
    ROLE=[("chain","In the chain","Executed on every run, in stage order. The conductor resolves each of these by name and fails loudly if one is missing."),
          ("reference","Reference and calibration data","The files the chain reads: the atlas, the floors, the identity loci, the maps, the bands, the age curve, the sky scales. These are what make a reading absolute."),
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
       "exclusivity, collinearity groups, pipeline maps, age curve, band, tiers, presence floors, sky mapping, "
       "directional panels, healthy reference, atlas provenance - plus the repository commit</td>"
       "<td>a reading is only meaningful against a stated instrument. When a layer is re-sealed these hashes change, "
       "so two findings can be compared only if their fingerprints match - and the record makes that checkable "
       "rather than assumed</td></tr>",
       "<tr><td><b>samples</b></td><td>one row per array: arm, age, every class reading, the departure, the sky "
       "summary, and the refusals that applied</td><td>the unit of this instrument is a per-sample absolute "
       "reading. Storing the samples means every summary above them can be recomputed, and an arm difference can "
       "never quietly become the result</td></tr>",
       "<tr><td><b>cells</b> and <b>groups</b></td><td>per atlas entry and per lineage group: where it was placed, "
       "the median reading by arm, the <b>direction</b> (above / below / within its own healthy range), the "
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
       "<td>the disease-matrix precursor. <b>Accumulating evidence, not a matching rule:</b> a matrix becomes "
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
         "atlas, the same band and the same age curve.</p>",
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
                 "cross-sample analysis should pass them (<code>--covariate cohort=NAME</code>), because "
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
    if o.get("lab_zero") is None:
        add("NO_LAB_ZERO", "WITHHELD", "Reading",
            "This laboratory has no commissioned zero, so no absolute reading is possible on any class.",
            "Commission the laboratory (PROC-MAHA-01) on its own healthy arrays, or read this run only for "
            "composition.")

    # 3. the laboratory's own false-alarm rate against the call that was made
    dep = o.get("departure") or {}
    fa = dep.get("lab_false_alarm_p95")
    for c, rec in (o.get("classes") or {}).items():
        pl = rec.get("placement")
        if pl and pl != "IN_BAND" and rec.get("reportable"):
            z = rec.get("z")
            bound = _lab_bound(fa)
            if bound and z is not None and abs(float(z)) < bound:
                add("CALL_INSIDE_LAB_NOISE", "CAUTION", "Reading",
                    "Class '%s' reads %s at z = %s, but this laboratory's own healthy arrays cross the "
                    "commissioned band %.1f %% of the time (nominal 5 %%), which puts its own 95 %% bound at "
                    "|z| = %.2f. This reading is inside that bound." % (c, pl, z, 100 * float(fa), bound),
                    "Do not read this as a departure. It is not distinguishable from this laboratory's own "
                    "healthy spread. A per-laboratory band would settle it (candidate procedure).")
            elif bound:
                add("CALL_OUTSIDE_LAB_NOISE", "NOTE", "Reading",
                    "Class '%s' reads %s at z = %s, outside this laboratory's own bound of |z| = %.2f "
                    "(its false-alarm rate is %.1f %%)." % (c, pl, z, bound, 100 * float(fa)),
                    "The call survives the laboratory's own noise. It is still a single array.")
    if fa is not None and float(fa) > 0.06:
        add("LAB_FALSE_ALARM_HIGH", "CAUTION", "Departure",
            "This laboratory's measured false-alarm rate is %.1f %% at p95, against a nominal 5 %%."
            % (100 * float(fa)),
            "Every departure from this laboratory is judged against a band its own healthy arrays cross "
            "more often than they should. Weigh accordingly.")

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
    if not tm.get("available"):
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
    ca = o.get("cellular_age") or {}
    if ca and not ca.get("reportable"):
        add("NO_CELLULAR_AGE", "WITHHELD", "Reading",
            "Cellular age in years is not reported: %s" % (ca.get("reason") or "single-array resolution"),
            "The age-matched healthy reference is what the chain uses instead.")

    F.sort(key=lambda x: (SEVERITY.get(x["severity"], 9), x["code"]))
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
            "<p class='m'>A rule that fails means a document has drifted from the tree - the SOP, the Issue "
            "003 manual, the commissioning register or the reviewer manifest no longer describes the code "
            "that produced this reading.</p>"
            % ("<b class='ok'>Every document was current.</b>" if ok else
               "<b class='warn'>At least one document had drifted.</b>",
               html.escape(str(st.get("when") or "?")), html.escape(str(st.get("commit") or "?")),
               len(st.get("rules", [])),
               "" if ok else "The failures are listed below.", rows))


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
    for n,d in [("release_check.py","all guards in one command; the Safeguards tab reads its output"),
                ("test_gauge_switch.py","the commissioned identity gauge reproduces on the cached commissioning arrays"),
                ("test_tiers.py","every tier boundary in tier_breakpoints.json, both sides, through the one tier function"),
                ("test_patient_sky.py","the sky stage: deterministic mapping, presence floors, refusal without a laboratory scale"),
                ("test_lab_zero.py","recovers a synthetic laboratory offset; refuses panels under 40 arrays"),
                ("PROC_ANCHOR_01.py","the sealed foundation-cohort anchors reproduce from raw GEO betas"),
                ("PROC_FORMULA_01.py","the A-score formula self-test, including the form the module refuses"),
                ("PROC_DECON_01.py","the composition solver against its answer key"),
                ("PROC_SEP_03.py","atlas separability by class - the measurement behind the blood caveat"),
                ("PROC_BIDIR_01.py","the directional detector, including re-extraction from the raw 5.1 GB GEO matrix"),
                ("finding_check.py","the protocol gate: every finding registered, every door taught, no unqualified detection claim")]:
        if n in R["files"]: H.append(f"<tr><td>{L(n)}</td><td>{_e(d)}</td></tr>")
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
       "<tr><td>declared age</td><td>healthy A rises about 0.47 milli-A per year; the decade term is subtracted before placement</td>"
       "<td>the age term cannot be removed and the class reading is not placed</td></tr>"
       "<tr><td>specimen</td><td>presence floors and the healthy band are specimen-specific</td><td>the run is refused - the chain will not guess</td></tr>"
       "<tr><td>laboratory identity</td><td>each laboratory has its own measured zero and residual scale</td>"
       "<td>lab_zero reads UNSET and every class prints NOT REPORTABLE with the reason</td></tr>"
       "<tr><td>pipeline name</td><td>a beta from a different normalisation sits on a different scale (LESSON-SCALE-01)</td>"
       "<td>the conductor refuses an unmapped reading rather than placing it</td></tr></table>")
    H.append("<h3>5. Commissioning your own laboratory</h3>"
       "<p>A laboratory the chain has not seen prints NOT REPORTABLE until its zero and sky scale are measured: <b>40 healthy arrays of that "
       "laboratory</b>, any age mix, through Stage 1, then the lab-zero procedure. That is the whole requirement, and it is a one-off per "
       "laboratory-and-pipeline, not per patient.</p><table class='t'><tr><th>file</th><th>what it is</th></tr>")
    for n,d in [("RUNBOOK.md","how to run the chain and how to commission a laboratory, step by step"),
                ("lab_zero.py","computes the zero from a 40-array healthy panel; refuses smaller panels"),
                ("CHAIN_COMMISSIONING.md","which stage is commissioned, by which sealed procedure, and what is still open"),
                ("HANDOFF.md","the state of the work, for the next reader"),
                ("reference_age_curve_v1.json","the four-laboratory age curve the decade term comes from"),
                ("identity_band_v3.json","the healthy band, pooled and per decade, with each laboratory's zero"),
                ("beta_scale_maps_v1.json","the pipeline maps"),
                ("iamatlas_gauge_identity_loci_v1_0.json","the identity loci and the eight frozen class floors"),
                ("percell_reference_v0.json","the per-entry healthy reference (exploration, unsealed)"),
                ("IAMAtlasREBUILD.csv.xz","the atlas: 483,092 CpGs x 115 cell types, posterior mean, SD and interval"),
                ("IAMAtlasREBUILD_provenance.json","how the atlas was built"),
                ("EPIC_plus_HM450_combined_manifest_normalized.csv","the array manifest with chromosome and position")]:
        if n in R["files"]: H.append(f"<tr><td>{L(n)}</td><td>{_e(d)}</td></tr>")
    H.append("</table>")
    H.append("<h3>6. Cohort mode</h3><p>For a validation run, every sample is read absolutely as above and the report then adds the distribution of "
       "readings by arm beside the sealed pre-registration bars. The protocol - seal before you run, register the finding, close it in code, teach "
       "every door, rebuild, read, push with copies - is in the RUNBOOK, and the gate that enforces it is <code>finding_check.py</code>.</p>")
    H.append(f"<p class='m'>Repository commit for this page: <code>{_e(R['sha'])}</code>. Every link above resolves at that commit, so a file that has "
       f"changed since will not silently substitute itself.</p>")
    H.append(deepdive(R,"running the chain"))
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
REFERENCE_TABS = ("howto", "story", "physics", "findings", "trouble", "reference", "roadmap", "coverage",
                  "record")
REFERENCE_BANNER = ("<p class='m' style='border-left:3px solid #bbb;padding-left:8px'>Reference material - "
                    "this tab is the same in every report and carries no measurement of this specimen.</p>")

TABS=[  # id, label, in the CLINICIAN print set, audience ("c" = both, "r" = researcher only)
 ("reading","Reading",True,"c"),("howto","How to read",True,"c"),("cells","Every cell",True,"c"),
 ("departure","Departure",True,"c"),("sky","Sky",True,"c"),("physics","Physics",False,"c"),("story","Story",False,"c"),
 ("reference","Healthy reference",False,"r"),("coverage","Coverage",False,"r"),("flags","Red flags",True,"c"),("safeguards","Safeguards",False,"r"),("trouble","Troubleshooting",False,"r"),
 ("integrity","Integrity",False,"r"),("chain","Chain",False,"r"),("files","Files",False,"r"),("findings","Findings",False,"r"),("roadmap","Roadmap",False,"r"),
 ("record","Record",False,"r"),("run","Run",False,"r")]

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
    sec={"reading":tab_reading(o,R,sample_id),"cells":tab_cells(o,R,percell_ref if percell_ref is not None else R.get("percell")),"departure":tab_departure(o,R),"sky":tab_sky(o,R,sample_id,wd),
         "reference":tab_reference(R,percell_status),"integrity":tab_integrity(o,R,refusals_from(o)),"chain":tab_chain(R),"files":tab_inventory(R),"findings":tab_findings(R),"physics":tab_physics(R),"howto":tab_howto(R),"coverage":tab_coverage(R),"safeguards":tab_safeguards(o,R),"flags":tab_redflags(o,R),"trouble":tab_troubleshooting(o,R),"roadmap":tab_roadmap(R),"story":tab_story(R),"record":tab_record(R),"run":tab_run(o,R)}
    for _rt in REFERENCE_TABS:
        if _rt in sec:
            sec[_rt] = REFERENCE_BANNER + sec[_rt]
    imm=o["classes"].get("immune",{}); head=(f"immune A'' {imm.get('A_abs')} · {imm.get('placement')} · {imm.get('tier')}" if imm.get("reportable") else "class gauge not reportable on this sample")
    nav="".join(f"<button class='{'resr' if a=='r' else ''}' data-t='{i}' onclick=\"tab('{i}')\">{n}</button>" for i,n,_,a in TABS)
    _rprint={"reading","howto","cells","departure","sky","reference","safeguards","trouble","integrity","chain","files","coverage"}
    body="".join(f"<section class='tab{' print' if p else ''}{' printr' if i in _rprint else ''}"
                 f"{' resr' if a=='r' else ''}' id='{i}'>{sec[i]}</section>" for i,n,p,a in TABS)
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
