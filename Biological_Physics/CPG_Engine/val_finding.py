#!/usr/bin/env python3
"""val_finding.py - the structured record of one validation run. Schema v1, 2026-09-22.

WHY THIS EXISTS AND WHY NOW. The author asked, before the first VAL on the commissioned chain: capture everything
a researcher will want to know, and capture exactly what a future disease matrix would need - which cells are
doing what, in which direction, at what magnitude, for every condition tested. A record designed after the runs
is a record shaped by what happened to be convenient to save. This is designed before them, so every run is
comparable to every other and nothing has to be reconstructed later.

THE UNIT OF A FINDING IS A PER-SAMPLE ABSOLUTE READING, NOT A GROUP DIFFERENCE. The chain measures each sample
against a fixed reference; arms are reported as distributions of absolute readings, never as an effect size that
would make the reference redundant. A finding therefore stores the per-sample readings and summarises them, in
that order.

WHAT A FINDING RECORD CONTAINS
  identity      val id, title, condition, specimen, date, operator, the PREREG path and its sha256
  instrument    every reference layer with its sha256: the floors, the identity loci, the markers, the pipeline
                map, the laboratory zero, the age curve, the band, the tier breakpoints, the presence floors,
                the sky mapping and residual scale, the atlas. A finding that cannot name the instrument it was
                measured on is not reproducible, and a later re-seal changes these hashes - which is the point.
  cohort        accession, laboratory, pipeline, specimen, arms with n, age and sex distribution, and the
                selection rule as it was written in the PREREG
  samples       one row per array: arm, age, the class readings, the departure, the sky summary, and the
                refusals that applied. The raw material of every number above it.
  classes       per arm and class: n reportable, median A'' with its interquartile range, the placement
                distribution, and the tier distribution
  cells         per atlas entry AND per lineage group: where it was placed, in how many samples, the median
                reading by arm, the DIRECTION (above / below / within its own healthy range), the MAGNITUDE in
                units of that entry's healthy spread, the prevalence (what fraction of the arm departed), the
                panel exclusivity, and the claim level this permits - individual, group-only, or withheld
  sky           per arm: the fraction of the genome beyond |z| = 2, the classes that were assessable, and the
                pixels most consistently departed across the arm
  bars          every pre-registered bar with its threshold, its measured value, and PASS / FAIL AS SEALED
  not_assessable  what could not be read and the reason - the honest half of any result
  matrix_evidence  the disease-matrix precursor: for this condition and specimen, the per-group direction,
                magnitude and prevalence, tagged with the instrument fingerprint. ACCUMULATING EVIDENCE, NOT A
                MATCHING RULE: a matrix becomes possible when enough conditions have been measured on one
                instrument to compare them; until then this block is a row in a ledger, and the chain does not
                read it back to classify anything.

Usage from a VAL runner:
    import val_finding as VF
    f = VF.Finding("VAL-131", "Colorectal adenoma, whole blood", condition="colorectal adenoma",
                   specimen="whole blood", prereg="Testing_and_Code/PROC_data/VAL-131/PREREG.md")
    for gsm, bundle, arm, age in runs: f.add_sample(gsm, bundle, arm=arm, age=age)
    f.add_bar("B1", "median A'' in the case arm >= 1.01", threshold=1.01, measured=1.037, passed=True)
    f.write()          # -> Testing_and_Code/VAL_FINDINGS/VAL-131_finding.json, sealed with its own sha256
"""
import os, json, hashlib, time, statistics as st

HERE=os.path.dirname(os.path.abspath(__file__)); BIO=os.path.dirname(HERE)
SCHEMA="val_finding_v1"

def _find(name, root=None):
    for dp,_,fs in os.walk(root or BIO):
        if "RETIRED" in dp or "__pycache__" in dp: continue
        if name in fs: return os.path.join(dp,name)

def _sha(p):
    if not p or not os.path.exists(p): return None
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

# every reference layer a reading depends on. A finding names all of them, so a re-seal is visible as a hash change.
INSTRUMENT_FILES=["iamatlas_gauge_identity_loci_v1_0.json","iamatlas_celltype_markers_v0_2.json",
 "percell_exclusivity_v0.json","percell_reference_v0.json","iamatlas_collinearity_groups_v0_1.json",
 "beta_scale_maps_v1.json","reference_age_curve_v1.json","identity_band_v3.json","tier_breakpoints.json",
 "presence_floors_v1.json","iamatlas_cpg_to_healpix_nside128.npz","directional_panels_v1_0.json",
 "mahalanobis_healthy_reference_v2_0_age_matched_derived.json","IAMAtlasREBUILD_provenance.json"]

def instrument_fingerprint():
    out={}
    for n in INSTRUMENT_FILES:
        p=_find(n); out[n]={"path":p.split("Biological_Physics/")[1] if p else None,"sha256":_sha(p)}
    try:
        import subprocess
        out["_commit"]=subprocess.run(["git","-C",BIO,"rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
    except Exception: out["_commit"]=None
    return out

def _q(v):
    v=[x for x in v if x is not None]
    if not v: return None
    v=sorted(v)
    return {"n":len(v),"median":round(st.median(v),4),
            "p25":round(v[int(.25*(len(v)-1))],4),"p75":round(v[int(.75*(len(v)-1))],4),
            "min":round(v[0],4),"max":round(v[-1],4)}

class Finding:
    def __init__(self, val_id, title, condition=None, specimen=None, prereg=None, operator=None, notes=None):
        self.rec={"schema":SCHEMA,"val_id":val_id,"title":title,"condition":condition,"specimen":specimen,
                  "date":time.strftime("%Y-%m-%d"),"operator":operator,"notes":notes,
                  "prereg":{"path":prereg,"sha256":_sha(os.path.join(BIO,prereg) if prereg and not os.path.isabs(prereg) else prereg)},
                  "instrument":instrument_fingerprint(),"cohort":{},"samples":[],"bars":[],
                  "not_assessable":[],"warnings":[]}
    # ---- inputs ----
    def set_cohort(self, **kw):  self.rec["cohort"].update(kw); return self
    def add_bar(self, bar_id, statement, threshold=None, measured=None, passed=None, note=None):
        self.rec["bars"].append({"bar":bar_id,"statement":statement,"threshold":threshold,
                                 "measured":measured,"passed":passed,"note":note}); return self
    def add_sample(self, sample_id, bundle, arm="case", age=None, sex=None):
        """bundle is cpg_conductor.run_full's output for that sample, unmodified."""
        cls={c:{"A":r.get("A_abs"),"placement":r.get("placement"),"tier":r.get("tier"),
                "reportable":r.get("reportable",r.get("A_abs") is not None)}
             for c,r in (bundle.get("classes") or {}).items()}
        cells={k:{"A":v.get("A"),"fraction":v.get("fraction"),"placed":bool((v.get("fraction") or 0)>0),
                  "coverage":v.get("coverage")} for k,v in (bundle.get("cells_all") or {}).items()}
        dep=bundle.get("departure") or {}
        sky=bundle.get("patient_sky") or {}
        self.rec["samples"].append({"sample":sample_id,"arm":arm,"age":age,"sex":sex,
            "classes":cls,"cells":cells,
            "departure":{"distance":dep.get("distance"),"axes":dep.get("n_axes"),
                         "lab_false_alarm_p95":dep.get("lab_false_alarm_p95")},
            "sky":{"available":sky.get("available"),
                   "beyond_2sigma":{k:(v.get("frac_beyond_2") if isinstance(v,dict) else None)
                                    for k,v in (sky.get("by_class") or {}).items()}},
            "refusals":bundle.get("refusals") or []})
        return self
    def add_not_assessable(self, what, reason): self.rec["not_assessable"].append({"what":what,"reason":reason}); return self
    # ---- derived ----
    def _summarise(self):
        R=self.rec; arms=sorted({s["arm"] for s in R["samples"]})
        R["cohort"].setdefault("arms",{a:sum(1 for s in R["samples"] if s["arm"]==a) for a in arms})
        # per class per arm
        cl={}
        for a in arms:
            ss=[s for s in R["samples"] if s["arm"]==a]
            for c in {c for s in ss for c in s["classes"]}:
                vals=[s["classes"][c]["A"] for s in ss if s["classes"].get(c,{}).get("reportable")]
                pl=[s["classes"][c].get("placement") for s in ss if s["classes"].get(c,{}).get("reportable")]
                ti=[s["classes"][c].get("tier") for s in ss if s["classes"].get(c,{}).get("reportable")]
                cl.setdefault(c,{})[a]={"A":_q(vals),"n_reportable":len(vals),"n_arm":len(ss),
                    "placement":{k:pl.count(k) for k in set(pl) if k},"tier":{k:ti.count(k) for k in set(ti) if k}}
        R["classes"]=cl
        # per cell and per group: direction, magnitude, prevalence
        try:
            ref=json.load(open(_find("percell_reference_v0.json")))["entries"]
            exc=json.load(open(_find("percell_exclusivity_v0.json")))["entries"]
            cg=json.load(open(_find("iamatlas_collinearity_groups_v0_1.json")))
            g_of=cg["cell_to_group"]; gm={g:(v.get("members") if isinstance(v,dict) else v) for g,v in cg["groups"].items()}
        except Exception: ref={}; exc={}; g_of={}; gm={}
        cells={}
        for cell in {c for s in R["samples"] for c in s["cells"]}:
            e=ref.get(cell,{}).get("pooled") or {}
            lo,hi,mid=e.get("A_p10"),e.get("A_p90"),e.get("A_p50")
            spread=(hi-lo) if (lo is not None and hi is not None) else None
            ex=exc.get(cell,{}); gid=g_of.get(cell); mem=gm.get(gid) or []
            claim=("withheld_panel_shared" if ex.get("individual_claim_ok") is False
                   else "group_only" if len(mem)>1 else "individual")
            row={"class":ex.get("class"),"exclusivity":ex.get("exclusivity"),"group":gid,
                 "group_members":mem,"claim_level":claim,"healthy_p10":lo,"healthy_p50":mid,"healthy_p90":hi,
                 "healthy_spread":spread,"by_arm":{}}
            for a in arms:
                ss=[s for s in R["samples"] if s["arm"]==a]
                vals=[s["cells"][cell]["A"] for s in ss if s["cells"].get(cell,{}).get("A") is not None]
                placed=sum(1 for s in ss if s["cells"].get(cell,{}).get("placed"))
                q=_q(vals)
                if q and lo is not None and hi is not None:
                    above=sum(1 for v in vals if v>hi); below=sum(1 for v in vals if v<lo)
                    direction=("above" if above>below and above/len(vals)>0.5 else
                               "below" if below>above and below/len(vals)>0.5 else "within")
                    mag=round((q["median"]-mid)/spread,3) if (spread and mid is not None) else None
                    row["by_arm"][a]={"A":q,"n_placed":placed,"direction":direction,
                        "magnitude_in_healthy_spreads":mag,
                        "prevalence_above":round(above/len(vals),3),"prevalence_below":round(below/len(vals),3)}
                elif q: row["by_arm"][a]={"A":q,"n_placed":placed,"direction":None,
                        "magnitude_in_healthy_spreads":None,"note":"no healthy reference for this entry"}
            cells[cell]=row
        R["cells"]=cells
        # groups: aggregate the members of each multi-member group (the honest unit where the atlas cannot separate)
        groups={}
        for gid,mem in gm.items():
            ms=[m for m in mem if m in cells]
            if not ms: continue
            groups[gid]={"members":mem,"label":(cg["groups"][gid].get("label") if isinstance(cg["groups"][gid],dict) else gid),
                "low_confidence":(cg["groups"][gid].get("low_confidence") if isinstance(cg["groups"][gid],dict) else None),
                "by_arm":{}}
            for a in arms:
                mags=[cells[m]["by_arm"].get(a,{}).get("magnitude_in_healthy_spreads") for m in ms]
                mags=[x for x in mags if x is not None]
                dirs=[cells[m]["by_arm"].get(a,{}).get("direction") for m in ms]
                groups[gid]["by_arm"][a]={"members_measured":len(mags),
                    "median_magnitude_in_healthy_spreads":round(st.median(mags),3) if mags else None,
                    "directions":{d:dirs.count(d) for d in set(dirs) if d}}
        R["groups"]=groups
        # the disease-matrix precursor row
        R["matrix_evidence"]={"_note":("ACCUMULATING EVIDENCE, NOT A MATCHING RULE. One condition measured on one instrument. "
            "A matrix becomes possible only when several conditions have been measured on the SAME instrument fingerprint, "
            "and the chain does not read this block back to classify anything."),
            "condition":R["condition"],"specimen":R["specimen"],"instrument_commit":R["instrument"].get("_commit"),
            # Two conditions to carry a group as evidence, both necessary:
            #  (1) the finding names a condition - a healthy or technical run contributes NOTHING to a matrix, and
            #      the first version of this filter wrongly carried all 94 groups from a run with condition=None;
            #  (2) at least one arm shows a DEPARTURE (direction above or below), not merely a non-zero magnitude -
            #      every group has some non-zero magnitude, so that test admitted everything.
            "per_group":({} if not R["condition"] else
                {gid:{a:v for a,v in g["by_arm"].items()} for gid,g in groups.items()
                 if any(any(d in (v.get("directions") or {}) for d in ("above","below")) for v in g["by_arm"].values())}),
            "carried":("nothing - this finding names no condition" if not R["condition"] else "groups showing a departure in at least one arm")}
        # sky by arm
        R["sky"]={a:{"n":sum(1 for s in R["samples"] if s["arm"]==a),
                     "available":sum(1 for s in R["samples"] if s["arm"]==a and (s["sky"] or {}).get("available"))}
                  for a in arms}
    def write(self, out_dir=None):
        self._summarise()
        d=out_dir or os.path.join(BIO,"Testing_and_Code","VAL_FINDINGS"); os.makedirs(d,exist_ok=True)
        p=os.path.join(d,f"{self.rec['val_id']}_finding.json")
        body=json.dumps(self.rec,indent=1,default=str)
        self.rec["_self_sha256"]=hashlib.sha256(body.encode()).hexdigest()
        json.dump(self.rec,open(p,"w"),indent=1,default=str)
        print(f"{self.rec['val_id']}: {len(self.rec['samples'])} samples, {len(self.rec.get('cells',{}))} entries, "
              f"{len(self.rec['bars'])} bars -> {p.split('Biological_Physics/')[1]}")
        return p
