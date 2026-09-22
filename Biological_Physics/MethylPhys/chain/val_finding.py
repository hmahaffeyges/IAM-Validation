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
                   specimen="whole blood", prereg="Record/PROC_data/VAL-131/PREREG.md")
    for gsm, bundle, arm, age in runs: f.add_sample(gsm, bundle, arm=arm, age=age)
    f.add_bar("B1", "median A'' in the case arm >= 1.01", threshold=1.01, measured=1.037, passed=True)
    f.write()          # -> Record/VAL_FINDINGS/VAL-131_finding.json, sealed with its own sha256
"""
import os, json, hashlib, time, statistics as st


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

HERE=os.path.dirname(os.path.abspath(__file__)); BIO=_bio_root()
SCHEMA="val_finding_v1_1"
# v1.1 (2026-09-22) adds the observation types the June disease cards and the straw-man wall recorded, so that a
# finding captures everything a matching layer would later need. What was read to decide these fields, and what
# each one is for, is in the block comment below - written after reading IAM_Disease_Wall_CROWN_JEWEL_v1_12.html,
# ad-immune_card_v3_1.json, breast-epic_card_v3_1.json, immune-atlas_card_v2_0.json and the residual /
# bimodality / PCA maps beside them. NOTHING numeric was imported from those files: their effect sizes are
# case-versus-control Cohen's d on the pre-atlas surface, which is the statistic this chain does not use. What
# was taken is the VOCABULARY OF OBSERVATIONS - the kinds of thing worth recording.
#
#   window              The wall's rows are disease x PHASE x substrate (">10 yr pre-dx", "5-10 yr", "0-2 yr",
#                       "at dx"), and its headline claim is a trajectory across those windows. Without a window
#                       field, findings from different distances to diagnosis pool into one number and the
#                       trajectory - the most interesting thing in the wall - cannot be reconstructed.
#   direction at class  The AD card's specificity arm is the sharpest thing in the whole set: on the SAME
#                       Mahalanobis metric, Alzheimer's departed outward, PSP/CBD departed INWARD (their words:
#                       "BELOW_NORMAL architectural compaction direction"), and FTD sat between. Direction, not
#                       magnitude, separated three conditions. This chain already computes placement above and
#                       below the band; a finding now records it explicitly per class so that contrast survives.
#   compartments        The wall splits immune into lymphoid and myeloid because they "move in opposite
#                       directions near diagnosis". A single pooled immune number averages that away. The
#                       compartment roll-up and an explicit opposition flag make it visible.
#   residual map        Every card carried one: a per-CpG map with cross-cohort concordance (their columns:
#                       cpg, d_<cohort> per cohort, concordant_strong, mean_abs_d, CHR, MAPINFO). Ours is the
#                       same object on an absolute footing - per-CpG mean residual z against the healthy
#                       reference per arm, plus agreement between cohorts where more than one is present.
#   bimodality          Their breast map decomposed each CpG into bc_hc, bc_case, delta_bc, mean and sd beta,
#                       delta_var, bimodal_in_hc, loss_of_bimodality. A mean shift and a change in bimodality
#                       are different events: a locus can hold its mean while splitting into two populations.
#                       Worth recording separately, and cheap once the residual z per sample is in hand.
#   coverage            Cards refused to match below 80 % coverage of the residual map (INSUFFICIENT_COVERAGE).
#                       A finding records the coverage it achieved so a later comparison can apply the same bar.
#   covariates          The immune-atlas card carries age, smoking and sex foreground layers - and discloses
#                       that smoking subtraction was NOT applied at beta level in v1.0. Smoking is a large blood
#                       methylation effect; a finding that does not say whether it was handled invites a
#                       confounded comparison. Recorded as available / handled / NOT handled, honestly.
#   specificity arm     The AD card's three-way differential only means anything because the contrast conditions
#                       were run on the same instrument. A finding names the other conditions measured on its
#                       own instrument fingerprint, so a contrast can be checked rather than assumed.
#   cell_of_origin      The wall marks it with a gold ring, as a DECLARED property of the condition, not an
#                       inference from the data. Recorded as declared, with its source.
#   conjunction rule    Both cards state it: "card never fires on one tile" - a route required at least two
#                       independent signals. Carried forward as a property of the record: no single row is a
#                       finding on its own.
#   honest_limitations  Both cards end with one. A finding without it is a press release.

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
    def __init__(self, val_id, title, condition=None, specimen=None, prereg=None, operator=None, notes=None,
                 window=None, cell_of_origin=None, cell_of_origin_source=None):
        self.rec={"schema":SCHEMA,"val_id":val_id,"title":title,"condition":condition,"specimen":specimen,
                  "date":time.strftime("%Y-%m-%d"),"operator":operator,"notes":notes,
                  "prereg":{"path":prereg,"sha256":_sha(os.path.join(BIO,prereg) if prereg and not os.path.isabs(prereg) else prereg)},
                  "instrument":instrument_fingerprint(),"cohort":{},"samples":[],"bars":[],
                  "not_assessable":[],"warnings":[],
                  "window":window,                      # phase relative to diagnosis, e.g. "long_pre_dx_gt_10yr", "at_dx", None for a cross-sectional run
                  "cell_of_origin":{"declared":cell_of_origin,"source":cell_of_origin_source,
                      "_note":"DECLARED from the clinical description of the condition, never inferred from these data"},
                  "covariates":{"_note":"what could confound this comparison, and whether the chain handled it. "
                      "An unhandled covariate is disclosed, not omitted - smoking in particular is a large blood "
                      "methylation effect and this chain does not subtract it."},
                  "specificity_arm":[], "honest_limitations":[],
                  "conjunction_rule":"No single row in this record is a finding on its own. A claim requires at "
                      "least two independent signals - e.g. a class-level placement AND a per-cell direction with "
                      "its prevalence, or agreement between two cohorts on the residual map."}
        self._z={}        # arm -> list of per-CpG residual z vectors; kept in memory, summarised on write, never serialised raw
        self._cpgs=None
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
        # The sky residual arrives as a pandas Series indexed by CpG id (not a bare array - an isinstance check
        # for list/ndarray silently skipped it on the first version). Align on the index so samples with
        # different probe coverage still stack.
        sk=(bundle.get("patient_sky") or {}).get("_sky") or {}
        z=sk.get("z")
        if z is not None:
            try:
                if hasattr(z,"index"):
                    self._z.setdefault(arm,[]).append(z.astype("float32"))
                else:
                    import pandas as _pd
                    ids=sk.get("cpgs") or sk.get("cpg_ids")
                    self._z.setdefault(arm,[]).append(_pd.Series(z,index=ids).astype("float32"))
            except Exception as e: self.rec["warnings"].append(f"residual z not captured for {sample_id}: {e!r}"[:200])
        self.rec["samples"].append({"sample":sample_id,"arm":arm,"age":age,"sex":sex,"window":self.rec.get("window"),
            "classes":cls,"cells":cells,
            "departure":{"distance":dep.get("distance"),"axes":dep.get("n_axes"),
                         "lab_false_alarm_p95":dep.get("lab_false_alarm_p95")},
            "sky":{"available":sky.get("available"),
                   "beyond_2sigma":{k:(v.get("frac_beyond_2") if isinstance(v,dict) else None)
                                    for k,v in (sky.get("by_class") or {}).items()}},
            "refusals":bundle.get("refusals") or []})
        return self
    def set_covariates(self, **kw):
        """e.g. age="handled - decade term subtracted (reference_age_curve_v1)", smoking="NOT handled - no beta-level
        subtraction in this chain; cohort smoking distribution unknown", sex="recorded, not adjusted"."""
        self.rec["covariates"].update(kw); return self
    def add_specificity_arm(self, condition, val_id=None, note=None):
        """Another condition measured on THIS instrument fingerprint, against which this one can be contrasted."""
        self.rec["specificity_arm"].append({"condition":condition,"val_id":val_id,"note":note}); return self
    def add_limitation(self, text): self.rec["honest_limitations"].append(text); return self
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
        # ---- compartments: the wall splits immune into lymphoid and myeloid because they can move in opposite
        # directions. Membership is DECLARED by name match (listed, so it is auditable) - the atlas carries no
        # adaptive/innate annotation. Entries that match neither are reported as unassigned rather than dropped.
        LYMPH=("cd4","cd8","tcell","t-cell","bcell","b-cell","treg","nk","plasma","bmem","bnv","lymph","ncd4","mcd4","ncd8","mcd8","mt")
        MYEL=("neutro","eosino","baso","granulo","mono","macroph","microglia","kupffer","dendritic","gmp","cmp","mep","myelo")
        comp={"_membership_rule":"declared by name match on the atlas entry; both member lists are printed here",
              "lymphoid":{"members":[],"by_arm":{}},"myeloid":{"members":[],"by_arm":{}},"unassigned_immune":[]}
        for cell,row in R["cells"].items():
            if (row.get("class") or "")!="immune": continue
            lc=cell.lower()
            if any(t in lc for t in LYMPH): comp["lymphoid"]["members"].append(cell)
            elif any(t in lc for t in MYEL): comp["myeloid"]["members"].append(cell)
            else: comp["unassigned_immune"].append(cell)
        for side in ("lymphoid","myeloid"):
            for a in arms:
                mags=[R["cells"][c]["by_arm"].get(a,{}).get("magnitude_in_healthy_spreads") for c in comp[side]["members"]]
                mags=[x for x in mags if x is not None]
                dirs=[R["cells"][c]["by_arm"].get(a,{}).get("direction") for c in comp[side]["members"]]
                comp[side]["by_arm"][a]={"n_measured":len(mags),
                    "median_magnitude_in_healthy_spreads":round(st.median(mags),3) if mags else None,
                    "directions":{d:dirs.count(d) for d in set(dirs) if d}}
        # the opposition flag the wall's headline rests on
        comp["opposition"]={}
        for a in arms:
            l=comp["lymphoid"]["by_arm"].get(a,{}).get("median_magnitude_in_healthy_spreads")
            m=comp["myeloid"]["by_arm"].get(a,{}).get("median_magnitude_in_healthy_spreads")
            # A guard, because on healthy arms this flag fires on noise: the first dry run reported opposite signs
            # in BOTH arms with the sides swapped, which is what sampling scatter looks like. The separation has to
            # clear the scatter of the member magnitudes themselves before it means anything, and the arm size is
            # printed beside it so a reader can judge.
            lm=[R["cells"][c]["by_arm"].get(a,{}).get("magnitude_in_healthy_spreads") for c in comp["lymphoid"]["members"]]
            mm=[R["cells"][c]["by_arm"].get(a,{}).get("magnitude_in_healthy_spreads") for c in comp["myeloid"]["members"]]
            lm=[x for x in lm if x is not None]; mm=[x for x in mm if x is not None]
            pooled=lm+mm
            spread=(st.quantiles(pooled,n=4)[2]-st.quantiles(pooled,n=4)[0]) if len(pooled)>=8 else None
            sep=(None if (l is None or m is None) else round(abs(l-m),3))
            n_arm=sum(1 for s2 in R["samples"] if s2["arm"]==a)
            comp["opposition"][a]={"lymphoid":l,"myeloid":m,"n_samples_in_arm":n_arm,
                "n_lymphoid_measured":len(lm),"n_myeloid_measured":len(mm),
                "opposite_sign":(None if (l is None or m is None) else (l>0)!=(m>0)),
                "separation_in_healthy_spreads":sep,
                "member_magnitude_iqr":(None if spread is None else round(spread,3)),
                "separation_clears_member_scatter":(None if (sep is None or spread is None) else bool(sep>spread)),
                "_reading_rule":("opposite_sign alone is not evidence - it fires on sampling scatter. Report an "
                    "opposition only when separation_clears_member_scatter is true AND the arm is large enough "
                    "that the median is stable; on a healthy arm it should fire in neither direction consistently.")}
        R["compartments"]=comp
        # ---- class-level DIRECTION of departure, the field that separated three conditions on one metric in the
        # AD card. Recorded per class per arm as the placement mix, plus the dominant direction where there is one.
        for c,per in R["classes"].items():
            for a,v in per.items():
                pl=v.get("placement") or {}
                ab=sum(n for k,n in pl.items() if "ABOVE" in (k or "").upper())
                be=sum(n for k,n in pl.items() if "BELOW" in (k or "").upper())
                inb=sum(n for k,n in pl.items() if "IN_BAND" in (k or "").upper())
                tot=ab+be+inb
                v["direction"]={"above":ab,"below":be,"in_band":inb,
                    "dominant":(None if not tot else "above" if ab/tot>0.5 else "below" if be/tot>0.5 else "in_band")}
        # ---- per-CpG residual map and bimodality, computed from the sky residual z rather than stubbed
        R["per_cpg"]={"_note":"per-CpG summaries. Ours is an ABSOLUTE map: mean residual z against the healthy "
            "reference per arm, not a case-minus-control effect size. Written beside this record as a CSV because "
            "it is one row per CpG; only the strongest rows are inlined here."}
        try:
            import numpy as np
            if self._z:
                import pandas as pd
                per={}
                for a,vecs in self._z.items():
                    df=pd.concat(vecs,axis=1,join="inner")             # CpGs x samples, aligned on the CpG index
                    self._cpgs=list(df.index); M=df.to_numpy(dtype="float32").T   # samples x CpGs
                    mu=np.nanmean(M,axis=0); sd=np.nanstd(M,axis=0)
                    # Sarle bimodality coefficient per CpG across the samples of this arm: (skew^2 + 1) / kurtosis
                    n=M.shape[0]
                    if n>=8:
                        c=M-mu; s2=np.nanmean(c**2,axis=0)+1e-12
                        sk=np.nanmean(c**3,axis=0)/s2**1.5; ku=np.nanmean(c**4,axis=0)/s2**2
                        bc=(sk**2+1.0)/np.maximum(ku,1e-9)
                    else: bc=np.full(mu.shape,np.nan)
                    per[a]={"mean_z":mu,"sd_z":sd,"bc":bc,"n":n}
                R["per_cpg"]["arms"]={a:{"n_samples":int(v["n"]),
                    "frac_abs_mean_z_gt_1":float(np.mean(np.abs(v["mean_z"])>1)),
                    "median_abs_mean_z":float(np.median(np.abs(v["mean_z"]))),
                    "bimodality_computed":bool(v["n"]>=8)} for a,v in per.items()}
                R["per_cpg"]["bimodality_note"]=("Sarle bimodality coefficient per CpG across the samples of an arm, "
                    "computed only where the arm has at least 8 samples. A change in bc with an unchanged mean is a "
                    "locus splitting into two populations - a different event from a mean shift, which is why the June "
                    "breast map carried bc_hc, bc_case and delta_bc as separate columns.")
                # inline the strongest rows only
                a0=sorted(per, key=lambda a:-per[a]["n"])[0]
                ids=self._cpgs; mu=per[a0]["mean_z"]; sd=per[a0]["sd_z"]
                def rows(idx):
                    return [{"cpg":(ids[i] if ids else int(i)),"arm":a0,
                             "mean_z":round(float(mu[i]),4),"sd_z":round(float(sd[i]),4),
                             "consistency":round(float(abs(mu[i])/(sd[i]+1e-6)),3),
                             "bc":(None if not np.isfinite(per[a0]["bc"][i]) else round(float(per[a0]["bc"][i]),4))}
                            for i in idx]
                # TWO rankings, because the first dry run showed why one is not enough: ranked by |mean z| alone the
                # top rows were mean 30.8 with sd 42.6 - high-variance probes, not consistent departures. The June
                # cards solved the same problem with a concordant_strong flag across cohorts. Here the second
                # ranking is |mean z| / sd across the samples of the arm: a departure most of the arm agrees on.
                R["per_cpg"]["top_200_by_abs_mean_z"]=rows(np.argsort(-np.abs(mu))[:200])
                # The consistency ranking needs enough samples for sd to mean anything. On the 3-sample dry run it
                # returned ratios of 3,800 because sd came out at 1e-4 - a division by an unestimated quantity, not
                # a consistent departure. Requires >= 8 samples in the arm, and sd is floored at a tenth of the
                # arm's median sd so a handful of freakishly tight probes cannot dominate the list.
                if per[a0]["n"]>=8:
                    fl=max(float(np.nanmedian(sd))*0.1,1e-4)
                    R["per_cpg"]["top_200_by_consistency"]=rows(np.argsort(-(np.abs(mu)/np.maximum(sd,fl)))[:200])
                    R["per_cpg"]["consistency_sd_floor"]=round(fl,6)
                else:
                    R["per_cpg"]["top_200_by_consistency"]=None
                    R["per_cpg"]["consistency_not_computed"]=(f"arm '{a0}' has {per[a0]['n']} samples; the "
                        "consistency ranking needs at least 8 for the per-CpG sd to be estimable")
                R["per_cpg"]["ranking_note"]=("top_200_by_abs_mean_z is the largest departures and is dominated by "
                    "unstable probes; top_200_by_consistency is |mean z| / sd across the arm and is the list to "
                    "carry into a candidate panel. With two or more cohorts, the intersection of the two "
                    "consistency lists is this chain's equivalent of the cards' concordant_strong flag.")
                self._per=per
            else:
                R["per_cpg"]["arms"]=None
                R["per_cpg"]["why_absent"]="no sky residual in the bundles - the laboratory has no commissioned scale, so there is no z to map"
        except Exception as e:
            R["per_cpg"]["error"]=repr(e)[:200]
        # ---- coverage, the gate the cards applied before matching
        cov=[s["classes"].get(c,{}) for s in R["samples"] for c in s["classes"]]
        R["coverage"]={"samples":len(R["samples"]),
            "samples_with_a_reportable_class":sum(1 for s in R["samples"] if any(v.get("reportable") for v in s["classes"].values())),
            "samples_with_a_sky":sum(1 for s in R["samples"] if (s.get("sky") or {}).get("available")),
            "_note":"the June cards refused to match below 80 % coverage of their residual map (INSUFFICIENT_COVERAGE). "
                    "Recorded here so a later comparison can apply the same bar rather than assume it."}
        # the disease-matrix precursor row
        R["matrix_evidence"]={"_note":("ACCUMULATING EVIDENCE, NOT A MATCHING RULE. One condition measured on one instrument. "
            "A matrix becomes possible only when several conditions have been measured on the SAME instrument fingerprint, "
            "and the chain does not read this block back to classify anything."),
            "condition":R["condition"],"specimen":R["specimen"],"window":R.get("window"),
            "instrument_commit":R["instrument"].get("_commit"),
            "cell_of_origin_declared":(R.get("cell_of_origin") or {}).get("declared"),
            "class_direction":{c:{a:(v.get("direction") or {}).get("dominant") for a,v in per.items()} for c,per in R["classes"].items()},
            "compartment_opposition":(R.get("compartments") or {}).get("opposition"),
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
        d=out_dir or os.path.join(BIO,"Record","VAL_FINDINGS"); os.makedirs(d,exist_ok=True)
        p=os.path.join(d,f"{self.rec['val_id']}_finding.json")
        body=json.dumps(self.rec,indent=1,default=str)
        self.rec["_self_sha256"]=hashlib.sha256(body.encode()).hexdigest()
        json.dump(self.rec,open(p,"w"),indent=1,default=str)
        print(f"{self.rec['val_id']}: {len(self.rec['samples'])} samples, {len(self.rec.get('cells',{}))} entries, "
              f"{len(self.rec['bars'])} bars -> {p.split('Biological_Physics/')[1]}")
        return p
