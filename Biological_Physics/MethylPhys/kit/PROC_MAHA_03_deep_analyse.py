#!/usr/bin/env python3
"""PROC-MAHA-03 deep arm analysis: reads whatever chips the calibration has completed.

Safe to run while the calibration continues: it writes its cache by atomic rename, so a reader never sees a
half-written file. Only chips with at least 9 arrays are used, so B1 is always a full-depth test.
"""
import json, glob, math, random, collections, gzip, re, os, sys
import numpy as np
CH="Biological_Physics/MethylPhys/chain"
ident=json.load(open(glob.glob(CH+"/**/iamatlas_gauge_identity_loci_v1_0.json",recursive=True)[0]))["immune"]
HMIN=float(ident["H_min"])
band=json.load(open(glob.glob(CH+"/**/identity_band_v3.json",recursive=True)[0]))
P=band["pooled"]; SB=(P["p90"]-P["p10"])/(2*1.2816)
coh=band["_meta"]["cohorts"]
if isinstance(coh,str):
    import ast; coh=ast.literal_eval(coh)
ZLAB={k.split("_")[0]:v["z_lab_full_cohort"] for k,v in coh.items()}["GSE87571"]
CURVE={int(k):v for k,v in json.load(open(glob.glob(CH+"/**/reference_age_curve_v1.json",recursive=True)[0]))["curve"].items()}
MAP=json.load(open(glob.glob(CH+"/**/beta_scale_maps_v1.json",recursive=True)[0]))["maps"]["stage1_noob_450K"]
SLOPE,INTER=MAP["slope"],MAP["intercept"]
THR=1.959964; random.seed(5011); np.random.seed(5011)
def H(b):
    b=min(max(b,1e-12),1-1e-12); return -b*math.log2(b)-(1-b)*math.log2(1-b)
d=json.load(open("results/percell/stage1_betamean_GSE87571.json"))
with gzip.open("geo/GSE87571_series_matrix.txt.gz","rt",errors="replace") as f:
    gsms=None; ages={}; chips=None
    for line in f:
        if line.startswith("!Sample_geo_accession"): gsms=re.findall(r'"([^"]+)"',line)
        elif line.startswith("!Sample_characteristics_ch1"):
            vals=re.findall(r'"([^"]*)"',line)
            if vals and sum(1 for v in vals[:5] if re.search(r"\bage\b",v,re.I))>=2:
                for i,v in enumerate(vals):
                    m=re.search(r"(\d{1,3})(?:\.\d+)?\s*$",v.strip())
                    if m: ages[i]=int(m.group(1))
        elif line.startswith("!Sample_supplementary_file") and "_R0" in line:
            got=[re.search(r"(\d{9,12})_(R0\dC0\d)",v) for v in re.findall(r'"([^"]+)"',line)]
            if any(got): chips=[m.group(1) if m else None for m in got]
        elif line.startswith("!series_matrix_table_begin"): break
age_of={g:ages[i] for i,g in enumerate(gsms) if i in ages}; chip_of={g:chips[i] for i,g in enumerate(gsms) if chips[i]}
byc=collections.defaultdict(list)
for g,b in d.items():
    if g not in age_of or g not in chip_of: continue
    A=H((b-INTER)/SLOPE)/HMIN
    dec=max(min(age_of[g]//10*10,max(CURVE)),min(CURVE))
    byc[chip_of[g]].append(A-CURVE[dec]-ZLAB)
byc={k:v for k,v in byc.items() if len(v)>=9}
v=np.array([x for l in byc.values() for x in l]); sizes=sorted(len(l) for l in byc.values())
print(f"complete chips: {len(byc)} | arrays: {len(v)} | per chip {sizes[0]}-{sizes[-1]} (median {sizes[len(sizes)//2]})")
print(f"median A_abs {np.median(v):.4f}  sd {v.std(ddof=1):.4f}  tail p95 {float((np.abs((v-1)/SB)>THR).mean()):.4f} "
      f"(published full-cohort 0.0561)")
def icc(groups,n_perm=2000):
    groups=[np.array(g) for g in groups]; allv=np.concatenate(groups); grand=allv.mean(); k=len(groups); n=len(allv)
    msb=sum(len(g)*(g.mean()-grand)**2 for g in groups)/(k-1)
    msw=sum(((g-g.mean())**2).sum() for g in groups)/(n-k)
    var_b=max((msb-msw)/(n/k),0.0); F=msb/msw; worse=0
    for _ in range(n_perm):
        sh=np.random.permutation(allv); i=0; gs=[]
        for g in groups: gs.append(sh[i:i+len(g)]); i+=len(g)
        gb=sum(len(g)*(g.mean()-grand)**2 for g in gs)/(k-1)
        gw=sum(((g-g.mean())**2).sum() for g in gs)/(n-k)
        if gb/gw>=F: worse+=1
    return {"chips":k,"arrays":int(n),"icc":round(var_b/(var_b+msw),4),"sd_between":round(math.sqrt(var_b),5),
            "sd_within":round(math.sqrt(msw),5),"F":round(F,3),"p_perm":round((worse+1)/(n_perm+1),5)}
b1=icc(list(byc.values()))
print(f"\n=== B1 ===\n  ICC {b1['icc']}  sd_between {b1['sd_between']}  sd_within {b1['sd_within']}  F {b1['F']}  "
      f"p_perm {b1['p_perm']}  ({b1['chips']} chips, {b1['arrays']} arrays)")
print(f"  bar p < 0.01: {'MET for this cohort' if b1['p_perm']<0.01 else 'not met'}")
print(f"  for comparison, the 80-array panel gave ICC 0.193, p 0.0975 on 28 chips of ~2 arrays")
def sig(x):
    x=np.array(x); return float((np.percentile(x,90)-np.percentile(x,10))/(2*1.2816))
raw=[x for l in byc.values() for x in l]
print(f"\n=== B2/B3 k-sweep, held-out references only (band sigma {SB:.5f}) ===")
print(f"  uncorrected: tail {float((np.abs((np.array(raw)-1)/SB)>THR).mean()):.4f}  own sigma {sig(raw):.5f}")
sweep={}
for k in (1,2,3,5):
    corr=[]
    for ch,l in byc.items():
        if len(l)<k+1: continue
        arr=list(l)
        for i in range(len(arr)):
            others=arr[:i]+arr[i+1:]
            corr.append(arr[i]-(float(np.mean(random.sample(others,k)))-1.0))
    if len(corr)<30: sweep[k]=None; print(f"  k={k}: not assessable"); continue
    s_re=sig(corr); t_band=float((np.abs((np.array(corr)-1)/SB)>THR).mean()); t_re=float((np.abs((np.array(corr)-1)/s_re)>THR).mean())
    sweep[k]={"n":len(corr),"tail_band_sigma":round(t_band,4),"tail_rederived":round(t_re,4),
              "sigma_rederived":round(s_re,5),"meets_bar":bool(t_re<=0.05)}
    print(f"  k={k}: {len(corr):>4} arrays  tail(band sigma) {t_band:.4f}  tail(re-derived) {t_re:.4f}  "
          f"sigma {s_re:.5f}  bar {'MET' if t_re<=0.05 else 'not met'}")
json.dump({"_meta":{"procedure":"PROC-MAHA-03 deep arm","cohort":"GSE87571","chips_complete":len(byc),
                    "arrays":len(v),"scale":"Stage 1 noob from IDATs with the scale map applied - commissioned scale",
                    "interim": len(byc)<62},
           "B1":b1,"B2_B3":{f"k={k}":s for k,s in sweep.items()}},
          open("handoff/maha03_deep_interim.json","w"),indent=1)
print("\nwrote handoff/maha03_deep_interim.json")
