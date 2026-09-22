#!/usr/bin/env python3
"""Per-cell healthy reference (exploration, unsealed; 2026-09-22). For every one of the 115 atlas entries, on each laboratory's 40 healthy PANEL
arrays (Stage 1 noob beta, mapped to the reference scale): mean beta over that entry's discriminative markers -> H -> per-cell A = H/H_min(class)
(the sealed-anchor statistic) AND H_ref(entry) = median H on the panel, so that A'(entry) = H/H_ref reads 1.0 healthy by construction.
Writes Runtime Matrices/Percell_Reference/percell_reference_v0.json: per entry per lab {n, H_median, H_p10, H_p90, A_p10, A_p50, A_p90} and the pooled
four-lab values, plus a HELD-OUT check on the 40 TEST arrays per lab (fraction of test arrays inside the panel p10-p90; nominal 0.80).
Inputs: results/percell/stage1_betas_<GSE>.pkl.xz (+MANIFEST) for the four laboratories."""
import os, sys, json, lzma, pickle, math, hashlib, time, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ENGINE=os.path.dirname(HERE); BIO=os.path.dirname(ENGINE); WS=sys.argv[1] if len(sys.argv)>1 else os.getcwd()
def find(n):
    for dp,_,fs in os.walk(BIO):
        if "RETIRED" in dp: continue
        if n in fs: return os.path.join(dp,n)
    raise FileNotFoundError(n)
mk=json.load(open(find("iamatlas_celltype_markers_v0_2.json"))); c2c=json.load(open(find("IAMAtlasREBUILD_celltype_to_class.json")))
ident=json.load(open(find("iamatlas_gauge_identity_loci_v1_0.json"))); HM={c:v["H_min"] for c,v in ident.items() if isinstance(v,dict) and "H_min" in v}
mp=json.load(open(find("beta_scale_maps_v1.json")))["maps"]["stage1_noob_450K"]; slope,inter=float(mp["slope"]),float(mp["intercept"])
def H(b): b=np.clip(np.asarray(b,float),1e-12,1-1e-12); return -b*np.log2(b)-(1-b)*np.log2(1-b)
labs=["GSE87571","GSE42861","GSE111629","GSE125105"]; sel=json.load(open(os.path.join(WS,"handoff/percell_selection.json")))
out={"_meta":{"built":time.strftime("%Y-%m-%d %H:%M"),"status":"EXPLORATION - unsealed working reference","statistic":"per entry: mean_i H(beta_i) over its markers on the mapped scale (the sealed-anchor form, NOT H(beta_mean)); A = that / H_min(class); H_ref = median on the 40-array panel; A prime = value / H_ref","labs":{},"markers":os.path.basename(find("iamatlas_celltype_markers_v0_2.json")),"map":"NONE - per-cell surface is scored on raw Stage 1 betas (stage_a_cells), so the reference is too; per-laboratory by construction"},"entries":{}}
panelH={}; testH={}
for g in labs:
    p=os.path.join(WS,f"results/percell/stage1_betas_{g}.pkl.xz")
    if not os.path.exists(p): print(g,"not available yet"); continue
    df=pickle.load(lzma.open(p,"rb")); man=json.load(open(p.replace(".pkl.xz","_MANIFEST.json")))
    # NO pipeline map here: cpg_conductor.stage_a_cells scores the per-cell surface on the RAW calibrated betas (only the class
    # gauge takes the Stage 1s map). A reference built on mapped betas sits on a different scale than the reading it is compared
    # with - which manufactured spurious 'below range' calls on healthy arrays. The per-cell reference is per laboratory for
    # exactly this reason: the scale offset stays inside the laboratory. Found 2026-09-22.
    B=df
    out["_meta"]["labs"][g]={"n_arrays":df.shape[1],"n_loci":df.shape[0],"sha256_pickle":man["sha256_pickle"],"panel":man["panel"],"test":man["test"]}
    for cell,loci in mk["markers_by_celltype"].items():
        l=[x for x in loci if x in B.index]
        if len(l)<10: continue
        # mean_i H(beta_i) over the cell's markers - NEVER H(beta_mean); the scoring module asserts against the latter (LESSON-ASCORE-02)
        h=np.nanmean(H(B.loc[l].to_numpy()),axis=0); s=pd.Series(h,index=df.columns)   # nanmean: a single missing marker must not void the array (plain mean propagated NaN and silently dropped most arrays)
        panelH.setdefault(cell,{})[g]=s.reindex([x for x in man["panel"] if x in s.index]).dropna(); testH.setdefault(cell,{})[g]=s.reindex([x for x in man["test"] if x in s.index]).dropna()
    print(g,"done:",df.shape,flush=True)
cover=0
for cell in mk["markers_by_celltype"]:
    if cell not in panelH: continue
    cls=c2c.get(cell); hm=HM.get(cls); rec={"class":cls,"H_min_class":hm,"n_markers":len(mk["markers_by_celltype"][cell]),"labs":{}}; allp=[]
    for g,s in panelH[cell].items():
        if len(s)<10: continue
        A=s.values/hm if hm else s.values*np.nan; t=testH[cell][g]; href=float(np.median(s)); lo,hi=np.percentile(s,[10,90]); inb=float(np.mean((t>=lo)&(t<=hi))) if len(t) else None
        rec["labs"][g]={"n_panel":int(len(s)),"H_ref":href,"H_p10":float(lo),"H_p90":float(hi),"A_p10":float(np.percentile(A,10)),"A_p50":float(np.median(A)),"A_p90":float(np.percentile(A,90)),"test_in_p10_p90":inb,"n_test":int(len(t))}; allp.append(s.values)
    if allp:
        a=np.concatenate(allp); rec["pooled"]={"n":int(len(a)),"H_ref":float(np.median(a)),"H_p10":float(np.percentile(a,10)),"H_p90":float(np.percentile(a,90)),"A_p10":float(np.percentile(a,10)/hm) if hm else None,"A_p50":float(np.median(a)/hm) if hm else None,"A_p90":float(np.percentile(a,90)/hm) if hm else None}; cover+=1
    out["entries"][cell]=rec
os.makedirs(os.path.join(ENGINE,"Runtime Matrices","Percell_Reference"),exist_ok=True); po=os.path.join(ENGINE,"Runtime Matrices","Percell_Reference","percell_reference_v0.json")
json.dump(out,open(po,"w"),indent=1); print(f"entries with a reference: {cover}/115 -> {po}")
# summary: held-out coverage and the alias spread question
cov=[v["test_in_p10_p90"] for e in out["entries"].values() for v in e["labs"].values() if v["test_in_p10_p90"] is not None]
print(f"held-out arrays inside the panel p10-p90 (nominal 0.80): median {np.median(cov):.2f}, IQR {np.percentile(cov,25):.2f}-{np.percentile(cov,75):.2f} over {len(cov)} entry x lab cells")
for cell in ["Neutrophils_reinius","Neu","granulocytes","Neutrophils_EPIC","neutrophil","CD4_T-cells","CD56_NK-cells"]:
    e=out["entries"].get(cell,{}); 
    if e.get("pooled"): print(f"  {cell:<20} pooled H_ref {e['pooled']['H_ref']:.4f}  A_p50 {e['pooled']['A_p50']:.3f}  per-lab A_p50 {[round(v['A_p50'],3) for v in e['labs'].values()]}")
