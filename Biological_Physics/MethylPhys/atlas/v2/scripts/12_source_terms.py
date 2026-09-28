#!/usr/bin/env python3
"""Atlas v2 source terms, closed form (2026-09-28). Stage A's joint NUTS fit did not converge (mu R-hat max 4.3, ESS min 2):
a sequencing source's a,b trade off against the means of every cell only that source measured, so a,b are pinned only by cells
measured on BOTH our arrays and that source. This estimates them directly from exactly those cells.
Scale: our Stage 1 array scale. For each cell measured by an array source AND sequencing source k, at each locus:
array-scale mean m_arr (mean over that cell's array samples) vs sequencing mean m_k (depth>=10). Fit m_k = a_k + b_k*m_arr by
Deming regression (errors in both). CI: bootstrap over CELLS (cells are the independent units), 2,000 resamples.
Array labs: d_k = median over overlap cells of (lab mean - mean of the other array labs' means), centred to sum to zero.
Sources with no overlap cell are reported NOT IDENTIFIED - never assumed."""
import os, json, numpy as np, pandas as pd
R="/home/ubuntu/data/atlas_sources"; SEED=20260928; NL=int(os.environ.get("NLOCI","50000"))
S=pd.read_csv("roster_samples.csv"); RO=pd.read_csv("roster.csv",index_col=0)
H=pd.read_csv("hsc_manifest.csv"); H=H[H.subject_status=="normal"]
S=pd.concat([S,pd.DataFrame({"cell":H.label.str.replace(" of normal bone marrow","",regex=False)+" (bone marrow)","source":"GSE63409","platform":"array","sample":H.gsm,"qc":H.call_rate>=0.93})])
adm=RO.index[RO.v2_status.str.startswith("IN")].tolist(); S=S[S.qc & S.cell.isin(adm)].copy()
B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
rng=np.random.default_rng(SEED); loci=np.sort(rng.choice(B.index[(C>=10).mean(axis=1)>0.9].values,NL,replace=False))
SUB={"Moss2018":"moss2018/shards","Salas2018":"blood/GSE110554/shards","Salas2022":"blood/GSE167998/shards","GSE63409":"hsc_gse63409/shards"}
def vec(r):
    if r.source=="Loyfer2023": return B[r["sample"]].where(C[r["sample"]]>=10).reindex(loci)
    if r.source=="Tian2023": d=pd.read_parquet(f"{R}/tian2023/{r['sample']}_array.parquet"); return d["beta"].where(d["cov"]>=10).reindex(loci)
    if r.source=="ENCODE": d=pd.read_parquet(r["sample"]); return d["beta"].where(d["cov"]>=10).reindex(loci)
    return pd.read_parquet(f"{R}/{SUB[r.source]}/{r['sample']}.parquet").iloc[:,0].reindex(loci)
M={}
for (cell,src),g in S.groupby(["cell","source"]): M[(cell,src)]=pd.concat([vec(r) for _,r in g.iterrows()],axis=1).mean(axis=1)
ARR=[s for s in SUB if any(k[1]==s for k in M)]; SEQ=sorted({k[1] for k in M}-set(ARR))
cells=sorted({k[0] for k in M}); arr_mean={c:pd.concat([M[(c,s)] for s in ARR if (c,s) in M],axis=1).mean(axis=1) for c in cells if any((c,s) in M for s in ARR)}
def deming(x,y):
    sx,sy=np.var(x),np.var(y); sxy=np.cov(x,y)[0,1]; b=(sy-sx+np.sqrt((sy-sx)**2+4*sxy**2))/(2*sxy); return float(np.mean(y)-b*np.mean(x)),float(b)
out={"scale":"our Stage 1 array scale","loci":NL,"array_sources":ARR,"sequencing":{},"array":{}}
for k in SEQ:
    ov=[c for c in cells if (c,k) in M and c in arr_mean]
    if not ov: out["sequencing"][k]={"status":"NOT IDENTIFIED - no cell measured on both an array and this source"}; print(k,"NOT IDENTIFIED",flush=True); continue
    pairs={c:pd.DataFrame({"x":arr_mean[c],"y":M[(c,k)]}).dropna() for c in ov}
    X=pd.concat(pairs.values()); a,b=deming(X.x.values,X.y.values)
    bs=[]
    for _ in range(2000):
        pick=rng.choice(ov,len(ov),replace=True); Z=pd.concat([pairs[c] for c in pick]); bs.append(deming(Z.x.values,Z.y.values))
    bs=np.array(bs); res=X.y-(a+b*X.x)
    out["sequencing"][k]={"status":"IDENTIFIED","overlap_cells":ov,"n_pairs":int(len(X)),"a":a,"b":b,"a_ci95":np.percentile(bs[:,0],[2.5,97.5]).tolist(),
        "b_ci95":np.percentile(bs[:,1],[2.5,97.5]).tolist(),"resid_sd":float(res.std()),"r":float(np.corrcoef(X.x,X.y)[0,1]),
        "per_cell_b":{c:round(deming(p.x.values,p.y.values)[1],4) for c,p in pairs.items() if len(p)>100}}
    print(k,json.dumps({x:out["sequencing"][k][x] for x in ("overlap_cells","n_pairs","a","b","a_ci95","b_ci95","resid_sd","r")}),flush=True)
d={}
for k in ARR:
    diffs=[]
    for c in cells:
        others=[s for s in ARR if s!=k and (c,s) in M]
        if (c,k) in M and others: diffs.append(float((M[(c,k)]-pd.concat([M[(c,s)] for s in others],axis=1).mean(axis=1)).median()))
    d[k]=(float(np.median(diffs)) if diffs else None,len(diffs))
ok=[v[0] for v in d.values() if v[0] is not None]; cen=float(np.mean(ok)) if ok else 0.0
out["array"]={k:({"d":v[0]-cen,"overlap_cells":v[1]} if v[0] is not None else {"status":"NOT IDENTIFIED - shares no cell with another array lab"}) for k,v in d.items()}
print("array labs:",json.dumps(out["array"]),flush=True)
json.dump(out,open("source_terms.json","w"),indent=1); print("DONE",flush=True)

# ---- BRIDGE (2026-09-28): a source with no cell shared with our arrays is chained through Loyfer (source -> Loyfer -> array),
# fitted only on cells that source shares with Loyfer. Composition: m_k = a' + b'*m_L and m_L = aL + bL*m_arr.
L=out["sequencing"].get("Loyfer2023",{})
if L.get("status")=="IDENTIFIED":
    aL,bL=L["a"],L["b"]
    for k in SEQ:
        if k=="Loyfer2023" or out["sequencing"][k]["status"]=="IDENTIFIED": continue
        ov=[c for c in cells if (c,k) in M and (c,"Loyfer2023") in M]
        if not ov: out["sequencing"][k]={"status":"NOT IDENTIFIED - shares no cell with our arrays or with Loyfer"}; print(k,"bridge: NOT IDENTIFIED",flush=True); continue
        X=pd.concat([pd.DataFrame({"x":M[(c,"Loyfer2023")],"y":M[(c,k)]}).dropna() for c in ov]); a1,b1=deming(X.x.values,X.y.values)
        out["sequencing"][k]={"status":"IDENTIFIED VIA LOYFER","overlap_cells_with_loyfer":ov,"n_pairs":int(len(X)),"a_vs_loyfer":a1,"b_vs_loyfer":b1,
            "a":a1+b1*aL,"b":b1*bL,"r_vs_loyfer":float(np.corrcoef(X.x,X.y)[0,1])}
        print(k,"bridge:",json.dumps({x:out["sequencing"][k][x] for x in ("overlap_cells_with_loyfer","n_pairs","a","b","r_vs_loyfer")}),flush=True)
    for k,v in list(out["array"].items()):
        if "d" in v: continue
        ov=[c for c in cells if (c,k) in M and (c,"Loyfer2023") in M]
        if not ov: out["array"][k]={"status":"NOT IDENTIFIED - shares no cell with another array lab or with Loyfer"}; print(k,"bridge: NOT IDENTIFIED",flush=True); continue
        diffs=[float((M[(c,k)]-(M[(c,"Loyfer2023")]-aL)/bL).median()) for c in ov]
        out["array"][k]={"d_via_loyfer":float(np.median(diffs)),"overlap_cells_with_loyfer":ov,"note":"offset of this lab vs Loyfer mapped onto the array scale; not centred with the directly measured labs"}
        print(k,"bridge:",json.dumps(out["array"][k]),flush=True)
json.dump(out,open("source_terms.json","w"),indent=1); print("BRIDGE DONE",flush=True)
