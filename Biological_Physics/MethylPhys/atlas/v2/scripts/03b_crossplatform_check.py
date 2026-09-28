#!/usr/bin/env python3
"""Same cell type measured two ways: Moss 2018 sorted cells on 450K/EPIC through our Stage 1, vs Loyfer 2023 sorted cells by WGBS.
For each shared cell type: Pearson r of the two mean profiles over shared CpGs (Loyfer depth >= 10), median |diff|, and the
linear fit loyfer = a + b*array - the first look at the pipeline map this source will need. Moss arrays below the 0.93 call-rate
line are excluded."""
import json, glob, os, numpy as np, pandas as pd
R="/home/ubuntu/data/atlas_sources"
B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); Cv=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
meta=json.load(open("loyfer_extract_report.json"))["per_sample"]
M=pd.read_csv("moss_manifest.csv"); M=M[(M.call_rate>=0.93)]
pairs={"hepatocytes":["Hepatocyte"],"adipocytes":["Adipocytes"],"pancreatic beta cells":["Beta"],"pancreatic acinar cells":["Acinar"],
       "pancreatic duct cells":["Duct"],"cortical neurons":["Neuronal"],"vascular endothelial cells":["Endothelium"],
       "colon epithelial cells":["Epithelium:Colon"],"lung epithelial cells":["Epithelium:Lung alveolar"],"leukocytes":["Granulocytes"]}
out=[]
for moss_lab,lk in pairs.items():
    bases=M[M.label==moss_lab].base.tolist()
    if not bases: continue
    arr=pd.concat([pd.read_parquet(f"{R}/moss2018/shards/{b}.parquet").iloc[:,0] for b in bases],axis=1).mean(axis=1)
    cols=[]
    for spec in lk:
        ct,_,tis=spec.partition(":")
        cols+=[g for g,v in meta.items() if v["cell_type"]==ct and (not tis or v["tissue"]==tis)]
    if not cols: continue
    lb=B[cols].where(Cv[cols]>=10).mean(axis=1)
    j=arr.index.intersection(lb.dropna().index); a=arr.loc[j]; w=lb.loc[j]
    b1,b0=np.polyfit(a,w,1)
    out.append({"cell":moss_lab,"moss_arrays":len(bases),"loyfer_samples":len(cols),"shared_cpgs":int(len(j)),
                "r":round(float(np.corrcoef(a,w)[0,1]),4),"median_abs_diff":round(float((a-w).abs().median()),4),
                "slope":round(float(b1),4),"intercept":round(float(b0),4),"mean_array":round(float(a.mean()),4),"mean_wgbs":round(float(w.mean()),4)})
    print(out[-1],flush=True)
pd.DataFrame(out).to_csv("crossplatform_check.csv",index=False)
