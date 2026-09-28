#!/usr/bin/env python3
"""Loyfer 2023 (GSE186458) hg19 .beta -> beta and read depth at every 450K/EPIC CpG in the atlas manifest.
.beta = uint8 [meth, cov] per CpG, rows in wgbstools hg19 CpG-index order (28,217,448). The index's 2nd column is the 1-based
position of the C. Array MAPINFO conventions differ by probe/manifest, so each probe is matched at offset 0, then +1, then -1,
and the offset used is recorded; the report shows the offset distribution so a convention error is visible, not silent.
Outputs on the box: /home/ubuntu/data/atlas_sources/loyfer2023/array_beta.parquet (probe x sample, float32; NaN where cov=0)
and array_cov.parquet (uint8). Job outputs: loyfer_extract_report.json, probe_map_summary.csv."""
import os, json, gzip, time, numpy as np, pandas as pd, concurrent.futures as cf
R="/home/ubuntu/data/atlas_sources/loyfer2023"; t0=time.time()
idx=pd.read_csv("/home/ubuntu/data/wgbs_tools/references/hg19/CpG.bed.gz",sep="\t",header=None,names=["chr","pos","i"],dtype={"chr":str,"pos":np.int64,"i":np.int64})
print("index", len(idx), f"{time.time()-t0:.0f}s", flush=True)
m=pd.read_csv("manifest.csv",dtype={"CHR":str}).dropna(subset=["CHR","MAPINFO"]).drop_duplicates("IlmnID")
m["chr"]="chr"+m.CHR.str.replace("chr","",regex=False); m["MAPINFO"]=m.MAPINFO.astype(np.int64)
key=pd.Series(idx.i.values,index=pd.MultiIndex.from_arrays([idx.chr.values,idx.pos.values]))
hit=pd.Series(np.nan,index=m.index); off=pd.Series(np.nan,index=m.index)
for o in (0,1,-1):
    need=hit.isna()
    k=pd.MultiIndex.from_arrays([m.loc[need,"chr"].values,(m.loc[need,"MAPINFO"]+o).values])
    got=key.reindex(k).values; ok=~np.isnan(got)
    hit.loc[need[need].index[ok]]=got[ok]; off.loc[need[need].index[ok]]=o
m["cpg_index"]=hit; m["offset"]=off
summ={"probes":int(len(m)),"mapped":int(m.cpg_index.notna().sum()),"offset_counts":{str(k):int(v) for k,v in m.offset.value_counts(dropna=False).items()}}
print("probe mapping:", summ, flush=True)
m[["IlmnID","chr","MAPINFO","cpg_index","offset","platform"]].to_csv("probe_map_summary.csv",index=False)
mm=m.dropna(subset=["cpg_index"]); rows=(mm.cpg_index.astype(np.int64)-1).values; ids=mm.IlmnID.values
L=json.load(open("loyfer_beta_list.json"))
def read(r):
    b=np.fromfile(f"{R}/beta/{r['file']}",dtype=np.uint8).reshape(-1,2)
    assert len(b)==len(idx), (r["file"],len(b))
    x=b[rows]; cov=x[:,1]; bet=np.where(cov>0, x[:,0]/np.maximum(cov,1), np.nan).astype(np.float32)
    return r["gsm"], bet, cov
B={}; Cv={}
with cf.ThreadPoolExecutor(16) as ex:
    for g,bet,cov in ex.map(read,L): B[g]=bet; Cv[g]=cov
Bdf=pd.DataFrame(B,index=ids); Cdf=pd.DataFrame(Cv,index=ids)
Bdf.to_parquet(f"{R}/array_beta.parquet"); Cdf.to_parquet(f"{R}/array_cov.parquet")
meta={r["gsm"]:{"cell_type":r["cell_type"],"tissue":r["tissue"]} for r in L}
cov_med=Cdf.median().to_dict(); frac=(Cdf>0).mean().to_dict(); ge10=(Cdf>=10).mean().to_dict()
rep={**summ,"samples":len(L),"seconds":round(time.time()-t0),
     "per_sample":{g:{**meta[g],"median_depth":float(cov_med[g]),"frac_covered":round(float(frac[g]),4),"frac_depth_ge10":round(float(ge10[g]),4)} for g in B}}
json.dump(rep,open("loyfer_extract_report.json","w"),indent=1)
# a sanity read: sorted blood B cells vs neutrophil-ish granulocytes at a few known loci is left to the analysis; here only coverage
print("DONE", f"{time.time()-t0:.0f}s", "| median depth across samples", float(np.median(list(cov_med.values()))), "| median frac covered", float(np.median(list(frac.values()))), flush=True)
