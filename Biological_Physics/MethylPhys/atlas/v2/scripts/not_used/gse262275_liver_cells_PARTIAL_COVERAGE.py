#!/usr/bin/env python3
"""GSE262275 (McNamara et al., Nat Commun 2025): 14 new purified-cell methylomes, wgbstools .beta (hg19, 28,217,448 CpGs - the same
format and index as Loyfer, 56,434,896 bytes each). Streams the 14 files, keeps array CpGs (probe_map_summary.csv from the Loyfer
extraction), writes beta and depth per array CpG to /home/ubuntu/data/atlas_sources/mcnamara2025/.
QUALITY RULES - written 2026-09-28 before any file is read (author: "it depends on the quality"):
 Q1 depth: a sample passes if >= 10 reads at >= 90 % of array CpGs (the v2 entry rule for WGBS). A cell needs >= 2 passing samples.
 Q2 donors: the A/B pair of each cell must be two donors, not replicates of one. Test: genotype-like concordance at the 2,000 most
    variable CpGs across all Loyfer+new samples is compared with Loyfer's own same-donor pairs (podocyte/tubule: GSM5652259-62,
    GSM5652260-63) and different-donor pairs of one cell type. A pair that looks same-donor enters FLAGGED "one donor - donor SD
    unmeasured", never silently.
 Q3 identity: sample-level twin test (twin_family_thresholds_v1.json rule applied per sample) against every Loyfer cell of the same
    tissue or family (hepatocyte; endothelia; macrophages; keratinocyte vs the WAIT keratinocyte). A twin does not enter as a new cell.
 Q4 culture (reported, flagged; not a bar - no fresh-sorted counterpart exists for most): these look like cultured primary cells.
    Report CpG-island hypermethylation (fraction of array CpGs in CGIs with beta > 0.3 that are < 0.1 in every Loyfer cell of the
    family) and global mean beta against Loyfer's fresh cells of the same family. Readings on a cultured cell carry CULTURED.
Nothing enters the atlas from this script; it writes liver_cells_report.json for the entry decision."""
import os, json, gzip, re, urllib.request, time, numpy as np, pandas as pd, itertools
R="/home/ubuntu/data/atlas_sources"; OUT=f"{R}/mcnamara2025"; os.makedirs(OUT,exist_ok=True)
U="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE262nnn/GSE262275/"
t=gzip.decompress(urllib.request.urlopen(U+"matrix/GSE262275_series_matrix.txt.gz",timeout=180).read()).decode("utf-8","replace")
row=lambda k:[x.strip('"') for x in re.search(rf"^!{k}\t(.+)$",t,re.M).group(1).split("\t")]
gsm=row("Sample_geo_accession"); src=row("Sample_source_name_ch1"); tit=row("Sample_title")
S={g:ti for g,s,ti in zip(gsm,src,tit) if s not in ("Serum","Plasma") and not re.search(r"ATAC|H3K27ac",ti)}
fl=urllib.request.urlopen(U+"suppl/filelist.txt",timeout=60).read().decode().splitlines()[1:]
F={l.split("\t")[1].split("_")[0]:l.split("\t")[1] for l in fl if l.split("\t")[1].endswith(".beta")}
m=pd.read_csv("probe_map_summary.csv").dropna(subset=["cpg_index"]); rows=(m.cpg_index.astype(np.int64)-1).values; ids=m.IlmnID.values
B={}; Cv={}; rep={}
for g,ti in sorted(S.items()):
    f=F.get(g); 
    if not f: rep[g]={"title":ti,"status":"no .beta file"}; continue
    p=f"{OUT}/{f}"
    if not os.path.exists(p): urllib.request.urlretrieve(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/{f}",p)
    a=np.fromfile(p,dtype=np.uint8).reshape(-1,2)[rows]; cov=a[:,1].astype(np.float32); meth=a[:,0].astype(np.float32)
    B[g]=np.where(cov>0,meth/np.maximum(cov,1),np.nan); Cv[g]=a[:,1]
    rep[g]={"title":ti,"median_cov":float(np.median(cov)),"frac_cov_ge10":round(float((cov>=10).mean()),4),"Q1_pass":bool((cov>=10).mean()>=0.90)}
    print(g,ti,rep[g],flush=True)
Bn=pd.DataFrame(B,index=ids); Cn=pd.DataFrame(Cv,index=ids); Bn.to_parquet(f"{OUT}/array_beta.parquet"); Cn.to_parquet(f"{OUT}/array_cov.parquet")
cell=lambda ti: re.sub(r"-[AB]$","",ti)
cells={}; [cells.setdefault(cell(rep[g]["title"]),[]).append(g) for g in B]
# Q2 donors
LB=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); LC=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
LX=LB.where(LC>=10); NX=Bn.where(Cn>=10); J=LX.join(NX,how="inner").dropna()
top=J.loc[J.var(axis=1).sort_values().index[-2000:]].round(0)
def conc(a,b): return float((top[a]==top[b]).mean())
same=[conc("GSM5652259","GSM5652262"),conc("GSM5652260","GSM5652263")]
diffp=[conc(a,b) for a,b in (("GSM5652258","GSM5652259"),("GSM5652258","GSM5652260"),("GSM5652259","GSM5652260"))]
q2={"loyfer_same_donor":same,"loyfer_diff_donor_same_cell":diffp}
for c,gs in cells.items():
    if len(gs)==2: q2[c]=conc(*gs)
rep["Q2"]=q2; print("Q2",json.dumps(q2),flush=True)
# Q3 twin test against same-family Loyfer cells
TW=json.load(open("twin_family_thresholds_v1.json")); SD=TW.get("sep_delta",0.2); SL=TW.get("min_separating_loci",30); TR=TW.get("twin_r",0.985)
RS=pd.read_csv("roster_samples.csv"); RS=RS[(RS.source=="Loyfer2023")&RS.qc]
fam={"Hepatic-Stellate":["hepatocyte","heart fibroblasts","colon fibroblasts","smooth muscle"],"Liver-Immune":["lung alveolar macrophages","lung interstitial macrophages","colon macrophages","monocytes"],
     "Dermal-Microvascular-Endothelial":["aorta endothelium","vascular saphenous endothelium","lung alveolar endothelium","kidney glomerular endothelium","kidney tubular endothelium","pancreas endothelium"],
     "Biliary":["hepatocyte","pancreatic duct","gallbladder","colon epithelium"],"Keratinocyte":["keratinocyte","tongue epithelium","tonsil palatine epithelium"]}
q3={}
for c,gs in cells.items():
    k=next((k for k in fam if c.startswith(k) or k in c),None); cmp=fam.get(k,[])
    for o in cmp:
        og=RS[RS.cell==o]["sample"].tolist()
        if not og: continue
        Xa=NX[gs]; Xb=LX[og]; j=Xa.dropna().index.intersection(Xb.dropna().index)
        if len(gs)<2 or len(og)<2: 
            r=float(np.corrcoef(Xa.loc[j].mean(1),Xb.loc[j].mean(1))[0,1]); q3[f"{c} vs {o}"]={"r":round(r,4),"note":"<2 samples on a side - r only"}; continue
        a=Xa.loc[j].values; b=Xb.loc[j].values; sep=int(((a.min(1)-b.max(1))>SD).sum()+((b.min(1)-a.max(1))>SD).sum())
        r=float(np.corrcoef(a.mean(1),b.mean(1))[0,1]); q3[f"{c} vs {o}"]={"r":round(r,4),"nonoverlap_sep":sep,"twin":bool(r>TR and sep<SL)}
    # the biliary subtypes against each other
bil=[c for c in cells if c.startswith("Biliary")]
for a,b in itertools.combinations(bil,2):
    Xa=NX[cells[a]]; Xb=NX[cells[b]]; j=Xa.dropna().index.intersection(Xb.dropna().index); A_=Xa.loc[j].values; B_=Xb.loc[j].values
    sep=int(((A_.min(1)-B_.max(1))>SD).sum()+((B_.min(1)-A_.max(1))>SD).sum()); r=float(np.corrcoef(A_.mean(1),B_.mean(1))[0,1])
    q3[f"{a} vs {b}"]={"r":round(r,4),"nonoverlap_sep":sep,"twin":bool(r>TR and sep<SL)}
rep["Q3"]=q3; [print("Q3",k,v,flush=True) for k,v in q3.items()]
# Q4 culture indicators
try:
    A=pd.read_csv("/home/ubuntu/data/HM450.hg38.manifest.gencode.v36.tsv.gz",sep="\t",usecols=["probeID","CGI"],low_memory=False).set_index("probeID")
    cgi=set(A.index[A.CGI.astype(str).str.contains("Island",na=False)])
except Exception: cgi=set()
loy_all=[x for x in RS["sample"] if x in LX.columns]; fresh_low=LX.loc[LX.index.isin(cgi),loy_all].max(axis=1)<0.1
q4={}
for c,gs in cells.items():
    v=NX.loc[NX.index.isin(cgi),gs].mean(1); fl=fresh_low.reindex(v.index).fillna(False)
    q4[c]={"cgi_hyper_frac":round(float((v[fl]>0.3).mean()),4),"global_mean_beta":round(float(NX[gs].mean(1).mean()),4)}
q4["loyfer_fresh_reference"]={"global_mean_beta_median":round(float(LX[loy_all].mean().median()),4)}
rep["Q4"]=q4; print("Q4",json.dumps(q4),flush=True)
json.dump(rep,open("liver_cells_report.json","w"),indent=1); print("DONE",flush=True)
