#!/usr/bin/env python3
"""ENCODE embryonic stem line WGBS (H1, H9, HUES64), GRCh38 'methylation state at CpG' bedMethyl, streamed; only array CpGs kept.
bedMethyl: chrom, start (0-based, the C), end, name, score, strand, ..., col 10 = reads, col 11 = % methylated.
Array CpG hg38 positions from Zhou's InfiniumAnnotation (CpG_beg = 0-based start of the CG): + strand C at CpG_beg, - strand C at
CpG_beg+1 (0-based). Strands summed. Out: /home/ubuntu/data/atlas_sources/encode_stem/<file>_array.parquet (mc, cov, beta);
job output stem_extract_report.json."""
import os, io, json, subprocess, time, urllib.request, concurrent.futures as cf
import numpy as np, pandas as pd
OUT="/home/ubuntu/data/atlas_sources/encode_stem"; os.makedirs(OUT,exist_ok=True)
q=("https://www.encodeproject.org/search/?type=File&assay_title=WGBS&status=released&assembly=GRCh38&output_type=methylation+state+at+CpG"
   "&file_format=bed&biosample_ontology.term_name=H1&biosample_ontology.term_name=H9&biosample_ontology.term_name=HUES64&limit=all&format=json")
g=json.load(urllib.request.urlopen(urllib.request.Request(q,headers={"Accept":"application/json"}),timeout=120))["@graph"]
files=[{"acc":f["accession"],"cell":(f.get("biosample_ontology") or {}).get("term_name"),"exp":f.get("dataset"),"reps":f.get("biological_replicates"),
        "size":f.get("file_size"),"url":"https://www.encodeproject.org"+f["href"]} for f in g]
for f in files: print(f["cell"],f["acc"],f["exp"],f["reps"],round((f["size"] or 0)/1e9,2),"GB",flush=True)
man=[]
for u in ("https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/HM450/HM450.hg38.manifest.tsv.gz",
          "https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/EPIC/EPIC.hg38.manifest.tsv.gz"):
    man.append(pd.read_csv(io.BytesIO(urllib.request.urlopen(u,timeout=300).read()),sep="\t",compression="gzip",usecols=["CpG_chrm","CpG_beg","Probe_ID"],dtype={"CpG_chrm":str}))
m=pd.concat(man).dropna().drop_duplicates("Probe_ID"); m=m[m.Probe_ID.str.startswith("cg")]; m["CpG_beg"]=m.CpG_beg.astype(np.int64)
pos={}
for pid,c,b in zip(m.Probe_ID,m.CpG_chrm,m.CpG_beg): pos[(c,b)]=pid; pos[(c,b+1)]=pid
def one(f):
    t0=time.time(); dst=f"{OUT}/{f['cell']}_{f['acc']}_array.parquet"
    if os.path.exists(dst):
        d=pd.read_parquet(dst)
    else:
        p=subprocess.Popen(f"curl -sL --retry 5 '{f['url']}' | gzip -dc",shell=True,stdout=subprocess.PIPE,bufsize=1<<20)
        mc={}; cov={}
        for line in io.TextIOWrapper(p.stdout,encoding="ascii",errors="ignore"):
            x=line.split("\t")
            if len(x)<11: continue
            pid=pos.get((x[0],int(x[1])))
            if pid is None: continue
            n=int(x[9]); k=round(n*float(x[10])/100.0)
            mc[pid]=mc.get(pid,0)+k; cov[pid]=cov.get(pid,0)+n
        p.wait()
        d=pd.DataFrame({"mc":pd.Series(mc),"cov":pd.Series(cov)}); d.to_parquet(dst+".counts")
        d["beta"]=d["mc"]/d["cov"].where(d["cov"]>0); d.to_parquet(dst)
    return {**f,"array_cpgs":int(len(d)),"median_cov":float(d["cov"].median()),"frac_cov_ge10":round(float((d["cov"]>=10).mean()),4),
            "mean_beta":round(float(d["beta"].mean()),4),"seconds":round(time.time()-t0)}
with cf.ThreadPoolExecutor(len(files)) as ex: rep=list(ex.map(one,files))
for r in rep: print(r["cell"],r["acc"],"CpGs",r["array_cpgs"],"median reads",r["median_cov"],">=10",r["frac_cov_ge10"],"mean beta",r["mean_beta"],flush=True)
json.dump(rep,open("stem_extract_report.json","w"),indent=1); print("DONE",flush=True)
