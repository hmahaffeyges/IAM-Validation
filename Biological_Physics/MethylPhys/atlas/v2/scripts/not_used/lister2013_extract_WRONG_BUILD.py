#!/usr/bin/env python3
"""Lister 2013 (GSE47966, hg19) sorted human frontal cortex: NeuN+ neurons and NeuN- glia, 2 donors each. Per-chromosome allC files
(chr, pos, strand, context, mc, h) STREAMED; only CG-context rows at array CpGs kept. Array CpG = the + strand C at hg19 position
P = MAPINFO + offset (probe_map_summary.csv from the Loyfer extraction, which matched every probe to the hg19 CpG index, 1-based).
The allC coordinate convention is not assumed: for each strand, three candidate offsets are counted on chr21 of the first sample and
the one with the most CG-context hits is used for everything (recorded in the report). Strands summed.
Out: /home/ubuntu/data/atlas_sources/lister2013/<gsm>_array.parquet (mc, cov, beta)."""
import os, io, gzip, json, time, subprocess, urllib.request, concurrent.futures as cf, collections
import numpy as np, pandas as pd
OUT="/home/ubuntu/data/atlas_sources/lister2013"; os.makedirs(OUT,exist_ok=True)
SAMPLES={"GSM1173773":"NeuN_pos female 53yr","GSM1173774":"NeuN_neg female 53yr","GSM1173776":"NeuN_pos male 55yr","GSM1173777":"NeuN_neg male 55yr"}
FL=urllib.request.urlopen("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE47nnn/GSE47966/suppl/filelist.txt",timeout=120).read().decode().split("\n")
files=collections.defaultdict(list)
for l in FL[1:]:
    x=l.split("\t")
    if len(x)>1 and x[1].split("_")[0] in SAMPLES and "allC" in x[1]: files[x[1].split("_")[0]].append(x[1])
M=pd.read_csv("probe_map_summary.csv").dropna(subset=["offset"]); M["P"]=(M.MAPINFO+M.offset).astype(np.int64)
M["c"]=M.chr.str.replace("chr","",regex=False)
url=lambda g,f: f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/{f}"
def stream(g,f):
    p=subprocess.Popen(f"curl -sL --retry 5 '{url(g,f)}' | gzip -dc",shell=True,stdout=subprocess.PIPE,bufsize=1<<20)
    for line in io.TextIOWrapper(p.stdout,encoding="ascii",errors="ignore"):
        x=line.rstrip("\n").split("\t")
        if len(x)>=6 and x[3].startswith("CG"): yield x
    p.wait()
# ---- offset calibration on chr21 of the first sample
g0=sorted(SAMPLES)[0]; f21=[f for f in files[g0] if ".chr21." in f][0]
m21=M[M.c=="21"]; cand={"+":{o:set(m21.P+o) for o in (-1,0,1)},"-":{o:set(m21.P+1+o) for o in (-1,0,1)}}
hits={s:collections.Counter() for s in "+-"}
for x in stream(g0,f21):
    pos=int(x[1]); s=x[2]
    if s in cand:
        for o,S in cand[s].items():
            if pos in S: hits[s][o]+=1
OFF={s:hits[s].most_common(1)[0][0] for s in "+-"}; print("offset calibration chr21:",{s:dict(hits[s]) for s in "+-"},"->",OFF,flush=True)
key={}
for pid,c,P in zip(M.IlmnID,M.c,M.P): key[(c,P+OFF["+"],"+")]=pid; key[(c,P+1+OFF["-"],"-")]=pid
def one(g):
    t0=time.time(); dst=f"{OUT}/{g}_array.parquet"
    if os.path.exists(dst): d=pd.read_parquet(dst)
    else:
        mc=collections.Counter(); cov=collections.Counter()
        for f in sorted(files[g]):
            ch=f.split(".chr")[1].split(".")[0]
            for x in stream(g,f):
                pid=key.get((ch,int(x[1]),x[2]))
                if pid: mc[pid]+=int(x[4]); cov[pid]+=int(x[5])
        d=pd.DataFrame({"mc":pd.Series(mc),"cov":pd.Series(cov)}); d["beta"]=d["mc"]/d["cov"].where(d["cov"]>0); d.to_parquet(dst)
    return g,{"label":SAMPLES[g],"array_cpgs":int(len(d)),"median_cov":float(d["cov"].median()),"frac_cov_ge10":round(float((d["cov"]>=10).mean()),4),"mean_beta":round(float(d["beta"].mean()),4),"seconds":round(time.time()-t0)}
with cf.ThreadPoolExecutor(4) as ex: rep=dict(ex.map(one,sorted(SAMPLES)))
for g,r in rep.items(): print(g,r,flush=True)
json.dump({"source":"Lister et al. 2013 Science, GSE47966 (hg19)","offsets":OFF,"calibration_hits":{s:dict(hits[s]) for s in "+-"},"samples":rep},open("lister_extract_report.json","w"),indent=1)
print("DONE",flush=True)
