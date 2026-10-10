set -e; mkdir -p remote_jobs/stool && cd remote_jobs/stool && cp ../colonmark/colon_blocks.py ../colonmark/probe_map_summary.csv.gz . && cp ../tumoursky/tumour_urls.json . && cp ../mu0wide/s3io.py . 2>/dev/null || true
python3 - <<'PY'
s=open("colon_blocks.py").read()
a='''    if g.startswith("Colon-Left-Epithelial") or g.startswith("Colon-Right-Epithelial"): k="COLON_EPI"
    elif g.startswith("Colon-") and "Endocrine" in g: continue          # colon endocrine cells are colon-derived: neither target nor background'''
b='''    if g.startswith("Colon-Left-Epithelial") or g.startswith("Colon-Right-Epithelial") or (TARGET=="lowerGI" and g.startswith("Small-int-Epithelial")): k="COLON_EPI"
    elif (g.startswith("Colon-") or g.startswith("Small-int-")) and "Endocrine" in g: continue   # gut endocrine cells: neither target nor background'''
assert s.count(a)==1; s=s.replace(a,b)
s=s.replace('MODE=os.environ.get("MODE","all")','MODE=os.environ.get("MODE","all"); TARGET=os.environ.get("TARGET","colon")')
a='''    elif MODE=="blood_stroma" and not (g.startswith("Blood-") or g in ("Colon-Fibroblasts","Colon-Macrophages")): continue'''
b='''    elif MODE=="blood_stroma" and not (g.startswith("Blood-") or g in ("Colon-Fibroblasts","Colon-Macrophages")): continue
    elif MODE=="stool" and not (g.startswith("Blood-") or "Macrophages" in g or "Fibroblasts" in g or "Endothel" in g or "Smooth-Muscle" in g or g.startswith("Gastric") or g.startswith("Esophagus") or g.startswith("Liver-Hep") or g.startswith("Pancreas")): continue'''
assert s.count(a)==1; s=s.replace(a,b)
# reverse polarity count (target unmethylated, others methylated), reported only
a='''okn=nother>=len(mean)-2'''
b='''okn=nother>=len(mean)-2
othmin=np.nanmin(oth,0)
for nm,mk in (("reverse strict",(col<=0.10)&(othmin>=0.80)&okn),("reverse loose",(col<=0.20)&(othmin>=0.70)&okn)):
    print(nm,"marker CpGs",int(mk.sum()),"blocks",len(blocks_of(mk,4 if "strict" in nm else 3)) if mk.sum() else 0,flush=True)'''
assert s.count(a)==1; s=s.replace(a,b)
# blocks_of must be defined before use: move definition above okn
i=s.index("def blocks_of"); j=s.index("okn=nother")
s=s[:i]+s[i:j]   # already in order? check
s=s.replace('B["background"]=MODE; B.to_csv(f"colon_epi_M_blocks_{MODE}.csv"','B["background"]=MODE; B["target"]=TARGET; B.to_csv(f"blocks_{TARGET}_{MODE}.csv"')
open("colon_blocks.py","w").write(s); print("def before okn:", s.index("def blocks_of")<s.index("okn=nother"))
PY
cat > tumour_scan.py <<'EOF'
#!/usr/bin/env python3
"""DEV-TUMOUR-REGIONS-01 (development, 2026-10-02). Where in the genome does the tumour's copy-error rise sit?
Early-onset CRC tumour vs the same patient's adjacent normal (PROC-TUMOUR-01 site tables, GRCh38; qualifying molecules >= 6 CpGs, >= 80 % methylated).
Genotype mask per patient as in score_tumour.py (site dropped if >= 5 opportunities and > 30 % error in either tissue).
Tiles of 2 kb. Per tile and patient: error rate in tumour and normal (opportunities summed over both run halves).
A tile is 'raised' when tumour > normal in >= 5 of the usable pairs, each tissue with >= 20 opportunities, and the pooled ratio >= 1.5.
Output: tile table, the raised tiles, and the overlap with lower-GI marker blocks if present (blocks_lowerGI_*.csv)."""
import os, json, glob, subprocess, numpy as np, pandas as pd
U=json.load(open("tumour_urls.json")); D="/home/ubuntu/data/tumour_scan"; os.makedirs(D,exist_ok=True)
pairs=sorted({k.split("_")[1] for k in U if k.startswith("EOCRC_") and k.endswith(".parquet")})
use=[p for p in pairs if f"EOCRC_{p}_WGBS_tumour.parquet" in U and f"EOCRC_{p}_WGBS_normal.parquet" in U]
print("pairs with both tissues:",use,flush=True)
for p in use:
    for t in ("tumour","normal"):
        f=f"EOCRC_{p}_WGBS_{t}.parquet"
        if not os.path.exists(f"{D}/{f}"): subprocess.run(["curl","-s","-f","-o",f"{D}/{f}",U[f]],check=True)
CH=[f"chr{i}" for i in range(1,23)]+["chrX","chrY"]; TILE=2000
rows=[]
for p in use:
    T=pd.read_parquet(f"{D}/EOCRC_{p}_WGBS_tumour.parquet"); N=pd.read_parquet(f"{D}/EOCRC_{p}_WGBS_normal.parquet")
    for X in (T,N): X["o"]=X.opp_A+X.opp_B; X["e"]=X.err_A+X.err_B
    bad=set(T.pos[(T.o>=5)&(T.e>0.3*T.o)])|set(N.pos[(N.o>=5)&(N.e>0.3*N.o)])
    for nm,X in (("T",T),("N",N)):
        X=X[~X.pos.isin(bad)]; ref=(X.pos.values>>32); bp=(X.pos.values&0xFFFFFFFF)
        g=pd.DataFrame(dict(chr=ref,tile=bp//TILE,o=X.o.values,e=X.e.values,m=X.m.values,t=X.t.values)).groupby(["chr","tile"]).sum().reset_index()
        g["pair"]=p; g["tissue"]=nm; rows.append(g)
    print(p,"done",flush=True)
A=pd.concat(rows,ignore_index=True)
W=A.pivot_table(index=["chr","tile"],columns=["pair","tissue"],values=["o","e","m","t"],aggfunc="sum").fillna(0)
out=[]
for (c,t),r in W.iterrows():
    up=0; ok=0; oT=eT=oN=eN=0; mN=tN=0
    for p in use:
        o1,e1,o2,e2=r[("o",p,"T")],r[("e",p,"T")],r[("o",p,"N")],r[("e",p,"N")]
        mN+=r[("m",p,"N")]; tN+=r[("t",p,"N")]
        if o1>=20 and o2>=20:
            ok+=1; up+=int(e1/o1>e2/o2); oT+=o1; eT+=e1; oN+=o2; eN+=e2
    if ok>=len(use)-1:
        out.append(dict(chr=CH[int(c)] if int(c)<len(CH) else str(c),start=int(t)*TILE,end=int(t+1)*TILE,pairs=ok,tumour_higher=up,
                        err_T=eT/oT,err_N=eN/oN,ratio=(eT/oT)/max(eN/oN,1e-9),opp_T=oT,opp_N=oN,meth_N=mN/max(tN,1)))
R=pd.DataFrame(out); R.to_csv("tumour_tiles.csv.gz",index=False)
k=len(use); hi=R[(R.tumour_higher>=k-1)&(R.ratio>=1.5)].sort_values("ratio",ascending=False); hi.to_csv("tumour_raised_tiles.csv",index=False)
print("tiles scored",len(R),"| raised",len(hi),"| genome-wide err T %.5f N %.5f"%(R.err_T.mul(R.opp_T).sum()/R.opp_T.sum(),R.err_N.mul(R.opp_N).sum()/R.opp_N.sum()),flush=True)
print("fraction of tiles tumour>normal in >=%d pairs: %.3f"%(k-1,(R.tumour_higher>=k-1).mean()))
print(hi.head(30).round(4).to_string(index=False))
for f in glob.glob("blocks_lowerGI_*.csv"):
    B=pd.read_csv(f); B=B[B.chr38.notna()]
    def tiles_of(b): return [(b.chr38,s) for s in range(int(b.start38)//TILE*TILE,int(b.end38)+1,TILE)]
    key=set(x for b in B.itertuples() for x in tiles_of(b)); R["inblk"]=[(a,b) in key for a,b in zip(R.chr,R.start)]
    print(f,"blocks",len(B),"| scored tiles in blocks",int(R.inblk.sum()),"| of those tumour>normal in >=%d pairs: %d, ratio median %.2f"%(k-1,int((R.inblk&(R.tumour_higher>=k-1)).sum()),R[R.inblk].ratio.median() if R.inblk.any() else float('nan')))
print("DONE")
EOF
cat > run_stool.sh <<'EOF'
#!/bin/bash
set -e
P=~/env/bin/python; $P -c "import pysam" 2>/dev/null || ~/env/bin/pip install -q pysam pyliftover
[ -d /home/ubuntu/data/atlas_sources/loyfer2023/beta ] && [ -s /home/ubuntu/data/hg19.bgz.fa.gz ] || echo "NEED_ATLAS"
for M in all stool; do TARGET=lowerGI MODE=$M nice -n 5 $P colon_blocks.py 2>&1 | grep -vE "Warning|nanm" | tail -14; done
nice -n 5 $P tumour_scan.py 2>&1 | tail -50
EOF
bash -n run_stool.sh && python3 -c "import ast;[ast.parse(open(f).read()) for f in ('colon_blocks.py','tumour_scan.py')];print('ok')"; ls