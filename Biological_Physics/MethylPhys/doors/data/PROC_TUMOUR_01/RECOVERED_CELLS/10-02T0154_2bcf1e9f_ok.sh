set -e; cd remote_jobs/stool
cat > tumour_scan.py <<'EOF'
#!/usr/bin/env python3
"""DEV-TUMOUR-REGIONS-01 (development, 2026-10-02). Where in the genome does the tumour's copy-error rise sit?
Early-onset CRC tumour vs the same patient's adjacent normal (PROC-TUMOUR-01 site tables, GRCh38; qualifying molecules >= 6 CpGs, >= 80 % methylated).
Genotype mask per patient as in score_tumour.py (site dropped if >= 5 opportunities and > 30 % error in either tissue). 2 kb tiles.
A tile is 'raised' when tumour > normal in all but at most one usable pair (each tissue >= 20 opportunities) and the pooled ratio >= 1.5.
Then the overlap with the lower-GI marker blocks (blocks_lowerGI_*.csv) is reported."""
import os, json, glob, subprocess, numpy as np, pandas as pd
U=json.load(open("tumour_urls.json")); D="/home/ubuntu/data/tumour_scan"; os.makedirs(D,exist_ok=True)
pairs=sorted({k.split("_")[1] for k in U if k.startswith("EOCRC_") and k.endswith(".parquet")})
use=[p for p in pairs if f"EOCRC_{p}_WGBS_tumour.parquet" in U and f"EOCRC_{p}_WGBS_normal.parquet" in U]
print("pairs with both tissues:",use,flush=True)
for p in use:
    for t in ("tumour","normal"):
        f=f"EOCRC_{p}_WGBS_{t}.parquet"
        if not os.path.exists(f"{D}/{f}"): subprocess.run(["curl","-s","-f","-o",f"{D}/{f}",U[f]],check=True)
CH=[f"chr{i}" for i in range(1,23)]+["chrX","chrY"]; TILE=2000; rows=[]
for p in use:
    T=pd.read_parquet(f"{D}/EOCRC_{p}_WGBS_tumour.parquet"); N=pd.read_parquet(f"{D}/EOCRC_{p}_WGBS_normal.parquet")
    for X in (T,N): X["o"]=X.opp_A+X.opp_B; X["e"]=X.err_A+X.err_B
    bad=np.union1d(T.pos[(T.o>=5)&(T.e>0.3*T.o)].values,N.pos[(N.o>=5)&(N.e>0.3*N.o)].values)
    for nm,X in (("T",T),("N",N)):
        X=X[~np.isin(X.pos.values,bad)]; ref=(X.pos.values>>32); bp=(X.pos.values&0xFFFFFFFF)
        g=pd.DataFrame(dict(chr=ref,tile=bp//TILE,o=X.o.values,e=X.e.values,m=X.m.values,t=X.t.values)).groupby(["chr","tile"]).sum().reset_index()
        g["pair"]=p; g["tissue"]=nm; rows.append(g)
    print(p,"done",flush=True)
A=pd.concat(rows,ignore_index=True)
k=len(use); idx=["chr","tile"]
def piv(col):
    return A.pivot_table(index=idx,columns=["pair","tissue"],values=col,aggfunc="sum").fillna(0)
O,E,Mm,Tt=piv("o"),piv("e"),piv("m"),piv("t")
oT=np.stack([O[(p,"T")].values for p in use],1); eT=np.stack([E[(p,"T")].values for p in use],1)
oN=np.stack([O[(p,"N")].values for p in use],1); eN=np.stack([E[(p,"N")].values for p in use],1)
ok=(oT>=20)&(oN>=20); up=ok&((eT/np.maximum(oT,1))>(eN/np.maximum(oN,1)))
nok=ok.sum(1); nup=up.sum(1); keep=nok>=k-1
sT=(oT*ok).sum(1); sN=(oN*ok).sum(1); rT=(eT*ok).sum(1)/np.maximum(sT,1); rN=(eN*ok).sum(1)/np.maximum(sN,1)
mN=Mm[[ (p,"N") for p in use]].sum(1).values/np.maximum(Tt[[ (p,"N") for p in use]].sum(1).values,1)
ii=O.index.to_frame(index=False)
R=pd.DataFrame(dict(chr=[CH[int(c)] if int(c)<len(CH) else str(c) for c in ii.chr],start=ii.tile.values*TILE,pairs=nok,tumour_higher=nup,
                    err_T=rT,err_N=rN,ratio=rT/np.maximum(rN,1e-9),opp_T=sT,opp_N=sN,meth_N=mN))[keep]
R["end"]=R.start+TILE; R.to_csv("tumour_tiles.csv.gz",index=False)
hi=R[(R.tumour_higher>=k-1)&(R.ratio>=1.5)].sort_values("ratio",ascending=False); hi.to_csv("tumour_raised_tiles.csv",index=False)
print("pairs",k,"| tiles scored",len(R),"| raised",len(hi),"| pooled err T %.5f N %.5f"%((R.err_T*R.opp_T).sum()/R.opp_T.sum(),(R.err_N*R.opp_N).sum()/R.opp_N.sum()))
print("tiles tumour>normal in >= %d pairs: %.3f | by normal methylation:"%(k-1,(R.tumour_higher>=k-1).mean()))
print(R.assign(mb=pd.cut(R.meth_N,[0,0.5,0.8,0.9,1.0])).groupby("mb",observed=True).agg(n=("ratio","size"),ratio=("ratio","median"),up=("tumour_higher",lambda x:(x>=k-1).mean())).round(3).to_string())
print(hi.head(25).round(4).to_string(index=False))
for f in glob.glob("blocks_lowerGI_*.csv"):
    B=pd.read_csv(f).dropna(subset=["chr38"]); key=set()
    for b in B.itertuples():
        for s in range(int(b.start38)//TILE*TILE,int(b.end38)+1,TILE): key.add((b.chr38,s))
    ib=np.array([(a,b) in key for a,b in zip(R.chr,R.start)])
    print(f,"| blocks",len(B),"| scored tiles in blocks",int(ib.sum()),"| tumour>normal in >= %d pairs: %d | ratio median %s"%(k-1,int((ib&(R.tumour_higher>=k-1).values).sum()),round(float(R[ib].ratio.median()),3) if ib.any() else "-"))
print("DONE")
EOF
cat > run_stool.sh <<'EOF'
#!/bin/bash
P=~/env/bin/python; $P -c "import pysam, pyliftover" 2>/dev/null || ~/env/bin/pip install -q pysam pyliftover
ls /home/ubuntu/data/atlas_sources/loyfer2023/beta 2>/dev/null | head -2; ls -la /home/ubuntu/data/hg19.bgz.fa.gz 2>/dev/null
for M in all stool; do TARGET=lowerGI MODE=$M nice -n 5 $P colon_blocks.py 2>&1 | grep -vE "Warning|nanm" | tail -14; done
nice -n 5 $P tumour_scan.py 2>&1 | tail -60
EOF
python3 -c "import ast;[ast.parse(open(f).read()) for f in ('colon_blocks.py','tumour_scan.py')];print('parse ok')"; bash -n run_stool.sh; ls