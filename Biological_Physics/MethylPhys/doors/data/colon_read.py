#!/usr/bin/env python3
"""Molecule-assigned IAM-A (development, 2026-10-01). In colon-epithelium M-blocks (colon methylated; blood and colon stroma unmethylated) the
qualifying molecules (>= 6 CpGs, >= 80 % methylated) are colon-epithelial molecules. Per specimen (EOCRC tumour/normal, OSCC as a non-colon
control): copy error on ALL qualifying molecules vs on the colon-block molecules only; colon content = mean beta at block CpGs / colon reference beta;
opportunities in blocks per million qualifying opportunities (sets the depth needed when colon cells are rare, e.g. stool).
Same genotype mask as PROC-TUMOUR-01 (site dropped if >= 5 qualifying molecules and > 30 % carry an error, per patient over both tissues)."""
import os, json, urllib.request, numpy as np, pandas as pd
U=json.load(open("tumour_urls.json")); os.makedirs("t",exist_ok=True)
for k,u in U.items():
    if k.endswith(".parquet") and not os.path.exists("t/"+k): urllib.request.urlretrieve(u,"t/"+k)
fai=[l.split("\t")[0] for l in open("/home/ubuntu/data/tumour/ref/hg38.fa.fai")]
BALL=pd.read_csv("colon_epi_M_blocks.csv")
tabs={k[:-8]:pd.read_parquet("t/"+k) for k in U if k.endswith(".parquet")}
print("columns",list(next(iter(tabs.values())).columns),flush=True)
def patient(k): return "_".join(k.split("_")[:3])
mask={}
for pt in sorted({patient(k) for k in tabs}):
    D=pd.concat([tabs[k] for k in tabs if patient(k)==pt]).groupby("pos")[["opp_A","err_A","opp_B","err_B"]].sum()
    o=D.opp_A+D.opp_B; e=D.err_A+D.err_B; mask[pt]=set(D.index[(o>=5)&(e/o.clip(lower=1)>0.30)])
rows=[]
for SET,B in BALL.groupby('set'):
  ref_beta=float(np.average(B.colon_beta,weights=B.n_cpg))
  iv={}
  for c,g in B.groupby("chr38"): g=g.sort_values("start38"); iv[fai.index(c)]=(g.start38.values,g.end38.values)
  def in_blocks(pos):
      rid=(pos>>32).astype(np.int64); p=(pos & 0xFFFFFFFF).astype(np.int64)+1; out=np.zeros(len(pos),bool)
      for r,(s,e) in iv.items():
          k=np.where(rid==r)[0]
          if not len(k): continue
          j=np.searchsorted(s,p[k],side="right")-1; ok=j>=0; jj=np.clip(j,0,None); out[k]=ok&(p[k]<=e[jj])
      return out
  for k,D in tabs.items():
      D=D[~D.pos.isin(mask[patient(k)])]; b=in_blocks(D.pos.values.astype(np.int64))
      o=D.opp_A+D.opp_B; e=D.err_A+D.err_B
      tot=D.t.where(D.t>0) if "t" in D else None
      beta_blk=float(D.m[b].sum()/max(D.t[b].sum(),1)) if "m" in D and "t" in D else None
      rows.append(dict(set=SET,specimen=k,sites=len(D),block_sites=int(b.sum()),opp_all=int(o.sum()),opp_blocks=int(o[b].sum()),
          eps_all=float(e.sum()/max(o.sum(),1)),eps_blocks=float(e[b].sum()/max(o[b].sum(),1)),
          beta_blocks=beta_blk,colon_content=(beta_blk/ref_beta if beta_blk is not None else None),
          blocks_opp_per_M=float(o[b].sum()/max(o.sum(),1)*1e6)))
R=pd.DataFrame(rows).sort_values(["set","specimen"]); R.to_csv("colon_read.csv",index=False)
pd.set_option("display.width",250); print(R.round(5).to_string(index=False)); print("DONE")
