#!/usr/bin/env python3
"""Atlas v2 stage B: one block of loci, source terms FIXED from source_terms_v1.json (closed form; ASSUMED entries flagged there). Loci are then
independent, so blocks run in parallel on any number of boxes; each block writes one parquet (mu mean/sd/2.5/97.5, donor sd, R-hat,
ESS per cell x locus) and is skipped if already present, so a spot interruption only loses the block in flight.
Same likelihood, scale (our Stage 1 array scale), noise floor and no-class rule as stage A.
Usage: BLOCK=<i> NBLOCK=<n> python stageB_block.py   (loci = every array CpG with Loyfer depth>=10 in >=90% of samples, split into n)."""
import os, sys, json, time, numpy as np, pandas as pd
os.environ.setdefault("XLA_FLAGS","--xla_force_host_platform_device_count=4")
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist, scipy.special as ssp
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import summary
numpyro.set_host_device_count(4)
R="/home/ubuntu/data/atlas_sources"; OUTD=os.environ.get("OUTD","v5_out"); os.makedirs(OUTD,exist_ok=True)
BI=int(os.environ["BLOCK"]); NB=int(os.environ["NBLOCK"]); dst=f"{OUTD}/block_{BI:05d}.parquet"
dst=f"{OUTD}/v5_block_{BI:05d}.json"
if os.path.exists(dst): print("exists",dst); sys.exit(0)

ST=json.load(open("source_terms_v1.json")); ST.pop("_meta")   # closed-form source terms (stage A did not converge)
S=pd.read_csv("roster_samples.csv"); RO=pd.read_csv("roster.csv",index_col=0)
H=pd.read_csv("hsc_manifest.csv"); H=H[H.subject_status=="normal"]
S=pd.concat([S,pd.DataFrame({"cell":H.label.str.replace(" of normal bone marrow","",regex=False)+" (bone marrow)","source":"GSE63409","platform":"array","sample":H.gsm,"qc":H.call_rate>=0.93,"metric":H.call_rate})])
adm=RO.index[RO.v2_status.str.startswith("IN")].tolist(); S=S[(S.qc==True)&S.cell.isin(adm)]
B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
allloci=np.sort(B.index[(C>=10).mean(axis=1)>0.9].values); loci=np.array_split(allloci,NB)[BI]; NL=len(loci); li={l:i for i,l in enumerate(loci)}
obs=[]
def add(cell,src,kind,vals,cov=None):
    v=vals.reindex(loci); cv=None if cov is None else cov.reindex(loci)
    for l,y in v.items():
        if np.isfinite(y) and (cv is None or (np.isfinite(cv[l]) and cv[l]>=5)): obs.append((cell,src,kind,li[l],float(y),float(cv[l]) if cv is not None else 0.0))
SUB={"Moss2018":"moss2018/shards","Salas2018":"blood/GSE110554/shards","Salas2022":"blood/GSE167998/shards","GSE63409":"hsc_gse63409/shards"}
for _,r in S.iterrows():
    if r.source=="Loyfer2023": add(r.cell,r.source,"wgbs",B[r["sample"]],C[r["sample"]])
    elif r.source=="Tian2023": d=pd.read_parquet(f"{R}/tian2023/{r['sample']}_array.parquet"); add(r.cell,r.source,"pooled",d["beta"],d["cov"])
    elif r.source=="ENCODE": d=pd.read_parquet(r["sample"]); add(r.cell,r.source,"wgbs",d["beta"],d["cov"])
    else: add(r.cell,r.source,"array",pd.read_parquet(f"{R}/{SUB[r.source]}/{r['sample']}.parquet").iloc[:,0])
O=pd.DataFrame(obs,columns=["cell","src","kind","l","y","cov"]); O["y"]=O.y.clip(1e-3,1-1e-3)
# ---- V5 HELD-OUT MASK (doors/PROC_V5_HELDOUT_PREREG.md): 5 % of observations whose (cell, locus) pair has >= 2 observations
_n=O.groupby(["cell","l"]).y.transform("size"); _elig=np.where(_n.values>=2)[0]
_rng=np.random.default_rng(5000+BI); _held=np.sort(_rng.choice(_elig,int(round(0.05*len(O))),replace=False))
HELD=O.iloc[_held].copy(); O=O.drop(O.index[_held]).reset_index(drop=True)
cells=sorted(O.cell.unique()); ci={c:i for i,c in enumerate(cells)}; NC=len(cells)
O["c"]=O.cell.map(ci); HELD["c"]=HELD.cell.map(ci)
a=jnp.array(O.src.map(lambda s:ST[s]["a"]).values); b=jnp.array(O.src.map(lambda s:ST[s]["b"]).values)
dd=jnp.array(O.src.map(lambda s:ST[s]["d"]).values); t=jnp.array(O.src.map(lambda s:ST[s]["t"]).values)
c_=jnp.array(O.c.values); l_=jnp.array(O.l.values); y_=jnp.array(O.y.values); cov_=jnp.array(O["cov"].values)
isarr=jnp.array((O.kind=="array").values); ispool=jnp.array((O.kind=="pooled").values); FLOOR=0.01**2
def model():
    nu=numpyro.sample("nu",dist.Normal(0.,2.).expand([NL])); tau=numpyro.sample("tau",dist.HalfNormal(2.))
    lmu=numpyro.sample("lmu",dist.Normal(nu[None,:],tau).expand([NC,NL])); mu=jax.nn.sigmoid(lmu)
    lsd=numpyro.sample("lsd",dist.Normal(jnp.log(0.03),0.7).expand([NC,NL])); sd=jnp.exp(lsd)
    m=mu[c_,l_]; s_=sd[c_,l_]
    loc=jnp.where(isarr,m+dd,a+b*m)
    var=jnp.where(isarr,s_**2+t**2+FLOOR,(b**2)*jnp.where(ispool,s_**2/3.,s_**2)+m*(1-m)/jnp.maximum(cov_,1.)+t**2+FLOOR)
    numpyro.sample("y",dist.Normal(loc,jnp.sqrt(var)),obs=y_)
lm=O.groupby("l").y.mean().reindex(range(NL)).fillna(0.5).clip(0.02,0.98).values
cm=O.groupby(["c","l"]).y.mean().unstack().reindex(index=range(NC),columns=range(NL))
l0=ssp.logit(cm.clip(0.02,0.98).fillna(pd.DataFrame(np.tile(lm,(NC,1))))).values
INIT=init_to_value(values={"nu":jnp.array(ssp.logit(lm)),"tau":jnp.array(1.0),"lmu":jnp.array(l0),"lsd":jnp.full((NC,NL),jnp.log(0.03))})
t0=time.time()
mc=MCMC(NUTS(model,target_accept_prob=0.9,init_strategy=INIT,max_tree_depth=8),num_warmup=int(os.environ.get("WARM","600")),num_samples=int(os.environ.get("DRAWS","400")),num_chains=4,chain_method="parallel",progress_bar=False)
mc.run(jax.random.PRNGKey(1000+BI)); G=mc.get_samples(group_by_chain=True); jax.block_until_ready(G); wall=time.time()-t0
mu=jax.nn.sigmoid(G["lmu"]); sm=summary({"mu":mu,"lsd":G["lsd"]},group_by_chain=True)
flat=np.asarray(mu).reshape(-1,NC,NL); sdf=np.exp(np.asarray(G["lsd"]).reshape(-1,NC,NL))
# ---- V5 POSTERIOR PREDICTIVE for each held-out value, using the model's own likelihood
lsd=np.asarray(G["lsd"]).reshape(-1,NC,NL); P=flat.shape[0]; rng=np.random.default_rng(7000+BI)
hc=HELD.c.values; hl=HELD.l.values; hy=HELD.y.values; hcov=HELD["cov"].values
ha=HELD.src.map(lambda s:ST[s]["a"]).values; hb=HELD.src.map(lambda s:ST[s]["b"]).values
hd=HELD.src.map(lambda s:ST[s]["d"]).values; ht=HELD.src.map(lambda s:ST[s]["t"]).values
harr=(HELD.kind=="array").values; hpool=(HELD.kind=="pooled").values
m=flat[:,hc,hl]; sd=np.exp(lsd[:,hc,hl])                                             # (P, H)
loc=np.where(harr,m+hd,ha+hb*m)
var=np.where(harr,sd**2+ht**2+FLOOR,(hb**2)*np.where(hpool,sd**2/3.,sd**2)+m*(1-m)/np.maximum(hcov,1.)+ht**2+FLOOR)
pred=loc+np.sqrt(var)*rng.standard_normal(loc.shape)
q05,q25,q75,q95=np.percentile(pred,[5,25,75,95],axis=0); med=np.median(pred,axis=0)
in90=(hy>=q05)&(hy<=q95); in50=(hy>=q25)&(hy<=q75)
res=dict(block=BI,loci=NL,cells=NC,n_train=int(len(O)),n_held=int(len(HELD)),cov90=float(in90.mean()),cov50=float(in50.mean()),
         mae=float(np.abs(hy-med).mean()),wall_s=round(wall),divergences=int(np.asarray(mc.get_extra_fields()["diverging"]).sum()),
         by_kind={k:dict(n=int((HELD.kind==k).sum()),cov90=float(in90[(HELD.kind==k).values].mean()),cov50=float(in50[(HELD.kind==k).values].mean())) for k in HELD.kind.unique()},
         by_cell={c:dict(n=int((HELD.cell==c).sum()),cov90=float(in90[(HELD.cell==c).values].mean())) for c in HELD.cell.unique()})
json.dump(res,open(dst+".tmp","w")); os.replace(dst+".tmp",dst)
print(json.dumps({k:v for k,v in res.items() if k!="by_cell"}),flush=True)
