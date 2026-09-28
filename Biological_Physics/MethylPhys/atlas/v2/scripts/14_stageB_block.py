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
R="/home/ubuntu/data/atlas_sources"; OUTD=os.environ.get("OUTD","/home/ubuntu/data/atlas_v2/blocks_v2"); os.makedirs(OUTD,exist_ok=True)
BI=int(os.environ["BLOCK"]); NB=int(os.environ["NBLOCK"]); dst=f"{OUTD}/block_{BI:05d}.parquet"
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
cells=sorted(O.cell.unique()); ci={c:i for i,c in enumerate(cells)}; NC=len(cells)
O["c"]=O.cell.map(ci)
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
# FLATNESS LESSON (IAMAtlas_FLATNESS_LESSON.md, 2026-05-25): a (cell, locus) pair with no observation has only the prior and
# returns the locus centre - the grand-mean fill that made v1 cells unfindable. Every pair records how many samples it rests on;
# a pair with none is written NOT MEASURED (NaN), never as a mean. The author, 2026-09-28: no partial cells, nothing filled in.
NOBS=O.groupby(["c","l"]).size().unstack().reindex(index=range(NC),columns=range(NL)).fillna(0).astype(int).values
rows=[]
for c in range(NC):
    rows.append(pd.DataFrame({"cell":cells[c],"cpg":loci,"n_obs":NOBS[c],"mu":flat[:,c,:].mean(0),"mu_sd":flat[:,c,:].std(0),"mu_lo":np.percentile(flat[:,c,:],2.5,0),"mu_hi":np.percentile(flat[:,c,:],97.5,0),
                              "donor_sd":sdf[:,c,:].mean(0),"rhat":np.asarray(sm["mu"]["r_hat"])[c],"ess":np.asarray(sm["mu"]["n_eff"])[c]}))
out=pd.concat(rows)
# POSTERIOR DRAWS (2026-09-28): 20 thinned draws of mu per (cell, locus), float16, so later tools carry the atlas's own uncertainty
# (G-002 never wrote its chains; this run does). NaN where the pair has no observation.
K=20; idx=np.linspace(0,flat.shape[0]-1,K).astype(int); DR=flat[idx].astype(np.float16)            # (K, NC, NL)
DR[:,NOBS==0]=np.nan
np.savez_compressed(dst.replace(".parquet","_draws.npz")+".tmp.npz",draws=DR,cells=np.array(cells),cpg=np.array(loci))
os.replace(dst.replace(".parquet","_draws.npz")+".tmp.npz",dst.replace(".parquet","_draws.npz"))
# PER-LOCUS PRIOR (nu, tau posterior means): what the append fit holds fixed when a cell is added later.
pd.DataFrame({"cpg":loci,"nu":np.asarray(G["nu"]).reshape(-1,NL).mean(0),"tau":float(np.asarray(G["tau"]).mean())}).to_parquet(dst.replace(".parquet","_prior.parquet"))
for col in [c for c in out.columns if c not in ("cell","cpg","n_obs")]: out.loc[out.n_obs==0,col]=np.nan
out.to_parquet(dst+".tmp"); os.replace(dst+".tmp",dst)
div=int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
print(json.dumps({"block":BI,"loci":NL,"cells":NC,"wall_s":round(wall),"s_per_locus":round(wall/NL,3),"divergences":div,"rhat_p99":float(np.nanpercentile(out.rhat,99)),"ess_p01":float(np.nanpercentile(out.ess,1))}),flush=True)
