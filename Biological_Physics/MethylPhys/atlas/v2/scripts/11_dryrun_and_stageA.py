#!/usr/bin/env python3
"""Atlas v2 dry run (ATLAS_V2_SPEC step 3): the joint model on 1,000 random array CpGs, every admitted cell, every source.
Purpose: does it converge, what does it cost per locus, and does it recover the measured platform transfer. NOT the atlas.

SCALE: our own Stage 1 array scale (author, 2026-09-28). mu is what an array read through our Stage 1 shows; sequencing is mapped
onto it. NO CLASSES in the fit (a class prior would put the class into the profile AND the floor - circular); cells only.
Per locus l, cell c:
  logit(mu[c,l]) ~ Normal(nu[l], tau)                     weak per-locus prior shared by all cells (no class level)
  sd[c,l] ~ HalfNormal(0.1)                               between-donor SD of the cell at that locus
Observations, sample s of cell c from source k:
  array  (Moss, Salas 2018/2022, GSE63409; all our Stage 1):  y ~ Normal(mu + d_k, sqrt(sd^2 + t_k^2)),  sum_k d_k = 0 (no lab privileged)
  WGBS   (Loyfer, ENCODE):  y = mc/cov ~ Normal(a_k + b_k*mu, sqrt((b_k*sd)^2 + mu(1-mu)/cov + t_k^2))
  pooled (Tian, 3 donors):  as WGBS with sd^2/3
a_k, b_k (sequencing), d_k (array labs), t_k are GLOBAL per source - the source term.
Covariance between loci is not in the dry run (it needs the full locus set). NUTS, 4 chains."""
import os, sys, json, time, numpy as np, pandas as pd
os.environ.setdefault("XLA_FLAGS","--xla_force_host_platform_device_count=4")
import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
numpyro.set_host_device_count(4)
R="/home/ubuntu/data/atlas_sources"; NL=int(os.environ.get("NLOCI","1000")); SEED=20260928
S=pd.read_csv("roster_samples.csv"); RO=pd.read_csv("roster.csv",index_col=0)
H=pd.read_csv("hsc_manifest.csv"); H=H[H.subject_status=="normal"]
S=pd.concat([S,pd.DataFrame({"cell":H.label.str.replace(" of normal bone marrow","",regex=False)+" (bone marrow)","source":"GSE63409","platform":"array","sample":H.gsm,"qc":H.call_rate>=0.93,"metric":H.call_rate})])
adm=RO.index[RO.v2_status.str.startswith("IN")].tolist(); S=S[S.qc & S.cell.isin(adm)].copy()
B=pd.read_parquet(f"{R}/loyfer2023/array_beta.parquet"); C=pd.read_parquet(f"{R}/loyfer2023/array_cov.parquet")
rng=np.random.default_rng(SEED); loci=np.sort(rng.choice(B.index[(C>=10).mean(axis=1)>0.9].values, NL, replace=False))
obs=[]  # (cell, source, kind, locus_idx, y, cov)
li={l:i for i,l in enumerate(loci)}
def add(cell,src,kind,vals,cov=None):
    v=vals.reindex(loci); cv=None if cov is None else cov.reindex(loci)
    for l,y in v.items():
        if np.isfinite(y) and (cv is None or (np.isfinite(cv[l]) and cv[l]>=5)): obs.append((cell,src,kind,li[l],float(y),float(cv[l]) if cv is not None else 0.0))
for _,r in S.iterrows():
    if r.source=="Loyfer2023": add(r.cell,"Loyfer2023","wgbs",B[r["sample"]],C[r["sample"]])
    elif r.source=="Tian2023":
        d=pd.read_parquet(f"{R}/tian2023/{r['sample']}_array.parquet"); add(r.cell,"Tian2023","pooled",d["beta"],d["cov"])
    elif r.source=="ENCODE":
        d=pd.read_parquet(r["sample"]); add(r.cell,"ENCODE","wgbs",d["beta"],d["cov"])
    else:
        sub={"Moss2018":"moss2018/shards","Salas2018":"blood/GSE110554/shards","Salas2022":"blood/GSE167998/shards","GSE63409":"hsc_gse63409/shards"}[r.source]
        add(r.cell,r.source,"array",pd.read_parquet(f"{R}/{sub}/{r['sample']}.parquet").iloc[:,0])
O=pd.DataFrame(obs,columns=["cell","src","kind","l","y","cov"]); O["y"]=O.y.clip(1e-3,1-1e-3)
cells=sorted(O.cell.unique()); srcs=sorted(O.src.unique()); ci={c:i for i,c in enumerate(cells)}; si={s:i for i,s in enumerate(srcs)}
O["c"]=O.cell.map(ci); O["k"]=O.src.map(si)
print(f"cells {len(cells)} | sources {srcs} | loci {NL} | observations {len(O):,}", flush=True)
arr_src=[si[x] for x in srcs if O[O.src==x].kind.iloc[0]=="array"]
c_=jnp.array(O.c.values); l_=jnp.array(O.l.values); k_=jnp.array(O.k.values); y_=jnp.array(O.y.values); cov_=jnp.array(O["cov"].values)
isarr=jnp.array((O.kind=="array").values); ispool=jnp.array((O.kind=="pooled").values)
NC,NS=len(cells),len(srcs); AM=jnp.zeros(NS).at[jnp.array(arr_src)].set(1.)
def model():
    nu=numpyro.sample("nu",dist.Normal(0.,2.).expand([NL]))
    tau=numpyro.sample("tau",dist.HalfNormal(2.))
    lmu=numpyro.sample("lmu",dist.Normal(nu[None,:],tau).expand([NC,NL]))   # centred: every cell is data-rich
    mu=jax.nn.sigmoid(lmu); numpyro.deterministic("mu",mu)
    lsd=numpyro.sample("lsd",dist.Normal(jnp.log(0.03),0.7).expand([NC,NL])); sd=jnp.exp(lsd); numpyro.deterministic("sd",sd)
    a=numpyro.sample("a",dist.Normal(0.,0.1).expand([NS])); b=numpyro.sample("b",dist.Normal(1.,0.1).expand([NS]))
    d_raw=numpyro.sample("d",dist.Normal(0.,0.05).expand([NS])); t=numpyro.sample("t",dist.HalfNormal(0.05).expand([NS]))
    d=(d_raw-jnp.sum(d_raw*AM)/jnp.sum(AM))*AM            # array-lab offsets, sum to zero over array sources
    m=mu[c_,l_]; s_=sd[c_,l_]
    loc=jnp.where(isarr, m+d[k_], a[k_]+b[k_]*m)
    FLOOR=0.01**2
    var_a=s_**2+t[k_]**2+FLOOR
    var_w=(b[k_]**2)*jnp.where(ispool,s_**2/3.,s_**2) + m*(1-m)/jnp.maximum(cov_,1.) + t[k_]**2 + FLOOR
    numpyro.sample("y",dist.Normal(loc,jnp.sqrt(jnp.where(isarr,var_a,var_w))),obs=y_)
import scipy.special as ssp
lm=O.groupby("l").y.mean().reindex(range(NL)).fillna(0.5).clip(0.02,0.98).values
cm=O.groupby(["c","l"]).y.mean().unstack().reindex(index=range(NC),columns=range(NL))
z0=(ssp.logit(cm.clip(0.02,0.98).fillna(pd.DataFrame(np.tile(lm,(NC,1))))).values-ssp.logit(lm)[None,:])/1.0
from numpyro.infer import init_to_value
INIT=init_to_value(values={"nu":jnp.array(ssp.logit(lm)),"tau":jnp.array(1.0),"lmu":jnp.array(ssp.logit(lm)[None,:]+z0),"lsd":jnp.full((NC,NL),jnp.log(0.03)),
    "a":jnp.zeros(NS),"b":jnp.ones(NS),"d":jnp.zeros(NS),"t":jnp.full(NS,0.01)})
t0=time.time()
mc=MCMC(NUTS(model,target_accept_prob=0.9,init_strategy=INIT,max_tree_depth=8),num_warmup=int(os.environ.get("WARM","500")),num_samples=int(os.environ.get("DRAWS","500")),num_chains=4,chain_method="parallel",progress_bar=False)
mc.run(jax.random.PRNGKey(SEED)); jax.block_until_ready(mc.get_samples()); wall=time.time()-t0
from numpyro.diagnostics import summary
sm=summary(mc.get_samples(group_by_chain=True),group_by_chain=True)
def worst(name): v=sm[name]; return float(np.nanmax(v["r_hat"])), float(np.nanmin(v["n_eff"]))
P=mc.get_samples()
G=mc.get_samples(group_by_chain=True); sm_mu=summary({"mu":G["mu"]},group_by_chain=True)["mu"]
rep={"mu_rhat_max":float(np.nanmax(sm_mu["r_hat"])),"mu_rhat_p99":float(np.nanpercentile(sm_mu["r_hat"],99)),"mu_ess_min":float(np.nanmin(sm_mu["n_eff"])),"mu_ess_p01":float(np.nanpercentile(sm_mu["n_eff"],1)),
     "cells":cells,"sources":srcs,"loci":NL,"observations":int(len(O)),"wall_s":round(wall),"s_per_locus":round(wall/NL,3),
     "rhat_max":{n:worst(n)[0] for n in ("lmu","lsd","a","b","d","t","nu")},"ess_min":{n:worst(n)[1] for n in ("lmu","lsd","a","b","d","t","nu")},
     "divergences":int(np.asarray(mc.get_extra_fields()["diverging"]).sum()),
     "source_terms":{s:{"a":float(P["a"][:,si[s]].mean()),"b":float(P["b"][:,si[s]].mean()),"d":float(P["d"][:,si[s]].mean()),"t":float(P["t"][:,si[s]].mean())} for s in srcs},
     "median_donor_sd":{c:float(np.median(P["sd"][:,ci[c],:].mean(0))) for c in cells},
     "full_run_estimate_hours_single_box":round(wall/NL*450000/3600,1)}
json.dump(rep,open("dryrun_report.json","w"),indent=1)
mu=np.asarray(P["mu"].mean(0)); pd.DataFrame(mu,index=cells,columns=loci).T.to_parquet("dryrun_mu.parquet")
print(json.dumps({k:rep[k] for k in ("wall_s","s_per_locus","mu_rhat_max","mu_rhat_p99","mu_ess_min","mu_ess_p01","rhat_max","ess_min","divergences","source_terms","full_run_estimate_hours_single_box")},indent=1),flush=True)
