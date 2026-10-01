#!/usr/bin/env python3
"""Single-neutrophil sky maps, Met-A (development, 2026-10-01). Salas EPIC purified neutrophils (12, our Stage 1).
For a held-out specimen: reference = the other 11. Sites = neutrophil reference sites (across-donor SD <= 0.05, own mean beta 0.75-0.95 or
0.05-0.25). Per site: residual r = H(beta_specimen) - mean H(beta_reference); noise s = SD of H over the reference specimens (shrunk toward the
pooled SD, k = 10); z = r / s. Sky: all EPIC probes in genomic order -> HEALPix NSIDE 64 RING pixels (sequential, as Stage 4.6); pixel value =
sum z / sqrt(n) over measured sites in the pixel; pixels with no site are masked.
Maps: healthy held-out (H1, H2); healthy-minus-healthy null; known positive: 2 % blur everywhere; localized positive: 5 % blur only inside
10 contiguous genomic blocks (5 % of sites). Clustering statistic: variance of binned z (bins of 50 consecutive sites), = 1 for noise."""
import glob, json, numpy as np, pandas as pd
R="/home/ubuntu/data/atlas_sources"; NSIDE=64; NPIX=12*NSIDE*NSIDE; rng=np.random.default_rng(20261001)
S=pd.read_csv("roster_samples.csv"); S=S[(S.qc==True)&(S.cell=="neutrophils")&S.source.isin(["Salas2018","Salas2022"])]
D={"Salas2018":f"{R}/blood/GSE110554/shards","Salas2022":f"{R}/blood/GSE167998/shards"}
B=pd.concat([pd.read_parquet(glob.glob(f"{D[r.source]}/{r['sample'].split('_')[0]}_*.parquet")[0]).iloc[:,0].rename(r["sample"]) for _,r in S.iterrows()],axis=1)
M=pd.read_csv("/home/ubuntu/data/EPIC.hg38.manifest.gencode.v36.tsv.gz",sep="\t",usecols=["probeID","CpG_chrm","CpG_beg"]).dropna()
M=M[M.CpG_chrm.str.match(r"^chr([0-9]+|X|Y)$")]
ck=M.CpG_chrm.str.replace("chr","").replace({"X":"23","Y":"24"}).astype(int); M=M.assign(ck=ck).sort_values(["ck","CpG_beg","probeID"],kind="mergesort").reset_index(drop=True)
M["pix"]=(np.arange(len(M))*NPIX//len(M)).astype(int); M["order"]=np.arange(len(M)); M=M.set_index("probeID")
B=B.loc[B.index.intersection(M.index)].astype("float64")
Hb=lambda x: -(np.clip(x,1e-6,1-1e-6)*np.log2(np.clip(x,1e-6,1-1e-6))+(1-np.clip(x,1e-6,1-1e-6))*np.log2(1-np.clip(x,1e-6,1-1e-6)))
cols=list(B.columns); held=[cols[0],cols[-1]]          # one from each study
out={}; stats=[]
def sites_for(ref):
    mu=B[ref].mean(axis=1); sd=B[ref].std(axis=1)
    ok=(sd<=0.05)&(((mu>=0.75)&(mu<=0.95))|((mu>=0.05)&(mu<=0.25))); return ok[ok].index
def zmap(x,ref,idx):
    Hr=Hb(B.loc[idx,ref]); m=Hr.mean(axis=1); s=Hr.std(axis=1); n=len(ref); sp=np.sqrt(np.nanmedian(s**2))
    s2=np.sqrt(((n-1)*s**2+10*sp**2)/(n-1+10)); return (Hb(x.loc[idx])-m)/s2
def sky(z):
    p=M.loc[z.index,"pix"]; g=pd.DataFrame({"p":p.values,"z":z.values}).groupby("p").z.agg(["sum","size"])
    v=np.full(NPIX,np.nan); v[g.index.values]=(g["sum"]/np.sqrt(g["size"])).values; return v
def clus(z,w=50):
    o=z.reindex(M.loc[z.index].sort_values("order").index).dropna().values; nb=len(o)//w
    b=o[:nb*w].reshape(nb,w).mean(axis=1)*np.sqrt(w); return float(np.var(b)/np.var(o))   # 1 = no clustering beyond site-level scatter
def blur(x,e): return x+e*(0.5-x)
for h in held:
    ref=[c for c in cols if c!=h]; idx=sites_for(ref); x=B[h]
    z0=zmap(x,ref,idx); A_site=float(Hb(x.loc[idx]).mean()/Hb(B.loc[idx,ref]).mean(axis=1).mean())
    z2=zmap(blur(x,0.02),ref,idx)
    # localized: 10 contiguous blocks covering 5 % of the sites
    o=M.loc[idx].sort_values("order").index; L=len(o); bl=int(0.005*L); starts=rng.choice(L-bl,10,replace=False)
    loc=pd.Index(np.unique(np.concatenate([o[s:s+bl] for s in starts]))); xl=x.copy(); xl.loc[loc]=blur(x.loc[loc],0.05)
    zl=zmap(xl,ref,idx)
    A=lambda xx: float(Hb(xx.loc[idx]).mean()/Hb(B.loc[idx,ref]).mean(axis=1).mean())
    tag=h.split("_")[0]
    for nm,z,xx in (("healthy",z0,x),("blur2pct",z2,blur(x,0.02)),("local5pct_blocks",zl,xl)):
        out[f"{tag}_{nm}"]=sky(z); stats.append(dict(specimen=tag,map=nm,n_sites=len(idx),A=A(xx),z_mean=float(z.mean()),z_sd=float(z.std()),frac_z_gt3=float((z.abs()>3).mean()),clustering=clus(z)))
# healthy-minus-healthy null: two held-out healthy maps on their common sites, difference / sqrt(2)
a,b=held; ref=[c for c in cols if c not in held]; idx=sites_for(ref)
zd=(zmap(B[a],ref,idx)-zmap(B[b],ref,idx))/np.sqrt(2); out["null_healthy_minus_healthy"]=sky(zd)
stats.append(dict(specimen="H1-H2",map="null",n_sites=len(idx),A=np.nan,z_mean=float(zd.mean()),z_sd=float(zd.std()),frac_z_gt3=float((zd.abs()>3).mean()),clustering=clus(zd)))
np.savez_compressed("sky_neut_maps.npz",nside=NSIDE,**out); T=pd.DataFrame(stats); T.to_csv("sky_neut_stats.csv",index=False)
print("neutrophils",len(cols),"| EPIC probes on sky",len(M)); print(T.round(4).to_string(index=False)); print("DONE")
