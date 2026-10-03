#!/usr/bin/env python3
"""probe_level_confounds.py -- run on the box next to the GSE223748 data.
Per array: recompute mean H(beta) on the primary and conserved panels (check against sample_stats_final.csv),
blood-composition proxies (mean beta over human sorted-cell lineage CpGs, GSE110554), detection rate.
Per species: probe-level between-animal SD in the young reference and the oldest-decile group (median over probes),
and the share of panel probes whose mean H is higher in the oldest-decile group than in the young reference.
Inputs (cwd): groups.csv, immune_marker_probes.csv, genome_assign.json, genome_probe_map.parquet,
detect_fraction_by_group.parquet, core_probes_final.txt.  DATA=/home/ubuntu/data/species_mmc"""
import os, json, numpy as np, pandas as pd, pyarrow.parquet as pq
D=os.environ.get('DATA','/home/ubuntu/data/species_mmc'); PTH=0.05; DFRAC=0.90
def Hb(b):
    b=np.clip(b,1e-12,1-1e-12); return -(b*np.log2(b)+(1-b)*np.log2(1-b))
G=pd.read_csv('groups.csv'); mk=pd.read_csv('immune_marker_probes.csv')
assign=json.load(open('genome_assign.json')); Gm=pd.read_parquet('genome_probe_map.parquet')
fr=pd.read_parquet('detect_fraction_by_group.parquet'); core=set(open('core_probes_final.txt').read().split())
pf=pq.ParquetFile(D+'/beta_all_float32.parquet'); names=pf.schema_arrow.names
idxcol=[n for n in names if not n.startswith('2')]
M=pq.read_table(D+'/beta_all_float32.parquet',columns=list(G.barcode)+idxcol).to_pandas()
if M.index.dtype!=object or not str(M.index[0]).startswith(('cg','ch','rs')):
    M=M.set_index(idxcol[0])
cg=np.array([p for p in M.index if str(p).startswith('cg')])
order=np.load(D+'/pdet/_probe_order.npy',allow_pickle=True)
pos=pd.Series(np.arange(len(order)),index=order).reindex(cg).values.astype(int)
Bm=M.loc[cg,G.barcode.values].values.astype(np.float64); del M
P=np.stack([np.load(f'{D}/pdet/{g}.npy')[pos] for g in G.gsm]).T
X=Bm.copy(); X[~(P<PTH)]=np.nan
print('beta',Bm.shape,flush=True)
cgi=pd.Index(cg); coremask=cgi.isin(list(core))
def gmask(files):
    m=np.ones(len(cg),bool)
    for f in files: m&=Gm.loc[f].reindex(cg).fillna(False).values.astype(bool)
    return m
mkidx={s:cgi.isin(mk.probe[mk.set==s]) for s in mk.set.unique()}
arr=[];spr=[]
for sp,g in G.groupby('species'):
    j=g.index.values; det=fr.loc[f'{sp}|Blood'].reindex(cg).values>=DFRAC
    a=assign[sp]; prim=det&(gmask(a['files']) if a['files'] else True); cor=coremask&det
    for tag,mask in [('primary',prim),('core',cor)]:
        Xs=X[np.ix_(mask,j)]; H=Hb(Xs)
        mh=np.nanmean(H,0)
        for jj,v in zip(j,mh): arr.append(dict(gsm=G.gsm[jj],panel=tag,mean_H_recomputed=v))
        ref=(g.role=='young_reference').values; old=g.oldest_decile.values.astype(bool)
        sd_ref=np.nanstd(Xs[:,ref],1,ddof=1); sd_old=np.nanstd(Xs[:,old],1,ddof=1)
        dH=np.nanmean(H[:,old],1)-np.nanmean(H[:,ref],1)
        # direction of beta change for probes moving toward 0.5 vs away
        mb_ref=np.nanmean(Xs[:,ref],1); mb_old=np.nanmean(Xs[:,old],1)
        spr.append(dict(species=sp,panel=tag,n_probes=int(mask.sum()),n_ref=int(ref.sum()),n_old=int(old.sum()),
            median_probe_sd_ref=float(np.nanmedian(sd_ref)),median_probe_sd_old=float(np.nanmedian(sd_old)),
            share_probes_H_higher_old=float(np.nanmean(dH>0)),mean_dH_old_minus_ref=float(np.nanmean(dH)),
            share_lowbeta_probes_gaining=float(np.nanmean((mb_old>mb_ref)[mb_ref<0.5])),
            share_highbeta_probes_losing=float(np.nanmean((mb_old<mb_ref)[mb_ref>=0.5]))))
    for s,mm in mkidx.items():
        use=mm&det
        v=np.nanmean(X[np.ix_(use,j)],0)
        for jj,vv in zip(j,v): arr.append(dict(gsm=G.gsm[jj],panel='marker_'+s,mean_H_recomputed=np.nan,marker_mean_beta=vv,n_marker_probes=int(use.sum())))
    print(sp,flush=True)
pd.DataFrame(arr).to_csv('per_array_probe_level.csv',index=False)
pd.DataFrame(spr).to_csv('per_species_probe_level.csv',index=False)
