#!/bin/bash
# Atlas v2 stage B, all 700 blocks (~814k loci), PAR blocks at a time x 4 chains (32 x 4 = 128 cores). A finished block is skipped, so the run
# can be restarted after any interruption and loses only the blocks in flight. Per-block logs in logs/; summary at the end.
# 4 chains per process on 4 virtual devices, ONE thread each: PAR processes x 4 = all cores. Without the caps each process started
# its own XLA/BLAS thread pools and the first launch (2026-09-28) ran at load 143 on 32 cores.
export XLA_FLAGS="--xla_force_host_platform_device_count=4 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
seq 0 699 | xargs -P ${PAR:-32} -I{} sh -c 'BLOCK={} NBLOCK=700 ~/mcmc/bin/python 14_stageB_block.py > logs/block_{}.log 2>&1; tail -1 logs/block_{}.log'
~/env/bin/python - <<'PY'
import json, glob
R=[]
for f in glob.glob("logs/block_*.log"):
    L=[l for l in open(f) if l.startswith('{"block"')]
    if L: R.append(json.loads(L[-1]))
done=len(glob.glob("/home/ubuntu/data/atlas_v2/blocks_v2/block_*.parquet"))
S={"blocks_logged":len(R),"blocks_on_disk":done,"loci":sum(r["loci"] for r in R),"divergences":sum(r["divergences"] for r in R),
   "worst_rhat_p99":max(r["rhat_p99"] for r in R) if R else None,"worst_ess_p01":min(r["ess_p01"] for r in R) if R else None,
   "blocks_rhat_p99_over_1.01":sum(r["rhat_p99"]>1.01 for r in R),"median_s_per_locus":sorted(r["s_per_locus"] for r in R)[len(R)//2] if R else None}
# DISTINCTNESS GATE (IAMAtlas_FLATNESS_LESSON.md): judge the atlas by its OUTPUT, never by R-hat. On measured pairs only, every cell
# pair's mean |difference| over shared loci (>= 200). FAIL if any pair < 0.005. Adjacent lineages (~0.01-0.02) are correct biology.
import pandas as pd, numpy as np, itertools
F=sorted(glob.glob("/home/ubuntu/data/atlas_v2/blocks_v2/block_*.parquet"))[::7]
D=pd.concat([pd.read_parquet(f,columns=["cell","cpg","mu","n_obs"]) for f in F]); D=D[D.n_obs>0]
M=D.pivot(index="cpg",columns="cell",values="mu"); res=[]
for a,b in itertools.combinations(M.columns,2):
    k=M[a].notna()&M[b].notna()
    if k.sum()>=200: res.append((a,b,float((M[a][k]-M[b][k]).abs().mean()),float(np.corrcoef(M[a][k],M[b][k])[0,1]),int(k.sum())))
T=pd.DataFrame(res,columns=["a","b","mean_abs_diff","r","n"]).sort_values("mean_abs_diff"); T.to_csv("distinctness.csv",index=False)
cov=(D.groupby("cell").cpg.nunique()/M.shape[0]).round(4).to_dict()
S["distinctness"]={"pairs":len(T),"near_identical_lt_0.005":int((T.mean_abs_diff<0.005).sum()),"median":round(float(T.mean_abs_diff.median()),4),
   "closest":T.head(5).round(4).to_dict("records"),"PASS":bool((T.mean_abs_diff>=0.005).all())}
S["measured_fraction_by_cell"]=cov
json.dump(S,open("stageB_summary.json","w"),indent=1); print(json.dumps({k:v for k,v in S.items() if k!="measured_fraction_by_cell"}))
PY
