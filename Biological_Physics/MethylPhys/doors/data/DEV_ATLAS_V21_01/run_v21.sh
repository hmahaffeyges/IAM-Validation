#!/bin/bash
# Atlas v2.1 stage B, all 700 blocks (readiness L9; committed before launch). v2's 15_stageB_run_all.sh with run_v21_block.py: PAR blocks at a
# time, 4 chains per process on 4 virtual devices, one thread each, each process pinned to CORES cores. PAR and CORES come from
# block_measure_v21.json (B1); a finished block is skipped, so the run restarts after any interruption and loses only blocks in flight.
# At the end: the run summary and v2's distinctness gate (every cell pair's mean |difference| >= 0.005 on shared measured loci).
set -u
export XLA_FLAGS="--xla_force_host_platform_device_count=4 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUTD=${OUTD:-/home/ubuntu/data/atlas_v21/blocks_v21}; export OUTD; mkdir -p logs $OUTD
PAR=${PAR:-$(~/env/bin/python -c "import json;print(json.load(open('block_measure_v21.json'))['par'])")}
CORES=${CORES:-$(~/env/bin/python -c "import json;print(json.load(open('block_measure_v21.json'))['cores_per_process'])")}
echo "$(date -u +%FT%TZ) START par=$PAR cores=$CORES outd=$OUTD" | tee -a run_v21.log
seq 0 699 | xargs -P $PAR --process-slot-var=SLOT -I{} sh -c 'taskset -c $((SLOT*'$CORES'))-$((SLOT*'$CORES'+'$CORES'-1)) env BLOCK={} NBLOCK=700 ~/mcmc/bin/python run_v21_block.py > logs/block_{}.log 2>&1; tail -1 logs/block_{}.log' >> run_v21.log
~/env/bin/python - <<'PY'
import json, glob, os, itertools, numpy as np, pandas as pd
OUTD = os.environ["OUTD"]; R = []
for f in glob.glob("logs/block_*.log"):
    L = [l for l in open(f) if l.startswith('{"block"')]
    if L: R.append(json.loads(L[-1]))
S = {"blocks_logged": len(R), "blocks_on_disk": len(glob.glob(f"{OUTD}/block_*[0-9].parquet")), "loci": sum(r["loci"] for r in R), "divergences": sum(r["divergences"] for r in R),
     "worst_rhat_p99": max(r["rhat_p99"] for r in R) if R else None, "worst_ess_p01": min(r["ess_p01"] for r in R) if R else None,
     "blocks_rhat_p99_over_1.01": sum(r["rhat_p99"] > 1.01 for r in R), "median_s_per_locus": sorted(r["s_per_locus"] for r in R)[len(R) // 2] if R else None}
F = sorted(glob.glob(f"{OUTD}/block_*[0-9].parquet"))[::7]; D = pd.concat([pd.read_parquet(f, columns=["cell", "cpg", "mu", "n_obs"]) for f in F]); D = D[D.n_obs > 0]
M = D.pivot(index="cpg", columns="cell", values="mu"); res = []
for a, b in itertools.combinations(M.columns, 2):
    k = M[a].notna() & M[b].notna()
    if k.sum() >= 200: res.append((a, b, float((M[a][k] - M[b][k]).abs().mean()), float(np.corrcoef(M[a][k], M[b][k])[0, 1]), int(k.sum())))
T = pd.DataFrame(res, columns=["a", "b", "mean_abs_diff", "r", "n"]).sort_values("mean_abs_diff"); T.to_csv("distinctness_v21.csv", index=False)
S["distinctness"] = {"pairs": len(T), "near_identical_lt_0.005": int((T.mean_abs_diff < 0.005).sum()), "median": round(float(T.mean_abs_diff.median()), 4),
                     "closest": T.head(5).round(4).to_dict("records"), "PASS": bool((T.mean_abs_diff >= 0.005).all())}
S["measured_fraction_by_cell"] = (D.groupby("cell").cpg.nunique() / M.shape[0]).round(4).to_dict()
json.dump(S, open("stageB_v21_summary.json", "w"), indent=1); print(json.dumps({k: v for k, v in S.items() if k != "measured_fraction_by_cell"}))
PY
echo "$(date -u +%FT%TZ) END" | tee -a run_v21.log
