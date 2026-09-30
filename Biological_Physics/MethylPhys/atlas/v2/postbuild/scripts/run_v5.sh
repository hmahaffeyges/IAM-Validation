#!/bin/bash
# PROC-V5-HELDOUT: 20 blocks in parallel, each pinned to 4 cores, one thread per chain.
export XLA_FLAGS="--xla_force_host_platform_device_count=4 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p v5_out logs
echo 18 54 68 122 123 208 244 250 254 320 437 443 455 488 501 547 574 580 598 628 | tr ' ' '\n' | xargs -P 20 --process-slot-var=SLOT -I{} sh -c 'taskset -c $((SLOT*4))-$((SLOT*4+3)) env BLOCK={} NBLOCK=700 ~/mcmc/bin/python v5_heldout.py > logs/v5_{}.log 2>&1; tail -1 logs/v5_{}.log; python3 -c "import json,subprocess;u=json.load(open(\"urls.json\"))[\"v5_put\"][\"{}\"];subprocess.run([\"curl\",\"-sS\",\"-f\",\"-X\",\"PUT\",\"-T\",\"v5_out/v5_block_%05d.json\" % {},u])"'
tar czf v5_out.tgz v5_out logs
