#!/usr/bin/env python3
"""Atlas v2.1 readiness B1 and L18 on the fit machine (committed 2026-10-14 before it runs).
B1: block 0 fitted alone with K = 100 saved draws, pinned to CORES = 8 cores with XLA_FLAGS exactly as run_v21.sh (4 chains on 4 devices, one thread each; v2 pinned 8 cores per block), under /usr/bin/time -v:
    wall time and peak memory -> par = min(cores // CORES, floor(0.80 * memory / peak)). Writes block_measure_v21.json.
L18: block 0 fitted again with K = 20 (same model, same seed 1000). The summary columns of the two saves must be IDENTICAL (the widened save changes
    only what is stored, never the fit). Writes outputs_v21_check_summary.txt. Also reports the block's file sizes, to size the disk for 700 blocks."""
import os, sys, json, subprocess, re, glob, time, pandas as pd, numpy as np
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
publish_code(f"results/ATLAS_V21/b1_l18/code/run{os.environ.get('RUN', '1')}/", ["v21_b1_l18_box.py", "run_v21_block.py", "atlas_block_v21.py", "transfer_v21_final.json", "final_samples_v21.csv", "publish_code.py"],
             job="atlas v2.1 B1 and L18 (one block, K 100 and K 20)", commit=os.environ.get("REPO_COMMIT"))
CORES = 8; NC = os.cpu_count(); MEM = int(open("/proc/meminfo").read().split()[1]) * 1024
def run(k, outd):
    os.makedirs(outd, exist_ok=True)
    env = dict(os.environ, BLOCK="0", NBLOCK="700", OUTD=outd, K=str(k), OMP_NUM_THREADS="1", XLA_FLAGS="--xla_force_host_platform_device_count=4 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    t = time.time(); p = subprocess.run(["/usr/bin/time", "-v", "taskset", "-c", f"0-{CORES-1}", os.path.expanduser("~/mcmc/bin/python"), "run_v21_block.py"],
                                        env=env, capture_output=True, text=True)
    open(f"b1_K{k}.log", "w").write(p.stdout + "\n" + p.stderr)
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr).group(1)) * 1024
    return p.returncode, time.time() - t, rss
rc100, w100, r100 = run(100, "/home/ubuntu/data/atlas_v21/b1_K100"); print("K100", rc100, round(w100), r100 / 1e9, flush=True)
rc20, w20, r20 = run(20, "/home/ubuntu/data/atlas_v21/b1_K20"); print("K20", rc20, round(w20), r20 / 1e9, flush=True)
assert rc100 == 0 and rc20 == 0, "a block fit failed; see b1_K*.log"
sizes = {os.path.basename(f): os.path.getsize(f) for f in glob.glob("/home/ubuntu/data/atlas_v21/b1_K100/block_00000*")}
par = int(min(NC // CORES, np.floor(0.80 * MEM / r100)))
B = {"block": 0, "K": 100, "cores_per_process": CORES, "wall_s": round(w100, 1), "peak_rss_gb": round(r100 / 1e9, 2), "machine_cores": NC, "machine_mem_gb": round(MEM / 1e9, 1),
     "par": par, "rule": "par = min(cores // cores_per_process, floor(0.80 * memory / peak))", "block_bytes": sizes, "bytes_per_block": sum(sizes.values()),
     "disk_for_700_blocks_gb": round(sum(sizes.values()) * 700 / 1e9, 1), "est_fit_hours": round(w100 * 700 / par / 3600, 1)}
json.dump(B, open("block_measure_v21.json", "w"), indent=1)
a = pd.read_parquet("/home/ubuntu/data/atlas_v21/b1_K100/block_00000.parquet"); b = pd.read_parquet("/home/ubuntu/data/atlas_v21/b1_K20/block_00000.parquet")
same_cols = list(a.columns) == list(b.columns); ident = same_cols and a.shape == b.shape and a.equals(b)
diff = {c: float(np.nanmax(np.abs(a[c].astype(float).values - b[c].astype(float).values))) for c in a.columns if same_cols and a[c].dtype.kind in "fi"} if not ident else {}
L = ["Atlas v2.1 L18: block 0 fitted with K = 100 and K = 20 saved draws, same model and seed 1000",
     f"summary columns {list(a.columns)}", f"rows {a.shape[0]} and {b.shape[0]}",
     f"VERDICT: {'IDENTICAL' if ident else 'DIFFERENT'}" + ("" if ident else f" (largest differences {diff})"),
     f"K = 100 draws file {sizes.get('block_00000_draws.npz', 0)} bytes"]
open("outputs_v21_check_summary.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(json.dumps(B), flush=True)
import boto3
s3 = boto3.client("s3", region_name="us-west-2")
for f in ("block_measure_v21.json", "outputs_v21_check_summary.txt", "b1_K100.log", "b1_K20.log"): s3.upload_file(f, "methylphys-data-945451304272-us-west-2-an", "results/ATLAS_V21/b1_l18/" + f)
