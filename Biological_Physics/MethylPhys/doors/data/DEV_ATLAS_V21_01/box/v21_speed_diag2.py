#!/usr/bin/env python3
"""Atlas v2.1 convergence diagnostic 2 (2026-10-15; diagnostic, no rule). Block 0's first 60 loci, same data, seed and sampler, three arms:
 A  v2's model (mode v2) with v2-form terms for every v2.1 source (terms_final_v21_run12.json: a, b per sequencing source; d per array lab)
 B  v2.1 model with the anchored transfer (transfer_v21_anchored.json)
 C  as B with the per-cell scatter s(m) set to zero
 S  v2.1 model with the smooth transfer (transfer_v21_smooth.json), 2026-10-15
Reports per arm: s per locus, share of draws at the maximum tree depth, divergences, R-hat p99. Writes diag2_<ARM>.json."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
os.environ.setdefault("BLOCK", "0"); os.environ.setdefault("NBLOCK", "700"); os.environ["DRYRUN"] = ""
src = open("run_v21_block.py").read()
exec(src[src.index("import os, sys, json, time"):src.index("if os.environ.get(\"DRYRUN\"):")])
ARM = os.environ["ARM"]; NSUB = 60
O = O[O.l < NSUB].reset_index(drop=True); sub = loci[:NSUB]
import atlas_block_v21 as AB
if ARM == "A":
    R12 = json.load(open("terms_final_v21_run12.json")); TT = {}
    for s_, v in R12["sequencing"].items(): TT[s_] = {"a": v["a"], "b": v["b"], "d": 0.0, "t": 0.0}
    for s_, v in R12["array"].items():
        # two labs have no direct offset: GSE116754 is assumed 0 and GSE137716 is measured via Loyfer, exactly as the v2.1 terms record them
        d_ = v["d"] if "d" in v else v.get("d_via_loyfer", 0.0)
        TT[s_] = {"a": 0.0, "b": 1.0, "d": d_, "t": 0.0}
    missing = sorted(set(O.src) - set(TT)); assert not missing, f"no v2-form term for {missing}"
    mode = "v2"
else:
    TT = json.load(open("transfer_v21_smooth.json" if ARM == "S" else "transfer_v21_anchored.json")); mode = "v21"   # ARM S: smooth transfer
    if ARM == "C": TT = dict(TT); TT["scatter_s"] = [0.0] * len(TT["scatter_s"])
t0 = time.time(); R = AB.fit_block(O, sub, TT, seed=1000, mode=mode, K=20); w = time.time() - t0
W = R["warnings"]; out = {"arm": ARM, "mode": mode, "loci": NSUB, "obs": len(O), "s_per_locus": round(w / NSUB, 3), "max_depth_share": W["draws_at_max_tree_depth"] / 1600,
                          "divergences": W["divergences"], "accept": W["accept_prob_mean"], "rhat_p99": R["rhat_p99"], "ess_p01": R["ess_p01"]}
json.dump(out, open(f"diag2_{ARM}.json", "w"), indent=1, default=float)
(R["summary"] if isinstance(R["summary"], pd.DataFrame) else pd.DataFrame(R["summary"])).to_parquet(f"diag2_{ARM}_summary.parquet"); print(json.dumps(out, default=float), flush=True)
