#!/usr/bin/env python3
"""Atlas v2.1 speed diagnostic (2026-10-15; not a test, no rule): block 0's first NSUB loci fitted in MODE (v2 or v21), K 20, same seed,
so the per-locus time and the sampler's tree depth of v2's model and the v2.1 model can be compared on identical data. B1 run 3 ran
7 times over the cost sheet's per-locus estimate; this finds why before the full fit is sized. Writes speed_<mode>.json."""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
os.environ.setdefault("BLOCK", "0"); os.environ.setdefault("NBLOCK", "700"); os.environ["DRYRUN"] = ""
src = open("run_v21_block.py").read()
exec(src[src.index("import os, sys, json, time"):src.index("if os.environ.get(\"DRYRUN\"):")])
NSUB = int(os.environ.get("NSUB", "60")); MODE = os.environ["MODE"]
O = O[O.l < NSUB].reset_index(drop=True); sub = loci[:NSUB]
import atlas_block_v21 as AB
t0 = time.time(); RES = AB.fit_block(O, sub, TERMS, seed=1000, mode=MODE, K=20); w = time.time() - t0
out = {"mode": MODE, "loci": NSUB, "obs": len(O), "wall_s": round(w, 1), "s_per_locus": round(w / NSUB, 3), "warnings": RES["warnings"], "rhat_p99": RES["rhat_p99"]}
json.dump(out, open(f"speed_{MODE}.json", "w"), indent=1, default=float); print(json.dumps(out, default=float), flush=True)
