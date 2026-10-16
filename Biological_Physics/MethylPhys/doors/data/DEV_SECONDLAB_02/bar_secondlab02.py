"""DEV-SECONDLAB-02 bar, computed by the chain before any Hansen count (author: 'Seal as drafted', 2026-10-15).
S1 bar = the 90th percentile of |held error - line| over Loyfer's 56 cell types, read through chain Stage Q2 from the stamped line (version 1)
and its stamped per-cell table. Writes bar_secondlab02.json (stamped). Score: python score_secondlab02.py (written with the shape rule)."""
import os, sys, json, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../.."))
sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
J = Q2.line(1); T = json.load(open(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_CHAIN_Q2_01/line_v1_cells.json")))
assert Q2.payload_sha(T) == T["chain_stamp"]["payload_sha256"], "line_v1_cells.json edited after the chain wrote it"
res = np.array([c["eps"] - Q2.predict(c["L"], 1) for c in T["cells"]])
B = dict(rule="S1: at least 2 of the 3 Hansen donors read within +/- bar of the chain's line (version 1); the lab shift (Hansen minus Loyfer B cells) is reported, not scored; the sample-shape check (COM-IAMA-053) is applied first and a refused donor is not scored",
         bar=float(np.quantile(np.abs(res), 0.90)), bar_basis="90th percentile of |residual| over the line's 56 cell types", line_version=J["version"], cells=len(res), need=2, of=3)
Q2.write_json(B, os.path.join(H, "bar_secondlab02.json")); print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in B.items() if k != "rule"}))
