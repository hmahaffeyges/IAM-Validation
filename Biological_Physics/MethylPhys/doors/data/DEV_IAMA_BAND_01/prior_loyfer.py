"""COM-IAMA-054 prior reading: Loyfer's 153 healthy counts (already counted, DEV_CHAIN_Q2_01/out/) read by chain Stage Q2 reading(), shown to
the author before the band rule was sealed. Writes prior_loyfer.json (stamped). No Hansen count is read."""
import os, sys, json, glob, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(H, "../../../../.."))
sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
O = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "Biological_Physics/MethylPhys/doors/data/DEV_CHAIN_Q2_01/out/GSM*.json")))]
a = np.array([Q2.reading(o, 1)["A"] for o in O])
J = dict(samples=len(a), median=round(float(np.median(a)), 3), p2_5=round(float(np.percentile(a, 2.5)), 3), p97_5=round(float(np.percentile(a, 97.5)), 3),
         p5=round(float(np.percentile(a, 5)), 3), p95=round(float(np.percentile(a, 95)), 3), min=round(float(a.min()), 3), max=round(float(a.max()), 3),
         b_cells=[round(Q2.reading(o, 1)["A"], 3) for o in O if o["cell"] == "b cells"])
Q2.write_json(J, os.path.join(H, "prior_loyfer.json"), manifest=["COM-IAMA-054"]); print(json.dumps(J))
