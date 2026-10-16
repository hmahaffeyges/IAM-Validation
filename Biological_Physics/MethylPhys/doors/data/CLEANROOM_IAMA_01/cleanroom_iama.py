"""PHY-055 / IAM-A clean room, part A (2026-10-16). Run inside a FRESH clone of the public repository, from an empty working folder, with public
data only: Loyfer's .pat.gz files from GEO and ENCODE's BAMs and GRCh38 reference from ENCODE's public sources. No file from our S3 is read.
 1. count: the clone's own chain Stage Q2 and its own driver (DEV_CHAIN_Q2_01/box_q2_line.py), Loyfer 153 then ENCODE 5, into ./out
 2. compare every count with the committed one (DEV_CHAIN_Q2_01/shape_out/): every field except the stamp's time must be equal
 3. rebuild, with the chain, the self-reference line and the sample-shape reference from the clean counts, and compare with the committed
    chain/Runtime Matrices/IAM_A_SelfRef/selfref_line_v1.json and shape_reference_v1.json: firm, slope, cells, margin, every per-type minimum
Writes cleanroom_iama_result.json (stamped). Usage: python cleanroom_iama.py compare <clone> <counts folder>"""
import os, sys, json, glob
CL = os.path.abspath(sys.argv[2]); OUT = os.path.abspath(sys.argv[3])
sys.path.insert(0, os.path.join(CL, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
import pandas as pd
DQ = os.path.join(CL, "Biological_Physics/MethylPhys/doors/data/DEV_CHAIN_Q2_01")
TIME = {"started", "finished", "seconds", "utc", "wall_s", "elapsed"}
def strip(o): return {k: v for k, v in o.items() if k != "chain_stamp" and k not in TIME}
new = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(OUT, "*.json"))}
old = {os.path.basename(f): json.load(open(f)) for f in glob.glob(os.path.join(DQ, "shape_out", "*.json"))}
diff = sorted(k for k in old if k not in new or strip(new[k]) != strip(old[k]))
L = [o for k, o in new.items() if k.startswith("GSM")]
J, _ = Q2.build_line(pd.DataFrame(L)[["sample", "cell", "loose_share", "held_err", "held_opp"]]); R = Q2.build_shape_reference(L)
LC = json.load(open(os.path.join(Q2.LINE_DIR, "selfref_line_v1.json"))); SC = json.load(open(os.path.join(Q2.LINE_DIR, "shape_reference_v1.json")))
line_same = all(J[k] == LC[k] for k in ("firm", "slope", "cells")); shape_same = R["margin"] == SC["margin"] and R["cells"] == SC["cells"]
res = dict(counts=len(new), committed=len(old), differing=diff, line_identical=line_same, shape_reference_identical=shape_same,
           line=dict(firm=J["firm"], slope=J["slope"], cells=J["cells"]), margin=R["margin"],
           verdict="REPRODUCED" if (not diff and len(new) == len(old) and line_same and shape_same) else "DIFFERS")
Q2.write_json(res, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleanroom_iama_result.json")); print(json.dumps({k: v for k, v in res.items() if k != "differing"}), "differing:", diff[:10])
