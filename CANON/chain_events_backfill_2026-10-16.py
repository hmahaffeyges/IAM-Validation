#!/usr/bin/env python3
"""One-time backfill (2026-10-16): chain results written before stamps carried manifest rows get their rows added to the stamp's manifest
field, with manifest_backfilled=true. The payload hash is untouched (the stamp is outside it); every payload hash is re-verified before and after.
The folder -> rows map is the only judgment here, and it is written out below."""
import os, sys, json, glob
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(R, "Biological_Physics/MethylPhys/chain")); import stage_q2_selfref as Q2
Q = "Biological_Physics/MethylPhys/doors/data/DEV_CHAIN_Q2_01/"; L = "Biological_Physics/MethylPhys/chain/Runtime Matrices/IAM_A_SelfRef/"
MAP = {Q + "out/*.json": ["OTH-057", "PHY-045"], Q + "shape_out/*.json": ["OTH-057", "COM-IAMA-053"],
       Q + "q2_synthetic.json": ["OTH-057"], Q + "line_v1_cells.json": ["PHY-045"], Q + "line_v1_vs_development.json": ["PHY-045"],
       Q + "encode_vs_line_v1.json": ["PHY-047"], Q + "encode_shape_v1.json": ["COM-IAMA-053", "PHY-047"],
       L + "selfref_line_v*.json": ["PHY-045", "OTH-057"], L + "shape_reference_v1.json": ["COM-IAMA-053"],
       "Biological_Physics/MethylPhys/doors/data/DEV_SECONDLAB_02/bar_secondlab02.json": ["PHY-047"]}
n = 0
for pat, rows in MAP.items():
    for f in glob.glob(os.path.join(R, pat)):
        o = json.load(open(f, encoding="utf-8")); st = o["chain_stamp"]
        assert Q2.payload_sha(o) == st["payload_sha256"], f
        if st.get("manifest"): continue
        st["manifest"] = sorted(rows); st["manifest_backfilled"] = True
        assert Q2.payload_sha(o) == st["payload_sha256"], f
        json.dump(o, open(f, "w", encoding="utf-8"), indent=1, default=float); n += 1
print("backfilled", n)
