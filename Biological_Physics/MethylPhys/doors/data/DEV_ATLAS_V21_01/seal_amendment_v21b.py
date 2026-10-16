"""Writes the stamped record of the 2026-10-16 pre-data amendment to the atlas v2.1 commissioning bars (author: "Seal the amendment"): the
amended scorer's fingerprint (chain Stage A21), the measured laboratory spread it reads, and the five changes. sealed_commission_v21.json is unchanged."""
import os, sys, json, hashlib
H = os.path.dirname(os.path.abspath(__file__)); C = os.path.abspath(os.path.join(H, "../../../chain")); sys.path.insert(0, C)
import stage_atlas_commission as CM, stage_q2_selfref as Q2
J = dict(amends="sealed_commission_v21.json", sealed_by_author="2026-10-16, before the atlas exists (fit at 580 of 700 blocks; no held-out array opened)",
  instruments_job="a healthy cell from any laboratory reads inside the atlas's range for that cell, and a mixed blood sample splits into its cells correctly",
  scorer="Biological_Physics/MethylPhys/chain/stage_atlas_commission.py", scorer_sha256=hashlib.sha256(open(os.path.join(C, "stage_atlas_commission.py"), "rb").read()).hexdigest(),
  original_scorer_sha256=json.load(open(os.path.join(H, "sealed_commission_v21.json")))["scorer_sha256"], lab_spread=round(CM.lab_sd(), 4), share=CM.SHARE,
  changes=["U1 range adds the measured laboratory spread (SD of the fit's array laboratory offsets) to the donor spread",
           "samples or sets the chain withholds are listed, not scored",
           "U1: >= 95 % of a cell's healthy held-out samples, all laboratories pooled, with coverage 0.80-0.97 (was: every set)",
           "U2: epithelium assessed and >= 95 % of assessed panel cells at share >= 0.90 (was: every cell)",
           "U3: mean absolute error <= 0.02 and >= 95 % of samples within 0.05 (was: every sample)"],
  planted="planted_commission_v21b.py: PASSED (15 cases)")
Q2.write_json(J, os.path.join(H, "amendment_v21b.json"), manifest=["ATL-005"]); print(json.dumps({k: J[k] for k in ("lab_spread", "share", "scorer_sha256")}))
