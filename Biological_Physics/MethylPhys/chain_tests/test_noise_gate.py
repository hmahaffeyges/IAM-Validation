#!/usr/bin/env python3
"""Stage T / noise gate (2026-10-02): the gauge state is withheld when the array's noise index N exceeds N_max and the reading is
untared; a tared reading keeps its state; nothing is fitted. Runs run_neutrophil on a synthetic isolated-neutrophil beta vector
(identity sites at the reference mean beta-equivalent, noise sites set to a chosen beta)."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chain")); import conductor_v3 as C
F = json.load(open(os.path.join(C.RM, "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
NS = C.noise_sites()["sites"]; Nmax = C.noise_gate()["N_max"]
def beta(noise_beta):
    idx = list(F["sites"]) + list(NS) + [f"cgfill{i:06d}" for i in range(760000)]
    b = pd.Series(0.9, index=idx, dtype="float64"); b.loc[NS] = noise_beta; return b
ok = True
for nb, refs, expect in ((0.005, None, "pass"), (0.06, None, "withheld"), (0.06, [1.0, 1.0, 1.0], "tared")):
    m = C.run_neutrophil(beta(nb), specimen="isolated neutrophils", ref_A=refs)
    mm = m.get("met_a") or m; st = mm.get("state", ""); N = mm.get("noise_index")
    got = "withheld" if st.startswith("withheld") else ("tared" if st.startswith("tared") else "pass")
    print(f"noise beta {nb}: N = {N} (N_max {Nmax}), refs {refs}: state '{st[:70]}' -> {got} (expected {expect})"); ok &= got == expect
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
