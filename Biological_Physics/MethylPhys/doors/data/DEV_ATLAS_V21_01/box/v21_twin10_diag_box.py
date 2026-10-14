#!/usr/bin/env python3
"""Diagnostic for job 5115bd05 (2026-10-14): its reproduction gate matched v2's twin table, null and source terms exactly but not the
sample-level twin test (10_twin_test_sample_level.csv), and the job did not save its own v2-mode table. This runs ONLY that step in v2 mode,
with the code taken verbatim from the committed v21_twin_terms_box.py (its loader section and its run10, read from the file and executed,
not retyped), saves the table and a row-by-row comparison with v2's record. Computes nothing for v2.1."""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
publish_code("results/ATLAS_V21/terms/diag10/code/", ["v21_twin10_diag_box.py", "v21_twin_terms_box.py", "publish_code.py", "roster_samples.csv",
             "twins.csv", "10_twin_test_sample_level.csv", "twin_family_thresholds_v1.json"], job="atlas v2.1 sample-level twin diagnostic", commit=os.environ.get("REPO_COMMIT"))
src = open("v21_twin_terms_box.py").read()
exec(src[src.index("BUCKET = "):src.index("# ---------------- 09:")])
exec(src[src.index("# ---------------- 10:"):src.index("# ---------------- 12:")])
import pandas as pd, numpy as np
S2 = pd.read_csv("roster_samples.csv"); ref = pd.read_csv("twins.csv"); R10 = pd.read_csv("10_twin_test_sample_level.csv")
P = run10(S2[S2.qc], ref, faithful=True); P.to_csv("v2mode_twin_sample_level.csv", index=False)
M = R10.merge(P, on=["a", "b"], how="outer", suffixes=("_v2", "_now"), indicator=True)
cols = ["n_a", "n_b", "shared", "nonoverlap_sep", "twin"]
M["same"] = M._merge.eq("both") & np.all([pd.to_numeric(M[c + "_v2"], errors="coerce").fillna(-1).values == pd.to_numeric(M[c + "_now"].astype(float) if c != "twin" else M[c + "_now"].astype(float), errors="coerce").fillna(-1).values for c in cols], axis=0)
M.to_csv("v2mode_twin_sample_level_compare.csv", index=False)
log("rows v2", len(R10), "| rows now", len(P), "| identical rows", int(M.same.sum()))
for _, r in M[~M.same].iterrows(): log("DIFF", r.a, "|", r.b, "|", {c: (r.get(c + "_v2"), r.get(c + "_now")) for c in cols})
for f in ("v2mode_twin_sample_level.csv", "v2mode_twin_sample_level_compare.csv"): s3.upload_file(f, BUCKET, "results/ATLAS_V21/terms/diag10/" + f)
open("DONE", "w").write("ok\n")
