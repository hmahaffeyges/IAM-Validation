#!/usr/bin/env python3
"""Readiness L8, decision 10 for GSE137716 (2026-10-14), committed BEFORE it runs: export the 904 EpiDISH sites for its 12 conditional build
arrays. Rule in l8_conditions_v21.md (addendum). Uses the twin/terms job's loaders unchanged."""
import os, sys
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
publish_code("results/ATLAS_V21/l8b/code/run2/", ["v21_l8b_box.py", "v21_twin_terms_box.py", "publish_code.py", "v21_samples_all.csv", "epidish_sites.txt", "atlas_v21_build_samples.csv", "roster_samples.csv", "roster.csv", "hsc_manifest.csv", "twin_family_thresholds_v1.json"],
             job="atlas v2.1 L8 D10 GSE137716 export", commit=os.environ.get("REPO_COMMIT"))
src = open("v21_twin_terms_box.py").read()
exec(src[src.index("import os, sys, json, itertools"):src.index("sys.path.insert(0, os.getcwd())")])
exec(src[src.index("BUCKET = "):src.index("# ---------------- 09:")])
S21 = pd.read_csv("v21_samples_all.csv", keep_default_na=False, dtype={"path": str}); S21["qc"] = S21.qc.astype(str).eq("True")
BL = pd.read_csv("atlas_v21_build_samples.csv"); cond = set(BL[(BL.series == "GSE137716") & (BL.state == "CONDITIONAL")]["sample"].str.split("_").str[0])
x = S21[(S21.source == "GSE137716") & S21.qc & S21["sample"].str.split("_").str[0].isin(cond)]
log("GSE137716 conditional qc arrays", len(x), "of", len(cond))
sites = [l.strip() for l in open("epidish_sites.txt") if l.strip()]
E = pd.concat([arr_load(r).reindex(sites).rename(f"{r.source}|{r.cell}|{r['sample']}") for _, r in x.iterrows()], axis=1)
E.to_csv("l8b_epidish_betas.csv"); log("export", E.shape)
s3.upload_file("l8b_epidish_betas.csv", BUCKET, "results/ATLAS_V21/l8b/l8b_epidish_betas.csv"); open("DONE", "w").write("ok\n")
