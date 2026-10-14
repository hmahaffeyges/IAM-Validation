#!/usr/bin/env python3
"""Readiness L8 data step (2026-10-14), committed BEFORE it runs; rules in l8_conditions_v21.md. Uses the twin/terms job's own loaders,
run09 profile code and run10 twin test, unchanged (executed from v21_twin_terms_box.py). Writes l8_d1_twin.csv, l8_d7_profile_r.csv and
l8_epidish_betas.csv (the 904 EpiDISH sites for GSE224807, GSE146552 and the Salas neutrophils)."""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
publish_code("results/ATLAS_V21/l8/code/run2/", ["v21_l8_box.py", "v21_twin_terms_box.py", "publish_code.py", "roster_samples.csv", "roster.csv", "hsc_manifest.csv",
             "twin_family_thresholds_v1.json", "v21_samples_all.csv", "epidish_sites.txt"], job="atlas v2.1 L8 conditions data step", commit=os.environ.get("REPO_COMMIT"))
src = open("v21_twin_terms_box.py").read()
exec(src[src.index("import os, sys, json, itertools"):src.index("sys.path.insert(0, os.getcwd())")])
exec(src[src.index("BUCKET = "):src.index("# ---------------- 09:")])
r09 = src[src.index("def run09"):src.index("# ---------------- 10:")]
hook = "    admitted = [c for c in admitted_order if entry_ok(c) and c in prof]"; assert r09.count(hook) == 1
PROF = {}
exec(r09.replace(hook, "    PROF.update(prof); return None\n" + hook), globals())
r10 = src[src.index("def run10"):src.index("# ---------------- 12:")]; exec(r10, globals())
S21 = pd.read_csv("v21_samples_all.csv", keep_default_na=False, dtype={"path": str}); S21["qc"] = S21.qc.astype(str).eq("True")   # run 2: v2 rows have an empty path, kept as "" (run 1 read it as NaN)
ok = S21[S21.qc].copy()
# D7: profiles exactly as run09 builds them, the GSE306226 astrocyte fraction kept out of every profile and read on its own
ast = ok[(ok.source == "GSE306226") & (ok.cell == "astrocytes")]
run09(ok.drop(ast.index), [], lambda c: True)
xa = pd.concat([arr_load(r) for _, r in ast.iterrows()], axis=1).mean(axis=1)
rows = []
for cell, (pb, xb) in PROF.items():
    j = xa.dropna().index.intersection(xb.dropna().index)
    if len(j) < 1000: continue
    u, v = xa.loc[j].values, xb.loc[j].values
    if pb != "array": s, i = np.polyfit(u, v, 1); u = i + s * u
    rows.append(dict(cell=cell, platform=pb, shared=len(j), r=round(float(np.corrcoef(u, v)[0, 1]), 4)))
R7 = pd.DataFrame(rows).sort_values("r", ascending=False); R7.to_csv("l8_d7_profile_r.csv", index=False); log("D7 top", R7.head(3).to_dict("records"))
# D1: GSE224807 neutrophils against the Salas neutrophils, v2's sample-level twin test
d1 = ok[(ok.cell == "neutrophils") & ok.source.isin(["GSE224807", "Salas2018", "Salas2022"])].copy()
d1["cell"] = np.where(d1.source == "GSE224807", "neutrophils GSE224807", "neutrophils Salas")
ma = pd.concat([arr_load(r) for _, r in d1[d1.cell == "neutrophils GSE224807"].iterrows()], axis=1).mean(axis=1)
mb = pd.concat([arr_load(r) for _, r in d1[d1.cell == "neutrophils Salas"].iterrows()], axis=1).mean(axis=1)
j = ma.dropna().index.intersection(mb.dropna().index); r1 = float(np.corrcoef(ma[j], mb[j])[0, 1])
T1 = pd.DataFrame([dict(a="neutrophils GSE224807", b="neutrophils Salas", r=round(r1, 4))])
P1 = run10(d1, T1.assign(r=T1.r.clip(lower=0.975)), faithful=True).assign(r_measured=round(r1, 4), twin_r=TW["twin_r"], min_sep=SL)
P1["twin"] = (r1 > TW["twin_r"]) & (P1.nonoverlap_sep < SL); P1.to_csv("l8_d1_twin.csv", index=False); log("D1", P1.to_dict("records"))
# EpiDISH sites
sites = [l.strip() for l in open("epidish_sites.txt") if l.strip()]
ex = pd.concat([ok[ok.source.isin(["GSE224807", "GSE146552"])], d1[d1.source != "GSE224807"]]).drop_duplicates("sample")
E = pd.concat([arr_load(r).reindex(sites).rename(f"{r.source}|{r.cell}|{r['sample']}") for _, r in ex.iterrows()], axis=1)
E.to_csv("l8_epidish_betas.csv"); log("EpiDISH export", E.shape, "| sites present", int(E.notna().any(axis=1).sum()))
for f in ("l8_d1_twin.csv", "l8_d7_profile_r.csv", "l8_epidish_betas.csv"): s3.upload_file(f, BUCKET, "results/ATLAS_V21/l8/" + f)
open("DONE", "w").write("ok\n")
