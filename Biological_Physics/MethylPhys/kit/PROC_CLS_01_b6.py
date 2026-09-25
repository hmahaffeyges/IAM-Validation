#!/usr/bin/env python3
"""PROC-CLS-01 bar B6, run properly: immune A'' recomputed on all 318 arrays and compared to PROC-BAND-01.

The pre-registration fixed B6 as a NUMBER: "immune A'' on these arrays must be unchanged from PROC-BAND-01 to
1e-9". The outcome first asserted it informally instead ("no chain file touched"), which is a deferred check
reported as a pass, and the informal half was false in the same sitting - chain/cmb_tools.py was edited to
record this procedure's own register note. This runs the test the pre-registration actually specified.
"""
import json, lzma, os, pickle, sys
sys.path.insert(0, "iamrepo/Biological_Physics/MethylPhys/chain")
import cpg_conductor as C

REF = "iamrepo/Biological_Physics/MethylPhys/reference_data"
atlas = "atlas_work/IAMAtlasREBUILD.csv"
band = json.load(open(C._find("identity_band_v3.json")))
coh = band["_meta"]["cohorts"]
if isinstance(coh, str):
    import ast; coh = ast.literal_eval(coh)
zeros = {k.split("_")[0]: v.get("z_lab_full_cohort") for k, v in coh.items()}
prior = {r["gsm"]: r for r in json.load(open("handoff/band01_arrays.json"))["arrays"]}

dec_mod = C._load_module("walther_iam_deconvolver", C._find("walther_iam_deconvolver.py"))
dec = dec_mod.WaltherIAMDeconvolver(atlas, celltype_class_map=str(
    C._find("IAMAtlasREBUILD_celltype_to_class.json")))

worst, n, missing = 0.0, 0, 0
worst_gsm = None
for gse in ["GSE87571", "GSE42861", "GSE111629", "GSE125105"]:
    with lzma.open(os.path.join(REF, "stage1_betas_%s.pkl.xz" % gse), "rb") as f:
        df = pickle.load(f)
    for gsm in df.columns:
        p = prior.get(gsm)
        if not p or p["immune"].get("A_abs") is None:
            missing += 1
            continue
        beta = df[gsm].dropna().to_dict()
        beta_rm, lab_scale = C.stage_1s_scale_map(beta, "stage1_noob_450K")
        fr = dict(dec.deconvolve(beta).class_fractions)
        bi = C.stage_b_identity(beta_rm, {"class_fractions": fr}, p["age"], lab_scale, lab_zero=zeros[gse])
        now = bi["immune"].get("A_abs")
        if now is None:
            missing += 1
            continue
        d = abs(now - p["immune"]["A_abs"])
        n += 1
        if d > worst:
            worst, worst_gsm = d, gsm
print("B6: %d arrays compared, %d without a prior reading" % (n, missing))
print("    largest |A''(now) - A''(PROC-BAND-01)| = %.3e  on %s" % (worst, worst_gsm))
print("    bar 1e-9 -> %s" % ("MET" if worst <= 1e-9 else "NOT MET"))
json.dump({"n_compared": n, "n_missing": missing, "max_abs_diff": worst, "worst_gsm": worst_gsm,
           "bar": 1e-9, "met": bool(worst <= 1e-9)}, open("handoff/cls01_b6.json", "w"), indent=1)
