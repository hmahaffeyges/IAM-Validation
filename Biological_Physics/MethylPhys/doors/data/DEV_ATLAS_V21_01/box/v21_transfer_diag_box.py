#!/usr/bin/env python3
"""DIAGNOSIS, not a test (2026-10-14): why the bounded transfer missed out of sample in L22. Nothing here is adopted or scored. Per cell
that sequencing and arrays both measured (run12's own pairs on the final table, the same 50,000 loci), per L20 bin with >= 200 pairs:
mean y against g(median x), (i) on run12's raw array mean, and (ii) with each array laboratory's offset d (terms_final_v21.json) removed
before averaging. Also each cell's array laboratories, so lab and cell effects can be told apart. Writes transfer_diag_v21_cells.csv."""
import os, sys, json
sys.path.insert(0, os.getcwd())
from publish_code import publish_code
FILES = ["v21_transfer_diag_box.py", "v21_twin_terms_box.py", "transfer_bounded_v21.py", "transfer_bounded_v21.json", "final_table_v21.py", "publish_code.py", "terms_final_v21.json",
         "roster_samples.csv", "roster.csv", "hsc_manifest.csv", "twin_family_thresholds_v1.json", "v21_samples_all.csv", "atlas_v21_build_samples.csv",
         "conditions_v21.json", "twin_sample_level_v21.csv", "stage1_v21_manifest.csv"]
publish_code("results/ATLAS_V21/transfer_diag/code/run1/", FILES, job="atlas v2.1 transfer diagnosis (L22 follow-up)", commit=os.environ.get("REPO_COMMIT"))
src = open("v21_twin_terms_box.py").read()
exec(src[src.index("import os, sys, json, itertools"):src.index("sys.path.insert(0, os.getcwd())")])
exec(src[src.index("BUCKET = "):src.index("# ---------------- 09:")])
import final_table_v21 as FT, transfer_bounded_v21 as TB
F, _ = FT.build(".")
r12 = src[src.index("# ---------------- 12:"):src.index('S2 = pd.read_csv("roster_samples.csv")')]
h2 = next(l for l in r12.split("\n") if l.startswith("    cells = sorted({k[0] for k in M})"))
MD = {}; exec(r12.replace(h2, h2 + "\n    MD.update(M=M, arr_mean=arr_mean, cells=cells, ARR=ARR, SEQ=SEQ)"), globals())
O = run12(F, list(SUBV2) + sorted(set(F[F.why != "v2"].source)))
TF = json.load(open("terms_final_v21.json")); dlab = {k: v["d"] for k, v in TF["array"].items()}
TJ = json.load(open("transfer_bounded_v21.json")); kx, ky = np.array(TJ["knots_x"]), np.array(TJ["knots_y"]); g = lambda x: np.interp(x, kx, ky)
M, ARR = MD["M"], MD["ARR"]; rows = []
for k in MD["SEQ"]:
    for c in MD["cells"]:
        labs = [s for s in ARR if (c, s) in M]
        if (c, k) not in M or not labs: continue
        xr = pd.concat([M[(c, s)] for s in labs], axis=1).mean(axis=1); xd = pd.concat([M[(c, s)] - dlab.get(s, 0.0) for s in labs], axis=1).mean(axis=1)
        for name, x in (("raw", xr), ("lab_offset_removed", xd)):
            X = pd.DataFrame({"x": x, "y": M[(c, k)]}).dropna(); X = X.assign(bin=pd.cut(X.x, TB.EDGES, include_lowest=True))
            T = X.groupby("bin", observed=True).agg(n=("x", "size"), xm=("x", "median"), ym=("y", "mean")).reset_index(); T = T[T.n >= 200]
            for _, r in T.iterrows():
                rows.append(dict(source=k, cell=c, labs="+".join(labs), in_L21=c in TJ["cells"], x_kind=name, bin=str(r["bin"]), n=int(r.n), xm=r.xm, ym=r.ym, g=float(g(r.xm)), resid=float(r.ym - g(r.xm))))
P = pd.DataFrame(rows); P.to_csv("transfer_diag_v21_cells.csv", index=False)
log("rows", len(P)); s3.upload_file("transfer_diag_v21_cells.csv", BUCKET, "results/ATLAS_V21/transfer_diag/transfer_diag_v21_cells.csv"); open("DONE", "w").write("ok\n")
