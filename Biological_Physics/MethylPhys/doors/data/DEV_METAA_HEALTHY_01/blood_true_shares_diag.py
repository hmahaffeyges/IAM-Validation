"""DIAGNOSTIC (2026-11-08, written and committed before it runs; development, nothing scored). Inside the rotation test's constructed bloods every
cell but the neutrophils reads about 1 % below 1, following the share the Deconvolver gives to cells not in the blood. This asks whether the
found shares cause it: the same bloods (blood_constructed.py MODE all, same seeds; 10 a part) are read by the one command, then each cell's
reading is made again with stage_metaa_atlas.cell_prediction twice, once with the shares the chain found and once with the true built shares,
the array's line (array_blur) measured with the same shares as the expectation. Output: per blood and cell, the chain's reading, the two remade.
Usage: python3 blood_true_shares_diag.py VALUES_DIR OUTDIR PART"""
import os, sys, json, subprocess, tempfile, shutil
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
sys.path.insert(0, CH); import stage_metaa_atlas as SA
SA.LINE_WIDE = True; SA.LINE_CENSORED = True   # as --dev-line-wide and the full chain set them
vd, out, PART = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(out, exist_ok=True)
SIX = ["neutrophils", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells"]
SH = pd.read_csv(os.path.join(H, "blood_shares_100.csv"), index_col=0)[SIX]; HV = pd.read_csv(os.path.join(H, "blood_donors_halves.csv"))
POOL = {c: [g for g in HV[HV.cell == c].gsm if os.path.exists(os.path.join(vd, g + "_betas.parquet"))] for c in SIX}
des, sets, _ = SA._ruler(); idx = des.index; cache = {}
def val(g):
    if g not in cache:
        b = pd.read_parquet(os.path.join(vd, g + "_betas.parquet")).iloc[:, 0]; b.index = b.index.astype(str); cache[g] = b
    return cache[g]
def remade(each, x, shares):
    F = {k: v for k, v in shares.items() if v}; MU = pd.concat([SA._col("mu", k).reindex(idx).rename(k) for k in F], axis=1); Fs = pd.Series(F)
    e = (MU.fillna(0).mul(Fs, axis=1).sum(1) / MU.notna().mul(Fs, axis=1).sum(1)).reindex(x.index)
    bl = SA.array_blur(x, e); E = json.loads(json.dumps(each)); E = SA.cell_prediction(E, x, F, bl)
    return {r["cell"]: r.get("ratio") for r in E["cells"]}
RF = os.path.join(out, f"true_shares_diag_p{PART}.json"); res = json.load(open(RF)) if os.path.exists(RF) else []; done_ = {r["blood"] for r in res}
for b in range(10 * PART, 10 * PART + 10):
    if b in done_: print(b, "already read", flush=True); continue
    rg = np.random.default_rng(20261170 + 9000 + b); pick = {c: POOL[c][rg.integers(len(POOL[c]))] for c in SIX}
    M = pd.concat([val(pick[c]).rename(c) for c in SIX], axis=1).dropna(); built = {c: float(SH.iloc[b % 100][c]) for c in SIX}; x = (M * SH.iloc[b % 100][SIX].values).sum(1)
    wd = tempfile.mkdtemp(); f = os.path.join(wd, "b.csv"); pd.DataFrame({"cpg_id": x.index, "beta": x.round(4).values}).to_csv(f, index=False); bf = os.path.join(wd, "b.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "whole blood", "--id", f"td{b}", "--out", os.path.join(wd, "b.html"), "--bundle", bf, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide"], capture_output=True, text=True)
    row = dict(blood=b, rc=r.returncode, built=built)
    if r.returncode == 0:
        B = json.load(open(bf)); found = {c["cell"]: c.get("share") for c in (B.get("cells_found") or {}).get("cells") or []}
        x2 = pd.read_csv(f, index_col=0)["beta"]; x2.index = x2.index.astype(str); each = B.get("met_a_each_cell") or {}
        row.update(found=found, chain={c["cell"]: c.get("met_a") for c in (B.get("cells_found") or {}).get("cells") or [] if c["cell"] in SIX},
                   remade_found=remade(each, x2, found), remade_built=remade(each, x2, built))
    res.append(row); print(b, r.returncode, flush=True); shutil.rmtree(wd, ignore_errors=True)
    json.dump(res, open(RF, "w"), indent=1)
