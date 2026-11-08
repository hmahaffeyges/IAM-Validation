"""A Normal for each cell inside a whole blood, from the atlas alone (written and committed 2026-11-08 before it runs). The atlas's own 146
blood-cell donor arrays (at least 85 % one cell; their calibrated values kept by run_series --keep-values, series bv2_donors) are split at random,
cell by cell, into two halves (blood_donors_halves.csv, seed 20261108): 'normal' builds the donor bloods that give each cell's Normal, 'new' builds
the new bloods that are scored, so no array is in both. One blood: one array of each of the six cells drawn from its half, mixed site by site in
the shares of one of the spent 100 healthy bloods (blood_shares_100.csv; subtypes under their parent cell), read by the one command as whole blood.
Sites missing in any of the six arrays are left out. MODE normal or new; PART p reads bloods 13p to 13p + 12 (8 parts).
Usage: python3 blood_constructed.py VALUES_DIR OUTDIR MODE PART"""
import os, sys, json, subprocess
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
vd, out, MODE, PART = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); os.makedirs(out, exist_ok=True)
SIX = ["neutrophils", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells"]
SH = pd.read_csv(os.path.join(H, "blood_shares_100.csv"), index_col=0)[SIX]; HV = pd.read_csv(os.path.join(H, "blood_donors_halves.csv"))
POOL = {c: [g for g in HV[(HV.cell == c) & (HV.half == MODE)].gsm if os.path.exists(os.path.join(vd, g + "_betas.parquet"))] for c in SIX}
print({c: len(v) for c, v in POOL.items()}, flush=True); cache = {}
def val(g):
    if g not in cache:
        b = pd.read_parquet(os.path.join(vd, g + "_betas.parquet")).iloc[:, 0]; b.index = b.index.astype(str); cache[g] = b
    return cache[g]
res = []
for b in range(13 * PART, min(13 * PART + 13, 100)):
    rg = np.random.default_rng(20261170 + (0 if MODE == "normal" else 5000) + b); pick = {c: POOL[c][rg.integers(len(POOL[c]))] for c in SIX}
    M = pd.concat([val(pick[c]).rename(c) for c in SIX], axis=1).dropna(); x = (M * SH.iloc[b][SIX].values).sum(1)
    tag = f"bc{MODE[1]}{b}"; f = os.path.join(out, tag + "_betas.csv"); pd.DataFrame({"cpg_id": x.index, "beta": x.round(4).values}).to_csv(f, index=False); bf = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "whole blood", "--id", tag, "--out", os.path.join(out, tag + ".html"), "--bundle", bf, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide"], capture_output=True, text=True)
    row = dict(blood=b, mode=MODE, rc=r.returncode, sites=len(x), arrays=pick, built={c: float(SH.iloc[b][c]) for c in SIX})
    if r.returncode == 0:
        B = json.load(open(bf)); row["cells"] = [{k: c.get(k) for k in ("cell", "share", "met_a")} for c in (B.get("cells_found") or {}).get("cells") or []]
    res.append(row); print(b, r.returncode, len(x), flush=True)
    for ff in (f, bf, os.path.join(out, tag + ".html")):
        if os.path.exists(ff): os.remove(ff)
    json.dump(res, open(os.path.join(out, f"blood_constructed_{MODE}_p{PART}.json"), "w"), indent=1)
