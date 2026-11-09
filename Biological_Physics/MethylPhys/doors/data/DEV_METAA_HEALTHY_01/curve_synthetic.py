"""Synthetic for each array's response curve on its own calibrators (run_sample.py --dev-curve-cal; written and committed 2026-11-08 before it
runs; development). The instrument's job: an array whose chemistry bends each probe design reads the same cells and the same Met-A as the same
blood measured without the bend. Six constructed healthy bloods (blood_synthetic.py's construction: the six cells planted from IAM-Atlas v2.1
with person spread and the array's line and noise, mixed in the shares of spent bloods 0 to 5, blood_shares_100.csv). Each blood measured
three ways: no bend; laboratory A (design II squeezed toward the middle with a bow, design I a smaller opposite bow); laboratory C (A's
bend times 7.5, sized to the real laboratories' top). Read by the one command (--full-chain --cell-panel atlas --dev-line-wide), with and without --dev-curve-cal; and with a 3 % loss
planted at the neutrophils' identity sites (each level moved 3 % of the way to 0.5), read unbent without the calibration and bent (A) with it.
PART p (0 to 11): blood p // 2, four of its eight readings, one at a time (three at once ran out of memory). Usage: curve_synthetic.py OUTDIR PART"""
import os, sys, json, subprocess, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
out, PART = sys.argv[1], int(sys.argv[2]); os.makedirs(out, exist_ok=True); b = PART // 2
SIX = ["neutrophils", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells"]; LINE = {"I": (-0.0033, 1.0026, 0.0066), "II": (-0.0121, 1.0175, 0.0098)}
SH = pd.read_csv(os.path.join(H, "blood_shares_100.csv"), index_col=0)[SIX]; des, _, _ = SA._ruler(); idx = des.index
COL = {c: (SA._col("mu", c).reindex(idx), SA._col("donor_sd", c).reindex(idx)) for c in SIX}
ok = np.ones(len(idx), bool)
for c in SIX: ok &= COL[c][0].notna().to_numpy()
ids = idx[ok]; d = des.reindex(ids).to_numpy(); NEU = np.isin(ids, list(SA.identity_sites("neutrophils")))
def blood(plant):
    g = np.random.default_rng(20261190 + b); x = np.zeros(len(ids))
    for c in SIX:
        mu, sd = COL[c][0].reindex(ids).to_numpy(), COL[c][1].reindex(ids); sd = sd.fillna(sd.median()).to_numpy()
        lev = np.clip(mu + sd * (np.sqrt(0.5) * g.standard_normal() + np.sqrt(0.5) * g.standard_normal(len(ids))), 0, 1)
        if plant and c == "neutrophils": lev = np.where(NEU, lev + 0.03 * (0.5 - lev), lev)
        y = np.empty(len(ids))
        for k, (a, bb, s) in LINE.items(): m = d == k; y[m] = a + bb * lev[m] + s * g.standard_normal(m.sum())
        x += SH.iloc[b][c] * np.clip(y, 0, 1)
    return x
def bend(v, lab):
    if lab == "none": return v
    # laboratory C: A's bend times 7.5, its design II projection on GMP's direction at the real laboratories' top (0.0621)
    s = {"A": 1, "B": -1, "C": 7.5}[lab]; y = v.copy(); m2 = d == "II"; m1 = d == "I"
    y[m2] = v[m2] + s * (-0.03 * (v[m2] - 0.5) * 2 * 0.5 + 0.10 * v[m2] * (1 - v[m2]) * (1 - 2 * v[m2]))
    y[m1] = v[m1] - s * 0.06 * v[m1] * (1 - v[m1]) * (1 - 2 * v[m1]); return np.clip(y, 0, 1)
COND = [("none", False, False), ("none", True, False), ("A", False, False), ("A", True, False), ("C", False, False), ("C", True, False), ("none", False, True), ("A", True, True)]
mine = COND[4 * (PART % 2): 4 * (PART % 2) + 4]
def read(cond):
    lab, cal, plant = cond; tag = f"cs{b}_{lab}_{'cal' if cal else 'raw'}_{'plant' if plant else 'healthy'}"; wd = tempfile.mkdtemp()
    f = os.path.join(wd, "b.csv"); pd.DataFrame({"cpg_id": ids, "beta": bend(blood(plant), lab).round(4)}).to_csv(f, index=False); bf = os.path.join(wd, "b.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "whole blood", "--id", tag, "--out", os.path.join(wd, "r.html"), "--bundle", bf, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide"]
                       + (["--dev-curve-cal"] if cal else []), capture_output=True, text=True)
    row = dict(blood=b, lab=lab, cal=cal, plant=plant, rc=r.returncode, err=r.stdout[-300:] if r.returncode else None)
    if r.returncode == 0:
        B = json.load(open(bf)); row["cells"] = [{k: c.get(k) for k in ("cell", "share", "met_a")} for c in (B.get("cells_found") or {}).get("cells") or []]; row["curve_cal"] = B.get("curve_cal")
    shutil.rmtree(wd, ignore_errors=True); print(tag, r.returncode, flush=True); return row
with ThreadPoolExecutor(int(os.environ.get("PAR", 1))) as ex: res = list(ex.map(read, mine))   # one at a time: three full-chain readings ran a 16 GB machine out of memory
json.dump(res, open(os.path.join(out, f"curve_synthetic_p{PART}.json"), "w"), indent=1)
