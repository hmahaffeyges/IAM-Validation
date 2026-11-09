"""Synthetic for one scale in the per-cell readings (run_sample.py --dev-one-scale; written and committed 2026-11-09 before it runs;
development). The instrument's job: a healthy blood measured through a laboratory's array line reads each cell's Met-A as it reads without
the line. Six constructed healthy bloods (blood_synthetic.py's construction, spent bloods 0 to 5's shares), each measured with no line, the
455 atlas donors' typical line (5 % and 95 % of their measured lines, per design) and their worst (the extremes); and with a 3 % neutrophil
loss planted (no line, worst line). Each read once by the one command with --dev-one-scale, which writes the one-scale per-cell readings and
the old self-tare path's beside them. PART p (0 to 11): blood p // 2, three readings, one at a time. Usage: one_scale_synthetic.py OUTDIR PART"""
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
LAB = {"none": None, "typical": {"I": (-0.0078, 1.0138), "II": (-0.0103, 1.0108)}, "worst": {"I": (0.0150, 0.9677), "II": (0.0361, 0.9301)}}   # the 455 atlas donors' measured lines: 5 %/95 % and the extremes
def bend(v, lab):
    if LAB[lab] is None: return v
    y = v.copy()
    for k, (a_, b_) in LAB[lab].items(): m = d == k; y[m] = a_ + b_ * v[m]
    return np.clip(y, 0, 1)
COND = [("none", True, False), ("typical", True, False), ("worst", True, False), ("none", True, True), ("worst", True, True), ("typical", True, True), ("none", True, False), ("none", True, False)][:6]
mine = COND[3 * (PART % 2): 3 * (PART % 2) + 3]
def read(cond):
    lab, cal, plant = cond; tag = f"os{b}_{lab}_{'cal' if cal else 'raw'}_{'plant' if plant else 'healthy'}"; wd = tempfile.mkdtemp()
    f = os.path.join(wd, "b.csv"); pd.DataFrame({"cpg_id": ids, "beta": bend(blood(plant), lab).round(4)}).to_csv(f, index=False); bf = os.path.join(wd, "b.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "whole blood", "--id", tag, "--out", os.path.join(wd, "r.html"), "--bundle", bf, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide"]
                       + ["--dev-one-scale"], capture_output=True, text=True)
    row = dict(blood=b, lab=lab, cal=cal, plant=plant, rc=r.returncode, err=r.stdout[-300:] if r.returncode else None)
    if r.returncode == 0:
        B = json.load(open(bf)); row["cells"] = [{k: c.get(k) for k in ("cell", "share", "met_a")} for c in (B.get("cells_found") or {}).get("cells") or []]; row["selftare_cells"] = [{k: c.get(k) for k in ("cell", "A", "ratio")} for c in (B.get("met_a_each_cell_selftare") or {}).get("cells") or []]; row["one_scale_cells"] = [{k: c.get(k) for k in ("cell", "A", "ratio")} for c in (B.get("met_a_each_cell") or {}).get("cells") or []]
    shutil.rmtree(wd, ignore_errors=True); print(tag, r.returncode, flush=True); return row
with ThreadPoolExecutor(int(os.environ.get("PAR", 1))) as ex: res = list(ex.map(read, mine))   # one at a time: three full-chain readings ran a 16 GB machine out of memory
json.dump(res, open(os.path.join(out, f"one_scale_synthetic_p{PART}.json"), "w"), indent=1)
