"""Synthetic for 'a Normal for each cell inside a whole blood, from the atlas alone' (written and committed 2026-11-08 before it runs). Known answer:
every constructed blood is healthy, so each cell found reads Normal. A person: for each of the six blood cells (neutrophils, monocytes, CD4 T,
CD8 T, B, NK) one sorted array planted from that cell's IAM-Atlas v2.1 column (mu + donor_sd x a standard normal per site, half shared by every
site of the person), with the array's line and noise (LINE, the width synthetic's center). The person's blood: the six arrays mixed site by site
in the shares the IAM-Deconvolver found in one of the spent 100 healthy bloods, each subtype counted under its parent cell
(blood_shares_parent_map.csv) and the six scaled to 1 (blood_shares_100.csv, row = blood index), read by the one command
as whole blood (--full-chain --cell-panel atlas --dev-line-wide). MODE donors: 100 bloods that give each cell's Normal; MODE new: 100 bloods of
new people (other seeds), scored. PART p reads bloods 13p to 13p + 12 (8 parts; blood index below 100).
Usage: python3 blood_synthetic.py OUTDIR MODE PART"""
import os, sys, json, subprocess, tempfile, shutil
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
out, MODE, PART = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(out, exist_ok=True)
SIX = ["neutrophils", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells"]; LINE = {"I": (-0.0033, 1.0026, 0.0066), "II": (-0.0121, 1.0175, 0.0098)}
SH = pd.read_csv(os.path.join(H, "blood_shares_100.csv"), index_col=0)[SIX]; des, _, _ = SA._ruler(); idx = des.index
COL = {c: (SA._col("mu", c).reindex(idx), SA._col("donor_sd", c).reindex(idx)) for c in SIX}
ok = np.ones(len(idx), bool)
for c in SIX: ok &= COL[c][0].notna().to_numpy()
ids = idx[ok]; d = des.reindex(ids).to_numpy(); S0 = {"donors": 7000, "new": 8000}[MODE]; res = []
for b in range(13 * PART, min(13 * PART + 13, 100)):
    g = np.random.default_rng(20261160 + S0 + b); x = np.zeros(len(ids))
    for c in SIX:
        mu, sd = COL[c][0].reindex(ids).to_numpy(), COL[c][1].reindex(ids); sd = sd.fillna(sd.median()).to_numpy()
        lev = np.clip(mu + sd * (np.sqrt(0.5) * g.standard_normal() + np.sqrt(0.5) * g.standard_normal(len(ids))), 0, 1); y = np.empty(len(ids))
        for k, (a, bb, s) in LINE.items(): m = d == k; y[m] = a + bb * lev[m] + s * g.standard_normal(m.sum())
        x += SH.iloc[b][c] * np.clip(y, 0, 1)
    tag = f"bs{MODE[0]}{b}"; wd = tempfile.mkdtemp(); f = os.path.join(wd, tag + "_betas.csv"); pd.DataFrame({"cpg_id": ids, "beta": x.round(4)}).to_csv(f, index=False); bf = os.path.join(wd, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "whole blood", "--id", tag, "--out", os.path.join(wd, tag + ".html"), "--bundle", bf, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide"], capture_output=True, text=True)
    row = dict(blood=b, mode=MODE, rc=r.returncode, built={c: float(SH.iloc[b][c]) for c in SIX})
    if r.returncode == 0:
        B = json.load(open(bf)); row["cells"] = [{k: c.get(k) for k in ("cell", "share", "met_a")} for c in (B.get("cells_found") or {}).get("cells") or []]
    res.append(row); print(b, r.returncode, flush=True)
    shutil.rmtree(wd, ignore_errors=True)   # 2026-11-08: the report's side files stay out of the uploaded folder (393 MB a part before)
    json.dump(res, open(os.path.join(out, f"blood_synthetic_{MODE}_p{PART}.json"), "w"), indent=1)
