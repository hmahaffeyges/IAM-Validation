"""THIRD DIAGNOSTIC (2026-11-08): MODE repeat with --dev-line-wide, the spread of one person read 20 times with fresh array noise, the
instrument's own measurement spread per reading on the line the chain now uses. The second diagnostic's text follows. SECOND DIAGNOSTIC of the width synthetic (2026-11-08; development, nothing scored): the no-noise arm read again, keeping each person's
raw and predicted Met-A of the main cell, the line the chain measured on the array (blur), the planted line, and the five largest shares,
to find which step adds spread that the planted people do not have. The first diagnostic's text follows: MODE nonoise reads the same 120 people with the array's line
but no random noise (if the chain is right, the readings' spread equals the truth's); MODE repeat reads person 0 of each cell and rho 20 times,
each with a fresh noise draw (that spread is the array's own noise alone). Usage: python3 width_diag.py OUTDIR CELL_INDEX RHO_INDEX FIRST MODE
The original synthetic, unchanged otherwise: each cell's healthy width from its donors, written and committed 2026-11-07 before it runs (the fourth step to a working chain;
development). Known answer: constructed sorted-cell arrays of 20 people per cell, each person's levels the cell's IAM-Atlas v2.1 level at every
site plus a planted person spread at the atlas's own measured donor spread there (donor_sd; its cell median where blank), either site by site
(rho 0) or half of it shared by the whole person (rho 0.5); through the real arrays' median line and noise per probe design; each read by
run_sample.py --betas --full-chain --cell-panel atlas. The truth for each person is that cell's Met-A on the person's noise-free levels
(stage_metaa_atlas.read); the reading is the Met-A of the cell the Deconvolver found most of (cells_found). Three cells (neutrophils, CD8 T
cells, B cells). Seed 20261150 + 10 * cell + rho index; persons FIRST..FIRST+4 per part.
Usage: python3 width_synthetic.py OUTDIR CELL_INDEX RHO_INDEX FIRST"""
import os, sys, json, subprocess
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
CELLS = ["neutrophils", "cd8 t cells", "b cells"]; RHOS = [0.0, 0.5]
out, ci, ri, first = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True); X, rho = CELLS[ci], RHOS[ri]; MODE = sys.argv[5]; assert MODE in ("nonoise", "repeat")
des, _, _ = SA._ruler(); idx = des.index
mu = SA._col("mu", X).reindex(idx).dropna(); sd = SA._col("donor_sd", X).reindex(mu.index); sd = sd.fillna(sd.median())
d = des.reindex(mu.index).to_numpy(); LINE = {"I": (-0.0033, 1.0026, 0.0066), "II": (-0.0121, 1.0175, 0.0098)}
g = np.random.default_rng(20261150 + 10 * ci + ri); Z = g.standard_normal(20); E = [g.standard_normal(len(mu)) for _ in range(20)]
res = []
for p in range(first, min(first + 5, 20)):
    pp = 0 if MODE == "repeat" else p
    lev = np.clip(mu.to_numpy() + sd.to_numpy() * (np.sqrt(rho) * Z[pp] + np.sqrt(1 - rho) * E[pp]), 0, 1); truth = SA.read(pd.Series(lev, index=mu.index), X).get("A")
    x = np.empty(len(lev)); gn = np.random.default_rng(20261150 + 1000 * ci + 100 * ri + p)
    for k, (a, b, s) in LINE.items():
        m = d == k; x[m] = a + b * lev[m] + (0.0 if MODE == "nonoise" else s) * gn.standard_normal(m.sum())
    tag = f"f{MODE[0]}{ci}r{ri}p{p}"; f = os.path.join(out, tag + "_betas.csv"); pd.DataFrame({"cpg_id": mu.index, "beta": np.clip(x, 0, 1).round(4)}).to_csv(f, index=False)
    bf = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "sorted cells", "--id", tag, "--out", os.path.join(out, tag + ".html"), "--bundle", bf, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide"], capture_output=True, text=True)
    os.remove(f); row = dict(mode=MODE, cell=X, rho=rho, person=p, truth=truth, rc=r.returncode)
    if r.returncode == 0:
        cf = (json.load(open(bf)).get("cells_found") or {}).get("cells") or []; top = max(cf, key=lambda c: c.get("share") or 0) if cf else {}
        row.update(main_cell=top.get("cell"), main_share=top.get("share"), reading=top.get("met_a"))
        B_ = json.load(open(bf)); W_ = B_.get("met_a_whole") or {}
        row.update(A_raw=top.get("met_a_untared"), A_pred=top.get("met_a_pred"), blur=(W_.get("healthy_prediction") or {}).get("blur"), planted_line=LINE,
                   whole=W_.get("A_ratio") if W_.get("A_ratio") is not None else W_.get("ratio"), shares=sorted(((c.get("cell"), c.get("share")) for c in cf), key=lambda z: -(z[1] or 0))[:5])
    else: row["err"] = (r.stdout + r.stderr)[-300:]
    res.append(row); print(json.dumps(row), flush=True)
    json.dump(res, open(os.path.join(out, f"width_diag3_{MODE}_c{ci}_r{ri}_p{first}.json"), "w"), indent=1)   # saved after every person
