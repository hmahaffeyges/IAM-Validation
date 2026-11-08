"""THE WIDTH SYNTHETIC WITH THE WIDE LINE (2026-11-08; the same people, bars and reading, with run_sample.py --dev-line-wide added: the
array's line measured on every site the sample's mix holds firmly instead of only the ruler's anchors; the second width diagnostic found the
anchors' line moving from person to person enough to triple the spread). The synthetic's own text follows. Each cell's healthy width from its donors: synthetic, written and committed 2026-11-07 before it runs (the fourth step to a working chain;
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
out, ci, ri, first = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True); X, rho = CELLS[ci], RHOS[ri]
des, _, _ = SA._ruler(); idx = des.index
mu = SA._col("mu", X).reindex(idx).dropna(); sd = SA._col("donor_sd", X).reindex(mu.index); sd = sd.fillna(sd.median())
d = des.reindex(mu.index).to_numpy(); LINE = {"I": (-0.0033, 1.0026, 0.0066), "II": (-0.0121, 1.0175, 0.0098)}
g = np.random.default_rng(20261150 + 10 * ci + ri); Z = g.standard_normal(20); E = [g.standard_normal(len(mu)) for _ in range(20)]
res = []
for p in range(first, min(first + 5, 20)):
    lev = np.clip(mu.to_numpy() + sd.to_numpy() * (np.sqrt(rho) * Z[p] + np.sqrt(1 - rho) * E[p]), 0, 1); truth = SA.read(pd.Series(lev, index=mu.index), X).get("A")
    x = np.empty(len(lev)); gn = np.random.default_rng(20261150 + 1000 * ci + 100 * ri + p)
    for k, (a, b, s) in LINE.items():
        m = d == k; x[m] = a + b * lev[m] + s * gn.standard_normal(m.sum())
    tag = f"w{ci}r{ri}p{p}"; f = os.path.join(out, tag + "_betas.csv"); pd.DataFrame({"cpg_id": mu.index, "beta": np.clip(x, 0, 1).round(4)}).to_csv(f, index=False)
    bf = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "sorted cells", "--id", tag, "--out", os.path.join(out, tag + ".html"), "--bundle", bf, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide"], capture_output=True, text=True)
    os.remove(f); row = dict(cell=X, rho=rho, person=p, truth=truth, rc=r.returncode)
    if r.returncode == 0:
        cf = (json.load(open(bf)).get("cells_found") or {}).get("cells") or []; top = max(cf, key=lambda c: c.get("share") or 0) if cf else {}
        row.update(main_cell=top.get("cell"), main_share=top.get("share"), reading=top.get("met_a"))
    else: row["err"] = (r.stdout + r.stderr)[-300:]
    res.append(row); print(json.dumps(row), flush=True)
    json.dump(res, open(os.path.join(out, f"width_wide_c{ci}_r{ri}_p{first}.json"), "w"), indent=1)   # saved after every person
