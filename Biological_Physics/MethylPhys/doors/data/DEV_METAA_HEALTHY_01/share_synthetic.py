"""When does a cell's own reading read that cell? (the third step to a working chain; written and committed 2026-11-07 before it runs;
development). The author: each cell the IAM-Deconvolver finds is scored against its own atlas reference. A cell's Met-A is read at its identity
sites in the whole sample, so at a small share those sites carry mostly other cells' DNA. Rule, nothing chosen: a cell's reading reads that
cell at a share where planting a loss of held pattern in that cell moves the cell's reading more than planting the same loss in every other
cell of the sample at the same sites. Construction as synthetic test 7 (synthetic_07.py): IAM-Atlas v2.1 mixed in the shares the chain found
for the spent healthy blood GSM7508429, through the real arrays' median line and noise per probe design; the target cell set to share S and the
others scaled to 1 - S; three arrays per (cell, share): healthy; a 10 % loss of held pattern (each level moved 10 % toward 0.5) in the target
cell at its own identity sites; the same loss in every other cell at those sites. Every array read by run_sample.py --betas --full-chain
--cell-panel atlas. Output per (cell, share): the target's Met-A on each array, the Deconvolver's share for it, and whether the target's own
loss moved it more than the others' loss did. Seed 20261140 + 100 * cell index + share index.
Usage: python3 share_synthetic.py OUTDIR CELL_INDEX SHARE_INDEX"""
import os, sys, json, re, subprocess, urllib.request, time
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
CELLS = ["neutrophils", "monocytes", "naive cd4 t cells", "cd8 central memory t cells", "naive b cells", "cd56bright nk cells", "hepatocyte", "colon epithelium", "vascular endothelium"]   # the blood cells the Deconvolver reports in blood, and three tissue cells
SHARES = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
out, ci, si = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True); X, S = CELLS[ci], SHARES[si]
F0 = pd.read_csv(os.path.join(H, "gsm7508429_shares.csv")).set_index("cell")["share"].to_dict()   # the shares the chain found for GSM7508429 (test 7, draw 1)
F = pd.Series({k: v for k, v in F0.items() if v > 0 and k != X}); F = F / F.sum() * (1 - S); F[X] = S
des, sets, _ = SA._ruler(); idx = des.index
MU = pd.concat([SA._col("mu", k).reindex(idx).rename(k) for k in F.index], axis=1); have = MU.notna()
cover = have.mul(F, axis=1).sum(1) / F.sum(); keep = cover >= SA.MIX_COVER
def mix(M): return (M.fillna(0).mul(F, axis=1).sum(1) / have.mul(F, axis=1).sum(1)).where(keep).dropna()
e = mix(MU); d = des.reindex(e.index).to_numpy(); g = np.random.default_rng(20261140 + 100 * ci + si)
ID = [s_ for s_ in SA.identity_sites(X) if s_ in MU.index]
LINE = {"I": (-0.0033, 1.0026, 0.0066), "II": (-0.0121, 1.0175, 0.0098)}   # the real arrays' medians (synthetic test 7, 'center')
def array(t):
    x = np.empty(len(t))
    for k, (a, b, s) in LINE.items():
        m = d == k; x[m] = a + b * t[m] + s * g.standard_normal(m.sum())
    return pd.Series(np.clip(x, 0, 1), index=e.index)
def loss(cols):
    M = MU.copy()
    for c in cols: M.loc[ID, c] = M.loc[ID, c] + 0.10 * (0.5 - M.loc[ID, c])
    return mix(M).reindex(e.index)
def run(tag, x):
    f = os.path.join(out, tag + "_betas.csv"); pd.DataFrame({"cpg_id": x.index, "beta": x.round(4).to_numpy()}).to_csv(f, index=False)
    b = os.path.join(out, tag + "_bundle.json")
    r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "whole blood", "--id", tag, "--out", os.path.join(out, tag + ".html"), "--bundle", b, "--full-chain", "--cell-panel", "atlas"], capture_output=True, text=True)
    os.remove(f)
    if r.returncode: return dict(rc=r.returncode, err=(r.stdout + r.stderr)[-400:])
    B = json.load(open(b)); m = {c["cell"]: c for c in (B.get("met_a_each_cell") or {}).get("cells", [])}.get(X) or {}
    return dict(rc=0, A=m.get("A"), share_found=m.get("share"), not_found=bool(m.get("not_found_flag")), note=m.get("coverage_flag") or m.get("reason"))
R = dict(cell=X, share=S, identity_sites=len(ID), seed=20261140 + 100 * ci + si)
R["healthy"] = run(f"c{ci}s{si}_healthy", array(e.to_numpy()))
R["own_loss"] = run(f"c{ci}s{si}_own", array(loss([X]).to_numpy()))
R["others_loss"] = run(f"c{ci}s{si}_others", array(loss([c for c in F.index if c != X]).to_numpy()))
a0, a1, a2 = (R[k].get("A") for k in ("healthy", "own_loss", "others_loss"))
R["own_move"] = None if None in (a0, a1) else round(a1 - a0, 5); R["others_move"] = None if None in (a0, a2) else round(a2 - a0, 5)
R["reads_this_cell"] = None if None in (R["own_move"], R["others_move"]) else bool(abs(R["own_move"]) > abs(R["others_move"]))
json.dump(R, open(os.path.join(out, f"share_synthetic_c{ci}_s{si}.json"), "w"), indent=1); print(json.dumps(R)[:600])
