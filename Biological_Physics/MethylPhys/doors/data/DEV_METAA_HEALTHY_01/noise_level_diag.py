"""DIAGNOSTIC of the noise lead (2026-11-08, written and committed before it runs; development, nothing scored). On the atlas's 455 donors a
cell's Met-A rises with the array's noise as the chain measures it (atlas_donor_noise_relation.csv); the lead: the chain measures the noise on
sites held below 0.05 or above 0.95, where a probe's noise is squeezed against its bound, and a cell's identity sites sit where it is larger.
Here 5 healthy people per cell (neutrophils, CD8 T cells, B cells; site-by-site person spread, as the width synthetic) are read with the
array's line and a noise that grows toward the middle: sd = S0 x sqrt(e (1 - e)) / sqrt(0.02 x 0.98), S0 the noise at a firmly held site (so
S0 is what the chain measures), for S0 = NOISE (0.004, 0.007, 0.010 or 0.014, the donors' measured range). The same people with one noise S0
at every level are read beside them (MODE flat). If the lead is right, the level-noise readings rise with S0 like the donors' (about 6.9 per
unit of measured noise) and the flat ones do not. Usage: python3 noise_level_diag.py OUTDIR NOISE"""
import os, sys, json, subprocess
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_metaa_atlas as SA
out, S0 = sys.argv[1], float(sys.argv[2]); os.makedirs(out, exist_ok=True)
des, _, _ = SA._ruler(); idx = des.index; LINE = {"I": (-0.0033, 1.0026), "II": (-0.0121, 1.0175)}; res = []
for ci, X in enumerate(["neutrophils", "cd8 t cells", "b cells"]):
    mu = SA._col("mu", X).reindex(idx).dropna(); sd = SA._col("donor_sd", X).reindex(mu.index); sd = sd.fillna(sd.median()); d = des.reindex(mu.index).to_numpy()
    g = np.random.default_rng(20261190 + ci)
    for p in range(5):
        lev = np.clip(mu.to_numpy() + sd.to_numpy() * g.standard_normal(len(mu)), 0, 1); truth = SA.read(pd.Series(lev, index=mu.index), X).get("A"); z = g.standard_normal(len(mu))
        for mode in ("level", "flat"):
            e = mu.to_numpy(); s = S0 * np.sqrt(np.clip(e * (1 - e), 0.02 * 0.98, None)) / np.sqrt(0.02 * 0.98) if mode == "level" else np.full(len(e), S0)
            x = np.empty(len(lev))
            for k, (a, b) in LINE.items(): m = d == k; x[m] = a + b * lev[m] + s[m] * z[m]
            tag = f"nl{mode[0]}{ci}p{p}"; f = os.path.join(out, tag + "_betas.csv"); pd.DataFrame({"cpg_id": mu.index, "beta": np.clip(x, 0, 1).round(4)}).to_csv(f, index=False)
            bf = os.path.join(out, tag + "_bundle.json")
            r = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "sorted cells", "--id", tag, "--out", os.path.join(out, tag + ".html"), "--bundle", bf, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide"], capture_output=True, text=True)
            os.remove(f); row = dict(cell=X, noise=S0, mode=mode, person=p, truth=truth, rc=r.returncode)
            if r.returncode == 0:
                B = json.load(open(bf)); cf = (B.get("cells_found") or {}).get("cells") or []; top = max(cf, key=lambda c: c.get("share") or 0) if cf else {}
                bl = ((B.get("met_a_whole") or {}).get("healthy_prediction") or {}).get("blur") or {}
                row.update(main_cell=top.get("cell"), reading=top.get("met_a"), measured_s_II=(bl.get("II") or {}).get("s"), measured_s_I=(bl.get("I") or {}).get("s"))
            res.append(row); print(row, flush=True)
            for ff in (bf, os.path.join(out, tag + ".html")):
                if os.path.exists(ff): os.remove(ff)
            json.dump(res, open(os.path.join(out, f"noise_level_{S0}.json"), "w"), indent=1)
