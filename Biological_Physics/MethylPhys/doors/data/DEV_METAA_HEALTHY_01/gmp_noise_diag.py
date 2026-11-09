"""Is the GMP share a measurement effect? (2026-11-08, written and committed before it runs; development, a diagnostic, nothing scored.)
In the atlas's sorted neutrophil donors the IAM-Deconvolver's GMP share follows the array's measured noise (design II, rank correlation
+0.78; gmp_noise_relation.txt) and the laboratory's neutrophil Met-A follows the GMP share. Here 12 neutrophil donors from the two quietest
laboratories (GMP share 0; gmp_noise_donors.csv; their calibrated values kept by the bv2_donors reading) are read again by the one command
with noise added to every probe at NOISE = 0, 0.003, 0.006 and 0.009 (normal, seeded per donor; values held to 0..1), which spans the noisy
laboratories' measured noise once the donors' own is counted. Kept per reading: the cells found and shares, the neutrophils' Met-A and the
measured line. If the GMP share and the neutrophils' Met-A rise with the added noise, the GMP share is made by the measurement.
Usage: gmp_noise_diag.py VALUES_DIR OUTDIR [PART 0-5, two donors each]"""
import os, sys, json, subprocess, tempfile, shutil
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); RS = os.path.abspath(os.path.join(H, "..", "..", "..", "chain", "MethylPhys_Interface", "run_sample.py"))
vd, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); PART = int(sys.argv[3]) if len(sys.argv) > 3 else None   # 2 donors a part
D = pd.read_csv(os.path.join(H, "gmp_noise_donors.csv")); RF = os.path.join(out, "gmp_noise_diag.json" if PART is None else f"gmp_noise_diag_p{PART}.json")
res = json.load(open(RF)) if os.path.exists(RF) else []; done_ = {(r["gsm"], r["noise"]) for r in res}
for i, g in enumerate(D.gsm):
    if PART is not None and i // 2 != PART: continue
    b = pd.read_parquet(os.path.join(vd, g + "_betas.parquet")).iloc[:, 0].dropna(); b.index = b.index.astype(str)
    for nz in (0.0, 0.003, 0.006, 0.009):
        if (g, nz) in done_: continue
        rg = np.random.default_rng(20261108 + 100 * i + int(nz * 1000)); x = np.clip(b.values + rg.normal(0, nz, len(b)) if nz else b.values, 0, 1)
        wd = tempfile.mkdtemp(); f = os.path.join(wd, "b.csv"); pd.DataFrame({"cpg_id": b.index, "beta": np.round(x, 4)}).to_csv(f, index=False); bf = os.path.join(wd, "b.json")
        p = subprocess.run([sys.executable, RS, "--betas", f, "--specimen", "unknown", "--id", f"{g}_{nz}", "--full-chain", "--cell-panel", "atlas", "--dev-line-wide", "--out", os.path.join(wd, "b.html"), "--bundle", bf], capture_output=True, text=True)
        row = dict(gsm=g, noise=nz, rc=p.returncode)
        if p.returncode == 0:
            B = json.load(open(bf)); cf = (B.get("cells_found") or {}).get("cells") or []
            row.update(cells={c["cell"]: c.get("share") for c in cf}, neutrophil_met_a=next((c.get("met_a") for c in cf if c["cell"] == "neutrophils"), None),
                       line=((B.get("met_a_whole") or {}).get("healthy_prediction") or {}).get("blur"))
        res.append(row); json.dump(res, open(RF, "w"), indent=1); shutil.rmtree(wd, ignore_errors=True); print(g, nz, p.returncode, flush=True)
