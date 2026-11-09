"""Cells or a probe-tied error? (2026-11-08, written and committed before it runs; development, a diagnostic, nothing scored.) Random array
noise does not make the IAM-Deconvolver's GMP share (gmp_noise_diag_summary.txt), yet among the atlas's sorted neutrophil donors the GMP
share follows the laboratory. For every neutrophil donor with kept values (gmp_source_donors.csv), at the Deconvolver's marker sites of the
full atlas panel: the residual r = the array minus the atlas's neutrophil column, and the direction d = the GMP column minus the neutrophil
column. Kept per donor: the projection f = sum(r d) / sum(d d) (the GMP fraction a pure two-cell fit would give); f again on design I and on
design II probes separately (a null test by detector: real GMP cells give the same f on both designs, an error of one probe design does not);
f again with the 5 % of probes carrying the largest |r d| left out (cells spread over many probes keep f; a few probes carrying it lose it);
and the rank correlation of r with d over the sites. Each donor is first read by the one command (run_sample.py --full-chain --cell-panel
atlas --dev-line-wide) so its GMP share and neutrophil Met-A are the chain's own. Usage: gmp_source_diag.py VALUES_DIR OUT.json PART (0-5) [errors: the second pass, no re-read | curve: each donor read with --dev-curve-cal and the projections taken on the calibrated values]"""
import os, sys, json, subprocess, tempfile, shutil
import numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); sys.path.insert(0, CH)
import stage_deconv_v21 as D21, stage_v4 as V4, stage_metaa_atlas as SA
from scipy.stats import spearmanr
vd, outf, PART = sys.argv[1], sys.argv[2], int(sys.argv[3]); CAL = len(sys.argv) > 4 and sys.argv[4] == "curve"; RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py")
panel = V4.atlas_panel(); R = D21.reference(panel); M = list(D21.markers(R)); des, _, _ = SA._ruler()
mu = R["mu"].loc[M]; d_all = (mu["GMP (bone marrow)"] - mu["neutrophils"]); n_all = mu["neutrophils"]
D = pd.read_csv(os.path.join(H, "gmp_source_donors.csv")); res = []
def proj(r, d): return float((r * d).sum() / (d * d).sum()) if len(d) and (d * d).sum() > 0 else None
res = json.load(open(outf)) if os.path.exists(outf) else []; done_ = {r_["gsm"] for r_ in res}
# 2026-11-08 late, second pass (no re-read): for donors already read here, the number of marker sites on each probe design and the bootstrap
# error (200 draws of sites, seeded) of each design's projection, so design I's null can be judged against its own noise
if len(sys.argv) > 4 and sys.argv[4] == "errors":
    rg = np.random.default_rng(20261108)
    for r_ in res:
        if "se_I" in r_ or not r_.get("read"): continue
        x = pd.read_parquet(os.path.join(vd, r_["gsm"] + "_betas.parquet")).iloc[:, 0]; x.index = x.index.astype(str); x = x.reindex(M).dropna()
        d, r = d_all.reindex(x.index), x - n_all.reindex(x.index); ok = d.notna() & r.notna(); d, r = d[ok], r[ok]; t_ = des.reindex(d.index)
        for k in ("I", "II"):
            dk, rk = d[t_ == k].values, r[t_ == k].values; r_["n_" + k] = int(len(dk))
            B = [proj(pd.Series(rk[i]), pd.Series(dk[i])) for i in (rg.integers(0, len(dk), len(dk)) for _ in range(200))] if len(dk) else []
            r_["se_" + k] = float(np.std(B)) if B else None
    json.dump(res, open(outf, "w"), indent=1); print("errors added", flush=True); sys.exit(0)
for i, (g, ser) in enumerate(zip(D.gsm, D.series)):
    if i % 6 != PART or g in done_: continue
    f = os.path.join(vd, g + "_betas.parquet")
    if not os.path.exists(f): res.append(dict(gsm=g, series=ser, read=False)); continue
    x0 = pd.read_parquet(f).iloc[:, 0].dropna(); x0.index = x0.index.astype(str)
    wd = tempfile.mkdtemp(); bf = os.path.join(wd, "b.csv"); pd.DataFrame({"cpg_id": x0.index, "beta": x0.round(4).values}).to_csv(bf, index=False); bj = os.path.join(wd, "b.json")
    p = subprocess.run([sys.executable, RS, "--betas", bf, "--specimen", "unknown", "--id", g, "--full-chain", "--cell-panel", "atlas", "--dev-line-wide"] + (["--dev-curve-cal"] if CAL else []) + ["--out", os.path.join(wd, "b.html"), "--bundle", bj], capture_output=True, text=True)
    ch = {}
    if p.returncode == 0:
        cf = (json.load(open(bj)).get("cells_found") or {}).get("cells") or []
        ch = dict(chain_gmp=next((c.get("share") for c in cf if c["cell"] == "GMP (bone marrow)"), 0) or 0, chain_neu=next((c.get("share") for c in cf if c["cell"] == "neutrophils"), None),
                  chain_met_a=next((c.get("met_a") for c in cf if c["cell"] == "neutrophils"), None))
    cc = None
    if CAL and p.returncode == 0:   # the values the chain read after its own calibration: the same function on the same cells the chain used
        import stage_t_selftare as STt; cc = json.load(open(bj)).get("curve_cal") if os.path.exists(bj) else None
    if cc and cc.get("cells"):
        x0, _ = STt.curve_cal(x0, cc["cells"], STt._json()["design"]); ch["curve_cal"] = {k: v.get("largest_correction") for k, v in (cc.get("designs") or {}).items()}
    shutil.rmtree(wd, ignore_errors=True)
    x = x0.reindex(M).dropna()
    d, r = d_all.reindex(x.index), x - n_all.reindex(x.index); ok = d.notna() & r.notna(); d, r = d[ok], r[ok]
    t = des.reindex(d.index); keep = (r * d).abs().rank(pct=True) <= 0.95
    res.append(dict(gsm=g, series=ser, read=True, sites=int(len(d)), f=proj(r, d), f_I=proj(r[t == "I"], d[t == "I"]), f_II=proj(r[t == "II"], d[t == "II"]),
                    f_trim=proj(r[keep], d[keep]), rho=float(spearmanr(r, d).statistic), resid_sd=float(r.std()), rc=p.returncode, **ch))
    print(g, ser, round(res[-1]["f"], 4), flush=True); json.dump(res, open(outf, "w"), indent=1)
json.dump(res, open(outf, "w"), indent=1)
