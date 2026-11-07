"""DEV-METAA-HEALTHY-01, diagnostic after synthetic test 6 (development; nothing scored; no chain code changed). Builds test 6's low-tail and
center arrays (the shares the chain found for GSM7508429 on every atlas cell, kept in synthetic_06_diag_draw1.json), runs the one command
(run_sample.py, the rehearsal's flags) IN THIS PROCESS with stage_metaa_atlas.predicted_healthy wrapped so the exact values the healthy line is
measured on (xs, es, w) and its result are saved, then the line is measured again on those saved values and on the built mix.
Usage: python3 synthetic_06_capture.py DIAG_DRAW1.json OUTDIR"""
import os, sys, json, runpy, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); CH = os.path.abspath(os.path.join(H, "..", "..", "..", "chain")); IF = os.path.join(CH, "MethylPhys_Interface")
sys.path.insert(0, IF); sys.path.insert(0, CH); import stage_metaa_atlas as SA
J, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True); built = json.load(open(J))["shares"]
des, sets, _ = SA._ruler(); idx = des.index; F = pd.Series(built)
MU = pd.concat([SA._col("mu", k).reindex(idx).rename(k) for k in built], axis=1); c = MU.notna().mul(F, axis=1).sum(1) / F.sum()
e = (MU.fillna(0).mul(F, axis=1).sum(1) / MU.notna().mul(F, axis=1).sum(1)).where(c >= SA.MIX_COVER).dropna(); d = des.reindex(e.index).to_numpy()
P = {"low tail": {"I": (0.0022, 0.9928, 0.0148), "II": (-0.0036, 0.9991, 0.0240)}, "center": {"I": (-0.0033, 1.0026, 0.0066), "II": (-0.0121, 1.0175, 0.0098)}}
CAP = {}
_orig = SA.predicted_healthy
def wrapped(xs, es, w, blur=None, **k):
    CAP["xs"], CAP["es"], CAP["w"] = xs.copy(), es.copy(), pd.Series(np.asarray(w), index=xs.index)
    r = _orig(xs, es, w, blur=blur, **k); CAP["r"] = r; return r
SA.predicted_healthy = wrapped
L = []
for nm, Pl in P.items():
    g = np.random.default_rng(20261128); x = np.empty(len(e))
    for kk, (a, b, s) in Pl.items():
        m = d == kk; x[m] = a + b * e.to_numpy()[m] + s * g.standard_normal(m.sum())
    f = os.path.join(out, "cap_betas.csv"); pd.DataFrame({"cpg_id": e.index, "beta": np.round(np.clip(x, 0, 1), 4)}).to_csv(f, index=False)
    bp = os.path.join(out, f"cap_{nm.split()[0]}_bundle.json"); CAP.clear()
    sys.argv = ["run_sample.py", "--betas", f, "--specimen", "whole blood", "--id", "cap", "--out", os.path.join(out, "cap.html"), "--bundle", bp,
                "--full-chain", "--cell-panel", "atlas", "--dev-line-censored"]
    cwd = os.getcwd(); os.chdir(IF)
    try: runpy.run_path(os.path.join(IF, "run_sample.py"), run_name="__main__")
    except SystemExit: pass
    finally: os.chdir(cwd)
    B = json.load(open(bp)); xs, es, w = CAP["xs"], CAP["es"], CAP["w"]
    xin = pd.Series(np.round(np.clip(x, 0, 1), 4), index=e.index).reindex(xs.index)
    L.append(dict(array=nm, planted=Pl, chain_line=CAP["r"].get("blur"), line_on_captured=SA.array_blur(xs, es),
                  line_on_captured_x_built_e=SA.array_blur(xs, e.reindex(xs.index)), sites=len(xs),
                  xs_max_diff_from_input=float(np.nanmax(np.abs(xs.to_numpy() - xin.to_numpy()))),
                  es_median_diff_from_built=float(np.nanmedian(np.abs(es.to_numpy() - e.reindex(xs.index).to_numpy()))),
                  ratio=((B.get("met_a_whole") or {}).get("healthy_prediction") or {}).get("A_over_pred"),
                  shares={x_["cell"]: x_["share"] for x_ in B["cell_shares"]["cells"]}, line_censored=SA.LINE_CENSORED))
    pd.DataFrame(dict(xs=xs, es=es, w=w, e_built=e.reindex(xs.index), x_input=xin)).to_parquet(os.path.join(out, f"cap_{nm.split()[0]}.parquet"))
    print(json.dumps(L[-1], default=str)[:900], flush=True)
json.dump(L, open(os.path.join(out, "synthetic_06_capture.json"), "w"), indent=1, default=str)
