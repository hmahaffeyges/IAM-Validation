#!/usr/bin/env python3
# INSTRUMENT-TEST: PROC-INTAKE-01 bars B1-B5 under the author's 0.93 line (intake_thresholds_v1.json). Produces bar results
# only; the reports it renders are test renders under results/intake01/ and are not filed.
import os, sys, json, glob, subprocess, time, re
import numpy as np, pandas as pd
W = os.path.dirname(os.path.abspath(__file__)); CH = os.path.join(W, "iamrepo/Biological_Physics/MethylPhys/chain")
RUN = os.path.join(CH, "MethylPhys_Interface/run_sample.py"); OUT = os.path.join(W, "results/intake01_clean"); os.makedirs(OUT, exist_ok=True)
ENV = {**os.environ, "HOME": os.path.join(W, "stage1/mp_home")}
os.environ["HOME"] = ENV["HOME"]   # in-process Stage 1 (B2, B3) needs the decoder home too - the first run raised on every B2 array without it
sys.path.insert(0, CH)
import stage_0_intake as S0
from stage_1_idat_calibration import calibrate_idat_to_beta
LINE = S0.CALL_RATE_BORDERLINE; assert abs(LINE - 0.93) < 1e-9, LINE
open(os.path.join(OUT, "PID"), "w").write(str(os.getpid()))
res = {"line": LINE, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

def pairs(lab):
    out = {}
    for g in sorted(glob.glob(os.path.join(W, "tare01/idats", f"{lab}__*_Grn.idat*"))):
        gsm = os.path.basename(g).split("__", 1)[1].split("_")[0]; out[gsm] = (g, g.replace("_Grn", "_Red"))
    return out

def render(grn, red, out, sid, extra=()):
    r = subprocess.run([sys.executable, RUN, "--grn", grn, "--red", red, "--age", "60", "--sex", "F", "--specimen", "whole blood",
                        "--lab", "TEST", "--id", sid, "--out", out, *extra], capture_output=True, text=True, env=ENV)
    return r.returncode, (r.stdout + r.stderr)

# ---- B1: the 12 Munich panel arrays - call rate reported; < 0.93 QUARANTINE (exit 2, no report); 0.93-0.98 PENALTY flag
b1 = []
for gsm, (g, r) in pairs("GSE125105").items():
    out = os.path.join(OUT, f"B1_{gsm}.html"); code, log = render(g, r, out, gsm)
    m = re.search(r"probes detected \((\d+\.\d+) %\)", log); cr = float(m.group(1))/100 if m else None
    open(os.path.join(OUT, f"B1_{gsm}.log"), "w").write(log)
    b1.append({"gsm": gsm, "exit": code, "report_written": os.path.exists(out), "call_rate": cr,
               "quarantined": code == 2, "penalty_flag": ("PENALTY" in log) or ("CALL_RATE_BORDERLINE" in log),
               "hard_failures": (re.search(r"hard failures: (\[.*?\])", log).group(1) if re.search(r"hard failures: (\[.*?\])", log) else None)})
    print(f"B1 {gsm}: exit {code} call_rate {cr} report {os.path.exists(out)}", flush=True)
q = sum(1 for x in b1 if x["quarantined"]); rep_when_q = sum(1 for x in b1 if x["quarantined"] and x["report_written"])
ok_b1 = all(x["call_rate"] is not None for x in b1) and rep_when_q == 0 and all(
    (x["quarantined"] == (x["call_rate"] < LINE)) for x in b1 if x["call_rate"] is not None)
res["B1"] = {"n": len(b1), "quarantined": q, "report_written_despite_quarantine": rep_when_q, "rows": b1, "met": ok_b1}

# ---- B2: first 100 GSE87571 IDAT pairs (accession order) - Stage 1 detection + call-rate gate, at least 95 PROCEED (>= 0.93)
up = sorted(glob.glob(os.path.join(W, "idats_full_GSE87571/*_Grn.idat*")))[:100]
b2 = []
for i, g in enumerate(up):
    try:
        beta, meta = calibrate_idat_to_beta(g, g.replace("_Grn", "_Red"), verbose=False)
        det = meta.get("detection") or {}; cr = det.get("pct_detected")
        n = det["n_probes"]; k = det["n_detected"]; mask = np.zeros(n, bool); mask[:k] = True
        v = S0.validate_call_rate(int(mask.sum()), n)
        b2.append({"gsm": os.path.basename(g).split("_")[0], "call_rate": cr, "status": v["call_rate_status"], "advance": v["advance"]})
    except Exception as e:
        b2.append({"gsm": os.path.basename(g).split("_")[0], "error": f"{type(e).__name__}: {e}"[:120]})
    if i % 10 == 9: print(f"B2 {i+1}/100  advance so far {sum(1 for x in b2 if x.get('advance'))}  errors {sum(1 for x in b2 if 'error' in x)}" + (f"  first error: {next(x['error'] for x in b2 if 'error' in x)}" if any('error' in x for x in b2) else ""), flush=True)
adv = sum(1 for x in b2 if x.get("advance")); crs = [x["call_rate"] for x in b2 if x.get("call_rate") is not None]
res["B2"] = {"n": len(b2), "advance": adv, "min_call_rate": min(crs) if crs else None, "median_call_rate": float(np.median(crs)) if crs else None,
             "status_counts": pd.Series([x.get("status", "error") for x in b2]).value_counts().to_dict(), "met": adv >= 95}

# ---- B3: masking does not move A on good arrays - 12 GSE87571 panel arrays, per present cell |A_masked - A_unmasked| < 0.002
import cpg_conductor as C
atlas = os.path.join(W, "atlas_work/IAMAtlasREBUILD.csv")
b3 = []
for gsm, (g, r) in list(pairs("GSE87571").items())[:12] or [(os.path.basename(x).split("_")[0], (x, x.replace("_Grn", "_Red"))) for x in up[:12]]:
    beta_m, meta = calibrate_idat_to_beta(g, r, verbose=False)
    beta_u, _ = calibrate_idat_to_beta(g, r, verbose=False, mask_detection=False) if "mask_detection" in calibrate_idat_to_beta.__code__.co_varnames else (None, None)
    if beta_u is None:
        # unmasked = masked vector with the removed probes restored from methylprep's own unmasked export is not available;
        # record as not assessable rather than fake it
        b3.append({"gsm": gsm, "note": "calibrate_idat_to_beta has no mask_detection switch - B3 NOT ASSESSED"}); break
    bm = (beta_m.iloc[:, 0] if hasattr(beta_m, "columns") else beta_m).dropna().to_dict(); bu = (beta_u.iloc[:, 0] if hasattr(beta_u, "columns") else beta_u).dropna().to_dict()
    om = C.run_full(bm, atlas, cfg={"age": 60, "pipeline": "stage1_noob_450K"}); ou = C.run_full(bu, atlas, cfg={"age": 60, "pipeline": "stage1_noob_450K"})
    cm = om.get("cells_all") or {}; cu = ou.get("cells_all") or {}
    d = {c: abs(cm[c]["A"] - cu[c]["A"]) for c in cm if cm[c].get("present") and c in cu and cm[c].get("A") is not None and cu[c].get("A") is not None}
    b3.append({"gsm": gsm, "n_present": len(d), "max_dA": max(d.values()) if d else None}); print(f"B3 {gsm}: max |dA| {max(d.values()) if d else None}", flush=True)
mx = [x["max_dA"] for x in b3 if x.get("max_dA") is not None]
res["B3"] = {"rows": b3, "max_dA_over_arrays": max(mx) if mx else None, "met": (bool(mx) and max(mx) < 0.002), "assessed": bool(mx)}

# ---- B4: a betas-only input renders with intake_verified False and the one printed line
csv = os.path.join(W, "results/e2e3/blood.csv"); out4 = os.path.join(OUT, "B4_betas_only.html")
r4 = subprocess.run([sys.executable, RUN, "--betas", csv, "--age", "60", "--sex", "F", "--specimen", "whole blood", "--pipeline", "stage1_noob_450K",
                     "--lab", "GSE87571", "--id", "B4", "--out", out4], capture_output=True, text=True, env=ENV)
bundle = json.load(open(out4.replace(".html", "_bundle.json"))) if os.path.exists(out4.replace(".html", "_bundle.json")) else {}
html_txt = open(out4, encoding="utf-8").read() if os.path.exists(out4) else ""
iv = (bundle.get("intake") or {}).get("intake_verified", bundle.get("intake_verified"))
res["B4"] = {"exit": r4.returncode, "intake_verified": iv, "line_printed": ("intake" in html_txt.lower() and "not verified" in html_txt.lower()) or ("intake_verified" in html_txt),
             "met": r4.returncode == 0 and iv is False}

# ---- B5: negative control - a Stage 1 return with the mask withheld must QUARANTINE (deferred never advances)
rec = {"status": "MANIFEST_COMPLETE", "flags": []}
rec = S0.step_0_7_call_rate(rec, None, None); rec = S0.step_0_9_decision_gate(rec, None)
res["B5"] = {"call_rate_status": rec.get("call_rate_status"), "verdict": rec.get("stage0_verdict"), "advance": rec.get("advance"),
             "met": (rec.get("call_rate_status") == "DEFERRED_PENDING_STAGE1_DECODER") and str(rec.get("stage0_verdict", "")).startswith("QUARANTINE")}

res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(res, open(os.path.join(W, "handoff/intake01_clean_results.json"), "w"), indent=1, default=str)
open(os.path.join(OUT, "PID"), "w").write(str(os.getpid()))
for k in ("B1", "B2", "B3", "B4", "B5"):
    v = res[k]; print(f"{k}: {'MET' if v.get('met') else ('NOT ASSESSED' if k == 'B3' and not v.get('assessed') else 'FAILED')}  ", {kk: vv for kk, vv in v.items() if kk not in ("rows",)})
