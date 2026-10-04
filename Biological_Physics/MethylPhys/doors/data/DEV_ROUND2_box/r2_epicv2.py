#!/usr/bin/env python3
"""DEVELOPMENT - not commissioned. DEV-EPIC-V2-01 checks 1-5 on GSE286313 (same venous bloods on EPIC v1 and v2)."""
import os, sys, json, glob, subprocess, re, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
W = os.getcwd(); CH = f"{W}/repo/Biological_Physics/MethylPhys/chain"; sys.path[:0] = [CH]; RSC = sys.argv[1]
OUT = f"{W}/v2out"; os.makedirs(OUT, exist_ok=True); D = "/home/ubuntu/data/round2v2/GSE286313"; os.makedirs(D, exist_ok=True)
URLS = json.load(open(f"{W}/urls.json")); MAN = pd.read_csv(f"{W}/manifest.csv").drop_duplicates("gsm"); MAN = MAN[MAN.series == "GSE286313"]
if not os.path.exists(f"{D}/.done"):
    subprocess.run(["bash", "-c", "set -o pipefail; ( " + "; ".join(f"curl -sSf --retry 5 '{u}'" for u in URLS["GSE286313"]) + f" ) | tar -x -C {D} --wildcards '*.idat*' && touch {D}/.done"], check=True)
import dev_stages as DV, conductor_v3 as C3
def pair(g):
    gr = sorted(glob.glob(f"{D}/**/{g}_*Grn.idat*", recursive=True)); return (gr[0], gr[0].replace("_Grn", "_Red")) if gr else (None, None)
def one(g):
    t0 = time.time(); grn, red = pair(g)
    if not grn: return dict(gsm=g, ok=False, err="no idat")
    try:
        b, meta = DV.epicv2_calibrate(grn, red, RSC); b.to_frame("beta").to_parquet(f"{OUT}/{g}_sesame.parquet")
        o = C3.run_neutrophil(b, specimen="whole blood", array_type=None); m = o.get("met_a") or {}; a = o.get("composition") or {}
        IS = pd.Index(C3._bc()["neutrophil_sites"]); MK = pd.Index(C3._bc()["markers"])
        return dict(gsm=g, ok=True, seconds=round(time.time() - t0, 1), n_cg=len(b), n_v2_probes=meta["n_v2_probes_detected"], identity_sites=int(b.reindex(IS).notna().sum()),
                    markers=int(b.reindex(MK).notna().sum()), A=m.get("A"), f_neu=m.get("fraction"), N=m.get("noise_index"), reason=m.get("reason") or o.get("refusal"))
    except Exception as e:
        return dict(gsm=g, ok=False, err=f"{type(e).__name__}: {str(e)[:300]}")
with ProcessPoolExecutor(32) as ex: R = pd.DataFrame(list(ex.map(one, MAN.gsm)))
R = R.merge(MAN[["gsm", "plat", "title"]], on="gsm")
def mp_A(g):
    p = f"/home/ubuntu/data/base_chain_01/betas/{g}.parquet"
    if not os.path.exists(p): return None
    b = pd.read_parquet(p).iloc[:, 0].astype(float); b.index = b.index.astype(str); return (C3.run_neutrophil(b, specimen="whole blood", array_type="EPIC_v1").get("met_a") or {}).get("A")
R["A_methylprep"] = [mp_A(g) if p == "EPIC_v1" else None for g, p in zip(R.gsm, R.plat)]
R["pair"] = R.title.str.replace(r"_EPICv[12]$", "", regex=True); R.to_csv(f"{OUT}/epicv2_arrays.csv", index=False)
P = R.pivot_table(index="pair", columns="plat", values="A", aggfunc="first").dropna() if "A" in R and R.A.notna().any() else pd.DataFrame()
d = (P["EPIC_v2"] - P["EPIC_v1"]) if {"EPIC_v1", "EPIC_v2"} <= set(P.columns) else pd.Series(dtype=float)
v1 = R[R.plat == "EPIC_v1"].dropna(subset=[c for c in ("A", "A_methylprep") if c in R]) if "A" in R else R.iloc[:0]; cal = v1.A - v1.A_methylprep
S = {"n_v2": int((R.plat == "EPIC_v2").sum()), "v2_calibrated": int(((R.plat == "EPIC_v2") & R.ok).sum()), "v1_calibrated": int(((R.plat == "EPIC_v1") & R.ok).sum()),
     "v2_identity_sites_median": float(R[R.plat == "EPIC_v2"].identity_sites.median()), "v2_markers_median": float(R[R.plat == "EPIC_v2"].markers.median()),
     "pairs": int(len(d)), "pair_diff_mean": float(d.mean()) if len(d) else None, "pair_diff_sd": float(d.std()) if len(d) > 1 else None, "target_sd": 0.020,
     "calibrator_v1_sesame_minus_methylprep_mean": float(cal.mean()) if len(cal) else None, "calibrator_sd": float(cal.std()) if len(cal) > 1 else None, "n_cal": int(len(cal)),
     "duplicate_titles": int(R.title.duplicated().sum()), "errors": R[~R.ok].err.value_counts().head(5).to_dict() if (~R.ok).any() else {}}
# the flag end to end on two v2 arrays
ff = []
for g in R[(R.plat == "EPIC_v2") & R.ok].gsm.head(2):
    grn, red = pair(g); out = f"{OUT}/flag_{g}.html"
    r = subprocess.run([sys.executable, f"{CH}/MethylPhys_Interface/run_sample.py", "--grn", grn, "--red", red, "--specimen", "whole blood", "--id", g, "--out", out,
                        "--dev-epic-v2", "--sesame-rscript", RSC], capture_output=True, text=True, cwd=f"{CH}/MethylPhys_Interface", env=dict(os.environ, PYTHONPATH=CH))
    try:
        b = json.load(open(out.replace(".html", "_bundle.json"))); ev = (b.get("development") or {}).get("epic_v2") or {}
        ff.append(dict(gsm=g, exit=r.returncode, refusal=str(b.get("refusal"))[:80], dev_status=ev.get("status"), dev_A=((ev.get("reading_cross_version") or {}).get("met_a") or {}).get("A")))
    except Exception as e:
        ff.append(dict(gsm=g, exit=r.returncode, err=(r.stdout + r.stderr)[-500:]))
S["flag_runs"] = ff; json.dump(S, open(f"{OUT}/epicv2_summary.json", "w"), indent=1, default=str); print(json.dumps(S, indent=1, default=str))
