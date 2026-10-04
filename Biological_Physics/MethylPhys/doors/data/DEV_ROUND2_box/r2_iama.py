#!/usr/bin/env python3
"""DEVELOPMENT - not commissioned. DEV-IAMA-REAL-01: Stage Q end to end on the Loyfer granulocyte .pat files (whole files, from GEO)."""
import os, sys, json, re, subprocess, urllib.request
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
W = os.getcwd(); CH = f"{W}/repo/Biological_Physics/MethylPhys/chain"; RS = f"{CH}/MethylPhys_Interface/run_sample.py"; PY = "/home/ubuntu/env/bin/python"
D = "/home/ubuntu/data/round2iama"; OUT = f"{W}/iama_hg19"; os.makedirs(D, exist_ok=True); os.makedirs(OUT, exist_ok=True)
G = ["GSM5652313", "GSM5652314", "GSM5652315"]
def fetch(g):
    """The hg19 build (no genome tag in the name) beside the hg38 file the first run fetched: same name without '.hg38'."""
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/"
    h38 = sorted(x for x in os.listdir(D) if x.startswith(g) and x.endswith(".hg38.pat.gz"))
    f = h38[0].replace(".hg38.pat.gz", ".pat.gz"); p = f"{D}/{f}"
    if not (os.path.exists(p) and os.path.exists(p + ".ok")):
        import time
        for k in range(12):   # GEO cuts long transfers (curl exit 18): resume where it stopped
            r = subprocess.run(["curl", "-sSfL", "-C", "-", "--retry", "5", "-o", p, base + f])
            if r.returncode == 0: break
            time.sleep(20)
        subprocess.run(["gzip", "-t", p], check=True); open(p + ".ok", "w").write("ok")
    return g, p, os.path.getsize(p)
def run(g, p, head):
    tag = f"{g}_{'head60MB' if head else 'whole'}"; out = f"{OUT}/{tag}.html"
    c = [PY, RS, "--pat", p, "--id", tag, "--out", out, "--ledger", f"{OUT}/ledger.jsonl"] + (["--pat-max-bytes", "60000000"] if head else [])
    r = subprocess.run(c, capture_output=True, text=True, cwd=os.path.dirname(RS), env=dict(os.environ, PYTHONPATH=CH))
    b = json.load(open(out.replace(".html", "_bundle.json"))) if os.path.exists(out.replace(".html", "_bundle.json")) else {}
    q = b.get("iam_a") or {}; cs = q.get("cscore") or {}
    return dict(gsm=g, part="head60MB" if head else "whole", exit=r.returncode, A=q.get("A"), state=q.get("state"), eps=q.get("eps"), opportunities=q.get("opportunities"),
                errors=round((q.get("eps") or 0) * (q.get("opportunities") or 0)), halves=json.dumps(q.get("halves")), n_sites=q.get("n_sites"), refusal=q.get("refusal"),
                C=cs.get("C"), C_blocks=cs.get("n_blocks"), C_se_null=cs.get("se_null"), C_A=((cs.get("halves") or {}).get("A") or {}).get("C"),
                C_B=((cs.get("halves") or {}).get("B") or {}).get("C"), n_molecules=(q.get("input") or {}).get("n_molecules"), tail=(r.stdout + r.stderr)[-400:])
files = [fetch(g) for g in G]; print(files, flush=True)
with ProcessPoolExecutor(6) as ex:
    res = list(ex.map(run, [f[0] for f in files] * 2, [f[1] for f in files] * 2, [True] * 3 + [False] * 3))
X = pd.DataFrame(res); X.to_csv(f"{OUT}/iama_real.csv", index=False)
F = pd.read_csv(f"{W}/iama_floor_granulocytes.csv")
h = X[X.part == "head60MB"].merge(F[["gsm", "iso", "opp"]], on="gsm")
S = {"files": [{"gsm": g, "bytes": s} for g, p, s in files], "rows": X.drop(columns=["tail"]).to_dict("records"),
     "head_reproduces": [dict(gsm=r.gsm, errors=r.errors, iso=r.iso, opp_chain=r.opportunities, opp_record=r.opp, equal=bool(r.errors == r.iso and r.opportunities == r.opp)) for r in h.itertuples()]}
json.dump(S, open(f"{OUT}/iama_summary.json", "w"), indent=1, default=str); print(json.dumps(S, indent=1, default=str)[:4000])
