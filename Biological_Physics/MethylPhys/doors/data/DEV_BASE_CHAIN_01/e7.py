#!/usr/bin/env python3
"""DEV-BASE-CHAIN-01 check (e): Stage 7 IAM-A through run_sample.py on the bundled (constructed) single-molecule test data."""
import gzip, json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd
CH = sys.argv[1]; RS = os.path.join(CH, "MethylPhys_Interface", "run_sample.py"); PY = sys.executable; tmp = tempfile.mkdtemp()
pos = json.load(open(os.path.join(CH, "Runtime Matrices", "IAM_A_Positions", "iama_positions_v1.json")))
P, e0 = pos["cells"]["neutrophils"]["P"], pos["eps0"]; pipe = pos["cells"]["neutrophils"]["pipeline"]
H = lambda e: -(e * np.log2(e) + (1 - e) * np.log2(1 - e)); target = P * H(e0); lo, hi = 1e-6, 0.5
for _ in range(200):
    mid = (lo + hi) / 2; (lo, hi) = (mid, hi) if H(mid) < target else (lo, mid)
eps = (lo + hi) / 2; n = 2000; opp = 100; k = int(round(eps * opp * n)); per = np.full(n, k // n); per[: k % n] += 1
T = pd.DataFrame({"pos": np.arange(n), "opp_A": opp, "err_A": per, "opp_B": opp, "err_B": per}); st = os.path.join(tmp, "sites.csv"); T.to_csv(st, index=False)
env = dict(os.environ, PYTHONPATH=CH)
def run(args, out):
    r = subprocess.run([PY, RS] + args + ["--out", out, "--ledger", os.path.join(tmp, "l.jsonl")], capture_output=True, text=True, env=env, cwd=os.path.dirname(RS))
    b = os.path.splitext(out)[0] + "_bundle.json"
    return r.returncode, (json.load(open(b)) if os.path.exists(b) else {}), r.stdout[-300:] + r.stderr[-300:]
c1, b1, l1 = run(["--site-table", st, "--seq-pipeline", pipe, "--id", "CONSTRUCTED_SEQ"], os.path.join(tmp, "q1.html"))
c2, b2, l2 = run(["--site-table", st, "--seq-pipeline", "another_pipeline", "--id", "CONSTRUCTED_SEQ_B"], os.path.join(tmp, "q2.html"))
pat = os.path.join(tmp, "c.pat.gz")
with gzip.open(pat, "wt") as fh:
    for i in range(300): fh.write("chr1\t%d\t%s\t1\n" % (1000 + i, "CCCTCCCCCC" if i % 10 == 0 else "CCCCCCCCCC"))
c3, b3, l3 = run(["--pat", pat, "--id", "CONSTRUCTED_PAT"], os.path.join(tmp, "q3.html"))
sys.path[:0] = [CH]; import stage_q_iam_a as Q
PT = Q.pat_site_table(pat); nerr = int(PT[["err_A", "err_B"]].sum().sum())
q1, q2, q3 = b1.get("iam_a") or {}, b2.get("iam_a") or {}, b3.get("iam_a") or {}
res = {"site_table": {"exit": c1, "A": q1.get("A"), "state": q1.get("state"), "eps": q1.get("eps"), "pass": c1 == 0 and q1.get("A") is not None and abs(q1["A"] - 1) <= 0.005 and q1.get("state") == "Normal"},
       "other_pipeline": {"exit": c2, "A": q2.get("A"), "refusal": q2.get("refusal"), "pass": c2 == 0 and q2.get("A") is None and bool(q2.get("refusal"))},
       "pat_extractor": {"sites": len(PT), "isolated_errors": nerr, "pass": nerr == 30},
       "pat_through_run_sample": {"exit": c3, "refusal": q3.get("refusal"), "A": q3.get("A"), "note": "300 constructed molecules < 100,000 opportunities: refusal expected"},
       "report_has_iam_a_section": "id='sec-iam-a'" in open(os.path.join(tmp, "q1.html")).read() if c1 == 0 else False}
res["pass"] = res["site_table"]["pass"] and res["other_pipeline"]["pass"] and res["pat_extractor"]["pass"]
json.dump(res, open("e7.json", "w"), indent=1); print(json.dumps(res))
