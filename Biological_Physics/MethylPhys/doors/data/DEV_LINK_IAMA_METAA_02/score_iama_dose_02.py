"""DEV-LINK-IAMA-METAA-02, IAM-A side and the comparison (written and committed 2026-10-10 before any EM-seq reading).
Runs (GSE237662 -> SRA): Veh_EM_1 SRR25322252, Veh_EM_2 SRR25322251, DAC30_EM_1 SRR25322250, DAC30_EM_2 SRR25322249,
DAC300_EM_1 SRR25322248, DAC300_EM_2 SRR25322247; .pat.gz from Box Run 2 session 5 (s3://.../results/BOXRUN2/session5/, pinned pipeline).
Per run: ε by Stage Q's pat_site_table (whole file); molecules with >= 6 calls, and those Stage Q reads (>= 80 % C), counted by Stage Q's rule.
IAM-A_rel = H(ε)/H(ε_vehicle), ε_vehicle = errors/opportunities pooled over the two vehicle runs.
share_read = (read / >= 6-call molecules) of the run ÷ the same for the pooled vehicles.
Comparison: the window from predict_iama_window_02.py with RESPONSE = insilico_loss_02.py on SRR25322252 (first 1,500,000 lines, as the stand-in).
Run: python3 score_iama_dose_02.py CHAIN PAT_DIR WINDOW_RESPONSE.csv OUT.csv"""
import os, sys, gzip, math, json, subprocess, numpy as np, pandas as pd
CH, PD, RESP, OUT = sys.argv[1:5]; sys.path.insert(0, CH); import stage_q_iam_a as Q
RUNS = {"SRR25322252": "vehicle", "SRR25322251": "vehicle", "SRR25322250": "DAC 30 nM", "SRR25322249": "DAC 30 nM",
        "SRR25322248": "DAC 300 nM", "SRR25322247": "DAC 300 nM"}
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
def counts(p):
    m6 = rd = 0
    with gzip.open(p, "rt") as f:
        for l in f:
            q = l.split("\t"); n = int(q[3]); c = [x for x in q[2] if x != "."]
            if len(c) < 6: continue
            m6 += n; rd += n * (c.count("C") >= 0.8 * len(c))
    return m6, rd
rows = []
for r, t in RUNS.items():
    p = os.path.join(PD, f"{r}.pat.gz")
    if not os.path.exists(p): print(r, t, "not yet"); continue
    T = Q.pat_site_table(p); e_ = float(T.err_A.sum() + T.err_B.sum()); o_ = float(T.opp_A.sum() + T.opp_B.sum()); m6, rd = counts(p)
    rows.append(dict(run=r, treatment=t, errors=e_, opportunities=o_, eps=e_ / o_, molecules_6=m6, molecules_read=rd)); print(rows[-1], flush=True)
R = pd.DataFrame(rows); V = R[R.treatment == "vehicle"]
if len(V) == 0: sys.exit("no vehicle run yet")
ev = V.errors.sum() / V.opportunities.sum(); sv = V.molecules_read.sum() / V.molecules_6.sum()
R["IAMA_rel"] = [H(e) / H(ev) for e in R.eps]; R["share_read"] = (R.molecules_read / R.molecules_6) / sv; R.to_csv(OUT, index=False)
print(f"vehicle eps {ev:.5f} (runs {', '.join(f'{x:.5f}' for x in V.eps)})")
print(R[["run", "treatment", "eps", "IAMA_rel", "share_read"]].round(4).to_string(index=False))
HERE = os.path.dirname(os.path.abspath(__file__))
print(subprocess.run([sys.executable, os.path.join(HERE, "predict_iama_window_02.py"), RESP], capture_output=True, text=True).stdout)
