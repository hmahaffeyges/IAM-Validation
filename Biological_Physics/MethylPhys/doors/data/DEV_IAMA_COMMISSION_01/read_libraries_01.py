"""DEV-IAMA-COMMISSION-01: per-library readings from Box Run 10 outputs (laptop; the sealed reading rules). For each library in RUNS.csv whose
.pat.gz, alignment_qc.json and flagstat are in S3 results/BOXRUN10_EPIQC/TAG/: Stage Q copy error (stage_q_iam_a.pat_site_table), Stage Q0
checks Q0.1-Q0.6 (stage_q0_intake.intake; Q0.7 not applied, see the note), conversion = conversion_rate_human_chr1, eps_planted (1 % of
methylated calls lost, insilico_loss_02.py rule, seeded), and, for Stage A libraries, c1_iama = H(eps at planted conversion 0.98) / H(eps)
(conversion_05.py rule, seeded). Appends to LIBS.csv (one row per library; existing rows kept). Usage: python3 read_libraries_01.py CHAIN RUNS.csv TAG WORK LIBS.csv"""
import sys, os, json, gzip, math, random, numpy as np, pandas as pd, boto3
CHAIN, RUNS, TAG, WORK, OUT = sys.argv[1:6]; sys.path.insert(0, CHAIN); import stage_q_iam_a as Q, stage_q0_intake as Q0
B = "methylphys-data-945451304272-us-west-2-an"; s3 = boto3.client("s3"); os.makedirs(WORK, exist_ok=True)
H = lambda e: -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
def eps_of(path):
    T = Q.pat_site_table(path); o = float(T.opp_A.sum() + T.opp_B.sum()); return float(T.err_A.sum() + T.err_B.sum()) / o, int(o)
def rewrite(src, dst, f, seed):
    rr = random.Random(seed)
    with gzip.open(src, "rt") as a, gzip.open(dst, "wt", compresslevel=1) as b:
        for l in a:
            q = l.rstrip("\n").split("\t"); n = int(q[3])
            for _ in range(n): b.write(f"{q[0]}\t{q[1]}\t{f(q[2], rr)}\t1\n")
loss = lambda p, rr: "".join("T" if ch == "C" and rr.random() < 0.01 else ch for ch in p)
unconv = lambda p, rr: "".join("C" if ch == "T" and rr.random() < 0.02 else ch for ch in p)
R = pd.read_csv(RUNS); have = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["run"])
for r in R.itertuples():
    if r.run in set(have.run): continue
    pre = f"results/BOXRUN10_EPIQC/{TAG}/{r.run}"; pat = os.path.join(WORK, f"{r.run}.pat.gz"); qcj = os.path.join(WORK, f"{r.run}_qc.json")
    try: s3.download_file(B, pre + ".pat.gz", pat); s3.download_file(B, pre + "_alignment_qc.json", qcj)
    except Exception as e: print(r.run, "not in S3", str(e)[:60], flush=True); continue
    qc = json.load(open(qcj)); conv = qc.get("conversion_rate_human_chr1")
    I = Q0.intake(pat, "isolated neutrophils", alignment_qc={"conversion_rate": conv, "duplicate_fraction": qc.get("duplicate_fraction")})
    bad = [c for c in I["checks"] if c["step"] != "Q0.7 specimen" and c["result"] == "REFUSED"]
    e, o = eps_of(pat); tmp = pat + ".plant.pat.gz"
    rewrite(pat, tmp, loss, 20261011); ep, _ = eps_of(tmp); os.remove(tmp)
    c1 = np.nan
    if r.stage == "A": rewrite(pat, tmp, unconv, 20261012); ec, _ = eps_of(tmp); os.remove(tmp); c1 = H(ec) / H(e)
    row = dict(kit=r.kit, lab=r.lab, line=r.line, rep=r.rep, run=r.run, eps=e, opp=o, conversion=conv, conversion_lambda=qc.get("conversion_rate_lambda"),
               duplicate_fraction=qc.get("duplicate_fraction"), q0="PROCEED" if not bad else bad[0]["step"], eps_planted=ep, c1_iama=c1)
    have = pd.concat([have, pd.DataFrame([row])], ignore_index=True); have.to_csv(OUT, index=False); os.remove(pat); print(row, flush=True)
