"""DEV-IAMA-CONVERSION-01: ε by Stage Q (whole file) for all 14 Box Run 2 session 4 whole-blood runs, with c from each run's
alignment_qc.json (lambda CHH). Prints ε_meas, ε/c, IAM-A rep2/rep1 before and after the correction. Run: python3 score_conversion_01.py CHAIN WORKDIR OUT.csv"""
import os, sys, json, math, re, boto3, pandas as pd
CH, W, OUT = sys.argv[1:4]; sys.path.insert(0, CH); import stage_q_iam_a as Q
B = "methylphys-data-945451304272-us-west-2-an"; s3 = boto3.client("s3"); HERE = os.path.dirname(os.path.abspath(__file__))
RR = pd.read_csv(os.path.join(HERE, "../../../boxruns/run2/GSE128731_runs.csv")).set_index("Run")
def H(e): return -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
rows = []
SESSION4 = [o["Key"].split("/")[-1][:10] for pg in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix="results/BOXRUN2/session4/")
            for o in pg.get("Contents", []) if o["Key"].endswith(".pat.gz")]   # the 14 runs of DEV-IAMA-WBTARE-01
for run in SESSION4:
    p = os.path.join(W, run + ".pat.gz")
    if not os.path.exists(p): s3.download_file(B, f"results/BOXRUN2/session4/{run}.pat.gz", p)
    c = json.loads(s3.get_object(Bucket=B, Key=f"results/BOXRUN2/session4/{run}_alignment_qc.json")["Body"].read())["conversion_rate"]
    T = Q.pat_site_table(p); e = float(T.err_A.sum() + T.err_B.sum()) / float(T.opp_A.sum() + T.opp_B.sum())
    m = re.match(r"(Sample\d)_(Swift|TruSeq)_HiSeqX(?:_(rep\d))?", RR.loc[run, "title"])
    rows.append(dict(run=run, donor=m.group(1), kit=m.group(2), rep=m.group(3) or "rep1", c=c, eps=e, eps_corr=e / c)); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
for (d, k), g in R.groupby(["donor", "kit"]):
    if set(g.rep) >= {"rep1", "rep2"}:
        a, b = g.set_index("rep").loc["rep1"], g.set_index("rep").loc["rep2"]
        print(f"{d} {k}: c {a.c:.4f}/{b.c:.4f} | eps ratio {b.eps / a.eps:.4f} (pred {b.c / a.c:.4f}) | IAM-A rep2/rep1 raw {H(b.eps) / H(a.eps):.4f}, corrected {H(b.eps_corr) / H(a.eps_corr):.4f}")
