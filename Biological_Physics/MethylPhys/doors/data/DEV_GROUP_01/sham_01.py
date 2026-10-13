"""DEV-GROUP-01 step 1, the sham arm: two HEALTHY whole-blood groups from two laboratories, compared on the chain's readings.

# NO CHAIN: arithmetic over readings the chain already produced (A untared, A_rel tared). No physics and no new reading here.

Inputs, both committed or in object storage from the runs that produced them:
  GSE315366  results/BOXRUN3/GSE315366_rows.csv            - clonal-haematopoiesis-NEGATIVE whole bloods only
  GSE184269  CSCORE_COMMISSIONING/band_test_GSE184269_rows.csv - whole bloods only, arrays the chain did not refuse

Bars are in DEV_GROUP_01.md, committed before this ran: the TARED difference must be under 2 % and under twice its standard error.
Writes sham_01_output.csv and sham_01_summary.txt. Usage: python3 sham_01.py
"""
import io, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
import boto3
B = "methylphys-data-945451304272-us-west-2-an"
G1 = pd.read_csv(io.BytesIO(boto3.client("s3").get_object(Bucket=B, Key="results/BOXRUN3/GSE315366_rows.csv")["Body"].read()))
G1 = G1[(G1["clonal hematopoiesis"].astype(str) == "False") & G1.A.notna()].copy(); G1["set"] = "GSE315366"
p2 = os.path.join(HERE, "..", "CSCORE_COMMISSIONING", "band_test_GSE184269_rows.csv")
G2 = pd.read_csv(p2); G2 = G2[G2.specimen.str.contains("blood", case=False) & G2.A.notna()].copy(); G2["set"] = "GSE184269"
L = ["DEV-GROUP-01 step 1: the sham arm - two healthy whole-blood groups, two laboratories", ""]
rows = []
for col, what in (("A", "UNTARED (raw reading)"), ("A_rel", "TARED (each array against healthy references in its own series)")):
    a, b = G1[col].dropna().astype(float), G2[col].dropna().astype(float)
    if not len(a) or not len(b): L.append(f"{what}: not available in one of the sets"); continue
    d = a.mean() - b.mean(); se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    rel = d / np.mean([a.mean(), b.mean()])
    rows.append(dict(reading=col, n1=len(a), n2=len(b), mean1=a.mean(), mean2=b.mean(), sd1=a.std(ddof=1), sd2=b.std(ddof=1),
                     difference=d, rel_difference=rel, se=se, t=d / se if se else np.nan))
    L += [f"{what}",
          f"  GSE315366 (lab 1): n {len(a):>3}  mean {a.mean():.4f}  SD {a.std(ddof=1):.4f}",
          f"  GSE184269 (lab 2): n {len(b):>3}  mean {b.mean():.4f}  SD {b.std(ddof=1):.4f}",
          f"  difference {d:+.4f} ({rel:+.2%} of the common mean), standard error {se:.4f}, that is {abs(d)/se:.2f} standard errors", ""]
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "sham_01_output.csv"), index=False)
tr = T[T.reading == "A_rel"]
if len(tr):
    r = tr.iloc[0]; met = (abs(r.rel_difference) < 0.02) and (abs(r.t) < 2)
    L += [f"BAR 1 (tared sham must be null: under 2 % AND under two standard errors): "
          f"{abs(r.rel_difference):.2%} and {abs(r.t):.2f} SE -> {'MET' if met else 'NOT MET'}"]
    ur = T[T.reading == "A"]
    if len(ur): L.append(f"What the tare does (reported, not barred): the untared difference is {abs(ur.iloc[0].rel_difference):.2%}, "
                         f"the tared difference is {abs(r.rel_difference):.2%}")
open(os.path.join(HERE, "sham_01_summary.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
