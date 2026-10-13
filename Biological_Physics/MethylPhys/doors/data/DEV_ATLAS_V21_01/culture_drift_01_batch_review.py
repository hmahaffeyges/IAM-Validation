"""DEV-ATLAS-V21-01 culture check step 1, review (descriptive, after the step; not a rule change). In five of the seven GSE197512 lines the arrays
were run on two dates, and date follows population doublings (rho 0.75-0.79 in four), so part of the drift set may be array batch. Two lines were
run on one date (Fetal_skin_fibroblast_1, Neonatal_foreskin_keratinocyte_1: two cell types). Here: the CpGs that drift in BOTH of those lines with
the same sign (|rho| >= 0.5, the step's cut), i.e. the step's rule on batch-free lines only, and its overlap with the step's set.
Writes culture_drift_01_batch_review_output.txt and culture_drift_01_batchfree_cpgs.csv (cpg, sign of the trend). Usage: python3 culture_drift_01_batch_review.py BETADIR"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); L = pd.read_csv(os.path.join(HERE, "culture_gse197512_baseline.csv"))
CLEAN = ["Fetal_skin_fibroblast_1", "Neonatal_foreskin_keratinocyte_1"]; CUT = 0.5
def rho(B, x):
    ok = B.notna().sum(1) >= 8; Bv = B[ok]
    R = Bv.rank(axis=1); xr = pd.Series(rankdata(x), index=B.columns)
    Rc = R.sub(R.mean(1), axis=0); xc = xr - xr.mean()
    return (Rc.mul(xc, axis=1).sum(1) / np.sqrt((Rc ** 2).sum(1) * (xc ** 2).sum()))
rs = {}
for c in CLEAN:
    g = L[L.cell_line == c]
    B = pd.concat({x: pd.read_parquet(os.path.join(sys.argv[1], f"{x}.parquet")).beta for x in g.gsm}, axis=1)
    rs[c] = rho(B, g.set_index("gsm").population_doublings.loc[B.columns].values)
R = pd.concat(rs, axis=1).dropna()
both = ((R.abs() >= CUT).all(1)) & (np.sign(R.iloc[:, 0]) == np.sign(R.iloc[:, 1]))
S = set(pd.read_csv(os.path.join(HERE, "culture_drift_01_cpgs.csv")).cpg)
clean = set(R.index[both])
out = [f"CpGs scored in both single-date lines: {len(R)}",
       f"drift in both single-date lines, same sign: {len(clean)} ({100 * len(clean) / len(R):.2f} %)",
       f"step 1 set: {len(S)}; of the batch-free set, in step 1's set: {len(clean & S)} ({100 * len(clean & S) / max(len(clean), 1):.1f} %)",
       "sign agreement of the two lines where both |rho| >= 0.5: " + f"{100 * float(((np.sign(R.iloc[:, 0]) == np.sign(R.iloc[:, 1])) & (R.abs() >= CUT).all(1)).sum() / max(int((R.abs() >= CUT).all(1).sum()), 1)):.1f} %"]
pd.DataFrame({"cpg": sorted(clean), "sign": [int(np.sign(R.loc[c].iloc[0])) for c in sorted(clean)]}).to_csv(os.path.join(HERE, "culture_drift_01_batchfree_cpgs.csv"), index=False)
open(os.path.join(HERE, "culture_drift_01_batch_review_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
