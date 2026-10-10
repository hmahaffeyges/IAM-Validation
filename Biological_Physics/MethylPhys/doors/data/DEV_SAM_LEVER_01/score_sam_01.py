"""DEV-SAM-LEVER-01 scoring (committed before any knockout file is read). GSE77079 mouse liver RRBS, Box Run 6 .pat files (s3 results/BOXRUN6_SAM).
Per run ε by Stage Q (pat_site_table); each mouse's runs pooled (errors / opportunities); IAM-A per mouse = H(ε) / H(median wild-type ε).
Ratio = median knockout vehicle IAM-A ÷ median wild-type IAM-A (= median KO vehicle IAM-A). Sealed window 1.016–1.135 (DEV_SAM_LEVER_01.md).
PASS: ratio inside the window and one-sided Mann-Whitney p < 0.05 (KO vehicle > wild type). Below 1.016: undecided (middle-cell power 0.46).
Above 1.135: beyond the calculation. Second prediction: knockout + SAMe median below knockout vehicle median (one-sided p reported).
Usage: score_sam_01.py PAT_DIR"""
import os, sys, math, json, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(MP, "chain")); import stage_q_iam_a as Q
H = lambda e: -(e * math.log2(e) + (1 - e) * math.log2(1 - e))
d = pd.read_csv(os.path.join(MP, "boxruns/run6_sam/GSE77079_runs.csv")); rows = []
for _, r in d.iterrows():
    T = Q.pat_site_table(os.path.join(sys.argv[1], r.run_accession + ".pat.gz"))
    rows.append(dict(run=r.run_accession, group=r.group, mouse=r.mouse, errors=float(T.err_A.sum() + T.err_B.sum()), opportunities=float(T.opp_A.sum() + T.opp_B.sum())))
R = pd.DataFrame(rows); R["eps"] = R.errors / R.opportunities; R.to_csv(os.path.join(HERE, "sam_01_runs.csv"), index=False)
M = R.groupby(["group", "mouse"])[["errors", "opportunities"]].sum().reset_index(); M["eps"] = M.errors / M.opportunities
e0 = M[M.group == "wild type"].eps.median(); M["IAMA"] = M.eps.map(H) / H(e0); M.to_csv(os.path.join(HERE, "sam_01_mice.csv"), index=False)
g = {k: M[M.group == k].IAMA.values for k in M.group.unique()}
ratio = float(np.median(g["knockout vehicle"])); p1 = mannwhitneyu(g["knockout vehicle"], g["wild type"], alternative="greater").pvalue
p2 = mannwhitneyu(g["knockout SAMe"], g["knockout vehicle"], alternative="less").pvalue
print(M.sort_values(["group", "IAMA"]).round(4).to_string(index=False))
v = "PASS" if (1.016 <= ratio <= 1.135 and p1 < 0.05) else ("UNDECIDED (below the window; power 0.46)" if ratio < 1.016 else ("BEYOND the calculation" if ratio > 1.135 else "inside the window, p >= 0.05: UNDECIDED"))
print(f"KO vehicle / WT ratio {ratio:.4f} (window 1.016-1.135) | p (KO > WT) {p1:.4f} | -> {v}")
print(f"KO + SAMe median {np.median(g['knockout SAMe']):.4f} vs KO vehicle {ratio:.4f} | p (SAMe < vehicle) {p2:.4f}")
