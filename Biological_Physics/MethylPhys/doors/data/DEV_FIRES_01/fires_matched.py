"""DEV-FIRES-01, added after the first reading (disclosed): breast cases against cancer-free women matched on EVERY quality measure at once.
Women only. Each case is paired with the cancer-free array on its own chip nearest in Stage 0's quality (standardized distance over the four
measures of fires_quality.py: bisulfite conversion I and II, red signal, call rate; standardized on the cancer-free), without replacement; a
pair is kept only if every measure is within a quarter of its cancer-free spread. The paired difference of the tared whole-sample Met-A, its
sign test and Wilcoxon p, by window. Nothing fitted.  Usage: python3 fires_matched.py FOLDER_OF_BUNDLES [OUT.txt]"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); OUT_M = sys.argv[2] if len(sys.argv) > 2 else os.path.join(sys.argv[1], "fires_matched_sexF.txt")
sys.argv = [sys.argv[0], sys.argv[1], "/dev/null"]
exec(open(os.path.join(H, "fires_quality.py")).read().split("L = [f\"DEV-FIRES-01 quality check")[0])
Qk = list(Q); Z = X.copy()
for k in Qk: Z[k] = pd.to_numeric(Z[k], errors="coerce")
cfZ = Z[Z.group == "cancer-free"]; mu, sd = cfZ[Qk].mean(), cfZ[Qk].std()
for k in Qk: Z[k + "_z"] = (Z[k] - mu[k]) / sd[k]
zk = [k + "_z" for k in Qk]; used, pairs = set(), []
cases = Z[Z.group == "breast"].dropna(subset=zk + ["A_rel"])
for g, c in cases.iterrows():
    pool = Z[(Z.group == "cancer-free") & (Z.slide == c.slide) & ~Z.index.isin(used)].dropna(subset=zk + ["A_rel"])
    if pool.empty: continue
    dist = np.sqrt(((pool[zk] - c[zk].astype(float)) ** 2).sum(1)); j = dist.idxmin()
    if (np.abs(pool.loc[j, zk].astype(float) - c[zk].astype(float)) <= 0.25).all(): used.add(j); pairs.append((g, j, c.A_rel - pool.loc[j, "A_rel"]))
P = pd.DataFrame(pairs, columns=["case", "control", "diff"]).set_index("case").join(X[["ttd"]])
win = pd.cut(pd.to_numeric(P.ttd, errors="coerce"), [0, 2, 5, 10, 100], labels=["0-2 y", "2-5 y", "5-10 y", "over 10 y"])
L = [f"DEV-FIRES-01 matched on every quality measure, women only: {len(P)} of {len(cases)} breast cases paired with a cancer-free array on the same chip (each measure within 0.25 of the cancer-free SD)"]
for wn in ["all", "0-2 y", "2-5 y", "5-10 y", "over 10 y"]:
    d = P["diff"] if wn == "all" else P[win == wn]["diff"]
    if len(d) < 6: L.append(f"{wn:9} {len(d)} pairs, too few"); continue
    L.append(f"{wn:9} {len(d):3} pairs | case minus control, tared Met-A: mean {100 * d.mean():+.2f} %, median {100 * d.median():+.2f} % | case higher in {int((d > 0).sum())} of {len(d)} (sign p {stats.binomtest(int((d > 0).sum()), len(d)).pvalue:.2g}) | Wilcoxon p {stats.wilcoxon(d).pvalue:.2g}")
q = Z.loc[P.index, zk].astype(float).values - Z.loc[P.control, zk].astype(float).values
L.append("quality after matching, case minus control (in cancer-free SD): " + ", ".join(f"{k} {m:+.3f}" for k, m in zip(Qk, q.mean(0))))
open(OUT_M, "w").write("\n".join(L) + "\n"); print("\n".join(L))
