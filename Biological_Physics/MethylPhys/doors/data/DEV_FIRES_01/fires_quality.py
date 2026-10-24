"""DEV-FIRES-01, added after the first reading (disclosed): is the breast shift the arrays' quality? Women only. Each array's Group-A tared
whole-sample Met-A (cancer-free on the same chip), and Stage 0's own quality measures. Breast cases against the cancer-free within thirds of
each quality measure (cut points from the cancer-free alone), and the shift's rank correlation with quality among the cancer-free. Nothing fitted.
Usage: python3 fires_quality.py FOLDER_OF_BUNDLES [OUT.txt]"""
import sys, os, glob, json, numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, "..", "..", "..", "chain")); import stage_groupa as GA
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "fires_quality_sexF.txt")
S = pd.read_csv(os.path.join(H, "gse51032_samples.csv"), dtype={"slide": str}).set_index("gsm")
Q = {"conversion II": ("controls", "bisulfite_conversion_II_median"), "conversion I": ("controls", "bisulfite_conversion_I_median"),
     "signal red": ("controls", "non_polymorphic_R_median"), "call rate": ("stage1_qc", "call_rate")}
rows, seen = [], set()
for f in sorted(glob.glob(os.path.join(src, "**", "GSM*.json"), recursive=True)):
    g = os.path.basename(f).split(".")[0]
    if g in seen or g not in S.index: continue
    seen.add(g); B = json.load(open(f)); it = B.get("intake") or {}; r = {"gsm": g, "A": (B.get("met_a_whole") or {}).get("A")}
    for k, (a, b) in Q.items(): r[k] = (it.get(a) or {}).get(b)
    rows.append(r)
X = pd.DataFrame(rows).set_index("gsm").join(S); X = X[X.sex == "F"]
X["group"] = np.where(X.cancer == "none", "cancer-free", np.where(X.cancer == "C50", "breast", "other")); X = X[X.group != "other"]
R = X.reset_index()[["gsm", "A", "slide", "group"]]; R["C"] = np.nan
T = GA.tare(R, (X.reset_index().ref.astype(str).str.lower() == "yes")).set_index("gsm"); X["A_rel"] = T.A_rel
L = [f"DEV-FIRES-01 quality check, women only: {int((X.group == 'breast').sum())} breast, {int((X.group == 'cancer-free').sum())} cancer-free; tared whole-sample Met-A"]
cf = X[X.group == "cancer-free"]
for k in Q:
    q = pd.to_numeric(X[k], errors="coerce"); rho, p = stats.spearmanr(q[cf.index], cf.A_rel, nan_policy="omit")
    L.append(f"-- {k}: among the cancer-free, rho with tared Met-A {rho:+.2f} (p {p:.2g}); cases' median {q[X.group == 'breast'].median():.4g}, cancer-free {q[cf.index].median():.4g}")
    cuts = q[cf.index].quantile([1 / 3, 2 / 3]).to_numpy(); th = pd.cut(q, [-np.inf, cuts[0], cuts[1], np.inf], labels=["low", "middle", "high"])
    for t in ("low", "middle", "high"):
        a, b = X[(th == t) & (X.group == "breast")].A_rel.dropna(), X[(th == t) & (X.group == "cancer-free")].A_rel.dropna()
        if len(a) < 5: L.append(f"   {t:6} third: {len(a)} cases, too few"); continue
        sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        L.append(f"   {t:6} third: {len(a):3} cases, {len(b):3} cancer-free | diff {100 * (a.mean() - b.mean()):+.2f} % | d {(a.mean() - b.mean()) / sd:+.2f} | p {stats.mannwhitneyu(a, b).pvalue:.2g}")
open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L))
