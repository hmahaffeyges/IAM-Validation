#!/usr/bin/env python3
"""DEV-COHO-CC-01 scorer (rules in DEV_COHO_CC_01_NOTE.md, committed before scoring).
Input: a folder of per-fish site tables <fish>.parquet and <fish>_extract.log (from extract_se.py, columns *_cc), and leluyer_runs.csv.
Output: coho_cc_fish.csv (per fish) and coho_cc_summary.json; prints the D1/D2 checks.
Usage: python3 score_coho_cc.py <folder> leluyer_runs.csv"""
import sys, os, json, ast, math, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

D, RUNS = sys.argv[1], sys.argv[2]
meta = pd.read_csv(RUNS).set_index("sample_title")
fish = sorted(os.path.basename(f)[:-8] for f in glob.glob(f"{D}/*.parquet"))
COLS = ["pos", "opp_A", "err_A", "opp_B", "err_B", "opp_A_cc", "err_A_cc", "opp_B_cc", "err_B_cc"]

# pass 1: common sites = >= 3 conversion-filtered opportunities in >= 90 % of fish
cnt = {}
for f in fish:
    T = pd.read_parquet(f"{D}/{f}.parquet", columns=["pos", "opp_A_cc", "opp_B_cc"])
    p = T.pos.values[(T.opp_A_cc.values + T.opp_B_cc.values) >= 3]
    u, c = np.unique(p, return_counts=True)
    for k in u: cnt[k] = cnt.get(k, 0) + 1
need = math.ceil(0.9 * len(fish))
common = np.array(sorted(k for k, v in cnt.items() if v >= need), dtype=np.int64)
del cnt

def rate(e, o): return float(e.sum() / o.sum()) if o.sum() > 0 else float("nan")
rows = []
for f in fish:
    T = pd.read_parquet(f"{D}/{f}.parquet", columns=COLS)
    L = open(f"{D}/{f}_extract.log").read().strip().splitlines()
    S = ast.literal_eval([l for l in L if l.startswith("{")][-1])
    C = T[np.isin(T.pos.values, common)]
    r = dict(fish=f, origin=meta.loc[f, "origin"], sex=meta.loc[f, "sex"], lane=str(meta.loc[f, "lane"]),
             qualifying=S["qualifying"], qualifying_cc=S["qualifying_cc"], conv_fail=S["conv_fail"], sub_err=S["sub_err"],
             eps_all=rate(T.err_A + T.err_B, T.opp_A + T.opp_B), eps_cc=rate(T.err_A_cc + T.err_B_cc, T.opp_A_cc + T.opp_B_cc),
             eps_all_common=rate(C.err_A + C.err_B, C.opp_A + C.opp_B),
             eps_cc_common=rate(C.err_A_cc + C.err_B_cc, C.opp_A_cc + C.opp_B_cc),
             eps_cc_common_A=rate(C.err_A_cc, C.opp_A_cc), eps_cc_common_B=rate(C.err_B_cc, C.opp_B_cc),
             opp_cc_common=int((C.opp_A_cc + C.opp_B_cc).sum()))
    r["E_kT"] = math.log((1 - r["eps_cc_common"]) / r["eps_cc_common"])
    rows.append(r)
F = pd.DataFrame(rows); F.to_csv("coho_cc_fish.csv", index=False)

def icc1(a, b):
    X = np.c_[a, b]; n, k = X.shape; gm = X.mean()
    msb = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msw = ((X - X.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw))
out = dict(n_fish=len(F), n_common_sites=int(len(common)),
           D1_icc=icc1(F.eps_cc_common_A, F.eps_cc_common_B))
for q in ("eps_cc_common", "eps_all_common"):
    out[f"rho_{q}_conv"] = float(spearmanr(F[q], F.conv_fail).correlation)
    out[f"rho_{q}_depth"] = float(spearmanr(F[q], F.qualifying).correlation)
    out[f"rho_{q}_suberr"] = float(spearmanr(F[q], F.sub_err).correlation)
out["D1_pass"] = out["D1_icc"] >= 0.9
out["D2_pass"] = abs(out["rho_eps_cc_common_conv"]) < 0.3 and abs(out["rho_eps_cc_common_depth"]) < 0.3
if out["D1_pass"] and out["D2_pass"]:
    for g, (a, b) in (("origin", ("hatchery", "wild")), ("sex", ("female", "male"))):
        x, y = F[F[g] == a].eps_cc_common, F[F[g] == b].eps_cc_common
        if len(x) and len(y):
            out[f"{g}_{a}_median"] = float(x.median()); out[f"{g}_{b}_median"] = float(y.median())
            out[f"{g}_n"] = [int(len(x)), int(len(y))]; out[f"{g}_mwu_p"] = float(mannwhitneyu(x, y).pvalue)
            out[f"{g}_dE_kT"] = float(math.log((1 - x.median()) / x.median()) - math.log((1 - y.median()) / y.median()))
json.dump(out, open("coho_cc_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print(F[["fish", "origin", "sex", "lane", "conv_fail", "eps_all_common", "eps_cc_common", "eps_cc_common_A", "eps_cc_common_B", "E_kT"]].round(5).to_string(index=False))
