"""DEV-IAMA-KIT-01 part 2: Swift (SRR9888332) and TruSeq (SRR9888333), donor 6, clip 0, read through Stage Q on (a) all their own sites and
(b) only the CpG positions both runs cover (opportunities > 0 in both). Also ε by CpG-density bin of the position's neighbourhood."""
import sys, json, numpy as np, pandas as pd
CH = sys.argv[1]; sys.path.insert(0, CH); import stage_q_iam_a as Q
W = "/mnt/scratch/clip/pat_c0/"
T = {r: Q.pat_site_table(W + r + ".pat.gz") for r in ("SRR9888332", "SRR9888333")}
out = {}
for r, t in T.items():
    t["opp"] = t.opp_A + t.opp_B; t["err"] = t.err_A + t.err_B
    out[r] = {"all_sites": {k: Q.read(t.drop(columns=["opp", "err"]), cell="neutrophils", pipeline="loyfer_pat_v1").get(k) for k in ("A", "eps", "opportunities")}, "n_pos": int((t.opp > 0).sum())}
common = set(T["SRR9888332"].pos[T["SRR9888332"].opp > 0]) & set(T["SRR9888333"].pos[T["SRR9888333"].opp > 0])
out["n_common_positions"] = len(common)
for r, t in T.items():
    tc = t[t.pos.isin(common)].drop(columns=["opp", "err"]).copy(); tc.attrs = dict(t.attrs)
    out[r]["common_sites"] = {k: Q.read(tc, cell="neutrophils", pipeline="loyfer_pat_v1").get(k) for k in ("A", "eps", "opportunities")}
    # per-position eps weighted by the OTHER run's opportunities (same weighting for both runs)
J = T["SRR9888332"][["pos", "opp", "err"]].merge(T["SRR9888333"][["pos", "opp", "err"]], on="pos", suffixes=("_s", "_t"))
J = J[(J.opp_s > 0) & (J.opp_t > 0)]
J["eps_s"] = J.err_s / J.opp_s; J["eps_t"] = J.err_t / J.opp_t; w = np.minimum(J.opp_s, J.opp_t)
out["same_weight_eps"] = {"swift": float(np.average(J.eps_s, weights=w)), "truseq": float(np.average(J.eps_t, weights=w))}
ix = pd.to_numeric(J.pos, errors="coerce"); J["bin"] = pd.qcut(J.opp_s / (J.opp_t + 1e-9), 5, labels=False, duplicates="drop")
out["eps_by_swift_over_truseq_depth_quintile"] = J.groupby("bin").apply(lambda x: {"swift": float(x.err_s.sum() / x.opp_s.sum()), "truseq": float(x.err_t.sum() / x.opp_t.sum()), "n": int(len(x))}).to_dict()
json.dump(out, open("/mnt/scratch/clip/same_sites.json", "w"), indent=1, default=float); print(json.dumps(out, default=float)[:2500])
