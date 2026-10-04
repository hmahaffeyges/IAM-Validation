#!/usr/bin/env python3
"""DEVELOPMENT - not commissioned. Round-2 follow-up: 11b interval on the GSE250556 pooled replicates (dev_stages fix: sites without an
expectation dropped), and the EPIC v2 cross-version pair reading at the identity sites both arrays of a pair measure (DEV-EPIC-V2-01 check 3)."""
import os, sys, json, glob
import numpy as np, pandas as pd
W = os.getcwd(); CH = f"{W}/repo/Biological_Physics/MethylPhys/chain"; sys.path[:0] = [CH]; OUT = f"{W}/fu"; os.makedirs(OUT, exist_ok=True)
import conductor_v3 as C3, dev_stages as DV
BET1 = "/home/ubuntu/data/base_chain_01/betas"
def beta(p):
    b = pd.read_parquet(p).iloc[:, 0].astype("float64"); b.index = b.index.astype(str); return b
R = pd.read_csv(f"{W}/repl.csv"); R = R[R.pooled == True]; rows = []
for g, per in zip(R.gsm, R.person):
    p = f"{BET1}/{g}.parquet"
    if not os.path.exists(p): continue
    b = beta(p); r = DV.brightness(b, "whole blood", C3.stage_a_composition(b))
    rows.append(dict(gsm=g, person=per, A=r.get("A"), lo=(r.get("interval_95") or [None, None])[0], hi=(r.get("interval_95") or [None, None])[1], hw=r.get("half_width"), status=r.get("status")))
BR = pd.DataFrame(rows); BR.to_csv(f"{OUT}/brightness11b.csv", index=False); pr = []
for p_, g in BR.dropna(subset=["A", "hw"]).groupby("person"):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        for j in range(i + 1, len(g)): pr.append(dict(person=p_, d=abs(g.A[i] - g.A[j]), lim=float(np.sqrt(g.hw[i] ** 2 + g.hw[j] ** 2))))
PR = pd.DataFrame(pr); PR.to_csv(f"{OUT}/brightness11b_pairs.csv", index=False)
S = {"brightness": {"n": int(BR.A.notna().sum()), "pairs": len(PR), "covered_frac": float((PR.d <= PR.lim).mean()) if len(PR) else None,
                    "half_width_median": float(BR.hw.median()), "pair_absdiff_median": float(PR.d.median()) if len(PR) else None, "bar": 0.95}}
V = sys.argv[1]; A2 = pd.read_csv(f"{V}/epicv2_arrays.csv"); A2["pair"] = A2.title.str.replace(r"_EPICv[12]$", "", regex=True)
B = C3._bc(); IS = pd.Index(B["neutrophil_sites"]); P = {k: pd.Series(v, index=IS, dtype="float64") for k, v in B["profiles_at_neutrophil_sites"].items()}
H = DV._H; out = []
for pr_, g in A2.groupby("pair"):
    if set(g.plat) != {"EPIC_v1", "EPIC_v2"}: continue
    g1, g2 = g[g.plat == "EPIC_v1"].gsm.iloc[0], g[g.plat == "EPIC_v2"].gsm.iloc[0]
    if not (os.path.exists(f"{V}/{g1}_sesame.parquet") and os.path.exists(f"{V}/{g2}_sesame.parquet")): continue
    b1, b2 = beta(f"{V}/{g1}_sesame.parquet"), beta(f"{V}/{g2}_sesame.parquet")
    c1, c2 = C3.stage_a_composition(b1), C3.stage_a_composition(b2)
    s = IS[b1.reindex(IS).notna().values & b2.reindex(IS).notna().values]
    def A(b, c):
        if not c.get("fractions"): return None
        e = sum(v * P[k] for k, v in c["fractions"].items() if k in P).reindex(s); ok = e.notna(); return float(H(b.reindex(s)[ok].values).mean() / H(e[ok].values).mean())
    mp = f"{BET1}/{g1}.parquet"; bm = beta(mp) if os.path.exists(mp) else None
    out.append(dict(pair=pr_, v1=g1, v2=g2, n_shared_sites=len(s), A_v1=A(b1, c1), A_v2=A(b2, c2), A_v1_methylprep=A(bm, C3.stage_a_composition(bm)) if bm is not None else None,
                    fneu_v1=(c1.get("fractions") or {}).get("NEU"), fneu_v2=(c2.get("fractions") or {}).get("NEU"),
                    A_v2_own_composition_from_v1=A(b2, c1)))
X = pd.DataFrame(out); X["d"] = X.A_v2 - X.A_v1; X["d_samecomp"] = X.A_v2_own_composition_from_v1 - X.A_v1; X.to_csv(f"{OUT}/epicv2_pairs.csv", index=False)
d = X.d.dropna(); d2 = X.d_samecomp.dropna()
S["epicv2_pairs"] = {"pairs": int(len(X)), "pairs_read": int(len(d)), "shared_sites_median": float(X.n_shared_sites.median()) if len(X) else None,
                     "diff_mean": float(d.mean()), "diff_sd": float(d.std()), "diff_samecomp_mean": float(d2.mean()), "diff_samecomp_sd": float(d2.std()), "target_sd": 0.020,
                     "note": "cross-version reading at the identity sites both arrays measure (below Stage M's 90 % coverage rule; development only)"}
json.dump(S, open(f"{OUT}/followup_summary.json", "w"), indent=1); print(json.dumps(S, indent=1))
