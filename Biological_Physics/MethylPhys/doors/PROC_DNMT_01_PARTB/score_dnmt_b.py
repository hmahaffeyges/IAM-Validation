#!/usr/bin/env python3
"""PROC-DNMT-01 Part B scoring, as pre-registered (PROC_DNMT_01_PREREG.md, Part B).
Copy error eps on qualifying molecules (same extraction and genotype mask as PROC-TUMOUR-01 score_tumour.py), corrected for the
substitution error rate. Reference = the same genotype's DMSO libraries (mean eps of its replicates).
Q1: H(eps) treated / H(eps) DMSO > 1.05 in every genotype, both replicates.  Q2: |conv_fail treated - conv_fail DMSO| < 0.005."""
import glob, os, re, ast, json, numpy as np, pandas as pd
O = "/home/ubuntu/data/tumour/out"
SP = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{O}/DNMT_*.parquet"))
def parse(sp):
    ds, pt, assay, kind = sp.split("_"); g, rep = pt.rsplit("-", 1); return dict(sp=sp, genotype=g, rep=rep, assay=assay, kind=kind)
meta = pd.DataFrame([parse(s) for s in SP])
def inst(sp):
    line = re.findall(r"\{'reads'.*\}", open(f"{O}/{sp}_extract.log").read())[-1]
    line = re.sub(r"np\.(?:float|int)\d*\(([^)]*)\)", r"\1", line); return ast.literal_eval(line)
def H(x): x = min(max(x, 1e-12), 1 - 1e-12); return float(-(x * np.log2(x) + (1 - x) * np.log2(1 - x)))
T = {s: pd.read_parquet(f"{O}/{s}.parquet") for s in SP}; mask = {}
for g, gg in meta.groupby("genotype"):
    D = pd.concat([T[s][["pos", "opp_A", "opp_B", "err_A", "err_B"]] for s in gg.sp]).groupby("pos").sum(); o = D.opp_A + D.opp_B; e = D.err_A + D.err_B
    mask[g] = set(D.index[(o >= 5) & (e > 0.30 * o)])
rows = []
for _, m in meta.iterrows():
    D = T[m.sp]; D = D[~D.pos.isin(mask[m.genotype])]; I = inst(m.sp)
    ep = (D.err_A.sum() + D.err_B.sum()) / (D.opp_A.sum() + D.opp_B.sum())
    rows.append(dict(**m, reads=I["reads"], qualifying=I["qualifying"], sites=len(D), conv_fail=I["conv_fail"], sub_err=I["sub_err"],
                     eps=ep, eps_corr=ep - I["sub_err"]))
R = pd.DataFrame(rows); R.to_csv("dnmt_b_readings.csv", index=False)
out = []
for g, gg in R.groupby("genotype"):
    ref = gg[gg.kind == "DMSO"]; e0 = ref.eps_corr.mean(); c0 = ref.conv_fail.mean()
    for _, t in gg[gg.kind != "DMSO"].iterrows():
        out.append(dict(genotype=g, rep=t.rep, kind=t.kind, eps_treated=t.eps_corr, eps_dmso=e0, n_dmso=len(ref),
                        A=H(t.eps_corr) / H(e0), conv_diff=abs(t.conv_fail - c0)))
P = pd.DataFrame(out); P["Q1"] = P.A > 1.05; P["Q2"] = P.conv_diff < 0.005; P.to_csv("dnmt_b_pairs.csv", index=False)
S = dict(n_libraries=len(R), genotypes=sorted(R.genotype.unique()), n_treated=len(P), Q1_pass=int(P.Q1.sum()), Q2_pass=int(P.Q2.sum()),
         Q1_all=bool(len(P) > 0 and P.Q1.all()), Q2_all=bool(len(P) > 0 and P.Q2.all()), A_range=[float(P.A.min()), float(P.A.max())] if len(P) else None)
json.dump(S, open("dnmt_b_summary.json", "w"), indent=1)
print(R.drop(columns=["sp"]).round(5).to_string(index=False)); print(P.round(4).to_string(index=False)); print(json.dumps(S)); print("DONE")
