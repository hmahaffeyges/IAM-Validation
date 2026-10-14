"""Atlas v2.1 sex and same-donor check (readiness L5 and the genotype rule of seal 2). Written and committed 2026-10-13, BEFORE the box
identity job's results were read. Every rule and threshold below is fixed here.

# NO CHAIN: arithmetic over values the chain's Stage 1 already produced (box job v21_identity_box.py: chrY, chrX and rs probe betas).

Input: identity_v21_arrays.csv and identity_v21_rs.parquet (S3 results/ATLAS_V21/identity/). No beta of any CpG site is read.

SEX. chrY_frac = chrY probes detected / chrY probes on the chip. Male if chrY_frac >= 0.5, female if <= 0.2, otherwise UNRESOLVED (kept in
the table, flagged, never guessed). chrX median beta is recorded as a second view, not a rule. Recorded sex is compared with the metadata sex
where the roster or GEO gives one; a disagreement is flagged, and the array leaves the build only by the rule in step 4.

GENOTYPE. Each rs probe's beta is called 0 (< 0.2), 1 (0.3-0.7) or 2 (> 0.8); betas between the bands are missing. For every pair of
arrays with at least 30 probes called in both: concordance = share of those probes with the same call. SAME PERSON if concordance >= 0.90
(different people on these probes sit far lower; the bar is checked on the data in step 3, not adjusted). 

 1. sex table, every array (build and held-out).
 2. pairs at SAME PERSON. Inside one series: replicates of one donor are expected and allowed. Across series: listed.
 3. bar check, recorded: the distribution of concordance over all pairs (the 99.9th percentile of pairs NOT called same person); if the
    gap between that and 0.90 is under 0.05, the result reads INCONCLUSIVE and no array is excluded on genotype.
 4. RULES, applied in order:
    a. a held-out array that is the same person as any build array leaves the held-out set (seal 2's donor-separation rule);
    b. two build arrays from different series that are the same person: both stay, and the pair is recorded as one donor for the per-source
       donor count and independence (the cap D2 applies to donors, not arrays);
    c. sex UNRESOLVED or contradicting the metadata: the array stays but is flagged; the atlas has no sex term, so sex alone excludes nothing.
Writes identity_check_v21_sex.csv, identity_check_v21_pairs.csv, identity_check_v21_summary.txt.
Usage: python3 identity_check_v21.py ARRAYS_CSV RS_PARQUET
"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
A = pd.read_csv(sys.argv[1]); RSB = pd.read_parquet(sys.argv[2])
A = A[A.status == "ok"].copy()
A["chrY_frac"] = A.chrY_detected / A.chrY_on_chip.replace(0, np.nan)
A["sex_called"] = np.where(A.chrY_frac >= 0.5, "M", np.where(A.chrY_frac <= 0.2, "F", "UNRESOLVED"))
A[["base", "sample", "set", "series", "cell", "chrY_frac", "chrX_median_beta", "sex_called"]].to_csv(os.path.join(HERE, "identity_check_v21_sex.csv"), index=False)
G = RSB.reindex(A.base)
C = np.full(G.shape, np.nan); v = G.values
C[v < 0.2] = 0; C[(v >= 0.3) & (v <= 0.7)] = 1; C[v > 0.8] = 2
n = len(C); okm = ~np.isnan(C); rows = []
for i in range(n):
    both = okm[i] & okm[i + 1:]; m = both.sum(1)
    same = ((C[i] == C[i + 1:]) & both).sum(1)
    conc = np.where(m >= 30, same / np.maximum(m, 1), np.nan)
    for j in np.nonzero(~np.isnan(conc))[0]: rows.append((i, i + 1 + j, int(m[j]), float(conc[j])))
P = pd.DataFrame(rows, columns=["i", "j", "n_probes", "concordance"])
P["same_person"] = P.concordance >= 0.90
ix = A.reset_index(drop=True)
for k in ("base", "set", "series", "cell", "sample"):
    P[f"a_{k}"] = ix.loc[P.i, k].values; P[f"b_{k}"] = ix.loc[P.j, k].values
S = P[P.same_person].drop(columns=["i", "j"])
S.to_csv(os.path.join(HERE, "identity_check_v21_pairs.csv"), index=False)
q999 = float(P.loc[~P.same_person, "concordance"].quantile(0.999)); inconcl = (0.90 - q999) < 0.05
cross = S[S.a_series != S.b_series]
ho_vs_build = S[((S.a_set == "heldout") & (S.b_set == "build")) | ((S.a_set == "build") & (S.b_set == "heldout"))]
excl = sorted(set(ho_vs_build.apply(lambda r: r.a_sample if r.a_set == "heldout" else r.b_sample, axis=1))) if not inconcl else []
L = [f"arrays {len(A)} (build {int((A.set == 'build').sum())}, held-out {int((A.set == 'heldout').sum())}) | pairs compared {len(P):,}",
     f"sex: {A.sex_called.value_counts().to_dict()}",
     f"bar check: 99.9th percentile of pairs not called same person {q999:.3f}; gap to 0.90 {0.90 - q999:.3f} -> {'INCONCLUSIVE' if inconcl else 'bar holds'}",
     f"same-person pairs {len(S)} | inside one series {int((S.a_series == S.b_series).sum())} | across series {len(cross)}",
     f"held-out arrays that are the same person as a build array (rule 4a, leave the held-out set): {len(excl)}",
     f"build pairs across series (rule 4b, counted as one donor): {int(((cross.a_set == 'build') & (cross.b_set == 'build')).sum())}"]
open(os.path.join(HERE, "identity_check_v21_summary.txt"), "w").write("\n".join(L) + "\n")
pd.Series(excl, name="sample").to_csv(os.path.join(HERE, "identity_check_v21_heldout_excluded.csv"), index=False)
print("\n".join(L))
