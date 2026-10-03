#!/usr/bin/env python3
"""DEV-ATLAS-EPIC-01 analysis: metrics vs truth, repeatability, method agreement, Stage M on healthy whole bloods, NILC selection on MIX18."""
import sys, json, numpy as np, pandas as pd
IN = sys.argv[1]; OUT = sys.argv[2]
S = pd.read_csv("stage/samples.csv")
G = pd.read_csv(f"{IN}/fractions_groups_long.csv"); PM = pd.read_csv(f"{IN}/per_sample_method.csv")
W = G.pivot_table(index=["gsm", "set", "method"], columns="group", values="est", aggfunc="first").reset_index()
TRUTH_SETS = ["MIX18", "MIX22", "MIX12", "FACS", "LONG", "MIX18_blood"]
_bl = set(S[(S.set == "MIX18") & (S.NEU >= 0.5)].gsm)   # the six blood-like Salas 2018 mixtures (neutrophils 63-75 %)
W = pd.concat([W, W[W.gsm.isin(_bl)].assign(set="MIX18_blood")], ignore_index=True)
GROUPS = ["NEU", "EOS", "BASO", "GRAN", "MONO", "B", "NK", "BNK", "CD4T", "CD8T", "CD4nv", "CD4mem", "Treg", "CD8nv", "CD8mem", "Bnv", "Bmem"]
T = S.set_index("gsm")
rows = []
for (m, st), X in W[W.set.isin(TRUTH_SETS)].groupby(["method", "set"]):
    for g in GROUPS:
        if g not in X or g not in T: continue
        t = T.loc[X.gsm, g].values.astype(float); e = X[g].values.astype(float); ok = np.isfinite(t) & np.isfinite(e)
        if ok.sum() < 3: continue
        d = e[ok] - t[ok]
        rows.append(dict(method=m, set=st, group=g, n=int(ok.sum()), bias=d.mean(), rmse=np.sqrt((d ** 2).mean()), mae=np.abs(d).mean(),
                         r=(np.corrcoef(e[ok], t[ok])[0, 1] if np.std(t[ok]) > 0 and np.std(e[ok]) > 0 else np.nan), truth_mean=t[ok].mean()))
MET = pd.DataFrame(rows); MET.to_csv(f"{OUT}/metrics_by_set.csv", index=False)
# NILC configuration chosen on MIX18 only (rule fixed before reading other sets): lowest mean RMSE over the six MIX18 groups, offset D0
six = ["NEU", "MONO", "B", "NK", "CD4T", "CD8T"]
sel = MET[(MET.set == "MIX18") & MET.group.isin(six) & MET.method.str.startswith("NILC_S")].groupby("method").rmse.mean().sort_values()
sel.to_csv(f"{OUT}/nilc_selection_on_MIX18.csv")
best_d0 = [m for m in sel.index if m.endswith("_D0")][0]; best_d1 = best_d0[:-2] + "D1"
sel8 = MET[(MET.set == "MIX18") & MET.group.isin(six) & MET.method.str.startswith("NILC_EPIC8")].groupby("method").rmse.mean().sort_values()
best8 = [m for m in sel8.index if m.endswith("_D0")][0]
sele = MET[(MET.set == "MIX18") & MET.group.isin(six) & MET.method.str.startswith("NILC_e_S")].groupby("method").rmse.mean().sort_values()
best_e = [m for m in sele.index if m.endswith("_D0")][0] if len(sele) else None
sele.to_csv(f"{OUT}/nilc_e_selection_on_MIX18.csv")
json.dump(dict(NILC_e_chosen=best_e, NILC_e_offset_twin=(best_e[:-2] + "D1" if best_e else None), NILC_atlas_chosen=best_d0, NILC_atlas_offset_twin=best_d1, NILC_EPIC8_chosen=best8, rule="lowest mean RMSE over NEU,MONO,B,NK,CD4T,CD8T on MIX18 (GSE110554) only"),
          open(f"{OUT}/nilc_choice.json", "w"), indent=1)
# repeatability GSE250556: within-subject SD of each group fraction, per method, pooled and unpooled separately
R = W[W.set == "REPL"].merge(S[["gsm", "subject", "rep_kind"]], on="gsm")
rr = []
for (m, k), X in R.groupby(["method", "rep_kind"]):
    for g in ["NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T", "NONBLOOD", "OTHER_IMMUNE"]:
        if g not in X: continue
        sd = X.groupby("subject")[g].std(ddof=1); rr.append(dict(method=m, rep_kind=k, group=g, within_subject_sd=np.sqrt((sd ** 2).mean()),
                                                                 between_subject_sd=X.groupby("subject")[g].mean().std(ddof=1), mean=X[g].mean()))
REP = pd.DataFrame(rr); REP.to_csv(f"{OUT}/repeatability_GSE250556.csv", index=False)
# agreement between methods on all whole bloods (FACS, LONG, REPL): mean difference and SD of difference vs NNLS8 and vs ATLAS_c
WB = W[W.set.isin(["FACS", "LONG", "REPL"])]
ag = []
for ref in ["NNLS8", "ATLAS_c"]:
    Rf = WB[WB.method == ref].set_index("gsm")
    for m, X in WB.groupby("method"):
        X = X.set_index("gsm")
        for g in ["NEU", "MONO", "B", "NK", "CD4T", "CD8T", "EOS", "BASO"]:
            if g not in X or g not in Rf: continue
            d = (X[g] - Rf[g].reindex(X.index)).dropna()
            ag.append(dict(reference=ref, method=m, group=g, n=len(d), mean_diff=d.mean(), sd_diff=d.std(ddof=1)))
pd.DataFrame(ag).to_csv(f"{OUT}/agreement_whole_blood.csv", index=False)
# Stage M on healthy whole bloods: untared A and median same-run tare (references: the other arrays of the same series, >= 3)
PM = PM.merge(S[["gsm", "gse"]], on="gsm")
ms = []
for (m, st), X in PM[PM.set.isin(["FACS", "LONG", "REPL"])].groupby(["method", "set"]):
    for col in ["A_EPIC8", "A_self"]:
        a = X.set_index("gsm")[col].astype(float).dropna()
        if len(a) < 4: continue
        tared = pd.Series({g: a[g] / np.median(a.drop(g)) for g in a.index})
        ms.append(dict(method=m, set=st, profiles=("EPIC8" if col == "A_EPIC8" else "same reference as composition"), n=len(a),
                       untared_median=a.median(), untared_sd=a.std(ddof=1), untared_min=a.min(), untared_max=a.max(),
                       tared_median=tared.median(), tared_sd=tared.std(ddof=1), tared_min=tared.min(), tared_max=tared.max(),
                       tared_in_normal=f"{int(((tared >= 0.95) & (tared <= 1.05)).sum())}/{len(tared)}",
                       untared_in_normal=f"{int(((a >= 0.95) & (a <= 1.05)).sum())}/{len(a)}"))
pd.DataFrame(ms).to_csv(f"{OUT}/stageM_healthy_whole_blood.csv", index=False)
W.to_csv(f"{OUT}/fractions_groups_wide.csv", index=False)
print(json.load(open(f"{OUT}/nilc_choice.json")))
