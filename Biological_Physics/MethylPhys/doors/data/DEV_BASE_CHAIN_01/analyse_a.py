#!/usr/bin/env python3
"""DEV-BASE-CHAIN-01 outcome: checks (a)-(e) from readings_all.csv (box driver), the frozen manifest and e7.json. Rules as written in the note."""
import json, sys, numpy as np, pandas as pd
RA, MANP, E7, OUT = sys.argv[1:5]
X = pd.read_csv(RA); M0 = pd.read_csv(MANP, dtype={"slide": str})
M = M0.assign(_sup=M0.series.eq("GSE181034")).sort_values("_sup").drop_duplicates("gsm").drop(columns="_sup")   # GSE181034 is a SuperSeries re-listing GSE167998/GSE182379 GSMs
X = X.merge(M[["gsm", "specimen", "healthy", "plat", "person", "slide", "no_intake"]], on="gsm", how="left")
R = {}
p1 = X[X.arm == "pass1"]
R["a"] = dict(n_manifest_rows=len(M0), n_manifest=len(M), n_pass1=len(p1), by_class=p1.cls.value_counts().to_dict(),
              crashes_all_arms=X[X.cls == "crash"][["arm", "series", "gsm"]].values.tolist(),
              environment=X[X.cls == "environment"][["arm", "series", "gsm"]].values.tolist(),
              stage0_reasons=p1.stage0_reason.value_counts().to_dict(),
              refusals_ok_reports=p1[p1.refusal.notna()].groupby("plat").size().to_dict() if "refusal" in p1 else {})
R["a"]["pass"] = (X.cls == "crash").sum() == 0 and len(p1) == len(M)
fin = {}
for arm1, arm2 in (("pass1", "pass2"), ("diag1", "diag2")):
    a = X[X.arm == arm1].set_index("gsm"); b = X[X.arm == arm2].set_index("gsm")
    f = a.copy()
    for col in ("A_rel", "tare_state", "n_refs", "ref_median", "ref_sd", "det_limit", "state"):
        f.loc[b.index.intersection(f.index), col] = b[col]
    fin[arm1] = f.reset_index()
F = fin["pass1"]; FD = fin["diag1"]
inN = lambda s: ((s >= 0.95) & (s <= 1.05))
INFLOOR = ("GSE110554", "GSE167998", "GSE181034")
setb = M[(M.specimen == "isolated neutrophils") & (M.healthy == True)]
def bsum(df, ids):
    d = df[df.gsm.isin(ids) & (df.cls == "ok")]
    t = d.A_rel.dropna()
    return dict(n_set=len(ids), reached_stage5=int(d.A.notna().sum()), tared=int(len(t)), tared_in_normal=int(inN(t).sum()),
                tared_range=[round(t.min(), 4), round(t.max(), 4)] if len(t) else None,
                untared_own_floor_states=d.state_own_floor.value_counts().to_dict(), untared_A_range=[round(d.A.min(), 4), round(d.A.max(), 4)] if d.A.notna().any() else None,
                per_series={s: dict(n=int(len(g)), tared=int(g.A_rel.notna().sum()), in_normal=int(inN(g.A_rel.dropna()).sum()),
                                    A_rel=[round(g.A_rel.min(), 3), round(g.A_rel.max(), 3)] if g.A_rel.notna().any() else None) for s, g in d.groupby("series")})
ids_in = setb[setb.series.isin(INFLOOR)].gsm; ids_out = setb[~setb.series.isin(INFLOOR)].gsm
R["b"] = dict(in_floor=bsum(F, ids_in), other_labs=bsum(F, ids_out),
              lost_at_stage0=p1[p1.gsm.isin(setb.gsm) & (p1.cls != "ok")][["series", "gsm", "stage0_reason"]].values.tolist(),
              diag_arm_in_floor=bsum(FD, ids_in), diag_arm_other_labs=bsum(FD, ids_out))
rb = F[F.gsm.isin(setb.gsm) & F.A.notna()]
R["b"]["pass"] = bool(len(rb) and rb.A_rel.notna().all() and inN(rb.A_rel).all())
R["b"]["note"] = "bar: every array of the set reaching Stage 5 reads Normal on its tared A_rel (pass-1/pass-2 arm only)"
g = F[(F.series == "GSE250556")]
gt = g[g.A_rel.notna()]
ss = sum(((d.A_rel - d.A_rel.mean()) ** 2).sum() for _, d in gt.groupby("person")); dof = len(gt) - gt.person.nunique()
R["c"] = dict(n=len(g), read=int(g.A.notna().sum()), tared=len(gt), within_person_sd=round(float(np.sqrt(ss / dof)), 4) if dof > 0 else None,
              per_person_sd={p: round(float(d.A_rel.std()), 4) for p, d in gt.groupby("person")}, sd_all=round(float(gt.A_rel.std()), 4),
              in_normal=int(inN(gt.A_rel).sum()), below=int((gt.A_rel < 0.95).sum()), above=int((gt.A_rel > 1.05).sum()),
              A_rel_range=[round(gt.A_rel.min(), 4), round(gt.A_rel.max(), 4)],
              withheld_by_gate_final=int(g.state.astype(str).str.startswith("withheld").sum()), N_above_Nmax=int((g.N > 0.149).sum()),
              N_range=[round(g.N.min(), 4), round(g.N.max(), 4)], not_read=g[g.A.isna()][["gsm", "cls", "stage0_reason"]].values.tolist())
R["c"]["targets"] = dict(within_person_sd_le_0_020=bool(R["c"]["within_person_sd"] is not None and R["c"]["within_person_sd"] <= 0.020),
                         in_normal_ge_95pct=bool(len(gt) and inN(gt.A_rel).mean() >= 0.95))
okall = X[X.cls == "ok"]
R["d"] = dict(n_reports=len(okall), by_arm=okall.arm.value_counts().to_dict(), sections_ok=int(okall.sections_ok.astype(str).eq("True").sum()),
              missing=okall[~okall.sections_ok.astype(str).eq("True")][["arm", "gsm", "sections_missing"]].head(20).values.tolist(),
              safeguards={c: okall[c].value_counts().to_dict() for c in okall.columns if c.startswith("sg_")},
              chain_commits=okall.chain_commit.value_counts().to_dict(), chain_dirty=okall.chain_dirty.value_counts().to_dict())
R["d"]["pass"] = R["d"]["sections_ok"] == len(okall)
R["e"] = json.load(open(E7))
# extra development observations (not bars)
wb = F[(F.cls == "ok") & (F.specimen == "whole blood")]
R["obs"] = dict(whole_blood_read=int(wb.A.notna().sum()), whole_blood_withheld_fraction=int(wb.state.astype(str).str.contains("fraction").sum()),
                whole_blood_tared=int(wb.A_rel.notna().sum()), whole_blood_tared_in_normal=int(inN(wb.A_rel.dropna()).sum()),
                untared_gate_withheld_all=int(F.state.astype(str).str.startswith("withheld").sum()),
                det_limit_median=float(F.det_limit.median()) if F.det_limit.notna().any() else None,
                C_range=[float(F.C.min()), float(F.C.max())], seconds_median=float(p1[p1.cls == "ok"].seconds.median()))
json.dump(R, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in R.items()}, default=str))
