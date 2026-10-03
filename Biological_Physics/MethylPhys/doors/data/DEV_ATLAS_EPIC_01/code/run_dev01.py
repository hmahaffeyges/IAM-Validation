#!/usr/bin/env python3
"""DEV-ATLAS-EPIC-01 runner (box). Builds every composition method once, runs all specimens in parallel, writes CSVs to ./out."""
import json, os, sys, time, glob, concurrent.futures as cf
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import comp_methods as CM
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
OUT = "out"; os.makedirs(OUT, exist_ok=True)
ATLAS = "/home/ubuntu/data/dev_atlas_epic_01/IAMAtlas_v2.parquet"
BC = json.load(open("blood_composition_EPIC_v1.json")); NS = pd.Index(BC["neutrophil_sites"])
ROSTER = pd.read_csv("roster.csv"); RS = pd.read_csv("roster_samples.csv"); TWIN = pd.read_csv("10_twin_test_sample_level.csv")
THR = json.load(open("twin_family_thresholds_v1.json"))
S = pd.read_csv("samples.csv")

A = CM.load_atlas(ATLAS); cells = CM.atlas_cells(A); log("atlas", A.shape, len(cells))
plat = ROSTER.set_index("cell")["platforms"].astype(str).to_dict()
WB = [c for c in cells if c not in CM.DROP_WB]
ARR = [c for c in WB if "array" in plat.get(c, "")]
SOL = {}
SOL["ATLAS_a"] = CM.build_deconv(A, WB, NS); log("ATLAS_a", SOL["ATLAS_a"].meta)
SOL["ATLAS_b"] = CM.build_deconv(A, ARR, NS); log("ATLAS_b", SOL["ATLAS_b"].meta)

# ---- (c) merges from the atlas's own twin / cross-source rule, on ATLAS_a's markers, among the blood-group cells
Da = SOL["ATLAS_a"]; bloodcells = [c for c in Da.cells if CM.grp(c) in CM.BLOOD8]
merge, P = CM.twin_merge(A, Da, ROSTER, TWIN, THR, bloodcells)
P.sort_values("r_markers", ascending=False).to_csv(f"{OUT}/twin_pairs_ATLAS_a_markers.csv", index=False)
log("cross-source / twin rule ->", merge)
# step 2: identifiability in a mixture (twin of a non-negative mixture of other cells, r >= twin_r), on the same markers
mix_removed, mrows = CM.mixture_twin(Da, [c for c in bloodcells if c not in merge], THR["twin_r"])
pd.DataFrame(mrows).to_csv(f"{OUT}/mixture_twin_test.csv", index=False); log("mixture-twin rule ->", mix_removed)
WBc = [c for c in WB if c not in merge and c not in mix_removed]
SOL["ATLAS_c"] = CM.build_deconv(A, WBc, NS); log("ATLAS_c", SOL["ATLAS_c"].meta)
BLOODc = [c for c in WBc if CM.grp(c) in CM.BLOOD8]
SOL["ATLAS_c_blood"] = CM.build_deconv(A, BLOODc, NS); log("ATLAS_c_blood", SOL["ATLAS_c_blood"].meta)
# (e) array-measured circulating blood cells only, then the same mixture-identifiability rule on their own markers (development combination of b and c)
ARRb = [c for c in ARR if CM.grp(c) in CM.BLOOD8]
De0 = CM.build_deconv(A, ARRb, NS); mix_removed_e, mrows_e = CM.mixture_twin(De0, ARRb, THR["twin_r"])
pd.DataFrame(mrows_e).to_csv(f"{OUT}/mixture_twin_test_array_blood.csv", index=False); log("mixture-twin rule (array blood) ->", mix_removed_e)
ARRe = [c for c in ARRb if c not in mix_removed_e]
SOL["ATLAS_e"] = CM.build_deconv(A, ARRe, NS); log("ATLAS_e", SOL["ATLAS_e"].meta)
json.dump(dict(cross_source_or_twin_merge=merge, mixture_twin_removed=mix_removed, mixture_twin_removed_array_blood=mix_removed_e, ATLAS_e_cells=ARRe,
               thresholds=THR, n_pairs=len(P), ATLAS_c_cells=WBc, ATLAS_c_blood_cells=BLOODc),
          open(f"{OUT}/merge_rule.json", "w"), indent=1)

# ---- identifiability (Cramer-Rao) of each solver's blood cells at a whole-blood-like composition
wbf = {"neutrophils": 0.60, "monocytes": 0.07, "eosinophils": 0.03, "basophils": 0.01, "nk cells": 0.04}
crows = []
for k, D in SOL.items():
    f = np.array([wbf.get(c, 0.0) for c in D.cells]); rest = [i for i, c in enumerate(D.cells) if c not in wbf and CM.grp(c) in CM.BLOOD8]
    f[rest] = (1 - f.sum()) / max(len(rest), 1); f = f / f.sum()
    sd, R = CM.crlb(D, f)
    for i, c in enumerate(D.cells):
        if CM.grp(c) not in CM.BLOOD8: continue
        j = np.argsort(R[i])[0]
        crows.append(dict(solver=k, cell=c, group=CM.grp(c), n_markers=D.meta["n_markers"], crlb_sd=float(sd[i]), most_anticorrelated=D.cells[j], corr=float(R[i, j])))
pd.DataFrame(crows).to_csv(f"{OUT}/crlb_identifiability.csv", index=False)

# ---- diagnostic: atlas mean vs the Salas purified arrays' own mean, for the blood cells, at ATLAS_a markers and at the neutrophil sites
sal = RS[RS.source.isin(["Salas2018", "Salas2022"]) & (RS.qc == True)].copy(); sal["gse"] = np.where(sal.source == "Salas2018", "GSE110554", "GSE167998")
def rd(r): return CM.read_beta(f"/home/ubuntu/data/atlas_sources/blood/{r.gse}/shards/{r['sample']}.parquet")
SAL = {}
for c, G in sal.groupby("cell"):
    seen = {}
    for _, r in G.iterrows():
        sid = r["sample"].split("_", 1)[1]           # Sentrix id: the same physical array deposited twice counts once
        if sid not in seen: seen[sid] = rd(r)
    SAL[c] = pd.concat(seen.values(), axis=1).mean(1)
drows = []
for c, m in SAL.items():
    if f"{c}_mean" not in A: continue
    for nm, sites in (("ATLAS_a_markers", pd.Index(Da.loci)), ("neutrophil_sites", NS)):
        a = A[f"{c}_mean"].reindex(sites); s = m.reindex(sites); ok = a.notna() & s.notna()
        drows.append(dict(cell=c, sites=nm, n=int(ok.sum()), n_physical_arrays=int(sal[sal.cell == c]["sample"].str.split("_", n=1).str[1].nunique()),
                          mean_atlas_minus_array=float((a[ok] - s[ok]).mean()), mean_abs=float((a[ok] - s[ok]).abs().mean()),
                          p95_abs=float((a[ok] - s[ok]).abs().quantile(0.95))))
pd.DataFrame(drows).to_csv(f"{OUT}/diag_atlas_vs_salas_arrays.csv", index=False); log("diag written")
# ---- NILC
def build_nilc(cellset, Dmark, tag, tech_from=None):
    Ab = CM.subset_atlas(A, cellset); Ab = Ab.loc[Ab.index.difference(NS)]
    mu_b = Ab[[f"{c}_mean" for c in cellset]]; v_b = Ab[[f"{c}_sd" for c in cellset]].values ** 2 + Ab[[f"{c}_donor_sd" for c in cellset]].values ** 2
    ok = mu_b.notna().all(1).values & np.isfinite(v_b).all(1); rng = (mu_b.max(1) - mu_b.min(1)).values
    sites = {"S1_markers": pd.Index(Dmark.loci), "S2_range0.2": mu_b.index[ok & (rng >= 0.2)]}
    log("NILC", tag, len(cellset), {k: len(v) for k, v in sites.items()}); return mu_b, v_b, sites
blood_c = [c for c in WBc if CM.grp(c) in CM.BLOOD8]
mu_b, v_b, SITES = build_nilc(blood_c, SOL["ATLAS_c_blood"], "c")
mu_e, v_e, SITES_E = build_nilc(ARRe, SOL["ATLAS_e"], "e")
# technical noise from GSE250556 POOLED replicates (same DNA pool, same person): residual about the subject's replicate mean
rep = S[(S.set == "REPL") & (S.rep_kind == "pooled")]; reps = {}
for r in rep.itertuples():
    if glob.glob(r.path): reps[r.gsm] = (r.subject, CM.read_beta(r.path))
allsites = SITES["S1_markers"].union(SITES["S2_range0.2"]).union(pd.Index(BC["markers"])).union(SITES_E["S1_markers"]).union(SITES_E["S2_range0.2"])
Rm = pd.DataFrame({g: b.reindex(allsites) for g, (s, b) in reps.items()}); subj = pd.Series({g: s for g, (s, b) in reps.items()})
res = Rm.copy()
for s in subj.unique(): cols = subj.index[subj == s]; res[cols] = Rm[cols].sub(Rm[cols].mean(1), axis=0)
dof = len(subj) - subj.nunique(); tech_var = (res ** 2).sum(1) / dof
tech_var = tech_var.fillna(tech_var.median())
log("tech noise: pooled replicates", len(subj), "subjects", subj.nunique(), "median tech SD", float(np.sqrt(tech_var.median())))
pd.DataFrame({"tech_sd": np.sqrt(tech_var)}).describe().to_csv(f"{OUT}/tech_noise_summary.csv")
Xs1 = res.loc[SITES["S1_markers"]].fillna(0).T.values * np.sqrt(len(subj) / dof)
LW, shrink = CM.ledoit_wolf(Xs1); log("Ledoit-Wolf shrinkage", shrink)
NIL = {}
for tag, (cellset, mu_x, v_x, sites_x) in {"": (blood_c, mu_b, v_b, SITES), "e_": (ARRe, mu_e, v_e, SITES_E)}.items():
    for sn, st in sites_x.items():
        ii = mu_x.index.get_indexer(st); mm = mu_x.values[ii]; vv = v_x[ii]
        covs = (["atlas", "atlas+tech", "atlas+techLW"] if (sn == "S1_markers" and tag == "") else ["atlas", "atlas+tech"])
        for cov in covs:
            for off in (False, True):
                NIL[f"NILC_{tag}{sn}_{cov}_{'D1' if off else 'D0'}"] = CM.NILC(st, cellset, mm, vv, cov=cov, tech_var=tech_var.reindex(st).values,
                                                                          tech_cov=(LW if cov == "atlas+techLW" else None), offset=off)
# NILC on the chain's 8 EPIC group templates (blood_composition_EPIC_v1.json markers; no template variance in the file)
mk = pd.Index(BC["markers"]); M8 = pd.DataFrame(BC["mu_markers"], index=mk)[BC["groups"]]
for cov in ("atlas", "atlas+tech"):
    for off in (False, True):
        NIL[f"NILC_EPIC8_{cov}_{'D1' if off else 'D0'}"] = CM.NILC(mk, BC["groups"], M8.values, np.zeros(M8.shape), cov=cov, tech_var=tech_var.reindex(mk).values, offset=off)
N8 = CM.NNLS8(BC)
# Stage M profiles at the neutrophil sites
P8 = {g: pd.Series(v, index=NS, dtype="float64") for g, v in BC["profiles_at_neutrophil_sites"].items()}
PA = {c: A[f"{c}_mean"].reindex(NS) for c in cells}
del A
log("methods built:", list(SOL) + list(NIL) + ["NNLS8"])

def run(r):
    if not glob.glob(r["path"]): return r["gsm"], None
    b = CM.read_beta(r["path"]); out = {}
    o = N8.deconvolve(b); out["NNLS8"] = dict(fr=o["fractions"], res=o["residual_mae"], n=o["n_markers_used"], prof="EPIC8")
    for k, D in SOL.items():
        o = D.deconvolve(b, n_boot=0); out[k] = dict(fr=o["fractions"], res=o["residual_mae"], n=o["n_markers_used"], prof="ATLAS")
    for k, N in NIL.items():
        o = N.deconvolve(b); out[k] = dict(fr=o["fractions"], res=o["residual_mae"], n=o["n_sites_used"], sum=o["sum"], off=o["offset"],
                                         nsd=o["noise_sd"], prof=("EPIC8" if "EPIC8" in k else "ATLAS"))
    # Stage M: both profile sets for every composition
    for k, d in out.items():
        frpos = {c: max(v, 0.0) for c, v in d["fr"].items()}
        g = CM.to_groups(frpos); g8 = {x: g.get(x, 0.0) for x in CM.BLOOD8}
        d["A_EPIC8"], d["n_ns"], d["mapped8"] = CM.met_a(b, g8, P8, NS)
        if d["prof"] == "ATLAS": d["A_self"], _, _ = CM.met_a(b, frpos, PA, NS)
        else: d["A_self"] = d["A_EPIC8"]
    return r["gsm"], out

if __name__ == "__main__":
    recs = S.to_dict("records")
    with cf.ProcessPoolExecutor(min(120, len(recs)), mp_context=__import__("multiprocessing").get_context("fork")) as ex:
        res_all = dict(ex.map(run, recs, chunksize=1))
    log("ran", sum(v is not None for v in res_all.values()), "of", len(recs))
    fl, gl, ml = [], [], []
    for r in recs:
        o = res_all.get(r["gsm"])
        if o is None: continue
        for k, d in o.items():
            for c, v in d["fr"].items(): fl.append(dict(gsm=r["gsm"], set=r["set"], method=k, cell=c, f=v, noise_sd=(d.get("nsd") or {}).get(c)))
            for gname, v in CM.to_groups(d["fr"]).items(): gl.append(dict(gsm=r["gsm"], set=r["set"], method=k, group=gname, est=v))
            ml.append(dict(gsm=r["gsm"], set=r["set"], subject=r.get("subject"), rep_kind=r.get("rep_kind"), method=k, residual_mae=d["res"], n_sites=d["n"],
                           sum_f=d.get("sum", sum(d["fr"].values())), offset=d.get("off"), A_EPIC8=d["A_EPIC8"], A_self=d["A_self"], n_ns=d["n_ns"], mapped8=d["mapped8"]))
    pd.DataFrame(fl).to_csv(f"{OUT}/fractions_cells_long.csv", index=False)
    pd.DataFrame(gl).to_csv(f"{OUT}/fractions_groups_long.csv", index=False)
    pd.DataFrame(ml).to_csv(f"{OUT}/per_sample_method.csv", index=False)
    json.dump({k: D.meta for k, D in SOL.items()} | {"NILC_cells": blood_c, "NILC_e_cells": ARRe, "NILC_sites": {k: len(v) for k, v in SITES.items()}, "NILC_e_sites": {k: len(v) for k, v in SITES_E.items()},
               "LW_shrinkage": shrink, "tech_pooled_arrays": int(len(subj)), "WB_cells": WB, "ARR_cells": ARR, "WBc_cells": WBc},
              open(f"{OUT}/solver_meta.json", "w"), indent=1)
    log("done")
