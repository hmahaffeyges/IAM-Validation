#!/usr/bin/env python3
"""DEVELOPMENT - not commissioned. Chain v3 development round 2 (2026-10-04), box analyses on Stage 1 beta vectors already on the box
(DEV-BASE-CHAIN-01 --save-betas). Every rule and bar is the one written in the doors/ notes before this ran.
  B1 type II fixed sites, design map, reference anchors -> Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json   (DEV-SELFTARE-02)
  B2 self-tare II readings (replicates, other-lab purified neutrophils, floor arrays)                                    (DEV-SELFTARE-02)
  B3 3b trace cell, 3c foreign cell (+ dev_foreign_placenta_EPIC_v1.json), 11b brightness                               (DEV-TOOLKIT-ADDED-02)
  B4 directional decomposition                                                                                           (DEV-DIRECTION-02)
  B5 sky against the block-shuffle null; look-elsewhere by simulation                                                   (DEV-SKY-02)
  B6 new-cell rule on monocytes and B cells                                                                              (DEV-NEWCELL-01)
  B7 GSE77797 composition truth (450K)                                                                                   (DEV-COMPOSITION-TRUTH-02)
  B8 development flags leave the reading unchanged                                                                       (DEV-FLAGS-01)
Outputs: ./r2/*.csv, ./r2/summary.json, ./r2/Development/*.json"""
import os, sys, json, glob, time, traceback, re, subprocess, shutil, gzip
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
W = os.getcwd(); MP = f"{W}/repo/Biological_Physics/MethylPhys"; CH = f"{MP}/chain"; OUT = f"{W}/r2"; os.makedirs(f"{OUT}/Development", exist_ok=True)
BET1 = "/home/ubuntu/data/base_chain_01/betas"; ATLAS = "/home/ubuntu/data/dev_atlas_epic_01/IAMAtlas_v2.parquet"; WORK = "/home/ubuntu/data/round2b"
os.makedirs(WORK, exist_ok=True); sys.path[:0] = [CH]
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
import conductor_v3 as C3, dev_stages as DV, stage_m_met_a as SM
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
MAN = pd.read_csv(f"{W}/manifest.csv", dtype={"slide": str}).drop_duplicates("gsm"); R1 = pd.read_csv(f"{W}/readings_all.csv")
REPL = pd.read_csv(f"{W}/repl.csv", dtype={"slide": str}); S = {}
REPL = REPL[[os.path.exists(f"{BET1}/{g}.parquet") for g in REPL.gsm]].reset_index(drop=True)
APP = f"{WORK}/atlas_parents.parquet"
BC = C3._bc(); NSITES = pd.Index(BC["neutrophil_sites"]); MARK = pd.Index(BC["markers"]); NOISE = pd.Index(C3.noise_sites()["sites"])
REFS6 = [r.split("_")[0] for r in SM._floors()["platforms"]["EPIC"]["neutrophils"]["refs"]]
H = DV._H; NORMAL = (0.95, 1.05); inN = lambda a: (a >= NORMAL[0]) & (a <= NORMAL[1])
STEPS = os.environ.get("STEPS", "B1,B2,B3,B4,B5,B6,B7,B8").split(",")


def beta(g):
    for p in (f"/home/ubuntu/data/round2/betas/{g}.parquet", f"{BET1}/{g}.parquet"):
        if os.path.exists(p):
            b = pd.read_parquet(p).iloc[:, 0].astype("float64"); b.index = b.index.astype(str); return b
    return None


def wsd(df, col, by="person"):
    d = df.dropna(subset=[col]); ss = sum(((g[col] - g[col].mean()) ** 2).sum() for _, g in d.groupby(by)); dof = len(d) - d[by].nunique()
    return float(np.sqrt(ss / dof)) if dof > 0 else None


def tare(df, col, scope):
    """median tare: col / median(col of the other rows sharing `scope` (list of columns, tried in order)); >= 3 required."""
    out = []
    for r in df.itertuples():
        v = getattr(r, col); val = np.nan
        if pd.notna(v):
            for sc in scope:
                pool = df[(df.gsm != r.gsm) & (df[sc] == getattr(r, sc))][col].dropna()
                if len(pool) >= 3: val = v / float(np.median(pool)); break
        out.append(val)
    return out


def tare_diff(df, col, scope, minref=2):
    out, spr = [], []
    for r in df.itertuples():
        v = getattr(r, col); val = sp = np.nan
        if pd.notna(v):
            for sc in scope:
                pool = df[(df.gsm != r.gsm) & (df[sc] == getattr(r, sc))][col].dropna()
                if len(pool) >= minref: val = v - float(np.median(pool)); sp = float(np.std(pool, ddof=1)); break
        out.append(val); spr.append(sp)
    return out, spr


def group_of(spec, title):
    """Purified group from the series record. T cells by title: GSE110554 Th = CD4, Tc = CD8; GSE167998 CD4nv/CD4mem/Treg = CD4,
    Tn/Tem = CD8 (naive / effector memory CD8 of the extended library); titles that name neither (PCA1547, ST2007) are left out."""
    t = str(title).upper().strip()
    g = {"isolated neutrophils": "NEU", "sorted eosinophils": "EOS", "sorted basophils": "BASO", "sorted monocytes": "MONO", "sorted B cells": "B",
         "sorted NK cells": "NK"}.get(spec)
    if g or spec != "sorted T cells": return g
    if t.startswith("TH") or "CD4" in t or t.startswith("TREG"): return "CD4T"
    if t.startswith("TC") or "CD8" in t or t.startswith("TEM") or t.startswith("TN"): return "CD8T"
    return None


def step(name, fn):
    if name not in STEPS: return
    try: fn(); log(name, "done")
    except Exception as e: S[name] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}; log(name, "ERROR", e)
    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, default=str)


# ------------------------------------------------------------------ B1
def B1():
    from stage_0_1_qc_handoff import _manifest
    man = _manifest("EPIC_v1").data_frame; des = man["Infinium_Design_Type"].astype(str).str.strip(); des.index = des.index.astype(str)
    des = des[~des.index.duplicated()]
    P = MAN[MAN.series == "GSE110554"].copy(); P["grp"] = [group_of(s, t) for s, t in zip(P.specimen, P.title)]; P = P[P.grp.notna()]
    Bm = {g: beta(g) for g in P.gsm}; Bm = {k: v for k, v in Bm.items() if v is not None}
    common = None
    for x in Bm.values(): common = x.index if common is None else common.intersection(x.index)
    X = pd.DataFrame({g: Bm[g].reindex(common) for g in Bm}); grp = P.set_index("gsm").grp.reindex(X.columns)
    mu = pd.DataFrame({k: X.loc[:, grp == k].mean(1) for k in sorted(grp.unique())}); sd = pd.DataFrame({k: X.loc[:, grp == k].std(1) for k in sorted(grp.unique())})
    cand = mu.index.difference(NSITES).difference(MARK); d2 = des.reindex(cand)
    ok = (sd.loc[cand].max(1) <= 0.02) & ((mu.loc[cand].max(1) - mu.loc[cand].min(1)) <= 0.03)
    low2 = cand[(d2 == "II").values & ok.values & (mu.loc[cand].max(1) <= 0.15).values]; high2 = cand[(d2 == "II").values & ok.values & (mu.loc[cand].min(1) >= 0.85).values]
    nmu = mu.reindex(NOISE).mean(1); dn = des.reindex(NOISE)
    sets = {"I_low": list(NOISE[(dn == "I").values & (nmu < 0.5).values]), "I_high": list(NOISE[(dn == "I").values & (nmu >= 0.5).values]),
            "II_low": sorted(set(low2) | set(NOISE[(dn == "II").values & (nmu < 0.5).values])), "II_high": sorted(set(high2) | set(NOISE[(dn == "II").values & (nmu >= 0.5).values]))}
    R6 = {g: beta(g) for g in REFS6}; allfix = sorted(set(sum(sets.values(), [])))
    rv = pd.concat([R6[g].reindex(allfix) for g in R6], axis=1).mean(1)
    st0 = {"sets": sets, "ref_value": {k: round(float(v), 6) for k, v in rv.dropna().items()}}
    ra = {k: float(np.mean([DV.anchors(R6[g], {"sets": sets})[k]["mean"] for g in R6])) for k in sets}
    design = {s: des.get(s, "II") for s in sorted(set(NSITES) | set(MARK))}
    st = {"version": "dev_selftare_typeII_EPIC_v1", "label": DV.DEV_LABEL, "date": "2026-10-04",
          "rule": "type II fixed sites: EPIC type II, not identity site, not composition marker; every GSE110554 purified group mean <= 0.15 (low) or >= 0.85 (high), group SD <= 0.02, spread of group means <= 0.03; type I sets = noise_sites_EPIC_v1 by design and state",
          "n": {k: len(v) for k, v in sets.items()}, "n_typeII_new": {"low": int(len(low2)), "high": int(len(high2))},
          "ref_arrays": REFS6, "ref_anchors": ra, **st0, "design": design, "source_arrays": sorted(X.columns), "groups": sorted(grp.unique())}
    json.dump(st, open(f"{OUT}/Development/dev_selftare_typeII_EPIC_v1.json", "w")); json.dump(st, open(f"{CH}/Runtime Matrices/Development/dev_selftare_typeII_EPIC_v1.json", "w"))
    DV._C.pop("dev_selftare_typeII_EPIC_v1.json", None)
    S["B1"] = {"n": st["n"], "n_typeII_new": st["n_typeII_new"], "ref_anchors": ra, "groups": st["groups"], "n_source_arrays": len(X.columns)}
    # 8-group purified profiles (GSE110554 + GSE167998, our Stage 1) for 3c
    P2 = MAN[MAN.series.isin(["GSE110554", "GSE167998"])].copy(); P2["grp"] = [group_of(s, t) for s, t in zip(P2.specimen, P2.title)]; P2 = P2[P2.grp.notna()]
    G8 = {}
    for k, g in P2.groupby("grp"):
        bs = [beta(x) for x in g.gsm]; bs = [b for b in bs if b is not None]
        G8[k] = pd.concat(bs, axis=1).mean(1)
    pd.DataFrame(G8).to_parquet(f"{WORK}/blood8_profiles.parquet"); S["B1"]["blood8_n"] = P2.groupby("grp").size().to_dict()


# ------------------------------------------------------------------ B2
def _read_all(gsm, specimen):
    b = beta(gsm)
    if b is None: return {"gsm": gsm, "missing": True}
    o = C3.run_neutrophil(b, specimen=specimen, array_type="EPIC_v1"); m = o.get("met_a") or {}
    st = DV.selftare_ii(b, specimen)
    return {"gsm": gsm, "A": m.get("A"), "N": m.get("noise_index"), "f_neu": m.get("fraction"), "A_st": st.get("A_selftared"),
            "slope_II": ((st.get("maps") or {}).get("II") or {}).get("slope"), "slope_I": ((st.get("maps") or {}).get("I") or {}).get("slope"),
            "L_II": ((st.get("maps") or {}).get("II") or {}).get("L"), "U_II": ((st.get("maps") or {}).get("II") or {}).get("U")}


def B2():
    rep = REPL.copy(); rep["specimen"] = "whole blood"
    oth = MAN[MAN.series.isin(["GSE247193", "GSE247195", "GSE122244"]) & (MAN.specimen == "isolated neutrophils") & MAN.healthy].copy()
    fl = pd.DataFrame({"gsm": REFS6, "specimen": "isolated neutrophils", "series": "GSE110554"})
    jobs = [(g, "whole blood") for g in rep.gsm] + [(g, "isolated neutrophils") for g in oth.gsm] + [(g, "isolated neutrophils") for g in REFS6]
    with ProcessPoolExecutor(48) as ex: res = list(ex.map(_read_all, *zip(*jobs)))
    Rr = pd.DataFrame(res).drop_duplicates("gsm").set_index("gsm")
    rep = rep.join(Rr, on="gsm"); rep["series"] = "GSE250556"
    for c in ("A", "A_st"): rep[f"{c}_tared"] = tare(rep, c, ["slide", "series"])
    oth = oth[["series", "gsm", "slide"]].join(Rr, on="gsm")
    for c in ("A", "A_st"): oth[f"{c}_tared"] = tare(oth, c, ["series"])
    fl = fl.join(Rr, on="gsm")
    # floor arrays read against the other five (self-tare reference anchors without the array): recorded as A_st_loo
    st = DV._json("dev_selftare_typeII_EPIC_v1.json")
    loo = []
    for g in REFS6:
        others = [x for x in REFS6 if x != g]; ra = {k: float(np.mean([DV.anchors(beta(x), st)[k]["mean"] for x in others])) for k in st["sets"]}
        b2, _ = DV.selftare_map(beta(g), dict(st, ref_anchors=ra)); m, _z = C3.stage_m_isolated(b2); loo.append(m.get("A"))
    fl["A_st_loo"] = loo
    rep.to_csv(f"{OUT}/selftare02_replicates.csv", index=False); oth.to_csv(f"{OUT}/selftare02_otherlab.csv", index=False); fl.to_csv(f"{OUT}/selftare02_floor.csv", index=False)
    summ = {}
    for nm, col in (("no tare", "A"), ("median tare", "A_tared"), ("self-tare II", "A_st"), ("self-tare II then median tare", "A_st_tared")):
        d = rep[col].dropna()
        summ[nm] = {"replicates_within_person_sd": wsd(rep, col), "replicates_in_normal": int(inN(d).sum()), "replicates_n": int(len(d)),
                    "replicates_median": float(d.median()) if len(d) else None, "otherlab_in_normal": int(inN(oth[col].dropna()).sum()), "otherlab_n": int(oth[col].notna().sum()),
                    "otherlab_range": [float(oth[col].min()), float(oth[col].max())] if oth[col].notna().any() else None,
                    "otherlab_by_series": {s: [int(inN(g[col].dropna()).sum()), int(g[col].notna().sum())] for s, g in oth.groupby("series")},
                    "r_with_N_replicates": float(rep[[col, "N"]].dropna().corr().iloc[0, 1]) if rep[[col, "N"]].dropna().shape[0] > 3 else None}
    summ["floor_arrays"] = {"A": fl.A.round(4).tolist(), "A_st": fl.A_st.round(4).tolist(), "A_st_loo": [round(x, 4) if x else None for x in loo],
                            "in_normal_A_st": int(inN(fl.A_st.dropna()).sum()), "in_normal_A_st_loo": int(inN(pd.Series(loo, dtype=float).dropna()).sum())}
    summ["bars"] = {"replicates_sd": 0.020, "replicates_normal_frac": 0.95, "otherlab": "every array in Normal", "floor": "every array in Normal"}
    S["B2"] = summ


# ------------------------------------------------------------------ B3
def _trace_job(spec_gsm, spike_gsm, cell, f):
    a, b = beta(spec_gsm), beta(spike_gsm)
    idx = a.index.intersection(b.index); m = (1 - f) * a.reindex(idx) + f * b.reindex(idx)
    r = DV.trace_cell(m); c = r["candidates"]; tgt = {"monocytes": "MONO", "B": "B", "T": "T"}[cell]
    hit = (tgt in r["called"]) if tgt != "T" else bool({"CD4T", "CD8T"} & set(r["called"]))
    return dict(specimen=spec_gsm, spike=spike_gsm, cell=cell, f=f, called=";".join(r["called"]), spiked_called=hit, any_called=bool(r["called"]),
                **{f"f_{k}": v["f"] for k, v in c.items()}, **{f"z_{k}": v["z"] for k, v in c.items()})


def _foreign_job(spec_gsm, spike_gsm, f):
    a, b = beta(spec_gsm), beta(spike_gsm); idx = a.index.intersection(b.index); m = (1 - f) * a.reindex(idx) + f * b.reindex(idx)
    r = DV.foreign_cell(m); return dict(specimen=spec_gsm, spike=spike_gsm, f=f, f_hat=r.get("f"), se=r.get("se"), called=r.get("called"), status=r.get("status"))


def _bright_job(gsm):
    b = beta(gsm); comp = C3.stage_a_composition(b); r = DV.brightness(b, "whole blood", comp)
    return dict(gsm=gsm, A=r.get("A"), lo=(r.get("interval_95") or [None, None])[0], hi=(r.get("interval_95") or [None, None])[1], hw=r.get("half_width"),
                **{f"s_{k}": v for k, v in (r.get("per_site_sd") or {}).items()})


def B3():
    # 3b
    specs = MAN[(MAN.series == "GSE247195") & (MAN.specimen == "isolated neutrophils")].gsm.tolist()
    sp = MAN[(MAN.series == "GSE122244") & MAN.healthy]; spikes = {"monocytes": sp[sp.specimen == "sorted monocytes"].gsm.tolist(), "B": sp[sp.specimen == "sorted B cells"].gsm.tolist(),
                                                              "T": sp[sp.specimen == "sorted T cells"].gsm.tolist()}
    pur = {}
    for cell, gs in spikes.items():
        for g in gs:
            fr = C3.stage_a_composition(beta(g)).get("fractions") or {}; pur[g] = {k: round(v, 3) for k, v in fr.items()}
    jobs = [(s, g, cell, f) for s in specs if beta(s) is not None for cell, gs in spikes.items() for g in gs for f in (0.0, 0.01, 0.02, 0.05, 0.10)]
    with ProcessPoolExecutor(64) as ex: T3 = pd.DataFrame(list(ex.map(_trace_job, *zip(*jobs))))
    T3.to_csv(f"{OUT}/trace3b.csv", index=False); json.dump(pur, open(f"{OUT}/trace3b_spike_purity_nnls8.json", "w"))
    t = T3.groupby("f").agg(spiked_called=("spiked_called", "mean"), any_called=("any_called", "mean"), n=("f", "size")).reset_index()
    at5 = T3[T3.f == 0.05]; called5 = at5[at5.any_called]
    S["B3_trace"] = {"by_f": t.to_dict("records"), "at5_spiked_called_frac": float(at5.spiked_called.mean()),
                     "at5_calls_naming_spiked_frac": float(called5.spiked_called.mean()) if len(called5) else None,
                     "by_cell_at5": T3[T3.f == 0.05].groupby("cell").spiked_called.mean().to_dict(), "bars": {"at5_spiked_called": 0.95, "at5_calls_naming_spiked": 0.95}}
    # 3c template
    pl = sorted(MAN[MAN.specimen == "placenta"].gsm.tolist()); half = len(pl) // 2; tg, sg = pl[:half], pl[half:]
    tb = [beta(g) for g in tg]; tb = [b for b in tb if b is not None]; tmpl = pd.concat(tb, axis=1).mean(1)
    G8 = pd.read_parquet(f"{WORK}/blood8_profiles.parquet")[BC["groups"]]
    common = tmpl.dropna().index.intersection(G8.dropna().index).difference(NSITES)
    dmin = (G8.loc[common].sub(tmpl.loc[common], axis=0)).abs().min(1); extra = dmin[dmin >= 0.25].sort_values(ascending=False).index[:500]
    sites = MARK.union(extra).intersection(common)
    T = {"version": "dev_foreign_placenta_EPIC_v1", "label": DV.DEV_LABEL, "name": "placenta (GSE271697 first half)", "template_arrays": tg,
         "sites": list(sites), "template": [float(tmpl[s]) for s in sites], "blood_profiles": {k: [float(G8.at[s, k]) for s in sites] for k in BC["groups"]},
         "rule": "963 composition markers + up to 500 sites where the template differs from every purified blood group by >= 0.25; blood profiles = GSE110554 + GSE167998 purified group means (our Stage 1)",
         "n_extra_sites": int(len(extra))}
    json.dump(T, open(f"{OUT}/Development/dev_foreign_placenta_EPIC_v1.json", "w")); json.dump(T, open(f"{CH}/Runtime Matrices/Development/dev_foreign_placenta_EPIC_v1.json", "w"))
    DV._C.pop("dev_foreign_placenta_EPIC_v1.json", None)
    sg = [g for g in sg if beta(g) is not None]
    jobs = [(g, sg[i % len(sg)], f) for i, g in enumerate(REPL.gsm) if beta(g) is not None for f in (0.0, 0.01, 0.02, 0.05, 0.10)]
    with ProcessPoolExecutor(64) as ex: F3 = pd.DataFrame(list(ex.map(_foreign_job, *zip(*jobs))))
    F3.to_csv(f"{OUT}/foreign3c.csv", index=False)
    S["B3_foreign"] = {"by_f": F3.groupby("f").agg(called=("called", "mean"), f_hat_median=("f_hat", "median"), se_median=("se", "median"), n=("f", "size")).reset_index().to_dict("records"),
                       "n_extra_sites": int(len(extra)), "n_sites": int(len(sites)), "bars": {"f0_called_max": 0.05, "f5_called_min": 0.95}}
    # 11b
    pooled = REPL[REPL.pooled == True]
    with ProcessPoolExecutor(32) as ex: BR = pd.DataFrame(list(ex.map(_bright_job, pooled.gsm)))
    BR = BR.merge(pooled[["gsm", "person"]], on="gsm"); BR.to_csv(f"{OUT}/brightness11b.csv", index=False)
    pr = []
    for p, g in BR.groupby("person"):
        g = g.dropna(subset=["A", "hw"]).reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pr.append(dict(person=p, d=abs(g.A[i] - g.A[j]), lim=float(np.sqrt(g.hw[i] ** 2 + g.hw[j] ** 2))))
    PR = pd.DataFrame(pr); PR.to_csv(f"{OUT}/brightness11b_pairs.csv", index=False)
    S["B3_brightness"] = {"n_arrays": len(BR), "n_pairs": len(PR), "covered_frac": float((PR.d <= PR.lim).mean()) if len(PR) else None,
                          "half_width_median": float(BR.hw.median()), "pair_absdiff_median": float(PR.d.median()) if len(PR) else None, "bar": 0.95}


# ------------------------------------------------------------------ B4
def _dir_wb(gsm):
    b = beta(gsm); comp = C3.stage_a_composition(b); r = DV.direction_record(b, "whole blood", comp); return dict(gsm=gsm, D=r.get("D"), n=r.get("n_sites"))


def B4():
    with ProcessPoolExecutor(48) as ex: Dd = pd.DataFrame(list(ex.map(_dir_wb, REPL.gsm)))
    rep = REPL.merge(Dd, on="gsm"); rep["series"] = "GSE250556"
    rep["D_rel"], rep["s"] = tare_diff(rep, "D", ["slide", "series"], minref=3)
    rep["call"] = np.where(rep.D_rel > 2 * rep.s, "toward disorder", np.where(rep.D_rel < -2 * rep.s, "toward over-order", "no direction"))
    rep.loc[rep.D_rel.isna(), "call"] = "no reference"; rep.to_csv(f"{OUT}/direction02_replicates.csv", index=False)
    out = {"replicates": {"n": int(rep.D_rel.notna().sum()), "no_direction": int((rep.call == "no direction").sum()), "bar_frac": 0.95,
                          "calls": rep.call.value_counts().to_dict()}}
    T = MAN[MAN.series == "GSE187291"].copy(); T["line"] = T.title.str.extract(r"^(\w+?)_")[0]; T["trt"] = T.title.str.extract(r"_(DMSO|DAC|NTX301)")[0]
    rows = []
    for line, g in T.groupby("line"):
        veh = [beta(x) for x in g[g.trt == "DMSO"].gsm]; veh = [v for v in veh if v is not None]
        if len(veh) < 2: continue
        V = pd.concat(veh, axis=1).dropna(); mu, sd = V.mean(1), V.std(1)
        cand = sd[sd <= 0.05].index
        hi = mu.loc[cand][(mu.loc[cand] >= 0.75) & (mu.loc[cand] <= 0.95)]; lo = mu.loc[cand][(mu.loc[cand] >= 0.05) & (mu.loc[cand] <= 0.25)]
        sites = sd.loc[hi.index].sort_values().index[:3000].union(sd.loc[lo.index].sort_values().index[:3000])
        for r in g.itertuples():
            b = beta(r.gsm)
            if b is None: continue
            d = DV.direction(b, mu, sites); rows.append(dict(line=line, gsm=r.gsm, trt=r.trt, title=r.title, D=d["D"], n_sites=d["n_sites"],
                                                         mean_beta_meth_sites=float(b.reindex(hi.index).mean())))
    X = pd.DataFrame(rows); calls = []
    for r in X.itertuples():
        ref = X[(X.line == r.line) & (X.trt == "DMSO") & (X.gsm != r.gsm)].D
        dr, s = r.D - ref.median(), ref.std(ddof=1)
        calls.append((dr, s, "toward disorder" if dr > 2 * s else "toward over-order" if dr < -2 * s else "no direction"))
    X["D_rel"], X["s"], X["call"] = zip(*calls) if calls else ([], [], []); X.to_csv(f"{OUT}/direction02_treated.csv", index=False)
    out["treated"] = {"rows": X[["line", "trt", "D", "D_rel", "s", "call"]].round(5).values.tolist(),
                      "treated_toward_disorder": int(((X.trt != "DMSO") & (X.call == "toward disorder")).sum()), "treated_n": int((X.trt != "DMSO").sum()),
                      "vehicle_no_direction": int(((X.trt == "DMSO") & (X.call == "no direction")).sum()), "vehicle_n": int((X.trt == "DMSO").sum())}
    S["B4"] = out


# ------------------------------------------------------------------ B5
def _sky_job(gsm, free):
    b = beta(gsm); comp = C3.stage_a_composition(b); fr = comp.get("fractions")
    if not fr: return dict(gsm=gsm, status="no composition")
    z = DV.residual_z(b, fr, DV.load_atlas_parents(APP)); r = DV.sky_from_z(z, free_null=free)
    d = dict(gsm=gsm, f_sky=r["f_sky"], n_sites=r["n_sites"], lee_T=r["lee_T"], lee_p=r["lee_p"], beyond=r["structure_beyond_null"])
    d.update({f"ratio_b{i+1}": x for i, x in enumerate(r["ratio_to_block_null"])})
    if free: d.update({f"free_b{i+1}": x for i, x in enumerate(r["ratio_to_free_null"])})
    return d


def B5():
    if not os.path.exists(APP): DV.load_atlas_parents(ATLAS).to_parquet(APP); DV._C.pop("AP", None)
    hw = R1[R1.arm.isin(["pass1", "diag1"]) & (R1.cls == "ok")].drop_duplicates("gsm").merge(MAN[["gsm", "specimen", "healthy"]], on="gsm")
    hw = hw[(hw.specimen == "whole blood") & (hw.healthy == True) & (hw.series != "GSE250556")].sort_values("gsm")
    others = [g for g in hw.gsm if beta(g) is not None][:100]
    jobs = [(g, True) for g in REPL.gsm if beta(g) is not None] + [(g, False) for g in others]
    with ProcessPoolExecutor(40) as ex: K = pd.DataFrame(list(ex.map(_sky_job, *zip(*jobs))))
    K["set"] = np.where(K.gsm.isin(REPL.gsm), "GSE250556", "other healthy whole blood"); K.to_csv(f"{OUT}/sky02.csv", index=False)
    r = K[K.set == "GSE250556"]; med = {f"b{i}": float(r[f"ratio_b{i}"].median()) for i in range(1, 7)}
    rate = float(K.beyond.mean()); n = int(K.beyond.notna().sum()); se = float(np.sqrt(0.05 * 0.95 / max(n, 1)))
    S["B5"] = {"median_ratio_block_null_GSE250556": med, "median_ratio_free_null_GSE250556": {f"b{i}": float(r[f"free_b{i}"].median()) for i in range(1, 7)},
               "band_bar": [0.9, 1.1], "lee_rate_all": rate, "lee_n": n, "lee_bar": 0.05 + 2 * se, "lee_rate_by_set": K.groupby("set").beyond.mean().to_dict()}


# ------------------------------------------------------------------ B6
def B6():
    FD = json.load(open(f"{CH}/Runtime Matrices/Met_A_Floors/metA_floors_v1_2_ALLCELLS_development.json"))["platforms"]["EPIC"]
    AP = DV.load_atlas_parents(ATLAS); out = {}
    sc = pd.read_csv(f"{W}/scores_r1.csv")
    for key, cell, spec, g8 in (("monocytes", "monocytes", "sorted monocytes", "MONO"), ("B", "b cells", "sorted B cells", "B")):
        fl = FD[cell]; sites = pd.Index(fl["sites"]); floor = float(fl["floor"])
        def A_of(g):
            b = beta(g)
            if b is None: return None
            x = b.reindex(sites).dropna(); return float(H(x.values).mean() / floor) if len(x) >= 0.9 * len(sites) else None
        pure = MAN[(MAN.specimen == spec) & MAN.healthy & ~MAN.series.isin(["GSE110554", "GSE167998", "GSE181034"])].copy(); pure["A"] = [A_of(g) for g in pure.gsm]
        pure["A_rel"] = tare(pure, "A", ["series"]); t1 = pure.A_rel.dropna()
        rr = []
        for t in REPL[REPL.pooled == True].itertuples():
            b = beta(t.gsm)
            if b is None: continue
            f = C3.stage_a_composition(b)["fractions"]; e = sum(f[g] * AP[f"{c}_mean"].reindex(sites) for g, c in DV.PARENTS.items()); x = b.reindex(sites); ok = x.notna() & e.notna()
            rr.append(dict(gsm=t.gsm, person=t.person, slide=t.slide, series="GSE250556", A=float(H(x[ok].values).mean() / H(e[ok].values).mean()), frac=f[g8]))
        Q = pd.DataFrame(rr); Q["A_rel"] = tare(Q, "A", ["slide", "series"])
        oth = MAN[MAN.healthy & MAN.specimen.isin(["isolated neutrophils", "sorted monocytes", "sorted B cells", "sorted NK cells", "sorted T cells", "sorted eosinophils", "sorted basophils"]) & (MAN.specimen != spec)].copy()
        oth["A"] = [A_of(g) for g in oth.gsm]; o3 = oth.A.dropna()
        h1 = sc[(sc.method == "NNLS8") & (sc.set == "H1") & (sc.group == g8)].rmse
        pure.to_csv(f"{OUT}/newcell_{key}_test1.csv", index=False); Q.to_csv(f"{OUT}/newcell_{key}_test2.csv", index=False); oth.to_csv(f"{OUT}/newcell_{key}_test3a.csv", index=False)
        out[key] = {"test1": {"n_tared": int(len(t1)), "in_normal": int(inN(t1).sum()), "frac": float(inN(t1).mean()) if len(t1) else None, "bar": 0.95,
                              "by_series": {s: [int(inN(g.A_rel.dropna()).sum()), int(g.A_rel.notna().sum())] for s, g in pure.groupby("series")}},
                    "test2": {"n": int(len(Q)), "within_person_sd": wsd(Q, "A_rel"), "bar": 0.020, "fraction_median": float(Q.frac.median()) if len(Q) else None,
                              "note": "Stage M read line (fraction >= 0.20) would withhold this reading"},
                    "test3a": {"n": int(len(o3)), "outside_normal": int((~inN(o3)).sum()), "frac": float((~inN(o3)).mean()) if len(o3) else None, "bar": 0.99},
                    "test3b": {"H1_NNLS8_rmse": float(h1.iloc[0]) if len(h1) else None, "bar": 0.03}}
    S["B6"] = out


# ------------------------------------------------------------------ B7
def _cal77(grn, red):
    import stage_1_idat_calibration as S1
    b, _ = S1.calibrate_idat_to_beta(grn, red, verbose=False); b = (b.iloc[:, 0] if hasattr(b, "columns") else b).dropna(); b.index = b.index.astype(str); return b


def B7():
    d = f"{WORK}/GSE77797"; os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/.done"):
        subprocess.run(["bash", "-c", f"cd {d} && curl -sSfL --retry 5 -o raw.tar https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77797/suppl/GSE77797_RAW.tar && tar -xf raw.tar && "
                        f"curl -sSfL --retry 5 -o sm.txt.gz https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77797/matrix/GSE77797_series_matrix.txt.gz && touch .done"], check=True)
    sm = [l.rstrip("\n").split("\t") for l in gzip.open(f"{d}/sm.txt.gz", "rt") if l.startswith("!Sample_")]
    smd = {}
    for row in sm: smd.setdefault(row[0], []).append([x.strip('"') for x in row[1:]])
    gsms = smd["!Sample_geo_accession"][0]; titles = smd["!Sample_title"][0]
    ch = pd.DataFrame({"gsm": gsms, "title": titles})
    for i, row in enumerate(smd.get("!Sample_characteristics_ch1", [])): ch[f"char{i}"] = row
    ch.to_csv(f"{OUT}/gse77797_characteristics.csv", index=False)
    # markers present on 450K (decided before truth is read)
    from stage_0_1_qc_handoff import _manifest
    m450 = set(_manifest("HM450K").data_frame.index.astype(str)); nm = int(sum(1 for x in BC["markers"] if x in m450))
    S["B7"] = {"markers_on_450K": nm, "required": C3.min_markers()}
    pairs = []
    for g in gsms:
        gr = sorted(glob.glob(f"{d}/{g}_*Grn.idat*"))
        if gr: pairs.append((g, gr[0], gr[0].replace("_Grn", "_Red")))
    with ProcessPoolExecutor(18) as ex: bs = dict(zip([p[0] for p in pairs], ex.map(_cal77, [p[1] for p in pairs], [p[2] for p in pairs])))
    rows = []
    import dev_comp_methods as CM
    N8 = CM.NNLS8(BC)
    for g, b in bs.items():
        n8 = N8.deconvolve(b)
        try: ae = DV.atlas_e(b, ATLAS)
        except Exception as e: ae = {"error": f"{type(e).__name__}: {e}"}
        try: ne = DV.nilc_e(b, ATLAS)
        except Exception as e: ne = {"error": f"{type(e).__name__}: {e}"}
        S["B7"].setdefault("errors", {}).update({k: v["error"] for k, v in (("ATLAS_e", ae), ("NILC_e", ne)) if "error" in v})
        for meth, fr in (("NNLS8", n8["fractions"]), ("ATLAS_e", ae.get("fractions") or {}), ("NILC_e", ne.get("fractions") or {})):
            rows.append(dict(gsm=g, method=meth, n_markers_used=n8["n_markers_used"] if meth == "NNLS8" else None, **{f"est_{k}": float(fr.get(k, 0.0)) for k in DV.BLOOD8}))
    pd.DataFrame(rows).to_csv(f"{OUT}/gse77797_fractions.csv", index=False)
    S["B7"]["n_arrays"] = len(bs)


# ------------------------------------------------------------------ B8
def _flag_job(gsm):
    b = beta(gsm); d = f"{WORK}/flags/{gsm}"; os.makedirs(d, exist_ok=True); csv = f"{d}/b.csv"; b.rename("beta").to_frame().to_csv(csv, index_label="cpg_id")
    RS = f"{CH}/MethylPhys_Interface/run_sample.py"; base = [sys.executable, RS, "--betas", csv, "--specimen", "whole blood", "--id", gsm, "--slide-ref-A", "1.15,1.2,1.25", "--array-type", "EPIC_v1"]
    fl = ["--dev-selftare-ii", "--dev-direction", "--dev-trace", "--dev-foreign", "--dev-brightness", "--dev-nilc", "--dev-atlas-e", "--dev-percell-b", "--dev-sky", "--atlas-v2", ATLAS]
    env = dict(os.environ, PYTHONPATH=CH)
    r0 = subprocess.run(base + ["--out", f"{d}/off.html"], capture_output=True, text=True, env=env, cwd=os.path.dirname(RS))
    r1 = subprocess.run(base + fl + ["--out", f"{d}/on.html"], capture_output=True, text=True, env=env, cwd=os.path.dirname(RS))
    try:
        a, c = json.load(open(f"{d}/off_bundle.json")), json.load(open(f"{d}/on_bundle.json"))
    except Exception as e:
        return dict(gsm=gsm, ok=False, err=(r1.stderr or r0.stderr)[-400:])
    key = lambda o: (o["met_a"].get("A"), o["met_a"].get("state"), o["tare"].get("A_rel"), o["tare"].get("state"), o["met_a_cscore"].get("C"))
    dev = c.get("development") or {}
    return dict(gsm=gsm, ok=True, reading_identical=key(a) == key(c), A=a["met_a"].get("A"),
                blocks=";".join(f"{k}:{(v or {}).get('status')}" for k, v in dev.items() if isinstance(v, dict)),
                all_labelled=all((v or {}).get("label") == DV.DEV_LABEL for k, v in dev.items() if isinstance(v, dict)),
                section_in_report="sec-development" in open(f"{d}/on.html").read())


def B8():
    with ProcessPoolExecutor(12) as ex: F = pd.DataFrame(list(ex.map(_flag_job, [g for g in REPL.gsm if beta(g) is not None])))
    F.to_csv(f"{OUT}/flags01.csv", index=False)
    S["B8"] = {"n": len(F), "ok": int(F.ok.sum()), "reading_identical": int(F.reading_identical.fillna(False).sum()), "all_labelled": int(F.all_labelled.fillna(False).sum()),
               "section_in_report": int(F.section_in_report.fillna(False).sum()), "blocks_example": F.blocks.dropna().iloc[0] if F.blocks.notna().any() else None}


for nm, fn in (("B1", B1), ("B2", B2), ("B3", B3), ("B4", B4), ("B5", B5), ("B6", B6), ("B7", B7), ("B8", B8)): step(nm, fn)
log("ALL DONE")
