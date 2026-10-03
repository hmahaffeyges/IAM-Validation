#!/usr/bin/env python3
"""Phase B toolkit checks on chain v3 (box). DEV-NILC-01 (stage 4), DEV-ATLAS-EPIC-02 (stage 3), DEV-PERCELL-01 (stage 5, qualifying cells),
DEV-SKY-01 (stage 11), DEV-TOOLKIT-ADDED-01 12b, and the per-group marker counts of blood_composition_EPIC_v1.json. Every rule and bar is the
one written in the doors/ notes before this ran. Outputs: ./tk/*.csv, ./tk/summary.json."""
import os, sys, json, glob, time, traceback
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
W = os.getcwd(); D = "/home/ubuntu/data/base_chain_01"; REPO = f"{D}/repo/Biological_Physics/MethylPhys"; CH = f"{REPO}/chain"
CODE = f"{REPO}/doors/data/DEV_ATLAS_EPIC_01/code"; BET = f"{D}/betas"; OUT = f"{W}/tk"; os.makedirs(OUT, exist_ok=True)
sys.path[:0] = [CODE, CH]
os.environ.setdefault("OMP_NUM_THREADS", "1")
import comp_methods as CM
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
ATLAS = "/home/ubuntu/data/dev_atlas_epic_01/IAMAtlas_v2.parquet"
BC = json.load(open(f"{CH}/Runtime Matrices/Met_A_Floors/blood_composition_EPIC_v1.json")); NS = pd.Index(BC["neutrophil_sites"])
ROSTER = pd.read_csv(f"{REPO}/atlas/v2/inputs/roster.csv"); THR = json.load(open(f"{CH}/Runtime Matrices/Celltype_Marker/twin_family_thresholds_v1.json"))
MAN = pd.read_csv(f"{W}/manifest.csv", dtype={"slide": str}); RD = pd.read_csv(f"{D}/out/readings_all.csv")
H1 = pd.read_csv(f"{W}/h1_truth.csv"); H2D = pd.read_csv(f"{W}/h2_design.csv", dtype=str); REPL = pd.read_csv(f"{W}/repl.csv", dtype={"slide": str})
INS = pd.read_csv(f"{W}/samples_and_truth.csv")
SUMMARY = {}

def beta(g):
    p = f"{BET}/{g}.parquet"
    if not os.path.exists(p): return None
    b = pd.read_parquet(p).iloc[:, 0].astype("float64"); b.index = b.index.astype(str); return b

# ---------------- methods ----------------
A = CM.load_atlas(ATLAS); cells = CM.atlas_cells(A); plat = ROSTER.set_index("cell")["platforms"].astype(str).to_dict()
WB = [c for c in cells if c not in CM.DROP_WB]; ARR = [c for c in WB if "array" in plat.get(c, "")]
ARRb = [c for c in ARR if CM.grp(c) in CM.BLOOD8]
De0 = CM.build_deconv(A, ARRb, NS); removed, _ = CM.mixture_twin(De0, ARRb, THR["twin_r"])
ARRe = [c for c in ARRb if c not in removed]
SUMMARY["rule_R"] = {"removed": sorted(removed), "expected": ["b cells", "cd4 t cells", "cd8 t cells"], "cells_e": ARRe,
                     "pass": sorted(removed) == ["b cells", "cd4 t cells", "cd8 t cells"]}
log("rule R", SUMMARY["rule_R"])
DE = CM.build_deconv(A, ARRe, NS)
Ab = CM.subset_atlas(A, ARRe); Ab = Ab.loc[Ab.index.difference(NS)]
mu = Ab[[f"{c}_mean" for c in ARRe]]; v = Ab[[f"{c}_sd" for c in ARRe]].values ** 2 + Ab[[f"{c}_donor_sd" for c in ARRe]].values ** 2
ok = mu.notna().all(1).values & np.isfinite(v).all(1); rng = (mu.max(1) - mu.min(1)).values; sel = ok & (rng >= 0.2)
NE = CM.NILC(mu.index[sel], ARRe, mu.values[sel], v[sel], cov="atlas")
N8 = CM.NNLS8(BC)
PARENTS = ["neutrophils", "eosinophils", "basophils", "monocytes", "b cells", "nk cells", "cd4 t cells", "cd8 t cells"]
P8 = dict(zip(PARENTS, ["NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T"]))
AP = A[[f"{c}_{s}" for c in PARENTS for s in ("mean", "sd", "donor_sd")]].copy(); del A
from nilc_celltype_deconvolver import NILCCelltypeDeconvolver
N1 = NILCCelltypeDeconvolver(f"{REPO}/atlas/IAMAtlasREBUILD.csv.xz", f"{CH}/Runtime Matrices/Celltype_Marker/iamatlas_celltype_markers_v0_2.json")
G1 = {}
for c in N1.celltypes:
    G1[c] = ("NEU" if c in ("Neu", "Neutro", "Neutrophils_EPIC", "Neutrophils_reinius", "neutrophil") else
             "EOS" if c in ("Eos", "Eosino", "Eosinophils_reinius", "eosinophil") else "BASO" if c == "Baso" else
             "MONO" if c in ("Mono", "Monocytes_EPIC", "CD14_monocytes", "monocyte") else
             "B" if c in ("B", "B-cells_EPIC", "Bcell", "Bmem", "Bnv", "CD19_B-cells") else "NK" if c in ("NK", "NK-cells_EPIC", "CD56_NK-cells") else
             "CD4T" if c in ("CD4T", "CD4T-cells_EPIC", "CD4Tmem", "CD4Tnv", "CD4_T-cells", "Treg") else
             "CD8T" if c in ("CD8T", "CD8T-cells_EPIC", "CD8Tmem", "CD8Tnv", "CD8_T-cells") else "T_unsplit" if c in ("Tcell", "tcell") else "OTHER")
pd.Series(G1, name="group").to_csv(f"{OUT}/N1_cell_to_group.csv")
log("methods built", len(ARRe), int(sel.sum()), len(N1.celltypes))

def groups8(fr):
    g = {}
    for c, x in fr.items(): g[CM.grp(c)] = g.get(CM.grp(c), 0.0) + x
    return g

def run_methods(b):
    o = {}
    n8 = N8.deconvolve(b)["fractions"]; o["NNLS8"] = n8
    a = DE.deconvolve(b, n_boot=0); o["ATLAS_e"] = groups8(a["fractions"]); o["ATLAS_e_cells"] = a["fractions"]
    n = NE.deconvolve(b); o["NILC_e"] = groups8(n["fractions"]); o["NILC_e_sum"] = n["sum"]
    r = N1.deconvolve(b.dropna().to_dict(), presence_bootstrap=False)
    g = {}
    for c, x in (r.get("fractions") or {}).items(): g[G1[c]] = g.get(G1[c], 0.0) + x
    o["N1"] = g; o["N1_status"] = r.get("status")
    return o

def rows_for(tag, gsm, b, truth=None, extra=None):
    o = run_methods(b); out = []
    for m in ("NNLS8", "ATLAS_e", "NILC_e", "N1"):
        f = dict(o[m])
        if m == "N1": f["T"] = f.get("CD4T", 0) + f.get("CD8T", 0) + f.get("T_unsplit", 0)
        else: f["T"] = f.get("CD4T", 0) + f.get("CD8T", 0)
        out.append(dict(set=tag, gsm=gsm, method=m, **{f"est_{k}": f.get(k, 0.0) for k in ("NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T", "T")},
                        **({f"true_{k}": vv for k, vv in truth.items()} if truth else {}), **(extra or {})))
    return out, o

R = []
# H1 cord-blood DNA mixtures
for t in H1.itertuples():
    b = beta(t.gsm)
    if b is None: log("H1 missing beta", t.gsm); continue
    rr, _ = rows_for("H1", t.gsm, b, {k: getattr(t, k) for k in ("NEU", "MONO", "B", "NK", "CD4T", "CD8T")}); R += rr
log("H1 done")
# H2 constructed mixtures of GSE122244 purified healthy arrays
FR = [(0.60, 0.10, 0.10, 0.20), (0.70, 0.05, 0.05, 0.20), (0.50, 0.15, 0.10, 0.25), (0.40, 0.10, 0.15, 0.35), (0.30, 0.20, 0.20, 0.30), (0.20, 0.10, 0.10, 0.60)]
for d in H2D.itertuples():
    bs = {k: beta(getattr(d, k)) for k in ("NEU", "MONO", "B", "T")}
    if any(x is None for x in bs.values()): log("H2 missing beta donor", d.donor); continue
    idx = bs["NEU"].dropna().index
    for k in ("MONO", "B", "T"): idx = idx.intersection(bs[k].dropna().index)
    for j, (fn, fm, fb, ft) in enumerate(FR):
        mix = fn * bs["NEU"].reindex(idx) + fm * bs["MONO"].reindex(idx) + fb * bs["B"].reindex(idx) + ft * bs["T"].reindex(idx)
        rr, _ = rows_for("H2", f"donor{d.donor}_mix{j+1}", mix, {"NEU": fn, "MONO": fm, "B": fb, "T": ft}); R += rr
log("H2 done")
# in-sample sets (reported, not scored)
for t in INS[INS.set.isin(["MIX18", "MIX22", "MIX12", "FACS"])].itertuples():
    b = beta(t.gsm)
    if b is None: continue
    tr = {k: getattr(t, k) for k in ("NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T") if pd.notna(getattr(t, k))}
    rr, _ = rows_for(t.set, t.gsm, b, tr); R += rr
log("in-sample done")
# repeatability GSE250556 + agreement on healthy whole bloods
CELLF = {}
for t in REPL.itertuples():
    b = beta(t.gsm)
    if b is None: continue
    rr, o = rows_for("REPL", t.gsm, b, None, {"person": t.person, "pooled": t.pooled}); R += rr; CELLF[t.gsm] = o
hw = RD[RD.arm.isin(["pass1", "diag1"]) & (RD.cls == "ok")].drop_duplicates("gsm").merge(MAN[["gsm", "specimen", "healthy"]], on="gsm")
hw = hw[(hw.specimen == "whole blood") & (hw.healthy == True) & (hw.series != "GSE250556")]
for g in hw.gsm:
    b = beta(g)
    if b is None: continue
    rr, _ = rows_for("AGREE", g, b); R += rr
log("REPL + AGREE done", len(hw))
X = pd.DataFrame(R); X.to_csv(f"{OUT}/fractions_long.csv", index=False)

# ---------------- scoring (bars from DEV-NILC-01 / DEV-ATLAS-EPIC-02) ----------------
def rmse(df, k): d = (df[f"est_{k}"] - df[f"true_{k}"]).dropna(); return float(np.sqrt((d ** 2).mean())) if len(d) else None
def bias(df, k): d = (df[f"est_{k}"] - df[f"true_{k}"]).dropna(); return float(d.mean()) if len(d) else None
SC = []
for m in ("NNLS8", "ATLAS_e", "NILC_e", "N1"):
    for st in ("H1", "H2", "MIX18", "MIX22", "MIX12", "FACS"):
        df = X[(X.method == m) & (X.set == st)]
        for k in ("NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T", "T"):
            if f"true_{k}" in df and df[f"true_{k}"].notna().any():
                SC.append(dict(method=m, set=st, group=k, n=int(df[f"true_{k}"].notna().sum()), rmse=rmse(df, k), bias=bias(df, k),
                               bar=(0.02 if k == "NEU" else 0.03) if st in ("H1", "H2") else None))
S = pd.DataFrame(SC); S["pass"] = np.where(S.bar.notna(), S.rmse <= S.bar, None); S.to_csv(f"{OUT}/scores.csv", index=False)
rep = X[(X.set == "REPL") & (X.pooled == True)]
RP = []
for m in ("NNLS8", "ATLAS_e", "NILC_e", "N1"):
    d = rep[rep.method == m]
    for k in ("NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T"):
        ss = sum(((g[f"est_{k}"] - g[f"est_{k}"].mean()) ** 2).sum() for _, g in d.groupby("person")); dof = len(d) - d.person.nunique()
        RP.append(dict(method=m, group=k, n=len(d), within_person_sd=float(np.sqrt(ss / dof)) if dof > 0 else None))
RP = pd.DataFrame(RP); RP["pass"] = RP.within_person_sd <= 0.010; RP.to_csv(f"{OUT}/repeatability.csv", index=False)
ag = X[X.set == "AGREE"].pivot_table(index="gsm", columns="method", values=[f"est_{k}" for k in ("NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T")])
AG = []
for k in ("NEU", "EOS", "BASO", "MONO", "B", "NK", "CD4T", "CD8T"):
    for m in ("ATLAS_e", "NILC_e", "N1"):
        dd = (ag[(f"est_{k}", m)] - ag[(f"est_{k}", "NNLS8")]).dropna()
        AG.append(dict(method=m, group=k, n=len(dd), mean_diff_vs_NNLS8=float(dd.mean()), sd=float(dd.std()), bar=0.02 if k == "NEU" else 0.03))
AG = pd.DataFrame(AG); AG["pass"] = AG.mean_diff_vs_NNLS8.abs() <= AG.bar; AG.to_csv(f"{OUT}/agreement.csv", index=False)
verdict = {}
for m in ("NILC_e", "N1", "ATLAS_e"):
    s = S[(S.method == m) & S.bar.notna()]; r = RP[RP.method == m]
    v_ = dict(truth_pass=bool(s["pass"].all()), truth_fail=s[~s["pass"].astype(bool)][["set", "group", "rmse"]].round(4).values.tolist(),
              repeat_pass=bool(r["pass"].all()), repeat_fail=r[~r["pass"]][["group", "within_person_sd"]].round(4).values.tolist())
    if m == "ATLAS_e":
        a = AG[AG.method == m]; v_.update(agree_pass=bool(a["pass"].all()), agree_fail=a[~a["pass"]][["group", "mean_diff_vs_NNLS8"]].round(4).values.tolist())
    v_["pass"] = v_["truth_pass"] and v_["repeat_pass"] and v_.get("agree_pass", True) and (SUMMARY["rule_R"]["pass"] if m == "ATLAS_e" else True)
    # per-cell qualification for DEV-PERCELL-01: the cell is in both held-out truths and every bar for that cell holds
    qual = []
    for k in ("MONO", "B"):
        sk = s[s.group == k]; rk = r[r.group == k]
        if set(sk.set) >= {"H1", "H2"} and sk["pass"].all() and rk["pass"].all() and (m != "ATLAS_e" or SUMMARY["rule_R"]["pass"]): qual.append(k)
    v_["qualifying_cells"] = qual; verdict[m] = v_
SUMMARY["verdict"] = verdict; log("verdicts", json.dumps(verdict)[:1500])

# ---------------- DEV-PERCELL-01 (only qualifying cells) ----------------
FD = json.load(open(f"{CH}/Runtime Matrices/Met_A_Floors/metA_floors_v1_2_ALLCELLS_development.json"))["platforms"]["EPIC"]
QUAL = sorted({c for m in verdict.values() for c in m["qualifying_cells"]})
H = CM.H; PC = {}
SPEC = {"MONO": ("monocytes", "sorted monocytes"), "B": ("b cells", "sorted B cells")}
for k in QUAL:
    cell, spec = SPEC[k]; fl = FD[cell]; sites = pd.Index(fl["sites"]); floor = float(fl["floor"])
    pure = MAN[(MAN.specimen == spec) & (MAN.healthy == True) & ~MAN.series.isin(["GSE110554", "GSE167998", "GSE181034"])]
    rows = []
    for t in pure.itertuples():
        b = beta(t.gsm)
        if b is None: continue
        x = b.reindex(sites).dropna()
        if len(x) < 0.9 * len(sites): continue
        rows.append(dict(series=t.series, gsm=t.gsm, A=float(H(x.values).mean() / floor)))
    P = pd.DataFrame(rows)
    if len(P):
        P["A_rel"] = [r.A / np.median(P[(P.series == r.series) & (P.gsm != r.gsm)].A) if ((P.series == r.series) & (P.gsm != r.gsm)).sum() >= 3 else np.nan for r in P.itertuples()]
    tared = P.A_rel.dropna() if len(P) else pd.Series(dtype=float)
    pure_pass = bool(len(tared) and ((tared >= 0.95) & (tared <= 1.05)).mean() >= 0.95)
    # own replicate test: whole blood GSE250556 pooled replicates, expectation from atlas parents x NNLS8 fractions at the cell's sites
    mus = {c: AP[f"{c}_mean"].reindex(sites) for c in PARENTS}; rr = []
    for t in REPL[REPL.pooled == True].itertuples():
        b = beta(t.gsm)
        if b is None or t.gsm not in CELLF: continue
        f = CELLF[t.gsm]["NNLS8"]; e = sum(f[P8[c]] * mus[c] for c in PARENTS); x = b.reindex(sites); okk = x.notna() & e.notna()
        rr.append(dict(gsm=t.gsm, person=t.person, slide=t.slide, A=float(H(x[okk].values).mean() / H(e[okk].values).mean())))
    Q = pd.DataFrame(rr)
    Q["A_rel"] = [r.A / np.median(Q[(Q.slide == r.slide) & (Q.gsm != r.gsm)].A) if ((Q.slide == r.slide) & (Q.gsm != r.gsm)).sum() >= 3 else
                  r.A / np.median(Q[Q.gsm != r.gsm].A) for r in Q.itertuples()]
    ss = sum(((g.A_rel - g.A_rel.mean()) ** 2).sum() for _, g in Q.groupby("person")); dof = len(Q) - Q.person.nunique(); wsd = float(np.sqrt(ss / dof))
    PC[k] = dict(cell=cell, n_pure=len(P), n_pure_tared=int(len(tared)), pure_in_normal=float(((tared >= 0.95) & (tared <= 1.05)).mean()) if len(tared) else None,
                 pure_pass=pure_pass, repl_n=len(Q), repl_within_person_sd=wsd, repl_pass=wsd <= 0.020, pass_=pure_pass and wsd <= 0.020)
    P.to_csv(f"{OUT}/percell_{k}_pure.csv", index=False); Q.to_csv(f"{OUT}/percell_{k}_repl.csv", index=False)
SUMMARY["percell"] = PC if QUAL else "no qualifying cell: not run"; log("percell", PC)

# ---------------- DEV-SKY-01 (stage 11) ----------------
try:
    import sky_statistics as SS, healpy as hp
    from stage_4_6_patient_cmb import load_mapping, NPIX
    mp = load_mapping(); mp = mp[~mp.index.duplicated()]
    mu8 = pd.DataFrame({c: AP[f"{c}_mean"] for c in PARENTS}); var8 = {c: AP[f"{c}_sd"] ** 2 + AP[f"{c}_donor_sd"] ** 2 for c in PARENTS}
    SK = []
    for t in REPL.itertuples():
        b = beta(t.gsm)
        if b is None or t.gsm not in CELLF: continue
        f = CELLF[t.gsm]["NNLS8"]; idx = b.index.intersection(mp.index).intersection(mu8.index)
        E = sum(f[P8[c]] * mu8[c].reindex(idx) for c in PARENTS); V = sum(f[P8[c]] ** 2 * var8[c].reindex(idx) for c in PARENTS) + 0.02 ** 2
        z = ((b.reindex(idx) - E) / np.sqrt(V)).dropna(); px = mp.reindex(z.index).astype(int).values
        s = np.bincount(px, weights=z.values, minlength=NPIX); n = np.bincount(px, minlength=NPIX); pix = np.full(NPIX, np.nan); pix[n > 0] = s[n > 0] / n[n > 0]
        recd = SS.sky_spectrum_record(pix, rng=np.random.default_rng(20261003))
        nb = recd["null_bandpowers"]; bp = recd["bandpowers"]
        SK.append(dict(gsm=t.gsm, person=t.person, f_sky=recd["f_sky"], n_sites=len(z), frac_abs_z_gt2=recd["frac_abs_z_gt2"],
                       **{f"ratio_b{i+1}": float(bp[i] / nb[:, i].mean()) for i in range(6)}, **{f"above_null_max_b{i+1}": bool(bp[i] > nb[:, i].max()) for i in range(6)}))
    K = pd.DataFrame(SK); K.to_csv(f"{OUT}/sky_GSE250556.csv", index=False)
    med = {f"b{i+1}": float(K[f"ratio_b{i+1}"].median()) for i in range(6)}
    SUMMARY["sky"] = dict(n=len(K), median_ratio=med, frac_above_null_max={f"b{i+1}": float(K[f"above_null_max_b{i+1}"].mean()) for i in range(6)},
                          bands=SS.BANDS, pass_=all(0.9 <= x <= 1.1 for x in med.values()))
except Exception as e:
    SUMMARY["sky"] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}
log("sky", SUMMARY.get("sky"))

# ---------------- 12b difference map ----------------
try:
    import serial_mode as SMd
    def adapt(h, at="EPIC_v1", pipe="methylprep noob"): return {"context": {"patient_hash": h, "array_type": at, "pipeline": pipe}}
    a1 = SMd.check_same_person(adapt("a" * 32), adapt("a" * 32))[0]; a2 = SMd.check_same_person(adapt("a" * 32), adapt("b" * 32))[0]
    pooled = REPL[REPL.pooled == True]; Bv = {g: beta(g) for g in pooled.gsm}; per = pooled.set_index("gsm").person.to_dict(); C2 = []
    for p, G in pooled.groupby("person"):
        gl = [g for g in G.gsm if Bv.get(g) is not None]
        for i in range(0, len(gl) - 1, 2):
            g1, g2 = gl[i], gl[i + 1]; _, same = SMd.delta_sky(Bv[g1], Bv[g2])
            for q in pooled.gsm:
                if per[q] == p or Bv.get(q) is None: continue
                _, oth = SMd.delta_sky(Bv[g1], Bv[q])
                C2.append(dict(person=p, a=g1, b=g2, other=q, same_q99=same["q99_abs_dbeta"], other_q99=oth["q99_abs_dbeta"], same_below=same["q99_abs_dbeta"] < oth["q99_abs_dbeta"]))
    C2 = pd.DataFrame(C2); C2.to_csv(f"{OUT}/diffmap_12b.csv", index=False)
    SUMMARY["diffmap_12b"] = dict(adapter_accepts_same=bool(a1), adapter_refuses_different=not a2, n_comparisons=len(C2),
                                  frac_same_below=float(C2.same_below.mean()), same_q99_median=float(C2.same_q99.median()), other_q99_median=float(C2.other_q99.median()),
                                  pass_=bool(a1 and not a2 and C2.same_below.mean() >= 0.95))
except Exception as e:
    SUMMARY["diffmap_12b"] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}
log("12b", SUMMARY.get("diffmap_12b"))

# ---------------- per-group marker counts of blood_composition_EPIC_v1.json (re-derived by the builder's own rule) ----------------
try:
    RS = pd.read_csv(f"{REPO}/atlas/v2/inputs/roster_samples.csv"); SRC = {"Salas2018": "GSE110554", "Salas2022": "GSE167998"}
    GRP = BC["group_of_cell"]
    St = RS[RS.source.isin(SRC) & (RS.qc == True) & RS.cell.isin(GRP)].copy(); St["gsm"] = St["sample"].str.split("_").str[0]
    Bm = {}
    for r in St.itertuples():
        p = glob.glob(f"/home/ubuntu/data/atlas_sources/blood/{SRC[r.source]}/shards/{r.gsm}_*.parquet")
        if p: Bm[r.gsm] = pd.read_parquet(p[0]).iloc[:, 0].astype("float64")
    grp = {r.gsm: GRP[r.cell] for r in St.itertuples()}
    common = None
    for x in Bm.values(): common = x.dropna().index if common is None else common.intersection(x.dropna().index)
    Mm = pd.DataFrame({g: Bm[g].reindex(common) for g in Bm}); groups = sorted({grp[g] for g in Mm})
    mu_ = pd.DataFrame({k: Mm[[g for g in Mm if grp[g] == k]].mean(1) for k in groups}); sd_ = pd.DataFrame({k: Mm[[g for g in Mm if grp[g] == k]].std(1).fillna(0) for k in groups})
    cand = mu_.index.difference(NS); mk = {}
    for k in groups:
        oth = mu_.loc[cand, [c for c in groups if c != k]]; mg = np.abs(mu_.loc[cand, k].values[:, None] - oth.values).min(1)
        s_ = pd.Series(mg, index=cand)[(sd_.loc[cand, k] <= 0.05).values]; mk[k] = list(s_[s_ >= 0.25].sort_values(ascending=False).index[:150])
    union = sorted(set(sum(mk.values(), [])))
    same = union == sorted(BC["markers"])
    mud = float(np.nanmax(np.abs(mu_.loc[BC["markers"], BC["groups"]].values - pd.DataFrame(BC["mu_markers"], index=BC["markers"])[BC["groups"]].values))) if same else None
    SUMMARY["marker_counts"] = dict(per_group={k: len(v) for k, v in mk.items()}, union=len(union), reproduces_frozen_markers=same, max_abs_mu_diff=mud,
                                    n_arrays=len(Bm), markers_in_two_groups=int(sum(len(v) for v in mk.values()) - len(union)))
except Exception as e:
    SUMMARY["marker_counts"] = {"error": f"{type(e).__name__}: {e}"}
log("marker counts", SUMMARY.get("marker_counts"))
json.dump(SUMMARY, open(f"{OUT}/summary.json", "w"), indent=1, default=str); log("DONE")
