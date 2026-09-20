#!/usr/bin/env python3
"""LAB-ZERO-02 — fourth lab (GSE111629 UCLA controls). Executes PREREG.md as sealed; nothing is fit on UCLA.
Runs in the methylprep env. Stage 1 via subprocess workers (stage1_worker.py), control probes via ctrl_worker.py."""
import os, sys, json, glob, time, subprocess, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); E = os.path.join(ROOT, "iamrepo/Biological_Physics/CPG_Engine")
OUT = os.path.join(ROOT, "results/labzero2"); os.makedirs(OUT, exist_ok=True)
IDAT = os.path.join(ROOT, "idats/GSE111629/idats"); SEL = json.load(open(os.path.join(ROOT, "idats/GSE111629/selected.json")))
def log(*a): print(*a, flush=True)
def H(b): b = np.clip(np.asarray(b, float), 1e-12, 1 - 1e-12); return -b * np.log2(b) - (1 - b) * np.log2(1 - b)

# ---------- Stage 1 ----------
def stage1(workers=6):
    cache = os.path.join(OUT, "betas_GSE111629_controls.pkl")
    if os.path.exists(cache): return pd.read_pickle(cache)
    pairs = {}
    for g in sorted(glob.glob(os.path.join(IDAT, "*_Grn.idat*"))):
        gsm = os.path.basename(g).split("_")[0]
        if gsm in SEL: pairs[gsm] = (g, g.replace("_Grn", "_Red"))
    log(f"control IDAT pairs: {len(pairs)}  (PD arrays NOT opened)")
    items = list(pairs.items()); procs = []; t = time.time()
    for k in range(workers):
        jf = os.path.join(OUT, f"jobs_{k}.json"); json.dump([(g, p[0], p[1]) for g, p in items[k::workers]], open(jf, "w"))
        procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "stage1_worker.py"), jf, os.path.join(OUT, f"part_{k}.pkl")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs: p.wait()
    df = pd.concat([pd.read_pickle(os.path.join(OUT, f"part_{k}.pkl")) for k in range(workers) if os.path.exists(os.path.join(OUT, f"part_{k}.pkl"))], axis=1)
    df.to_pickle(cache); log(f"  stage1 {df.shape[1]}/{len(pairs)} in {time.time()-t:.0f}s"); return df

# ---------- control probes ----------
def controls(workers=6):
    cache = os.path.join(OUT, "control_features_GSE111629.pkl")
    if os.path.exists(cache): return pd.read_pickle(cache)
    files = [g for g in sorted(glob.glob(os.path.join(IDAT, "*_Grn.idat*"))) if os.path.basename(g).split("_")[0] in SEL]
    procs = []
    for k in range(workers):
        jf = os.path.join(OUT, f"cjobs_{k}.json"); json.dump(files[k::workers], open(jf, "w"))
        procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "ctrl_worker.py"), jf, os.path.join(OUT, f"cpart_{k}.pkl")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs: p.wait()
    F = pd.concat([pd.read_pickle(os.path.join(OUT, f"cpart_{k}.pkl")) for k in range(workers)]); F.to_pickle(cache); return F

def ridge(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Xt = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]; Xe = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
    I = np.eye(Xt.shape[1]); I[0, 0] = 0; w = np.linalg.solve(Xt.T @ Xt + lam * I, Xt.T @ ytr); return Xe @ w, Xt @ w

def main():
    df = stage1(); n_pairs = len(SEL)
    ident = json.load(open(os.path.join(E, "Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json")))
    ident = {k: v for k, v in ident.items() if isinstance(v, dict) and "loci" in v}
    m = json.load(open(os.path.join(E, "Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json")))["maps"]["stage1_noob_450K"]; slope, icpt = m["slope"], m["intercept"]
    IMM = [c for c in ident["immune"]["loci"] if c in df.index]; hmI = ident["immune"]["H_min"]
    # deconvolution presence gate
    sys.path.insert(0, os.path.join(E, "Walther_iam_deconvolver")); from walther_iam_deconvolver import WaltherIAMDeconvolver
    W = WaltherIAMDeconvolver(os.path.join(E, "../IAM_Atlas/IAMAtlasREBUILD.csv"), celltype_class_map=os.path.join(E, "../IAM_Atlas/IAMAtlasREBUILD_celltype_to_class.json"), verbose=False)
    EPI = ["cycling", "secretory", "terminal", "stromal"]; rows = {}
    for gsm in df.columns:
        b = df[gsm].dropna(); r = W.deconvolve({k: float(v) for k, v in b.items()}); fr = r.class_fractions
        mi = float(df.loc[IMM, gsm].mean()); A_raw = float(H(mi) / hmI); A_map = float(H((mi - icpt) / slope) / hmI)
        rows[gsm] = dict(immune_frac=fr.get("immune", 0), joint_frac=fr.get("progenitor", 0) + fr.get("stem_adult", 0), epi_frac=sum(fr.get(c, 0) for c in EPI),
                         A_raw=A_raw, A_mapped=A_map, age=float(SEL[gsm].get("age", "nan") or "nan"), sex=SEL[gsm].get("gender", ""), eth=SEL[gsm].get("ethnicity", ""))
    S = pd.DataFrame(rows).T; S.to_csv(os.path.join(OUT, "per_sample.csv"))
    for c in ("immune_frac", "joint_frac", "epi_frac", "A_raw", "A_mapped", "age"): S[c] = S[c].astype(float)
    gate = (S.immune_frac + S.joint_frac >= 0.85) & (S.epi_frac <= 0.02); Sb = S[gate]
    res = dict(P1_coverage=round(df.shape[1] / n_pairs, 4), P2_presence=round(float(gate.mean()), 4), n_after_gate=int(gate.sum()),
               P3_median_mapped_A=round(float(Sb.A_mapped.median()), 4), unmapped_median_A=round(float(Sb.A_raw.median()), 4))
    # ---------- four-lab lab-zero test ----------
    F4 = controls(); F4 = F4.drop(columns=[c for c in F4.columns if c == "error"]).astype(float)
    F3 = pd.read_pickle(os.path.join(ROOT, "results/labzero/control_features.pkl")).astype(float)
    P3 = pd.read_csv(os.path.join(ROOT, "results/band_v2/pooled_per_sample.csv"), index_col=0)[["A_imm", "lab", "age"]]
    M3 = pd.read_csv(os.path.join(ROOT, "results/band_v2_test/per_sample.csv"), index_col=0); M3 = M3[(M3.immune_frac + M3.joint_frac >= 0.85) & (M3.epi_frac <= 0.02)]
    M3 = pd.DataFrame({"A_imm": M3.A_immune_mapped, "lab": "GSE125105_Munich", "age": M3.age})
    U = pd.DataFrame({"A_imm": Sb.A_mapped, "lab": "GSE111629_UCLA", "age": Sb.age})
    ALL = pd.concat([pd.concat([P3, M3]).join(F3, how="inner"), U.join(F4, how="inner")]).dropna(); feats = list(F3.columns)
    ref = ALL[ALL.lab == "GSE87571_Uppsala"].A_imm.median(); obs = {l: round(float(ALL[ALL.lab == l].A_imm.median() - ref), 4) for l in ALL.lab.unique()}
    res["observed_offsets_vs_Uppsala"] = obs
    def loco(held):
        tr = ALL[ALL.lab != held]; te = ALL[ALL.lab == held]
        pred, pred_tr = ridge(tr[feats].values, tr.A_imm.values, te[feats].values, 1.0)
        ref_lab = "GSE87571_Uppsala" if held != "GSE87571_Uppsala" else "GSE42861_Karolinska"
        pred_off = float(np.median(pred) - np.median(pred_tr[(tr.lab == ref_lab).values])); obs_off = float(te.A_imm.median() - tr[tr.lab == ref_lab].A_imm.median())
        return dict(obs=round(obs_off, 4), pred=round(pred_off, 4), err=round(abs(pred_off - obs_off), 4))
    res["P4_UCLA_from_three_labs"] = loco("GSE111629_UCLA"); res["P4_pass"] = res["P4_UCLA_from_three_labs"]["err"] <= 0.010
    res["P4b_LOO_four"] = {l: loco(l) for l in sorted(ALL.lab.unique())}; res["P4b_pass"] = all(v["err"] <= 0.010 for v in res["P4b_LOO_four"].values())
    # ---------- P6 within-cohort narrowing ----------
    tr = ALL[ALL.lab != "GSE111629_UCLA"].copy(); tr["y"] = tr.A_imm - tr.groupby("lab").A_imm.transform("mean")
    te = ALL[ALL.lab == "GSE111629_UCLA"]
    pred, _ = ridge(tr[feats].values, tr.y.values, te[feats].values, 1.0); corr = te.A_imm.values - (pred - pred.mean())
    b2 = json.load(open(os.path.join(E, "Runtime Matrices/A_Scoring_Module/identity_band_v2_PROVISIONAL.json")))["immune"]
    band = {e["decade"]: (e["p10"], e["p90"]) for e in b2 if e.get("p10") is not None and e["n"] >= 30}
    def inb(A, age):
        for k, (lo, hi) in band.items():
            a, b_ = map(int, k.split("-"))
            if a <= age <= b_: return lo <= A <= hi
        return None
    ib0 = [inb(a, g) for a, g in zip(te.A_imm, te.age)]; ib1 = [inb(a, g) for a, g in zip(corr, te.age)]
    res["P6_within_cohort"] = dict(sd_before=round(float(te.A_imm.std()), 4), sd_after=round(float(np.std(corr, ddof=1)), 4),
                                  narrowing=round(1 - float(np.std(corr, ddof=1)) / float(te.A_imm.std()), 3),
                                  median_shift=round(float(np.median(corr) - te.A_imm.median()), 4),
                                  in_band_before=round(float(np.mean([x for x in ib0 if x is not None])), 3), in_band_after=round(float(np.mean([x for x in ib1 if x is not None])), 3))
    res["P6_pass"] = res["P6_within_cohort"]["narrowing"] >= 0.20 and abs(res["P6_within_cohort"]["median_shift"]) <= 0.005
    # ---------- nulls ----------
    chips = pd.Series({g: os.path.basename(f).split("_")[1] for f in glob.glob(os.path.join(IDAT, "*_Grn.idat*")) for g in [os.path.basename(f).split("_")[0]]})
    Sb = Sb.assign(chip=chips.reindex(Sb.index)); grp = [g.A_mapped.values for _, g in Sb.groupby("chip") if len(g) >= 3]
    from scipy import stats
    if len(grp) >= 2: Fst, pv = stats.f_oneway(*grp); res["N_plate"] = dict(n_chips=len(grp), F=round(float(Fst), 3), p=float(pv))
    def sd(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) < 5 or len(b) < 5: return None
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)); return round(float((a.mean() - b.mean()) / sp), 3)
    res["N_ethnicity_signed_d_hisp_minus_cauc"] = sd(Sb[Sb.eth == "Hispanic"].A_mapped, Sb[Sb.eth == "Caucasian"].A_mapped)
    res["N_sex_signed_d_f_minus_m"] = {f"{lo}-{lo+9}": sd(Sb[(Sb.age >= lo) & (Sb.age <= lo + 9) & (Sb.sex == "Female")].A_mapped, Sb[(Sb.age >= lo) & (Sb.age <= lo + 9) & (Sb.sex == "Male")].A_mapped) for lo in (40, 50, 60, 70, 80)}
    json.dump(res, open(os.path.join(OUT, "labzero2_results.json"), "w"), indent=1, default=str); log(json.dumps(res, indent=1, default=str))

if __name__ == "__main__": main()
