#!/usr/bin/env python3
"""DEV-SELFTARE-01 (development prototype; NOT part of chain v3; conductor_v3.py is not changed).

Per-array self-tare for neutrophil Met-A. The array's own 48,528 fixed sites (noise_sites_EPIC_v1.json) give its measurement
offsets; the expected entropy those offsets add at each identity site is removed before Met-A is formed. Nothing is fitted; no
statistic is taken across people. Inputs: Stage-1 betas of each array (chain calibrate_idat_to_beta), the frozen chain files in
Runtime Matrices/Met_A_Floors, and the six frozen purified neutrophil reference arrays (GSE110554, listed in metA_floors_v1_3.json).

Model (derivation: doors/DEV_SELFTARE_01.md)
  measured beta = b (1 - dU - dM) + dU     b = true beta of the site; dU = beta read at a site whose true state is 0,
                                           dM = 1 - beta read at a site whose true state is 1 (same array, same probe design I / II)
  mean inversion      b_hat = (beta - mean dU) / (1 - mean dU - mean dM)
  entropy inflation   I(b) = E[ H(b (1 - dU - dM) + dU) ] - H(b)    (E over this array's own dU, dM: convolution)
  self-tared site H   H_st = H(beta) - I(b_hat)
  isolated:    A_st = mean H_st / floor_st,  floor_st = mean over the 6 reference arrays of their own mean H_st
  whole blood: A_st = mean H_st / mean H(e*), e* = (e - dU_ref) / (1 - dU_ref - dM_ref), e = sum_g f_g mu_g (frozen profiles,
               this specimen's fractions from the chain's Stage A); dU_ref, dM_ref = mean offsets of the 6 reference arrays.
Variant recorded beside it (mean inversion only, no convolution): H_inv = H(b_hat).
Variant B (per probe, from intensities; methylprep beta = M / (M + U), noob M and U):
  rM = median M at this array's fixed unmethylated sites, rU = median U at its fixed methylated sites, per probe class
  (type II; type I green; type I red).  b_hat_B = (M - rM)+ / ((M - rM)+ + (U - rU)+),  site entropy H(b_hat_B).
  isolated: A_B = mean H(b_hat_B) / floor_B (floor_B = mean over the 6 reference arrays); whole blood: A_B = mean H(b_hat_B) / mean H(e*).
Variant I: as the self-tare, but every identity site uses the type I offsets (40,827 + 7,553 fixed sites instead of 55 + 93 type II).
A reading needs >= 90 % of the 6,000 identity sites measured (the chain's SITE_COVERAGE_MIN); otherwise A is not formed.

Usage: python dev_selftare_01.py --data <dir: beta_subset.parquet probe_design.csv untared_records.csv jobs.csv>
         --chain <chain dir (conductor_v3.py, Runtime Matrices)> --man <GSE250556 manifest_from_series_matrix.csv>
         --sm <dir with GSE247193/5 series matrices> --out <dir>
"""
import json, os, sys, gzip, argparse
import numpy as np, pandas as pd

NORMAL = (0.95, 1.05); MIN_REFS = 3; NQ = 200; BGRID = np.linspace(0, 1, 401); DES = ("I", "II")

def H(b):
    b = np.clip(np.asarray(b, dtype="float64"), 1e-6, 1 - 1e-6)
    return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))

def qgrid(x, n=NQ):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return np.quantile(x, (np.arange(n) + 0.5) / n)

def inflation_curve(dU, dM):
    """I(b) on BGRID = E[H(b(1-dU-dM)+dU)] - H(b), dU and dM independent draws from this array's own fixed-site readings
    (NQ x NQ quantile grid of each)."""
    u = qgrid(dU)[:, None]; m = qgrid(dM)[None, :]
    return np.array([H(b * (1 - u - m) + u).mean() - H(b) for b in BGRID])

def array_noise(bns, state, design):
    """This array's offsets at its fixed sites, per probe design."""
    out = {}
    for d in DES:
        u = bns[(state == "U") & (design == d)].dropna().values; m = 1 - bns[(state == "M") & (design == d)].dropna().values
        out[d] = dict(dU=u, dM=m, mU=float(u.mean()), mM=float(m.mean()))
        out[d]["curve"] = inflation_curve(u, m)
    return out

def self_tared_H(x, design, nz):
    """x: measured beta at identity sites. Returns (H_st, H(b_hat)) per site."""
    hst = pd.Series(np.nan, index=x.index); hinv = hst.copy()
    for d in DES:
        s = (design == d) & x.notna()
        if not s.any(): continue
        bh = ((x[s] - nz[d]["mU"]) / (1 - nz[d]["mU"] - nz[d]["mM"])).clip(0, 1)
        hst[s] = H(x[s]) - np.interp(bh, BGRID, nz[d]["curve"]); hinv[s] = H(bh)
    return hst, hinv

def denoise(e, design, dref):
    out = e.copy()
    for d in DES:
        s = design == d; out[s] = ((e[s] - dref[d][0]) / (1 - dref[d][0] - dref[d][1])).clip(0, 1)
    return out

def median_tare(sub, col):
    """A / median(A of the other arrays on the same slide), self excluded, >= MIN_REFS; NaN otherwise."""
    v, n = [], []
    for g, r in sub.iterrows():
        refs = sub.loc[(sub.slide == r.slide) & (sub.index != g), col].dropna(); n.append(len(refs))
        v.append(r[col] / refs.median() if len(refs) >= MIN_REFS and pd.notna(r[col]) else np.nan)
    return pd.Series(v, index=sub.index), pd.Series(n, index=sub.index)

def pooled_within_sd(v, g):
    d = pd.DataFrame({"v": v, "g": g}).dropna()
    ss = d.groupby("g")["v"].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum(); dof = (d.groupby("g").size() - 1).sum()
    return float(np.sqrt(ss / dof))

def titles_from_series_matrix(path):
    L = [l.rstrip("\n").split("\t") for l in gzip.open(path, "rt") if l.startswith(("!Sample_title", "!Sample_geo_accession"))]
    d = {l[0]: [x.strip('"') for x in l[1:]] for l in L}; return dict(zip(d["!Sample_geo_accession"], d["!Sample_title"]))

def run(a):
    sys.path.insert(0, a.chain); import conductor_v3 as C
    RM = os.path.join(a.chain, "Runtime Matrices", "Met_A_Floors")
    Bt = pd.read_parquet(os.path.join(a.data, "beta_subset.parquet"))
    design = pd.read_csv(os.path.join(a.data, "probe_design.csv"), index_col=0)["Infinium_Design_Type"].reindex(Bt.index)
    recs = pd.read_csv(os.path.join(a.data, "untared_records.csv")).set_index("gsm")
    jobs = pd.read_csv(os.path.join(a.data, "jobs.csv")).set_index("gsm")
    NS = pd.Index(json.load(open(os.path.join(RM, "noise_sites_EPIC_v1.json")))["sites"])
    BC = json.load(open(os.path.join(RM, "blood_composition_EPIC_v1.json")))
    FL = json.load(open(os.path.join(RM, "metA_floors_v1_3.json")))["platforms"]["EPIC"]["neutrophils"]
    REFS = [r.split("_")[0] for r in FL["refs"]]; S = pd.Index(BC["neutrophil_sites"]); FS = pd.Index(FL["sites"])
    P = {g: pd.Series(v, index=S, dtype="float64") for g, v in BC["profiles_at_neutrophil_sites"].items()}
    dS = design.reindex(S); dFS = design.reindex(FS); dN = design.reindex(NS)

    # true state of each fixed site: the purified GSE110554 arrays' mean beta (every group <= 0.03 or >= 0.97 by construction)
    pur = [g for g in Bt.columns if str(jobs.loc[g, "group"]).startswith("purified_")]
    mp = Bt.loc[NS, pur].mean(1); state = pd.Series(np.where(mp < 0.5, "U", "M"), index=NS)
    meta = dict(n_noise_sites=len(NS), n_U=int((state == "U").sum()), n_M=int((state == "M").sum()),
                noise_sites_by_state_design={f"{s}_{d}": int(((state == s) & (dN == d)).sum()) for s in "UM" for d in DES},
                identity_sites_by_design={str(k): int(v) for k, v in dS.value_counts().items()},
                floor_sites_same_as_blood_neutrophil_sites=bool(set(FS) == set(S)), n_purified_arrays_for_state=len(pur),
                purified_mean_beta_range_U=[float(mp[state == "U"].min()), float(mp[state == "U"].max())],
                purified_mean_beta_range_M=[float(mp[state == "M"].min()), float(mp[state == "M"].max())], reference_arrays=REFS)

    rows, curves, NZ = [], [], {}
    for g in Bt.columns:
        b = Bt[g]; nz = array_noise(b.reindex(NS), state, dN); NZ[g] = nz
        r = dict(gsm=g, series=jobs.loc[g, "series"], group=jobs.loc[g, "group"], specimen=jobs.loc[g, "specimen"],
                 slide=os.path.basename(str(jobs.loc[g, "grn"])).split("_")[1], N=float(H(b.reindex(NS).dropna()).mean()))
        for d in DES:
            r[f"dU_{d}"] = nz[d]["mU"]; r[f"dM_{d}"] = nz[d]["mM"]
            for bt in (0.10, 0.85): r[f"I_{d}_b{bt:.2f}"] = float(np.interp(bt, BGRID, nz[d]["curve"]))
        x = b.reindex(FS); hst, hinv = self_tared_H(x, dFS, nz); ok = x.notna()
        r["meanH_stI"] = float(self_tared_H(x, pd.Series("I", index=FS), nz)[0][ok].mean())
        r.update(n_floor_sites=int(ok.sum()), meanH=float(H(x[ok]).mean()), meanH_st=float(hst[ok].mean()), meanH_inv=float(hinv[ok].mean()))
        rows.append(r); curves += [dict(gsm=g, design=d, b=bb, I=v) for d in DES for bb, v in zip(BGRID[::10], NZ[g][d]["curve"][::10])]
    D = pd.DataFrame(rows).set_index("gsm")
    if a.inten:
        W = pd.read_parquet(a.inten); pz = pd.read_csv(os.path.join(a.data, "probe_design.csv"), index_col=0)
        cls = pd.Series(np.where(pz["Infinium_Design_Type"] == "II", "II", "I" + pz["Color_Channel"].fillna("").str[:3]), index=pz.index)
        stW = state.reindex(W.index); clW = cls.reindex(W.index); hB = {}
        for g in D.index:
            Mi, Ui = W[f"{g}|M"].astype(float), W[f"{g}|U"].astype(float); rM = pd.Series(np.nan, index=W.index); rU = rM.copy()
            for k in ("II", "IGrn", "IRed"):
                rm = float(Mi[(stW == "U") & (clW == k)].median()); ru = float(Ui[(stW == "M") & (clW == k)].median())
                rM[clW == k] = rm; rU[clW == k] = ru; D.loc[g, f"rM_{k}"] = rm; D.loc[g, f"rU_{k}"] = ru
            m_ = (Mi - rM).clip(lower=0); u_ = (Ui - rU).clip(lower=0); bB = m_ / (m_ + u_)
            hB[g] = pd.Series(H(bB), index=W.index).where(bB.notna())
            D.loc[g, "meanH_B"] = float(hB[g].reindex(FS).mean())
    D = D.join(recs[["A", "N", "f_neu"]].rename(columns={"A": "A_chain", "N": "N_chain", "f_neu": "f_neu_chain"}))

    R = D.loc[REFS]
    dref = {d: (float(R[f"dU_{d}"].mean()), float(R[f"dM_{d}"].mean())) for d in DES}
    floor_chain = float(FL["floor"]); floor_st = float(R["meanH_st"].mean()); floor_inv = float(R["meanH_inv"].mean())
    meta.update(floor_chain=floor_chain, floor_recomputed_here=float(R["meanH"].mean()), floor_st=floor_st, floor_inv=floor_inv,
                reference_offsets={d: {"mean_dU": dref[d][0], "mean_dM": dref[d][1]} for d in DES},
                purified_group_offsets=D[D.group.astype(str).str.startswith("purified_")].groupby("group")[[f"d{s}_{d}" for d in DES for s in "UM"]].mean().round(5).to_dict("index"))

    iso = D.specimen == "isolated neutrophils"
    D.loc[iso, "A_untared"] = D.loc[iso, "meanH"] / floor_chain
    D.loc[iso, "A_st"] = D.loc[iso, "meanH_st"] / floor_st
    D.loc[iso, "A_inv"] = D.loc[iso, "meanH_inv"] / floor_inv
    floor_stI = float(R["meanH_stI"].mean()); meta["floor_stI"] = floor_stI; D.loc[iso, "A_stI"] = D.loc[iso, "meanH_stI"] / floor_stI
    if a.inten:
        floor_B = float(R["meanH_B"].mean()); meta["floor_B"] = floor_B; D.loc[iso, "A_B"] = D.loc[iso, "meanH_B"] / floor_B
    for g in REFS:   # reference arrays read against the other five
        o = R.drop(g)
        D.loc[g, "A_untared_loo"] = D.loc[g, "meanH"] / o["meanH"].mean(); D.loc[g, "A_st_loo"] = D.loc[g, "meanH_st"] / o["meanH_st"].mean()
        if a.inten: D.loc[g, "A_B_loo"] = D.loc[g, "meanH_B"] / o["meanH_B"].mean()
    for g in D.index[D.specimen == "whole blood"]:
        b = Bt[g]; comp = C.stage_a_composition(b.dropna()); fr = comp["fractions"]
        x = b.reindex(S); e = sum(v * P[k] for k, v in fr.items() if k in P); ok = x.notna() & e.notna()
        es = denoise(e, dS, dref); hst, hinv = self_tared_H(x, dS, NZ[g])
        D.loc[g, "f_neu"] = fr.get("NEU"); D.loc[g, "n_id_sites"] = int(ok.sum())
        D.loc[g, "num_untared"] = float(H(x[ok]).mean()); D.loc[g, "num_st"] = float(hst[ok].mean())
        D.loc[g, "den_chain"] = float(H(e[ok]).mean()); D.loc[g, "den_st"] = float(H(es[ok]).mean())
        D.loc[g, "A_untared"] = D.loc[g, "num_untared"] / D.loc[g, "den_chain"]
        D.loc[g, "A_st"] = D.loc[g, "num_st"] / D.loc[g, "den_st"]; D.loc[g, "A_inv"] = float(hinv[ok].mean()) / D.loc[g, "den_st"]
        if a.inten: D.loc[g, "A_B"] = float(hB[g].reindex(S)[ok].mean()) / D.loc[g, "den_st"]
        esI = denoise(e, pd.Series("I", index=S), dref); hstI = self_tared_H(x, pd.Series("I", index=S), NZ[g])[0]
        D.loc[g, "A_stI"] = float(hstI[ok].mean() / H(esI[ok]).mean())
    need = int(np.ceil(0.9 * len(S))); D["n_id"] = np.where(D.specimen == "whole blood", D.get("n_id_sites"), D["n_floor_sites"])
    D["coverage_ok"] = D["n_id"] >= need
    for c_ in ("A_untared", "A_st", "A_inv", "A_B", "A_stI"):
        if c_ in D: D.loc[~D["coverage_ok"] & (D.specimen != "none"), c_] = np.nan
    D["A_untared_minus_chain"] = D["A_untared"] - D["A_chain"]

    t = {}
    if a.man: t.update(pd.read_csv(a.man).set_index("gsm")["title"].to_dict())
    for gse in ("GSE247193", "GSE247195"):
        p = os.path.join(a.sm, f"{gse}_series_matrix.txt.gz")
        if os.path.exists(p): t.update(titles_from_series_matrix(p))
    D["title"] = pd.Series(t).reindex(D.index)
    D["person"] = np.where(D.series == "GSE250556", D["title"].str.extract(r"(subject[A-D])")[0],
                   np.where(D.series == "GSE247193", "Neu30yo", np.where(D.series == "GSE247195", "Neu54yo", None)))
    D["timepoint"] = D["title"].str.extract(r"(Neu\d+yo_ZT\d+)")[0]
    for ser in ("GSE250556", "GSE247193", "GSE247195"):
        m = D.series == ser
        for col, new in (("A_untared", "A_median"), ("A_st", "A_st_median"), ("A_inv", "A_inv_median"), ("A_B", "A_B_median"), ("A_stI", "A_stI_median")):
            if col not in D: continue
            v, n = median_tare(D[m], col); D.loc[m, new] = v; D.loc[m, "n_refs_same_slide"] = n
    return D, meta, pd.DataFrame(curves)

def summarise(D):
    out = []
    cols = [("no tare", "A_untared"), ("median tare", "A_median"), ("self-tare", "A_st"), ("self-tare then median tare", "A_st_median"),
            ("mean inversion only (variant)", "A_inv"), ("mean inversion then median tare (variant)", "A_inv_median"),
            ("type I offsets for every site, variant I", "A_stI"), ("variant I then median tare", "A_stI_median"),
            ("per-probe background, variant B", "A_B"), ("variant B then median tare", "A_B_median")]
    for ser, grp in (("GSE250556", "person"), ("GSE247193+GSE247195", "timepoint")):
        m = D.series.isin(ser.split("+"))
        for lab, c in cols:
            if c not in D: continue
            v = D.loc[m, c]; ok = v.notna()
            out.append(dict(set=ser, reading=lab, n=int(ok.sum()), median=v.median(), min=v.min(), max=v.max(), sd_all=v.std(),
                            within_group_sd=pooled_within_sd(v, D.loc[m, grp]), within_group=grp,
                            in_normal=int(((v >= NORMAL[0]) & (v <= NORMAL[1])).sum()),
                            r_with_N=float(np.corrcoef(v[ok], D.loc[m & ok, "N"])[0, 1]) if ok.sum() > 2 else np.nan))
    return pd.DataFrame(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("data", "chain", "man", "sm", "out", "inten"): ap.add_argument("--" + k)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D, meta, curves = run(a)
    D.to_csv(os.path.join(a.out, "dev_selftare_01_readings.csv")); summarise(D).to_csv(os.path.join(a.out, "dev_selftare_01_summary.csv"), index=False)
    curves.to_csv(os.path.join(a.out, "dev_selftare_01_inflation_curves.csv"), index=False)
    json.dump(meta, open(os.path.join(a.out, "dev_selftare_01_meta.json"), "w"), indent=1, default=float)
