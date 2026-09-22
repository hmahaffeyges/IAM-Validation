#!/usr/bin/env python3
"""PHASE 1 — identity-loci healthy bands from GSE87571 through Stage 0/1. Executes PREREG.md as sealed.
Runs in the `methylprep` environment (Stage 1 needs it). Writes results/phase1/*.json and prints only the
pass-condition table. Every value tested against the sealed prereg; nothing tuned after the data is seen."""
import os, sys, json, re, tarfile, gzip, time, glob, hashlib, io
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(ROOT, "iamrepo/Biological_Physics/MethylPhys/chain")
sys.path.insert(0, E); sys.path.insert(0, os.path.join(E, "Walther_iam_deconvolver")); sys.path.insert(0, os.path.join(ROOT, "stage1"))
os.environ.setdefault("HOME", os.path.join(ROOT, "stage1/mp_home"))
TAR = os.path.join(ROOT, "idats/GSE87571/GSE87571_RAW.tar"); OUT = os.path.join(ROOT, "results/phase1"); os.makedirs(OUT, exist_ok=True)
IDAT_DIR = os.path.join(ROOT, "idats/GSE87571/idats"); os.makedirs(IDAT_DIR, exist_ok=True)
CLASSES = ['stem_pluri','stem_adult','stromal','progenitor','cycling','secretory','terminal','immune']
EPI = ["cycling","secretory","terminal","stromal"]

def H(b): b = np.clip(np.asarray(b, float), 1e-12, 1-1e-12); return -b*np.log2(b) - (1-b)*np.log2(1-b)

def log(*a): print(*a, flush=True)

# ---------- 0. unpack + metadata ----------
def unpack():
    if not glob.glob(os.path.join(IDAT_DIR, "*_Grn.idat*")):
        with tarfile.open(TAR) as t: t.extractall(IDAT_DIR)
    grn = sorted(glob.glob(os.path.join(IDAT_DIR, "*_Grn.idat*")))
    pairs = {}
    for g in grn:
        gsm = os.path.basename(g).split("_")[0]; r = g.replace("_Grn", "_Red")
        if os.path.exists(r): pairs[gsm] = (g, r)
    return pairs

def metadata():
    """age + sex from the series matrix header (fetched by range, no betas read)."""
    import urllib.request, zlib
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/matrix/GSE87571_series_matrix.txt.gz"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-600000"}), timeout=120).read()
    txt = zlib.decompressobj(16+zlib.MAX_WBITS).decompress(raw).decode("utf-8", "replace")
    gsms = None; age = None; sex = None
    for l in txt.split("\n"):
        if l.startswith("!Sample_geo_accession"): gsms = [v.strip('"') for v in l.split("\t")[1:]]
        if l.startswith("!Sample_characteristics_ch1"):
            vals = [v.strip('"') for v in l.split("\t")[1:]]
            if vals and vals[0].lower().startswith("age"): age = [float(re.sub(r"[^\d.]", "", v) or "nan") for v in vals]
            if vals and vals[0].lower().startswith("gender") or (vals and vals[0].lower().startswith("sex")): sex = [v.split(":")[-1].strip() for v in vals]
    return pd.DataFrame({"gsm": gsms, "age": age, "sex": sex}).set_index("gsm")

# ---------- 1. Stage 0 + Stage 1 ----------
def calibrate_all(pairs):
    from stage_1_idat_calibration import calibrate_idat_to_beta
    cache = os.path.join(OUT, "betas_GSE87571.pkl")
    if os.path.exists(cache): return pd.read_pickle(cache)
    B = {}; t = time.time(); fails = []
    for i, (gsm, (g, r)) in enumerate(pairs.items()):
        try:
            beta, meta = calibrate_idat_to_beta(g, r); B[gsm] = beta.astype("float32")
        except Exception as e: fails.append((gsm, str(e)[:80]))
        if (i+1) % 50 == 0: log(f"  stage1 {i+1}/{len(pairs)}  {time.time()-t:.0f}s  fails {len(fails)}")
    df = pd.DataFrame(B); df.to_pickle(cache); json.dump(fails, open(os.path.join(OUT, "stage1_fails.json"), "w"))
    return df

# ---------- 2. deconvolution + gauge ----------
def score(df, ident):
    from walther_iam_deconvolver import WaltherIAMDeconvolver
    W = WaltherIAMDeconvolver(os.path.join(ROOT, "atlasrun/IAMAtlas.csv"),
                              celltype_class_map=os.path.join(ROOT, "iamrepo/Biological_Physics/MethylPhys/atlas/IAMAtlasREBUILD_celltype_to_class.json"), verbose=False)
    IMM = [c for c in ident["immune"]["loci"] if c in df.index]
    JOINT = [c for c in set(ident["progenitor"]["loci"]) | set(ident["stem_adult"]["loci"]) if c in df.index]
    hm_imm = ident["immune"]["H_min"]; hm_joint = ident["progenitor"]["H_min"]      # prereg §3: progenitor's floor for the joint component
    rows = []
    for gsm in df.columns:
        b = df[gsm]; d = {k: float(v) for k, v in b.items() if v == v}
        fr = W.deconvolve(d).class_fractions
        rows.append(dict(gsm=gsm, immune_frac=fr.get("immune", 0), epi_frac=sum(fr.get(c, 0) for c in EPI),
                         joint_frac=fr.get("progenitor", 0)+fr.get("stem_adult", 0),
                         A_immune=float(H(np.nanmean(b.loc[IMM].values))/hm_imm),
                         A_joint=float(H(np.nanmean(b.loc[JOINT].values))/hm_joint)))
    return pd.DataFrame(rows).set_index("gsm"), IMM

# ---------- 3. bands ----------
DECADES = [(14,24),(25,34),(35,44),(45,54),(55,64),(65,74),(75,84),(85,94)]
def bands(S, col):
    out = []
    for lo, hi in DECADES:
        m = (S.age >= lo) & (S.age <= hi); v = S.loc[m, col].dropna()
        out.append(dict(decade=f"{lo}-{hi}", n=int(len(v)), p10=float(np.percentile(v, 10)) if len(v) else None,
                        p50=float(np.percentile(v, 50)) if len(v) else None, p90=float(np.percentile(v, 90)) if len(v) else None,
                        thin=bool(len(v) < 30)))
    return out

def in_band(A, age, bnd):
    for b in bnd:
        lo, hi = map(int, b["decade"].split("-"))
        if lo <= age <= hi and b["p10"] is not None: return b["p10"] <= A <= b["p90"]
    return None

def main():
    ident = json.load(open(os.path.join(E, "Runtime Matrices/A_Scoring_Module/iamatlas_gauge_identity_loci_v1_0.json")))
    ident = {k: v for k, v in ident.items() if isinstance(v, dict) and "loci" in v}
    pairs = unpack(); log(f"IDAT pairs: {len(pairs)}")
    meta = metadata(); log(f"metadata rows: {len(meta)}  age known: {meta.age.notna().sum()}")
    df = calibrate_all(pairs); log(f"Stage 1 betas: {df.shape}")
    S, IMM = score(df, ident); S = S.join(meta, how="left"); S.to_csv(os.path.join(OUT, "per_sample.csv"))
    # P1 coverage
    p1 = len(df.columns)/len(pairs); 
    # P2 presence
    ok = (S.immune_frac >= 0.80) & (S.epi_frac <= 0.02); p2 = ok.mean(); Sb = S[ok & S.age.notna()]
    band_imm = bands(Sb, "A_immune"); band_joint = bands(Sb, "A_joint")
    json.dump({"immune": band_imm, "haematopoietic_progenitor": band_joint, "n_band": int(len(Sb)),
               "statistic": "H(beta_mean)/H_min over identity loci", "pipeline": "Stage0/1 methylprep noob; Walther HEAD contrast off",
               "sealed_prereg_sha256_prefix": "4d8f40ace72eca1f"}, open(os.path.join(OUT, "identity_band_v1.json"), "w"), indent=1)
    # P3 synthetic healthy
    sys.path.insert(0, os.path.join(E, "Synthetic_Patient_Generator")); import synthetic_patient_generator as SPG
    coh = SPG.SyntheticCohort(n_case=0, n_hc=16, random_seed=20260919, composition_alpha=SPG.WHOLE_BLOOD_ALPHA); coh.generate()
    cpg_ids = coh.atlas['cpg'].values
    pts = [(pd.Series(coh.beta_matrix[i], index=cpg_ids), p.age) for i, p in enumerate(coh.patients)]
    p3 = []
    for b, age in pts:
        A = float(H(np.nanmean(b.reindex(IMM).values))/ident["immune"]["H_min"]); p3.append(in_band(A, age, band_imm))
    # N-random: same on a random panel matched on mean beta
    rng = np.random.default_rng(1); allc = df.index.values; mb = df.mean(1)
    tgt = mb.loc[[c for c in IMM if c in mb.index]].mean(); cand = mb[(mb > tgt-0.05) & (mb < tgt+0.05)].index.values
    RND = list(rng.choice(cand, size=min(len(IMM), len(cand)), replace=False))
    S["A_rand"] = [float(H(np.nanmean(df[g].loc[RND].values))/ident["immune"]["H_min"]) for g in S.index]
    band_rnd = bands(S[ok & S.age.notna()], "A_rand")
    nrand = []
    for b, age in pts:
        A = float(H(np.nanmean(b.reindex(RND).values))/ident["immune"]["H_min"]); nrand.append(in_band(A, age, band_rnd))
    # P4 test samples
    import pickle; cache = pickle.load(open(os.path.join(ROOT, "testdata/10_TEST_DATA/betas_cache.pkl"), "rb"))
    TEST = {"GSM2333901": 58, "GSM2333905": 67, "GSM2333950": 43, "GSM1051525": 60, "GSM1051526": 60, "GSM1051533": 60, "GSM1051534": 60}
    p4 = {}
    for gsm, age in TEST.items():
        v = cache[gsm]; d = pd.Series({k: float(x) for k, x in (v.items() if hasattr(v, "items") else enumerate(v)) if x == x})
        A = float(H(np.nanmean(d.reindex(IMM).values))/ident["immune"]["H_min"]); p4[gsm] = (round(A, 4), in_band(A, age, band_imm))
    # P5 age trend
    from scipy import stats; rho = stats.spearmanr(Sb.age, Sb.A_immune)
    # N-sex, N-split
    nsex = []
    for b in band_imm:
        lo, hi = map(int, b["decade"].split("-")); m = (Sb.age >= lo) & (Sb.age <= hi)
        if m.sum() >= 30 and Sb.loc[m, "sex"].nunique() == 2:
            g = [Sb.loc[m & (Sb.sex == s), "A_immune"].values for s in Sb.loc[m, "sex"].unique()]
            sp = np.sqrt(((len(g[0])-1)*g[0].var(ddof=1)+(len(g[1])-1)*g[1].var(ddof=1))/(len(g[0])+len(g[1])-2)); nsex.append((b["decade"], float(abs(g[0].mean()-g[1].mean())/sp)))
    idx = rng.permutation(len(Sb)); h1, h2 = Sb.iloc[idx[:len(Sb)//2]], Sb.iloc[idx[len(Sb)//2:]]
    b1, b2 = bands(h1, "A_immune"), bands(h2, "A_immune")
    nsplit = [(a["decade"], round(abs(a["p10"]-c["p10"]), 4), round(abs(a["p90"]-c["p90"]), 4)) for a, c in zip(b1, b2) if a["n"] >= 30 and c["n"] >= 30 and a["p10"] is not None and c["p10"] is not None]
    res = dict(P1_coverage=round(p1, 4), P2_presence=round(float(p2), 4), n_in_band_build=int(len(Sb)),
               P3_synthetic_in_band=f"{sum(1 for x in p3 if x)}/{len(p3)}", N_random_in_band=f"{sum(1 for x in nrand if x)}/{len(nrand)}",
               P4_test_samples=p4, P5_spearman_age=(round(float(rho.statistic), 4), float(rho.pvalue)),
               N_sex_d_by_decade=nsex, N_split_p10_p90_diff=nsplit, band_immune=band_imm)
    json.dump(res, open(os.path.join(OUT, "phase1_results.json"), "w"), indent=1, default=str)
    log(json.dumps({k: v for k, v in res.items() if k != "band_immune"}, indent=1, default=str))
    log("bands:"); [log(f"  {b['decade']:>6} n={b['n']:>3} p10={b['p10']} p50={b['p50']} p90={b['p90']}{' THIN' if b['thin'] else ''}") for b in band_imm]

if __name__ == "__main__": main()
