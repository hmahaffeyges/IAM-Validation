#!/usr/bin/env python3
"""band_v2 TEST — pooled two-lab band + map, tested on GSE125105 controls (Munich, n=210). NOTHING is fit here.
Runs in the `methylprep` environment (Stage 1 needs it). Writes results/phase1/*.json and prints only the
pass-condition table. Every value tested against the sealed prereg; nothing tuned after the data is seen."""
import os, sys, json, re, tarfile, gzip, time, glob, hashlib, io
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(ROOT, "iamrepo/Biological_Physics/CPG_Engine")
sys.path.insert(0, E); sys.path.insert(0, os.path.join(E, "Walther_iam_deconvolver")); sys.path.insert(0, os.path.join(ROOT, "stage1"))
os.environ.setdefault("HOME", os.path.join(ROOT, "stage1/mp_home"))
TAR = os.path.join(ROOT, "idats/GSE125105/GSE125105_RAW.tar"); OUT = os.path.join(ROOT, "results/band_v2_test"); os.makedirs(OUT, exist_ok=True)
IDAT_DIR = os.path.join(ROOT, "idats/GSE125105/idats"); os.makedirs(IDAT_DIR, exist_ok=True)
CLASSES = ['stem_pluri','stem_adult','stromal','progenitor','cycling','secretory','terminal','immune']
EPI = ["cycling","secretory","terminal","stromal"]

def H(b): b = np.clip(np.asarray(b, float), 1e-12, 1-1e-12); return -b*np.log2(b) - (1-b)*np.log2(1-b)

def log(*a): print(*a, flush=True)

# ---------- 0. unpack + metadata ----------
def unpack():
    pass  # IDATs fetched per-sample by tools/geo_fetch_idats.py (controls only)
    grn = sorted(glob.glob(os.path.join(IDAT_DIR, "*_Grn.idat*")))
    pairs = {}
    for g in grn:
        gsm = os.path.basename(g).split("_")[0]; r = g.replace("_Grn", "_Red")
        if os.path.exists(r): pairs[gsm] = (g, r)
    return pairs

def metadata():
    """age + sex from the series matrix header (fetched by range, no betas read)."""
    import urllib.request, zlib
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE125nnn/GSE125105/matrix/GSE125105_series_matrix.txt.gz"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-600000"}), timeout=120).read()
    txt = zlib.decompressobj(16+zlib.MAX_WBITS).decompress(raw).decode("utf-8", "replace")
    gsms = None; age = None; sex = None; dz = None; smk = None
    for l in txt.split("\n"):
        if l.startswith("!Sample_geo_accession"): gsms = [v.strip('"') for v in l.split("\t")[1:]]
        if l.startswith("!Sample_characteristics_ch1"):
            vals = [v.strip('"') for v in l.split("\t")[1:]]
            if vals and vals[0].lower().startswith("age"): age = [float(re.sub(r"[^\d.]", "", v) or "nan") for v in vals]
            if vals and vals[0].lower().startswith("gender") or (vals and vals[0].lower().startswith("sex")): sex = [v.split(":")[-1].strip() for v in vals]
            if vals and (vals[0].lower().startswith("disease state") or vals[0].lower().startswith("diagnosis")): dz = [v.split(":")[-1].strip() for v in vals]
            if vals and vals[0].lower().startswith("smoking"): smk = [v.split(":")[-1].strip() for v in vals]
    return pd.DataFrame({"gsm": gsms, "age": age, "sex": sex, "disease": dz, "smoking": smk}).set_index("gsm")

# ---------- 1. Stage 0 + Stage 1 ----------
def calibrate_all(pairs, workers=6):
    """Stage 1 in parallel via independent subprocesses (the sandbox forbids multiprocessing semaphores).
    Same per-sample code path as PROC-CAL-01; only the scheduling differs. 6 workers on 8 cores."""
    import subprocess
    cache = os.path.join(OUT, "betas_GSE125105_controls.pkl")
    if os.path.exists(cache): return pd.read_pickle(cache)
    items = list(pairs.items()); slices = [items[k::workers] for k in range(workers)]; procs = []; t = time.time()
    for k, sl in enumerate(slices):
        jf = os.path.join(OUT, f"jobs_{k}.json"); json.dump([(g, p[0], p[1]) for g, p in sl], open(jf, "w"))
        procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "stage1_worker.py"), jf, os.path.join(OUT, f"part_{k}.pkl")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs: p.wait()
    parts = [pd.read_pickle(os.path.join(OUT, f"part_{k}.pkl")) for k in range(workers) if os.path.exists(os.path.join(OUT, f"part_{k}.pkl"))]
    fails = [f for k in range(workers) if os.path.exists(os.path.join(OUT, f"part_{k}.pkl.fails.json")) for f in json.load(open(os.path.join(OUT, f"part_{k}.pkl.fails.json")))]
    df = pd.concat(parts, axis=1); df.to_pickle(cache); json.dump(fails, open(os.path.join(OUT, "stage1_fails.json"), "w"))
    log(f"  stage1 {df.shape[1]}/{len(pairs)} in {time.time()-t:.0f}s, fails {len(fails)}"); return df

# ---------- 2. deconvolution + gauge ----------
def score(df, ident):
    from walther_iam_deconvolver import WaltherIAMDeconvolver
    W = WaltherIAMDeconvolver(os.path.join(ROOT, "atlasrun/IAMAtlas.csv"),
                              celltype_class_map=os.path.join(ROOT, "iamrepo/Biological_Physics/IAM_Atlas/IAMAtlasREBUILD_celltype_to_class.json"), verbose=False)
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
    maps = json.load(open(os.path.join(E, "Runtime Matrices/A_Scoring_Module/beta_scale_maps_v1.json")))["maps"]["stage1_noob_450K"]
    slope, icpt = maps["slope"], maps["intercept"]
    b2 = json.load(open(os.path.join(E, "Runtime Matrices/A_Scoring_Module/identity_band_v2_PROVISIONAL.json")))["immune"]
    mband = {e["decade"]: (e["p10"], e["p90"], e["n"], e["p50"]) for e in b2 if e.get("p10") is not None}   # band_v2, frozen
    meta = metadata(); ctrl = set(meta.index[meta.disease.astype(str).str.lower() == "control"]); log(f"metadata {len(meta)} | controls {len(ctrl)}")
    pairs = {g: p for g, p in unpack().items() if g in ctrl}; log(f"control IDAT pairs: {len(pairs)}  (RA arrays NOT opened)")
    df = calibrate_all(pairs); log(f"Stage 1 betas: {df.shape}")
    S, IMM = score(df, ident); S = S.join(meta, how="left")
    # apply the FROZEN map on the gauge path
    loci = [c for c in IMM if c in df.index]
    S["A_immune_mapped"] = [float(H((np.nanmean(df[g].loc[loci].values) - icpt) / slope) / ident["immune"]["H_min"]) for g in S.index]
    S.to_csv(os.path.join(OUT, "per_sample.csv"))
    p1 = len(df.columns) / len(pairs)
    ok = (S.immune_frac + S.joint_frac >= 0.85) & (S.epi_frac <= 0.02); p2 = float(ok.mean()); Sb = S[ok & S.age.notna()]
    def bnd(age):
        for k, v in mband.items():
            lo, hi = map(int, k.split("-"))
            if lo <= age <= hi: return k, v[0], v[1], v[2]
        return None
    med = float(Sb.A_immune_mapped.median()); p3 = abs(med - 1.0) <= 0.02
    inb = []; per_dec = {}
    for g, r in Sb.iterrows():
        b = bnd(r.age)
        if b and b[3] >= 30: inb.append(b[1] <= r.A_immune_mapped <= b[2]); per_dec.setdefault(b[0], []).append(r.A_immune_mapped)
    p4 = float(np.mean(inb)) if inb else None
    # P5: GSE125105 control median per decade minus the pooled band p50
    p5 = {k: dict(n125105=len(v), med125105=round(float(np.median(v)), 4), pooled_p50=mband[k][3], offset=round(float(np.median(v) - mband[k][3]), 4)) for k, v in per_dec.items() if k in mband}
    gse87571_median_affine = None
    # P6 the four TEST_DATA arrays vs their own cohort
    import pickle; cache = pickle.load(open(os.path.join(ROOT, "testdata/10_TEST_DATA/betas_cache.pkl"), "rb"))
    lo5, hi95 = float(Sb.A_immune_mapped.quantile(.05)), float(Sb.A_immune_mapped.quantile(.95)); p6 = {}
    for gsm in ("GSM1051525", "GSM1051526", "GSM1051533", "GSM1051534"):
        v = cache[gsm]; d = pd.Series({k: float(x) for k, x in (v.items() if hasattr(v, "items") else enumerate(v)) if x == x}).reindex(loci).dropna()
        A = float(H((d.mean() - icpt) / slope) / ident["immune"]["H_min"]); p6[gsm] = (round(A, 4), bool(lo5 <= A <= hi95), gsm in df.columns)
    # N-random
    rng = np.random.default_rng(1); RND = list(rng.choice(df.index.values, size=len(loci), replace=False))   # SIZE-matched only (band_v2 PREREG)
    Arnd = pd.Series([float(H((np.nanmean(df[g].loc[RND].values) - icpt) / slope) / ident["immune"]["H_min"]) for g in Sb.index], index=Sb.index)
    nr = [ (bnd(a)[1] <= x <= bnd(a)[2]) for a, x in zip(Sb.age, Arnd) if bnd(a) and bnd(a)[3] >= 30 ]; nrand = float(np.mean(nr)) if nr else None
    nrand_level = float(Arnd.median()); nrand_pass = abs(nrand_level - 1.0) > 0.05
    # N-sex, N-smoke, N-plate
    def d_(a, b): a, b = np.asarray(a, float), np.asarray(b, float); sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2)); return float(abs(a.mean()-b.mean())/sp)
    nsex = {}
    for k in per_dec:
        lo, hi = map(int, k.split("-")); m = (Sb.age >= lo) & (Sb.age <= hi)
        if m.sum() >= 30 and Sb.loc[m, "sex"].nunique() == 2:
            g_ = [Sb.loc[m & (Sb.sex == sx), "A_immune_mapped"].values for sx in sorted(Sb.loc[m, "sex"].unique())]
            if min(len(x) for x in g_) >= 5: nsex[k] = round(d_(*g_), 3)
    nsmoke = {k: (int(len(v)), round(float(v.median()), 4)) for k, v in Sb.groupby("smoking").A_immune_mapped if len(v) >= 10}
    chips = pd.Series({g: os.path.basename(pairs[g][0]).split("_")[1] if len(os.path.basename(pairs[g][0]).split("_")) > 2 else None for g in Sb.index})
    nplate = None
    if chips.notna().sum() > 50 and chips.nunique() > 3:
        from scipy import stats; grp = [Sb.loc[chips == c, "A_immune_mapped"].values for c in chips.dropna().unique() if (chips == c).sum() >= 4]
        if len(grp) >= 3: F = stats.f_oneway(*grp); nplate = dict(n_chips=len(grp), F=round(float(F.statistic), 3), p=float(F.pvalue))
    res = dict(P1_coverage=round(p1, 4), P2_presence=round(p2, 4), n_after_gate=int(len(Sb)), P3_median_mapped_A=round(med, 4), P3_pass=bool(p3),
               P4_frac_in_GSE87571_band=None if p4 is None else round(p4, 4), P4_pass=bool(p4 is not None and p4 >= 0.80), P5_lab_layer=p5, GSE87571_median_affine=gse87571_median_affine,
               P6_test_arrays=p6, P6_cohort_p5_p95=(round(lo5, 4), round(hi95, 4)), N_random_in_band=None if nrand is None else round(nrand, 4), N_random_level_median=round(nrand_level, 4), N_random_pass_level=bool(nrand_pass), N_sex_d=nsex, N_smoke=nsmoke, N_plate=nplate,
               unmapped_median_A=round(float(Sb.A_immune.median()), 4))
    json.dump(res, open(os.path.join(OUT, "band_v2_test_results.json"), "w"), indent=1, default=str)
    log(json.dumps(res, indent=1, default=str))

if __name__ == "__main__": main()
