"""DEV-ATLAS-LAVAGE-01 (written 2026-10-10 before any array is downloaded): atlas v2 templates in the lung against cytospin differential counts.
Data: GSE133062 (Karolinska; 70 bronchoalveolar lavage EPIC arrays; per sample percent macrophages, lymphocytes, neutrophils, eosinophils).
Method = the 10-09 simulation (development/sims/atlas_sims_01.py), unchanged: the same 60 atlas blocks (atlas_blocks_60.json), the 11-cell LUNG
panel, the 8,000 loci with the largest spread of posterior means across the panel, non-negative least squares on posterior means.
Betas: chain Stage 1 on the GEO IDATs. Comparison on leukocytes only (cytospin counts exclude epithelium): fractions renormalised without the two
epithelium templates; macrophages = alveolar + interstitial macrophages + monocytes; lymphocytes = CD4 + CD8 + B + NK.
Bars (from DEV-ATLAS-COMMISSION-01 bar 1, applied to the lung templates; evidence for the templates, not blood commissioning):
  neutrophil fraction mean |error| vs the count <= 0.02, and every sample within 0.05. Lymphocytes and macrophages reported.
Run: HOME=<methylprep manifest dir> python3 score_lavage_01.py WORKDIR calib k n | score OUT.csv"""
import os, sys, json, re, urllib.request, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(MP, "chain")); sys.path.insert(0, os.path.join(MP, "../../development/sims"))
W = sys.argv[1]; os.makedirs(os.path.join(W, "betas"), exist_ok=True)
def samples():
    p = os.path.join(W, "samples.csv")
    if os.path.exists(p): return pd.read_csv(p)
    t = urllib.request.urlopen("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133062&targ=gsm&form=text&view=brief").read().decode().replace("\r", "")
    rows = []
    for b in t.split("^SAMPLE = ")[1:]:
        g = b.split("\n")[0].strip(); ch = dict(l.split("=", 1)[1].strip().split(": ", 1) for l in b.split("\n") if l.startswith("!Sample_characteristics") and ": " in l)
        sup = [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith("!Sample_supplementary_file") and "_Grn.idat" in l]
        rows.append(dict(gsm=g, grn=sup[0] if sup else None, **{k: ch.get(k) for k in ("percent_macrophages", "percent_lymphocytes", "percent_neutrophils", "percent_eosinophils", "smoking_status")}))
    d = pd.DataFrame(rows); d.to_csv(p, index=False); return d
S = samples()
if sys.argv[2] == "calib":
    from stage_1_idat_calibration import calibrate_idat_to_beta
    k, n = int(sys.argv[3]), int(sys.argv[4])
    for _, r in S.sort_values("gsm").reset_index(drop=True).iloc[k::n].iterrows():
        dst = os.path.join(W, "betas", r.gsm + ".parquet")
        if os.path.exists(dst) or not isinstance(r.grn, str): continue
        p = {}
        for ch in ("Grn", "Red"):
            u = r.grn.replace("ftp://", "https://").replace("_Grn.idat", f"_{ch}.idat"); p[ch] = os.path.join(W, u.rsplit("/", 1)[-1]); urllib.request.urlretrieve(u, p[ch])
        o = calibrate_idat_to_beta(p["Grn"], p["Red"], verbose=False); b = (o[0] if isinstance(o, tuple) else o).astype("float64"); b.index = b.index.astype(str)
        b.rename("beta").to_frame().to_parquet(dst); [os.remove(x) for x in p.values()]; print(r.gsm, "ok", flush=True)
    print("SHARD_DONE", k, flush=True)
else:
    from scipy.optimize import nnls
    import atlas_sims_01 as A
    ps = A.blocks(); D = np.concatenate([np.load(p, allow_pickle=True)["draws"].astype(np.float32) for p in ps], axis=2)
    cpg = np.concatenate([np.load(p, allow_pickle=True)["cpg"] for p in ps]); cells = np.load(ps[0], allow_pickle=True)["cells"].tolist()
    ix = [cells.index(c) for c in A.LUNG]; Mu = D[:, ix, :].mean(0); ok = np.isfinite(Mu).all(0); inf = np.argsort(-Mu[:, ok].std(0))[:8000]
    Mu = Mu[:, ok][:, inf]; loci = cpg[ok][inf]
    num = lambda x: float(x) / 100 if str(x).strip().lower() not in ("na", "nan", "", "none") else np.nan   # missing counts stay missing (2026-10-10, before scoring)
    rows = []
    for _, r in S.iterrows():
        f = os.path.join(W, "betas", r.gsm + ".parquet")
        if not os.path.exists(f): continue
        b = pd.read_parquet(f).beta.reindex(loci).values; m = np.isfinite(b); w, _ = nnls(Mu[:, m].T, b[m]); w = w / w.sum()
        leu = w[:9] / w[:9].sum()
        rows.append(dict(gsm=r.gsm, smoking=r.smoking_status, loci=int(m.sum()), epithelium=float(w[9] + w[10]),
                         neu_atlas=leu[7], neu_count=num(r.percent_neutrophils), lym_atlas=leu[3:7].sum(), lym_count=num(r.percent_lymphocytes),
                         mac_atlas=leu[0:3].sum(), mac_count=num(r.percent_macrophages), eos_atlas=leu[8], eos_count=num(r.percent_eosinophils)))
    R = pd.DataFrame(rows); R.to_csv(sys.argv[3], index=False)
    for c in ("neu", "lym", "mac", "eos"):
        Q = R[R[f"{c}_count"].notna()]; e = Q[f"{c}_atlas"] - Q[f"{c}_count"]; print(f"{c}: n {len(Q)} | mean |error| {e.abs().mean():.4f} | bias {e.mean():+.4f} | within 0.05 {int((e.abs() <= 0.05).sum())}/{len(Q)} | r {np.corrcoef(Q[f'{c}_atlas'], Q[f'{c}_count'])[0, 1]:.3f}")
    e = (R.neu_atlas - R.neu_count).dropna().abs(); print(f"n with a neutrophil count {len(e)}"); print(f"BARS neutrophils: mean |error| {e.mean():.4f} (<= 0.02: {e.mean() <= 0.02}); every sample within 0.05: {bool((e <= 0.05).all())}; median loci {int(R.loci.median())}")
