#!/usr/bin/env python3
"""First v2 chain run (PREVIEW): 24 GSE87571 whole-blood arrays fetched from GEO through our Stage 1, plus the GSE63409 AML arrays on the box."""
import os, re, gzip, glob, json, urllib.request, multiprocessing as mp, sys, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
W = "/home/ubuntu/data/v2run01"; os.makedirs(f"{W}/idats", exist_ok=True); os.makedirs(f"{W}/shards", exist_ok=True)
U = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/"
t = gzip.decompress(urllib.request.urlopen(U + "matrix/GSE87571_series_matrix.txt.gz", timeout=300).read()).decode("utf-8", "replace")
row = lambda k: [x.strip('"') for x in re.search(rf"^!{k}\t(.+)$", t, re.M).group(1).split("\t")]
gsm = row("Sample_geo_accession"); chs = [[x.strip('"') for x in l.split("\t")[1:]] for l in re.findall(r"^!Sample_characteristics_ch1\t.+$", t, re.M)]
age = []
for i in range(len(gsm)):
    v = np.nan
    for r in chs:
        m = re.match(r"\s*age[^:]*:\s*([0-9.]+)", r[i], re.I)
        if m: v = float(m.group(1)); break
    age.append(v)
M = pd.DataFrame(dict(gsm=gsm, age=age)).dropna().sort_values("age")
pick = M.iloc[np.linspace(0, len(M) - 1, 24).astype(int)].copy()
files = {}
for g in pick.gsm:
    d = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:-3]}nnn/{g}/suppl/"
    h = urllib.request.urlopen(d, timeout=120).read().decode("utf-8", "replace")
    files[g] = [d + x for x in sorted(set(re.findall(r'href="([^"]+\.idat(?:\.gz)?)"', h)))]
pick = pick[pick.gsm.map(lambda g: len(files[g]) == 2)]
print("GSE87571 samples", len(M), "| picked with 2 IDATs", len(pick), "| ages", pick.age.min(), "-", pick.age.max(), flush=True)
def calib(g):
    sh = f"{W}/shards/{g}.parquet"
    if os.path.exists(sh): return g, "exists", None
    loc = []
    for u in files[g]:
        p = f"{W}/idats/{os.path.basename(u)}"
        if not os.path.exists(p): urllib.request.urlretrieve(u, p)
        loc.append(p)
    grn = [p for p in loc if "_Grn" in p][0]; red = [p for p in loc if "_Red" in p][0]
    from stage_1_idat_calibration import calibrate_idat_to_beta
    beta, meta = calibrate_idat_to_beta(grn, red, verbose=False); beta = beta.iloc[:, 0] if hasattr(beta, "columns") else beta
    det = meta.get("detection") or {}; cr = det.get("n_detected", 0) / max(det.get("n_probes", 1), 1)
    beta.to_frame(g).to_parquet(sh); return g, "ok", cr
with mp.get_context("fork").Pool(8) as pool: C = pool.map(calib, pick.gsm.tolist())
cr = {g: c for g, _, c in C}; print("calibrated", sum(1 for _, s, _ in C if s in ("ok", "exists")), flush=True)
from methylphys_v2 import ReaderV2
P = "/home/ubuntu/data/IAMAtlas_v2.parquet"
if not os.path.exists(P): urllib.request.urlretrieve(json.load(open("atlas_url.json"))["atlas_get"], P)
R = ReaderV2(P, "iamatlas_v2_identity_loci_v1_1.json"); print("reader ready", json.dumps(R.D.meta), flush=True)
specs = [("GSE87571", g, f"{W}/shards/{g}.parquet", float(a)) for g, a in zip(pick.gsm, pick.age)]
HM = pd.read_csv("hsc_manifest.csv"); aml = HM[HM.subject_status != "normal"]
for r in aml.itertuples():
    p = glob.glob(f"/home/ubuntu/data/atlas_sources/hsc_gse63409/shards/{r.gsm}*.parquet")
    if p: specs.append(("GSE63409_AML:" + r.label, r.gsm, p[0], np.nan))
print("specimens", len(specs), flush=True)
out = []
for study, g, p, a in specs:
    b = pd.read_parquet(p).iloc[:, 0]; o = R.read(b)
    rec = dict(study=study, gsm=g, age=a, call_rate=cr.get(g), residual_mae=o["residual_mae"], cells=o["cells"])
    out.append(rec)
    top = sorted(o["cells"].items(), key=lambda x: -x[1].get("fraction", 0))[:4]
    print(study, g, a, "| " + " | ".join(f"{c} f={v.get('fraction')} A={v.get('A')} {v.get('tier','')}" for c, v in top), flush=True)
json.dump(out, open("v2run01.json", "w"), indent=1, default=float)
