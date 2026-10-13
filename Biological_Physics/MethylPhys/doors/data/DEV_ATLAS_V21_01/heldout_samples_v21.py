"""Atlas v2.1 held-out array list (readiness B2): one row per array of every held-out and truth-set roster row on a platform Stage 1
reads (450K, EPIC v1). Sequencing rows are a separate alignment job (B3). IDAT names from the committed GEO snapshots (geo_snapshots/)
and ArrayExpress SDRFs. These arrays are downloaded, calibrated and genotyped (the same-donor check) but NOT scored until the atlas they
judge is built and its commissioning bars are sealed (L15). Writes atlas_v21_heldout_samples.csv. Usage: python3 heldout_samples_v21.py"""
import os, re, gzip, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(os.path.join(HERE, "atlas_v21_roster.csv")); H = R[R.role.astype(str).str.startswith("held-out") | (R.role == "truth set")]
seq = H.platform.astype(str).str.contains(r"seq|bisulfite|RRBS|WGBS|nanopore|EPIC v2", case=False)
def snap(g):
    p = os.path.join(HERE, "geo_snapshots", f"{g}.txt.gz"); out = {}
    if not os.path.exists(p): return out
    for b in gzip.open(p, "rt", encoding="utf-8").read().replace("\r", "").split("^SAMPLE = ")[1:]:
        gsm = b.split("\n")[0].strip(); get = lambda k: [l.split("=", 1)[1].strip() for l in b.split("\n") if l.startswith(k)]
        out[gsm] = ";".join(sorted(f.split("/")[-1] for f in get("!Sample_supplementary_file") if ".idat" in f.lower()))
    return out
def sdrf(acc):
    p = os.path.join(HERE, f"{acc}.sdrf.txt"); out = {}
    if not os.path.exists(p): return out
    T = pd.read_csv(p, sep="\t"); fc = [c for c in T.columns if c.startswith("Array Data File")]
    for _, r in T.iterrows():
        k = str(r.iloc[0]); out[k] = ";".join(sorted(set(out.get(k, "").split(";")) | {str(r[c]) for c in fc if str(r[c]).endswith(".idat")} - {""}))
    return out
rows, missing = [], []
for _, r in H[~seq].iterrows():
    meta = sdrf(r.series) if str(r.series).startswith("E-MTAB") else snap(r.series)
    for x in [x for x in str(r.samples).replace(" ", "").split(";") if x and x != "nan"]:
        f = meta.get(x, "")
        rows.append(dict(sample=x, series=r.series, cell_type=r.cell_type, role=r.role, platform=r.platform, idat_files=f))
        if not f: missing.append((r.series, x))
T = pd.DataFrame(rows).drop_duplicates("sample").sort_values(["series", "sample"])
T.to_csv(os.path.join(HERE, "atlas_v21_heldout_samples.csv"), index=False)
print("held-out arrays", len(T), "| series", T.series.nunique(), "| without IDAT names", len(missing), sorted({m[0] for m in missing})[:10])
