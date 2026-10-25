"""stage_provisional.py - provisional reference columns beside IAM-Atlas v2.1 (DEVELOPMENT; written 2026-10-24, plan 40.18).

A cell type the atlas does not hold (first: megakaryocytes, a large source of plasma DNA) cannot be named by the IAM-Deconvolver; its DNA is
absorbed into the cells that look most like it (absorption test 1: lung interstitial macrophages, erythrocyte progenitors, MEP, eosinophils).
Until purified data good enough for a full atlas fit exist, such a cell gets a PROVISIONAL column, kept apart from the atlas and never written
into it: the deconvolution is run a second time with the column added, and every share from that run is flagged. The atlas's own reading is
unchanged.

The column, from pooled sequencing reads (m methylated of n at each atlas site), nothing fitted:
  level   x = g^-1(m / n), g IAM-Atlas v2.1's smooth transfer, inverted exactly as the atlas converted its sequencing readings (mode v21inv);
  spread  sd^2 = q(1 - q) / (n k^2) + d^2 + lab^2 + FLOOR, q = (m + 1/2) / (n + 1) (so a site read 3 of 3 is not given zero spread), k the
          transfer's slope at x floored at 0.151 (the atlas's rule), d the median donor spread of the 79 atlas cells at that site (one donor
          cannot give its own), lab and FLOOR the atlas reference's own terms (stage_atlas_commission).
Sites with no reads stay empty; a site empty in any column is not used as a marker (stage_deconv_v21.markers)."""
import os, json, gzip, hashlib, urllib.request, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.join(HERE, "..")
ATLAS = os.path.join(MP, "atlas", "v2.1"); DIR = os.path.join(HERE, "Runtime Matrices", "Provisional_References")
SMOOTH = os.path.join(MP, "doors", "data", "DEV_ATLAS_V21_01", "transfer_v21_smooth.json")
ANN = "https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/EPIC/EPIC.hg38.manifest.tsv.gz"
FLAG = "PROVISIONAL REFERENCE, not part of IAM-Atlas v2.1 (development): shares from a second deconvolution with the column added"

def pool_hg38(samples_csv, work):
    """Pool per-CpG methylated/total read files (GEO, GRCh38, chr pos m n) at the atlas's sites. Returns probe, m, n."""
    os.makedirs(work, exist_ok=True); af = os.path.join(work, "EPIC.hg38.manifest.tsv.gz")
    if not os.path.exists(af): urllib.request.urlretrieve(ANN, af)
    K = pd.read_parquet(os.path.join(ATLAS, "sites_sequencing_key.parquet")).drop_duplicates("probe")
    A = pd.read_csv(af, sep="\t", usecols=["CpG_chrm", "CpG_beg", "Probe_ID"]).dropna().rename(columns={"Probe_ID": "probe"})
    A = A[A.probe.isin(K.probe)]; A["pos"] = A.CpG_beg.astype(np.int64) + 1; A = A.set_index(["CpG_chrm", "pos"])
    m = np.zeros(len(A), np.int64); n = np.zeros(len(A), np.int64); S = pd.read_csv(samples_csv)
    for g, f in zip(S.gsm, S.file):
        p = os.path.join(work, f)
        for _ in range(4):
            try: urllib.request.urlretrieve(f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{g[:7]}nnn/{g}/suppl/{f}", p); break
            except Exception: time.sleep(30)
        M = pd.read_csv(p, sep="\t", header=None, names=["chr", "pos", "m", "n"]).groupby(["chr", "pos"]).sum()   # a position listed twice is summed
        R = M.reindex(A.index); m += R["m"].fillna(0).to_numpy(np.int64); n += R["n"].fillna(0).to_numpy(np.int64); os.remove(p)
        print(f"  pooled {g}", flush=True)
    return pd.DataFrame({"probe": A.probe.to_numpy(), "m": m, "n": n}), hashlib.sha256(open(af, "rb").read()).hexdigest()

def column(P):
    """P: probe, m, n. Returns a DataFrame indexed by atlas site id with mu and sd (array scale)."""
    import stage_atlas_commission as A21
    from scipy.interpolate import PchipInterpolator
    T = json.load(open(SMOOTH)); g = PchipInterpolator(T["knots_x"], T["knots_y"]); gx = np.linspace(0, 1, 200001); gv = g(gx) + 1e-12 * gx
    P = P[P.n > 0].copy(); y = P.m / P.n; x = np.interp(y, gv, gx); k = np.maximum(g.derivative()(x), 0.151); q = (P.m + 0.5) / (P.n + 1)
    sites = pd.read_parquet(os.path.join(ATLAS, "sites.parquet")).cpg.astype(str).values; ds = []
    for i in (1, 2, 3):
        D = pd.read_parquet(os.path.join(ATLAS, f"donor_sd_part{i}.parquet")); ds.append(D.where(D != 65535) / 10000.0)
    d = pd.concat(ds, axis=1).median(1); d.index = sites
    sd = np.sqrt(q * (1 - q) / (P.n * k ** 2) + d.reindex(P.probe).fillna(d.median()).to_numpy() ** 2 + A21.lab_sd() ** 2 + A21.FLOOR)
    return pd.DataFrame({"mu": x.astype("float32"), "sd": np.asarray(sd, "float32"), "n": P.n.to_numpy(np.int32)}, index=P.probe.astype(str).to_numpy())

MIN_READS_V2 = 5   # stated before version 2 was read (2026-10-24)

def column_v2(P):
    """Version 2 (2026-10-24, after provisional reference test 1 was not met; disclosed). Version 1 put the counting spread on the array scale
    with the transfer's local slope, which at the ends (where every single-read site sits, read as exactly 0 or 1) understated it: sites with
    1 or 2 reads came out NARROWER (median 0.111, 0.075) than sites with 3 (0.208), so the Deconvolver trusted the thinnest sites most and
    chose donor 1's chance extremes as markers. Version 2: (1) the level from the shrunk proportion q = (m + 1/2)/(n + 1), so no site sits at
    exactly 0 or 1 on one read; (2) the counting spread carried through the atlas's own inverse transfer by the two ends of its interval,
    q -/+ sqrt(q (1 - q)/(n + 1)), half their distance on the array scale, no slope; (3) only sites with at least MIN_READS_V2 reads. The
    donor spread, laboratory and floor terms are version 1's. Nothing fitted."""
    import stage_atlas_commission as A21
    from scipy.interpolate import PchipInterpolator
    T = json.load(open(SMOOTH)); g = PchipInterpolator(T["knots_x"], T["knots_y"]); gx = np.linspace(0, 1, 200001); gv = g(gx) + 1e-12 * gx
    P = P[P.n >= MIN_READS_V2].copy(); q = (P.m + 0.5) / (P.n + 1); e = np.sqrt(q * (1 - q) / (P.n + 1))
    inv = lambda v: np.interp(np.clip(v, 0, 1), gv, gx); x = inv(q); half = (inv(q + e) - inv(q - e)) / 2
    sites = pd.read_parquet(os.path.join(ATLAS, "sites.parquet")).cpg.astype(str).values; ds = []
    for i in (1, 2, 3):
        D = pd.read_parquet(os.path.join(ATLAS, f"donor_sd_part{i}.parquet")); ds.append(D.where(D != 65535) / 10000.0)
    d = pd.concat(ds, axis=1).median(1); d.index = sites
    sd = np.sqrt(half ** 2 + d.reindex(P.probe).fillna(d.median()).to_numpy() ** 2 + A21.lab_sd() ** 2 + A21.FLOOR)
    return pd.DataFrame({"mu": np.asarray(x, "float32"), "sd": np.asarray(sd, "float32"), "n": P.n.to_numpy(np.int32)}, index=P.probe.astype(str).to_numpy())

def _name_version(cell):
    """'megakaryocytes@2' -> ('megakaryocytes', 2); a bare name is version 1."""
    c, _, v = str(cell).partition("@"); return c, int(v or 1)

def build(cell, samples_csv, work, out=None):
    cell, ver = _name_version(cell)
    P, ann = pool_hg38(samples_csv, work); C = column_v2(P) if ver == 2 else column(P); out = out or os.path.join(DIR, f"{cell}_v{ver}.parquet"); os.makedirs(os.path.dirname(out), exist_ok=True)
    C.to_parquet(out); meta = dict(cell=cell, version=ver, flag=FLAG, samples=os.path.basename(samples_csv), n_samples=int(len(pd.read_csv(samples_csv))), sites=int(len(C)),
        median_reads=int(np.median(C.n)), annotation=ANN, annotation_sha256=ann, transfer=os.path.basename(SMOOTH), column_sha256=hashlib.sha256(open(out, "rb").read()).hexdigest())
    json.dump(meta, open(out.replace(".parquet", ".json"), "w"), indent=1); return meta

def load(cell):
    c, v = _name_version(cell); p = os.path.join(DIR, f"{c}_v{v}.parquet"); return pd.read_parquet(p) if os.path.exists(p) else None

def with_provisional(R, cells):
    """The atlas reference R (stage_deconv_v21.reference) with provisional columns added; R itself is not changed."""
    mu, sd = R["mu"].copy(), R["sd"].copy(); added = []
    for c in cells:
        C = load(c)
        if C is None: continue
        nm = _name_version(c)[0] + " (provisional)"   # the version is in the bundle's flag, not the cell's name
        mu[nm] = C.mu.reindex(mu.index).astype("float32"); sd[nm] = C.sd.reindex(sd.index).astype("float32"); added.append(nm)
    return dict(R, mu=mu, sd=sd, cells=list(R["cells"]) + added), added
