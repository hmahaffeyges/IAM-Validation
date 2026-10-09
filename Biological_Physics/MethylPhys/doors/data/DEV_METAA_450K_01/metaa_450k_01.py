"""DEV-METAA-450K-01 — every 450K number logged on 2026-10-09, rebuilt from pinned inputs.
Inputs (s3://methylphys-data-945451304272-us-west-2-an/, sha256 in inputs_sha256.json): Stage 1 betas of GSE88824 (reference laboratory,
calibrated by calib450.py), GSE124565 (first other laboratory), GSE224807 non-smoker CD15 arrays (second other laboratory; IDAT list in
gse224807_ns_cd15_idats.txt, calibrated with chain/stage_1_idat_calibration.py), and the 450K manifest (probe design types).
Sample sheets: samples_GSE*.csv (from GEO). Run: python3 metaa_450k_01.py [cache_dir]"""
import os, sys, json, gzip, hashlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_cache")
BUCKET = "methylphys-data-945451304272-us-west-2-an"; PINS = json.load(open(os.path.join(HERE, "inputs_sha256.json")))
def fetch(key):
    os.makedirs(CACHE, exist_ok=True); p = os.path.join(CACHE, key.rsplit("/", 1)[1])
    if not os.path.exists(p):
        import boto3; boto3.client("s3").download_file(BUCKET, key, p)
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == PINS[key], f"checksum differs: {key}"
    return p
def Hb(b):
    b = np.clip(b, 1e-9, 1 - 1e-9); return -(b * np.log2(b) + (1 - b) * np.log2(1 - b))
def sheet(a): return pd.read_csv(os.path.join(HERE, f"samples_{a}.csv"))
def match(columns, keys): return [c for c in columns if any(c.startswith(str(k).split("_")[0]) for k in keys)]

# ---- A. reference laboratory (GSE88824): identity sites, healthy reference, held-out spread
G8 = pd.read_parquet(fetch("results/K450_COMMISSION/betas_GSE88824.parquet")); m8 = sheet("GSE88824")
cols = match(G8.columns, m8[m8.src == "Control-Neutrophil"].key.dropna()); R8 = G8[cols].dropna(how="any")
def sites450(R):
    m = R.mean(1); s = R.std(1); ok = s <= 0.05
    return s[ok & (m >= 0.75) & (m <= 0.95)].sort_values().index[:3000].union(s[ok & (m >= 0.05) & (m <= 0.25)].sort_values().index[:3000])
S450 = sites450(R8); floor450 = float(np.mean([Hb(R8.loc[S450, c]).mean() for c in cols]))
loo = []
for c in cols:
    o = [x for x in cols if x != c]; s = sites450(R8[o]); fl = float(np.mean([Hb(R8.loc[s, x]).mean() for x in o])); loo.append(Hb(R8.loc[s, c]).mean() / fl)
print(f"A. reference arrays {len(cols)} | identity sites {len(S450)} | healthy reference {floor450:.5f} bits | held-out SD {np.std(loo, ddof=1):.4f}")

# ---- B. self-tare on invariant sites, by probe design type
des = {}
with gzip.open(fetch("reference/manifests/HumanMethylation450k_15017482_v3.csv.gz"), "rt", errors="replace") as fh:
    hdr = None
    for l in fh:
        if hdr is None:
            if l.startswith("IlmnID"): hdr = l.rstrip("\n").split(","); iD = hdr.index("Infinium_Design_Type")
            continue
        p = l.split(",")
        if len(p) > iD and p[0].startswith("cg"): des[p[0]] = p[iD]
DES = pd.Series(des)
grp = {g: match(G8.columns, m8[m8.src == f"Control-{g}"].key.dropna()) for g in ("Neutrophil", "CD4T", "CD8T", "NKcell", "CD19B", "Monocyte")}
GM = pd.DataFrame({g: G8[v].mean(1) for g, v in grp.items()}); GS = pd.DataFrame({g: G8[v].std(1) for g, v in grp.items()})
ok = (GS.max(1) <= 0.02) & ((GM.max(1) - GM.min(1)) <= 0.03) & (~GM.index.isin(S450))
low = ok & (GM.max(1) <= 0.15); high = ok & (GM.min(1) >= 0.85)
SETS = {f"{d}_{lh}": [s for s in GM.index[m] if DES.get(s) == d] for d in ("I", "II") for lh, m in (("low", low), ("high", high))}
def anch(b): return {k: float(b.reindex(v).dropna().mean()) for k, v in SETS.items()}
REFA = {k: float(np.mean([anch(G8[c])[k] for c in cols])) for k in SETS}
def selftare(b):
    b2 = b.copy(); a = anch(b)
    for d in ("I", "II"):
        L, U, Lr, Ur = a[f"{d}_low"], a[f"{d}_high"], REFA[f"{d}_low"], REFA[f"{d}_high"]
        idx = DES.index[DES == d].intersection(b.index); b2[idx] = Lr + (b[idx] - L) * (Ur - Lr) / (U - L)
    return b2.clip(0.0005, 0.9995)
def A450(b):
    x = selftare(b).reindex(S450).dropna(); return float(Hb(x).mean() / floor450), len(x) / len(S450)
print("B. invariant anchor sites:", {k: len(v) for k, v in SETS.items()})
loo2 = []
for c in cols:
    o = [x for x in cols if x != c]; s_ = sites450(R8[o]); ra = {k: float(np.mean([anch(G8[x])[k] for x in o])) for k in SETS}
    def st_(b, ra=ra):
        b2 = b.copy(); a_ = anch(b)
        for d in ("I", "II"):
            L, U, Lr, Ur = a_[f"{d}_low"], a_[f"{d}_high"], ra[f"{d}_low"], ra[f"{d}_high"]
            idx = DES.index[DES == d].intersection(b.index); b2[idx] = Lr + (b[idx] - L) * (Ur - Lr) / (U - L)
        return b2.clip(0.0005, 0.9995)
    fl = float(np.mean([Hb(st_(G8[x]).reindex(s_)).mean() for x in o])); loo2.append(float(Hb(st_(G8[c]).reindex(s_)).mean()) / fl)
R8t = pd.DataFrame({c: selftare(G8[c]) for c in cols}).loc[R8.index]
loo1 = []
for c in cols:
    o = [x for x in cols if x != c]; s_ = sites450(R8t[o]); fl = float(np.mean([Hb(R8t.loc[s_, x]).mean() for x in o])); loo1.append(float(Hb(R8t.loc[s_, c]).mean()) / fl)
print(f"   held-out after self-tare (anchors from all 8, as logged): SD {np.std(loo1, ddof=1):.4f} ({min(loo1):.3f}-{max(loo1):.3f})")
print(f"   held-out after self-tare (anchors also held out, stricter): SD {np.std(loo2, ddof=1):.4f} ({min(loo2):.3f}-{max(loo2):.3f})")

# ---- C. first other laboratory (GSE124565): healthy readings, same-run tare, loss detection
G2 = pd.read_parquet(fetch("results/K450_COMMISSION/betas_GSE124565.parquet")); m2 = sheet("GSE124565")
m2["status"] = m2.characteristics.str.split(r" \| ").str[1].str.split(": ").str[1]
hk = match(G2.columns, m2[m2.status == "healthy"].key.dropna())
A2 = pd.Series({c: A450(G2[c])[0] for c in hk}); Arel2 = pd.Series({c: A2[c] / A2.drop(c).median() for c in hk})
print(f"C. healthy arrays {len(hk)} | tared Normal {int(Arel2.between(0.95, 1.05).sum())} of {len(hk)} | range {Arel2.min():.4f}-{Arel2.max():.4f}")
for de in (0.01, 0.02, 0.03):
    out = []
    for c in hk:
        x = selftare(G2[c]).reindex(S450).dropna(); hi = x > 0.5; x[hi] = x[hi] * (1 - de); out.append(float(Hb(x).mean() / floor450) / A2.drop(c).median())
    print(f"   {de:.0%} loss: A_rel {min(out):.3f}-{max(out):.3f} | outside Normal {sum(o > 1.05 for o in out)}/{len(out)}")

# ---- D. second other laboratory (GSE224807 non-smoker CD15): readings, slides, noise gate
G7 = pd.read_parquet(fetch("results/K450_COMMISSION/betas_GSE224807_NS_CD15.parquet"))
names = open(os.path.join(HERE, "gse224807_ns_cd15_idats.txt")).read().split(); slide = {n.split("_")[0]: n.split("_")[1] for n in names if "_Grn" in n}
A7 = {c: A450(G7[c]) for c in G7.columns}
T7 = pd.DataFrame({"A": pd.Series({c: v[0] for c, v in A7.items()}), "cov": pd.Series({c: v[1] for c, v in A7.items()}), "slide": pd.Series(slide)}).dropna(subset=["A"])
def tare(T):
    out = {}
    for c, x in T.iterrows():
        same = T[(T.slide == x.slide) & (T.index != c)].A; refs = same if len(same) >= 3 else T.drop(c).A; out[c] = x.A / refs.median()
    return pd.Series(out)
T7["A_rel"] = tare(T7)
print(f"D. arrays {len(T7)} | slides {T7.slide.nunique()} | tared Normal {int(T7.A_rel.between(0.95, 1.05).sum())} of {len(T7)} | SD {T7.A_rel.std():.4f}")

# ---- E. slide diagnosis: same-slide pairs vs random pairs; one direction carries the spread
from scipy.stats import mannwhitneyu
r = np.random.default_rng(3)
pairs = [g.index.tolist() for _, g in T7.groupby("slide") if len(g) == 2]
d_same = np.array([abs(T7.A_rel[a] - T7.A_rel[b]) for a, b in pairs])
d_rand = np.array([abs(T7.A_rel[a] - T7.A_rel[b]) for a, b in (r.choice(T7.index.tolist(), 2, replace=False) for _ in range(5000))])
B7 = pd.DataFrame({c: selftare(G7[c]).reindex(S450) for c in T7.index}).dropna(); Rz = B7.sub(B7.median(1), axis=0)
U, Sv, Vt = np.linalg.svd(Rz.values - Rz.values.mean(1, keepdims=True), full_matrices=False); ev = Sv ** 2 / np.sum(Sv ** 2)
pc1 = pd.Series(Vt[0], index=Rz.columns); load = pd.Series(U[:, 0] * Sv[0], index=Rz.index); mu7 = B7.median(1)
print(f"E. same-slide pairs {len(pairs)} median |diff| {np.median(d_same):.4f} | random {np.median(d_rand):.4f} | p {mannwhitneyu(d_same, d_rand, alternative='less').pvalue:.2g}")
print(f"   PC1 {ev[0]:.3f} of variance | corr(PC1, A_rel) {abs(np.corrcoef(pc1, T7.A_rel.loc[pc1.index])[0, 1]):.3f} | loading: methylated sites {load[mu7 > 0.5].mean():+.4f}, unmethylated {load[mu7 <= 0.5].mean():+.4f}")

# ---- F. deviation load (DEV-SYNTH-LEVERS-01 simulation 7)
from scipy.optimize import nnls  # noqa: F401  (kept for parity with the notebook imports)
Rf = pd.DataFrame({c: selftare(G8[c]) for c in cols}); Hb2 = pd.DataFrame({c: selftare(G2[c]) for c in hk}); H7 = pd.DataFrame({c: selftare(G7[c]) for c in T7.index})
common = Rf.dropna().index.intersection(Hb2.dropna().index).intersection(H7.dropna().index); Rf, Hb2, H7 = Rf.loc[common], Hb2.loc[common], H7.loc[common]
def shrunk_sd(ref):
    mu = ref.mean(1); sd = ref.std(1, ddof=1); bn = pd.qcut(mu, 40, labels=False, duplicates="drop"); sdb = sd.groupby(bn).transform("median")
    n = ref.shape[1]; return mu, np.sqrt(((n - 1) * sd ** 2 + 8 * sdb ** 2) / (n - 1 + 8)).clip(lower=0.01)
def count(b, mu, sd, z):
    X = np.c_[np.ones(len(b)), mu.values]; a0, a1 = np.linalg.lstsq(X, b.values, rcond=None)[0]; return int((((b - a0) / a1 - mu) / sd).abs().gt(z).sum())
mu_r, sd_r = shrunk_sd(Rf)
L2 = [count(Hb2[c], mu_r, sd_r, 5) for c in Hb2]; L7 = [count(H7[c], mu_r, sd_r, 5) for c in H7]
print(f"F. sites {len(common)} | load vs another lab's reference: GSE124565 median {np.median(L2):.0f} max {max(L2)}; GSE224807 max {max(L7)}")
def loo_load(B, c, z, override=None):
    ref = B.drop(columns=[c]); mu, sd = shrunk_sd(ref); return count(override if override is not None else B[c], mu, sd, z)
r = np.random.default_rng(5)
for z in (5, 6, 8, 10):
    Lh = pd.Series({c: loo_load(Hb2, c, z) for c in Hb2})
    for k, db in ((300, 0.2), (300, 0.1), (1000, 0.1)):
        if z in (5, 6) and db == 0.1: continue
        sel = r.choice(len(common), k, replace=False); m = mu_r.values[sel]; caught = 0
        for c in Hb2:
            v = Hb2[c].values.copy(); v[sel] = np.clip(v[sel] + np.where(m > 0.5, -db, db), 0.001, 0.999)
            caught += loo_load(Hb2, c, z, override=pd.Series(v, index=Hb2.index)) > Lh.drop(c).max()
        print(f"   |z| > {z:2d}: healthy {Lh.min()}-{Lh.max()} | {k} sites, Δβ {db}: caught {caught}/{Hb2.shape[1]}")
