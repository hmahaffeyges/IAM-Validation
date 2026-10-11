"""DEV-TISSUE-COLON-01 synthetic tests 1-5 (2026-10-11; healthy mucosa and adjacent normal arrays only, no tumour array read).
Reading: Met-A = mean binary entropy H(beta) at composition-blind sites (atlas v2: within 0.05 of the colon epithelium value in every panel cell;
epithelium SD <= 0.05; 3,000 methylated (0.75-0.95) + 3,000 unmethylated (0.05-0.25), smallest cross-cell spread first). Pair ratio
R = Met-A(test) / Met-A(reference) for two arrays on one slide. Stand-in pairs: two different adjacent normals on one slide (j tested, k reference).
1 spread: R over stand-in pairs (no blend); per-pair bar = its 95th percentile; plate and slide effects on Met-A.
2 composition shift (no methylation change): normal j blended with a healthy donor's mucosa array from the same plate (a ~ U(0.3, 0.6)), and with
  the atlas colon epithelium template. PASS (mucosa blend): share of pairs above the bar <= 0.10.
3 independence: per normal, Met-A tared by the median of the other normals on its slide, against its fitted composition (NNLS on the 8,000 most
  cell-variable sites). PASS: |Spearman| < 0.2 and p > 0.05 for the epithelial share.
4 planted change d in the blended mucosa part (methylated sites lose a share d, unmethylated gain d(1-beta)), d = 0.02, 0.05, 0.10.
  PASS: d = 0.05 above the bar in >= 0.80 of pairs.
5 power: group rule over 91 pairs, FINGERPRINT if >= CRIT pairs above the bar (CRIT: binomial p < 0.05 at 0.05 per pair); detection share over
  2,000 draws of 91 pairs. PASS: NULL and mucosa-shift group false calls <= 0.05; d = 0.05 group detection >= 0.80.
Writes colon_synthetic_output.csv and colon_synthetic_summary.txt. Usage: python3 colon_synthetic.py POSTDIR BETADIR"""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, kruskal, binom
POST, BD = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__)); rg = np.random.default_rng(20261011)
R_ = lambda n: pd.read_parquet(os.path.join(POST, f"atlas_{n.replace(' ', '_')}_posterior.parquet")).set_index("cpg")
names = {"epithelium": ["colon epithelium"], "fibroblast": ["colon fibroblasts"], "macrophage": ["colon macrophages"], "smooth_muscle": ["smooth muscle"],
         "endothelium": ["vascular endothelium"], "T": ["cd4 t cells", "cd8 t cells"], "B": ["b cells"], "neutrophil": ["neutrophils"], "adipocytes": ["adipocytes"]}
P = {k: [R_(n) for n in v] for k, v in names.items()}; idx = P["epithelium"][0].index
for v in P.values():
    for x in v: idx = idx.intersection(x.index)
S = pd.read_csv(os.path.join(HERE, "colon_samples.csv")); S = S[S.tissue.isin(["Normal", "Mucosa"]) & (S.quality_control_status == "OK")].set_index("gsm")
Bt = {}
for g in S.index:
    f = os.path.join(BD, g + ".parquet")
    if os.path.exists(f): b = pd.read_parquet(f).beta; b.index = b.index.astype(str); Bt[g] = b
S = S.loc[list(Bt)]; common = idx
for b in Bt.values(): common = common.intersection(b.index[b.notna().values])
Mu = pd.DataFrame({k: np.mean([x.loc[common, "mean"].values for x in v], 0) for k, v in P.items()}, index=common); E = P["epithelium"][0].loc[common]
ok = Mu.notna().all(1) & E["mean"].notna() & E["sd"].notna(); Mu, E = Mu[ok], E[ok]; common = Mu.index
spread = Mu.sub(Mu["epithelium"], axis=0).abs().max(1); good = (spread <= 0.05) & (E.sd <= 0.05)
hi = E[good & E["mean"].between(.75, .95)].assign(sp=spread).sort_values("sp").index[:3000]; lo = E[good & E["mean"].between(.05, .25)].assign(sp=spread).sort_values("sp").index[:3000]
Sx = common.get_indexer(hi.union(lo)); hi_ = common.get_indexer(hi); lo_ = common.get_indexer(lo)
inf_ = common.get_indexer(Mu.std(1).sort_values(ascending=False).index[:8000])
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
X = {g: Bt[g].loc[common].values for g in S.index}; base = {g: H(X[g][Sx]).mean() for g in S.index}
Nm = S[S.tissue == "Normal"]; Mc = S[S.tissue == "Mucosa"]
pairs = [(j, k) for j, k in itertools.permutations(Nm.index, 2) if Nm.id_sentrix[j] == Nm.id_sentrix[k]]
out = [f"arrays: normal {len(Nm)}, mucosa {len(Mc)} (QC OK) | loci {len(common)} | blind sites {len(hi)} + {len(lo)} | stand-in pairs {len(pairs)} on {Nm.id_sentrix.nunique()} slides"]
# 1 spread
null = np.array([base[j] / base[k] for j, k in pairs]); bar = float(np.quantile(null, .95))
out.append(f"1 spread: stand-in R SD {null.std(ddof=1):.4f}, 95th percentile (pair bar) {bar:.4f}")
for col in ("plate", "id_sentrix"):
    g = [Nm.index[Nm[col] == v] for v in Nm[col].unique()]; g = [[base[x] for x in v] for v in g if len(v) >= 3]
    out.append(f"  normals' Met-A by {col}: Kruskal-Wallis p {kruskal(*g).pvalue:.2e} ({len(g)} groups with >= 3)")
out.append(f"  Met-A median: normal {np.median([base[g] for g in Nm.index]):.4f}, mucosa {np.median([base[g] for g in Mc.index]):.4f} (descriptive)")
mpl = {p: list(Mc.index[Mc.plate == p]) for p in S.plate.unique()}
def blend(j, d, tpl):
    a = rg.uniform(.3, .6); T = tpl.copy(); T[lo_] = T[lo_] + d * (1 - T[lo_]); T[hi_] = T[hi_] * (1 - d)
    return np.clip((1 - a) * X[j] + a * T, 0, 1)
epi_t = Mu["epithelium"].values; rows = []
crit = int(binom.ppf(0.95, 91, 0.05)) + 1
def case(name, f):
    v = np.array([f(j, k) for j, k in pairs]); above = v > bar
    det = float(np.mean([above[rg.integers(len(v), size=91)].sum() >= crit for _ in range(2000)]))
    rows.append(dict(case=name, pairs=len(v), mean_R=round(float(v.mean()), 4), share_above_bar=round(float(above.mean()), 3), group_call=round(det, 3), bar=round(bar, 4), crit_of_91=crit))
    out.append(f"  {name}: mean R {v.mean():.4f}, share above bar {above.mean():.3f}, group call {det:.3f}")
out.append("2-5 cases (pairs; group rule >= %d of 91):" % crit)
case("NULL", lambda j, k: base[j] / base[k])
case("SHIFT mucosa", lambda j, k: H(blend(j, 0, X[rg.choice(mpl[Nm.plate[j]])])[Sx]).mean() / base[k])
case("SHIFT atlas epithelium", lambda j, k: H(blend(j, 0, epi_t)[Sx]).mean() / base[k])
for d in (.02, .05, .10): case(f"d{d:.2f} mucosa", lambda j, k, d=d: H(blend(j, d, X[rg.choice(mpl[Nm.plate[j]])])[Sx]).mean() / base[k])
# 3 independence
tar, epi = [], []
for g in Nm.index:
    oth = [base[x] for x in Nm.index if Nm.id_sentrix[x] == Nm.id_sentrix[g] and x != g]
    if len(oth) < 2: continue
    w, _ = nnls(Mu.values[inf_], X[g][inf_]); w = w / w.sum(); tar.append(base[g] / np.median(oth)); epi.append(w[0])
rho, p = spearmanr(tar, epi); out.append(f"3 independence: tared Met-A vs fitted epithelial share ({min(epi):.3f}-{max(epi):.3f}), n {len(tar)}: Spearman {rho:+.3f} (p {p:.2f})")
T = pd.DataFrame(rows).set_index("case")
verd = {"2 composition (mucosa blend) <= 0.10": T.loc["SHIFT mucosa", "share_above_bar"] <= 0.10, "3 independence": abs(rho) < 0.2 and p > 0.05,
        "4 planted 5 % >= 0.80": T.loc["d0.05 mucosa", "share_above_bar"] >= 0.80,
        "5 power": T.loc["NULL", "group_call"] <= 0.05 and T.loc["SHIFT mucosa", "group_call"] <= 0.05 and T.loc["d0.05 mucosa", "group_call"] >= 0.80}
out += [f"{k}: {'PASSED' if v else 'FAILED'}" for k, v in verd.items()]
T.to_csv(os.path.join(HERE, "colon_synthetic_output.csv")); open(os.path.join(HERE, "colon_synthetic_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
