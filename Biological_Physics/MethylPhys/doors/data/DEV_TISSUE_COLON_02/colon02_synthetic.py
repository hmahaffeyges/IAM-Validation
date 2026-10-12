"""DEV-TISSUE-COLON-02 synthetic tests 1-6 (written 2026-10-12, before any array of this set is read; non-cancer arrays only, tumours locked).
In GSE199057 no tumour shares a slide with its own normal (layout_02_output.txt), so the COLON-01 same-slide reading cannot be repeated.
Reading, unchanged from DEV-TISSUE-COLON-01 except the pairing: Met-A = mean binary entropy H(beta) at the composition-blind sites (atlas v2:
within 0.05 of the colon epithelium value in every panel cell; epithelium SD <= 0.05; 3,000 methylated (0.75-0.95) + 3,000 unmethylated
(0.05-0.25), smallest cross-cell spread first). Pair ratio R = Met-A(test) / Met-A(reference).
Stand-in pairs: two DIFFERENT patients' normals on DIFFERENT slides (j tested, k reference), so the slide-to-slide spread is inside the bar.
1 spread: R over stand-in pairs; per-pair bar = 95th percentile; slide effect on the normals' Met-A (Kruskal-Wallis).
2 composition shift (no methylation change): normal j blended with a non-CRC normal array from another slide (a ~ U(0.3, 0.6)), and with
  the atlas colon epithelium template. PASS (array blend): share of pairs above the bar <= 0.10.
3 independence: per normal, Met-A tared by the median of the normals on OTHER slides, against its fitted composition (NNLS, 8,000 most
  cell-variable sites). PASS: |Spearman| < 0.2 and p > 0.05 for the epithelial share.
4 planted change d in the blended part (methylated sites lose a share d, unmethylated gain d(1-beta)), d = 0.02, 0.05, 0.10.
  PASS: d = 0.05 above the bar in >= 0.80 of pairs.
5 power: group rule over 77 pairs, FINGERPRINT if >= CRIT pairs above the bar (binomial p < 0.01 at 0.05 per pair); detection over 2,000 draws.
  PASS: NULL and blend-shift group false calls <= 0.05; d = 0.05 group detection >= 0.80.
6 slide offset: the tumours sit on slides of their own. Each normal j is shifted by the slide offset of a randomly drawn OTHER slide (that
  slide's median Met-A over the all-normal median), on top of its own; this doubles the slide effect a tumour-only slide could carry.
  PASS: NULL group false call <= 0.05 with the doubled slide effect.
Writes colon02_synthetic_output.csv and colon02_synthetic_summary.txt. Usage: python3 colon02_synthetic.py POSTDIR BETADIR"""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, kruskal, binom
POST, BD = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__)); rg = np.random.default_rng(20261012)
R_ = lambda n: pd.read_parquet(os.path.join(POST, f"atlas_{n.replace(' ', '_')}_posterior.parquet")).set_index("cpg")
names = {"epithelium": ["colon epithelium"], "fibroblast": ["colon fibroblasts"], "macrophage": ["colon macrophages"], "smooth_muscle": ["smooth muscle"],
         "endothelium": ["vascular endothelium"], "T": ["cd4 t cells", "cd8 t cells"], "B": ["b cells"], "neutrophil": ["neutrophils"], "adipocytes": ["adipocytes"]}
P = {k: [R_(n) for n in v] for k, v in names.items()}; idx = P["epithelium"][0].index
for v in P.values():
    for x in v: idx = idx.intersection(x.index)
S = pd.read_csv(os.path.join(HERE, "gse199057_samples.csv")); S = S[~S.ctrl & ~S.kind.str.startswith("colon tumor")].set_index("gsm")
S["grp"] = np.where(S.kind.str.contains("non-CRC"), "H", "N")
common = idx; Bt = {}
for g in S.index:
    f = os.path.join(BD, g + ".parquet")
    if os.path.exists(f):
        b = pd.read_parquet(f).beta; b.index = b.index.astype(str); b = b.reindex(common); Bt[g] = b
for b in Bt.values(): common = common[b.loc[common].notna().values]
S = S.loc[list(Bt)]
Mu = pd.DataFrame({k: np.mean([x.loc[common, "mean"].values for x in v], 0) for k, v in P.items()}, index=common); E = P["epithelium"][0].loc[common]
ok = Mu.notna().all(1) & E["mean"].notna() & E["sd"].notna(); Mu, E = Mu[ok], E[ok]; common = Mu.index
spread = Mu.sub(Mu["epithelium"], axis=0).abs().max(1); good = (spread <= 0.05) & (E.sd <= 0.05)
hi = E[good & E["mean"].between(.75, .95)].assign(sp=spread).sort_values("sp").index[:3000]; lo = E[good & E["mean"].between(.05, .25)].assign(sp=spread).sort_values("sp").index[:3000]
Sx = common.get_indexer(hi.union(lo)); hi_ = common.get_indexer(hi); lo_ = common.get_indexer(lo)
inf_ = common.get_indexer(Mu.std(1).sort_values(ascending=False).index[:8000])
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
X = {g: Bt[g].loc[common].values for g in S.index}; del Bt; base = {g: H(X[g][Sx]).mean() for g in S.index}
Nm = S[S.grp == "N"]; Hc = S[S.grp == "H"]
pairs = [(j, k) for j, k in itertools.permutations(Nm.index, 2) if Nm.slide[j] != Nm.slide[k] and Nm.pid[j] != Nm.pid[k]]
pairs = [pairs[i] for i in rg.choice(len(pairs), size=min(4000, len(pairs)), replace=False)]
out = [f"arrays: patient normals {len(Nm)}, non-CRC normals {len(Hc)} | loci {len(common)} | blind sites {len(hi)} + {len(lo)} | cross-slide stand-in pairs {len(pairs)} (sampled) on {Nm.slide.nunique()} slides"]
null = np.array([base[j] / base[k] for j, k in pairs]); bar = float(np.quantile(null, .95))
out.append(f"1 spread: stand-in R SD {null.std(ddof=1):.4f}, 95th percentile (pair bar) {bar:.4f}")
g_ = [[base[x] for x in Nm.index[Nm.slide == v]] for v in Nm.slide.unique()]; g_ = [v for v in g_ if len(v) >= 3]
out.append(f"  normals' Met-A by slide: Kruskal-Wallis p {kruskal(*g_).pvalue:.2e} ({len(g_)} slides with >= 3)")
out.append(f"  Met-A median: patient normal {np.median([base[g] for g in Nm.index]):.4f}, non-CRC normal {np.median([base[g] for g in Hc.index]):.4f} (descriptive)")
med = np.median([base[g] for g in Nm.index]); off = {s: np.median([base[x] for x in Nm.index[Nm.slide == s]]) / med for s in Nm.slide.unique()}
def blend(j, d, tpl):
    a = rg.uniform(.3, .6); T = tpl.copy(); T[lo_] = T[lo_] + d * (1 - T[lo_]); T[hi_] = T[hi_] * (1 - d)
    return np.clip((1 - a) * X[j] + a * T, 0, 1)
def other_H(j): c = Hc.index[Hc.slide != Nm.slide[j]]; return X[rg.choice(c)]
epi_t = Mu["epithelium"].values; rows = []; N_ = 77; crit = int(binom.ppf(0.99, N_, 0.05)) + 1
def case(name, f):
    v = np.array([f(j, k) for j, k in pairs]); above = v > bar
    det = float(np.mean([above[rg.integers(len(v), size=N_)].sum() >= crit for _ in range(2000)]))
    rows.append(dict(case=name, pairs=len(v), mean_R=round(float(v.mean()), 4), share_above_bar=round(float(above.mean()), 3), group_call=round(det, 3), bar=round(bar, 4), crit_of_77=crit))
    out.append(f"  {name}: mean R {v.mean():.4f}, share above bar {above.mean():.3f}, group call {det:.3f}")
out.append("2-6 cases (pairs; group rule >= %d of %d, binomial p < 0.01):" % (crit, N_))
case("NULL", lambda j, k: base[j] / base[k])
case("SHIFT non-CRC array", lambda j, k: H(blend(j, 0, other_H(j))[Sx]).mean() / base[k])
case("SHIFT atlas epithelium", lambda j, k: H(blend(j, 0, epi_t)[Sx]).mean() / base[k])
for d in (.02, .05, .10): case(f"d{d:.2f} non-CRC array", lambda j, k, d=d: H(blend(j, d, other_H(j))[Sx]).mean() / base[k])
sl = list(off); case("SLIDE x2 NULL", lambda j, k: base[j] * off[sl[rg.integers(len(sl))]] / base[k])
tar, epi = [], []
for g in Nm.index:
    oth = [base[x] for x in Nm.index if Nm.slide[x] != Nm.slide[g]]
    w, _ = nnls(Mu.values[inf_], X[g][inf_]); w = w / w.sum(); tar.append(base[g] / np.median(oth)); epi.append(w[0])
rho, p = spearmanr(tar, epi); out.append(f"3 independence: tared Met-A vs fitted epithelial share ({min(epi):.3f}-{max(epi):.3f}), n {len(tar)}: Spearman {rho:+.3f} (p {p:.2f})")
T = pd.DataFrame(rows).set_index("case")
verd = {"2 composition (array blend) <= 0.10": T.loc["SHIFT non-CRC array", "share_above_bar"] <= 0.10, "3 independence": abs(rho) < 0.2 and p > 0.05,
        "4 planted 5 % >= 0.80": T.loc["d0.05 non-CRC array", "share_above_bar"] >= 0.80,
        "5 power": T.loc["NULL", "group_call"] <= 0.05 and T.loc["SHIFT non-CRC array", "group_call"] <= 0.05 and T.loc["d0.05 non-CRC array", "group_call"] >= 0.80,
        "6 slide offset x2": T.loc["SLIDE x2 NULL", "group_call"] <= 0.05}
out += [f"{k}: {'PASSED' if v else 'FAILED'}" for k, v in verd.items()]
T.to_csv(os.path.join(HERE, "colon02_synthetic_output.csv")); open(os.path.join(HERE, "colon02_synthetic_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
