"""DEV-TISSUE-LIVER-01 synthetic tests 1-6 (2026-10-12; TCGA-LIHC normal livers only, no tumor array fetched or read).
Derived from DEV-TISSUE-LUNG-01 lung_synthetic.py, with the three changes of LESSONS T15 (from the lung result).
Reading: Met-A = mean binary entropy H(beta) at composition-blind sites: every panel cell within 0.05 of the atlas hepatocyte value; hepatocyte SD
<= 0.05; 3,000 methylated (0.75-0.95) + 3,000 unmethylated (0.05-0.25), smallest cross-cell spread first. The atlas has no cholangiocyte (bile-duct
cell), so bile-duct content is not covered by the panel. Pair ratio R = Met-A(test) / Met-A(reference), two arrays on one plate.
Stand-in pairs: two different patients' normal livers on one plate (j tested, k reference).
1 spread (T15 step 1): R over stand-in pairs; per-pair bar = 95th percentile; plate effect; the spread reported beside colon and lung.
2 composition shift (no methylation change): normal j blended (a ~ U(0.3, 0.6)) with another patient's normal liver from its plate, with the atlas
  hepatocyte, and with immune cells. PASS: normal-liver and immune blends, share above the bar <= 0.10.
3 independence: per normal, Met-A tared by the median of the other normals on its plate, against its fitted hepatocyte share. PASS: |Spearman| < 0.2, p > 0.05.
4 planted change d in the blended normal-liver part, d = 0.01, 0.02, 0.05, 0.10 (T15 step 2: down to 1 %). PASS: d = 0.05 above the bar in >= 0.80.
5 two group rules over N = the number of sealed pairs (T15 step 3), each by 2,000 draws of N stand-in pairs:
  COUNT: FINGERPRINT if >= CRIT pairs above the bar (binomial p < 0.01 at 0.05 per pair);
  EVERY-PAIR: FINGERPRINT if the one-sided Wilcoxon signed-rank test of log R > 0 over all N pairs gives p < 0.01.
  A rule is eligible if its false calls are <= 0.05 under NULL, normal-liver shift and immune shift. The eligible rule with the higher detection at
  d = 0.02 is the one sealed (ties: COUNT). PASS: the chosen rule detects d = 0.05 in >= 0.80 of draws.
6 plate offset doubled (NULL with each array's plate offset doubled): chosen rule's false calls <= 0.05.
Writes liver_synthetic_output.csv and liver_synthetic_summary.txt. Usage: python3 liver_synthetic.py POSTDIR BETADIR"""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, kruskal, binom, wilcoxon
POST, BD = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__)); rg = np.random.default_rng(20261012)
R_ = lambda n: pd.read_parquet(os.path.join(POST, f"atlas_{n.replace(' ', '_')}_posterior.parquet")).set_index("cpg")
names = {"hepatocyte": ["hepatocyte"], "fibroblast": ["colon fibroblasts", "heart fibroblasts"], "macrophage": ["colon macrophages"],
         "smooth_muscle": ["smooth muscle"], "endothelium": ["vascular endothelium"], "T": ["cd4 t cells", "cd8 t cells"], "B": ["b cells"],
         "neutrophil": ["neutrophils"], "monocyte": ["monocytes"], "nk": ["nk cells"], "adipocytes": ["adipocytes"]}
P = {k: [R_(n) for n in v] for k, v in names.items()}; idx = P["hepatocyte"][0].index
for v in P.values():
    for x in v: idx = idx.intersection(x.index)
F = pd.read_csv(os.path.join(HERE, "tcga_lihc_450k_files.csv")); S = F[F.sample_type == "Solid Tissue Normal"].drop_duplicates("sample").set_index("sample")
Bt = {}
for g in S.index:
    f = os.path.join(BD, g + ".parquet")
    if os.path.exists(f): b = pd.read_parquet(f).beta; b.index = b.index.astype(str); Bt[g] = b
S = S.loc[list(Bt)]; common = idx
for b in Bt.values(): common = common.intersection(b.index[b.notna().values])
Mu = pd.DataFrame({k: np.mean([x.loc[common, "mean"].values for x in v], 0) for k, v in P.items()}, index=common); E = P["hepatocyte"][0].loc[common]
ok = Mu.notna().all(1) & E["mean"].notna() & E["sd"].notna(); Mu, E = Mu[ok], E[ok]; common = Mu.index
spread = Mu.sub(Mu["hepatocyte"], axis=0).abs().max(1); good = (spread <= 0.05) & (E.sd <= 0.05)
hi = E[good & E["mean"].between(.75, .95)].assign(sp=spread).sort_values("sp").index[:3000]; lo = E[good & E["mean"].between(.05, .25)].assign(sp=spread).sort_values("sp").index[:3000]
Sx = common.get_indexer(hi.union(lo)); hi_ = common.get_indexer(hi); lo_ = common.get_indexer(lo)
inf_ = common.get_indexer(Mu.std(1).sort_values(ascending=False).index[:8000])
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
X = {g: Bt[g].loc[common].values for g in S.index}; base = {g: H(X[g][Sx]).mean() for g in S.index}
pairs = [(j, k) for j, k in itertools.permutations(S.index, 2) if S.plate[j] == S.plate[k]]
T_ = F[F.sample_type == "Primary Tumor"].drop_duplicates("sample"); NP = int(sum(((T_.case == c) & (T_.plate == S.plate[g])).any() for g, c in S.case.items()))
out = [f"arrays: normal livers {len(S)} on {S.plate.nunique()} plates | loci {len(common)} | blind sites {len(hi)} + {len(lo)} | stand-in pairs {len(pairs)} | sealed pairs N {NP}"]
null = np.array([base[j] / base[k] for j, k in pairs]); bar = float(np.quantile(null, .95))
out.append(f"1 spread: stand-in R SD {null.std(ddof=1):.4f}, 95th percentile (pair bar) {bar:.4f}  [earlier tissues: colon SD 0.0502, lung SD 0.0860]")
g_ = [[base[x] for x in S.index[S.plate == v]] for v in S.plate.unique()]; g_ = [v for v in g_ if len(v) >= 3]
if len(g_) >= 2: out.append(f"  normals' Met-A by plate: Kruskal-Wallis p {kruskal(*g_).pvalue:.2e} ({len(g_)} plates with >= 3)")
out.append(f"  Met-A median, normal liver: {np.median(list(base.values())):.4f}")
pl = {p: list(S.index[S.plate == p]) for p in S.plate.unique()}
def blend(j, d, tpl):
    a = rg.uniform(.3, .6); T = tpl.copy(); T[lo_] = T[lo_] + d * (1 - T[lo_]); T[hi_] = T[hi_] * (1 - d)
    return np.clip((1 - a) * X[j] + a * T, 0, 1)
def other(j): c = [x for x in pl[S.plate[j]] if x != j]; return X[rg.choice(c)]
imm = Mu[["macrophage", "T", "B", "neutrophil", "monocyte", "nk"]].mean(1).values
crit = int(binom.ppf(0.99, NP, 0.05)) + 1; rows = []
def calls(v):
    above = v > bar; lr = np.log(v); c, w = [], []
    for _ in range(2000):
        i = rg.integers(len(v), size=NP); c.append(above[i].sum() >= crit)
        w.append(wilcoxon(lr[i], alternative="greater").pvalue < 0.01)
    return float(np.mean(c)), float(np.mean(w)), float(above.mean())
def case(name, f):
    v = np.array([f(j, k) for j, k in pairs]); cc, ww, sh = calls(v)
    rows.append(dict(case=name, pairs=len(v), mean_R=round(float(v.mean()), 4), share_above_bar=round(sh, 3), count_call=round(cc, 3), every_pair_call=round(ww, 3), bar=round(bar, 4), crit=crit, N=NP))
    out.append(f"  {name}: mean R {v.mean():.4f}, share above bar {sh:.3f}, COUNT call {cc:.3f}, EVERY-PAIR call {ww:.3f}")
out.append(f"2-6 cases (N {NP} pairs; COUNT >= {crit} above the bar; EVERY-PAIR one-sided signed-rank p < 0.01):")
case("NULL", lambda j, k: base[j] / base[k])
case("SHIFT normal liver", lambda j, k: H(blend(j, 0, other(j))[Sx]).mean() / base[k])
case("SHIFT atlas hepatocyte", lambda j, k: H(blend(j, 0, Mu["hepatocyte"].values)[Sx]).mean() / base[k])
case("SHIFT immune", lambda j, k: H(blend(j, 0, imm)[Sx]).mean() / base[k])
for d in (.01, .02, .05, .10): case(f"d{d:.2f} normal liver", lambda j, k, d=d: H(blend(j, d, other(j))[Sx]).mean() / base[k])
off = {g: base[g] / np.median([base[x] for x in pl[S.plate[g]]]) for g in S.index}
case("PLATE x2 NULL", lambda j, k: (base[j] * off[j]) / (base[k] * off[k]))
tar, hep = [], []
for g in S.index:
    oth = [base[x] for x in pl[S.plate[g]] if x != g]
    if len(oth) < 2: continue
    w, _ = nnls(Mu.values[inf_], X[g][inf_]); w = w / w.sum(); tar.append(base[g] / np.median(oth)); hep.append(w[0])
if len(tar) >= 5: rho, p = spearmanr(tar, hep); out.append(f"3 independence: tared Met-A vs fitted hepatocyte share ({min(hep):.3f}-{max(hep):.3f}), n {len(tar)}: Spearman {rho:+.3f} (p {p:.2f})")
else: rho, p = np.nan, np.nan; out.append(f"3 independence: only {len(tar)} normals share a plate with >= 2 others: NOT TESTABLE")
T = pd.DataFrame(rows).set_index("case")
elig = {r: all(T.loc[c, col] <= 0.05 for c in ("NULL", "SHIFT normal liver", "SHIFT immune")) for r, col in (("COUNT", "count_call"), ("EVERY-PAIR", "every_pair_call"))}
det2 = {"COUNT": T.loc["d0.02 normal liver", "count_call"], "EVERY-PAIR": T.loc["d0.02 normal liver", "every_pair_call"]}
ch = [r for r in ("COUNT", "EVERY-PAIR") if elig[r]]; chosen = max(ch, key=lambda r: (det2[r], r == "COUNT")) if ch else None
col = {"COUNT": "count_call", "EVERY-PAIR": "every_pair_call"}.get(chosen)
out.append(f"5 rules: eligible {elig}; detection at d = 0.02 {det2}; CHOSEN RULE: {chosen}")
verd = {"2 composition (normal-liver and immune blends) <= 0.10": T.loc["SHIFT normal liver", "share_above_bar"] <= 0.10 and T.loc["SHIFT immune", "share_above_bar"] <= 0.10,
        "3 independence": bool(abs(rho) < 0.2 and p > 0.05) if rho == rho else False, "4 planted 5 % >= 0.80": T.loc["d0.05 normal liver", "share_above_bar"] >= 0.80,
        "5 power (chosen rule detects d = 0.05 >= 0.80)": bool(chosen) and T.loc["d0.05 normal liver", col] >= 0.80,
        "6 plate offset x2 (chosen rule false calls <= 0.05)": bool(chosen) and T.loc["PLATE x2 NULL", col] <= 0.05}
out += [f"{k}: {'PASSED' if v else 'FAILED'}" for k, v in verd.items()]
T.to_csv(os.path.join(HERE, "liver_synthetic_output.csv")); open(os.path.join(HERE, "liver_synthetic_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
