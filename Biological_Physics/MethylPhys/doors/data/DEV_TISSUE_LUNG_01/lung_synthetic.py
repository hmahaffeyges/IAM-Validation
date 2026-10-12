"""DEV-TISSUE-LUNG-01 synthetic tests 1-5 (2026-10-12; patients' normal lung and the non-cancer control lungs only, no tumour array read).
Derived from DEV-TISSUE-COLON-01 colon_synthetic.py (same-slide design); changes: the panel and the healthy-donor source.
Reading: Met-A = mean binary entropy H(beta) at composition-blind sites. GEO gives no tumour type, so the blind sites are those where BOTH atlas
lung epithelia (alveolar, bronchial) and every other panel cell lie within 0.05 of the mean of the two epithelia; both epithelial SDs <= 0.05;
3,000 methylated (0.75-0.95) + 3,000 unmethylated (0.05-0.25), smallest cross-cell spread first. Pair ratio R = Met-A(test) / Met-A(reference) for
two arrays on one slide. Stand-in pairs: two different patients' normal lungs on one slide (j tested, k reference).
1 spread: R over stand-in pairs (no blend); per-pair bar = its 95th percentile; slide effect on Met-A.
2 composition shift (no methylation change): normal j blended (a ~ U(0.3, 0.6)) with a non-cancer control lung, with each atlas lung epithelium,
  and with immune cells (mean of macrophages, CD4/CD8 T, B, neutrophils). PASS: control-lung blend AND immune blend, share above the bar <= 0.10.
3 independence: per normal, Met-A tared by the median of the other normals on its slide, against its fitted epithelial share (alveolar + bronchial;
  NNLS on the 8,000 most cell-variable sites). PASS: |Spearman| < 0.2 and p > 0.05.
4 planted change d in the blended control-lung part (methylated sites lose a share d, unmethylated gain d(1-beta)), d = 0.02, 0.05, 0.10.
  PASS: d = 0.05 above the bar in >= 0.80 of pairs.
5 power: group rule over 28 pairs, FINGERPRINT if >= CRIT pairs above the bar (binomial p < 0.01 at 0.05 per pair); detection share over 2,000
  draws of 28 pairs. PASS: NULL, control-lung shift and immune shift group false calls <= 0.05; d = 0.05 group detection >= 0.80.
Writes lung_synthetic_output.csv and lung_synthetic_summary.txt. Usage: python3 lung_synthetic.py POSTDIR BETADIR"""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, kruskal, binom
POST, BD = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__)); rg = np.random.default_rng(20261012)
R_ = lambda n: pd.read_parquet(os.path.join(POST, f"atlas_{n.replace(' ', '_')}_posterior.parquet")).set_index("cpg")
names = {"alveolar": ["lung alveolar epithelium"], "bronchial": ["lung bronchus epithelium"], "fibroblast": ["colon fibroblasts", "heart fibroblasts"],
         "macrophage": ["colon macrophages"], "smooth_muscle": ["smooth muscle"], "endothelium": ["vascular endothelium"], "T": ["cd4 t cells", "cd8 t cells"],
         "B": ["b cells"], "neutrophil": ["neutrophils"], "monocyte": ["monocytes"], "adipocytes": ["adipocytes"]}
P = {k: [R_(n) for n in v] for k, v in names.items()}; idx = P["alveolar"][0].index
for v in P.values():
    for x in v: idx = idx.intersection(x.index)
S = pd.read_csv(os.path.join(HERE, "gse94785_samples.csv")); S = S[S.kind.isin(["normal", "control"])].set_index("gsm")
Bt = {}
for g in S.index:
    f = os.path.join(BD, g + ".parquet")
    if os.path.exists(f): b = pd.read_parquet(f).beta; b.index = b.index.astype(str); Bt[g] = b
S = S.loc[list(Bt)]; common = idx
for b in Bt.values(): common = common.intersection(b.index[b.notna().values])
Mu = pd.DataFrame({k: np.mean([x.loc[common, "mean"].values for x in v], 0) for k, v in P.items()}, index=common)
SDa, SDb = P["alveolar"][0].loc[common, "sd"], P["bronchial"][0].loc[common, "sd"]
ok = Mu.notna().all(1) & SDa.notna() & SDb.notna(); Mu, SDa, SDb = Mu[ok], SDa[ok], SDb[ok]; common = Mu.index
Em = (Mu["alveolar"] + Mu["bronchial"]) / 2
spread = Mu.sub(Em, axis=0).abs().max(1); good = (spread <= 0.05) & (SDa <= 0.05) & (SDb <= 0.05)
hi = Em[good & Em.between(.75, .95)].to_frame("m").assign(sp=spread).sort_values("sp").index[:3000]
lo = Em[good & Em.between(.05, .25)].to_frame("m").assign(sp=spread).sort_values("sp").index[:3000]
Sx = common.get_indexer(hi.union(lo)); hi_ = common.get_indexer(hi); lo_ = common.get_indexer(lo)
inf_ = common.get_indexer(Mu.std(1).sort_values(ascending=False).index[:8000])
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
X = {g: Bt[g].loc[common].values for g in S.index}; base = {g: H(X[g][Sx]).mean() for g in S.index}
Nm = S[S.kind == "normal"]; Ct = S[S.kind == "control"]
pairs = [(j, k) for j, k in itertools.permutations(Nm.index, 2) if Nm.slide[j] == Nm.slide[k]]
out = [f"arrays: patient normals {len(Nm)}, control lungs {len(Ct)} | loci {len(common)} | blind sites {len(hi)} + {len(lo)} | stand-in pairs {len(pairs)} on {Nm.slide.nunique()} slides"]
null = np.array([base[j] / base[k] for j, k in pairs]); bar = float(np.quantile(null, .95))
out.append(f"1 spread: stand-in R SD {null.std(ddof=1):.4f}, 95th percentile (pair bar) {bar:.4f}")
g_ = [[base[x] for x in Nm.index[Nm.slide == v]] for v in Nm.slide.unique()]; g_ = [v for v in g_ if len(v) >= 3]
out.append(f"  normals' Met-A by slide: Kruskal-Wallis p {kruskal(*g_).pvalue:.2e} ({len(g_)} slides with >= 3)")
out.append(f"  Met-A median: patient normal {np.median([base[g] for g in Nm.index]):.4f}, control lung {np.median([base[g] for g in Ct.index]):.4f} (descriptive)")
def blend(j, d, tpl):
    a = rg.uniform(.3, .6); T = tpl.copy(); T[lo_] = T[lo_] + d * (1 - T[lo_]); T[hi_] = T[hi_] * (1 - d)
    return np.clip((1 - a) * X[j] + a * T, 0, 1)
imm = Mu[["macrophage", "T", "B", "neutrophil", "monocyte"]].mean(1).values; rows = []
NP = 28; crit = int(binom.ppf(0.99, NP, 0.05)) + 1
def case(name, f):
    v = np.array([f(j, k) for j, k in pairs]); above = v > bar
    det = float(np.mean([above[rg.integers(len(v), size=NP)].sum() >= crit for _ in range(2000)]))
    rows.append(dict(case=name, pairs=len(v), mean_R=round(float(v.mean()), 4), share_above_bar=round(float(above.mean()), 3), group_call=round(det, 3), bar=round(bar, 4), crit_of_28=crit))
    out.append(f"  {name}: mean R {v.mean():.4f}, share above bar {above.mean():.3f}, group call {det:.3f}")
out.append("2-5 cases (pairs; group rule >= %d of %d, binomial p < 0.01):" % (crit, NP))
ctl = list(Ct.index)
case("NULL", lambda j, k: base[j] / base[k])
case("SHIFT control lung", lambda j, k: H(blend(j, 0, X[rg.choice(ctl)])[Sx]).mean() / base[k])
case("SHIFT atlas alveolar", lambda j, k: H(blend(j, 0, Mu["alveolar"].values)[Sx]).mean() / base[k])
case("SHIFT atlas bronchial", lambda j, k: H(blend(j, 0, Mu["bronchial"].values)[Sx]).mean() / base[k])
case("SHIFT immune", lambda j, k: H(blend(j, 0, imm)[Sx]).mean() / base[k])
for d in (.02, .05, .10): case(f"d{d:.2f} control lung", lambda j, k, d=d: H(blend(j, d, X[rg.choice(ctl)])[Sx]).mean() / base[k])
tar, epi = [], []
for g in Nm.index:
    oth = [base[x] for x in Nm.index if Nm.slide[x] == Nm.slide[g] and x != g]
    if len(oth) < 2: continue
    w, _ = nnls(Mu.values[inf_], X[g][inf_]); w = w / w.sum(); tar.append(base[g] / np.median(oth)); epi.append(w[0] + w[1])
rho, p = spearmanr(tar, epi); out.append(f"3 independence: tared Met-A vs fitted epithelial share ({min(epi):.3f}-{max(epi):.3f}), n {len(tar)}: Spearman {rho:+.3f} (p {p:.2f})")
T = pd.DataFrame(rows).set_index("case")
verd = {"2 composition (control lung and immune blends) <= 0.10": T.loc["SHIFT control lung", "share_above_bar"] <= 0.10 and T.loc["SHIFT immune", "share_above_bar"] <= 0.10,
        "3 independence": abs(rho) < 0.2 and p > 0.05, "4 planted 5 % >= 0.80": T.loc["d0.05 control lung", "share_above_bar"] >= 0.80,
        "5 power": all(T.loc[c, "group_call"] <= 0.05 for c in ("NULL", "SHIFT control lung", "SHIFT immune")) and T.loc["d0.05 control lung", "group_call"] >= 0.80}
out += [f"{k}: {'PASSED' if v else 'FAILED'}" for k, v in verd.items()]
T.to_csv(os.path.join(HERE, "lung_synthetic_output.csv")); open(os.path.join(HERE, "lung_synthetic_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
