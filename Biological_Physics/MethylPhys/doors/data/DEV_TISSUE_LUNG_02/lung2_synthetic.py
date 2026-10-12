"""DEV-TISSUE-LUNG-02 synthetic tests 1-7 (written 2026-10-12 before any array of this set is read; patients' normal lungs only).
Reading as DEV-TISSUE-LUNG-01: Met-A = mean binary entropy H(beta) at composition-blind sites (both atlas lung epithelia and every other panel cell
within 0.05 of the epithelial mean; both epithelial SDs <= 0.05; 3,000 + 3,000 sites). Pair ratio R = Met-A(test) / Met-A(reference), one slide.
Stand-in pairs: two different patients' normal lungs on one slide. Design changes from LESSONS T15 (lung 1) and DEV-TISSUE-LIVER-01:
1 spread: stand-in R SD and the per-pair bar (95th percentile), beside lung 1 (SD 0.0860) and colon (0.0502).
2 composition: normal j blended (a ~ U(0.3, 0.6)) with another patient's normal lung from its slide, each atlas lung epithelium, immune cells.
  PASS: other-normal and immune blends, share above the bar <= 0.10.
4 planted change d in the blended normal part, d = 0.005, 0.0075, 0.01, 0.02, 0.05 (lung 1's median tumor lean, R 1.038, is about d 0.0075 on this
  reading). PASS: d = 0.05 above the bar in >= 0.80 of pairs.
5 two group rules over N = 12 pairs, 2,000 draws each: COUNT (>= CRIT above the bar, binomial p < 0.01 at 0.05 per pair) and EVERY-PAIR (one-sided
  Wilcoxon signed-rank of log R > 0, p < 0.01). Eligible if false calls <= 0.05 under NULL, other-normal and immune shifts; the eligible rule with
  the higher detection at d = 0.0075 is sealed (ties: COUNT). PASS: the chosen rule detects d = 0.05 in >= 0.80 of draws.
7 direction of composition (as DEV-TISSUE-LIVER-01 test 7): stroma, fibroblasts, smooth muscle, endothelium, immune, each epithelium, fat cells,
  another patient's normal lung, at a ~ U(0.3, 0.6) and U(0.6, 0.9). PASS: the chosen rule's false call <= 0.05 for every mix.
  (Independence, test 3, is reported but replaced by test 7 from the start, the rule DEV-TISSUE-LIVER-01 established.)
Reported, written now: the chosen rule's detection at d = 0.0075. If it is < 0.50, a NO FINGERPRINT here is reported as 'not decidable at lung 1's
lean', not as absence.
Draws: 12 DIFFERENT stand-in pairs per draw (without replacement), as the real test has 12 different patients; exact signed-rank p (fix 2026-10-12
before any output existed: with replacement, repeated pairs made ties and scipy's default switched to a permutation method about 700x slower).
Writes lung2_synthetic_output.csv and lung2_synthetic_summary.txt. Usage: python3 lung2_synthetic.py POSTDIR BETADIR"""
import os, sys, itertools, numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, binom, wilcoxon
POST, BD = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__)); rg = np.random.default_rng(20261012)
R_ = lambda n: pd.read_parquet(os.path.join(POST, f"atlas_{n.replace(' ', '_')}_posterior.parquet")).set_index("cpg")
names = {"alveolar": ["lung alveolar epithelium"], "bronchial": ["lung bronchus epithelium"], "fibroblast": ["colon fibroblasts", "heart fibroblasts"],
         "macrophage": ["colon macrophages"], "smooth_muscle": ["smooth muscle"], "endothelium": ["vascular endothelium"], "T": ["cd4 t cells", "cd8 t cells"],
         "B": ["b cells"], "neutrophil": ["neutrophils"], "monocyte": ["monocytes"], "adipocytes": ["adipocytes"]}
P = {k: [R_(n) for n in v] for k, v in names.items()}; idx = P["alveolar"][0].index
for v in P.values():
    for x in v: idx = idx.intersection(x.index)
S = pd.read_csv(os.path.join(HERE, "gse83842_samples.csv"), dtype={"slide": str}); S = S[S.kind == "normal"].set_index("gsm")
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
pairs = [(j, k) for j, k in itertools.permutations(S.index, 2) if S.slide[j] == S.slide[k]]; pl = {p: list(S.index[S.slide == p]) for p in S.slide.unique()}
NP = 12; crit = int(binom.ppf(0.99, NP, 0.05)) + 1
out = [f"arrays: patient normals {len(S)} on {S.slide.nunique()} slides | loci {len(common)} | blind sites {len(hi)} + {len(lo)} | stand-in pairs {len(pairs)} | N {NP}, CRIT {crit}"]
null = np.array([base[j] / base[k] for j, k in pairs]); bar = float(np.quantile(null, .95))
out.append(f"1 spread: stand-in R SD {null.std(ddof=1):.4f}, 95th percentile (pair bar) {bar:.4f}  [lung 1 SD 0.0860, colon SD 0.0502]")
def other(j): c = [x for x in pl[S.slide[j]] if x != j]; return X[rg.choice(c)]
def blend(j, d, tpl, lo_a=.3, hi_a=.6):
    a = rg.uniform(lo_a, hi_a); T = tpl.copy(); T[lo_] = T[lo_] + d * (1 - T[lo_]); T[hi_] = T[hi_] * (1 - d)
    return np.clip((1 - a) * X[j] + a * T, 0, 1)
imm = Mu[["macrophage", "T", "B", "neutrophil", "monocyte"]].mean(1).values; rows = []
def calls(v):
    above = v > bar; lr = np.log(v); c, w = [], []
    for _ in range(2000):
        i = rg.choice(len(v), size=NP, replace=False); c.append(above[i].sum() >= crit); w.append(wilcoxon(lr[i], alternative="greater", method="exact").pvalue < 0.01)
    return float(np.mean(c)), float(np.mean(w)), float(above.mean())
def case(name, f):
    v = np.array([f(j, k) for j, k in pairs]); cc, ww, sh = calls(v)
    rows.append(dict(case=name, mean_R=round(float(v.mean()), 4), share_above_bar=round(sh, 3), count_call=round(cc, 3), every_pair_call=round(ww, 3)))
    out.append(f"  {name}: mean R {v.mean():.4f}, share above bar {sh:.3f}, COUNT {cc:.3f}, EVERY-PAIR {ww:.3f}")
out.append("2, 4, 5 cases:")
case("NULL", lambda j, k: base[j] / base[k])
case("SHIFT other normal", lambda j, k: H(blend(j, 0, other(j))[Sx]).mean() / base[k])
case("SHIFT immune", lambda j, k: H(blend(j, 0, imm)[Sx]).mean() / base[k])
for d in (.005, .0075, .01, .02, .05): case(f"d{d} other normal", lambda j, k, d=d: H(blend(j, d, other(j))[Sx]).mean() / base[k])
T = pd.DataFrame(rows).set_index("case")
elig = {r: all(T.loc[c, col] <= 0.05 for c in ("NULL", "SHIFT other normal", "SHIFT immune")) for r, col in (("COUNT", "count_call"), ("EVERY-PAIR", "every_pair_call"))}
det = {"COUNT": T.loc["d0.0075 other normal", "count_call"], "EVERY-PAIR": T.loc["d0.0075 other normal", "every_pair_call"]}
ch = [r for r in ("COUNT", "EVERY-PAIR") if elig[r]]; chosen = max(ch, key=lambda r: (det[r], r == "COUNT")) if ch else None
col = {"COUNT": "count_call", "EVERY-PAIR": "every_pair_call"}.get(chosen)
out.append(f"5 rules: eligible {elig}; detection at d 0.0075 {det}; CHOSEN RULE: {chosen}")
comp = {"stroma": Mu[["fibroblast", "smooth_muscle", "endothelium"]].mean(1).values, "fibroblasts": Mu["fibroblast"].values, "smooth muscle": Mu["smooth_muscle"].values,
        "endothelium": Mu["endothelium"].values, "immune": imm, "alveolar": Mu["alveolar"].values, "bronchial": Mu["bronchial"].values, "adipocytes": Mu["adipocytes"].values, "other normal lung": None}
D7 = []
if chosen:
    for nm, tpl in comp.items():
        for lo_a, hi_a in ((.3, .6), (.6, .9)):
            v = np.array([H(blend(j, 0, other(j) if tpl is None else tpl, lo_a, hi_a)[Sx]).mean() / base[k] for j, k in pairs]); cc, ww, sh = calls(v)
            D7.append(dict(component=nm, share=f"{lo_a}-{hi_a}", mean_R=round(float(v.mean()), 4), call=round(cc if chosen == "COUNT" else ww, 3)))
    W7 = pd.DataFrame(D7); w = W7.sort_values("call").iloc[-1]
    out.append(f"7 direction: mean R above 1 in {int((W7.mean_R > 1).sum())} of {len(W7)} mixes (max {W7.mean_R.max():.4f}); worst {w.component} {w.share}, chosen-rule false call {w.call}")
tar, epi = [], []
for g in S.index:
    oth = [base[x] for x in pl[S.slide[g]] if x != g]
    if len(oth) < 2: continue
    wgt, _ = nnls(Mu.values[inf_], X[g][inf_]); wgt = wgt / wgt.sum(); tar.append(base[g] / np.median(oth)); epi.append(wgt[0] + wgt[1])
rho, p = spearmanr(tar, epi) if len(tar) >= 5 else (np.nan, np.nan); out.append(f"3 independence (reported; replaced by test 7): Spearman {rho:+.3f} (p {p:.2f}), n {len(tar)}")
verd = {"2 composition <= 0.10": T.loc["SHIFT other normal", "share_above_bar"] <= 0.10 and T.loc["SHIFT immune", "share_above_bar"] <= 0.10,
        "4 planted 5 % >= 0.80": T.loc["d0.05 other normal", "share_above_bar"] >= 0.80,
        "5 power (chosen rule detects d 0.05 >= 0.80)": bool(chosen) and T.loc["d0.05 other normal", col] >= 0.80,
        "7 direction (no mix fakes the chosen rule)": bool(chosen) and bool(D7) and max(x["call"] for x in D7) <= 0.05}
out += [f"{k}: {'PASSED' if v else 'FAILED'}" for k, v in verd.items()]
if chosen: out.append(f"reported: chosen rule detects lung 1's lean (d 0.0075) in {T.loc['d0.0075 other normal', col]:.3f} of draws" + (" -> a NO FINGERPRINT would read 'not decidable at lung 1's lean'" if T.loc['d0.0075 other normal', col] < 0.5 else ""))
T.to_csv(os.path.join(HERE, "lung2_synthetic_output.csv")); pd.DataFrame(D7).to_csv(os.path.join(HERE, "lung2_direction_output.csv"), index=False)
open(os.path.join(HERE, "lung2_synthetic_summary.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
