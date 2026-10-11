"""DEV-TISSUE-BREAST-01 synthetic test 5 (2026-10-11; real normal arrays only). Composition-blind sites: loci whose atlas posterior mean is within
0.05 of the luminal value in EVERY breast-tissue panel cell (luminal, basal, adipocytes, fibroblast, endothelium, macrophage, T, B) and whose
luminal value is 0.75-0.95 or 0.05-0.25 with SD <= 0.05 (up to 3,000 per channel, smallest spread across cells). A change in composition cannot move
them; a change in methylation can. Reading: plain Met-A at those sites, same-plate tare (median of the other normals on the plate). Same plants as
plant_real_normals.py (composition shift toward luminal 30-60 %, planted change d in the luminal part). PASS needs SHIFT to cross the healthy 95th
percentile at <= 0.10 and d = 0.05 at >= 0.80. Writes blind_sites_test_output.csv. Usage: python3 blind_sites_test.py POSTDIR BETADIR"""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); argv = sys.argv[:3]; sys.argv = argv
src = open(os.path.join(HERE, "null_normals.py")).read(); exec(src[:src.index("hi = L[")])            # panel Mu, L, common, Bt
spread = (Mu.sub(Mu["luminal"], axis=0)).abs().max(1); ok = (spread <= 0.05) & (L.sd <= 0.05)
hi = L[ok & L["mean"].between(.75, .95)].assign(sp=spread).sort_values("sp").index[:3000]; lo = L[ok & L["mean"].between(.05, .25)].assign(sp=spread).sort_values("sp").index[:3000]
S = hi.union(lo); S_ = Mu.index.get_indexer(S); hi_ = Mu.index.get_indexer(hi); lo_ = Mu.index.get_indexer(lo)
H = lambda b: -(np.clip(b, 1e-6, 1 - 1e-6) * np.log2(np.clip(b, 1e-6, 1 - 1e-6)) + (1 - np.clip(b, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(b, 1e-6, 1 - 1e-6)))
T = pd.read_csv(os.path.join(HERE, "null_normals_output.csv")).set_index("sample"); plates = T.plate; keep = T.index[plates.map(plates.value_counts()) >= 3]
X = {s: Bt[s].loc[common].values for s in keep}; base = {s: H(X[s][S_]).mean() for s in keep}
ref = {s: np.median([base[o] for o in keep if plates[o] == plates[s] and o != s]) for s in keep}
tared = np.array([base[s] / ref[s] for s in keep]); bar = np.quantile(tared, .95); TPL = os.environ.get("TEMPLATE")   # optional: a real 450K epithelial array (parquet) as the plant template instead of the atlas luminal mean
lum = (lambda t: t.reindex(Mu.index).fillna(Mu["luminal"]).values)(pd.read_parquet(TPL).beta.rename(lambda x: str(x))) if TPL else Mu["luminal"].values; rg = np.random.default_rng(20261011); rows = []
print(f"blind sites: {len(hi)} methylated + {len(lo)} unmethylated | tared normals SD {tared.std(ddof=1):.4f}, 95th percentile {bar:.4f}")
for case, (sh, d) in {"NULL": (0, 0), "SHIFT": (1, 0), "d0.02": (1, .02), "d0.05": (1, .05), "d0.10": (1, .10)}.items():
    v = []
    for s in keep:
        a = rg.uniform(.3, .6) if sh else 0.0; L_ = lum.copy(); L_[lo_] = L_[lo_] + d * (1 - L_[lo_]); L_[hi_] = L_[hi_] * (1 - d)
        v.append(H(np.clip((1 - a) * X[s] + a * L_, 0, 1)[S_]).mean() / ref[s])
    v = np.array(v); rows.append(dict(case=case, n=len(v), mean=round(v.mean(), 4), sd=round(v.std(ddof=1), 4), share_above_bar=round((v > bar).mean(), 3), bar=round(bar, 4))); print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "blind_sites_test_output.csv" if not os.environ.get("TEMPLATE") else "blind_sites_test_" + os.path.basename(os.environ["TEMPLATE"]).split(".")[0] + "_output.csv"), index=False)
ok_ = R.set_index("case"); print("SYNTHETIC TEST 5", "PASSED" if ok_.loc["SHIFT", "share_above_bar"] <= 0.10 and ok_.loc["d0.05", "share_above_bar"] >= 0.80 else "FAILED")
