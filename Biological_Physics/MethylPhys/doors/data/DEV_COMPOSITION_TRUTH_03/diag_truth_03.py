"""DEV-COMPOSITION-TRUTH-03 diagnosis (2026-10-10, after reading; development). Is the laboratory-template truth precise enough for a
0.02 bar? 450K arrays only (30 people). Inputs: betas_GPL13534.parquet (inputs_sha256.json), manifest.csv, truth03_rows.csv (score_truth_03.py).
Run: python3 diag_truth_03.py BETAS_GPL13534.parquet TRUTH03_ROWS.csv"""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import nnls
HERE = os.path.dirname(os.path.abspath(__file__)); X = pd.read_parquet(sys.argv[1]); R3 = pd.read_csv(sys.argv[2])
m = pd.read_csv(os.path.join(HERE, "manifest.csv")); m = m[m.gpl == "GPL13534"]; CELLS = ["CD15", "CD14", "CD19", "CD4", "CD56", "CD8"]
T = pd.DataFrame({c: X[m[m.cell == c].gsm].mean(1) for c in CELLS}).dropna(); W = X[m[m.cell == "WB"].gsm].loc[T.index].dropna(); T = T.loc[W.index]
def truth(s): return np.array([(lambda f: f[0] / f.sum())(nnls(T.loc[s].values, W.loc[s, g].values)[0]) for g in W.columns])
order = T.std(1).sort_values(ascending=False).index
fit = [nnls(T.loc[order[:6000]].values, W.loc[order[:6000], g].values) for g in W.columns]
rms = [np.sqrt(np.mean((W.loc[order[:6000], g].values - T.loc[order[:6000]].values @ f) ** 2)) for (f, _), g in zip(fit, W.columns)]
print(f"1. whole blood as a mixture of the six sorted templates: RMS residual median {np.median(rms):.4f} (the simulation assumed array noise 0.02-0.06 and fitted it)")
d = T.drop(columns="CD15"); spec = T.index[((T.CD15 - d.max(1)) > 0.4) | ((d.min(1) - T.CD15) > 0.4)]
print(f"2. granulocyte signal in the sorted templates ({len(spec)} granulocyte-specific sites):")
for c in CELLS[1:]:
    o = [k for k in CELLS[1:] if k != c]; b = T.loc[spec, o].median(1); print(f"   sorted {c:4s}: {((T.loc[spec, c] - b) / (T.loc[spec, 'CD15'] - b)).median():+.3f}")
base = truth(order[:6000]); ae = R3[R3.gpl == "GPL13534"].set_index("gsm").loc[W.columns].atlas_e_gran.values
print("3. truth (CD15 share) by site set, untared betas:")
for k, s in (("top 6000 (as written)", order[:6000]), ("top 2000", order[:2000]), ("top 20000", order[:20000]), ("granulocyte-specific", spec),
             ("ranked 6001-12000", order[6000:12000]), ("random 20000", T.index[np.random.default_rng(1).choice(len(T), 20000, replace=False)])):
    t = truth(s); print(f"   {k:22s} median {np.median(t):.3f} | vs as written {np.median(t - base):+.3f} (max |{np.abs(t - base).max():.3f}|) | atlas_e - truth {np.median(ae - t):+.3f}")
