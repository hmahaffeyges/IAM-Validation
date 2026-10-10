"""Rebuilds the values of neutrophil_reference_v1_2.json that the chain reads (neutrophil_H_mean, neutrophil_H_sd_shrunk, healthy_clustering_LOO,
healthy_clustering_median) with freeze_v13.py's own formulas (shrunk SD k = 10; clustering = variance of block means x sqrt(w) over the variance of
the site z-scores), block w = 10 as the v1.2 change states, on the same 6 GSE110554 arrays through chain Stage 1. Usage: reproduce_reference_v12.py BETAS_DIR"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); MP = os.path.abspath(os.path.join(HERE, "../../.."))
R = json.load(open(os.path.join(MP, "chain/Runtime Matrices/Met_A_Floors/neutrophil_reference_v1_2.json")))
H = lambda x: -(np.clip(x, 1e-6, 1 - 1e-6) * np.log2(np.clip(x, 1e-6, 1 - 1e-6)) + (1 - np.clip(x, 1e-6, 1 - 1e-6)) * np.log2(1 - np.clip(x, 1e-6, 1 - 1e-6)))
uniq = ["GSM2998021", "GSM2998057", "GSM2998116", "GSM2998023", "GSM2998143", "GSM2998030"]
M = pd.DataFrame({g: pd.read_parquet(os.path.join(sys.argv[1], g + ".parquet")).iloc[:, 0].astype("float64") for g in uniq})
So = pd.Index(R["sites_ordered"]); HN = pd.concat([H(M.loc[So, g]) for g in uniq], axis=1); sd = HN.std(1); sp = np.sqrt(np.nanmedian(sd ** 2)); n = len(uniq)
s_sh = np.sqrt(((n - 1) * sd ** 2 + 10 * sp ** 2) / (n - 1 + 10))
def clus(z, w):
    o = z.dropna().values; nb = len(o) // w; b = o[:nb * w].reshape(nb, w).mean(1) * np.sqrt(w); return float(np.var(b) / np.var(o))
base = [round(clus((H(M.loc[So, g]) - pd.concat([H(M.loc[So, x]) for x in uniq if x != g], axis=1).mean(1)) / s_sh, R["clustering_block"]), 4) for g in uniq]
dm = np.nanmax(np.abs(np.round(HN.mean(1).values, 6) - np.array(R["neutrophil_H_mean"]))); ds = np.nanmax(np.abs(np.round(s_sh.values, 6) - np.array(R["neutrophil_H_sd_shrunk"])))
print(f"sites {len(So)} | H_mean max diff {dm:.1e} | H_sd_shrunk max diff {ds:.1e}")
print("clustering LOO rebuilt", base, "| chain", R["healthy_clustering_LOO"]); print("median rebuilt", round(float(np.median(base)), 4), "| chain", R["healthy_clustering_median"])
