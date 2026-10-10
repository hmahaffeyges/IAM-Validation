"""DEV-WRITER-02 descriptive numbers of the outcome note (recorded, not tested): per strand-pooled context class, pooled over the 153 samples,
copy error and de novo rate; minimum counts; Spearman correlations with the effective discrimination and with each other.
Usage: python3 describe_contexts.py   (reads context_counts.csv, enzyme_D_256.csv)"""
import os, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
C = pd.read_csv(os.path.join(HERE, "context_counts.csv"), keep_default_na=False); Z = pd.read_csv(os.path.join(HERE, "enzyme_D_256.csv")).set_index("ctx")
C = C[C.ctx != "NA"].copy(); C["cls"] = [min(c, Z.loc[c, "rc"]) for c in C.ctx]
G = C.groupby("cls")[["meth_err", "meth_opp", "ctrl_err", "ctrl_opp"]].sum(); G["eps"] = G.meth_err / G.meth_opp; G["ctrl"] = G.ctrl_err / G.ctrl_opp
G["Deff"] = 1 / Z.pred1.reindex(G.index) - 1
print(f"classes {len(G)} | min opportunities {int(G.meth_opp.min())} | min errors {int(G.meth_err.min())}")
print(f"Spearman: copy error vs D_eff {spearmanr(G.eps, G.Deff)[0]:.3f} | de novo vs D_eff {spearmanr(G.ctrl, G.Deff)[0]:.3f} | copy error vs de novo {spearmanr(G.eps, G.ctrl)[0]:.3f}")
