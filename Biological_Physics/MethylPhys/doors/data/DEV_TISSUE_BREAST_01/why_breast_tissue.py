"""DEV-TISSUE-BREAST-01 review (descriptive, after scoring): distribution of R = Met-A(tumour) / Met-A(own normal) over the 53 sealed pairs,
sites read per pair, the binomial p of the count above the bar at 0.05 per pair, and R by plate. Reads scored_breast_tissue.json.
Writes why_breast_tissue_output.txt. Usage: python3 why_breast_tissue.py"""
import os, json, numpy as np, pandas as pd
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__)); O = json.load(open(os.path.join(HERE, "scored_breast_tissue.json"))); R = pd.DataFrame(O["rows"])
q = R.R.quantile([0, .1, .25, .5, .75, .9, 1]).round(4).to_dict()
out = [f"pairs {len(R)} | above bar {O['above_bar']} (bar {O['bar']:.4f}; needed {O['crit']}) | binomial p at 0.05 per pair {binom.sf(O['above_bar'] - 1, len(R), .05):.1e}",
       f"R quantiles {q}", f"pairs with R > 1: {int((R.R > 1).sum())} of {len(R)}; R < 1: {int((R.R < 1).sum())}",
       f"sites read per pair: {R.sites.min()}-{R.sites.max()} of 6000",
       "R by plate (n, median, above bar): " + "; ".join(f"{p} {len(g)}, {g.R.median():.3f}, {int((g.R > O['bar']).sum())}" for p, g in R.groupby("plate"))]
open(os.path.join(HERE, "why_breast_tissue_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
