"""DEV-TISSUE-COLON-01 review (descriptive, after scoring): distribution of R = Met-A(tumour) / Met-A(own adjacent normal) over the 88 sealed pairs,
sites read per pair, binomial p of the count above the bar at 0.05 per pair, and R by plate and by stage. Reads scored_colon_tissue.json.
Writes why_colon_tissue_output.txt. Usage: python3 why_colon_tissue.py"""
import os, json, pandas as pd
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__)); O = json.load(open(os.path.join(HERE, "scored_colon_tissue.json"))); R = pd.DataFrame(O["rows"]); b = O["bar"]
out = [f"pairs {len(R)} | above bar {O['above_bar']} (bar {b:.4f}; needed {O['crit']}) | binomial p at 0.05 per pair {binom.sf(O['above_bar'] - 1, len(R), .05):.1e}",
       f"R quantiles {R.R.quantile([0, .1, .25, .5, .75, .9, 1]).round(4).to_dict()}",
       f"tumours above their own normal: {int((R.R > 1).sum())} of {len(R)}; below: {int((R.R < 1).sum())}",
       f"sites read per pair: {R.sites.min()}-{R.sites.max()} of 6000"]
for k in ("plate", "stage"): out.append(f"R by {k} (n, median, above bar): " + "; ".join(f"{v} {len(g)}, {g.R.median():.3f}, {int((g.R > b).sum())}" for v, g in R.groupby(k)))
open(os.path.join(HERE, "why_colon_tissue_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
