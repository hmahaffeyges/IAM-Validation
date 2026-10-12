"""DEV-TISSUE-LIVER-01 review (descriptive, after scoring; not a test): distribution of R = Met-A(tumor) / Met-A(own normal liver) over the 50
sealed pairs, binomial p of the count above the bar at 0.05 per pair, pairs above 1, R by plate, sites read, and the stand-in spread the bar came
from. Reads scored_liver_tissue.json and liver_synthetic_summary.txt. Writes why_liver_tissue_output.txt. Usage: python3 why_liver_tissue.py"""
import os, json, math, pandas as pd
from scipy.stats import binom, wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); O = json.load(open(os.path.join(HERE, "scored_liver_tissue.json"))); R = pd.DataFrame(O["rows"]); b = O["bar"]
q = ", ".join(f"{x:.3f}" for x in R.R.quantile([0, .25, .5, .75, 1]))
out = [f"verdict {O['VERDICT']}: {O['above_bar']} of {O['pairs']} above the bar {b:.4f} (needed {O['crit']}); binomial p at 0.05 per pair {binom.sf(O['above_bar'] - 1, len(R), .05):.1e}",
       f"R quantiles (min, 25 %, median, 75 %, max) {q}",
       f"pairs with R > 1: {int((R.R > 1).sum())} of {len(R)}; below 1: {int((R.R < 1).sum())}",
       f"sites read per pair: {R.sites.min()}-{R.sites.max()} of {len(json.load(open(os.path.join(HERE, 'sealed_liver_tissue.json')))['sites_methylated']) + 3000}",
       "R by plate (n, median, above bar): " + "; ".join(f"{p} {len(g)}, {g.R.median():.3f}, {int((g.R > b).sum())}" for p, g in R.groupby("plate")),
       [l for l in open(os.path.join(HERE, "liver_synthetic_summary.txt")).read().split("\n") if l.startswith("1 spread")][0]]
open(os.path.join(HERE, "why_liver_tissue_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
