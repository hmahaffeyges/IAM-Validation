"""DEV-TISSUE-LUNG-01 review (descriptive, after scoring; not a test): distribution of R = Met-A(tumour) / Met-A(own normal lung) over the 28
sealed pairs, binomial p of the count above the bar at 0.05 per pair, R by slide and by asbestos exposure, and the stand-in spread the bar came
from against the colon and breast designs. Reads scored_lung_tissue.json and lung_synthetic_summary.txt. Writes why_lung_tissue_output.txt.
Usage: python3 why_lung_tissue.py"""
import os, json, pandas as pd
from scipy.stats import binom, wilcoxon
HERE = os.path.dirname(os.path.abspath(__file__)); O = json.load(open(os.path.join(HERE, "scored_lung_tissue.json"))); R = pd.DataFrame(O["rows"]); b = O["bar"]
q = ", ".join(f"{x:.3f}" for x in R.R.quantile([0, .25, .5, .75, 1]))
out = [f"verdict {O['VERDICT']}: {O['above_bar']} of {O['pairs']} above the bar {b:.4f} (needed {O['crit']}); binomial p at 0.05 per pair {binom.sf(O['above_bar'] - 1, len(R), .05):.3f}",
       f"R quantiles (min, 25 %, median, 75 %, max) {q}",
       f"pairs with R > 1: {int((R.R > 1).sum())} of {len(R)}; Wilcoxon signed-rank on log R (two-sided) p {wilcoxon(R.R.apply(lambda x: __import__('math').log(x))).pvalue:.4f}",
       f"sites read per pair: {R.sites.min()}-{R.sites.max()} of 6000",
       "R by slide (n, median, above bar): " + "; ".join(f"{s} {len(g)}, {g.R.median():.3f}, {int((g.R > b).sum())}" for s, g in R.groupby("slide")),
       "R by asbestos exposure (n, median, above bar): " + "; ".join(f"{s} {len(g)}, {g.R.median():.3f}, {int((g.R > b).sum())}" for s, g in R.groupby("asbestos")),
       [l for l in open(os.path.join(HERE, "lung_synthetic_summary.txt")).read().split("\n") if l.startswith("1 spread")][0]]
open(os.path.join(HERE, "why_lung_tissue_output.txt"), "w").write("\n".join(out) + "\n"); print("\n".join(out))
